#include "jllama.h"

#include "arg.h"
#include "build-info.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <thread>
#include <stdexcept>

// We store some references to Java classes and their fields/methods here to speed up things for later and to fail
// early on if anything can't be found. This happens when the JVM loads the shared library (see `JNI_OnLoad`).
// The references remain valid throughout the whole life of the shared library, on `JNI_OnUnload` they are released.

namespace {

// Sentinel value used by llama.cpp (since b7433) to indicate that n_parallel
// should be resolved automatically by the host application. Introduced in:
// common_params_parser_init() for LLAMA_EXAMPLE_SERVER in common/arg.cpp.
static constexpr int N_PARALLEL_AUTO = -1;

// Default n_parallel for the embedded Java library. Unlike the standalone
// llama.cpp server (which resolves auto to 4 for multi-client throughput),
// the Java bindings run in-process with a single caller, so 1 slot is the
// appropriate default and preserves pre-b7433 behaviour.
static constexpr int N_PARALLEL_DEFAULT = 1;

// jllama_context is defined in jni_helpers.hpp.

JavaVM *g_vm = nullptr;

// classes
jclass c_llama_model = nullptr;
jclass c_standard_charsets = nullptr;
jclass c_string = nullptr;
jclass c_hash_map = nullptr;
jclass c_map = nullptr;
jclass c_set = nullptr;
jclass c_entry = nullptr;
jclass c_iterator = nullptr;
jclass c_integer = nullptr;
jclass c_float = nullptr;
jclass c_biconsumer = nullptr;
jclass c_llama_error = nullptr;
jclass c_log_level = nullptr;
jclass c_log_format = nullptr;
jclass c_error_oom = nullptr;

// constructors
jmethodID cc_hash_map = nullptr;
jmethodID cc_integer = nullptr;
jmethodID cc_float = nullptr;

// methods
jmethodID m_get_bytes = nullptr;
jmethodID m_entry_set = nullptr;
jmethodID m_set_iterator = nullptr;
jmethodID m_iterator_has_next = nullptr;
jmethodID m_iterator_next = nullptr;
jmethodID m_entry_key = nullptr;
jmethodID m_entry_value = nullptr;
jmethodID m_map_put = nullptr;
jmethodID m_int_value = nullptr;
jmethodID m_float_value = nullptr;
jmethodID m_biconsumer_accept = nullptr;

// fields
jfieldID f_model_pointer = nullptr;
jfieldID f_utf_8 = nullptr;
jfieldID f_log_level_debug = nullptr;
jfieldID f_log_level_info = nullptr;
jfieldID f_log_level_warn = nullptr;
jfieldID f_log_level_error = nullptr;
jfieldID f_log_format_json = nullptr;
jfieldID f_log_format_text = nullptr;

// objects
jobject o_utf_8 = nullptr;
jobject o_log_level_debug = nullptr;
jobject o_log_level_info = nullptr;
jobject o_log_level_warn = nullptr;
jobject o_log_level_error = nullptr;
jobject o_log_format_json = nullptr;
jobject o_log_format_text = nullptr;
jobject o_log_callback = nullptr;

/**
 * Returns the jllama_context wrapper for the Java LlamaModel object.
 * Used by the delete path and any method that needs jctx directly.
 * Returns nullptr silently on a null handle (valid no-op for a destructor).
 */
[[nodiscard]] static jllama_context *get_jllama_context(JNIEnv *env, jobject obj) {
    return get_jllama_context_impl(env, obj, f_model_pointer);
}

/**
 * Formats e as a JSON invalid-request error and throws it via JNI.
 */
static void throw_invalid_request(JNIEnv *env, const std::exception &e) {
    const auto &err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
    env->ThrowNew(c_llama_error, err.dump().c_str());
}

/**
 * Returns true if result is non-null and not an error.
 * On failure throws via JNI and returns false.  Callers must return immediately.
 */
[[nodiscard]] static bool result_ok_or_throw(JNIEnv *env,
                                              const server_task_result_ptr &result) {
    if (!result || result->is_error()) {
        env->ThrowNew(c_llama_error,
                      result ? get_result_error_message(result).c_str() : "No result");
        return false;
    }
    return true;
}

/**
 * Returns true if the batch completed without a task-level error.
 * On failure throws via JNI and returns false.  Callers must return immediately.
 */
[[nodiscard]] static bool batch_ok_or_throw(
        JNIEnv *env,
        const server_response_reader::batch_response &br) {
    if (br.error) {
        env->ThrowNew(c_llama_error, get_result_error_message(br.error).c_str());
        return false;
    }
    return true;
}

/**
 * Parse the OAI chat-completion body through oaicompat_chat_params_parse and
 * write the result into `out`.  Returns true on success; on failure throws and
 * returns false.
 */
[[nodiscard]] static bool parse_oai_chat_params(JNIEnv *env,
                                                 server_context *ctx_server,
                                                 json &body,
                                                 json &out) {
    try {
        std::vector<raw_buffer> files;
        auto meta = ctx_server->get_meta();
        out = oaicompat_chat_params_parse(body, meta.chat_params, files);
        return true;
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return false;
    }
}

// Tokenise the prompt in `data` and fill task.tokens + task.params.
// Callers must wrap this in try/catch (params_from_json_cmpl can throw).
static void populate_completion_task(server_task                          &task,
                                     jllama_context                       *jctx,
                                     int                                   n_ctx_slot,
                                     const std::vector<llama_logit_bias>  &logit_bias_eog,
                                     const json                           &data) {
    auto tokenized_prompts = tokenize_input_prompts(
            jctx->vocab, nullptr, data.at("prompt"), true, true);
    if (!tokenized_prompts.empty()) {
        task.tokens = std::move(tokenized_prompts[0]);
    }
    task.params = server_task::params_from_json_cmpl(
            jctx->vocab, jctx->params, n_ctx_slot, logit_bias_eog, data);
}

[[nodiscard]] static jint dispatch_streaming_completion(JNIEnv           *env,
                                                         jllama_context   *jctx,
                                                         const json       &data,
                                                         server_task_type  task_type,
                                                         task_response_type res_type) {
    server_context *ctx_server = &jctx->server;
    auto meta = ctx_server->get_meta();
    auto *rd  = new server_response_reader(ctx_server->get_response_reader());
    int   tid = rd->get_new_id();
    try {
        server_task task(task_type);
        task.id     = tid;
        populate_completion_task(task, jctx, meta.slot_n_ctx, meta.logit_bias_eog, data);
        task.params.res_type = res_type;
        rd->post_task(std::move(task));
    } catch (const std::exception &e) {
        delete rd;
        throw_invalid_request(env, e);
        return 0;
    }
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    jctx->readers[tid].reset(rd);
    return static_cast<jint>(tid);
}

/**
 * Build one completion/infill task from `data`, post it, wait for all results,
 * and serialise them to a jstring.
 * Used by handleCompletions, handleCompletionsOai, handleChatCompletions,
 * handleInfill — the blocking completion path.
 * On error: throws via JNI and returns nullptr.
 */
[[nodiscard]] static jstring dispatch_blocking_completion(JNIEnv           *env,
                                                           jllama_context   *jctx,
                                                           const json       &data,
                                                           server_task_type  task_type,
                                                           task_response_type res_type) {
    server_context *ctx_server = &jctx->server;
    auto meta = ctx_server->get_meta();
    auto rd   = ctx_server->get_response_reader();
    server_task task(task_type);
    task.id = rd.get_new_id();
    try {
        populate_completion_task(task, jctx, meta.slot_n_ctx, meta.logit_bias_eog, data);
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }
    task.params.res_type = res_type;
    rd.post_task(std::move(task));
    auto br = rd.wait_for_all([] { return false; });
    if (!batch_ok_or_throw(env, br)) return nullptr;
    return results_to_jstring_impl(env, br.results);
}

/**
 * Convert a Java string to a std::string
 */
std::string parse_jstring(JNIEnv *env, jstring java_string) {
    auto *const string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);

    auto length = (size_t)env->GetArrayLength(string_bytes);
    jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char *)byte_elements, length);

    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
    env->DeleteLocalRef(string_bytes);

    return string;
}

/**
 * Convert a Java string to a parsed JSON object.
 * Combines parse_jstring + json::parse, which every parameter-taking JNI
 * function needs before it can read its arguments.
 */
static json parse_json_params(JNIEnv *env, jstring jparams) {
    return json::parse(parse_jstring(env, jparams));
}

/**
 * Convenience wrapper around require_json_field_impl (jni_helpers.hpp).
 * Returns false and throws if `field` is absent from `data`.
 */
[[nodiscard]] static bool require_json_field(JNIEnv *env, const json &data,
                                              const char *field) {
    return require_json_field_impl(env, data, field, c_llama_error);
}

// Build a single indexed token task for batch submission (rerank and embedding).
// Assigns the reader-allocated id; moves tokens into the task.
[[nodiscard]] static server_task build_indexed_token_task(server_response_reader &rd,
                                                           server_task_type        type,
                                                           server_tokens         &&tokens,
                                                           int                     index,
                                                           task_response_type      res_type) {
    server_task task(type);
    task.id              = rd.get_new_id();
    task.tokens          = std::move(tokens);
    task.index           = index;
    task.params.res_type = res_type;
    return task;
}

// Post a single pre-built task, wait for its result, and return JSON as a jstring.
// The task's id field is assigned here; callers must not set it beforehand.
[[nodiscard]] static jstring dispatch_one_shot_task(JNIEnv           *env,
                                                     server_context   *ctx_server,
                                                     server_task       task) {
    auto rd    = ctx_server->get_response_reader();
    task.id    = rd.get_new_id();
    rd.post_task(std::move(task));
    auto result = rd.next([] { return false; });
    if (!result_ok_or_throw(env, result)) return nullptr;
    return json_to_jstring_impl(env, result->to_json());
}

// Post a single slot file task (SAVE or RESTORE), wait for its result, and
// return the result JSON as a jstring.
[[nodiscard]] static jstring exec_slot_file_task(JNIEnv           *env,
                                                  server_context   *ctx_server,
                                                  jint              slotId,
                                                  jstring           jfilename,
                                                  server_task_type  task_type,
                                                  const char       *empty_filename_error) {
    const std::string filename = jfilename != nullptr ? parse_jstring(env, jfilename) : "";
    if (filename.empty()) {
        env->ThrowNew(c_llama_error, empty_filename_error);
        return nullptr;
    }
    server_task task(task_type);
    task.slot_action.id_slot  = slotId;
    task.slot_action.filename = filename;
    task.slot_action.filepath = filename;
    return dispatch_one_shot_task(env, ctx_server, std::move(task));
}

char **parse_string_array(JNIEnv *env, const jobjectArray string_array, const jsize length) {
    auto *const result = static_cast<char **>(malloc(length * sizeof(char *)));

    if (result == nullptr) {
        return nullptr;
    }

    for (jsize i = 0; i < length; i++) {
        auto *const javaString = static_cast<jstring>(env->GetObjectArrayElement(string_array, i));
        const char *cString = env->GetStringUTFChars(javaString, nullptr);
        result[i] = strdup(cString);
        env->ReleaseStringUTFChars(javaString, cString);
    }

    return result;
}

void free_string_array(char **array, jsize length) {
    if (array != nullptr) {
        for (jsize i = 0; i < length; i++) {
            free(array[i]);
        }
        free(array);
    }
}

/**
 * Since Java expects utf16 but std::strings are utf8, we can't directly use `env->NewString` or `env-NewString`,
 * but we directly send the bytes and do the conversion in Java. Unfortunately, there isn't a nice/standardized way to
 * do this conversion in C++
 */
jbyteArray parse_jbytes(JNIEnv *env, const std::string &string) {
    jsize length = string.size(); // NOLINT(*-narrowing-conversions)
    jbyteArray bytes = env->NewByteArray(length);
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte *>(string.c_str()));
    return bytes;
}

/**
 * Map a llama.cpp log level to its Java enumeration option.
 */
jobject log_level_to_jobject(ggml_log_level level) {
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        return o_log_level_error;
    case GGML_LOG_LEVEL_WARN:
        return o_log_level_warn;
    default:
    case GGML_LOG_LEVEL_INFO:
        return o_log_level_info;
    case GGML_LOG_LEVEL_DEBUG:
        return o_log_level_debug;
    }
}

/**
 * Returns the JNIEnv of the current thread.
 */
JNIEnv *get_jni_env() {
    JNIEnv *env = nullptr;
    if (g_vm == nullptr || g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        throw std::runtime_error("Thread is not attached to the JVM");
    }
    return env;
}

bool log_json;
std::function<void(ggml_log_level, const char *, void *)> log_callback;

/**
 * Format a log message as JSON.
 */
std::string format_log_as_json(ggml_log_level level, const char *text) {
    std::string level_str;
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        level_str = "ERROR";
        break;
    case GGML_LOG_LEVEL_WARN:
        level_str = "WARN";
        break;
    case GGML_LOG_LEVEL_INFO:
        level_str = "INFO";
        break;
    default:
    case GGML_LOG_LEVEL_DEBUG:
        level_str = "DEBUG";
        break;
    }
    nlohmann::json log_obj = {{"timestamp", std::time(nullptr)}, {"level", level_str}, {"message", text}};
    return log_obj.dump();
}

/**
 * Invoke the log callback if there is any. When JSON mode is enabled,
 * the message is formatted as a JSON object before forwarding.
 */
void log_callback_trampoline(ggml_log_level level, const char *text, void *user_data) {
    if (log_callback != nullptr) {
        if (log_json) {
            std::string json_text = format_log_as_json(level, text);
            log_callback(level, json_text.c_str(), user_data);
        } else {
            log_callback(level, text, user_data);
        }
    }
}
} // namespace

// Validates the jllama_context at every JNI entry point.  Declares both
// `jctx` and `ctx_server` in the caller's scope; returns the given sentinel
// (omit for void functions) if the model is not loaded.
#define REQUIRE_SERVER_CONTEXT(...) \
    auto *jctx = get_jllama_context(env, obj); \
    if (!jctx) { env->ThrowNew(c_llama_error, "Model is not loaded"); return __VA_ARGS__; } \
    server_context *ctx_server = &jctx->server

/**
 * The VM calls JNI_OnLoad when the native library is loaded (for example, through `System.loadLibrary`).
 * `JNI_OnLoad` must return the JNI version needed by the native library.
 * In order to use any of the new JNI functions, a native library must export a `JNI_OnLoad` function that returns
 * `JNI_VERSION_1_2`. If the native library does not export a JNI_OnLoad function, the VM assumes that the library
 * only requires JNI version `JNI_VERSION_1_1`. If the VM does not recognize the version number returned by
 `JNI_OnLoad`, the VM will unload the library and act as if the library was never loaded.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    g_vm = vm;
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1)) {
        goto error;
    }

    // find classes
    c_llama_model = env->FindClass("de/kherud/llama/LlamaModel");
    c_standard_charsets = env->FindClass("java/nio/charset/StandardCharsets");
    c_string = env->FindClass("java/lang/String");
    c_hash_map = env->FindClass("java/util/HashMap");
    c_map = env->FindClass("java/util/Map");
    c_set = env->FindClass("java/util/Set");
    c_entry = env->FindClass("java/util/Map$Entry");
    c_iterator = env->FindClass("java/util/Iterator");
    c_integer = env->FindClass("java/lang/Integer");
    c_float = env->FindClass("java/lang/Float");
    c_biconsumer = env->FindClass("java/util/function/BiConsumer");
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");
    c_log_level = env->FindClass("de/kherud/llama/LogLevel");
    c_log_format = env->FindClass("de/kherud/llama/args/LogFormat");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_standard_charsets && c_string && c_hash_map && c_map &&
          c_set && c_entry && c_iterator && c_integer && c_float && c_biconsumer && c_llama_error && c_log_level &&
          c_log_format && c_error_oom)) {
        goto error;
    }

    // create references
    c_llama_model = (jclass)env->NewGlobalRef(c_llama_model);
    c_string = (jclass)env->NewGlobalRef(c_string);
    c_hash_map = (jclass)env->NewGlobalRef(c_hash_map);
    c_map = (jclass)env->NewGlobalRef(c_map);
    c_set = (jclass)env->NewGlobalRef(c_set);
    c_entry = (jclass)env->NewGlobalRef(c_entry);
    c_iterator = (jclass)env->NewGlobalRef(c_iterator);
    c_integer = (jclass)env->NewGlobalRef(c_integer);
    c_float = (jclass)env->NewGlobalRef(c_float);
    c_biconsumer = (jclass)env->NewGlobalRef(c_biconsumer);
    c_llama_error = (jclass)env->NewGlobalRef(c_llama_error);
    c_log_level = (jclass)env->NewGlobalRef(c_log_level);
    c_log_format = (jclass)env->NewGlobalRef(c_log_format);
    c_error_oom = (jclass)env->NewGlobalRef(c_error_oom);

    // find constructors
    cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
    cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
    cc_float = env->GetMethodID(c_float, "<init>", "(F)V");

    if (!(cc_hash_map && cc_integer && cc_float)) {
        goto error;
    }

    // find methods
    m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
    m_entry_set = env->GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
    m_set_iterator = env->GetMethodID(c_set, "iterator", "()Ljava/util/Iterator;");
    m_iterator_has_next = env->GetMethodID(c_iterator, "hasNext", "()Z");
    m_iterator_next = env->GetMethodID(c_iterator, "next", "()Ljava/lang/Object;");
    m_entry_key = env->GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
    m_entry_value = env->GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
    m_map_put = env->GetMethodID(c_map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    m_int_value = env->GetMethodID(c_integer, "intValue", "()I");
    m_float_value = env->GetMethodID(c_float, "floatValue", "()F");
    m_biconsumer_accept = env->GetMethodID(c_biconsumer, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key &&
          m_entry_value && m_map_put && m_int_value && m_float_value && m_biconsumer_accept)) {
        goto error;
    }

    // find fields
    f_model_pointer = env->GetFieldID(c_llama_model, "ctx", "J");
    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");
    f_log_format_json = env->GetStaticFieldID(c_log_format, "JSON", "Lde/kherud/llama/args/LogFormat;");
    f_log_format_text = env->GetStaticFieldID(c_log_format, "TEXT", "Lde/kherud/llama/args/LogFormat;");

    if (!(f_model_pointer && f_utf_8 && f_log_level_debug && f_log_level_info &&
          f_log_level_warn && f_log_level_error && f_log_format_json && f_log_format_text)) {
        goto error;
    }

    o_utf_8 = env->NewStringUTF("UTF-8");
    o_log_level_debug = env->GetStaticObjectField(c_log_level, f_log_level_debug);
    o_log_level_info = env->GetStaticObjectField(c_log_level, f_log_level_info);
    o_log_level_warn = env->GetStaticObjectField(c_log_level, f_log_level_warn);
    o_log_level_error = env->GetStaticObjectField(c_log_level, f_log_level_error);
    o_log_format_json = env->GetStaticObjectField(c_log_format, f_log_format_json);
    o_log_format_text = env->GetStaticObjectField(c_log_format, f_log_format_text);

    if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error &&
          o_log_format_json && o_log_format_text)) {
        goto error;
    }

    o_utf_8 = env->NewGlobalRef(o_utf_8);
    o_log_level_debug = env->NewGlobalRef(o_log_level_debug);
    o_log_level_info = env->NewGlobalRef(o_log_level_info);
    o_log_level_warn = env->NewGlobalRef(o_log_level_warn);
    o_log_level_error = env->NewGlobalRef(o_log_level_error);
    o_log_format_json = env->NewGlobalRef(o_log_format_json);
    o_log_format_text = env->NewGlobalRef(o_log_format_text);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        goto error;
    }

    llama_backend_init();

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_6;
}

/**
 * The VM calls `JNI_OnUnload` when the class loader containing the native library is garbage collected.
 * This function can be used to perform cleanup operations. Because this function is called in an unknown context
 * (such as from a finalizer), the programmer should be conservative on using Java VM services, and refrain from
 * arbitrary Java call-backs.
 * Note that `JNI_OnLoad` and `JNI_OnUnload` are two functions optionally supplied by JNI libraries, not exported from
 * the VM.
 */
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_6)) {
        return;
    }

    env->DeleteGlobalRef(c_llama_model);
    env->DeleteGlobalRef(c_string);
    env->DeleteGlobalRef(c_hash_map);
    env->DeleteGlobalRef(c_map);
    env->DeleteGlobalRef(c_set);
    env->DeleteGlobalRef(c_entry);
    env->DeleteGlobalRef(c_iterator);
    env->DeleteGlobalRef(c_integer);
    env->DeleteGlobalRef(c_float);
    env->DeleteGlobalRef(c_biconsumer);
    env->DeleteGlobalRef(c_llama_error);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_log_format);
    env->DeleteGlobalRef(c_error_oom);

    env->DeleteGlobalRef(o_utf_8);
    env->DeleteGlobalRef(o_log_level_debug);
    env->DeleteGlobalRef(o_log_level_info);
    env->DeleteGlobalRef(o_log_level_warn);
    env->DeleteGlobalRef(o_log_level_error);
    env->DeleteGlobalRef(o_log_format_json);
    env->DeleteGlobalRef(o_log_format_text);

    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    llama_backend_free();
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jobjectArray jparams) {
    common_params params;

    const jsize argc = env->GetArrayLength(jparams);
    char **argv = parse_string_array(env, jparams, argc);
    if (argv == nullptr) {
        env->ThrowNew(c_error_oom, "Failed to allocate memory for parameters");
        return;
    }

    // Strip --vocab-only before common_params_parse (not a common_params flag).
    bool vocab_only = false;
    std::vector<char *> filtered_argv = strip_flag_from_argv(argv, static_cast<int>(argc), "--vocab-only", &vocab_only);
    int filtered_argc = static_cast<int>(filtered_argv.size());
    const auto parsed_params = common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_SERVER);
    free_string_array(argv, argc);
    if (!parsed_params) {
        env->ThrowNew(c_llama_error, "Failed to parse model parameters");
        return;
    }

    common_init();

    auto *jctx        = new jllama_context();
    jctx->vocab_only  = vocab_only;
    jctx->params      = params;

    auto fail_load = [&](const char *msg) {
        if (jctx->vocab_only_model) {
            llama_model_free(jctx->vocab_only_model);
        }
        delete jctx;
        env->ThrowNew(c_llama_error, msg);
    };

    // Vocab-only mode: load just the model vocab, skip inference setup.
    if (vocab_only) {
        SRV_INF("loading tokenizer from '%s'\n", params.model.path.c_str());
        llama_model_params mparams = llama_model_default_params();
        mparams.vocab_only = true;
        jctx->vocab_only_model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (!jctx->vocab_only_model) {
            fail_load("could not load tokenizer from given file path");
            return;
        }
        jctx->vocab = llama_model_get_vocab(jctx->vocab_only_model);
        env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(jctx));
        return;
    }

    SRV_INF("loading model '%s'\n", params.model.path.c_str());

    llama_numa_init(params.numa);

    LOG_INF("build_info: %s\n", llama_build_info());
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Resolve the auto sentinel before loading the model.
    if (params.n_parallel <= N_PARALLEL_AUTO) {
        params.n_parallel = N_PARALLEL_DEFAULT;
        jctx->params.n_parallel = N_PARALLEL_DEFAULT;
    }

    LOG_INF("%s: loading model\n", __func__);

    if (!jctx->server.load_model(params)) {
        fail_load("could not load model from given file path");
        return;
    }

    jctx->vocab = llama_model_get_vocab(llama_get_model(jctx->server.get_llama_context()));

    LOG_INF("%s: model loaded\n", __func__);

    {
        auto meta = jctx->server.get_meta();
        LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
                common_chat_templates_source(meta.chat_params.tmpls.get()).c_str(),
                common_chat_format_example(meta.chat_params.tmpls.get(),
                                           jctx->params.use_jinja,
                                           jctx->params.default_template_kwargs).c_str());
    }

    jctx->worker = std::thread([jctx]() {
        JNIEnv *tenv;
        jint res = g_vm->GetEnv((void **)&tenv, JNI_VERSION_1_6);
        bool attached = false;
        if (res == JNI_EDETACHED) {
            res = g_vm->AttachCurrentThread((void **)&tenv, nullptr);
            if (res != JNI_OK) {
                jctx->worker_ready.store(true);
                return;
            }
            attached = true;
        }
        jctx->worker_ready.store(true);
        jctx->server.start_loop();
        if (attached) {
            g_vm->DetachCurrentThread();
        }
    });

    while (!jctx->worker_ready.load()) {
        std::this_thread::yield();
    }

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(jctx));
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_getModelMetaJson(JNIEnv *env, jobject obj) {
    REQUIRE_SERVER_CONTEXT(nullptr);
    if (jctx->vocab_only) {
        json meta = {
            {"vocab_type", llama_vocab_type(jctx->vocab)},
            {"n_vocab",    llama_vocab_n_tokens(jctx->vocab)},
        };
        return json_to_jstring_impl(env, meta);
    }
    auto m = ctx_server->get_meta();
    // Read general.architecture from GGUF metadata via the llama C API.
    char arch_buf[128] = {};
    const llama_model *mdl = llama_get_model(ctx_server->get_llama_context());
    if (mdl) {
        llama_model_meta_val_str(mdl, "general.architecture", arch_buf, sizeof(arch_buf));
    }
    json j = {
        {"vocab_type",   m.model_vocab_type},
        {"n_vocab",      m.model_vocab_n_tokens},
        {"n_ctx_train",  m.model_n_ctx_train},
        {"n_embd",       m.model_n_embd_inp},
        {"n_params",     m.model_n_params},
        {"size",         m.model_size},
        {"modalities",   {{"vision", m.has_inp_image}, {"audio", m.has_inp_audio}}},
        {"name",         m.model_name},
        {"architecture", std::string(arch_buf)},
    };
    return json_to_jstring_impl(env, j);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj, jstring jparams) {
    REQUIRE_SERVER_CONTEXT(0);

    json data = parse_json_params(env, jparams);

    const server_task_type type = is_infill_request(data)
                                       ? SERVER_TASK_TYPE_INFILL
                                       : SERVER_TASK_TYPE_COMPLETION;

    return dispatch_streaming_completion(env, jctx, data, type, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask(JNIEnv *env, jobject obj, jint id_task) {
    REQUIRE_SERVER_CONTEXT();
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    jctx->readers.erase(id_task);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletionJson(JNIEnv *env, jobject obj,
                                                                               jint id_task) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    server_response_reader *rd;
    {
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        auto it = jctx->readers.find(id_task);
        if (it == jctx->readers.end()) {
            env->ThrowNew(c_llama_error, "Task not found");
            return nullptr;
        }
        rd = it->second.get();
    }

    server_task_result_ptr result = rd->next([] { return false; });

    if (!result_ok_or_throw(env, result)) {
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        jctx->readers.erase(id_task);
        return nullptr;
    }

    json response      = result->to_json();
    response["stop"]   = result->is_stop();

    if (result->is_stop()) {
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        jctx->readers.erase(id_task);
    }

    return json_to_jstring_impl(env, response);
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    if (!jctx->params.embedding) {
        env->ThrowNew(c_llama_error,
                      "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
        return nullptr;
    }

    const std::string prompt = parse_jstring(env, jprompt);
    SRV_INF("Calling embedding '%s'\n", prompt.c_str());

    auto tokens = tokenize_mixed(jctx->vocab, prompt, true, true);
    auto rd     = ctx_server->get_response_reader();
    server_task task(SERVER_TASK_TYPE_EMBEDDING);
    task.id     = rd.get_new_id();
    task.tokens = server_tokens(tokens, false);
    task.index  = 0;
    rd.post_task(std::move(task));

    auto br = rd.wait_for_all([] { return false; });
    if (!batch_ok_or_throw(env, br)) return nullptr;

    auto *embd_result = dynamic_cast<server_task_result_embd *>(br.results[0].get());
    if (!embd_result || embd_result->embedding.empty() || embd_result->embedding[0].empty()) {
        env->ThrowNew(c_llama_error, "embedding result is empty");
        return nullptr;
    }
    const std::vector<float> &first_row = embd_result->embedding[0];

    SRV_INF("Embedding has %d columns\n", static_cast<jsize>(first_row.size()));
    return embedding_to_jfloat_array_impl(env, first_row, c_error_oom);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleRerank(JNIEnv *env, jobject obj, jstring jprompt,
                                                                      jobjectArray documents) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    {
        auto meta = ctx_server->get_meta();
        if (!jctx->params.embedding || meta.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            env->ThrowNew(c_llama_error,
                          "This server does not support reranking. Start it with `--reranking` and without `--embedding`");
            return nullptr;
        }
    }

    const std::string prompt = parse_jstring(env, jprompt);

    const jsize amount_documents = env->GetArrayLength(documents);
    auto *document_array = parse_string_array(env, documents, amount_documents);
    auto  document_vector = std::vector<std::string>(document_array, document_array + amount_documents);
    free_string_array(document_array, amount_documents);

    const llama_model *model = llama_get_model(ctx_server->get_llama_context());
    auto rd = ctx_server->get_response_reader();
    std::vector<server_task> tasks;
    tasks.reserve(document_vector.size());
    for (size_t i = 0; i < document_vector.size(); i++) {
        tasks.push_back(build_indexed_token_task(rd, SERVER_TASK_TYPE_RERANK,
            format_prompt_rerank(model, jctx->vocab, nullptr, prompt, document_vector[i]),
            static_cast<int>(i), TASK_RESPONSE_TYPE_NONE));
    }
    rd.post_tasks(std::move(tasks));

    auto br = rd.wait_for_all([] { return false; });
    if (!batch_ok_or_throw(env, br)) return nullptr;
    return json_to_jstring_impl(env, rerank_results_to_json(br.results, document_vector));
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate(JNIEnv *env, jobject obj, jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json data = parse_json_params(env, jparams);

    json templateData;
    if (!parse_oai_chat_params(env, ctx_server, data, templateData)) return nullptr;

    std::string tok_str = templateData.at("prompt");
    return env->NewStringUTF(tok_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleChatCompletions(JNIEnv *env, jobject obj,
                                                                                jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json body = parse_json_params(env, jparams);
    json data;
    if (!parse_oai_chat_params(env, ctx_server, body, data)) return nullptr;

    return dispatch_blocking_completion(env, jctx, data,
                                        SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CHAT);
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestChatCompletion(JNIEnv *env, jobject obj,
                                                                             jstring jparams) {
    REQUIRE_SERVER_CONTEXT(0);

    json body = parse_json_params(env, jparams);
    // Chat template already applied by parse_oai_chat_params; no OAI wrapping on the streaming path.
    json data;
    if (!parse_oai_chat_params(env, ctx_server, body, data)) return 0;

    return dispatch_streaming_completion(env, jctx, data,
                                         SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const std::string c_prompt = parse_jstring(env, jprompt);
    llama_tokens tokens = tokenize_mixed(jctx->vocab, c_prompt, false, true);
    return tokens_to_jint_array_impl(env, tokens, c_error_oom);
}

// Detokenise a token sequence to UTF-8, dispatching on vocab-only vs full context.
static std::string detokenize(jllama_context *jctx, const std::vector<llama_token> &tokens) {
    if (jctx->vocab_only) {
        return tokens_to_str(jctx->vocab, tokens);
    }
    return tokens_to_str(jctx->server.get_llama_context(), tokens);
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                         jintArray java_tokens) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const auto tokens = jint_array_to_tokens_impl(env, java_tokens);
    return parse_jbytes(env, detokenize(jctx, tokens));
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv *env, jobject obj) {
    auto *jctx = get_jllama_context(env, obj);
    if (!jctx) return;

    env->SetLongField(obj, f_model_pointer, 0);

    if (!jctx->vocab_only) {
        // Cancel any pending streaming readers before stopping the server.
        {
            std::lock_guard<std::mutex> lk(jctx->readers_mutex);
            jctx->readers.clear();
        }
        while (!jctx->worker_ready.load()) {
            std::this_thread::yield();
        }
        // Signal the background thread to stop. Call twice with a brief sleep
        // to close the race where the thread signalled ready but start_loop()
        // hasn't yet set its internal running flag.
        jctx->server.terminate();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        jctx->server.terminate();
        if (jctx->worker.joinable()) {
            jctx->worker.join();
        }
    }

    if (jctx->vocab_only_model) {
        llama_model_free(jctx->vocab_only_model);
    }
    delete jctx;
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task) {
    REQUIRE_SERVER_CONTEXT();
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    jctx->readers.erase(id_task);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject log_format,
                                                                 jobject jcallback) {
    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    log_json = env->IsSameObject(log_format, o_log_format_json);

    if (jcallback == nullptr) {
        log_callback = nullptr;
        llama_log_set(nullptr, nullptr);
    } else {
        o_log_callback = env->NewGlobalRef(jcallback);
        log_callback = [](enum ggml_log_level level, const char *text, void *user_data) {
            JNIEnv *env = get_jni_env();
            jstring message = env->NewStringUTF(text);
            jobject log_level = log_level_to_jobject(level);
            env->CallVoidMethod(o_log_callback, m_biconsumer_accept, log_level, message);
            env->DeleteLocalRef(message);
        };
        // Always set the trampoline — it handles JSON formatting internally
        llama_log_set(log_callback_trampoline, nullptr);
    }
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_jsonSchemaToGrammarBytes(JNIEnv *env, jclass clazz,
                                                                                      jstring j_schema) {
    const std::string c_schema = parse_jstring(env, j_schema);
    nlohmann::ordered_json c_schema_json = nlohmann::ordered_json::parse(c_schema);
    const std::string c_grammar = json_schema_to_grammar(c_schema_json);
    return parse_jbytes(env, c_grammar);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleCompletions(JNIEnv *env, jobject obj,
                                                                            jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json data = parse_json_params(env, jparams);
    return dispatch_blocking_completion(env, jctx, data,
                                        SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleCompletionsOai(JNIEnv *env, jobject obj,
                                                                               jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json body = parse_json_params(env, jparams);
    json data;
    try {
        data = oaicompat_completion_params_parse(body);
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    return dispatch_blocking_completion(env, jctx, data,
                                        SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CMPL);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleInfill(JNIEnv *env, jobject obj, jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    // Check FIM token support via server_context_meta (populated from the
    // same llama_vocab_fim_* calls inside server-context).
    auto meta = ctx_server->get_meta();
    if (meta.fim_pre_token == LLAMA_TOKEN_NULL ||
        meta.fim_sub_token == LLAMA_TOKEN_NULL ||
        meta.fim_mid_token == LLAMA_TOKEN_NULL) {
        env->ThrowNew(c_llama_error, "Model does not support fill-in-the-middle infill");
        return nullptr;
    }

    json data = parse_json_params(env, jparams);

    if (!require_json_field(env, data, "input_prefix")) return nullptr;
    if (!require_json_field(env, data, "input_suffix")) return nullptr;

    json input_extra = json_value(data, "input_extra", json::array());
    data["input_extra"] = input_extra;

    std::string prompt = json_value(data, "prompt", std::string());
    std::vector<server_tokens> tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, prompt, false, true);

    data["prompt"] = format_prompt_infill(jctx->vocab,
                                   data.at("input_prefix"), data.at("input_suffix"),
                                   data.at("input_extra"),
                                   jctx->params.n_batch, jctx->params.n_predict,
                                   meta.slot_n_ctx, jctx->params.spm_infill,
                                   tokenized_prompts.empty() ? llama_tokens()
                                                             : tokenized_prompts[0].get_tokens());

    return dispatch_blocking_completion(env, jctx, data,
                                        SERVER_TASK_TYPE_INFILL, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleEmbeddings(JNIEnv *env, jobject obj,
                                                                           jstring jparams, jboolean joaiCompat) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    if (!jctx->params.embedding) {
        env->ThrowNew(c_llama_error,
                      "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
        return nullptr;
    }

    task_response_type res_type = joaiCompat ? TASK_RESPONSE_TYPE_OAI_EMBD : TASK_RESPONSE_TYPE_NONE;

    {
        auto meta = ctx_server->get_meta();
        if (res_type != TASK_RESPONSE_TYPE_NONE && meta.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            env->ThrowNew(c_llama_error,
                          "Pooling type 'none' is not OAI compatible. Please use a different pooling type");
            return nullptr;
        }
    }

    json body = parse_json_params(env, jparams);

    bool force_no_oaicompat = false;
    json prompt;
    bool use_base64 = false;
    try {
        prompt     = extract_embedding_prompt(body, force_no_oaicompat);
        use_base64 = parse_encoding_format(body);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
    if (force_no_oaicompat) res_type = TASK_RESPONSE_TYPE_NONE;

    std::vector<server_tokens> tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, prompt, true, true);

    for (const auto &toks : tokenized_prompts) {
        if (toks.get_tokens().empty()) {
            env->ThrowNew(c_llama_error, "Input content cannot be empty");
            return nullptr;
        }
    }

    auto rd = ctx_server->get_response_reader();
    std::vector<server_task> tasks;
    tasks.reserve(tokenized_prompts.size());
    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        tasks.push_back(build_indexed_token_task(rd, SERVER_TASK_TYPE_EMBEDDING,
            server_tokens(tokenized_prompts[i].get_tokens(), false),
            static_cast<int>(i), res_type));
    }
    rd.post_tasks(std::move(tasks));

    auto br = rd.wait_for_all([] { return false; });
    if (!batch_ok_or_throw(env, br)) return nullptr;

    json responses = json::array();
    for (const auto &result : br.results) {
        responses.push_back(result->to_json());
    }
    json out = (res_type == TASK_RESPONSE_TYPE_OAI_EMBD)
        ? format_embeddings_response_oaicompat(body, json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), responses, use_base64)
        : responses;
    return json_to_jstring_impl(env, out);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleTokenize(JNIEnv *env, jobject obj, jstring jcontent,
                                                                         jboolean jaddSpecial,
                                                                         jboolean jwithPieces) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const std::string content    = parse_jstring(env, jcontent);
    const bool        add_special = jaddSpecial;
    const bool        with_pieces = jwithPieces;

    llama_tokens tokens = tokenize_mixed(jctx->vocab, content, add_special, true);

    json tokens_response = json::array();

    if (with_pieces) {
        llama_context *lctx = jctx->vocab_only ? nullptr : jctx->server.get_llama_context();
        for (const auto &token : tokens) {
            std::string piece;
            if (lctx) {
                piece = common_token_to_piece(lctx, token);
            } else {
                char buf[256];
                int n = llama_token_to_piece(jctx->vocab, token, buf, static_cast<int>(sizeof(buf)), 0, false);
                piece = n > 0 ? std::string(buf, n) : std::string();
            }
            tokens_response.push_back({{"id", token}, {"piece", token_piece_value(piece)}});
        }
    } else {
        tokens_response = tokens;
    }

    return json_to_jstring_impl(env, format_tokenizer_response(tokens_response));
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleDetokenize(JNIEnv *env, jobject obj,
                                                                           jintArray jtokens) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const auto tokens = jint_array_to_tokens_impl(env, jtokens);
    return json_to_jstring_impl(env, format_detokenized_response(detokenize(jctx, tokens)));
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleSlotAction(JNIEnv *env, jobject obj, jint action,
                                                                           jint slotId, jstring jfilename) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    switch (action) {
    case 0: // LIST — get slot info via metrics task
        return dispatch_one_shot_task(env, ctx_server, server_task(SERVER_TASK_TYPE_METRICS));
    case 1: // SAVE
        return exec_slot_file_task(env, ctx_server, slotId, jfilename,
                                    SERVER_TASK_TYPE_SLOT_SAVE,
                                    "Filename is required for slot save");
    case 2: // RESTORE
        return exec_slot_file_task(env, ctx_server, slotId, jfilename,
                                    SERVER_TASK_TYPE_SLOT_RESTORE,
                                    "Filename is required for slot restore");
    case 3: { // ERASE
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.slot_action.id_slot = slotId;
        return dispatch_one_shot_task(env, ctx_server, std::move(task));
    }
    default:
        env->ThrowNew(c_llama_error, "Invalid slot action");
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_configureParallelInference(JNIEnv *env, jobject obj,
                                                                                      jstring jconfig) {
    REQUIRE_SERVER_CONTEXT(JNI_FALSE);
    (void)obj;

    json config = parse_json_params(env, jconfig);

    std::optional<float> slot_sim_opt;
    std::optional<int>   n_threads_opt;
    std::optional<int>   n_threads_batch_opt;
    try {
        slot_sim_opt        = parse_slot_prompt_similarity(config);
        n_threads_opt       = parse_positive_int_config(config, "n_threads");
        n_threads_batch_opt = parse_positive_int_config(config, "n_threads_batch");
    } catch (const std::invalid_argument &e) {
        env->ThrowNew(c_llama_error, e.what());
        return JNI_FALSE;
    }

    // Apply n_threads / n_threads_batch via the public llama.h API.  The setter
    // requires both values; fill any missing one from the cached common_params
    // captured at load_model time so a single-field update behaves as a no-op
    // for the unspecified field.
    if (n_threads_opt.has_value() || n_threads_batch_opt.has_value()) {
        llama_context *lctx = ctx_server->get_llama_context();
        if (lctx == nullptr) {
            env->ThrowNew(c_llama_error,
                          "configureParallelInference: llama_context not available "
                          "(model sleeping or not loaded)");
            return JNI_FALSE;
        }
        const int n  = n_threads_opt.value_or(jctx->params.cpuparams.n_threads);
        const int nb = n_threads_batch_opt.value_or(jctx->params.cpuparams_batch.n_threads);
        llama_set_n_threads(lctx, n, nb);
        // Keep the cached params in sync so a follow-up call that supplies only
        // the other field reads back the value just applied, not the original.
        jctx->params.cpuparams.n_threads       = n;
        jctx->params.cpuparams_batch.n_threads = nb;
    }

    // slot_prompt_similarity: validated above (the [0.0, 1.0] range check still
    // throws for out-of-range values, preserving the existing exception
    // contract).  Live mutation requires an upstream setter that does not yet
    // exist at b8913 — upstream PR: https://github.com/ggml-org/llama.cpp/pull/22393
    // adds server_context::set_slot_prompt_similarity().  Once that lands and
    // the pinned llama.cpp version is bumped, uncomment the block below:
    //
    // if (slot_sim_opt.has_value()) {
    //     ctx_server->set_slot_prompt_similarity(*slot_sim_opt);
    // }
    (void)slot_sim_opt;

    return JNI_TRUE;
}
