#include "jllama.h"

#include "arg.h"
#include "json-schema-to-grammar.h"
#include "jni_helpers.hpp"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "server.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <thread>
#include <iostream>
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
 * Convenience wrapper: extracts and validates the server_context from the
 * Java-side model object using the module-level field-ID and error-class
 * globals.  Returns nullptr (with a JNI exception pending) when the model
 * is not loaded.
 */
static server_context *get_server_context(JNIEnv *env, jobject obj) {
    return get_server_context_impl(env, obj, f_model_pointer, c_llama_error);
}

/**
 * Convenience wrapper for the delete path only: returns the jllama_context
 * wrapper itself (not its inner .server) so the caller can call `delete jctx`.
 * Returns nullptr silently when the handle is 0 — a valid no-op for a dtor.
 * See get_jllama_context_impl in jni_helpers.hpp for the full contract.
 */
static jllama_context *get_jllama_context(JNIEnv *env, jobject obj) {
    return get_jllama_context_impl(env, obj, f_model_pointer);
}

/**
 * Formats e as a JSON invalid-request error and throws it via JNI.
 * Call inside catch(const std::exception &) blocks that must propagate
 * request-parse failures back to Java as LlamaException.
 */
static void throw_invalid_request(JNIEnv *env, const std::exception &e) {
    const auto &err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
    env->ThrowNew(c_llama_error, err.dump().c_str());
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
    env->DeleteGlobalRef(c_log_level);
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

    auto *jctx = new jllama_context();
    jctx->server = new server_context();
    jctx->vocab_only = vocab_only;
    auto *ctx_server = jctx->server;

    // Vocab-only mode: load just the tokenizer, skip inference setup.
    if (vocab_only) {
        SRV_INF("loading tokenizer from '%s'\n", params.model.path.c_str());
        if (!ctx_server->load_tokenizer(params)) {
            delete ctx_server;
            delete jctx;
            llama_backend_free();
            env->ThrowNew(c_llama_error, "could not load tokenizer from given file path");
            return;
        }
        env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(jctx));
        return;
    }

    SRV_INF("loading model '%s'\n", params.model.path.c_str());

    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads,
            params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // Necessary similarity of prompt for slot selection
    ctx_server->slot_prompt_similarity = params.slot_prompt_similarity;

    // Resolve the auto sentinel before loading the model.
    if (params.n_parallel <= N_PARALLEL_AUTO) {
        params.n_parallel = N_PARALLEL_DEFAULT;
    }

    LOG_INF("%s: loading model\n", __func__);

    // load the model
    if (!ctx_server->load_model(params)) {
        delete ctx_server;
        delete jctx;
        llama_backend_free();
        env->ThrowNew(c_llama_error, "could not load model from given file path");
        return;
    }

    ctx_server->init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    const auto model_meta = ctx_server->model_meta();

    // print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
            common_chat_templates_source(ctx_server->chat_templates.get()).c_str(),
            common_chat_format_example(ctx_server->chat_templates.get(), ctx_server->params_base.use_jinja, ctx_server->params_base.default_template_kwargs).c_str());

    // print sample chat example to make it clear which template is used
    //    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
    //         common_chat_templates_source(ctx_server->chat_templates.get()),
    //        common_chat_format_example(*ctx_server->chat_templates.template_default,
    //        ctx_server->params_base.use_jinja) .c_str());

    ctx_server->queue_tasks.on_new_task(
        std::bind(&server_context::process_single_task, ctx_server, std::placeholders::_1));
    ctx_server->queue_tasks.on_update_slots(std::bind(&server_context::update_slots, ctx_server));

    jctx->worker = std::thread([jctx, ctx_server]() {
        JNIEnv *env;
        jint res = g_vm->GetEnv((void **)&env, JNI_VERSION_1_6);
        bool attached = false;
        if (res == JNI_EDETACHED) {
            res = g_vm->AttachCurrentThread((void **)&env, nullptr);
            if (res != JNI_OK) {
                jctx->worker_ready.store(true); // Signal even on failure so close() doesn't hang
                return;
            }
            attached = true;
        }
        // Signal that we're about to enter start_loop(). This must happen
        // after AttachCurrentThread but before start_loop() sets running=true,
        // so that close() can safely call terminate() knowing the thread is ready.
        jctx->worker_ready.store(true);
        ctx_server->queue_tasks.start_loop();
        // Detach from JVM before thread exits to prevent writing to closed pipes
        if (attached) {
            g_vm->DetachCurrentThread();
        }
    });

    // Wait for the worker thread to be ready before returning. This prevents
    // a race where close() calls terminate() before start_loop() has set
    // running=true, which would cause start_loop() to override the terminate
    // and result in a deadlock on join().
    while (!jctx->worker_ready.load()) {
        std::this_thread::yield();
    }

    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(jctx));
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj, jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return 0;

    std::string c_params = parse_jstring(env, jparams);
    json data = json::parse(c_params);

    server_task_type type = SERVER_TASK_TYPE_COMPLETION;

    if (data.contains("input_prefix") || data.contains("input_suffix")) {
        type = SERVER_TASK_TYPE_INFILL;
    }

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        const auto &prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(type);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = OAICOMPAT_TYPE_NONE;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return 0;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);

    ctx_server->queue_tasks.post(std::move(tasks));

    if (task_ids.size() != 1) {
        env->ThrowNew(c_llama_error, "multitasking currently not supported");
        return 0;
    }

    return *task_ids.begin();
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask(JNIEnv *env, jobject obj, jint id_task) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return;
    ctx_server->queue_results.remove_waiting_task_id(id_task);
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletionJson(JNIEnv *env, jobject obj,
                                                                               jint id_task) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    server_task_result_ptr result = ctx_server->queue_results.recv(id_task);

    if (result->is_error()) {
        std::string response = result->to_json()["message"].get<std::string>();
        ctx_server->queue_results.remove_waiting_task_id(id_task);
        env->ThrowNew(c_llama_error, response.c_str());
        return nullptr;
    }

    json response = result->to_json();
    response["stop"] = result->is_stop();

    if (result->is_stop()) {
        ctx_server->queue_results.remove_waiting_task_id(id_task);
    }

    std::string response_str = response.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    if (!ctx_server->params_base.embedding) {
        env->ThrowNew(c_llama_error,
                      "model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
        return nullptr;
    }

    const std::string prompt = parse_jstring(env, jprompt);

    SRV_INF("Calling embedding '%s'\n", prompt.c_str());

    auto tokens = tokenize_mixed(ctx_server->vocab, prompt, true, true);
    std::vector<server_task> tasks;

    server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

    task.id = ctx_server->queue_tasks.get_new_id();
    task.index = 0;
    task.prompt_tokens = server_tokens(tokens, false);

    // OAI-compat
    task.params.oaicompat = OAICOMPAT_TYPE_NONE;

    tasks.push_back(std::move(task));

    ctx_server->queue_results.add_waiting_tasks(tasks);
    std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

    ctx_server->queue_tasks.post(std::move(tasks));
    const auto id_task = *task_ids.begin();
    json responses = json::array();

    json error = nullptr;

    server_task_result_ptr result = ctx_server->queue_results.recv(id_task);

    json response_str = result->to_json();
    if (result->is_error()) {
        std::string response = result->to_json()["message"].get<std::string>();
        ctx_server->queue_results.remove_waiting_task_id(id_task);
        env->ThrowNew(c_llama_error, response.c_str());
        return nullptr;
    }

    if (result->is_stop()) {
        ctx_server->queue_results.remove_waiting_task_id(id_task);
    }

    const auto out_res = result->to_json();

    // Extract "embedding" as a vector of vectors (2D array)
    std::vector<std::vector<float>> embedding = out_res["embedding"].get<std::vector<std::vector<float>>>();

    // Get total number of rows in the embedding
    jsize embedding_rows = embedding.size();

    // Get total number of columns in the first row (assuming all rows are of equal length)
    jsize embedding_cols = embedding_rows > 0 ? embedding[0].size() : 0;

    SRV_INF("Embedding has %d rows and %d columns\n", embedding_rows, embedding_cols);

    // Ensure embedding is not empty
    if (embedding.empty() || embedding[0].empty()) {
        env->ThrowNew(c_error_oom, "embedding array is empty");
        return nullptr;
    }

    // Extract only the first row
    const std::vector<float> &first_row = embedding[0]; // Reference to avoid copying

    // Create a new float array in JNI
    jfloatArray j_embedding = env->NewFloatArray(embedding_cols);
    if (j_embedding == nullptr) {
        env->ThrowNew(c_error_oom, "could not allocate embedding");
        return nullptr;
    }

    // Copy the first row into the JNI float array
    env->SetFloatArrayRegion(j_embedding, 0, embedding_cols, reinterpret_cast<const jfloat *>(first_row.data()));

    return j_embedding;
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleRerank(JNIEnv *env, jobject obj, jstring jprompt,
                                                                      jobjectArray documents) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    if (!ctx_server->params_base.embedding || ctx_server->params_base.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        env->ThrowNew(c_llama_error,
                      "This server does not support reranking. Start it with `--reranking` and without `--embedding`");
        return nullptr;
    }

    const std::string prompt = parse_jstring(env, jprompt);

    const auto tokenized_query = tokenize_mixed(ctx_server->vocab, prompt, true, true);

    std::vector<server_task> tasks;
    const jsize amount_documents = env->GetArrayLength(documents);
    auto *document_array = parse_string_array(env, documents, amount_documents);
    auto document_vector = std::vector<std::string>(document_array, document_array + amount_documents);
    free_string_array(document_array, amount_documents);

    std::vector<llama_tokens> tokenized_docs = tokenize_input_prompts(ctx_server->vocab, document_vector, true, true);

    tasks.reserve(tokenized_docs.size());
    for (size_t i = 0; i < tokenized_docs.size(); i++) {
        auto task = server_task(SERVER_TASK_TYPE_RERANK);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.index = i;
        auto tokens = format_rerank(ctx_server->vocab, tokenized_query, tokenized_docs[i]);
        task.prompt_tokens = server_tokens(tokens, false);
        tasks.push_back(std::move(task));
    }
    ctx_server->queue_results.add_waiting_tasks(tasks);
    std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

    ctx_server->queue_tasks.post(std::move(tasks));

    json results_json = json::array();

    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);
        if (result->is_error()) {
            auto response = result->to_json()["message"].get<std::string>();
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            env->ThrowNew(c_llama_error, response.c_str());
            return nullptr;
        }

        const auto out_res = result->to_json();
        int index = out_res["index"].get<int>();
        float score = out_res["score"].get<float>();

        results_json.push_back({
            {"document", document_vector[index]},
            {"index", index},
            {"score", score}
        });
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    std::string response_str = results_json.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate(JNIEnv *env, jobject obj, jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    std::string c_params = parse_jstring(env, jparams);
    json data = json::parse(c_params);

    std::vector<raw_buffer> files;
    json templateData = oaicompat_chat_params_parse(data, ctx_server->oai_parser_opt, files);

    std::string tok_str = templateData.at("prompt");
    jstring jtok_str = env->NewStringUTF(tok_str.c_str());

    return jtok_str;
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleChatCompletions(JNIEnv *env, jobject obj,
                                                                                jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    std::string c_params = parse_jstring(env, jparams);
    json body = json::parse(c_params);

    // Apply chat template via OAI-compatible parser
    json data;
    try {
        std::vector<raw_buffer> files;
        data = oaicompat_chat_params_parse(body, ctx_server->oai_parser_opt, files);
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        const auto &prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat = OAICOMPAT_TYPE_CHAT;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    // Collect all results (blocking)
    std::vector<server_task_result_ptr> results;
    results.reserve(task_ids.size());

    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

        if (result->is_error()) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        results.push_back(std::move(result));
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    // Build response JSON
    json response;
    if (results.size() == 1) {
        response = results[0]->to_json();
    } else {
        response = json::array();
        for (auto &res : results) {
            response.push_back(res->to_json());
        }
    }

    std::string response_str = response.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestChatCompletion(JNIEnv *env, jobject obj,
                                                                             jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return 0;

    std::string c_params = parse_jstring(env, jparams);
    json body = json::parse(c_params);

    // Apply chat template via OAI-compatible parser
    json data;
    try {
        std::vector<raw_buffer> files;
        data = oaicompat_chat_params_parse(body, ctx_server->oai_parser_opt, files);
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return 0;
    }

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        const auto &prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // Use NONE so receiveCompletion gets the simple {"content":"..."} format.
            // The chat template was already applied by oaicompat_chat_params_parse above.
            task.params.oaicompat = OAICOMPAT_TYPE_NONE;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return 0;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    if (task_ids.size() != 1) {
        env->ThrowNew(c_llama_error, "multitasking currently not supported");
        return 0;
    }

    return *task_ids.begin();
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    const std::string c_prompt = parse_jstring(env, jprompt);

    llama_tokens tokens = tokenize_mixed(ctx_server->vocab, c_prompt, false, true);
    jsize token_size = tokens.size(); // NOLINT(*-narrowing-conversions)

    jintArray java_tokens = env->NewIntArray(token_size);
    if (java_tokens == nullptr) {
        env->ThrowNew(c_error_oom, "could not allocate token memory");
        return nullptr;
    }

    env->SetIntArrayRegion(java_tokens, 0, token_size, reinterpret_cast<const jint *>(tokens.data()));

    return java_tokens;
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                         jintArray java_tokens) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    jsize length = env->GetArrayLength(java_tokens);
    jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
    std::vector<llama_token> tokens(elements, elements + length);

    std::string text;
    if (!ctx_server->is_vocab_only()) {
        text = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());
    } else {
        // vocab-only mode: detokenize using vocabulary directly
        text = tokens_to_str(ctx_server->vocab, tokens.cbegin(), tokens.cend());
    }

    env->ReleaseIntArrayElements(java_tokens, elements, 0);

    return parse_jbytes(env, text);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv *env, jobject obj) {
    auto *jctx = get_jllama_context(env, obj);
    if (!jctx) return; // Already deleted or never initialized

    // Clear the pointer first to prevent double-free from concurrent calls
    env->SetLongField(obj, f_model_pointer, 0);

    if (!jctx->vocab_only) {
        // Wait for the worker thread to be ready (entered start_loop).
        while (!jctx->worker_ready.load()) {
            std::this_thread::yield();
        }
        // Signal the background thread to stop. We call terminate() twice with
        // a brief sleep in between to close the race window where the thread
        // signalled ready but start_loop() hasn't yet set running=true.
        jctx->server->queue_tasks.terminate();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        jctx->server->queue_tasks.terminate();
        if (jctx->worker.joinable()) {
            jctx->worker.join();
        }
    }

    delete jctx->server;
    delete jctx;
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return;
    std::unordered_set<int> id_tasks = {id_task};
    ctx_server->cancel_tasks(id_tasks);
    ctx_server->queue_results.remove_waiting_task_id(id_task);
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
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    std::string c_params = parse_jstring(env, jparams);
    json data = json::parse(c_params);

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        const auto &prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat = OAICOMPAT_TYPE_NONE;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    // Collect all results (blocking)
    std::vector<server_task_result_ptr> results;
    results.reserve(task_ids.size());

    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

        if (result->is_error()) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        results.push_back(std::move(result));
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    json response;
    if (results.size() == 1) {
        response = results[0]->to_json();
    } else {
        response = json::array();
        for (auto &res : results) {
            response.push_back(res->to_json());
        }
    }

    std::string response_str = response.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleCompletionsOai(JNIEnv *env, jobject obj,
                                                                               jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    std::string c_params = parse_jstring(env, jparams);
    json body = json::parse(c_params);

    // Parse OAI-compatible completion parameters
    json data = oaicompat_completion_params_parse(body);

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        const auto &prompt = data.at("prompt");

        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat = OAICOMPAT_TYPE_COMPLETION;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    std::vector<server_task_result_ptr> results;
    results.reserve(task_ids.size());

    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

        if (result->is_error()) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        results.push_back(std::move(result));
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    json response;
    if (results.size() == 1) {
        response = results[0]->to_json();
    } else {
        response = json::array();
        for (auto &res : results) {
            response.push_back(res->to_json());
        }
    }

    std::string response_str = response.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleInfill(JNIEnv *env, jobject obj, jstring jparams) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    // Check model compatibility for infill
    std::string err;
    if (llama_vocab_fim_pre(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
        err += "prefix token is missing. ";
    }
    if (llama_vocab_fim_suf(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
        err += "suffix token is missing. ";
    }
    if (llama_vocab_fim_mid(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
        err += "middle token is missing. ";
    }
    if (!err.empty()) {
        env->ThrowNew(c_llama_error, ("Infill is not supported by this model: " + err).c_str());
        return nullptr;
    }

    std::string c_params = parse_jstring(env, jparams);
    json data = json::parse(c_params);

    if (!data.contains("input_prefix")) {
        env->ThrowNew(c_llama_error, "\"input_prefix\" is required");
        return nullptr;
    }
    if (!data.contains("input_suffix")) {
        env->ThrowNew(c_llama_error, "\"input_suffix\" is required");
        return nullptr;
    }

    json input_extra = json_value(data, "input_extra", json::array());
    data["input_extra"] = input_extra;

    // Format the infill prompt
    std::string prompt = json_value(data, "prompt", std::string());
    std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, false, true);

    data["prompt"] = format_infill(ctx_server->vocab, data.at("input_prefix"), data.at("input_suffix"),
                                   data.at("input_extra"), ctx_server->params_base.n_batch,
                                   ctx_server->params_base.n_predict, ctx_server->slots[0].n_ctx,
                                   ctx_server->params_base.spm_infill,
                                   tokenized_prompts.empty() ? llama_tokens() : tokenized_prompts[0]);

    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    try {
        std::vector<llama_tokens> infill_prompts =
            tokenize_input_prompts(ctx_server->vocab, data.at("prompt"), true, true);

        tasks.reserve(infill_prompts.size());
        for (size_t i = 0; i < infill_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_INFILL);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = server_tokens(infill_prompts[i], false);
            task.params = server_task::params_from_json_cmpl(ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat = OAICOMPAT_TYPE_NONE;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    std::vector<server_task_result_ptr> results;
    results.reserve(task_ids.size());

    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

        if (result->is_error()) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        results.push_back(std::move(result));
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    json response;
    if (results.size() == 1) {
        response = results[0]->to_json();
    } else {
        response = json::array();
        for (auto &res : results) {
            response.push_back(res->to_json());
        }
    }

    std::string response_str = response.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleEmbeddings(JNIEnv *env, jobject obj,
                                                                           jstring jparams, jboolean joaiCompat) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    if (!ctx_server->params_base.embedding) {
        env->ThrowNew(c_llama_error,
                      "Model was not loaded with embedding support (see ModelParameters#enableEmbedding())");
        return nullptr;
    }

    oaicompat_type oaicompat = joaiCompat ? OAICOMPAT_TYPE_EMBEDDING : OAICOMPAT_TYPE_NONE;

    if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server->ctx) == LLAMA_POOLING_TYPE_NONE) {
        env->ThrowNew(c_llama_error,
                      "Pooling type 'none' is not OAI compatible. Please use a different pooling type");
        return nullptr;
    }

    std::string c_params = parse_jstring(env, jparams);
    json body = json::parse(c_params);

    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        oaicompat = OAICOMPAT_TYPE_NONE;
        prompt = body.at("content");
    } else {
        env->ThrowNew(c_llama_error, "\"input\" or \"content\" must be provided");
        return nullptr;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string &format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            env->ThrowNew(c_llama_error, "encoding_format must be \"float\" or \"base64\"");
            return nullptr;
        }
    }

    std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

    for (const auto &tokens : tokenized_prompts) {
        if (tokens.empty()) {
            env->ThrowNew(c_llama_error, "Input content cannot be empty");
            return nullptr;
        }
    }

    std::vector<server_task> tasks;
    tasks.reserve(tokenized_prompts.size());

    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

        task.id = ctx_server->queue_tasks.get_new_id();
        task.index = i;
        task.prompt_tokens = server_tokens(tokenized_prompts[i], false);
        task.params.oaicompat = oaicompat;

        tasks.push_back(std::move(task));
    }

    ctx_server->queue_results.add_waiting_tasks(tasks);
    std::unordered_set<int> task_ids = server_task::get_list_id(tasks);
    ctx_server->queue_tasks.post(std::move(tasks));

    json responses = json::array();

    for (size_t i = 0; i < tasks.size(); i++) {
        server_task_result_ptr result = ctx_server->queue_results.recv(task_ids);

        if (result->is_error()) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        responses.push_back(result->to_json());
    }

    ctx_server->queue_results.remove_waiting_task_ids(task_ids);

    json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
                    ? format_embeddings_response_oaicompat(body, responses, use_base64)
                    : json(responses);

    std::string response_str = root.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleTokenize(JNIEnv *env, jobject obj, jstring jcontent,
                                                                         jboolean jaddSpecial,
                                                                         jboolean jwithPieces) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    const std::string content = parse_jstring(env, jcontent);
    const bool add_special = jaddSpecial;
    const bool with_pieces = jwithPieces;

    llama_tokens tokens = tokenize_mixed(ctx_server->vocab, content, add_special, true);

    json tokens_response = json::array();

    if (with_pieces) {
        for (const auto &token : tokens) {
            std::string piece = common_token_to_piece(ctx_server->ctx, token);
            json piece_json;

            if (is_valid_utf8(piece)) {
                piece_json = piece;
            } else {
                piece_json = json::array();
                for (unsigned char c : piece) {
                    piece_json.push_back(static_cast<int>(c));
                }
            }

            tokens_response.push_back({{"id", token}, {"piece", piece_json}});
        }
    } else {
        tokens_response = tokens;
    }

    json data = format_tokenizer_response(tokens_response);

    std::string response_str = data.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleDetokenize(JNIEnv *env, jobject obj,
                                                                           jintArray jtokens) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    jsize length = env->GetArrayLength(jtokens);
    jint *elements = env->GetIntArrayElements(jtokens, nullptr);
    std::vector<llama_token> tokens(elements, elements + length);
    env->ReleaseIntArrayElements(jtokens, elements, JNI_ABORT);

    std::string content;
    if (!ctx_server->is_vocab_only()) {
        content = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());
    } else {
        content = tokens_to_str(ctx_server->vocab, tokens.cbegin(), tokens.cend());
    }

    json data = format_detokenized_response(content);

    std::string response_str = data.dump();
    return env->NewStringUTF(response_str.c_str());
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_handleSlotAction(JNIEnv *env, jobject obj, jint action,
                                                                           jint slotId, jstring jfilename) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return nullptr;

    switch (action) {
    case 0: { // LIST — get slot info via metrics
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = ctx_server->queue_tasks.get_new_id();
        ctx_server->queue_results.add_waiting_task_id(task.id);
        int id = task.id;
        ctx_server->queue_tasks.post(std::move(task), true);

        server_task_result_ptr result = ctx_server->queue_results.recv(id);
        ctx_server->queue_results.remove_waiting_task_id(id);

        if (result->is_error()) {
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        std::string resp = result->to_json().dump();
        return env->NewStringUTF(resp.c_str());
    }
    case 1: { // SAVE
        std::string filename = jfilename != nullptr ? parse_jstring(env, jfilename) : "";
        if (filename.empty()) {
            env->ThrowNew(c_llama_error, "Filename is required for slot save");
            return nullptr;
        }

        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.slot_action.id_slot = slotId;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filename;

        int tid = task.id;
        ctx_server->queue_results.add_waiting_task_id(tid);
        ctx_server->queue_tasks.post(std::move(task));

        server_task_result_ptr result = ctx_server->queue_results.recv(tid);
        ctx_server->queue_results.remove_waiting_task_id(tid);

        if (result->is_error()) {
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        std::string resp = result->to_json().dump();
        return env->NewStringUTF(resp.c_str());
    }
    case 2: { // RESTORE
        std::string filename = jfilename != nullptr ? parse_jstring(env, jfilename) : "";
        if (filename.empty()) {
            env->ThrowNew(c_llama_error, "Filename is required for slot restore");
            return nullptr;
        }

        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.slot_action.id_slot = slotId;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filename;

        int tid = task.id;
        ctx_server->queue_results.add_waiting_task_id(tid);
        ctx_server->queue_tasks.post(std::move(task));

        server_task_result_ptr result = ctx_server->queue_results.recv(tid);
        ctx_server->queue_results.remove_waiting_task_id(tid);

        if (result->is_error()) {
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        std::string resp = result->to_json().dump();
        return env->NewStringUTF(resp.c_str());
    }
    case 3: { // ERASE
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.slot_action.id_slot = slotId;

        int tid = task.id;
        ctx_server->queue_results.add_waiting_task_id(tid);
        ctx_server->queue_tasks.post(std::move(task));

        server_task_result_ptr result = ctx_server->queue_results.recv(tid);
        ctx_server->queue_results.remove_waiting_task_id(tid);

        if (result->is_error()) {
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(c_llama_error, error_msg.c_str());
            return nullptr;
        }

        std::string resp = result->to_json().dump();
        return env->NewStringUTF(resp.c_str());
    }
    default:
        env->ThrowNew(c_llama_error, "Invalid slot action");
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_de_kherud_llama_LlamaModel_configureParallelInference(JNIEnv *env, jobject obj,
                                                                                      jstring jconfig) {
    auto *ctx_server = get_server_context(env, obj);
    if (!ctx_server) return JNI_FALSE;

    std::string config_str = parse_jstring(env, jconfig);
    json config = json::parse(config_str);

    if (config.contains("slot_prompt_similarity")) {
        float similarity = config["slot_prompt_similarity"].get<float>();
        if (similarity < 0.0f || similarity > 1.0f) {
            env->ThrowNew(c_llama_error, "slot_prompt_similarity must be between 0.0 and 1.0");
            return JNI_FALSE;
        }
        ctx_server->slot_prompt_similarity = similarity;
    }

    if (config.contains("n_threads")) {
        int n_threads = config["n_threads"].get<int>();
        if (n_threads <= 0) {
            env->ThrowNew(c_llama_error, "n_threads must be greater than 0");
            return JNI_FALSE;
        }
        ctx_server->params_base.cpuparams.n_threads = n_threads;
    }

    if (config.contains("n_threads_batch")) {
        int n_threads_batch = config["n_threads_batch"].get<int>();
        if (n_threads_batch <= 0) {
            env->ThrowNew(c_llama_error, "n_threads_batch must be greater than 0");
            return JNI_FALSE;
        }
        ctx_server->params_base.cpuparams_batch.n_threads = n_threads_batch;
    }

    return JNI_TRUE;
}
