// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

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
#include "server-schema.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"
#include "log_helpers.hpp"
#include "tts_engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

// Guards the Java -> native jllama_context handle so acquire_jllama_context_impl (every entry
// point) and delete() (close) cannot race on the same field. Declared extern in jni_helpers.hpp.
// Defined here at file scope (external linkage) so the header's extern reference resolves.
std::mutex g_ctx_mutex;

// We store some references to Java classes and their fields/methods here to speed up things for later and to fail
// early on if anything can't be found. This happens when the JVM loads the shared library (see `JNI_OnLoad`).
// The references remain valid throughout the whole life of the shared library, on `JNI_OnUnload` they are released.

// Forward-declared here (defined with external linkage below, after the anonymous namespace
// closes) so the in-namespace call sites below can see it before its definition point, and so
// native_server.cpp — which forward-declares it the same way — can link against it.
std::string parse_jstring(JNIEnv *env, jstring java_string);

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
// String(byte[], String charsetName) — the standard-UTF-8-safe way to build a
// Java String from native bytes (NewStringUTF expects Modified UTF-8 and is
// spec-invalid for supplementary-plane characters such as 4-byte emoji).
jmethodID cc_string_bytes_charset = nullptr;

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

// ---------------------------------------------------------------------------
// JNI global-ref lifecycle tables
//
// Every entry here is promoted to a JNI global ref in JNI_OnLoad and released
// in JNI_OnUnload. Keep these in sync with the declarations above; ordering
// within each table only matters for the human reader.
// ---------------------------------------------------------------------------
static jclass *const g_global_class_refs[] = {
    &c_llama_model, &c_string, &c_hash_map,   &c_map,         &c_set,       &c_entry,      &c_iterator,
    &c_integer,     &c_float,  &c_biconsumer, &c_llama_error, &c_log_level, &c_log_format, &c_error_oom,
};

static jobject *const g_global_object_refs[] = {
    &o_utf_8,           &o_log_level_debug, &o_log_level_info,  &o_log_level_warn,
    &o_log_level_error, &o_log_format_json, &o_log_format_text,
};

// Maps every object that is fetched from a Java static field on load to the
// (class, field) pair it should be looked up from.
struct static_object_binding {
    jobject *target;
    jclass *cls;
    jfieldID *field;
};

static const static_object_binding g_static_object_bindings[] = {
    {&o_log_level_debug, &c_log_level, &f_log_level_debug},  {&o_log_level_info, &c_log_level, &f_log_level_info},
    {&o_log_level_warn, &c_log_level, &f_log_level_warn},    {&o_log_level_error, &c_log_level, &f_log_level_error},
    {&o_log_format_json, &c_log_format, &f_log_format_json}, {&o_log_format_text, &c_log_format, &f_log_format_text},
};

/**
 * Returns the jllama_context wrapper for the Java LlamaModel object.
 * Used by the delete path and any method that needs jctx directly.
 * Returns nullptr silently on a null handle (valid no-op for a destructor).
 */
[[nodiscard]] static jllama_context *get_jllama_context(JNIEnv *env, jobject obj) {
    return get_jllama_context_impl(env, obj, f_model_pointer);
}

/**
 * Builds a Java String from raw standard-UTF-8 bytes via the cached
 * String(byte[], "UTF-8") constructor (see utf8_to_jstring_impl for why
 * NewStringUTF must not be used for payload text).
 */
[[nodiscard]] static jstring utf8_to_jstring(JNIEnv *env, const std::string &s) {
    return utf8_to_jstring_impl(env, s, c_string, cc_string_bytes_charset, o_utf_8);
}

/**
 * Serialises a json value to a Java String (UTF-8-safe on both the dump and
 * the JNI crossing; see json_to_jstring_impl).
 */
[[nodiscard]] static jstring json_to_jstring(JNIEnv *env, const json &j) {
    return json_to_jstring_impl(env, j, c_string, cc_string_bytes_charset, o_utf_8);
}

/**
 * Serialises a vector of task results to a Java String (see
 * results_to_jstring_impl).
 */
[[nodiscard]] static jstring results_to_jstring(JNIEnv *env, const std::vector<server_task_result_ptr> &results) {
    return results_to_jstring_impl(env, results, c_string, cc_string_bytes_charset, o_utf_8);
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
[[nodiscard]] static bool result_ok_or_throw(JNIEnv *env, const server_task_result_ptr &result) {
    if (!result || result->is_error()) {
        env->ThrowNew(c_llama_error, result ? get_result_error_message(result).c_str() : "No result");
        return false;
    }
    return true;
}

/**
 * Returns true if the batch completed without a task-level error.
 * On failure throws via JNI and returns false.  Callers must return immediately.
 */
[[nodiscard]] static bool batch_ok_or_throw(JNIEnv *env, const server_response_reader::batch_response &br) {
    if (br.error) {
        env->ThrowNew(c_llama_error, get_result_error_message(br.error).c_str());
        return false;
    }
    if (br.is_terminated) {
        // wait_for_all was cut short because close() set jctx->closing: the results vector is
        // only partially filled (null entries), so don't dereference it - unwind cleanly instead.
        env->ThrowNew(c_llama_error, "inference interrupted by close()");
        return false;
    }
    return true;
}

/**
 * Parse the OAI chat-completion body through oaicompat_chat_params_parse and
 * write the result into `out`, preserving decoded media in `files`. Returns
 * true on success; on failure throws and returns false.
 */
[[nodiscard]] static bool parse_oai_chat_params(JNIEnv *env, server_context *ctx_server, json &body, json &out,
                                                std::vector<raw_buffer> &files) {
    try {
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
static void populate_completion_task(server_task &task, jllama_context *jctx,
                                     const std::vector<llama_logit_bias> &logit_bias_eog, const json &data,
                                     bool has_mtmd, std::vector<raw_buffer> files = {}) {
    if (!configure_multimodal_task_impl(task, has_mtmd, data, std::move(files))) {
        auto tokenized_prompts = tokenize_input_prompts(jctx->vocab, nullptr, data.at("prompt"), true, true);
        if (!tokenized_prompts.empty()) {
            task.tokens = std::move(tokenized_prompts[0]);
        }
    }
    task.params = server_schema::eval_llama_cmpl_schema(jctx->vocab, jctx->params, logit_bias_eog, data);
    configure_task_slot_impl(task, data);
}

[[nodiscard]] static jint dispatch_streaming_completion(JNIEnv *env, jllama_context *jctx, const json &data,
                                                        server_task_type task_type, task_response_type res_type,
                                                        std::vector<raw_buffer> files = {}) {
    server_context *ctx_server = &jctx->server;
    auto meta = ctx_server->get_meta();
    auto *rd = new server_response_reader(ctx_server->get_response_reader());
    int tid = rd->get_new_id();
    try {
        server_task task(task_type);
        task.id = tid;
        populate_completion_task(task, jctx, meta.logit_bias_eog, data, meta.has_mtmd, std::move(files));
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
[[nodiscard]] static jstring dispatch_blocking_completion(JNIEnv *env, jllama_context *jctx, const json &data,
                                                          server_task_type task_type, task_response_type res_type,
                                                          std::vector<raw_buffer> files = {}) {
    server_context *ctx_server = &jctx->server;
    auto meta = ctx_server->get_meta();
    auto rd = ctx_server->get_response_reader();
    server_task task(task_type);
    task.id = rd.get_new_id();
    try {
        populate_completion_task(task, jctx, meta.logit_bias_eog, data, meta.has_mtmd, std::move(files));
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }
    task.params.res_type = res_type;
    rd.post_task(std::move(task));
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    if (!batch_ok_or_throw(env, br))
        return nullptr;
    return results_to_jstring(env, br.results);
}

/**
 * Convert a Java string to a parsed JSON object.
 * Combines parse_jstring + json::parse, which every parameter-taking JNI
 * function needs before it can read its arguments.
 */
// Parse the JSON request body. On malformed input, converts the C++ parse error into a Java
// LlamaException (so it never crosses the JNI boundary — which is undefined behavior) and returns
// false; callers must return their sentinel when this returns false.
[[nodiscard]] static bool parse_json_params(JNIEnv *env, jstring jparams, json &out) {
    try {
        out = json::parse(parse_jstring(env, jparams));
        return true;
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return false;
    }
}

/**
 * Convenience wrapper around require_json_field_impl (jni_helpers.hpp).
 * Returns false and throws if `field` is absent from `data`.
 */
[[nodiscard]] static bool require_json_field(JNIEnv *env, const json &data, const char *field) {
    return require_json_field_impl(env, data, field, c_llama_error);
}

// Build a single indexed token task for batch submission (rerank and embedding).
// Assigns the reader-allocated id; moves tokens into the task.
[[nodiscard]] static server_task build_indexed_token_task(server_response_reader &rd, server_task_type type,
                                                          server_tokens &&tokens, int index,
                                                          task_response_type res_type) {
    server_task task(type);
    task.id = rd.get_new_id();
    task.tokens = std::move(tokens);
    task.index = index;
    task.params.res_type = res_type;
    return task;
}

// Post a single pre-built task, wait for its result, and return JSON as a jstring.
// The task's id field is assigned here; callers must not set it beforehand.
[[nodiscard]] static jstring dispatch_one_shot_task(JNIEnv *env, server_context *ctx_server, server_task task) {
    auto rd = ctx_server->get_response_reader();
    task.id = rd.get_new_id();
    rd.post_task(std::move(task));
    auto result = rd.next([] { return false; });
    if (!result_ok_or_throw(env, result))
        return nullptr;
    return json_to_jstring(env, result->to_json());
}

// Post a single slot file task (SAVE or RESTORE), wait for its result, and
// return the result JSON as a jstring.
[[nodiscard]] static jstring exec_slot_file_task(JNIEnv *env, server_context *ctx_server, jint slotId,
                                                 jstring jfilename, server_task_type task_type,
                                                 const char *empty_filename_error) {
    const std::string filename = jfilename != nullptr ? parse_jstring(env, jfilename) : "";
    if (filename.empty()) {
        env->ThrowNew(c_llama_error, empty_filename_error);
        return nullptr;
    }
    server_task task(task_type);
    task.slot_action.id_slot = slotId;
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
        // Default to a null slot so an OOM/early-exit path below (e.g. GetByteArrayElements
        // returning nullptr) cannot leave result[i] as uninitialized malloc garbage that
        // free_string_array would later free().
        result[i] = nullptr;
        auto *const javaString = static_cast<jstring>(env->GetObjectArrayElement(string_array, i));
        // A null array element (legal from a Java String[]) must not reach the getBytes call.
        if (javaString == nullptr) {
            result[i] = strdup("");
            continue;
        }
        // Use the standard-UTF-8 byte path (String.getBytes("UTF-8")), NOT GetStringUTFChars
        // which yields Modified-UTF8 (CESU-8) and would mangle non-BMP chars (emoji, many CJK
        // extensions) in model paths and rerank documents.
        jbyteArray bytes = (jbyteArray)env->CallObjectMethod(javaString, m_get_bytes, o_utf_8);
        // Clear any pending exception from the call (e.g. OOM) before making further JNI calls
        // in this loop — the JNI spec forbids most calls while an exception is pending. The
        // element degrades to "" like the other failure paths.
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            if (bytes != nullptr) {
                env->DeleteLocalRef(bytes);
                bytes = nullptr;
            }
        }
        if (bytes == nullptr) {
            result[i] = strdup("");
        } else {
            const jsize blen = env->GetArrayLength(bytes);
            jbyte *bytesElems = env->GetByteArrayElements(bytes, nullptr);
            if (bytesElems != nullptr) {
                result[i] = static_cast<char *>(malloc(static_cast<size_t>(blen) + 1));
                if (result[i] != nullptr) {
                    memcpy(result[i], bytesElems, static_cast<size_t>(blen));
                    result[i][blen] = '\0';
                }
                env->ReleaseByteArrayElements(bytes, bytesElems, JNI_ABORT);
            }
            if (result[i] == nullptr) {
                result[i] = strdup("");
            }
        }
        // Each GetObjectArrayElement returns a fresh local ref; release it so a large
        // array (e.g. a rerank with many documents) cannot overflow the local-ref table.
        env->DeleteLocalRef(javaString);
        if (bytes != nullptr) {
            env->DeleteLocalRef(bytes);
        }
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

/**
 * Like get_jni_env() but returns nullptr instead of throwing when the calling
 * thread is not attached to the JVM. Used from the log callback, which can fire
 * on internal ggml/llama worker threads that were never AttachCurrentThread-ed —
 * throwing there would unwind through C call frames (undefined behavior).
 */
JNIEnv *get_jni_env_or_null() noexcept {
    JNIEnv *env = nullptr;
    if (g_vm == nullptr || g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return nullptr;
    }
    return env;
}

bool log_json;
std::function<void(ggml_log_level, const char *, void *)> log_callback;
// Guards the logger globals so concurrent setLogger calls (and the trampoline
// reading them from arbitrary worker threads) cannot race.
static std::mutex g_log_mutex;
// Number of trampolines currently invoking a copied callback OUTSIDE the lock.
// setLogger drains this to 0 before DeleteGlobalRef-ing the old callback ref —
// without the drain, a trampoline that copied the old std::function could still
// call into a just-deleted global ref (a use-after-free on the JNI ref).
static int g_log_active = 0;
static std::condition_variable g_log_cv;

/**
 * Invoke the log callback if there is any. When JSON mode is enabled,
 * the message is formatted as a JSON object before forwarding.
 */
void log_callback_trampoline(ggml_log_level level, const char *text, void *user_data) {
    // Copy the (small) callback state under the lock so we don't hold it across the
    // user-provided callback, which may itself do arbitrary work. While the copy is
    // in use, g_log_active keeps setLogger from deleting the captured global ref.
    std::function<void(ggml_log_level, const char *, void *)> cb;
    bool json_mode;
    {
        std::lock_guard<std::mutex> lk(g_log_mutex);
        cb = log_callback;
        json_mode = log_json;
        if (cb != nullptr) {
            ++g_log_active;
        }
    }
    if (cb != nullptr) {
        if (json_mode) {
            std::string json_text = format_log_as_json(level, text, std::time(nullptr));
            cb(level, json_text.c_str(), user_data);
        } else {
            cb(level, text, user_data);
        }
        {
            std::lock_guard<std::mutex> lk(g_log_mutex);
            --g_log_active;
        }
        g_log_cv.notify_all();
    }
}
} // namespace

/**
 * Convert a Java string to a std::string
 *
 * External linkage on purpose: native_server.cpp forward-declares this and calls it, so it
 * must live outside the anonymous namespace above (whose members have internal linkage and
 * are invisible to other translation units).
 */
std::string parse_jstring(JNIEnv *env, jstring java_string) {
    // A null receiver would make CallObjectMethod raise a pending JNI exception; bail out
    // cleanly so callers (e.g. parse_json_params) can surface their own error instead of
    // invoking ThrowNew under an already-pending exception.
    if (java_string == nullptr) {
        return std::string();
    }

    auto *const string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);

    // CallObjectMethod may have raised a pending exception (e.g. OOM) leaving string_bytes
    // null; dereferencing it via GetArrayLength would be undefined behaviour. Clear the pending
    // exception (calling arbitrary JNI functions — including the caller's ThrowNew — with one
    // pending is not allowed by the JNI spec) and bail out so the caller (parse_json_params)
    // surfaces a parse error as a LlamaException instead of crashing.
    if (env->ExceptionCheck() || string_bytes == nullptr) {
        env->ExceptionClear();
        return std::string();
    }

    auto length = (size_t)env->GetArrayLength(string_bytes);
    jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

    std::string string = std::string((char *)byte_elements, length);

    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
    env->DeleteLocalRef(string_bytes);

    return string;
}

// Validates the jllama_context at every JNI entry point.  Declares both
// `jctx` and `ctx_server` in the caller's scope; returns the given sentinel
// (omit for void functions) if the model is not loaded.
#define REQUIRE_SERVER_CONTEXT(...)                                                                                    \
    jllama_context_guard _jctx_guard{acquire_jllama_context_impl(env, obj, f_model_pointer)};                          \
    if (!_jctx_guard.ptr) {                                                                                            \
        env->ThrowNew(c_llama_error, "Model is not loaded");                                                           \
        return __VA_ARGS__;                                                                                            \
    }                                                                                                                  \
    auto *jctx = _jctx_guard.ptr;                                                                                      \
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
    c_llama_model = env->FindClass("net/ladenthin/llama/LlamaModel");
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
    c_llama_error = env->FindClass("net/ladenthin/llama/exception/LlamaException");
    c_log_level = env->FindClass("net/ladenthin/llama/value/LogLevel");
    c_log_format = env->FindClass("net/ladenthin/llama/args/LogFormat");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_standard_charsets && c_string && c_hash_map && c_map && c_set && c_entry && c_iterator &&
          c_integer && c_float && c_biconsumer && c_llama_error && c_log_level && c_log_format && c_error_oom)) {
        goto error;
    }

    // Promote local class refs (from FindClass) to global refs in one pass.
    // c_standard_charsets is intentionally omitted: only used to look up
    // f_utf_8 in this function and never referenced again.
    for (jclass *p : g_global_class_refs) {
        *p = (jclass)env->NewGlobalRef(*p);
    }

    // find constructors
    cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
    cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
    cc_float = env->GetMethodID(c_float, "<init>", "(F)V");
    cc_string_bytes_charset = env->GetMethodID(c_string, "<init>", "([BLjava/lang/String;)V");

    if (!(cc_hash_map && cc_integer && cc_float && cc_string_bytes_charset)) {
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
    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lnet/ladenthin/llama/value/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lnet/ladenthin/llama/value/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lnet/ladenthin/llama/value/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lnet/ladenthin/llama/value/LogLevel;");
    f_log_format_json = env->GetStaticFieldID(c_log_format, "JSON", "Lnet/ladenthin/llama/args/LogFormat;");
    f_log_format_text = env->GetStaticFieldID(c_log_format, "TEXT", "Lnet/ladenthin/llama/args/LogFormat;");

    if (!(f_model_pointer && f_utf_8 && f_log_level_debug && f_log_level_info && f_log_level_warn &&
          f_log_level_error && f_log_format_json && f_log_format_text)) {
        goto error;
    }

    o_utf_8 = env->NewStringUTF("UTF-8");
    for (const auto &b : g_static_object_bindings) {
        *b.target = env->GetStaticObjectField(*b.cls, *b.field);
    }

    if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error &&
          o_log_format_json && o_log_format_text)) {
        goto error;
    }

    for (jobject *p : g_global_object_refs) {
        *p = env->NewGlobalRef(*p);
    }

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

    for (jclass *p : g_global_class_refs) {
        env->DeleteGlobalRef(*p);
    }
    for (jobject *p : g_global_object_refs) {
        env->DeleteGlobalRef(*p);
    }

    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    llama_backend_free();
}

// Trampoline state for llama.cpp's load_progress_callback. The native loader runs
// on the calling JNI thread so we can capture JNIEnv directly. Lifetime is bounded
// by the single load_model_impl call.
namespace {
struct load_progress_ud {
    JNIEnv *env;
    jobject callback;
    jmethodID on_progress;
};

bool jni_load_progress_trampoline(float progress, void *user_data) {
    auto *ud = static_cast<load_progress_ud *>(user_data);
    return ud->env->CallBooleanMethod(ud->callback, ud->on_progress, progress) == JNI_TRUE;
}
} // namespace

// Shared implementation of loadModel and loadModelWithProgress. When `progress` is
// non-null, installs a load-progress trampoline; otherwise behaves identically to
// the no-callback path.
static void load_model_impl(JNIEnv *env, jobject obj, jobjectArray jparams, jobject progress) {
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
    jctx->vocab_only = vocab_only;
    jctx->params = params;

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

    common_params_print_info(params);

    // Resolve the auto sentinel before loading the model.
    if (params.n_parallel <= N_PARALLEL_AUTO) {
        params.n_parallel = N_PARALLEL_DEFAULT;
        jctx->params.n_parallel = N_PARALLEL_DEFAULT;
    }

    LOG_INF("%s: loading model\n", __func__);

    // Install the load-progress trampoline if the caller supplied a callback.
    load_progress_ud progress_ud{};
    if (progress != nullptr) {
        jclass cb_cls = env->GetObjectClass(progress);
        progress_ud.env = env;
        progress_ud.callback = progress;
        progress_ud.on_progress = env->GetMethodID(cb_cls, "onProgress", "(F)Z");
        if (progress_ud.on_progress == nullptr) {
            fail_load("LoadProgressCallback.onProgress(float) not found");
            return;
        }
        params.load_progress_callback = jni_load_progress_trampoline;
        params.load_progress_callback_user_data = &progress_ud;
    }

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
                common_chat_format_example(meta.chat_params.tmpls.get(), jctx->params.use_jinja,
                                           jctx->params.default_template_kwargs)
                    .c_str());
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

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jobjectArray jparams) {
    load_model_impl(env, obj, jparams, nullptr);
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_loadModelWithProgress(JNIEnv *env, jobject obj,
                                                                                 jobjectArray jparams,
                                                                                 jobject callback) {
    load_model_impl(env, obj, jparams, callback);
}

// Build the special-token id map (a token is -1 / LLAMA_TOKEN_NULL when the model defines none).
static json special_tokens_json(const llama_vocab *vocab) {
    return {
        {"bos", llama_vocab_bos(vocab)}, {"eos", llama_vocab_eos(vocab)}, {"eot", llama_vocab_eot(vocab)},
        {"sep", llama_vocab_sep(vocab)}, {"nl", llama_vocab_nl(vocab)},   {"pad", llama_vocab_pad(vocab)},
    };
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_getModelMetaJson(JNIEnv *env, jobject obj) {
    REQUIRE_SERVER_CONTEXT(nullptr);
    if (jctx->vocab_only) {
        json meta = {
            {"vocab_type", llama_vocab_type(jctx->vocab)},
            {"n_vocab", llama_vocab_n_tokens(jctx->vocab)},
            {"special_tokens", special_tokens_json(jctx->vocab)},
        };
        return json_to_jstring(env, meta);
    }
    auto m = ctx_server->get_meta();
    // Read general.architecture from GGUF metadata via the llama C API. Size the buffer
    // dynamically: llama_model_meta_val_str returns the required length when given a null/0
    // buffer, so a long architecture name is never silently truncated. A char vector (not a
    // std::string written in place) keeps the API's terminating '\0' write off the string's
    // internal terminator slot, which the standard does not allow to be written.
    std::string arch;
    const llama_model *mdl = llama_get_model(ctx_server->get_llama_context());
    if (mdl) {
        const int need = llama_model_meta_val_str(mdl, "general.architecture", nullptr, 0);
        if (need > 0) {
            std::vector<char> buf(static_cast<size_t>(need) + 1, '\0');
            llama_model_meta_val_str(mdl, "general.architecture", buf.data(), buf.size());
            arch.assign(buf.data());
        }
    }
    json j = {
        {"vocab_type", m.model_vocab_type},
        {"n_vocab", m.model_vocab_n_tokens},
        {"n_ctx_train", m.model_n_ctx_train},
        {"n_embd", m.model_n_embd_inp},
        {"n_params", m.model_n_params},
        {"size", m.model_size},
        {"modalities", {{"vision", m.has_inp_image}, {"audio", m.has_inp_audio}}},
        {"name", m.model_name},
        {"architecture", arch},
        {"ftype", m.model_ftype},
    };
    // Resolved default chat template (Jinja); empty when the model ships none.
    const char *chat_tmpl = mdl != nullptr ? llama_model_chat_template(mdl, /*name*/ nullptr) : nullptr;
    j["chat_template"] = chat_tmpl != nullptr ? std::string(chat_tmpl) : std::string();
    j["special_tokens"] = special_tokens_json(jctx->vocab);
    // Full GGUF metadata key/value map.
    if (mdl != nullptr) {
        json meta_map = json::object();
        const int meta_count = llama_model_meta_count(mdl);
        for (int i = 0; i < meta_count; i++) {
            char key_buf[256] = {};
            // ponytail: 2 KB/value cap — scalar metadata fits; huge array values
            // (tokenizer tokens/merges) truncate rather than bloating the JSON.
            char val_buf[2048] = {};
            if (llama_model_meta_key_by_index(mdl, i, key_buf, sizeof(key_buf)) >= 0 &&
                llama_model_meta_val_str_by_index(mdl, i, val_buf, sizeof(val_buf)) >= 0) {
                meta_map[std::string(key_buf)] = std::string(val_buf);
            }
        }
        j["metadata"] = std::move(meta_map);
    }
    return json_to_jstring(env, j);
}

JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj,
                                                                             jstring jparams) {
    REQUIRE_SERVER_CONTEXT(0);

    json data;
    if (!parse_json_params(env, jparams, data)) {
        return 0;
    }

    const server_task_type type = is_infill_request(data) ? SERVER_TASK_TYPE_INFILL : SERVER_TASK_TYPE_COMPLETION;

    return dispatch_streaming_completion(env, jctx, data, type, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_releaseTask(JNIEnv *env, jobject obj, jint id_task) {
    REQUIRE_SERVER_CONTEXT();
    erase_reader(jctx, id_task);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_receiveCompletionJson(JNIEnv *env, jobject obj,
                                                                                    jint id_task) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    // Copy the shared_ptr out under the lock so the reader stays alive across next() below,
    // which runs without the lock and may race a concurrent erase_reader()/close().
    std::shared_ptr<server_response_reader> rd;
    {
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        auto it = jctx->readers.find(id_task);
        if (it == jctx->readers.end()) {
            env->ThrowNew(c_llama_error, "Task not found");
            return nullptr;
        }
        rd = it->second;
    }

    // Upstream b9437 added is_begin partial results whose to_json() returns
    // a nullptr sentinel meaning "HTTP-headers-only, no body". Loop past
    // those so the Java iterator only ever sees real events.
    json response;
    try {
        while (true) {
            server_task_result_ptr result = rd->next([jctx] { return jctx->closing.load(); });

            if (!result_ok_or_throw(env, result)) {
                erase_reader(jctx, id_task);
                return nullptr;
            }

            response = result->to_json();
            if (response.is_null()) {
                continue;
            }
            response["stop"] = result->is_stop();

            if (result->is_stop()) {
                erase_reader(jctx, id_task);
            }
            break;
        }
    } catch (const std::exception &e) {
        // A throwing to_json() must surface as a LlamaException, not abort the JVM. The task is
        // over for the Java caller, so release its reader entry like the other error paths.
        erase_reader(jctx, id_task);
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }

    return json_to_jstring(env, response);
}

// Streaming OpenAI chat: poll one step of a chat.completion.chunk stream.
// Returns {"data": <chunk-object-or-array>, "stop": <bool>} — `data` is exactly
// what the streaming result's to_json() produced (a single chunk object for a
// partial token, or a JSON array of chunks for the final delta + usage). The
// uniform envelope avoids injecting a "stop" key into an array. Skips the
// header-only nullptr sentinels (upstream b9437+) and releases the reader on stop.
JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_receiveChatCompletionChunk(JNIEnv *env, jobject obj,
                                                                                         jint id_task) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    // Copy the shared_ptr out under the lock so the reader stays alive across next() below,
    // which runs without the lock and may race a concurrent erase_reader()/close().
    std::shared_ptr<server_response_reader> rd;
    {
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        auto it = jctx->readers.find(id_task);
        if (it == jctx->readers.end()) {
            env->ThrowNew(c_llama_error, "Task not found");
            return nullptr;
        }
        rd = it->second;
    }

    json payload;
    bool stop = false;
    try {
        while (true) {
            server_task_result_ptr result = rd->next([jctx] { return jctx->closing.load(); });

            if (!result_ok_or_throw(env, result)) {
                erase_reader(jctx, id_task);
                return nullptr;
            }

            json chunk = result->to_json();
            if (chunk.is_null()) {
                continue;
            }
            payload = std::move(chunk);
            stop = result->is_stop();
            if (stop) {
                erase_reader(jctx, id_task);
            }
            break;
        }
    } catch (const std::exception &e) {
        // A throwing to_json() must surface as a LlamaException, not abort the JVM. The task is
        // over for the Java caller, so release its reader entry like the other error paths.
        erase_reader(jctx, id_task);
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }

    return json_to_jstring(env, wrap_stream_chunk(std::move(payload), stop));
}

JNIEXPORT jfloatArray JNICALL Java_net_ladenthin_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    if (!require_embedding_support(env, jctx->params.embedding, c_llama_error)) {
        return nullptr;
    }

    const std::string prompt = parse_jstring(env, jprompt);
    SRV_INF("Calling embedding '%s'\n", prompt.c_str());

    llama_tokens tokens;
    try {
        tokens = tokenize_mixed(jctx->vocab, prompt, true, true);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
    auto rd = ctx_server->get_response_reader();
    server_task task(SERVER_TASK_TYPE_EMBEDDING);
    task.id = rd.get_new_id();
    task.tokens = server_tokens(tokens, false);
    task.index = 0;
    rd.post_task(std::move(task));

    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    if (!batch_ok_or_throw(env, br))
        return nullptr;

    if (br.results.empty()) {
        env->ThrowNew(c_llama_error, "embedding result is empty");
        return nullptr;
    }
    auto *embd_result = dynamic_cast<server_task_result_embd *>(br.results[0].get());
    if (!embd_result || embd_result->embedding.empty() || embd_result->embedding[0].empty()) {
        env->ThrowNew(c_llama_error, "embedding result is empty");
        return nullptr;
    }
    const std::vector<float> &first_row = embd_result->embedding[0];

    SRV_INF("Embedding has %d columns\n", static_cast<jsize>(first_row.size()));
    return embedding_to_jfloat_array_impl(env, first_row, c_error_oom);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleRerank(JNIEnv *env, jobject obj, jstring jprompt,
                                                                           jobjectArray documents) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    {
        auto meta = ctx_server->get_meta();
        if (!jctx->params.embedding || meta.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            env->ThrowNew(
                c_llama_error,
                "This server does not support reranking. Start it with `--reranking` and without `--embedding`");
            return nullptr;
        }
    }

    const std::string prompt = parse_jstring(env, jprompt);

    const jsize amount_documents = env->GetArrayLength(documents);
    auto *document_array = parse_string_array(env, documents, amount_documents);
    auto document_vector = std::vector<std::string>(document_array, document_array + amount_documents);
    free_string_array(document_array, amount_documents);

    const llama_model *model = llama_get_model(ctx_server->get_llama_context());
    auto rd = ctx_server->get_response_reader();
    std::vector<server_task> tasks;
    tasks.reserve(document_vector.size());
    for (size_t i = 0; i < document_vector.size(); i++) {
        tasks.push_back(build_indexed_token_task(
            rd, SERVER_TASK_TYPE_RERANK, format_prompt_rerank(model, jctx->vocab, nullptr, prompt, document_vector[i]),
            static_cast<int>(i), TASK_RESPONSE_TYPE_NONE));
    }
    rd.post_tasks(std::move(tasks));

    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    if (!batch_ok_or_throw(env, br))
        return nullptr;
    // rerank_results_to_json throws std::invalid_argument on a malformed/out-of-range
    // result index; unwrap it into a LlamaException instead of letting it cross
    // the JNI boundary (undefined behaviour / JVM abort).
    try {
        return json_to_jstring(env, rerank_results_to_json(br.results, document_vector));
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_applyTemplate(JNIEnv *env, jobject obj, jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json data;
    if (!parse_json_params(env, jparams, data)) {
        return nullptr;
    }

    json templateData;
    std::vector<raw_buffer> files;
    if (!parse_oai_chat_params(env, ctx_server, data, templateData, files))
        return nullptr;

    if (!templateData.contains("prompt") || !templateData.at("prompt").is_string()) {
        env->ThrowNew(c_llama_error, "applyTemplate did not produce a string prompt");
        return nullptr;
    }
    std::string tok_str = templateData.at("prompt");
    return utf8_to_jstring(env, tok_str);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleChatCompletions(JNIEnv *env, jobject obj,
                                                                                    jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json body;
    if (!parse_json_params(env, jparams, body)) {
        return nullptr;
    }
    json data;
    std::vector<raw_buffer> files;
    if (!parse_oai_chat_params(env, ctx_server, body, data, files))
        return nullptr;

    return dispatch_blocking_completion(env, jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CHAT,
                                        std::move(files));
}

JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestChatCompletion(JNIEnv *env, jobject obj,
                                                                                 jstring jparams) {
    REQUIRE_SERVER_CONTEXT(0);

    json body;
    if (!parse_json_params(env, jparams, body)) {
        return 0;
    }
    // Chat template already applied by parse_oai_chat_params; no OAI wrapping on the streaming path.
    json data;
    std::vector<raw_buffer> files;
    if (!parse_oai_chat_params(env, ctx_server, body, data, files))
        return 0;

    return dispatch_streaming_completion(env, jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_NONE,
                                         std::move(files));
}

// Streaming OpenAI chat with OAI-formatted chunks. Mirrors requestChatCompletion
// but sets TASK_RESPONSE_TYPE_OAI_CHAT so each polled result formats as an
// OpenAI chat.completion.chunk (including streamed delta.tool_calls). The params
// must carry "stream": true so the upstream formatter emits chunk deltas; poll
// the returned task id with receiveChatCompletionChunk.
JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestChatCompletionStream(JNIEnv *env, jobject obj,
                                                                                       jstring jparams) {
    REQUIRE_SERVER_CONTEXT(0);

    json body;
    if (!parse_json_params(env, jparams, body)) {
        return 0;
    }
    json data;
    std::vector<raw_buffer> files;
    if (!parse_oai_chat_params(env, ctx_server, body, data, files))
        return 0;

    return dispatch_streaming_completion(env, jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CHAT,
                                         std::move(files));
}

JNIEXPORT jintArray JNICALL Java_net_ladenthin_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const std::string c_prompt = parse_jstring(env, jprompt);
    try {
        llama_tokens tokens = tokenize_mixed(jctx->vocab, c_prompt, false, true);
        return tokens_to_jint_array_impl(env, tokens, c_error_oom);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

// Detokenise a token sequence to UTF-8, dispatching on vocab-only vs full context.
static std::string detokenize(jllama_context *jctx, const std::vector<llama_token> &tokens) {
    if (jctx->vocab_only) {
        return tokens_to_str(jctx->vocab, tokens);
    }
    return tokens_to_str(jctx->server.get_llama_context(), tokens);
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                             jintArray java_tokens) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const auto tokens = jint_array_to_tokens_impl(env, java_tokens);
    return parse_jbytes(env, detokenize(jctx, tokens));
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_delete(JNIEnv *env, jobject obj) {
    auto *jctx = get_jllama_context(env, obj);
    if (!jctx)
        return;

    // Null the Java handle under g_ctx_mutex so no NEW entry point can acquire this context
    // (acquire_jllama_context_impl will see 0 and return null). In-flight calls that already
    // hold a user reference keep running until they release on their own scope exit.
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        env->SetLongField(obj, f_model_pointer, 0);
    }

    // Signal teardown to any in-flight reader. The streaming / blocking should_stop lambdas
    // observe this and make next()/wait_for_all() return within one poll (~1s), so the
    // in-flight JNI call unwinds and releases its user reference — otherwise the reader would
    // poll forever (the worker only stops the task queue, not the results queue) and close()
    // would hang.
    jctx->closing.store(true, std::memory_order_release);
    if (!jctx->vocab_only) {
        // Cancel any pending streaming readers before stopping the server.
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        jctx->readers.clear();
    }

    // Wait for every in-flight JNI call (an entry point still inside its jllama_context_guard)
    // to release its user reference BEFORE tearing anything down — a concurrent call could
    // otherwise still be using the worker, the server_context, or the vocab-only model while
    // they are being stopped/freed (use-after-free). New calls cannot start (the handle is
    // already nulled above), and the closing flag bounds this wait to ~1s.
    {
        std::unique_lock<std::mutex> lk(jctx->m);
        jctx->cv.wait(lk, [&] { return jctx->users.load() == 0; });
    }

    if (!jctx->vocab_only) {
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

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task) {
    REQUIRE_SERVER_CONTEXT();
    erase_reader(jctx, id_task);
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject log_format,
                                                                     jobject jcallback) {
    // Serialize the whole swap under the logger mutex: first clear the live callback so no NEW
    // trampoline invocation can copy a lambda that still references the global ref we are about
    // to delete, then DRAIN trampolines already executing a copied callback (g_log_active), then
    // delete the old ref, then install the new one. Without the drain, an in-flight trampoline
    // could still call into the just-deleted global ref. Note: this makes setLogger block until
    // running log callbacks return — do not call setLogger from within a log callback.
    std::unique_lock<std::mutex> lk(g_log_mutex);
    log_callback = nullptr;
    g_log_cv.wait(lk, [] { return g_log_active == 0; });
    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
        o_log_callback = nullptr;
    }

    log_json = env->IsSameObject(log_format, o_log_format_json);

    if (jcallback == nullptr) {
        log_callback = nullptr;
        llama_log_set(nullptr, nullptr);
    } else {
        o_log_callback = env->NewGlobalRef(jcallback);
        // Capture copies of the global ref and method id so the callback never dereferences the
        // logger globals at call time (those may be swapped by a concurrent setLogger).
        jobject cb_ref = o_log_callback;
        log_callback = [cb_ref](enum ggml_log_level level, const char *text, void *user_data) noexcept {
            // Logging can fire from internal native threads with no JNIEnv; skip rather than
            // throw (an exception here would unwind through llama.cpp's C frames).
            JNIEnv *env = get_jni_env_or_null();
            if (env == nullptr || text == nullptr) {
                return;
            }
            // Log lines can embed payload text (prompts, model metadata), so the
            // message must cross as standard UTF-8, not Modified UTF-8.
            jstring message = utf8_to_jstring(env, text);
            if (message == nullptr) {
                env->ExceptionClear(); // allocation failed; drop this log line
                return;
            }
            jobject log_level = log_level_to_jobject(level);
            env->CallVoidMethod(cb_ref, m_biconsumer_accept, log_level, message);
            env->DeleteLocalRef(message);
        };
        // Always set the trampoline — it handles JSON formatting internally
        llama_log_set(log_callback_trampoline, nullptr);
    }
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_LlamaModel_jsonSchemaToGrammarBytes(JNIEnv *env, jclass clazz,
                                                                                          jstring j_schema) {
    try {
        const std::string c_schema = parse_jstring(env, j_schema);
        nlohmann::ordered_json c_schema_json = nlohmann::ordered_json::parse(c_schema);
        const std::string c_grammar = json_schema_to_grammar(c_schema_json);
        return parse_jbytes(env, c_grammar);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

// Returns llama.cpp's own build identifier ("b<number>-<commit>") from the linked-in build-info
// symbols. Pure-ASCII, so NewStringUTF is safe here (the modified-UTF-8 caveat only applies to
// model-generated payload text).
JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_nativeLlamaCppBuildInfo(JNIEnv *env, jclass clazz) {
    return env->NewStringUTF(llama_build_info());
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleCompletions(JNIEnv *env, jobject obj,
                                                                                jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json data;
    if (!parse_json_params(env, jparams, data)) {
        return nullptr;
    }
    return dispatch_blocking_completion(env, jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleCompletionsOai(JNIEnv *env, jobject obj,
                                                                                   jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json body;
    if (!parse_json_params(env, jparams, body)) {
        return nullptr;
    }
    json data;
    try {
        data = oaicompat_completion_params_parse(body);
    } catch (const std::exception &e) {
        throw_invalid_request(env, e);
        return nullptr;
    }

    return dispatch_blocking_completion(env, jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CMPL);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleInfill(JNIEnv *env, jobject obj, jstring jparams) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    // Check FIM token support via server_context_meta (populated from the
    // same llama_vocab_fim_* calls inside server-context).
    auto meta = ctx_server->get_meta();
    if (meta.fim_pre_token == LLAMA_TOKEN_NULL || meta.fim_sub_token == LLAMA_TOKEN_NULL ||
        meta.fim_mid_token == LLAMA_TOKEN_NULL) {
        env->ThrowNew(c_llama_error, "Model does not support fill-in-the-middle infill");
        return nullptr;
    }

    json data;
    if (!parse_json_params(env, jparams, data)) {
        return nullptr;
    }

    if (!require_json_field(env, data, "input_prefix"))
        return nullptr;
    if (!require_json_field(env, data, "input_suffix"))
        return nullptr;

    json input_extra = json_value(data, "input_extra", json::array());
    data["input_extra"] = input_extra;

    std::string prompt = json_value(data, "prompt", std::string());
    try {
        std::vector<server_tokens> tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, prompt, false, true);

        data["prompt"] =
            format_prompt_infill(jctx->vocab, data.at("input_prefix"), data.at("input_suffix"), data.at("input_extra"),
                                 jctx->params.n_batch, jctx->params.n_predict, meta.slot_n_ctx, jctx->params.spm_infill,
                                 tokenized_prompts.empty() ? llama_tokens() : tokenized_prompts[0].get_tokens());
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }

    return dispatch_blocking_completion(env, jctx, data, SERVER_TASK_TYPE_INFILL, TASK_RESPONSE_TYPE_NONE);
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleEmbeddings(JNIEnv *env, jobject obj,
                                                                               jstring jparams, jboolean joaiCompat) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    if (!require_embedding_support(env, jctx->params.embedding, c_llama_error)) {
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

    json body;
    if (!parse_json_params(env, jparams, body)) {
        return nullptr;
    }

    bool force_no_oaicompat = false;
    json prompt;
    bool use_base64 = false;
    try {
        prompt = extract_embedding_prompt(body, force_no_oaicompat);
        use_base64 = parse_encoding_format(body);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
    if (force_no_oaicompat)
        res_type = TASK_RESPONSE_TYPE_NONE;

    std::vector<server_tokens> tokenized_prompts;
    try {
        tokenized_prompts = tokenize_input_prompts(jctx->vocab, nullptr, prompt, true, true);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }

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

    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    if (!batch_ok_or_throw(env, br))
        return nullptr;

    json responses = json::array();
    for (const auto &result : br.results) {
        responses.push_back(result->to_json());
    }
    json out = (res_type == TASK_RESPONSE_TYPE_OAI_EMBD)
                   ? format_embeddings_response_oaicompat(
                         body, json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), responses, use_base64)
                   : responses;
    try {
        return json_to_jstring(env, out);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleTokenize(JNIEnv *env, jobject obj, jstring jcontent,
                                                                             jboolean jaddSpecial,
                                                                             jboolean jwithPieces) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const std::string content = parse_jstring(env, jcontent);
    const bool add_special = jaddSpecial;
    const bool with_pieces = jwithPieces;

    llama_tokens tokens;
    try {
        tokens = tokenize_mixed(jctx->vocab, content, add_special, true);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }

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

    return json_to_jstring(env, format_tokenizer_response(tokens_response));
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleDetokenize(JNIEnv *env, jobject obj,
                                                                               jintArray jtokens) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    const auto tokens = jint_array_to_tokens_impl(env, jtokens);
    return json_to_jstring(env, format_detokenized_response(detokenize(jctx, tokens)));
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleSlotAction(JNIEnv *env, jobject obj, jint action,
                                                                               jint slotId, jstring jfilename) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    switch (action) {
    case 0: // LIST — get slot info via the dedicated slot-get task
        // b10519 (upstream #27376) split the old METRICS task in two: METRICS now carries only the
        // cumulative counters (rendered as Prometheus text by to_metrics(); its to_json() returns an
        // empty object), and the /slots payload moved to SERVER_TASK_TYPE_SLOT_GET ->
        // server_task_result_slots::to_json(), which returns the slot array verbatim.
        return dispatch_one_shot_task(env, ctx_server, server_task(SERVER_TASK_TYPE_SLOT_GET));
    case 1: // SAVE
        return exec_slot_file_task(env, ctx_server, slotId, jfilename, SERVER_TASK_TYPE_SLOT_SAVE,
                                   "Filename is required for slot save");
    case 2: // RESTORE
        return exec_slot_file_task(env, ctx_server, slotId, jfilename, SERVER_TASK_TYPE_SLOT_RESTORE,
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

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_getLoraAdaptersJson(JNIEnv *env, jobject obj) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    return dispatch_one_shot_task(env, ctx_server, server_task(SERVER_TASK_TYPE_GET_LORA));
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_setLoraAdaptersJson(JNIEnv *env, jobject obj,
                                                                                  jstring jadapters) {
    REQUIRE_SERVER_CONTEXT(nullptr);

    json data;
    if (!parse_json_params(env, jadapters, data)) {
        return nullptr;
    }
    if (!data.is_array()) {
        // Same contract as the upstream POST /lora-adapters route body.
        env->ThrowNew(c_llama_error, "LoRA adapter list must be a JSON array of {id, scale} objects");
        return nullptr;
    }
    server_task task(SERVER_TASK_TYPE_SET_LORA);
    task.set_lora = parse_lora_request(data);
    return dispatch_one_shot_task(env, ctx_server, std::move(task));
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaQuantizer_quantizeNative(JNIEnv *env, jclass, jstring jinput,
                                                                              jstring joutput, jint ftype, jint nthread,
                                                                              jboolean allowRequantize) {
    try {
        const std::string input_path = parse_jstring(env, jinput);
        const std::string output_path = parse_jstring(env, joutput);
        // Idempotent; intentionally never paired with llama_backend_free here — a LlamaModel
        // loaded in the same JVM shares the backend and must not have it freed underneath it.
        llama_backend_init();
        llama_model_quantize_params qparams = llama_model_quantize_default_params();
        qparams.ftype = static_cast<llama_ftype>(ftype);
        qparams.nthread = nthread;
        qparams.allow_requantize = (allowRequantize == JNI_TRUE);
        const uint32_t rc = llama_model_quantize(input_path.c_str(), output_path.c_str(), &qparams);
        if (rc != 0) {
            const std::string msg = "Quantization of '" + input_path + "' failed with code " + std::to_string(rc);
            env->ThrowNew(c_llama_error, msg.c_str());
        }
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
    } catch (...) {
        env->ThrowNew(c_llama_error, "Unknown C++ exception during quantization");
    }
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_LlamaModel_configureParallelInference(JNIEnv *env, jobject obj,
                                                                                          jstring jconfig) {
    REQUIRE_SERVER_CONTEXT(JNI_FALSE);
    (void)obj;

    json config;
    if (!parse_json_params(env, jconfig, config)) {
        return JNI_FALSE;
    }

    std::optional<float> slot_sim_opt;
    std::optional<int> n_threads_opt;
    std::optional<int> n_threads_batch_opt;
    try {
        slot_sim_opt = parse_slot_prompt_similarity(config);
        n_threads_opt = parse_positive_int_config(config, "n_threads");
        n_threads_batch_opt = parse_positive_int_config(config, "n_threads_batch");
    } catch (const std::exception &e) {
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
            env->ThrowNew(c_llama_error, "configureParallelInference: llama_context not available "
                                         "(model sleeping or not loaded)");
            return JNI_FALSE;
        }
        const int n = n_threads_opt.value_or(jctx->params.cpuparams.n_threads);
        const int nb = n_threads_batch_opt.value_or(jctx->params.cpuparams_batch.n_threads);
        llama_set_n_threads(lctx, n, nb);
        // Keep the cached params in sync so a follow-up call that supplies only
        // the other field reads back the value just applied, not the original.
        jctx->params.cpuparams.n_threads = n;
        jctx->params.cpuparams_batch.n_threads = nb;
    }

    // slot_prompt_similarity: validated above (the [0.0, 1.0] range check still
    // throws for out-of-range values, preserving the existing exception
    // contract).  Live mutation uses server_context::set_slot_prompt_similarity(),
    // added upstream by https://github.com/ggml-org/llama.cpp/pull/22393 and carried
    // in this repo as patches/0003-pr22393-... until it merges upstream (the patch
    // applies against the pinned llama.cpp GIT_TAG in CMakeLists.txt).  not thread-safe
    // per the upstream contract — main-thread only, which this JNI call is.
    if (slot_sim_opt.has_value()) {
        ctx_server->set_slot_prompt_similarity(*slot_sim_opt);
    }

    return JNI_TRUE;
}

// ---------------------------------------------------------------------------
// TextToSpeech (upstream Qwen3-TTS via mtmd_helper::gen_audio) — native methods for the
// backbone+mmproj TTS pipeline. Separate Java type (net.ladenthin.llama.TextToSpeech);
// implemented here so it can reuse parse_jstring / c_llama_error / c_error_oom from this TU.
// ---------------------------------------------------------------------------
extern "C" {

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_TextToSpeech_loadNative(JNIEnv *env, jclass clazz, jstring jmodel,
                                                                         jstring jmmproj, jint gpu_layers,
                                                                         jint threads) {
    (void)clazz;
    try {
        const std::string model = parse_jstring(env, jmodel);
        const std::string mmproj = parse_jstring(env, jmmproj);
        std::string err;
        jllama_tts::tts_engine *engine =
            jllama_tts::engine_init(model, mmproj, static_cast<int>(gpu_layers), static_cast<int>(threads), err);
        if (engine == nullptr) {
            env->ThrowNew(c_llama_error, err.c_str());
            return 0;
        }
        return reinterpret_cast<jlong>(engine);
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return 0;
    }
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_TextToSpeech_synthesizeNative(JNIEnv *env, jclass clazz,
                                                                                    jlong handle, jstring jtext,
                                                                                    jstring jspeaker_reference_path,
                                                                                    jstring jlang, jint max_frames,
                                                                                    jint top_k, jint seed) {
    (void)clazz;
    try {
        auto *engine = reinterpret_cast<jllama_tts::tts_engine *>(handle);
        if (engine == nullptr) {
            env->ThrowNew(c_llama_error, "TextToSpeech handle is null");
            return nullptr;
        }
        const std::string text = parse_jstring(env, jtext);
        const std::string speaker_reference_path =
            jspeaker_reference_path == nullptr ? std::string() : parse_jstring(env, jspeaker_reference_path);
        const std::string lang = jlang == nullptr ? std::string() : parse_jstring(env, jlang);
        std::vector<uint8_t> wav;
        std::string err;
        if (!jllama_tts::engine_synthesize(engine, text, speaker_reference_path, lang, static_cast<int>(max_frames),
                                           static_cast<int>(top_k), static_cast<uint32_t>(seed), wav, err)) {
            env->ThrowNew(c_llama_error, err.c_str());
            return nullptr;
        }
        jbyteArray out = env->NewByteArray(static_cast<jsize>(wav.size()));
        if (out == nullptr) {
            env->ThrowNew(c_error_oom, "could not allocate WAV byte array");
            return nullptr;
        }
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(wav.size()), reinterpret_cast<const jbyte *>(wav.data()));
        return out;
    } catch (const std::exception &e) {
        env->ThrowNew(c_llama_error, e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_TextToSpeech_deleteNative(JNIEnv *env, jclass clazz, jlong handle) {
    (void)env;
    (void)clazz;
    jllama_tts::engine_free(reinterpret_cast<jllama_tts::tts_engine *>(handle));
}

} // extern "C"
