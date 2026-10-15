// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

#include "jllama.h"

#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"
#include "jllama_engine.hpp"
#include "jllama_api.h"
#include "log_helpers.hpp"

#ifdef GGML_BACKEND_DL
// load_backends_next_to_this_library() needs this library's own path: dladdr on POSIX,
// GetModuleHandleExW + GetModuleFileNameW on Windows. On Android the modules are loaded by soname
// instead (load_backends_by_soname, JLLAMA_MODULES_BY_SONAME), through dlopen/dlsym.
#ifdef JLLAMA_MODULES_BY_SONAME
#include "jllama_modules.h"
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX // or windows.h's min/max macros break std::min / std::max below
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

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

// N_PARALLEL_AUTO / N_PARALLEL_DEFAULT moved to jllama_engine.cpp with the load.

// jllama_context is defined in jllama_context.hpp (via jni_helpers.hpp).

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
jclass c_llama_invalid_request = nullptr;
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
    &c_llama_model, &c_string,     &c_hash_map,  &c_map,        &c_set,         &c_entry,
    &c_iterator,    &c_integer,    &c_float,     &c_biconsumer, &c_llama_error, &c_llama_invalid_request,
    &c_log_level,   &c_log_format, &c_error_oom,
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

// Every model operation of LlamaModel is a shell over the C API (jllama_api.h): the Java arguments
// cross as C strings and arrays, the status comes back as a Java exception (throw_api_status), the
// out-strings as Java strings (api_string). The engine (jllama_engine.hpp) behind the C API is the
// one implementation; nothing in this file reads the server any more.

/**
 * Throws the Java exception a failed C API call maps to (jllama_last_error() carries the message).
 */
static void throw_api_status(JNIEnv *env, jllama_status status) {
    env->ThrowNew(status == JLLAMA_STATUS_INVALID_REQUEST ? c_llama_invalid_request : c_llama_error,
                  jllama_last_error());
}

/** The handle a Java field holds, as the C API's opaque model. */
[[nodiscard]] static jllama_model *api(jllama_context *jctx) { return reinterpret_cast<jllama_model *>(jctx); }

/**
 * The Java string of a C API out-string (released here), or the Java exception of a failed call
 * and nullptr. Every string-returning shell ends in this.
 */
[[nodiscard]] static jstring api_string(JNIEnv *env, jllama_status status, char *response) {
    if (status != JLLAMA_STATUS_OK) {
        jllama_string_free(response);
        throw_api_status(env, status);
        return nullptr;
    }
    jstring result = utf8_to_jstring(env, response);
    jllama_string_free(response);
    return result;
}

/** The task id of a C API stream request, or the Java exception of a failed call and 0. */
[[nodiscard]] static jint api_task(JNIEnv *env, jllama_status status, int task_id) {
    if (status != JLLAMA_STATUS_OK) {
        throw_api_status(env, status);
        return 0;
    }
    return static_cast<jint>(task_id);
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

/**
 * A JNIEnv for the current thread, attaching the thread to the JVM once if it is not attached yet.
 *
 * The log sink runs on common_log's worker thread, a plain std::thread that llama.cpp creates (and
 * re-creates on every pause/resume) and that has never seen the JVM. Until 5.2.x the trampoline
 * attached and detached per log line -- leak-free, but every attach creates a java.lang.Thread
 * object and fires the JVMTI thread events (measured: 13 lines of a failed load, 13 Thread objects),
 * which is noticeable at --verbose volumes. Now the attach lasts for the life of the thread: a
 * thread_local guard records it, and its destructor detaches when the thread exits.
 * common_log::pause() joins the worker, a normal exit on which C++ TLS destructors run (glibc,
 * macOS, MSVC -- also for a dlopen'd library, whose registered TLS destructors keep it mapped until
 * those threads are gone, and JNI_OnUnload joins the worker through common_log_set_callback(nullptr)
 * before anything else). Attached as a daemon, so DestroyJavaVM never waits for the singleton
 * logger's worker; should the worker outlive the VM (the static common_log is destroyed at process
 * exit, after DestroyJavaVM), HotSpot's DetachCurrentThread returns as soon as it sees the VM has
 * exited. A thread the JVM already owns (GetEnv == JNI_OK) is used as it is and never detached.
 * Returns nullptr (nothing to log through) when the JVM is gone or refuses the attach.
 */
namespace {
struct jni_thread_attachment {
    bool attached = false;
    ~jni_thread_attachment() {
        if (attached && g_vm != nullptr) {
            g_vm->DetachCurrentThread();
        }
    }
};
thread_local jni_thread_attachment t_jni_attachment;
} // namespace

JNIEnv *get_jni_env_attaching() noexcept {
    JNIEnv *env = nullptr;
    if (g_vm == nullptr) {
        return nullptr;
    }
    const jint res = g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (res == JNI_OK) {
        return env;
    }
    if (res != JNI_EDETACHED) {
        return nullptr;
    }
    char name[] = "jllama-log-worker"; // the JVM copies it; shows in thread dumps instead of Thread-N
    JavaVMAttachArgs args = {JNI_VERSION_1_6, name, nullptr};
    if (g_vm->AttachCurrentThreadAsDaemon(reinterpret_cast<void **>(&env), &args) != JNI_OK) {
        return nullptr;
    }
    t_jni_attachment.attached = true;
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
// Serializes setLogger callers against each other. It is NOT g_log_mutex: the swap pauses and
// resumes common_log's worker (a join + a fresh std::thread), and two unserialized swaps race on
// that std::thread -- one caller joins it while the other assigns a new thread over the still
// joinable object, which is std::terminate. The trampoline never takes this mutex, so holding it
// across the join cannot deadlock.
static std::mutex g_set_logger_mutex;

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

// The C API's log callback shape over the JNI trampoline above (jllama_set_log_callback takes an
// int level; the values are ggml_log_level's).
void jni_log_sink(int level, const char *text, void *user_data) {
    log_callback_trampoline(static_cast<ggml_log_level>(level), text, user_data);
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

#ifdef GGML_BACKEND_DL
// What the process ended up with after the modules were loaded: a module whose runtime is missing
// fails silently (ggml logs it at GGML_LOG_DEBUG only, and on Windows SEM_FAILCRITICALERRORS
// suppresses the dialog), so say which backends ggml registered and which devices they report.
// Reaches LlamaModel.setLogger.
static void log_backend_registry(const char *where, const std::string &from) {
    std::string devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        devices += (i == 0 ? "" : ", ");
        devices += ggml_backend_dev_name(dev);
        devices += " (";
        devices += ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        devices += ")";
    }
    std::string backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        backends += (i == 0 ? "" : ", ");
        backends += ggml_backend_reg_name(ggml_backend_reg_get(i));
    }
    LOG_INF("%s: %zu backend(s) registered from %s [%s], %zu device(s): %s\n", where, ggml_backend_reg_count(),
            from.c_str(), backends.c_str(), ggml_backend_dev_count(), devices.c_str());
}

#ifdef JLLAMA_MODULES_BY_SONAME
// Android: an app's native libraries stay inside the APK (extractNativeLibs=false is the default from
// API 23 on), so there is no directory to scan -- dladdr() reports a path into the zip, and ggml's
// directory iteration finds nothing. bionic resolves a bare soname against the app's own libraries
// (the same lookup System.loadLibrary uses), so the modules are opened by name instead: the CPU
// variants of this build (generated into jllama_modules.h at configure time) are scored the way
// ggml_backend_load_best() scores them and the best one is registered, every other module of the
// build (ggml-rpc) and every GPU module another AAR may add next to this one (the fixed list below)
// is registered when it is there. A missing one is silent, as a missing module is on the desktop.
static int score_module_by_soname(const char *soname) {
    void *handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return -1;
    }
    int score = 1; // ggml treats a module without a score function as always usable
    using score_fn_t = int (*)();
    if (auto score_fn = reinterpret_cast<score_fn_t>(dlsym(handle, "ggml_backend_score"))) {
        score = score_fn();
    }
    dlclose(handle);
    return score;
}

static void load_backends_by_soname() {
    const std::vector<const char *> cpu_modules = JLLAMA_CPU_MODULE_SONAMES;
    const std::vector<const char *> other_modules = JLLAMA_OTHER_MODULE_SONAMES;
    static const char *const gpu_modules[] = {"libggml-opencl.so", "libggml-vulkan.so"};
    const char *best = nullptr;
    int best_score = 0;
    for (const char *soname : cpu_modules) {
        const int score = score_module_by_soname(soname);
        if (score > best_score) {
            best = soname;
            best_score = score;
        }
    }
    if (best != nullptr) {
        ggml_backend_load(best);
    } else {
        LOG_ERR("%s: none of the %zu CPU backend module(s) of this build is usable on this device\n", __func__,
                cpu_modules.size());
    }
    for (const char *soname : other_modules) {
        if (score_module_by_soname(soname) > 0) {
            ggml_backend_load(soname);
        }
    }
    for (const char *soname : gpu_modules) {
        if (score_module_by_soname(soname) > 0) {
            ggml_backend_load(soname);
        }
    }
    log_backend_registry(__func__, "the APK (by soname)");
}
#else
// With GGML_BACKEND_DL (JLLAMA_BACKEND_DL) the CPU and RPC backends -- and every GPU backend whose
// natives jar is on the classpath -- are modules ggml loads at run time, and ggml_backend_load_all()
// looks next to the executable -- `java`, here. LlamaLoader extracts the modules next to this
// library, so they are loaded from this library's own directory before anything touches the backend
// registry. ggml_backend_load_best() picks the CPU module whose instruction set the running CPU
// supports best, and loads the GPU modules in its fixed order (cuda, hip, sycl, vulkan, opencl,
// openvino): one whose vendor runtime is missing fails to open and registers nothing.
static void load_backends_next_to_this_library() {
#ifdef _WIN32
    // GetModuleHandleExW on an address inside this DLL -- GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT
    // so the handle needs no release. The path must reach ggml as UTF-8: it opens the modules with
    // the narrow CRT, and a directory holding a non-ASCII character (an umlaut in a Windows user
    // name reaches the temp directory LlamaLoader extracts into) loaded nothing at all when the
    // path was handed over in the active ANSI code page -- without any error being logged.
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&load_backends_next_to_this_library), &self)) {
        return;
    }
    std::wstring wpath(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(self, wpath.data(), static_cast<DWORD>(wpath.size()));
        if (n == 0) {
            return;
        }
        if (n < wpath.size()) {
            wpath.resize(n);
            break;
        }
        if (wpath.size() > 32768) { // the documented ceiling for an extended-length path
            return;
        }
        wpath.resize(wpath.size() * 2);
    }
    const size_t sep = wpath.find_last_of(L"\\/");
    const std::wstring wdir = sep == std::wstring::npos ? std::wstring(L".") : wpath.substr(0, sep);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) {
        return;
    }
    std::string dir(static_cast<size_t>(bytes - 1), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, dir.data(), bytes, nullptr, nullptr) <= 0) {
        return;
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&load_backends_next_to_this_library), &info) == 0 ||
        info.dli_fname == nullptr) {
        return;
    }
    std::string dir(info.dli_fname);
    const size_t slash = dir.find_last_of('/');
    dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
#endif
    ggml_backend_load_all_from_path(dir.c_str());
    log_backend_registry(__func__, dir);
}
#endif // JLLAMA_MODULES_BY_SONAME
#endif // GGML_BACKEND_DL

/**
 * The VM calls JNI_OnLoad when the native library is loaded (for example, through `System.loadLibrary`).
 * `JNI_OnLoad` must return the JNI version needed by the native library.
 * In order to use any of the new JNI functions, a native library must export a `JNI_OnLoad` function that returns
 * `JNI_VERSION_1_2`. If the native library does not export a JNI_OnLoad function, the VM assumes that the library
 * only requires JNI version `JNI_VERSION_1_1`. If the VM does not recognize the version number returned by
 `JNI_OnLoad`, the VM will unload the library and act as if the library was never loaded.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) try {
    g_vm = vm;
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1)) {
        goto error;
    }

#ifdef JLLAMA_MODULES_BY_SONAME
    load_backends_by_soname();
#elif defined(GGML_BACKEND_DL)
    load_backends_next_to_this_library();
#endif

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
    c_llama_invalid_request = env->FindClass("net/ladenthin/llama/exception/InvalidRequestException");
    c_log_level = env->FindClass("net/ladenthin/llama/value/LogLevel");
    c_log_format = env->FindClass("net/ladenthin/llama/args/LogFormat");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_standard_charsets && c_string && c_hash_map && c_map && c_set && c_entry && c_iterator &&
          c_integer && c_float && c_biconsumer && c_llama_error && c_llama_invalid_request && c_log_level &&
          c_log_format && c_error_oom)) {
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
} catch (...) {
    // A function-try-block, not jni_guard_impl: this runs BEFORE c_llama_error is cached, so there
    // is no class to throw into Java with. Refusing to load is the only correct outcome — the JVM
    // surfaces it to the caller as an UnsatisfiedLinkError. (It is also the one shape that leaves
    // the goto/label flow above untouched.)
    return JNI_ERR;
}

/**
 * The VM calls `JNI_OnUnload` when the class loader containing the native library is garbage collected.
 * This function can be used to perform cleanup operations. Because this function is called in an unknown context
 * (such as from a finalizer), the programmer should be conservative on using Java VM services, and refrain from
 * arbitrary Java call-backs.
 * Note that `JNI_OnLoad` and `JNI_OnUnload` are two functions optionally supplied by JNI libraries, not exported from
 * the VM.
 */
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) try {
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

    // Detach the sink first (this drains the queue through it), then drop the ref it calls into.
    jllama_set_log_callback(nullptr, nullptr);
    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    llama_backend_free();
} catch (...) {
    // Mirrors JNI_OnLoad: the cached class references are being released here, so there is nothing
    // to throw into Java. Unload is best-effort by definition — swallow rather than abort the JVM.
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

// Shared implementation of loadModel and loadModelWithProgress: the argv crosses into the C API
// (jllama_model_load), which runs the load the engine owns; a Java LoadProgressCallback becomes the
// C progress callback through jni_load_progress_trampoline. The handle the C API returns is the
// jllama_context the Java field holds.
static void load_model_impl(JNIEnv *env, jobject obj, jobjectArray jparams, jobject progress) {
    const jsize argc = env->GetArrayLength(jparams);
    char **argv = parse_string_array(env, jparams, argc);
    if (argv == nullptr) {
        env->ThrowNew(c_error_oom, "Failed to allocate memory for parameters");
        return;
    }

    load_progress_ud progress_ud{};
    jllama_load_progress_fn progress_fn = nullptr;
    if (progress != nullptr) {
        jclass cb_cls = env->GetObjectClass(progress);
        progress_ud.env = env;
        progress_ud.callback = progress;
        progress_ud.on_progress = env->GetMethodID(cb_cls, "onProgress", "(F)Z");
        if (progress_ud.on_progress == nullptr) {
            free_string_array(argv, argc);
            env->ThrowNew(c_llama_error, "LoadProgressCallback.onProgress(float) not found");
            return;
        }
        progress_fn = jni_load_progress_trampoline;
    }

    jllama_model *model = nullptr;
    const jllama_status status = jllama_model_load(static_cast<int>(argc), argv, progress_fn,
                                                   progress != nullptr ? &progress_ud : nullptr, &model);
    free_string_array(argv, argc);
    if (status != JLLAMA_STATUS_OK) {
        // An unreachable --rpc server and a refused argv are both LlamaException here, as before.
        env->ThrowNew(c_llama_error, jllama_last_error());
        return;
    }
    env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(model));
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jobjectArray jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void { load_model_impl(env, obj, jparams, nullptr); });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_loadModelWithProgress(JNIEnv *env, jobject obj,
                                                                                 jobjectArray jparams,
                                                                                 jobject callback) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void { load_model_impl(env, obj, jparams, callback); });
}

// Build the special-token id map (a token is -1 / LLAMA_TOKEN_NULL when the model defines none).
static json special_tokens_json(const llama_vocab *vocab) {
    return {
        {"bos", llama_vocab_bos(vocab)}, {"eos", llama_vocab_eos(vocab)}, {"eot", llama_vocab_eot(vocab)},
        {"sep", llama_vocab_sep(vocab)}, {"nl", llama_vocab_nl(vocab)},   {"pad", llama_vocab_pad(vocab)},
    };
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_getModelMetaJson(JNIEnv *env, jobject obj) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        char *response = nullptr;
        const jllama_status status = jllama_model_meta(api(jctx), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj,
                                                                             jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jint {
        REQUIRE_SERVER_CONTEXT(0);
        const std::string body = parse_jstring(env, jparams);
        int task_id = 0;
        const jllama_status status = jllama_completions_stream_request(api(jctx), body.c_str(), &task_id);
        return api_task(env, status, task_id);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_releaseTask(JNIEnv *env, jobject obj, jint id_task) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
        REQUIRE_SERVER_CONTEXT();
        jllama_task_release(reinterpret_cast<jllama_model *>(jctx), static_cast<int>(id_task));
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_receiveCompletionJson(JNIEnv *env, jobject obj,
                                                                                    jint id_task) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        char *response = nullptr;
        bool stop = false;
        const jllama_status status =
            jllama_completions_stream_receive(api(jctx), static_cast<int>(id_task), &response, &stop);
        return api_string(env, status, response);
    });
}

// Streaming OpenAI chat: poll one step of a chat.completion.chunk stream through the C API.
// Returns {"data": <chunk-object-or-array>, "stop": <bool>} -- `data` is exactly what the streaming
// result's to_json() produced (a single chunk object for a partial token, or a JSON array of chunks
// for the final delta + usage). The uniform envelope avoids injecting a "stop" key into an array.
// Header-only sentinels (upstream b9437+) are skipped and the reader released on stop, inside.
JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_receiveChatCompletionChunk(JNIEnv *env, jobject obj,
                                                                                         jint id_task) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        char *envelope = nullptr;
        bool stop = false;
        const jllama_status status = jllama_chat_completions_stream_receive(
            reinterpret_cast<jllama_model *>(jctx), static_cast<int>(id_task), &envelope, &stop);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        jstring result = utf8_to_jstring(env, envelope);
        jllama_string_free(envelope);
        return result;
    });
}

JNIEXPORT jfloatArray JNICALL Java_net_ladenthin_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jfloatArray {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string prompt = parse_jstring(env, jprompt);
        float *values = nullptr;
        size_t count = 0;
        const jllama_status status = jllama_embed(api(jctx), prompt.c_str(), &values, &count);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        const std::vector<float> row(values, values + count);
        jllama_floats_free(values);
        return embedding_to_jfloat_array_impl(env, row, c_error_oom);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleRerank(JNIEnv *env, jobject obj, jstring jprompt,
                                                                           jobjectArray documents) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string query = parse_jstring(env, jprompt);
        const jsize amount_documents = env->GetArrayLength(documents);
        char **document_array = parse_string_array(env, documents, amount_documents);
        if (document_array == nullptr) {
            env->ThrowNew(c_error_oom, "Failed to allocate memory for documents");
            return nullptr;
        }
        char *response = nullptr;
        const jllama_status status =
            jllama_rerank(api(jctx), query.c_str(), static_cast<int>(amount_documents), document_array, &response);
        free_string_array(document_array, amount_documents);
        return api_string(env, status, response);
    });
}

/**
 * `POST /v1/systemone` (llama.cpp b11361, #29818): answers typed questions about a state with a
 * decision model. Forwarded to the upstream route handler rather than re-implemented -- the request
 * parsing, the per-model prompt layout, the shared-prefix grouping and the answer formatting all live
 * in upstream's `server_decision_context`, which is private to `server_context_impl` and reachable
 * only through `server_routes`. The handler waits for the server to leave sleep itself
 * (`create_response()`), and every failure it reports as an error body becomes a LlamaException
 * carrying upstream's message ("This model is not a decision model", a malformed question, ...).
 */
JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleSystemOne(JNIEnv *env, jobject obj,
                                                                              jstring jrequest) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jrequest);
        char *response = nullptr;
        const jllama_status status = jllama_systemone(api(jctx), body.c_str(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_applyTemplate(JNIEnv *env, jobject obj, jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *prompt = nullptr;
        const jllama_status status = jllama_apply_template(api(jctx), body.c_str(), &prompt);
        return api_string(env, status, prompt);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleChatCompletions(JNIEnv *env, jobject obj,
                                                                                    jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *response = nullptr;
        const jllama_status status =
            jllama_chat_completions(reinterpret_cast<jllama_model *>(jctx), body.c_str(), &response);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        jstring result = utf8_to_jstring(env, response);
        jllama_string_free(response);
        return result;
    });
}

JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestChatCompletion(JNIEnv *env, jobject obj,
                                                                                 jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jint {
        REQUIRE_SERVER_CONTEXT(0);
        // The chat template is applied inside; the results are raw (no OAI chunk formatting), polled
        // with receiveCompletionJson.
        const std::string body = parse_jstring(env, jparams);
        int task_id = 0;
        const jllama_status status = jllama_chat_completions_request(api(jctx), body.c_str(), &task_id);
        return api_task(env, status, task_id);
    });
}

// Streaming OpenAI chat with OAI-formatted chunks, through the C API: each polled result formats as
// an OpenAI chat.completion.chunk (including streamed delta.tool_calls). The params must carry
// "stream": true so the upstream formatter emits chunk deltas; poll the returned task id with
// receiveChatCompletionChunk.
JNIEXPORT jint JNICALL Java_net_ladenthin_llama_LlamaModel_requestChatCompletionStream(JNIEnv *env, jobject obj,
                                                                                       jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jint {
        REQUIRE_SERVER_CONTEXT(0);
        const std::string body = parse_jstring(env, jparams);
        int task_id = 0;
        const jllama_status status =
            jllama_chat_completions_stream_request(reinterpret_cast<jllama_model *>(jctx), body.c_str(), &task_id);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return 0;
        }
        return static_cast<jint>(task_id);
    });
}

JNIEXPORT jintArray JNICALL Java_net_ladenthin_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jintArray {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string text = parse_jstring(env, jprompt);
        int *tokens = nullptr;
        size_t count = 0;
        const jllama_status status = jllama_tokenize(api(jctx), text.c_str(), false, &tokens, &count);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        const std::vector<int32_t> ids(tokens, tokens + count);
        jllama_tokens_free(tokens);
        return tokens_to_jint_array_impl(env, ids, c_error_oom);
    });
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                             jintArray java_tokens) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jbyteArray {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::vector<int32_t> tokens = jint_array_to_tokens_impl(env, java_tokens);
        char *text = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_detokenize(api(jctx), tokens.data(), tokens.size(), &text, &length);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        const std::string bytes(text, length);
        jllama_string_free(text);
        return parse_jbytes(env, bytes);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_delete(JNIEnv *env, jobject obj) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
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

        // The rest -- closing flag, cancelling the readers, waiting for every in-flight call to
        // release its user reference, stopping the worker, freeing the model -- is the C API's
        // jllama_model_free, the same teardown a C caller gets.
        jllama_model_free(reinterpret_cast<jllama_model *>(jctx));
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
        REQUIRE_SERVER_CONTEXT();
        jllama_task_release(reinterpret_cast<jllama_model *>(jctx), static_cast<int>(id_task));
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject log_format,
                                                                     jobject jcallback) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
        // One swap at a time (see g_set_logger_mutex); the trampoline never takes this lock.
        std::lock_guard<std::mutex> swap_lock(g_set_logger_mutex);

        // The Java logger is a sink on common_log (patches/0014), not a llama_log_set() callback.
        // Every line reaches common_log -- the server's SRV_*/SLT_* macros write into it directly
        // and llama/ggml lines arrive through common_log_default_callback -- and common_init(),
        // which every model load runs, re-points llama_log_set() at that same default callback
        // without touching the sink. So the logger survives the load whichever order the caller
        // chose, and it sees the slot/srv lines llama_log_set() never carried.
        //
        // Step 1, outside the logger mutex: detach the current sink (jllama_set_log_callback(null)
        // -> the engine's set_log_sink). This pauses common_log's worker, which first drains every
        // queued entry through the OLD callback (so a caller that sets null gets a synchronous
        // flush) and then joins. The worker needs g_log_mutex to read the callback, so holding it
        // here would deadlock the join.
        jllama_set_log_callback(nullptr, nullptr);

        {
            // Step 2: swap the Java-side state. No new trampoline can start now (the sink is off);
            // still DRAIN any trampoline that copied the old std::function before that (g_log_active)
            // so the old global ref is never used after it is deleted.
            std::unique_lock<std::mutex> lk(g_log_mutex);
            log_callback = nullptr;
            g_log_cv.wait(lk, [] { return g_log_active == 0; });
            if (o_log_callback != nullptr) {
                env->DeleteGlobalRef(o_log_callback);
                o_log_callback = nullptr;
            }

            log_json = env->IsSameObject(log_format, o_log_format_json);

            if (jcallback != nullptr) {
                o_log_callback = env->NewGlobalRef(jcallback);
                // Capture copies of the global ref and method id so the callback never dereferences
                // the logger globals at call time (those may be swapped by a concurrent setLogger).
                jobject cb_ref = o_log_callback;
                log_callback = [cb_ref](enum ggml_log_level level, const char *text, void *user_data) noexcept {
                    // common_log delivers from its own worker thread, which is not attached to the
                    // JVM; it is attached once per worker thread and detached when that thread exits
                    // (see get_jni_env_attaching).
                    JNIEnv *env = get_jni_env_attaching();
                    if (env == nullptr || text == nullptr) {
                        return;
                    }
                    // The thread stays attached and has no Java frame, so local references would
                    // accumulate until it exits: scope every call's references to a local frame.
                    if (env->PushLocalFrame(4) != JNI_OK) {
                        env->ExceptionClear(); // out of memory; drop this log line
                        return;
                    }
                    // Log lines can embed payload text (prompts, model metadata), so the
                    // message must cross as standard UTF-8, not Modified UTF-8.
                    jstring message = utf8_to_jstring(env, text);
                    if (message == nullptr) {
                        env->ExceptionClear(); // allocation failed; drop this log line
                    } else {
                        jobject log_level = log_level_to_jobject(level);
                        env->CallVoidMethod(cb_ref, m_biconsumer_accept, log_level, message);
                        if (env->ExceptionCheck()) {
                            env->ExceptionClear(); // a throwing logger must not poison the worker
                        }
                    }
                    env->PopLocalFrame(nullptr);
                };
            }
        }

        // Step 3: install the sink through the C API (the JNI trampoline handles JSON formatting
        // internally; the engine routes llama/ggml lines into common_log even before the first model
        // load runs common_init()). A null callback leaves the sink detached: common_log prints to
        // the console again.
        if (jcallback != nullptr) {
            jllama_set_log_callback(jni_log_sink, nullptr);
        }
    });
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_LlamaModel_jsonSchemaToGrammarBytes(JNIEnv *env, jclass clazz,
                                                                                          jstring j_schema) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jbyteArray {
        (void)clazz;
        const std::string schema = parse_jstring(env, j_schema);
        char *grammar = nullptr;
        const jllama_status status = jllama_json_schema_to_grammar(schema.c_str(), &grammar);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        const std::string text(grammar);
        jllama_string_free(grammar);
        return parse_jbytes(env, text);
    });
}

// Returns llama.cpp's own build identifier ("b<number>-<commit>") from the linked-in build-info
// symbols. Pure-ASCII, so NewStringUTF is safe here (the modified-UTF-8 caveat only applies to
// model-generated payload text).
JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_nativeLlamaCppBuildInfo(JNIEnv *env, jclass clazz) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring { return env->NewStringUTF(jllama_build_info()); });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleCompletions(JNIEnv *env, jobject obj,
                                                                                jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *response = nullptr;
        const jllama_status status = jllama_completions(api(jctx), body.c_str(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleCompletionsOai(JNIEnv *env, jobject obj,
                                                                                   jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *response = nullptr;
        const jllama_status status = jllama_completions_oai(api(jctx), body.c_str(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleInfill(JNIEnv *env, jobject obj, jstring jparams) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *response = nullptr;
        const jllama_status status = jllama_infill(api(jctx), body.c_str(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleEmbeddings(JNIEnv *env, jobject obj,
                                                                               jstring jparams, jboolean joaiCompat) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string body = parse_jstring(env, jparams);
        char *response = nullptr;
        const jllama_status status = jllama_embeddings(api(jctx), body.c_str(), joaiCompat == JNI_TRUE, &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleTokenize(JNIEnv *env, jobject obj, jstring jcontent,
                                                                             jboolean jaddSpecial,
                                                                             jboolean jwithPieces) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string content = parse_jstring(env, jcontent);
        char *response = nullptr;
        const jllama_status status = jllama_tokenize_json(api(jctx), content.c_str(), jaddSpecial == JNI_TRUE,
                                                          jwithPieces == JNI_TRUE, &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleDetokenize(JNIEnv *env, jobject obj,
                                                                               jintArray jtokens) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::vector<int32_t> tokens = jint_array_to_tokens_impl(env, jtokens);
        char *response = nullptr;
        const jllama_status status = jllama_detokenize_json(api(jctx), tokens.data(), tokens.size(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_handleSlotAction(JNIEnv *env, jobject obj, jint action,
                                                                               jint slotId, jstring jfilename) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        char *response = nullptr;
        if (action == 0) { // LIST: the full server-introspection payload (getMetrics)
            const jllama_status status = jllama_metrics(api(jctx), &response);
            return api_string(env, status, response);
        }
        // 1 SAVE, 2 RESTORE, 3 ERASE: the values of jllama_slot_op; anything else is refused by
        // the C API ("Invalid slot action").
        const std::string filename = jfilename != nullptr ? parse_jstring(env, jfilename) : "";
        const jllama_status status = jllama_slot_action(api(jctx), static_cast<jllama_slot_op>(action),
                                                        static_cast<int>(slotId), filename.c_str(), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_getLoraAdaptersJson(JNIEnv *env, jobject obj) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        char *response = nullptr;
        const jllama_status status = jllama_lora_adapters(api(jctx), &response);
        return api_string(env, status, response);
    });
}

JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaModel_setLoraAdaptersJson(JNIEnv *env, jobject obj,
                                                                                  jstring jadapters) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jstring {
        REQUIRE_SERVER_CONTEXT(nullptr);
        const std::string adapters = parse_jstring(env, jadapters);
        char *response = nullptr;
        const jllama_status status = jllama_set_lora_adapters(api(jctx), adapters.c_str(), &response);
        return api_string(env, status, response);
    });
}

// LlamaQuantizer is not part of the javac-generated jllama.h (that header only covers LlamaModel),
// so nothing else gives this definition C linkage and it would be exported under its C++-mangled
// name -- which the JVM cannot resolve, failing every call with UnsatisfiedLinkError on every
// platform. Same reason train_engine.cpp marks LlamaTrainer_finetuneNative and native_server.cpp
// wraps the NativeServer entry points in an extern "C" block.
extern "C" JNIEXPORT void JNICALL Java_net_ladenthin_llama_LlamaQuantizer_quantizeNative(
    JNIEnv *env, jclass, jstring jinput, jstring joutput, jint ftype, jint nthread, jboolean allowRequantize) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
        const std::string input_path = parse_jstring(env, jinput);
        const std::string output_path = parse_jstring(env, joutput);
        const jllama_status status = jllama_quantize(input_path.c_str(), output_path.c_str(), static_cast<int>(ftype),
                                                     static_cast<int>(nthread), allowRequantize == JNI_TRUE);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
        }
    });
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_LlamaModel_configureParallelInference(JNIEnv *env, jobject obj,
                                                                                          jstring jconfig) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jboolean {
        REQUIRE_SERVER_CONTEXT(JNI_FALSE);
        const std::string config = parse_jstring(env, jconfig);
        const jllama_status status = jllama_configure(api(jctx), config.c_str());
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return JNI_FALSE;
        }
        return JNI_TRUE;
    });
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
    return jni_guard_impl(env, c_llama_error, [&]() -> jlong {
        (void)clazz;
        const std::string model = parse_jstring(env, jmodel);
        const std::string mmproj = parse_jstring(env, jmmproj);
        jllama_tts_model *tts = nullptr;
        const jllama_status status = jllama_tts_load(model.c_str(), mmproj.c_str(), static_cast<int>(gpu_layers),
                                                     static_cast<int>(threads), &tts);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return 0;
        }
        return reinterpret_cast<jlong>(tts);
    });
}

JNIEXPORT jbyteArray JNICALL Java_net_ladenthin_llama_TextToSpeech_synthesizeNative(JNIEnv *env, jclass clazz,
                                                                                    jlong handle, jstring jtext,
                                                                                    jstring jspeaker_reference_path,
                                                                                    jstring jlang, jint max_frames,
                                                                                    jint top_k, jint seed) {
    return jni_guard_impl(env, c_llama_error, [&]() -> jbyteArray {
        (void)clazz;
        const std::string text = parse_jstring(env, jtext);
        const std::string speaker_reference_path =
            jspeaker_reference_path == nullptr ? std::string() : parse_jstring(env, jspeaker_reference_path);
        const std::string lang = jlang == nullptr ? std::string() : parse_jstring(env, jlang);
        unsigned char *wav = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize(
            reinterpret_cast<jllama_tts_model *>(handle), text.c_str(), speaker_reference_path.c_str(), lang.c_str(),
            static_cast<int>(max_frames), static_cast<int>(top_k), static_cast<unsigned int>(seed), &wav, &length);
        if (status != JLLAMA_STATUS_OK) {
            throw_api_status(env, status);
            return nullptr;
        }
        jbyteArray out = env->NewByteArray(static_cast<jsize>(length));
        if (out == nullptr) {
            jllama_bytes_free(wav);
            env->ThrowNew(c_error_oom, "could not allocate WAV byte array");
            return nullptr;
        }
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(length), reinterpret_cast<const jbyte *>(wav));
        jllama_bytes_free(wav);
        return out;
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_TextToSpeech_deleteNative(JNIEnv *env, jclass clazz, jlong handle) {
    return jni_guard_impl(env, c_llama_error, [&]() -> void {
        (void)env;
        (void)clazz;
        jllama_tts_free(reinterpret_cast<jllama_tts_model *>(handle));
    });
}

} // extern "C"
