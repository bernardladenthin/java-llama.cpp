#pragma once

// jni_helpers.hpp — JNI bridge helpers for jllama.cpp.
//
// Two layers live here:
//
//   Layer A — JNI handle management:
//     jllama_context struct, get_server_context_impl, get_jllama_context_impl,
//     require_json_field_impl, jint_array_to_tokens_impl
//
//   Layer B — JNI + server orchestration:
//     json_to_jstring_impl, results_to_jstring_impl,
//     embedding_to_jfloat_array_impl, tokens_to_jint_array_impl
//
// Pure JSON transforms (no JNI, no llama state) live in json_helpers.hpp,
// which is included at the bottom of this file.
//
// Include order: upstream server headers (server-context.h, server-queue.h,
// server-task.h, server-common.h, server-chat.h) must be included by the
// including translation unit BEFORE this header.

#include "jni.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations.
struct server_context;
struct server_response_reader;

// ===========================================================================
// Layer A — JNI handle management
// ===========================================================================

// ---------------------------------------------------------------------------
// jllama_context
//
// Owns a server_context (value member, pimpl inside) and the background
// worker thread.  Stored as the Java-side `ctx` (jlong) pointer.
// ---------------------------------------------------------------------------
struct jllama_context {
    server_context    server;                 // value member (pimpl inside)
    std::thread       worker;
    bool              vocab_only        = false;
    std::atomic<bool> worker_ready{false};

    // Cached after load_model() — valid for the lifetime of this context.
    const llama_vocab *vocab             = nullptr;
    // Non-null only in vocab-only mode (bypasses server_context entirely).
    llama_model       *vocab_only_model  = nullptr;

    // Saved copy of common_params used to load the model.
    // Required by server_task::params_from_json_cmpl which takes common_params&.
    common_params      params;

    // Per-streaming-task response readers, keyed by task id.
    // Guarded by readers_mutex.
    std::mutex         readers_mutex;
    std::map<int, std::unique_ptr<server_response_reader>> readers;
};

// ---------------------------------------------------------------------------
// get_server_context_impl
//
// Reads the native handle stored in the Java LlamaModel object, validates it,
// and returns a pointer to the embedded server_context value member.
//
// On success: returns a non-null server_context*.
// On failure: throws "Model is not loaded" via JNI and returns nullptr.
// ---------------------------------------------------------------------------
[[nodiscard]] inline server_context *get_server_context_impl(JNIEnv   *env,
                                                              jobject   obj,
                                                              jfieldID  field_id,
                                                              jclass    error_class) {
    const jlong handle = env->GetLongField(obj, field_id);
    if (handle == 0) {
        env->ThrowNew(error_class, "Model is not loaded");
        return nullptr;
    }
    return &(reinterpret_cast<jllama_context *>(handle)->server); // NOLINT(*-no-int-to-ptr)
}

// ---------------------------------------------------------------------------
// get_jllama_context_impl
//
// Like get_server_context_impl but returns the jllama_context wrapper itself.
// Used ONLY by the delete path and methods that need jctx directly.
//
// Intentionally does NOT throw on null: a zero handle means the model was
// already deleted (or never fully initialised), which is a valid no-op for
// a destructor-style call.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jllama_context *get_jllama_context_impl(JNIEnv   *env,
                                                              jobject   obj,
                                                              jfieldID  field_id) {
    const jlong handle = env->GetLongField(obj, field_id);
    if (handle == 0) {
        return nullptr;
    }
    return reinterpret_cast<jllama_context *>(handle); // NOLINT(*-no-int-to-ptr)
}

// ---------------------------------------------------------------------------
// require_json_field_impl
//
// Checks that `data` contains the given key.  Returns true if present.
// On missing key: throws "<field> is required" via JNI and returns false.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool require_json_field_impl(JNIEnv               *env,
                                                   const nlohmann::json &data,
                                                   const char           *field,
                                                   jclass                error_class) {
    if (data.contains(field)) {
        return true;
    }
    const std::string msg = std::string("\"") + field + "\" is required";
    env->ThrowNew(error_class, msg.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// jint_array_to_tokens_impl
//
// Reads a Java int array into a std::vector<int32_t> and releases the JNI
// array elements with JNI_ABORT (read-only — no writeback needed).
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::vector<int32_t> jint_array_to_tokens_impl(
        JNIEnv *env, jintArray array) {
    const jsize length = env->GetArrayLength(array);
    jint *elements     = env->GetIntArrayElements(array, nullptr);
    std::vector<int32_t> tokens(elements, elements + length);
    env->ReleaseIntArrayElements(array, elements, JNI_ABORT);
    return tokens;
}

// ===========================================================================
// Layer B — JNI + server orchestration
// (server.hpp must be included by the TU before this header)
// ===========================================================================

// json_helpers.hpp provides get_result_error_message, results_to_json, and
// the other pure JSON transforms used by the functions below.
#include "json_helpers.hpp"

// ---------------------------------------------------------------------------
// json_to_jstring_impl
//
// Serialises any json value to a JNI string via dump() + NewStringUTF.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring json_to_jstring_impl(JNIEnv *env, const json &j) {
    std::string s = j.dump();
    return env->NewStringUTF(s.c_str());
}

// ---------------------------------------------------------------------------
// results_to_jstring_impl
//
// Serialises a vector of task results to a jstring by delegating JSON
// construction to results_to_json (json_helpers.hpp) and serialisation to
// json_to_jstring_impl.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring results_to_jstring_impl(
        JNIEnv                                    *env,
        const std::vector<server_task_result_ptr> &results) {
    return json_to_jstring_impl(env, results_to_json(results));
}

// ---------------------------------------------------------------------------
// vec_to_jarray_impl
//
// Generic helper: converts a C++ vector to a JNI primitive array.
// Parameterized on JNI array/element types and the alloc/copy member fns.
// On allocation failure: throws via JNI with oom_class and returns nullptr.
// ---------------------------------------------------------------------------
template <typename JArray, typename JElem, typename CppElem>
[[nodiscard]] inline JArray vec_to_jarray_impl(
        JNIEnv                     *env,
        const std::vector<CppElem> &values,
        jclass                      oom_class,
        const char                 *oom_msg,
        JArray (JNIEnv_::*alloc)(jsize),
        void (JNIEnv_::*copy)(JArray, jsize, jsize, const JElem *)) {
    const jsize len = static_cast<jsize>(values.size());
    JArray arr = (env->*alloc)(len);
    if (arr == nullptr) {
        env->ThrowNew(oom_class, oom_msg);
        return nullptr;
    }
    (env->*copy)(arr, 0, len, reinterpret_cast<const JElem *>(values.data()));
    return arr;
}

// Converts a float vector to a Java jfloatArray.
[[nodiscard]] inline jfloatArray embedding_to_jfloat_array_impl(
        JNIEnv                   *env,
        const std::vector<float> &values,
        jclass                    oom_class) {
    return vec_to_jarray_impl<jfloatArray, jfloat>(
            env, values, oom_class, "could not allocate embedding",
            &JNIEnv_::NewFloatArray, &JNIEnv_::SetFloatArrayRegion);
}

// Converts a token vector to a Java jintArray.
[[nodiscard]] inline jintArray tokens_to_jint_array_impl(
        JNIEnv                       *env,
        const std::vector<int32_t>   &tokens,
        jclass                        oom_class) {
    return vec_to_jarray_impl<jintArray, jint>(
            env, tokens, oom_class, "could not allocate token memory",
            &JNIEnv_::NewIntArray, &JNIEnv_::SetIntArrayRegion);
}
