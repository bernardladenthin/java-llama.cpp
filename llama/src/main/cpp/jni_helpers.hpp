// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// jni_helpers.hpp — JNI bridge helpers for jllama.cpp.
//
// Two layers live here:
//
//   Layer A — JNI handle management:
//     get_jllama_context_impl, acquire_jllama_context_impl (the Java-handle side of the
//     user-reference protocol whose JNI-free half is jllama_context.hpp),
//     require_json_field_impl, jint_array_to_tokens_impl
//
//   Layer B — JNI + server orchestration:
//     utf8_to_jstring_impl, json_to_jstring_impl, results_to_jstring_impl,
//     embedding_to_jfloat_array_impl, tokens_to_jint_array_impl
//
// Pure JSON transforms (no JNI, no llama state) live in json_helpers.hpp,
// which is included at the bottom of this file.
//
// Include order: upstream server headers (server-context.h, server-queue.h,
// server-task.h, server-common.h, server-chat.h) must be included by the
// including translation unit BEFORE this header.

#include "jni.h"
// The JNI-free half of layer A -- jllama_context, erase_reader, the user-reference release and
// its guard -- lives in jllama_context.hpp, and the task helpers that used to sit in layer B here
// (configure_multimodal_task_impl, configure_task_slot_impl) in jllama_engine.hpp, both shared with
// the C API (jllama_api.cpp) and the engine (jllama_engine.cpp); only what reads a Java object
// stays here.
#include "jllama_engine.hpp"

#include <string>
#include <type_traits>
#include <vector>

// ===========================================================================
// Layer A — JNI handle management
// ===========================================================================

// Guard: throw and return false if the model was loaded without embedding
// support enabled. Used by every JNI entry point that produces embeddings.
[[nodiscard]] inline bool require_embedding_support(JNIEnv *env, bool embedding_enabled, jclass error_class) {
    if (embedding_enabled) {
        return true;
    }
    env->ThrowNew(error_class,
                  "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
    return false;
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
[[nodiscard]] inline jllama_context *get_jllama_context_impl(JNIEnv *env, jobject obj, jfieldID field_id) {
    const jlong handle = env->GetLongField(obj, field_id);
    if (handle == 0) {
        return nullptr;
    }
    return reinterpret_cast<jllama_context *>(handle); // NOLINT(*-no-int-to-ptr)
}

// ---------------------------------------------------------------------------
// acquire / release_jllama_context_impl  +  jllama_context_guard
//
// Reference-count in-flight JNI calls against the jllama_context so that
// close() can wait for all of them to drain before deleting the context. Every
// entry point that dereferences jctx goes through REQUIRE_SERVER_CONTEXT, which
// constructs a jllama_context_guard; the guard's destructor calls
// release_jllama_context_impl (even on early return), decrementing users.
//
// close() (delete()) first nulls the Java handle under g_ctx_mutex so no NEW
// acquire can succeed, sets the closing flag so parked readers unwind, waits on
// jctx->cv for users to reach 0, and only then tears down the worker/models and
// frees the context — guaranteeing no call is mid-use during any part of the
// teardown. g_ctx_mutex is defined in jllama.cpp and declared extern here.
// ---------------------------------------------------------------------------
extern std::mutex g_ctx_mutex;

[[nodiscard]] inline jllama_context *acquire_jllama_context_impl(JNIEnv *env, jobject obj, jfieldID field_id) {
    std::lock_guard<std::mutex> lk(g_ctx_mutex);
    const jlong handle = env->GetLongField(obj, field_id);
    if (handle == 0) {
        return nullptr;
    }
    auto *jctx = reinterpret_cast<jllama_context *>(handle); // NOLINT(*-no-int-to-ptr)
    jctx->users.fetch_add(1, std::memory_order_relaxed);
    return jctx;
}

// ---------------------------------------------------------------------------
// require_json_field_impl
//
// Checks that `data` contains the given key.  Returns true if present.
// On missing key: throws "<field> is required" via JNI and returns false.
//
// Templated on the JSON type so the upstream `json` alias binds directly.  Since llama.cpp
// b10585 (upstream #27511) that alias is `common_json`, not `nlohmann::ordered_json`, and a
// non-template `const nlohmann::json &` parameter would still COMPILE for it — silently, via
// common_json::operator std::string() feeding nlohmann's string-constructible converting
// constructor — turning the presence check into a `json::type_error 302` thrown out of the JNI
// frame.  Keep this a template.
// ---------------------------------------------------------------------------
template <typename JsonT>
[[nodiscard]] inline bool require_json_field_impl(JNIEnv *env, const JsonT &data, const char *field,
                                                  jclass error_class) {
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
[[nodiscard]] inline std::vector<int32_t> jint_array_to_tokens_impl(JNIEnv *env, jintArray array) {
    const jsize length = env->GetArrayLength(array);
    jint *elements = env->GetIntArrayElements(array, nullptr);
    std::vector<int32_t> tokens(elements, elements + length);
    env->ReleaseIntArrayElements(array, elements, JNI_ABORT);
    return tokens;
}

// ---------------------------------------------------------------------------
// report_cxx_exception_impl / jni_guard_impl — the JNI exception boundary
//
// An exception that escapes a native method and unwinds across the JNI boundary
// is undefined behaviour and aborts the JVM on most implementations. Every
// `Java_*` entry point in this project therefore runs its body inside
// jni_guard_impl, which converts anything that escapes into a Java exception and
// returns a zero/nullptr sentinel.
//
// The guard is ADDITIVE: an entry point that already converts std::exception
// itself keeps doing so and never reaches the handlers here. What the guard adds
// everywhere is the `catch (...)` arm — the case for an exception type not
// derived from std::exception, which otherwise has no backstop at all.
//
// Two rules the handlers must keep:
//   * Never call ThrowNew while a Java exception is already pending. The JNI
//     spec forbids most calls in that state, and the pending exception is the
//     more precise error anyway — so it is left in place.
//   * Never call ThrowNew with a null class. JNI_OnLoad has not cached the
//     exception class yet and JNI_OnUnload has already released it, so those two
//     carry their own local handlers rather than routing through here.
// ---------------------------------------------------------------------------
inline void report_cxx_exception_impl(JNIEnv *env, jclass exception_class, const char *what) {
    if (env->ExceptionCheck()) {
        return;
    }
    if (exception_class == nullptr) {
        return;
    }
    env->ThrowNew(exception_class, what);
}

template <typename Fn> auto jni_guard_impl(JNIEnv *env, jclass exception_class, Fn &&fn) -> decltype(fn()) {
    using result_type = decltype(fn());
    try {
        return fn();
    } catch (const std::exception &e) {
        report_cxx_exception_impl(env, exception_class, e.what());
    } catch (...) {
        report_cxx_exception_impl(env, exception_class, "unknown C++ exception crossed the JNI boundary");
    }
    if constexpr (!std::is_void<result_type>::value) {
        return result_type{};
    }
}

// ===========================================================================
// Layer B — JNI + server orchestration
// (upstream server headers must be included by the TU before this header)
// ===========================================================================

// json_helpers.hpp provides get_result_error_message, results_to_json, and
// the other pure JSON transforms used by the functions below.
#include "json_helpers.hpp"

// ---------------------------------------------------------------------------
// utf8_to_jstring_impl
//
// Builds a java.lang.String from raw standard-UTF-8 bytes via the
// String(byte[], String charsetName) constructor. NewStringUTF must NOT be
// used for payload text: JNI specifies *Modified* UTF-8 input, where
// supplementary-plane characters (e.g. every 4-byte emoji sequence) are
// encoded as CESU-8 surrogate pairs — standard UTF-8 payloads containing
// them are spec-invalid (Android CheckJNI aborts, HotSpot mangles). Routing
// through byte[] + charset keeps the bytes standard UTF-8 end to end and is
// the mirror of parse_jstring's String.getBytes("UTF-8") input path.
//
// string_init_bytes_charset is the cached method id of
// String(byte[], String) and charset_name a cached "UTF-8" jstring.
// Returns nullptr with a pending OOM exception if the byte array cannot be
// allocated.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring utf8_to_jstring_impl(JNIEnv *env, const std::string &s, jclass string_class,
                                                  jmethodID string_init_bytes_charset, jobject charset_name) {
    const jsize length = static_cast<jsize>(s.size());
    jbyteArray bytes = env->NewByteArray(length);
    if (bytes == nullptr) {
        return nullptr; // OOM exception already pending
    }
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte *>(s.data()));
    auto result = (jstring)env->NewObject(string_class, string_init_bytes_charset, bytes, charset_name);
    env->DeleteLocalRef(bytes);
    return result;
}

// ---------------------------------------------------------------------------
// json_to_jstring_impl
//
// Serialises any json value to a JNI string. Serialisation goes through
// upstream safe_json_to_str (dump with error_handler_t::replace) so a
// payload string that ends in an incomplete UTF-8 sequence — possible on the
// non-stream path when generation stops mid-codepoint at the token limit —
// is replaced with U+FFFD instead of throwing json::type_error 316. The
// resulting UTF-8 bytes cross into Java via utf8_to_jstring_impl.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring json_to_jstring_impl(JNIEnv *env, const json &j, jclass string_class,
                                                  jmethodID string_init_bytes_charset, jobject charset_name) {
    return utf8_to_jstring_impl(env, safe_json_to_str(j), string_class, string_init_bytes_charset, charset_name);
}

// ---------------------------------------------------------------------------
// results_to_jstring_impl
//
// Serialises a vector of task results to a jstring by delegating JSON
// construction to results_to_json (json_helpers.hpp) and serialisation to
// json_to_jstring_impl.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring results_to_jstring_impl(JNIEnv *env, const std::vector<server_task_result_ptr> &results,
                                                     jclass string_class, jmethodID string_init_bytes_charset,
                                                     jobject charset_name) {
    return json_to_jstring_impl(env, results_to_json(results), string_class, string_init_bytes_charset, charset_name);
}

// ---------------------------------------------------------------------------
// vec_to_jarray_impl
//
// Generic helper: converts a C++ vector to a JNI primitive array.
// Parameterized on JNI array/element types and the alloc/copy member fns.
// On allocation failure: throws via JNI with oom_class and returns nullptr.
// ---------------------------------------------------------------------------
template <typename JArray, typename JElem, typename CppElem>
[[nodiscard]] inline JArray vec_to_jarray_impl(JNIEnv *env, const std::vector<CppElem> &values, jclass oom_class,
                                               const char *oom_msg, JArray (JNIEnv_::*alloc)(jsize),
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
[[nodiscard]] inline jfloatArray embedding_to_jfloat_array_impl(JNIEnv *env, const std::vector<float> &values,
                                                                jclass oom_class) {
    return vec_to_jarray_impl<jfloatArray, jfloat>(env, values, oom_class, "could not allocate embedding",
                                                   &JNIEnv_::NewFloatArray, &JNIEnv_::SetFloatArrayRegion);
}

// Converts a token vector to a Java jintArray.
[[nodiscard]] inline jintArray tokens_to_jint_array_impl(JNIEnv *env, const std::vector<int32_t> &tokens,
                                                         jclass oom_class) {
    return vec_to_jarray_impl<jintArray, jint>(env, tokens, oom_class, "could not allocate token memory",
                                               &JNIEnv_::NewIntArray, &JNIEnv_::SetIntArrayRegion);
}
