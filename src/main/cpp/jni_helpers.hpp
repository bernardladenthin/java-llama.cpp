#pragma once

// jni_helpers.hpp — JNI bridge helpers for jllama.cpp.
//
// This file is the single project-side helper header for all JNI bridge code.
// It was formed by merging the former jni_helpers.hpp (handle management) and
// the former jni_server_helpers.hpp (server orchestration) into one coherent file.
//
// Two layers live here:
//
//   Layer A — JNI handle management (no server.hpp required):
//     jllama_context struct, get_server_context_impl, get_jllama_context_impl,
//     require_single_task_id_impl, require_json_field_impl,
//     jint_array_to_tokens_impl
//
//   Layer B — JNI + server orchestration (server.hpp must precede this header):
//     json_to_jstring_impl, results_to_jstring_impl,
//     build_completion_tasks_impl, recv_slot_task_result_impl,
//     collect_task_results_impl, check_infill_support_impl, append_task
//
// Pure JSON transforms (no JNI, no llama state) live in json_helpers.hpp,
// which is included at the bottom of this file so all bridge helpers can
// call them directly.
//
// IMPORTANT — include order for Layer B:
//   server.hpp must be included by the including translation unit BEFORE this
//   header.  server.hpp has no include guard, so including it here would cause
//   redefinition errors in any TU that already includes server.hpp directly.
//
// All parameters are passed explicitly (no module-level globals) so every
// function can be exercised in unit tests using a mock JNIEnv.
//
// Declaration order (each function must be defined before its first caller):
//   Layer A:
//     1.  jllama_context struct
//     2.  get_server_context_impl
//     3.  get_jllama_context_impl
//     4.  require_single_task_id_impl
//     5.  require_json_field_impl
//     6.  jint_array_to_tokens_impl
//   Layer B (needs server.hpp in TU):
//     7.  json_to_jstring_impl
//     8.  build_completion_tasks_impl
//     9.  recv_slot_task_result_impl     — uses get_result_error_message (json_helpers), json_to_jstring_impl
//    10.  collect_task_results_impl      — uses get_result_error_message (json_helpers)
//    11.  results_to_jstring_impl        — uses results_to_json (json_helpers), json_to_jstring_impl
//    12.  check_infill_support_impl
//    13.  append_task
//    14.  embedding_to_jfloat_array_impl
//    15.  tokens_to_jint_array_impl

#include "jni.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// Forward declaration — Layer A helpers only hold/cast pointers to
// server_context; they never dereference it, so a full definition is not
// needed here.  TUs that call Layer B functions must include server.hpp first.
struct server_context;

// ===========================================================================
// Layer A — JNI handle management
// ===========================================================================

// ---------------------------------------------------------------------------
// jllama_context
//
// Owns a server_context and the background worker thread.  Stored as the
// Java-side `ctx` (jlong) pointer.  Using a wrapper allows us to join the
// thread on close() instead of detaching it, which eliminates the race
// between thread teardown and JVM shutdown.
// ---------------------------------------------------------------------------
struct jllama_context {
    server_context *server     = nullptr;
    std::thread     worker;
    bool            vocab_only = false;
    // Signals that the worker thread has entered start_loop() and is ready.
    // Without this, terminate() can race with start_loop() setting running=true.
    std::atomic<bool> worker_ready{false};
};

// ---------------------------------------------------------------------------
// get_server_context_impl
//
// Reads the native handle stored in the Java LlamaModel object, validates it,
// and returns the embedded server_context pointer.
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
    return reinterpret_cast<jllama_context *>(handle)->server; // NOLINT(*-no-int-to-ptr)
}

// ---------------------------------------------------------------------------
// get_jllama_context_impl
//
// Like get_server_context_impl but returns the jllama_context wrapper itself.
// Used ONLY by the delete path, which must call `delete jctx`.
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
// require_single_task_id_impl
//
// Validates that exactly one task was created after dispatch and returns its
// ID.  Returns 0 (with a JNI exception pending) when the count is not 1.
// ---------------------------------------------------------------------------
[[nodiscard]] inline int require_single_task_id_impl(
        JNIEnv                        *env,
        const std::unordered_set<int> &task_ids,
        jclass                         error_class) {
    if (task_ids.size() != 1) {
        env->ThrowNew(error_class, "multitasking currently not supported");
        return 0;
    }
    return *task_ids.begin();
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
// build_completion_tasks_impl
//
// Reads data["prompt"], tokenises it, and appends one server_task per prompt
// token sequence to `tasks`.  task_type and oaicompat are caller-specified.
//
// IMPORTANT: data["prompt"] is read before any ctx_server member is accessed,
// so passing ctx_server=nullptr is safe in tests that exercise the error path
// (missing "prompt" key).
//
// On success: `tasks` is populated, returns true.
// On error:   throws via JNI using error_class, returns false.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool build_completion_tasks_impl(
        JNIEnv                   *env,
        server_context           *ctx_server,
        const json               &data,
        const std::string        &completion_id,
        server_task_type          task_type,
        oaicompat_type            oaicompat,
        std::vector<server_task> &tasks,
        jclass                    error_class) {
    try {
        const auto &prompt = data.at("prompt"); // throws before ctx_server is touched

        std::vector<llama_tokens> tokenized_prompts =
            tokenize_input_prompts(ctx_server->vocab, prompt, true, true);

        tasks.reserve(tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(task_type);
            task.id    = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens    = server_tokens(tokenized_prompts[i], false);
            task.params           = server_task::params_from_json_cmpl(
                                        ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            task.params.oaicompat         = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception &e) {
        const auto &err = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
        env->ThrowNew(error_class, err.dump().c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// recv_slot_task_result_impl
//
// Receives a single slot-action result from the response queue, checks for
// an error, and returns the result JSON as a JNI string.
//
// On success: returns a new jstring containing result->to_json().dump().
// On error:   removes the waiting task id, throws via JNI, returns nullptr.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring recv_slot_task_result_impl(JNIEnv          *env,
                                                         server_response &queue,
                                                         int              task_id,
                                                         jclass           error_class) {
    server_task_result_ptr result = queue.recv(task_id);
    queue.remove_waiting_task_id(task_id);
    if (result->is_error()) {
        env->ThrowNew(error_class, get_result_error_message(result).c_str());
        return nullptr;
    }
    return json_to_jstring_impl(env, result->to_json());
}

// ---------------------------------------------------------------------------
// collect_task_results_impl
//
// Precondition: each ID in task_ids has already been registered with
//   queue.add_waiting_task_id() (or add_waiting_tasks()).
//
// On success: appends all results to `out`, removes waiting ids, returns true.
// On error:   removes waiting ids, throws via JNI, returns false.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool collect_task_results_impl(
        JNIEnv                               *env,
        server_response                      &queue,
        const std::unordered_set<int>        &task_ids,
        std::vector<server_task_result_ptr>  &out,
        jclass                                error_class) {
    out.reserve(task_ids.size());
    for (size_t i = 0; i < task_ids.size(); i++) {
        server_task_result_ptr result = queue.recv(task_ids);
        if (result->is_error()) {
            queue.remove_waiting_task_ids(task_ids);
            env->ThrowNew(error_class, get_result_error_message(result).c_str());
            return false;
        }
        out.push_back(std::move(result));
    }
    queue.remove_waiting_task_ids(task_ids);
    return true;
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
// check_infill_support_impl
//
// Checks that the model vocabulary has all three fill-in-the-middle (FIM)
// tokens (prefix, suffix, middle).  Returns true if infill is supported.
// On failure: throws via JNI and returns false.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool check_infill_support_impl(JNIEnv            *env,
                                                     const llama_vocab *vocab,
                                                     jclass             error_class) {
    std::string err;
    if (llama_vocab_fim_pre(vocab) == LLAMA_TOKEN_NULL) { err += "prefix token is missing. "; }
    if (llama_vocab_fim_suf(vocab) == LLAMA_TOKEN_NULL) { err += "suffix token is missing. "; }
    if (llama_vocab_fim_mid(vocab) == LLAMA_TOKEN_NULL) { err += "middle token is missing. "; }
    if (!err.empty()) {
        env->ThrowNew(error_class, ("Infill is not supported by this model: " + err).c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// append_task
//
// Constructs a server_task of the given type and appends it to `tasks`.
// The caller is responsible for pre-computing `prompt_tokens`.
// `oaicompat` defaults to NONE so rerank call sites need no explicit argument.
// ---------------------------------------------------------------------------
inline void append_task(server_context           *ctx_server,
                        std::vector<server_task> &tasks,
                        server_task_type          type,
                        llama_tokens              prompt_tokens,
                        size_t                    index,
                        oaicompat_type            oaicompat = OAICOMPAT_TYPE_NONE) {
    server_task task(type);
    task.id               = ctx_server->queue_tasks.get_new_id();
    task.index            = index;
    task.prompt_tokens    = server_tokens(prompt_tokens, false);
    task.params.oaicompat = oaicompat;
    tasks.push_back(std::move(task));
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
