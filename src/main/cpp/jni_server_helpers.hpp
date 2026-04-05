#pragma once

// jni_server_helpers.hpp — JNI helpers that need server.hpp types.
//
// Kept separate from jni_helpers.hpp intentionally: jni_helpers.hpp has a
// deliberately minimal include surface (only jni.h + stdlib) so it can be
// unit-tested without the full llama.cpp stack.  Any helper that must reach
// into server.hpp types belongs here instead.
//
// Public entry points:
//   build_completion_tasks_impl  — tokenise and build a server_task vector
//   collect_task_results_impl    — drain results from the response queue
//   recv_slot_task_result_impl   — recv + check a single slot-action result
//   append_task                  — construct and push a single server_task
//
// All parameters are explicit (no module-level globals) so each function can
// be exercised in unit tests using local server objects and a mock JNIEnv.
//
// IMPORTANT — include order:
//   server.hpp must be included by the including translation unit BEFORE this
//   header.  server.hpp has no include guard, so including it here would cause
//   redefinition errors in any TU that already includes server.hpp directly.
//
// Declaration order (each function must be defined before its first caller):
//   1. get_result_error_message    — used by recv_slot_task_result_impl,
//                                    collect_task_results_impl
//   2. json_to_jstring_impl        — used by recv_slot_task_result_impl,
//                                    results_to_jstring_impl
//   3. build_completion_tasks_impl — no dependencies on helpers above
//   4. recv_slot_task_result_impl  — uses 1 + 2
//   5. collect_task_results_impl   — uses 1
//   6. results_to_json_impl        — no dependencies on helpers above
//   7. results_to_jstring_impl     — uses 2 + 6
//   8. check_infill_support_impl   — no dependencies on helpers above
//   9. append_task                 — no dependencies on helpers above

#include "jni.h"

#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// get_result_error_message
//
// Extracts the human-readable error string from a failed task result.
// Equivalent to result->to_json()["message"].get<std::string>(), which
// appears verbatim in five places:
//
//   receiveCompletionJson, embed, handleRerank   (in jllama.cpp)
//   collect_task_results_impl, recv_slot_task_result_impl  (in this header)
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string get_result_error_message(
        const server_task_result_ptr &result) {
    return result->to_json()["message"].get<std::string>();
}

// ---------------------------------------------------------------------------
// json_to_jstring_impl
//
// Serialises any json value to a JNI string via dump() + NewStringUTF.
// Extracted from the repeated two-line pattern:
//
//   std::string response_str = some_json.dump();
//   return env->NewStringUTF(response_str.c_str());
//
// Used by recv_slot_task_result_impl, results_to_jstring_impl,
// receiveCompletionJson, handleRerank, handleEmbeddings,
// handleTokenize, and handleDetokenize.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring json_to_jstring_impl(JNIEnv *env, const json &j) {
    std::string s = j.dump();
    return env->NewStringUTF(s.c_str());
}

// ---------------------------------------------------------------------------
// build_completion_tasks_impl
//
// Reads data["prompt"], tokenises it, and appends one server_task per prompt
// token sequence to `tasks`.  task_type and oaicompat are caller-specified,
// covering all six JNI call sites:
//   requestCompletion       → COMPLETION or INFILL / NONE  (type from caller)
//   handleCompletions       → COMPLETION / NONE
//   handleCompletionsOai    → COMPLETION / COMPLETION
//   handleChatCompletions   → COMPLETION / CHAT
//   requestChatCompletion   → COMPLETION / NONE  (template already applied)
//   handleInfill            → INFILL     / NONE
//
// IMPORTANT: data["prompt"] is read in its own statement before any
// ctx_server member is accessed, so passing ctx_server=nullptr is safe in
// tests that only exercise the error path (missing "prompt" key).
//
// On success: `tasks` is populated, returns true.
// On error:   throws via JNI using error_class, returns false.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool build_completion_tasks_impl(JNIEnv                   *env,
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
// Used by all four handleSlotAction switch cases (LIST / SAVE / RESTORE /
// ERASE).  The caller is responsible for constructing the task, registering
// the task ID with queue.add_waiting_task_id(), and posting it to the task
// queue; this helper only covers the recv → check → return leg.
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
//   queue.add_waiting_task_id() (or add_waiting_tasks()) so that
//   remove_waiting_task_ids() performs correct cleanup.
//
// On success: appends all results to `out`, removes waiting ids, returns true.
// On error:   removes waiting ids, throws via JNI using error_class,
//             returns false.  The caller must return nullptr (or equivalent
//             sentinel) immediately — the JNI exception is already pending.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool collect_task_results_impl(JNIEnv                               *env,
                                       server_response                     &queue,
                                       const std::unordered_set<int>       &task_ids,
                                       std::vector<server_task_result_ptr> &out,
                                       jclass                               error_class) {
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
// results_to_json_impl
//
// Converts a vector of task results to a json value without touching JNI.
//
// When there is exactly one result, the top-level JSON is that result's object
// directly.  When there are multiple results, they are wrapped in a JSON array.
// This mirrors the OpenAI API convention used by handleCompletions,
// handleCompletionsOai, handleChatCompletions, and handleInfill.
//
// Separated from results_to_jstring_impl so the construction logic is
// unit-testable without any JNI mock.  The caller is responsible for checking
// that `results` is non-empty before calling (an empty vector produces an
// empty JSON array).
// ---------------------------------------------------------------------------
[[nodiscard]] inline json results_to_json_impl(
        const std::vector<server_task_result_ptr> &results) {
    if (results.size() == 1) {
        return results[0]->to_json();
    }
    json arr = json::array();
    for (const auto &res : results) {
        arr.push_back(res->to_json());
    }
    return arr;
}

// ---------------------------------------------------------------------------
// results_to_jstring_impl
//
// Serialises a vector of task results to a jstring by delegating JSON
// construction to results_to_json_impl and serialisation to
// json_to_jstring_impl.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring results_to_jstring_impl(
        JNIEnv *env,
        const std::vector<server_task_result_ptr> &results) {
    return json_to_jstring_impl(env, results_to_json_impl(results));
}

// ---------------------------------------------------------------------------
// check_infill_support_impl
//
// Checks that the model vocabulary has all three fill-in-the-middle (FIM)
// tokens (prefix, suffix, middle).  Returns true if infill is supported.
// On failure: populates a descriptive error message, throws via JNI using
// error_class, and returns false.
//
// Extracted from the 10-line compatibility block in handleInfill so it can
// be unit-tested independently of the JNI dispatch layer.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// append_task
//
// Constructs a server_task of the given type and appends it to `tasks`.
// Captures the repeated 5–6-line block that appears in embed (single task),
// handleEmbeddings (loop), and handleRerank (loop):
//
//   server_task task(type);
//   task.id            = ctx_server->queue_tasks.get_new_id();
//   task.index         = index;
//   task.prompt_tokens = server_tokens(prompt_tokens, false);
//   task.params.oaicompat = oaicompat;
//   tasks.push_back(std::move(task));
//
// The caller is responsible for pre-computing `prompt_tokens` (e.g. via
// format_rerank() for rerank tasks).  Taken by value because server_tokens
// constructor requires a non-const lvalue reference.  `oaicompat` defaults
// to NONE so the rerank call site needs no explicit argument.
//
// Unit-testable without JNI: takes only C++ objects, no JNIEnv calls.
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
