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
//
// All parameters are explicit (no module-level globals) so each function can
// be exercised in unit tests using local server objects and a mock JNIEnv.
//
// IMPORTANT — include order:
//   server.hpp must be included by the including translation unit BEFORE this
//   header.  server.hpp has no include guard, so including it here would cause
//   redefinition errors in any TU that already includes server.hpp directly.

#include "jni.h"

#include <unordered_set>
#include <vector>

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
        std::string error_msg = result->to_json()["message"].get<std::string>();
        env->ThrowNew(error_class, error_msg.c_str());
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
            std::string error_msg = result->to_json()["message"].get<std::string>();
            env->ThrowNew(error_class, error_msg.c_str());
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
// Serialises a vector of task results to a jstring.
//
// When there is exactly one result, the top-level JSON is that result's object
// directly.  When there are multiple results, they are wrapped in a JSON array.
// This mirrors the OpenAI API convention used by handleCompletions,
// handleCompletionsOai, handleChatCompletions, and handleInfill.
//
// Parameters are passed explicitly so the function is testable without a real
// JVM.  The caller is responsible for checking that `results` is non-empty
// before calling (an empty vector produces an empty JSON array).
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring results_to_jstring_impl(
        JNIEnv *env,
        const std::vector<server_task_result_ptr> &results) {
    json response;
    if (results.size() == 1) {
        response = results[0]->to_json();
    } else {
        response = json::array();
        for (const auto &res : results) {
            response.push_back(res->to_json());
        }
    }
    std::string s = response.dump();
    return env->NewStringUTF(s.c_str());
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
// Used by receiveCompletionJson, handleRerank, handleEmbeddings,
// handleTokenize, and handleDetokenize.
// ---------------------------------------------------------------------------
[[nodiscard]] inline jstring json_to_jstring_impl(JNIEnv *env, const json &j) {
    std::string s = j.dump();
    return env->NewStringUTF(s.c_str());
}
