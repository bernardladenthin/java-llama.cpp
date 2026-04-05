#pragma once

// jni_server_helpers.hpp — JNI helpers that need server.hpp types.
//
// Kept separate from jni_helpers.hpp intentionally: jni_helpers.hpp has a
// deliberately minimal include surface (only jni.h + stdlib) so it can be
// unit-tested without the full llama.cpp stack.  Any helper that must reach
// into server.hpp types belongs here instead.
//
// The single public entry point is collect_task_results_impl(), which drains
// one server_task_result_ptr per task ID from a server_response queue and
// propagates any server-side error back to Java via JNI.
//
// All parameters are explicit (no module-level globals) so the function can
// be exercised in unit tests using a local server_response and a mock JNIEnv.

#include "jni.h"
#include "server.hpp"

#include <unordered_set>
#include <vector>

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
inline bool collect_task_results_impl(JNIEnv                               *env,
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
