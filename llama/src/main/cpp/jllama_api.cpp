// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The C API (jllama_api.h) over the engine (jllama_engine.hpp): status codes and a thread-local
// error message instead of exceptions, malloc'd strings instead of std::string, the user-reference
// guard around every call so jllama_model_free can wait for the calls in flight.

#include "build-info.h"
#include "llama.h"
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "jllama_engine.hpp"
#include "jllama_api.h"

#include <cstdlib>
#include <cstring>
#include <string>

#ifndef JLLAMA_VERSION_STRING
#error "JLLAMA_VERSION_STRING must be defined by the build (llama/CMakeLists.txt sets it from the pom)"
#endif

namespace {

thread_local std::string t_last_error;

jllama_status fail(jllama_status status, const char *what) {
    t_last_error = what != nullptr ? what : "";
    return status;
}

// Runs `fn` inside the user-reference guard and maps what it throws to a status.
template <typename Fn> jllama_status guarded(jllama_model *model, Fn &&fn) {
    auto *jctx = reinterpret_cast<jllama_context *>(model);
    if (jctx == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "model is null");
    }
    jllama_context_guard guard{acquire_jllama_context(jctx)};
    try {
        fn(jctx);
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const jllama_invalid_request &e) {
        return fail(JLLAMA_STATUS_INVALID_REQUEST, e.what());
    } catch (const jllama_interrupted &e) {
        return fail(JLLAMA_STATUS_INTERRUPTED, e.what());
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

// A caller-owned copy of a string (jllama_string_free releases it).
char *copy_out(const std::string &s) {
    auto *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out == nullptr) {
        throw std::bad_alloc();
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

// The one adapter between the C callback and common_log's sink: the user's function and data are
// read when a line arrives, and set_log_sink's detach-first ordering (the engine) is what keeps a
// line queued for the previous callback from reaching the new one.
jllama_log_fn g_log_fn = nullptr;
void *g_log_user_data = nullptr;

void log_adapter(enum ggml_log_level level, const char *text, void * /*user_data*/) {
    jllama_log_fn fn = g_log_fn;
    if (fn != nullptr && text != nullptr) {
        fn(static_cast<int>(level), text, g_log_user_data);
    }
}

} // namespace

extern "C" {

const char *jllama_api_version(void) { return JLLAMA_VERSION_STRING; }

const char *jllama_build_info(void) { return llama_build_info(); }

const char *jllama_last_error(void) { return t_last_error.c_str(); }

void jllama_string_free(char *s) { std::free(s); }

jllama_status jllama_model_load(int argc, const char *const *argv, jllama_load_progress_fn progress,
                                void *progress_user_data, jllama_model **out) {
    if (out == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "out is null");
    }
    *out = nullptr;
    if (argc < 1 || argv == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "argv must hold at least the program name");
    }
    try {
        std::vector<std::string> args;
        args.reserve(static_cast<size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i] != nullptr ? argv[i] : "");
        }
        // llama_progress_callback and jllama_load_progress_fn have the same shape on purpose.
        *out = reinterpret_cast<jllama_model *>(jllama::engine::load(args, progress, progress_user_data));
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::invalid_argument &e) {
        return fail(JLLAMA_STATUS_INVALID_REQUEST, e.what());
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

void jllama_model_free(jllama_model *model) { jllama::engine::free(reinterpret_cast<jllama_context *>(model)); }

jllama_status jllama_chat_completions(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::chat_completions(jctx, request_json));
    });
}

jllama_status jllama_chat_completions_stream_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, [&](jllama_context *jctx) {
        *task_id = jllama::engine::chat_completions_stream_request(jctx, request_json, true);
    });
}

jllama_status jllama_chat_completions_stream_receive(jllama_model *model, int task_id, char **envelope_json,
                                                     bool *stop) {
    if (envelope_json == nullptr || stop == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "envelope_json and stop must not be null");
    }
    *envelope_json = nullptr;
    *stop = false;
    return guarded(model, [&](jllama_context *jctx) {
        bool s = false;
        std::string envelope = jllama::engine::chat_completions_stream_receive(jctx, task_id, s);
        *envelope_json = copy_out(envelope);
        *stop = s;
    });
}

jllama_status jllama_task_release(jllama_model *model, int task_id) {
    return guarded(model, [&](jllama_context *jctx) { jllama::engine::task_release(jctx, task_id); });
}

jllama_status jllama_set_log_callback(jllama_log_fn callback, void *user_data) {
    try {
        if (callback == nullptr) {
            // Detach (and drain to the old callback) BEFORE forgetting it.
            jllama::engine::set_log_sink(nullptr, nullptr);
            g_log_fn = nullptr;
            g_log_user_data = nullptr;
        } else {
            // Drain the previous callback first, then point the adapter at the new one and attach.
            jllama::engine::set_log_sink(nullptr, nullptr);
            g_log_fn = callback;
            g_log_user_data = user_data;
            jllama::engine::set_log_sink(log_adapter, nullptr);
        }
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    }
}

} // extern "C"
