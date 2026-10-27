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
#include "rpc_support.hpp"
#include "tts_engine.h"
#include "train_engine.h"
#ifdef JLLAMA_HAS_NATIVE_SERVER
#include "native_server_engine.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifndef JLLAMA_VERSION_STRING
#error "JLLAMA_VERSION_STRING must be defined by the build (llama/CMakeLists.txt sets it from the pom)"
#endif

namespace {

thread_local std::string t_last_error;

jllama_status fail(jllama_status status, const char *what) {
    t_last_error = what != nullptr ? what : "";
    return status;
}

// Runs `fn` inside the user-reference guard and maps what it throws to a status. `operation` is the
// C entry point (__func__); a model loaded with --vocab-only has no server context, so an operation
// that needs one is refused here, before `fn` reaches the never-loaded server_context (a SIGSEGV in
// llama_vocab::token_bos(), or a wait for a result no inference loop produces, before F266). The
// refusal is the default: only the operations guarded_tokenizer() runs answer a vocab-only model.
template <typename Fn>
jllama_status run_guarded(jllama_model *model, const char *operation, bool needs_server, Fn &&fn) {
    auto *jctx = reinterpret_cast<jllama_context *>(model);
    if (jctx == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "model is null");
    }
    jllama_context_guard guard{acquire_jllama_context(jctx)};
    try {
        if (needs_server) {
            jllama::engine::require_server(jctx, operation);
        }
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

// An operation that needs the server context (and therefore a model loaded without --vocab-only).
template <typename Fn> jllama_status guarded(jllama_model *model, const char *operation, Fn &&fn) {
    return run_guarded(model, operation, true, std::forward<Fn>(fn));
}

// An operation the tokenizer alone answers: the model meta, tokenize and detokenize.
template <typename Fn> jllama_status guarded_tokenizer(jllama_model *model, Fn &&fn) {
    return run_guarded(model, nullptr, false, std::forward<Fn>(fn));
}

// A C argv as strings (a null element is "").
std::vector<std::string> to_strings(int argc, const char *const *argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i] != nullptr ? argv[i] : "");
    }
    return args;
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

// jllama::rpc::set_serve_hook_for_testing().
std::atomic<void (*)()> g_rpc_serve_hook{nullptr};
// jllama::rpc::set_stop_hook_for_testing().
std::atomic<void (*)()> g_rpc_stop_hook{nullptr};

// The serves in progress, for jllama_rpc_server_stop(). ggml clears its stop flag when it binds the
// socket (patches/0015), so a stop of ggml's alone is lost when it lands before the bind (device
// initialization, transport start) and the server goes on listening. A stop while a serve is in
// progress is therefore kept here: serve does not start ggml's server once it is set, and the
// stopping thread repeats ggml's stop until serve has returned. A stop while none is in progress
// stays a no-op, and a stop is for the serves in progress when it is called: every serve call gets
// an id, the stop records the newest one as its target and waits for the serves up to it, so a
// serve that starts while the stop still waits (a restart from another thread) is neither told to
// stop nor reached by the repeated ggml stop -- that one is issued under the mutex the new serve's
// registration takes first, and ggml clears its flag when the new serve binds. Before, the stop
// waited for the count of serves to reach zero and the request was reset only then, so a serve
// that took over right after the targeted one had ended was ended too.
struct rpc_serve_state {
    std::mutex mutex;
    std::condition_variable ended;
    std::uint64_t next_id = 1;
    std::vector<std::uint64_t> active; // the ids of the serves in progress, oldest first
    std::uint64_t stop_up_to = 0;      // the serves with an id up to this one were asked to stop
};

// Whether a serve with an id up to `target` is still in progress.
bool serving_up_to(const rpc_serve_state &state, std::uint64_t target) {
    return !state.active.empty() && state.active.front() <= target;
}

rpc_serve_state &rpc_serve() {
    static rpc_serve_state state;
    return state;
}

// Whether this thread is inside jllama_rpc_server_serve(): a stop from there cannot wait for it.
thread_local bool t_in_rpc_serve = false;

constexpr auto RPC_STOP_RETRY = std::chrono::milliseconds(50);
constexpr auto RPC_STOP_TIMEOUT = std::chrono::seconds(30);

// Marks one jllama_rpc_server_serve() call in progress, for its whole duration, under an id of its own.
class rpc_serving_scope {
  public:
    rpc_serving_scope() {
        rpc_serve_state &state = rpc_serve();
        std::lock_guard<std::mutex> lock(state.mutex);
        id_ = state.next_id++;
        state.active.push_back(id_);
        t_in_rpc_serve = true;
    }
    ~rpc_serving_scope() {
        t_in_rpc_serve = false;
        rpc_serve_state &state = rpc_serve();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.active.erase(std::remove(state.active.begin(), state.active.end(), id_), state.active.end());
        state.ended.notify_all();
    }
    rpc_serving_scope(const rpc_serving_scope &) = delete;
    rpc_serving_scope &operator=(const rpc_serving_scope &) = delete;

    // Whether a stop was issued for this serve (not for one that started after the stop).
    bool stop_requested() const {
        rpc_serve_state &state = rpc_serve();
        std::lock_guard<std::mutex> lock(state.mutex);
        return id_ <= state.stop_up_to;
    }

  private:
    std::uint64_t id_ = 0;
};

} // namespace

void jllama::rpc::set_serve_hook_for_testing(void (*hook)()) { g_rpc_serve_hook.store(hook); }
void jllama::rpc::set_stop_hook_for_testing(void (*hook)()) { g_rpc_stop_hook.store(hook); }

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
        // llama_progress_callback and jllama_load_progress_fn have the same shape on purpose.
        *out = reinterpret_cast<jllama_model *>(
            jllama::engine::load(to_strings(argc, argv), progress, progress_user_data));
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
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::chat_completions(jctx, request_json));
    });
}

jllama_status jllama_chat_completions_stream_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, __func__, [&](jllama_context *jctx) {
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
    return guarded(model, __func__, [&](jllama_context *jctx) {
        bool s = false;
        std::string envelope = jllama::engine::chat_completions_stream_receive(jctx, task_id, s);
        *envelope_json = copy_out(envelope);
        *stop = s;
    });
}

jllama_status jllama_task_release(jllama_model *model, int task_id) {
    return guarded(model, __func__, [&](jllama_context *jctx) { jllama::engine::task_release(jctx, task_id); });
}

// ------------------------------------------------------------------------------------------------
// The rest of LlamaModel.
// ------------------------------------------------------------------------------------------------

void jllama_floats_free(float *values) { std::free(values); }

void jllama_tokens_free(int *tokens) { std::free(tokens); }

jllama_status jllama_model_meta(jllama_model *model, char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    return guarded_tokenizer(
        model, [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::model_meta(jctx)); });
}

jllama_status jllama_completions(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::completions(jctx, request_json));
    });
}

jllama_status jllama_completions_oai(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::completions_oai(jctx, request_json));
    });
}

jllama_status jllama_infill(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::infill(jctx, request_json));
    });
}

jllama_status jllama_completions_stream_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *task_id = jllama::engine::completions_stream_request(jctx, request_json);
    });
}

jllama_status jllama_completions_oai_stream_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *task_id = jllama::engine::completions_oai_stream_request(jctx, request_json);
    });
}

jllama_status jllama_chat_completions_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *task_id = jllama::engine::chat_completions_stream_request(jctx, request_json, false);
    });
}

jllama_status jllama_completions_stream_receive(jllama_model *model, int task_id, char **response_json, bool *stop) {
    if (response_json == nullptr || stop == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json and stop must not be null");
    }
    *response_json = nullptr;
    *stop = false;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        bool s = false;
        std::string response = jllama::engine::completions_stream_receive(jctx, task_id, s);
        *response_json = copy_out(response);
        *stop = s;
    });
}

jllama_status jllama_apply_template(jllama_model *model, const char *request_json, char **prompt) {
    if (request_json == nullptr || prompt == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and prompt must not be null");
    }
    *prompt = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *prompt = copy_out(jllama::engine::apply_template(jctx, request_json));
    });
}

jllama_status jllama_embed(jllama_model *model, const char *prompt, float **values, size_t *count) {
    if (prompt == nullptr || values == nullptr || count == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "prompt, values and count must not be null");
    }
    *values = nullptr;
    *count = 0;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        const std::vector<float> embedding = jllama::engine::embed(jctx, prompt);
        auto *out = static_cast<float *>(std::malloc(std::max<size_t>(embedding.size(), 1) * sizeof(float)));
        if (out == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(out, embedding.data(), embedding.size() * sizeof(float));
        *values = out;
        *count = embedding.size();
    });
}

jllama_status jllama_embeddings(jllama_model *model, const char *request_json, bool oai_compat, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::embeddings(jctx, request_json, oai_compat));
    });
}

jllama_status jllama_rerank(jllama_model *model, const char *query, int n_documents, const char *const *documents,
                            char **response_json) {
    if (query == nullptr || response_json == nullptr || (n_documents > 0 && documents == nullptr) || n_documents < 0) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "query, documents and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        std::vector<std::string> docs;
        docs.reserve(static_cast<size_t>(n_documents));
        for (int i = 0; i < n_documents; ++i) {
            docs.emplace_back(documents[i] != nullptr ? documents[i] : "");
        }
        *response_json = copy_out(jllama::engine::rerank(jctx, query, docs));
    });
}

jllama_status jllama_tokenize(jllama_model *model, const char *text, bool add_special, int **tokens, size_t *count) {
    if (text == nullptr || tokens == nullptr || count == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "text, tokens and count must not be null");
    }
    *tokens = nullptr;
    *count = 0;
    return guarded_tokenizer(model, [&](jllama_context *jctx) {
        const std::vector<llama_token> ids = jllama::engine::tokenize(jctx, text, add_special);
        auto *out = static_cast<int *>(std::malloc(std::max<size_t>(ids.size(), 1) * sizeof(int)));
        if (out == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(out, ids.data(), ids.size() * sizeof(int));
        *tokens = out;
        *count = ids.size();
    });
}

jllama_status jllama_tokenize_json(jllama_model *model, const char *text, bool add_special, bool with_pieces,
                                   char **response_json) {
    if (text == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "text and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded_tokenizer(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::tokenize_json(jctx, text, add_special, with_pieces));
    });
}

jllama_status jllama_detokenize(jllama_model *model, const int *tokens, size_t count, char **text, size_t *length) {
    if ((tokens == nullptr && count > 0) || text == nullptr || length == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "tokens, text and length must not be null");
    }
    *text = nullptr;
    *length = 0;
    return guarded_tokenizer(model, [&](jllama_context *jctx) {
        const std::vector<llama_token> ids(tokens, tokens + count);
        const std::string bytes = jllama::engine::detokenize(jctx, ids);
        *text = copy_out(bytes);
        *length = bytes.size();
    });
}

jllama_status jllama_detokenize_json(jllama_model *model, const int *tokens, size_t count, char **response_json) {
    if ((tokens == nullptr && count > 0) || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "tokens and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded_tokenizer(model, [&](jllama_context *jctx) {
        const std::vector<llama_token> ids(tokens, tokens + count);
        *response_json = copy_out(jllama::engine::detokenize_json(jctx, ids));
    });
}

jllama_status jllama_metrics(jllama_model *model, char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__,
                   [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::metrics(jctx)); });
}

jllama_status jllama_slot_action(jllama_model *model, jllama_slot_op action, int slot_id, const char *filename,
                                 char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    jllama::engine::slot_op op;
    switch (action) {
    case JLLAMA_SLOT_SAVE:
        op = jllama::engine::slot_op::save;
        break;
    case JLLAMA_SLOT_RESTORE:
        op = jllama::engine::slot_op::restore;
        break;
    case JLLAMA_SLOT_ERASE:
        op = jllama::engine::slot_op::erase;
        break;
    default:
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "Invalid slot action");
    }
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::slot_action(jctx, op, slot_id, filename != nullptr ? filename : ""));
    });
}

jllama_status jllama_lora_adapters(jllama_model *model, char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__,
                   [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::lora_adapters(jctx)); });
}

jllama_status jllama_set_lora_adapters(jllama_model *model, const char *adapters_json, char **response_json) {
    if (adapters_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "adapters_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::set_lora_adapters(jctx, adapters_json));
    });
}

jllama_status jllama_configure(jllama_model *model, const char *config_json) {
    if (config_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "config_json must not be null");
    }
    return guarded(model, __func__, [&](jllama_context *jctx) { jllama::engine::configure(jctx, config_json); });
}

jllama_status jllama_systemone(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::systemone(jctx, request_json));
    });
}

jllama_status jllama_json_schema_to_grammar(const char *schema_json, char **grammar) {
    if (schema_json == nullptr || grammar == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "schema_json and grammar must not be null");
    }
    *grammar = nullptr;
    try {
        *grammar = copy_out(jllama::engine::json_schema_to_grammar_text(schema_json));
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

jllama_status jllama_quantize(const char *input_path, const char *output_path, int ftype, int nthread,
                              bool allow_requantize) {
    if (input_path == nullptr || output_path == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "input_path and output_path must not be null");
    }
    try {
        jllama::engine::quantize(input_path, output_path, ftype, nthread, allow_requantize);
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "Unknown C++ exception during quantization");
    }
}

// ------------------------------------------------------------------------------------------------
// The embedded HTTP server (native_server_engine.hpp; not on Android).
// ------------------------------------------------------------------------------------------------

#ifdef JLLAMA_HAS_NATIVE_SERVER

jllama_status jllama_server_start(int argc, const char *const *argv, jllama_server **out) {
    if (out == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "out is null");
    }
    *out = nullptr;
    if (argc < 1 || argv == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "argv must hold at least the program name");
    }
    try {
        *out = reinterpret_cast<jllama_server *>(jllama::server_engine::start(to_strings(argc, argv)));
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

jllama_status jllama_server_start_attached(jllama_model *model, int argc, const char *const *argv,
                                           jllama_server **out) {
    if (out == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "out is null");
    }
    *out = nullptr;
    if (argc < 1 || argv == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "argv must hold at least the program name");
    }
    return guarded(model, __func__, [&](jllama_context *jctx) {
        *out = reinterpret_cast<jllama_server *>(jllama::server_engine::start_attached(jctx, to_strings(argc, argv)));
    });
}

jllama_status jllama_server_stop(jllama_server *server) {
    try {
        jllama::server_engine::stop(reinterpret_cast<jllama::server_engine::native_server *>(server));
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    }
}

bool jllama_server_is_running(const jllama_server *server) {
    return jllama::server_engine::running(reinterpret_cast<const jllama::server_engine::native_server *>(server));
}

jllama_status jllama_server_set_worker_command(const char *command) {
    if (command == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "command must not be null (\"\" clears it)");
    }
    jllama::server_engine::set_worker_command(command);
    t_last_error.clear();
    return JLLAMA_STATUS_OK;
}

#endif // JLLAMA_HAS_NATIVE_SERVER

// ------------------------------------------------------------------------------------------------
// The RPC server (rpc_support.hpp over ggml's own server; patches/0015).
// ------------------------------------------------------------------------------------------------

void jllama_strings_free(char **strings, size_t count) {
    if (strings == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(strings[i]);
    }
    std::free(strings);
}

jllama_status jllama_rpc_server_serve(const char *host, int port, int threads, const char *cache_dir,
                                      const char *devices) {
    if (host == nullptr || devices == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "host and devices must not be null");
    }
    rpc_serving_scope serving;
    try {
        const std::string endpoint = std::string(host) + ":" + std::to_string(port);
        auto selected = jllama::rpc::server_devices(jllama::rpc::split_endpoints(devices));
        if (selected.empty()) {
            throw std::runtime_error("no device to serve over RPC");
        }
        if (auto *hook = g_rpc_serve_hook.load()) {
            hook();
        }
        if (serving.stop_requested()) {
            t_last_error.clear();
            return JLLAMA_STATUS_OK; // stopped before the server started
        }
        // Blocks until jllama_rpc_server_stop(), or returns at once when the socket cannot be bound.
        jllama::rpc::rpc_proc<decltype(&ggml_backend_rpc_start_server)>("ggml_backend_rpc_start_server")(
            endpoint.c_str(), cache_dir != nullptr && cache_dir[0] != '\0' ? cache_dir : nullptr,
            static_cast<size_t>(threads), selected.size(), selected.data());
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

jllama_status jllama_rpc_server_stop(void) {
    try {
        auto *stop_server =
            jllama::rpc::rpc_proc<decltype(&ggml_backend_rpc_stop_server)>("ggml_backend_rpc_stop_server");
        rpc_serve_state &state = rpc_serve();
        std::unique_lock<std::mutex> lock(state.mutex);
        // The serves in progress now; one that starts later gets a greater id and is not meant.
        const std::uint64_t target = state.next_id - 1;
        if (!state.active.empty()) {
            state.stop_up_to = std::max(state.stop_up_to, target);
        }
        if (state.active.empty() || t_in_rpc_serve) {
            // Nothing to wait for: none runs, or this is the serving thread itself, before ggml's
            // server started (serve sees the request and does not start it).
            stop_server();
            t_last_error.clear();
            return JLLAMA_STATUS_OK;
        }
        if (auto *hook = g_rpc_stop_hook.load()) {
            lock.unlock();
            hook();
            lock.lock();
        }
        const auto deadline = std::chrono::steady_clock::now() + RPC_STOP_TIMEOUT;
        for (;;) {
            if (!serving_up_to(state, target)) {
                t_last_error.clear();
                return JLLAMA_STATUS_OK;
            }
            // Under the mutex, so a serve that starts now registers only afterwards and then binds,
            // which clears ggml's flag: the stop reaches the serves it is for. Repeated, because it is
            // lost while their socket is not bound yet.
            stop_server();
            const auto next = std::min(std::chrono::steady_clock::now() + RPC_STOP_RETRY, deadline);
            if (state.ended.wait_until(lock, next, [&state, target]() { return !serving_up_to(state, target); })) {
                t_last_error.clear();
                return JLLAMA_STATUS_OK;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return fail(JLLAMA_STATUS_ERROR, "the RPC server did not stop within 30 s of the stop request");
            }
        }
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    }
}

bool jllama_rpc_server_listening(void) {
    try {
        return jllama::rpc::rpc_proc<decltype(&ggml_backend_rpc_server_listening)>(
            "ggml_backend_rpc_server_listening")();
    } catch (...) {
        return false;
    }
}

jllama_status jllama_rpc_server_devices(const char *devices, char ***names, size_t *count) {
    if (devices == nullptr || names == nullptr || count == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "devices, names and count must not be null");
    }
    *names = nullptr;
    *count = 0;
    try {
        const auto selected = jllama::rpc::server_devices(jllama::rpc::split_endpoints(devices));
        auto **out = static_cast<char **>(std::calloc(std::max<size_t>(selected.size(), 1), sizeof(char *)));
        if (out == nullptr) {
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < selected.size(); ++i) {
            out[i] = copy_out(ggml_backend_dev_name(selected[i]));
        }
        *names = out;
        *count = selected.size();
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

// ------------------------------------------------------------------------------------------------
// Text to speech (tts_engine.h).
// ------------------------------------------------------------------------------------------------

void jllama_bytes_free(unsigned char *bytes) { std::free(bytes); }

jllama_status jllama_tts_load(const char *model_path, const char *mmproj_path, int n_gpu_layers, int n_threads,
                              jllama_tts_model **out) {
    if (model_path == nullptr || mmproj_path == nullptr || out == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "model_path, mmproj_path and out must not be null");
    }
    *out = nullptr;
    try {
        std::string err;
        jllama_tts::tts_engine *engine = jllama_tts::engine_init(model_path, mmproj_path, n_gpu_layers, n_threads, err);
        if (engine == nullptr) {
            return fail(JLLAMA_STATUS_ERROR, err.c_str());
        }
        *out = reinterpret_cast<jllama_tts_model *>(engine);
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

jllama_status jllama_tts_synthesize(jllama_tts_model *tts, const char *text, const char *speaker_reference_path,
                                    const char *lang, int max_frames, int top_k, unsigned int seed, unsigned char **wav,
                                    size_t *length) {
    if (tts == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "TextToSpeech handle is null");
    }
    if (text == nullptr || wav == nullptr || length == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "text, wav and length must not be null");
    }
    *wav = nullptr;
    *length = 0;
    try {
        std::vector<uint8_t> bytes;
        std::string err;
        if (!jllama_tts::engine_synthesize(reinterpret_cast<jllama_tts::tts_engine *>(tts), text,
                                           speaker_reference_path != nullptr ? speaker_reference_path : "",
                                           lang != nullptr ? lang : "", max_frames, top_k, seed, bytes, err)) {
            return fail(JLLAMA_STATUS_ERROR, err.c_str());
        }
        auto *out = static_cast<unsigned char *>(std::malloc(std::max<size_t>(bytes.size(), 1)));
        if (out == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(out, bytes.data(), bytes.size());
        *wav = out;
        *length = bytes.size();
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

void jllama_tts_free(jllama_tts_model *tts) {
    jllama_tts::engine_free(reinterpret_cast<jllama_tts::tts_engine *>(tts));
}

// ------------------------------------------------------------------------------------------------
// Fine-tuning (train_engine.h).
// ------------------------------------------------------------------------------------------------

jllama_status jllama_finetune(const char *config_json) {
    if (config_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "config_json must not be null");
    }
    try {
        std::string err;
        if (!jllama_train::finetune_json(config_json, err)) {
            return fail(JLLAMA_STATUS_ERROR, err.c_str());
        }
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception during fine-tuning");
    }
}

jllama_status jllama_set_cpu_moe_batched(bool batched) {
    try {
        jllama::engine::set_cpu_moe_batched(batched);
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
}

jllama_status jllama_get_cpu_moe_batched(bool *batched) {
    if (batched == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "batched must not be null");
    }
    try {
        *batched = jllama::engine::cpu_moe_batched();
        t_last_error.clear();
        return JLLAMA_STATUS_OK;
    } catch (const std::exception &e) {
        return fail(JLLAMA_STATUS_ERROR, e.what());
    } catch (...) {
        return fail(JLLAMA_STATUS_ERROR, "unknown C++ exception");
    }
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
