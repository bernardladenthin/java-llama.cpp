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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
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
    return guarded(model, [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::model_meta(jctx)); });
}

jllama_status jllama_completions(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::completions(jctx, request_json));
    });
}

jllama_status jllama_completions_oai(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::completions_oai(jctx, request_json));
    });
}

jllama_status jllama_infill(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(
        model, [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::infill(jctx, request_json)); });
}

jllama_status jllama_completions_stream_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, [&](jllama_context *jctx) {
        *task_id = jllama::engine::completions_stream_request(jctx, request_json);
    });
}

jllama_status jllama_chat_completions_request(jllama_model *model, const char *request_json, int *task_id) {
    if (request_json == nullptr || task_id == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and task_id must not be null");
    }
    *task_id = 0;
    return guarded(model, [&](jllama_context *jctx) {
        *task_id = jllama::engine::chat_completions_stream_request(jctx, request_json, false);
    });
}

jllama_status jllama_completions_stream_receive(jllama_model *model, int task_id, char **response_json, bool *stop) {
    if (response_json == nullptr || stop == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json and stop must not be null");
    }
    *response_json = nullptr;
    *stop = false;
    return guarded(model, [&](jllama_context *jctx) {
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
    return guarded(
        model, [&](jllama_context *jctx) { *prompt = copy_out(jllama::engine::apply_template(jctx, request_json)); });
}

jllama_status jllama_embed(jllama_model *model, const char *prompt, float **values, size_t *count) {
    if (prompt == nullptr || values == nullptr || count == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "prompt, values and count must not be null");
    }
    *values = nullptr;
    *count = 0;
    return guarded(model, [&](jllama_context *jctx) {
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
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::embeddings(jctx, request_json, oai_compat));
    });
}

jllama_status jllama_rerank(jllama_model *model, const char *query, int n_documents, const char *const *documents,
                            char **response_json) {
    if (query == nullptr || response_json == nullptr || (n_documents > 0 && documents == nullptr) || n_documents < 0) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "query, documents and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) {
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
    return guarded(model, [&](jllama_context *jctx) {
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
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::tokenize_json(jctx, text, add_special, with_pieces));
    });
}

jllama_status jllama_detokenize(jllama_model *model, const int *tokens, size_t count, char **text, size_t *length) {
    if ((tokens == nullptr && count > 0) || text == nullptr || length == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "tokens, text and length must not be null");
    }
    *text = nullptr;
    *length = 0;
    return guarded(model, [&](jllama_context *jctx) {
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
    return guarded(model, [&](jllama_context *jctx) {
        const std::vector<llama_token> ids(tokens, tokens + count);
        *response_json = copy_out(jllama::engine::detokenize_json(jctx, ids));
    });
}

jllama_status jllama_metrics(jllama_model *model, char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::metrics(jctx)); });
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
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::slot_action(jctx, op, slot_id, filename != nullptr ? filename : ""));
    });
}

jllama_status jllama_lora_adapters(jllama_model *model, char **response_json) {
    if (response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model,
                   [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::lora_adapters(jctx)); });
}

jllama_status jllama_set_lora_adapters(jllama_model *model, const char *adapters_json, char **response_json) {
    if (adapters_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "adapters_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(model, [&](jllama_context *jctx) {
        *response_json = copy_out(jllama::engine::set_lora_adapters(jctx, adapters_json));
    });
}

jllama_status jllama_configure(jllama_model *model, const char *config_json) {
    if (config_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "config_json must not be null");
    }
    return guarded(model, [&](jllama_context *jctx) { jllama::engine::configure(jctx, config_json); });
}

jllama_status jllama_systemone(jllama_model *model, const char *request_json, char **response_json) {
    if (request_json == nullptr || response_json == nullptr) {
        return fail(JLLAMA_STATUS_INVALID_ARGUMENT, "request_json and response_json must not be null");
    }
    *response_json = nullptr;
    return guarded(
        model, [&](jllama_context *jctx) { *response_json = copy_out(jllama::engine::systemone(jctx, request_json)); });
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
