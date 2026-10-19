// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// jllama_engine.hpp — the one implementation behind both call paths into this library.
//
// Layer B of the delivery roadmap (TODO.md, item 7): everything a LlamaModel can do is C++ over
// llama.cpp's server_context, and until 5.2.x that C++ lived inside the JNI entry points of
// jllama.cpp, callable only from a JVM. This header is that logic with the JNI taken out: it takes
// and returns C++ values, it reports failure by throwing, and it never sees a JNIEnv. Two thin
// shells sit on top of it -- jllama_api.h (extern "C", status codes, the surface a C, Rust or
// Python caller and a future Panama binding use) and the Java_* entry points of jllama.cpp, which
// call the C API for the operations it already covers and these functions directly for the rest.
//
// Error contract, which both shells map: jllama_invalid_request is "the request's fault" (what
// upstream's HTTP server answers with 400 -- InvalidRequestException in Java,
// JLLAMA_STATUS_INVALID_REQUEST in C); jllama_interrupted is a wait that close()/free cut short;
// every other std::exception is a server-side error (LlamaException, JLLAMA_STATUS_ERROR).
//
// Include order: upstream server headers (server-context.h, server-queue.h, server-task.h,
// server-common.h, server-chat.h) must be included by the translation unit BEFORE this header.

#include "jllama_context.hpp"
#include "log.h"

#include <stdexcept>
#include <string>
#include <vector>

/** The request is invalid: what llama.cpp's own server answers with 400. */
struct jllama_invalid_request : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/** A wait on the model's queue was cut short because the model is being freed. */
struct jllama_interrupted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace jllama::engine {

/**
 * Loads a model from a llama-server argv (argv[0] is the program name, as common_params_parse
 * expects; `--vocab-only` is this library's own flag and is stripped first). `progress` is
 * llama.cpp's load-progress callback (return false to abort the load) or null. Throws
 * std::invalid_argument for an unreachable `--rpc` server, std::runtime_error for everything else
 * (argv the server parser refuses, a model that does not load). The returned handle is owned by
 * the caller and released with free().
 */
[[nodiscard]] jllama_context *load(const std::vector<std::string> &argv, llama_progress_callback progress,
                                   void *progress_user_data);

/**
 * Tears a handle down: marks it closing so every parked reader unwinds, waits until no call is in
 * flight (the user references of jllama_context_guard), stops the worker, frees the model, deletes
 * the handle. The caller must already have made the handle unreachable for new calls (the JNI
 * side nulls the Java field first; a C caller simply stops using the pointer).
 */
void free(jllama_context *jctx);

/** Leaves the idle-sleep state (if any) and re-reads what a sleep/resume cycle replaces. */
void wake_server(jllama_context *jctx);

/** wake_server(), then the server's metadata. */
[[nodiscard]] server_context_meta wake_and_get_meta(jllama_context *jctx);

/**
 * Parses an OpenAI chat-completion body into the completion request upstream's
 * oaicompat_chat_params_parse produces, keeping decoded media in `files` and the chat session
 * (prompt, parser state) in `session`. Throws jllama_invalid_request.
 */
[[nodiscard]] json parse_oai_chat_params(jllama_context *jctx, json &body, std::vector<raw_buffer> &files,
                                         common_chat_session &session);

/** Tokenises the prompt of `data` and fills task.tokens and task.params. May throw (the schema). */
void populate_completion_task(server_task &task, jllama_context *jctx,
                              const std::vector<llama_logit_bias> &logit_bias_eog, const json &data, bool has_mtmd,
                              std::vector<raw_buffer> files = {});

/** Wakes the queue, then posts (see the comment in the .cpp for why the wake is not optional). */
void wake_and_post(server_response_reader &rd, server_task &&task, bool front = false);
void wake_and_post(server_response_reader &rd, std::vector<server_task> &&tasks,
                   const common_chat_session &session = {});

/**
 * One completion task, posted and waited for; every result of it. Throws jllama_invalid_request
 * when the request does not parse, jllama_interrupted when free() cut the wait, std::runtime_error
 * for a failed result (jllama_invalid_request when the server reports it as 400).
 */
[[nodiscard]] std::vector<server_task_result_ptr>
blocking_completion(jllama_context *jctx, const json &data, server_task_type task_type, task_response_type res_type,
                    std::vector<raw_buffer> files = {}, const common_chat_session &session = {});

/**
 * One streaming completion task, posted; its id, under which stream_next() yields the results and
 * task_release() drops it. Throws jllama_invalid_request when the request does not parse.
 */
[[nodiscard]] int streaming_request(jllama_context *jctx, const json &data, server_task_type task_type,
                                    task_response_type res_type, std::vector<raw_buffer> files = {},
                                    const common_chat_session &session = {});

/**
 * The next result of a streaming task as JSON (header-only sentinels skipped), with `stop` set on
 * the last one, after which the task is released. A failed result or an unknown task id throws
 * (the task is released on failure too).
 */
[[nodiscard]] json stream_next(jllama_context *jctx, int id_task, bool &stop);

/** Drops a streaming task (cancel / release); unknown ids are ignored. */
void task_release(jllama_context *jctx, int id_task);

/**
 * The three chat operations end to end, over JSON text -- what the C API and the JNI chat entry
 * points call. The stream variant returns the {"data": <chunk>, "stop": <bool>} envelope the Java
 * iterator consumes (wrap_stream_chunk), so neither shell parses a chunk only to wrap it.
 */
[[nodiscard]] std::string chat_completions(jllama_context *jctx, const std::string &body);
[[nodiscard]] int chat_completions_stream_request(jllama_context *jctx, const std::string &body, bool oai_chunks);
[[nodiscard]] std::string chat_completions_stream_receive(jllama_context *jctx, int id_task, bool &stop);

/**
 * Installs (or, with null, removes) the log sink of common_log (patches/0014), the sink
 * LlamaModel.setLogger and jllama_set_log_callback share. Swaps are serialised: a swap pauses and
 * resumes common_log's worker thread, and two concurrent swaps raced on that std::thread
 * (std::terminate, measured by LlamaLoggerTest before the lock existed). Removing the sink first
 * drains every queued entry to the previous callback, which is what makes "set null" a
 * synchronous flush. With a sink installed, llama/ggml lines are routed into common_log as well,
 * so they reach it before the first model load runs common_init().
 */
void set_log_sink(common_log_callback callback, void *user_data);

// ---------------------------------------------------------------------------
// The rest of LlamaModel, operation by operation (TODO.md item 7, B.3). Every function takes and
// returns JSON text or plain C++ values, wakes the server from idle sleep where it touches the
// model, and reports failure by throwing under the contract above.
// ---------------------------------------------------------------------------

/** The model's metadata object (LlamaModel.getModelMeta()): vocab, context, modalities, template. */
[[nodiscard]] std::string model_meta(jllama_context *jctx);

/** A raw completion (llama.cpp's /completion body), blocking; the array of result objects. */
[[nodiscard]] std::string completions(jllama_context *jctx, const std::string &body);

/** An OpenAI /v1/completions request, blocking; the text_completion object(s). */
[[nodiscard]] std::string completions_oai(jllama_context *jctx, const std::string &body);

/** A fill-in-the-middle request (`input_prefix`, `input_suffix`, optional `input_extra`), blocking. */
[[nodiscard]] std::string infill(jllama_context *jctx, const std::string &body);

/**
 * A raw streaming completion (an infill one when the body carries the infill fields); its task id,
 * received with completions_stream_receive(): each result as JSON with a "stop" key added.
 */
[[nodiscard]] int completions_stream_request(jllama_context *jctx, const std::string &body);

/**
 * A streaming OpenAI /v1/completions request, parsed as the blocking one is; its task id, received
 * with chat_completions_stream_receive(): text_completion chunks, the last with usage.
 */
[[nodiscard]] int completions_oai_stream_request(jllama_context *jctx, const std::string &body);
[[nodiscard]] std::string completions_stream_receive(jllama_context *jctx, int id_task, bool &stop);

/** The chat template applied to an OpenAI chat body: the prompt text the model would see. */
[[nodiscard]] std::string apply_template(jllama_context *jctx, const std::string &body);

/** The embedding vector of one prompt (needs `--embedding`). */
[[nodiscard]] std::vector<float> embed(jllama_context *jctx, const std::string &prompt);

/** An /embeddings (or, with oai_compat, /v1/embeddings) request, blocking; the response body. */
[[nodiscard]] std::string embeddings(jllama_context *jctx, const std::string &body, bool oai_compat);

/** Reranks documents against a query (needs `--reranking`); the result array. */
[[nodiscard]] std::string rerank(jllama_context *jctx, const std::string &query,
                                 const std::vector<std::string> &documents);

/** Tokenisation: the ids, or the /tokenize response (optionally with pieces). */
[[nodiscard]] std::vector<llama_token> tokenize(jllama_context *jctx, const std::string &text, bool add_special);
[[nodiscard]] std::string tokenize_json(jllama_context *jctx, const std::string &text, bool add_special,
                                        bool with_pieces);

/** Detokenisation: the UTF-8 bytes (may hold any byte), or the /detokenize response. */
[[nodiscard]] std::string detokenize(jllama_context *jctx, const std::vector<llama_token> &tokens);
[[nodiscard]] std::string detokenize_json(jllama_context *jctx, const std::vector<llama_token> &tokens);

/** The server metrics object (LlamaModel.getMetrics()): counters plus the slots array. */
[[nodiscard]] std::string metrics(jllama_context *jctx);

enum class slot_op { save, restore, erase };
/** Saves, restores or erases a slot's KV cache (filename for save/restore); the result object. */
[[nodiscard]] std::string slot_action(jllama_context *jctx, slot_op op, int id_slot, const std::string &filename);

/** The LoRA adapter list, and setting the scales from a JSON array of {id, scale}. */
[[nodiscard]] std::string lora_adapters(jllama_context *jctx);
[[nodiscard]] std::string set_lora_adapters(jllama_context *jctx, const std::string &adapters_json);

/** Runtime tuning: `n_threads`, `n_threads_batch`, `slot_prompt_similarity` (each optional). */
void configure(jllama_context *jctx, const std::string &config_json);

/** A /v1/systemone request (a decision model), forwarded to upstream's route handler. */
[[nodiscard]] std::string systemone(jllama_context *jctx, const std::string &body);

/** A JSON schema as a GBNF grammar; needs no model. */
[[nodiscard]] std::string json_schema_to_grammar_text(const std::string &schema_json);

/** Quantises a GGUF file (llama_model_quantize); `ftype` is a llama_ftype. Needs no model. */
void quantize(const std::string &input_path, const std::string &output_path, int ftype, int nthread,
              bool allow_requantize);

} // namespace jllama::engine

// ===========================================================================
// Task helpers shared by the engine and the not-yet-migrated JNI entry points (formerly layer B of
// jni_helpers.hpp; JNI-free, so they live with the engine).
// ===========================================================================

// ---------------------------------------------------------------------------
// configure_multimodal_task_impl
//
// Upstream keeps mtmd_context private inside server_context. Its CLI task
// path exists specifically for callers that have a formatted prompt plus raw
// media but cannot access that context directly. Attach those inputs to the
// task so server_context::tokenize_cli_input() invokes process_mtmd_prompt()
// on the worker thread. Returns false for a text-only request so callers can
// retain their normal eager-tokenization path.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool configure_multimodal_task_impl(server_task &task, bool has_mtmd, const json &data,
                                                         std::vector<raw_buffer> files) {
    if (files.empty()) {
        return false;
    }
    if (!has_mtmd) {
        throw std::invalid_argument("Image input requires a loaded multimodal projector (--mmproj)");
    }
    const auto &prompt = data.at("prompt");
    if (!prompt.is_string()) {
        throw std::invalid_argument("Multimodal chat prompt must be a string");
    }
    task.cli = true;
    task.cli_prompt = prompt.get<std::string>();
    task.cli_files = std::move(files);
    return true;
}

// Match server_routes::handle_completions_impl(): slot selection is task
// metadata, not part of task_params, so eval_llama_cmpl_schema() does not set it.
inline void configure_task_slot_impl(server_task &task, const json &data) {
    task.id_slot = json_value(data, "id_slot", -1);
}
