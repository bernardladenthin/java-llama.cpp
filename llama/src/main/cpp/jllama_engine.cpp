// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "arg.h"
#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-schema.h"
#include "server-common.h"
#include "server-chat.h"
#include "server-http.h"
#include "base64.hpp"
#include "json-schema-to-grammar.h"
#include "utils.hpp"
#include "jllama_engine.hpp"
#include "json_helpers.hpp"
#include "rpc_support.hpp"

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace jllama::engine {

namespace {

// Sentinel value used by llama.cpp (since b7433) to indicate that n_parallel should be resolved
// automatically by the host application (common_params_parser_init() for LLAMA_EXAMPLE_SERVER).
constexpr int N_PARALLEL_AUTO = -1;

// Default n_parallel for the embedded library. Unlike the standalone llama.cpp server (which
// resolves auto to 4 for multi-client throughput), this library runs in-process with a single
// caller, so 1 slot is the appropriate default and preserves pre-b7433 behaviour.
constexpr int N_PARALLEL_DEFAULT = 1;

// Serialises set_log_sink callers; see the header.
std::mutex g_set_log_sink_mutex;

// Throws the exception a failed task result maps to: an error the server reports with code 400
// (ERROR_TYPE_INVALID_REQUEST, ERROR_TYPE_EXCEED_CONTEXT_SIZE) is the request's fault.
[[noreturn]] void throw_result_error(const server_task_result_ptr &result) {
    const std::string message = get_result_error_message(result);
    if (get_result_error_code(result) == 400) {
        throw jllama_invalid_request(message);
    }
    throw std::runtime_error(message);
}

void require_result_ok(const server_task_result_ptr &result) {
    if (!result) {
        throw std::runtime_error("No result");
    }
    if (result->is_error()) {
        throw_result_error(result);
    }
}

// A JSON request body; a body that does not parse is the request's fault.
json parse_json_body(const std::string &body_text) {
    try {
        return json::parse(body_text);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
}

// Rethrows `e` under the rule of upstream's ex_wrapper (tools/server/server.cpp): a
// std::invalid_argument or a common_json_error out of tokenization, prompt formatting or a route
// handler is the request's fault, anything else a server error.
[[noreturn]] void rethrow_by_type(const std::exception &e) {
    if (dynamic_cast<const std::invalid_argument *>(&e) != nullptr ||
        dynamic_cast<const common_json_error *>(&e) != nullptr) {
        throw jllama_invalid_request(e.what());
    }
    throw std::runtime_error(e.what());
}

// The batch completed without a task-level error, or this throws.
void require_batch_ok(const server_response_reader::batch_response &br) {
    if (br.error) {
        throw_result_error(br.error);
    }
    if (br.is_terminated) {
        // wait_for_all was cut short because free() set closing: the results vector is only
        // partially filled (null entries), so don't dereference it -- unwind cleanly instead.
        throw jllama_interrupted("inference interrupted by close()");
    }
}

// One indexed token task for batch submission (rerank and embeddings).
server_task indexed_token_task(server_response_reader &rd, server_task_type type, server_tokens &&tokens, int index,
                               task_response_type res_type) {
    server_task task(type);
    task.id = rd.get_new_id();
    task.tokens = std::move(tokens);
    task.index = index;
    task.params.res_type = res_type;
    return task;
}

// Posts one pre-built task at the front of the queue (as upstream does for its own introspection
// endpoints) and waits for its result; the task's id is assigned here.
//
// wake_and_post() has already left any idle-sleep state, so the task is guaranteed to be seen by a
// running loop. The is_sleeping() disjunct is only a backstop for the narrow window in which the
// queue could fall asleep again between the wake and this wait -- it takes a full idle period
// (>= 1s) of quiet to re-enter. Should it ever happen, the result is a thrown "No result" rather
// than a block until free(); upstream's /metrics handler adds the same guard for the same reason.
server_task_result_ptr post_and_wait(jllama_context *jctx, server_task task) {
    auto rd = jctx->server.get_response_reader();
    task.id = rd.get_new_id();
    wake_and_post(rd, std::move(task), true);
    auto result = rd.next([jctx, &rd] { return jctx->closing.load() || rd.queue_tasks.is_sleeping(); });
    require_result_ok(result);
    return result;
}

std::string one_shot(jllama_context *jctx, server_task task) {
    return safe_json_to_str(post_and_wait(jctx, std::move(task))->to_json());
}

// The special-token id map (a token is -1 / LLAMA_TOKEN_NULL when the model defines none).
json special_tokens_json(const llama_vocab *vocab) {
    return {
        {"bos", llama_vocab_bos(vocab)}, {"eos", llama_vocab_eos(vocab)}, {"eot", llama_vocab_eot(vocab)},
        {"sep", llama_vocab_sep(vocab)}, {"nl", llama_vocab_nl(vocab)},   {"pad", llama_vocab_pad(vocab)},
    };
}

// The streaming reader of a task, copied out under the lock so it stays alive across next(),
// which runs without the lock and may race a concurrent task_release()/free().
std::shared_ptr<server_response_reader> reader_of(jllama_context *jctx, int id_task) {
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    auto it = jctx->readers.find(id_task);
    if (it == jctx->readers.end()) {
        throw std::runtime_error("Task not found");
    }
    return it->second;
}

} // namespace

jllama_context *load(const std::vector<std::string> &argv_in, llama_progress_callback progress,
                     void *progress_user_data) {
    common_params params;

    // Strip --vocab-only before common_params_parse (not a common_params flag).
    std::vector<char *> raw;
    std::vector<std::string> argv_copy = argv_in;
    raw.reserve(argv_copy.size());
    for (auto &a : argv_copy) {
        raw.push_back(a.data());
    }
    bool vocab_only = false;
    std::vector<char *> filtered_argv =
        strip_flag_from_argv(raw.data(), static_cast<int>(raw.size()), "--vocab-only", &vocab_only);
    // Register the `--rpc` servers up front (a clear error for an unreachable one, as
    // std::invalid_argument) and keep RPC devices another model registered in this process out of
    // this load -- see rpc_support.hpp.
    std::vector<std::string> rpc_argv =
        jllama::rpc::prepare_argv(std::vector<std::string>(filtered_argv.begin(), filtered_argv.end()));
    std::vector<char *> parse_argv;
    parse_argv.reserve(rpc_argv.size());
    for (auto &arg : rpc_argv) {
        parse_argv.push_back(const_cast<char *>(arg.c_str()));
    }
    if (!common_params_parse(static_cast<int>(parse_argv.size()), parse_argv.data(), params, LLAMA_EXAMPLE_SERVER)) {
        throw std::runtime_error("Failed to parse model parameters");
    }

    common_init();

    std::unique_ptr<jllama_context> jctx(new jllama_context());
    jctx->vocab_only = vocab_only;
    jctx->params = params;

    // Vocab-only mode: load just the model vocab, skip inference setup.
    if (vocab_only) {
        SRV_INF("loading tokenizer from '%s'\n", params.model.path.c_str());
        llama_model_params mparams = llama_model_default_params();
        mparams.vocab_only = true;
        jctx->vocab_only_model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (!jctx->vocab_only_model) {
            throw std::runtime_error("could not load tokenizer from given file path");
        }
        jctx->vocab = llama_model_get_vocab(jctx->vocab_only_model);
        return jctx.release();
    }

    SRV_INF("loading model '%s'\n", params.model.path.c_str());

    llama_numa_init(params.numa);

    common_params_print_info(params);

    // Resolve the auto sentinel before loading the model.
    if (params.n_parallel <= N_PARALLEL_AUTO) {
        params.n_parallel = N_PARALLEL_DEFAULT;
        jctx->params.n_parallel = N_PARALLEL_DEFAULT;
    }

    LOG_INF("%s: loading model\n", __func__);

    if (progress != nullptr) {
        params.load_progress_callback = progress;
        params.load_progress_callback_user_data = progress_user_data;
    }

    // Before load_model(), as upstream's llama_server() does -- see jllama_context::routes. It keeps
    // a reference to jctx->params, which lives as long as it does.
    jctx->routes = std::make_unique<server_routes>(jctx->params, jctx->server);

    if (!jctx->server.load_model(params)) {
        // The failed load may have left a vocab-only model? No: that is the other branch. Nothing
        // to free besides the context (the server owns what it loaded and frees it in its
        // destructor).
        throw std::runtime_error("could not load model from given file path");
    }

    jctx->vocab = llama_model_get_vocab(llama_get_model(jctx->server.get_llama_context()));
    // The handlers read the model's metadata from this copy; taken once, as upstream does after a load.
    jctx->routes->update_meta(jctx->server);

    LOG_INF("%s: model loaded\n", __func__);

    {
        // Direct get_meta() is correct here and must NOT go through wake_and_get_meta(): this runs on
        // the load path, where the model is freshly loaded and the server cannot be asleep.
        auto meta = jctx->server.get_meta();
        LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
                common_chat_templates_source(meta.chat_params.tmpls.get()).c_str(),
                common_chat_format_example(meta.chat_params.tmpls.get(), jctx->params.use_jinja,
                                           jctx->params.default_template_kwargs)
                    .c_str());
    }

    // The worker runs the server's task loop. It is a plain thread: nothing it runs calls back
    // into a host (the log sink attaches itself to the JVM when there is one, on its own thread),
    // which is also what lets the same loop serve a C caller without a JVM at all.
    jllama_context *raw_ctx = jctx.get();
    jctx->worker = std::thread([raw_ctx]() {
        raw_ctx->worker_ready.store(true);
        raw_ctx->server.start_loop();
    });

    while (!jctx->worker_ready.load()) {
        std::this_thread::yield();
    }

    return jctx.release();
}

void free(jllama_context *jctx) {
    if (jctx == nullptr) {
        return;
    }
    // Signal teardown to any in-flight reader. The streaming / blocking should_stop lambdas observe
    // this and make next()/wait_for_all() return within one poll (~1s), so the in-flight call
    // unwinds and releases its user reference -- otherwise the reader would poll forever (the
    // worker only stops the task queue, not the results queue) and free() would hang.
    jctx->closing.store(true, std::memory_order_release);
    if (!jctx->vocab_only) {
        // Cancel any pending streaming readers before stopping the server.
        std::lock_guard<std::mutex> lk(jctx->readers_mutex);
        jctx->readers.clear();
    }

    // Wait for every in-flight call (an entry point still inside its jllama_context_guard) to
    // release its user reference BEFORE tearing anything down -- a concurrent call could otherwise
    // still be using the worker, the server_context, or the vocab-only model while they are being
    // stopped/freed (use-after-free). New calls cannot start (the caller made the handle
    // unreachable), and the closing flag bounds this wait to ~1s.
    {
        std::unique_lock<std::mutex> lk(jctx->m);
        jctx->cv.wait(lk, [&] { return jctx->users.load() == 0; });
    }

    if (!jctx->vocab_only) {
        while (!jctx->worker_ready.load()) {
            std::this_thread::yield();
        }
        // Signal the background thread to stop. Call twice with a brief sleep to close the race
        // where the thread signalled ready but start_loop() hasn't yet set its internal running flag.
        jctx->server.terminate();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        jctx->server.terminate();
        if (jctx->worker.joinable()) {
            jctx->worker.join();
        }
    }

    if (jctx->vocab_only_model) {
        llama_model_free(jctx->vocab_only_model);
    }

    delete jctx;
}

void require_server(const jllama_context *jctx, const char *operation) {
    if (jctx->vocab_only) {
        throw std::runtime_error(std::string(operation != nullptr ? operation : "This operation") +
                                 " is not available in vocab-only mode: the model was loaded with --vocab-only, "
                                 "which loads the tokenizer only (model meta, tokenize, detokenize)");
    }
}

// Wake the server from the idle-sleep state and re-sync the state this layer caches from the model.
//
// Upstream's `handle_sleeping_state(true)` calls `destroy()`, which frees the model and context and
// sets `ctx_tgt`/`model_tgt` to nullptr; resuming calls `load_model()` again, producing a *new*
// model. Two consequences for this layer, and both crashed or corrupted before this helper existed:
//
//  1. `server_context::get_meta()` dereferences `ctx_tgt`/`model_tgt`, so calling it while asleep is
//     a null dereference (SIGSEGV at `llama_context::get_model()` on all six CI platforms the first
//     time a request followed an idle window).
//  2. `jctx->vocab` is captured once after the initial load. The reload replaces the model, so the
//     cached pointer dangles -- use-after-free on every tokenize/detokenize/rerank path.
//
// `wake_and_post()` is NOT sufficient for either: every dispatch site reads `get_meta()` and
// tokenizes *before* it posts, so the wake has to happen earlier than the post. Call this at the
// top of every operation that touches the model. A no-op when sleeping is off (the default, `-1`),
// and in vocab-only mode, which owns its own model and never has a server.
//
// Teardown ordering: a caller parked in `wait_until_no_sleep()` when the queue terminates is never
// released. That is safe only because every caller runs inside a `jllama_context_guard`, and free()
// waits for `users == 0` before `terminate()`. Do not call this from the teardown path.
void wake_server(jllama_context *jctx) {
    if (jctx->vocab_only) {
        return; // no server_context in vocab-only mode; the model is owned directly and never sleeps
    }
    server_response_reader rd = jctx->server.get_response_reader();
    rd.queue_tasks.wait_until_no_sleep();
    // Re-read the vocab: a sleep/resume cycle replaced the model this pointer came from.
    jctx->vocab = llama_model_get_vocab(llama_get_model(jctx->server.get_llama_context()));
}

server_context_meta wake_and_get_meta(jllama_context *jctx) {
    wake_server(jctx);
    return jctx->server.get_meta();
}

json parse_oai_chat_params(jllama_context *jctx, json &body, std::vector<raw_buffer> &files,
                           common_chat_session &session) {
    try {
        auto meta = wake_and_get_meta(jctx);
        return oaicompat_chat_params_parse(jctx->vocab, body, meta.chat_params, files, session);
    } catch (const jllama_invalid_request &) {
        throw;
    } catch (const std::exception &e) {
        // An invalid request whatever the parser threw: upstream's parsers mix std::invalid_argument
        // and std::runtime_error for the same kind of defect.
        throw jllama_invalid_request(e.what());
    }
}

void populate_completion_task(server_task &task, jllama_context *jctx,
                              const std::vector<llama_logit_bias> &logit_bias_eog, const json &data, bool has_mtmd,
                              std::vector<raw_buffer> files) {
    // upstream's raw multimodal prompt, {"prompt_string": "...<marker>...", "multimodal_data": [base64]},
    // is tokenized by its HTTP handler with the projector, which this engine cannot reach; the task's
    // CLI fields carry it to the worker, which tokenizes it with the projector as the chat paths do.
    // Tokenized here without one, it was refused as if the model had no projector.
    const json *prompt_field = data.contains("prompt") ? &data.at("prompt") : nullptr;
    if (files.empty() && prompt_field != nullptr && prompt_field->is_object() &&
        prompt_field->contains("prompt_string") && prompt_field->contains("multimodal_data")) {
        const json &prompt = *prompt_field;
        if (!has_mtmd) {
            throw std::invalid_argument("Multimodal data provided, but model does not support multimodal requests.");
        }
        const json &media = prompt.at("multimodal_data");
        if (!media.is_array() || media.empty()) {
            throw std::invalid_argument("\"multimodal_data\" must be a non-empty array of base64 strings");
        }
        for (const auto &entry : media) {
            if (!entry.is_string()) {
                throw std::invalid_argument("\"multimodal_data\" must be a non-empty array of base64 strings");
            }
            const std::string bytes = base64::decode(entry.get<std::string>());
            files.emplace_back(bytes.begin(), bytes.end());
        }
        // A prompt that ends with its media leaves the slot no text token to take the logits from, and
        // the server aborts the process on it (GGML_ASSERT(batch.size() > 0) in update_slots); upstream's
        // chat paths never get there because the template puts text after the last image.
        const json &prompt_string = prompt.at("prompt_string");
        if (prompt_string.is_string() && string_ends_with(prompt_string.get<std::string>(), get_media_marker())) {
            throw std::invalid_argument("a multimodal prompt must have text after its last media marker");
        }
        json multimodal = data;
        multimodal["prompt"] = prompt.at("prompt_string");
        if (!configure_multimodal_task_impl(task, has_mtmd, multimodal, std::move(files))) {
            throw std::logic_error("a multimodal prompt carried no media");
        }
    } else if (!configure_multimodal_task_impl(task, has_mtmd, data, std::move(files))) {
        auto tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, data.at("prompt"), true, true, mtmd_helper_init_opt_default());
        // A task carries one prompt. Upstream answers several prompts with one choice each; taking the
        // first here dropped the others without a word, so they are refused instead.
        if (tokenized_prompts.size() > 1) {
            throw jllama_invalid_request("a request takes one prompt, this one has " +
                                         std::to_string(tokenized_prompts.size()) +
                                         " prompts; send them as separate requests");
        }
        if (!tokenized_prompts.empty()) {
            task.tokens = std::move(tokenized_prompts[0]);
        }
    }
    task.params = server_schema::eval_llama_cmpl_schema(jctx->vocab, jctx->params, logit_bias_eog, data);
    // The completion schema sets the sequence the reasoning budget forces only in the handler of
    // reasoning_budget_message, so a request naming the end tags without a message armed a budget that
    // forced nothing when it ran out. The chat paths always pass the message (the server's default,
    // --reasoning-budget-message, empty unless set); a request without one gets the same default here.
    common_params_sampling &sampling = task.params.sampling;
    if (!sampling.reasoning_budget_end.empty() && sampling.reasoning_budget_forced.empty()) {
        llama_tokens forced = sampling.reasoning_budget_end.front();
        const std::string &message = jctx->params.sampling.reasoning_budget_message;
        if (!message.empty()) {
            const llama_tokens message_tokens = common_tokenize(jctx->vocab, message, false, true);
            forced.insert(forced.begin(), message_tokens.begin(), message_tokens.end());
        }
        sampling.reasoning_budget_forced = std::move(forced);
    }
    // Several completions of one prompt ("n", upstream's n_cmpl): upstream answers them in child tasks
    // the server creates, the engine posts this one task alone, so the others were dropped without a
    // word. Refused like several prompts.
    if (task.params.n_cmpl > 1) {
        throw jllama_invalid_request("a request takes one completion, this one asks for 'n' = " +
                                     std::to_string(task.params.n_cmpl) + "; send separate requests");
    }
    configure_task_slot_impl(task, data);
}

// The id and the model name every OpenAI answer and chunk carries, set as upstream's
// handle_completions_impl sets them: a fresh completion id, and the name the server answers to (the
// alias, else the model's name). Left unset, they were "" in every answer.
static void name_completion_task(server_task &task, const server_context_meta &meta) {
    task.params.oaicompat_cmpl_id = gen_chatcmplid();
    task.params.oaicompat_model = meta.model_name;
}

// Wake the queue before posting, then post.
//
// With ModelParameters.setSleepIdleSeconds(> 0) the queue enters a sleeping state after the
// configured idle period, and server_queue::post() does NOT leave it: post() only calls
// condition_tasks.notify_one(), while the sleeping wait predicate is `(!running ||
// req_stop_sleeping)`. The loop wakes, re-tests, and goes straight back to sleep with the task still
// queued. Only wait_until_no_sleep() sets req_stop_sleeping. Upstream does this wake for the caller
// in server_res_generator's constructor; this layer builds its readers with
// server_context::get_response_reader(), the CLI-facing accessor, which performs no wake.
//
// Ordering invariant this depends on: wait_until_no_sleep() blocks until the worker loop clears
// `sleeping`, and upstream's loop leaves that flag SET when it exits on !running. So a caller parked
// here when the queue is terminated is never released. It is safe only because every caller holds a
// jllama_context_guard user reference and free() waits for users == 0 BEFORE calling terminate().
void wake_and_post(server_response_reader &rd, server_task &&task, bool front) {
    rd.queue_tasks.wait_until_no_sleep();
    rd.post_task(std::move(task), front);
}

// The chat session (if any) is what the reader builds its task_result_state from -- the parser that
// turns generated text into content, reasoning and tool calls. A default session is the non-chat
// case (content only).
void wake_and_post(server_response_reader &rd, std::vector<server_task> &&tasks, const common_chat_session &session) {
    rd.queue_tasks.wait_until_no_sleep();
    rd.post_tasks(std::move(tasks), session);
}

// A raw task posts no session, so what apply_chat_session gives a chat task is applied here for a raw
// completion with "messages". The generation prompt is what the sampler of a chat task is primed with
// (common_sampler_init feeds it to the reasoning budget), so a template that opens the thinking block
// in the generation prompt starts the budget counting at the first generated token. An output-format
// grammar (json_schema) is primed with the same prompt, which it cannot accept; beside one the prompt
// is left unset, and the budget then counts only from a start tag the model writes itself. The
// template's preserved tokens join the request's own (the schema has put those in already): the chat
// path drops a request's preserved_tokens because the session owns them, but on a raw completion the
// field is a completion field of its own, so both are kept.
static void apply_raw_chat_sampling(server_task &task, const raw_chat_sampling &raw_chat) {
    if (!raw_chat.generation_prompt.empty() && !common_grammar_needs_prefill(task.params.sampling.grammar)) {
        task.params.sampling.generation_prompt = raw_chat.generation_prompt;
    }
    task.params.sampling.preserved_tokens.insert(raw_chat.preserved_tokens.begin(), raw_chat.preserved_tokens.end());
}

std::vector<server_task_result_ptr> blocking_completion(jllama_context *jctx, const json &data,
                                                        server_task_type task_type, task_response_type res_type,
                                                        std::vector<raw_buffer> files,
                                                        const common_chat_session &session,
                                                        const raw_chat_sampling &raw_chat) {
    auto meta = wake_and_get_meta(jctx);
    auto rd = jctx->server.get_response_reader();
    server_task task(task_type);
    task.id = rd.get_new_id();
    try {
        populate_completion_task(task, jctx, meta.logit_bias_eog, data, meta.has_mtmd, std::move(files));
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
    task.apply_chat_session(session);
    apply_raw_chat_sampling(task, raw_chat);
    task.params.res_type = res_type;
    name_completion_task(task, meta);
    std::vector<server_task> tasks;
    tasks.push_back(std::move(task));
    wake_and_post(rd, std::move(tasks), session);
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    require_batch_ok(br);
    return std::move(br.results);
}

int streaming_request(jllama_context *jctx, const json &data, server_task_type task_type, task_response_type res_type,
                      std::vector<raw_buffer> files, const common_chat_session &session,
                      const raw_chat_sampling &raw_chat) {
    auto meta = wake_and_get_meta(jctx);
    auto rd = std::make_shared<server_response_reader>(jctx->server.get_response_reader());
    const int tid = rd->get_new_id();
    try {
        server_task task(task_type);
        task.id = tid;
        populate_completion_task(task, jctx, meta.logit_bias_eog, data, meta.has_mtmd, std::move(files));
        task.apply_chat_session(session);
        apply_raw_chat_sampling(task, raw_chat);
        task.params.res_type = res_type;
        name_completion_task(task, meta);
        std::vector<server_task> tasks;
        tasks.push_back(std::move(task));
        wake_and_post(*rd, std::move(tasks), session);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    jctx->readers[tid] = std::move(rd);
    return tid;
}

json stream_next(jllama_context *jctx, int id_task, bool &stop) {
    std::shared_ptr<server_response_reader> rd = reader_of(jctx, id_task);
    try {
        // Upstream b9437 added is_begin partial results whose to_json() returns a nullptr sentinel
        // meaning "HTTP-headers-only, no body". Loop past those so a consumer only ever sees real
        // events.
        while (true) {
            server_task_result_ptr result = rd->next([jctx] { return jctx->closing.load(); });
            require_result_ok(result);
            json payload = result->to_json();
            if (payload.is_null()) {
                continue;
            }
            stop = result->is_stop();
            if (stop) {
                erase_reader(jctx, id_task);
            }
            return payload;
        }
    } catch (...) {
        // The task is over for the caller whatever failed (a failed result, a throwing to_json()):
        // release its reader entry like every other error path.
        erase_reader(jctx, id_task);
        throw;
    }
}

void task_release(jllama_context *jctx, int id_task) { erase_reader(jctx, id_task); }

std::string chat_completions(jllama_context *jctx, const std::string &body_text) {
    json body = parse_json_body(body_text);
    std::vector<raw_buffer> files;
    common_chat_session session;
    json data = parse_oai_chat_params(jctx, body, files, session);
    auto results = blocking_completion(jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CHAT,
                                       std::move(files), session);
    // Serialisation goes through upstream safe_json_to_str (dump with error_handler_t::replace) so
    // content ending in an incomplete UTF-8 sequence yields U+FFFD instead of throwing.
    return safe_json_to_str(results_to_json(results));
}

int chat_completions_stream_request(jllama_context *jctx, const std::string &body_text, bool oai_chunks) {
    json body = parse_json_body(body_text);
    std::vector<raw_buffer> files;
    common_chat_session session;
    json data = parse_oai_chat_params(jctx, body, files, session);
    return streaming_request(jctx, data, SERVER_TASK_TYPE_COMPLETION,
                             oai_chunks ? TASK_RESPONSE_TYPE_OAI_CHAT : TASK_RESPONSE_TYPE_NONE, std::move(files),
                             session);
}

std::string chat_completions_stream_receive(jllama_context *jctx, int id_task, bool &stop) {
    stop = false;
    json payload = stream_next(jctx, id_task, stop);
    return safe_json_to_str(wrap_stream_chunk(std::move(payload), stop));
}

void set_log_sink(common_log_callback callback, void *user_data) {
    std::lock_guard<std::mutex> swap_lock(g_set_log_sink_mutex);
    // Detach the current sink first: this pauses common_log's worker, which drains every queued
    // entry through the OLD callback (a synchronous flush for the caller that sets null) and joins.
    common_log_set_callback(common_log_main(), nullptr, nullptr);
    if (callback != nullptr) {
        common_log_set_callback(common_log_main(), callback, user_data);
        // llama/ggml lines feed common_log even before the first model load runs common_init(),
        // which re-points llama_log_set() at this same default callback anyway.
        llama_log_set(common_log_default_callback, nullptr);
    }
}

// ---------------------------------------------------------------------------
// The rest of LlamaModel (B.3).
// ---------------------------------------------------------------------------

std::string model_meta(jllama_context *jctx) {
    if (jctx->vocab_only) {
        json meta = {
            // static_cast: an unscoped enum binds to common_json_value(bool) and would serialise as
            // true/false -- see the "vocab_type" note in docs/history/llama-cpp-breaking-changes.md
            {"vocab_type", static_cast<int>(llama_vocab_type(jctx->vocab))},
            {"n_vocab", llama_vocab_n_tokens(jctx->vocab)},
            {"special_tokens", special_tokens_json(jctx->vocab)},
        };
        return safe_json_to_str(meta);
    }
    auto m = wake_and_get_meta(jctx);
    // general.architecture from the GGUF metadata. Sized dynamically: llama_model_meta_val_str
    // returns the required length for a null buffer, so a long name is never truncated; a char
    // vector keeps the API's terminating '\0' off a std::string's internal terminator slot.
    std::string arch;
    const llama_model *mdl = llama_get_model(jctx->server.get_llama_context());
    if (mdl != nullptr) {
        const int need = llama_model_meta_val_str(mdl, "general.architecture", nullptr, 0);
        if (need > 0) {
            std::vector<char> buf(static_cast<size_t>(need) + 1, '\0');
            llama_model_meta_val_str(mdl, "general.architecture", buf.data(), buf.size());
            arch.assign(buf.data());
        }
    }
    json j = {
        // static_cast: see the vocab_only branch above
        {"vocab_type", static_cast<int>(m.model_vocab_type)},
        {"n_vocab", m.model_vocab_n_tokens},
        {"n_ctx_train", m.model_n_ctx_train},
        {"n_embd", m.model_n_embd_inp},
        {"n_params", m.model_n_params},
        {"size", m.model_size},
        // All three modalities upstream tracks, as its own /props emits them.
        {"modalities", {{"vision", m.has_inp_image}, {"audio", m.has_inp_audio}, {"video", m.has_inp_video}}},
        {"name", m.model_name},
        {"architecture", arch},
        {"ftype", m.model_ftype},
        // The placeholder a raw multimodal prompt puts where each image goes ({"prompt_string": ...,
        // "multimodal_data": [...]}). Random per process unless LLAMA_MEDIA_MARKER pins it, so a caller
        // reads it here, as upstream's /props publishes it.
        {"media_marker", get_media_marker()},
        // What the loaded pair can do beyond reading the modalities above, as upstream's server
        // decides it from mtmd_helper_get_capabilities() (b11901, #31183): chat is false only for a
        // TTS-only model (an audio-generating projector and no chat template), audio_gen is speech
        // output (the embedded server's POST /tts), audio_stream realtime transcription (/v1/realtime).
        {"capabilities",
         {{"chat", m.has_cap_chat}, {"audio_gen", m.has_cap_tts}, {"audio_stream", m.has_audio_stream}}},
    };
    // The input/output modalities of upstream's GET /models `architecture` object (b11429, #29987),
    // flattened to the top level because "architecture" above already names the GGUF architecture
    // string. A native decision model reports ["decisions"].
    {
        const json modalities = server_model_architecture_json(m.has_inp_image, m.has_inp_audio, m.has_inp_video,
                                                               m.model_output_modalities);
        j["input_modalities"] = modalities.at("input_modalities");
        j["output_modalities"] = modalities.at("output_modalities");
    }
    // Resolved default chat template (Jinja); empty when the model ships none.
    const char *chat_tmpl = mdl != nullptr ? llama_model_chat_template(mdl, /*name*/ nullptr) : nullptr;
    j["chat_template"] = chat_tmpl != nullptr ? std::string(chat_tmpl) : std::string();
    j["special_tokens"] = special_tokens_json(jctx->vocab);
    // Full GGUF metadata key/value map.
    if (mdl != nullptr) {
        json meta_map = json::object();
        const int meta_count = llama_model_meta_count(mdl);
        for (int i = 0; i < meta_count; i++) {
            char key_buf[256] = {};
            // 2 KB/value cap: scalar metadata fits; huge array values (tokenizer tokens/merges)
            // truncate rather than bloating the JSON.
            char val_buf[2048] = {};
            if (llama_model_meta_key_by_index(mdl, i, key_buf, sizeof(key_buf)) >= 0 &&
                llama_model_meta_val_str_by_index(mdl, i, val_buf, sizeof(val_buf)) >= 0) {
                meta_map[std::string(key_buf)] = std::string(val_buf);
            }
        }
        j["metadata"] = std::move(meta_map);
    }
    return safe_json_to_str(j);
}

std::string completions_oai(jllama_context *jctx, const std::string &body_text) {
    const json body = parse_json_body(body_text);
    json data;
    try {
        data = oaicompat_completion_params_parse(body);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
    return safe_json_to_str(
        results_to_json(blocking_completion(jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CMPL)));
}

// The FIM check of llama.cpp's /infill handler (handle_infill). Upstream answers 501 here, a
// server-side error. The tokens come from server_context_meta (populated from the same
// llama_vocab_fim_* calls inside server-context).
static void require_fim_tokens(const server_context_meta &meta) {
    if (meta.fim_pre_token == LLAMA_TOKEN_NULL || meta.fim_sub_token == LLAMA_TOKEN_NULL ||
        meta.fim_mid_token == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Model does not support fill-in-the-middle infill");
    }
}

// The rest of handle_infill's preparation: both fields required, input_extra defaulted, and the
// prompt formatted around the FIM tokens into data["prompt"] (an optional "prompt" of the request is
// the text after the middle token). Shared by the infill endpoint and the raw completion paths,
// which take input_prefix/input_suffix as an infill request -- upstream formats the infill prompt in
// its HTTP handler alone and runs an infill task in the slot like a completion, so a completion
// request carrying the fields used to complete its (mostly empty) prompt and drop both.
static void prepare_infill_prompt(jllama_context *jctx, const server_context_meta &meta, json &data) {
    for (const char *field : {"input_prefix", "input_suffix"}) {
        if (!data.contains(field)) {
            throw jllama_invalid_request(std::string("\"") + field + "\" is required");
        }
    }
    data["input_extra"] = json_value(data, "input_extra", json::array());
    const std::string prompt = json_value(data, "prompt", std::string());
    try {
        std::vector<server_tokens> tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, prompt, false, true, mtmd_helper_init_opt_default());
        data["prompt"] =
            format_prompt_infill(jctx->vocab, data.at("input_prefix"), data.at("input_suffix"), data.at("input_extra"),
                                 jctx->params.n_batch, jctx->params.n_predict, meta.slot_n_ctx, jctx->params.spm_infill,
                                 tokenized_prompts.empty() ? llama_tokens() : tokenized_prompts[0].get_tokens());
    } catch (const std::exception &e) {
        rethrow_by_type(e);
    }
}

// The request fields oaicompat_chat_params_parse consumes that have no effect on a raw completion:
// response_format becomes the session's grammar, tools/tool_choice/parallel_tool_calls its tool-call
// grammar with its triggers and preserved tokens, top_logprobs the n_probs of a chat answer and
// thinking_budget_tokens an alias of reasoning_budget_tokens, which the raw paths take under its own
// name. The completion schema reads none of them, and the raw paths take from the parser only the
// rendered prompt, the template's stops and its reasoning budget, so each used to be dropped without
// a word. The fields that act while rendering (chat_template_kwargs, reasoning_effort,
// add_generation_prompt, continue_final_message) and the completion fields the schema reads itself
// (json_schema, grammar, logprobs as n_probs, reasoning_budget_tokens, stop) stay allowed.
static constexpr const char *CHAT_ONLY_FIELDS[] = {"response_format",     "tools",        "tool_choice",
                                                   "parallel_tool_calls", "top_logprobs", "thinking_budget_tokens"};

// The stop strings a chat template adds of its own (upstream's Laguna workaround adds "</assistant>",
// which the model can spell out as text), which the chat paths append to the request's stops in
// server_task::apply_chat_session. The raw paths post no session, so they are written into "stop"
// here, after the stops the completion schema would take: the request's (a string or an array), or
// the CLI defaults when it names none -- the schema falls back to those only when "stop" holds no
// word. An array element that is no string is left for the schema to refuse.
static void add_template_stops(jllama_context *jctx, json &data, const std::vector<std::string> &template_stops) {
    if (template_stops.empty()) {
        return;
    }
    json stops = json::array();
    const json requested = data.contains("stop") ? data.at("stop") : json();
    if (requested.is_string()) {
        stops.push_back(requested);
    } else if (requested.is_array()) {
        for (const auto &word : requested) {
            if (!word.is_string() || !word.get<std::string>().empty()) {
                stops.push_back(word);
            }
        }
    }
    if (stops.empty()) {
        for (const std::string &word : jctx->params.antiprompt) {
            stops.push_back(word);
        }
    }
    for (const std::string &word : template_stops) {
        bool present = false;
        for (const auto &existing : stops) {
            present = present || (existing.is_string() && existing.get<std::string>() == word);
        }
        if (!present) {
            stops.push_back(word);
        }
    }
    data["stop"] = std::move(stops);
}

// The reasoning-budget fields oaicompat_chat_params_parse derives from a template that declares
// thinking tags: the budget (the request's reasoning_budget_tokens, else the server's default), the
// template's start and end tags, the message forced before the end tag and reasoning_control. The
// completion schema reads each of them, so a raw completion with "messages" takes them as the chat
// paths do -- the parser's values win over the request's, as in the parser's own copy of the request.
// A template without thinking tags yields none of them, and the budget then has nothing to act on,
// on the chat paths as here.
static constexpr const char *REASONING_BUDGET_FIELDS[] = {"reasoning_budget_tokens", "reasoning_budget_start_tag",
                                                          "reasoning_budget_end_tags", "reasoning_budget_message",
                                                          "reasoning_control"};

// A raw completion request with "messages" completes the prompt the chat template renders from them,
// as applyTemplate renders it; the other fields stay completion fields. The field used to be read on
// the chat paths alone (oaicompat_chat_params_parse), so a raw completion ignored it and completed
// the prompt next to it, mostly the empty one. A non-empty prompt, the infill fields or one of
// CHAT_ONLY_FIELDS beside the messages are refused rather than dropped; an empty prompt is no prompt
// (InferenceParameters("") writes one). The media files of array-form content come back in files.
// The template's stop strings and its reasoning budget are carried over (add_template_stops,
// REASONING_BUDGET_FIELDS), and what the chat paths hand to the sampler is returned for the raw paths to
// apply (see apply_raw_chat_sampling): the template's preserved tokens, and its generation prompt when
// the budget is carried over.
static raw_chat_sampling render_messages_into_prompt(jllama_context *jctx, json &data, std::vector<raw_buffer> &files) {
    for (const char *field : CHAT_ONLY_FIELDS) {
        if (data.contains(field) && !data.at(field).is_null()) {
            throw jllama_invalid_request(std::string("'") + field +
                                         "' is a chat field and has no effect on a raw completion with 'messages': "
                                         "send the request as a chat completion");
        }
    }
    if (is_infill_request(data)) {
        throw jllama_invalid_request(
            "'messages' cannot be combined with 'input_prefix'/'input_suffix': send an infill request or a chat");
    }
    if (data.contains("prompt") && !(data.at("prompt").is_string() && data.at("prompt").get<std::string>().empty())) {
        throw jllama_invalid_request(
            "'messages' and 'prompt' cannot both be set: the messages are rendered into the prompt");
    }
    json body = data;
    common_chat_session session; // the raw path parses no chat output; the prompt and the stops are used
    const json chat = parse_oai_chat_params(jctx, body, files, session);
    if (!chat.contains("prompt") || !chat.at("prompt").is_string()) {
        throw std::runtime_error("the chat template did not produce a string prompt");
    }
    data["prompt"] = chat.at("prompt");
    add_template_stops(jctx, data, session.additional_stops());
    raw_chat_sampling raw_chat;
    common_params_sampling applied; // the session's sampler fields, as apply_chat_session takes them
    session.apply_sampling(applied);
    raw_chat.preserved_tokens = std::move(applied.preserved_tokens);
    if (!chat.contains("reasoning_budget_end_tags")) {
        return raw_chat;
    }
    for (const char *field : REASONING_BUDGET_FIELDS) {
        if (chat.contains(field)) {
            data[field] = chat.at(field);
        }
    }
    raw_chat.generation_prompt = session.generation_prompt();
    return raw_chat;
}

// A raw completion request with input_prefix/input_suffix is an infill one: refused like the infill
// endpoint refuses it (no FIM tokens, a field missing), otherwise its prompt formatted the same way.
// One with "messages" completes their templated prompt, and raw_chat receives what
// render_messages_into_prompt returns. Returns the task type to post.
static server_task_type prepare_raw_completion(jllama_context *jctx, json &data, std::vector<raw_buffer> &files,
                                               raw_chat_sampling &raw_chat) {
    if (data.contains("messages")) {
        raw_chat = render_messages_into_prompt(jctx, data, files);
        return SERVER_TASK_TYPE_COMPLETION;
    }
    if (!is_infill_request(data)) {
        return SERVER_TASK_TYPE_COMPLETION;
    }
    const auto meta = wake_and_get_meta(jctx);
    require_fim_tokens(meta);
    prepare_infill_prompt(jctx, meta, data);
    return SERVER_TASK_TYPE_INFILL;
}

std::string infill(jllama_context *jctx, const std::string &body_text) {
    const auto meta = wake_and_get_meta(jctx);
    require_fim_tokens(meta);
    json data = parse_json_body(body_text);
    prepare_infill_prompt(jctx, meta, data);
    return safe_json_to_str(
        results_to_json(blocking_completion(jctx, data, SERVER_TASK_TYPE_INFILL, TASK_RESPONSE_TYPE_NONE)));
}

std::string completions(jllama_context *jctx, const std::string &body_text) {
    json data = parse_json_body(body_text);
    std::vector<raw_buffer> files;
    raw_chat_sampling raw_chat;
    const server_task_type type = prepare_raw_completion(jctx, data, files, raw_chat);
    return safe_json_to_str(results_to_json(
        blocking_completion(jctx, data, type, TASK_RESPONSE_TYPE_NONE, std::move(files), {}, raw_chat)));
}

int completions_stream_request(jllama_context *jctx, const std::string &body_text) {
    json data = parse_json_body(body_text);
    std::vector<raw_buffer> files;
    raw_chat_sampling raw_chat;
    const server_task_type type = prepare_raw_completion(jctx, data, files, raw_chat);
    return streaming_request(jctx, data, type, TASK_RESPONSE_TYPE_NONE, std::move(files), {}, raw_chat);
}

int completions_oai_stream_request(jllama_context *jctx, const std::string &body_text) {
    const json body = parse_json_body(body_text);
    json data;
    try {
        data = oaicompat_completion_params_parse(body);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
    data["stream"] = true;
    return streaming_request(jctx, data, SERVER_TASK_TYPE_COMPLETION, TASK_RESPONSE_TYPE_OAI_CMPL);
}

std::string completions_stream_receive(jllama_context *jctx, int id_task, bool &stop) {
    stop = false;
    json payload = stream_next(jctx, id_task, stop);
    payload["stop"] = stop;
    return safe_json_to_str(payload);
}

std::string apply_template(jllama_context *jctx, const std::string &body_text) {
    json body = parse_json_body(body_text);
    std::vector<raw_buffer> files;
    common_chat_session session; // built with the prompt, not needed afterwards
    const json data = parse_oai_chat_params(jctx, body, files, session);
    if (!data.contains("prompt") || !data.at("prompt").is_string()) {
        throw std::runtime_error("applyTemplate did not produce a string prompt");
    }
    const std::string prompt = data.at("prompt");
    return prompt;
}

std::vector<float> embed(jllama_context *jctx, const std::string &prompt) {
    // Idle sleep frees the model and invalidates the cached vocab; wake and re-sync first.
    wake_server(jctx);
    if (!jctx->params.embedding) {
        throw std::runtime_error(
            "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
    }
    SRV_INF("Calling embedding '%s'\n", prompt.c_str());
    llama_tokens tokens;
    try {
        tokens = tokenize_mixed(jctx->vocab, prompt, true, true);
    } catch (const std::exception &e) {
        rethrow_by_type(e);
    }
    auto rd = jctx->server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_EMBEDDING);
    task.id = rd.get_new_id();
    task.tokens = server_tokens(tokens, false);
    task.index = 0;
    wake_and_post(rd, std::move(task));
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    require_batch_ok(br);
    if (br.results.empty()) {
        throw std::runtime_error("embedding result is empty");
    }
    auto *embd_result = dynamic_cast<server_task_result_embd *>(br.results[0].get());
    if (embd_result == nullptr || embd_result->embedding.empty() || embd_result->embedding[0].empty()) {
        throw std::runtime_error("embedding result is empty");
    }
    SRV_INF("Embedding has %d columns\n", static_cast<int>(embd_result->embedding[0].size()));
    return embd_result->embedding[0];
}

std::string embeddings(jllama_context *jctx, const std::string &body_text, bool oai_compat) {
    if (!jctx->params.embedding) {
        throw std::runtime_error(
            "Model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
    }
    task_response_type res_type = oai_compat ? TASK_RESPONSE_TYPE_OAI_EMBD : TASK_RESPONSE_TYPE_NONE;
    {
        auto meta = wake_and_get_meta(jctx);
        if (res_type != TASK_RESPONSE_TYPE_NONE && meta.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            throw jllama_invalid_request(
                "Pooling type 'none' is not OAI compatible. Please use a different pooling type");
        }
    }
    const json body = parse_json_body(body_text);
    bool force_no_oaicompat = false;
    json prompt;
    bool use_base64 = false;
    try {
        prompt = extract_embedding_prompt(body, force_no_oaicompat);
        use_base64 = parse_encoding_format(body);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
    if (force_no_oaicompat) {
        res_type = TASK_RESPONSE_TYPE_NONE;
    }
    std::vector<server_tokens> tokenized_prompts;
    try {
        tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, prompt, true, true, mtmd_helper_init_opt_default());
    } catch (const std::exception &e) {
        // `"prompt" must not be empty` is a std::invalid_argument, as upstream throws it
        rethrow_by_type(e);
    }
    for (const auto &toks : tokenized_prompts) {
        if (toks.get_tokens().empty()) {
            throw jllama_invalid_request("Input content cannot be empty");
        }
    }
    auto rd = jctx->server.get_response_reader();
    std::vector<server_task> tasks;
    tasks.reserve(tokenized_prompts.size());
    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        tasks.push_back(indexed_token_task(rd, SERVER_TASK_TYPE_EMBEDDING,
                                           server_tokens(tokenized_prompts[i].get_tokens(), false), static_cast<int>(i),
                                           res_type));
    }
    wake_and_post(rd, std::move(tasks));
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    require_batch_ok(br);
    json responses = json::array();
    for (const auto &result : br.results) {
        responses.push_back(result->to_json());
    }
    const json out =
        (res_type == TASK_RESPONSE_TYPE_OAI_EMBD)
            ? format_embeddings_response_oaicompat(
                  body, json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), responses, use_base64)
            : responses;
    return safe_json_to_str(out);
}

std::string rerank(jllama_context *jctx, const std::string &query, const std::vector<std::string> &documents) {
    {
        auto meta = wake_and_get_meta(jctx);
        if (!jctx->params.embedding || meta.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            throw std::runtime_error(
                "This server does not support reranking. Start it with `--reranking` and without `--embedding`");
        }
    }
    const llama_model *model = llama_get_model(jctx->server.get_llama_context());
    auto rd = jctx->server.get_response_reader();
    std::vector<server_task> tasks;
    tasks.reserve(documents.size());
    for (size_t i = 0; i < documents.size(); i++) {
        tasks.push_back(indexed_token_task(rd, SERVER_TASK_TYPE_RERANK,
                                           format_prompt_rerank(model, jctx->vocab, nullptr, query, documents[i], {},
                                                                {}, mtmd_helper_init_opt_default()),
                                           static_cast<int>(i), TASK_RESPONSE_TYPE_NONE));
    }
    wake_and_post(rd, std::move(tasks));
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    require_batch_ok(br);
    // rerank_results_to_json throws std::invalid_argument on a malformed/out-of-range result index:
    // a broken result, not the caller's fault.
    try {
        return safe_json_to_str(rerank_results_to_json(br.results, documents));
    } catch (const std::exception &e) {
        throw std::runtime_error(e.what());
    }
}

std::vector<llama_token> tokenize(jllama_context *jctx, const std::string &text, bool add_special) {
    // Idle sleep frees the model and invalidates the cached vocab; wake and re-sync first.
    wake_server(jctx);
    return tokenize_mixed(jctx->vocab, text, add_special, true);
}

std::string tokenize_json(jllama_context *jctx, const std::string &text, bool add_special, bool with_pieces) {
    const llama_tokens tokens = tokenize(jctx, text, add_special);
    json tokens_response = json::array();
    if (with_pieces) {
        llama_context *lctx = jctx->vocab_only ? nullptr : jctx->server.get_llama_context();
        for (const auto &token : tokens) {
            std::string piece;
            if (lctx != nullptr) {
                piece = common_token_to_piece(lctx, token);
            } else {
                char buf[256];
                const int n = llama_token_to_piece(jctx->vocab, token, buf, static_cast<int>(sizeof(buf)), 0, false);
                piece = n > 0 ? std::string(buf, n) : std::string();
            }
            tokens_response.push_back({{"id", token}, {"piece", token_piece_value(piece)}});
        }
    } else {
        tokens_response = tokens;
    }
    return safe_json_to_str(format_tokenizer_response(tokens_response));
}

std::string detokenize(jllama_context *jctx, const std::vector<llama_token> &tokens) {
    wake_server(jctx);
    if (jctx->vocab_only) {
        return tokens_to_str(jctx->vocab, tokens);
    }
    return tokens_to_str(jctx->server.get_llama_context(), tokens);
}

std::string detokenize_json(jllama_context *jctx, const std::vector<llama_token> &tokens) {
    return safe_json_to_str(format_detokenized_response(detokenize(jctx, tokens)));
}

std::string metrics(jllama_context *jctx) {
    // b10408 (upstream #26920) reduced server_task_result_metrics::to_json() to the slot array and
    // b10519 (#27376) split the task in two: METRICS keeps only the counters (its to_json() is unused
    // and returns JSON null; to_metrics() renders them as Prometheus text) and SLOT_GET carries the
    // slot array plus the idle-slot count. Post both and merge them so getMetrics() keeps returning
    // the single documented object.
    auto metrics_result = post_and_wait(jctx, server_task(SERVER_TASK_TYPE_METRICS));
    auto slots_result = post_and_wait(jctx, server_task(SERVER_TASK_TYPE_SLOT_GET));
    const auto *counters = dynamic_cast<const server_task_result_metrics *>(metrics_result.get());
    const auto *slots = dynamic_cast<const server_task_result_slots *>(slots_result.get());
    if (counters == nullptr || slots == nullptr) {
        throw std::runtime_error("Unexpected result type for server metrics");
    }
    return safe_json_to_str(server_metrics_to_json(*counters, *slots));
}

// Saving and restoring follow upstream's slots route (handle_slots_save/handle_slots_restore): the
// file is a name, validated by fs_validate_filename (no directory part, no ".."), inside the
// --slot-save-path directory, and without that directory the actions are not offered (upstream
// answers 501 there, before the name is looked at). The C API used to take the name as the path, so
// the directory had no effect and a file could be written anywhere the process may write. Erasing
// needs no file and stays available.
std::string slot_action(jllama_context *jctx, slot_op op, int id_slot, const std::string &filename) {
    if (op == slot_op::erase) {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.slot_action.id_slot = id_slot;
        return one_shot(jctx, std::move(task));
    }
    const std::string &directory = jctx->params.slot_save_path; // the parser ends it with a separator
    if (directory.empty()) {
        throw std::runtime_error("Saving or restoring a slot needs a slot directory: load the model with "
                                 "--slot-save-path <dir> (ModelParameters.setSlotSavePath)");
    }
    if (!fs_validate_filename(filename)) {
        throw jllama_invalid_request("Invalid slot file name '" + filename +
                                     "': name a file inside the --slot-save-path directory, without a directory part");
    }
    server_task task(op == slot_op::save ? SERVER_TASK_TYPE_SLOT_SAVE : SERVER_TASK_TYPE_SLOT_RESTORE);
    task.slot_action.id_slot = id_slot;
    task.slot_action.filename = filename;
    task.slot_action.filepath = directory + filename;
    return one_shot(jctx, std::move(task));
}

std::string lora_adapters(jllama_context *jctx) { return one_shot(jctx, server_task(SERVER_TASK_TYPE_GET_LORA)); }

std::string set_lora_adapters(jllama_context *jctx, const std::string &adapters_json) {
    const json data = parse_json_body(adapters_json);
    if (!data.is_array()) {
        // Same contract as the upstream POST /lora-adapters route body.
        throw std::runtime_error("LoRA adapter list must be a JSON array of {id, scale} objects");
    }
    server_task task(SERVER_TASK_TYPE_SET_LORA);
    task.set_lora = parse_lora_request(data);
    return one_shot(jctx, std::move(task));
}

void configure(jllama_context *jctx, const std::string &config_json) {
    const json config = parse_json_body(config_json);
    std::optional<float> slot_sim_opt;
    std::optional<int> n_threads_opt;
    std::optional<int> n_threads_batch_opt;
    try {
        slot_sim_opt = parse_slot_prompt_similarity(config);
        n_threads_opt = parse_positive_int_config(config, "n_threads");
        n_threads_batch_opt = parse_positive_int_config(config, "n_threads_batch");
    } catch (const std::exception &e) {
        throw std::runtime_error(e.what());
    }
    // n_threads / n_threads_batch through the public llama.h API. The setter requires both values;
    // a missing one is filled from the cached common_params so a single-field update is a no-op for
    // the other field, and the cache is kept in sync so a follow-up call reads back what was applied.
    if (n_threads_opt.has_value() || n_threads_batch_opt.has_value()) {
        llama_context *lctx = jctx->server.get_llama_context();
        if (lctx == nullptr) {
            throw std::runtime_error("configureParallelInference: llama_context not available "
                                     "(model sleeping or not loaded)");
        }
        const int n = n_threads_opt.value_or(jctx->params.cpuparams.n_threads);
        const int nb = n_threads_batch_opt.value_or(jctx->params.cpuparams_batch.n_threads);
        llama_set_n_threads(lctx, n, nb);
        jctx->params.cpuparams.n_threads = n;
        jctx->params.cpuparams_batch.n_threads = nb;
    }
    // slot_prompt_similarity: validated above ([0.0, 1.0]). Live mutation uses
    // server_context::set_slot_prompt_similarity(), carried as patches/0003 (upstream #22393).
    if (slot_sim_opt.has_value()) {
        jctx->server.set_slot_prompt_similarity(*slot_sim_opt);
    }
}

std::string systemone(jllama_context *jctx, const std::string &body) {
    // Forwarded to upstream's route handler (b11361, #29818) rather than re-implemented: the request
    // parsing, the per-model prompt layout, the shared-prefix grouping and the answer formatting live
    // in server_decision_context, private to server_context_impl and reachable only through
    // server_routes. The handler waits for the server to leave sleep itself (create_response()).
    require_server(jctx, "systemone");
    const std::function<bool()> should_stop = [jctx] { return jctx->closing.load(); };
    const server_http_req req{{}, {}, "/v1/systemone", "", body, {}, should_stop};
    server_http_res_ptr res;
    try {
        res = jctx->routes->post_systemone(req);
    } catch (const std::exception &e) {
        rethrow_by_type(e);
    }
    if (res->status != 200) {
        // the handler's own verdict: server_res_generator::error() sets the status from the error code
        const std::string message = route_error_message(res->data);
        if (res->status == 400) {
            throw jllama_invalid_request(message);
        }
        throw std::runtime_error(message);
    }
    return res->data;
}

std::string json_schema_to_grammar_text(const std::string &schema_json) {
    const json schema = json::parse(schema_json);
    return json_schema_to_grammar(schema);
}

void quantize(const std::string &input_path, const std::string &output_path, int ftype, int nthread,
              bool allow_requantize) {
    // Idempotent; intentionally never paired with llama_backend_free here -- a model loaded in the
    // same process shares the backend and must not have it freed underneath it.
    llama_backend_init();
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.ftype = static_cast<llama_ftype>(ftype);
    qparams.nthread = nthread;
    qparams.allow_requantize = allow_requantize;
    const uint32_t rc = llama_model_quantize(input_path.c_str(), output_path.c_str(), &qparams);
    if (rc != 0) {
        throw std::runtime_error("Quantization of '" + input_path + "' failed with code " + std::to_string(rc));
    }
}

} // namespace jllama::engine
