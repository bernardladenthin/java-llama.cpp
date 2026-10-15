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
#include "utils.hpp"
#include "jllama_engine.hpp"
#include "json_helpers.hpp"
#include "rpc_support.hpp"

#include <chrono>
#include <mutex>
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
    if (!configure_multimodal_task_impl(task, has_mtmd, data, std::move(files))) {
        auto tokenized_prompts =
            tokenize_input_prompts(jctx->vocab, nullptr, data.at("prompt"), true, true, mtmd_helper_init_opt_default());
        if (!tokenized_prompts.empty()) {
            task.tokens = std::move(tokenized_prompts[0]);
        }
    }
    task.params = server_schema::eval_llama_cmpl_schema(jctx->vocab, jctx->params, logit_bias_eog, data);
    configure_task_slot_impl(task, data);
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

std::vector<server_task_result_ptr> blocking_completion(jllama_context *jctx, const json &data,
                                                        server_task_type task_type, task_response_type res_type,
                                                        std::vector<raw_buffer> files,
                                                        const common_chat_session &session) {
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
    task.params.res_type = res_type;
    std::vector<server_task> tasks;
    tasks.push_back(std::move(task));
    wake_and_post(rd, std::move(tasks), session);
    auto br = rd.wait_for_all([jctx] { return jctx->closing.load(); });
    if (br.error) {
        throw_result_error(br.error);
    }
    if (br.is_terminated) {
        // wait_for_all was cut short because free() set closing: the results vector is only
        // partially filled (null entries), so don't dereference it -- unwind cleanly instead.
        throw jllama_interrupted("inference interrupted by close()");
    }
    return std::move(br.results);
}

int streaming_request(jllama_context *jctx, const json &data, server_task_type task_type, task_response_type res_type,
                      std::vector<raw_buffer> files, const common_chat_session &session) {
    auto meta = wake_and_get_meta(jctx);
    auto rd = std::make_shared<server_response_reader>(jctx->server.get_response_reader());
    const int tid = rd->get_new_id();
    try {
        server_task task(task_type);
        task.id = tid;
        populate_completion_task(task, jctx, meta.logit_bias_eog, data, meta.has_mtmd, std::move(files));
        task.apply_chat_session(session);
        task.params.res_type = res_type;
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
    json body;
    try {
        body = json::parse(body_text);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
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
    json body;
    try {
        body = json::parse(body_text);
    } catch (const std::exception &e) {
        throw jllama_invalid_request(e.what());
    }
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

} // namespace jllama::engine
