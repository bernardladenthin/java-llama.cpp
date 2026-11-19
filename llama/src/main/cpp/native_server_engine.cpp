// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Upstream server headers first (include order rule, see CLAUDE.md): they provide server_context
// for jllama_context, whose server the attach path drives.
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "native_server_engine.hpp"
#include "jllama_engine.hpp"
#include "native_server_bridge.h"
#include "rpc_support.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace jllama::server_engine {

namespace {

// Points srv->argv into srv->args. Called once, after the last change to srv->args: the pointers
// reference the std::string storage, which is never mutated afterwards.
void bind_argv(native_server *srv) {
    srv->argv.clear();
    srv->argv.reserve(srv->args.size());
    for (auto &arg : srv->args) {
        srv->argv.push_back(const_cast<char *>(arg.c_str()));
    }
}

} // namespace

native_server *start(const std::vector<std::string> &argv) {
    std::unique_ptr<native_server> owned(new native_server());
    // Same RPC preparation as a model load (rpc_support.hpp): an unreachable `--rpc` server fails
    // here with a clear message (std::invalid_argument), and RPC devices another model registered
    // in this process stay out of this server's model.
    owned->args = jllama::rpc::prepare_argv(argv);
    // The argv is parsed here, once, and the worker runs on these params (llama_server_parsed): a
    // refused argv, or a batch llama.cpp would abort the process on, fails this call -- before the
    // argv's log file is opened. A --log-file is opened here too; the server owns the file until it
    // returns, then the file is closed (unless a model or another server took it).
    owned->params = std::make_shared<common_params>();
    const std::uint64_t log_file = jllama::engine::prepare_server_argv(owned->args, *owned->params);
    bind_argv(owned.get());

    // Embedded mode: no process signal handlers, honour the forwarded argv (patches/0006).
    llama_server_set_embedded(true);
    native_server *srv = owned.get();
    try {
        srv->worker = std::thread([srv, log_file]() {
            srv->exit_code = llama_server_parsed(*srv->params, static_cast<int>(srv->argv.size()), srv->argv.data());
            jllama::engine::release_log_file_of_server(log_file);
            srv->finished.store(true);
        });
    } catch (...) {
        jllama::engine::release_log_file_of_server(log_file);
        throw;
    }
    return owned.release();
}

native_server *start_attached(jllama_context *jctx, const std::vector<std::string> &argv) {
    // Contract of the attach path: llama_server_attach drives the HTTP frontend on a NEW worker
    // thread but must NOT start a second task-processing loop on the shared server_context -- the
    // model's own worker already runs start_loop(). Likewise, stop()'s llama_server_request_shutdown()
    // terminates only the HTTP frontend, not the attached model's worker, so the model remains
    // usable after the server closes. The routes are the model's own (jllama_context::routes, built
    // before its load as upstream does), never a second server_routes: its constructor registers a
    // sleeping-state callback on the model's task queue, which has no unregister, so a server_routes
    // that died with the attach left a callback into freed memory that the model's next idle sleep
    // ran. Consequently the endpoint toggles and every other route-level setting come from the
    // model's parameters; the attach argv configures the HTTP side.
    if (!jctx->routes) {
        throw std::runtime_error("a vocab-only model has no server context to attach to");
    }
    std::unique_ptr<native_server> owned(new native_server());
    // The parse below sees the argv without its --rpc pairs, so their servers are registered (and an
    // unreachable one refused) here, before it takes the log lock -- as for start().
    owned->args = jllama::rpc::prepare_argv(argv);
    owned->params = std::make_shared<common_params>();
    const std::uint64_t log_file = jllama::engine::prepare_server_argv(owned->args, *owned->params);
    bind_argv(owned.get());

    // The attach entry always parses the forwarded argv; set the embedded flag anyway so any shared
    // embedded-mode behaviour in server.cpp stays consistent with start().
    llama_server_set_embedded(true);
    server_context *ctx_server = &jctx->server;
    server_routes *routes = jctx->routes.get();
    native_server *srv = owned.get();
    // The user reference held for the worker's lifetime: the model's free() waits for users == 0,
    // so a model is never freed under a running attach (close the server before the model). Taken
    // last: a start that fails before the worker runs gives it and the log file back.
    jllama_context *held = acquire_jllama_context(jctx);
    try {
        srv->worker = std::thread([srv, ctx_server, routes, held, log_file]() {
            srv->exit_code = llama_server_attach_parsed(*srv->params, *ctx_server, *routes);
            jllama::engine::release_log_file_of_server(log_file);
            srv->finished.store(true);
            release_jllama_context_impl(held);
        });
    } catch (...) {
        release_jllama_context_impl(held);
        jllama::engine::release_log_file_of_server(log_file);
        throw;
    }
    return owned.release();
}

void stop(native_server *srv) {
    if (srv == nullptr) {
        return;
    }
    // Signal shutdown, retrying until the worker actually returns: a stop issued before the server
    // finished starting (shutdown_handler not yet installed by llama_server) would otherwise be
    // lost. Once the handler is installed the first signal takes effect; if the model failed to
    // load, llama_server has already returned and `finished` is set.
    while (!srv->finished.load()) {
        llama_server_request_shutdown();
        if (srv->finished.load()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (srv->worker.joinable()) {
        srv->worker.join();
    }
    delete srv;
}

bool running(const native_server *srv) { return srv != nullptr && !srv->finished.load(); }

void set_worker_command(const std::string &value) {
#if defined(_WIN32)
    _putenv_s("LLAMA_SERVER_WORKER_CMD", value.c_str()); // empty value removes the variable
#else
    if (value.empty()) {
        unsetenv("LLAMA_SERVER_WORKER_CMD");
    } else {
        setenv("LLAMA_SERVER_WORKER_CMD", value.c_str(), 1);
    }
#endif
}

} // namespace jllama::server_engine
