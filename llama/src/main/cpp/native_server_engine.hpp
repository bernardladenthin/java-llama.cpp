// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// native_server_engine.hpp -- the embedded llama-server (NativeServer) without JNI: upstream's
// llama_server() (patches/0006) and llama_server_attach() (patches/0007) on a worker thread of
// their own, stoppable out of band. The C API (jllama_server_*) and the JNI shells of
// native_server.cpp are two thin layers over this. Compiled on every platform but Android, where
// server.cpp cannot be built (subprocess.h / posix_spawn_*).
//
// Only ONE server may run per process: server.cpp keeps its shutdown_handler / is_terminating
// state in file-scope globals, so a second concurrent llama_server() would clobber them. The Java
// side enforces that; a C caller has to.
//
// Include order: the upstream server headers must be included by the translation unit before this
// header (it needs jllama_context, i.e. server_context).

#include "jllama_context.hpp"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace jllama::server_engine {

// A running (or finished) server: the argv storage for its lifetime plus the worker thread.
struct native_server {
    std::vector<std::string> args; // args[0] is the program name
    std::vector<char *> argv;      // points into `args`
    std::thread worker;
    std::atomic<bool> finished{false};
    int exit_code = -1;
};

/**
 * Starts the full upstream server on a worker thread from a llama-server argv (argv[0] is a
 * program name, any). The argv is prepared like a model load (rpc_support.hpp): an unreachable
 * `--rpc` server throws std::invalid_argument. The caller owns the handle and ends it with stop().
 */
[[nodiscard]] native_server *start(const std::vector<std::string> &argv);

/**
 * Starts the HTTP frontend over an already-loaded model's server context (attach mode): the
 * model's own worker keeps driving inference, the routes are the model's own. Holds a user
 * reference on the model for the worker's lifetime, so freeing the model blocks until this server
 * is stopped (stop the server first). Throws std::runtime_error for a vocab-only model.
 */
[[nodiscard]] native_server *start_attached(jllama_context *jctx, const std::vector<std::string> &argv);

/** Signals shutdown until the worker has returned, joins it and deletes the handle; null is ignored. */
void stop(native_server *srv);

/** True while the worker runs (false for null and after the server returned on its own). */
[[nodiscard]] bool running(const native_server *srv);

/**
 * Sets (or, with an empty value, clears) LLAMA_SERVER_WORKER_CMD in the process environment, the
 * command router mode spawns its workers with (patches/0008).
 */
void set_worker_command(const std::string &value);

} // namespace jllama::server_engine
