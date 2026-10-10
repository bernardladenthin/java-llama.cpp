// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// JNI bridge for net.ladenthin.llama.server.NativeServer: runs the full upstream llama.cpp HTTP
// server (llama_server(), including its embedded WebUI) inside libjllama, driven over JNI. The
// argv is forwarded verbatim from Java, so every llama-server flag is supported. This is an
// independent server lifecycle (it loads its own model from the argv), distinct from LlamaModel
// and the Java-side OpenAiCompatServer.
//
// Only ONE native server may run per process: server.cpp keeps its shutdown_handler /
// is_terminating state in file-scope globals, so a second concurrent llama_server() would clobber
// them. NativeServer enforces this on the Java side.

// Upstream server headers must precede jni_helpers.hpp (include order rule, see CLAUDE.md);
// they provide server_context so the attach path can reach a LlamaModel's jllama_context.
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"

#include "native_server_bridge.h"
#include "rpc_support.hpp"

#include <jni.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Standard-UTF-8 jstring extraction, defined in jllama.cpp (String.getBytes("UTF-8") via the
// method/charset handles cached in JNI_OnLoad — always initialized before any native method can
// run). Reused here instead of a second extraction path: GetStringUTFChars yields Modified-UTF-8
// (CESU-8), which would mangle non-BMP chars (emoji, CJK extensions) in argv such as model paths.
std::string parse_jstring(JNIEnv *env, jstring java_string);

namespace {

// Owns the argv storage for the lifetime of the running server plus the worker thread that runs
// llama_server(). The argv pointers reference the std::string storage in `args`, which is filled
// once (with reserve) and never mutated afterwards, so the pointers stay valid.
struct native_server {
    std::vector<std::string> args; // args[0] is the program name ("llama-server")
    std::vector<char *> argv;      // points into `args`
    std::thread worker;
    std::atomic<bool> finished{false};
    int exit_code = -1;
};

// Copies the forwarded Java argv into srv->args/argv with a synthetic argv[0]. The argv pointers
// reference the std::string storage in `args`, which is filled once (with reserve) and never
// mutated afterwards, so the pointers stay valid for the worker's lifetime.
void fill_native_server_args(JNIEnv *env, jobjectArray jargs, native_server *srv) {
    const jsize n = (jargs != nullptr) ? env->GetArrayLength(jargs) : 0;
    srv->args.reserve(static_cast<size_t>(n) + 1);
    srv->args.emplace_back("llama-server"); // argv[0]
    for (jsize i = 0; i < n; ++i) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
        if (js != nullptr) {
            srv->args.emplace_back(parse_jstring(env, js));
            env->DeleteLocalRef(js);
        } else {
            srv->args.emplace_back("");
        }
    }
}

// Points srv->argv into srv->args. Called once, after the last change to srv->args.
void bind_native_server_argv(native_server *srv) {
    srv->argv.clear();
    srv->argv.reserve(srv->args.size());
    for (auto &arg : srv->args) {
        srv->argv.push_back(const_cast<char *>(arg.c_str()));
    }
}

// Resolves net.ladenthin.llama.exception.LlamaException. Unlike jllama.cpp this TU caches no
// global class reference, so the lookup happens per call; every entry point here is a cold
// lifecycle operation (start/stop/isRunning), never an inference hot path.
jclass llama_exception_class(JNIEnv *env) { return env->FindClass("net/ladenthin/llama/exception/LlamaException"); }

// Throws net.ladenthin.llama.exception.LlamaException with the given message (best-effort: if the
// class cannot be resolved the pending NoClassDefFoundError is surfaced instead).
void throw_llama_exception(JNIEnv *env, const char *message) {
    jclass exception_class = llama_exception_class(env);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startNativeServer(JNIEnv *env, jclass,
                                                                                       jobjectArray jargs) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> jlong {
        std::unique_ptr<native_server> owned(new native_server());
        fill_native_server_args(env, jargs, owned.get());
        // Same RPC preparation as LlamaModel's load (rpc_support.hpp): an unreachable `--rpc`
        // server fails here with a clear message, and RPC devices another model registered in
        // this JVM stay out of this server's model. Throws std::invalid_argument -> LlamaException.
        owned->args = jllama::rpc::prepare_argv(owned->args);
        bind_native_server_argv(owned.get());
        auto *srv = owned.release();

        // Embedded mode: no process signal handlers, honor the forwarded argv (see patches/0006).
        llama_server_set_embedded(true);

        srv->worker = std::thread([srv]() {
            srv->exit_code = llama_server(static_cast<int>(srv->argv.size()), srv->argv.data());
            srv->finished.store(true);
        });

        return reinterpret_cast<jlong>(srv);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_stopNativeServer(JNIEnv *env, jclass,
                                                                                     jlong handle) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> void {
        auto *srv = reinterpret_cast<native_server *>(handle);
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
    });
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_server_NativeServer_isRunningNative(JNIEnv *env, jclass,
                                                                                        jlong handle) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> jboolean {
        auto *srv = reinterpret_cast<native_server *>(handle);
        return (srv != nullptr && !srv->finished.load()) ? JNI_TRUE : JNI_FALSE;
    });
}

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startAttachedNativeServer(JNIEnv *env, jclass,
                                                                                               jobject jmodel,
                                                                                               jobjectArray jargs) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> jlong {
        if (jmodel == nullptr) {
            throw_llama_exception(env, "model must not be null");
            return 0;
        }
        // Resolve the LlamaModel's native context handle (the same "ctx" long field jllama.cpp
        // caches in JNI_OnLoad; this TU resolves it itself to stay decoupled from those globals).
        jclass model_class = env->GetObjectClass(jmodel);
        jfieldID ctx_field = env->GetFieldID(model_class, "ctx", "J");
        if (ctx_field == nullptr) {
            return 0; // NoSuchFieldError already pending
        }
        // Acquire a jllama_context user reference through the same handle protocol as every other
        // JNI entry point (field read + user increment under g_ctx_mutex), so a concurrent
        // LlamaModel.close() cannot free jctx between the handle read and the increment. The
        // reference is held for the lifetime of the attach worker: delete() in jllama.cpp waits on
        // the user-count condition variable, so close() blocks until this thread exits (close the
        // server before the model — see the NativeServer attach-constructor Javadoc).
        jllama_context *jctx = acquire_jllama_context_impl(env, jmodel, ctx_field);
        if (jctx == nullptr) {
            throw_llama_exception(env, "model is not loaded (or already closed)");
            return 0;
        }

        // Contract of the attach path: llama_server_attach drives the HTTP frontend on a NEW worker
        // thread but must NOT start a second task-processing loop on the shared server_context — the
        // model's own worker (spawned in load_model_impl) already runs start_loop(). Likewise,
        // stopNativeServer's llama_server_request_shutdown() must terminate only the HTTP frontend,
        // not the attached model's worker, so the LlamaModel remains usable after the server closes
        // (documented order: close server first, then model). If a future upstream change breaks
        // either assumption, this attach path needs revisiting.
        // The routes are the model's own (jllama_context::routes, built before its load as upstream
        // does), never a second server_routes: its constructor registers a sleeping-state callback on
        // the model's task queue, which has no unregister, so a server_routes that died with the attach
        // left a callback into freed memory that the model's next idle sleep ran. Consequently the
        // endpoint toggles (--metrics, --props, --slots) and every other route-level setting come from
        // the model's ModelParameters; the attach argv configures the HTTP side.
        if (!jctx->routes) {
            release_jllama_context_impl(jctx);
            throw_llama_exception(env, "a vocab-only model has no server context to attach to");
            return 0;
        }
        auto *srv = new native_server();
        fill_native_server_args(env, jargs, srv);
        bind_native_server_argv(srv);

        // The attach entry always parses the forwarded argv; set the embedded flag anyway so any
        // shared embedded-mode behavior in server.cpp stays consistent with startNativeServer.
        llama_server_set_embedded(true);

        server_context *ctx_server = &jctx->server;
        server_routes *routes = jctx->routes.get();
        srv->worker = std::thread([srv, ctx_server, routes, jctx]() {
            srv->exit_code =
                llama_server_attach(static_cast<int>(srv->argv.size()), srv->argv.data(), *ctx_server, *routes);
            srv->finished.store(true);
            release_jllama_context_impl(jctx);
        });

        return reinterpret_cast<jlong>(srv);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_setWorkerCommandNative(JNIEnv *env, jclass,
                                                                                           jstring jcommand) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> void {
        // Sets/clears LLAMA_SERVER_WORKER_CMD in the process environment, which the router-mode
        // model manager (server-models.cpp, patches/0008) reads when spawning worker instances.
        std::string value;
        if (jcommand != nullptr) {
            value = parse_jstring(env, jcommand);
        }
#if defined(_WIN32)
        _putenv_s("LLAMA_SERVER_WORKER_CMD", value.c_str()); // empty value removes the variable
#else
        if (value.empty()) {
            unsetenv("LLAMA_SERVER_WORKER_CMD");
        } else {
            setenv("LLAMA_SERVER_WORKER_CMD", value.c_str(), 1);
        }
#endif
    });
}

} // extern "C"
