// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// jllama_context.hpp — the native model handle, with no JNI in it.
//
// jllama_context is what both call paths into this library hold: the Java side as the `ctx` jlong
// field of LlamaModel, a C consumer as the opaque `jllama_model *` of jllama_api.h. Everything here
// is JNI-free on purpose -- the C API (jllama_api.cpp) and the engine behind both paths
// (jllama_engine.cpp) include this header; jni_helpers.hpp includes it and adds the Java-handle
// half of the user-reference protocol (acquire_jllama_context_impl reads the field).
//
// Include order: upstream server headers (server-context.h, server-queue.h, server-task.h,
// server-common.h, server-chat.h) must be included by the translation unit BEFORE this header.

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

// ---------------------------------------------------------------------------
// jllama_context
//
// Owns a server_context (value member, pimpl inside) and the background
// worker thread.  Stored as the Java-side `ctx` (jlong) pointer, and handed
// to C callers as the opaque jllama_model of jllama_api.h.
// ---------------------------------------------------------------------------
struct jllama_context {
    server_context server; // value member (pimpl inside)
    // The upstream HTTP route handlers over `server`, for endpoints this layer forwards to upstream
    // instead of re-implementing them (`handleSystemOne` -> `post_systemone`). Constructed before
    // `server.load_model()`, as upstream does: its constructor registers a sleeping-state callback,
    // and upstream relies on that callback running before the server's own, which frees the model.
    // Declared after `server`, so it is destroyed first. Null in vocab-only mode.
    std::unique_ptr<server_routes> routes;
    std::thread worker;
    bool vocab_only = false;
    std::atomic<bool> worker_ready{false};

    // Cached after load_model() — valid for the lifetime of this context.
    const llama_vocab *vocab = nullptr;
    // Non-null only in vocab-only mode (bypasses server_context entirely).
    llama_model *vocab_only_model = nullptr;

    // Saved copy of common_params used to load the model.
    // Required by server_schema::eval_llama_cmpl_schema which takes common_params&.
    common_params params;

    // Per-streaming-task response readers, keyed by task id.
    // Guarded by readers_mutex.
    std::mutex readers_mutex;
    // shared_ptr (not unique_ptr): the streaming receive paths copy the reader out under
    // readers_mutex and then call next() *without* the lock, so a concurrent close()/erase
    // that removes the map entry must not free a reader still in use — the receiver's copy
    // keeps it alive until next() returns.
    std::map<int, std::shared_ptr<server_response_reader>> readers;

    // In-flight call reference count (JNI entry points and C API calls alike), for safe teardown. close() waits for
    // this to reach 0 before tearing down the worker/models and deleting the context, so a concurrent inference/receive
    // call can never dereference freed state (close()-vs-inference use-after-free, close()-during-streaming hang).
    // acquire_jllama_context_impl increments it; the jllama_context_guard (set up by REQUIRE_SERVER_CONTEXT) decrements
    // it on scope exit.
    std::atomic<int> users{0};
    // Set by delete() (teardown) so any in-flight reader parked in next()/wait_for_all()
    // observes a stop signal and unwinds instead of polling forever (which would otherwise
    // keep `users` > 0 and hang close()). The streaming/blocking should_stop lambdas read this.
    std::atomic<bool> closing{false};
    // Serializes the "last user released" notification (paired with cv). The users
    // decrement in release_jllama_context_impl happens UNDER this lock so delete()'s
    // cv.wait cannot observe users==0 and free m/cv before the releaser finishes
    // touching them (otherwise a use-after-free on the condvar/mutex themselves).
    std::mutex m;
    std::condition_variable cv;
};

// Removes the streaming reader entry for `id_task` under the readers_mutex.
// No-op if the id is not in the map. Used to release the JNI side of a
// completed, cancelled, or failed streaming task.
inline void erase_reader(jllama_context *jctx, int id_task) {
    std::lock_guard<std::mutex> lk(jctx->readers_mutex);
    jctx->readers.erase(id_task);
}

// The user-reference acquire on a handle that is already known to be live -- what the C API
// does with the pointer its caller holds (a C caller owns the handle's lifetime; jllama_model_free
// still waits for in-flight calls, the way close() does for JNI calls). The JNI entry points go
// through acquire_jllama_context_impl (jni_helpers.hpp), which reads the handle from the Java
// field under g_ctx_mutex so a close() that nulled it is never raced.
[[nodiscard]] inline jllama_context *acquire_jllama_context(jllama_context *jctx) {
    if (jctx != nullptr) {
        jctx->users.fetch_add(1, std::memory_order_relaxed);
    }
    return jctx;
}

inline void release_jllama_context_impl(jllama_context *jctx) {
    if (jctx == nullptr) {
        return;
    }
    // Decrement under jctx->m so delete()'s cv.wait (which holds jctx->m while checking the
    // predicate) cannot observe users==0 and free m/cv before this releaser has finished
    // notifying — that would be a use-after-free on the condvar/mutex themselves.
    std::lock_guard<std::mutex> lk(jctx->m);
    // fetch_sub returns the previous value; when it was 1 we just dropped the last reference.
    if (jctx->users.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        jctx->cv.notify_all();
    }
}

// RAII guard: releases the jllama_context user reference at scope exit, including
// on early return. REQUIRE_SERVER_CONTEXT (JNI) and the C API's calls use it so every
// entry point is covered.
struct jllama_context_guard {
    jllama_context *ptr;
    ~jllama_context_guard() { release_jllama_context_impl(ptr); }
};
