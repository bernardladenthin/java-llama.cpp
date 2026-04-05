#pragma once

// jni_helpers.hpp — JNI utility helpers for jllama.cpp
//
// Extracted from jllama.cpp so that the core logic can be tested without a
// running JVM.  The single public entry point is get_server_context_impl(),
// which validates the Java-side model handle and returns the native
// server_context pointer.  All module-level globals are passed explicitly so
// the function is self-contained and unit-testable with mock JNI environments.

#include "jni.h"

#include <atomic>
#include <thread>

// Forward declaration — callers that need the full definition must include
// server.hpp themselves.
struct server_context;

// ---------------------------------------------------------------------------
// jllama_context
//
// Owns a server_context and the background worker thread.  Stored as the
// Java-side `ctx` (jlong) pointer.  Using a wrapper allows us to join the
// thread on close() instead of detaching it, which eliminates the race
// between thread teardown and JVM shutdown.
// ---------------------------------------------------------------------------
struct jllama_context {
    server_context *server       = nullptr;
    std::thread     worker;
    bool            vocab_only   = false;
    // Signals that the worker thread has entered start_loop() and is ready.
    // Without this, terminate() can race with start_loop() setting running=true.
    std::atomic<bool> worker_ready{false};
};

// ---------------------------------------------------------------------------
// get_server_context_impl
//
// Reads the native handle stored in the Java LlamaModel object, validates it,
// and returns the embedded server_context pointer.
//
// On success: returns a non-null server_context*.
// On failure: throws "Model is not loaded" via JNI and returns nullptr.
//
// Parameters are passed explicitly (no module-level globals) so the function
// can be exercised from unit tests using a mock JNIEnv.
// ---------------------------------------------------------------------------
inline server_context *get_server_context_impl(JNIEnv *env, jobject obj,
                                               jfieldID field_id,
                                               jclass   error_class) {
    const jlong handle = env->GetLongField(obj, field_id);
    if (handle == 0) {
        env->ThrowNew(error_class, "Model is not loaded");
        return nullptr;
    }
    return reinterpret_cast<jllama_context *>(handle)->server; // NOLINT(*-no-int-to-ptr)
}
