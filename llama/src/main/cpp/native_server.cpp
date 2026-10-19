// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// JNI bridge for net.ladenthin.llama.server.NativeServer: runs the full upstream llama.cpp HTTP
// server (llama_server(), including its embedded WebUI) inside libjllama, driven over JNI. The
// argv is forwarded verbatim from Java, so every llama-server flag is supported. This is an
// independent server lifecycle (it loads its own model from the argv), distinct from LlamaModel
// and the Java-side OpenAiCompatServer.
//
// Every entry point is a shell over the C API (jllama_server_* in jllama_api.h), whose engine is
// native_server_engine.cpp. Only ONE native server may run per process (server.cpp keeps its
// shutdown state in file-scope globals); NativeServer enforces this on the Java side.

// Upstream server headers must precede jni_helpers.hpp (include order rule, see CLAUDE.md);
// they provide server_context so the attach path can reach a LlamaModel's jllama_context.
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"
#include "jllama_api.h"

#include <jni.h>

#include <string>
#include <vector>

// Standard-UTF-8 jstring extraction, defined in jllama.cpp (String.getBytes("UTF-8") via the
// method/charset handles cached in JNI_OnLoad — always initialized before any native method can
// run). Reused here instead of a second extraction path: GetStringUTFChars yields Modified-UTF-8
// (CESU-8), which would mangle non-BMP chars (emoji, CJK extensions) in argv such as model paths.
std::string parse_jstring(JNIEnv *env, jstring java_string);

namespace {

// The forwarded Java argv with a synthetic argv[0], as C strings for the C API.
struct forwarded_argv {
    std::vector<std::string> args;
    std::vector<const char *> argv;
    forwarded_argv(JNIEnv *env, jobjectArray jargs) {
        const jsize n = (jargs != nullptr) ? env->GetArrayLength(jargs) : 0;
        args.reserve(static_cast<size_t>(n) + 1);
        args.emplace_back("llama-server"); // argv[0]
        for (jsize i = 0; i < n; ++i) {
            auto js = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
            if (js != nullptr) {
                args.emplace_back(parse_jstring(env, js));
                env->DeleteLocalRef(js);
            } else {
                args.emplace_back("");
            }
        }
        argv.reserve(args.size());
        for (const auto &arg : args) {
            argv.push_back(arg.c_str());
        }
    }
    int argc() const { return static_cast<int>(argv.size()); }
};

// Resolves net.ladenthin.llama.exception.LlamaException. Unlike jllama.cpp this TU caches no
// global class reference, so the lookup happens per call; every entry point here is a cold
// lifecycle operation (start/stop/isRunning), never an inference hot path.
jclass llama_exception_class(JNIEnv *env) { return env->FindClass("net/ladenthin/llama/exception/LlamaException"); }

// Throws net.ladenthin.llama.exception.LlamaException with the given message (best-effort: if the
// class cannot be resolved the pending NoClassDefFoundError is surfaced instead).
void throw_llama_exception(JNIEnv *env, const char *message) {
    jclass exception_class = llama_exception_class(env);
    if (exception_class != nullptr) {
        throw_utf8_impl(env, exception_class, message);
    }
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startNativeServer(JNIEnv *env, jclass,
                                                                                       jobjectArray jargs) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> jlong {
        const forwarded_argv args(env, jargs);
        jllama_server *srv = nullptr;
        if (jllama_server_start(args.argc(), args.argv.data(), &srv) != JLLAMA_STATUS_OK) {
            throw_llama_exception(env, jllama_last_error());
            return 0;
        }
        return reinterpret_cast<jlong>(srv);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_stopNativeServer(JNIEnv *env, jclass,
                                                                                     jlong handle) {
    return jni_guard_impl(env, llama_exception_class(env),
                          [&]() -> void { jllama_server_stop(reinterpret_cast<jllama_server *>(handle)); });
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_server_NativeServer_isRunningNative(JNIEnv *env, jclass,
                                                                                        jlong handle) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> jboolean {
        return jllama_server_is_running(reinterpret_cast<jllama_server *>(handle)) ? JNI_TRUE : JNI_FALSE;
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
        // The handle protocol of every other JNI entry point (field read + user increment under
        // g_ctx_mutex), so a concurrent LlamaModel.close() cannot free jctx between the read and
        // the start; the engine takes its own reference for the worker's lifetime.
        jllama_context *jctx = acquire_jllama_context_impl(env, jmodel, ctx_field);
        if (jctx == nullptr) {
            throw_llama_exception(env, "model is not loaded (or already closed)");
            return 0;
        }
        jllama_context_guard guard{jctx};
        const forwarded_argv args(env, jargs);
        jllama_server *srv = nullptr;
        if (jllama_server_start_attached(reinterpret_cast<jllama_model *>(jctx), args.argc(), args.argv.data(), &srv) !=
            JLLAMA_STATUS_OK) {
            throw_llama_exception(env, jllama_last_error());
            return 0;
        }
        return reinterpret_cast<jlong>(srv);
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_setWorkerCommandNative(JNIEnv *env, jclass,
                                                                                           jstring jcommand) {
    return jni_guard_impl(env, llama_exception_class(env), [&]() -> void {
        std::string value;
        if (jcommand != nullptr) {
            value = parse_jstring(env, jcommand);
        }
        if (jllama_server_set_worker_command(value.c_str()) != JLLAMA_STATUS_OK) {
            throw_llama_exception(env, jllama_last_error());
        }
    });
}

} // extern "C"
