// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// JNI bridge for net.ladenthin.llama.RpcServerNative, the native half of RpcServer: serves this
// process's devices to llama.cpp RPC clients (`--rpc host:port` on another llama.cpp /
// java-llama.cpp instance), the in-JVM counterpart of upstream's rpc-server binary.
//
// Every entry point is a shell over the C API (jllama_rpc_server_* in jllama_api.h). The server
// loop is ggml's own ggml_backend_rpc_start_server(), which blocks for the life of the server;
// RpcServer runs it on a Java thread of its own and ends it with ggml_backend_rpc_stop_server()
// (added by patches/0015, together with ggml_backend_rpc_server_listening()). ggml keeps the server
// state in file-scope globals, so only ONE server may run per process; RpcServer enforces that.
//
// Security: the protocol has no authentication and no encryption. RpcServer binds to loopback
// unless the caller asks for a network address explicitly.

// Upstream server headers must precede jni_helpers.hpp (include order rule, see CLAUDE.md).
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

// Standard-UTF-8 jstring extraction, defined in jllama.cpp (see native_server.cpp for why).
std::string parse_jstring(JNIEnv *env, jstring java_string);

namespace {

jclass rpc_exception_class(JNIEnv *env) { return env->FindClass("net/ladenthin/llama/exception/LlamaException"); }

void throw_rpc_exception(JNIEnv *env, const char *message) {
    jclass exception_class = rpc_exception_class(env);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL Java_net_ladenthin_llama_RpcServerNative_serveNative(JNIEnv *env, jclass, jstring jhost,
                                                                            jint port, jint threads, jstring jcache_dir,
                                                                            jstring jdevices) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> void {
        const std::string host = parse_jstring(env, jhost);
        const std::string cache_dir = jcache_dir != nullptr ? parse_jstring(env, jcache_dir) : std::string();
        const std::string devices = parse_jstring(env, jdevices);
        // Blocks until the server is stopped, or returns at once when the socket cannot be bound;
        // RpcServer tells the two apart with serverListeningNative().
        if (jllama_rpc_server_serve(host.c_str(), static_cast<int>(port), static_cast<int>(threads),
                                    cache_dir.empty() ? nullptr : cache_dir.c_str(),
                                    devices.c_str()) != JLLAMA_STATUS_OK) {
            throw_rpc_exception(env, jllama_last_error());
        }
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_RpcServerNative_stopNative(JNIEnv *env, jclass) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> void {
        if (jllama_rpc_server_stop() != JLLAMA_STATUS_OK) {
            throw_rpc_exception(env, jllama_last_error());
        }
    });
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_RpcServerNative_serverListeningNative(JNIEnv *env, jclass) {
    return jni_guard_impl(env, rpc_exception_class(env),
                          [&]() -> jboolean { return jllama_rpc_server_listening() ? JNI_TRUE : JNI_FALSE; });
}

JNIEXPORT jobjectArray JNICALL Java_net_ladenthin_llama_RpcServerNative_serverDevicesNative(JNIEnv *env, jclass,
                                                                                            jstring jdevices) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> jobjectArray {
        const std::string devices = parse_jstring(env, jdevices);
        char **names = nullptr;
        size_t count = 0;
        if (jllama_rpc_server_devices(devices.c_str(), &names, &count) != JLLAMA_STATUS_OK) {
            throw_rpc_exception(env, jllama_last_error());
            return nullptr;
        }
        jclass string_class = env->FindClass("java/lang/String");
        jobjectArray out =
            string_class != nullptr ? env->NewObjectArray(static_cast<jsize>(count), string_class, nullptr) : nullptr;
        if (out != nullptr) {
            for (size_t i = 0; i < count; ++i) {
                // device names are ASCII identifiers such as "CPU" or "CUDA0"
                jstring name = env->NewStringUTF(names[i]);
                env->SetObjectArrayElement(out, static_cast<jsize>(i), name);
                env->DeleteLocalRef(name);
            }
        }
        jllama_strings_free(names, count);
        return out;
    });
}

} // extern "C"
