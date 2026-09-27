// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// JNI bridge for net.ladenthin.llama.RpcServer: serves this process's devices to llama.cpp RPC
// clients (`--rpc host:port` on another llama.cpp / java-llama.cpp instance), the in-JVM
// counterpart of upstream's rpc-server binary.
//
// The server loop is ggml's own ggml_backend_rpc_start_server(), which blocks for the life of the
// server; RpcServer runs it on a Java thread of its own and ends it with ggml_backend_rpc_stop_server()
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
#include "rpc_support.hpp"

#include <jni.h>

#include <stdexcept>
#include <string>
#include <vector>

// Standard-UTF-8 jstring extraction, defined in jllama.cpp (see native_server.cpp for why).
std::string parse_jstring(JNIEnv *env, jstring java_string);

namespace {

jclass rpc_exception_class(JNIEnv *env) { return env->FindClass("net/ladenthin/llama/exception/LlamaException"); }

} // namespace

extern "C" {

JNIEXPORT void JNICALL Java_net_ladenthin_llama_RpcServer_serveNative(JNIEnv *env, jclass, jstring jhost, jint port,
                                                                      jint threads, jstring jcache_dir) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> void {
        const std::string endpoint = parse_jstring(env, jhost) + ":" + std::to_string(port);
        std::string cache_dir;
        if (jcache_dir != nullptr) {
            cache_dir = parse_jstring(env, jcache_dir);
        }
        auto devices = jllama::rpc::server_devices();
        if (devices.empty()) {
            throw std::runtime_error("no device to serve over RPC");
        }
        // Blocks until ggml_backend_rpc_stop_server(), or returns at once when the socket cannot be
        // bound; RpcServer tells the two apart with serverListeningNative().
        ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir.empty() ? nullptr : cache_dir.c_str(),
                                      static_cast<size_t>(threads), devices.size(), devices.data());
    });
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_RpcServer_stopNative(JNIEnv *env, jclass) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> void { ggml_backend_rpc_stop_server(); });
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_RpcServer_serverListeningNative(JNIEnv *env, jclass) {
    return jni_guard_impl(env, rpc_exception_class(env),
                          [&]() -> jboolean { return ggml_backend_rpc_server_listening() ? JNI_TRUE : JNI_FALSE; });
}

JNIEXPORT jobjectArray JNICALL Java_net_ladenthin_llama_RpcServer_serverDevicesNative(JNIEnv *env, jclass) {
    return jni_guard_impl(env, rpc_exception_class(env), [&]() -> jobjectArray {
        const auto devices = jllama::rpc::server_devices();
        jclass string_class = env->FindClass("java/lang/String");
        if (string_class == nullptr) {
            return nullptr;
        }
        jobjectArray out = env->NewObjectArray(static_cast<jsize>(devices.size()), string_class, nullptr);
        if (out == nullptr) {
            return nullptr;
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            // device names are ASCII identifiers such as "CPU" or "CUDA0"
            jstring name = env->NewStringUTF(ggml_backend_dev_name(devices[i]));
            env->SetObjectArrayElement(out, static_cast<jsize>(i), name);
            env->DeleteLocalRef(name);
        }
        return out;
    });
}

} // extern "C"
