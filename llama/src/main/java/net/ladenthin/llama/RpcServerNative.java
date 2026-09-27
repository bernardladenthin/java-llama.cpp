// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.loader.LlamaLoader;
import org.jspecify.annotations.Nullable;

/**
 * The native half of {@link RpcServer} ({@code rpc_bridge.cpp}): ggml's own RPC server loop plus
 * the two entry points {@code patches/0015} adds to it. Kept apart from {@link RpcServer} so that
 * class loads, and its lifecycle is testable, without the native library.
 */
final class RpcServerNative implements RpcServer.Backend {

    static {
        LlamaLoader.initialize();
    }

    /** The one instance; the native server state is process-wide anyway. */
    static final RpcServerNative INSTANCE = new RpcServerNative();

    private RpcServerNative() {}

    @Override
    public void serve(String host, int port, int threads, @Nullable String cacheDir) {
        serveNative(host, port, threads, cacheDir);
    }

    @Override
    public void stop() {
        stopNative();
    }

    @Override
    public boolean listening() {
        return serverListeningNative();
    }

    @Override
    public String @Nullable [] devices() {
        return serverDevicesNative();
    }

    private static native void serveNative(String host, int port, int threads, @Nullable String cacheDir);

    private static native void stopNative();

    private static native boolean serverListeningNative();

    private static native String @Nullable [] serverDevicesNative();
}
