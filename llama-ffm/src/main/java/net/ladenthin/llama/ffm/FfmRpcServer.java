// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.longSlot;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;
import static net.ladenthin.llama.ffm.JllamaNative.takeStrings;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import net.ladenthin.llama.exception.LlamaException;

/**
 * The in-process ggml RPC server ({@code RpcServer}): serves this process's devices to llama.cpp RPC
 * clients ({@code --rpc host:port}). One server per process, state in ggml's globals, no
 * authentication and no encryption: bind to loopback unless the network is trusted.
 */
public final class FfmRpcServer {

    private FfmRpcServer() {}

    /**
     * Serves until {@link #stop()}: <strong>blocks the calling thread</strong> for the life of the server,
     * or returns at once when the socket cannot be bound (check {@link #isListening()} from another
     * thread to tell the two apart, as {@code RpcServer} does). A caller that gives up waiting for the
     * server to listen stops it the way {@link #stop()} describes: repeatedly until this call has
     * returned, since a stop before the serving thread has entered it is lost.
     *
     * @param host the IPv4 literal to bind
     * @param port the port
     * @param threads the compute threads, {@code 0} for the default
     * @param cacheDir the tensor cache directory, or {@code null}
     * @param devices the devices to serve, comma-separated ({@code "CPU"}, {@code "CUDA0,CPU"}), or empty
     *        for every accelerator else the CPU
     * @throws LlamaException when the server cannot start
     */
    public static void serve(String host, int port, int threads, String cacheDir, String devices) {
        Objects.requireNonNull(host, "host");
        try (Arena arena = Arena.ofConfined()) {
            JllamaNative.call(JllamaNative.RPC_SERVER_SERVE, cString(arena, host), port, threads,
                    cString(arena, cacheDir), cString(arena, devices == null ? "" : devices));
        }
    }

    /**
     * Stops the server and returns once {@link #serve} has returned, also for a stop that arrives before
     * the socket is bound. A no-op when no {@link #serve} call is in progress, and it is for the
     * {@link #serve} calls in progress when it is made: one started while the stop still waits (a
     * restart from another thread) is not stopped. A stop issued before the serving thread has entered
     * {@link #serve} is lost, so stopping a server that may not have come up yet means repeating this
     * until {@link #serve} has returned.
     *
     * @throws LlamaException when {@link #serve} has not returned 30 seconds after the request
     */
    public static void stop() {
        JllamaNative.call(JllamaNative.RPC_SERVER_STOP);
    }

    /**
     * Whether the server listens.
     *
     * @return {@code true} while it accepts clients
     */
    public static boolean isListening() {
        return (boolean) invoke(JllamaNative.RPC_SERVER_LISTENING);
    }

    /**
     * The names of the devices a selection resolves to, with the rule of {@link #serve}.
     *
     * @param devices the selection, or empty for the default choice
     * @return the device names
     * @throws LlamaException for an unknown name, with the available ones in the message
     */
    public static List<String> devices(String devices) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment names = pointerSlot(arena);
            MemorySegment count = longSlot(arena);
            JllamaNative.call(JllamaNative.RPC_SERVER_DEVICES, cString(arena, devices == null ? "" : devices), names,
                    count);
            return takeStrings(names, count);
        }
    }
}
