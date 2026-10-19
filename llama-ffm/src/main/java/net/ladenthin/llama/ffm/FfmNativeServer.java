// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.cStrings;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import net.ladenthin.llama.exception.LlamaException;

/**
 * The embedded llama-server ({@code NativeServer}): upstream's full HTTP server, WebUI included, on a
 * worker thread inside the library. The argv is llama-server's own ({@code --port}, {@code -m}, ...);
 * {@link #startAttached} serves an already loaded {@link FfmLlamaModel} instead of loading its own.
 * One server per process; the Java side does not enforce it, a second start fails or misbehaves.
 */
public final class FfmNativeServer implements AutoCloseable {

    /** The two native calls on a running server's handle; a seam so the locking is testable without it. */
    interface Backend {

        /**
         * Whether the worker behind {@code handle} still runs.
         *
         * @param handle the server handle
         * @return {@code true} while it serves
         */
        boolean isRunning(MemorySegment handle);

        /**
         * Stops the worker, waits for it and frees {@code handle}.
         *
         * @param handle the server handle
         */
        void stop(MemorySegment handle);
    }

    /** The real calls into the library. */
    private static final Backend NATIVE = new Backend() {
        @Override
        public boolean isRunning(MemorySegment handle) {
            return (boolean) invoke(JllamaNative.SERVER_IS_RUNNING, handle);
        }

        @Override
        public void stop(MemorySegment handle) {
            JllamaNative.call(JllamaNative.SERVER_STOP, handle);
        }
    };

    private final MemorySegment handle;
    private final Backend backend;

    /**
     * Serializes {@link #isRunning()} and {@link #close()} over {@link #handle}, as in
     * {@code NativeServer}: the native stop frees the handle, so a query that is inside its native
     * call when another thread closes would read freed memory (shown with this seam: {@code close()}
     * returned while {@code isRunning()} was still inside the call).
     */
    private final Object lifecycleLock = new Object();

    /** Set by {@link #close()} under {@link #lifecycleLock}. */
    private boolean closed;

    private FfmNativeServer(MemorySegment handle) {
        this(handle, NATIVE);
    }

    /**
     * A server over {@code handle} that calls {@code backend}; for tests.
     *
     * @param handle the server handle
     * @param backend the native calls
     */
    FfmNativeServer(MemorySegment handle, Backend backend) {
        this.handle = handle;
        this.backend = backend;
    }

    /**
     * Starts the server with its own model (classic mode), or in router mode without a model argument.
     * The call returns as soon as the server's worker runs; a model that then fails to load ends the
     * worker without an exception, visible as {@link #isRunning()} being {@code false}.
     *
     * @param args llama-server's arguments, without the program name
     * @return the server; check {@link #isRunning()} when the model may fail to load
     * @throws LlamaException when the worker cannot be started at all
     */
    public static FfmNativeServer start(String... args) {
        Objects.requireNonNull(args, "args");
        try (Arena arena = Arena.ofConfined()) {
            String[] argv = withProgramName(args);
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.SERVER_START, argv.length, cStrings(arena, argv), out);
            return new FfmNativeServer(out.get(ADDRESS, 0));
        }
    }

    /**
     * Starts the HTTP frontend over a loaded model (attach mode): the model's own worker keeps driving
     * inference, the argv configures the HTTP side only. Close the server before the model.
     *
     * @param model the loaded model
     * @param args the HTTP-side arguments ({@code --port}, {@code --host}, ...)
     * @return the running server
     * @throws LlamaException when it cannot start
     */
    public static FfmNativeServer startAttached(FfmLlamaModel model, String... args) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(args, "args");
        try (Arena arena = Arena.ofConfined()) {
            String[] argv = withProgramName(args);
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.SERVER_START_ATTACHED, model.handle(), argv.length, cStrings(arena, argv),
                    out);
            return new FfmNativeServer(out.get(ADDRESS, 0));
        }
    }

    /**
     * Sets (or clears, with {@code null}) the command router mode spawns its model workers with
     * ({@code LLAMA_SERVER_WORKER_CMD}), e.g. a {@code java -cp ... NativeServer} line.
     *
     * @param command the command, or {@code null} to clear it
     */
    public static void setWorkerCommand(String command) {
        try (Arena arena = Arena.ofConfined()) {
            JllamaNative.call(JllamaNative.SERVER_SET_WORKER_COMMAND, cString(arena, command == null ? "" : command));
        }
    }

    /**
     * Whether the server's worker still runs.
     *
     * @return {@code true} while it serves
     */
    public boolean isRunning() {
        synchronized (lifecycleLock) {
            return !closed && backend.isRunning(handle);
        }
    }

    /**
     * Signals shutdown, waits for the worker and releases the handle; a second call is a no-op. Waits
     * for an {@link #isRunning()} call in progress, which must not query a handle being freed.
     */
    @Override
    public void close() {
        synchronized (lifecycleLock) {
            if (!closed) {
                closed = true;
                backend.stop(handle);
            }
        }
    }

    private static String[] withProgramName(String[] args) {
        String[] argv = new String[args.length + 1];
        argv[0] = "llama-server";
        System.arraycopy(args, 0, argv, 1, args.length);
        return argv;
    }
}
