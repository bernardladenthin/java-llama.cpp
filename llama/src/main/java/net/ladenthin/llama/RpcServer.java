// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.RpcEndpoint;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Serves this machine's devices to llama.cpp RPC clients: the in-JVM counterpart of upstream's
 * {@code rpc-server}. A client offloads model layers here with
 * {@link net.ladenthin.llama.parameters.ModelParameters#setRpcServers(RpcEndpoint...)} (or
 * {@code --rpc host:port} on any llama.cpp tool), so one model can use the GPUs of several
 * machines.
 *
 * <pre>{@code
 * try (RpcServer server = RpcServer.startLocal(RpcEndpoint.DEFAULT_PORT)) {
 *     // clients on this machine can now use --rpc 127.0.0.1:50052
 *     server.awaitTermination();
 * }
 * }</pre>
 *
 * <p>The server offers every accelerator this library found (CUDA, Vulkan, Metal, ...), or the
 * CPU when there is none; see {@link #servedDevices()}. It runs on a daemon thread of its own until
 * {@link #close()}.
 *
 * <p><strong>Security.</strong> The RPC protocol has no authentication and no encryption: anyone
 * who reaches the port can use the devices and read or write the tensors on them. The factories
 * therefore bind to loopback; {@link #startOnNetwork} is the explicit opt-in for another interface
 * and logs a warning. Use a trusted network or a tunnel (SSH, WireGuard).
 *
 * <p>Only one server may run per process (ggml keeps the server state in globals); a second
 * {@code start} while one is running throws {@link IllegalStateException}. It is not available on
 * Android builds without the {@code INTERNET} permission, which even a loopback socket needs there.
 */
public final class RpcServer implements AutoCloseable {

    /** Loopback, the only address the non-explicit factories bind to. */
    public static final String LOOPBACK = "127.0.0.1";

    /** How long {@code start} waits for the socket to be bound before it gives up. */
    static final long START_TIMEOUT_MILLIS = 10_000L;

    /** How long {@link #close()} waits for the server thread to end. */
    static final long STOP_TIMEOUT_MILLIS = 30_000L;

    private static final Logger LOGGER = LoggerFactory.getLogger(RpcServer.class);

    private static final AtomicBoolean ACTIVE = new AtomicBoolean();

    /**
     * The native server operations. A seam of its own so the lifecycle here -- single instance,
     * bind failure, start timeout, stop -- is testable without the native library; production code
     * always uses {@link RpcServerNative}.
     */
    interface Backend {
        /** Blocks for the life of the server, or returns at once when the socket cannot be bound. */
        void serve(String host, int port, int threads, @Nullable String cacheDir);

        /** Makes a running {@link #serve} return. */
        void stop();

        /** Whether {@link #serve} is accepting connections. */
        boolean listening();

        /** Names of the devices {@link #serve} would offer. */
        String @Nullable [] devices();
    }

    private final RpcEndpoint endpoint;
    private final Thread thread;
    private final Backend backend;
    private final AtomicBoolean closed = new AtomicBoolean();

    private RpcServer(RpcEndpoint endpoint, Thread thread, Backend backend) {
        this.endpoint = endpoint;
        this.thread = thread;
        this.backend = backend;
    }

    /**
     * Starts a server on loopback with the default thread count and no tensor cache.
     *
     * @param port the TCP port, e.g. {@link RpcEndpoint#DEFAULT_PORT}
     * @return the running server
     * @throws LlamaException when the port cannot be bound
     * @throws IllegalStateException when a server is already running in this process
     */
    public static RpcServer startLocal(int port) {
        return startLocal(port, Options.defaultThreads(), null);
    }

    /**
     * Starts a server on loopback.
     *
     * @param port the TCP port
     * @param threads CPU threads the served CPU device uses, at least 1
     * @param cacheDir a directory where the server caches large tensors a client uploads, so a
     *     repeated load of the same model sends them only once; {@code null} for no cache
     * @return the running server
     * @throws LlamaException when the port cannot be bound
     * @throws java.io.UncheckedIOException when the cache directory cannot be created
     * @throws IllegalStateException when a server is already running in this process
     */
    public static RpcServer startLocal(int port, int threads, @Nullable Path cacheDir) {
        return start(RpcEndpoint.of(LOOPBACK, port), threads, cacheDir, RpcServerNative.INSTANCE, START_TIMEOUT_MILLIS);
    }

    /**
     * Starts a server on another interface. The protocol is unauthenticated and unencrypted, so
     * this is the explicit opt-in: expose it only on a trusted network.
     *
     * @param bindAddress an IPv4 address of this machine, or {@code 0.0.0.0} for every interface
     * @param port the TCP port
     * @param threads CPU threads the served CPU device uses, at least 1
     * @param cacheDir tensor cache directory, or {@code null}
     * @return the running server
     * @throws LlamaException when the port cannot be bound
     * @throws IllegalArgumentException when the address is not an IPv4 literal
     * @throws IllegalStateException when a server is already running in this process
     */
    public static RpcServer startOnNetwork(String bindAddress, int port, int threads, @Nullable Path cacheDir) {
        Options.requireIpv4Literal(bindAddress);
        if (!LOOPBACK.equals(bindAddress)) {
            LOGGER.warn(
                    "RpcServer listens on {}:{}. The RPC protocol has no authentication and no encryption"
                            + " -- never expose it to an untrusted network.",
                    bindAddress,
                    port);
        }
        return start(
                RpcEndpoint.of(bindAddress, port), threads, cacheDir, RpcServerNative.INSTANCE, START_TIMEOUT_MILLIS);
    }

    /** The lifecycle of {@link #startLocal}/{@link #startOnNetwork}, on an explicit backend and timeout. */
    static RpcServer start(
            RpcEndpoint endpoint, int threads, @Nullable Path cacheDir, Backend backend, long startTimeoutMillis) {
        if (threads < 1) {
            throw new IllegalArgumentException("threads must be at least 1, was " + threads);
        }
        if (!ACTIVE.compareAndSet(false, true)) {
            throw new IllegalStateException("cannot start an RPC server on " + endpoint
                    + ": an RpcServer is already running in this process; close it first");
        }
        boolean started = false;
        try {
            String cache = cacheDir == null ? null : createDirectories(cacheDir).toString();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread thread = new Thread(
                    () -> {
                        try {
                            backend.serve(endpoint.getHost(), endpoint.getPort(), threads, cache);
                        } catch (Throwable t) {
                            failure.set(t);
                        }
                    },
                    "jllama-rpc-server-" + endpoint.getPort());
            thread.setDaemon(true);
            thread.start();
            if (!awaitListening(thread, backend, startTimeoutMillis)) {
                joinQuietly(thread, STOP_TIMEOUT_MILLIS);
                Throwable cause = failure.get();
                throw new LlamaException("RPC server could not listen on " + endpoint
                        + (cause == null ? " (is the port already in use?)" : ": " + cause.getMessage()));
            }
            RpcServer server = new RpcServer(endpoint, thread, backend);
            started = true;
            return server;
        } finally {
            if (!started) {
                ACTIVE.set(false);
            }
        }
    }

    /**
     * A short description for logs.
     *
     * @return the endpoint and whether the server still runs
     */
    @Override
    public String toString() {
        return "RpcServer[" + endpoint + (isRunning() ? ", running]" : ", stopped]");
    }

    /**
     * The address clients use for this server.
     *
     * @return {@code host:port}
     */
    public RpcEndpoint getEndpoint() {
        return endpoint;
    }

    /**
     * Whether the server still accepts clients.
     *
     * @return {@code false} once closed
     */
    public boolean isRunning() {
        return !closed.get() && thread.isAlive();
    }

    /**
     * Blocks until the server is closed, e.g. by a shutdown hook.
     *
     * @throws InterruptedException when the waiting thread is interrupted
     */
    public void awaitTermination() throws InterruptedException {
        thread.join();
    }

    /**
     * Stops the server: the client being served is disconnected, the port is released. Idempotent.
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        try {
            backend.stop();
            joinQuietly(thread, STOP_TIMEOUT_MILLIS);
        } finally {
            ACTIVE.set(false);
        }
    }

    /**
     * The devices a server started now would offer: every accelerator, else the CPU.
     *
     * @return device names as llama.cpp reports them, e.g. {@code [CUDA0]} or {@code [CPU]}
     */
    public static List<String> servedDevices() {
        return servedDevices(RpcServerNative.INSTANCE);
    }

    static List<String> servedDevices(Backend backend) {
        String[] devices = backend.devices();
        return devices == null ? Collections.<String>emptyList() : Collections.unmodifiableList(Arrays.asList(devices));
    }

    /**
     * Command-line entry: {@code java -cp <jar> net.ladenthin.llama.RpcServer [--host 127.0.0.1]
     * [--port 50052] [--threads N] [--cache DIR]}. Runs until the process is stopped.
     *
     * @param args the command line
     * @throws InterruptedException when interrupted while serving
     */
    public static void main(String[] args) throws InterruptedException {
        Options options = Options.parse(args);
        if (options.help) {
            System.out.println(Options.USAGE);
            return;
        }
        RpcServer server = LOOPBACK.equals(options.host)
                ? startLocal(options.port, options.threads, options.cacheDir)
                : startOnNetwork(options.host, options.port, options.threads, options.cacheDir);
        Runtime.getRuntime().addShutdownHook(new Thread(server::close, "jllama-rpc-server-shutdown"));
        System.out.println("RpcServer listening on " + server.getEndpoint() + ", serving " + servedDevices());
        server.awaitTermination();
    }

    /** The parsed command line of {@link #main}. */
    static final class Options {

        static final String USAGE = "usage: java -cp <jar> net.ladenthin.llama.RpcServer"
                + " [--host 127.0.0.1] [--port " + RpcEndpoint.DEFAULT_PORT + "] [--threads N] [--cache DIR]";

        String host = LOOPBACK;
        int port = RpcEndpoint.DEFAULT_PORT;
        int threads = defaultThreads();

        @Nullable
        Path cacheDir;

        boolean help;

        static Options parse(String... args) {
            Options options = new Options();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                switch (arg) {
                    case "-h":
                    case "--help":
                        options.help = true;
                        break;
                    case "-H":
                    case "--host":
                        options.host = value(args, ++i, arg);
                        requireIpv4Literal(options.host);
                        break;
                    case "-p":
                    case "--port":
                        options.port = number(value(args, ++i, arg), arg);
                        break;
                    case "-t":
                    case "--threads":
                        options.threads = number(value(args, ++i, arg), arg);
                        break;
                    case "-c":
                    case "--cache":
                        options.cacheDir = Paths.get(value(args, ++i, arg));
                        break;
                    default:
                        throw new IllegalArgumentException("unknown argument: " + arg + "\n" + USAGE);
                }
            }
            return options;
        }

        /** Upstream's rpc-server default: half the hardware threads, at least one. */
        static int defaultThreads() {
            return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        }

        /**
         * The native server binds with {@code inet_addr}, so only a dotted IPv4 literal works; a host
         * name would fail inside the native layer with no indication why.
         */
        static void requireIpv4Literal(String address) {
            String[] parts = address.split("\\.", -1);
            boolean valid = parts.length == 4;
            for (int i = 0; valid && i < parts.length; i++) {
                String part = parts[i];
                valid = !part.isEmpty()
                        && part.length() <= 3
                        && part.chars().allMatch(c -> c >= '0' && c <= '9')
                        && Integer.parseInt(part) <= 255;
            }
            if (!valid) {
                throw new IllegalArgumentException(
                        "bind address '" + address + "' is not an IPv4 literal such as 127.0.0.1 or 0.0.0.0");
            }
        }

        private static String value(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException(flag + " needs a value");
            }
            return args[index];
        }

        private static int number(String text, String flag) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(flag + " expects a number, got '" + text + "'", e);
            }
        }
    }

    private static boolean awaitListening(Thread thread, Backend backend, long timeoutMillis) {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        while (System.nanoTime() < deadline) {
            if (backend.listening()) {
                return true;
            }
            if (!thread.isAlive()) {
                return false;
            }
            try {
                // returns at once when the server thread ends (bind failed), else polls again
                thread.join(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        // still binding after the timeout, or interrupted: make sure the thread does not outlive us
        backend.stop();
        return false;
    }

    private static void joinQuietly(Thread thread, long millis) {
        try {
            thread.join(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static Path createDirectories(Path dir) {
        try {
            return Files.createDirectories(dir);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot create the RPC tensor cache directory " + dir, e);
        }
    }
}
