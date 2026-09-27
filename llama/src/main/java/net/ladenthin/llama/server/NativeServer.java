// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.ToString;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.loader.LlamaLoader;
import org.jspecify.annotations.Nullable;

/**
 * Runs the <em>full</em> upstream llama.cpp HTTP server — including its embedded
 * <strong>WebUI</strong> — inside {@code libjllama}, driven over JNI, with no separate
 * {@code llama-server} executable. It is the second of two server modes, the native counterpart to
 * the Java-transport {@link OpenAiCompatServer}.
 *
 * <p>The constructor takes the raw llama-server command-line arguments and forwards them verbatim
 * to the native entry point ({@code llama_server}), so <em>every</em> llama-server flag is supported
 * ({@code -m}, {@code -c}, {@code -b}, {@code -ub}, {@code -ngl}, {@code -t}, {@code -tb},
 * {@code -ctk}, {@code -ctv}, {@code --jinja}, {@code --chat-template-kwargs}, {@code --host},
 * {@code --port}, {@code --ui}/{@code --no-ui}, …). Unlike {@link OpenAiCompatServer}, no per-flag
 * Java mapping is involved.</p>
 *
 * <p><strong>Two lifecycles.</strong> The classic constructor ({@link #NativeServer(String...)})
 * loads its <em>own</em> model from the forwarded arguments — exactly like running
 * {@code llama-server.exe} — and is unrelated to any {@link LlamaModel} you may also have open;
 * while it runs it owns the process-wide llama backend and routes llama.cpp logging to
 * stderr/file (llama-server's own logging), not the JNI log callback. The <em>attach</em>
 * constructor ({@link #NativeServer(LlamaModel, String...)}) instead serves an
 * <strong>already-loaded</strong> {@code LlamaModel} — no second copy of the weights, no second
 * model load: the model's own worker thread keeps driving inference and the HTTP routes post
 * tasks to its queue. In attach mode the arguments carry only the HTTP-side flags
 * ({@code --host}, {@code --port}, {@code --api-key}, …; no {@code -m}), and the caller keeps
 * full ownership of the model: <strong>do not {@code close()} the model while the server is
 * attached</strong> — stop the server first (server before model, like unwinding
 * try-with-resources).</p>
 *
 * <p><strong>Router mode.</strong> Starting without any model argument puts the upstream server
 * in router mode ({@code --models-dir}, {@code GET/POST /models}, per-request model selection).
 * The router serves each model from a <em>worker subprocess</em> that upstream spawns by
 * re-executing its own binary — which, inside a JVM, is {@code java}, not a llama-server. Set
 * {@link #setWorkerCommand(String...)} before starting a router so workers relaunch through this
 * library instead, e.g. {@code setWorkerCommand(javaBin, "-cp", classpath,
 * "net.ladenthin.llama.server.NativeServer")} — each worker is then a fresh JVM running the
 * classic single-model {@code NativeServer}.</p>
 *
 * <p><strong>Single instance per process.</strong> The upstream server keeps its shutdown state in
 * file-scope globals, so only one {@code NativeServer} may run at a time; {@link #start()} throws if
 * another instance is already running.</p>
 *
 * <p>Typical use:</p>
 * <pre>{@code
 * try (NativeServer server = new NativeServer(
 *         "-m", "models/model.gguf", "--host", "127.0.0.1", "--port", "8080", "-c", "65536").start()) {
 *     // Server (and WebUI at http://127.0.0.1:8080/) runs on a native worker thread.
 *     // Readiness: poll GET /health until it returns {"status":"ok"}.
 *     Thread.currentThread().join();
 * }
 * }</pre>
 *
 * <p><strong>Platform note.</strong> The native methods are compiled into {@code libjllama} on all
 * platforms except Android (the upstream server pulls in {@code posix_spawn_*}, unavailable there);
 * on Android use {@link OpenAiCompatServer}. No SSL: the embedded server is plain HTTP — bind
 * localhost or front it with a TLS proxy.</p>
 */
@ToString
public final class NativeServer implements AutoCloseable {

    /** Guards the process-wide single-instance invariant (upstream uses file-scope globals). */
    private static final AtomicBoolean RUNNING = new AtomicBoolean(false);

    /** Default bind host reported by {@link #getHost()} when {@code --host} is not passed. */
    private static final String DEFAULT_HOST = "127.0.0.1";

    /** Default port reported by {@link #getPort()} when no port flag is passed. */
    private static final int DEFAULT_PORT = 8080;

    /** The llama-server argument vector, forwarded verbatim to the native entry point. */
    private final String[] args;

    /**
     * The already-loaded model this server attaches to, or {@code null} for the classic
     * standalone lifecycle (the server loads its own model from {@link #args}). Excluded from
     * {@code toString} — rendering it would dump the model's own identity block on every log line.
     */
    @ToString.Exclude
    private final @Nullable LlamaModel attachedModel;

    /** Native handle (pointer) while running, or {@code 0} when not started / stopped. */
    private volatile long handle;

    /**
     * Creates a native-server bridge for the given llama-server arguments.
     *
     * <p>Construction performs no native work and binds no socket; it only captures the arguments.
     * Call {@link #start()} to launch the server.</p>
     *
     * @param args the llama-server command-line arguments (e.g. {@code "-m", "model.gguf",
     *             "--port", "8080"}); must not be {@code null} and must not contain {@code null}
     *             elements
     */
    public NativeServer(String... args) {
        Objects.requireNonNull(args, "args");
        for (final String arg : args) {
            Objects.requireNonNull(arg, "args element");
        }
        this.args = args.clone();
        this.attachedModel = null;
    }

    /**
     * Creates a native-server bridge that <em>attaches</em> the full upstream HTTP frontend —
     * route table, WebUI, resumable streaming — to an already-loaded {@link LlamaModel}, instead
     * of loading a second copy of the weights.
     *
     * <p>The arguments carry only the HTTP-side llama-server flags ({@code --host},
     * {@code --port}, {@code --api-key}, {@code --slots}, …); no model argument is needed or
     * used. Because the model is already loaded, the server reports ready on {@code GET /health}
     * as soon as the socket is up.</p>
     *
     * <p><strong>Lifecycle contract:</strong> the caller keeps full ownership of {@code model}.
     * The model must stay open for as long as the server runs — close the server first, then the
     * model. The attached server holds a native usage reference on the model, so a
     * {@link LlamaModel#close()} issued while the server is still running does not free native
     * state out from under the HTTP routes — instead it <em>blocks indefinitely</em> until the
     * server is stopped. Always {@link #close()} this server before closing the model.</p>
     *
     * @param model the loaded model whose native server context this server should serve; must
     *              not be {@code null} and must not be closed while the server runs
     * @param args  the HTTP-side llama-server arguments (e.g. {@code "--host", "127.0.0.1",
     *              "--port", "8080"}); must not be {@code null} or contain {@code null} elements
     */
    public NativeServer(LlamaModel model, String... args) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(args, "args");
        for (final String arg : args) {
            Objects.requireNonNull(arg, "args element");
        }
        this.args = args.clone();
        this.attachedModel = model;
    }

    /**
     * Starts the native HTTP server (and its embedded WebUI) on a background thread and returns
     * immediately. The server binds and begins serving {@code GET /health} before the model finishes
     * loading; poll {@code /health} for readiness.
     *
     * @return this server instance (for fluent / try-with-resources use)
     * @throws IllegalStateException if this instance was already started, or another
     *                               {@code NativeServer} is already running in this process
     */
    public NativeServer start() {
        if (handle != 0) {
            throw new IllegalStateException("NativeServer already started");
        }
        if (!RUNNING.compareAndSet(false, true)) {
            throw new IllegalStateException(
                    "another NativeServer is already running in this process (only one is supported)");
        }
        try {
            // Load libjllama lazily here (not in a static initializer) so construction, argument
            // parsing and close() stay usable — and unit-testable — without the native library.
            LlamaLoader.initialize();
            handle = attachedModel != null ? startAttachedNativeServer(attachedModel, args) : startNativeServer(args);
        } catch (final RuntimeException | Error e) {
            RUNNING.set(false);
            throw e;
        }
        return this;
    }

    /**
     * Reports whether the native server worker is currently running.
     *
     * <p>Note: this becomes {@code true} as soon as the worker thread starts, which is before the
     * socket is necessarily accepting connections — use {@code GET /health} to detect readiness.</p>
     *
     * @return {@code true} if the server has been started and its worker has not yet exited
     */
    public boolean isRunning() {
        final long h = handle;
        return h != 0 && isRunningNative(h);
    }

    /**
     * Returns the bind host parsed from the arguments ({@code --host}), or {@code 127.0.0.1} when
     * absent. Best-effort convenience for logging; the authoritative value is what the native server
     * parsed. When {@code --host} lists several addresses, this is the first of them — see
     * {@link #getHosts()}.
     *
     * @return the configured bind host
     */
    public String getHost() {
        return getHosts().get(0);
    }

    /**
     * Returns every bind address parsed from the arguments ({@code --host}), or a single
     * {@code 127.0.0.1} when absent. Since llama.cpp b11104 {@code --host} takes a comma-separated
     * list (IP addresses and/or UNIX socket paths ending in {@code .sock}) and the server listens on
     * all of them; blanks around the commas are ignored, as upstream does. Best-effort convenience for
     * logging, like {@link #getHost()}.
     *
     * @return the configured bind addresses, never empty
     */
    public List<String> getHosts() {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--host".equals(args[i])) {
                final List<String> hosts = new ArrayList<>();
                for (final String host : args[i + 1].split(",", -1)) {
                    final String trimmed = host.trim();
                    if (!trimmed.isEmpty()) {
                        hosts.add(trimmed);
                    }
                }
                if (!hosts.isEmpty()) {
                    return Collections.unmodifiableList(hosts);
                }
            }
        }
        return Collections.singletonList(DEFAULT_HOST);
    }

    /**
     * Returns the port parsed from the arguments ({@code --port} / {@code -p}), or {@code 8080} when
     * absent or unparseable. Best-effort convenience for logging.
     *
     * @return the configured port
     */
    public int getPort() {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--port".equals(args[i]) || "-p".equals(args[i])) {
                try {
                    return Integer.parseInt(args[i + 1].trim());
                } catch (final NumberFormatException e) {
                    return DEFAULT_PORT;
                }
            }
        }
        return DEFAULT_PORT;
    }

    /**
     * Stops the native server if it is running and releases the native handle. Blocks until the
     * server has fully shut down. Safe to call more than once and from try-with-resources even if
     * {@link #start()} was never called (no-op then).
     */
    @Override
    public void close() {
        final long h = handle;
        if (h == 0) {
            return;
        }
        handle = 0;
        try {
            stopNativeServer(h);
        } finally {
            RUNNING.set(false);
        }
    }

    /**
     * Fat-jar entry point (the assembly JAR's {@code Main-Class}): starts the full native llama.cpp
     * server — WebUI included — forwarding every argument to it verbatim, and blocks until the
     * server exits or the JVM is asked to shut down (Ctrl-C / SIGTERM), stopping the server cleanly
     * on the way out.
     *
     * <p>This is the default runnable server. The Java-transport {@link OpenAiCompatServer} remains
     * available via its own {@code main} — run it explicitly with
     * {@code java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …}.</p>
     *
     * @param args the llama-server command-line arguments, forwarded verbatim (e.g. {@code -m
     *             model.gguf --host 127.0.0.1 --port 8080}); pass {@code --help} for the full
     *             llama-server option list
     * @throws InterruptedException if interrupted while waiting for the server to exit
     */
    public static void main(String[] args) throws InterruptedException {
        // Own the server in a try/finally so close() is guaranteed on normal or exceptional exit of
        // the block (satisfies S2095 via the "close in a finally clause" option — try-with-resources
        // is not used because the shutdown hook must also call close() explicitly, which javac flags
        // under -Werror as an "explicit call to close() on an auto-closeable resource"). close() is
        // idempotent (guards on a zero handle), so the finally and the hook both firing is safe.
        final NativeServer server = new NativeServer(args);
        try {
            // Signalled by the shutdown hook so the main thread wakes immediately on Ctrl-C / SIGTERM
            // rather than waiting out a poll tick — and so the wait uses a bounded latch await instead
            // of Thread.sleep (banned by LlamaArchitectureTest.noThreadSleep).
            final CountDownLatch stopSignal = new CountDownLatch(1);
            // Graceful Ctrl-C / SIGTERM: the embedded server installs no signal handlers of its own
            // (see patches/0006), so the JVM-level shutdown hook is what stops it before exit.
            Runtime.getRuntime()
                    .addShutdownHook(new Thread(
                            () -> {
                                server.close();
                                stopSignal.countDown();
                            },
                            "jllama-native-server-shutdown"));
            server.start();
            // Keep the JVM alive until the native worker exits — on its own (e.g. a fatal startup/model
            // error that llama_server has already logged) or because the shutdown hook stopped it. The
            // bounded await returns early when the hook fires; on timeout we re-check isRunning() to
            // catch a self-terminated worker.
            while (server.isRunning() && !stopSignal.await(200L, TimeUnit.MILLISECONDS)) {
                // wait for the native worker to exit or the shutdown hook to fire
            }
        } finally {
            server.close();
        }
    }

    /**
     * Sets (or clears) the router-mode <em>worker command</em> for this process — the command
     * line prefix used to spawn each model-worker subprocess when this server runs in router
     * mode. By default the upstream router re-executes its own binary, which inside a JVM is
     * {@code java} itself and cannot serve a model; point it at this library's bootstrap instead:
     *
     * <pre>{@code
     * String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
     * NativeServer.setWorkerCommand(javaBin, "-cp", System.getProperty("java.class.path"),
     *         "net.ladenthin.llama.server.NativeServer");
     * }</pre>
     *
     * <p>The tokens are stored in the process environment ({@code LLAMA_SERVER_WORKER_CMD},
     * whitespace-joined) and read by the native router when it spawns a worker; the router's
     * computed worker arguments ({@code --host}, {@code --port}, alias, model flags) are appended
     * after them. Because the variable is whitespace-split natively, <strong>no token may contain
     * whitespace</strong> (e.g. a classpath with spaces is unsupported).</p>
     *
     * <p>Calling with no tokens clears the override (workers re-exec the process binary again,
     * the upstream default).</p>
     *
     * @param command the worker command tokens, e.g. {@code "java", "-cp", "app.jar",
     *                "net.ladenthin.llama.server.NativeServer"}; empty clears the override
     * @throws IllegalArgumentException if a token is null, empty, or contains whitespace
     */
    public static void setWorkerCommand(String... command) {
        Objects.requireNonNull(command, "command");
        StringBuilder joined = new StringBuilder();
        for (final String token : command) {
            if (token == null || token.isEmpty() || token.matches(".*\\s.*")) {
                throw new IllegalArgumentException(
                        "worker command tokens must be non-empty and must not contain whitespace, got: " + token);
            }
            if (joined.length() > 0) {
                joined.append(' ');
            }
            joined.append(token);
        }
        LlamaLoader.initialize();
        setWorkerCommandNative(joined.length() == 0 ? null : joined.toString());
    }

    /**
     * Starts the native server on a worker thread and returns an opaque handle. The argv is
     * forwarded verbatim (with a synthetic {@code argv[0]}).
     */
    private static native long startNativeServer(String[] args);

    /**
     * Starts the attach-mode server (HTTP frontend over the given model's native server context)
     * on a worker thread and returns an opaque handle compatible with
     * {@link #stopNativeServer(long)} / {@link #isRunningNative(long)}.
     */
    private static native long startAttachedNativeServer(LlamaModel model, String[] args);

    /** Sets/clears the {@code LLAMA_SERVER_WORKER_CMD} process environment variable. */
    private static native void setWorkerCommandNative(@Nullable String command);

    /** Signals shutdown, joins the worker thread, and frees the handle. */
    private static native void stopNativeServer(long handle);

    /** Whether the worker thread for the given handle is still running. */
    private static native boolean isRunningNative(long handle);
}
