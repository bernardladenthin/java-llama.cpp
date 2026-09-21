// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import net.ladenthin.llama.args.LogFormat;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.OSInfo;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LogLevel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Model-free guard for {@link LlamaModel#setLogger}: a logger set <em>before</em> a load keeps
 * receiving lines through the load. Every load runs llama.cpp's {@code common_init()}, which
 * re-points {@code llama_log_set} at its own default callback, and until patches/0014 that silently
 * dropped a previously set Java logger. The logger is now a sink on {@code common_log}, behind that
 * default callback, so it sees both the server's own {@code srv …} lines and the llama/ggml lines.
 * The load here is made to fail on purpose (a file that is not a GGUF), which needs no model, no GPU
 * and no network and still produces an INFO line from the server and an ERROR line from llama.
 * Skips cleanly when {@code libjllama} is not on the classpath (pure-Java checkout).
 */
@ClaudeGenerated(
        purpose = "Model-free guard that LlamaModel.setLogger survives a model load (common_init re-points "
                + "llama_log_set) and receives the server's own srv/slot lines, in both TEXT and JSON mode, "
                + "without a GGUF: the load fails on a non-GGUF file and its log lines are asserted.")
class LlamaLoggerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private static boolean nativeLibraryOnClasspath() {
        String resource = "/net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/"
                + System.mapLibraryName("jllama");
        return LlamaLoggerTest.class.getResource(resource) != null;
    }

    private static final class Line {
        private final LogLevel level;
        private final String text;

        private Line(LogLevel level, String text) {
            this.level = level;
            this.text = text;
        }

        @Override
        public String toString() {
            return level + ": " + text.trim();
        }
    }

    @AfterEach
    void restoreConsoleLogging() {
        LlamaModel.setLogger(LogFormat.TEXT, null);
    }

    private Path notAGguf() throws IOException {
        Path file = tempDir.resolve("not-a-model.gguf");
        Files.write(file, "this is not a GGUF file".getBytes(StandardCharsets.UTF_8));
        return file;
    }

    /** Logs through a failing load and drains the queue; the drain is what makes the assertions safe. */
    private List<Line> linesOfAFailedLoad(LogFormat format) throws IOException {
        return linesOfAFailedLoad(format, new ModelParameters());
    }

    private List<Line> linesOfAFailedLoad(LogFormat format, ModelParameters parameters) throws IOException {
        List<Line> lines = Collections.synchronizedList(new ArrayList<>());
        LlamaModel.setLogger(format, (level, text) -> lines.add(new Line(level, text)));
        failingLoad(parameters);
        // Removing the logger flushes every queued message to the previous callback before returning.
        LlamaModel.setLogger(LogFormat.TEXT, null);
        return lines;
    }

    private void failingLoad(ModelParameters parameters) throws IOException {
        Path file = notAGguf();
        assertThrows(
                LlamaException.class,
                () -> new LlamaModel(parameters.setModel(file.toString()).setDevices("none")).close());
    }

    /** The server's own {@code srv … loading model '…'} INFO line — not llama's {@code error loading model}. */
    private static boolean sawLoadingModel(List<Line> lines) {
        return lines.stream().anyMatch(l -> l.text.startsWith("srv ") && l.text.contains("loading model '"));
    }

    @Test
    void loggerSetBeforeTheLoadReceivesTheLoadsOwnLines() throws IOException {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");

        List<Line> lines = linesOfAFailedLoad(LogFormat.TEXT);

        assertThat("a failed load must log something: " + lines, lines, not(empty()));
        assertThat(
                "the server's own INFO line ('srv … loading model') must reach a logger set before the load: " + lines,
                sawLoadingModel(lines),
                is(true));
        assertThat(
                "llama's own error line must reach the logger too: " + lines,
                lines.stream().map(l -> l.level).collect(Collectors.toList()),
                hasItem(LogLevel.ERROR));
        assertThat(
                "text mode hands over the bare message, no prefix/timestamp: " + lines,
                lines.stream().noneMatch(l -> l.text.matches("^\\d+\\.\\d+\\.\\d+\\.\\d+ [IWED] .*")),
                is(true));
    }

    @Test
    void jsonModeWrapsEveryLineIntoOneObject() throws IOException {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");

        List<Line> lines = linesOfAFailedLoad(LogFormat.JSON);

        assertThat("a failed load must log something: " + lines, lines, not(empty()));
        for (Line line : lines) {
            JsonNode node = MAPPER.readTree(line.text);
            assertThat("every line is one JSON object: " + line.text, node.isObject(), is(true));
            assertThat(toMap(node), hasKey("level"));
            assertThat(toMap(node), hasKey("message"));
            assertThat(toMap(node), hasKey("timestamp"));
        }
    }

    @Test
    void anEmptyCallbackDiscardsWithoutFailingTheLoad() throws IOException {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");
        LlamaModel.setLogger(LogFormat.TEXT, (level, text) -> {});
        Path file = notAGguf();

        // The load still fails for its own reason; the muted logger must not change that or hang the drain.
        assertThrows(
                LlamaException.class,
                () -> new LlamaModel(
                                new ModelParameters().setModel(file.toString()).setDevices("none"))
                        .close());
        LlamaModel.setLogger(LogFormat.TEXT, null);
    }

    private static Map<String, JsonNode> toMap(JsonNode node) {
        Map<String, JsonNode> map = new HashMap<>();
        node.fields().forEachRemaining(e -> map.put(e.getKey(), e.getValue()));
        return map;
    }

    /**
     * Concurrent {@code setLogger} calls must be serialized natively. The sink swap pauses and
     * resumes llama.cpp's log worker, and two unserialized swaps race on that {@code std::thread}:
     * one caller joins it while the other assigns a fresh thread over the still-joinable object,
     * which is {@code std::terminate} — the whole JVM dies, not a test. Before the fix this hammered
     * the race hard enough to reproduce it.
     */
    @Test
    void concurrentSetLoggerCallsDoNotRaceOnTheLogWorker() throws Exception {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");
        final int threads = 4;
        final int rounds = 200;
        java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            java.util.List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(pool.submit(() -> {
                    for (int i = 0; i < rounds; i++) {
                        LlamaModel.setLogger(LogFormat.TEXT, (level, text) -> {});
                        LlamaModel.setLogger(LogFormat.JSON, null);
                    }
                }));
            }
            for (java.util.concurrent.Future<?> f : futures) {
                f.get(2, java.util.concurrent.TimeUnit.MINUTES);
            }
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * The verbosity threshold is process-wide and applies before the sink: {@code -lv 1} hides the
     * server's INFO line but keeps llama's ERROR line. It is set by <em>every</em> load, not only by
     * one that passes {@code -lv}: {@code common_params_parse} ends with
     * {@code common_log_set_verbosity_thold(params.verbosity)}, whose default is 3, so a load without
     * the flag resets the threshold to llama.cpp's default. The last model loaded wins, whichever
     * way it was loaded. (Written down because the opposite was assumed once, in a review.)
     */
    @Test
    void verbosityThresholdIsProcessWideAndEveryLoadSetsIt() throws IOException {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");
        try {
            List<Line> errorsOnly = linesOfAFailedLoad(LogFormat.TEXT, new ModelParameters().setLogVerbosity(1));
            assertThat("-lv 1 must drop the server's INFO line: " + errorsOnly, sawLoadingModel(errorsOnly), is(false));
            assertThat(
                    "-lv 1 must keep llama's ERROR line: " + errorsOnly,
                    errorsOnly.stream().map(l -> l.level).collect(Collectors.toList()),
                    hasItem(LogLevel.ERROR));

            List<Line> reset = linesOfAFailedLoad(LogFormat.TEXT, new ModelParameters());
            assertThat(
                    "a load without -lv resets the threshold to llama.cpp's default (3), so the INFO line is back: "
                            + reset,
                    sawLoadingModel(reset),
                    is(true));
        } finally {
            // Belt and braces for the other tests: leave the process at llama.cpp's default.
            linesOfAFailedLoad(LogFormat.TEXT, new ModelParameters().setLogVerbosity(3));
        }
    }

    /**
     * Messages are delivered on llama.cpp's log worker thread, never on the thread that logged
     * or the one that installed the logger; and removing the logger returns only after every queued
     * message has been delivered. Both are the facts behind the two deadlock rules in the Javadoc
     * (no {@code setLogger} from a callback; no lock held that the previous callback needs). The
     * distinct-thread count is recorded, not pinned: today every line attaches the worker afresh
     * (one {@code java.lang.Thread} per line), a {@code thread_local} guard would make it one.
     */
    @Test
    void deliveryIsAsynchronousOnTheLogWorkerAndRemovingTheLoggerDrains() throws Exception {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");
        Thread caller = Thread.currentThread();
        Set<Thread> deliveringThreads = Collections.synchronizedSet(new HashSet<>());
        AtomicInteger delivered = new AtomicInteger();
        LlamaModel.setLogger(LogFormat.TEXT, (level, text) -> {
            deliveringThreads.add(Thread.currentThread());
            delivered.incrementAndGet();
        });

        failingLoad(new ModelParameters());
        LlamaModel.setLogger(LogFormat.TEXT, null);
        int atReturn = delivered.get();
        Thread.sleep(200);

        assertThat("a failed load logs at least one line", atReturn, greaterThan(0));
        assertThat("nothing may arrive after setLogger(format, null) returned", delivered.get(), is(atReturn));
        assertThat("delivery never runs on the caller's thread", deliveringThreads.contains(caller), is(false));
        assertThat(
                "every delivering thread is a native-attached one, not a Java-created one",
                deliveringThreads.stream().allMatch(t -> t.getName().startsWith("Thread-")),
                is(true));
        System.out.println("[LlamaLoggerTest] " + atReturn + " lines delivered on " + deliveringThreads.size()
                + " distinct Thread object(s)");
    }
}
