// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
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
import java.util.List;
import java.util.Map;
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
        List<Line> lines = Collections.synchronizedList(new ArrayList<>());
        LlamaModel.setLogger(format, (level, text) -> lines.add(new Line(level, text)));
        Path file = notAGguf();
        assertThrows(
                LlamaException.class,
                () -> new LlamaModel(
                                new ModelParameters().setModel(file.toString()).setDevices("none"))
                        .close());
        // Removing the logger flushes every queued message to the previous callback before returning.
        LlamaModel.setLogger(LogFormat.TEXT, null);
        return lines;
    }

    @Test
    void loggerSetBeforeTheLoadReceivesTheLoadsOwnLines() throws IOException {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping logger guard");

        List<Line> lines = linesOfAFailedLoad(LogFormat.TEXT);

        assertThat("a failed load must log something: " + lines, lines, not(empty()));
        assertThat(
                "the server's own INFO line ('srv … loading model') must reach a logger set before the load: " + lines,
                lines.stream().anyMatch(l -> l.text.contains("loading model")),
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
}
