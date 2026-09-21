// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The real thing: Atmosphere's built-in runtime driving a llama.cpp model through java-llama.cpp's
 * {@link OpenAiCompatServer} — plain chat, streaming, a tool call with its result fed back, and a
 * multi-round read/write/read loop over a temp file.
 *
 * <p>Model: the Qwen2.5-1.5B-Instruct tool model the core's {@code OpenAiServerToolCallingIntegrationTest}
 * uses (llama.cpp's own tool-call test matrix), resolved from {@code -Dnet.ladenthin.llama.tool.model}
 * (module-relative, then reactor-root). Self-skips when the GGUF is absent so a model-free checkout
 * stays green; CI runs it in a validation-only job (a small model's exact wording is not a release
 * gate). GPU layers come from {@code -Dnet.ladenthin.llama.test.ngl} (default 0 = CPU, device
 * {@code none}).
 *
 * <p>Every assertion here is about the <em>loop</em> — did a tool run, did the result reach the model,
 * did the model answer afterwards, did the file change — not about exact prose, which a 1.5B model does
 * not produce deterministically. The deterministic wire shape is pinned model-free in
 * {@link AtmosphereWireContractTest}.
 */
class AtmosphereToolLoopIntegrationTest {

    private static final String PROP_TOOL_MODEL = "net.ladenthin.llama.tool.model";
    private static final String DEFAULT_TOOL_MODEL = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
    private static final String PROP_NGL = "net.ladenthin.llama.test.ngl";
    private static final String API_KEY = "sk-local";
    private static final String MODEL_ID = "qwen25-tools";
    private static final Duration TURN_TIMEOUT = Duration.ofMinutes(10);
    private static final String SYSTEM_PROMPT =
            "You are a precise assistant. When a tool can answer the request, call it. Keep answers short.";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static String baseUrl;

    @TempDir
    Path workspace;

    @BeforeAll
    static void startServer() throws Exception {
        Path modelPath = TestModelPaths.resolve(System.getProperty(PROP_TOOL_MODEL, DEFAULT_TOOL_MODEL));
        Assumptions.assumeTrue(
                modelPath != null && Files.exists(modelPath),
                "Tool-calling model (Qwen2.5-1.5B) not found, skipping Atmosphere tool-loop test: " + modelPath);
        int gpuLayers = Integer.getInteger(PROP_NGL, 0);
        ModelParameters parameters = new ModelParameters()
                .setModel(modelPath.toString())
                .setCtxSize(8192)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .setParallel(1)
                .enableJinja();
        if (gpuLayers == 0) {
            parameters.setDevices("none");
        }
        model = new LlamaModel(parameters);
        server = new OpenAiCompatServer(
                        model,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .apiKey(API_KEY)
                                .modelId(MODEL_ID)
                                .build())
                .start();
        baseUrl = "http://127.0.0.1:" + server.getPort() + "/v1";
    }

    @AfterAll
    static void stopServer() {
        if (server != null) {
            server.close();
        }
        if (model != null) {
            model.close();
        }
    }

    private AgentRunner runner(List<ToolDefinition> tools) {
        return new AgentRunner(baseUrl, API_KEY, MODEL_ID, tools, SYSTEM_PROMPT, 0.0, 256, 8);
    }

    private ConsoleSession session() {
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        return new ConsoleSession(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8), fs);
    }

    @Test
    void plainChatStreamsAnAnswer() throws Exception {
        ConsoleSession session = session();

        runner(List.of()).run("Reply with exactly this word and nothing else: ATMOSPHERE_OK", List.of(), session);

        assertThat(session.await(TURN_TIMEOUT), is(true));
        assertThat(session.failure(), is(nullValue()));
        assertThat(
                "streamed text: " + session.text(), session.text().toUpperCase().contains("ATMOSPHERE_OK"), is(true));
        assertThat(
                "the answer must arrive as several SSE chunks, not one blob",
                session.chunks().size(),
                greaterThanOrEqualTo(2));
    }

    @Test
    void toolCallResultIsFedBackAndAnswered() throws Exception {
        AtomicInteger invocations = new AtomicInteger();
        ToolDefinition tool = ToolDefinition.builder(
                        "get_current_test_value", "Returns the current test value. Call it to learn the value.")
                .executor(args -> {
                    invocations.incrementAndGet();
                    return "ATMOSPHERE_TOOL_OK";
                })
                .build();
        ConsoleSession session = session();

        runner(List.of(tool))
                .run(
                        "Call the tool get_current_test_value and then tell me the value it returned.",
                        List.of(),
                        session);

        assertThat(session.await(TURN_TIMEOUT), is(true));
        assertThat(session.failure(), is(nullValue()));
        assertThat("the model must call the tool", invocations.get(), greaterThanOrEqualTo(1));
        assertThat(
                "final answer after the tool round: " + session.text(),
                session.text().trim().isEmpty(),
                is(false));
        assertThat(session.text(), containsString("ATMOSPHERE_TOOL_OK"));
    }

    @Test
    void multiRoundReadWriteReadLoopChangesTheFile() throws Exception {
        Path file = workspace.resolve("test.txt");
        Files.writeString(file, "VALUE=1\n");
        List<String> trace = new CopyOnWriteArrayList<>();
        ToolDefinition read = ToolDefinition.builder("read_test_file", "Read the content of test.txt")
                .executor(args -> {
                    trace.add("read");
                    return Files.readString(file);
                })
                .build();
        ToolDefinition write = ToolDefinition.builder(
                        "write_test_file", "Replace the whole content of test.txt with the given content")
                .parameter("content", "The new full content of the file", "string", true)
                .executor(args -> {
                    trace.add("write");
                    Files.writeString(file, String.valueOf(args.get("content")));
                    return "written";
                })
                .build();
        ConsoleSession session = session();

        runner(List.of(read, write))
                .run(
                        "Use the tools: first read test.txt, then change the line VALUE=1 to VALUE=2 by writing the"
                                + " file, then read the file again and tell me the new value.",
                        List.of(),
                        session);

        assertThat(session.await(TURN_TIMEOUT), is(true));
        assertThat(session.failure(), is(nullValue()));
        assertThat("tool trace: " + trace, trace.contains("write"), is(true));
        assertThat("at least two tool rounds: " + trace, trace.size(), greaterThanOrEqualTo(2));
        assertThat(Files.readString(file), containsString("VALUE=2"));
        assertThat("final answer: " + session.text(), session.text().trim().isEmpty(), is(false));
    }
}
