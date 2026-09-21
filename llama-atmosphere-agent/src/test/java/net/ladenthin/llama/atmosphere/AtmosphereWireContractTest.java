// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The wire contract between Atmosphere's built-in OpenAI-compatible runtime and java-llama.cpp's
 * {@link OpenAiCompatServer}, proven over a real loopback socket with no model: the server's routing,
 * bearer authentication, {@code /v1/models} and Server-Sent-Events framing are the real thing, only the
 * inference engine is a {@link ScriptedBackend} replaying llama.cpp-shaped chunks.
 *
 * <p>What this pins, on every PR and in seconds: Atmosphere accumulates streamed {@code tool_calls}
 * deltas by {@code index}, executes the Java tool, re-submits the conversation with the assistant's
 * {@code tool_calls} message (no {@code content} key — accepted by llama.cpp's
 * {@code common_chat_msgs_parse_oaicompat}, which requires {@code content} <em>or</em>
 * {@code tool_calls}) plus one {@code role:"tool"} message per call carrying {@code tool_call_id},
 * keeps every earlier round in the history, and completes on {@code finish_reason:"stop"}. The
 * model-backed counterpart is {@link AtmosphereToolLoopIntegrationTest}.
 */
class AtmosphereWireContractTest {

    private static final String API_KEY = "secret-key";
    private static final String MODEL_ID = "local-model";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    @TempDir
    Path workspace;

    private static OpenAiServerConfig config(String apiKey) {
        return OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey(apiKey)
                .modelId(MODEL_ID)
                .build();
    }

    private static String baseUrl(OpenAiCompatServer server) {
        return "http://127.0.0.1:" + server.getPort() + "/v1";
    }

    private AgentRunner runner(OpenAiCompatServer server, String apiKey, List<ToolDefinition> tools) {
        return new AgentRunner(baseUrl(server), apiKey, MODEL_ID, tools, "You are a test agent.", 0.0, 64, 10)
                .retryPolicy(RetryPolicy.NONE);
    }

    private ConsoleSession session() {
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        return new ConsoleSession(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8), fs);
    }

    private static ToolDefinition tool(String name, List<String> invocations, String result) {
        return ToolDefinition.builder(name, "Test tool " + name)
                .parameter("value", "An optional value", "string", false)
                .executor(args -> {
                    invocations.add(name + ":" + args.getOrDefault("value", ""));
                    return result;
                })
                .build();
    }

    private static List<String> roles(JsonNode request) {
        List<String> roles = new ArrayList<>();
        for (JsonNode message : request.path("messages")) {
            roles.add(message.path("role").asText());
        }
        return roles;
    }

    @Test
    void oneToolRoundTravelsThroughTheRealServer() throws Exception {
        List<String> invocations = new CopyOnWriteArrayList<>();
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", "get_current_test_value", "{}")
                : ScriptedBackend.textTurn("Result: ", "ATMOSPHERE_TOOL_OK"));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            AgentRunner runner =
                    runner(server, API_KEY, List.of(tool("get_current_test_value", invocations, "ATMOSPHERE_TOOL_OK")));
            ConsoleSession session = session();

            runner.run("Call get_current_test_value and print its result.", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(session.failure(), is(nullValue()));
            assertThat(session.text(), is("Result: ATMOSPHERE_TOOL_OK"));
            assertThat(invocations, contains("get_current_test_value:"));

            List<JsonNode> requests = backend.requests();
            assertThat(requests, hasSize(2));
            JsonNode first = requests.get(0);
            assertThat(first.path("stream").asBoolean(), is(true));
            assertThat(first.path("model").asText(), is(MODEL_ID));
            assertThat(first.path("tools").size(), is(1));
            assertThat(first.path("tools").get(0).path("function").path("name").asText(), is("get_current_test_value"));
            assertThat(roles(first), contains("system", "user"));

            JsonNode second = requests.get(1);
            assertThat(roles(second), contains("system", "user", "assistant", "tool"));
            JsonNode assistant = second.path("messages").get(2);
            // llama.cpp accepts an assistant message that carries tool_calls without a content key.
            assertThat(assistant.has("content"), is(false));
            assertThat(assistant.path("tool_calls").get(0).path("id").asText(), is("call_1"));
            assertThat(
                    assistant
                            .path("tool_calls")
                            .get(0)
                            .path("function")
                            .path("name")
                            .asText(),
                    is("get_current_test_value"));
            assertThat(
                    assistant
                            .path("tool_calls")
                            .get(0)
                            .path("function")
                            .path("arguments")
                            .isTextual(),
                    is(true));
            JsonNode toolMessage = second.path("messages").get(3);
            assertThat(toolMessage.path("tool_call_id").asText(), is("call_1"));
            assertThat(toolMessage.path("content").asText(), is("ATMOSPHERE_TOOL_OK"));
        }
    }

    @Test
    void threeToolRoundsWithAParallelPairKeepTheWholeConversation() throws Exception {
        List<String> invocations = new CopyOnWriteArrayList<>();
        ScriptedBackend backend = new ScriptedBackend((call, request) -> switch (call) {
            case 1 ->
                List.of(
                        ScriptedBackend.roleChunk(),
                        ScriptedBackend.toolCallStart(0, "call_a", "read_test_file"),
                        ScriptedBackend.toolCallStart(1, "call_b", "list_test_files"),
                        ScriptedBackend.toolCallArguments(0, "{\"value\":"),
                        ScriptedBackend.toolCallArguments(1, "{}"),
                        ScriptedBackend.toolCallArguments(0, "\"test.txt\"}"),
                        ScriptedBackend.finish("tool_calls"));
            case 2 -> ScriptedBackend.toolCallTurn("call_c", "write_test_file", "{\"value\":\"VALUE=2\"}");
            case 3 -> ScriptedBackend.toolCallTurn("call_d", "read_test_file", "{\"value\":\"test.txt\"}");
            default -> ScriptedBackend.textTurn("The new value is VALUE=2.");
        });
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            AgentRunner runner = runner(
                    server,
                    API_KEY,
                    List.of(
                            tool("read_test_file", invocations, "VALUE=1"),
                            tool("list_test_files", invocations, "test.txt"),
                            tool("write_test_file", invocations, "ok")));
            ConsoleSession session = session();

            runner.run("Read test.txt, change VALUE=1 to VALUE=2, read it again.", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(session.failure(), is(nullValue()));
            assertThat(session.text(), is("The new value is VALUE=2."));
            // Fragmented arguments were reassembled per index; both parallel calls ran, in index order.
            assertThat(
                    invocations,
                    contains(
                            "read_test_file:test.txt",
                            "list_test_files:",
                            "write_test_file:VALUE=2",
                            "read_test_file:test.txt"));
            assertThat(session.toolCalls(), is(4));

            List<JsonNode> requests = backend.requests();
            assertThat(requests, hasSize(4));
            // Every earlier round is still in the history of the last request.
            assertThat(
                    roles(requests.get(3)),
                    contains("system", "user", "assistant", "tool", "tool", "assistant", "tool", "assistant", "tool"));
            JsonNode lastRequest = requests.get(3);
            assertThat(lastRequest.path("messages").get(2).path("tool_calls").size(), is(2));
            assertThat(lastRequest.path("messages").get(3).path("tool_call_id").asText(), is("call_a"));
            assertThat(lastRequest.path("messages").get(4).path("tool_call_id").asText(), is("call_b"));
            assertThat(lastRequest.path("messages").get(6).path("tool_call_id").asText(), is("call_c"));
            assertThat(lastRequest.path("messages").get(8).path("tool_call_id").asText(), is("call_d"));
        }
    }

    @Test
    void streamedTextArrivesChunkByChunkAndHistoryIsReplayed() throws Exception {
        ScriptedBackend backend =
                new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("ATMOS", "PHERE", "_OK"));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            AgentRunner runner = runner(server, API_KEY, List.of());
            ConsoleSession session = session();
            List<ChatMessage> history =
                    List.of(ChatMessage.user("earlier question"), ChatMessage.assistant("earlier answer"));

            runner.run("Answer exactly with ATMOSPHERE_OK", history, session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(session.text(), is("ATMOSPHERE_OK"));
            assertThat(session.chunks(), contains("ATMOS", "PHERE", "_OK"));
            JsonNode request = backend.requests().get(0);
            assertThat(roles(request), contains("system", "user", "assistant", "user"));
            assertThat(request.path("messages").get(1).path("content").asText(), is("earlier question"));
            assertThat(
                    request.path("messages").get(3).path("content").asText(), is("Answer exactly with ATMOSPHERE_OK"));
            assertThat(request.path("temperature").asDouble(), is(0.0));
            assertThat(request.path("max_tokens").asInt(), is(64));
        }
    }

    @Test
    void modelsAreEnumeratedFromTheServer() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("unused"));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            assertThat(runner(server, API_KEY, List.of()).models(), contains(MODEL_ID));
        }
    }

    @Test
    void wrongApiKeyIsRejectedBeforeReachingTheBackend() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("never"));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            AgentRunner runner = runner(server, "wrong-key", List.of());
            ConsoleSession session = session();

            runner.run("hello", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(session.failure(), is(notNullValue()));
            assertThat(session.text(), is(""));
            assertThat(backend.requests(), is(empty()));
        }
    }

    /**
     * Known gap, pinned so a change on either side is noticed. When the engine fails after the stream
     * started, java-llama.cpp (like upstream llama-server) has already sent HTTP 200 and reports the
     * failure as an SSE {@code data: {"error":{...}}} object without a terminating {@code [DONE]}.
     * Atmosphere's SSE parser only reads {@code choices[0]}, so it ignores that object and completes
     * the session normally with whatever text arrived before — an empty answer here, not an error.
     * An in-stream error event would be a "SHOULD" for Atmosphere's {@code OpenAiCompatibleClient}.
     */
    @Test
    void midStreamEngineFailureCompletesSilentlyRatherThanErroring() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            throw new IllegalStateException("model exploded");
        });
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config(API_KEY)).start()) {
            AgentRunner runner = runner(server, API_KEY, List.of());
            ConsoleSession session = session();

            runner.run("hello", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(backend.requests(), hasSize(1));
            assertThat(session.text(), is(""));
            assertThat("Atmosphere does not surface an in-stream error object", session.failure(), is(nullValue()));
        }
    }
}
