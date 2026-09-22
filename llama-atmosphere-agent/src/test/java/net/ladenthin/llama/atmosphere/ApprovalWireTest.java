// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The approval gate over the real wire: a denial must stop the tool <em>and</em> reach the model as a
 * tool result, so it replans instead of believing the command ran.
 *
 * <p>Both halves matter and neither is ours: Atmosphere decides whether to ask
 * ({@link ConsoleApprovalStrategy#policy()}), and Atmosphere turns the answer into the {@code role:
 * "tool"} message. This test drives the whole path — scripted llama.cpp chunks through the real
 * {@link OpenAiCompatServer}, the real tool loop, a console answer of {@code n} or {@code y} — and
 * asserts what ends up on the wire.
 */
class ApprovalWireTest {

    private static final String MODEL_ID = "local-model";
    private static final String API_KEY = "k";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    @TempDir
    Path workspace;

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();

    private AgentRunner runner(OpenAiCompatServer server, List<ToolDefinition> tools, String typed) {
        AtomicReference<ApprovalMode> mode = new AtomicReference<>(ApprovalMode.MANUAL);
        PrintStream out = new PrintStream(console, true, StandardCharsets.UTF_8);
        return new AgentRunner(
                        "http://127.0.0.1:" + server.getPort() + "/v1",
                        API_KEY,
                        MODEL_ID,
                        tools,
                        "You are a test agent.",
                        0.0,
                        64,
                        10)
                .retryPolicy(RetryPolicy.NONE)
                .approval(
                        new ConsoleApprovalStrategy(mode, new BufferedReader(new StringReader(typed)), out, Ansi.PLAIN),
                        ConsoleApprovalStrategy.policy());
    }

    private ConsoleSession session() {
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        return new ConsoleSession(new PrintStream(console, true, StandardCharsets.UTF_8), fs);
    }

    private static OpenAiServerConfig config() {
        return OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey(API_KEY)
                .modelId(MODEL_ID)
                .build();
    }

    private static ToolDefinition shellLike(String name, List<String> invocations) {
        return ToolDefinition.builder(name, "Test tool " + name)
                .parameter("command", "The command line", "string", true)
                .executor(args -> {
                    invocations.add(String.valueOf(args.get("command")));
                    return "ran";
                })
                .build();
    }

    private static ScriptedBackend backend(String toolName) {
        return new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", toolName, "{\"command\":\"rm -rf build\"}")
                : ScriptedBackend.textTurn("Understood."));
    }

    private static String toolResult(JsonNode request) {
        for (JsonNode message : request.path("messages")) {
            if ("tool".equals(message.path("role").asText())) {
                return message.path("content").asText();
            }
        }
        return "";
    }

    @Test
    void aDeniedShellCallNeverRunsAndTheModelIsToldItWasCancelled() throws Exception {
        List<String> invocations = new CopyOnWriteArrayList<>();
        ScriptedBackend backend = backend(ShellTool.TOOL_NAME);
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            AgentRunner runner =
                    runner(server, List.of(shellLike(ShellTool.TOOL_NAME, invocations)), "n" + System.lineSeparator());
            ConsoleSession session = session();

            runner.run("Delete the build directory.", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat("the executor must not have run", invocations, is(empty()));
            List<JsonNode> requests = backend.requests();
            assertThat(requests, hasSize(2));
            // Atmosphere's own wording; the point is that the model is told, not what it says.
            assertThat(toolResult(requests.get(1)), containsString("cancelled"));
            assertThat(console.toString(StandardCharsets.UTF_8), containsString("rm -rf build"));
        }
    }

    @Test
    void anApprovedShellCallRunsAndItsRealOutputIsSentBack() throws Exception {
        List<String> invocations = new CopyOnWriteArrayList<>();
        ScriptedBackend backend = backend(ShellTool.TOOL_NAME);
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            AgentRunner runner =
                    runner(server, List.of(shellLike(ShellTool.TOOL_NAME, invocations)), "y" + System.lineSeparator());
            ConsoleSession session = session();

            runner.run("Delete the build directory.", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(invocations, contains("rm -rf build"));
            assertThat(toolResult(backend.requests().get(1)), is("ran"));
        }
    }

    @Test
    void aReadingToolIsNotGatedAndRunsWithoutAnyAnswer() throws Exception {
        List<String> invocations = new CopyOnWriteArrayList<>();
        ScriptedBackend backend = backend("ls");
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            // empty console input: if this tool asked, the strategy would read EOF and deny
            AgentRunner runner = runner(server, List.of(shellLike("ls", invocations)), "");
            ConsoleSession session = session();

            runner.run("List the files.", List.of(), session);

            assertThat(session.await(TIMEOUT), is(true));
            assertThat(invocations, contains("rm -rf build"));
            assertThat(toolResult(backend.requests().get(1)), is("ran"));
        }
    }
}
