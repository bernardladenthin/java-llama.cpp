// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Drives {@link LocalAgent#run} end to end — option parsing, the workspace-confined built-in file
 * tools, the console rendering and the exit code — against the real {@link OpenAiCompatServer} with a
 * scripted engine. This is the one test that proves Atmosphere's own {@code FileSystemTools} work
 * headless: they resolve the {@code AgentFileSystem} from the session's injectables, which
 * {@link ConsoleSession} supplies.
 */
class LocalAgentTest {

    @TempDir
    Path workspace;

    private static OpenAiCompatServer server(ScriptedBackend backend) throws Exception {
        return new OpenAiCompatServer(
                        backend,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .apiKey("sk-local")
                                .modelId("local-model")
                                .build())
                .start();
    }

    @Test
    void oneShotTurnReadsAWorkspaceFileThroughTheBuiltInFileTools() throws Exception {
        Files.writeString(workspace.resolve("hello.txt"), "VALUE=42\n");
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", "read_file", "{\"path\":\"hello.txt\"}")
                : ScriptedBackend.textTurn("The file says VALUE=42."));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url",
                "http://127.0.0.1:" + server.getPort() + "/v1",
                "--workspace",
                workspace.toString(),
                "--prompt",
                "What does hello.txt say?"
            });

            int exit = LocalAgent.run(
                    options,
                    null,
                    new PrintStream(out, true, StandardCharsets.UTF_8),
                    new PrintStream(err, true, StandardCharsets.UTF_8));

            assertThat(exit, is(0));
        }
        String console = out.toString(StandardCharsets.UTF_8);
        assertThat(console, containsString("⚙ read_file {path=hello.txt}"));
        assertThat(console, containsString("↳ VALUE=42"));
        assertThat(console, containsString("The file says VALUE=42."));
        List<JsonNode> requests = backend.requests();
        assertThat(requests, hasSize(2));
        // The built-in file tools were offered to the model ...
        assertThat(requests.get(0).path("tools").toString(), containsString("\"name\":\"read_file\""));
        assertThat(requests.get(0).path("tools").toString(), containsString("\"name\":\"edit_file\""));
        assertThat(requests.get(0).path("tools").toString().contains(ShellTool.TOOL_NAME), is(false));
        // ... and the tool's real result (the file content) travelled back to the model.
        JsonNode toolMessage = requests.get(1).path("messages").get(3);
        assertThat(toolMessage.path("role").asText(), is("tool"));
        assertThat(toolMessage.path("content").asText(), containsString("VALUE=42"));
        assertThat(err.toString(StandardCharsets.UTF_8), containsString("tools=[ls, read_file"));
    }

    @Test
    void interactiveModeRunsTurnsUntilExitAndKeepsHistory() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url",
                "http://127.0.0.1:" + server.getPort() + "/v1",
                "--workspace",
                workspace.toString(),
                "--allow-shell"
            });

            int exit = LocalAgent.run(
                    options,
                    new StringReader("first\n\nsecond\n/exit\n"),
                    new PrintStream(out, true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));

            assertThat(exit, is(0));
        }
        List<JsonNode> requests = backend.requests();
        assertThat(requests, hasSize(2));
        // The second turn carries the first turn as history: system, user, assistant, user.
        assertThat(requests.get(1).path("messages").size(), is(4));
        assertThat(requests.get(1).path("messages").get(2).path("content").asText(), is("answer 1"));
        assertThat(requests.get(1).path("messages").get(3).path("content").asText(), is("second"));
        assertThat(
                requests.get(0).path("tools").toString(), containsString("\"name\":\"" + ShellTool.TOOL_NAME + "\""));
        assertThat(out.toString(StandardCharsets.UTF_8), containsString("answer 2"));
    }

    @Test
    void failedTurnExitsNonZero() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("never"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url",
                "http://127.0.0.1:" + server.getPort() + "/v1",
                "--api-key",
                "wrong",
                "--workspace",
                workspace.toString(),
                "-p",
                "hello"
            });

            int exit = LocalAgent.run(
                    options,
                    null,
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));

            assertThat(exit, is(1));
        }
    }
}
