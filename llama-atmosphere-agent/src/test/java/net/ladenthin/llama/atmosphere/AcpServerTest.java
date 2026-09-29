// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.not;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The editor front end end to end: an {@link AcpServer} on two pipes, driven with the JSON-RPC an editor
 * sends ({@link AcpTestClient}), the real agent session behind it, and the real {@link OpenAiCompatServer}
 * with a scripted engine behind that.
 */
class AcpServerTest {

    private static final Duration WAIT = Duration.ofSeconds(20);

    @TempDir
    Path workspace;

    private final List<AutoCloseable> open = new ArrayList<>();

    @AfterEach
    void closeAll() throws Exception {
        for (int i = open.size() - 1; i >= 0; i--) {
            open.get(i).close();
        }
    }

    private AcpTestClient start(ScriptedBackend backend, String... extra) throws Exception {
        OpenAiCompatServer model = new OpenAiCompatServer(
                        backend,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .apiKey("sk-local")
                                .modelId("local-model")
                                .build())
                .start();
        open.add(model);
        String baseUrl = "http://127.0.0.1:" + model.getPort() + "/v1";
        List<String> args = new ArrayList<>(List.of("--base-url", baseUrl, "--acp"));
        args.addAll(List.of(extra));
        AgentOptions options = AgentOptions.parse(args.toArray(String[]::new));
        AcpTestClient editor = new AcpTestClient();
        AcpServer server = AcpServer.start(
                cwd -> AgentSession.open(options.withWorkspace(cwd), baseUrl, 10_000),
                editor.agentIn,
                editor.agentOut,
                "test");
        open.add(server);
        // closed first: the editor hanging up is how an ACP session ends, and it gives the agent a clean
        // end of input rather than an interrupted read
        open.add(editor);
        return editor;
    }

    private static String initializeAndOpen(AcpTestClient editor, Path cwd) {
        JsonNode init = editor.response(
                editor.request(
                        "initialize",
                        AcpTestClient.object()
                                .put("protocolVersion", 1)
                                .set("clientCapabilities", AcpTestClient.object())),
                WAIT);
        assertThat(init.path("result").path("protocolVersion").asInt(), is(1));
        ObjectNode newSession = AcpTestClient.object().put("cwd", cwd.toString());
        newSession.putArray("mcpServers");
        JsonNode created = editor.response(editor.request("session/new", newSession), WAIT);
        return created.path("result").path("sessionId").asText();
    }

    private static int prompt(AcpTestClient editor, String sessionId, String text) {
        ObjectNode params = AcpTestClient.object().put("sessionId", sessionId);
        params.putArray("prompt").addObject().put("type", "text").put("text", text);
        return editor.request("session/prompt", params);
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
    void initializeNamesTheAgentAndANewSessionOffersTheModesAndTheCommands() throws Exception {
        AcpTestClient editor = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        JsonNode init = editor.response(
                editor.request(
                        "initialize",
                        AcpTestClient.object()
                                .put("protocolVersion", 1)
                                .set("clientCapabilities", AcpTestClient.object())),
                WAIT);
        assertThat(init.path("result").path("agentInfo").path("name").asText(), is("java-llama.cpp-agent"));

        ObjectNode newSession = AcpTestClient.object().put("cwd", workspace.toString());
        newSession.putArray("mcpServers");
        JsonNode result =
                editor.response(editor.request("session/new", newSession), WAIT).path("result");
        assertThat(result.path("sessionId").asText().isEmpty(), is(false));
        assertThat(result.path("modes").path("currentModeId").asText(), is("manual"));
        assertThat(result.path("modes").path("availableModes").toString(), containsString("\"auto\""));

        editor.await(message -> !editor.updates("available_commands_update").isEmpty(), WAIT);
        List<String> names = new ArrayList<>();
        editor.updates("available_commands_update")
                .get(0)
                .path("availableCommands")
                .forEach(command -> names.add(command.path("name").asText()));
        assertThat(names, hasItem("compact"));
        assertThat(names, hasItem("loop"));
        assertThat("nothing to exit in an editor", names, not(hasItem("exit")));
    }

    @Test
    void aPromptStreamsItsAnswerAndEndsTheTurn() throws Exception {
        AcpTestClient editor =
                start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("Hello ", "from the agent")));
        String session = initializeAndOpen(editor, workspace);

        JsonNode response = editor.response(prompt(editor, session, "hi"), WAIT);

        assertThat(response.path("result").path("stopReason").asText(), is("end_turn"));
        assertThat(editor.messageText(), is("Hello from the agent"));
    }

    @Test
    void theToolsWorkInTheDirectoryTheEditorNamed() throws Exception {
        Files.writeString(workspace.resolve("hello.txt"), "VALUE=42\n");
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", "read_file", "{\"file_path\":\"hello.txt\"}")
                : ScriptedBackend.textTurn("It says 42."));
        AcpTestClient editor = start(backend);
        String session = initializeAndOpen(editor, workspace);

        editor.response(prompt(editor, session, "read it"), WAIT);

        assertThat(toolResult(backend.requests().get(1)), containsString("VALUE=42"));
        JsonNode call = editor.updates("tool_call").get(0);
        assertThat(call.path("kind").asText(), is("read"));
        assertThat(call.path("title").asText(), containsString("hello.txt"));
        assertThat(
                "the editor can open the file",
                call.path("locations").get(0).path("path").asText(),
                is(workspace.resolve("hello.txt").toString()));
        JsonNode done = editor.updates("tool_call_update").get(0);
        assertThat(done.path("toolCallId").asText(), is(call.path("toolCallId").asText()));
        assertThat(done.path("status").asText(), is("completed"));
    }

    @Test
    void aGatedCallAsksTheEditorAndAllowRunsIt() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", ShellTool.TOOL_NAME, "{\"command\":\"echo from-the-editor\"}")
                : ScriptedBackend.textTurn("Done."));
        AcpTestClient editor = start(backend, "--allow-shell");
        String session = initializeAndOpen(editor, workspace);

        int id = prompt(editor, session, "run it");
        JsonNode ask = editor.await(
                message -> "session/request_permission"
                        .equals(message.path("method").asText()),
                WAIT);
        assertThat(ask.path("params").path("toolCall").path("kind").asText(), is("execute"));
        assertThat(ask.path("params").path("options").toString(), containsString("allow_always"));
        editor.respond(ask.path("id"), (ObjectNode) AcpTestClient.object()
                .set(
                        "outcome",
                        AcpTestClient.object().put("outcome", "selected").put("optionId", "allow")));
        JsonNode response = editor.response(id, WAIT);

        assertThat(response.path("result").path("stopReason").asText(), is("end_turn"));
        assertThat(toolResult(backend.requests().get(1)), containsString("from-the-editor"));
    }

    @Test
    void rejectKeepsTheCommandFromRunningAndACancelledDialogIsANo() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call % 2 == 1
                ? ScriptedBackend.toolCallTurn("call_" + call, ShellTool.TOOL_NAME, "{\"command\":\"echo nope\"}")
                : ScriptedBackend.textTurn("Understood."));
        AcpTestClient editor = start(backend, "--allow-shell");
        String session = initializeAndOpen(editor, workspace);

        int first = prompt(editor, session, "run it");
        JsonNode ask = editor.await(
                message -> "session/request_permission"
                        .equals(message.path("method").asText()),
                WAIT);
        editor.respond(ask.path("id"), (ObjectNode) AcpTestClient.object()
                .set(
                        "outcome",
                        AcpTestClient.object().put("outcome", "selected").put("optionId", "reject")));
        editor.response(first, WAIT);
        assertThat(toolResult(backend.requests().get(1)), containsString("cancelled"));

        int second = prompt(editor, session, "run it again");
        JsonNode again = editor.await(
                message -> "session/request_permission"
                                .equals(message.path("method").asText())
                        && !message.path("id").equals(ask.path("id")),
                WAIT);
        editor.respond(again.path("id"), (ObjectNode)
                AcpTestClient.object().set("outcome", AcpTestClient.object().put("outcome", "cancelled")));
        editor.response(second, WAIT);
        assertThat(toolResult(backend.requests().get(3)), containsString("cancelled"));
    }

    @Test
    void allowAlwaysSwitchesTheSessionToAutoAndTheEditorIsTold() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call % 2 == 1
                ? ScriptedBackend.toolCallTurn("call_" + call, ShellTool.TOOL_NAME, "{\"command\":\"echo ok\"}")
                : ScriptedBackend.textTurn("Done."));
        AcpTestClient editor = start(backend, "--allow-shell");
        String session = initializeAndOpen(editor, workspace);

        int first = prompt(editor, session, "run it");
        JsonNode ask = editor.await(
                message -> "session/request_permission"
                        .equals(message.path("method").asText()),
                WAIT);
        editor.respond(ask.path("id"), (ObjectNode) AcpTestClient.object()
                .set(
                        "outcome",
                        AcpTestClient.object().put("outcome", "selected").put("optionId", "always")));
        editor.response(first, WAIT);
        assertThat(
                editor.updates("current_mode_update")
                        .get(0)
                        .path("currentModeId")
                        .asText(),
                is("auto"));

        editor.response(prompt(editor, session, "and again"), WAIT);
        long questions = editor.received.stream()
                .filter(message -> "session/request_permission"
                        .equals(message.path("method").asText()))
                .count();
        assertThat("auto mode does not ask again", questions, is(1L));
        assertThat(toolResult(backend.requests().get(3)), containsString("ok"));
    }

    @Test
    void theEditorCanSwitchTheModeAndCommandsWork() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x"));
        AcpTestClient editor = start(backend);
        String session = initializeAndOpen(editor, workspace);

        editor.response(
                editor.request(
                        "session/set_mode",
                        AcpTestClient.object().put("sessionId", session).put("modeId", "auto")),
                WAIT);
        editor.response(prompt(editor, session, "/status"), WAIT);

        assertThat(editor.messageText(), containsString(ApprovalMode.AUTO.badge()));
        assertThat(editor.messageText(), containsString("history: 0 messages"));
        assertThat(backend.requests(), is(empty()));
    }

    @Test
    void cancelEndsARunningPromptWithoutWaitingForIt() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            if (call == 1) {
                try {
                    Thread.sleep(8_000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            return ScriptedBackend.textTurn("answer " + call);
        });
        AcpTestClient editor = start(backend);
        String session = initializeAndOpen(editor, workspace);
        long started = System.nanoTime();

        int id = prompt(editor, session, "take your time");
        Thread.sleep(500);
        editor.notify("session/cancel", AcpTestClient.object().put("sessionId", session));
        JsonNode response = editor.response(id, WAIT);

        assertThat(response.path("result").path("stopReason").asText(), is("cancelled"));
        assertThat(Duration.ofNanos(System.nanoTime() - started).toMillis(), lessThan(6_000L));
    }

    @Test
    void anAttachedFileBecomesPartOfTheMessage() {
        String text = AcpServer.text(List.of(
                new com.agentclientprotocol.sdk.spec.AcpSchema.TextContent("explain this"),
                new com.agentclientprotocol.sdk.spec.AcpSchema.Resource(
                        "resource",
                        new com.agentclientprotocol.sdk.spec.AcpSchema.TextResourceContents(
                                "int x;", "file:///a/B.java", "text/x-java"),
                        null,
                        null)));

        assertThat(text, containsString("explain this"));
        assertThat(text, containsString("file:///a/B.java"));
        assertThat(text, containsString("int x;"));
    }
}
