// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.not;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalResolution;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.PendingApproval;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The contract every front end relies on: {@link AgentSession} driven through a {@link SessionFrontend}
 * that is none of the real ones, against the real {@link OpenAiCompatServer} with a scripted engine.
 *
 * <p>If one of these fails, every front end is affected at once — which is the point of testing the
 * session on its own rather than only through the console.
 */
class AgentSessionTest {

    private static final String MODEL_ID = "local-model";

    @TempDir
    Path workspace;

    private static OpenAiCompatServer server(ScriptedBackend backend) throws Exception {
        return new OpenAiCompatServer(
                        backend,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .apiKey("sk-local")
                                .modelId(MODEL_ID)
                                .build())
                .start();
    }

    private AgentSession session(OpenAiCompatServer server, String... extra) {
        List<String> args = new java.util.ArrayList<>(List.of(
                "--base-url",
                "http://127.0.0.1:" + server.getPort() + "/v1",
                "--workspace",
                workspace.toString(),
                "--api-key",
                "sk-local"));
        args.addAll(List.of(extra));
        return AgentSession.open(
                AgentOptions.parse(args.toArray(String[]::new)),
                "http://127.0.0.1:" + server.getPort() + "/v1",
                10_000);
    }

    private static String toolResult(JsonNode request) {
        for (JsonNode message : request.path("messages")) {
            if ("tool".equals(message.path("role").asText())) {
                return message.path("content").asText();
            }
        }
        return "";
    }

    /** Approves or denies every gated call, and counts the questions. */
    private static final class Answering implements ApprovalStrategy {
        final List<String> asked = new CopyOnWriteArrayList<>();
        private final boolean approve;

        Answering(boolean approve) {
            this.approve = approve;
        }

        @Override
        public ApprovalOutcome awaitApproval(PendingApproval approval, StreamingSession session) {
            return awaitApprovalDetailed(approval, session).outcome();
        }

        @Override
        public ApprovalResolution awaitApprovalDetailed(PendingApproval approval, StreamingSession session) {
            asked.add(approval.toolName());
            return approve ? ApprovalResolution.approve() : ApprovalResolution.deny();
        }
    }

    @Test
    void aMessageRunsATurnAndBothHalvesEndUpInTheHistory() throws Exception {
        ScriptedBackend backend =
                new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("hello ", "from ", "turn " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            assertThat(session.submit("hi", frontend), is(AgentSession.Result.CONTINUE));

            assertThat(String.join("", frontend.streamed), is("hello from turn 1"));
            assertThat("the renderer is told the turn is over", frontend.ends, contains("complete"));
            assertThat(session.history(), hasSize(2));
            assertThat(session.history().get(0).content(), is("hi"));
            assertThat(session.history().get(1).content(), is("hello from turn 1"));
            assertThat(session.isBusy(), is(false));
        }
    }

    @Test
    void commandsAnswerOnTheFrontEndAndNeverReachTheModel() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            session.submit("/help", frontend);
            session.submit("/status", frontend);
            session.submit("/tools", frontend);
            session.submit("/calls", frontend);

            assertThat(backend.requests(), is(empty()));
            assertThat(frontend.allLines(), containsString("/compact"));
            assertThat(frontend.allLines(), containsString("workspace: " + workspace));
            assertThat(frontend.allLines(), containsString("tools: "));
            assertThat(session.submit("/exit", frontend), is(AgentSession.Result.EXIT));
        }
    }

    @Test
    void anUnknownSlashWordIsAMessageForTheModel() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("ok"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);

            session.submit("/usr/bin/java -version", new RecordingFrontend());

            assertThat(backend.requests(), hasSize(1));
        }
    }

    @Test
    void theModeCanBeSetByCommandByCycleAndDirectly() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            assertThat(session.mode(), is(ApprovalMode.MANUAL));
            session.submit("/mode auto", frontend);
            assertThat(session.mode(), is(ApprovalMode.AUTO));
            assertThat(session.cycleMode(), is(ApprovalMode.MANUAL));
            session.mode(ApprovalMode.AUTO);
            assertThat(session.status(), containsString(ApprovalMode.AUTO.badge()));
            session.submit("/mode nonsense", frontend);
            assertThat("a bad mode leaves the mode alone", session.mode(), is(ApprovalMode.AUTO));
        }
    }

    @Test
    void manualModeAsksTheFrontEndAndADenialNeverRuns() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", ShellTool.TOOL_NAME, "{\"command\":\"echo ran-it\"}")
                : ScriptedBackend.textTurn("Understood."));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server, "--allow-shell");
            Answering no = new Answering(false);
            RecordingFrontend frontend = new RecordingFrontend(no, null);

            session.submit("run it", frontend);

            assertThat(no.asked, contains(ShellTool.TOOL_NAME));
            assertThat(toolResult(backend.requests().get(1)), containsString("cancelled"));
            assertThat("nothing ran, so nothing was printed", frontend.commandOutput, is(empty()));
        }
    }

    @Test
    void anApprovalRunsTheCommandAndItsOutputReachesTheFrontEndLive() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", ShellTool.TOOL_NAME, "{\"command\":\"echo ran-it\"}")
                : ScriptedBackend.textTurn("Understood."));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server, "--allow-shell");
            Answering yes = new Answering(true);
            RecordingFrontend frontend = new RecordingFrontend(yes, null);

            session.submit("run it", frontend);

            assertThat(yes.asked, contains(ShellTool.TOOL_NAME));
            assertThat(frontend.commandOutput, hasItem(containsString("ran-it")));
            assertThat(toolResult(backend.requests().get(1)), containsString("ran-it"));
            assertThat(frontend.events, hasItem("ToolStart " + ShellTool.TOOL_NAME));
            assertThat(frontend.events, hasItem("ToolResult " + ShellTool.TOOL_NAME));
        }
    }

    @Test
    void aFrontEndThatCannotAskDeniesAndAutoModeNeverAsks() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call % 2 == 1
                ? ScriptedBackend.toolCallTurn("call_" + call, ShellTool.TOOL_NAME, "{\"command\":\"echo ran-it\"}")
                : ScriptedBackend.textTurn("Understood."));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server, "--allow-shell");

            session.submit("run it", new RecordingFrontend());
            assertThat(toolResult(backend.requests().get(1)), containsString("cancelled"));

            session.mode(ApprovalMode.AUTO);
            Answering never = new Answering(false);
            session.submit("run it again", new RecordingFrontend(never, null));
            assertThat(never.asked, is(empty()));
            assertThat(toolResult(backend.requests().get(3)), containsString("ran-it"));
        }
    }

    @Test
    void cancelFromAnotherThreadEndsTheTurnAndFreesTheSession() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            if (call == 1) {
                try {
                    Thread.sleep(5_000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            return ScriptedBackend.textTurn("answer " + call);
        });
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend().stoppedBy(session);
            long started = System.nanoTime();
            Future<AgentSession.Result> first = pool.submit(() -> session.submit("slow", frontend));
            long deadline = System.nanoTime() + Duration.ofSeconds(3).toNanos();
            while (!session.isBusy() && System.nanoTime() < deadline) {
                Thread.sleep(10);
            }
            Thread.sleep(300);

            session.cancel();

            assertThat(first.get(3, TimeUnit.SECONDS), is(AgentSession.Result.CONTINUE));
            assertThat(
                    "it did not wait for the slow answer",
                    Duration.ofNanos(System.nanoTime() - started).toMillis(),
                    lessThan(4_000L));
            // The next request is served normally: the stop belonged to the request it was aimed at.
            session.submit("next", frontend);
            assertThat(session.history().get(session.history().size() - 1).content(), is("answer 2"));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void retryAsksAgainAndTheRecordHasTheQuestionOnce() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            session.submit("same question", frontend);
            session.submit("/retry", frontend);
            session.submit("same question", frontend);
            session.submit("/save record.txt", frontend);

            String record = Files.readString(workspace.resolve("record.txt"), StandardCharsets.UTF_8);
            assertThat(
                    "asked by the user twice, retried once", record.split("you: same question", -1).length - 1, is(2));
            assertThat(record, containsString("retrying: same question"));
            assertThat(
                    "the retry replaced the first answer instead of following it",
                    session.history().stream().map(m -> m.content()).toList(),
                    contains("same question", "answer 2", "same question", "answer 3"));
        }
    }

    @Test
    void clearForgetsTheConversationAndClearsTheScreen() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            session.submit("remember me", frontend);
            session.submit("/clear", frontend);
            session.submit("/retry", frontend);

            assertThat(session.history(), is(empty()));
            assertThat(frontend.clears(), is(1));
            assertThat(frontend.allLines(), containsString("nothing to retry yet"));
            assertThat(backend.requests(), hasSize(1));
        }
    }

    @Test
    void aLoopInManualModeWithNobodyToAskIsRefusedRatherThanRunUnattended() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);
            RecordingFrontend frontend = new RecordingFrontend();

            session.submit("/loop write a poem", frontend);

            assertThat(frontend.questions, hasSize(1));
            assertThat(frontend.allLines(), containsString("loop: cancelled"));
            assertThat(session.mode(), is(ApprovalMode.MANUAL));
            assertThat(backend.requests(), is(empty()));
        }
    }

    @Test
    void aLoopRunsItsStepsThroughTheFrontEndUntilTheMarker() throws Exception {
        ScriptedBackend backend = new ScriptedBackend(
                (call, request) -> ScriptedBackend.textTurn("working" + System.lineSeparator() + TaskLoop.SENTINEL));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server, "--auto");
            RecordingFrontend frontend = new RecordingFrontend();

            session.submit("/loop write a poem", frontend);

            assertThat(frontend.questions, is(empty()));
            assertThat(frontend.allLines(), containsString("loop: done after 1 steps"));
            assertThat("a loop does not touch the conversation", session.history(), is(empty()));
        }
    }

    @Test
    void aOneShotMessageIsSentAsItIsEvenWhenItLooksLikeACommand() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("done"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentSession session = session(server);

            TurnRecorder turn = session.send("/help me", new RecordingFrontend());

            assertThat(turn.failure(), is((Throwable) null));
            assertThat(turn.text(), is("done"));
            assertThat(backend.requests().get(0).path("messages").toString(), containsString("/help me"));
            assertThat(session.history().get(0).content(), not(containsString("Record of the tools")));
        }
    }
}
