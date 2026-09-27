// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Typing while the agent works stops that turn — and the turn after it has to run.
 *
 * <p>Reported as "it does not carry on by itself": the interruption printed its line and then nothing
 * followed. The sequence is driven here end to end against the real {@link OpenAiCompatServer} with
 * scripted llama.cpp chunks, because every part of it is a different piece of machinery — the
 * cancellable entry point of the runtime, the queue the console keeps, and the history the next turn
 * is sent with — and a defect in any of them looks identical from the outside.
 */
class InterruptedTurnTest {

    private static final String MODEL_ID = "local-model";

    /** Longer than one activity tick, so the wait actually looks for typed input. */
    private static final java.time.Duration SLOW_ENOUGH_TO_INTERRUPT = java.time.Duration.ofMillis(700);

    @TempDir
    Path workspace;

    /** A console that answers "yes, something was typed" whenever the test says so. */
    private final class Typing implements AgentTerminal {
        private volatile boolean pending;
        private final List<String> lines = new ArrayList<>();

        @Override
        public void line(String text) {
            lines.add(text);
        }

        @Override
        public String readLine(String prompt) {
            return null;
        }

        @Override
        public String readKey(String prompt) {
            return null;
        }

        @Override
        public boolean hasPendingInput() {
            return pending;
        }

        @Override
        public void status(List<String> statusLines) {}

        @Override
        public boolean pinsStatus() {
            return false;
        }

        @Override
        public Ansi ansi() {
            return Ansi.PLAIN;
        }

        @Override
        public void close() {}
    }

    private final Typing terminal = new Typing();

    private OpenAiCompatServer server(ScriptedBackend backend) throws Exception {
        return new OpenAiCompatServer(
                        backend,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .modelId(MODEL_ID)
                                .build())
                .start();
    }

    private AgentRunner runner(OpenAiCompatServer server) {
        return new AgentRunner(
                        "http://127.0.0.1:" + server.getPort() + "/v1",
                        "k",
                        MODEL_ID,
                        List.of(),
                        "You are a test agent.",
                        0.0,
                        64,
                        4)
                .retryPolicy(RetryPolicy.NONE);
    }

    /**
     * A turn that takes long enough to be interrupted.
     *
     * <p>Without this the test proves nothing: the interruption is only looked for while waiting for
     * the turn, and a scripted answer arrives before the first look. That is also the behaviour in a
     * real session — a turn that is already finished is not cut short — so the delay is what makes
     * this the reported situation rather than a different one.
     *
     * @param call which model call this is
     * @return the scripted answer, after a pause on the first call
     */
    private static List<String> slowFirstTurn(int call) {
        if (call == 1) {
            try {
                Thread.sleep(SLOW_ENOUGH_TO_INTERRUPT.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return ScriptedBackend.textTurn("answer " + call);
    }

    @Test
    void theTurnAfterAnInterruptedOneRunsAndCarriesTheHistory() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> slowFirstTurn(call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentRunner runner = runner(server);
            AgentFileSystem files = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
            List<ChatMessage> history = new ArrayList<>();
            ToolCallLog log = new ToolCallLog();

            // The user types while the first turn is still running.
            terminal.pending = true;
            ConsoleSession first = LocalAgent.turn(
                    runner, files, "a poem please", history, terminal, log, 1, ignored -> "", new TurnActivity());

            assertThat(
                    "the interruption is announced",
                    terminal.lines.stream().anyMatch(line -> line.contains("interrupted")),
                    is(true));

            // What was typed is now the next message, and nothing is pending any more.
            terminal.pending = false;
            ConsoleSession second = LocalAgent.turn(
                    runner, files, "make it longer", history, terminal, log, 2, ignored -> "", new TurnActivity());

            assertThat("the turn after the interruption produced an answer", second.text(), containsString("answer"));
            assertThat(
                    "and it was not itself reported as failed",
                    second.failure(),
                    is(org.hamcrest.Matchers.nullValue()));
            assertThat(
                    "the interrupted turn is in the history as asked",
                    history.get(0).content(),
                    is("a poem please"));

            List<JsonNode> requests = backend.requests();
            assertThat("both turns reached the server", requests.size() >= 2, is(true));
            JsonNode last = requests.get(requests.size() - 1).path("messages");
            assertThat(
                    "the second request carries the typed line",
                    last.get(last.size() - 1).path("content").asText(),
                    is("make it longer"));
            assertThat("and the interrupted question before it", last.toString(), containsString("a poem please"));
            assertThat(first.rounds(), hasSize(0));
        }
    }

    @Test
    void aSecondLineTypedDuringTheReplacementTurnStopsThatOneToo() throws Exception {
        // Not a defect: each typed line overtakes the turn it arrived in. It is pinned because the
        // symptom -- an interruption that seems to lead nowhere -- looks the same as a turn that never
        // starts, and telling the two apart afterwards is what took the longest.
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            try {
                Thread.sleep(SLOW_ENOUGH_TO_INTERRUPT.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return ScriptedBackend.textTurn("answer " + call);
        });
        try (OpenAiCompatServer server = server(backend)) {
            AgentRunner runner = runner(server);
            AgentFileSystem files = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
            List<ChatMessage> history = new ArrayList<>();
            ToolCallLog log = new ToolCallLog();

            terminal.pending = true;
            LocalAgent.turn(runner, files, "one", history, terminal, log, 1, ignored -> "", new TurnActivity());
            LocalAgent.turn(runner, files, "two", history, terminal, log, 2, ignored -> "", new TurnActivity());

            assertThat(
                    "both were cut short, and both said so",
                    terminal.lines.stream()
                            .filter(line -> line.contains("interrupted"))
                            .count(),
                    is(2L));
        }
    }
}
