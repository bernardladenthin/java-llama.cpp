// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The loop: when it stops, when it does not, and what it sends. */
class TaskLoopTest {

    private static final String MODEL_ID = "local-model";
    private static final Duration BUDGET = Duration.ofMinutes(5);

    @TempDir
    Path workspace;

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();

    private AgentTerminal terminal() {
        return new PlainTerminal(new PrintStream(console, true, StandardCharsets.UTF_8), null, Ansi.PLAIN);
    }

    private TaskLoop.Outcome runLoop(ScriptedBackend backend, LoopOptions options) throws Exception {
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("k")
                .modelId(MODEL_ID)
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config).start()) {
            AgentRunner runner = new AgentRunner(
                            "http://127.0.0.1:" + server.getPort() + "/v1",
                            "k",
                            MODEL_ID,
                            List.of(),
                            "You are a test agent.",
                            0.0,
                            64,
                            5)
                    .retryPolicy(RetryPolicy.NONE);
            AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
            return TaskLoop.run(runner, fs, terminal(), workspace, options, () -> false, BUDGET, new ToolCallLog());
        }
    }

    // ----- the stop marker -----

    @Test
    void onlyAWholeLineEndsTheLoop() {
        assertThat(TaskLoop.isComplete("all done\n" + TaskLoop.SENTINEL), is(true));
        assertThat(TaskLoop.isComplete("  " + TaskLoop.SENTINEL + "  "), is(true));
        // the failure this guards against: the model talking ABOUT the marker
        assertThat(TaskLoop.isComplete("I will answer " + TaskLoop.SENTINEL + " when I am done."), is(false));
        assertThat(TaskLoop.isComplete("TASK_COMPLETE"), is(false));
        assertThat(TaskLoop.isComplete("still working"), is(false));
    }

    @Test
    void everyStepSendsTheTaskVerbatimAndNamesTheFile() {
        String prompt = TaskLoop.stepPrompt(new LoopOptions("rename the class", null, 5, null));

        assertThat(prompt, containsString("rename the class"));
        assertThat(prompt, containsString(TaskLoop.LOOP_FILE));
        assertThat(prompt, containsString(TaskLoop.SENTINEL));
        assertThat(prompt.contains("{"), is(false));
    }

    @Test
    void theCheckCommandIsPartOfTheInstructionsWhenThereIsOne() {
        assertThat(
                TaskLoop.stepPrompt(new LoopOptions("fix it", null, 5, "mvn -q test")), containsString("mvn -q test"));
    }

    // ----- the file -----

    @Test
    void theLoopFileIsCreatedOnceAndKeepsWhatIsInIt() throws Exception {
        Path file = TaskLoop.ensureLoopFile(workspace, "write a parser");

        assertThat(Files.readString(file), containsString("write a parser"));
        Files.writeString(file, "edited by the agent");
        assertThat(TaskLoop.ensureLoopFile(workspace, "write a parser"), is(file));
        assertThat(Files.readString(file), is("edited by the agent"));
    }

    @Test
    void theFingerprintChangesWithTheContent() throws Exception {
        Path file = TaskLoop.ensureLoopFile(workspace, "task");
        long before = TaskLoop.fingerprint(file);

        Files.writeString(file, "something else");

        assertThat(TaskLoop.fingerprint(file) != before, is(true));
        assertThat(TaskLoop.fingerprint(workspace.resolve("absent.md")), is(-1L));
    }

    // ----- the loop itself, over the real server -----

    @Test
    void theLoopEndsWhenTheModelAnswersWithTheMarker() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call < 3
                ? ScriptedBackend.textTurn("working on step ", String.valueOf(call))
                : ScriptedBackend.textTurn(TaskLoop.SENTINEL));

        TaskLoop.Outcome outcome = runLoop(backend, new LoopOptions("do the thing", null, 10, null));

        assertThat(outcome.completed(), is(true));
        assertThat(outcome.reason(), containsString("3 steps"));
        assertThat(backend.requests().size(), is(3));
    }

    @Test
    void everyStepStartsFromAnEmptyHistorySoTheContextCannotGrow() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) ->
                call < 2 ? ScriptedBackend.textTurn("step") : ScriptedBackend.textTurn(TaskLoop.SENTINEL));

        runLoop(backend, new LoopOptions("do the thing", null, 10, null));

        // system + user, every single time: the file is the memory, the conversation is not
        for (var request : backend.requests()) {
            assertThat(request.path("messages").size(), is(2));
        }
    }

    @Test
    void aModelThatRepeatsItselfWithoutChangingAnythingIsStopped() throws Exception {
        // no tool calls, no file change: exactly the shape of a model looping on its own answer
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("thinking…"));

        TaskLoop.Outcome outcome = runLoop(backend, new LoopOptions("do the thing", null, 50, null));

        assertThat(outcome.completed(), is(false));
        assertThat(outcome.reason(), containsString("no progress"));
        assertThat(backend.requests().size(), is(TaskLoop.STALL_LIMIT));
    }

    @Test
    void theStepLimitStopsALoopThatWouldOtherwiseRunOn() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            // something changes every step, so the stall detector never fires
            Files.writeString(workspace.resolve(TaskLoop.LOOP_FILE), "step " + call);
            return ScriptedBackend.textTurn("still working");
        });

        TaskLoop.Outcome outcome = runLoop(backend, new LoopOptions("endless", null, 2, null));

        assertThat(outcome.completed(), is(false));
        assertThat(outcome.reason(), containsString("step limit"));
        assertThat(backend.requests().size(), is(2));
    }

    @Test
    void aFailingCheckRejectsTheMarkerAndFeedsTheOutputBack() throws Exception {
        String failing = ShellTool.isWindows() ? "exit 7" : "exit 7";
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            Files.writeString(workspace.resolve(TaskLoop.LOOP_FILE), "step " + call);
            return ScriptedBackend.textTurn(TaskLoop.SENTINEL);
        });

        TaskLoop.Outcome outcome = runLoop(backend, new LoopOptions("claim it works", null, 2, failing));

        assertThat("the model said done, the check said otherwise", outcome.completed(), is(false));
        assertThat(
                backend.requests()
                        .get(1)
                        .path("messages")
                        .get(1)
                        .path("content")
                        .asText(),
                containsString("the check"));
    }

    // ----- the argument parser -----

    @Test
    void theTaskIsEverythingAfterTheFlags() {
        LoopOptions plain = LoopOptions.parse("keep the README in sync with the code");

        assertThat(plain.task(), is("keep the README in sync with the code"));
        assertThat(plain.interval(), is(nullValue()));
        assertThat(plain.maxSteps(), is(LoopOptions.DEFAULT_MAX_STEPS));
        assertThat(plain.check(), is(nullValue()));
    }

    @Test
    void flagsAreReadFromTheFrontAndAQuotedCheckKeepsItsSpaces() {
        LoopOptions options = LoopOptions.parse("--every 90s --max 3 --check 'mvn -q test' fix the build");

        assertThat(options.interval(), is(Duration.ofSeconds(90)));
        assertThat(options.maxSteps(), is(3));
        assertThat(options.check(), is("mvn -q test"));
        assertThat(options.task(), is("fix the build"));
    }

    @Test
    void durationsAreWrittenAsPeopleWriteThem() {
        assertThat(LoopOptions.parseDuration("30s"), is(Duration.ofSeconds(30)));
        assertThat(LoopOptions.parseDuration("5m"), is(Duration.ofMinutes(5)));
        assertThat(LoopOptions.parseDuration("2h"), is(Duration.ofHours(2)));
        assertThat(LoopOptions.parseDuration("10"), is(Duration.ofMinutes(10)));
    }

    @Test
    void aMalformedLoopCommandSaysWhatIsWrong() {
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class, () -> LoopOptions.parse(""))
                        .getMessage(),
                containsString("Usage: /loop"));
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class, () -> LoopOptions.parse("--every soon do it"))
                        .getMessage(),
                containsString("Not a duration"));
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class, () -> LoopOptions.parse("--max zero do it"))
                        .getMessage(),
                containsString("--max expects a number"));
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class, () -> LoopOptions.parse("--bogus x do it"))
                        .getMessage(),
                containsString("Unknown /loop flag"));
    }
}
