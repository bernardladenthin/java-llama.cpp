// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.not;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
                ? ScriptedBackend.toolCallTurn("call_1", "read_file", "{\"file_path\":\"hello.txt\"}")
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
        // the tool line and its result, as ConsoleSession renders them (unstyled here: not a terminal)
        assertThat(console, containsString("● read_file {file_path=hello.txt}"));
        // our read_file numbers the lines, so the result is "  1: VALUE=42"
        assertThat(console, containsString("↳   1: VALUE=42"));
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
    void theSessionIsRecordedAndSavedWhereTheToolsWork() throws Exception {
        // End to end through the real server: what was typed, what came back, written by /save with a
        // timestamp on every line. /compact keeps the record -- it rewrites what the model is sent,
        // not what happened -- and only /clear empties it.
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            int exit = LocalAgent.run(
                    options,
                    new StringReader("what is two plus two\n/compact\n/save session.txt\n/exit\n"),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));

            assertThat(exit, is(0));
            java.nio.file.Path written = workspace.resolve("session.txt");
            assertThat("the file lands in the workspace", java.nio.file.Files.exists(written), is(true));
            String text = java.nio.file.Files.readString(written, StandardCharsets.UTF_8);
            assertThat(text, containsString("you: what is two plus two"));
            assertThat("the answer survived the compaction", text, containsString("agent: answer 1"));
            assertThat("and the compaction is noted rather than hidden", text, containsString("compacted"));
            assertThat("every line is stamped", text.startsWith("["), is(true));
        }
    }

    @Test
    void clearingEmptiesTheRecordAsWell() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader("remember this\n/clear\n/save after-clear.txt\n/exit\n"),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));

            String text = java.nio.file.Files.readString(workspace.resolve("after-clear.txt"), StandardCharsets.UTF_8);
            assertThat("forget the session means the record too", text, not(containsString("remember this")));
        }
    }

    @Test
    void retryAsksAgainWithoutTheAnswerThatCameBack() throws Exception {
        // The point of a retry: the model must not see what it said last time, or it says it again.
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("answer " + call));
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader("the question\n/retry\n/exit\n"),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        }
        List<JsonNode> requests = backend.requests();
        assertThat("it was asked twice", requests, hasSize(2));
        JsonNode second = requests.get(1).path("messages");
        assertThat("system and the question, and nothing else", second.size(), is(2));
        assertThat(second.get(1).path("content").asText(), is("the question"));
        assertThat(
                "the first answer is gone from the conversation", second.toString(), not(containsString("answer 1")));
    }

    @Test
    void retryBeforeAnythingWasAskedSaysSo() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("never"));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader("/retry\n/exit\n"),
                    new PrintStream(out, true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        }
        assertThat(out.toString(StandardCharsets.UTF_8), containsString("nothing to retry"));
        assertThat("and nothing was sent", backend.requests(), hasSize(0));
    }

    @Test
    void aSystemPromptCanComeFromAFile() throws Exception {
        java.nio.file.Path promptFile = workspace.resolve("persona.txt");
        java.nio.file.Files.writeString(promptFile, "You answer only in haiku.");
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("ok"));
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url",
                "http://127.0.0.1:" + server.getPort() + "/v1",
                "--workspace",
                workspace.toString(),
                "--system-file",
                promptFile.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader("hello\n/exit\n"),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        }
        JsonNode system = backend.requests().get(0).path("messages").get(0);
        assertThat(system.path("role").asText(), is("system"));
        assertThat(system.path("content").asText(), containsString("only in haiku"));
    }

    @Test
    void aSystemFileThatIsNotThereIsAUsageError() {
        // Read at startup rather than at first use: a typo in a path is easy to miss, and a prompt long
        // enough to be worth a file is long enough that its absence should not be a surprise mid-turn.
        IllegalArgumentException thrown = org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () -> AgentOptions.parse(new String[] {
                    "--base-url",
                    "http://localhost:1/v1",
                    "--system-file",
                    workspace.resolve("gone.txt").toString()
                }));

        assertThat(thrown.getMessage(), containsString("--system-file cannot be read"));
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

    @Test
    void inProcessModelIsLoadedWithAQuietLogThresholdByDefault() {
        // llama.cpp prints its per-request INFO lines to stderr, the very console the streamed answer
        // goes to; the default threshold has to stay below INFO (3) or the two interleave again.
        List<String> args = List.of(LocalAgent.modelParameters(AgentOptions.parse(new String[] {"--model", "m.gguf"}))
                .toArray());

        assertThat(args, hasItem("--log-verbosity"));
        assertThat(
                args.get(args.indexOf("--log-verbosity") + 1), is(String.valueOf(AgentOptions.DEFAULT_LOG_VERBOSITY)));
        assertThat(AgentOptions.DEFAULT_LOG_VERBOSITY, lessThan(3));
        assertThat(args, not(hasItem("--verbose")));
    }

    @Test
    void verboseReplacesTheThresholdWithLlamaCppsOwnVerboseFlag() {
        List<String> args = List.of(LocalAgent.modelParameters(
                        AgentOptions.parse(new String[] {"--model", "m.gguf", "--log-verbosity", "1", "--verbose"}))
                .toArray());

        assertThat(args, hasItem("--verbose"));
        assertThat(args, not(hasItem("--log-verbosity")));
    }

    @Test
    void mavenJvmConfigPinsAUtf8ConsoleForExecJava() throws Exception {
        // On Windows llama.cpp's common_init() switches the console to UTF-8 after the JVM fixed its
        // stdout encoding from the old code page; exec:java runs in Maven's JVM, so the fix has to
        // live in .mvn/jvm.config (Maven reads it from the project root the user runs mvn in).
        Path jvmConfig = Path.of(".mvn", "jvm.config").toAbsolutePath();
        assertThat("expected " + jvmConfig, Files.exists(jvmConfig), is(true));
        String content = Files.readString(jvmConfig);

        assertThat(content, containsString("-Dstdout.encoding=UTF-8"));
        assertThat(content, containsString("-Dstderr.encoding=UTF-8"));
    }

    @Test
    void aToolCallStaysInTheHistorySoTheNextTurnSeesItHappened() throws Exception {
        // The failure this pins: with only user text and the model's prose in the history, a small
        // model stops calling tools after a few turns and starts DESCRIBING the work instead --
        // reporting exit codes and files that never existed. The evidence has to stay in the context.
        Files.writeString(workspace.resolve("hello.txt"), "VALUE=42\n");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        ScriptedBackend backend = new ScriptedBackend((call, request) -> switch (call) {
            case 1 -> ScriptedBackend.toolCallTurn("call_1", "read_file", "{\"file_path\":\"hello.txt\"}");
            case 2 -> ScriptedBackend.textTurn("The file says VALUE=42.");
            default -> ScriptedBackend.textTurn("Understood.");
        });
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader(
                            "what does hello.txt say?" + System.lineSeparator() + "and now?" + System.lineSeparator()),
                    new PrintStream(out, true, StandardCharsets.UTF_8),
                    new PrintStream(err, true, StandardCharsets.UTF_8));
        }

        // third request = second turn: the calls of turn one ride along with the new user message, so
        // the model sees that they happened. Not as tool_calls messages (Atmosphere's assembleMessages
        // rebuilds history as new ChatMessage(role, content) and drops the rest), and not as assistant
        // text either (the model copied that into its own answers).
        List<JsonNode> requests = backend.requests();
        assertThat(requests.size(), is(3));
        List<String> roles = new ArrayList<>();
        for (JsonNode message : requests.get(2).path("messages")) {
            roles.add(message.path("role").asText());
        }
        assertThat(
                "strict alternation keeps every chat template happy",
                roles,
                contains("system", "user", "assistant", "user"));
        String secondUserMessage =
                requests.get(2).path("messages").get(3).path("content").asText();
        assertThat(secondUserMessage, containsString("read_file"));
        assertThat(secondUserMessage, containsString("VALUE=42"));
        assertThat(secondUserMessage, containsString("do not repeat it"));
        assertThat("and the user's own words are still there", secondUserMessage, containsString("and now?"));
        assertThat(
                "the answer itself stays the model's own",
                requests.get(2).path("messages").get(2).path("content").asText(),
                is("The file says VALUE=42."));
    }

    @Test
    void theActivityLineNamesTheRunningToolSoALongBuildLooksAlive() {
        // "working… (90s)" during a two-minute mvn test is indistinguishable from a hang, so the line
        // names the tool and how long IT has been running, next to the turn's total.
        assertThat(
                LocalAgent.activityLine('x', "Fettling", 12, "run_command", 9, 2),
                is("x Fettling… (run_command 9s of 12s · 2 tool calls)"));
        assertThat(LocalAgent.activityLine('x', "Fettling", 5, null, 0, 0), is("x Fettling… (5s)"));
        assertThat(LocalAgent.activityLine('x', "Fettling", 30, null, 0, 3), is("x Fettling… (30s · 3 tool calls)"));
    }

    @Test
    void theSpinnerWordsAreOursAndHarmless() {
        List<String> words = LocalAgent.prompt(LocalAgent.SPINNER_WORDS)
                .lines()
                .map(String::strip)
                .filter(word -> !word.isEmpty())
                .toList();

        assertThat("enough variety to not repeat every other turn", words.size() > 15, is(true));
        assertThat("no duplicates", words.size(), is((int)
                words.stream().distinct().count()));
        for (String word : words) {
            assertThat(word, word.matches("[A-Z][a-z-]+"), is(true));
        }
        // Claude Code's own list is extracted from a proprietary binary and the public copies of it are
        // unlicensed or CC BY-NC-SA; none of its words may appear here.
        for (String theirs : List.of("Razzmatazzing", "Clauding", "Flibbertigibbeting", "Simmering", "Vibing")) {
            assertThat(words.contains(theirs), is(false));
        }
        assertThat(words.contains(LocalAgent.spinnerWord()), is(true));
    }

    @Test
    void theToolNoteIsAddressedToTheModelAndNotWrittenAsItsOwnWords() {
        // It first rode in front of the assistant's answer -- and the model copied it into its next
        // reply, so the user read "(tools I actually ran this turn: …)" as the first line of an answer.
        String note = LocalAgent.toolNote(
                List.of(new ConsoleSession.ToolRound("grep", "{pattern=Test}", "3 matches in 2 files")));

        assertThat(note, containsString("do not repeat it"));
        assertThat(note, containsString("grep"));
        assertThat(note, containsString("3 matches in 2 files"));
        assertThat(LocalAgent.toolNote(List.of()), is(""));
    }

    @Test
    void compactionIsDecidedBeforeTheRequestAndOnlyWhenTheWindowIsKnown() {
        AgentOptions on = AgentOptions.parse(new String[] {"--base-url", "u"});
        assertThat("on by default", on.isAutoCompact(), is(true));
        assertThat(on.getCompactAt(), is(AgentOptions.DEFAULT_COMPACT_AT));

        // 70 % of 1000 tokens: 699 still fits, 700 does not
        assertThat(LocalAgent.needsCompaction(on, 1000, 699), is(false));
        assertThat(LocalAgent.needsCompaction(on, 1000, 700), is(true));

        // an unknown window is never guessed at
        assertThat(LocalAgent.needsCompaction(on, StatusLine.UNKNOWN_CONTEXT, 1_000_000), is(false));

        AgentOptions off = AgentOptions.parse(new String[] {"--base-url", "u", "--auto-compact", "false"});
        assertThat(off.isAutoCompact(), is(false));
        assertThat(LocalAgent.needsCompaction(off, 1000, 999), is(false));

        AgentOptions early = AgentOptions.parse(new String[] {"--base-url", "u", "--compact-at", "50"});
        assertThat(LocalAgent.needsCompaction(early, 1000, 500), is(true));
        assertThat(LocalAgent.needsCompaction(early, 1000, 499), is(false));
    }

    @Test
    void aMalformedCompactionFlagIsRejectedWithItsReason() {
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class,
                                () -> AgentOptions.parse(new String[] {"--base-url", "u", "--auto-compact", "maybe"}))
                        .getMessage(),
                containsString("Expected true or false"));
        assertThat(
                org.junit.jupiter.api.Assertions.assertThrows(
                                IllegalArgumentException.class,
                                () -> AgentOptions.parse(new String[] {"--base-url", "u", "--compact-at", "99"}))
                        .getMessage(),
                containsString("between 10 and 95"));
    }

    @Test
    void compactingAnAlreadyCompactedHistoryIsRefusedInsteadOfRepeated() throws Exception {
        // A compacted history is the summary plus its acknowledgement. Summarizing that again returns
        // the same text for another model call -- and re-sends a byte-identical prompt, which llama.cpp
        // answers with "need to evaluate at least 1 token for each active slot".
        ScriptedBackend backend = new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("a summary"));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (OpenAiCompatServer server = server(backend)) {
            AgentOptions options = AgentOptions.parse(new String[] {
                "--base-url", "http://127.0.0.1:" + server.getPort() + "/v1", "--workspace", workspace.toString()
            });

            LocalAgent.run(
                    options,
                    new StringReader("hello" + System.lineSeparator() + "/compact" + System.lineSeparator() + "/compact"
                            + System.lineSeparator()),
                    new PrintStream(out, true, StandardCharsets.UTF_8),
                    new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8));
        }

        // one turn + one compaction = two requests; the second /compact must not add a third
        assertThat(backend.requests(), hasSize(2));
        assertThat(
                "the first compaction really happened",
                out.toString(StandardCharsets.UTF_8),
                containsString("compacted"));
        assertThat(out.toString(StandardCharsets.UTF_8), containsString("already a summary"));
    }
}
