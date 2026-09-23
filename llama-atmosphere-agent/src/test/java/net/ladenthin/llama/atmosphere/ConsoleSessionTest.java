// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * What the console is handed for one turn, with a terminal that only records.
 *
 * <p>The rule every test here defends: <b>one call to the terminal is one screen line</b>. The pinned
 * block at the bottom is sized in lines, so a single "line" carrying ten newlines pushes the screen
 * ten rows further than the terminal accounted for and the block is drawn across the output — which is
 * what a {@code write_file} call with a whole file in its arguments did.
 */
class ConsoleSessionTest {

    @TempDir
    Path workspace;

    /** A terminal that records what it was told to print. */
    private static final class RecordingTerminal implements AgentTerminal {
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

    private final RecordingTerminal terminal = new RecordingTerminal();

    private ConsoleSession session() {
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        return new ConsoleSession(terminal, fs);
    }

    private void assertEveryLineIsOneLine() {
        for (String line : terminal.lines) {
            assertThat("a printed line must not contain a newline: " + line, line.contains("\n"), is(false));
            assertThat(line.contains("\r"), is(false));
        }
    }

    @Test
    void aToolCallWithAWholeFileInItsArgumentsStaysOnOneLine() {
        String fileContent = "# Project Summary\n\n## Build\n\nmvn package\n".repeat(20);
        ConsoleSession session = session();

        session.emit(new AiEvent.ToolStart("write_file", Map.of("path", "project_summary.md", "content", fileContent)));

        assertEveryLineIsOneLine();
        assertThat(terminal.lines, is(not(List.of())));
        assertThat(terminal.lines.get(0), containsString("write_file"));
        assertThat(terminal.lines.get(0), containsString("project_summary.md"));
        assertThat("and it is cut, not merely joined", terminal.lines.get(0).length() < 400, is(true));
    }

    @Test
    void aMultiLineToolResultStaysOnOneLineToo() {
        ConsoleSession session = session();

        session.emit(new AiEvent.ToolStart("run_command", Map.of("command", "mvn -version")));
        session.emit(new AiEvent.ToolResult("run_command", "exit code: 0\nline one\r\nline two\n"));
        session.emit(new AiEvent.ToolError("run_command", "boom\nand more"));

        assertEveryLineIsOneLine();
    }

    @Test
    void theContextFigureGrowsWhileTheTurnRuns() {
        // it used to be rendered once before the turn and handed over as a fixed string, so it stood
        // still through every tool round and only moved at the next prompt
        ConsoleSession session = session();
        assertThat(LocalAgent.liveTokens(1000, session), is(1000L));

        session.send("a".repeat(400));
        session.emit(new AiEvent.ToolStart("read_file", Map.of("file_path", "x")));
        session.emit(new AiEvent.ToolResult("read_file", "b".repeat(4000)));

        assertThat(
                "the tool output counts too, it is in the next call's prompt",
                LocalAgent.liveTokens(1000, session) > 2000L,
                is(true));
    }

    @Test
    void aReportedCountWinsOverTheEstimate() {
        ConsoleSession session = session();
        session.send("a".repeat(4000));
        session.usage(new org.atmosphere.ai.TokenUsage(7777, 10, 0, 7787, "m"));

        assertThat(LocalAgent.liveTokens(1000, session), is(7777L));
    }

    @Test
    void theRecordedRoundKeepsTheFullArgumentsEvenThoughTheConsoleShowsLess() {
        String fileContent = "a".repeat(5000);
        ConsoleSession session = session();

        session.emit(new AiEvent.ToolStart("write_file", Map.of("content", fileContent)));

        assertThat("the console is cut …", terminal.lines.get(0).length() < 400, is(true));
        assertThat(
                "… but what the model is told later is not cut here",
                session.rounds().get(0).argumentsJson().length() > 4000,
                is(true));
    }
}
