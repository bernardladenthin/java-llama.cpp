// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.TokenUsage;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The recording half every front end shares: what a turn keeps, and that everything still reaches the
 * front end that shows it.
 */
class TurnRecorderTest {

    @TempDir
    Path workspace;

    private final RecordingFrontend frontend = new RecordingFrontend();

    private TurnRecorder recorder() {
        return new TurnRecorder(
                new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults()), frontend.renderer());
    }

    @Test
    void textIsKeptRawAndForwardedChunkByChunk() {
        TurnRecorder turn = recorder();

        turn.send("Hello ");
        turn.emit(new AiEvent.TextDelta("world"));

        assertThat(turn.text(), is("Hello world"));
        assertThat(turn.chunks(), contains("Hello ", "world"));
        assertThat(frontend.streamed, contains("Hello ", "world"));
    }

    @Test
    void toolRoundsAreRecordedAndTheEventsStillReachTheFrontEnd() {
        TurnRecorder turn = recorder();

        turn.emit(new AiEvent.ToolStart("read_file", Map.of("file_path", "a.txt")));
        assertThat(turn.runningTool(), is("read_file"));
        turn.emit(new AiEvent.ToolResult("read_file", "content"));
        turn.emit(new AiEvent.ToolStart("grep", Map.of("pattern", "x")));
        turn.emit(new AiEvent.ToolError("grep", "bad pattern"));

        assertThat(turn.toolCalls(), is(2));
        assertThat(turn.runningTool(), is((String) null));
        assertThat(turn.rounds().get(0).result(), is("content"));
        assertThat(turn.rounds().get(1).result(), is("error: bad pattern"));
        assertThat(
                frontend.events,
                contains("ToolStart read_file", "ToolResult read_file", "ToolStart grep", "ToolError grep"));
    }

    @Test
    void theEndIsRecordedAndForwardedOnce() throws Exception {
        TurnRecorder done = recorder();
        done.complete("the summary");
        assertThat(done.await(Duration.ofMillis(10)), is(true));
        assertThat("a summary stands in for an empty answer", done.text(), is("the summary"));
        assertThat(done.isClosed(), is(true));

        TurnRecorder failed = recorder();
        IllegalStateException cause = new IllegalStateException("boom");
        failed.error(cause);
        assertThat(failed.failure(), sameInstance(cause));
        assertThat(failed.hasErrored(), is(true));
        assertThat(frontend.ends, contains("complete", "error: boom"));
    }

    @Test
    void theLastReportedInputCountWinsAndTheFileSystemIsInjected() {
        TurnRecorder turn = recorder();

        turn.usage(new TokenUsage(100, 5, 0, 105, "m"));
        turn.usage(new TokenUsage(250, 5, 0, 255, "m"));
        turn.usage(new TokenUsage(0, 0, 0, 0, "m"));

        assertThat(turn.inputTokens(), is(250L));
        assertThat(turn.injectables().get(AgentFileSystem.class) instanceof WorkspaceAgentFileSystem, is(true));
        assertThat(turn.sessionId(), is("recording"));
        assertThat(List.copyOf(turn.injectables().keySet()), contains(AgentFileSystem.class));
    }
}
