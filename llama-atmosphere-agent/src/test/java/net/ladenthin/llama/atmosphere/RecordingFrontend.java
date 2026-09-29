// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.jspecify.annotations.Nullable;

/**
 * A {@link SessionFrontend} that writes everything down, for tests of {@link AgentSession} that must not
 * depend on any one real front end.
 *
 * <p>It waits for turns the way a front end without a console does ({@link AgentSession#awaitQuietly}), so
 * the session's own {@link AgentSession#cancel()} is what stops a turn — exactly the path a browser or an
 * editor takes.
 */
final class RecordingFrontend implements SessionFrontend {

    /** Every finished line, in order. */
    final List<String> lines = new CopyOnWriteArrayList<>();

    /** Every streamed text chunk of every turn. */
    final List<String> streamed = new CopyOnWriteArrayList<>();

    /** Every event the renderer saw (tool starts, results, errors), by type name. */
    final List<String> events = new CopyOnWriteArrayList<>();

    /** Every running command's output line. */
    final List<String> commandOutput = new CopyOnWriteArrayList<>();

    /** How many times each turn's renderer saw {@code complete()}. */
    final List<String> ends = new CopyOnWriteArrayList<>();

    /** The questions {@link #ask} was asked. */
    final List<String> questions = new CopyOnWriteArrayList<>();

    private final @Nullable ApprovalStrategy approvals;
    private final @Nullable String answer;
    private volatile BooleanSupplier stopped = () -> false;
    private volatile int clears;

    /**
     * A front end that answers nothing: gated calls are denied, questions go unanswered.
     */
    RecordingFrontend() {
        this(null, null);
    }

    /**
     * A front end with an approval strategy and an answer for plain questions.
     *
     * @param approvals who answers gated calls, or {@code null}
     * @param answer the answer to every {@link #ask}, or {@code null} for nobody there
     */
    RecordingFrontend(@Nullable ApprovalStrategy approvals, @Nullable String answer) {
        this.approvals = approvals;
        this.answer = answer;
    }

    /**
     * Let the session's stop request reach the waiting turn.
     *
     * @param session the session this front end drives
     * @return this front end
     */
    RecordingFrontend stoppedBy(AgentSession session) {
        this.stopped = session::stopRequested;
        return this;
    }

    int clears() {
        return clears;
    }

    String allLines() {
        return String.join("\n", lines);
    }

    @Override
    public Ansi ansi() {
        return Ansi.PLAIN;
    }

    @Override
    public void line(String text) {
        lines.add(text);
    }

    @Override
    public StreamingSession renderer() {
        return new StreamingSession() {
            @Override
            public String sessionId() {
                return "recording";
            }

            @Override
            public void send(String text) {
                streamed.add(text);
            }

            @Override
            public void sendMetadata(String key, Object value) {}

            @Override
            public void progress(String message) {}

            @Override
            public void complete() {
                ends.add("complete");
            }

            @Override
            public void complete(String summary) {
                ends.add("complete");
            }

            @Override
            public void error(Throwable t) {
                ends.add("error: " + t.getMessage());
            }

            @Override
            public boolean isClosed() {
                return false;
            }

            @Override
            public void emit(AiEvent event) {
                events.add(event.getClass().getSimpleName() + describe(event));
            }
        };
    }

    private static String describe(AiEvent event) {
        return switch (event) {
            case AiEvent.ToolStart start -> " " + start.toolName();
            case AiEvent.ToolResult result -> " " + result.toolName();
            case AiEvent.ToolError error -> " " + error.toolName();
            default -> "";
        };
    }

    @Override
    public TurnEnd await(TurnRecorder turn, ExecutionHandle handle, Supplier<String> stateLine)
            throws InterruptedException {
        return AgentSession.awaitQuietly(turn, handle, stopped);
    }

    @Override
    public @Nullable ApprovalStrategy approvals() {
        return approvals;
    }

    @Override
    public @Nullable String ask(String question) {
        questions.add(question);
        return answer;
    }

    @Override
    public void clearScreen() {
        clears++;
    }

    @Override
    public void commandOutput(String line) {
        commandOutput.add(line);
    }
}
