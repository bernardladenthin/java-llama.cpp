// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.function.Supplier;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.jspecify.annotations.Nullable;

/**
 * What an {@link AgentSession} needs from whoever shows it: a console, a browser, an editor.
 *
 * <p>The session owns everything a conversation is — the history, the approval mode, the record, the
 * tools, the commands. A front end only answers five questions, which is what makes the same conversation
 * reachable from a terminal, a browser tab and an IDE without three copies of the logic:
 *
 * <ul>
 *   <li>where a finished line of output goes ({@link #line}),
 *   <li>where a turn's streamed events go ({@link #renderer}),
 *   <li>how to wait for a running turn, and when to give up on it ({@link #await}),
 *   <li>who answers a tool call that needs approval ({@link #approvals}),
 *   <li>and who answers a plain question, if anybody can ({@link #ask}).
 * </ul>
 */
public interface SessionFrontend {

    /**
     * How this front end colours text; {@link Ansi#PLAIN} for anything that is not a terminal.
     *
     * @return the colours
     */
    Ansi ansi();

    /**
     * One finished line of output: a command's answer, a notice, an error.
     *
     * @param text the line, without a terminator
     */
    void line(String text);

    /**
     * Where the streamed events of the next turn go. Asked once per turn, so a renderer may keep state for
     * the turn (a half-written Markdown line).
     *
     * <p>The session wraps it in a {@link TurnRecorder}, which forwards every call including the final
     * {@code complete()}/{@code error()}; a front end whose connection must outlive one turn wraps its
     * connection so that those two do not close it.
     *
     * @return the session that shows the turn
     */
    StreamingSession renderer();

    /**
     * Wait for a running turn.
     *
     * @param turn the running turn
     * @param handle stops the turn; call {@link ExecutionHandle#cancel()} to cut it short
     * @param stateLine the session's state, current while the turn runs (context use, mode)
     * @return how the turn ended
     * @throws InterruptedException if interrupted while waiting
     */
    TurnEnd await(TurnRecorder turn, ExecutionHandle handle, Supplier<String> stateLine) throws InterruptedException;

    /**
     * Who answers a tool call that needs approval while the session is in {@link ApprovalMode#MANUAL} mode.
     * In {@link ApprovalMode#AUTO} mode it is never asked.
     *
     * @return the strategy, or {@code null} when nobody can answer, which denies the call
     */
    @Nullable
    ApprovalStrategy approvals();

    /**
     * Ask a question that is not about a tool call, e.g. whether {@code /loop} may switch to auto mode.
     *
     * @param question the question
     * @return the answer in lower case, or {@code null} when nobody can answer
     */
    @Nullable
    String ask(String question);

    /**
     * Clear what is on screen, for {@code /clear} and {@code /cls}. Nothing for a front end without a
     * screen of its own.
     */
    default void clearScreen() {
        // nothing to clear
    }

    /**
     * One line of a running command's output, as it arrives.
     *
     * @param line the output line
     */
    default void commandOutput(String line) {
        line(ansi().dim("  │ " + line));
    }
}
