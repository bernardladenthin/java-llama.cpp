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
 * A session shown on a console — the plain line-oriented one or the JLine one, which differ only in the
 * {@link AgentTerminal} underneath.
 *
 * <p>While a turn runs it keeps the activity line moving and watches for typed input, which cuts the turn
 * short (see {@link LocalAgent#awaitWithActivity}).
 */
public final class ConsoleFrontend implements SessionFrontend {

    private final AgentTerminal terminal;
    private final TurnActivity activity;
    private final @Nullable ApprovalStrategy approvals;

    /**
     * Show a session on {@code terminal}.
     *
     * @param terminal the console
     * @param activity paused while an approval question is open
     * @param approvals who answers a gated tool call in manual mode, or {@code null} to deny them
     */
    public ConsoleFrontend(AgentTerminal terminal, TurnActivity activity, @Nullable ApprovalStrategy approvals) {
        this.terminal = terminal;
        this.activity = activity;
        this.approvals = approvals;
    }

    @Override
    public Ansi ansi() {
        return terminal.ansi();
    }

    @Override
    public void line(String text) {
        terminal.line(text);
    }

    @Override
    public StreamingSession renderer() {
        return new ConsoleRenderer(terminal);
    }

    @Override
    public TurnEnd await(TurnRecorder turn, ExecutionHandle handle, Supplier<String> stateLine)
            throws InterruptedException {
        return LocalAgent.awaitWithActivity(turn, terminal, stateLine, activity, handle);
    }

    @Override
    public @Nullable ApprovalStrategy approvals() {
        return approvals;
    }

    @Override
    public @Nullable String ask(String question) {
        return terminal.readKey(question);
    }

    @Override
    public void clearScreen() {
        terminal.clearScreen();
    }
}
