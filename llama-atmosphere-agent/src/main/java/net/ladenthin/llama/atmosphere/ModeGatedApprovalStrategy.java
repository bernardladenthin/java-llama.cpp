// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalResolution;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.PendingApproval;
import org.jspecify.annotations.Nullable;

/**
 * The one approval strategy an {@link AgentSession}'s runner carries: auto mode approves, manual mode asks
 * whichever front end is running the turn.
 *
 * <p>The runner is built once, but the front end changes: a conversation started in a browser can be
 * continued in the same browser after a reconnect, and each connection answers approvals in its own way —
 * a console question, buttons, an editor dialog. So the question of <em>who</em> to ask is looked up per
 * call rather than fixed when the runner is built.
 *
 * <p>Nobody to ask means no: a gated call is denied, never silently allowed. That keeps the direction
 * Atmosphere itself takes when no strategy is wired, and an unattended run must not be the most permissive
 * one.
 */
public final class ModeGatedApprovalStrategy implements ApprovalStrategy {

    private final AtomicReference<ApprovalMode> mode;
    private final Supplier<@Nullable ApprovalStrategy> asker;

    /**
     * Create the strategy.
     *
     * @param mode the session's approval mode, read per call
     * @param asker who to ask in manual mode, looked up per call; {@code null} denies
     */
    public ModeGatedApprovalStrategy(AtomicReference<ApprovalMode> mode, Supplier<@Nullable ApprovalStrategy> asker) {
        this.mode = mode;
        this.asker = asker;
    }

    @Override
    public ApprovalOutcome awaitApproval(PendingApproval approval, StreamingSession session) {
        return awaitApprovalDetailed(approval, session).outcome();
    }

    @Override
    public ApprovalResolution awaitApprovalDetailed(PendingApproval approval, StreamingSession session) {
        if (mode.get() == ApprovalMode.AUTO) {
            return ApprovalResolution.approve();
        }
        ApprovalStrategy delegate = asker.get();
        return delegate == null ? ApprovalResolution.deny() : delegate.awaitApprovalDetailed(approval, session);
    }
}
