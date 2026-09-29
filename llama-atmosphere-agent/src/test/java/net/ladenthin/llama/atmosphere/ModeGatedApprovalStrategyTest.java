// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalResolution;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.PendingApproval;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

/** Auto approves without asking; manual asks whoever is there right now, and nobody means no. */
class ModeGatedApprovalStrategyTest {

    private final AtomicReference<ApprovalMode> mode = new AtomicReference<>(ApprovalMode.MANUAL);
    private final AtomicInteger asked = new AtomicInteger();
    private final AtomicReference<@Nullable ApprovalStrategy> asker = new AtomicReference<>();

    private final ApprovalStrategy approving = new ApprovalStrategy() {
        @Override
        public ApprovalOutcome awaitApproval(PendingApproval approval, StreamingSession session) {
            return awaitApprovalDetailed(approval, session).outcome();
        }

        @Override
        public ApprovalResolution awaitApprovalDetailed(PendingApproval approval, StreamingSession session) {
            asked.incrementAndGet();
            return ApprovalResolution.approve();
        }
    };

    private ApprovalStrategy.ApprovalOutcome decide() {
        PendingApproval approval = new PendingApproval(
                "id-1", ShellTool.TOOL_NAME, Map.of("command", "ls"), "run it?", "session", Instant.MAX);
        return new ModeGatedApprovalStrategy(mode, asker::get).awaitApproval(approval, null);
    }

    @Test
    void autoApprovesWithoutAsking() {
        mode.set(ApprovalMode.AUTO);
        asker.set(approving);

        assertThat(decide(), is(ApprovalStrategy.ApprovalOutcome.APPROVED));
        assertThat(asked.get(), is(0));
    }

    @Test
    void manualWithNobodyToAskDenies() {
        assertThat(decide(), is(ApprovalStrategy.ApprovalOutcome.DENIED));
    }

    @Test
    void manualAsksWhoeverIsThereAtTheTimeOfTheCall() {
        assertThat(decide(), is(ApprovalStrategy.ApprovalOutcome.DENIED));
        asker.set(approving);
        assertThat(decide(), is(ApprovalStrategy.ApprovalOutcome.APPROVED));
        assertThat(asked.get(), is(1));
    }
}
