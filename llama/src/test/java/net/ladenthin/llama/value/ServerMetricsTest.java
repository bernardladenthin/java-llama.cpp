// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify ServerMetrics typed getters map all fields emitted by server_task_result_metrics::to_json, "
                + "including cumulative Usage and derived cumulative Timings.")
public class ServerMetricsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ServerMetrics parse(String json) throws Exception {
        return new ServerMetrics(MAPPER.readTree(json));
    }

    private static final String SAMPLE = "{\"idle\":2,\"processing\":1,\"deferred\":3,\"t_start\":1234567890,"
            + "\"n_prompt_tokens_processed_total\":100,\"t_prompt_processing_total\":50,"
            + "\"n_tokens_predicted_total\":200,\"t_tokens_generation_total\":80,"
            + "\"n_prompt_tokens_processed\":10,\"t_prompt_processing\":5,"
            + "\"n_tokens_predicted\":20,\"t_tokens_generation\":8,"
            + "\"n_decode_total\":300,\"n_busy_slots_total\":4,\"n_tokens_max\":4096,"
            // Counters that only became reachable as JSON once the JNI layer started merging the
            // METRICS and SLOT_GET halves back together (upstream split them at b10408/b10519).
            + "\"n_prompt_tokens_cached_total\":700,\"n_draft_tokens_total\":40,"
            + "\"n_draft_accepted_total\":30,\"n_draft_verif_steps_total\":12,"
            + "\"n_accepted_per_pos_total\":[20,8,2],"
            // next_token is an ARRAY of one object — this mirrors llama.cpp's server_slot::to_json
            // at b9739, not a bare object; SlotMetrics must unwrap next_token[0].
            + "\"slots\":[{\"id\":0,\"n_ctx\":4096,\"is_processing\":true,"
            + "\"n_prompt_tokens\":100,\"n_prompt_tokens_processed\":20,"
            + "\"n_prompt_tokens_cache\":80,\"next_token\":[{\"n_decoded\":7,\"n_remain\":9}]},"
            + "{\"id\":1}]}";

    @Test
    public void slotCountsAndTimestamp() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(2, m.getIdleSlots());
        assertEquals(1, m.getProcessingSlots());
        assertEquals(3, m.getDeferredTasks());
        assertEquals(1234567890L, m.getStartTimestamp());
    }

    @Test
    public void totalsAndMaxTokens() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(300L, m.getDecodeTotal());
        assertEquals(4L, m.getBusySlotsTotal());
        assertEquals(4096, m.getTokensMax());
    }

    @Test
    public void cumulativeUsage() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Usage u = m.getCumulativeUsage();
        assertEquals(100L, u.getPromptTokens());
        assertEquals(200L, u.getCompletionTokens());
        assertEquals(300L, u.getTotalTokens());
        assertEquals(100L, m.getCumulativeProcessedPromptTokens());
        assertEquals(200L, m.getCumulativeGeneratedTokens());
    }

    @Test
    public void windowUsage() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Usage u = m.getWindowUsage();
        assertEquals(10L, u.getPromptTokens());
        assertEquals(20L, u.getCompletionTokens());
        assertEquals(10L, m.getWindowProcessedPromptTokens());
    }

    @Test
    public void cumulativeTimingsDerivesRates() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Timings t = m.getCumulativeTimings();
        assertEquals(100, t.getPromptN());
        assertEquals(50.0, t.getPromptMs(), 1e-9);
        // 100 tokens / 50ms = 2000/s
        assertEquals(2000.0, t.getPromptPerSecond(), 1e-9);
        // 200 / 80ms = 2500/s
        assertEquals(2500.0, t.getPredictedPerSecond(), 1e-9);
    }

    @Test
    public void cumulativeTimingsDoNotOverflowIntOnBusyServer() throws Exception {
        // A long-running server easily exceeds Integer.MAX_VALUE tokens; an int cast would wrap
        // these cumulative counters to negative values.
        ServerMetrics m =
                parse("{\"n_prompt_tokens_processed_total\":3000000000," + "\"n_tokens_predicted_total\":4000000000}");
        Timings t = m.getCumulativeTimings();
        assertEquals(3_000_000_000L, t.getPromptN());
        assertEquals(4_000_000_000L, t.getPredictedN());
    }

    @Test
    public void cumulativeTimingsZeroMsYieldsZeroRate() throws Exception {
        ServerMetrics m = parse("{\"n_prompt_tokens_processed_total\":5,\"t_prompt_processing_total\":0}");
        assertEquals(0.0, m.getCumulativeTimings().getPromptPerSecond(), 1e-9);
    }

    @Test
    public void slotsArrayExposed() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertTrue(m.getSlots().isArray());
        assertEquals(2, m.getSlots().size());
    }

    @Test
    public void typedSlotMetricsExposeCacheCounts() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(2, m.getSlotMetrics().size());
        SlotMetrics slot = m.getSlotMetrics().get(0);
        assertEquals(0, slot.getId());
        assertEquals(4096, slot.getContextSize());
        assertTrue(slot.isProcessing());
        assertEquals(100L, slot.getPromptTokens());
        assertEquals(20L, slot.getProcessedPromptTokens());
        assertEquals(80L, slot.getCachedPromptTokens());
        assertEquals(7L, slot.getDecodedTokens());
        assertEquals(9L, slot.getRemainingTokens());
        assertEquals(0, slot.asJson().path("id").asInt());
        assertTrue(slot.toString().contains("n_prompt_tokens_cache"));

        // Assert against the SECOND slot (id=1, no is_processing) so the id and is_processing
        // accessors are pinned to non-default values a constant-return mutant cannot satisfy:
        // slot 0's id==0 and is_processing==true coincide with the mutated constants.
        SlotMetrics idle = m.getSlotMetrics().get(1);
        assertEquals(1, idle.getId());
        assertFalse(idle.isProcessing());
        // next_token absent on the idle slot — accessors fall back to zero, not throw.
        assertEquals(0L, idle.getDecodedTokens());
        assertEquals(0L, idle.getRemainingTokens());
    }

    @Test
    public void missingFieldsDefaultToZero() throws Exception {
        ServerMetrics m = parse("{}");
        assertEquals(0, m.getIdleSlots());
        assertEquals(0L, m.getDecodeTotal());
        assertEquals(0, m.getTokensMax());
        assertEquals(0L, m.getCumulativeUsage().getTotalTokens());
    }

    @Test
    public void cumulativeTimingsZeroPredictedMsYieldsZeroRate() throws Exception {
        // Pins the predictedMs > 0.0 boundary: with predictedN>0 but predictedMs=0 the rate must be 0.0
        // (a >= boundary mutant would divide by zero and produce a non-zero / NaN rate).
        ServerMetrics m = parse("{\"n_tokens_predicted_total\":5,\"t_tokens_generation_total\":0}");
        assertEquals(0.0, m.getCumulativeTimings().getPredictedPerSecond(), 1e-9);
    }

    @Test
    public void asJsonExposesBackingNode() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // Dereferencing the returned node kills the "return null" mutant on asJson().
        assertEquals(2, m.asJson().get("idle").asInt());
    }

    @Test
    public void toStringSerializesNode() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        // Assert content (not just non-null) so the empty-string return mutant on toString is killed.
        assertTrue(m.toString().contains("idle"));
    }

    @Test
    public void cachedPromptTokenTotalIsSeparateFromTheProcessedTotal() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(700L, m.getCumulativeCachedPromptTokens());
        // The processed total counts only tokens actually evaluated, so the two must not alias.
        assertEquals(100L, m.getCumulativeProcessedPromptTokens());
    }

    @Test
    public void speculativeDecodingCounters() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        assertEquals(40L, m.getDraftTokensTotal());
        assertEquals(30L, m.getDraftAcceptedTotal());
        assertEquals(12L, m.getDraftVerifyStepsTotal());
        assertEquals(0.75, m.getDraftAcceptanceRate(), 1e-9);
    }

    @Test
    public void draftAcceptanceRateIsZeroWhenNothingWasDrafted() throws Exception {
        // Pins the drafted > 0L guard: without it this divides by zero and yields NaN.
        ServerMetrics m = parse("{\"n_draft_accepted_total\":5,\"n_draft_tokens_total\":0}");
        assertEquals(0.0, m.getDraftAcceptanceRate(), 1e-9);
    }

    @Test
    public void draftAcceptedPerPositionIsExposedInOrder() throws Exception {
        List<Long> perPos = parse(SAMPLE).getDraftAcceptedPerPosition();
        assertEquals(3, perPos.size());
        assertEquals(20L, perPos.get(0).longValue());
        assertEquals(8L, perPos.get(1).longValue());
        assertEquals(2L, perPos.get(2).longValue());
    }

    @Test
    public void draftAcceptedPerPositionIsEmptyWhenAbsentOrNotAnArray() throws Exception {
        assertTrue(parse("{}").getDraftAcceptedPerPosition().isEmpty());
        assertTrue(parse("{\"n_accepted_per_pos_total\":7}")
                .getDraftAcceptedPerPosition()
                .isEmpty());
    }

    @Test
    public void newCountersDefaultToZeroWhenAbsent() throws Exception {
        ServerMetrics m = parse("{}");
        assertEquals(0L, m.getCumulativeCachedPromptTokens());
        assertEquals(0L, m.getDraftTokensTotal());
        assertEquals(0L, m.getDraftAcceptedTotal());
        assertEquals(0L, m.getDraftVerifyStepsTotal());
    }

    @Test
    public void windowTimingsReadTheWindowKeysNotTheTotals() throws Exception {
        // The fixture deliberately gives the window and the total different values, so a getter
        // wired to the wrong half is visible rather than coincidentally equal.
        ServerMetrics m = parse(SAMPLE);
        assertEquals(5.0, m.getWindowPromptProcessingMillis(), 1e-9);
        assertEquals(8.0, m.getWindowTokenGenerationMillis(), 1e-9);
    }

    @Test
    public void windowTimingsDeriveRatesFromTheWindowCounts() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Timings t = m.getWindowTimings();
        assertEquals(10L, t.getPromptN());
        assertEquals(5.0, t.getPromptMs(), 1e-9);
        assertEquals(10 * 1000.0 / 5.0, t.getPromptPerSecond(), 1e-9);
        assertEquals(20L, t.getPredictedN());
        assertEquals(8.0, t.getPredictedMs(), 1e-9);
        assertEquals(20 * 1000.0 / 8.0, t.getPredictedPerSecond(), 1e-9);
    }

    @Test
    public void windowTimingsAreZeroRatherThanInfiniteWhenNothingRanYet() throws Exception {
        ServerMetrics m = parse("{}");
        assertEquals(0.0, m.getWindowPromptProcessingMillis(), 1e-9);
        assertEquals(0.0, m.getWindowTokenGenerationMillis(), 1e-9);
        Timings t = m.getWindowTimings();
        assertEquals(0.0, t.getPromptPerSecond(), 1e-9);
        assertEquals(0.0, t.getPredictedPerSecond(), 1e-9);
    }
}
