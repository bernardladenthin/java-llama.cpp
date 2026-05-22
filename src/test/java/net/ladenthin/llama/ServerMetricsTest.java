// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ClaudeGenerated(
        purpose = "Verify ServerMetrics typed getters map all fields emitted by server_task_result_metrics::to_json, "
                + "including cumulative Usage and derived cumulative Timings."
)
public class ServerMetricsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ServerMetrics parse(String json) throws Exception {
        return new ServerMetrics(MAPPER.readTree(json));
    }

    private static final String SAMPLE =
            "{\"idle\":2,\"processing\":1,\"deferred\":3,\"t_start\":1234567890,"
                    + "\"n_prompt_tokens_processed_total\":100,\"t_prompt_processing_total\":50,"
                    + "\"n_tokens_predicted_total\":200,\"t_tokens_generation_total\":80,"
                    + "\"n_prompt_tokens_processed\":10,\"t_prompt_processing\":5,"
                    + "\"n_tokens_predicted\":20,\"t_tokens_generation\":8,"
                    + "\"n_decode_total\":300,\"n_busy_slots_total\":4,\"n_tokens_max\":4096,"
                    + "\"slots\":[{\"id\":0},{\"id\":1}]}";

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
    }

    @Test
    public void windowUsage() throws Exception {
        ServerMetrics m = parse(SAMPLE);
        Usage u = m.getWindowUsage();
        assertEquals(10L, u.getPromptTokens());
        assertEquals(20L, u.getCompletionTokens());
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
    public void missingFieldsDefaultToZero() throws Exception {
        ServerMetrics m = parse("{}");
        assertEquals(0, m.getIdleSlots());
        assertEquals(0L, m.getDecodeTotal());
        assertEquals(0, m.getTokensMax());
        assertEquals(0L, m.getCumulativeUsage().getTotalTokens());
    }
}
