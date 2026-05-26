// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ClaudeGenerated(
        purpose = "Verify Timings.fromJson maps every result_timings field and treats missing nodes as zero."
)
public class TimingsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    public void parsesAllBaseFields() throws Exception {
        String json = "{\"cache_n\":7,\"prompt_n\":10,\"prompt_ms\":200.0,"
                + "\"prompt_per_second\":50.0,\"predicted_n\":5,\"predicted_ms\":100.0,"
                + "\"predicted_per_second\":50.0}";
        Timings t = Timings.fromJson(MAPPER.readTree(json));
        assertEquals(7, t.getCacheN());
        assertEquals(10, t.getPromptN());
        assertEquals(t.getPromptMs(), 1e-9, 200.0);
        assertEquals(t.getPromptPerSecond(), 1e-9, 50.0);
        assertEquals(5, t.getPredictedN());
        assertEquals(t.getPredictedMs(), 1e-9, 100.0);
        assertEquals(t.getPredictedPerSecond(), 1e-9, 50.0);
        assertEquals(0, t.getDraftN());
        assertEquals(0, t.getDraftNAccepted());
    }

    @Test
    public void parsesSpeculativeFields() throws Exception {
        String json = "{\"prompt_n\":1,\"predicted_n\":1,\"draft_n\":50,\"draft_n_accepted\":35}";
        Timings t = Timings.fromJson(MAPPER.readTree(json));
        assertEquals(50, t.getDraftN());
        assertEquals(35, t.getDraftNAccepted());
    }

    @Test
    public void missingNodeYieldsAllZeroes() {
        Timings t = Timings.fromJson(null);
        assertEquals(0, t.getPromptN());
        assertEquals(t.getPromptMs(), 1e-9, 0.0);
        assertEquals(0, t.getDraftN());
    }

    @Test
    public void missingFieldsDefaultToZero() throws Exception {
        Timings t = Timings.fromJson(MAPPER.readTree("{}"));
        assertEquals(0, t.getCacheN());
        assertEquals(t.getPredictedPerSecond(), 1e-9, 0.0);
    }
}
