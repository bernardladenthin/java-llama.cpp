// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.json.CompletionResponseParser;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ClaudeGenerated(
        purpose = "Verify CompletionResponseParser.parseLogprobs populates TokenLogprob entries "
                + "including token id and nested top_logprobs/top_probs alternatives, "
                + "for both post-sampling (prob) and pre-sampling (logprob) modes."
)
public class TokenLogprobTest {

    private final CompletionResponseParser parser = new CompletionResponseParser();

    @Test
    public void emptyWhenAbsent() {
        LlamaOutput out = parser.parse("{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}");
        assertTrue(out.logprobs.isEmpty());
    }

    @Test
    public void parsesPostSamplingWithTopProbs() {
        String json = "{\"content\":\"hi\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"prob\":0.82,"
                + "\"top_probs\":[{\"token\":\"Hi\",\"id\":9932,\"prob\":0.10},"
                + "              {\"token\":\"Hey\",\"id\":12,\"prob\":0.05}]}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertEquals(1, out.logprobs.size());
        TokenLogprob first = out.logprobs.get(0);
        assertEquals("Hello", first.getToken());
        assertEquals(15043, first.getTokenId());
        assertEquals(0.82f, first.getLogprob(), 1e-4f);
        assertEquals(2, first.getTopLogprobs().size());
        assertEquals("Hi", first.getTopLogprobs().get(0).getToken());
        assertEquals(9932, first.getTopLogprobs().get(0).getTokenId());
        assertEquals(0.10f, first.getTopLogprobs().get(0).getLogprob(), 1e-4f);
    }

    @Test
    public void parsesPreSamplingWithTopLogprobs() {
        String json = "{\"content\":\"hi\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"logprob\":-0.20,"
                + "\"top_logprobs\":[{\"token\":\"Hi\",\"id\":9932,\"logprob\":-2.3}]}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertEquals(1, out.logprobs.size());
        TokenLogprob first = out.logprobs.get(0);
        assertEquals(-0.20f, first.getLogprob(), 1e-4f);
        assertEquals(1, first.getTopLogprobs().size());
        assertEquals(-2.3f, first.getTopLogprobs().get(0).getLogprob(), 1e-4f);
    }

    @Test
    public void preservesOrder() {
        String json = "{\"content\":\"\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"A\",\"id\":1,\"prob\":0.5},"
                + "{\"token\":\"B\",\"id\":2,\"prob\":0.4},"
                + "{\"token\":\"C\",\"id\":3,\"prob\":0.1}"
                + "]}";
        List<TokenLogprob> lp = parser.parse(json).logprobs;
        assertEquals("A", lp.get(0).getToken());
        assertEquals("B", lp.get(1).getToken());
        assertEquals("C", lp.get(2).getToken());
    }

    @Test
    public void mapAndListBothPopulated() {
        String json = "{\"content\":\"\",\"stop\":false,"
                + "\"completion_probabilities\":["
                + "{\"token\":\"hello\",\"id\":1,\"prob\":0.9}"
                + "]}";
        LlamaOutput out = parser.parse(json);
        assertEquals(1, out.logprobs.size());
        assertEquals(0.9f, out.probabilities.get("hello"), 1e-4f);
    }

    @Test
    public void backwardsCompatibleConstructor() {
        LlamaOutput out = new LlamaOutput("hi", java.util.Collections.<String, Float>emptyMap(),
                false, StopReason.NONE);
        assertNotNull(out.logprobs);
        assertTrue(out.logprobs.isEmpty());
    }
}
