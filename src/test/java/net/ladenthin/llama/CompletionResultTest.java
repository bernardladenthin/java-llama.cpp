// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.json.CompletionResponseParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ClaudeGenerated(
        purpose = "Verify CompletionResponseParser.parseCompletionResult maps the non-OAI completion JSON "
                + "(content + tokens_evaluated/predicted + timings + completion_probabilities + stop_type) "
                + "into a typed CompletionResult, and handles malformed input gracefully."
)
public class CompletionResultTest {

    private final CompletionResponseParser parser = new CompletionResponseParser();

    @Test
    public void parsesFullResponse() {
        String json = "{\"content\":\"hello world\",\"tokens_evaluated\":12,\"tokens_predicted\":5,"
                + "\"stop\":true,\"stop_type\":\"eos\","
                + "\"timings\":{\"prompt_n\":12,\"prompt_ms\":200.0,\"prompt_per_second\":60.0,"
                + "\"predicted_n\":5,\"predicted_ms\":50.0,\"predicted_per_second\":100.0,"
                + "\"cache_n\":3},"
                + "\"completion_probabilities\":["
                + "{\"token\":\"hello\",\"id\":15043,\"prob\":0.9,"
                + "\"top_probs\":[{\"token\":\"hi\",\"id\":9932,\"prob\":0.05}]}]}";

        CompletionResult r = parser.parseCompletionResult(json);
        assertEquals("hello world", r.getText());
        assertEquals(12L, r.getUsage().getPromptTokens());
        assertEquals(5L,  r.getUsage().getCompletionTokens());
        assertEquals(17L, r.getUsage().getTotalTokens());
        assertEquals(12, r.getTimings().getPromptN());
        assertEquals(3, r.getTimings().getCacheN());
        assertEquals(r.getTimings().getPredictedPerSecond(), 1e-9, 100.0);
        assertEquals(StopReason.EOS, r.getStopReason());

        assertEquals(1, r.getLogprobs().size());
        TokenLogprob lp = r.getLogprobs().get(0);
        assertEquals("hello", lp.getToken());
        assertEquals(15043, lp.getTokenId());
        assertEquals(lp.getLogprob(), 1e-4f, 0.9f);
        assertEquals(1, lp.getTopLogprobs().size());

        assertNotNull(r.getRawJson());
    }

    @Test
    public void missingFieldsDefaultToZero() {
        CompletionResult r = parser.parseCompletionResult("{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}");
        assertEquals("hi", r.getText());
        assertEquals(0L, r.getUsage().getTotalTokens());
        assertEquals(0, r.getTimings().getPromptN());
        assertTrue(r.getLogprobs().isEmpty());
        assertEquals(StopReason.EOS, r.getStopReason());
    }

    @Test
    public void stopReasonLimit() {
        CompletionResult r = parser.parseCompletionResult(
                "{\"content\":\"\",\"stop\":true,\"stop_type\":\"limit\",\"truncated\":true,"
                + "\"tokens_evaluated\":1,\"tokens_predicted\":10}");
        assertEquals(StopReason.MAX_TOKENS, r.getStopReason());
        assertEquals(10L, r.getUsage().getCompletionTokens());
    }

    @Test
    public void stopReasonStopString() {
        CompletionResult r = parser.parseCompletionResult(
                "{\"content\":\"abc\",\"stop\":true,\"stop_type\":\"word\",\"stopping_word\":\"END\"}");
        assertEquals(StopReason.STOP_STRING, r.getStopReason());
    }

    @Test
    public void malformedInputYieldsEmptyResult() {
        CompletionResult r = parser.parseCompletionResult("{not json");
        assertEquals("", r.getText());
        assertEquals(0L, r.getUsage().getTotalTokens());
        assertEquals(StopReason.NONE, r.getStopReason());
        assertTrue(r.getLogprobs().isEmpty());
    }
}
