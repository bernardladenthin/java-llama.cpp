// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.value.CompletionResult;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.StopReason;
import net.ladenthin.llama.value.TokenLogprob;
import nl.altindag.log.LogCaptor;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link CompletionResponseParser}.
 *
 * <p>All tests use JSON string literals — no JVM native library or model file is needed.
 * This mirrors the pattern established by {@code test_json_helpers.cpp} on the C++ side.
 */
public class CompletionResponseParserTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final CompletionResponseParser parser = new CompletionResponseParser();

    // ------------------------------------------------------------------
    // parse(String)
    // ------------------------------------------------------------------

    @Test
    public void testParseString_text() throws Exception {
        String json = "{\"content\":\"Hello world\",\"stop\":false}";
        LlamaOutput out = parser.parse(json);
        assertEquals("Hello world", out.text);
    }

    @Test
    public void testParseString_stopFalse() {
        String json = "{\"content\":\"partial\",\"stop\":false}";
        LlamaOutput out = parser.parse(json);
        assertFalse(out.stop);
        assertEquals(StopReason.NONE, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueEos() {
        String json = "{\"content\":\"done\",\"stop\":true,\"stop_type\":\"eos\"}";
        LlamaOutput out = parser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.EOS, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueWord() {
        String json = "{\"content\":\"end\",\"stop\":true,\"stop_type\":\"word\",\"stopping_word\":\"END\"}";
        LlamaOutput out = parser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.STOP_STRING, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueLimit() {
        String json = "{\"content\":\"truncated\",\"stop\":true,\"stop_type\":\"limit\",\"truncated\":true}";
        LlamaOutput out = parser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.MAX_TOKENS, out.stopReason);
    }

    @Test
    public void testParseString_malformedReturnsEmptyNonStop() {
        LlamaOutput out = parser.parse("{not valid json");
        assertEquals("", out.text);
        assertFalse(out.stop);
        assertEquals(StopReason.NONE, out.stopReason);
        assertTrue(out.probabilities.isEmpty());
    }

    @Test
    public void testParseString_escapedContent() {
        String json = "{\"content\":\"line1\\nline2\\t\\\"quoted\\\"\",\"stop\":false}";
        LlamaOutput out = parser.parse(json);
        assertEquals("line1\nline2\t\"quoted\"", out.text);
    }

    @Test
    public void testParseString_unicodeEscape() {
        String json = "{\"content\":\"caf\\u00e9\",\"stop\":false}";
        LlamaOutput out = parser.parse(json);
        assertEquals("café", out.text);
    }

    @Test
    public void testParseString_emptyContent() {
        String json = "{\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}";
        LlamaOutput out = parser.parse(json);
        assertEquals("", out.text);
        assertTrue(out.stop);
    }

    // ------------------------------------------------------------------
    // parse(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testParseNode_delegatesCorrectly() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}");
        LlamaOutput out = parser.parse(node);
        assertEquals("hi", out.text);
        assertTrue(out.stop);
        assertEquals(StopReason.EOS, out.stopReason);
    }

    // ------------------------------------------------------------------
    // extractContent
    // ------------------------------------------------------------------

    @Test
    public void testExtractContent_present() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hello\",\"stop\":false}");
        assertEquals("hello", parser.extractContent(node));
    }

    @Test
    public void testExtractContent_absent() throws Exception {
        JsonNode node = MAPPER.readTree("{\"stop\":false}");
        assertEquals("", parser.extractContent(node));
    }

    @Test
    public void testExtractContent_empty() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"\",\"stop\":true}");
        assertEquals("", parser.extractContent(node));
    }

    // ------------------------------------------------------------------
    // parseProbabilities
    // ------------------------------------------------------------------

    @Test
    public void testParseProbabilities_absentKey() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true}");
        assertTrue(parser.parseProbabilities(node).isEmpty());
    }

    @Test
    public void testParseProbabilities_emptyArray() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true,\"completion_probabilities\":[]}");
        assertTrue(parser.parseProbabilities(node).isEmpty());
    }

    @Test
    public void testParseProbabilities_postSampling() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"prob\":0.82,"
                + "\"top_probs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"prob\":0.1}]},"
                + "{\"token\":\" world\",\"bytes\":[32,119],\"id\":1917,\"prob\":0.65,"
                + "\"top_probs\":[]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = parser.parseProbabilities(node);
        assertEquals(2, probs.size());
        assertEquals(0.82f, probs.get("Hello"), 0.001f);
        assertEquals(0.65f, probs.get(" world"), 0.001f);
    }

    @Test
    public void testParseProbabilities_preSampling() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," + "\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"logprob\":-0.2,"
                + "\"top_logprobs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"logprob\":-2.3}]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = parser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertEquals(-0.2f, probs.get("Hello"), 0.001f);
    }

    @Test
    public void testParseProbabilities_escapedToken() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," + "\"completion_probabilities\":["
                + "{\"token\":\"say \\\"yes\\\"\",\"bytes\":[],\"id\":1,\"prob\":0.5,"
                + "\"top_probs\":[]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = parser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertEquals(0.5f, probs.get("say \"yes\""), 0.001f);
    }

    @Test
    public void testParseProbabilities_topProbs_notIncluded() throws Exception {
        // top_probs entries must NOT appear in the outer map — only the outer token/prob
        String json = "{\"content\":\"hi\",\"stop\":true," + "\"completion_probabilities\":["
                + "{\"token\":\"A\",\"bytes\":[],\"id\":1,\"prob\":0.9,"
                + "\"top_probs\":[{\"token\":\"B\",\"bytes\":[],\"id\":2,\"prob\":0.05}]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = parser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertTrue(probs.containsKey("A"), "only outer token 'A' should be present");
        assertFalse(probs.containsKey("B"), "inner top_probs token 'B' must not appear");
    }

    // ------------------------------------------------------------------
    // parseLogprobs — typed per-token entries
    // ------------------------------------------------------------------

    @Test
    public void testParseLogprobs_postSamplingWithNestedTopProbs() throws Exception {
        String json = "{\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"prob\":0.82,"
                + "\"top_probs\":[{\"token\":\"Hi\",\"id\":9932,\"prob\":0.1}]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        List<TokenLogprob> lp = parser.parseLogprobs(node);

        assertEquals(1, lp.size());
        TokenLogprob e = lp.get(0);
        assertEquals("Hello", e.getToken());
        assertEquals(15043, e.getTokenId());
        assertEquals(0.82f, e.getLogprob(), 0.001f);
        // Nested alternatives are parsed recursively from top_probs.
        assertEquals(1, e.getTopLogprobs().size());
        assertEquals("Hi", e.getTopLogprobs().get(0).getToken());
        assertEquals(9932, e.getTopLogprobs().get(0).getTokenId());
        assertEquals(0.1f, e.getTopLogprobs().get(0).getLogprob(), 0.001f);
    }

    @Test
    public void testParseLogprobs_preSamplingUsesLogprobAndTopLogprobs() throws Exception {
        // No "prob"/"top_probs" — the parser falls back to "logprob"/"top_logprobs".
        String json = "{\"completion_probabilities\":["
                + "{\"token\":\"Hello\",\"id\":15043,\"logprob\":-0.2,"
                + "\"top_logprobs\":[{\"token\":\"Hi\",\"id\":9932,\"logprob\":-2.3}]}"
                + "]}";
        JsonNode node = MAPPER.readTree(json);
        List<TokenLogprob> lp = parser.parseLogprobs(node);

        assertEquals(1, lp.size());
        TokenLogprob e = lp.get(0);
        assertEquals(-0.2f, e.getLogprob(), 0.001f);
        assertEquals(1, e.getTopLogprobs().size());
        assertEquals("Hi", e.getTopLogprobs().get(0).getToken());
        assertEquals(-2.3f, e.getTopLogprobs().get(0).getLogprob(), 0.001f);
    }

    @Test
    public void testParseLogprobs_entryWithoutAlternatives_hasEmptyTopLogprobs() throws Exception {
        String json = "{\"completion_probabilities\":[" + "{\"token\":\"x\",\"id\":1,\"prob\":0.5}" + "]}";
        JsonNode node = MAPPER.readTree(json);
        List<TokenLogprob> lp = parser.parseLogprobs(node);

        assertEquals(1, lp.size());
        assertEquals(1, lp.get(0).getTokenId());
        assertTrue(lp.get(0).getTopLogprobs().isEmpty(), "no top_probs/top_logprobs → empty alternatives");
    }

    @Test
    public void testParseLogprobs_missingId_defaultsToMinusOne() throws Exception {
        String json = "{\"completion_probabilities\":[" + "{\"token\":\"x\",\"prob\":0.5}" + "]}";
        JsonNode node = MAPPER.readTree(json);
        List<TokenLogprob> lp = parser.parseLogprobs(node);
        assertEquals(-1, lp.get(0).getTokenId());
    }

    @Test
    public void testParseLogprobs_absentArray_returnsMutableEmptyList() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true}");
        List<TokenLogprob> lp = parser.parseLogprobs(node);
        assertTrue(lp.isEmpty());
        // Documented to be a mutable empty list — adding must not throw
        // (kills the immutable-emptyList() return mutant).
        lp.add(new TokenLogprob("x", 1, 0.5f, Collections.<TokenLogprob>emptyList()));
        assertEquals(1, lp.size());
    }

    @Test
    public void testParseLogprobs_emptyArray_returnsEmptyList() throws Exception {
        JsonNode node = MAPPER.readTree("{\"completion_probabilities\":[]}");
        assertTrue(parser.parseLogprobs(node).isEmpty());
    }

    // ------------------------------------------------------------------
    // parseCompletionResult(String) — non-streaming typed result
    // ------------------------------------------------------------------

    @Test
    public void testParseCompletionResult_fullResult() {
        String json = "{\"content\":\"final answer\","
                + "\"tokens_evaluated\":11,\"tokens_predicted\":4,"
                + "\"timings\":{\"cache_n\":8},"
                + "\"stop_type\":\"eos\","
                + "\"completion_probabilities\":[{\"token\":\"final\",\"id\":1,\"prob\":0.7}]}";
        CompletionResult r = parser.parseCompletionResult(json);

        assertEquals("final answer", r.getText());
        assertEquals(11L, r.getUsage().getPromptTokens());
        assertEquals(4L, r.getUsage().getCompletionTokens());
        assertEquals(8L, r.getUsage().getCachedTokens());
        assertEquals(3L, r.getUsage().getProcessedPromptTokens());
        assertEquals(StopReason.EOS, r.getStopReason());
        assertEquals(1, r.getLogprobs().size());
        assertEquals("final", r.getLogprobs().get(0).getToken());
        assertEquals(json, r.getRawJson());
    }

    @Test
    public void testParseCompletionResult_limitStopType() {
        String json = "{\"content\":\"trunc\",\"tokens_evaluated\":2,\"tokens_predicted\":8,\"stop_type\":\"limit\"}";
        CompletionResult r = parser.parseCompletionResult(json);
        assertEquals(StopReason.MAX_TOKENS, r.getStopReason());
        assertTrue(r.getLogprobs().isEmpty());
    }

    @Test
    public void testParseCompletionResult_malformedJson_returnsEmptyResultPreservingRawJson() {
        String bad = "{not valid json";
        CompletionResult r = parser.parseCompletionResult(bad);
        assertEquals("", r.getText());
        assertEquals(0L, r.getUsage().getPromptTokens());
        assertEquals(0L, r.getUsage().getCompletionTokens());
        assertEquals(StopReason.NONE, r.getStopReason());
        assertTrue(r.getLogprobs().isEmpty());
        assertEquals(bad, r.getRawJson());
    }

    /**
     * Parsing a completion result carrying real timings must emit exactly one
     * per-run timing line — pins the {@code TimingsLogger.log(...)} side-effect.
     */
    @Test
    public void testParseCompletionResult_emitsTimingLine() {
        String json = "{\"content\":\"done\",\"tokens_evaluated\":7,\"tokens_predicted\":3,\"stop_type\":\"eos\","
                + "\"timings\":{\"prompt_n\":7,\"prompt_ms\":10.0,\"prompt_per_second\":700.0,"
                + "\"predicted_n\":3,\"predicted_ms\":20.0,\"predicted_per_second\":150.0}}";

        try (LogCaptor captor = LogCaptor.forName(TimingsLogger.LOGGER_NAME)) {
            CompletionResult r = parser.parseCompletionResult(json);
            assertEquals(7, r.getTimings().getPromptN());
            assertEquals(1, captor.getInfoLogs().size(), "exactly one timing line must be emitted");
        }
    }
}
