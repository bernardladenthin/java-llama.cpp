package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.StopReason;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link CompletionResponseParser}.
 *
 * <p>All tests use JSON string literals — no JVM native library or model file is needed.
 * This mirrors the pattern established by {@code test_json_helpers.cpp} on the C++ side.
 */
public class CompletionResponseParserTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ------------------------------------------------------------------
    // parse(String)
    // ------------------------------------------------------------------

    @Test
    public void testParseString_text() throws Exception {
        String json = "{\"content\":\"Hello world\",\"stop\":false}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertEquals("Hello world", out.text);
    }

    @Test
    public void testParseString_stopFalse() {
        String json = "{\"content\":\"partial\",\"stop\":false}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertFalse(out.stop);
        assertEquals(StopReason.NONE, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueEos() {
        String json = "{\"content\":\"done\",\"stop\":true,\"stop_type\":\"eos\"}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.EOS, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueWord() {
        String json = "{\"content\":\"end\",\"stop\":true,\"stop_type\":\"word\",\"stopping_word\":\"END\"}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.STOP_STRING, out.stopReason);
    }

    @Test
    public void testParseString_stopTrueLimit() {
        String json = "{\"content\":\"truncated\",\"stop\":true,\"stop_type\":\"limit\",\"truncated\":true}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertTrue(out.stop);
        assertEquals(StopReason.MAX_TOKENS, out.stopReason);
    }

    @Test
    public void testParseString_malformedReturnsEmptyNonStop() {
        LlamaOutput out = CompletionResponseParser.parse("{not valid json");
        assertEquals("", out.text);
        assertFalse(out.stop);
        assertEquals(StopReason.NONE, out.stopReason);
        assertTrue(out.probabilities.isEmpty());
    }

    @Test
    public void testParseString_escapedContent() {
        String json = "{\"content\":\"line1\\nline2\\t\\\"quoted\\\"\",\"stop\":false}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertEquals("line1\nline2\t\"quoted\"", out.text);
    }

    @Test
    public void testParseString_unicodeEscape() {
        String json = "{\"content\":\"caf\\u00e9\",\"stop\":false}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertEquals("café", out.text);
    }

    @Test
    public void testParseString_emptyContent() {
        String json = "{\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}";
        LlamaOutput out = CompletionResponseParser.parse(json);
        assertEquals("", out.text);
        assertTrue(out.stop);
    }

    // ------------------------------------------------------------------
    // parse(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testParseNode_delegatesCorrectly() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}");
        LlamaOutput out = CompletionResponseParser.parse(node);
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
        assertEquals("hello", CompletionResponseParser.extractContent(node));
    }

    @Test
    public void testExtractContent_absent() throws Exception {
        JsonNode node = MAPPER.readTree("{\"stop\":false}");
        assertEquals("", CompletionResponseParser.extractContent(node));
    }

    @Test
    public void testExtractContent_empty() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"\",\"stop\":true}");
        assertEquals("", CompletionResponseParser.extractContent(node));
    }

    // ------------------------------------------------------------------
    // parseProbabilities
    // ------------------------------------------------------------------

    @Test
    public void testParseProbabilities_absentKey() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true}");
        assertTrue(CompletionResponseParser.parseProbabilities(node).isEmpty());
    }

    @Test
    public void testParseProbabilities_emptyArray() throws Exception {
        JsonNode node = MAPPER.readTree("{\"content\":\"hi\",\"stop\":true,\"completion_probabilities\":[]}");
        assertTrue(CompletionResponseParser.parseProbabilities(node).isEmpty());
    }

    @Test
    public void testParseProbabilities_postSampling() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," +
                "\"completion_probabilities\":[" +
                "{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"prob\":0.82," +
                "\"top_probs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"prob\":0.1}]}," +
                "{\"token\":\" world\",\"bytes\":[32,119],\"id\":1917,\"prob\":0.65," +
                "\"top_probs\":[]}" +
                "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = CompletionResponseParser.parseProbabilities(node);
        assertEquals(2, probs.size());
        assertEquals(0.82f, probs.get("Hello"), 0.001f);
        assertEquals(0.65f, probs.get(" world"), 0.001f);
    }

    @Test
    public void testParseProbabilities_preSampling() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," +
                "\"completion_probabilities\":[" +
                "{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"logprob\":-0.2," +
                "\"top_logprobs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"logprob\":-2.3}]}" +
                "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = CompletionResponseParser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertEquals(-0.2f, probs.get("Hello"), 0.001f);
    }

    @Test
    public void testParseProbabilities_escapedToken() throws Exception {
        String json = "{\"content\":\"hi\",\"stop\":true," +
                "\"completion_probabilities\":[" +
                "{\"token\":\"say \\\"yes\\\"\",\"bytes\":[],\"id\":1,\"prob\":0.5," +
                "\"top_probs\":[]}" +
                "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = CompletionResponseParser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertEquals(0.5f, probs.get("say \"yes\""), 0.001f);
    }

    @Test
    public void testParseProbabilities_topProbs_notIncluded() throws Exception {
        // top_probs entries must NOT appear in the outer map — only the outer token/prob
        String json = "{\"content\":\"hi\",\"stop\":true," +
                "\"completion_probabilities\":[" +
                "{\"token\":\"A\",\"bytes\":[],\"id\":1,\"prob\":0.9," +
                "\"top_probs\":[{\"token\":\"B\",\"bytes\":[],\"id\":2,\"prob\":0.05}]}" +
                "]}";
        JsonNode node = MAPPER.readTree(json);
        Map<String, Float> probs = CompletionResponseParser.parseProbabilities(node);
        assertEquals(1, probs.size());
        assertTrue("only outer token 'A' should be present", probs.containsKey("A"));
        assertFalse("inner top_probs token 'B' must not appear", probs.containsKey("B"));
    }
}
