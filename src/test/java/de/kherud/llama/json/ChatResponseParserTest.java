package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link ChatResponseParser}.
 * No JVM native library or model file needed — JSON string literals only.
 */
public class ChatResponseParserTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ------------------------------------------------------------------
    // extractChoiceContent(String)
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceContent_typical() {
        String json = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"OK\"}," +
                "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}";
        assertEquals("OK", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_emptyContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\"}}]}";
        assertEquals("", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_escapedContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\"," +
                "\"content\":\"line1\\nline2\\t\\\"quoted\\\"\"}}]}";
        assertEquals("line1\nline2\t\"quoted\"", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_unicodeInContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"caf\\u00e9\"}}]}";
        assertEquals("café", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_missingChoices() {
        String json = "{\"id\":\"x\",\"object\":\"chat.completion\"}";
        assertEquals("", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_emptyChoicesArray() {
        String json = "{\"choices\":[]}";
        assertEquals("", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_missingContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\"}}]}";
        assertEquals("", ChatResponseParser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_malformedJson() {
        assertEquals("", ChatResponseParser.extractChoiceContent("{not json"));
    }

    @Test
    public void testExtractChoiceContent_multilineResponse() {
        String content = "First line.\\nSecond line.\\nThird line.";
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" + content + "\"}}]}";
        assertEquals("First line.\nSecond line.\nThird line.", ChatResponseParser.extractChoiceContent(json));
    }

    // ------------------------------------------------------------------
    // extractChoiceContent(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceContent_node() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}");
        assertEquals("Hello", ChatResponseParser.extractChoiceContent(node));
    }

    @Test
    public void testExtractChoiceContent_nodeMultipleChoices_takesFirst() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"choices\":[" +
                        "{\"message\":{\"content\":\"First\"}}," +
                        "{\"message\":{\"content\":\"Second\"}}" +
                        "]}");
        assertEquals("First", ChatResponseParser.extractChoiceContent(node));
    }

    // ------------------------------------------------------------------
    // extractUsageField
    // ------------------------------------------------------------------

    @Test
    public void testExtractUsageField_promptTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(12, ChatResponseParser.extractUsageField(node, "prompt_tokens"));
    }

    @Test
    public void testExtractUsageField_completionTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(5, ChatResponseParser.extractUsageField(node, "completion_tokens"));
    }

    @Test
    public void testExtractUsageField_totalTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(17, ChatResponseParser.extractUsageField(node, "total_tokens"));
    }

    @Test
    public void testExtractUsageField_missingUsage_returnsZero() throws Exception {
        JsonNode node = MAPPER.readTree("{\"id\":\"x\"}");
        assertEquals(0, ChatResponseParser.extractUsageField(node, "prompt_tokens"));
    }

    @Test
    public void testExtractUsageField_missingField_returnsZero() throws Exception {
        JsonNode node = MAPPER.readTree("{\"usage\":{}}");
        assertEquals(0, ChatResponseParser.extractUsageField(node, "prompt_tokens"));
    }

    // ------------------------------------------------------------------
    // countChoices
    // ------------------------------------------------------------------

    @Test
    public void testCountChoices_one() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[{\"message\":{\"content\":\"hi\"}}]}");
        assertEquals(1, ChatResponseParser.countChoices(node));
    }

    @Test
    public void testCountChoices_multiple() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[{},{},{}]}");
        assertEquals(3, ChatResponseParser.countChoices(node));
    }

    @Test
    public void testCountChoices_empty() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[]}");
        assertEquals(0, ChatResponseParser.countChoices(node));
    }

    @Test
    public void testCountChoices_absent() throws Exception {
        JsonNode node = MAPPER.readTree("{\"id\":\"x\"}");
        assertEquals(0, ChatResponseParser.countChoices(node));
    }
}
