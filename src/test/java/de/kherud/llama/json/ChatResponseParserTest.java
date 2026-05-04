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
    private final ChatResponseParser parser = new ChatResponseParser();

    // ------------------------------------------------------------------
    // extractChoiceContent(String)
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceContent_typical() {
        String json = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"OK\"}," +
                "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}";
        assertEquals("OK", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_emptyContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\"}}]}";
        assertEquals("", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_escapedContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\"," +
                "\"content\":\"line1\\nline2\\t\\\"quoted\\\"\"}}]}";
        assertEquals("line1\nline2\t\"quoted\"", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_unicodeInContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"caf\\u00e9\"}}]}";
        assertEquals("café", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_missingChoices() {
        String json = "{\"id\":\"x\",\"object\":\"chat.completion\"}";
        assertEquals("", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_emptyChoicesArray() {
        String json = "{\"choices\":[]}";
        assertEquals("", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_missingContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\"}}]}";
        assertEquals("", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_malformedJson() {
        assertEquals("", parser.extractChoiceContent("{not json"));
    }

    @Test
    public void testExtractChoiceContent_multilineResponse() {
        String content = "First line.\\nSecond line.\\nThird line.";
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" + content + "\"}}]}";
        assertEquals("First line.\nSecond line.\nThird line.", parser.extractChoiceContent(json));
    }

    // ------------------------------------------------------------------
    // extractChoiceContent(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceContent_node() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}");
        assertEquals("Hello", parser.extractChoiceContent(node));
    }

    @Test
    public void testExtractChoiceContent_nodeMultipleChoices_takesFirst() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"choices\":[" +
                        "{\"message\":{\"content\":\"First\"}}," +
                        "{\"message\":{\"content\":\"Second\"}}" +
                        "]}");
        assertEquals("First", parser.extractChoiceContent(node));
    }

    // ------------------------------------------------------------------
    // extractChoiceReasoningContent
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceReasoningContent_present() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"The answer is 42.\"," +
                "\"reasoning_content\":\"Let me think step by step...\"}}]}";
        assertEquals("Let me think step by step...", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_absent_returnsEmpty() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}";
        assertEquals("", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_emptyString() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hi\"," +
                "\"reasoning_content\":\"\"}}]}";
        assertEquals("", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_missingChoices_returnsEmpty() {
        String json = "{\"id\":\"x\",\"object\":\"chat.completion\"}";
        assertEquals("", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_malformedJson_returnsEmpty() {
        assertEquals("", parser.extractChoiceReasoningContent("{not json"));
    }

    @Test
    public void testExtractChoiceReasoningContent_multiline() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"42\"," +
                "\"reasoning_content\":\"Step 1: identify the question.\\nStep 2: answer it.\"}}]}";
        assertEquals("Step 1: identify the question.\nStep 2: answer it.",
                parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_node() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"," +
                "\"reasoning_content\":\"thinking...\"}}]}");
        assertEquals("thinking...", parser.extractChoiceReasoningContent(node));
    }

    // ------------------------------------------------------------------
    // extractUsageField
    // ------------------------------------------------------------------

    @Test
    public void testExtractUsageField_promptTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(12, parser.extractUsageField(node, "prompt_tokens"));
    }

    @Test
    public void testExtractUsageField_completionTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(5, parser.extractUsageField(node, "completion_tokens"));
    }

    @Test
    public void testExtractUsageField_totalTokens() throws Exception {
        JsonNode node = MAPPER.readTree(
                "{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(17, parser.extractUsageField(node, "total_tokens"));
    }

    @Test
    public void testExtractUsageField_missingUsage_returnsZero() throws Exception {
        JsonNode node = MAPPER.readTree("{\"id\":\"x\"}");
        assertEquals(0, parser.extractUsageField(node, "prompt_tokens"));
    }

    @Test
    public void testExtractUsageField_missingField_returnsZero() throws Exception {
        JsonNode node = MAPPER.readTree("{\"usage\":{}}");
        assertEquals(0, parser.extractUsageField(node, "prompt_tokens"));
    }

    // ------------------------------------------------------------------
    // countChoices
    // ------------------------------------------------------------------

    @Test
    public void testCountChoices_one() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[{\"message\":{\"content\":\"hi\"}}]}");
        assertEquals(1, parser.countChoices(node));
    }

    @Test
    public void testCountChoices_multiple() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[{},{},{}]}");
        assertEquals(3, parser.countChoices(node));
    }

    @Test
    public void testCountChoices_empty() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[]}");
        assertEquals(0, parser.countChoices(node));
    }

    @Test
    public void testCountChoices_absent() throws Exception {
        JsonNode node = MAPPER.readTree("{\"id\":\"x\"}");
        assertEquals(0, parser.countChoices(node));
    }
}
