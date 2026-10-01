// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import net.ladenthin.llama.value.ChatChoice;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ToolCall;
import nl.altindag.log.LogCaptor;
import org.junit.jupiter.api.Test;

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
        String json = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"OK\"},"
                + "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}";
        assertEquals("OK", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_emptyContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\"}}]}";
        assertEquals("", parser.extractChoiceContent(json));
    }

    @Test
    public void testExtractChoiceContent_escapedContent() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\","
                + "\"content\":\"line1\\nline2\\t\\\"quoted\\\"\"}}]}";
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
        JsonNode node = MAPPER.readTree("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}");
        assertEquals("Hello", parser.extractChoiceContent(node));
    }

    @Test
    public void testExtractChoiceContent_nodeMultipleChoices_takesFirst() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[" + "{\"message\":{\"content\":\"First\"}},"
                + "{\"message\":{\"content\":\"Second\"}}"
                + "]}");
        assertEquals("First", parser.extractChoiceContent(node));
    }

    // ------------------------------------------------------------------
    // extractChoiceReasoningContent
    // ------------------------------------------------------------------

    @Test
    public void testExtractChoiceReasoningContent_present() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"The answer is 42.\","
                + "\"reasoning_content\":\"Let me think step by step...\"}}]}";
        assertEquals("Let me think step by step...", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_absent_returnsEmpty() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}";
        assertEquals("", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_emptyString() {
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hi\","
                + "\"reasoning_content\":\"\"}}]}";
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
        String json = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"42\","
                + "\"reasoning_content\":\"Step 1: identify the question.\\nStep 2: answer it.\"}}]}";
        assertEquals("Step 1: identify the question.\nStep 2: answer it.", parser.extractChoiceReasoningContent(json));
    }

    @Test
    public void testExtractChoiceReasoningContent_node() throws Exception {
        JsonNode node = MAPPER.readTree("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\","
                + "\"reasoning_content\":\"thinking...\"}}]}");
        assertEquals("thinking...", parser.extractChoiceReasoningContent(node));
    }

    // ------------------------------------------------------------------
    // extractUsageField
    // ------------------------------------------------------------------

    @Test
    public void testExtractUsageField_promptTokens() throws Exception {
        JsonNode node =
                MAPPER.readTree("{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(12, parser.extractUsageField(node, "prompt_tokens"));
    }

    @Test
    public void testExtractUsageField_completionTokens() throws Exception {
        JsonNode node =
                MAPPER.readTree("{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
        assertEquals(5, parser.extractUsageField(node, "completion_tokens"));
    }

    @Test
    public void testExtractUsageField_totalTokens() throws Exception {
        JsonNode node =
                MAPPER.readTree("{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17}}");
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

    // ------------------------------------------------------------------
    // parseResponse(String) — full typed parse
    // ------------------------------------------------------------------

    @Test
    public void testParseResponse_fullResponse() {
        String json = "{\"id\":\"chatcmpl-abc\",\"choices\":[{\"index\":0,"
                + "\"message\":{\"role\":\"assistant\",\"content\":\"Hi there\"},"
                + "\"finish_reason\":\"stop\"}],"
                + "\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":5}}}";
        ChatResponse r = parser.parseResponse(json);

        assertEquals("chatcmpl-abc", r.getId());
        assertEquals(1, r.getChoices().size());
        ChatChoice c = r.getChoices().get(0);
        assertEquals(0, c.getIndex());
        assertEquals("assistant", c.getMessage().getRole());
        assertEquals("Hi there", c.getMessage().getContent());
        assertEquals("stop", c.getFinishReason());
        assertEquals(7L, r.getUsage().getPromptTokens());
        assertEquals(3L, r.getUsage().getCompletionTokens());
        assertEquals(5L, r.getUsage().getCachedTokens());
        assertEquals(2L, r.getUsage().getProcessedPromptTokens());
        assertEquals(json, r.getRawJson());
    }

    @Test
    public void testParseResponse_multipleChoicesPreserveIndexAndOrder() {
        String json = "{\"id\":\"x\",\"choices\":["
                + "{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"first\"},\"finish_reason\":\"stop\"},"
                + "{\"index\":1,\"message\":{\"role\":\"assistant\",\"content\":\"second\"},\"finish_reason\":\"length\"}"
                + "]}";
        ChatResponse r = parser.parseResponse(json);

        assertEquals(2, r.getChoices().size());
        assertEquals(0, r.getChoices().get(0).getIndex());
        assertEquals("first", r.getChoices().get(0).getMessage().getContent());
        assertEquals(1, r.getChoices().get(1).getIndex());
        assertEquals("second", r.getChoices().get(1).getMessage().getContent());
        assertEquals("length", r.getChoices().get(1).getFinishReason());
    }

    @Test
    public void testParseResponse_toolCallsWithStringArguments() {
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,"
                + "\"message\":{\"role\":\"assistant\",\"content\":\"\","
                + "\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}";
        ChatResponse r = parser.parseResponse(json);

        ChatMessage m = r.getChoices().get(0).getMessage();
        List<ToolCall> tcs = m.getToolCalls();
        assertEquals(1, tcs.size());
        assertEquals("call_1", tcs.get(0).getId());
        assertEquals("get_weather", tcs.get(0).getName());
        // arguments is a JSON string in the wire form → unwrapped verbatim, not re-quoted.
        assertEquals("{\"city\":\"NYC\"}", tcs.get(0).getArgumentsJson());
    }

    @Test
    public void testParseResponse_toolCallsWithObjectArguments() {
        // Some shapes emit arguments as a nested object rather than a string;
        // the parser serialises it back to its JSON text.
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,"
                + "\"message\":{\"role\":\"assistant\",\"content\":\"\","
                + "\"tool_calls\":[{\"id\":\"call_2\","
                + "\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}]}}]}";
        ChatResponse r = parser.parseResponse(json);

        ToolCall tc = r.getChoices().get(0).getMessage().getToolCalls().get(0);
        assertEquals("{\"a\":1}", tc.getArgumentsJson());
    }

    @Test
    public void testParseResponse_noToolCalls_plainAssistantMessage() {
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,"
                + "\"message\":{\"role\":\"assistant\",\"content\":\"plain\"}}]}";
        ChatResponse r = parser.parseResponse(json);

        ChatMessage m = r.getChoices().get(0).getMessage();
        assertEquals("plain", m.getContent());
        assertTrue(m.getToolCalls().isEmpty(), "plain message carries no tool calls");
    }

    @Test
    public void testParseResponse_emptyChoicesArray_returnsMutableEmptyList() {
        ChatResponse r = parser.parseResponse("{\"id\":\"x\",\"choices\":[]}");
        assertTrue(r.getChoices().isEmpty());
        // The choices list is exposed by reference and documented as mutable —
        // adding to it must not throw (kills the immutable-emptyList() mutant).
        r.getChoices().add(new ChatChoice(0, new ChatMessage("assistant", "added"), "stop"));
        assertEquals(1, r.getChoices().size());
    }

    @Test
    public void testParseResponse_absentChoices_returnsEmptyList() {
        ChatResponse r = parser.parseResponse("{\"id\":\"x\"}");
        assertEquals("x", r.getId());
        assertTrue(r.getChoices().isEmpty());
    }

    @Test
    public void testParseResponse_malformedJson_returnsEmptyResponsePreservingRawJson() {
        String bad = "{not valid json";
        ChatResponse r = parser.parseResponse(bad);
        assertEquals("", r.getId());
        assertTrue(r.getChoices().isEmpty());
        assertEquals(0L, r.getUsage().getPromptTokens());
        assertEquals(0L, r.getUsage().getCompletionTokens());
        // Raw JSON is preserved verbatim even on parse failure (escape hatch).
        assertEquals(bad, r.getRawJson());
    }

    /**
     * Parsing a response carrying real timings must emit exactly one per-run
     * timing line through the dedicated SLF4J logger — pins the {@code
     * TimingsLogger.log(...)} side-effect so its removal (VoidMethodCall mutant)
     * is detected.
     */
    @Test
    public void testParseResponse_emitsTimingLine() {
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,"
                + "\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}],"
                + "\"timings\":{\"prompt_n\":7,\"prompt_ms\":10.0,\"prompt_per_second\":700.0,"
                + "\"predicted_n\":3,\"predicted_ms\":20.0,\"predicted_per_second\":150.0}}";

        try (LogCaptor captor = LogCaptor.forName(TimingsLogger.LOGGER_NAME)) {
            ChatResponse r = parser.parseResponse(json);
            assertEquals(7, r.getTimings().getPromptN());
            assertEquals(1, captor.getInfoLogs().size(), "exactly one timing line must be emitted");
        }
    }
}
