// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.json.ChatResponseParser;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ClaudeGenerated(
        purpose = "Verify ChatResponseParser.parseResponse maps the OAI-compatible chat completion JSON "
                + "into ChatResponse / ChatChoice / ChatMessage / ToolCall, surfaces Usage and Timings, "
                + "and falls back gracefully on malformed input."
)
public class ChatResponseTest {

    private final ChatResponseParser parser = new ChatResponseParser();

    @Test
    public void parsesPlainAssistantReply() {
        String json = "{\"id\":\"chatcmpl-1\","
                + "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"Hello!\"},"
                + "\"finish_reason\":\"stop\"}],"
                + "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17},"
                + "\"timings\":{\"prompt_n\":12,\"prompt_ms\":100.0,\"prompt_per_second\":120.0,"
                + "\"predicted_n\":5,\"predicted_ms\":50.0,\"predicted_per_second\":100.0}}";
        ChatResponse r = parser.parseResponse(json);

        assertEquals("chatcmpl-1", r.getId());
        assertEquals(1, r.getChoices().size());
        ChatChoice c = r.getChoices().get(0);
        assertEquals(0, c.getIndex());
        assertEquals("assistant", c.getMessage().getRole());
        assertEquals("Hello!", c.getMessage().getContent());
        assertEquals("stop", c.getFinishReason());
        assertTrue(c.getMessage().getToolCalls().isEmpty());

        assertEquals(12L, r.getUsage().getPromptTokens());
        assertEquals(5L,  r.getUsage().getCompletionTokens());
        assertEquals(17L, r.getUsage().getTotalTokens());

        assertEquals(12, r.getTimings().getPromptN());
        assertEquals(100.0, r.getTimings().getPromptMs(), 1e-9);
        assertEquals(100.0, r.getTimings().getPredictedPerSecond(), 1e-9);

        assertEquals("Hello!", r.getFirstContent());
    }

    @Test
    public void parsesToolCalls() {
        String json = "{\"id\":\"x\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                + "\"content\":\"\",\"tool_calls\":["
                + "{\"id\":\"call_a\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Berlin\\\"}\"}},"
                + "{\"id\":\"call_b\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"get_time\",\"arguments\":\"{}\"}}"
                + "]},\"finish_reason\":\"tool_calls\"}],"
                + "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":7}}";
        ChatResponse r = parser.parseResponse(json);
        ChatMessage m = r.getFirstMessage();
        assertEquals("assistant", m.getRole());
        List<ToolCall> tc = m.getToolCalls();
        assertEquals(2, tc.size());
        assertEquals("call_a", tc.get(0).getId());
        assertEquals("get_weather", tc.get(0).getName());
        assertEquals("{\"city\":\"Berlin\"}", tc.get(0).getArgumentsJson());
        assertEquals("get_time", tc.get(1).getName());
        assertEquals("tool_calls", r.getChoices().get(0).getFinishReason());
    }

    @Test
    public void parsesObjectShapedArguments() {
        // Some upstream variants emit arguments as an object instead of a string
        String json = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                + "\"tool_calls\":[{\"id\":\"x\",\"type\":\"function\",\"function\":"
                + "{\"name\":\"f\",\"arguments\":{\"a\":1,\"b\":2}}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}";
        ChatResponse r = parser.parseResponse(json);
        String args = r.getFirstMessage().getToolCalls().get(0).getArgumentsJson();
        // exact text isn't guaranteed, but must contain both fields
        assertTrue("expected serialized object, got: " + args, args.contains("\"a\":1"));
        assertTrue(args.contains("\"b\":2"));
    }

    @Test
    public void malformedInputYieldsEmptyResponse() {
        ChatResponse r = parser.parseResponse("{not json");
        assertEquals("", r.getId());
        assertTrue(r.getChoices().isEmpty());
        assertEquals(0L, r.getUsage().getTotalTokens());
    }

    @Test
    public void buildMessagesJsonRoundTripsToolTurns() {
        ChatRequest req = new ChatRequest()
                .addMessage("system", "be terse")
                .addMessage("user", "two plus two?")
                .addMessage(ChatMessage.assistantToolCalls("",
                        java.util.Collections.singletonList(new ToolCall("c1", "add", "{\"a\":2,\"b\":2}"))))
                .addMessage(ChatMessage.toolResult("c1", "4"));

        String msgs = req.buildMessagesJson();
        assertTrue(msgs, msgs.contains("\"tool_calls\""));
        assertTrue(msgs, msgs.contains("\"tool_call_id\":\"c1\""));
        assertTrue(msgs, msgs.contains("\"name\":\"add\""));
    }

    @Test
    public void buildToolsJsonEmptyWhenNoTools() {
        ChatRequest req = new ChatRequest().addMessage("user", "hi");
        org.junit.Assert.assertNull(req.buildToolsJson());
    }

    @Test
    public void buildToolsJsonInlinesParameterSchema() {
        ChatRequest req = new ChatRequest().addTool(new ToolDefinition(
                "echo", "Echo a string",
                "{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}}}"));
        String tools = req.buildToolsJson();
        assertTrue(tools, tools.contains("\"type\":\"function\""));
        assertTrue(tools, tools.contains("\"name\":\"echo\""));
        assertTrue(tools, tools.contains("\"properties\""));
    }
}
