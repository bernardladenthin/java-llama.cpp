// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ToolCallDeltaAccumulator}: reconstructing whole tool calls from fragmented
 * OpenAI streaming {@code delta.tool_calls}. Pure — JSON literals only.
 */
public class ToolCallDeltaAccumulatorTest {

    @Test
    public void mergesNameAndArgumentFragmentsByIndex() {
        ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        accumulator.accept("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}");
        accumulator.accept(
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}");
        assertThat(accumulator.hasToolCalls(), is(true));
        ArrayNode toolCalls = accumulator.toOpenAiToolCalls();
        assertThat(toolCalls.size(), is(1));
        JsonNode toolCall = toolCalls.get(0);
        assertThat(toolCall.path("id").asText(), is("call_1"));
        assertThat(toolCall.path("type").asText(), is("function"));
        assertThat(toolCall.path("function").path("name").asText(), is("get_weather"));
        // Arguments are the concatenated JSON-encoded string (not a parsed object).
        assertThat(toolCall.path("function").path("arguments").isTextual(), is(true));
        assertThat(toolCall.path("function").path("arguments").asText(), is("{\"city\":\"Paris\"}"));
    }

    @Test
    public void tracksMultipleParallelToolCallsInIndexOrder() {
        ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        accumulator.accept("{\"choices\":[{\"delta\":{\"tool_calls\":["
                + "{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}},"
                + "{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}}]}}]}");
        ArrayNode toolCalls = accumulator.toOpenAiToolCalls();
        assertThat(toolCalls.size(), is(2));
        // Emitted in index order (0 then 1), regardless of arrival order.
        assertThat(toolCalls.get(0).path("id").asText(), is("a"));
        assertThat(toolCalls.get(1).path("id").asText(), is("b"));
    }

    @Test
    public void ignoresChunksWithoutToolCalls() {
        ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        accumulator.accept("{\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}");
        accumulator.accept("not json");
        assertThat(accumulator.hasToolCalls(), is(false));
        assertThat(accumulator.toOpenAiToolCalls().size(), is(0));
    }
}
