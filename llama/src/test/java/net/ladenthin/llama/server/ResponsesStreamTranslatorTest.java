// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ResponsesStreamTranslator}: the OpenAI-chunk to Responses-SSE-event sequence
 * (response.created → output_item/content_part → output_text.delta* → done events → response.completed).
 * Pure.
 */
public class ResponsesStreamTranslatorTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** The {@code data:} payloads of the framed SSE events, in order. */
    private static List<JsonNode> events(String sse) throws IOException {
        List<JsonNode> events = new ArrayList<>();
        for (String line : sse.split("\n")) {
            if (line.startsWith("data: ")) {
                events.add(MAPPER.readTree(line.substring("data: ".length())));
            }
        }
        return events;
    }

    /** The first event of {@code type} whose {@code output_index} is {@code outputIndex}. */
    private static JsonNode eventAt(List<JsonNode> events, String type, int outputIndex) {
        for (JsonNode event : events) {
            if (event.path("type").asText().equals(type)
                    && event.path("output_index").asInt(-1) == outputIndex) {
                return event;
            }
        }
        throw new AssertionError("no " + type + " at output_index " + outputIndex);
    }

    @Test
    public void beginEmitsResponseCreated() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        String begin = translator.begin();
        assertThat(begin, containsString("event: response.created"));
        assertThat(begin, containsString("\"sequence_number\":0"));
    }

    @Test
    public void firstTextDeltaOpensItemAndPartThenStreamsDelta() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        String first = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
        assertThat(first, containsString("event: response.output_item.added"));
        assertThat(first, containsString("event: response.content_part.added"));
        assertThat(first, containsString("event: response.output_text.delta"));
        assertThat(first, containsString("\"delta\":\"he\""));
        String second = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}");
        assertThat(second.contains("output_item.added"), is(false));
        assertThat(second, containsString("\"delta\":\"llo\""));
    }

    @Test
    public void endEmitsDoneEventsAndCompleted() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}");
        translator.onChunk("{\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":8}}}");
        String end = translator.end();
        assertThat(end, containsString("event: response.output_text.done"));
        assertThat(end, containsString("event: response.content_part.done"));
        assertThat(end, containsString("event: response.output_item.done"));
        assertThat(end, containsString("event: response.completed"));
        assertThat(end, containsString("\"text\":\"hi\""));
        assertThat(end, containsString("\"input_tokens\":12"));
        assertThat(end, containsString("\"output_tokens\":3"));
        assertThat(end, containsString("\"cached_tokens\":8"));
    }

    @Test
    public void reasoningDeltasStreamAsAReasoningItemLikeUpstream() {
        // Upstream's to_json_oaicompat_resp_stream opens a reasoning output item, streams
        // response.reasoning_text.delta and puts the item first in the completed output.
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        String first = translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"6 times 7\"}}]}");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"42\"}}]}");
        String end = translator.end();

        assertThat(first, containsString("event: response.output_item.added"));
        assertThat(first, containsString("\"type\":\"reasoning\""));
        assertThat(first, containsString("event: response.reasoning_text.delta"));
        assertThat(first, containsString("\"delta\":\"6 times 7\""));
        String completed = end.substring(end.indexOf("event: response.completed"));
        assertThat(
                completed.indexOf("\"type\":\"reasoning\"") >= 0
                        && completed.indexOf("\"type\":\"reasoning\"") < completed.indexOf("\"type\":\"message\""),
                is(true));
        assertThat(completed, containsString("\"type\":\"reasoning_text\",\"text\":\"6 times 7\""));
    }

    @Test
    public void reasoningTextAndToolCallGetConsecutiveOutputIndicesAndSequenceNumbers() throws IOException {
        // Responses clients key the items by output_index (the OpenAI SDK raises on a duplicate) and
        // order the events by sequence_number; reasoning, message and function_call are items 0, 1, 2.
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        List<JsonNode> events = events(translator.begin()
                + translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"6 times 7\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"42\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                        + "\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}")
                + translator.end());

        assertThat(events.size(), is(15));
        List<Integer> sequence = new ArrayList<>();
        List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < events.size(); i++) {
            sequence.add(events.get(i).path("sequence_number").asInt(-1));
            expected.add(i);
        }
        assertThat(sequence, is(expected));

        String[] itemTypes = {"reasoning", "message", "function_call"};
        for (int index = 0; index < itemTypes.length; index++) {
            assertThat(
                    eventAt(events, "response.output_item.added", index)
                            .path("item")
                            .path("type")
                            .asText(),
                    is(itemTypes[index]));
            assertThat(
                    eventAt(events, "response.output_item.done", index)
                            .path("item")
                            .path("type")
                            .asText(),
                    is(itemTypes[index]));
        }

        JsonNode reasoningDone = eventAt(events, "response.reasoning_text.done", 0);
        assertThat(reasoningDone.path("item_id").asText(), is("rs_resp_1"));
        assertThat(reasoningDone.path("content_index").asInt(-1), is(0));
        assertThat(reasoningDone.path("text").asText(), is("6 times 7"));
        assertThat(reasoningDone.path("sequence_number").asInt(-1), is(6));

        JsonNode completed = events.get(events.size() - 1);
        assertThat(completed.path("type").asText(), is("response.completed"));
        List<String> outputTypes = new ArrayList<>();
        for (JsonNode item : completed.path("response").path("output")) {
            outputTypes.add(item.path("type").asText());
        }
        assertThat(outputTypes, contains("reasoning", "message", "function_call"));
    }

    @Test
    public void failedContinuesTheSequenceNumber() throws IOException {
        // response.failed ends a stream that already emitted events; its sequence_number follows them.
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        List<JsonNode> events = events(translator.begin()
                + translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}")
                + translator.failed("boom", "server_error"));

        assertThat(events.size(), is(5));
        JsonNode failed = events.get(4);
        assertThat(failed.path("type").asText(), is("response.failed"));
        assertThat(failed.path("sequence_number").asInt(-1), is(4));
        assertThat(failed.path("response").path("status").asText(), is("failed"));
        assertThat(failed.path("response").path("error").path("code").asText(), is("server_error"));
        assertThat(failed.path("response").path("error").path("message").asText(), is("boom"));
    }

    @Test
    public void toolCallsBecomeFunctionCallItemsBeforeCompleted() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}");
        String end = translator.end();
        assertThat(end, containsString("\"type\":\"function_call\""));
        assertThat(end, containsString("event: response.function_call_arguments.done"));
        assertThat(end, containsString("event: response.completed"));
    }

    @Test
    public void maxTokensCutOffEndsWithResponseIncomplete() {
        // finish_reason "length" is a cut-off answer; it was reported as response.completed, so a client
        // that hit max_output_tokens could not tell. The usage-only chunk the server appends afterwards
        // ("choices":[]) must not reset the recorded reason.
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}");
        translator.onChunk("{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}");
        translator.onChunk("{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}");
        String end = translator.end();

        assertThat(end, containsString("event: response.incomplete"));
        assertThat(end, not(containsString("event: response.completed")));
        String terminal = end.substring(end.indexOf("event: response.incomplete"));
        assertThat(terminal, containsString("\"status\":\"incomplete\""));
        assertThat(terminal, containsString("\"incomplete_details\":{\"reason\":\"max_output_tokens\"}"));
        String messageDone = end.substring(end.indexOf("event: response.output_item.done"));
        assertThat(
                messageDone.substring(0, messageDone.indexOf("\n\n")),
                containsString("\"type\":\"message\",\"id\":\"msg_resp_1\",\"status\":\"incomplete\""));
    }

    @Test
    public void aStoppedAnswerStaysCompleted() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}");
        String end = translator.end();

        assertThat(end, containsString("event: response.completed"));
        assertThat(end, not(containsString("incomplete")));
    }
}
