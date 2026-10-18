// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ResponsesStreamTranslator}: the OpenAI-chunk to Responses-SSE-event sequence
 * (response.created → output_item/content_part → output_text.delta* → done events → response.completed).
 * Pure.
 */
public class ResponsesStreamTranslatorTest {

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
    public void toolCallsBecomeFunctionCallItemsBeforeCompleted() {
        ResponsesStreamTranslator translator = new ResponsesStreamTranslator("m", "resp_1");
        translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}");
        String end = translator.end();
        assertThat(end, containsString("\"type\":\"function_call\""));
        assertThat(end, containsString("event: response.function_call_arguments.done"));
        assertThat(end, containsString("event: response.completed"));
    }
}
