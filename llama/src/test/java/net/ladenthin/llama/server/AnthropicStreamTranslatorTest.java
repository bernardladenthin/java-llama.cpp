// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link AnthropicStreamTranslator}: the OpenAI-chunk to Anthropic-SSE-event sequence
 * (message_start → text content block → tool_use blocks → message_delta → message_stop). Pure.
 */
public class AnthropicStreamTranslatorTest {

    @Test
    public void beginEmitsMessageStart() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        assertThat(translator.begin(), containsString("event: message_start"));
    }

    @Test
    public void firstTextDeltaOpensBlockThenSubsequentDeltasAppend() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        String first = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
        assertThat(first, containsString("event: content_block_start"));
        assertThat(first, containsString("event: content_block_delta"));
        assertThat(first, containsString("\"text\":\"he\""));
        String second = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}");
        // No second block start; just another delta.
        assertThat(second.contains("content_block_start"), is(false));
        assertThat(second, containsString("\"text\":\"llo\""));
    }

    @Test
    public void theMatchedStopWordIsTheStopSequenceOfTheMessageDelta() {
        // upstream puts __verbose on the first chunk of the final result (F183)
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}");
        String last = translator.onChunk("{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],"
                + "\"__verbose\":{\"stopping_word\":\"END\"}}");
        assertThat(last.contains("__verbose"), is(false));
        String end = translator.end();
        assertThat(end, containsString("\"stop_reason\":\"stop_sequence\""));
        assertThat(end, containsString("\"stop_sequence\":\"END\""));
    }

    @Test
    public void aMatchedStopWordIsTheStopReasonStopSequenceOfTheMessageDelta() {
        // As the Messages API specifies for a matched stop sequence; upstream says end_turn (F221).
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}");
        translator.onChunk("{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],"
                + "\"__verbose\":{\"stopping_word\":\"END\"}}");
        String end = translator.end();
        assertThat(end, containsString("\"stop_reason\":\"stop_sequence\""));
        assertThat(end.contains("\"stop_reason\":\"end_turn\""), is(false));
    }

    @Test
    public void withoutAStopWordTheStopSequenceIsNull() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"
                + "\"__verbose\":{\"stopping_word\":\"\"}}");
        assertThat(translator.end(), containsString("\"stop_sequence\":null"));
    }

    @Test
    public void endClosesTextBlockAndEmitsStopReasonAndMessageStop() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}");
        translator.onChunk("{\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":8}}}");
        String end = translator.end();
        assertThat(end, containsString("event: content_block_stop"));
        assertThat(end, containsString("event: message_delta"));
        assertThat(end, containsString("\"stop_reason\":\"end_turn\""));
        assertThat(end, containsString("event: message_stop"));
        assertThat(end, containsString("\"input_tokens\":4"));
        assertThat(end, containsString("\"output_tokens\":3"));
        assertThat(end, containsString("\"cache_read_input_tokens\":8"));
    }

    @Test
    public void reasoningDeltasStreamAsAThinkingBlockClosedWithASignatureBeforeTheText() {
        // Upstream's to_json_anthropic_stream streams reasoning_content as a thinking block
        // (thinking_delta) and closes it with an empty signature_delta, which the protocol requires.
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        String first = translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"6 times\"}}]}");
        String second = translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\" 7\"}}]}");
        String text = translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"42\"}}]}");
        String events = first + second + text + translator.end();

        assertThat(first, containsString("\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}"));
        assertThat(first, containsString("\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"6 times\"}"));
        assertThat(second, containsString("\"thinking\":\" 7\""));
        int signature = events.indexOf("\"type\":\"signature_delta\"");
        int thinkingStop = events.indexOf("\"type\":\"content_block_stop\",\"index\":0");
        int textStart = events.indexOf("\"content_block\":{\"type\":\"text\"");
        assertThat(signature >= 0 && signature < thinkingStop && thinkingStop < textStart, is(true));
        assertThat(events, containsString("\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"42\"}"));
    }

    @Test
    public void reasoningFollowedByAToolCallIsClosedBeforeTheToolUseBlock() {
        // The common agent turn of a reasoning model: thinking, then a tool call and no text. No content
        // chunk closes the thinking block, so end() must, before the tool_use block opens; otherwise
        // block 1 starts while block 0 is still open, without its signature_delta and stop.
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        String events = translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                        + "\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}")
                + translator.end();

        int signature = events.indexOf("\"type\":\"signature_delta\"");
        int thinkingStop = events.indexOf("\"type\":\"content_block_stop\",\"index\":0");
        int toolStart = events.indexOf("\"index\":1,\"content_block\":{\"type\":\"tool_use\"");
        assertThat("signature_delta is emitted", signature >= 0, is(true));
        assertThat("the thinking block is stopped after its signature", thinkingStop > signature, is(true));
        assertThat("the tool_use block opens at index 1 after the stop", toolStart > thinkingStop, is(true));
        assertThat(events, containsString("\"stop_reason\":\"tool_use\""));
    }

    @Test
    public void aThinkingOnlyAnswerIsClosedAtTheEnd() {
        // A thinking-only answer (e.g. max_tokens spent inside the thinking block) has no content chunk
        // either; end() closes the block with its signature before message_delta and opens no other.
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}");
        String end = translator.end();

        int signature = end.indexOf("\"type\":\"signature_delta\"");
        int thinkingStop = end.indexOf("\"type\":\"content_block_stop\",\"index\":0");
        int messageDelta = end.indexOf("event: message_delta");
        assertThat("end() emits the signature_delta", signature >= 0, is(true));
        assertThat("then the stop of block 0", thinkingStop > signature, is(true));
        assertThat("then message_delta", messageDelta > thinkingStop, is(true));
        assertThat(end.contains("content_block_start"), is(false));
    }

    @Test
    public void reasoningAfterTextClosesTheTextBlockBeforeANewThinkingBlock() {
        // A parser can grow the reasoning after the content started (interleaved thinking: GPT-OSS,
        // Gemma 4). The blocks must still follow one another, each start, deltas, stop, as Claude itself
        // streams several thinking blocks: 0 thinking, 1 text, 2 thinking, 3 text (F222).
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        String events = translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"a\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"b\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"reasoning_content\":\"c\"}}]}")
                + translator.onChunk("{\"choices\":[{\"delta\":{\"content\":\"d\"}}]}");
        String end = translator.end();
        events += end;

        assertThat(events, containsString("\"index\":2,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}"));
        assertThat(events, containsString("\"index\":3,\"delta\":{\"type\":\"text_delta\",\"text\":\"d\"}"));
        int textStop = events.indexOf("\"type\":\"content_block_stop\",\"index\":1");
        int secondThinkingStart =
                events.indexOf("{\"type\":\"thinking\",\"thinking\":\"\"}", events.indexOf("\"text\":\"b\""));
        assertThat(
                "the text block stops before the second thinking block starts",
                textStop >= 0 && textStop < secondThinkingStart,
                is(true));
        assertThat(
                "no delta reaches the text block after its stop",
                events.lastIndexOf("\"index\":1,\"delta\"") < textStop,
                is(true));
        int messageDelta = end.indexOf("event: message_delta");
        String closing = end.substring(0, messageDelta);
        assertThat(
                "end() closes exactly one block, the last text block",
                closing.split("event: content_block_stop", -1).length - 1,
                is(1));
        assertThat(closing, containsString("\"type\":\"content_block_stop\",\"index\":3"));
    }

    @Test
    public void accumulatedToolCallsBecomeToolUseBlocksAtEnd() {
        AnthropicStreamTranslator translator = new AnthropicStreamTranslator("msg_1", "m");
        translator.onChunk("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}");
        String end = translator.end();
        assertThat(end, containsString("event: content_block_start"));
        assertThat(end, containsString("\"type\":\"tool_use\""));
        assertThat(end, containsString("\"name\":\"get_weather\""));
        assertThat(end, containsString("event: content_block_delta"));
        assertThat(end, containsString("input_json_delta"));
        assertThat(end, containsString("\"stop_reason\":\"tool_use\""));
    }
}
