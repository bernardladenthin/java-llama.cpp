// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Model-free tests for the chunk-stream state machine behind {@code JllamaStreamingChatModel}:
 * canned OpenAI {@code chat.completion.chunk} JSON drives text/thinking/tool-call event
 * forwarding and final-response assembly.
 */
class StreamingChunkAssemblerTest {

    private static final class RecordingHandler implements StreamingChatResponseHandler {
        final List<String> partials = new ArrayList<>();
        final List<String> thinking = new ArrayList<>();
        final List<PartialToolCall> partialToolCalls = new ArrayList<>();
        final List<CompleteToolCall> completeToolCalls = new ArrayList<>();

        @Override
        public void onPartialResponse(String partialResponse) {
            partials.add(partialResponse);
        }

        @Override
        public void onPartialThinking(PartialThinking partialThinking) {
            thinking.add(partialThinking.text());
        }

        @Override
        public void onPartialToolCall(PartialToolCall partialToolCall) {
            partialToolCalls.add(partialToolCall);
        }

        @Override
        public void onCompleteToolCall(CompleteToolCall completeToolCall) {
            completeToolCalls.add(completeToolCall);
        }

        @Override
        public void onCompleteResponse(ChatResponse completeResponse) {
            throw new AssertionError("assembler never calls onCompleteResponse itself");
        }

        @Override
        public void onError(Throwable error) {
            throw new AssertionError("assembler never calls onError itself", error);
        }
    }

    private static String contentChunk(String content) {
        return "{\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"content\":\"" + content + "\"},\"finish_reason\":null}]}";
    }

    @Test
    void assemblesPlainTextStream() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept(contentChunk("Hel"));
        assembler.accept(contentChunk("lo"));
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(handler.partials, contains("Hel", "lo"));
        assertThat(response.aiMessage().text(), is("Hello"));
        assertThat(response.finishReason(), is(FinishReason.STOP));
        assertThat(response.aiMessage().hasToolExecutionRequests(), is(false));
    }

    @Test
    void assemblesStreamedToolCallAcrossFragments() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        // Fragment 1: id + name + start of arguments. Fragment 2: rest of arguments.
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"id\":\"call_1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}}]},\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"function\":{\"arguments\":\"ty\\\":\\\"Berlin\\\"}\"}}]},\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],"
                + "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}");
        ChatResponse response = assembler.complete();

        assertThat(handler.partialToolCalls.size(), is(2));
        assertThat(handler.partialToolCalls.get(0).name(), is("get_weather"));
        assertThat(handler.partialToolCalls.get(0).partialArguments(), is("{\"ci"));
        assertThat(handler.partialToolCalls.get(1).partialArguments(), is("ty\":\"Berlin\"}"));
        assertThat(handler.completeToolCalls.size(), is(1));
        assertThat(
                handler.completeToolCalls.get(0).toolExecutionRequest().arguments(),
                is("{\"city\":\"Berlin\"}"));
        assertThat(response.aiMessage().hasToolExecutionRequests(), is(true));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).id(), is("call_1"));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("get_weather"));
        assertThat(response.finishReason(), is(FinishReason.TOOL_EXECUTION));
        assertThat(response.tokenUsage().inputTokenCount(), is(10));
        assertThat(response.tokenUsage().outputTokenCount(), is(5));
    }

    @Test
    void assemblesParallelToolCallsByIndex() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":["
                + "{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}},"
                + "{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}"
                + "]},\"finish_reason\":\"tool_calls\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(handler.completeToolCalls.size(), is(2));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("first"));
        assertThat(response.aiMessage().toolExecutionRequests().get(1).name(), is("second"));
    }

    @Test
    void forwardsThinkingDeltasAndKeepsThinkingOnFinalMessage() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"let me \"},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"},"
                + "\"finish_reason\":null}]}");
        assembler.accept(contentChunk("42"));
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(handler.thinking, contains("let me ", "think"));
        assertThat(response.aiMessage().thinking(), is("let me think"));
        assertThat(response.aiMessage().text(), is("42"));
    }

    @Test
    void thinkingOnlyStreamLeavesTextNullAndDeliversNoContentFragments() {
        // A reasoning model that spends its whole output budget inside <think> streams only
        // reasoning_content. complete() then never calls message.text(...), so AiMessage.text()
        // is null rather than "" -- which is exactly what made
        // JllamaChatModelIntegrationTest#streamingDeliversTokensThenCompletes fail with
        // "expected \"\" but was null" the first time it actually ran against Qwen3-0.6B.
        // Pinned here, model-free, so the null is a known contract and not a surprise.
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"still \"},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(response.aiMessage().text(), is(nullValue()));
        assertThat(response.aiMessage().thinking(), is("still thinking"));
        assertThat(handler.partials, is(empty()));
        assertThat(handler.thinking, contains("still ", "thinking"));
    }

    @Test
    void noUsageChunkMeansNoTokenUsage() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept(contentChunk("x"));
        ChatResponse response = assembler.complete();

        assertThat(response.tokenUsage(), is(nullValue()));
    }

    @Test
    void unparseableChunkFailsLoud() {
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(new RecordingHandler());

        assertThrows(UncheckedIOException.class, () -> assembler.accept("not json"));
    }
}
