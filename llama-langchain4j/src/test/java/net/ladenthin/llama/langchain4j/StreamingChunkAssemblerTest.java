// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static java.util.stream.Collectors.joining;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
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
        /** Every tool-call event in arrival order, e.g. {@code "partial:0"}, {@code "complete:0"}. */
        final List<String> toolCallEvents = new ArrayList<>();

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
            toolCallEvents.add("partial:" + partialToolCall.index());
        }

        @Override
        public void onCompleteToolCall(CompleteToolCall completeToolCall) {
            completeToolCalls.add(completeToolCall);
            toolCallEvents.add("complete:" + completeToolCall.index());
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

    /**
     * langchain4j's contract for the partial events of one call: concatenated, the argument fragments
     * are the complete call's arguments, and each carries the complete call's id and name.
     */
    private static void assertPartialsMakeUpTheCompleteCall(RecordingHandler handler) {
        ToolExecutionRequest complete = handler.completeToolCalls.get(0).toolExecutionRequest();
        assertThat(
                handler.partialToolCalls.stream().map(PartialToolCall::partialArguments).collect(joining()),
                is(complete.arguments()));
        for (PartialToolCall partial : handler.partialToolCalls) {
            assertThat(partial.id(), is(complete.id()));
            assertThat(partial.name(), is(complete.name()));
        }
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

    /**
     * llama.cpp's first delta of a tool call usually carries the id and the name only: its diff
     * writer leaves {@code arguments} out while they are still empty. langchain4j's
     * {@link PartialToolCall} refuses empty partial arguments, so that delta must not become one.
     */
    @Test
    void toolCallDeltaWithoutArguments_doesNotFailTheStream() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assertDoesNotThrow(() -> assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}]},"
                + "\"finish_reason\":null}]}"));
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("get_weather"));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).arguments(), is("{}"));
        // The header delta is withheld; the id and name it carried go with the first argument fragment.
        assertThat(handler.partialToolCalls.size(), is(1));
        assertThat(handler.partialToolCalls.get(0).index(), is(0));
        assertThat(handler.partialToolCalls.get(0).id(), is("call_1"));
        assertThat(handler.partialToolCalls.get(0).name(), is("get_weather"));
        assertThat(handler.partialToolCalls.get(0).partialArguments(), is("{}"));
        assertThat(handler.completeToolCalls.size(), is(1));
        assertPartialsMakeUpTheCompleteCall(handler);
    }

    /**
     * llama.cpp's incremental parser may discover a tool call before its name ({@code compute_diffs}
     * allows the name to go from empty to non-empty); {@link PartialToolCall} refuses a blank name.
     */
    @Test
    void toolCallDeltaBeforeItsName_doesNotFailTheStream() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assertDoesNotThrow(() -> assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"function\":{\"arguments\":\"{\\\"ci\"}}]},\"finish_reason\":null}]}"));
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"ty\\\":1}\"}}]},"
                + "\"finish_reason\":\"tool_calls\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("get_weather"));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).arguments(), is("{\"city\":1}"));
        // Every partial that is reported carries the final id and name. Which fragments are reported
        // here is not pinned: the prefix that arrives before the name is not reported at all, so the
        // partials do not make up the arguments yet (assertPartialsMakeUpTheCompleteCall is red).
        // At least one is reported, so the loop below cannot pass on an empty list.
        assertThat(handler.completeToolCalls.size(), is(1));
        assertThat(handler.partialToolCalls, is(not(empty())));
        for (PartialToolCall partial : handler.partialToolCalls) {
            assertThat(partial.id(), is("call_1"));
            assertThat(partial.name(), is("get_weather"));
        }
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
        // Call 0 is complete once the fragment of call 1 arrives, even inside one chunk.
        assertThat(handler.toolCallEvents, contains("partial:0", "complete:0", "partial:1", "complete:1"));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("first"));
        assertThat(response.aiMessage().toolExecutionRequests().get(1).name(), is("second"));
    }

    /**
     * llama.cpp only ever extends the last tool call of a message ({@code compute_diffs} in
     * {@code common/chat.cpp}), so call 0 is final once call 1 starts. langchain4j's handler contract
     * and its own OpenAI binding complete a call at that point, before the next call's first partial,
     * so that a handler can run the tool at once.
     */
    @Test
    void completesEachToolCallAsSoonAsTheNextOneStarts() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"id\":\"a\",\"function\":{\"name\":\"first\",\"arguments\":\"{\\\"x\\\":1}\"}}]},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,"
                + "\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{\\\"y\\\":2}\"}}]},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(handler.toolCallEvents, contains("partial:0", "complete:0", "partial:1", "complete:1"));
        assertThat(handler.completeToolCalls.get(0).toolExecutionRequest(),
                is(response.aiMessage().toolExecutionRequests().get(0)));
        assertThat(handler.completeToolCalls.get(1).toolExecutionRequest(),
                is(response.aiMessage().toolExecutionRequests().get(1)));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).arguments(), is("{\"x\":1}"));
        assertThat(response.aiMessage().toolExecutionRequests().get(1).arguments(), is("{\"y\":2}"));
    }

    /**
     * A call that has no name yet when the next one starts is not completed early: a
     * {@link CompleteToolCall} with a blank name would reach the handler. It is completed by
     * {@code complete()}, with the name that arrived in the meantime.
     */
    @Test
    void defersCompletionOfAToolCallThatHasNoNameYetWhenTheNextOneStarts() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,"
                + "\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}]},"
                + "\"finish_reason\":null}]}");
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                + "\"id\":\"a\",\"function\":{\"name\":\"first\"}}]},\"finish_reason\":\"tool_calls\"}]}");
        ChatResponse response = assembler.complete();

        assertThat(handler.toolCallEvents, contains("partial:1", "complete:0", "complete:1"));
        assertThat(handler.completeToolCalls.get(0).toolExecutionRequest().name(), is("first"));
        assertThat(response.aiMessage().toolExecutionRequests().get(0).name(), is("first"));
        assertThat(response.aiMessage().toolExecutionRequests().get(1).name(), is("second"));
    }

    /**
     * Arguments for a call that was already reported complete would make the final request differ
     * from the {@link CompleteToolCall} the handler got; that is a native contract violation and
     * fails the stream instead of being merged silently.
     */
    @Test
    void argumentsForAToolCallThatIsAlreadyCompleteFailTheStream() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);
        assembler.accept("{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":["
                + "{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"first\",\"arguments\":\"{\"}},"
                + "{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}"
                + "]},\"finish_reason\":null}]}");

        IllegalStateException e = assertThrows(IllegalStateException.class, () -> assembler.accept(
                "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,"
                        + "\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":null}]}"));

        assertThat(e.getMessage(), containsString("tool call 0"));
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

    /**
     * With {@code include_usage} (which the adapter requests) upstream sends the usage in a chunk of
     * its own after the finish chunk, with an empty {@code choices} array as the OpenAI spec asks,
     * carrying {@code timings} as well ({@code to_json_oaicompat_chat_stream} in
     * {@code tools/server/server-task.cpp}). The usage must be read from it, and the finish reason
     * of the chunk before must survive a chunk that has no choice.
     */
    @Test
    void usageChunkWithEmptyChoicesCarriesTokenUsage() {
        RecordingHandler handler = new RecordingHandler();
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(handler);

        assembler.accept(contentChunk("ok"));
        assembler.accept("{\"choices\":[{\"finish_reason\":\"length\",\"index\":0,\"delta\":{}}],"
                + "\"created\":1760000000,\"id\":\"chatcmpl-abc\",\"model\":\"m\","
                + "\"system_fingerprint\":\"b11639-a0e6dc738\",\"object\":\"chat.completion.chunk\"}");
        assembler.accept("{\"choices\":[],\"created\":1760000000,\"id\":\"chatcmpl-abc\",\"model\":\"m\","
                + "\"system_fingerprint\":\"b11639-a0e6dc738\",\"object\":\"chat.completion.chunk\","
                + "\"usage\":{\"completion_tokens\":3,\"prompt_tokens\":7,\"total_tokens\":10,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":0}},"
                + "\"timings\":{\"cache_n\":0,\"prompt_n\":7,\"predicted_n\":3}}");
        ChatResponse response = assembler.complete();

        assertThat(response.tokenUsage(), is(notNullValue()));
        assertThat(response.tokenUsage().inputTokenCount(), is(7));
        assertThat(response.tokenUsage().outputTokenCount(), is(3));
        // "length", not "stop": no finish reason maps to STOP, so a reset would hide behind "stop".
        assertThat(response.finishReason(), is(FinishReason.LENGTH));
        assertThat(handler.partials, contains("ok"));
        assertThat(response.aiMessage().text(), is("ok"));
    }

    @Test
    void unparseableChunkFailsLoud() {
        StreamingChunkAssembler assembler = new StreamingChunkAssembler(new RecordingHandler());

        assertThrows(UncheckedIOException.class, () -> assembler.accept("not json"));
    }
}
