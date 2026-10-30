// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Arrays;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ToolCall;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ChatStreamAssembler}. No JVM native library or model file needed — the
 * streams are string literals in the shapes llama.cpp's server, {@code OpenAiCompatServer} and
 * OpenRouter send.
 */
public class ChatStreamAssemblerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String DONE = "data: [DONE]\n\n";

    private static String event(String json) {
        return "data: " + json + "\n\n";
    }

    private static String token(String content) {
        return event("{\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"content\":" + content + "},\"finish_reason\":null}],\"usage\":null}");
    }

    private static JsonNode completion(ChatStreamAssembler assembler) throws Exception {
        return MAPPER.readTree(assembler.toCompletionJson());
    }

    private static ChatStreamAssembler fed(String... chunks) {
        ChatStreamAssembler assembler = new ChatStreamAssembler();
        for (String chunk : chunks) {
            assembler.acceptChunk(chunk);
        }
        return assembler;
    }

    // ----- parseStream -----

    /** llama.cpp's stream with {@code include_usage}: role chunk, tokens, finish chunk with timings, usage chunk. */
    @Test
    public void parseStream_llamaServerStream_isTheNonStreamingAnswer() {
        String sse = event("{\"id\":\"chatcmpl-1\",\"created\":7,\"model\":\"m\",\"object\":\"chat.completion.chunk\","
                        + "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},"
                        + "\"finish_reason\":null}]}")
                + token("\"Hel\"")
                + token("\"lo.\"")
                + event("{\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],"
                        + "\"timings\":{\"prompt_n\":5,\"predicted_n\":2}}")
                + event("{\"id\":\"chatcmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,"
                        + "\"total_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":3}}}")
                + DONE;

        ChatResponse response = ChatStreamAssembler.parseStream(sse);

        assertThat(response.getId(), is("chatcmpl-1"));
        assertThat(response.getFirstContent(), is("Hello."));
        assertThat(response.getChoices().get(0).getFinishReason(), is("length"));
        assertThat(response.getChoices().get(0).getMessage().getRole(), is("assistant"));
        assertThat(response.getUsage().getPromptTokens(), is(5L));
        assertThat(response.getUsage().getCompletionTokens(), is(2L));
        assertThat(response.getUsage().getCachedTokens(), is(3L));
        assertThat(response.getTimings().getPredictedN(), is(2L));
    }

    /** Heartbeat comments, other fields, CRLF and CR line ends, and {@code data:} without the space. */
    @Test
    public void parseStream_commentsFieldsAndLineEnds_areSkipped() {
        String sse = ": ping\r\n\r\n"
                + "event: message\r\nid: 7\r\nretry: 100\r\n"
                + "data:{\"choices\":[{\"delta\":{\"content\":\"o\"}}]}\r\n\r\n"
                + "data: {\"choices\":[{\"delta\":{\"content\":\"k\"},\"finish_reason\":\"stop\"}]}\r\r"
                + ": ping\n\n"
                + "data: [DONE]";
        ChatResponse response = ChatStreamAssembler.parseStream(sse);
        assertThat(response.getFirstContent(), is("ok"));
        assertThat(response.getChoices().get(0).getFinishReason(), is("stop"));
    }

    @Test
    public void parseStream_lfOnlyAndCrOnlyLineEnds_eachEndALine() {
        String lf = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\ndata: [DONE]";
        String cr = "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\rdata: [DONE]\r";
        assertThat(ChatStreamAssembler.parseStream(lf).getFirstContent(), is("a"));
        assertThat(ChatStreamAssembler.parseStream(cr).getFirstContent(), is("b"));
    }

    @Test
    public void parseStream_linesAfterDone_areNotRead() {
        ChatResponse response = ChatStreamAssembler.parseStream(token("\"kept\"") + DONE + "data: not json\n\n");
        assertThat(response.getFirstContent(), is("kept"));
    }

    @Test
    public void parseStream_withoutDone_failsInsteadOfPassingForAShortAnswer() {
        String sse = token("\"half\"");
        ChatStreamAssembler.InvalidStreamException thrown = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class, () -> ChatStreamAssembler.parseStream(sse));
        assertThat(
                thrown.getMessage(),
                is("the stream ended without 'data: [DONE]' after " + sse.length() + " characters"));
    }

    @Test
    public void parseStream_aPlainJsonAnswer_failsForWantOfDone() {
        // a server that ignored stream:true
        assertThrows(
                ChatStreamAssembler.InvalidStreamException.class,
                () -> ChatStreamAssembler.parseStream("{\"choices\":[{\"message\":{\"content\":\"x\"}}]}"));
    }

    // ----- acceptLine -----

    @Test
    public void acceptLine_reportsTheEndAndIgnoresEverythingAfterIt() {
        ChatStreamAssembler assembler = new ChatStreamAssembler();
        assertThat(assembler.acceptLine("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}"), is(false));
        assertThat(assembler.isDone(), is(false));
        assertThat(assembler.acceptLine(": ping"), is(false));
        assertThat(assembler.acceptLine("data: [DONE]"), is(true));
        assertThat(assembler.isDone(), is(true));
        // not JSON, and still no exception: nothing after [DONE] is read
        assertThat(assembler.acceptLine("data: {broken"), is(true));
        assertThat(assembler.toResponse().getFirstContent(), is("a"));
    }

    @Test
    public void acceptLine_emptyData_isSkipped() throws Exception {
        ChatStreamAssembler assembler = new ChatStreamAssembler();
        assertThat(assembler.acceptLine("data:"), is(false));
        assertThat(assembler.acceptLine("data:   "), is(false));
        assertThat(completion(assembler).path("choices").size(), is(0));
    }

    // ----- acceptChunk -----

    @Test
    public void acceptChunk_notJson_failsNamingTheChunk() {
        ChatStreamAssembler.InvalidStreamException thrown = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class, () -> new ChatStreamAssembler().acceptChunk("{oops"));
        assertThat(thrown.getMessage(), is("a stream chunk is not JSON: '{oops'"));
    }

    @Test
    public void acceptChunk_notAnObject_fails() {
        ChatStreamAssembler.InvalidStreamException thrown = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class, () -> new ChatStreamAssembler().acceptChunk("[1]"));
        assertThat(thrown.getMessage(), is("a stream chunk is not a JSON object: '[1]'"));
    }

    @Test
    public void acceptChunk_errorChunk_fails() {
        ChatStreamAssembler assembler = fed("{\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}");
        ChatStreamAssembler.InvalidStreamException thrown = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class,
                () -> assembler.acceptChunk("{\"error\":{\"message\":\"Provider returned error\",\"code\":502}}"));
        assertThat(
                thrown.getMessage(),
                is("the stream reported an error: {\"message\":\"Provider returned error\",\"code\":502}"));
    }

    @Test
    public void acceptChunk_aNullError_isNoError() {
        assertThat(
                fed("{\"error\":null,\"choices\":[{\"delta\":{\"content\":\"fine\"}}]}")
                        .toResponse()
                        .getFirstContent(),
                is("fine"));
    }

    @Test
    public void acceptChunk_aLongRejectedText_isQuotedByABoundedExcerpt() {
        char[] exact = new char[200];
        Arrays.fill(exact, 'x');
        String twoHundred = new String(exact);
        ChatStreamAssembler.InvalidStreamException atLimit = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class,
                () -> new ChatStreamAssembler().acceptChunk(twoHundred));
        assertThat(atLimit.getMessage(), is("a stream chunk is not JSON: '" + twoHundred + "'"));
        ChatStreamAssembler.InvalidStreamException overLimit = assertThrows(
                ChatStreamAssembler.InvalidStreamException.class,
                () -> new ChatStreamAssembler().acceptChunk(twoHundred + "y"));
        assertThat(overLimit.getMessage(), is("a stream chunk is not JSON: '" + twoHundred + "...'"));
    }

    @Test
    public void acceptChunk_nullDeltasAndALaterNullFinishReason_changeNothing() {
        ChatResponse response = fed(
                        "{\"choices\":[{\"delta\":{\"content\":\"a\"}}]}",
                        "{\"choices\":[{\"delta\":{\"content\":null,\"role\":null},\"finish_reason\":null}]}",
                        "{\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}",
                        "{\"choices\":[{\"delta\":{},\"finish_reason\":null}]}")
                .toResponse();
        assertThat(response.getFirstContent(), is("ab"));
        assertThat(response.getChoices().get(0).getMessage().getRole(), is("assistant"));
        assertThat(response.getChoices().get(0).getFinishReason(), is("stop"));
    }

    @Test
    public void acceptChunk_theFirstIdCreatedAndModelWin() throws Exception {
        JsonNode json = completion(fed(
                "{\"choices\":[]}",
                "{\"id\":\"first\",\"created\":1,\"model\":\"m1\",\"choices\":[]}",
                "{\"id\":\"second\",\"created\":2,\"model\":\"m2\",\"choices\":[]}"));
        assertThat(json.path("id").asText(), is("first"));
        assertThat(json.path("created").asInt(), is(1));
        assertThat(json.path("model").asText(), is("m1"));
    }

    @Test
    public void acceptChunk_theLastUsageAndTimingsObjectsWin_andALaterNullKeepsThem() throws Exception {
        JsonNode json = completion(fed(
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":1},\"timings\":{\"predicted_n\":1}}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":2},\"timings\":{\"predicted_n\":2}}",
                "{\"choices\":[],\"usage\":null,\"timings\":null}"));
        assertThat(json.path("usage").path("prompt_tokens").asInt(), is(2));
        assertThat(json.path("timings").path("predicted_n").asInt(), is(2));
    }

    @Test
    public void acceptChunk_reasoningContent_isJoinedIntoTheMessage() throws Exception {
        ChatStreamAssembler assembler = fed(
                "{\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}",
                "{\"choices\":[{\"delta\":{\"reasoning_content\":\"hard\"}}]}",
                "{\"choices\":[{\"delta\":{\"content\":\"42\"},\"finish_reason\":\"stop\"}]}");
        JsonNode message = completion(assembler).path("choices").get(0).path("message");
        assertThat(message.path("reasoning_content").asText(), is("think hard"));
        assertThat(message.path("content").asText(), is("42"));
        assertThat(
                new ChatResponseParser().extractChoiceReasoningContent(assembler.toCompletionJson()), is("think hard"));
    }

    @Test
    public void acceptChunk_toolCallDeltas_areRebuiltIntoWholeCalls() {
        ChatResponse response = fed(
                        "{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","
                                + "\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}",
                        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}",
                        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}")
                .toResponse();
        assertThat(response.getChoices().get(0).getFinishReason(), is("tool_calls"));
        ToolCall call = response.getChoices().get(0).getMessage().getToolCalls().get(0);
        assertThat(call.getId(), is("call_1"));
        assertThat(call.getName(), is("get_weather"));
        assertThat(call.getArgumentsJson(), is("{\"city\":\"Paris\"}"));
    }

    @Test
    public void acceptChunk_severalChoices_areKeptApartInIndexOrder() throws Exception {
        JsonNode choices = completion(fed(
                        "{\"choices\":[{\"index\":1,\"delta\":{\"content\":\"B\"}},{\"index\":0,\"delta\":{\"content\":\"A\"}}]}",
                        "{\"choices\":[{\"index\":1,\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}"))
                .path("choices");
        assertThat(choices.size(), is(2));
        assertThat(choices.get(0).path("index").asInt(), is(0));
        assertThat(choices.get(0).path("message").path("content").asText(), is("A"));
        assertThat(choices.get(0).path("finish_reason").isNull(), is(true));
        assertThat(choices.get(1).path("index").asInt(), is(1));
        assertThat(choices.get(1).path("message").path("content").asText(), is("Bb"));
        assertThat(choices.get(1).path("finish_reason").asText(), is("stop"));
    }

    // ----- toCompletionJson -----

    @Test
    public void toCompletionJson_aBareStream_carriesOnlyWhatItSaw() throws Exception {
        JsonNode json = completion(fed("{\"choices\":[{\"delta\":{\"content\":\"x\"}}]}"));
        assertThat(json.path("object").asText(), is("chat.completion"));
        assertThat(json.has("id"), is(false));
        assertThat(json.has("created"), is(false));
        assertThat(json.has("model"), is(false));
        assertThat(json.has("usage"), is(false));
        assertThat(json.has("timings"), is(false));
        JsonNode choice = json.path("choices").get(0);
        assertThat(choice.path("index").asInt(), is(0));
        assertThat(choice.path("finish_reason").isNull(), is(true));
        assertThat(choice.path("message").has("reasoning_content"), is(false));
        assertThat(choice.path("message").has("tool_calls"), is(false));
        assertThat(choice.path("message").path("content").asText(), is("x"));
    }

    @Test
    public void toResponse_ofAnEmptyAssembler_hasNoChoices() {
        ChatResponse response = new ChatStreamAssembler().toResponse();
        assertThat(response.getChoices().isEmpty(), is(true));
        assertThat(response.getUsage().getPromptTokens(), is(0L));
    }

    @Test
    public void invalidStreamException_keepsItsCause() {
        IllegalStateException cause = new IllegalStateException("x");
        assertThat(new ChatStreamAssembler.InvalidStreamException("m", cause).getCause(), is(cause));
        assertThat(new ChatStreamAssembler.InvalidStreamException("m", null).getCause(), is(nullValue()));
    }
}
