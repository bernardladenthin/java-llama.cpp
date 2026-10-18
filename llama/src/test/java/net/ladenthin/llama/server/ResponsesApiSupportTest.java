// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Unit tests for {@link ResponsesApiSupport}: the OpenAI Responses &harr; OpenAI chat request/response
 * translation (input items, instructions, function tools, function_call output items). Pure — no model.
 */
public class ResponsesApiSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    /**
     * The text parts of one input item must stay separate words. llama.cpp's own {@code /v1/responses}
     * keeps them as content parts, which are joined with a newline ({@code concat_content_parts} in
     * {@code common/chat.cpp}); concatenated without a separator, {@code "first"} and {@code "second"}
     * reached the model as {@code "firstsecond"}.
     */
    @Test
    public void requestJoinsTheTextPartsOfOneInputItemWithANewline() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":[{\"type\":\"message\","
                + "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"first\"},"
                + "{\"type\":\"input_text\",\"text\":\"second\"}]}]}"));

        assertThat(openAi.path("messages").path(0).path("content").asText(), is("first\nsecond"));
    }

    /**
     * {@code created_at} is the Unix time the response was created, as in the Responses API and in
     * llama.cpp's own {@code /v1/responses} ({@code to_json_oaicompat_resp}). It was the constant
     * {@code 0}, so clients showed every response as created on 1 January 1970.
     */
    @Test
    public void reasoningInputItemsBecomeAssistantReasoningNotUserTextLikeUpstream() throws IOException {
        // A reasoning item carries no role; read as a message it became a user turn holding the model's
        // own earlier reasoning. Upstream's convert_responses_to_chatcmpl attaches it as
        // reasoning_content to the preceding assistant message, or to a new assistant message. The
        // sequence is a tool loop as a Responses client sends it back.
        JsonNode messages = ResponsesApiSupport.toOpenAiChatRequest(
                        read(
                                "{\"input\":["
                                        + "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
                                        + "{\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"plan\"}]},"
                                        + "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                                        + "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"r\"},"
                                        + "{\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"more\"}]},"
                                        + "{\"type\":\"reasoning\",\"summary\":[],\"encrypted_content\":\"x\"},"
                                        + "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"42\"}]}]}"))
                .path("messages");
        for (JsonNode message : messages) {
            if ("user".equals(message.path("role").asText())) {
                assertThat(message.path("content").asText(), is("q"));
            }
        }
        assertThat(messages.size(), is(4));
        assertThat(messages.path(1).path("role").asText(), is("assistant"));
        assertThat(messages.path(1).path("reasoning_content").asText(), is("plan"));
        assertThat(messages.path(1).path("tool_calls").size(), is(1));
        assertThat(messages.path(2).path("role").asText(), is("tool"));
        assertThat(messages.path(3).path("role").asText(), is("assistant"));
        assertThat(messages.path(3).path("reasoning_content").asText(), is("more"));
        assertThat(messages.path(3).path("content").asText(), is("42"));
    }

    @Test
    public void functionCallOutputArrayKeepsItsPartsLikeUpstream() throws IOException {
        // A tool result may be an array of input_text / input_image parts; upstream's
        // convert_responses_to_chatcmpl turns them into text / image_url parts of the tool message.
        JsonNode messages = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":["
                        + "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                        + "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":["
                        + "{\"type\":\"input_text\",\"text\":\"sunny\"},"
                        + "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AA==\"}]}]}"))
                .path("messages");
        JsonNode tool = messages.get(1);
        assertThat(tool.path("role").asText(), is("tool"));
        assertThat(tool.path("tool_call_id").asText(), is("c1"));
        assertThat(tool.path("content").path(0).path("type").asText(), is("text"));
        assertThat(tool.path("content").path(0).path("text").asText(), is("sunny"));
        assertThat(tool.path("content").path(1).path("type").asText(), is("image_url"));
        assertThat(
                tool.path("content").path(1).path("image_url").path("url").asText(), is("data:image/png;base64,AA=="));
    }

    @Test
    public void consecutiveFunctionCallsJoinThePrecedingAssistantMessageLikeUpstream() throws IOException {
        // Parallel tool calls follow the assistant's text as separate items; upstream's
        // convert_responses_to_chatcmpl appends each to the preceding assistant message (merge_prev),
        // so the template sees one assistant turn with two tool calls.
        JsonNode messages = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":["
                        + "{\"type\":\"message\",\"role\":\"user\",\"content\":\"weather?\"},"
                        + "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"checking\"}]},"
                        + "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                        + "{\"type\":\"function_call\",\"call_id\":\"c2\",\"name\":\"g\",\"arguments\":\"{}\"},"
                        + "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"a\"},"
                        + "{\"type\":\"function_call_output\",\"call_id\":\"c2\",\"output\":\"b\"}]}"))
                .path("messages");
        assertThat(messages.size(), is(4));
        assertThat(messages.path(1).path("content").asText(), is("checking"));
        assertThat(messages.path(1).path("tool_calls").size(), is(2));
        assertThat(messages.path(1).path("tool_calls").path(1).path("id").asText(), is("c2"));
        assertThat(messages.path(2).path("role").asText(), is("tool"));
        assertThat(messages.path(3).path("role").asText(), is("tool"));
    }

    @Test
    public void anAssistantMessageJoinsThePrecedingAssistantTurnLikeUpstream() throws IOException {
        // Each earlier turn comes back as a reasoning item followed by the assistant's message item;
        // upstream's convert_responses_to_chatcmpl merges the message into the preceding assistant
        // message (merge_prev), so the template sees one turn with reasoning and answer.
        JsonNode messages = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":["
                        + "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
                        + "{\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"plan\"}]},"
                        + "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"42\"}]},"
                        + "{\"type\":\"message\",\"role\":\"user\",\"content\":\"and?\"}]}"))
                .path("messages");
        assertThat(messages.size(), is(3));
        assertThat(messages.path(1).path("role").asText(), is("assistant"));
        assertThat(messages.path(1).path("reasoning_content").asText(), is("plan"));
        assertThat(messages.path(1).path("content").asText(), is("42"));
        assertThat(messages.path(2).path("content").asText(), is("and?"));
    }

    @Test
    public void requestMapsTheReasoningEffortLikeUpstream() throws IOException {
        // Upstream's Responses converter maps reasoning.effort to reasoning_effort, which the chat
        // parser reads ("none" switches the model's thinking off).
        JsonNode openAi =
                ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":\"hi\",\"reasoning\":{\"effort\":\"none\"}}"));
        assertThat(openAi.path("reasoning_effort").asText(), is("none"));
    }

    @Test
    public void responseCarriesTheReasoningAsAReasoningItemFirstLikeUpstream() throws IOException {
        // Upstream's to_json_oaicompat_resp emits the reasoning as the first output item,
        // {"type":"reasoning","content":[{"type":"reasoning_text","text":...}],"summary":[]}.
        JsonNode output = read(ResponsesApiSupport.toResponsesResponse(
                        "{\"choices\":[{\"message\":{\"content\":\"42\",\"reasoning_content\":\"6 times 7\"}}]}",
                        "m",
                        "resp_1"))
                .path("output");
        assertThat(output.size(), is(2));
        assertThat(output.get(0).path("type").asText(), is("reasoning"));
        assertThat(output.get(0).path("content").get(0).path("type").asText(), is("reasoning_text"));
        assertThat(output.get(0).path("content").get(0).path("text").asText(), is("6 times 7"));
        assertThat(output.get(0).path("summary").isArray(), is(true));
        assertThat(output.get(1).path("type").asText(), is("message"));
    }

    @Test
    public void toolCallOnlyResponseHasNoEmptyMessageItemLikeUpstreamAndTheStream() throws IOException {
        // upstream's to_json_oaicompat_resp adds a message item only when there is text, and the
        // streamed form (ResponsesStreamTranslator) opens one only on the first text delta; an empty
        // message beside the function calls made the blocking answer differ from both.
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\","
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\","
                + "\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}";
        JsonNode output = read(ResponsesApiSupport.toResponsesResponse(openAi, "m", "resp_1"))
                .path("output");
        assertThat(output.size(), is(1));
        assertThat(output.get(0).path("type").asText(), is("function_call"));
    }

    @ParameterizedTest
    @CsvSource(
            delimiter = '|',
            value = {
                "top_k|40",
                "seed|7",
                "presence_penalty|0.2",
                "frequency_penalty|0.3",
                "stop|[\"END\"]",
                "chat_template_kwargs|{\"enable_thinking\":false}",
                "response_format|{\"type\":\"json_object\"}",
                "cache_prompt|false",
                "n_cache_reuse|64",
                "id_slot|1",
                "min_p|0.05",
                "typical_p|0.9",
                "repeat_penalty|1.1",
                "repeat_last_n|32",
                "top_n_sigma|1.5",
                "xtc_probability|0.3",
                "xtc_threshold|0.2",
                "dynatemp_range|0.4",
                "dynatemp_exponent|1.2",
                "min_keep|2",
                "n_keep|8",
                "ignore_eos|true",
                "grammar|\"root ::= x\"",
                "json_schema|{\"type\":\"object\"}",
                "n_predict|16",
                "reasoning_budget_tokens|100",
                "thinking_budget_tokens|100"
            })
    public void requestFieldsTheChatMapperReadsPassThroughLikeUpstream(String field, String json) throws IOException {
        // upstream's convert_responses_to_chatcmpl copies the whole body into the chat request; the
        // shared chat mapper reads these fields, so dropping them here changed the sampling silently.
        JsonNode value = read(json);
        com.fasterxml.jackson.databind.node.ObjectNode request =
                (com.fasterxml.jackson.databind.node.ObjectNode) read("{\"model\":\"m\",\"input\":\"hi\"}");
        request.set(field, value);
        assertThat(ResponsesApiSupport.toOpenAiChatRequest(request).path(field), is(value));
    }

    @Test
    public void previousResponseIdIsRejectedLikeUpstream() throws IOException {
        // There is no response store: answering a turn that continues by previous_response_id without
        // its history would silently answer something else. upstream rejects it with a 400.
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> ResponsesApiSupport.toOpenAiChatRequest(
                        read("{\"model\":\"m\",\"input\":\"and then?\",\"previous_response_id\":\"resp_1\"}")));
        assertThat(e.getMessage(), containsString("previous_response_id"));
    }

    @Test
    public void responseCarriesTheCurrentUnixTimeAsCreatedAt() throws IOException {
        long before = System.currentTimeMillis() / 1000L;

        JsonNode response = read(ResponsesApiSupport.toResponsesResponse("{}", "m", "resp_1"));

        long createdAt = response.path("created_at").asLong();
        long after = System.currentTimeMillis() / 1000L;
        assertThat(createdAt, both(greaterThanOrEqualTo(before)).and(lessThanOrEqualTo(after)));
    }

    @Test
    public void isStreamingDefaultsFalse() throws IOException {
        assertThat(ResponsesApiSupport.isStreaming(read("{}")), is(false));
        assertThat(ResponsesApiSupport.isStreaming(read("{\"stream\":true}")), is(true));
    }

    @Test
    public void requestMapsInstructionsStringInputAndTools() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"instructions\":\"be brief\",\"input\":\"hi\",\"max_output_tokens\":32,"
                + "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]}"));
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("system"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("be brief"));
        assertThat(openAi.path("messages").get(1).path("role").asText(), is("user"));
        assertThat(openAi.path("messages").get(1).path("content").asText(), is("hi"));
        // max_output_tokens -> max_tokens.
        assertThat(openAi.path("max_tokens").asInt(), is(32));
        // Responses function tool (flat) -> OpenAI nested function tool.
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("f"));
        assertThat(
                openAi.path("tools")
                        .get(0)
                        .path("function")
                        .path("parameters")
                        .path("type")
                        .asText(),
                is("object"));
    }

    @Test
    public void requestMapsInputArrayMessageAndFunctionCallItems() throws IOException {
        String responses = "{\"input\":["
                + "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"q\"}]},"
                + "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                + "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"ok\"}]}";
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read(responses));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("q"));
        // function_call -> assistant tool_calls
        JsonNode toolCall = openAi.path("messages").get(1).path("tool_calls").get(0);
        assertThat(toolCall.path("id").asText(), is("c1"));
        assertThat(toolCall.path("function").path("name").asText(), is("f"));
        // function_call_output -> role:"tool" message
        assertThat(openAi.path("messages").get(2).path("role").asText(), is("tool"));
        assertThat(openAi.path("messages").get(2).path("tool_call_id").asText(), is("c1"));
        assertThat(openAi.path("messages").get(2).path("content").asText(), is("ok"));
    }

    @Test
    public void requestTurnsInputImagePartsIntoImageUrlParts() throws IOException {
        String responses = "{\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":["
                + "{\"type\":\"input_text\",\"text\":\"what is this?\"},"
                + "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,iVBORw0K\"}]}]}";
        JsonNode content = ResponsesApiSupport.toOpenAiChatRequest(read(responses))
                .path("messages")
                .get(0)
                .path("content");
        assertThat(content.isArray(), is(true));
        assertThat(content.get(0).path("type").asText(), is("text"));
        assertThat(content.get(0).path("text").asText(), is("what is this?"));
        assertThat(content.get(1).path("type").asText(), is("image_url"));
        assertThat(content.get(1).path("image_url").path("url").asText(), is("data:image/png;base64,iVBORw0K"));
    }

    @Test
    public void requestWithoutInstructionsHasNoSystemMessage() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":\"hi\"}"));
        // First (and only) message is the user input — no leading system message.
        assertThat(openAi.path("messages").size(), is(1));
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("user"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("hi"));
    }

    @Test
    public void requestMapsAssistantMessageItemAndSkipsNonFunctionTools() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"input\":["
                + "{\"type\":\"message\",\"role\":\"assistant\","
                + "\"content\":[{\"type\":\"output_text\",\"text\":\"prior\"}]}],"
                + "\"tools\":[{\"type\":\"web_search\"},{\"type\":\"function\",\"name\":\"f\"}]}"));
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("assistant"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("prior"));
        // Only the function tool survives; the non-function (web_search) tool is dropped.
        assertThat(openAi.path("tools").size(), is(1));
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("f"));
    }

    @Test
    public void responseWrapsOutputMessageWithOutputTextAndUsage() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},"
                + "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":3}}}";
        JsonNode out = read(ResponsesApiSupport.toResponsesResponse(openAi, "m", "resp_1"));
        assertThat(out.path("object").asText(), is("response"));
        assertThat(out.path("status").asText(), is("completed"));
        JsonNode messageItem = out.path("output").get(0);
        assertThat(messageItem.path("type").asText(), is("message"));
        assertThat(messageItem.path("content").get(0).path("type").asText(), is("output_text"));
        assertThat(messageItem.path("content").get(0).path("text").asText(), is("hello"));
        assertThat(out.path("usage").path("input_tokens").asInt(), is(4));
        assertThat(out.path("usage").path("total_tokens").asInt(), is(5));
        assertThat(
                out.path("usage")
                        .path("input_tokens_details")
                        .path("cached_tokens")
                        .asInt(),
                is(3));
    }

    @Test
    public void responseEmitsFunctionCallItemsForToolCalls() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\","
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\","
                + "\"arguments\":\"{\\\"a\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}";
        JsonNode out = read(ResponsesApiSupport.toResponsesResponse(openAi, "m", "resp_1"));
        // The function_call item is the only output item: upstream adds a message item only for text.
        JsonNode functionCall = out.path("output").get(0);
        assertThat(functionCall.path("type").asText(), is("function_call"));
        assertThat(functionCall.path("call_id").asText(), is("c1"));
        assertThat(functionCall.path("name").asText(), is("f"));
        assertThat(functionCall.path("arguments").asText(), is("{\"a\":1}"));
    }

    @Test
    public void requestForwardsToolChoiceAndParallelToolCalls() throws IOException {
        // The Responses API uses the same tool_choice + parallel_tool_calls fields as chat.
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\",\"input\":\"hi\","
                + "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}],"
                + "\"tool_choice\":\"required\",\"parallel_tool_calls\":false}"));
        assertThat(openAi.path("tool_choice").asText(), is("required"));
        assertThat(openAi.path("parallel_tool_calls").isBoolean(), is(true));
        assertThat(openAi.path("parallel_tool_calls").asBoolean(), is(false));
    }

    @Test
    public void requestOmitsToolChoiceAndParallelToolCallsWhenAbsent() throws IOException {
        JsonNode openAi = ResponsesApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\",\"input\":\"hi\","
                + "\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]}"));
        assertThat(openAi.has("tool_choice"), is(false));
        assertThat(openAi.has("parallel_tool_calls"), is(false));
    }
}
