// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;

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
        // output[0] is the (empty) message, output[1] is the function_call item.
        JsonNode functionCall = out.path("output").get(1);
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
