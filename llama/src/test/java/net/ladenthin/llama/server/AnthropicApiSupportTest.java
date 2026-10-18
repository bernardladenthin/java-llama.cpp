// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link AnthropicApiSupport}: the Anthropic Messages &harr; OpenAI chat request/response
 * translation (content blocks, tool_use/tool_result, tools, stop reasons) and the SSE event builders.
 * Pure — no model.
 */
public class AnthropicApiSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    @Test
    public void requestTurnsImageBlocksIntoImageUrlParts() throws IOException {
        // Anthropic clients send images as {type:image, source:{type:base64, media_type, data}}. The
        // OpenAI parser reads images only from image_url parts, so dropping the block left a vision
        // model answering as if no image had been sent.
        String anthropic = "{\"messages\":[{\"role\":\"user\",\"content\":["
                + "{\"type\":\"text\",\"text\":\"what is this?\"},"
                + "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\","
                + "\"data\":\"iVBORw0K\"}}]}]}";
        JsonNode content = AnthropicApiSupport.toOpenAiChatRequest(read(anthropic))
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
    public void responseCarriesTheReasoningAsAThinkingBlockFirstLikeUpstream() throws IOException {
        // Upstream's to_json_anthropic puts the reasoning first, as {"type":"thinking","thinking":...,
        // "signature":""}; a client that enabled thinking (F76) otherwise never sees it.
        JsonNode response = read(AnthropicApiSupport.toAnthropicResponse(
                "{\"choices\":[{\"message\":{\"content\":\"42\",\"reasoning_content\":\"6 times 7\"},"
                        + "\"finish_reason\":\"stop\"}]}",
                "m"));
        JsonNode content = response.path("content");
        assertThat(content.size(), is(2));
        assertThat(content.get(0).path("type").asText(), is("thinking"));
        assertThat(content.get(0).path("thinking").asText(), is("6 times 7"));
        assertThat(content.get(0).path("signature").isTextual(), is(true));
        assertThat(content.get(1).path("type").asText(), is("text"));
        assertThat(content.get(1).path("text").asText(), is("42"));
    }

    @Test
    public void requestKeepsThinkingBlocksOfTheHistoryAsReasoningContentLikeUpstream() throws IOException {
        // Upstream's convert_anthropic_to_oai collects an assistant turn's thinking blocks into
        // reasoning_content, which the chat parser hands to templates that render earlier reasoning.
        JsonNode messages = AnthropicApiSupport.toOpenAiChatRequest(read("{\"messages\":["
                        + "{\"role\":\"user\",\"content\":\"q\"},"
                        + "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"6 times 7\","
                        + "\"signature\":\"\"},{\"type\":\"text\",\"text\":\"42\"}]},"
                        + "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"only\"}]}]}"))
                .path("messages");
        assertThat(messages.get(1).path("reasoning_content").asText(), is("6 times 7"));
        assertThat(messages.get(1).path("content").asText(), is("42"));
        assertThat(messages.get(2).path("reasoning_content").asText(), is("only"));
        assertThat(messages.get(2).path("content").asText(), is(""));
    }

    @Test
    public void requestMapsAnEnabledThinkingBudgetLikeUpstream() throws IOException {
        // Upstream's convert_anthropic_to_oai turns thinking.budget_tokens into thinking_budget_tokens,
        // which the chat parser hands to the sampler as the reasoning budget.
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read("{\"max_tokens\":64,"
                + "\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024},"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"));
        assertThat(openAi.path("thinking_budget_tokens").asInt(), is(1024));
    }

    /**
     * Separate text blocks of one message must stay separate words. Anthropic clients split a turn into
     * several blocks (a reminder block, then the user's own text); llama.cpp's own {@code /v1/messages}
     * keeps them as content parts and joins them with a newline ({@code concat_content_parts} in
     * {@code common/chat.cpp}). Concatenated without a separator, {@code "first"} and {@code "second"}
     * reached the model as {@code "firstsecond"}.
     */
    @Test
    public void requestJoinsTheTextBlocksOfOneMessageWithANewline() throws IOException {
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read("{\"messages\":[{\"role\":\"user\","
                + "\"content\":[{\"type\":\"text\",\"text\":\"first\"},{\"type\":\"text\",\"text\":\"second\"}]}]}"));

        assertThat(openAi.path("messages").path(0).path("content").asText(), is("first\nsecond"));
    }

    @Test
    public void isStreamingDefaultsFalse() throws IOException {
        assertThat(AnthropicApiSupport.isStreaming(read("{}")), is(false));
        assertThat(AnthropicApiSupport.isStreaming(read("{\"stream\":true}")), is(true));
    }

    @Test
    public void requestMapsSystemMessagesToolsAndSampling() throws IOException {
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\",\"max_tokens\":64,"
                + "\"system\":\"be brief\",\"temperature\":0.3,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"tools\":[{\"name\":\"get_weather\",\"description\":\"d\","
                + "\"input_schema\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"auto\"}}"));
        // system becomes the first OpenAI message.
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("system"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("be brief"));
        assertThat(openAi.path("messages").get(1).path("content").asText(), is("hi"));
        assertThat(openAi.path("max_tokens").asInt(), is(64));
        assertThat(openAi.path("temperature").asDouble(), is(0.3));
        // Anthropic tool input_schema -> OpenAI function parameters.
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("get_weather"));
        assertThat(
                openAi.path("tools")
                        .get(0)
                        .path("function")
                        .path("parameters")
                        .path("type")
                        .asText(),
                is("object"));
        assertThat(openAi.path("tool_choice").asText(), is("auto"));
    }

    @Test
    public void requestFlattensToolUseAndToolResultBlocks() throws IOException {
        String anthropic = "{\"messages\":["
                + "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"c1\","
                + "\"name\":\"get_weather\",\"input\":{\"city\":\"Paris\"}}]},"
                + "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"c1\","
                + "\"content\":\"sunny\"}]}]}";
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read(anthropic));
        // assistant tool_use -> OpenAI tool_calls with arguments as a JSON string.
        JsonNode toolCall = openAi.path("messages").get(0).path("tool_calls").get(0);
        assertThat(toolCall.path("id").asText(), is("c1"));
        assertThat(toolCall.path("function").path("arguments").isTextual(), is(true));
        assertThat(
                read(toolCall.path("function").path("arguments").asText())
                        .path("city")
                        .asText(),
                is("Paris"));
        // user tool_result -> separate OpenAI role:"tool" message.
        JsonNode toolMessage = openAi.path("messages").get(1);
        assertThat(toolMessage.path("role").asText(), is("tool"));
        assertThat(toolMessage.path("tool_call_id").asText(), is("c1"));
        assertThat(toolMessage.path("content").asText(), is("sunny"));
    }

    @Test
    public void requestConcatenatesSystemBlocksAndMapsStopSequences() throws IOException {
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(
                read("{\"system\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}],"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                        + "\"stop_sequences\":[\"X\",\"Y\"]}"));
        // system blocks are concatenated into one system message.
        assertThat(openAi.path("messages").get(0).path("role").asText(), is("system"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("ab"));
        // stop_sequences -> OpenAI stop.
        assertThat(openAi.path("stop").size(), is(2));
        assertThat(openAi.path("stop").get(0).asText(), is("X"));
    }

    @Test
    public void toolChoiceAnyMapsToRequired() throws IOException {
        JsonNode openAi =
                AnthropicApiSupport.toOpenAiChatRequest(read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                        + "\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}],"
                        + "\"tool_choice\":{\"type\":\"any\"}}"));
        assertThat(openAi.path("tool_choice").asText(), is("required"));
    }

    @Test
    public void toolResultOnlyUserMessageEmitsOnlyToolMessage() throws IOException {
        // A user turn that carries only tool_result blocks must become exactly one role:"tool"
        // message — not a tool message plus a spurious empty user message.
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(
                read("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
                        + "\"tool_use_id\":\"c1\",\"content\":[{\"type\":\"text\",\"text\":\"su\"},"
                        + "{\"type\":\"text\",\"text\":\"nny\"}]}]}]}"));
        assertThat(openAi.path("messages").size(), is(1));
        JsonNode toolMessage = openAi.path("messages").get(0);
        assertThat(toolMessage.path("role").asText(), is("tool"));
        assertThat(toolMessage.path("tool_call_id").asText(), is("c1"));
        // tool_result content blocks are flattened to text.
        assertThat(toolMessage.path("content").asText(), is("sunny"));
    }

    @Test
    public void responseEmitsTextAndToolUseBlocksAndStopReason() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\","
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\","
                + "\"arguments\":\"{\\\"a\\\":1}\"}}]},\"finish_reason\":\"tool_calls\"}],"
                + "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,"
                + "\"prompt_tokens_details\":{\"cached_tokens\":3}}}";
        JsonNode out = read(AnthropicApiSupport.toAnthropicResponse(openAi, "m"));
        assertThat(out.path("type").asText(), is("message"));
        assertThat(out.path("role").asText(), is("assistant"));
        assertThat(out.path("content").get(0).path("type").asText(), is("text"));
        assertThat(out.path("content").get(0).path("text").asText(), is("hi"));
        JsonNode toolUse = out.path("content").get(1);
        assertThat(toolUse.path("type").asText(), is("tool_use"));
        assertThat(toolUse.path("name").asText(), is("f"));
        assertThat(toolUse.path("input").path("a").asInt(), is(1));
        // finish_reason "tool_calls" -> stop_reason "tool_use".
        assertThat(out.path("stop_reason").asText(), is("tool_use"));
        assertThat(out.path("usage").path("input_tokens").asInt(), is(2));
        assertThat(out.path("usage").path("cache_read_input_tokens").asInt(), is(3));
        assertThat(out.path("usage").path("output_tokens").asInt(), is(2));
    }

    @Test
    public void toAnthropicResponseTurnsEmptyToolCallArgumentsIntoAnEmptyInputObject() throws IOException {
        // a parameterless call can come back with "arguments": "" -- tool_use.input must be an object
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,"
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"get_time\","
                + "\"arguments\":\"\"}}]},\"finish_reason\":\"tool_calls\"}]}";
        JsonNode out = read(AnthropicApiSupport.toAnthropicResponse(openAi, "m"));
        JsonNode input = out.path("content").get(0).path("input");
        assertThat(input.isObject(), is(true));
        assertThat(input.size(), is(0));
    }

    @Test
    public void stopReasonMapping() {
        assertThat(AnthropicApiSupport.anthropicStopReason("stop"), is("end_turn"));
        assertThat(AnthropicApiSupport.anthropicStopReason("length"), is("max_tokens"));
        assertThat(AnthropicApiSupport.anthropicStopReason("tool_calls"), is("tool_use"));
    }

    @Test
    public void sseEventBuildersAreWellFormed() throws IOException {
        String start = AnthropicApiSupport.messageStartEvent("msg_1", "m");
        assertThat(start.startsWith("event: message_start\ndata: "), is(true));
        assertThat(
                read(start.substring(start.indexOf('{')))
                        .path("message")
                        .path("role")
                        .asText(),
                is("assistant"));
        assertThat(AnthropicApiSupport.messageStopEvent().startsWith("event: message_stop"), is(true));
    }

    @Test
    public void requestMapsDisableParallelToolUseToParallelToolCallsFalse() throws IOException {
        // Anthropic tool_choice.disable_parallel_tool_use=true -> OpenAI parallel_tool_calls=false.
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"tools\":[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"}}],"
                + "\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":true}}"));
        assertThat(openAi.path("parallel_tool_calls").isBoolean(), is(true));
        assertThat(openAi.path("parallel_tool_calls").asBoolean(), is(false));
    }

    @Test
    public void requestOmitsParallelToolCallsWhenParallelToolUseAllowed() throws IOException {
        // disable_parallel_tool_use absent -> default (parallel allowed) -> no override emitted.
        JsonNode openAi = AnthropicApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"tools\":[{\"name\":\"get_weather\",\"input_schema\":{\"type\":\"object\"}}],"
                + "\"tool_choice\":{\"type\":\"auto\"}}"));
        assertThat(openAi.has("parallel_tool_calls"), is(false));
    }
}
