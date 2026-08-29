// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.value.ChatChoice;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import net.ladenthin.llama.value.Timings;
import net.ladenthin.llama.value.ToolCall;
import net.ladenthin.llama.value.ToolDefinition;
import net.ladenthin.llama.value.Usage;
import org.junit.jupiter.api.Test;

/** Model-free tests for the pure langchain4j&lt;-&gt;java-llama.cpp transforms. */
class LangChain4jMappingTest {

    @Test
    void mapsEveryRoleAndContent() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(
                                SystemMessage.from("you are terse"),
                                UserMessage.from("hi"),
                                AiMessage.from("hello"),
                                ToolExecutionResultMessage.from("call_1", "search", "42"))
                        .build();

        List<ChatMessage> messages = LangChain4jMapping.toJllamaRequest(request).getMessages();

        List<String> roles = new ArrayList<>();
        List<String> contents = new ArrayList<>();
        for (ChatMessage message : messages) {
            roles.add(message.getRole());
            contents.add(message.getContent());
        }
        assertThat(roles, contains("system", "user", "assistant", "tool"));
        assertThat(contents, contains("you are terse", "hi", "hello", "42"));
    }

    @Test
    void flattensMultiTextUserMessageToText() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(UserMessage.from(TextContent.from("Hello "), TextContent.from("world")))
                        .build();

        ChatMessage mapped = LangChain4jMapping.toJllamaRequest(request).getMessages().get(0);

        assertThat(mapped.getRole(), is("user"));
        assertThat(mapped.getContent(), is("Hello world"));
        // Text-only content needs no multimodal array form.
        assertThat(mapped.hasParts(), is(false));
    }

    @Test
    void appliesSamplingParametersToInferenceJson() {
        ChatRequest request =
                ChatRequest.builder()
                        .messages(UserMessage.from("hi"))
                        .temperature(0.3)
                        .topK(40)
                        .maxOutputTokens(64)
                        .frequencyPenalty(0.5)
                        .presencePenalty(0.25)
                        .stopSequences(Arrays.asList("STOP"))
                        .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        assertThat(json, containsString("\"temperature\""));
        assertThat(json, containsString("\"top_k\""));
        assertThat(json, containsString("\"n_predict\""));
        assertThat(json, containsString("\"frequency_penalty\""));
        assertThat(json, containsString("\"presence_penalty\""));
        assertThat(json, containsString("\"stop\""));
        // Messages must survive into the streaming parameter blob too.
        assertThat(json, containsString("hi"));
    }

    @Test
    void mapsFinishReasonStrings() {
        assertThat(LangChain4jMapping.toFinishReason("stop"), is(FinishReason.STOP));
        assertThat(LangChain4jMapping.toFinishReason("length"), is(FinishReason.LENGTH));
        assertThat(LangChain4jMapping.toFinishReason("tool_calls"), is(FinishReason.TOOL_EXECUTION));
        assertThat(LangChain4jMapping.toFinishReason("content_filter"), is(FinishReason.CONTENT_FILTER));
        assertThat(LangChain4jMapping.toFinishReason("something_new"), is(FinishReason.OTHER));
        // No choices / absent reason is the normal terminal state.
        assertThat(LangChain4jMapping.toFinishReason(null), is(FinishReason.STOP));
    }

    @Test
    void rerankScoresAlignToInputOrderNotResponseOrder() {
        // Native results arrive out of order; "index" is the input position.
        String json =
                "[{\"document\":\"b\",\"index\":1,\"score\":0.9},"
                        + "{\"document\":\"a\",\"index\":0,\"score\":0.1}]";

        double[] scores = LangChain4jMapping.parseRerankScores(json, 2);

        assertThat(scores.length, is(2));
        assertThat(scores[0], is(0.1));
        assertThat(scores[1], is(0.9));
    }

    @Test
    void rerankScoresDefaultToZeroForMissingEntries() {
        double[] scores = LangChain4jMapping.parseRerankScores("[]", 3);

        assertThat(scores.length, is(3));
        assertThat(scores[0], is(0.0));
        assertThat(scores[1], is(0.0));
        assertThat(scores[2], is(0.0));
    }

    @Test
    void rerankScoresFallBackToArrayOrderWhenIndexAbsent() {
        // No "index" field: array position is used, so scores are not silently all-zero.
        String json = "[{\"document\":\"a\",\"score\":0.7},{\"document\":\"b\",\"score\":0.2}]";

        double[] scores = LangChain4jMapping.parseRerankScores(json, 2);

        assertThat(scores[0], is(0.7));
        assertThat(scores[1], is(0.2));
    }

    // ------------------------------------------------------------------
    // Tool calling
    // ------------------------------------------------------------------

    @Test
    void forwardsToolSpecificationsAndToolChoice() {
        ToolSpecification weather = ToolSpecification.builder()
                .name("get_weather")
                .description("Current weather for a city")
                .parameters(JsonObjectSchema.builder()
                        .addProperty("city", JsonStringSchema.builder()
                                .description("city name")
                                .build())
                        .required("city")
                        .build())
                .build();
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from("weather in Berlin?"))
                .toolSpecifications(weather)
                .toolChoice(ToolChoice.REQUIRED)
                .build();

        net.ladenthin.llama.parameters.ChatRequest jllama = LangChain4jMapping.toJllamaRequest(request);

        assertThat(jllama.getTools().size(), is(1));
        ToolDefinition tool = jllama.getTools().get(0);
        assertThat(tool.getName(), is("get_weather"));
        assertThat(tool.getDescription(), is("Current weather for a city"));
        assertThat(tool.getParametersSchemaJson(), containsString("\"city\""));
        assertThat(tool.getParametersSchemaJson(), containsString("\"required\""));
        assertThat(jllama.getToolChoice().orElse(""), is("required"));
        // The OAI tools array must materialize for LlamaModel.chat to forward it natively.
        assertThat(jllama.buildToolsJson().orElse(""), containsString("\"function\""));
    }

    @Test
    void toolWithoutParametersGetsEmptyObjectSchema() {
        ToolSpecification noArgs =
                ToolSpecification.builder().name("get_time").build();

        ToolDefinition tool = LangChain4jMapping.toToolDefinition(noArgs);

        assertThat(tool.getParametersSchemaJson(), is(LangChain4jMapping.EMPTY_PARAMETERS_SCHEMA));
    }

    @Test
    void streamingParametersCarryToolsAndToolChoice() {
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from("weather?"))
                .toolSpecifications(ToolSpecification.builder()
                        .name("get_weather")
                        .parameters(JsonObjectSchema.builder()
                                .addProperty("city", new JsonStringSchema())
                                .build())
                        .build())
                .toolChoice(ToolChoice.REQUIRED)
                .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        // The streaming blob must carry the same tools wiring the blocking path applies.
        assertThat(json, containsString("\"tools\""));
        assertThat(json, containsString("get_weather"));
        assertThat(json, containsString("\"tool_choice\""));
        assertThat(json, containsString("required"));
        // Jinja is a load-time option (--jinja). Upstream's request parser never reads a "use_jinja"
        // key and silently discards unknown fields, so a re-added withUseChatTemplate(true) here
        // would be invisible at runtime and uncatchable by any integration test. Mirrors
        // OpenAiRequestMapperTest#toolsEnableChatTemplateAndForwardChoice.
        assertThat(json, not(containsString("\"use_jinja\"")));
    }

    @Test
    void mapsToolChoiceEnumToOaiStrings() {
        assertThat(LangChain4jMapping.toToolChoiceString(ToolChoice.AUTO), is("auto"));
        assertThat(LangChain4jMapping.toToolChoiceString(ToolChoice.REQUIRED), is("required"));
        assertThat(LangChain4jMapping.toToolChoiceString(ToolChoice.NONE), is("none"));
    }

    @Test
    void mapsAssistantToolCallTurnIntoHistory() {
        // Multi-turn tool history: the assistant's earlier tool-call turn must survive verbatim
        // so the model sees the full call/result exchange.
        AiMessage toolCallTurn = AiMessage.from(ToolExecutionRequest.builder()
                .id("call_7")
                .name("get_weather")
                .arguments("{\"city\":\"Berlin\"}")
                .build());
        ChatRequest request = ChatRequest.builder()
                .messages(
                        UserMessage.from("weather in Berlin?"),
                        toolCallTurn,
                        ToolExecutionResultMessage.from("call_7", "get_weather", "22C"))
                .build();

        List<ChatMessage> messages = LangChain4jMapping.toJllamaRequest(request).getMessages();

        ChatMessage assistant = messages.get(1);
        assertThat(assistant.getRole(), is("assistant"));
        assertThat(assistant.getToolCalls().size(), is(1));
        ToolCall call = assistant.getToolCalls().get(0);
        assertThat(call.getId(), is("call_7"));
        assertThat(call.getName(), is("get_weather"));
        assertThat(call.getArgumentsJson(), is("{\"city\":\"Berlin\"}"));
        ChatMessage result = messages.get(2);
        assertThat(result.getRole(), is("tool"));
        assertThat(result.getToolCallId().orElse(""), is("call_7"));
    }

    @Test
    void toolCallResponseBecomesAiMessageWithToolExecutionRequests() {
        ChatMessage assistant = ChatMessage.assistantToolCalls(
                "", Arrays.asList(new ToolCall("call_1", "get_weather", "{\"city\":\"Berlin\"}")));
        net.ladenthin.llama.value.ChatResponse response = new net.ladenthin.llama.value.ChatResponse(
                "id-1",
                Arrays.asList(new ChatChoice(0, assistant, "tool_calls")),
                new Usage(10, 5),
                new Timings(0, 0, 0, 0, 0, 0, 0, 0, 0),
                "{}");

        dev.langchain4j.model.chat.response.ChatResponse mapped =
                LangChain4jMapping.toLangChainResponse(response);

        assertThat(mapped.aiMessage().hasToolExecutionRequests(), is(true));
        ToolExecutionRequest request = mapped.aiMessage().toolExecutionRequests().get(0);
        assertThat(request.id(), is("call_1"));
        assertThat(request.name(), is("get_weather"));
        assertThat(request.arguments(), is("{\"city\":\"Berlin\"}"));
        assertThat(mapped.finishReason(), is(FinishReason.TOOL_EXECUTION));
    }

    // ------------------------------------------------------------------
    // response_format / JSON mode
    // ------------------------------------------------------------------

    @Test
    void jsonResponseFormatWithoutSchemaMapsToJsonObjectMode() {
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from("emit json"))
                .responseFormat(ResponseFormat.JSON)
                .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        assertThat(json, containsString("\"response_format\""));
        assertThat(json, containsString("json_object"));
    }

    @Test
    void jsonResponseFormatWithSchemaMapsToJsonSchemaConstraint() {
        ResponseFormat format = ResponseFormat.builder()
                .type(ResponseFormatType.JSON)
                .jsonSchema(JsonSchema.builder()
                        .name("Person")
                        .rootElement(JsonObjectSchema.builder()
                                .addProperty("name", new JsonStringSchema())
                                .required("name")
                                .build())
                        .build())
                .build();
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from("extract the person"))
                .responseFormat(format)
                .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        assertThat(json, containsString("\"json_schema\""));
        assertThat(json, containsString("\"name\""));
    }

    @Test
    void textResponseFormatAddsNoConstraint() {
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from("hi"))
                .responseFormat(ResponseFormat.TEXT)
                .build();

        String json = LangChain4jMapping.toStreamingParameters(request).toString();

        assertThat(json, not(containsString("\"response_format\"")));
        assertThat(json, not(containsString("\"json_schema\"")));
    }

    // ------------------------------------------------------------------
    // Multimodal user content
    // ------------------------------------------------------------------

    @Test
    void multimodalUserMessageMapsToContentParts() {
        // 1x1 transparent PNG, base64.
        String base64Png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                + "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
        ChatRequest request = ChatRequest.builder()
                .messages(UserMessage.from(
                        TextContent.from("What is in this image?"), ImageContent.from(base64Png, "image/png")))
                .build();

        ChatMessage mapped = LangChain4jMapping.toJllamaRequest(request).getMessages().get(0);

        assertThat(mapped.hasParts(), is(true));
        List<ContentPart> parts = mapped.getParts().orElse(java.util.Collections.emptyList());
        assertThat(parts.size(), is(2));
        assertThat(parts.get(0).getType(), is(ContentPart.Type.TEXT));
        assertThat(parts.get(0).getText(), is("What is in this image?"));
        assertThat(parts.get(1).getType(), is(ContentPart.Type.IMAGE_URL));
        assertThat(parts.get(1).getImageUrl(), is("data:image/png;base64," + base64Png));
    }

    @Test
    void audioContentMapsToInputAudioPart() {
        byte[] fakeWav = new byte[] {'R', 'I', 'F', 'F'};
        AudioContent audio = AudioContent.from(Audio.builder()
                .base64Data(java.util.Base64.getEncoder().encodeToString(fakeWav))
                .mimeType("audio/wav")
                .build());

        ContentPart part = LangChain4jMapping.toContentPart(audio);

        assertThat(part.getType(), is(ContentPart.Type.INPUT_AUDIO));
        assertThat(part.getAudioFormat(), is("wav"));
        assertThat(part.getAudioData(), is(java.util.Base64.getEncoder().encodeToString(fakeWav)));
    }

    @Test
    void unsupportedAudioMimeTypeFailsLoud() {
        AudioContent audio = AudioContent.from(Audio.builder()
                .base64Data("AAAA")
                .mimeType("audio/ogg")
                .build());

        assertThrows(IllegalArgumentException.class, () -> LangChain4jMapping.toContentPart(audio));
    }

    @Test
    void imageWithoutDataOrUrlFailsLoud() {
        ImageContent image = ImageContent.from(
                dev.langchain4j.data.image.Image.builder().build());

        assertThrows(IllegalArgumentException.class, () -> LangChain4jMapping.toContentPart(image));
    }
}
