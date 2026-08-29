// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import net.ladenthin.llama.json.RerankResponseParser;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.value.ContentPart;
import net.ladenthin.llama.value.ToolCall;
import net.ladenthin.llama.value.ToolDefinition;

/**
 * Pure (model-free) translation between langchain4j chat types and java-llama.cpp parameters.
 *
 * <p>Every method here is a deterministic data transform with no JNI and no loaded model, so the
 * mapping is unit-testable on its own (see {@code LangChain4jMappingTest}). The adapters keep the
 * live-model calls; this class only reshapes their inputs and outputs.
 */
final class LangChain4jMapping {

    /**
     * Parameter schema used when a {@link ToolSpecification} declares no parameters: a
     * no-argument tool still needs a syntactically valid JSON Schema object for the OAI
     * {@code tools} array.
     */
    static final String EMPTY_PARAMETERS_SCHEMA = "{\"type\":\"object\",\"properties\":{}}";

    /** OAI {@code response_format} payload selecting plain JSON-object mode (no schema). */
    static final String JSON_OBJECT_RESPONSE_FORMAT = "{\"type\":\"json_object\"}";

    private LangChain4jMapping() {}

    /**
     * Build a java-llama.cpp typed chat request from a langchain4j chat request. Messages map by
     * role (including assistant tool-call turns and multimodal user turns); tool specifications and
     * the {@code toolChoice} hint are forwarded; sampling parameters ({@code temperature}/{@code
     * topP}/{@code topK}/{@code maxOutputTokens}/{@code stopSequences}) and a JSON {@code
     * responseFormat} ride along as an inference customizer.
     */
    static net.ladenthin.llama.parameters.ChatRequest toJllamaRequest(ChatRequest request) {
        net.ladenthin.llama.parameters.ChatRequest jllama =
                net.ladenthin.llama.parameters.ChatRequest.empty();
        for (ChatMessage message : request.messages()) {
            jllama = jllama.appendMessage(toJllamaMessage(message));
        }
        List<ToolSpecification> tools = request.toolSpecifications();
        if (tools != null) {
            for (ToolSpecification tool : tools) {
                jllama = jllama.appendTool(toToolDefinition(tool));
            }
        }
        if (request.toolChoice() != null) {
            jllama = jllama.withToolChoice(toToolChoiceString(request.toolChoice()));
        }
        return jllama.withInferenceCustomizer(params -> applyResponseFormat(applySampling(params, request), request));
    }

    /**
     * Build the streaming inference parameters (messages JSON + tools + sampling) for
     * {@code streamChatCompletion}. Shares {@link #toJllamaRequest(ChatRequest)} so blocking and
     * streaming stay in lockstep: tool specifications, the {@code toolChoice} hint, JSON
     * {@code responseFormat}, and the sampling parameters all ride along — the same wiring
     * {@code LlamaModel.chat} applies on the blocking path.
     */
    static InferenceParameters toStreamingParameters(ChatRequest request) {
        net.ladenthin.llama.parameters.ChatRequest jllama = toJllamaRequest(request);
        InferenceParameters params =
                InferenceParameters.empty().withMessagesJson(jllama.buildMessagesJson());
        java.util.Optional<String> toolsJson = jllama.buildToolsJson();
        if (toolsJson.isPresent()) {
            params = params.withToolsJson(toolsJson.get());
            java.util.Optional<String> toolChoice = jllama.getToolChoice();
            if (toolChoice.isPresent()) {
                params = params.withToolChoice(toolChoice.get());
            }
        }
        return jllama.applyCustomizer(params);
    }

    /**
     * Wrap a java-llama.cpp chat result as a langchain4j {@link ChatResponse}. An assistant turn
     * carrying {@code tool_calls} becomes an {@link AiMessage} with
     * {@link AiMessage#toolExecutionRequests()} populated (and {@code finishReason} mapped to
     * {@code TOOL_EXECUTION} by the native {@code "tool_calls"} finish string).
     */
    static ChatResponse toLangChainResponse(net.ladenthin.llama.value.ChatResponse response) {
        ChatResponse.Builder builder = ChatResponse.builder().aiMessage(toAiMessage(response));
        net.ladenthin.llama.value.Usage usage = response.getUsage();
        if (usage != null) {
            builder.tokenUsage(
                    new TokenUsage((int) usage.getPromptTokens(), (int) usage.getCompletionTokens()));
        }
        List<net.ladenthin.llama.value.ChatChoice> choices = response.getChoices();
        String finishReason = choices.isEmpty() ? null : choices.get(0).getFinishReason();
        return builder.finishReason(toFinishReason(finishReason)).build();
    }

    /**
     * Map java-llama.cpp's OpenAI-style finish-reason string to the langchain4j enum. A {@code null}
     * (no choices / reason absent) is treated as a normal {@code STOP}; an unrecognized value maps to
     * {@code OTHER} rather than guessing.
     */
    static FinishReason toFinishReason(String reason) {
        if (reason == null) {
            return FinishReason.STOP;
        }
        switch (reason) {
            case "stop":
                return FinishReason.STOP;
            case "length":
                return FinishReason.LENGTH;
            case "tool_calls":
                return FinishReason.TOOL_EXECUTION;
            case "content_filter":
                return FinishReason.CONTENT_FILTER;
            default:
                return FinishReason.OTHER;
        }
    }

    /**
     * Align native rerank scores to input order. The native response is a JSON array of
     * {@code {document, index, score}} objects whose {@code index} is the position in the input
     * documents array; results may arrive in any order, so we place each score at its index.
     *
     * @param json the raw native rerank JSON array
     * @param count the number of input documents (output length)
     * @return scores indexed by input position; positions absent from the response stay {@code 0.0}
     */
    static double[] parseRerankScores(String json, int count) {
        double[] scores = new double[count];
        try {
            JsonNode array = RerankResponseParser.OBJECT_MAPPER.readTree(json);
            if (array.isArray()) {
                int position = 0;
                for (JsonNode entry : array) {
                    // "index" is the input position; fall back to array order when the field is
                    // absent so a response without it never silently yields all-zero scores.
                    int index = entry.path("index").asInt(position);
                    if (index >= 0 && index < count) {
                        scores[index] = entry.path("score").asDouble(0.0);
                    }
                    position++;
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to parse rerank response", e);
        }
        return scores;
    }

    /** Convert a langchain4j tool specification to the jllama typed tool definition. */
    static ToolDefinition toToolDefinition(ToolSpecification spec) {
        String parametersJson = spec.parameters() == null
                ? EMPTY_PARAMETERS_SCHEMA
                : JsonSchemaElementSerializer.toJson(spec.parameters());
        return new ToolDefinition(
                spec.name(), spec.description() == null ? "" : spec.description(), parametersJson);
    }

    /** Map the langchain4j {@link ToolChoice} enum to the OAI {@code tool_choice} string. */
    static String toToolChoiceString(ToolChoice choice) {
        switch (choice) {
            case REQUIRED:
                return "required";
            case NONE:
                return "none";
            case AUTO:
            default:
                return "auto";
        }
    }

    private static AiMessage toAiMessage(net.ladenthin.llama.value.ChatResponse response) {
        net.ladenthin.llama.value.ChatMessage first =
                response.getFirstMessage().orElse(null);
        if (first == null || first.getToolCalls().isEmpty()) {
            return AiMessage.from(response.getFirstContent());
        }
        List<ToolExecutionRequest> requests = new ArrayList<>(first.getToolCalls().size());
        for (ToolCall call : first.getToolCalls()) {
            requests.add(ToolExecutionRequest.builder()
                    .id(call.getId())
                    .name(call.getName())
                    .arguments(call.getArgumentsJson())
                    .build());
        }
        AiMessage.Builder builder = AiMessage.builder().toolExecutionRequests(requests);
        String text = first.getContent();
        if (text != null && !text.isEmpty()) {
            builder.text(text);
        }
        return builder.build();
    }

    private static net.ladenthin.llama.value.ChatMessage toJllamaMessage(ChatMessage message) {
        switch (message.type()) {
            case SYSTEM:
                return new net.ladenthin.llama.value.ChatMessage(
                        "system", ((SystemMessage) message).text());
            case USER:
                return toJllamaUserMessage((UserMessage) message);
            case AI:
                return toJllamaAssistantMessage((AiMessage) message);
            case TOOL_EXECUTION_RESULT:
                ToolExecutionResultMessage tool = (ToolExecutionResultMessage) message;
                return net.ladenthin.llama.value.ChatMessage.toolResult(tool.id(), tool.text());
            default:
                // CUSTOM and any future type: no faithful chat-role mapping exists.
                throw new IllegalArgumentException("Unsupported message type: " + message.type());
        }
    }

    private static net.ladenthin.llama.value.ChatMessage toJllamaAssistantMessage(AiMessage message) {
        String text = message.text();
        if (!message.hasToolExecutionRequests()) {
            return new net.ladenthin.llama.value.ChatMessage("assistant", text == null ? "" : text);
        }
        List<ToolCall> calls = new ArrayList<>(message.toolExecutionRequests().size());
        for (ToolExecutionRequest request : message.toolExecutionRequests()) {
            calls.add(new ToolCall(
                    request.id() == null ? "" : request.id(),
                    request.name(),
                    request.arguments() == null ? "{}" : request.arguments()));
        }
        return net.ladenthin.llama.value.ChatMessage.assistantToolCalls(text == null ? "" : text, calls);
    }

    private static net.ladenthin.llama.value.ChatMessage toJllamaUserMessage(UserMessage message) {
        if (message.hasSingleText()) {
            return new net.ladenthin.llama.value.ChatMessage("user", message.singleText());
        }
        if (!hasNonTextContent(message)) {
            // Multiple text parts, no media: keep the flat-string form (no array-content needed).
            StringBuilder text = new StringBuilder();
            for (Content content : message.contents()) {
                text.append(((TextContent) content).text());
            }
            return new net.ladenthin.llama.value.ChatMessage("user", text.toString());
        }
        List<ContentPart> parts = new ArrayList<>(message.contents().size());
        for (Content content : message.contents()) {
            parts.add(toContentPart(content));
        }
        return new net.ladenthin.llama.value.ChatMessage("user", parts);
    }

    private static boolean hasNonTextContent(UserMessage message) {
        for (Content content : message.contents()) {
            if (content.type() != ContentType.TEXT) {
                return true;
            }
        }
        return false;
    }

    /**
     * Map one langchain4j content part to the jllama multimodal part. Media requires either an
     * inline payload (base64 + MIME type) or, for images, a URL; anything the upstream mtmd
     * pipeline cannot consume fails loud instead of being silently dropped.
     */
    static ContentPart toContentPart(Content content) {
        if (content instanceof TextContent) {
            return ContentPart.text(((TextContent) content).text());
        }
        if (content instanceof ImageContent) {
            return toImagePart(((ImageContent) content).image());
        }
        if (content instanceof AudioContent) {
            return toAudioPart(((AudioContent) content).audio());
        }
        throw new IllegalArgumentException("Unsupported user content type: " + content.type());
    }

    private static ContentPart toImagePart(Image image) {
        if (image.base64Data() != null && image.mimeType() != null) {
            return ContentPart.imageUrl("data:" + image.mimeType() + ";base64," + image.base64Data());
        }
        if (image.url() != null) {
            return ContentPart.imageUrl(image.url().toString());
        }
        throw new IllegalArgumentException(
                "ImageContent carries neither base64 data (with MIME type) nor a URL: " + image);
    }

    private static ContentPart toAudioPart(Audio audio) {
        String format = toAudioFormat(audio.mimeType());
        byte[] bytes = audio.binaryData();
        if (bytes == null && audio.base64Data() != null) {
            bytes = Base64.getDecoder().decode(audio.base64Data());
        }
        if (bytes == null) {
            throw new IllegalArgumentException(
                    "AudioContent carries no inline audio data (URL-only audio is not supported): " + audio);
        }
        return ContentPart.inputAudio(bytes, format);
    }

    private static String toAudioFormat(String mimeType) {
        if (mimeType != null) {
            String normalized = mimeType.toLowerCase(Locale.ROOT);
            if (normalized.equals("audio/wav") || normalized.equals("audio/x-wav") || normalized.equals("audio/wave")) {
                return "wav";
            }
            if (normalized.equals("audio/mpeg") || normalized.equals("audio/mp3")) {
                return "mp3";
            }
        }
        throw new IllegalArgumentException(
                "Unsupported audio MIME type (only wav/mp3 reach the mtmd pipeline): " + mimeType);
    }

    private static InferenceParameters applyResponseFormat(InferenceParameters params, ChatRequest request) {
        ResponseFormat format = request.responseFormat();
        if (format == null || format.type() != ResponseFormatType.JSON) {
            return params;
        }
        if (format.jsonSchema() != null && format.jsonSchema().rootElement() != null) {
            return params.withJsonSchema(
                    JsonSchemaElementSerializer.toJson(format.jsonSchema().rootElement()));
        }
        return params.withResponseFormat(JSON_OBJECT_RESPONSE_FORMAT);
    }

    private static InferenceParameters applySampling(InferenceParameters params, ChatRequest request) {
        if (request.temperature() != null) {
            params = params.withTemperature(request.temperature().floatValue());
        }
        if (request.topP() != null) {
            params = params.withTopP(request.topP().floatValue());
        }
        if (request.topK() != null) {
            params = params.withTopK(request.topK());
        }
        if (request.maxOutputTokens() != null) {
            params = params.withNPredict(request.maxOutputTokens());
        }
        if (request.frequencyPenalty() != null) {
            params = params.withFrequencyPenalty(request.frequencyPenalty().floatValue());
        }
        if (request.presencePenalty() != null) {
            params = params.withPresencePenalty(request.presencePenalty().floatValue());
        }
        List<String> stops = request.stopSequences();
        if (stops != null && !stops.isEmpty()) {
            params = params.withStopStrings(stops.toArray(new String[0]));
        }
        return params;
    }
}
