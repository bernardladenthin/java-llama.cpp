// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Assembles the native OpenAI {@code chat.completion.chunk} stream
 * ({@code LlamaModel.streamChatCompletion}) into langchain4j streaming events and the final
 * {@link ChatResponse}:
 *
 * <ul>
 *   <li>{@code delta.content} → {@link StreamingChatResponseHandler#onPartialResponse(String)}
 *   <li>{@code delta.reasoning_content} →
 *       {@link StreamingChatResponseHandler#onPartialThinking(PartialThinking)}
 *   <li>{@code delta.tool_calls[*]} fragments → accumulated per OpenAI {@code index} (id and
 *       name arrive on the first fragment, arguments split across fragments), each forwarded as
 *       {@link StreamingChatResponseHandler#onPartialToolCall(PartialToolCall)} and completed as
 *       {@link StreamingChatResponseHandler#onCompleteToolCall(CompleteToolCall)}
 *   <li>{@code finish_reason} / trailing {@code usage} chunk → finish reason and
 *       {@link TokenUsage} on the final response
 * </ul>
 *
 * <p>Pure data transform over chunk JSON strings — no JNI, no model — so the whole state
 * machine is unit-testable with canned chunks (see {@code StreamingChunkAssemblerTest}).
 * Not thread-safe; one instance per streamed request (chunks arrive in order on one thread).</p>
 */
final class StreamingChunkAssembler {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Per-index accumulation state for one streamed tool call. */
    private static final class PartialTool {
        String id = "";
        String name = "";
        final StringBuilder arguments = new StringBuilder();
    }

    private final StreamingChatResponseHandler handler;
    private final StringBuilder text = new StringBuilder();
    private final StringBuilder thinking = new StringBuilder();
    private final Map<Integer, PartialTool> toolsByIndex = new TreeMap<>();
    private String finishReason = "";
    private TokenUsage tokenUsage;

    /**
     * Creates an assembler forwarding intermediate events to {@code handler}.
     *
     * @param handler the langchain4j streaming handler to notify per chunk
     */
    StreamingChunkAssembler(StreamingChatResponseHandler handler) {
        this.handler = handler;
    }

    /**
     * Consume one {@code chat.completion.chunk} JSON string, forwarding the matching
     * streaming events.
     *
     * @param chunkJson the chunk as emitted by {@code streamChatCompletion}
     * @throws UncheckedIOException if the chunk is not parseable JSON (native contract
     *     violation — surfaces via {@code onError} in the adapter)
     */
    void accept(String chunkJson) {
        JsonNode chunk;
        try {
            chunk = MAPPER.readTree(chunkJson);
        } catch (IOException e) {
            throw new UncheckedIOException("Unparseable chat.completion.chunk: " + chunkJson, e);
        }
        JsonNode usage = chunk.path("usage");
        if (usage.isObject()) {
            tokenUsage = new TokenUsage(
                    usage.path("prompt_tokens").asInt(0), usage.path("completion_tokens").asInt(0));
        }
        JsonNode choice = chunk.path("choices").path(0);
        if (choice.path("finish_reason").isTextual()) {
            finishReason = choice.path("finish_reason").asText();
        }
        JsonNode delta = choice.path("delta");
        JsonNode content = delta.path("content");
        if (content.isTextual() && !content.asText().isEmpty()) {
            text.append(content.asText());
            handler.onPartialResponse(content.asText());
        }
        JsonNode reasoning = delta.path("reasoning_content");
        if (reasoning.isTextual() && !reasoning.asText().isEmpty()) {
            thinking.append(reasoning.asText());
            handler.onPartialThinking(new PartialThinking(reasoning.asText()));
        }
        JsonNode toolCalls = delta.path("tool_calls");
        if (toolCalls.isArray()) {
            for (JsonNode fragment : toolCalls) {
                acceptToolCallFragment(fragment);
            }
        }
    }

    private void acceptToolCallFragment(JsonNode fragment) {
        int index = fragment.path("index").asInt(0);
        PartialTool tool = toolsByIndex.get(index);
        if (tool == null) {
            tool = new PartialTool();
            toolsByIndex.put(index, tool);
        }
        if (fragment.path("id").isTextual() && !fragment.path("id").asText().isEmpty()) {
            tool.id = fragment.path("id").asText();
        }
        JsonNode function = fragment.path("function");
        if (function.path("name").isTextual() && !function.path("name").asText().isEmpty()) {
            tool.name = function.path("name").asText();
        }
        String argumentsFragment =
                function.path("arguments").isTextual() ? function.path("arguments").asText() : "";
        tool.arguments.append(argumentsFragment);
        // PartialToolCall refuses empty arguments and a blank name, and llama.cpp sends both: the
        // first delta of a call usually carries id and name only, and its incremental parser may
        // report a call before its name. Such a delta still feeds the assembled call.
        if (argumentsFragment.isEmpty() || tool.name.isEmpty()) {
            return;
        }
        handler.onPartialToolCall(PartialToolCall.builder()
                .index(index)
                .id(tool.id)
                .name(tool.name)
                .partialArguments(argumentsFragment)
                .build());
    }

    /**
     * Finalize the stream: emit {@code onCompleteToolCall} for every accumulated call and
     * build the final {@link ChatResponse} (text and/or tool execution requests, thinking,
     * finish reason, token usage when the stream carried a usage chunk).
     *
     * @return the assembled final response for {@code onCompleteResponse}
     */
    ChatResponse complete() {
        AiMessage.Builder message = AiMessage.builder();
        if (text.length() > 0) {
            message.text(text.toString());
        }
        if (thinking.length() > 0) {
            message.thinking(thinking.toString());
        }
        if (!toolsByIndex.isEmpty()) {
            List<ToolExecutionRequest> requests = new ArrayList<>(toolsByIndex.size());
            for (Map.Entry<Integer, PartialTool> entry : toolsByIndex.entrySet()) {
                PartialTool tool = entry.getValue();
                ToolExecutionRequest request = ToolExecutionRequest.builder()
                        .id(tool.id)
                        .name(tool.name)
                        .arguments(tool.arguments.toString())
                        .build();
                handler.onCompleteToolCall(new CompleteToolCall(entry.getKey(), request));
                requests.add(request);
            }
            message.toolExecutionRequests(requests);
        }
        ChatResponse.Builder response = ChatResponse.builder()
                .aiMessage(message.build())
                .finishReason(LangChain4jMapping.toFinishReason(finishReason.isEmpty() ? null : finishReason));
        if (tokenUsage != null) {
            response.tokenUsage(tokenUsage);
        }
        return response.build();
    }
}
