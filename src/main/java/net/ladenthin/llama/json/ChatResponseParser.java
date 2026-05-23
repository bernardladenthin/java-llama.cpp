// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.ladenthin.llama.ChatChoice;
import net.ladenthin.llama.ChatMessage;
import net.ladenthin.llama.ChatResponse;
import net.ladenthin.llama.Timings;
import net.ladenthin.llama.ToolCall;
import net.ladenthin.llama.Usage;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Pure JSON transforms for OAI-compatible chat completion responses.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code ChatResponseParserTest}).
 *
 * <p>The native server produces an OAI-compatible chat completion JSON:
 * <pre>{@code
 * {
 *   "id": "chatcmpl-...",
 *   "object": "chat.completion",
 *   "choices": [
 *     {
 *       "index": 0,
 *       "message": {"role": "assistant", "content": "Hello!"},
 *       "finish_reason": "stop"
 *     }
 *   ],
 *   "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
 * }
 * }</pre>
 */
public class ChatResponseParser {

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Extract the reasoning/thinking content from an OAI chat completion JSON string.
     * Navigates {@code choices[0].message.reasoning_content}.
     *
     * <p>Thinking models (DeepSeek-R1, QwQ, Qwen3) populate this field when
     * {@code reasoning_format} is {@code "deepseek"} or {@code "auto"}. Returns an
     * empty string when no reasoning content is present or when the JSON is malformed.
     *
     * @param json OAI-compatible chat completion JSON string
     * @return the reasoning content string, or {@code ""} on any failure
     */
    public String extractChoiceReasoningContent(String json) {
        try {
            return extractChoiceReasoningContent(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return "";
        }
    }

    /**
     * Extract the reasoning/thinking content from a pre-parsed OAI chat completion node.
     * Navigates {@code choices[0].message.reasoning_content} via Jackson path API.
     *
     * @param node pre-parsed OAI chat completion response node
     * @return the reasoning content string, or {@code ""} if absent
     */
    public String extractChoiceReasoningContent(JsonNode node) {
        return node.path("choices").path(0).path("message").path("reasoning_content").asText("");
    }

    /**
     * Extract the assistant's reply text from an OAI chat completion JSON string.
     * Navigates {@code choices[0].message.content} via Jackson.
     *
     * <p>Returns an empty string when: the JSON is malformed, {@code choices} is absent
     * or empty, or {@code content} is null/absent.
     *
     * @param json OAI-compatible chat completion JSON string
     * @return the assistant content string, or {@code ""} on any failure
     */
    public String extractChoiceContent(String json) {
        try {
            return extractChoiceContent(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return "";
        }
    }

    /**
     * Extract the assistant's reply text from a pre-parsed OAI chat completion node.
     * Navigates {@code choices[0].message.content} via Jackson path API.
     *
     * @param node pre-parsed OAI chat completion response node
     * @return the assistant content string, or {@code ""} if absent
     */
    public String extractChoiceContent(JsonNode node) {
        return node.path("choices").path(0).path("message").path("content").asText("");
    }

    /**
     * Read a numeric usage field from the {@code "usage"} object in a chat completion node.
     * Common field names: {@code "prompt_tokens"}, {@code "completion_tokens"},
     * {@code "total_tokens"}.
     *
     * @param node  the parsed chat completion response
     * @param field the field name within {@code "usage"}
     * @return the integer value, or {@code 0} if the field or the {@code "usage"} object is absent
     */
    public int extractUsageField(JsonNode node, String field) {
        return node.path("usage").path(field).asInt(0);
    }

    /**
     * Count the number of choices returned in the response.
     * Returns {@code 0} when the {@code "choices"} array is absent or not an array.
     *
     * @param node pre-parsed OAI chat completion response node
     * @return the number of choices, or {@code 0} if absent
     */
    public int countChoices(JsonNode node) {
        JsonNode choices = node.path("choices");
        return choices.isArray() ? choices.size() : 0;
    }

    /**
     * Parse a full OAI chat completion JSON string into a typed {@link ChatResponse}.
     * Carries the {@code id}, choices, {@link Usage}, and {@link Timings}. The original
     * JSON is preserved on {@link ChatResponse#getRawJson()}.
     *
     * @param json the OAI-compatible chat completion JSON string
     * @return a parsed {@link ChatResponse} (empty choices on malformed input)
     */
    public ChatResponse parseResponse(String json) {
        try {
            JsonNode node = OBJECT_MAPPER.readTree(json);
            String id = node.path("id").asText("");
            List<ChatChoice> choices = parseChoices(node.path("choices"));
            Usage usage = new Usage(
                    node.path("usage").path("prompt_tokens").asLong(0L),
                    node.path("usage").path("completion_tokens").asLong(0L));
            Timings timings = Timings.fromJson(node.path("timings"));
            return new ChatResponse(id, choices, usage, timings, json);
        } catch (IOException e) {
            return new ChatResponse("",
                    Collections.<ChatChoice>emptyList(),
                    new Usage(0L, 0L),
                    Timings.fromJson(null),
                    json);
        }
    }

    private List<ChatChoice> parseChoices(JsonNode arr) {
        if (!arr.isArray() || arr.size() == 0) return Collections.emptyList();
        List<ChatChoice> out = new ArrayList<ChatChoice>(arr.size());
        for (JsonNode c : arr) {
            int index = c.path("index").asInt(0);
            JsonNode msg = c.path("message");
            String role = msg.path("role").asText("assistant");
            String content = msg.path("content").asText("");
            List<ToolCall> toolCalls = parseToolCalls(msg.path("tool_calls"));
            ChatMessage message = toolCalls.isEmpty()
                    ? new ChatMessage(role, content)
                    : ChatMessage.assistantToolCalls(content, toolCalls);
            String finishReason = c.path("finish_reason").asText("");
            out.add(new ChatChoice(index, message, finishReason));
        }
        return out;
    }

    private List<ToolCall> parseToolCalls(JsonNode arr) {
        if (!arr.isArray() || arr.size() == 0) return Collections.emptyList();
        List<ToolCall> out = new ArrayList<ToolCall>(arr.size());
        for (JsonNode tc : arr) {
            String id = tc.path("id").asText("");
            JsonNode fn = tc.path("function");
            String name = fn.path("name").asText("");
            JsonNode argsNode = fn.path("arguments");
            // OAI emits arguments as a string; some shapes emit a nested object.
            String args = argsNode.isTextual() ? argsNode.asText("") : argsNode.toString();
            out.add(new ToolCall(id, name, args));
        }
        return out;
    }
}
