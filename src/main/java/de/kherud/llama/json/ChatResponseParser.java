package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;

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
public final class ChatResponseParser {

    /** Shared Jackson mapper; all methods are stateless and thread-safe. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private ChatResponseParser() {}

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
    public static String extractChoiceContent(String json) {
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
    public static String extractChoiceContent(JsonNode node) {
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
    public static int extractUsageField(JsonNode node, String field) {
        return node.path("usage").path(field).asInt(0);
    }

    /**
     * Count the number of choices returned in the response.
     * Returns {@code 0} when the {@code "choices"} array is absent or not an array.
     *
     * @param node pre-parsed OAI chat completion response node
     * @return the number of choices, or {@code 0} if absent
     */
    public static int countChoices(JsonNode node) {
        JsonNode choices = node.path("choices");
        return choices.isArray() ? choices.size() : 0;
    }
}
