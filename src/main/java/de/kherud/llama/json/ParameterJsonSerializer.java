package de.kherud.llama.json;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.kherud.llama.Pair;
import de.kherud.llama.args.Sampler;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Pure JSON builders for inference request parameters.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with plain Java values alone (see
 * {@code ParameterJsonSerializerTest}).
 *
 * <p>Methods return Jackson {@link ArrayNode} or {@link ObjectNode}. Callers that need a JSON
 * string (e.g. callers in {@code JsonParameters}) call {@code node.toString()}.
 *
 * <p>This class replaces hand-rolled {@code StringBuilder} loops and the
 * {@code org.json}-derived {@code toJsonString()} escaper previously embedded in
 * {@code JsonParameters}.
 */
public final class ParameterJsonSerializer {

    /** Shared Jackson mapper; all methods are stateless and thread-safe. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private ParameterJsonSerializer() {}

    // ------------------------------------------------------------------
    // String escaping
    // ------------------------------------------------------------------

    /**
     * Serialize a Java string to a quoted, properly escaped JSON string literal
     * (e.g. {@code "hello\nworld"} → {@code "\"hello\\nworld\""}).
     * Returns {@code "null"} for a {@code null} input.
     *
     * <p>Replaces the hand-rolled {@code toJsonString()} method in {@code JsonParameters}.
     *
     * @param value the Java string to serialize, or {@code null}
     * @return a JSON string literal, or {@code "null"} if the input is {@code null}
     */
    public static String toJsonString(String value) {
        if (value == null) return "null";
        try {
            return OBJECT_MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return "null";
        }
    }

    // ------------------------------------------------------------------
    // Message array
    // ------------------------------------------------------------------

    /**
     * Build an OAI-compatible {@code messages} array node.
     *
     * <p>An optional system message is prepended when non-null and non-empty.
     * Each message in {@code messages} must have role {@code "user"} or {@code "assistant"}.
     *
     * @param systemMessage optional system prompt; skipped when {@code null} or empty
     * @param messages list of user/assistant message pairs (role as key, content as value)
     * @return a Jackson {@link ArrayNode} of {@code {"role", "content"}} message objects
     * @throws IllegalArgumentException if any message has an invalid role
     */
    public static ArrayNode buildMessages(String systemMessage, List<Pair<String, String>> messages) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        if (systemMessage != null && !systemMessage.isEmpty()) {
            ObjectNode sys = OBJECT_MAPPER.createObjectNode();
            sys.put("role", "system");
            sys.put("content", systemMessage);
            arr.add(sys);
        }
        for (Pair<String, String> message : messages) {
            String role = message.getKey();
            String content = message.getValue();
            if (!"user".equals(role) && !"assistant".equals(role)) {
                throw new IllegalArgumentException(
                        "Invalid role: " + role + ". Role must be 'user' or 'assistant'.");
            }
            ObjectNode msg = OBJECT_MAPPER.createObjectNode();
            msg.put("role", role);
            msg.put("content", content);
            arr.add(msg);
        }
        return arr;
    }

    // ------------------------------------------------------------------
    // Simple array builders
    // ------------------------------------------------------------------

    /**
     * Build a JSON string array from the given stop strings
     * (e.g. {@code ["<|endoftext|>", "\n"]}).
     *
     * @param stops one or more stop strings
     * @return a Jackson {@link ArrayNode} of stop string values
     */
    public static ArrayNode buildStopStrings(String... stops) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (String stop : stops) arr.add(stop);
        return arr;
    }

    /**
     * Build a JSON string array from the given sampler sequence
     * (e.g. {@code ["top_k", "top_p", "temperature"]}).
     *
     * @param samplers one or more samplers in the desired order
     * @return a Jackson {@link ArrayNode} of sampler name strings
     */
    public static ArrayNode buildSamplers(Sampler... samplers) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Sampler sampler : samplers) {
            switch (sampler) {
                case TOP_K:       arr.add("top_k");       break;
                case TOP_P:       arr.add("top_p");       break;
                case MIN_P:       arr.add("min_p");       break;
                case TEMPERATURE: arr.add("temperature"); break;
            }
        }
        return arr;
    }

    /**
     * Build a JSON integer array from a primitive {@code int[]}
     * (used for penalty-prompt token sequences).
     *
     * @param values the token IDs to include
     * @return a Jackson {@link ArrayNode} of integer values
     */
    public static ArrayNode buildIntArray(int[] values) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (int v : values) arr.add(v);
        return arr;
    }

    // ------------------------------------------------------------------
    // Logit-bias pair arrays — [[key, value], ...]
    // ------------------------------------------------------------------

    /**
     * Build a logit-bias array for integer token IDs:
     * {@code [[15043, 1.0], [50256, -0.5]]}.
     *
     * @param biases map from token ID to logit bias value
     * @return a Jackson {@link ArrayNode} of {@code [tokenId, biasValue]} pairs
     */
    public static ArrayNode buildTokenIdBiasArray(Map<Integer, Float> biases) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Map.Entry<Integer, Float> entry : biases.entrySet()) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(entry.getKey());
            pair.add(entry.getValue());
            arr.add(pair);
        }
        return arr;
    }

    /**
     * Build a logit-bias array for string tokens:
     * {@code [["Hello", 1.0], [" world", -0.5]]}.
     *
     * @param biases map from token string to logit bias value
     * @return a Jackson {@link ArrayNode} of {@code ["token", biasValue]} pairs
     */
    public static ArrayNode buildTokenStringBiasArray(Map<String, Float> biases) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Map.Entry<String, Float> entry : biases.entrySet()) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(entry.getKey());
            pair.add(entry.getValue());
            arr.add(pair);
        }
        return arr;
    }

    /**
     * Build a disable-token array for integer token IDs:
     * {@code [[15043, false], [50256, false]]}.
     *
     * @param ids collection of integer token IDs to disable
     * @return a Jackson {@link ArrayNode} of {@code [tokenId, false]} pairs
     */
    public static ArrayNode buildDisableTokenIdArray(Collection<Integer> ids) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Integer id : ids) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(id);
            pair.add(false);
            arr.add(pair);
        }
        return arr;
    }

    /**
     * Build a disable-token array for string tokens:
     * {@code [["Hello", false], [" world", false]]}.
     *
     * @param tokens collection of token strings to disable
     * @return a Jackson {@link ArrayNode} of {@code ["token", false]} pairs
     */
    public static ArrayNode buildDisableTokenStringArray(Collection<String> tokens) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (String token : tokens) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(token);
            pair.add(false);
            arr.add(pair);
        }
        return arr;
    }

    // ------------------------------------------------------------------
    // Object with pre-serialized JSON values
    // ------------------------------------------------------------------

    /**
     * Build a JSON object where each map value is a <em>pre-serialized JSON string</em>
     * (not a plain Java string). For example, a map entry {@code ("enable_thinking", "true")}
     * produces {@code {"enable_thinking": true}}, not {@code {"enable_thinking": "true"}}.
     *
     * <p>Used for {@code chat_template_kwargs} which stores raw JSON values.
     * If a value cannot be parsed as JSON, it is stored as a JSON string literal.
     *
     * @param map map of key to pre-serialized JSON value strings
     * @return a Jackson {@link ObjectNode} with each value embedded as a parsed JSON node
     */
    public static ObjectNode buildRawValueObject(Map<String, String> map) {
        ObjectNode node = OBJECT_MAPPER.createObjectNode();
        for (Map.Entry<String, String> entry : map.entrySet()) {
            try {
                JsonNode val = OBJECT_MAPPER.readTree(entry.getValue());
                node.set(entry.getKey(), val);
            } catch (IOException e) {
                node.put(entry.getKey(), entry.getValue());
            }
        }
        return node;
    }
}
