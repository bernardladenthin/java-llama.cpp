// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.args.Sampler;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import net.ladenthin.llama.value.Pair;
import net.ladenthin.llama.value.ToolCall;
import org.jspecify.annotations.Nullable;

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
public class ParameterJsonSerializer {

    /** Creates a new {@link ParameterJsonSerializer}. */
    public ParameterJsonSerializer() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

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
    public String toJsonString(String value) {
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
    public ArrayNode buildMessages(@Nullable String systemMessage, List<Pair<String, String>> messages) {
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
                throw new IllegalArgumentException("Invalid role: " + role + ". Role must be 'user' or 'assistant'.");
            }
            ObjectNode msg = OBJECT_MAPPER.createObjectNode();
            msg.put("role", role);
            msg.put("content", content);
            arr.add(msg);
        }
        return arr;
    }

    /**
     * Multimodal-capable variant of {@link #buildMessages(String, List)}. Accepts
     * {@link net.ladenthin.llama.value.ChatMessage} objects directly so messages with non-null
     * {@link net.ladenthin.llama.value.ChatMessage#getParts()} can be serialized as the OAI-compatible
     * array-form {@code content} the upstream {@code mtmd} pipeline expects.
     * Plain text messages still emit the legacy string-form {@code content}.
     *
     * @param messages messages in order; must not be {@code null}
     * @return a Jackson {@link ArrayNode} ready for the {@code messages} request field
     */
    public ArrayNode buildMessages(List<ChatMessage> messages) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (ChatMessage message : messages) {
            ObjectNode msg = OBJECT_MAPPER.createObjectNode();
            msg.put("role", message.getRole());
            if (message.hasParts()) {
                ArrayNode parts = OBJECT_MAPPER.createArrayNode();
                for (ContentPart p : message.getParts()
                        .orElseThrow(() -> new IllegalStateException("hasParts() was true but getParts() was empty"))) {
                    ObjectNode part = OBJECT_MAPPER.createObjectNode();
                    if (p.getType() == ContentPart.Type.TEXT) {
                        part.put("type", "text");
                        final String text = p.getText();
                        part.put("text", text != null ? text : "");
                    } else if (p.getType() == ContentPart.Type.INPUT_AUDIO) {
                        part.put("type", "input_audio");
                        ObjectNode inputAudio = OBJECT_MAPPER.createObjectNode();
                        final String data = p.getAudioData();
                        final String format = p.getAudioFormat();
                        inputAudio.put("data", data != null ? data : "");
                        inputAudio.put("format", format != null ? format : "wav");
                        part.set("input_audio", inputAudio);
                    } else {
                        part.put("type", "image_url");
                        ObjectNode imageUrl = OBJECT_MAPPER.createObjectNode();
                        final String url = p.getImageUrl();
                        imageUrl.put("url", url != null ? url : "");
                        part.set("image_url", imageUrl);
                    }
                    parts.add(part);
                }
                msg.set("content", parts);
            } else {
                msg.put("content", message.getContent());
            }
            // Tool-calling round trips: without these the model sees a tool turn it cannot pair with
            // the call it made (and InferenceParameters.withMessages dropped them, unlike ChatRequest).
            message.getToolCallId().ifPresent(id -> msg.put("tool_call_id", id));
            if (!message.getToolCalls().isEmpty()) {
                ArrayNode toolCalls = msg.putArray("tool_calls");
                for (ToolCall call : message.getToolCalls()) {
                    ObjectNode entry = toolCalls.addObject();
                    entry.put("id", call.getId());
                    entry.put("type", "function");
                    ObjectNode function = entry.putObject("function");
                    function.put("name", call.getName());
                    // Upstream parses the arguments of the history with json::parse, so an empty
                    // string fails the whole request with 400; a call without arguments is "{}".
                    final String arguments = call.getArgumentsJson();
                    function.put("arguments", arguments.trim().isEmpty() ? "{}" : arguments);
                }
            }
            arr.add(msg);
        }
        return arr;
    }

    // ------------------------------------------------------------------
    // Simple array builders
    // ------------------------------------------------------------------

    /**
     * Build a JSON array of strings.
     *
     * @param values the strings
     * @return a Jackson {@link ArrayNode} of the strings
     */
    public ArrayNode buildStringArray(String... values) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (String value : values) arr.add(value);
        return arr;
    }

    /**
     * Build a JSON string array from the given stop strings
     * (e.g. {@code ["<|endoftext|>", "\n"]}).
     *
     * @param stops one or more stop strings
     * @return a Jackson {@link ArrayNode} of stop string values
     */
    public ArrayNode buildStopStrings(String... stops) {
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
    public ArrayNode buildSamplers(Sampler... samplers) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Sampler sampler : samplers) {
            arr.add(sampler.getArgValue());
        }
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
    public ArrayNode buildTokenIdBiasArray(Map<Integer, Float> biases) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Map.Entry<Integer, Float> entry : biases.entrySet()) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(entry.getKey());
            addBias(pair, entry.getValue());
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
    public ArrayNode buildTokenStringBiasArray(Map<String, Float> biases) {
        ArrayNode arr = OBJECT_MAPPER.createArrayNode();
        for (Map.Entry<String, Float> entry : biases.entrySet()) {
            ArrayNode pair = OBJECT_MAPPER.createArrayNode();
            pair.add(entry.getKey());
            addBias(pair, entry.getValue());
            arr.add(pair);
        }
        return arr;
    }

    /**
     * Appends one bias value in a form llama.cpp's {@code logit_bias} parser accepts: a number, or
     * {@code false} for negative infinity (a ban). JSON has no infinity, and Jackson writes a
     * non-finite float as a string, which that parser skips -- a {@code -Infinity} bias then
     * silently did nothing.
     *
     * @param pair the {@code [token, bias]} pair to append to
     * @param bias the bias value
     * @throws IllegalArgumentException if {@code bias} is NaN or positive infinity
     */
    private static void addBias(ArrayNode pair, float bias) {
        if (bias == Float.NEGATIVE_INFINITY) {
            pair.add(false);
        } else if (Float.isFinite(bias)) {
            pair.add(bias);
        } else {
            throw new IllegalArgumentException("logit bias must be finite or negative infinity, got " + bias);
        }
    }

    /**
     * Build a disable-token array for integer token IDs:
     * {@code [[15043, false], [50256, false]]}.
     *
     * @param ids collection of integer token IDs to disable
     * @return a Jackson {@link ArrayNode} of {@code [tokenId, false]} pairs
     */
    public ArrayNode buildDisableTokenIdArray(Collection<Integer> ids) {
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
    public ArrayNode buildDisableTokenStringArray(Collection<String> tokens) {
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
    public ObjectNode buildRawValueObject(Map<String, String> map) {
        ObjectNode node = OBJECT_MAPPER.createObjectNode();
        for (Map.Entry<String, String> entry : map.entrySet()) {
            try {
                JsonNode val = OBJECT_MAPPER.readTree(entry.getValue());
                // An empty or blank value is no JSON either, but Jackson returns MissingNode (written as null).
                if (val.isMissingNode()) {
                    node.put(entry.getKey(), entry.getValue());
                } else {
                    node.set(entry.getKey(), val);
                }
            } catch (IOException e) {
                node.put(entry.getKey(), entry.getValue());
            }
        }
        return node;
    }
}
