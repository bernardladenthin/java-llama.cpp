package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.StopReason;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Pure JSON transforms for native completion/streaming responses.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code CompletionResponseParserTest}).
 *
 * <p>The native server produces one JSON object per streamed token:
 * <pre>{@code
 * {
 *   "content": "Hello",
 *   "stop": false,
 *   "stop_type": "none",
 *   "completion_probabilities": [
 *     {"token": "Hello", "bytes": [...], "id": 15043, "prob": 0.82,
 *      "top_probs": [{"token": "Hi", "bytes": [...], "id": 9932, "prob": 0.1}]}
 *   ]
 * }
 * }</pre>
 *
 * <p>This is the Java analogue of {@code json_helpers.hpp} in the C++ layer.
 */
public final class CompletionResponseParser {

    /** Shared Jackson mapper; all methods are stateless and thread-safe. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private CompletionResponseParser() {}

    /**
     * Parse a {@link LlamaOutput} from a raw JSON string returned by the native
     * {@code receiveCompletionJson} method. Delegates to {@link #parse(JsonNode)} after
     * a single {@code readTree} call so the string is parsed only once.
     *
     * @param json raw JSON string from the native completion response
     * @return parsed {@link LlamaOutput}; empty output on parse failure
     */
    public static LlamaOutput parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new LlamaOutput("", Collections.<String, Float>emptyMap(), false, StopReason.NONE);
        }
    }

    /**
     * Parse a {@link LlamaOutput} from a pre-parsed {@link JsonNode}.
     * Callers that already hold a parsed node should prefer this overload to avoid re-parsing.
     *
     * @param node pre-parsed completion response node
     * @return parsed {@link LlamaOutput}
     */
    public static LlamaOutput parse(JsonNode node) {
        String content = extractContent(node);
        boolean stop = node.path("stop").asBoolean(false);
        Map<String, Float> probabilities = parseProbabilities(node);
        StopReason stopReason = stop ? StopReason.fromJson(node) : StopReason.NONE;
        return new LlamaOutput(content, probabilities, stop, stopReason);
    }

    /**
     * Extract the {@code "content"} string from a completion response node.
     * Returns an empty string if the field is absent.
     *
     * @param node completion response node
     * @return the content string, or {@code ""} if absent
     */
    public static String extractContent(JsonNode node) {
        return node.path("content").asText("");
    }

    /**
     * Parse the {@code completion_probabilities} array into a {@code token → probability} map.
     *
     * <p>Each array entry carries the generated token and either a {@code "prob"} value
     * (post-sampling mode) or {@code "logprob"} (pre-sampling mode). The nested
     * {@code top_probs}/{@code top_logprobs} arrays are invisible at the outer entry level
     * and do not interfere with field lookup.
     *
     * <p>Returns an empty map when the field is absent or the array is empty.
     * Requires {@code InferenceParameters#setNProbs(int)} to be configured before inference.
     *
     * @param root the top-level completion response node
     * @return map from token string to probability; empty when no probability data is present
     */
    public static Map<String, Float> parseProbabilities(JsonNode root) {
        JsonNode array = root.path("completion_probabilities");
        if (!array.isArray() || array.size() == 0) {
            return Collections.emptyMap();
        }
        Map<String, Float> result = new HashMap<String, Float>();
        for (JsonNode entry : array) {
            String token = entry.path("token").asText("");
            if (token.isEmpty()) continue;

            // "prob" (post-sampling) or "logprob" (pre-sampling)
            JsonNode probNode = entry.path("prob");
            if (probNode.isMissingNode() || probNode.isNull()) {
                probNode = entry.path("logprob");
            }
            if (probNode.isMissingNode() || probNode.isNull()) continue;

            result.put(token, (float) probNode.asDouble(0.0));
        }
        return result.isEmpty() ? Collections.<String, Float>emptyMap() : result;
    }
}
