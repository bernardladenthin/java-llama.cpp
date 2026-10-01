// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.value.CompletionResult;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.StopReason;
import net.ladenthin.llama.value.Timings;
import net.ladenthin.llama.value.TokenLogprob;
import net.ladenthin.llama.value.Usage;

/**
 * Pure JSON transforms for native completion/streaming responses.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code CompletionResponseParserTest}).
 *
 * <p>The native server produces one JSON object per streamed token. By default only the
 * core fields are present:
 * <pre>{@code
 * {
 *   "content": "Hello",
 *   "stop": false,
 *   "stop_type": "none"
 * }
 * }</pre>
 *
 * <p>When inference is configured with {@link net.ladenthin.llama.parameters.InferenceParameters#withNProbs(int)} &gt; 0,
 * each chunk additionally carries a {@code completion_probabilities} array:
 * <pre>{@code
 * {
 *   "content": "Hello",
 *   "stop": false,
 *   "completion_probabilities": [
 *     {"token": "Hello", "bytes": [...], "id": 15043, "prob": 0.82,
 *      "top_probs": [{"token": "Hi", "bytes": [...], "id": 9932, "prob": 0.1}]}
 *   ]
 * }
 * }</pre>
 *
 * <p>This is the Java analogue of {@code json_helpers.hpp} in the C++ layer.
 */
public class CompletionResponseParser {

    /** Creates a new {@link CompletionResponseParser}. */
    public CompletionResponseParser() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Parse a {@link net.ladenthin.llama.value.LlamaOutput} from a raw JSON string returned by the native
     * {@code receiveCompletionJson} method. Delegates to {@link #parse(JsonNode)} after
     * a single {@code readTree} call so the string is parsed only once.
     *
     * @param json raw JSON string from the native completion response
     * @return parsed {@link net.ladenthin.llama.value.LlamaOutput}; empty output on parse failure
     */
    public LlamaOutput parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new LlamaOutput(
                    "",
                    Collections.<String, Float>emptyMap(),
                    Collections.<TokenLogprob>emptyList(),
                    false,
                    StopReason.NONE);
        }
    }

    /**
     * Parse a {@link net.ladenthin.llama.value.LlamaOutput} from a pre-parsed {@link JsonNode}.
     * Callers that already hold a parsed node should prefer this overload to avoid re-parsing.
     *
     * @param node pre-parsed completion response node
     * @return parsed {@link net.ladenthin.llama.value.LlamaOutput}
     */
    public LlamaOutput parse(JsonNode node) {
        String content = extractContent(node);
        boolean stop = node.path("stop").asBoolean(false);
        Map<String, Float> probabilities = parseProbabilities(node);
        List<TokenLogprob> logprobs = parseLogprobs(node);
        StopReason stopReason =
                stop ? StopReason.fromStopType(node.path("stop_type").asText("")) : StopReason.NONE;
        return new LlamaOutput(content, probabilities, logprobs, stop, stopReason);
    }

    /**
     * Extract the {@code "content"} string from a completion response node.
     * Returns an empty string if the field is absent.
     *
     * @param node completion response node
     * @return the content string, or {@code ""} if absent
     */
    public String extractContent(JsonNode node) {
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
     * Requires {@code InferenceParameters#withNProbs(int)} to be configured before inference.
     *
     * <p><b>Lossy on duplicate token text:</b> this flat map is keyed by token string, so when the
     * same token text occurs more than once in a generation (common for repeated words/punctuation)
     * only the <em>last</em> occurrence's probability survives. Use {@link #parseLogprobs(JsonNode)}
     * (an order-preserving list) when every per-token value matters.
     *
     * @param root the top-level completion response node
     * @return map from token string to probability; empty when no probability data is present
     */
    public Map<String, Float> parseProbabilities(JsonNode root) {
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

    /**
     * Parse the {@code completion_probabilities} array into a list of typed {@link net.ladenthin.llama.value.TokenLogprob}
     * entries, preserving order, token ids, and the nested alternatives array
     * ({@code top_probs} for post-sampling mode or {@code top_logprobs} for pre-sampling).
     *
     * <p>Returns an empty list when the field is absent or empty. Requires
     * {@link net.ladenthin.llama.parameters.InferenceParameters#withNProbs(int)} to be configured.
     *
     * @param root the top-level completion response node
     * @return list of {@link net.ladenthin.llama.value.TokenLogprob}; empty when no probability data is present
     */
    public List<TokenLogprob> parseLogprobs(JsonNode root) {
        JsonNode array = root.path("completion_probabilities");
        // Single mutable-ArrayList return: an empty (or absent) array falls
        // through the loop and returns the same empty ArrayList, keeping the
        // return type consistent (Error Prone MixedMutabilityReturnType) and
        // leaving no equivalent empty-branch mutant for PIT to flag.
        List<TokenLogprob> result = new ArrayList<>();
        if (array.isArray()) {
            for (JsonNode entry : array) {
                result.add(parseLogprobEntry(entry));
            }
        }
        return result;
    }

    /**
     * Parse a {@link net.ladenthin.llama.value.CompletionResult} from the non-streaming, non-OAI completion JSON
     * emitted by {@code server_task_result_cmpl_final::to_json_non_oaicompat}.
     * <p>
     * Maps {@code content} → text, {@code tokens_evaluated}/{@code tokens_predicted} →
     * {@link net.ladenthin.llama.value.Usage}, the {@code timings} sub-object → {@link net.ladenthin.llama.value.Timings},
     * {@code completion_probabilities} → {@link net.ladenthin.llama.value.TokenLogprob} list, and
     * {@code stop_type} → {@link net.ladenthin.llama.value.StopReason}.
     *
     * @param json raw JSON string from the native completion response
     * @return a populated {@link net.ladenthin.llama.value.CompletionResult}; fields default to empty/zero on parse failure
     */
    public CompletionResult parseCompletionResult(String json) {
        try {
            JsonNode node = OBJECT_MAPPER.readTree(json);
            String text = extractContent(node);
            Timings timings = Timings.fromJson(node.path("timings"));
            Usage usage = new Usage(
                    node.path("tokens_evaluated").asLong(0L),
                    node.path("tokens_predicted").asLong(0L),
                    Math.max(0, timings.getCacheN()));
            TimingsLogger.log(timings);
            List<TokenLogprob> logprobs = parseLogprobs(node);
            StopReason stopReason =
                    StopReason.fromStopType(node.path("stop_type").asText(""));
            return new CompletionResult(text, usage, timings, logprobs, stopReason, json);
        } catch (IOException e) {
            return new CompletionResult(
                    "",
                    new Usage(0L, 0L),
                    Timings.fromJson(null),
                    Collections.<TokenLogprob>emptyList(),
                    StopReason.NONE,
                    json);
        }
    }

    private TokenLogprob parseLogprobEntry(JsonNode entry) {
        String token = entry.path("token").asText("");
        int tokenId = entry.path("id").asInt(-1);
        JsonNode probNode = entry.path("prob");
        if (probNode.isMissingNode() || probNode.isNull()) {
            probNode = entry.path("logprob");
        }
        float logprob = (float) probNode.asDouble(0.0);

        JsonNode top = entry.path("top_probs");
        if (!top.isArray()) {
            top = entry.path("top_logprobs");
        }
        // Single mutable-ArrayList accumulation: a missing or empty nested array
        // skips the loop and yields an empty ArrayList, so there is no equivalent
        // empty-branch mutant (the prior emptyList()/ArrayList ternary left one).
        List<TokenLogprob> topLogprobs = new ArrayList<>();
        if (top.isArray()) {
            for (JsonNode t : top) {
                topLogprobs.add(parseLogprobEntry(t));
            }
        }
        return new TokenLogprob(token, tokenId, logprob, topLogprobs);
    }
}
