package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An output of the LLM providing access to the generated text and the associated probabilities. You have to configure
 * {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
 */
public final class LlamaOutput {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * The last bit of generated text that is representable as text (i.e., cannot be individual utf-8 multibyte code
     * points).
     */
    @NotNull
    public final String text;

    /**
     * Note, that you have to configure {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
     */
    @NotNull
    public final Map<String, Float> probabilities;

    /** Whether this is the final token of the generation. */
    public final boolean stop;

    /**
     * The reason generation stopped. {@link StopReason#NONE} on intermediate streaming tokens.
     * Only meaningful when {@link #stop} is {@code true}.
     */
    @NotNull
    public final StopReason stopReason;

    LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities, boolean stop, @NotNull StopReason stopReason) {
        this.text = text;
        this.probabilities = probabilities;
        this.stop = stop;
        this.stopReason = stopReason;
    }

    @Override
    public String toString() {
        return text;
    }

    /**
     * Parse a LlamaOutput from a JSON string returned by the native receiveCompletionJson method.
     * The JSON has the structure: {"content": "...", "stop": true/false, ...}
     */
    static LlamaOutput fromJson(String json) {
        try {
            return fromJson(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new LlamaOutput("", Collections.<String, Float>emptyMap(), false, StopReason.NONE);
        }
    }

    /**
     * Parse a LlamaOutput from a pre-parsed JsonNode. This is the primary implementation;
     * {@link #fromJson(String)} is a thin wrapper that parses the string once and delegates here.
     */
    static LlamaOutput fromJson(JsonNode node) {
        String content = node.path("content").asText("");
        boolean stop = node.path("stop").asBoolean(false);
        Map<String, Float> probabilities = parseProbabilities(node);
        StopReason stopReason = stop ? StopReason.fromJson(node) : StopReason.NONE;
        return new LlamaOutput(content, probabilities, stop, stopReason);
    }

    /**
     * Extract the "content" field value from a JSON string.
     *
     * <p>For well-formed JSON objects, Jackson is used directly. For substring fragments
     * (used by {@code chatCompleteText()} and test helpers that pass
     * {@code json.substring(contentIdx)} starting at the {@code "content"} key), the
     * method falls back to a manual character scan so those callers continue to work
     * without modification.
     */
    static String getContentFromJson(String json) {
        // Fast path: try Jackson for a complete JSON object.
        try {
            JsonNode root = OBJECT_MAPPER.readTree(json);
            if (root != null && root.isObject()) {
                return root.path("content").asText("");
            }
        } catch (IOException ignored) {
            // Fall through to the substring scanner below.
        }

        // Fallback: manual scan for callers that pass a substring fragment beginning at
        // the "content" key rather than a complete JSON object.
        int keyIdx = json.indexOf("\"content\"");
        if (keyIdx < 0) {
            return "";
        }
        int colonIdx = json.indexOf(':', keyIdx + 9);
        if (colonIdx < 0) {
            return "";
        }
        int startQuote = json.indexOf('"', colonIdx + 1);
        if (startQuote < 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = startQuote + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case '"':  sb.append('"');  i++; break;
                    case '\\': sb.append('\\'); i++; break;
                    case '/':  sb.append('/');  i++; break;
                    case 'n':  sb.append('\n'); i++; break;
                    case 'r':  sb.append('\r'); i++; break;
                    case 't':  sb.append('\t'); i++; break;
                    case 'b':  sb.append('\b'); i++; break;
                    case 'f':  sb.append('\f'); i++; break;
                    case 'u':
                        if (i + 5 < json.length()) {
                            String hex = json.substring(i + 2, i + 6);
                            sb.append((char) Integer.parseInt(hex, 16));
                            i += 5;
                        }
                        break;
                    default: sb.append('\\').append(next); i++; break;
                }
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Parse token probabilities from a parsed JSON node. Returns an empty map when
     * {@code completion_probabilities} is absent or empty.
     *
     * <p>Each array entry has the structure:
     * <pre>{"token":"txt","bytes":[...],"id":N,"prob":F,"top_probs":[...]}</pre>
     * or with {@code "logprob"} instead of {@code "prob"} when post-sampling mode is off.
     * Jackson's field access is scoped to the outer object, so the nested
     * {@code top_probs}/{@code top_logprobs} arrays are invisible at this level.
     */
    private static Map<String, Float> parseProbabilities(JsonNode root) {
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
     * Parse rerank results from a JSON array string.
     * Expected format: [{"document": "...", "index": 0, "score": 0.95}, ...]
     */
    static List<Pair<String, Float>> parseRerankResults(String json) {
        try {
            JsonNode arr = OBJECT_MAPPER.readTree(json);
            if (!arr.isArray()) return Collections.emptyList();
            List<Pair<String, Float>> results = new ArrayList<Pair<String, Float>>();
            for (JsonNode entry : arr) {
                String doc = entry.path("document").asText("");
                float score = (float) entry.path("score").asDouble(0.0);
                results.add(new Pair<String, Float>(doc, score));
            }
            return results;
        } catch (IOException e) {
            return Collections.emptyList();
        }
    }
}
