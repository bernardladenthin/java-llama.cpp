package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Pure JSON transforms for native rerank responses.
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code RerankResponseParserTest}).
 *
 * <p>The native server produces a JSON array of reranked results:
 * <pre>{@code
 * [
 *   {"document": "The quick brown fox", "index": 0, "score": 0.92},
 *   {"document": "Another document",    "index": 1, "score": 0.43}
 * ]
 * }</pre>
 */
public final class RerankResponseParser {

    /** Shared Jackson mapper; all methods are stateless and thread-safe. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private RerankResponseParser() {}

    /**
     * Parse rerank results from a raw JSON array string. Delegates to {@link #parse(JsonNode)}
     * after a single {@code readTree} call.
     *
     * @param json raw JSON array string from the native rerank response
     * @return list of document/score pairs; empty list on parse failure or empty array
     */
    public static List<Pair<String, Float>> parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return Collections.emptyList();
        }
    }

    /**
     * Parse rerank results from a pre-parsed {@link JsonNode} array.
     * Each element must contain {@code "document"} (string) and {@code "score"} (number).
     * Returns an empty list when the node is not an array or is empty.
     *
     * @param arr pre-parsed {@link JsonNode} array of rerank result objects
     * @return list of document/score pairs; empty list if the node is not an array or is empty
     */
    public static List<Pair<String, Float>> parse(JsonNode arr) {
        if (!arr.isArray() || arr.size() == 0) {
            return Collections.emptyList();
        }
        List<Pair<String, Float>> results = new ArrayList<Pair<String, Float>>();
        for (JsonNode entry : arr) {
            String doc = entry.path("document").asText("");
            float score = (float) entry.path("score").asDouble(0.0);
            results.add(new Pair<String, Float>(doc, score));
        }
        return results;
    }
}
