// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pure request-parsing and response-shaping helpers for the {@code POST /v1/rerank} route.
 *
 * <p>Reads the Jina/Cohere-style rerank request ({@code query} + {@code documents} [+ {@code top_n}]) and
 * reshapes the native llama.cpp rerank array ({@code [{document,index,score}]}) into the OpenAI-style
 * rerank response. The response carries both a {@code results} array (the standard llama.cpp/Jina shape)
 * and a {@code data} alias of the same entries, because Continue expects {@code data} and errors on
 * {@code results} (continuedev/continue #6478).
 *
 * <p>Stateless and free of JNI / model dependencies, so each helper is unit-testable with JSON literals.
 */
final class OaiRerankSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private OaiRerankSupport() {}

    /**
     * Read the required {@code query} string.
     *
     * @param request the parsed rerank request
     * @return the query text
     * @throws IllegalArgumentException if {@code query} is missing or not a string
     */
    static String readQuery(JsonNode request) {
        JsonNode query = request.path("query");
        if (!query.isTextual()) {
            throw new IllegalArgumentException("'query' must be a string");
        }
        return query.asText();
    }

    /**
     * Read the required {@code documents} array. Each entry may be a plain string or an object carrying a
     * {@code "text"} string (the Cohere/Jina document shape).
     *
     * @param request the parsed rerank request
     * @return the documents, in request order
     * @throws IllegalArgumentException if {@code documents} is absent, empty, or holds an unsupported entry
     */
    static String[] readDocuments(JsonNode request) {
        // Hugging Face text-embeddings-inference names them "texts"; upstream reads it when "documents" is absent.
        JsonNode documents = request.has("documents") ? request.path("documents") : request.path("texts");
        if (!documents.isArray() || documents.size() == 0) {
            throw new IllegalArgumentException("'documents' (or TEI's 'texts') must be a non-empty array");
        }
        List<String> out = new ArrayList<>(documents.size());
        for (JsonNode document : documents) {
            if (document.isTextual()) {
                out.add(document.asText());
            } else if (document.isObject() && document.path("text").isTextual()) {
                out.add(document.path("text").asText());
            } else {
                throw new IllegalArgumentException("each document must be a string or an object with a 'text' string");
            }
        }
        return out.toArray(new String[0]);
    }

    /**
     * Read the optional {@code top_n} cap.
     *
     * @param request the parsed rerank request
     * @return the requested cap, or {@code -1} when absent or not an integer
     */
    static int readTopN(JsonNode request) {
        JsonNode topN = request.path("top_n");
        return topN.isInt() ? topN.asInt() : -1;
    }

    /**
     * Shape the native rerank array as the response the request's format expects.
     *
     * @param request the parsed rerank request
     * @param nativeArrayJson the native {@code [{document,index,score}]} array as JSON
     * @param model the model id to echo (omitted when empty)
     * @return the rerank response JSON
     */
    static String toResponse(JsonNode request, String nativeArrayJson, String model) {
        if (request.has("texts")) {
            return toTeiResponse(
                    nativeArrayJson,
                    readTopN(request),
                    request.path("return_text").asBoolean(false));
        }
        return toOaiResponse(nativeArrayJson, model, readTopN(request));
    }

    /**
     * Reshape the native rerank array into the text-embeddings-inference response, as upstream does for a
     * request with {@code texts}: a bare array of {@code {index, score}}, best first, with the document as
     * {@code text} when {@code return_text} is set.
     *
     * @param nativeArrayJson the native {@code [{document,index,score}]} array as JSON
     * @param topN keep only the top-N highest-scoring entries, or {@code <= 0} to keep all
     * @param returnText whether to include each document's text
     * @return the TEI rerank response JSON
     */
    static String toTeiResponse(String nativeArrayJson, int topN, boolean returnText) {
        final List<ObjectNode> entries = new ArrayList<>();
        try {
            JsonNode arr = OBJECT_MAPPER.readTree(nativeArrayJson);
            if (arr.isArray()) {
                for (JsonNode entry : arr) {
                    ObjectNode result = OBJECT_MAPPER.createObjectNode();
                    result.put("index", entry.path("index").asInt());
                    result.put("score", entry.path("score").asDouble());
                    if (returnText) {
                        result.put("text", entry.path("document").asText(""));
                    }
                    entries.add(result);
                }
            }
        } catch (IOException e) {
            // The native call already succeeded; an unexpected body just yields an empty array.
            entries.clear();
        }
        entries.sort((a, b) ->
                Double.compare(b.path("score").asDouble(), a.path("score").asDouble()));
        final ArrayNode results = OBJECT_MAPPER.createArrayNode();
        final int limit = topN > 0 ? Math.min(topN, entries.size()) : entries.size();
        for (int i = 0; i < limit; i++) {
            results.add(entries.get(i));
        }
        return results.toString();
    }

    /**
     * Reshape the native rerank array into the OpenAI-style rerank response.
     *
     * @param nativeArrayJson the native {@code [{document,index,score}]} array as JSON
     * @param model the model id to echo (omitted when empty)
     * @param topN keep only the top-N highest-scoring entries, or {@code <= 0} to keep all
     * @return the rerank response JSON with sorted {@code results} and a {@code data} alias
     */
    static String toOaiResponse(String nativeArrayJson, String model, int topN) {
        final List<ObjectNode> entries = new ArrayList<>();
        try {
            JsonNode arr = OBJECT_MAPPER.readTree(nativeArrayJson);
            if (arr.isArray()) {
                for (JsonNode entry : arr) {
                    ObjectNode result = OBJECT_MAPPER.createObjectNode();
                    result.put("index", entry.path("index").asInt());
                    result.put("relevance_score", entry.path("score").asDouble());
                    entries.add(result);
                }
            }
        } catch (IOException e) {
            // The native call already succeeded; an unexpected body just yields empty results.
            entries.clear();
        }
        entries.sort((a, b) -> Double.compare(
                b.path("relevance_score").asDouble(), a.path("relevance_score").asDouble()));

        final ArrayNode results = OBJECT_MAPPER.createArrayNode();
        final int limit = topN > 0 ? Math.min(topN, entries.size()) : entries.size();
        for (int i = 0; i < limit; i++) {
            results.add(entries.get(i));
        }

        final ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("object", "list");
        if (!model.isEmpty()) {
            root.put("model", model);
        }
        root.set("results", results);
        root.set("data", results.deepCopy()); // alias for Continue (#6478), independent copy
        return root.toString();
    }
}
