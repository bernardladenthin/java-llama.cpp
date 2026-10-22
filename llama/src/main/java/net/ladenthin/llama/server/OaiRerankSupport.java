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
import org.jspecify.annotations.Nullable;

/**
 * Pure request-parsing and response-shaping helpers for the {@code POST /v1/rerank} route.
 *
 * <p>Reads either the Jina/Cohere-style rerank request ({@code query} + {@code documents} [+
 * {@code top_n}]) or Hugging Face text-embeddings-inference's ({@code query} + {@code texts} [+
 * {@code top_n}, {@code return_text}]) and reshapes the native llama.cpp rerank array
 * ({@code [{document,index,score}]}). {@link #toResponse} picks the answer by the presence of
 * {@code texts}, as upstream's {@code format_response_rerank} does. The OpenAI-style answer carries a
 * {@code results} array (the standard llama.cpp/Jina shape), a {@code data} alias of the same entries,
 * because Continue expects {@code data} and errors on {@code results} (continuedev/continue #6478), and a
 * {@code usage} object ({@code prompt_tokens} = {@code total_tokens} = the sum of the documents'
 * {@code tokens_evaluated}). The TEI answer is a bare array of {@code {index, score}}, best first, plus
 * {@code text} when {@code return_text} is true.
 *
 * <p>Stateless and free of JNI / model dependencies, so each helper is unit-testable with JSON literals.
 */
final class OaiRerankSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** How much of an unexpected native body a refusal quotes. */
    private static final int NATIVE_BODY_EXCERPT = 80;

    private OaiRerankSupport() {}

    /**
     * Read the required {@code query}: a string, or an object whose {@code "text"} is a string or an
     * array of strings (concatenated), the item shape llama.cpp's rerank route reads since b11820 (#29625;
     * see {@link #readItem}).
     *
     * @param request the parsed rerank request
     * @return the query text
     * @throws IllegalArgumentException if {@code query} is missing or not such an item, or carries an image
     */
    static String readQuery(JsonNode request) {
        JsonNode query = request.path("query");
        if (query.isMissingNode() || query.isNull()) {
            throw new IllegalArgumentException("'query' must be provided");
        }
        return readItem(query, "query");
    }

    /**
     * Read the required {@code documents} array. Each entry may be a plain string or an object carrying a
     * {@code "text"} string or array of strings (the Cohere/Jina document shape, read as {@link #readItem}
     * reads it).
     *
     * @param request the parsed rerank request
     * @return the documents, in request order
     * @throws IllegalArgumentException if {@code documents} (or, when it is absent or {@code null}, TEI's
     *     {@code texts}) is absent, empty, or holds an unsupported entry
     */
    static String[] readDocuments(JsonNode request) {
        // Hugging Face text-embeddings-inference names them "texts"; upstream reads it when "documents" is
        // absent or null (its json_value treats an explicit null as absent).
        JsonNode documents = request.hasNonNull("documents") ? request.path("documents") : request.path("texts");
        if (!documents.isArray() || documents.size() == 0) {
            throw new IllegalArgumentException("'documents' (or TEI's 'texts') must be a non-empty array");
        }
        List<String> out = new ArrayList<>(documents.size());
        for (JsonNode document : documents) {
            out.add(readItem(document, "document"));
        }
        return out.toArray(new String[0]);
    }

    /**
     * Read one rerank input item as llama.cpp's {@code handle_rerank_impl} reads it (b11820, #29625): a
     * string, or an object whose {@code "text"} is a string or an array of strings (concatenated) and is
     * empty text when absent or {@code null}; {@code "video"}, {@code "fps"} and {@code "max_frames"} are
     * refused, as upstream refuses them.
     *
     * <p>One difference, a refusal rather than another answer: an {@code "image"} (a string, or an array
     * with a string) upstream ranks through the multimodal projector, which this transport does not drive
     * ({@link net.ladenthin.llama.LlamaModel}'s rerank takes text), so it is refused -- ranking the text
     * alone would answer another question than the one asked. {@link NativeServer} serves llama.cpp's own
     * route. An {@code "image"} that is {@code null} or an empty array carries no image there either, and
     * is ignored here as there.
     *
     * @param item the query or one document
     * @param what {@code "query"} or {@code "document"}, for the message
     * @return the item's text
     * @throws IllegalArgumentException if the item has none of these shapes or carries an image
     */
    private static String readItem(JsonNode item, String what) {
        if (item.isTextual()) {
            return item.asText();
        }
        if (!item.isObject()) {
            throw new IllegalArgumentException(("query".equals(what) ? "'query'" : "each " + what)
                    + " must be a string or an object with a 'text' string");
        }
        final String in = " (in the " + what + ")";
        if (item.has("video") || item.has("fps") || item.has("max_frames")) {
            throw new IllegalArgumentException("video/fps/max_frames are not supported for rerank input" + in);
        }
        final StringBuilder out = new StringBuilder();
        final JsonNode text = item.path("text");
        if (text.isTextual()) {
            out.append(text.asText());
        } else if (text.isArray()) {
            for (JsonNode part : text) {
                if (!part.isTextual()) {
                    throw new IllegalArgumentException("\"text\" array elements must be strings" + in);
                }
                out.append(part.asText());
            }
        } else if (!text.isMissingNode() && !text.isNull()) {
            throw new IllegalArgumentException("\"text\" must be a string or an array of strings" + in);
        }
        // upstream decodes the images in order, so the first element decides which refusal comes first
        final JsonNode image = item.path("image");
        final boolean list = image.isArray() && image.size() > 0;
        if (list && !image.get(0).isTextual()) {
            throw new IllegalArgumentException("\"image\" array elements must be strings" + in);
        }
        if (image.isTextual() || list) {
            throw new IllegalArgumentException("an 'image' in a rerank " + what
                    + " is ranked by llama.cpp's own server only (NativeServer); this transport ranks text");
        }
        if (!image.isArray() && !image.isMissingNode() && !image.isNull()) {
            throw new IllegalArgumentException("\"image\" must be a string or an array of strings" + in);
        }
        return out.toString();
    }

    /**
     * Read the optional {@code top_n} cap. {@code 0} keeps no result, as upstream's {@code
     * format_response_rerank} does ({@code min(top_n, n)} entries).
     *
     * <p>A JSON number is read as upstream reads it ({@code get<int>()}, a {@code static_cast}): a float
     * is truncated toward zero, so {@code 2.0} and {@code 2.7} keep 2. Two deliberate differences: a
     * value that is no number (a string, or a boolean, which upstream would convert to 1 or 0) keeps
     * all, and so does a number outside the {@code int} range, which upstream's cast wraps (and the
     * request then fails or keeps nothing in the server).
     *
     * @param request the parsed rerank request
     * @return the requested cap, or {@code -1} when absent, not a number or outside the {@code int}
     *     range
     * @throws IllegalArgumentException if {@code top_n} is negative, integer or float (upstream refuses
     *     it too since b11820, #29625, with {@code "top_n" must be >= 0}; before, it resized its result
     *     vector to {@code min(top_n, n)}, a huge size for a negative value, and failed inside the server)
     */
    static int readTopN(JsonNode request) {
        JsonNode topN = request.path("top_n");
        if (!topN.isNumber() || !topN.canConvertToInt()) {
            return -1;
        }
        final int value = topN.asInt();
        if (value < 0) {
            throw new IllegalArgumentException("'top_n' must not be negative but was " + value);
        }
        return value;
    }

    /** The number of entries a {@code top_n} keeps: all for {@code -1} (absent), else at most it. */
    private static int limit(int topN, int size) {
        return topN < 0 ? size : Math.min(topN, size);
    }

    /**
     * Parse the native rerank body, which must be a JSON array. Anything else is a broken result, not
     * an empty one: answering it with no results would read as "nothing matched". An empty body is
     * checked as well, since Jackson reads it as a missing node rather than failing. The failure is an
     * {@link IOException}, the type {@link OpenAiBackend#rerank} declares for a server error.
     *
     * @param nativeArrayJson the native {@code [{document,index,score}]} array as JSON
     * @return the parsed array
     * @throws IOException if the body is not a JSON array
     */
    private static JsonNode readNativeArray(String nativeArrayJson) throws IOException {
        final JsonNode arr;
        try {
            arr = OBJECT_MAPPER.readTree(nativeArrayJson);
        } catch (IOException e) {
            throw notANativeArray(nativeArrayJson, e);
        }
        if (!arr.isArray()) {
            throw notANativeArray(nativeArrayJson, null);
        }
        return arr;
    }

    private static IOException notANativeArray(String nativeArrayJson, @Nullable IOException cause) {
        final String excerpt = nativeArrayJson.length() <= NATIVE_BODY_EXCERPT
                ? nativeArrayJson
                : nativeArrayJson.substring(0, NATIVE_BODY_EXCERPT) + "...";
        return new IOException(
                "Failed to parse the native rerank response, expected a JSON array but got: '" + excerpt + "'", cause);
    }

    /**
     * Shape the native rerank array as the response the request's format expects.
     *
     * @param request the parsed rerank request
     * @param nativeArrayJson the native {@code [{document,index,score}]} array as JSON
     * @param model the model id to echo (omitted when empty)
     * @return the rerank response JSON
     * @throws IOException if {@code nativeArrayJson} is not a JSON array
     */
    static String toResponse(JsonNode request, String nativeArrayJson, String model) throws IOException {
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
     * @param topN keep only the top-N highest-scoring entries ({@code 0} keeps none), or {@code -1} to
     *     keep all
     * @param returnText whether to include each document's text
     * @return the TEI rerank response JSON
     * @throws IOException if {@code nativeArrayJson} is not a JSON array
     */
    static String toTeiResponse(String nativeArrayJson, int topN, boolean returnText) throws IOException {
        final List<ObjectNode> entries = new ArrayList<>();
        for (JsonNode entry : readNativeArray(nativeArrayJson)) {
            ObjectNode result = OBJECT_MAPPER.createObjectNode();
            result.put("index", entry.path("index").asInt());
            result.put("score", entry.path("score").asDouble());
            if (returnText) {
                result.put("text", entry.path("document").asText(""));
            }
            entries.add(result);
        }
        entries.sort((a, b) ->
                Double.compare(b.path("score").asDouble(), a.path("score").asDouble()));
        final ArrayNode results = OBJECT_MAPPER.createArrayNode();
        final int limit = limit(topN, entries.size());
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
     * @param topN keep only the top-N highest-scoring entries ({@code 0} keeps none), or {@code -1} to
     *     keep all
     * @return the rerank response JSON with sorted {@code results}, a {@code data} alias and the
     *     {@code usage} of the evaluated tokens
     * @throws IOException if {@code nativeArrayJson} is not a JSON array
     */
    static String toOaiResponse(String nativeArrayJson, String model, int topN) throws IOException {
        final List<ObjectNode> entries = new ArrayList<>();
        long promptTokens = 0;
        for (JsonNode entry : readNativeArray(nativeArrayJson)) {
            ObjectNode result = OBJECT_MAPPER.createObjectNode();
            result.put("index", entry.path("index").asInt());
            result.put("relevance_score", entry.path("score").asDouble());
            entries.add(result);
            promptTokens += entry.path("tokens_evaluated").asLong(0L);
        }
        entries.sort((a, b) -> Double.compare(
                b.path("relevance_score").asDouble(), a.path("relevance_score").asDouble()));

        final ArrayNode results = OBJECT_MAPPER.createArrayNode();
        final int limit = limit(topN, entries.size());
        for (int i = 0; i < limit; i++) {
            results.add(entries.get(i));
        }

        final ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("object", "list");
        if (!model.isEmpty()) {
            root.put("model", model);
        }
        // upstream's usage: every token evaluated for a document is a prompt token
        root.putObject("usage").put("prompt_tokens", promptTokens).put("total_tokens", promptTokens);
        root.set("results", results);
        root.set("data", results.deepCopy()); // alias for Continue (#6478), independent copy
        return root.toString();
    }
}
