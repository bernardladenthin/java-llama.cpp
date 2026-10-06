// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.value.RouterModel;

/**
 * Pure JSON transform for the router-mode model registry wire format (upstream
 * {@code GET /models} in router mode).
 *
 * <p>All methods are stateless and have zero dependency on JNI, native libraries, or llama
 * model state — they can be tested with JSON string literals alone (see
 * {@code RouterModelsResponseParserTest}).
 *
 * <p>The router response has the shape (see {@code get_router_models} in upstream
 * {@code tools/server/server-models.cpp}):
 * <pre>{@code
 * {
 *   "object": "list",
 *   "data": [
 *     {"id": "Qwen3-0.6B-Q4_K_M",
 *      "status": {"value": "loaded", "args": [...]},
 *      "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
 *      "source": "models_dir", ...},
 *     {"id": "broken-model",
 *      "status": {"value": "unloaded", "failed": true, "exit_code": 1}, ...}
 *   ]
 * }
 * }</pre>
 */
public class RouterModelsResponseParser {

    /** Creates a new {@link RouterModelsResponseParser}. */
    public RouterModelsResponseParser() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Parse the model list from a raw {@code GET /models} JSON response string. Delegates to
     * {@link #parse(JsonNode)} after a single {@code readTree} call.
     *
     * @param json raw router {@code GET /models} response JSON
     * @return list of models; empty list on parse failure
     */
    public List<RouterModel> parse(String json) {
        try {
            return parse(OBJECT_MAPPER.readTree(json));
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }

    /**
     * Parse the model list from a pre-parsed response node. Entries are read from the
     * {@code "data"} array (falling back to {@code "models"} for older/alternate shapes); the
     * identifier is read from {@code "id"} (falling back to {@code "name"}). The lifecycle
     * status comes from {@code status.value}; a missing status maps to
     * {@link RouterModel.Status#UNKNOWN} with an empty raw value. The failure marker is read
     * from {@code status.failed} / {@code status.exit_code}, the modalities from
     * {@code architecture.input_modalities} / {@code architecture.output_modalities} (empty when
     * absent, as from a server before llama.cpp b11429).
     *
     * @param root pre-parsed router {@code GET /models} response
     * @return list of models; empty list when no entry array is present
     */
    public List<RouterModel> parse(JsonNode root) {
        List<RouterModel> models = new ArrayList<RouterModel>();
        JsonNode data = root.path("data");
        if (!data.isArray()) {
            data = root.path("models");
        }
        if (!data.isArray()) {
            return models;
        }
        for (JsonNode entry : data) {
            String id = entry.path("id").asText(entry.path("name").asText(""));
            JsonNode status = entry.path("status");
            String statusValue = status.path("value").asText("");
            JsonNode architecture = entry.path("architecture");
            models.add(new RouterModel(
                    id,
                    RouterModel.Status.fromValue(statusValue),
                    statusValue,
                    status.path("failed").asBoolean(false),
                    status.path("exit_code").asInt(0),
                    strings(architecture.path("input_modalities")),
                    strings(architecture.path("output_modalities"))));
        }
        return models;
    }

    /** The string elements of a JSON array; empty for a missing node (servers before b11429). */
    private static List<String> strings(JsonNode array) {
        List<String> values = new ArrayList<String>();
        for (JsonNode value : array) {
            values.add(value.asText());
        }
        return values;
    }
}
