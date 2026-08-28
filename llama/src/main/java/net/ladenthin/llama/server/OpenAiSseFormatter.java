// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import org.jspecify.annotations.Nullable;

/**
 * Pure formatting helpers for the OpenAI HTTP surface: Server-Sent-Events framing, the {@code [DONE]}
 * terminator, heartbeat comments, the {@code GET /v1/models} body, and the OpenAI error envelope.
 *
 * <p>Stateless and free of JNI / model dependencies, so each helper is unit-testable with literals.
 */
final class OpenAiSseFormatter {

    /** Shared Jackson mapper; thread-safe and reused. */
    static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private OpenAiSseFormatter() {}

    /**
     * Frame a chunk's JSON as one SSE {@code data:} event.
     *
     * @param json the chunk JSON to send
     * @return the SSE event text, terminated by a blank line
     */
    static String sseData(String json) {
        return "data: " + json + "\n\n";
    }

    /**
     * The terminating SSE event that marks the end of an OpenAI stream.
     *
     * @return {@code "data: [DONE]\n\n"}
     */
    static String sseDone() {
        return "data: [DONE]\n\n";
    }

    /**
     * An SSE comment line used as a keep-alive heartbeat. OpenAI clients ignore comment lines, but the
     * bytes reset the client's stream-inactivity timer during long prompt prefill.
     *
     * @return {@code ": ping\n\n"}
     */
    static String heartbeat() {
        return ": ping\n\n";
    }

    /**
     * Build an OpenAI error envelope: {@code {"error":{"message":…,"type":…,"code":…}}}.
     *
     * @param message human-readable error message
     * @param type OpenAI error type (e.g. {@code "invalid_request_error"}, {@code "server_error"})
     * @param code optional machine-readable code; {@code null} renders as JSON {@code null}
     * @return the error envelope serialized as JSON
     */
    static String errorJson(String message, String type, @Nullable String code) {
        ObjectNode error = OBJECT_MAPPER.createObjectNode();
        error.put("message", message);
        error.put("type", type);
        if (code != null) {
            error.put("code", code);
        } else {
            error.putNull("code");
        }
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.set("error", error);
        return root.toString();
    }

    /**
     * Guarantee a streamed chunk's usage object carries {@code usage.prompt_tokens_details.cached_tokens}.
     *
     * <p>When {@code stream_options.include_usage} is set, the OpenAI streaming protocol emits a trailing
     * usage chunk. The VS&nbsp;Code Copilot custom endpoint throws
     * {@code Cannot read properties of undefined (reading 'cached_tokens')} (microsoft/vscode #273482) if
     * {@code usage.prompt_tokens_details.cached_tokens} is missing, and upstream llama.cpp does not always
     * populate it. This fills a default of {@code 0} when absent. Token-delta chunks (which carry no
     * non-null usage object) are returned unchanged and unparsed, so the streaming hot path is untouched.
     *
     * @param chunkJson one {@code chat.completion.chunk} serialized as JSON
     * @return the chunk JSON with {@code cached_tokens} guaranteed present inside any non-null usage object
     */
    static String ensureUsageCachedTokens(String chunkJson) {
        // Fast path: only the trailing usage chunk carries a non-null usage object — skip the rest unparsed.
        if (!chunkJson.contains("\"usage\"") || chunkJson.contains("\"usage\":null")) {
            return chunkJson;
        }
        try {
            JsonNode root = OBJECT_MAPPER.readTree(chunkJson);
            if (!root.isObject() || !root.path("usage").isObject()) {
                return chunkJson;
            }
            ObjectNode usage = (ObjectNode) root.get("usage");
            JsonNode details = usage.path("prompt_tokens_details");
            if (details.isObject()) {
                if (details.has("cached_tokens")) {
                    return chunkJson; // already correct — emit verbatim
                }
                ((ObjectNode) details).put("cached_tokens", 0);
            } else {
                usage.putObject("prompt_tokens_details").put("cached_tokens", 0);
            }
            return root.toString();
        } catch (IOException e) {
            // Never break a live stream over a formatting nicety.
            return chunkJson;
        }
    }

    /**
     * Build the {@code GET /v1/models} body advertising a single model.
     *
     * @param modelId the model id to advertise
     * @return an OpenAI model-list object serialized as JSON
     */
    static String modelsJson(String modelId) {
        return modelsJson(modelId, "");
    }

    /**
     * Build the {@code GET /v1/models} body advertising a single model, including the model's file
     * type (quantization) as a {@code data[].ftype} field when known — mirroring the upstream
     * llama.cpp server's {@code get_model_info()}.
     *
     * @param modelId the model id to advertise
     * @param ftype the model's file-type (quantization) label, or {@code ""}/{@code null} to omit it
     * @return an OpenAI model-list object serialized as JSON
     */
    static String modelsJson(String modelId, @Nullable String ftype) {
        ObjectNode model = OBJECT_MAPPER.createObjectNode();
        model.put("id", modelId);
        model.put("object", "model");
        model.put("owned_by", "llama.cpp");
        if (ftype != null && !ftype.isEmpty()) {
            model.put("ftype", ftype);
        }
        ArrayNode data = OBJECT_MAPPER.createArrayNode();
        data.add(model);
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("object", "list");
        root.set("data", data);
        return root.toString();
    }

    /**
     * Build one OpenAI {@code text_completion} streaming chunk for {@code POST /v1/completions}.
     *
     * @param id the completion id, stable across the whole stream
     * @param created the creation timestamp in epoch seconds
     * @param model the served model id
     * @param text the incremental token text carried by this chunk
     * @param finishReason the finish reason on the final chunk, or {@code null} for intermediate chunks
     * @return the chunk serialized as JSON
     */
    static String completionChunk(String id, long created, String model, String text, @Nullable String finishReason) {
        ObjectNode choice = OBJECT_MAPPER.createObjectNode();
        choice.put("text", text);
        choice.put("index", 0);
        choice.putNull("logprobs");
        if (finishReason == null) {
            choice.putNull("finish_reason");
        } else {
            choice.put("finish_reason", finishReason);
        }
        ArrayNode choices = OBJECT_MAPPER.createArrayNode();
        choices.add(choice);
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("id", id);
        root.put("object", "text_completion");
        root.put("created", created);
        root.put("model", model);
        root.set("choices", choices);
        return root.toString();
    }

    /**
     * Build the llama.cpp-native {@code GET /props} body. Autocomplete clients (e.g. llama.vscode) read
     * {@code default_generation_settings.n_ctx} from here to size their context window, and newer clients
     * read the {@code modalities} block to gate image input.
     *
     * <p>All three modality keys upstream emits are present, so a client that gates on
     * {@code modalities.video} sees an explicit {@code false} rather than a missing key.
     * {@code audio} and {@code video} are hard-coded {@code false} because this server exposes no
     * configuration for them &mdash; {@link OpenAiServerConfig} carries only
     * {@code supportsVision}.</p>
     *
     * @param modelId the served model id
     * @param nCtx the advertised context length
     * @param vision whether image input is supported
     * @return the props object serialized as JSON
     */
    static String propsJson(String modelId, int nCtx, boolean vision) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        ObjectNode defaults = root.putObject("default_generation_settings");
        defaults.put("n_ctx", nCtx);
        defaults.put("model", modelId);
        root.put("total_slots", 1);
        root.put("model_alias", modelId);
        root.put("chat_template", "");
        ObjectNode modalities = root.putObject("modalities");
        modalities.put("vision", vision);
        modalities.put("audio", false);
        modalities.put("video", false);
        return root.toString();
    }
}
