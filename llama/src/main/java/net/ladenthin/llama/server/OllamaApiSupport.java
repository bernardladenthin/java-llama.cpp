// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.time.Instant;
import org.jspecify.annotations.Nullable;

/**
 * Pure translators between Ollama's native HTTP API and the OpenAI chat shape used internally, so the
 * server can present an Ollama-compatible surface (which Copilot's built-in <em>Ollama</em> provider and
 * Ollama-hardcoded tools target) without a second inference path.
 *
 * <p>Covers the discovery endpoints ({@code /api/version}, {@code /api/tags}, {@code /api/show}) and the
 * {@code /api/chat} request/response translation, including the NDJSON streaming shape (one JSON object
 * per line, terminated by a {@code "done":true} line). Tool-call {@code arguments} are objects in Ollama
 * but JSON-encoded strings in OpenAI, so they are converted on the way in and out.
 *
 * <p>Stateless and free of JNI / model dependencies; unit-testable with JSON literals.
 */
final class OllamaApiSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** Advertised Ollama API version (clients only check that {@code /api/version} responds with one). */
    static final String OLLAMA_VERSION = "0.1.0";

    private OllamaApiSupport() {}

    private static String nowIso() {
        return Instant.now().toString();
    }

    /**
     * The {@code GET /api/version} body.
     *
     * @return {@code {"version":"<v>"}}
     */
    static String versionJson() {
        return OBJECT_MAPPER.createObjectNode().put("version", OLLAMA_VERSION).toString();
    }

    /**
     * The {@code GET /api/tags} body advertising the single served model.
     *
     * @param modelId the model id
     * @return an Ollama model-list object serialized as JSON
     */
    static String tagsJson(String modelId) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        ArrayNode models = root.putArray("models");
        ObjectNode model = models.addObject();
        model.put("name", modelId);
        model.put("model", modelId);
        model.put("modified_at", nowIso());
        model.put("size", 0L);
        model.put("digest", "");
        ObjectNode details = model.putObject("details");
        details.put("family", "llama");
        details.put("parameter_size", "");
        details.put("quantization_level", "");
        return root.toString();
    }

    /**
     * The {@code POST /api/show} body advertising the model's capabilities and context length — the
     * fields Copilot's Ollama provider reads to enable tools/vision and size prompts.
     *
     * @param modelId the model id
     * @param contextLength the advertised context window
     * @param vision whether image input is supported
     * @return an Ollama show object serialized as JSON
     */
    static String showJson(String modelId, int contextLength, boolean vision) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("license", "");
        root.put("modelfile", "");
        root.put("parameters", "");
        root.put("template", "");
        ObjectNode details = root.putObject("details");
        details.put("family", "llama");
        details.putArray("families").add("llama");
        details.put("parameter_size", "");
        details.put("quantization_level", "");
        ObjectNode modelInfo = root.putObject("model_info");
        modelInfo.put("general.architecture", "llama");
        modelInfo.put("llama.context_length", contextLength);
        ArrayNode capabilities = root.putArray("capabilities");
        capabilities.add("completion");
        capabilities.add("tools");
        capabilities.add("insert"); // fill-in-the-middle via /infill
        if (vision) {
            capabilities.add("vision");
        }
        root.put("model", modelId);
        return root.toString();
    }

    /**
     * Whether the Ollama request asks for a streamed response. Ollama defaults {@code stream} to
     * {@code true} when the field is absent.
     *
     * @param ollamaRequest the parsed Ollama request
     * @return {@code true} unless {@code "stream"} is explicitly {@code false}
     */
    static boolean isStreaming(JsonNode ollamaRequest) {
        JsonNode stream = ollamaRequest.path("stream");
        return !stream.isBoolean() || stream.asBoolean();
    }

    /**
     * Translate an Ollama {@code /api/chat} request into the internal OpenAI chat request shape.
     *
     * @param ollamaRequest the parsed Ollama request
     * @return an OpenAI {@code /v1/chat/completions} request object
     */
    static ObjectNode toOpenAiChatRequest(JsonNode ollamaRequest) {
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        if (ollamaRequest.path("model").isTextual()) {
            openAi.put("model", ollamaRequest.path("model").asText());
        }

        // Messages: copy through, converting any assistant tool_calls.arguments object to the OpenAI
        // JSON-encoded string form.
        ArrayNode messages = openAi.putArray("messages");
        for (JsonNode message : ollamaRequest.path("messages")) {
            messages.add(toOpenAiMessage(message));
        }

        JsonNode tools = ollamaRequest.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            openAi.set("tools", tools.deepCopy());
        }

        // Ollama's "think": false switches a model's thinking off, a string picks a level (gpt-oss);
        // true leaves the template's default. The chat mapper applies reasoning_effort as upstream does.
        JsonNode think = ollamaRequest.path("think");
        if (think.isBoolean() && !think.asBoolean()) {
            openAi.put("reasoning_effort", "none");
        } else if (think.isTextual() && !think.asText().isEmpty()) {
            openAi.put("reasoning_effort", think.asText());
        }

        // Ollama nests sampling under "options"; map the common knobs onto OpenAI top-level fields.
        JsonNode options = ollamaRequest.path("options");
        copyNumber(options, "temperature", openAi, "temperature");
        copyNumber(options, "top_p", openAi, "top_p");
        copyNumber(options, "top_k", openAi, "top_k");
        copyNumber(options, "seed", openAi, "seed");
        copyNumber(options, "num_predict", openAi, "max_tokens");
        if (options.path("stop").isArray()) {
            openAi.set("stop", options.path("stop").deepCopy());
        }

        // Ollama "format": "json" or a JSON schema → OpenAI response_format.
        JsonNode format = ollamaRequest.path("format");
        if (format.isTextual() && "json".equals(format.asText())) {
            openAi.putObject("response_format").put("type", "json_object");
        } else if (format.isObject()) {
            ObjectNode responseFormat = openAi.putObject("response_format");
            responseFormat.put("type", "json_schema");
            responseFormat.putObject("json_schema").set("schema", format.deepCopy());
        }
        return openAi;
    }

    private static ObjectNode toOpenAiMessage(JsonNode ollamaMessage) {
        ObjectNode message = ollamaMessage.deepCopy();
        // Ollama carries images as an "images" array of bare base64 strings beside "content"; the
        // OpenAI parser reads only image_url content parts, so without this the images were dropped
        // and a vision model answered as if none had been sent.
        JsonNode images = message.remove("images");
        if (images != null && images.isArray() && images.size() > 0) {
            ArrayNode parts = OBJECT_MAPPER.createArrayNode();
            JsonNode content = message.path("content");
            if (content.isTextual() && !content.asText().isEmpty()) {
                parts.addObject().put("type", "text").put("text", content.asText());
            }
            for (JsonNode image : images) {
                if (image.isTextual()) {
                    parts.addObject()
                            .put("type", "image_url")
                            .putObject("image_url")
                            .put("url", toImageDataUrl(image.asText()));
                }
            }
            message.set("content", parts);
        }
        JsonNode toolCalls = message.path("tool_calls");
        if (toolCalls.isArray()) {
            for (JsonNode toolCall : toolCalls) {
                // Ollama's tool calls carry no type; llama.cpp rejects a tool call without one.
                if (toolCall.isObject() && !toolCall.has("type")) {
                    ((ObjectNode) toolCall).put("type", "function");
                }
                JsonNode arguments = toolCall.path("function").path("arguments");
                if (arguments.isObject() || arguments.isArray()) {
                    ((ObjectNode) toolCall.path("function")).put("arguments", arguments.toString());
                }
            }
        }
        return message;
    }

    /**
     * Turns an Ollama image (bare base64, or already a data URL) into the data URL an OpenAI
     * {@code image_url} part carries. The MIME type is read from the encoded magic bytes; llama.cpp
     * decodes by content and only requires the {@code data:image/...;base64,} shape, so an unknown
     * format falls back to {@code image/png}.
     */
    private static String toImageDataUrl(String image) {
        if (image.startsWith("data:")) {
            return image;
        }
        String mimeType;
        if (image.startsWith("/9j/")) {
            mimeType = "image/jpeg";
        } else if (image.startsWith("R0lGOD")) {
            mimeType = "image/gif";
        } else if (image.startsWith("UklGR")) {
            mimeType = "image/webp";
        } else {
            mimeType = "image/png";
        }
        return "data:" + mimeType + ";base64," + image;
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = from.path(fromKey);
        if (value.isNumber()) {
            to.set(toKey, value);
        }
    }

    /**
     * Translate a non-streaming OpenAI {@code chat.completion} into an Ollama {@code /api/chat} response.
     *
     * @param openAiCompletionJson the OpenAI completion body
     * @param model the model id to echo
     * @return the Ollama chat response serialized as JSON
     */
    static String toOllamaChatResponse(String openAiCompletionJson, String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        ObjectNode message = root.putObject("message");
        message.put("role", "assistant");
        message.put("content", "");
        String doneReason = "stop";
        try {
            JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
            JsonNode choice = completion.path("choices").path(0);
            JsonNode openAiMessage = choice.path("message");
            message.put("content", openAiMessage.path("content").asText(""));
            ArrayNode ollamaToolCalls = toOllamaToolCalls(openAiMessage.path("tool_calls"));
            if (ollamaToolCalls.size() > 0) {
                message.set("tool_calls", ollamaToolCalls);
            }
            if (choice.path("finish_reason").isTextual()) {
                doneReason = choice.path("finish_reason").asText();
            }
            JsonNode usage = completion.path("usage");
            if (usage.isObject()) {
                root.put("prompt_eval_count", usage.path("prompt_tokens").asLong(0));
                root.put("eval_count", usage.path("completion_tokens").asLong(0));
            }
        } catch (IOException e) {
            // Defensive: an unexpected body still yields a valid, empty Ollama "done" response.
            doneReason = "stop";
        }
        root.put("done", true);
        root.put("done_reason", doneReason);
        return root.toString();
    }

    /**
     * Translate one streamed OpenAI chunk into an Ollama NDJSON content line, or return {@code null} when
     * the chunk carries no assistant text to emit (role-only, finish-only or usage-only chunks).
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @param model the model id to echo
     * @return the Ollama NDJSON line (with trailing newline), or {@code null} if nothing to emit
     */
    static @Nullable String toOllamaContentLine(String openAiChunkJson, String model) {
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            JsonNode content = chunk.path("choices").path(0).path("delta").path("content");
            if (!content.isTextual() || content.asText().isEmpty()) {
                return null;
            }
            ObjectNode root = OBJECT_MAPPER.createObjectNode();
            root.put("model", model);
            root.put("created_at", nowIso());
            ObjectNode message = root.putObject("message");
            message.put("role", "assistant");
            message.put("content", content.asText());
            root.put("done", false);
            return root.toString() + "\n";
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Build the terminating Ollama NDJSON line ({@code "done":true}), attaching any tool calls
     * reconstructed from the stream.
     *
     * @param model the model id to echo
     * @param accumulator the tool-call accumulator fed with the stream's chunks
     * @return the final Ollama NDJSON line (with trailing newline)
     */
    static String toOllamaDoneLine(String model, ToolCallDeltaAccumulator accumulator) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        ObjectNode message = root.putObject("message");
        message.put("role", "assistant");
        message.put("content", "");
        if (accumulator.hasToolCalls()) {
            ArrayNode ollamaToolCalls = toOllamaToolCalls(accumulator.toOpenAiToolCalls());
            if (ollamaToolCalls.size() > 0) {
                message.set("tool_calls", ollamaToolCalls);
            }
        }
        root.put("done", true);
        root.put("done_reason", "stop");
        return root.toString() + "\n";
    }

    /** Convert OpenAI tool calls (arguments = JSON string) to Ollama tool calls (arguments = object). */
    private static ArrayNode toOllamaToolCalls(JsonNode openAiToolCalls) {
        ArrayNode out = OBJECT_MAPPER.createArrayNode();
        if (!openAiToolCalls.isArray()) {
            return out;
        }
        for (JsonNode openAiToolCall : openAiToolCalls) {
            JsonNode function = openAiToolCall.path("function");
            ObjectNode ollamaToolCall = out.addObject();
            ObjectNode ollamaFunction = ollamaToolCall.putObject("function");
            ollamaFunction.put("name", function.path("name").asText(""));
            ollamaFunction.set("arguments", parseArgumentsToObject(function.path("arguments")));
        }
        return out;
    }

    private static JsonNode parseArgumentsToObject(JsonNode arguments) {
        if (arguments.isObject() || arguments.isArray()) {
            return arguments;
        }
        if (arguments.isTextual()) {
            try {
                // readTree("") returns a MissingNode, which serializes as null, rather than throwing;
                // a parameterless call's empty arguments must still reach Ollama as an object.
                JsonNode parsed = OBJECT_MAPPER.readTree(arguments.asText());
                if (parsed.isObject() || parsed.isArray()) {
                    return parsed;
                }
            } catch (IOException e) {
                // Fall through to an empty object on unparseable arguments.
            }
        }
        return OBJECT_MAPPER.createObjectNode();
    }

    // ----- /api/generate (prompt completion / FIM) -----

    /**
     * Whether the {@code /api/generate} request carries a {@code suffix} (a fill-in-the-middle request).
     *
     * @param request the parsed Ollama generate request
     * @return {@code true} if a textual {@code suffix} is present
     */
    static boolean hasSuffix(JsonNode request) {
        return request.path("suffix").isTextual();
    }

    /**
     * Translate an Ollama {@code /api/generate} request into the internal OpenAI {@code /v1/completions}
     * request shape ({@code prompt} + sampling). Used when there is no {@code suffix}.
     *
     * @param request the parsed Ollama generate request
     * @return an OpenAI completion request object
     */
    static ObjectNode toOpenAiCompletionRequest(JsonNode request) {
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        openAi.put("prompt", request.path("prompt").asText(""));
        JsonNode options = request.path("options");
        copyNumber(options, "temperature", openAi, "temperature");
        copyNumber(options, "top_p", openAi, "top_p");
        copyNumber(options, "top_k", openAi, "top_k");
        copyNumber(options, "seed", openAi, "seed");
        copyNumber(options, "num_predict", openAi, "max_tokens");
        if (options.path("stop").isArray()) {
            openAi.set("stop", options.path("stop").deepCopy());
        }
        return openAi;
    }

    /**
     * Translate an Ollama {@code /api/generate} request with a {@code suffix} into the native infill
     * request shape ({@code input_prefix} / {@code input_suffix}).
     *
     * @param request the parsed Ollama generate request
     * @return a native {@code /infill} request object
     */
    static ObjectNode toInfillRequest(JsonNode request) {
        ObjectNode infill = OBJECT_MAPPER.createObjectNode();
        infill.put("input_prefix", request.path("prompt").asText(""));
        infill.put("input_suffix", request.path("suffix").asText(""));
        JsonNode options = request.path("options");
        copyNumber(options, "temperature", infill, "temperature");
        copyNumber(options, "top_p", infill, "top_p");
        copyNumber(options, "top_k", infill, "top_k");
        copyNumber(options, "seed", infill, "seed");
        copyNumber(options, "num_predict", infill, "n_predict");
        if (options.path("stop").isArray()) {
            infill.set("stop", options.path("stop").deepCopy());
        }
        return infill;
    }

    /**
     * Extract the generated text from an OpenAI completion body ({@code choices[0].text}).
     *
     * @param openAiCompletionJson the OpenAI {@code /v1/completions} response
     * @return the completion text, or empty on an unexpected body
     */
    static String extractCompletionText(String openAiCompletionJson) {
        try {
            return OBJECT_MAPPER
                    .readTree(openAiCompletionJson)
                    .path("choices")
                    .path(0)
                    .path("text")
                    .asText("");
        } catch (IOException e) {
            return "";
        }
    }

    /**
     * Extract the generated text from a native infill body ({@code content}).
     *
     * @param infillJson the native {@code /infill} response
     * @return the infill content, or empty on an unexpected body
     */
    static String extractInfillContent(String infillJson) {
        try {
            return OBJECT_MAPPER.readTree(infillJson).path("content").asText("");
        } catch (IOException e) {
            return "";
        }
    }

    /**
     * Build a non-streaming Ollama {@code /api/generate} response wrapping {@code text}.
     *
     * @param text the generated text
     * @param model the model id to echo
     * @return the Ollama generate response serialized as JSON
     */
    static String toOllamaGenerateResponse(String text, String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        root.put("response", text);
        root.put("done", true);
        root.put("done_reason", "stop");
        return root.toString();
    }

    /**
     * Build the streamed Ollama {@code /api/generate} NDJSON: a single response line carrying {@code text}
     * followed by a terminating {@code "done":true} line. (Generation completes before emission, so this
     * is one content chunk rather than token-by-token streaming.)
     *
     * @param text the generated text
     * @param model the model id to echo
     * @return the two NDJSON lines, concatenated (each with a trailing newline)
     */
    static String toOllamaGenerateStream(String text, String model) {
        ObjectNode line = OBJECT_MAPPER.createObjectNode();
        line.put("model", model);
        line.put("created_at", nowIso());
        line.put("response", text);
        line.put("done", false);
        ObjectNode done = OBJECT_MAPPER.createObjectNode();
        done.put("model", model);
        done.put("created_at", nowIso());
        done.put("response", "");
        done.put("done", true);
        done.put("done_reason", "stop");
        return line + "\n" + done + "\n";
    }
}
