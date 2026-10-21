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

    /** The fields of an {@code /api/generate} request that {@code /api/chat} carries the same way. */
    private static final String[] GENERATE_CHAT_FIELDS = {"model", "options", "format", "think"};

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** Ollama's answer to {@code "raw": true} with a {@code system}, {@code template} or {@code context}. */
    static final String RAW_CONFLICT_MESSAGE = "raw mode does not support template, system, or context";

    /**
     * Advertised Ollama API version. Clients gate on it: VS Code Copilot's Ollama provider refuses a server
     * below 0.6.4. 0.9.0 is the first Ollama with {@code think}, the newest request feature these routes
     * translate.
     */
    static final String OLLAMA_VERSION = "0.9.0";

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
        // Ollama's "<key> <value>" lines; Continue's Ollama provider reads the context length only from here.
        root.put("parameters", "num_ctx " + contextLength);
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
     * @throws IllegalArgumentException when {@code messages} is present but not an array, or one of its
     *     elements is not an object (upstream's {@code common/chat.cpp} refuses both the same way), or a
     *     number option holds anything but a number (see {@link OpenAiRequestMapper#optionalNumber})
     */
    static ObjectNode toOpenAiChatRequest(JsonNode ollamaRequest) {
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        if (ollamaRequest.path("model").isTextual()) {
            openAi.put("model", ollamaRequest.path("model").asText());
        }

        // Messages: copy through, converting any assistant tool_calls.arguments object to the OpenAI
        // JSON-encoded string form.
        ArrayNode messages = openAi.putArray("messages");
        JsonNode ollamaMessages = ollamaRequest.path("messages");
        if (!ollamaMessages.isArray() && !ollamaMessages.isMissingNode() && !ollamaMessages.isNull()) {
            throw new IllegalArgumentException("Expected 'messages' to be an array, got " + ollamaMessages);
        }
        for (int i = 0; i < ollamaMessages.size(); i++) {
            JsonNode message = ollamaMessages.get(i);
            if (!message.isObject()) {
                throw new IllegalArgumentException("Expected 'messages[" + i + "]' to be an object, got " + message);
            }
            messages.add(toOpenAiMessage(message));
        }

        JsonNode tools = ollamaRequest.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            openAi.set("tools", tools.deepCopy());
        }

        // Ollama's "think": false switches a model's thinking off, a string picks a level (gpt-oss); the
        // chat mapper applies reasoning_effort as upstream does. "think": true switches it on, also over
        // a server whose thinking is off by default (--reasoning off): upstream lets a request's
        // enable_thinking override that, and it must stay a JSON boolean (a string is refused there).
        JsonNode think = ollamaRequest.path("think");
        if (think.isBoolean() && !think.asBoolean()) {
            openAi.put("reasoning_effort", "none");
        } else if (think.isBoolean()) {
            openAi.putObject("chat_template_kwargs").put("enable_thinking", true);
        } else if (think.isTextual() && !think.asText().isEmpty()) {
            openAi.put("reasoning_effort", think.asText());
        }

        // Ollama nests sampling under "options"; map the common knobs onto OpenAI top-level fields.
        JsonNode options = ollamaRequest.path("options");
        copyNumber(options, "temperature", openAi, "temperature");
        copyNumber(options, "top_p", openAi, "top_p");
        copyNumber(options, "top_k", openAi, "top_k");
        copyNumber(options, "seed", openAi, "seed");
        copyNumPredict(options, openAi, "max_tokens");
        copyLlamaCppOptions(options, openAi);
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
        // Ollama returns a model's thinking as message.thinking and clients send it back; llama.cpp's
        // chat parser reads it as reasoning_content.
        JsonNode thinking = message.remove("thinking");
        if (thinking != null && thinking.isTextual() && !thinking.asText().isEmpty()) {
            message.put("reasoning_content", thinking.asText());
        }
        // Ollama names the tool of a result "tool_name"; llama.cpp's chat parser reads "name".
        JsonNode toolName = message.remove("tool_name");
        if (toolName != null && toolName.isTextual() && !message.has("name")) {
            message.put("name", toolName.asText());
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

    /**
     * Copy the sampling options Ollama names as llama.cpp does, which every route hands on to llama.cpp's
     * request schema (the chat route through the shared mapper). Ollama's {@code repeat_last_n} of
     * {@code -1} (the context size) is left to the server default: upstream accepts {@code 0} and up only.
     */
    private static void copyLlamaCppOptions(JsonNode options, ObjectNode to) {
        copyNumber(options, "min_p", to, "min_p");
        copyNumber(options, "typical_p", to, "typical_p");
        copyNumber(options, "repeat_penalty", to, "repeat_penalty");
        copyNumber(options, "presence_penalty", to, "presence_penalty");
        copyNumber(options, "frequency_penalty", to, "frequency_penalty");
        copyNumber(options, "num_keep", to, "n_keep");
        JsonNode repeatLastN = OpenAiRequestMapper.optionalNumber(options, "repeat_last_n");
        if (repeatLastN != null && repeatLastN.asInt() >= 0) {
            to.set("repeat_last_n", repeatLastN);
        }
    }

    /**
     * Copy Ollama's {@code num_predict}; its {@code -2} ("fill the context") becomes upstream's {@code -1},
     * whose hard limit starts there.
     */
    private static void copyNumPredict(JsonNode options, ObjectNode to, String toKey) {
        JsonNode numPredict = OpenAiRequestMapper.optionalNumber(options, "num_predict");
        if (numPredict != null && numPredict.asInt() < -1) {
            to.put(toKey, -1);
        } else {
            copyNumber(options, "num_predict", to, toKey);
        }
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = OpenAiRequestMapper.optionalNumber(from, fromKey);
        if (value != null) {
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
            String reasoning = openAiMessage.path("reasoning_content").asText("");
            if (!reasoning.isEmpty()) {
                message.put("thinking", reasoning);
            }
            ArrayNode ollamaToolCalls = toOllamaToolCalls(openAiMessage.path("tool_calls"));
            if (ollamaToolCalls.size() > 0) {
                message.set("tool_calls", ollamaToolCalls);
            }
            doneReason = toOllamaDoneReason(choice.path("finish_reason").asText(""));
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
     * Translate one streamed OpenAI chunk into an Ollama NDJSON content line (reasoning deltas as
     * {@code message.thinking}, as Ollama streams them), or return {@code null} when the chunk carries
     * neither assistant text nor thinking (role-only, finish-only or usage-only chunks).
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @param model the model id to echo
     * @return the Ollama NDJSON line (with trailing newline), or {@code null} if nothing to emit
     */
    static @Nullable String toOllamaContentLine(String openAiChunkJson, String model) {
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            JsonNode delta = chunk.path("choices").path(0).path("delta");
            String content = delta.path("content").asText("");
            String thinking = delta.path("reasoning_content").asText("");
            if (content.isEmpty() && thinking.isEmpty()) {
                return null;
            }
            ObjectNode root = OBJECT_MAPPER.createObjectNode();
            root.put("model", model);
            root.put("created_at", nowIso());
            ObjectNode message = root.putObject("message");
            message.put("role", "assistant");
            message.put("content", content);
            if (!thinking.isEmpty()) {
                message.put("thinking", thinking);
            }
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
        return toOllamaDoneLine(model, accumulator, null);
    }

    /**
     * Build the terminating Ollama NDJSON line ({@code "done":true}) with the token counts of the
     * stream's usage chunk, as the blocking answer reports them.
     *
     * @param model the model id to echo
     * @param accumulator the tool-call accumulator fed with the stream's chunks
     * @param usage the OpenAI {@code usage} object of the stream's trailing chunk, or {@code null}
     * @return the final Ollama NDJSON line (with trailing newline)
     */
    static String toOllamaDoneLine(String model, ToolCallDeltaAccumulator accumulator, @Nullable JsonNode usage) {
        return toOllamaDoneLine(model, accumulator, usage, null);
    }

    /**
     * Build the terminating Ollama NDJSON line ({@code "done":true}) with the token counts and the
     * {@code done_reason} of the stream: {@code "length"} when the last chunk's {@code finish_reason}
     * says the answer was cut off, else {@code "stop"} (see {@link #toOllamaDoneReason}).
     *
     * @param model the model id to echo
     * @param accumulator the tool-call accumulator fed with the stream's chunks
     * @param usage the OpenAI {@code usage} object of the stream's trailing chunk, or {@code null}
     * @param finishReason the OpenAI {@code finish_reason} of the stream, or {@code null} if none came
     * @return the final Ollama NDJSON line (with trailing newline)
     */
    static String toOllamaDoneLine(
            String model,
            ToolCallDeltaAccumulator accumulator,
            @Nullable JsonNode usage,
            @Nullable String finishReason) {
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
        if (usage != null && usage.isObject()) {
            root.put("prompt_eval_count", usage.path("prompt_tokens").asLong(0));
            root.put("eval_count", usage.path("completion_tokens").asLong(0));
        }
        root.put("done", true);
        root.put("done_reason", toOllamaDoneReason(finishReason));
        return root.toString() + "\n";
    }

    /**
     * Ollama's {@code done_reason} for an OpenAI {@code finish_reason}: {@code "length"} for an answer
     * cut off by the token limit, {@code "stop"} for everything else -- Ollama reports a turn that ends
     * in tool calls as {@code "stop"}, it has no {@code "tool_calls"}.
     *
     * @param openAiFinishReason the OpenAI finish reason, or {@code null} if unknown
     * @return {@code "length"} or {@code "stop"}
     */
    static String toOllamaDoneReason(@Nullable String openAiFinishReason) {
        return "length".equals(openAiFinishReason) ? "length" : "stop";
    }

    /**
     * Read the {@code finish_reason} of a streamed OpenAI chunk (the chunk that ends the answer carries
     * it), or {@code null} when the chunk has none.
     *
     * @param openAiChunkJson one streamed OpenAI chunk
     * @return the finish reason, or {@code null}
     */
    static @Nullable String finishReasonOf(String openAiChunkJson) {
        try {
            JsonNode reason = OBJECT_MAPPER
                    .readTree(openAiChunkJson)
                    .path("choices")
                    .path(0)
                    .path("finish_reason");
            return reason.isTextual() ? reason.asText() : null;
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Read the {@code usage} object of a streamed OpenAI chunk (the trailing chunk the stream emits for
     * {@code stream_options.include_usage}).
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @return the usage object, or {@code null} when the chunk carries none or does not parse
     */
    static @Nullable JsonNode usageOf(String openAiChunkJson) {
        try {
            JsonNode usage = OBJECT_MAPPER.readTree(openAiChunkJson).path("usage");
            return usage.isObject() ? usage : null;
        } catch (IOException e) {
            return null;
        }
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
     * Refuses an {@code /api/generate} request whose {@code prompt} is present but not a string. Ollama
     * binds the body into a string field and answers such a request with a 400; here Jackson's
     * {@code asText("")} would read an array or an object as an empty prompt, which the load answer and
     * the three translators then took for "no prompt". A {@code null} prompt counts as absent.
     *
     * @param request the parsed Ollama generate request
     * @throws IllegalArgumentException if {@code prompt} is present, not {@code null} and not a string
     */
    static void requireTextualPrompt(JsonNode request) {
        JsonNode prompt = request.path("prompt");
        if (!prompt.isMissingNode() && !prompt.isNull() && !prompt.isTextual()) {
            throw new IllegalArgumentException("'prompt' must be a string");
        }
    }

    /**
     * Whether the {@code /api/generate} request only asks for the model to be loaded: Ollama answers a
     * request without a prompt with {@code done} at once ({@code done_reason "load"}), which is how its
     * clients preload a model. The model of this server is always loaded, so there is nothing to do.
     *
     * <p>Two deliberate differences from Ollama, whose check reads the prompt alone. A request with an
     * empty prompt and a {@code suffix} is not load-only here but fill-in-the-middle at the start of a
     * file, a request editors send; Ollama answers it with the load answer. That includes an empty
     * {@code suffix} ({@link #hasSuffix} only asks for a string), which Ollama would not treat as
     * fill-in-the-middle either. And {@code keep_alive} is not read: Ollama unloads the model on a
     * {@code keep_alive} of {@code 0} without a prompt and answers {@code done_reason "unload"}, while
     * this server keeps its one model loaded and gives the load answer.
     *
     * <p>Only an absent, {@code null} or empty string {@code prompt} counts as no prompt; a prompt that
     * is an array, an object or a number is a malformed request, which {@link #requireTextualPrompt}
     * refuses before this check runs.
     *
     * @param request the parsed Ollama generate request
     * @return {@code true} when the request carries no prompt and no suffix
     */
    static boolean isLoadOnly(JsonNode request) {
        JsonNode prompt = request.path("prompt");
        boolean noPrompt = prompt.isMissingNode()
                || prompt.isNull()
                || (prompt.isTextual() && prompt.asText().isEmpty());
        return noPrompt && !hasSuffix(request);
    }

    /**
     * The answer to a load-only {@code /api/generate} request, in the form of Ollama's load answer: one
     * object, empty {@code response}, {@code done} and {@code done_reason "load"}, whether streaming was
     * asked for or not. It is also the answer to a {@code keep_alive} of {@code 0}, which Ollama answers
     * with {@code done_reason "unload"}: nothing is unloaded here, so the server does not claim it (see
     * {@link #isLoadOnly}).
     *
     * @param model the model id to echo
     * @return the answer
     */
    static String toOllamaLoadResponse(String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        root.put("response", "");
        root.put("done", true);
        root.put("done_reason", "load");
        return root.toString();
    }

    /**
     * Whether the {@code /api/chat} request only asks for the model to be loaded: Ollama answers a chat
     * without messages ({@code messages} absent, {@code null} or empty) with {@code done} at once, which
     * is how its clients preload a model through the chat route. Ollama unloads instead when
     * {@code keep_alive} is {@code 0}; the model of this server stays loaded, so that is a load answer
     * here too, as on {@code /api/generate}.
     *
     * @param request the parsed Ollama chat request
     * @return {@code true} when the request carries no messages
     */
    static boolean isChatLoadOnly(JsonNode request) {
        JsonNode messages = request.path("messages");
        return messages.isMissingNode() || messages.isNull() || (messages.isArray() && messages.size() == 0);
    }

    /**
     * The answer to a load-only {@code /api/chat} request, as Ollama gives it: one object with an empty
     * assistant message, {@code done} and {@code done_reason "load"}, whether streaming was asked for or
     * not.
     *
     * @param model the model id to echo
     * @return the answer
     */
    static String toOllamaChatLoadResponse(String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        root.putObject("message").put("role", "assistant").put("content", "");
        root.put("done", true);
        root.put("done_reason", "load");
        return root.toString();
    }

    /**
     * Whether the {@code /api/generate} request asks for the bare prompt ({@code "raw": true}). Without
     * it Ollama applies the model's chat template, with {@code system} as the system message; with it a
     * {@code system}, {@code template} or {@code context} is refused ({@link #hasRawConflict}).
     *
     * @param request the parsed Ollama generate request
     * @return {@code true} if {@code raw} is {@code true}
     */
    static boolean isRaw(JsonNode request) {
        return request.path("raw").asBoolean(false);
    }

    /**
     * Whether the {@code /api/generate} request combines {@code "raw": true} with a non-empty
     * {@code system} or {@code template} or a non-empty {@code context}, which Ollama refuses with
     * {@value #RAW_CONFLICT_MESSAGE} before anything else (its {@code GenerateHandler}): a raw prompt
     * goes to the model unformatted, so there is nothing to apply them to.
     *
     * @param request the parsed Ollama generate request
     * @return {@code true} if the request must be refused
     */
    static boolean hasRawConflict(JsonNode request) {
        if (!isRaw(request)) {
            return false;
        }
        JsonNode context = request.path("context");
        return !request.path("system").asText("").isEmpty()
                || !request.path("template").asText("").isEmpty()
                || (context.isArray() && context.size() > 0);
    }

    /**
     * Translate an Ollama {@code /api/generate} request without {@code raw} into the internal OpenAI chat
     * request, as Ollama applies the chat template: {@code system} (when given) as the system message,
     * {@code prompt} with its {@code images} as the user message, and {@code options}, {@code format}
     * and {@code think} as {@code /api/chat} carries them.
     *
     * @param request the parsed Ollama generate request
     * @return an OpenAI {@code /v1/chat/completions} request object
     */
    static ObjectNode toOpenAiChatRequestFromGenerate(JsonNode request) {
        ObjectNode chat = OBJECT_MAPPER.createObjectNode();
        for (String field : GENERATE_CHAT_FIELDS) {
            if (request.has(field)) {
                chat.set(field, request.get(field).deepCopy());
            }
        }
        ArrayNode messages = chat.putArray("messages");
        String system = request.path("system").asText("");
        if (!system.isEmpty()) {
            messages.addObject().put("role", "system").put("content", system);
        }
        ObjectNode user = messages.addObject()
                .put("role", "user")
                .put("content", request.path("prompt").asText(""));
        if (request.path("images").isArray()) {
            user.set("images", request.path("images").deepCopy());
        }
        return toOpenAiChatRequest(chat);
    }

    /**
     * Translate an Ollama {@code /api/generate} request into the internal OpenAI {@code /v1/completions}
     * request shape ({@code prompt} + sampling). Used for {@code "raw": true} without {@code suffix}.
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
        copyNumPredict(options, openAi, "max_tokens");
        copyLlamaCppOptions(options, openAi);
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
        copyNumPredict(options, infill, "n_predict");
        copyLlamaCppOptions(options, infill);
        if (options.path("stop").isArray()) {
            infill.set("stop", options.path("stop").deepCopy());
        }
        return infill;
    }

    /**
     * What an {@code /api/generate} answer reports: the text, why it stopped and the token counts.
     * Built from the backend's answer, which is an OpenAI completion or a native infill answer.
     */
    static final class Generated {
        final String text;
        /** The model's reasoning, or {@code ""} when it gave none. */
        final String thinking;

        final String doneReason;
        /** {@code prompt_eval_count}, or {@code -1} when the answer has none. */
        final long promptEvalCount;
        /** {@code eval_count}, or {@code -1} when the answer has none. */
        final long evalCount;

        Generated(String text, String doneReason, long promptEvalCount, long evalCount) {
            this(text, "", doneReason, promptEvalCount, evalCount);
        }

        Generated(String text, String thinking, String doneReason, long promptEvalCount, long evalCount) {
            this.text = text;
            this.thinking = thinking;
            this.doneReason = doneReason;
            this.promptEvalCount = promptEvalCount;
            this.evalCount = evalCount;
        }

        /**
         * From an OpenAI {@code /v1/completions} answer: {@code choices[0].text}, its
         * {@code finish_reason} and {@code usage}.
         *
         * @param openAiCompletionJson the OpenAI completion answer
         * @return what the generate answer reports; an empty {@code "stop"} on an unexpected body
         */
        static Generated fromCompletion(String openAiCompletionJson) {
            try {
                JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
                JsonNode choice = completion.path("choices").path(0);
                JsonNode usage = completion.path("usage");
                return new Generated(
                        choice.path("text").asText(""),
                        toOllamaDoneReason(choice.path("finish_reason").asText("")),
                        usage.path("prompt_tokens").asLong(-1L),
                        usage.path("completion_tokens").asLong(-1L));
            } catch (IOException e) {
                return new Generated("", "stop", -1L, -1L);
            }
        }

        /**
         * From an OpenAI chat answer (the {@code /api/generate} request without {@code raw}): the
         * message's {@code content} and {@code reasoning_content}, its {@code finish_reason} and
         * {@code usage}.
         *
         * @param openAiChatJson the OpenAI chat completion answer
         * @return what the generate answer reports; an empty {@code "stop"} on an unexpected body
         */
        static Generated fromChat(String openAiChatJson) {
            try {
                JsonNode completion = OBJECT_MAPPER.readTree(openAiChatJson);
                JsonNode choice = completion.path("choices").path(0);
                JsonNode usage = completion.path("usage");
                return new Generated(
                        choice.path("message").path("content").asText(""),
                        choice.path("message").path("reasoning_content").asText(""),
                        toOllamaDoneReason(choice.path("finish_reason").asText("")),
                        usage.path("prompt_tokens").asLong(-1L),
                        usage.path("completion_tokens").asLong(-1L));
            } catch (IOException e) {
                return new Generated("", "stop", -1L, -1L);
            }
        }

        /** Put the text and, when there is one, the reasoning into an Ollama generate line. */
        private void putText(ObjectNode line) {
            line.put("response", text);
            if (!thinking.isEmpty()) {
                line.put("thinking", thinking);
            }
        }

        /**
         * From a native {@code /infill} answer: {@code content}, its {@code stop_type} ({@code "limit"}
         * is a cut-off answer) and {@code tokens_evaluated} / {@code tokens_predicted}.
         *
         * @param infillJson the native infill answer
         * @return what the generate answer reports; an empty {@code "stop"} on an unexpected body
         */
        static Generated fromInfill(String infillJson) {
            try {
                JsonNode infill = OBJECT_MAPPER.readTree(infillJson);
                return new Generated(
                        infill.path("content").asText(""),
                        "limit".equals(infill.path("stop_type").asText("")) ? "length" : "stop",
                        infill.path("tokens_evaluated").asLong(-1L),
                        infill.path("tokens_predicted").asLong(-1L));
            } catch (IOException e) {
                return new Generated("", "stop", -1L, -1L);
            }
        }

        private void putDone(ObjectNode done) {
            if (promptEvalCount >= 0) {
                done.put("prompt_eval_count", promptEvalCount);
            }
            if (evalCount >= 0) {
                done.put("eval_count", evalCount);
            }
            done.put("done", true);
            done.put("done_reason", doneReason);
        }
    }

    /**
     * Build a non-streaming Ollama {@code /api/generate} response.
     *
     * @param generated the text, stop reason and token counts of the answer
     * @param model the model id to echo
     * @return the Ollama generate response serialized as JSON
     */
    static String toOllamaGenerateResponse(Generated generated, String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("model", model);
        root.put("created_at", nowIso());
        generated.putText(root);
        generated.putDone(root);
        return root.toString();
    }

    /**
     * Build the streamed Ollama {@code /api/generate} NDJSON: a single response line carrying the text
     * followed by a terminating {@code "done":true} line with the stop reason and token counts.
     * (Generation completes before emission, so this is one content chunk rather than token-by-token
     * streaming.)
     *
     * @param generated the text, stop reason and token counts of the answer
     * @param model the model id to echo
     * @return the two NDJSON lines, concatenated (each with a trailing newline)
     */
    static String toOllamaGenerateStream(Generated generated, String model) {
        ObjectNode line = OBJECT_MAPPER.createObjectNode();
        line.put("model", model);
        line.put("created_at", nowIso());
        generated.putText(line);
        line.put("done", false);
        ObjectNode done = OBJECT_MAPPER.createObjectNode();
        done.put("model", model);
        done.put("created_at", nowIso());
        done.put("response", "");
        generated.putDone(done);
        return line + "\n" + done + "\n";
    }
}
