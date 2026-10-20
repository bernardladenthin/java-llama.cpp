// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.jspecify.annotations.Nullable;

/**
 * Pure translators between the OpenAI Responses API ({@code POST /v1/responses}) and the internal OpenAI
 * chat shape, plus builders for the Responses streaming SSE events. Lets clients/editors that use the
 * newer Responses protocol (e.g. Copilot's {@code responses} apiType) drive the local model.
 *
 * <p>Request mapping: {@code instructions} becomes a system message; {@code input} (a string, or an array
 * of {@code message} / {@code function_call} / {@code function_call_output} items) is flattened to OpenAI
 * messages; Responses function tools ({@code {type:"function",name,description,parameters}}) become OpenAI
 * function tools. Responses replies wrap the assistant turn in a {@code response} object whose
 * {@code output} array holds a {@code message} item (with {@code output_text} content) and one
 * {@code function_call} item per tool call.
 *
 * <p>Stateless and free of JNI / model dependencies; unit-testable with JSON literals. Streaming state is
 * held by {@link ResponsesStreamTranslator}.
 */
final class ResponsesApiSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** The keys of a Responses {@code json_schema} format that chat nests under {@code json_schema}. */
    private static final List<String> JSON_SCHEMA_FORMAT_FIELDS =
            Collections.unmodifiableList(Arrays.asList("name", "description", "schema", "strict"));

    /**
     * Request fields handed on to the chat request unchanged. Upstream's {@code
     * convert_responses_to_chatcmpl} copies the whole body; these are the ones the shared chat mapper
     * reads (OpenAI sampling and llama.cpp's extensions). The structural fields ({@code input}, {@code
     * tools}, {@code tool_choice}) are translated above, not copied.
     */
    private static final List<String> CHAT_PASSTHROUGH_FIELDS = Collections.unmodifiableList(Arrays.asList(
            "top_k",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "chat_template_kwargs",
            "response_format",
            "cache_prompt",
            "n_cache_reuse",
            "id_slot",
            "min_p",
            "typical_p",
            "repeat_penalty",
            "repeat_last_n",
            "top_n_sigma",
            "xtc_probability",
            "xtc_threshold",
            "dynatemp_range",
            "dynatemp_exponent",
            "min_keep",
            "n_keep",
            "ignore_eos",
            "grammar",
            "json_schema",
            "n_predict",
            "reasoning_budget_tokens",
            "thinking_budget_tokens"));

    private ResponsesApiSupport() {}

    /**
     * Whether the Responses request asks for a streamed response ({@code "stream"} defaults to false).
     *
     * @param request the parsed Responses request
     * @return {@code true} if {@code "stream"} is explicitly true
     */
    static boolean isStreaming(JsonNode request) {
        return request.path("stream").asBoolean(false);
    }

    /**
     * Translate an OpenAI Responses request into the internal OpenAI chat request shape.
     *
     * @param request the parsed Responses request
     * @return an OpenAI {@code /v1/chat/completions} request object
     * @throws IllegalArgumentException if the request continues an earlier response by {@code previous_response_id}
     */
    static ObjectNode toOpenAiChatRequest(JsonNode request) {
        // As upstream's convert_responses_to_chatcmpl: there is no response store, so a turn that refers
        // to an earlier one by id would be answered without its history.
        String previousResponseId = request.path("previous_response_id").asText("");
        if (!previousResponseId.isEmpty()) {
            throw new IllegalArgumentException("llama.cpp does not support 'previous_response_id' (got "
                    + previousResponseId + "); send the earlier turns in 'input' instead");
        }
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        if (request.path("model").isTextual()) {
            openAi.put("model", request.path("model").asText());
        }

        ArrayNode messages = openAi.putArray("messages");
        if (request.path("instructions").isTextual()) {
            ObjectNode system = messages.addObject();
            system.put("role", "system");
            system.put("content", request.path("instructions").asText());
        }
        appendInput(messages, request.path("input"));

        JsonNode tools = request.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            ArrayNode openAiTools = openAi.putArray("tools");
            for (JsonNode tool : tools) {
                if (!"function".equals(tool.path("type").asText(""))) {
                    continue;
                }
                ObjectNode openAiTool = openAiTools.addObject();
                openAiTool.put("type", "function");
                ObjectNode function = openAiTool.putObject("function");
                function.put("name", tool.path("name").asText(""));
                if (tool.path("description").isTextual()) {
                    function.put("description", tool.path("description").asText());
                }
                if (tool.path("parameters").isObject()) {
                    function.set("parameters", tool.path("parameters").deepCopy());
                }
            }
            // The Responses API uses the same tool_choice + parallel_tool_calls fields as chat; forward
            // them so the shared chat core honors them. The mapper consumes the string form of
            // tool_choice ("auto"/"none"/"required"), which is what we forward here.
            JsonNode toolChoice = request.path("tool_choice");
            if (toolChoice.isTextual()) {
                openAi.put("tool_choice", toolChoice.asText());
            } else if ("function".equals(toolChoice.path("type").asText(""))) {
                // {"type":"function","name":...}: the chat side knows only strings, "required" is closest.
                openAi.put("tool_choice", "required");
            }
            if (request.path("parallel_tool_calls").isBoolean()) {
                openAi.put(
                        "parallel_tool_calls",
                        request.path("parallel_tool_calls").asBoolean());
            }
        }

        copyNumber(request, "temperature", openAi, "temperature");
        copyNumber(request, "top_p", openAi, "top_p");
        copyNumber(request, "max_output_tokens", openAi, "max_tokens");
        for (String field : CHAT_PASSTHROUGH_FIELDS) {
            JsonNode value = request.path(field);
            if (!value.isMissingNode() && !value.isNull()) {
                openAi.set(field, value.deepCopy());
            }
        }
        if (!openAi.has("response_format")) {
            appendResponseFormat(openAi, request.path("text").path("format"));
        }
        // Upstream's converter maps reasoning.effort to the chat field reasoning_effort ("none" turns
        // the model's thinking off); the other reasoning settings have no chat equivalent there either.
        if (request.path("reasoning").path("effort").isTextual()) {
            openAi.put(
                    "reasoning_effort", request.path("reasoning").path("effort").asText());
        }
        return openAi;
    }

    /**
     * Translate Responses' structured-output setting {@code text.format} into chat's {@code
     * response_format}: {@code json_schema} keeps its name, schema, strict flag and description, now
     * nested under {@code json_schema}; {@code json_object} carries over; {@code text} needs nothing.
     * Upstream's converter copies {@code text} unchanged, where the chat parser does not read it.
     *
     * @param openAi the chat request being built
     * @param format the request's {@code text.format}
     */
    private static void appendResponseFormat(ObjectNode openAi, JsonNode format) {
        String type = format.path("type").asText("");
        if ("json_object".equals(type)) {
            openAi.putObject("response_format").put("type", "json_object");
        } else if ("json_schema".equals(type)) {
            ObjectNode responseFormat = openAi.putObject("response_format");
            responseFormat.put("type", "json_schema");
            ObjectNode jsonSchema = responseFormat.putObject("json_schema");
            for (String key : JSON_SCHEMA_FORMAT_FIELDS) {
                JsonNode value = format.path(key);
                if (!value.isMissingNode() && !value.isNull()) {
                    jsonSchema.set(key, value.deepCopy());
                }
            }
        }
    }

    private static void appendInput(ArrayNode messages, JsonNode input) {
        if (input.isTextual()) {
            ObjectNode message = messages.addObject();
            message.put("role", "user");
            message.put("content", input.asText());
            return;
        }
        if (!input.isArray()) {
            return;
        }
        for (JsonNode item : input) {
            String type = item.path("type").asText("message");
            switch (type) {
                case "function_call":
                    ObjectNode toolCall = assistantToolCalls(messages).addObject();
                    toolCall.put(
                            "id", item.path("call_id").asText(item.path("id").asText("")));
                    toolCall.put("type", "function");
                    ObjectNode function = toolCall.putObject("function");
                    function.put("name", item.path("name").asText(""));
                    // upstream json::parse()s every arguments string of the history, so an empty one
                    // fails the whole request; a call without arguments is "{}" (F166). Arguments are a
                    // JSON string: upstream answers 400 for any other value, and read as text an object
                    // was "" and became "{}", a call the model never made that way.
                    JsonNode argumentsNode = item.path("arguments");
                    if (!argumentsNode.isMissingNode() && !argumentsNode.isNull() && !argumentsNode.isTextual()) {
                        throw new IllegalArgumentException("'arguments' of a function_call must be a JSON string (got "
                                + argumentsNode.getNodeType() + ", call_id "
                                + item.path("call_id").asText("") + ")");
                    }
                    String arguments = argumentsNode.asText("");
                    function.put("arguments", arguments.trim().isEmpty() ? "{}" : arguments);
                    break;
                case "function_call_output":
                    ObjectNode toolMessage = messages.addObject();
                    toolMessage.put("role", "tool");
                    toolMessage.put("tool_call_id", item.path("call_id").asText(""));
                    JsonNode output = item.path("output");
                    if (output.isArray()) {
                        toolMessage.set(
                                "content",
                                toolOutputParts(output, item.path("call_id").asText("")));
                    } else {
                        toolMessage.put("content", output.asText(""));
                    }
                    break;
                case "reasoning":
                    appendReasoningItem(messages, item);
                    break;
                case "message":
                default:
                    rejectInputFiles(item.path("content"));
                    String role = item.path("role").asText("user");
                    JsonNode imageContent = inputContentWithImages(item.path("content"));
                    JsonNode content =
                            imageContent != null ? imageContent : new TextNode(inputContentText(item.path("content")));
                    ObjectNode previous = "assistant".equals(role) ? lastAssistant(messages) : null;
                    if (previous != null) {
                        previous.set("content", joinContent(previous.path("content"), content));
                    } else {
                        ObjectNode message = messages.addObject();
                        message.put("role", role);
                        message.set("content", content);
                    }
                    break;
            }
        }
    }

    /**
     * The OpenAI content-part array for a Responses content array that carries {@code input_image}
     * parts, or {@code null} when it carries none. The OpenAI parser reads images only from
     * {@code image_url} parts, so flattening such a message to its text dropped the image and a
     * vision model answered as if none had been sent. An {@code input_image} without a URL (one given
     * by {@code file_id}) is refused as upstream refuses it; dropping it answered the text alone.
     *
     * @throws IllegalArgumentException for an {@code input_image} part without a URL
     */
    private static @Nullable ArrayNode inputContentWithImages(JsonNode content) {
        if (!content.isArray()) {
            return null;
        }
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        boolean hasImage = false;
        for (JsonNode part : content) {
            if ("input_image".equals(part.path("type").asText(""))) {
                String url = imageUrl(part);
                if (url.isEmpty()) {
                    throw new IllegalArgumentException("'image_url' is required (input_image with file_id "
                            + part.path("file_id").asText("") + " is not supported)");
                }
                parts.addObject()
                        .put("type", "image_url")
                        .putObject("image_url")
                        .put("url", url);
                hasImage = true;
            } else {
                String text = partText(part);
                if (text != null) {
                    parts.addObject().put("type", "text").put("text", text);
                }
            }
        }
        return hasImage ? parts : null;
    }

    /**
     * The {@code tool_calls} array a {@code function_call} item joins: the preceding assistant message's
     * (created if it has none), as upstream's {@code convert_responses_to_chatcmpl} merges parallel calls
     * into one assistant turn, else a new assistant message with {@code null} content.
     */
    private static ArrayNode assistantToolCalls(ArrayNode messages) {
        int size = messages.size();
        JsonNode last = size > 0 ? messages.get(size - 1) : null;
        if (last instanceof ObjectNode && "assistant".equals(last.path("role").asText(""))) {
            JsonNode existing = last.path("tool_calls");
            return existing instanceof ArrayNode ? (ArrayNode) existing : ((ObjectNode) last).putArray("tool_calls");
        }
        ObjectNode assistant = messages.addObject();
        assistant.put("role", "assistant");
        assistant.putNull("content");
        return assistant.putArray("tool_calls");
    }

    /**
     * The content parts of a {@code function_call_output} whose {@code output} is an array: {@code
     * input_text} becomes a {@code text} part, {@code input_image} an {@code image_url} part, as in
     * upstream's {@code convert_responses_to_chatcmpl}. Reading such an output as text gave the tool
     * message an empty content. A part without a type, of another type, or an image without a URL is
     * refused with upstream's messages; skipping it handed the model an incomplete tool result.
     *
     * @param output the {@code output} array
     * @param callId the {@code call_id} of the item, named in a refusal
     * @return the content parts of the tool message
     * @throws IllegalArgumentException for a part upstream refuses
     */
    private static ArrayNode toolOutputParts(JsonNode output, String callId) {
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        for (JsonNode part : output) {
            if (!part.has("type")) {
                throw new IllegalArgumentException("Output of tool call missing 'type' field (call_id " + callId + ")");
            }
            String type = part.path("type").asText("");
            if ("input_text".equals(type)) {
                parts.addObject()
                        .put("type", "text")
                        .put("text", part.path("text").asText(""));
            } else if ("input_image".equals(type)) {
                String url = imageUrl(part);
                if (url.isEmpty()) {
                    throw new IllegalArgumentException(
                            "'image_url' is required (call_id " + callId + ", input_image with file_id "
                                    + part.path("file_id").asText("") + " is not supported)");
                }
                parts.addObject()
                        .put("type", "image_url")
                        .putObject("image_url")
                        .put("url", url);
            } else {
                throw new IllegalArgumentException("Output of tool call should be 'Input text' or 'Input image' (got '"
                        + type + "', call_id " + callId + ")");
            }
        }
        return parts;
    }

    /**
     * The URL of an {@code input_image} part: {@code image_url} as the string the Responses API defines,
     * or as {@code {"url": ...}} (accepted on both paths alike); {@code ""} when there is none.
     */
    private static String imageUrl(JsonNode part) {
        JsonNode imageUrl = part.path("image_url");
        return imageUrl.isTextual() ? imageUrl.asText() : imageUrl.path("url").asText("");
    }

    /** The last message if it is an assistant message, else {@code null} (upstream's {@code merge_prev}). */
    private static @Nullable ObjectNode lastAssistant(ArrayNode messages) {
        int size = messages.size();
        JsonNode last = size > 0 ? messages.get(size - 1) : null;
        return last instanceof ObjectNode
                        && "assistant".equals(last.path("role").asText(""))
                ? (ObjectNode) last
                : null;
    }

    /**
     * The content of an assistant message after an assistant message item joined it, as upstream's
     * {@code convert_responses_to_chatcmpl} appends the item's content to the preceding assistant
     * message: an empty or {@code null} content is replaced, two texts are joined with a newline (as
     * {@code concat_content_parts} joins text parts), otherwise the parts are concatenated.
     */
    private static JsonNode joinContent(JsonNode previous, JsonNode added) {
        if (previous.isNull()
                || previous.isMissingNode()
                || (previous.isTextual() && previous.asText().isEmpty())) {
            return added;
        }
        if (previous.isTextual() && added.isTextual()) {
            return added.asText().isEmpty() ? previous : new TextNode(previous.asText() + "\n" + added.asText());
        }
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        for (JsonNode node : new JsonNode[] {previous, added}) {
            if (node.isArray()) {
                parts.addAll((ArrayNode) node);
            } else if (node.isTextual() && !node.asText().isEmpty()) {
                parts.addObject().put("type", "text").put("text", node.asText());
            }
        }
        return parts;
    }

    /**
     * Rejects an {@code input_file} content part the way upstream's {@code convert_responses_to_chatcmpl}
     * does (400, "not supported"); dropping it let the model answer as if the file had not been sent.
     */
    private static void rejectInputFiles(JsonNode content) {
        if (!content.isArray()) {
            return;
        }
        for (JsonNode part : content) {
            if ("input_file".equals(part.path("type").asText(""))) {
                throw new IllegalArgumentException("'input_file' is not supported by llamacpp at this moment (file "
                        + part.path("file_id").asText(part.path("filename").asText("")) + ")");
            }
        }
    }

    /**
     * A {@code reasoning} input item (an earlier answer's reasoning, sent back by the client) becomes
     * {@code reasoning_content} of the preceding assistant message, or of a new one, as in upstream's
     * {@code convert_responses_to_chatcmpl}. The item has no role, so reading it as a message made the
     * model's own reasoning a user turn. An item without readable text (only {@code summary} or
     * {@code encrypted_content}) is skipped; upstream rejects it.
     */
    private static void appendReasoningItem(ArrayNode messages, JsonNode item) {
        JsonNode text = item.path("content").path(0).path("text");
        if (!text.isTextual()) {
            return;
        }
        String reasoning = text.asText();
        int size = messages.size();
        JsonNode last = size > 0 ? messages.get(size - 1) : null;
        if (last instanceof ObjectNode && "assistant".equals(last.path("role").asText(""))) {
            ((ObjectNode) last).put("reasoning_content", reasoning);
            return;
        }
        ObjectNode assistant = messages.addObject();
        assistant.put("role", "assistant");
        assistant.put("content", "");
        assistant.put("reasoning_content", reasoning);
    }

    /**
     * The text of a content part, or {@code null} for a part without one. A {@code refusal} part of an
     * assistant message carries its text under {@code refusal}; upstream turns it into a part of type
     * {@code refusal}, which its own chat parser then rejects with 400 (it accepts text and media only),
     * so as text it is what the assistant said.
     */
    private static @Nullable String partText(JsonNode part) {
        if (part.path("text").isTextual()) {
            return part.path("text").asText();
        }
        if ("refusal".equals(part.path("type").asText(""))
                && part.path("refusal").isTextual()) {
            return part.path("refusal").asText();
        }
        return null;
    }

    private static String inputContentText(JsonNode content) {
        if (content.isTextual()) {
            return content.asText();
        }
        if (content.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode part : content) {
                String text = partText(part);
                if (text != null) {
                    // Joined with a newline, as llama.cpp's concat_content_parts joins text parts.
                    if (sb.length() > 0) {
                        sb.append('\n');
                    }
                    sb.append(text);
                }
            }
            return sb.toString();
        }
        return "";
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = from.path(fromKey);
        if (value.isNumber()) {
            to.set(toKey, value);
        }
    }

    /**
     * Translate a non-streaming OpenAI {@code chat.completion} into a Responses API response object.
     *
     * @param openAiCompletionJson the OpenAI completion body
     * @param model the model id to echo
     * @param responseId the response id to assign
     * @return the Responses object serialized as JSON
     */
    static String toResponsesResponse(String openAiCompletionJson, String model, String responseId) {
        ObjectNode root = newResponseShell(model, responseId, "completed");
        ArrayNode output = root.putArray("output");
        ObjectNode usage = root.putObject("usage");
        usage.put("input_tokens", 0);
        usage.put("output_tokens", 0);
        usage.put("total_tokens", 0);
        ObjectNode inputTokenDetails = usage.putObject("input_tokens_details");
        inputTokenDetails.put("cached_tokens", 0);
        try {
            JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
            JsonNode message = completion.path("choices").path(0).path("message");
            // As upstream's to_json_oaicompat_resp: the reasoning comes first, as a reasoning item.
            String reasoning = message.path("reasoning_content").asText("");
            if (!reasoning.isEmpty()) {
                ObjectNode reasoningItem = output.addObject();
                reasoningItem.put("type", "reasoning");
                reasoningItem.put("id", "rs_" + responseId);
                reasoningItem.putArray("summary");
                ObjectNode reasoningText = reasoningItem.putArray("content").addObject();
                reasoningText.put("type", "reasoning_text");
                reasoningText.put("text", reasoning);
                reasoningItem.put("encrypted_content", "");
                reasoningItem.put("status", "completed");
            }
            // As upstream (and the streamed form): a message item only when there is text.
            String text = message.path("content").asText("");
            if (!text.isEmpty()) {
                ObjectNode messageItem = output.addObject();
                messageItem.put("type", "message");
                messageItem.put("id", "msg_" + responseId);
                messageItem.put("status", "completed");
                messageItem.put("role", "assistant");
                ArrayNode content = messageItem.putArray("content");
                ObjectNode textPart = content.addObject();
                textPart.put("type", "output_text");
                textPart.put("text", text);
                textPart.putArray("annotations");
            }
            JsonNode toolCalls = message.path("tool_calls");
            if (toolCalls.isArray()) {
                for (JsonNode toolCall : toolCalls) {
                    output.add(functionCallItem(toolCall));
                }
            }
            JsonNode openAiUsage = completion.path("usage");
            if (openAiUsage.isObject()) {
                long in = openAiUsage.path("prompt_tokens").asLong(0);
                long out = openAiUsage.path("completion_tokens").asLong(0);
                usage.put("input_tokens", in);
                usage.put("output_tokens", out);
                usage.put("total_tokens", in + out);
                inputTokenDetails.put(
                        "cached_tokens",
                        openAiUsage
                                .path("prompt_tokens_details")
                                .path("cached_tokens")
                                .asLong(0));
            }
        } catch (IOException e) {
            // Defensive: an unexpected body still yields a valid, empty completed response.
            output.removeAll();
        }
        return root.toString();
    }

    /** Build a Responses {@code function_call} output item from an OpenAI tool call. */
    static ObjectNode functionCallItem(JsonNode openAiToolCall) {
        JsonNode function = openAiToolCall.path("function");
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "function_call");
        item.put("id", "fc_" + openAiToolCall.path("id").asText(""));
        item.put("call_id", openAiToolCall.path("id").asText(""));
        item.put("name", function.path("name").asText(""));
        item.put("arguments", function.path("arguments").asText(""));
        item.put("status", "completed");
        return item;
    }

    /** A bare Responses API object shell (no output/usage), used by both the final reply and events. */
    static ObjectNode newResponseShell(String model, String responseId, String status) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("id", responseId);
        root.put("object", "response");
        root.put("created_at", System.currentTimeMillis() / 1000L);
        root.put("status", status);
        root.put("model", model);
        return root;
    }

    // ----- streaming SSE event builders -----

    /**
     * Frame a Responses SSE event: {@code event: <type>\ndata: <json>\n\n}, where the data object carries
     * the event {@code type} and {@code sequence_number}.
     *
     * @param type the event type (e.g. {@code response.output_text.delta})
     * @param sequenceNumber the monotonic event sequence number
     * @param data the event payload (the {@code type}/{@code sequence_number} are added here)
     * @return the framed SSE event
     */
    static String sseEvent(String type, int sequenceNumber, ObjectNode data) {
        data.put("type", type);
        data.put("sequence_number", sequenceNumber);
        return "event: " + type + "\ndata: " + data + "\n\n";
    }

    /** New empty data object for an event payload. */
    static ObjectNode dataObject() {
        return OBJECT_MAPPER.createObjectNode();
    }
}
