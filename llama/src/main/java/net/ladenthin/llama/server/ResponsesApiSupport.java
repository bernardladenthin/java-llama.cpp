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
            if (request.path("tool_choice").isTextual()) {
                openAi.put("tool_choice", request.path("tool_choice").asText());
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
        // Upstream's converter maps reasoning.effort to the chat field reasoning_effort ("none" turns
        // the model's thinking off); the other reasoning settings have no chat equivalent there either.
        if (request.path("reasoning").path("effort").isTextual()) {
            openAi.put(
                    "reasoning_effort", request.path("reasoning").path("effort").asText());
        }
        return openAi;
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
                    function.put("arguments", item.path("arguments").asText(""));
                    break;
                case "function_call_output":
                    ObjectNode toolMessage = messages.addObject();
                    toolMessage.put("role", "tool");
                    toolMessage.put("tool_call_id", item.path("call_id").asText(""));
                    JsonNode output = item.path("output");
                    if (output.isArray()) {
                        toolMessage.set("content", toolOutputParts(output));
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
     * vision model answered as if none had been sent.
     */
    private static @Nullable ArrayNode inputContentWithImages(JsonNode content) {
        if (!content.isArray()) {
            return null;
        }
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        boolean hasImage = false;
        for (JsonNode part : content) {
            if ("input_image".equals(part.path("type").asText(""))) {
                JsonNode imageUrl = part.path("image_url");
                String url = imageUrl.isTextual()
                        ? imageUrl.asText()
                        : imageUrl.path("url").asText("");
                if (!url.isEmpty()) {
                    parts.addObject()
                            .put("type", "image_url")
                            .putObject("image_url")
                            .put("url", url);
                    hasImage = true;
                }
            } else if (part.path("text").isTextual()) {
                parts.addObject()
                        .put("type", "text")
                        .put("text", part.path("text").asText());
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
     * message an empty content. Other part types are skipped (upstream rejects them).
     */
    private static ArrayNode toolOutputParts(JsonNode output) {
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        for (JsonNode part : output) {
            String type = part.path("type").asText("");
            if ("input_text".equals(type)) {
                parts.addObject()
                        .put("type", "text")
                        .put("text", part.path("text").asText(""));
            } else if ("input_image".equals(type) && part.path("image_url").isTextual()) {
                parts.addObject()
                        .put("type", "image_url")
                        .putObject("image_url")
                        .put("url", part.path("image_url").asText());
            }
        }
        return parts;
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

    private static String inputContentText(JsonNode content) {
        if (content.isTextual()) {
            return content.asText();
        }
        if (content.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode part : content) {
                if (part.path("text").isTextual()) {
                    // Joined with a newline, as llama.cpp's concat_content_parts joins text parts.
                    if (sb.length() > 0) {
                        sb.append('\n');
                    }
                    sb.append(part.path("text").asText());
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
