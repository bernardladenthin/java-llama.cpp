// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;

/**
 * Pure translators between the Anthropic Messages API ({@code POST /v1/messages}) and the internal
 * OpenAI chat shape, plus builders for the Anthropic streaming SSE events. Lets clients that speak the
 * Anthropic protocol (Claude Code, Copilot's {@code messages} apiType) drive the local model without a
 * second inference path.
 *
 * <p>Request mapping covers Anthropic's content-block model: a {@code system} string/blocks becomes an
 * OpenAI system message; message {@code content} that is a string or an array of {@code text} /
 * {@code tool_use} / {@code tool_result} blocks is flattened to OpenAI messages (a user message's
 * {@code tool_result} blocks become separate {@code role:"tool"} messages); Anthropic {@code tools}
 * ({@code name}/{@code description}/{@code input_schema}) become OpenAI function tools. Responses map the
 * other way: OpenAI {@code content} + {@code tool_calls} become Anthropic {@code text} +
 * {@code tool_use} content blocks.
 *
 * <p>Stateless and free of JNI / model dependencies; unit-testable with JSON literals. Streaming state
 * is held by {@link AnthropicStreamTranslator}.
 */
final class AnthropicApiSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private AnthropicApiSupport() {}

    /**
     * Whether the Anthropic request asks for a streamed response ({@code "stream"} defaults to false).
     *
     * @param request the parsed Anthropic request
     * @return {@code true} if {@code "stream"} is explicitly true
     */
    static boolean isStreaming(JsonNode request) {
        return request.path("stream").asBoolean(false);
    }

    /**
     * Translate an Anthropic {@code /v1/messages} request into the internal OpenAI chat request shape.
     *
     * @param request the parsed Anthropic request
     * @return an OpenAI {@code /v1/chat/completions} request object
     */
    static ObjectNode toOpenAiChatRequest(JsonNode request) {
        ObjectNode openAi = OBJECT_MAPPER.createObjectNode();
        if (request.path("model").isTextual()) {
            openAi.put("model", request.path("model").asText());
        }

        ArrayNode messages = openAi.putArray("messages");
        String system = systemText(request.path("system"));
        if (!system.isEmpty()) {
            ObjectNode systemMessage = messages.addObject();
            systemMessage.put("role", "system");
            systemMessage.put("content", system);
        }
        for (JsonNode message : request.path("messages")) {
            appendOpenAiMessages(messages, message);
        }

        JsonNode tools = request.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            ArrayNode openAiTools = openAi.putArray("tools");
            for (JsonNode tool : tools) {
                ObjectNode openAiTool = openAiTools.addObject();
                openAiTool.put("type", "function");
                ObjectNode function = openAiTool.putObject("function");
                function.put("name", tool.path("name").asText(""));
                if (tool.path("description").isTextual()) {
                    function.put("description", tool.path("description").asText());
                }
                if (tool.path("input_schema").isObject()) {
                    function.set("parameters", tool.path("input_schema").deepCopy());
                }
            }
            String toolChoice = toOpenAiToolChoice(request.path("tool_choice"));
            if (toolChoice != null) {
                openAi.put("tool_choice", toolChoice);
            }
            // Anthropic expresses "no parallel tool use" via tool_choice.disable_parallel_tool_use;
            // OpenAI's equivalent is parallel_tool_calls=false. Map it so the shared chat core honors
            // a client's request to serialize tool calls (default stays parallel when unset/false).
            if (request.path("tool_choice").path("disable_parallel_tool_use").asBoolean(false)) {
                openAi.put("parallel_tool_calls", false);
            }
        }

        copyNumber(request, "max_tokens", openAi, "max_tokens");
        copyNumber(request, "temperature", openAi, "temperature");
        copyNumber(request, "top_p", openAi, "top_p");
        copyNumber(request, "top_k", openAi, "top_k");
        if (request.path("stop_sequences").isArray()) {
            openAi.set("stop", request.path("stop_sequences").deepCopy());
        }
        return openAi;
    }

    private static String systemText(JsonNode system) {
        if (system.isTextual()) {
            return system.asText();
        }
        if (system.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode block : system) {
                if (block.path("text").isTextual()) {
                    sb.append(block.path("text").asText());
                }
            }
            return sb.toString();
        }
        return "";
    }

    private static void appendOpenAiMessages(ArrayNode out, JsonNode anthropicMessage) {
        String role = anthropicMessage.path("role").asText("user");
        JsonNode content = anthropicMessage.path("content");
        if (content.isTextual()) {
            ObjectNode message = out.addObject();
            message.put("role", role);
            message.put("content", content.asText());
            return;
        }
        if (!content.isArray()) {
            return;
        }

        StringBuilder text = new StringBuilder();
        // OpenAI content parts in block order; only used when the message carries an image, since the
        // OpenAI parser reads images from image_url parts alone.
        ArrayNode parts = OBJECT_MAPPER.createArrayNode();
        boolean hadImage = false;
        ArrayNode toolCalls = OBJECT_MAPPER.createArrayNode();
        boolean hadToolResult = false;
        for (JsonNode block : content) {
            String type = block.path("type").asText("");
            switch (type) {
                case "text":
                    text.append(block.path("text").asText(""));
                    parts.addObject()
                            .put("type", "text")
                            .put("text", block.path("text").asText(""));
                    break;
                case "image":
                    String url = imageUrl(block.path("source"));
                    if (url != null) {
                        parts.addObject()
                                .put("type", "image_url")
                                .putObject("image_url")
                                .put("url", url);
                        hadImage = true;
                    }
                    break;
                case "tool_use":
                    // Assistant tool call: Anthropic input (object) -> OpenAI arguments (JSON string).
                    ObjectNode toolCall = toolCalls.addObject();
                    toolCall.put("id", block.path("id").asText(""));
                    toolCall.put("type", "function");
                    ObjectNode function = toolCall.putObject("function");
                    function.put("name", block.path("name").asText(""));
                    function.put("arguments", block.path("input").toString());
                    break;
                case "tool_result":
                    // A user-message tool_result becomes a separate OpenAI role:"tool" message.
                    ObjectNode toolMessage = out.addObject();
                    toolMessage.put("role", "tool");
                    toolMessage.put("tool_call_id", block.path("tool_use_id").asText(""));
                    toolMessage.put("content", toolResultText(block.path("content")));
                    hadToolResult = true;
                    break;
                default:
                    break;
            }
        }
        if (hadImage) {
            ObjectNode message = out.addObject();
            message.put("role", role);
            message.set("content", parts);
            if (toolCalls.size() > 0) {
                message.set("tool_calls", toolCalls);
            }
        } else if (text.length() > 0 || toolCalls.size() > 0) {
            ObjectNode message = out.addObject();
            message.put("role", role);
            if (toolCalls.size() > 0 && text.length() == 0) {
                message.putNull("content"); // assistant tool-call turn carries null content
            } else {
                message.put("content", text.toString());
            }
            if (toolCalls.size() > 0) {
                message.set("tool_calls", toolCalls);
            }
        } else if (!hadToolResult) {
            // Genuinely empty/plain content (no text, no tool calls, no tool_result) — keep a slot.
            // A content array of only tool_result blocks emits no extra message (they became tool messages).
            ObjectNode message = out.addObject();
            message.put("role", role);
            message.put("content", "");
        }
    }

    /**
     * The OpenAI {@code image_url} URL for an Anthropic image {@code source}: a data URL for a base64
     * source, the URL itself for a url source, {@code null} for anything else.
     */
    private static @org.jspecify.annotations.Nullable String imageUrl(JsonNode source) {
        String sourceType = source.path("type").asText("");
        if ("base64".equals(sourceType) && source.path("data").isTextual()) {
            return "data:" + source.path("media_type").asText("image/png") + ";base64,"
                    + source.path("data").asText();
        }
        if ("url".equals(sourceType) && source.path("url").isTextual()) {
            return source.path("url").asText();
        }
        return null;
    }

    private static String toolResultText(JsonNode content) {
        if (content.isTextual()) {
            return content.asText();
        }
        if (content.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode block : content) {
                if (block.path("text").isTextual()) {
                    sb.append(block.path("text").asText());
                }
            }
            return sb.toString();
        }
        return content.toString();
    }

    private static @org.jspecify.annotations.Nullable String toOpenAiToolChoice(JsonNode toolChoice) {
        String type = toolChoice.path("type").asText("");
        if ("auto".equals(type)) {
            return "auto";
        }
        if ("any".equals(type) || "tool".equals(type)) {
            // OpenAI's textual tool_choice cannot name a specific function; "required" is the closest.
            return "required";
        }
        return null;
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = from.path(fromKey);
        if (value.isNumber()) {
            to.set(toKey, value);
        }
    }

    /**
     * Translate a non-streaming OpenAI {@code chat.completion} into an Anthropic message response.
     *
     * @param openAiCompletionJson the OpenAI completion body
     * @param model the model id to echo
     * @return the Anthropic message serialized as JSON
     */
    static String toAnthropicResponse(String openAiCompletionJson, String model) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("id", "msg_" + Integer.toHexString(openAiCompletionJson.hashCode()));
        root.put("type", "message");
        root.put("role", "assistant");
        root.put("model", model);
        ArrayNode content = root.putArray("content");
        String stopReason = "end_turn";
        ObjectNode usage = root.putObject("usage");
        usage.put("input_tokens", 0);
        usage.put("output_tokens", 0);
        try {
            JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
            JsonNode choice = completion.path("choices").path(0);
            JsonNode message = choice.path("message");
            String text = message.path("content").asText("");
            if (!text.isEmpty()) {
                ObjectNode textBlock = content.addObject();
                textBlock.put("type", "text");
                textBlock.put("text", text);
            }
            JsonNode toolCalls = message.path("tool_calls");
            if (toolCalls.isArray()) {
                for (JsonNode toolCall : toolCalls) {
                    content.add(toolUseBlock(toolCall));
                }
            }
            stopReason = anthropicStopReason(choice.path("finish_reason").asText("stop"));
            JsonNode openAiUsage = completion.path("usage");
            if (openAiUsage.isObject()) {
                long promptTokens = openAiUsage.path("prompt_tokens").asLong(0);
                long cachedTokens = openAiUsage
                        .path("prompt_tokens_details")
                        .path("cached_tokens")
                        .asLong(0);
                usage.put("input_tokens", Math.max(0L, promptTokens - cachedTokens));
                usage.put("cache_read_input_tokens", cachedTokens);
                usage.put("output_tokens", openAiUsage.path("completion_tokens").asLong(0));
            }
        } catch (IOException e) {
            stopReason = "end_turn";
        }
        root.put("stop_reason", stopReason);
        root.putNull("stop_sequence");
        return root.toString();
    }

    /** Build an Anthropic {@code tool_use} content block from an OpenAI tool call. */
    static ObjectNode toolUseBlock(JsonNode openAiToolCall) {
        JsonNode function = openAiToolCall.path("function");
        ObjectNode block = OBJECT_MAPPER.createObjectNode();
        block.put("type", "tool_use");
        block.put("id", openAiToolCall.path("id").asText(""));
        block.put("name", function.path("name").asText(""));
        block.set("input", parseToObject(function.path("arguments")));
        return block;
    }

    private static JsonNode parseToObject(JsonNode arguments) {
        if (arguments.isObject() || arguments.isArray()) {
            return arguments;
        }
        if (arguments.isTextual()) {
            try {
                return OBJECT_MAPPER.readTree(arguments.asText());
            } catch (IOException e) {
                return OBJECT_MAPPER.createObjectNode();
            }
        }
        return OBJECT_MAPPER.createObjectNode();
    }

    /** Map an OpenAI finish_reason to an Anthropic stop_reason. */
    static String anthropicStopReason(String openAiFinishReason) {
        switch (openAiFinishReason) {
            case "length":
                return "max_tokens";
            case "tool_calls":
                return "tool_use";
            case "stop":
            default:
                return "end_turn";
        }
    }

    // ----- streaming SSE event builders -----

    /**
     * Frame an Anthropic SSE event: {@code event: <type>\ndata: <json>\n\n}.
     *
     * @param type the event type
     * @param dataJson the event data object serialized as JSON
     * @return the framed SSE event
     */
    static String sseEvent(String type, String dataJson) {
        return "event: " + type + "\ndata: " + dataJson + "\n\n";
    }

    /** {@code message_start} event for a new assistant message. */
    static String messageStartEvent(String id, String model) {
        ObjectNode message = OBJECT_MAPPER.createObjectNode();
        message.put("id", id);
        message.put("type", "message");
        message.put("role", "assistant");
        message.put("model", model);
        message.putArray("content");
        message.putNull("stop_reason");
        message.putNull("stop_sequence");
        ObjectNode usage = message.putObject("usage");
        usage.put("input_tokens", 0);
        usage.put("output_tokens", 0);
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "message_start");
        data.set("message", message);
        return sseEvent("message_start", data.toString());
    }

    /** {@code content_block_start} event opening a text block at {@code index}. */
    static String textBlockStartEvent(int index) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_start");
        data.put("index", index);
        ObjectNode block = data.putObject("content_block");
        block.put("type", "text");
        block.put("text", "");
        return sseEvent("content_block_start", data.toString());
    }

    /** {@code content_block_delta} event appending {@code text} to the block at {@code index}. */
    static String textDeltaEvent(int index, String text) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_delta");
        data.put("index", index);
        ObjectNode delta = data.putObject("delta");
        delta.put("type", "text_delta");
        delta.put("text", text);
        return sseEvent("content_block_delta", data.toString());
    }

    /** {@code content_block_start} event opening a {@code tool_use} block at {@code index}. */
    static String toolUseBlockStartEvent(int index, String id, String name) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_start");
        data.put("index", index);
        ObjectNode block = data.putObject("content_block");
        block.put("type", "tool_use");
        block.put("id", id);
        block.put("name", name);
        block.putObject("input");
        return sseEvent("content_block_start", data.toString());
    }

    /** {@code content_block_delta} event carrying the tool-call arguments as an {@code input_json_delta}. */
    static String inputJsonDeltaEvent(int index, String partialJson) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_delta");
        data.put("index", index);
        ObjectNode delta = data.putObject("delta");
        delta.put("type", "input_json_delta");
        delta.put("partial_json", partialJson);
        return sseEvent("content_block_delta", data.toString());
    }

    /** {@code content_block_stop} event closing the block at {@code index}. */
    static String blockStopEvent(int index) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_stop");
        data.put("index", index);
        return sseEvent("content_block_stop", data.toString());
    }

    /** {@code message_delta} event carrying the final stop reason. */
    static String messageDeltaEvent(String stopReason) {
        return messageDeltaEvent(stopReason, 0, 0, 0);
    }

    /** Final message delta carrying token usage collected from the trailing OpenAI usage chunk. */
    static String messageDeltaEvent(String stopReason, long inputTokens, long outputTokens, long cachedTokens) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "message_delta");
        ObjectNode delta = data.putObject("delta");
        delta.put("stop_reason", stopReason);
        delta.putNull("stop_sequence");
        ObjectNode usage = data.putObject("usage");
        usage.put("input_tokens", inputTokens);
        usage.put("output_tokens", outputTokens);
        usage.put("cache_read_input_tokens", cachedTokens);
        return sseEvent("message_delta", data.toString());
    }

    /** {@code message_stop} event ending the stream. */
    static String messageStopEvent() {
        return sseEvent("message_stop", "{\"type\":\"message_stop\"}");
    }
}
