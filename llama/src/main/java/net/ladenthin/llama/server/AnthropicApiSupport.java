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
 * Pure translators between the Anthropic Messages API ({@code POST /v1/messages}) and the internal
 * OpenAI chat shape, plus builders for the Anthropic streaming SSE events. Lets clients that speak the
 * Anthropic protocol (Claude Code, Copilot's {@code messages} apiType) drive the local model without a
 * second inference path.
 *
 * <p>Request mapping covers Anthropic's content-block model: a {@code system} string/blocks becomes an
 * OpenAI system message; message {@code content} that is a string or an array of {@code text} /
 * {@code image} / {@code thinking} / {@code tool_use} / {@code tool_result} blocks is mapped to OpenAI
 * messages: text is flattened to a string, a message carrying an image keeps its blocks as {@code text}
 * / {@code image_url} content parts (also inside a {@code tool_result}), thinking blocks become
 * {@code reasoning_content}, and a user message's {@code tool_result} blocks become separate
 * {@code role:"tool"} messages; Anthropic {@code tools} ({@code name}/{@code description}/
 * {@code input_schema}) become OpenAI function tools. Responses map the other way:
 * {@code reasoning_content} becomes a leading {@code thinking} block with an empty {@code signature},
 * OpenAI {@code content} + {@code tool_calls} become {@code text} + {@code tool_use} content blocks.
 *
 * <p>Stateless and free of JNI / model dependencies; unit-testable with JSON literals. Streaming state
 * is held by {@link AnthropicStreamTranslator}.
 */
final class AnthropicApiSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** First line of Claude Code's system prompt, whose {@code cch} stamp changes per request. */
    private static final String BILLING_HEADER_PREFIX = "x-anthropic-billing-header:";

    /** Length of the {@code cch} stamp upstream overwrites. */
    private static final int BILLING_STAMP_LENGTH = 5;

    /** Output cap of a request without {@code max_tokens} (upstream's default). */
    private static final int DEFAULT_MAX_TOKENS = 4096;

    /** Reasoning budget of an enabled {@code thinking} block without {@code budget_tokens} (upstream's default). */
    private static final int DEFAULT_THINKING_BUDGET_TOKENS = 10_000;

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
     * @throws IllegalArgumentException if {@code max_tokens}, {@code temperature}, {@code top_p} or
     *     {@code top_k} is present, not {@code null} and not a number
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
            JsonNode toolChoice = toOpenAiToolChoice(request.path("tool_choice"));
            if (toolChoice != null) {
                openAi.set("tool_choice", toolChoice);
            }
            // Anthropic expresses "no parallel tool use" via tool_choice.disable_parallel_tool_use;
            // OpenAI's equivalent is parallel_tool_calls=false. Map it so the shared chat core honors
            // a client's request to serialize tool calls (default stays parallel when unset/false).
            if (request.path("tool_choice").path("disable_parallel_tool_use").asBoolean(false)) {
                openAi.put("parallel_tool_calls", false);
            }
        }

        // As upstream's convert_anthropic_to_oai: the protocol requires max_tokens, a request without
        // it is capped at upstream's default instead of generating until the context is full. null
        // counts as absent, as in upstream's schema ("clients can send null to request the server
        // default"); any other value that is not a number upstream's schema refuses, and so does this.
        JsonNode maxTokens = OpenAiRequestMapper.optionalNumber(request, "max_tokens");
        if (maxTokens != null) {
            openAi.set("max_tokens", maxTokens);
        } else {
            openAi.put("max_tokens", DEFAULT_MAX_TOKENS);
        }
        // Upstream passes these through and its schema refuses a value that is not a number.
        copyNumber(request, "temperature", openAi, "temperature");
        copyNumber(request, "top_p", openAi, "top_p");
        copyNumber(request, "top_k", openAi, "top_k");
        if (request.path("stop_sequences").isArray()) {
            openAi.set("stop", request.path("stop_sequences").deepCopy());
        }
        // Upstream answers with the matched stop word as stop_sequence; the OpenAI answer this route
        // translates from does not carry it, upstream's debug object does (__verbose.stopping_word).
        // Only that word: the whole object echoes the rendered prompt and the generation settings
        // through JNI with every answer, to be dropped here.
        openAi.put("verbose", true);
        openAi.putArray("response_fields").add("stopping_word");
        // Passed through like upstream; the chat mapper hands it to the template.
        JsonNode templateKwargs = request.path("chat_template_kwargs");
        if (templateKwargs.isObject()) {
            openAi.set("chat_template_kwargs", templateKwargs.deepCopy());
        }
        // As upstream's convert_anthropic_to_oai: an enabled thinking block becomes the reasoning budget
        // the chat parser hands to the sampler, with upstream's default when budget_tokens is absent.
        JsonNode thinking = request.path("thinking");
        if ("enabled".equals(thinking.path("type").asText(""))) {
            openAi.put("thinking_budget_tokens", thinking.path("budget_tokens").asInt(DEFAULT_THINKING_BUDGET_TOKENS));
        }
        return openAi;
    }

    private static String systemText(JsonNode system) {
        if (system.isTextual()) {
            return normalizeBillingHeader(system.asText());
        }
        if (system.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode block : system) {
                if (block.path("text").isTextual()) {
                    sb.append(normalizeBillingHeader(block.path("text").asText()));
                }
            }
            return sb.toString();
        }
        return "";
    }

    /**
     * Overwrites the {@code cch} stamp of Claude Code's {@code x-anthropic-billing-header} system line,
     * as upstream's {@code normalize_anthropic_billing_header} (llama.cpp #21793) does. The stamp changes
     * on every request, so leaving it in place made every prompt differ at its start and the KV prefix
     * cache never hit: each turn re-processed the whole system prompt. Any other text is returned as is.
     */
    static String normalizeBillingHeader(String systemText) {
        if (!systemText.startsWith(BILLING_HEADER_PREFIX)) {
            return systemText;
        }
        int stamp = systemText.indexOf("cch=", BILLING_HEADER_PREFIX.length());
        if (stamp < 0) {
            return systemText;
        }
        int start = stamp + "cch=".length();
        int end = start + BILLING_STAMP_LENGTH;
        if (end >= systemText.length() || systemText.charAt(end) != ';') {
            return systemText;
        }
        return systemText.substring(0, start) + "fffff" + systemText.substring(end);
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
        // Thinking blocks of earlier turns, collected as reasoning_content like upstream's converter.
        StringBuilder reasoning = new StringBuilder();
        for (JsonNode block : content) {
            String type = block.path("type").asText("");
            switch (type) {
                case "text":
                    // Joined with a newline, as llama.cpp's concat_content_parts joins text parts.
                    if (text.length() > 0) {
                        text.append('\n');
                    }
                    text.append(block.path("text").asText(""));
                    parts.addObject()
                            .put("type", "text")
                            .put("text", block.path("text").asText(""));
                    break;
                case "thinking":
                    reasoning.append(block.path("thinking").asText(""));
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
                    // A missing or null input is no arguments; upstream json::parse()s every arguments
                    // string of the history, so an empty one fails the whole request (F166).
                    JsonNode input = block.path("input");
                    function.put("arguments", input.isMissingNode() || input.isNull() ? "{}" : input.toString());
                    break;
                case "tool_result":
                    // A user-message tool_result becomes a separate OpenAI role:"tool" message.
                    ObjectNode toolMessage = out.addObject();
                    toolMessage.put("role", "tool");
                    toolMessage.put("tool_call_id", block.path("tool_use_id").asText(""));
                    toolMessage.set("content", toolResultContent(block.path("content")));
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
            putReasoning(message, reasoning);
        } else if (text.length() > 0 || toolCalls.size() > 0 || reasoning.length() > 0) {
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
            putReasoning(message, reasoning);
        } else if (!hadToolResult) {
            // Genuinely empty/plain content (no text, no tool calls, no tool_result) — keep a slot.
            // A content array of only tool_result blocks emits no extra message (they became tool messages).
            ObjectNode message = out.addObject();
            message.put("role", role);
            message.put("content", "");
        }
    }

    private static void putReasoning(ObjectNode message, CharSequence reasoning) {
        if (reasoning.length() > 0) {
            message.put("reasoning_content", reasoning.toString());
        }
    }

    /**
     * The OpenAI {@code image_url} URL for an Anthropic image {@code source}: a data URL for a base64
     * source, the URL itself for a url source, {@code null} for anything else. A base64 source without
     * {@code media_type} is named by its encoded magic bytes ({@link ImageMediaTypes#ofBase64}), and as
     * {@code image/jpeg} when they are not recognised, upstream's default
     * ({@code server_chat_convert_anthropic_to_oai}, {@code tools/server/server-chat.cpp:423}, b11639).
     */
    private static @Nullable String imageUrl(JsonNode source) {
        String sourceType = source.path("type").asText("");
        if ("base64".equals(sourceType) && source.path("data").isTextual()) {
            String data = source.path("data").asText();
            String mediaType = source.path("media_type").asText("");
            if (mediaType.isEmpty()) {
                String sniffed = ImageMediaTypes.ofBase64(data);
                mediaType = sniffed != null ? sniffed : "image/jpeg";
            }
            return "data:" + mediaType + ";base64," + data;
        }
        if ("url".equals(sourceType) && source.path("url").isTextual()) {
            return source.path("url").asText();
        }
        return null;
    }

    /**
     * The content of the tool message for a {@code tool_result}: its text, or, when it carries an image,
     * text and {@code image_url} parts in block order, as upstream's {@code convert_anthropic_to_oai}
     * does. Flattening to text alone dropped an image a tool returned (a screenshot, an image file).
     */
    private static JsonNode toolResultContent(JsonNode content) {
        if (content.isArray()) {
            ArrayNode parts = OBJECT_MAPPER.createArrayNode();
            boolean hadImage = false;
            for (JsonNode block : content) {
                String type = block.path("type").asText("");
                if ("text".equals(type)) {
                    parts.addObject()
                            .put("type", "text")
                            .put("text", block.path("text").asText(""));
                } else if ("image".equals(type)) {
                    String url = imageUrl(block.path("source"));
                    if (url != null) {
                        parts.addObject()
                                .put("type", "image_url")
                                .putObject("image_url")
                                .put("url", url);
                        hadImage = true;
                    }
                }
            }
            if (hadImage) {
                return parts;
            }
        }
        return new TextNode(toolResultText(content));
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

    /**
     * The OpenAI {@code tool_choice} for an Anthropic one: {@code auto}, {@code none} (upstream's
     * converter leaves it out and runs the request as {@code auto}), {@code any} as {@code required}, and
     * {@code {"type":"tool","name":…}} as OpenAI's named function, which the chat mapper enforces by
     * forwarding that tool alone with {@code required} (upstream makes it {@code required} over every
     * tool). Any other type ({@code allowed_tools}, {@code mcp}, …) and a value that is no object are
     * refused, where upstream runs the request as {@code auto}.
     *
     * @param toolChoice the request's {@code tool_choice}
     * @return the OpenAI {@code tool_choice}, or {@code null} when the request carries none
     * @throws IllegalArgumentException for a {@code tool_choice} this translation cannot honour
     */
    private static @Nullable JsonNode toOpenAiToolChoice(JsonNode toolChoice) {
        if (toolChoice.isMissingNode() || toolChoice.isNull()) {
            return null;
        }
        if (!toolChoice.isObject()) {
            throw new IllegalArgumentException("'tool_choice' must be an object (got " + toolChoice + ")");
        }
        String type = toolChoice.path("type").asText("");
        switch (type) {
            case "auto":
                return TextNode.valueOf("auto");
            case "none":
                return TextNode.valueOf("none");
            case "any":
                return TextNode.valueOf("required");
            case "tool":
                JsonNode name = toolChoice.path("name");
                if (!name.isTextual() || name.asText().isEmpty()) {
                    throw new IllegalArgumentException(
                            "tool_choice of type 'tool' needs the tool's 'name' (got " + toolChoice + ")");
                }
                ObjectNode named = OBJECT_MAPPER.createObjectNode();
                named.put("type", "function");
                named.putObject("function").put("name", name.asText());
                return named;
            default:
                throw new IllegalArgumentException(
                        "tool_choice type '" + type + "' is not supported; use auto, any, tool or none");
        }
    }

    private static void copyNumber(JsonNode from, String fromKey, ObjectNode to, String toKey) {
        JsonNode value = OpenAiRequestMapper.optionalNumber(from, fromKey);
        if (value != null) {
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
        String stopSequence = null;
        ObjectNode usage = root.putObject("usage");
        usage.put("input_tokens", 0);
        usage.put("output_tokens", 0);
        try {
            JsonNode completion = OBJECT_MAPPER.readTree(openAiCompletionJson);
            JsonNode choice = completion.path("choices").path(0);
            JsonNode message = choice.path("message");
            // As upstream's to_json_anthropic: the reasoning comes first as a thinking block, with an
            // empty signature (a local model has nothing to sign).
            String reasoning = message.path("reasoning_content").asText("");
            if (!reasoning.isEmpty()) {
                ObjectNode thinkingBlock = content.addObject();
                thinkingBlock.put("type", "thinking");
                thinkingBlock.put("thinking", reasoning);
                thinkingBlock.put("signature", "");
            }
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
            stopSequence = stopSequenceOf(completion);
            stopReason = anthropicStopReason(choice.path("finish_reason").asText("stop"), stopSequence);
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
        if (stopSequence != null) {
            root.put("stop_sequence", stopSequence);
        } else {
            root.putNull("stop_sequence");
        }
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
                // readTree("") returns a MissingNode, which serializes as null, rather than throwing;
                // a parameterless call's empty arguments must still become an input object.
                JsonNode parsed = OBJECT_MAPPER.readTree(arguments.asText());
                if (parsed.isObject() || parsed.isArray()) {
                    return parsed;
                }
            } catch (IOException e) {
                // fall through to an empty object
            }
        }
        return OBJECT_MAPPER.createObjectNode();
    }

    /**
     * The stop word that ended the answer, as upstream reports it in {@code stop_sequence} (the stop
     * reason that goes with it is {@link #anthropicStopReason(String, String)}'s): read from the
     * {@code __verbose} object the translated request asks for, {@code null} when generation ended
     * otherwise or the object is absent.
     *
     * @param openAiAnswer an OpenAI chat answer or stream chunk
     * @return the matched stop word, or {@code null}
     */
    static @Nullable String stopSequenceOf(JsonNode openAiAnswer) {
        String word = openAiAnswer.path("__verbose").path("stopping_word").asText("");
        return word.isEmpty() ? null : word;
    }

    /**
     * Map an OpenAI finish_reason and the matched stop word to an Anthropic stop_reason: a {@code stop}
     * that a stop word ended is {@code stop_sequence}, as the Messages API specifies ("If the model
     * encounters one of the custom sequences, the response stop_reason value will be "stop_sequence""),
     * every other case is {@link #anthropicStopReason(String)}'s.
     *
     * <p>This departs from upstream llama.cpp on purpose: its {@code to_json_anthropic} and
     * {@code to_json_anthropic_stream} (b11639 and master) answer {@code end_turn} for
     * {@code STOP_TYPE_WORD} as for the end-of-generation token, which leaves a client that branches on
     * {@code stop_reason} unable to tell a stop-word stop from a natural end. The word is non-null
     * exactly for a stop-word stop (upstream sets {@code stopping_word} only then), and such a stop
     * always has the finish reason {@code stop}, so {@code tool_use} and {@code max_tokens} stay as they
     * are.
     *
     * @param openAiFinishReason the OpenAI finish reason
     * @param stopSequence the matched stop word ({@link #stopSequenceOf(JsonNode)}), or {@code null}
     * @return the Anthropic stop reason
     */
    static String anthropicStopReason(String openAiFinishReason, @Nullable String stopSequence) {
        if (stopSequence != null && "stop".equals(openAiFinishReason)) {
            return "stop_sequence";
        }
        return anthropicStopReason(openAiFinishReason);
    }

    /** Map an OpenAI finish_reason to an Anthropic stop_reason, without regard to a stop word. */
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

    /** {@code content_block_start} event opening an empty {@code thinking} block at {@code index}. */
    static String thinkingBlockStartEvent(int index) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_start");
        data.put("index", index);
        ObjectNode block = data.putObject("content_block");
        block.put("type", "thinking");
        block.put("thinking", "");
        return sseEvent("content_block_start", data.toString());
    }

    /** {@code content_block_delta} event appending reasoning text to the thinking block at {@code index}. */
    static String thinkingDeltaEvent(int index, String thinking) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_delta");
        data.put("index", index);
        ObjectNode delta = data.putObject("delta");
        delta.put("type", "thinking_delta");
        delta.put("thinking", thinking);
        return sseEvent("content_block_delta", data.toString());
    }

    /**
     * {@code content_block_delta} event carrying the (empty) signature the protocol requires before a
     * thinking block closes; a local model has nothing to sign, as in upstream's stream.
     */
    static String signatureDeltaEvent(int index) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "content_block_delta");
        data.put("index", index);
        ObjectNode delta = data.putObject("delta");
        delta.put("type", "signature_delta");
        delta.put("signature", "");
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
        return messageDeltaEvent(stopReason, null, 0, 0, 0);
    }

    /** Final message delta carrying token usage collected from the trailing OpenAI usage chunk. */
    static String messageDeltaEvent(
            String stopReason, @Nullable String stopSequence, long inputTokens, long outputTokens, long cachedTokens) {
        ObjectNode data = OBJECT_MAPPER.createObjectNode();
        data.put("type", "message_delta");
        ObjectNode delta = data.putObject("delta");
        delta.put("stop_reason", stopReason);
        if (stopSequence != null) {
            delta.put("stop_sequence", stopSequence);
        } else {
            delta.putNull("stop_sequence");
        }
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
