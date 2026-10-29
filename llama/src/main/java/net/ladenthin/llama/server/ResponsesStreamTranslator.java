// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import lombok.ToString;
import net.ladenthin.llama.json.ToolCallDeltaAccumulator;
import org.jspecify.annotations.Nullable;

/**
 * Stateful translator that turns the OpenAI streaming chat chunks into the OpenAI Responses SSE event
 * sequence: {@code response.created} → (for {@code reasoning_content}) a {@code reasoning} item with
 * {@code response.reasoning_text.delta}* + {@code response.reasoning_text.done} → (for a text message)
 * {@code response.output_item.added} +
 * {@code response.content_part.added} + {@code response.output_text.delta}* +
 * {@code response.output_text.done} + {@code response.content_part.done} +
 * {@code response.output_item.done} → (per tool call) a {@code function_call} item with
 * {@code response.function_call_arguments.done} → {@code response.completed}, or {@code
 * response.incomplete} for an answer cut off by a limit ({@code finish_reason} {@code "length"}, see
 * {@link ResponsesApiSupport#isTruncated}). Each event carries a monotonic {@code sequence_number}.
 *
 * <p>Text deltas are emitted live; tool calls are reconstructed via {@link ToolCallDeltaAccumulator} and
 * emitted as whole {@code function_call} items at the end. Free of JNI / model dependencies;
 * unit-testable by feeding chunk JSON.
 */
@ToString
final class ResponsesStreamTranslator {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final String model;
    private final String responseId;
    private final String messageItemId;
    private final String reasoningItemId;
    private final ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
    private final StringBuilder text = new StringBuilder();
    private final StringBuilder reasoning = new StringBuilder();

    private int sequence;
    private boolean reasoningOpen;
    private int reasoningOutputIndex = -1;
    private boolean messageOpen;
    private int nextOutputIndex;
    private int messageOutputIndex = -1;
    private long inputTokens;
    private long outputTokens;
    private long cachedTokens;
    /** The last textual {@code finish_reason} seen; the trailing usage-only chunk carries none. */
    private String finishReason = "";

    ResponsesStreamTranslator(String model, String responseId) {
        this.model = model;
        this.responseId = responseId;
        this.messageItemId = "msg_" + responseId;
        this.reasoningItemId = "rs_" + responseId;
    }

    /**
     * The opening {@code response.created} event.
     *
     * @return the framed SSE event
     */
    String begin() {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.set("response", ResponsesApiSupport.newResponseShell(model, responseId, "in_progress"));
        return ResponsesApiSupport.sseEvent("response.created", sequence++, data);
    }

    /**
     * Translate one OpenAI chunk into the Responses events it produces (opening the message item and
     * content part on first text, then text deltas), accumulating tool-call fragments. Returns an empty
     * string when the chunk yields no event.
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @return zero or more framed SSE events, concatenated
     */
    String onChunk(String openAiChunkJson) {
        StringBuilder out = new StringBuilder();
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            accumulator.accept(chunk);
            captureUsage(chunk.path("usage"));
            JsonNode chunkFinishReason = chunk.path("choices").path(0).path("finish_reason");
            if (chunkFinishReason.isTextual()) {
                finishReason = chunkFinishReason.asText();
            }
            // Reasoning streams as a reasoning output item, as in upstream's to_json_oaicompat_resp_stream.
            JsonNode deltaNode = chunk.path("choices").path(0).path("delta");
            JsonNode reasoningDelta = deltaNode.path("reasoning_content");
            String reasoningText = reasoningDelta.isTextual() ? reasoningDelta.asText() : "";
            if (!reasoningText.isEmpty()) {
                if (!reasoningOpen) {
                    reasoningOutputIndex = nextOutputIndex++;
                    out.append(outputItemAdded(reasoningOutputIndex, reasoningItem("in_progress", null)));
                    reasoningOpen = true;
                }
                reasoning.append(reasoningText);
                out.append(deltaEvent(
                        "response.reasoning_text.delta", reasoningItemId, reasoningOutputIndex, reasoningText));
            }
            JsonNode content = deltaNode.path("content");
            if (content.isTextual() && !content.asText().isEmpty()) {
                if (!messageOpen) {
                    messageOutputIndex = nextOutputIndex++;
                    out.append(outputItemAdded(messageOutputIndex, messageItemShell()));
                    out.append(contentPartAdded());
                    messageOpen = true;
                }
                String delta = content.asText();
                text.append(delta);
                out.append(deltaEvent("response.output_text.delta", messageItemId, messageOutputIndex, delta));
            }
        } catch (IOException e) {
            // A malformed chunk produces no events.
        }
        return out.toString();
    }

    /**
     * The closing events: finish the text content part / message item, emit a {@code function_call} item
     * per accumulated tool call, then {@code response.completed} carrying the assembled output and usage
     * ({@code response.incomplete}, with the message item {@code incomplete} too, for a cut-off answer).
     *
     * @return the framed SSE events, concatenated
     */
    String end() {
        boolean truncated = ResponsesApiSupport.isTruncated(finishReason);
        StringBuilder out = new StringBuilder();
        ArrayNode output = OBJECT_MAPPER.createArrayNode();

        if (reasoningOpen) {
            ObjectNode reasoningDone = ResponsesApiSupport.dataObject();
            reasoningDone.put("item_id", reasoningItemId);
            reasoningDone.put("output_index", reasoningOutputIndex);
            reasoningDone.put("content_index", 0);
            reasoningDone.put("text", reasoning.toString());
            out.append(ResponsesApiSupport.sseEvent("response.reasoning_text.done", sequence++, reasoningDone));

            ObjectNode item = reasoningItem("completed", reasoning.toString());
            output.add(item);
            out.append(outputItemDone(reasoningOutputIndex, item));
        }

        if (messageOpen) {
            ObjectNode textDone = ResponsesApiSupport.dataObject();
            textDone.put("item_id", messageItemId);
            textDone.put("output_index", messageOutputIndex);
            textDone.put("content_index", 0);
            textDone.put("text", text.toString());
            out.append(ResponsesApiSupport.sseEvent("response.output_text.done", sequence++, textDone));

            ObjectNode partDone = ResponsesApiSupport.dataObject();
            partDone.put("item_id", messageItemId);
            partDone.put("output_index", messageOutputIndex);
            partDone.put("content_index", 0);
            ObjectNode part = partDone.putObject("part");
            part.put("type", "output_text");
            part.put("text", text.toString());
            out.append(ResponsesApiSupport.sseEvent("response.content_part.done", sequence++, partDone));

            ObjectNode messageItem = finishedMessageItem(
                    truncated ? ResponsesApiSupport.STATUS_INCOMPLETE : ResponsesApiSupport.STATUS_COMPLETED);
            output.add(messageItem);
            out.append(outputItemDone(messageOutputIndex, messageItem));
        }

        for (JsonNode toolCall : accumulator.toOpenAiToolCalls()) {
            int index = nextOutputIndex++;
            ObjectNode functionCall = ResponsesApiSupport.functionCallItem(toolCall);
            out.append(outputItemAdded(index, functionCall));
            ObjectNode argsDone = ResponsesApiSupport.dataObject();
            argsDone.put("item_id", functionCall.path("id").asText());
            argsDone.put("output_index", index);
            argsDone.put(
                    "arguments", toolCall.path("function").path("arguments").asText(""));
            out.append(ResponsesApiSupport.sseEvent("response.function_call_arguments.done", sequence++, argsDone));
            out.append(outputItemDone(index, functionCall));
            output.add(functionCall);
        }

        ObjectNode completed = ResponsesApiSupport.dataObject();
        ObjectNode response =
                ResponsesApiSupport.newResponseShell(model, responseId, ResponsesApiSupport.STATUS_COMPLETED);
        if (truncated) {
            ResponsesApiSupport.markIncomplete(response);
        }
        response.set("output", output);
        ObjectNode usage = response.putObject("usage");
        usage.put("input_tokens", inputTokens);
        usage.put("output_tokens", outputTokens);
        usage.put("total_tokens", inputTokens + outputTokens);
        usage.putObject("input_tokens_details").put("cached_tokens", cachedTokens);
        completed.set("response", response);
        out.append(ResponsesApiSupport.sseEvent(
                truncated ? "response.incomplete" : "response.completed", sequence++, completed));
        return out.toString();
    }

    /**
     * The terminal {@code response.failed} event for a stream that ended in an error. A Responses client
     * dispatches on the event {@code type} and reads the failure from {@code response.error}; without
     * this event it sees a stream that ends before {@code response.completed} and loses the message.
     *
     * @param message the error message
     * @param code the error code
     * @return the framed SSE event
     */
    String failed(String message, String code) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        ObjectNode response = ResponsesApiSupport.newResponseShell(model, responseId, "failed");
        ObjectNode error = response.putObject("error");
        error.put("code", code);
        error.put("message", message);
        data.set("response", response);
        return ResponsesApiSupport.sseEvent("response.failed", sequence++, data);
    }

    private void captureUsage(JsonNode usage) {
        if (!usage.isObject()) {
            return;
        }
        inputTokens = usage.path("prompt_tokens").asLong(0);
        outputTokens = usage.path("completion_tokens").asLong(0);
        cachedTokens = usage.path("prompt_tokens_details").path("cached_tokens").asLong(0);
    }

    /** A {@code *.delta} event appending {@code delta} to content part 0 of the item {@code itemId}. */
    private String deltaEvent(String event, String itemId, int outputIndex, String delta) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("item_id", itemId);
        data.put("output_index", outputIndex);
        data.put("content_index", 0);
        data.put("delta", delta);
        return ResponsesApiSupport.sseEvent(event, sequence++, data);
    }

    /** A reasoning output item; {@code reasoningText} {@code null} gives the empty in-progress shell. */
    private ObjectNode reasoningItem(String status, @Nullable String reasoningText) {
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "reasoning");
        item.put("id", reasoningItemId);
        item.putArray("summary");
        ArrayNode content = item.putArray("content");
        if (reasoningText != null) {
            ObjectNode part = content.addObject();
            part.put("type", "reasoning_text");
            part.put("text", reasoningText);
        }
        item.put("encrypted_content", "");
        item.put("status", status);
        return item;
    }

    private ObjectNode messageItemShell() {
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "message");
        item.put("id", messageItemId);
        item.put("status", "in_progress");
        item.put("role", "assistant");
        item.putArray("content");
        return item;
    }

    private ObjectNode finishedMessageItem(String status) {
        ObjectNode item = OBJECT_MAPPER.createObjectNode();
        item.put("type", "message");
        item.put("id", messageItemId);
        item.put("status", status);
        item.put("role", "assistant");
        ObjectNode textPart = item.putArray("content").addObject();
        textPart.put("type", "output_text");
        textPart.put("text", text.toString());
        textPart.putArray("annotations");
        return item;
    }

    private String outputItemAdded(int outputIndex, ObjectNode item) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("output_index", outputIndex);
        data.set("item", item);
        return ResponsesApiSupport.sseEvent("response.output_item.added", sequence++, data);
    }

    private String outputItemDone(int outputIndex, ObjectNode item) {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("output_index", outputIndex);
        data.set("item", item);
        return ResponsesApiSupport.sseEvent("response.output_item.done", sequence++, data);
    }

    private String contentPartAdded() {
        ObjectNode data = ResponsesApiSupport.dataObject();
        data.put("item_id", messageItemId);
        data.put("output_index", messageOutputIndex);
        data.put("content_index", 0);
        ObjectNode part = data.putObject("part");
        part.put("type", "output_text");
        part.put("text", "");
        return ResponsesApiSupport.sseEvent("response.content_part.added", sequence++, data);
    }
}
