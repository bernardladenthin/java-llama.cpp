// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import lombok.ToString;
import net.ladenthin.llama.json.ToolCallDeltaAccumulator;
import org.jspecify.annotations.Nullable;

/**
 * Stateful translator that turns the OpenAI streaming chat chunks into the Anthropic Messages SSE event
 * sequence: {@code message_start} → (a {@code thinking} block for {@code reasoning_content}, closed with an
 * empty {@code signature_delta}) → (a {@code text} content block with {@code content_block_start} +
 * {@code content_block_delta}s + {@code content_block_stop}) → one {@code tool_use} block per tool call
 * (start + {@code input_json_delta} + stop) → {@code message_delta} (stop reason) → {@code message_stop}.
 *
 * <p>The blocks follow one another, at most one is open at a time: reasoning that arrives after text
 * closes the text block and opens a new thinking block, and text after it a new text block (a parser
 * that interleaves thinking and text, e.g. GPT-OSS or Gemma 4, gives thinking, text, thinking, text), as
 * Claude streams several thinking blocks. The non-streamed {@link AnthropicApiSupport#toAnthropicResponse}
 * cannot mirror that, because the chat message carries the reasoning as one concatenated string; an
 * interleaved generation therefore has more blocks streamed than blocking, deliberately.
 *
 * <p>Text deltas are emitted live; tool calls are reconstructed via {@link ToolCallDeltaAccumulator} and
 * emitted as whole blocks at the end (Anthropic expects each tool_use block's input as one
 * {@code input_json_delta}). Free of JNI / model dependencies; unit-testable by feeding chunk JSON.
 */
@ToString
final class AnthropicStreamTranslator {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final String id;
    private final String model;
    private final ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();

    private boolean thinkingBlockOpen;
    private int thinkingBlockIndex = -1;
    private boolean textBlockOpen;
    private int textBlockIndex = -1;
    private int nextIndex;
    private String finishReason = "stop";
    private @Nullable String stopSequence;
    private long inputTokens;
    private long outputTokens;
    private long cachedTokens;

    AnthropicStreamTranslator(String id, String model) {
        this.id = id;
        this.model = model;
    }

    /**
     * The opening {@code message_start} event.
     *
     * @return the framed SSE event
     */
    String begin() {
        return AnthropicApiSupport.messageStartEvent(id, model);
    }

    /**
     * Translate one OpenAI chunk into the SSE events it produces, accumulating tool-call fragments and
     * capturing the finish reason: {@code reasoning_content} opens a {@code thinking} block (closing an
     * open text block first) and adds a {@code thinking_delta}; non-empty {@code content} closes an open
     * thinking block (an empty {@code signature_delta} + {@code content_block_stop}), opens a text block
     * if none is open and adds a {@code text_delta}. Returns an empty string when the chunk yields no
     * event (role-only / finish-only / tool-call-only chunks).
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk}
     * @return zero or more framed SSE events, concatenated
     */
    String onChunk(String openAiChunkJson) {
        StringBuilder out = new StringBuilder();
        try {
            JsonNode chunk = OBJECT_MAPPER.readTree(openAiChunkJson);
            accumulator.accept(chunk);
            JsonNode usage = chunk.path("usage");
            if (usage.isObject()) {
                long promptTokens = usage.path("prompt_tokens").asLong(0);
                cachedTokens = usage.path("prompt_tokens_details")
                        .path("cached_tokens")
                        .asLong(0);
                inputTokens = Math.max(0L, promptTokens - cachedTokens);
                outputTokens = usage.path("completion_tokens").asLong(0);
            }
            String word = AnthropicApiSupport.stopSequenceOf(chunk);
            if (word != null) {
                stopSequence = word;
            }
            JsonNode choice = chunk.path("choices").path(0);
            // Reasoning streams as a thinking block, as in upstream's to_json_anthropic_stream.
            JsonNode reasoning = choice.path("delta").path("reasoning_content");
            if (reasoning.isTextual() && !reasoning.asText().isEmpty()) {
                closeTextBlock(out);
                if (!thinkingBlockOpen) {
                    thinkingBlockIndex = nextIndex++;
                    out.append(AnthropicApiSupport.thinkingBlockStartEvent(thinkingBlockIndex));
                    thinkingBlockOpen = true;
                }
                out.append(AnthropicApiSupport.thinkingDeltaEvent(thinkingBlockIndex, reasoning.asText()));
            }
            JsonNode content = choice.path("delta").path("content");
            if (content.isTextual() && !content.asText().isEmpty()) {
                closeThinkingBlock(out);
                if (!textBlockOpen) {
                    textBlockIndex = nextIndex++;
                    out.append(AnthropicApiSupport.textBlockStartEvent(textBlockIndex));
                    textBlockOpen = true;
                }
                out.append(AnthropicApiSupport.textDeltaEvent(textBlockIndex, content.asText()));
            }
            if (choice.path("finish_reason").isTextual()) {
                finishReason = choice.path("finish_reason").asText();
            }
        } catch (IOException e) {
            // A malformed chunk produces no events.
        }
        return out.toString();
    }

    /**
     * The closing events: close a thinking block that is still open (a reasoning-only answer, or
     * reasoning followed directly by a tool call; an empty {@code signature_delta} +
     * {@code content_block_stop}), stop the open text block, emit a {@code tool_use} block per
     * accumulated tool call, then {@code message_delta} (mapped stop reason) and {@code message_stop}.
     *
     * @return the framed SSE events, concatenated
     */
    String end() {
        StringBuilder out = new StringBuilder();
        closeThinkingBlock(out);
        closeTextBlock(out);
        if (accumulator.hasToolCalls()) {
            for (JsonNode toolCall : accumulator.toOpenAiToolCalls()) {
                int index = nextIndex++;
                String callId = toolCall.path("id").asText("");
                String name = toolCall.path("function").path("name").asText("");
                String arguments = toolCall.path("function").path("arguments").asText("");
                out.append(AnthropicApiSupport.toolUseBlockStartEvent(index, callId, name));
                out.append(AnthropicApiSupport.inputJsonDeltaEvent(index, arguments));
                out.append(AnthropicApiSupport.blockStopEvent(index));
            }
        }
        out.append(AnthropicApiSupport.messageDeltaEvent(
                AnthropicApiSupport.anthropicStopReason(finishReason, stopSequence),
                stopSequence,
                inputTokens,
                outputTokens,
                cachedTokens));
        out.append(AnthropicApiSupport.messageStopEvent());
        return out.toString();
    }

    /** Close an open thinking block: the empty signature the protocol requires, then the block stop. */
    private void closeThinkingBlock(StringBuilder out) {
        if (thinkingBlockOpen) {
            out.append(AnthropicApiSupport.signatureDeltaEvent(thinkingBlockIndex));
            out.append(AnthropicApiSupport.blockStopEvent(thinkingBlockIndex));
            thinkingBlockOpen = false;
        }
    }

    /** Close an open text block, so the next text delta opens a new one after the block in between. */
    private void closeTextBlock(StringBuilder out) {
        if (textBlockOpen) {
            out.append(AnthropicApiSupport.blockStopEvent(textBlockIndex));
            textBlockOpen = false;
        }
    }
}
