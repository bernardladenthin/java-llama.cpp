// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.Map;
import java.util.TreeMap;
import lombok.ToString;

/**
 * Reconstructs whole tool calls from the incremental {@code delta.tool_calls} fragments of an OpenAI
 * streaming chat completion. Across a stream, the first fragment for a given {@code index} carries the
 * call {@code id} and {@code function.name}, and subsequent fragments append {@code function.arguments}
 * string pieces. This accumulator merges them by index so the non-OpenAI protocol shims of the server
 * (Ollama, Anthropic, OpenAI Responses) — which deliver tool calls whole rather than fragmented — can
 * emit a complete tool-call list once the stream finishes, and so {@link ChatStreamAssembler} can
 * rebuild the message of a streamed answer.
 *
 * <p>Stateful but free of JNI / model dependencies; unit-testable by feeding chunk JSON literals.
 */
@ToString
public final class ToolCallDeltaAccumulator {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** Per-index partial tool call: id/name captured once, argument fragments concatenated. */
    private static final class Partial {
        private String id = "";
        private String name = "";
        private final StringBuilder arguments = new StringBuilder();
    }

    private final Map<Integer, Partial> byIndex = new TreeMap<>();

    /** Creates an empty accumulator. */
    public ToolCallDeltaAccumulator() {}

    /**
     * Feed one OpenAI chunk from its raw JSON; unparseable chunks are ignored. Convenience for streaming
     * sinks that hold the chunk as a string.
     *
     * @param openAiChunkJson one OpenAI {@code chat.completion.chunk} serialized as JSON
     */
    public void accept(String openAiChunkJson) {
        try {
            accept(OBJECT_MAPPER.readTree(openAiChunkJson));
        } catch (java.io.IOException e) {
            // A malformed chunk simply contributes no tool-call fragments.
        }
    }

    /**
     * Feed one OpenAI {@code chat.completion.chunk}; merges any {@code delta.tool_calls} fragments it
     * carries. Chunks without tool-call deltas are ignored.
     *
     * @param openAiChunk a parsed OpenAI streaming chunk
     */
    public void accept(JsonNode openAiChunk) {
        acceptToolCalls(openAiChunk.path("choices").path(0).path("delta").path("tool_calls"));
    }

    /**
     * Feed the {@code delta.tool_calls} array of one choice, for a caller that reads the chunk itself
     * (e.g. one with several choices). Anything but an array is ignored.
     *
     * @param toolCalls the {@code delta.tool_calls} node of one choice
     */
    public void acceptToolCalls(JsonNode toolCalls) {
        if (!toolCalls.isArray()) {
            return;
        }
        for (JsonNode toolCall : toolCalls) {
            int index = toolCall.path("index").asInt(0);
            Partial partial = byIndex.computeIfAbsent(index, k -> new Partial());
            if (toolCall.path("id").isTextual()) {
                partial.id = toolCall.path("id").asText();
            }
            JsonNode function = toolCall.path("function");
            if (function.path("name").isTextual()) {
                partial.name = function.path("name").asText();
            }
            if (function.path("arguments").isTextual()) {
                partial.arguments.append(function.path("arguments").asText());
            }
        }
    }

    /**
     * Whether any tool-call fragments were accumulated.
     *
     * @return {@code true} if at least one tool call was seen
     */
    public boolean hasToolCalls() {
        return !byIndex.isEmpty();
    }

    /**
     * The reconstructed tool calls as an OpenAI-shaped array
     * ({@code [{id,type:"function",function:{name,arguments}}]}), in index order. {@code arguments}
     * is the concatenated JSON-encoded string, exactly as the OpenAI non-streaming message carries it.
     *
     * @return the reconstructed tool-call array (empty when none were seen)
     */
    public ArrayNode toOpenAiToolCalls() {
        ArrayNode out = OBJECT_MAPPER.createArrayNode();
        for (Partial partial : byIndex.values()) {
            ObjectNode toolCall = out.addObject();
            toolCall.put("id", partial.id);
            toolCall.put("type", "function");
            ObjectNode function = toolCall.putObject("function");
            function.put("name", partial.name);
            function.put("arguments", partial.arguments.toString());
        }
        return out;
    }
}
