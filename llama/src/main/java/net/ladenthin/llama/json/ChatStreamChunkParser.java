// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.function.Consumer;

/**
 * Pure JSON transform for the streaming-chat envelope produced by the native
 * {@code receiveChatCompletionChunk} method.
 *
 * <p>The native side wraps each polled streaming result in a uniform envelope so the
 * Java side never has to distinguish a single chunk object from the final chunk array:
 * <pre>{@code
 * { "data": <chunk-object-or-array-of-chunks>, "stop": <boolean> }
 * }</pre>
 *
 * <p>{@code data} is exactly what an OpenAI streaming chat result's {@code to_json()}
 * produced: a single {@code chat.completion.chunk} object for a partial token, or a JSON
 * array of chunk objects for the final step (final delta chunk plus an optional usage
 * chunk). This parser emits each chunk as its own JSON string so a caller can forward it
 * verbatim as one SSE {@code data:} event.
 *
 * <p>Stateless and free of JNI / native / model dependencies — testable with JSON string
 * literals alone (see {@code ChatStreamChunkParserTest}). This is the Java analogue of
 * {@code wrap_stream_chunk} in {@code json_helpers.hpp}.
 */
public class ChatStreamChunkParser {

    /** Creates a new {@link ChatStreamChunkParser}. */
    public ChatStreamChunkParser() {}

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Parse one streaming envelope and feed each contained {@code chat.completion.chunk}
     * JSON string to {@code chunkSink}, in order.
     *
     * <p>A {@code data} array yields one {@code chunkSink} call per element; a {@code data}
     * object yields a single call; any other shape (absent/null) yields no calls. An
     * unparseable envelope is treated as end-of-stream (returns {@code true}) so a polling
     * loop cannot spin forever on malformed input.
     *
     * @param envelopeJson the raw {@code {"data":…,"stop":…}} string from the native layer
     * @param chunkSink receiver for each chunk's JSON string (one OpenAI SSE event each)
     * @return {@code true} if this envelope marks the end of the stream, else {@code false}
     */
    public boolean feed(String envelopeJson, Consumer<String> chunkSink) {
        final JsonNode root;
        try {
            root = OBJECT_MAPPER.readTree(envelopeJson);
        } catch (IOException e) {
            return true;
        }
        JsonNode data = root.path("data");
        if (data.isArray()) {
            for (JsonNode element : data) {
                chunkSink.accept(element.toString());
            }
        } else if (data.isObject()) {
            chunkSink.accept(data.toString());
        }
        return root.path("stop").asBoolean(false);
    }
}
