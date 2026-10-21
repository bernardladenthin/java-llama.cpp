// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import java.io.IOException;
import java.util.function.Consumer;
import org.jspecify.annotations.Nullable;

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
 * array of chunk objects for the final step (any remaining tool-call delta chunks, the
 * chunk carrying {@code finish_reason}, then an optional usage chunk). This parser emits
 * each chunk as its own JSON string so a caller can forward it verbatim as one SSE
 * {@code data:} event.
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

    /** Reads exactly one JSON value: {@code FAIL_ON_TRAILING_TOKENS} refuses text after it. */
    private static final ObjectReader STRICT_READER =
            OBJECT_MAPPER.reader().with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    /** How much of a rejected envelope its message quotes. */
    private static final int ENVELOPE_EXCERPT = 80;

    /**
     * Parse one streaming envelope and feed each contained {@code chat.completion.chunk}
     * JSON string to {@code chunkSink}, in order.
     *
     * <p>A {@code data} array yields one {@code chunkSink} call per element (none for an empty array);
     * a {@code data} object yields a single call. The envelope must be exactly what the native side
     * writes -- one JSON object with {@code data} an object or an array and {@code stop} a boolean;
     * anything else is a broken transport and is rejected rather than taken for the end of the stream,
     * which would make a broken envelope look like a short answer.
     *
     * @param envelopeJson the raw {@code {"data":…,"stop":…}} string from the native layer
     * @param chunkSink receiver for each chunk's JSON string (one OpenAI SSE event each)
     * @return {@code true} if this envelope marks the end of the stream, else {@code false}
     * @throws MalformedEnvelopeException if the envelope is not of that form; the message quotes it (at
     *     most 80 characters), nothing is passed to {@code chunkSink}
     */
    public boolean feed(String envelopeJson, Consumer<String> chunkSink) {
        final JsonNode root;
        try {
            root = STRICT_READER.readTree(envelopeJson);
        } catch (IOException e) {
            throw malformed(envelopeJson, "it is not one well-formed JSON value", e);
        }
        // Jackson returns MissingNode for an empty or blank text
        if (!root.isObject()) {
            throw malformed(envelopeJson, "it is not a JSON object", null);
        }
        JsonNode stop = root.path("stop");
        if (!stop.isBoolean()) {
            throw malformed(envelopeJson, "'stop' is not a boolean", null);
        }
        JsonNode data = root.path("data");
        if (data.isArray()) {
            for (JsonNode element : data) {
                chunkSink.accept(element.toString());
            }
        } else if (data.isObject()) {
            chunkSink.accept(data.toString());
        } else {
            throw malformed(envelopeJson, "'data' is neither an object nor an array", null);
        }
        return stop.booleanValue();
    }

    private static MalformedEnvelopeException malformed(String envelopeJson, String why, @Nullable Throwable cause) {
        String excerpt = envelopeJson.length() > ENVELOPE_EXCERPT
                ? envelopeJson.substring(0, ENVELOPE_EXCERPT) + "..."
                : envelopeJson;
        return new MalformedEnvelopeException(
                "malformed streaming envelope from the native layer (" + why + "), expected "
                        + "{\"data\": <object or array>, \"stop\": <boolean>} but got: '" + excerpt + "'",
                cause);
    }

    /**
     * An envelope that is not what the native layer writes. An {@link IllegalArgumentException}, since
     * the envelope is this parser's argument; its own type lets {@code LlamaModel} tell it from an
     * exception of the caller's chunk sink and report it as the failure of the native layer it is.
     */
    public static final class MalformedEnvelopeException extends IllegalArgumentException {

        private static final long serialVersionUID = 1L;

        /**
         * Creates the exception.
         *
         * @param message what is wrong, with an excerpt of the envelope
         * @param cause the parser's exception, or {@code null}
         */
        public MalformedEnvelopeException(String message, @Nullable Throwable cause) {
            super(message, cause);
        }
    }
}
