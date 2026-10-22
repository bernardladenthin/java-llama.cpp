// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.function.Consumer;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.json.ChatStreamChunkParser;

/**
 * The envelopes of {@code receiveChatCompletionChunk}, as {@link LlamaModel}'s streaming loops read
 * them. A class of its own, without a native library behind it: {@link LlamaModel} loads the library
 * when it is initialized, and this translation is pure Java, tested without one.
 */
final class StreamEnvelope {

    private StreamEnvelope() {}

    /**
     * Feed one envelope of {@code receiveChatCompletionChunk} to {@code chunkSink}. An envelope that is
     * not what the native layer writes is a failure of the native layer, not of the caller's arguments,
     * so it is a {@link LlamaException} (a server answers it with {@code 500}, where an
     * {@link IllegalArgumentException} would be the request's fault); an exception of {@code chunkSink}
     * passes through unchanged.
     *
     * @param parser the envelope parser
     * @param envelope the envelope the native layer returned
     * @param chunkSink the caller's receiver of each chunk
     * @return {@code true} if the envelope ends the stream
     * @throws LlamaException if the envelope is malformed
     */
    static boolean feed(ChatStreamChunkParser parser, String envelope, Consumer<String> chunkSink) {
        try {
            return parser.feed(envelope, chunkSink);
        } catch (ChatStreamChunkParser.MalformedEnvelopeException e) {
            throw new LlamaException(String.valueOf(e.getMessage()), e);
        }
    }
}
