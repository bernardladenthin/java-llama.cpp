// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;

/**
 * Receiver for the individual {@code chat.completion.chunk} JSON strings produced while a streaming
 * chat completion runs.
 *
 * <p>Distinct from {@link java.util.function.Consumer} because writing a chunk to an HTTP response can
 * fail with {@link IOException} (for example when the client disconnects); a checked exception lets that
 * failure propagate so the in-flight generation can be cancelled.
 */
@FunctionalInterface
public interface ChunkSink {

    /**
     * Accept one streaming chunk's JSON text.
     *
     * @param chunkJson a single {@code chat.completion.chunk} object serialized as JSON
     * @throws IOException if the chunk cannot be delivered (e.g. the client closed the connection)
     */
    void accept(String chunkJson) throws IOException;
}
