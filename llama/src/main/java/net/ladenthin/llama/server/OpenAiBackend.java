// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;

/**
 * The inference engine seam behind {@link OpenAiCompatServer}.
 *
 * <p>Decoupling the HTTP layer from {@link net.ladenthin.llama.LlamaModel} lets the whole server —
 * routing, authentication, Server-Sent-Events framing, heartbeats — be exercised by tests with a fake
 * backend, with no native library and no model loaded. The production implementation is
 * {@link LlamaModelBackend}.
 *
 * <p>Every method receives the parsed OpenAI request object (already validated as a JSON object by the
 * handler) and returns the OpenAI-shaped response JSON, except {@link #stream} which delivers chunks
 * incrementally. The {@code GET /v1/models} response is built from configuration alone and so is not
 * part of this seam.
 *
 * <p>Public so that sibling modules can drive the real HTTP surface with a scripted backend — the
 * {@code llama-atmosphere-agent} wire-contract tests replay llama.cpp-shaped
 * {@code chat.completion.chunk} sequences through {@link OpenAiCompatServer} to prove an OpenAI client's
 * tool-calling loop end to end without a model.
 */
public interface OpenAiBackend {

    /**
     * Return llama.cpp server metrics, including per-slot cache counters.
     * Test backends may rely on the empty default.
     *
     * @return metrics JSON
     * @throws IOException if metrics cannot be read
     */
    default String metrics() throws IOException {
        return "{\"slots\":[]}";
    }

    /**
     * Run a non-streaming chat completion ({@code POST /v1/chat/completions}).
     *
     * @param request the parsed OpenAI {@code /v1/chat/completions} request
     * @return the complete OpenAI {@code chat.completion} response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String complete(JsonNode request) throws IOException;

    /**
     * Run a streaming chat completion, delivering each {@code chat.completion.chunk} to {@code sink}
     * in order. Implementations must not emit the terminating {@code [DONE]} marker; the caller adds it.
     *
     * @param request the parsed OpenAI {@code /v1/chat/completions} request
     * @param sink receiver for each streamed chunk's JSON
     * @throws IOException if a chunk cannot be delivered or generation fails
     */
    void stream(JsonNode request, ChunkSink sink) throws IOException;

    /**
     * Run a (non-streaming) text completion ({@code POST /v1/completions}). The request body is
     * forwarded verbatim to the native OpenAI-compatible completion handler.
     *
     * @param request the parsed OpenAI {@code /v1/completions} request (must contain {@code "prompt"})
     * @return the OpenAI {@code text_completion} response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String completions(JsonNode request) throws IOException;

    /**
     * Run a <em>streaming</em> text completion ({@code POST /v1/completions} with {@code stream:true}),
     * delivering each OpenAI {@code text_completion} chunk to {@code sink} in order. Implementations must
     * not emit the terminating {@code [DONE]} marker; the caller adds it. The default throws
     * {@link UnsupportedOperationException}; backends that support streaming completions override it.
     *
     * @param request the parsed OpenAI {@code /v1/completions} request (must contain {@code "prompt"})
     * @param sink receiver for each streamed chunk's JSON
     * @throws IOException if a chunk cannot be delivered or generation fails
     */
    default void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
        throw new UnsupportedOperationException("streaming /v1/completions is not supported by this backend");
    }

    /**
     * Generate embeddings ({@code POST /v1/embeddings}). Requires the model to have been loaded in
     * embedding mode; otherwise the native call fails and the caller surfaces a server error.
     *
     * @param request the parsed OpenAI {@code /v1/embeddings} request (must contain {@code "input"})
     * @return the OpenAI embeddings response serialized as JSON
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String embeddings(JsonNode request) throws IOException;

    /**
     * Rerank documents against a query ({@code POST /v1/rerank}). Requires the model to have been loaded
     * in reranking mode; otherwise the native call fails and the caller surfaces a server error.
     *
     * @param request the parsed rerank request ({@code query} string + {@code documents} array, optional
     *                {@code top_n})
     * @return the rerank response serialized as JSON ({@code results}/{@code data} of
     *         {@code {index, relevance_score}})
     * @throws IOException if reranking fails in a way the caller should surface as a server error
     */
    String rerank(JsonNode request) throws IOException;

    /**
     * Run a (non-streaming) fill-in-the-middle completion ({@code POST /infill}). The request body is
     * forwarded verbatim to the native llama.cpp infill handler, which applies the model's FIM control
     * tokens server-side from GGUF metadata — so callers send raw {@code input_prefix} /
     * {@code input_suffix} (and optional {@code input_extra} / {@code prompt}). This is the endpoint
     * that drives local ghost-text autocomplete clients (llama.vscode, llama.vim, Twinny, Tabby,
     * Continue's {@code llama.cpp} provider).
     *
     * @param request the parsed llama.cpp {@code /infill} request (typically {@code input_prefix} +
     *                {@code input_suffix})
     * @return the infill response serialized as JSON (clients read the {@code "content"} field)
     * @throws IOException if generation fails in a way the caller should surface as a server error
     */
    String infill(JsonNode request) throws IOException;
}
