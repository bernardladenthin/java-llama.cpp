// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.function.Consumer;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;

/**
 * Production {@link OpenAiBackend} that runs requests against a loaded {@link LlamaModel}.
 *
 * <p>Non-streaming chat reuses {@link LlamaModel#chatComplete(InferenceParameters)}, whose return value
 * is already a verbatim OpenAI {@code chat.completion} body. Streaming chat uses
 * {@link LlamaModel#streamChatCompletion(InferenceParameters, java.util.function.Consumer)}, which emits
 * OpenAI {@code chat.completion.chunk} objects (including {@code delta.tool_calls}). Text completions and
 * embeddings forward the request body verbatim to {@link LlamaModel#handleCompletionsOai(String)} /
 * {@link LlamaModel#handleEmbeddings(String, boolean)}, which already return OpenAI-shaped JSON.
 *
 * <p>The streaming sink may fail with {@link IOException} (client disconnect); because the underlying
 * model API takes a {@link java.util.function.Consumer} (no checked exceptions), that failure is
 * relayed across the boundary via {@link UncheckedIOException} and unwrapped here so the
 * in-flight native task is cancelled ({@link #relay}).
 *
 * <p>A request the native layer rejects -- a body that is not valid JSON, a missing field, an empty
 * embedding input, a prompt over the context size -- arrives as
 * {@link net.ladenthin.llama.exception.InvalidRequestException} and is left untouched: the server maps
 * that type to {@code 400}, as upstream's HTTP server does for the same failures. Every other
 * {@link net.ladenthin.llama.exception.LlamaException} stays a {@code 500}.
 */
final class LlamaModelBackend implements OpenAiBackend {

    private final LlamaModel model;
    private final OpenAiRequestMapper mapper;

    /**
     * Create a backend over the given model.
     *
     * @param model the loaded model to run completions against
     * @param mapper the OpenAI-request to {@link InferenceParameters} mapper
     */
    LlamaModelBackend(LlamaModel model, OpenAiRequestMapper mapper) {
        this.model = model;
        this.mapper = mapper;
    }

    @Override
    public String metrics() {
        return model.getMetrics();
    }

    @Override
    public String complete(JsonNode request) {
        return model.chatComplete(mapper.toInferenceParameters(request));
    }

    @Override
    public void stream(JsonNode request, ChunkSink sink) throws IOException {
        InferenceParameters params = mapper.toInferenceParameters(request);
        relay(chunkConsumer -> model.streamChatCompletion(params, chunkConsumer), sink);
    }

    @Override
    public String completions(JsonNode request) {
        // The native /v1/completions handler parses the OpenAI body itself; forward it verbatim.
        return model.handleCompletionsOai(request.toString());
    }

    @Override
    public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
        // The native handler parses the OpenAI body itself, as for the blocking request; forward it.
        ObjectNode body = request.deepCopy();
        body.put("stream", true);
        relay(chunkConsumer -> model.streamCompletionsOai(body.toString(), chunkConsumer), sink);
    }

    /**
     * Run a streaming model call, handing its chunks to {@code sink}, and give back an
     * {@link IOException} of the sink as that same exception.
     *
     * <p>The model API takes a {@link Consumer}, which cannot throw a checked exception, so the sink's
     * {@link IOException} (a client that disconnected) crosses it as an {@link UncheckedIOException}.
     * The model call cancels its native task when that exception unwinds it; afterwards the original
     * {@link IOException} is rethrown, so that the server takes its disconnect path rather than the one
     * for a failed completion. An {@link UncheckedIOException} the sink did not cause, and every other
     * exception of the model call, propagate unchanged.
     *
     * @param modelCall the model call, given the consumer its chunks go to
     * @param sink where each chunk is written
     * @throws IOException the sink's own exception, if it threw one
     */
    static void relay(Consumer<Consumer<String>> modelCall, ChunkSink sink) throws IOException {
        // Holds an IOException thrown by the sink so it can be rethrown after the model API (which
        // only understands unchecked exceptions) unwinds and cancels the native task.
        final IOException[] sinkFailure = new IOException[1];
        try {
            modelCall.accept(chunkJson -> {
                try {
                    sink.accept(chunkJson);
                } catch (IOException e) {
                    sinkFailure[0] = e;
                    throw new UncheckedIOException(e);
                }
            });
        } catch (UncheckedIOException e) {
            IOException cause = sinkFailure[0];
            if (cause != null) {
                throw cause;
            }
            throw e;
        }
    }

    @Override
    public String mediaMarker() {
        return model.getModelMeta().getMediaMarker();
    }

    @Override
    public String embeddings(JsonNode request) {
        // oaiCompat=true so the response uses the OpenAI {"object":"list","data":[{embedding}]} shape.
        return model.handleEmbeddings(request.toString(), true);
    }

    @Override
    public String infill(JsonNode request) {
        // The native /infill handler parses the body itself (input_prefix/input_suffix/...) and applies
        // the model's FIM tokens from GGUF metadata; forward verbatim.
        return model.handleInfill(request.toString());
    }

    @Override
    public String rerank(JsonNode request) throws IOException {
        final String query = OaiRerankSupport.readQuery(request);
        final String[] documents = OaiRerankSupport.readDocuments(request);
        OaiRerankSupport.readTopN(request); // a negative top_n is refused before anything is evaluated
        final String requestModel = request.path("model").asText("");
        final String nativeJson = model.handleRerank(query, documents);
        return OaiRerankSupport.toResponse(request, nativeJson, requestModel);
    }
}
