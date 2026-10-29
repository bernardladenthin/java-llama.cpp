// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;
import lombok.ToString;
import net.ladenthin.llama.value.ChatResponse;
import org.jspecify.annotations.Nullable;

/**
 * Assembles an OpenAI streaming chat completion into the chat completion the same request would have
 * been answered with without {@code stream}, so a streamed answer goes through the one
 * {@link ChatResponseParser} as well.
 *
 * <p>It reads the two forms a stream arrives in:
 * <ul>
 *   <li>the lines of a Server-Sent Events body, as an OpenAI-compatible server sends it (llama.cpp's
 *       {@code llama-server}, {@code OpenAiCompatServer}, OpenRouter): {@link #acceptLine(String)}, or
 *       {@link #parseStream(String)} for a whole body;</li>
 *   <li>the {@code chat.completion.chunk} JSON strings {@code LlamaModel#streamChatCompletion} hands
 *       its sink: {@link #acceptChunk(String)}.</li>
 * </ul>
 *
 * <p>Per choice ({@code index}, default 0) it joins {@code delta.content} and
 * {@code delta.reasoning_content}, keeps the role, rebuilds {@code delta.tool_calls} with a
 * {@link ToolCallDeltaAccumulator} and keeps the last finish reason. Of the whole stream it keeps the
 * first {@code id}, {@code created} and {@code model} and the last {@code usage} and {@code timings}
 * objects (the usage chunk that {@code stream_options.include_usage} adds, llama.cpp's timings on the
 * final chunk). {@link #toCompletionJson()} writes them in the shape of a non-streaming answer.</p>
 *
 * <p>Strict where a lenient reading would turn a broken stream into a plausible short answer: a data
 * line that is not JSON, a chunk carrying an {@code error} (a provider failing mid-stream) and, for
 * {@link #parseStream(String)}, a body without {@code data: [DONE]} throw {@link InvalidStreamException}.
 * Comments ({@code : ping} heartbeats), blank lines and the other SSE fields are skipped; lines after
 * {@code [DONE]} are not read.</p>
 *
 * <p>Stateful and not thread-safe: one instance per stream. Free of JNI / native / model dependencies,
 * testable with string literals alone (see {@code ChatStreamAssemblerTest}).</p>
 */
@ToString
public class ChatStreamAssembler {

    /** The field prefix of a Server-Sent Events line that carries data. */
    public static final String SSE_DATA_PREFIX = "data:";

    /** The data of the event that ends an OpenAI stream. */
    public static final String SSE_DONE = "[DONE]";

    /** Shared Jackson mapper; thread-safe and reused across all instances. */
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** How much of a rejected data line or error its message quotes. */
    private static final int EXCERPT = 200;

    /** The role of an assembled message whose stream named none. */
    private static final String DEFAULT_ROLE = "assistant";

    /** The state of one choice across the stream. */
    @ToString
    private static final class Choice {
        private String role = DEFAULT_ROLE;
        private final StringBuilder content = new StringBuilder();
        private final StringBuilder reasoningContent = new StringBuilder();
        private final ToolCallDeltaAccumulator toolCalls = new ToolCallDeltaAccumulator();
        private @Nullable String finishReason;
    }

    private final Map<Integer, Choice> choices = new TreeMap<>();
    private @Nullable JsonNode id;
    private @Nullable JsonNode created;
    private @Nullable JsonNode model;
    private @Nullable JsonNode usage;
    private @Nullable JsonNode timings;
    private boolean done;

    /** Creates an assembler for one stream. */
    public ChatStreamAssembler() {}

    /**
     * Assembles a whole Server-Sent Events body.
     *
     * @param sse the response body of a streamed chat completion
     * @return the parsed chat response of the assembled completion
     * @throws InvalidStreamException if a data line is not JSON, a chunk carries an error, or the body
     *     ends without {@code data: [DONE]}
     */
    public static ChatResponse parseStream(String sse) {
        ChatStreamAssembler assembler = new ChatStreamAssembler();
        // An event stream ends a line with LF, CRLF or CR. Ending one at each CR and each LF reads a
        // CRLF as a line and an empty one, and acceptLine skips an empty line.
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < sse.length(); i++) {
            char c = sse.charAt(i);
            if (c == '\n' || c == '\r') {
                assembler.acceptLine(line.toString());
                line.setLength(0);
            } else {
                line.append(c);
            }
        }
        assembler.acceptLine(line.toString());
        if (!assembler.isDone()) {
            throw new InvalidStreamException(
                    "the stream ended without '" + SSE_DATA_PREFIX + " " + SSE_DONE + "' after " + sse.length()
                            + " characters",
                    null);
        }
        return assembler.toResponse();
    }

    /**
     * Feeds one line of a Server-Sent Events body (without its line end). Only {@code data:} lines
     * count; once {@code data: [DONE]} was read, every further line is ignored.
     *
     * @param line one line of the body
     * @return {@code true} once the stream has ended with {@code data: [DONE]}
     * @throws InvalidStreamException if the line's data is not JSON or carries an error
     */
    public boolean acceptLine(String line) {
        if (done || !line.startsWith(SSE_DATA_PREFIX)) {
            return done;
        }
        String data = line.substring(SSE_DATA_PREFIX.length()).trim();
        if (SSE_DONE.equals(data)) {
            done = true;
        } else if (!data.isEmpty()) {
            acceptChunk(data);
        }
        return done;
    }

    /**
     * Feeds one {@code chat.completion.chunk} as its JSON string.
     *
     * @param chunkJson one chunk, e.g. the data of an SSE line or what {@code LlamaModel} hands a sink
     * @throws InvalidStreamException if the string is not one JSON object or the chunk carries an error
     */
    public void acceptChunk(String chunkJson) {
        JsonNode chunk;
        try {
            chunk = OBJECT_MAPPER.readTree(chunkJson);
        } catch (IOException e) {
            throw new InvalidStreamException("a stream chunk is not JSON: '" + excerpt(chunkJson) + "'", e);
        }
        if (!chunk.isObject()) {
            throw new InvalidStreamException("a stream chunk is not a JSON object: '" + excerpt(chunkJson) + "'", null);
        }
        acceptChunk(chunk);
    }

    /**
     * Feeds one parsed {@code chat.completion.chunk}.
     *
     * @param chunk one chunk object
     * @throws InvalidStreamException if the chunk carries an error
     */
    public void acceptChunk(JsonNode chunk) {
        if (chunk.hasNonNull("error")) {
            throw new InvalidStreamException(
                    "the stream reported an error: "
                            + excerpt(chunk.get("error").toString()),
                    null);
        }
        if (id == null && chunk.hasNonNull("id")) {
            id = chunk.get("id");
        }
        if (created == null && chunk.hasNonNull("created")) {
            created = chunk.get("created");
        }
        if (model == null && chunk.hasNonNull("model")) {
            model = chunk.get("model");
        }
        for (JsonNode choiceNode : chunk.path("choices")) {
            acceptChoice(choiceNode);
        }
        if (chunk.path("usage").isObject()) {
            usage = chunk.get("usage");
        }
        if (chunk.path("timings").isObject()) {
            timings = chunk.get("timings");
        }
    }

    private void acceptChoice(JsonNode choiceNode) {
        Choice choice = choices.computeIfAbsent(choiceNode.path("index").asInt(0), k -> new Choice());
        JsonNode delta = choiceNode.path("delta");
        if (delta.path("role").isTextual()) {
            choice.role = delta.path("role").asText();
        }
        if (delta.path("content").isTextual()) {
            choice.content.append(delta.path("content").asText());
        }
        if (delta.path("reasoning_content").isTextual()) {
            choice.reasoningContent.append(delta.path("reasoning_content").asText());
        }
        choice.toolCalls.acceptToolCalls(delta.path("tool_calls"));
        if (choiceNode.path("finish_reason").isTextual()) {
            choice.finishReason = choiceNode.path("finish_reason").asText();
        }
    }

    /**
     * Whether {@code data: [DONE]} was read by {@link #acceptLine(String)}. A caller that feeds chunks
     * (where the end is the end of its own loop) does not need it.
     *
     * @return {@code true} once the stream has ended
     */
    public boolean isDone() {
        return done;
    }

    /**
     * The assembled completion as the JSON of a non-streaming answer: {@code object}
     * {@code chat.completion}, one choice per index with {@code message.role}, {@code message.content},
     * {@code message.reasoning_content} and {@code message.tool_calls} when the stream carried any, and
     * {@code finish_reason} ({@code null} when none came), then {@code usage} and {@code timings} when
     * the stream carried them.
     *
     * @return the chat-completion JSON
     */
    public String toCompletionJson() {
        ObjectNode completion = OBJECT_MAPPER.createObjectNode();
        if (id != null) {
            completion.set("id", id);
        }
        completion.put("object", "chat.completion");
        if (created != null) {
            completion.set("created", created);
        }
        if (model != null) {
            completion.set("model", model);
        }
        ArrayNode choiceArray = completion.putArray("choices");
        for (Map.Entry<Integer, Choice> entry : choices.entrySet()) {
            Choice choice = entry.getValue();
            ObjectNode choiceNode = choiceArray.addObject();
            choiceNode.put("index", entry.getKey());
            ObjectNode message = choiceNode.putObject("message");
            message.put("role", choice.role);
            message.put("content", choice.content.toString());
            if (choice.reasoningContent.length() > 0) {
                message.put("reasoning_content", choice.reasoningContent.toString());
            }
            if (choice.toolCalls.hasToolCalls()) {
                message.set("tool_calls", choice.toolCalls.toOpenAiToolCalls());
            }
            String finishReason = choice.finishReason;
            if (finishReason == null) {
                choiceNode.putNull("finish_reason");
            } else {
                choiceNode.put("finish_reason", finishReason);
            }
        }
        if (usage != null) {
            completion.set("usage", usage);
        }
        if (timings != null) {
            completion.set("timings", timings);
        }
        return completion.toString();
    }

    /**
     * The assembled completion, parsed by {@link ChatResponseParser}.
     *
     * @return the chat response
     */
    public ChatResponse toResponse() {
        return new ChatResponseParser().parseResponse(toCompletionJson());
    }

    private static String excerpt(String text) {
        return text.length() > EXCERPT ? text.substring(0, EXCERPT) + "..." : text;
    }

    /**
     * A stream that cannot be assembled: a data line that is not JSON, a chunk that carries an error,
     * or a body without {@code data: [DONE]}. An {@link IllegalArgumentException}, since the stream is
     * this assembler's argument, like {@link ChatStreamChunkParser.MalformedEnvelopeException}.
     */
    public static final class InvalidStreamException extends IllegalArgumentException {

        private static final long serialVersionUID = 1L;

        /**
         * Creates the exception.
         *
         * @param message what is wrong, with an excerpt of the offending text
         * @param cause the parser's exception, or {@code null}
         */
        public InvalidStreamException(String message, @Nullable Throwable cause) {
            super(message, cause);
        }
    }
}
