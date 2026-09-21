// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.server.ChunkSink;
import net.ladenthin.llama.server.OpenAiBackend;

/**
 * A model-free {@link OpenAiBackend} that answers each streaming chat request with a scripted sequence
 * of {@code chat.completion.chunk} objects — the shapes llama.cpp's server emits (role delta first,
 * {@code tool_calls} deltas keyed by {@code index}, {@code finish_reason:"tool_calls"} on the terminal
 * chunk) — and records every request it received, so a test can assert what an OpenAI client actually
 * put on the wire after the real {@code OpenAiCompatServer} routing, authentication and SSE framing.
 */
final class ScriptedBackend implements OpenAiBackend {

    /** Decides the chunks for the n-th request (1-based). */
    @FunctionalInterface
    interface Script {
        List<String> chunksFor(int call, JsonNode request) throws IOException;
    }

    private final Script script;
    private final List<JsonNode> requests = Collections.synchronizedList(new ArrayList<>());

    ScriptedBackend(Script script) {
        this.script = script;
    }

    /** Every {@code /v1/chat/completions} body received, in order. */
    List<JsonNode> requests() {
        return List.copyOf(requests);
    }

    @Override
    public void stream(JsonNode request, ChunkSink sink) throws IOException {
        requests.add(request.deepCopy());
        for (String chunk : script.chunksFor(requests.size(), request)) {
            sink.accept(chunk);
        }
    }

    @Override
    public String complete(JsonNode request) {
        throw new UnsupportedOperationException("Atmosphere always streams; a blocking request is a contract change");
    }

    @Override
    public String completions(JsonNode request) {
        throw new UnsupportedOperationException("not part of the agent contract");
    }

    @Override
    public String embeddings(JsonNode request) {
        throw new UnsupportedOperationException("not part of the agent contract");
    }

    @Override
    public String rerank(JsonNode request) {
        throw new UnsupportedOperationException("not part of the agent contract");
    }

    @Override
    public String infill(JsonNode request) {
        throw new UnsupportedOperationException("not part of the agent contract");
    }

    // ----- chunk builders (llama.cpp server shapes) -----

    static String roleChunk() {
        return "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null}]}";
    }

    static String textChunk(String content) {
        return "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"content\":" + quote(content) + "},\"finish_reason\":null}]}";
    }

    /** First delta of a tool call: carries index, id, type and name; arguments start empty. */
    static String toolCallStart(int index, String id, String name) {
        return "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"tool_calls\":[{\"index\":" + index + ",\"id\":" + quote(id)
                + ",\"type\":\"function\",\"function\":{\"name\":" + quote(name) + ",\"arguments\":\"\"}}]},"
                + "\"finish_reason\":null}]}";
    }

    /** A later delta of the same tool call: only an arguments fragment, addressed by index. */
    static String toolCallArguments(int index, String fragment) {
        return "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,"
                + "\"delta\":{\"tool_calls\":[{\"index\":" + index + ",\"function\":{\"arguments\":" + quote(fragment)
                + "}}]},\"finish_reason\":null}]}";
    }

    static String finish(String reason) {
        return "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,"
                + "\"delta\":{},\"finish_reason\":" + quote(reason) + "}]}";
    }

    /** A complete single-tool-call turn: role, start, arguments, {@code finish_reason:"tool_calls"}. */
    static List<String> toolCallTurn(String id, String name, String argumentsJson) {
        return List.of(
                roleChunk(), toolCallStart(0, id, name), toolCallArguments(0, argumentsJson), finish("tool_calls"));
    }

    /** A complete text turn split into one chunk per element, then {@code finish_reason:"stop"}. */
    static List<String> textTurn(String... pieces) {
        List<String> chunks = new ArrayList<>();
        chunks.add(roleChunk());
        for (String piece : pieces) {
            chunks.add(textChunk(piece));
        }
        chunks.add(finish("stop"));
        return chunks;
    }

    private static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                default -> sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
