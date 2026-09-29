// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

/**
 * An editor, as far as the wire is concerned: newline-delimited JSON-RPC 2.0 on two pipes.
 *
 * <p>Deliberately <em>not</em> the SDK's own client. The point of the tests is what an editor such as
 * IntelliJ or Zed actually receives, so the messages are built and read as plain JSON, the way the protocol
 * describes them — an SDK on both ends would agree with itself even where it disagrees with the spec.
 */
final class AcpTestClient implements AutoCloseable {

    private static final ObjectMapper JSON = new ObjectMapper();

    /** Every message the agent sent, in order. */
    final List<JsonNode> received = new CopyOnWriteArrayList<>();

    private final OutputStream toAgent;
    private final AtomicInteger ids = new AtomicInteger();
    private final Thread reader;

    /** The agent's end of the pipes. */
    final PipedInputStream agentIn;

    /** The agent's end of the pipes. */
    final PipedOutputStream agentOut;

    AcpTestClient() throws IOException {
        this.agentIn = new PipedInputStream(1 << 16);
        this.toAgent = new PipedOutputStream(agentIn);
        this.agentOut = new PipedOutputStream();
        PipedInputStream fromAgent = new PipedInputStream(agentOut, 1 << 16);
        this.reader = Thread.ofVirtual().start(() -> {
            try (BufferedReader lines = new BufferedReader(new InputStreamReader(fromAgent, StandardCharsets.UTF_8))) {
                String line;
                while ((line = lines.readLine()) != null) {
                    if (!line.isBlank()) {
                        received.add(JSON.readTree(line));
                    }
                }
            } catch (IOException e) {
                // the agent closed its end
            }
        });
    }

    /**
     * Send a request and return its id.
     *
     * @param method the JSON-RPC method
     * @param params the parameters
     * @return the request id, to wait for the response with
     */
    int request(String method, ObjectNode params) {
        int id = ids.incrementAndGet();
        ObjectNode message =
                JSON.createObjectNode().put("jsonrpc", "2.0").put("id", id).put("method", method);
        message.set("params", params);
        write(message);
        return id;
    }

    /**
     * Send a notification.
     *
     * @param method the JSON-RPC method
     * @param params the parameters
     */
    void notify(String method, ObjectNode params) {
        ObjectNode message = JSON.createObjectNode().put("jsonrpc", "2.0").put("method", method);
        message.set("params", params);
        write(message);
    }

    /**
     * Answer a request the agent sent.
     *
     * @param id the agent's request id
     * @param result the result
     */
    void respond(JsonNode id, ObjectNode result) {
        ObjectNode message = JSON.createObjectNode().put("jsonrpc", "2.0");
        message.set("id", id);
        message.set("result", result);
        write(message);
    }

    private synchronized void write(ObjectNode message) {
        try {
            toAgent.write((message.toString() + "\n").getBytes(StandardCharsets.UTF_8));
            toAgent.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Wait for the response to one of our requests.
     *
     * @param id the request id
     * @param timeout how long
     * @return the response message
     */
    JsonNode response(int id, Duration timeout) {
        return await(message -> !message.has("method") && message.path("id").asInt(-1) == id, timeout);
    }

    /**
     * Wait for a message.
     *
     * @param condition what to wait for
     * @param timeout how long
     * @return the first matching message
     */
    JsonNode await(Predicate<JsonNode> condition, Duration timeout) {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            for (JsonNode message : received) {
                if (condition.test(message)) {
                    return message;
                }
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new AssertionError(e);
            }
        }
        throw new AssertionError("no matching message within " + timeout + "; got " + received);
    }

    /**
     * Every {@code session/update} of one kind, in order.
     *
     * @param kind the {@code sessionUpdate} value, e.g. {@code agent_message_chunk}
     * @return the updates' {@code update} objects
     */
    List<JsonNode> updates(String kind) {
        return received.stream()
                .filter(message ->
                        "session/update".equals(message.path("method").asText()))
                .map(message -> message.path("params").path("update"))
                .filter(update -> kind.equals(update.path("sessionUpdate").asText()))
                .toList();
    }

    /**
     * The streamed answer: every {@code agent_message_chunk}'s text, joined.
     *
     * @return the text
     */
    String messageText() {
        StringBuilder text = new StringBuilder();
        for (JsonNode update : updates("agent_message_chunk")) {
            text.append(update.path("content").path("text").asText());
        }
        return text.toString();
    }

    static ObjectNode object() {
        return JSON.createObjectNode();
    }

    @Override
    public void close() throws IOException {
        toAgent.close();
        reader.interrupt();
    }
}
