// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * What a browser running the Atmosphere console does on the wire, in plain JDK code: a WebSocket to the
 * agent's endpoint with atmosphere.js's query parameters, messages split by their length prefix.
 *
 * <p>The parameters are the ones the console sends. {@code X-Atmosphere-TrackMessageSize} makes every
 * message arrive as {@code <length>|<payload>}, possibly several in one frame, which is why frames are not
 * taken as messages directly.
 */
final class AtmosphereTestClient implements AutoCloseable {

    private static final Pattern APPROVAL_ID = Pattern.compile("\"approvalId\"\\s*:\\s*\"([^\"]+)\"");

    /** Every message received, in order. */
    final List<String> messages = new CopyOnWriteArrayList<>();

    private volatile WebSocket socket;
    private final StringBuilder pending = new StringBuilder();

    private AtmosphereTestClient() {}

    /**
     * Connect as the console does.
     *
     * @param port the server's port
     * @param token the access token, sent as a bearer header
     * @return the connected client
     */
    static AtmosphereTestClient connect(int port, String token) {
        return connect(port, builder -> builder.header("Authorization", "Bearer " + token));
    }

    /**
     * Connect with whatever the caller adds to the handshake.
     *
     * @param port the server's port
     * @param handshake adds headers
     * @return the connected client
     */
    static AtmosphereTestClient connect(int port, java.util.function.UnaryOperator<WebSocket.Builder> handshake) {
        URI uri = URI.create("ws://127.0.0.1:" + port + WebServer.AGENT_PATH + "?X-Atmosphere-tracking-id=0"
                + "&X-Atmosphere-Framework=5.0.7&X-Atmosphere-Transport=websocket&X-atmo-protocol=true"
                + "&X-Atmosphere-TrackMessageSize=true");
        // The client exists before the socket does: the handshake message can arrive before buildAsync()
        // returns, and a listener with nowhere to put it would drop it and never ask for the next one.
        AtmosphereTestClient client = new AtmosphereTestClient();
        client.socket = handshake
                .apply(HttpClient.newHttpClient().newWebSocketBuilder().connectTimeout(Duration.ofSeconds(10)))
                .buildAsync(uri, new WebSocket.Listener() {
                    @Override
                    public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
                        client.receive(data);
                        webSocket.request(1);
                        return null;
                    }
                })
                .orTimeout(10, TimeUnit.SECONDS)
                .join();
        // The handshake message comes first; nothing sent before it is guaranteed to be routed.
        client.awaitMessage(message -> true, Duration.ofSeconds(10));
        return client;
    }

    private synchronized void receive(CharSequence data) {
        pending.append(data);
        while (true) {
            int bar = pending.indexOf("|");
            if (bar <= 0) {
                return;
            }
            int length;
            try {
                length = Integer.parseInt(pending.substring(0, bar));
            } catch (NumberFormatException e) {
                messages.add(pending.toString());
                pending.setLength(0);
                return;
            }
            if (pending.length() < bar + 1 + length) {
                return;
            }
            messages.add(pending.substring(bar + 1, bar + 1 + length));
            pending.delete(0, bar + 1 + length);
        }
    }

    /**
     * Send one message, as the console's input box does.
     *
     * @param text the message
     */
    void send(String text) {
        socket.sendText(text, true).orTimeout(10, TimeUnit.SECONDS).join();
    }

    /**
     * Wait until a message matching {@code condition} has arrived.
     *
     * @param condition what to wait for
     * @param timeout how long
     * @return the first matching message
     * @throws AssertionError when none arrives in time
     */
    String awaitMessage(Predicate<String> condition, Duration timeout) {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            for (String message : messages) {
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
        throw new AssertionError("no matching message within " + timeout + "; got " + messages);
    }

    /**
     * Wait for the {@code complete} message that ends a request.
     *
     * @param timeout how long
     */
    void awaitComplete(Duration timeout) {
        awaitMessage(message -> message.contains("\"type\":\"complete\""), timeout);
    }

    /**
     * Wait until {@code count} requests have completed on this connection.
     *
     * @param count how many {@code complete} messages
     * @param timeout how long
     */
    void awaitCompletes(int count, Duration timeout) {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (messages.stream()
                        .filter(message -> message.contains("\"type\":\"complete\""))
                        .count()
                < count) {
            if (System.nanoTime() > deadline) {
                throw new AssertionError(count + " completions expected within " + timeout + "; got " + messages);
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new AssertionError(e);
            }
        }
    }

    /**
     * Wait for an approval request and answer it, as the console's buttons do.
     *
     * @param approve {@code true} for Approve, {@code false} for Deny
     * @param timeout how long to wait for the request
     */
    void answerApproval(boolean approve, Duration timeout) {
        String request = awaitMessage(message -> message.contains("approval-required"), timeout);
        Matcher id = APPROVAL_ID.matcher(request);
        if (!id.find()) {
            throw new AssertionError("no approvalId in " + request);
        }
        send("/__approval/" + id.group(1) + (approve ? "/approve" : "/deny"));
    }

    /**
     * Everything streamed as text, concatenated.
     *
     * @return the text of every {@code streaming-text} message
     */
    String streamedText() {
        StringBuilder text = new StringBuilder();
        Pattern data = Pattern.compile("\"type\":\"streaming-text\",\"data\":\"((?:[^\"\\\\]|\\\\.)*)\"");
        for (String message : messages) {
            Matcher matcher = data.matcher(message);
            if (matcher.find()) {
                text.append(matcher.group(1).replace("\\n", "\n").replace("\\\"", "\""));
            }
        }
        return text.toString();
    }

    @Override
    public void close() {
        socket.sendClose(WebSocket.NORMAL_CLOSURE, "done")
                .orTimeout(5, TimeUnit.SECONDS)
                .exceptionally(e -> null);
    }
}
