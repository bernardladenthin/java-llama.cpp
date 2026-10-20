// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

/**
 * {@code ResponseStream}, the lazily started response of the streaming routes, over a recording
 * {@link HttpExchange}: no server, no model.
 */
public class ResponseStreamTest {

    /** Records the status and the body; everything else is inert. */
    private static final class RecordingExchange extends HttpExchange {

        final Headers responseHeaders = new Headers();
        final ByteArrayOutputStream body = new ByteArrayOutputStream();
        int status = -1;

        @Override
        public Headers getRequestHeaders() {
            return new Headers();
        }

        @Override
        public Headers getResponseHeaders() {
            return responseHeaders;
        }

        @Override
        public URI getRequestURI() {
            return URI.create("/");
        }

        @Override
        public String getRequestMethod() {
            return "POST";
        }

        @Override
        public @Nullable HttpContext getHttpContext() {
            return null;
        }

        @Override
        public void close() {}

        @Override
        public InputStream getRequestBody() {
            return new ByteArrayInputStream(new byte[0]);
        }

        @Override
        public OutputStream getResponseBody() {
            return body;
        }

        @Override
        public void sendResponseHeaders(int rCode, long responseLength) {
            status = rCode;
        }

        @Override
        public InetSocketAddress getRemoteAddress() {
            return InetSocketAddress.createUnresolved("localhost", 0);
        }

        @Override
        public int getResponseCode() {
            return status;
        }

        @Override
        public InetSocketAddress getLocalAddress() {
            return InetSocketAddress.createUnresolved("localhost", 0);
        }

        @Override
        public String getProtocol() {
            return "HTTP/1.1";
        }

        @Override
        public @Nullable Object getAttribute(String name) {
            return null;
        }

        @Override
        public void setAttribute(String name, @Nullable Object value) {}

        @Override
        public void setStreams(@Nullable InputStream i, @Nullable OutputStream o) {}

        @Override
        public @Nullable HttpPrincipal getPrincipal() {
            return null;
        }

        String text() {
            return new String(body.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void theOpeningIsWrittenFirstWhenItIsSetBeforeTheFirstWrite() throws Exception {
        RecordingExchange exchange = new RecordingExchange();
        try (OpenAiCompatServer.ResponseStream out =
                new OpenAiCompatServer.ResponseStream(exchange, "text/event-stream")) {
            out.startWith("OPEN\n");
            out.writeQuietly(": ping\n");
            out.writeStrict("DATA\n");
        }

        assertThat(exchange.status, is(200));
        assertThat(exchange.text(), is("OPEN\n: ping\nDATA\n"));
    }

    /**
     * A heartbeat that fires before the route set its opening starts the stream. The opening used to be
     * lost then: it was only ever written on the first write, which had already happened, so an
     * Anthropic stream lacked {@code message_start} and a Responses stream {@code response.created}. A
     * heartbeat is an SSE comment, not an event, so the opening written after it is still the first
     * event of the stream.
     */
    @Test
    void anOpeningSetAfterAHeartbeatStartedTheStreamIsStillWritten() throws Exception {
        RecordingExchange exchange = new RecordingExchange();
        try (OpenAiCompatServer.ResponseStream out =
                new OpenAiCompatServer.ResponseStream(exchange, "text/event-stream")) {
            out.writeQuietly(": ping\n");
            out.startWith("OPEN\n");
            out.writeStrict("DATA\n");
        }

        assertThat(exchange.text(), is(": ping\nOPEN\nDATA\n"));
    }

    @Test
    void anErrorBeforeTheFirstWriteIsAnsweredWithItsStatusAndNoOpening() throws Exception {
        RecordingExchange exchange = new RecordingExchange();
        try (OpenAiCompatServer.ResponseStream out =
                new OpenAiCompatServer.ResponseStream(exchange, "text/event-stream")) {
            out.startWith("OPEN\n");
            assertThat(out.answerIfNotStarted(400, "{\"error\":\"bad\"}"), is(true));
            out.writeQuietly(": ping\n");
        }

        assertThat(exchange.status, is(400));
        assertThat(exchange.text(), is("{\"error\":\"bad\"}"));
    }

    @Test
    void anErrorAfterTheFirstWriteIsLeftToTheStream() throws Exception {
        RecordingExchange exchange = new RecordingExchange();
        try (OpenAiCompatServer.ResponseStream out =
                new OpenAiCompatServer.ResponseStream(exchange, "text/event-stream")) {
            out.writeQuietly(": ping\n");
            assertThat(out.answerIfNotStarted(400, "{\"error\":\"bad\"}"), is(false));
        }

        assertThat(exchange.status, is(200));
    }
}
