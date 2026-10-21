// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * What the agent's HTTP retry policy does with an error status, against an endpoint that answers every
 * chat request with the same status and counts them.
 *
 * <p>The other wire tests run with {@link RetryPolicy#NONE}, so nothing there tells a {@code 4xx} apart
 * from a {@code 5xx}. The agent itself runs with {@link RetryPolicy#DEFAULT}; the policy here keeps
 * DEFAULT's set of retryable error types and only sets the delays to zero, so the counts are exact and
 * no test waits for a backoff. A request the server refuses as the caller's fault ({@code 400}) must
 * not be sent again; an unavailable server ({@code 503}) is asked again until the retries are used up,
 * and the turn then ends with the error rather than hanging.
 */
class RetryPolicyWireTest {

    private static final String MODEL_ID = "local-model";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);
    private static final int MAX_RETRIES = 2;
    private static final String CHAT_PATH = "/v1/chat/completions";

    /** DEFAULT's retryable error types, without its one-second backoff. */
    private static final RetryPolicy IMMEDIATE_RETRIES =
            new RetryPolicy(MAX_RETRIES, Duration.ZERO, Duration.ZERO, 1.0, RetryPolicy.DEFAULT.retryableErrors());

    @TempDir
    Path workspace;

    private HttpServer server;
    private final AtomicInteger chatRequests = new AtomicInteger();

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    /** Answers every chat request with {@code status} and an OpenAI-style error body; 404 elsewhere. */
    private String startServerAnswering(int status) throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", exchange -> {
            try (HttpExchange e = exchange) {
                e.getRequestBody().readAllBytes();
                boolean chat = "POST".equals(e.getRequestMethod())
                        && CHAT_PATH.equals(e.getRequestURI().getPath());
                if (chat) {
                    chatRequests.incrementAndGet();
                }
                int code = chat ? status : 404;
                byte[] body = ("{\"error\":{\"code\":" + code + ",\"message\":\"status " + code
                                + "\",\"type\":\"test_error\"}}")
                        .getBytes(StandardCharsets.UTF_8);
                e.getResponseHeaders().set("Content-Type", "application/json");
                e.sendResponseHeaders(code, body.length);
                try (OutputStream out = e.getResponseBody()) {
                    out.write(body);
                }
            }
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort() + "/v1";
    }

    private ConsoleSession runTurn(String baseUrl) throws InterruptedException {
        AgentRunner runner = new AgentRunner(baseUrl, "k", MODEL_ID, List.of(), "You are a test agent.", 0.0, 64, 4)
                .retryPolicy(IMMEDIATE_RETRIES);
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        ConsoleSession session = new ConsoleSession(
                new PlainTerminal(
                        new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8), null, Ansi.PLAIN),
                fs);
        runner.run("Hello", List.of(), session);
        assertThat("the turn ends", session.await(TIMEOUT), is(true));
        return session;
    }

    @Test
    void aBadRequestIsSentExactlyOnce() throws Exception {
        String baseUrl = startServerAnswering(400);

        ConsoleSession session = runTurn(baseUrl);

        assertThat(chatRequests.get(), is(1));
        assertThat(session.failure(), is(notNullValue()));
        assertThat(String.valueOf(session.failure().getMessage()), containsString("400"));
    }

    @Test
    void anUnavailableServerIsAskedAgainUntilTheRetriesAreUsedUp() throws Exception {
        String baseUrl = startServerAnswering(503);

        ConsoleSession session = runTurn(baseUrl);

        assertThat(chatRequests.get(), is(1 + MAX_RETRIES));
        assertThat(session.failure(), is(notNullValue()));
        assertThat(String.valueOf(session.failure().getMessage()), containsString("503"));
    }
}
