// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletionException;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The browser front end end to end: the real Jetty, the real Atmosphere endpoint and console, the real
 * agent session, and the real {@link OpenAiCompatServer} with a scripted engine behind it. What a browser
 * does is done over plain HTTP and a WebSocket speaking atmosphere.js's protocol ({@link
 * AtmosphereTestClient}).
 *
 * <p>Two halves: that nothing gets in without the token (the agent can run commands, so this is the part
 * that must not regress quietly), and that a conversation works — streaming, commands, approvals through
 * the console's buttons, and stopping a request.
 */
class WebServerTest {

    private static final String TOKEN = "test-token-0123456789abcdef";
    private static final Duration WAIT = Duration.ofSeconds(20);

    @TempDir
    Path workspace;

    private final List<AutoCloseable> open = new ArrayList<>();
    private final HttpClient http = HttpClient.newHttpClient();

    @AfterEach
    void closeAll() throws Exception {
        for (int i = open.size() - 1; i >= 0; i--) {
            open.get(i).close();
        }
    }

    private WebServer start(ScriptedBackend backend, String... extra) throws Exception {
        OpenAiCompatServer model = new OpenAiCompatServer(
                        backend,
                        OpenAiServerConfig.builder()
                                .host("127.0.0.1")
                                .port(0)
                                .apiKey("sk-local")
                                .modelId("local-model")
                                .build())
                .start();
        open.add(model);
        List<String> args = new ArrayList<>(List.of(
                "--base-url",
                "http://127.0.0.1:" + model.getPort() + "/v1",
                "--workspace",
                workspace.toString(),
                "--web"));
        args.addAll(List.of(extra));
        AgentOptions options = AgentOptions.parse(args.toArray(String[]::new));
        AgentSession session = AgentSession.open(options, "http://127.0.0.1:" + model.getPort() + "/v1", 10_000);
        WebServer server = WebServer.start(session, "127.0.0.1", 0, TOKEN, "test");
        open.add(server);
        return server;
    }

    private HttpResponse<String> get(WebServer server, String path, String... headers) throws Exception {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path));
        if (headers.length > 0) {
            request.headers(headers);
        }
        return http.send(request.build(), HttpResponse.BodyHandlers.ofString());
    }

    private static String toolResult(JsonNode request) {
        for (JsonNode message : request.path("messages")) {
            if ("tool".equals(message.path("role").asText())) {
                return message.path("content").asText();
            }
        }
        return "";
    }

    // ----- the door -----

    @Test
    void withoutTheTokenNothingIsServed() throws Exception {
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        assertThat(get(server, "/").statusCode(), is(401));
        assertThat(get(server, WebConsole.PATH + "/").statusCode(), is(401));
        assertThat(get(server, WebConsole.Info.PATH).statusCode(), is(401));
        assertThat(get(server, "/?token=wrong-token-0000000000").statusCode(), is(401));
        assertThat(get(server, "/", "Authorization", "Bearer wrong").statusCode(), is(401));
        CompletionException refused = assertThrows(
                CompletionException.class, () -> AtmosphereTestClient.connect(server.port(), builder -> builder));
        assertThat(
                "the WebSocket upgrade is behind the same door",
                ((java.net.http.WebSocketHandshakeException) refused.getCause())
                        .getResponse()
                        .statusCode(),
                is(401));
    }

    @Test
    void theTokenIsExchangedForACookieAndLeavesTheAddressBar() throws Exception {
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        HttpResponse<String> exchange = get(server, "/?token=" + TOKEN);

        assertThat(exchange.statusCode(), is(302));
        assertThat(exchange.headers().firstValue("Location").orElse(""), containsString(WebConsole.PATH + "/"));
        String cookie = exchange.headers().firstValue("Set-Cookie").orElse("");
        assertThat(cookie, startsWith(WebAccessGuard.COOKIE + "=" + TOKEN));
        assertThat(cookie, containsString("HttpOnly"));
        assertThat(cookie, containsString("SameSite=Strict"));

        HttpResponse<String> page = get(server, WebConsole.PATH + "/", "Cookie", WebAccessGuard.COOKIE + "=" + TOKEN);
        assertThat(page.statusCode(), is(200));
        assertThat(page.body(), containsString("Atmosphere AI Console"));
        assertThat("the nonce placeholder is filled in", page.body(), not(containsString("__ATMO_CSP_NONCE__")));
        assertThat(page.headers().firstValue("Content-Security-Policy").orElse(""), containsString("nonce-"));
    }

    @Test
    void theConsoleIsToldWhereTheAgentIs() throws Exception {
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        HttpResponse<String> info = get(server, WebConsole.Info.PATH, "Authorization", "Bearer " + TOKEN);

        assertThat(info.statusCode(), is(200));
        assertThat(info.body(), containsString("\"endpoint\":\"" + WebServer.AGENT_PATH + "\""));
        assertThat(info.body(), containsString("\"hasAdmin\":false"));
    }

    @Test
    void anotherPageInTheSameBrowserIsTurnedAway() throws Exception {
        // Cross-site WebSocket hijacking: a page on another origin opening a socket to localhost. The cookie
        // would not even be sent (SameSite=Strict); this pins that the Origin check stops it regardless.
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        assertThat(
                get(server, "/", "Authorization", "Bearer " + TOKEN, "Origin", "http://evil.example")
                        .statusCode(),
                is(403));
        assertThat(
                get(server, "/", "Authorization", "Bearer " + TOKEN, "Origin", "http://127.0.0.1:" + server.port())
                        .statusCode(),
                is(302));
    }

    @Test
    void aForeignHostNameIsTurnedAwayOnLoopback() {
        // DNS rebinding keeps the attacker's name in the Host header even when it resolves to 127.0.0.1.
        assertThat(WebAccessGuard.isLoopback(WebAccessGuard.hostName("127.0.0.1:8787")), is(true));
        assertThat(WebAccessGuard.isLoopback(WebAccessGuard.hostName("localhost")), is(true));
        assertThat(WebAccessGuard.isLoopback(WebAccessGuard.hostName("[::1]:8787")), is(true));
        assertThat(WebAccessGuard.isLoopback(WebAccessGuard.hostName("rebind.example:8787")), is(false));
        assertThat(WebAccessGuard.sameOrigin("http://127.0.0.1:8787", "127.0.0.1:8787"), is(true));
        assertThat(WebAccessGuard.sameOrigin("http://127.0.0.1:9999", "127.0.0.1:8787"), is(false));
        assertThat(WebAccessGuard.sameOrigin("null", "127.0.0.1:8787"), is(false));
    }

    @Test
    void onlyOneBrowserFrontEndPerProcess() throws Exception {
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        IllegalStateException twice = assertThrows(
                IllegalStateException.class,
                () -> WebServer.start(WebAgentEndpoint.SESSION.get(), "127.0.0.1", 0, TOKEN, "second"));
        assertThat(twice.getMessage(), containsString("already running"));
        assertThat(server.url(), containsString("?token=" + TOKEN));
    }

    // ----- the conversation -----

    @Test
    void aMessageStreamsItsAnswerAndCompletes() throws Exception {
        WebServer server =
                start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("Hello ", "from ", "the agent")));

        try (AtmosphereTestClient browser = AtmosphereTestClient.connect(server.port(), TOKEN)) {
            browser.send("hi");
            browser.awaitComplete(WAIT);

            assertThat(browser.streamedText(), is("Hello from the agent"));
        }
    }

    @Test
    void commandsAnswerInTheBrowser() throws Exception {
        WebServer server = start(new ScriptedBackend((call, request) -> ScriptedBackend.textTurn("x")));

        try (AtmosphereTestClient browser = AtmosphereTestClient.connect(server.port(), TOKEN)) {
            browser.send("/status");
            browser.awaitComplete(WAIT);

            assertThat(browser.streamedText(), containsString("history: 0 messages"));
        }
    }

    @Test
    void theApproveButtonRunsTheCommandAndShowsItsCard() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", ShellTool.TOOL_NAME, "{\"command\":\"echo from-the-web\"}")
                : ScriptedBackend.textTurn("Done."));
        WebServer server = start(backend, "--allow-shell");

        try (AtmosphereTestClient browser = AtmosphereTestClient.connect(server.port(), TOKEN)) {
            browser.send("run it");
            browser.answerApproval(true, WAIT);
            browser.awaitComplete(WAIT);

            assertThat(toolResult(backend.requests().get(1)), containsString("from-the-web"));
            browser.awaitMessage(message -> message.contains("\"tool-start\""), WAIT);
            browser.awaitMessage(message -> message.contains("\"tool-result\""), WAIT);
            assertThat(browser.streamedText(), containsString("Done."));
        }
    }

    @Test
    void theDenyButtonKeepsTheCommandFromRunning() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> call == 1
                ? ScriptedBackend.toolCallTurn("call_1", ShellTool.TOOL_NAME, "{\"command\":\"echo from-the-web\"}")
                : ScriptedBackend.textTurn("Understood."));
        WebServer server = start(backend, "--allow-shell");

        try (AtmosphereTestClient browser = AtmosphereTestClient.connect(server.port(), TOKEN)) {
            browser.send("run it");
            browser.answerApproval(false, WAIT);
            browser.awaitComplete(WAIT);

            assertThat(toolResult(backend.requests().get(1)), containsString("cancelled"));
        }
    }

    @Test
    void stopEndsARunningRequestWithoutWaitingForIt() throws Exception {
        ScriptedBackend backend = new ScriptedBackend((call, request) -> {
            if (call == 1) {
                try {
                    Thread.sleep(8_000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            return ScriptedBackend.textTurn("answer " + call);
        });
        WebServer server = start(backend);

        try (AtmosphereTestClient browser = AtmosphereTestClient.connect(server.port(), TOKEN)) {
            long started = System.nanoTime();
            browser.send("take your time");
            Thread.sleep(500);
            browser.send(WebAgentEndpoint.STOP);
            browser.awaitMessage(message -> message.contains("(stopped)"), WAIT);
            // both requests are over: the stop itself, and the one it stopped
            browser.awaitCompletes(2, WAIT);

            assertThat(Duration.ofNanos(System.nanoTime() - started).toMillis(), lessThan(6_000L));
        }
    }
}
