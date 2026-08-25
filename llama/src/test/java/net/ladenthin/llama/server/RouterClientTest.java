// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.value.RouterModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Model-free verification of RouterClient against a stub HTTP server speaking "
                + "the upstream router wire format: list/find parsing, load/unload request "
                + "bodies, error surfacing with the router's error body, and the "
                + "awaitModelLoaded state machine (poll-until-loaded, fail-fast on failed "
                + "worker or unknown model, timeout), and the optional API-key bearer header.")
public class RouterClientTest {

    private HttpServer server;
    private RouterClient client;

    /** Body served for GET /models; test cases swap it (or a supplier-driven sequence). */
    private final AtomicReference<String> modelsBody = new AtomicReference<>("{\"data\":[],\"object\":\"list\"}");

    /** Counts GET /models calls so await tests can serve a status sequence. */
    private final AtomicInteger modelsCalls = new AtomicInteger();

    /** Authorization header seen by the stub on the most recent request; "" when absent. */
    private final AtomicReference<String> lastAuthHeader = new AtomicReference<>("");

    private final AtomicReference<String> lastLoadBody = new AtomicReference<>("");
    private final AtomicReference<String> lastUnloadBody = new AtomicReference<>("");

    @BeforeEach
    public void startStub() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/models", exchange -> {
            String auth = exchange.getRequestHeaders().getFirst("Authorization");
            lastAuthHeader.set(auth == null ? "" : auth);
            String path = exchange.getRequestURI().getPath();
            if ("/models/load".equals(path)) {
                lastLoadBody.set(readBody(exchange));
                respond(exchange, 200, "{\"success\":true}");
            } else if ("/models/unload".equals(path)) {
                lastUnloadBody.set(readBody(exchange));
                respond(exchange, 200, "{\"success\":true}");
            } else {
                modelsCalls.incrementAndGet();
                respond(exchange, 200, modelsBody.get());
            }
        });
        server.start();
        client = new RouterClient("127.0.0.1", server.getAddress().getPort());
    }

    @AfterEach
    public void stopStub() {
        if (server != null) {
            server.stop(0);
        }
    }

    private static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int read;
            while ((read = in.read(chunk)) != -1) {
                buffer.write(chunk, 0, read);
            }
            return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private static void respond(HttpExchange exchange, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
        exchange.close();
    }

    private static String entry(String id, String status) {
        return "{\"id\":\"" + id + "\",\"status\":{\"value\":\"" + status + "\"}}";
    }

    @Test
    public void listModels_parsesRouterResponse() throws IOException {
        modelsBody.set(
                "{\"object\":\"list\",\"data\":[" + entry("qwen", "loaded") + "," + entry("llama", "unloaded") + "]}");

        List<RouterModel> models = client.listModels();

        assertThat(models.size(), is(2));
        assertThat(models.get(0).getId(), is("qwen"));
        assertThat(models.get(0).getStatus(), is(RouterModel.Status.LOADED));
        assertThat(models.get(1).getStatus(), is(RouterModel.Status.UNLOADED));
    }

    @Test
    public void findModel_matchesById() throws IOException {
        modelsBody.set("{\"data\":[" + entry("qwen", "loading") + "]}");

        assertThat(client.findModel("qwen").isPresent(), is(true));
        assertThat(client.findModel("qwen").get().getStatus(), is(RouterModel.Status.LOADING));
        assertThat(client.findModel("absent").isPresent(), is(false));
    }

    @Test
    public void loadModel_postsModelFieldToLoadEndpoint() throws IOException {
        client.loadModel("qwen");

        assertThat(lastLoadBody.get(), is("{\"model\":\"qwen\"}"));
    }

    @Test
    public void unloadModel_postsModelFieldToUnloadEndpoint() throws IOException {
        client.unloadModel("qwen");

        assertThat(lastUnloadBody.get(), is("{\"model\":\"qwen\"}"));
    }

    @Test
    public void non2xxResponseSurfacesRouterErrorBody() throws IOException {
        // Dedicated stub that always rejects, mirroring upstream's error JSON shape.
        HttpServer errorServer = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        errorServer.createContext(
                "/models",
                exchange -> respond(exchange, 400, "{\"error\":{\"message\":\"model is already running\"}}"));
        errorServer.start();
        try {
            RouterClient errorClient =
                    new RouterClient("127.0.0.1", errorServer.getAddress().getPort());

            IOException thrown = assertThrows(IOException.class, () -> errorClient.loadModel("qwen"));

            assertThat(thrown.getMessage(), containsString("HTTP 400"));
            assertThat(thrown.getMessage(), containsString("model is already running"));
        } finally {
            errorServer.stop(0);
        }
    }

    @Test
    public void awaitModelLoaded_pollsUntilLoaded() throws Exception {
        modelsCalls.set(0);
        // Serve "loading" for the first two polls, then "loaded" — via a body that depends on
        // the call counter (swapped inside the handler through modelsBody on each poll).
        modelsBody.set("{\"data\":[" + entry("qwen", "loading") + "]}");
        Thread flipper = new Thread(() -> {
            try {
                while (modelsCalls.get() < 2) {
                    Thread.sleep(20L);
                }
                modelsBody.set("{\"data\":[" + entry("qwen", "loaded") + "]}");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        flipper.start();

        RouterModel loaded = client.awaitModelLoaded("qwen", 30_000L);

        flipper.join(5_000L);
        assertThat(loaded.getStatus(), is(RouterModel.Status.LOADED));
    }

    @Test
    public void awaitModelLoaded_failsFastOnFailedWorker() {
        modelsBody.set("{\"data\":[{\"id\":\"qwen\","
                + "\"status\":{\"value\":\"unloaded\",\"failed\":true,\"exit_code\":137}}]}");

        IllegalStateException thrown =
                assertThrows(IllegalStateException.class, () -> client.awaitModelLoaded("qwen", 30_000L));

        assertThat(thrown.getMessage(), containsString("exit code 137"));
    }

    @Test
    public void awaitModelLoaded_failsFastOnUnknownModel() {
        modelsBody.set("{\"data\":[" + entry("other", "loaded") + "]}");

        IllegalStateException thrown =
                assertThrows(IllegalStateException.class, () -> client.awaitModelLoaded("qwen", 30_000L));

        assertThat(thrown.getMessage(), containsString("does not list model 'qwen'"));
    }

    @Test
    public void awaitModelLoaded_timesOutWithLastStatusInMessage() {
        modelsBody.set("{\"data\":[" + entry("qwen", "loading") + "]}");

        IllegalStateException thrown =
                assertThrows(IllegalStateException.class, () -> client.awaitModelLoaded("qwen", 600L));

        assertThat(thrown.getMessage(), containsString("did not reach LOADED"));
        assertThat(thrown.getMessage(), containsString("loading"));
    }

    @Test
    public void unlistedModelMessageNamesTheHiddenCacheModelCause() {
        // A model hidden by a preset with dedup-cache-models (upstream #27346, b10507) never shows
        // up in GET /models even though it loads and serves, so the message must not send the
        // caller off to double-check --models-dir alone.
        modelsBody.set("{\"data\":[]}");

        IllegalStateException thrown =
                assertThrows(IllegalStateException.class, () -> client.awaitModelLoaded("qwen", 30_000L));

        assertThat(thrown.getMessage(), containsString("dedup-cache-models"));
        assertThat(thrown.getMessage(), containsString("--models-dir"));
    }

    @Test
    public void noAuthorizationHeaderWithoutAnApiKey() throws IOException {
        client.listModels();
        assertThat(lastAuthHeader.get(), is(""));
    }

    @Test
    public void apiKeyIsSentAsABearerTokenOnReads() throws IOException {
        RouterClient authed = new RouterClient("127.0.0.1", server.getAddress().getPort(), "secret");
        authed.listModels();
        assertThat(lastAuthHeader.get(), is("Bearer secret"));
    }

    @Test
    public void apiKeyIsSentAsABearerTokenOnWrites() throws IOException {
        RouterClient authed = new RouterClient(server.getAddress().getPort(), "secret");
        authed.loadModel("qwen");
        assertThat(lastAuthHeader.get(), is("Bearer secret"));
        authed.unloadModel("qwen");
        assertThat(lastAuthHeader.get(), is("Bearer secret"));
    }

    @Test
    public void anEmptyApiKeySendsNoHeader() throws IOException {
        // Distinguishes "" from a real credential: an empty key must behave exactly like none,
        // not send "Bearer ".
        new RouterClient("127.0.0.1", server.getAddress().getPort(), "").listModels();
        assertThat(lastAuthHeader.get(), is(""));
    }

    @Test
    public void clientsDifferingOnlyInApiKeyAreNotEqual() {
        int port = server.getAddress().getPort();
        assertThat(
                new RouterClient("127.0.0.1", port, "a").equals(new RouterClient("127.0.0.1", port, "b")), is(false));
        assertThat(new RouterClient("127.0.0.1", port, "a").equals(new RouterClient("127.0.0.1", port, "a")), is(true));
        assertThat(new RouterClient(port).equals(new RouterClient("127.0.0.1", port)), is(true));
    }

    @Test
    public void toStringNeverLeaksTheApiKey() {
        assertThat(
                new RouterClient("127.0.0.1", 8080, "super-secret").toString(),
                is("RouterClient(http://127.0.0.1:8080)"));
    }
}
