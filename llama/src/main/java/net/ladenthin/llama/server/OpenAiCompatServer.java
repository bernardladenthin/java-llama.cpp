// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.LlamaModel;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An OpenAI-compatible HTTP endpoint over a loaded {@link LlamaModel}, built only on the JDK's
 * {@code com.sun.net.httpserver.HttpServer} (no new runtime dependency). It is both embeddable and the
 * {@code Main-Class} of the {@code -jar-with-dependencies} assembly.
 *
 * <p>Routes:
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — streaming (Server-Sent Events) and non-streaming chat
 *       completions, forwarded faithfully (messages/tools verbatim; streamed {@code delta.tool_calls}
 *       preserved).</li>
 *   <li>{@code POST /v1/completions} — text completion, streaming (Server-Sent Events, token by token
 *       via {@code generate(...)}) when {@code stream:true} and non-streaming otherwise.</li>
 *   <li>{@code POST /v1/embeddings} — embeddings (requires the model to be loaded in embedding
 *       mode).</li>
 *   <li>{@code GET /v1/models} — advertises the single configured model.</li>
 *   <li>{@code GET /metrics} — server and per-slot token/cache counters as JSON.</li>
 *   <li>{@code GET /slots} — the per-slot metrics array as JSON.</li>
 *   <li>{@code GET /health} — liveness probe returning {@code {"status":"ok"}} (no authentication).</li>
 * </ul>
 *
 * <p>During streaming, the server emits SSE comment heartbeats on a timer so a long prompt prefill on
 * CPU does not trip a client's stream-inactivity timeout before the first token. It binds to loopback by
 * default and can require a bearer API key. The endpoint is a pass-through: tools are provided and
 * executed by the client, not here.
 *
 * <p>Typical use:
 * <pre>{@code
 * try (LlamaModel model = new LlamaModel(new ModelParameters().setModel("models/model.gguf"));
 *      OpenAiCompatServer server = new OpenAiCompatServer(
 *              model, OpenAiServerConfig.builder().port(8080).modelId("local").build()).start()) {
 *     Thread.currentThread().join();
 * }
 * }</pre>
 */
public final class OpenAiCompatServer implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(OpenAiCompatServer.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /** The chat-completions route. */
    public static final String PATH_CHAT_COMPLETIONS = "/v1/chat/completions";

    /** The text-completions route. */
    public static final String PATH_COMPLETIONS = "/v1/completions";

    /** The embeddings route. */
    public static final String PATH_EMBEDDINGS = "/v1/embeddings";

    /** The rerank route (requires the model loaded in reranking mode). */
    public static final String PATH_RERANK = "/v1/rerank";

    /** The Anthropic Messages API route. */
    public static final String PATH_MESSAGES = "/v1/messages";

    /** The OpenAI Responses API route. */
    public static final String PATH_RESPONSES = "/v1/responses";

    /**
     * The fill-in-the-middle (autocomplete) route. Deliberately the llama.cpp-native bare path (no
     * {@code /v1}) so ghost-text clients such as llama.vscode and Tabby reach it unchanged.
     */
    public static final String PATH_INFILL = "/infill";

    /** The model-list route. */
    public static final String PATH_MODELS = "/v1/models";

    /** The liveness-probe route. */
    public static final String PATH_HEALTH = "/health";

    /** The llama.cpp-native server-properties route (context length + modalities). */
    public static final String PATH_PROPS = "/props";

    /** llama.cpp server metrics as JSON, including per-slot token/cache counters. */
    public static final String PATH_METRICS = "/metrics";

    /** llama.cpp slot state array as JSON. */
    public static final String PATH_SLOTS = "/slots";

    /** Ollama-native discovery route (version). */
    public static final String PATH_OLLAMA_VERSION = "/api/version";

    /** Ollama-native discovery route (model list). */
    public static final String PATH_OLLAMA_TAGS = "/api/tags";

    /** Ollama-native discovery route (model capabilities). */
    public static final String PATH_OLLAMA_SHOW = "/api/show";

    /** Ollama-native chat route. */
    public static final String PATH_OLLAMA_CHAT = "/api/chat";

    /** Ollama-native generate route (prompt completion / fill-in-the-middle). */
    public static final String PATH_OLLAMA_GENERATE = "/api/generate";

    private static final String CONTENT_TYPE_NDJSON = "application/x-ndjson";

    private static final int HTTP_OK = 200;
    private static final int HTTP_BAD_REQUEST = 400;
    private static final int HTTP_UNAUTHORIZED = 401;
    private static final int HTTP_NOT_FOUND = 404;
    private static final int HTTP_METHOD_NOT_ALLOWED = 405;
    private static final int HTTP_SERVER_ERROR = 500;
    private static final int HTTP_PAYLOAD_TOO_LARGE = 413;

    private static final String CONTENT_TYPE_JSON = "application/json; charset=utf-8";
    private static final String CONTENT_TYPE_SSE = "text/event-stream; charset=utf-8";
    private static final String BEARER_PREFIX = "Bearer ";
    private static final String ERROR_TYPE_REQUEST = "invalid_request_error";
    private static final String ERROR_TYPE_SERVER = "server_error";
    private static final String HEALTH_BODY = "{\"status\":\"ok\"}";

    private final OpenAiServerConfig config;
    private final OpenAiBackend backend;
    private final HttpServer http;
    private final Filter corsFilter;
    private final ExecutorService requestExecutor;
    private final ScheduledExecutorService heartbeatExecutor;

    /**
     * Create a server backed by a loaded model.
     *
     * @param model the model to serve completions from (owned by the caller; not closed by the server)
     * @param config the server configuration
     * @throws IOException if the listening socket cannot be bound
     */
    public OpenAiCompatServer(LlamaModel model, OpenAiServerConfig config) throws IOException {
        this(new LlamaModelBackend(model, new OpenAiRequestMapper()), config);
    }

    /**
     * Create a server backed by an arbitrary {@link OpenAiBackend}. Used by tests to drive the full HTTP
     * surface without a native library or model.
     *
     * @param backend the inference engine seam
     * @param config the server configuration
     * @throws IOException if the listening socket cannot be bound
     */
    OpenAiCompatServer(OpenAiBackend backend, OpenAiServerConfig config) throws IOException {
        this.config = config;
        this.backend = backend;
        this.requestExecutor = Executors.newCachedThreadPool(namedFactory("jllama-openai-http"));
        // Sized for concurrency: a heartbeat write blocks on a slow/stalled client, so a single
        // shared thread would let one such client starve every other stream's keep-alives. Bound
        // the impact to roughly one stalled client per thread.
        this.heartbeatExecutor = Executors.newScheduledThreadPool(
                Math.max(2, Runtime.getRuntime().availableProcessors()), namedFactory("jllama-openai-hb"));
        this.http = HttpServer.create(new InetSocketAddress(config.getHost(), config.getPort()), 0);
        this.corsFilter = buildCorsFilter(config.getCorsAllowOrigin());
        register("/", this::handleNotFound);
        register(PATH_HEALTH, this::handleHealth);
        register(PATH_PROPS, this::handleProps);
        register(PATH_METRICS, this::handleMetrics);
        register(PATH_SLOTS, this::handleSlots);
        // Each route is registered under its canonical path and a bare alias (clients disagree on
        // whether to include the /v1 prefix), so both forms resolve to the same handler.
        register(PATH_MODELS, this::handleModels);
        register("/models", this::handleModels);
        register(PATH_CHAT_COMPLETIONS, this::handleChatCompletions);
        register("/chat/completions", this::handleChatCompletions);
        register(PATH_COMPLETIONS, this::handleCompletions);
        register("/completions", this::handleCompletions);
        register(PATH_EMBEDDINGS, this::handleEmbeddings);
        register("/embeddings", this::handleEmbeddings);
        register(PATH_RERANK, this::handleRerank);
        register("/rerank", this::handleRerank);
        register("/reranking", this::handleRerank);
        register(PATH_INFILL, this::handleInfill);
        register("/v1/infill", this::handleInfill);
        register(PATH_MESSAGES, this::handleAnthropicMessages);
        register("/messages", this::handleAnthropicMessages);
        register(PATH_RESPONSES, this::handleResponses);
        register("/responses", this::handleResponses);
        // Ollama-native surface (Copilot's built-in Ollama provider + Ollama-hardcoded tools).
        register(PATH_OLLAMA_VERSION, this::handleOllamaVersion);
        register(PATH_OLLAMA_TAGS, this::handleOllamaTags);
        register(PATH_OLLAMA_SHOW, this::handleOllamaShow);
        register(PATH_OLLAMA_CHAT, this::handleOllamaChat);
        register(PATH_OLLAMA_GENERATE, this::handleOllamaGenerate);
        http.setExecutor(requestExecutor);
    }

    /**
     * Start accepting connections.
     *
     * @return this server, for chaining
     */
    public OpenAiCompatServer start() {
        http.start();
        LOG.info("OpenAI-compatible server listening on http://{}:{}", config.getHost(), getPort());
        return this;
    }

    /**
     * The actual bound port (useful when configured with port {@code 0} for an ephemeral port).
     *
     * @return the port the server is listening on
     */
    public int getPort() {
        return http.getAddress().getPort();
    }

    /** Stop the server and release its thread pools. The backing model is not closed. */
    @Override
    public void close() {
        http.stop(0);
        requestExecutor.shutdownNow();
        heartbeatExecutor.shutdownNow();
    }

    /**
     * Register {@code handler} for {@code path} with the CORS filter attached. Centralised so the
     * cross-cutting CORS/preflight wiring applies uniformly to every route (including the catch-all).
     */
    private void register(String path, HttpHandler handler) {
        http.createContext(path, handler).getFilters().add(corsFilter);
    }

    /**
     * Build a CORS filter that stamps {@code Access-Control-Allow-Origin} on every response and answers
     * {@code OPTIONS} preflights with {@code 204} + the allowed methods/headers — so browser- and
     * webview-based clients (which preflight an {@code Authorization} header) are not blocked.
     */
    private static Filter buildCorsFilter(String allowOrigin) {
        return new Filter() {
            @Override
            public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
                exchange.getResponseHeaders().set("Access-Control-Allow-Origin", allowOrigin);
                if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                    exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                    exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, Authorization");
                    exchange.getResponseHeaders().set("Access-Control-Max-Age", "86400");
                    exchange.sendResponseHeaders(204, -1);
                    exchange.close();
                    return;
                }
                chain.doFilter(exchange);
            }

            @Override
            public String description() {
                return "CORS preflight + Access-Control-Allow-Origin";
            }
        };
    }

    // ----- handlers -----

    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            JsonNode messages = request.path("messages");
            if (!messages.isArray() || messages.size() == 0) {
                sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, "'messages' must be a non-empty array");
                return;
            }
            if (request.path("stream").asBoolean(false)) {
                streamChat(exchange, request);
            } else {
                completeNonStreaming(exchange, request, backend::complete);
            }
        } finally {
            exchange.close();
        }
    }

    private void handleCompletions(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                if (request.path("stream").asBoolean(false)) {
                    streamCompletions(exchange, request);
                } else {
                    completeNonStreaming(exchange, request, backend::completions);
                }
            }
        } finally {
            exchange.close();
        }
    }

    private void streamCompletions(HttpExchange exchange, JsonNode request) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_SSE);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        try (ResponseStream out = new ResponseStream(exchange.getResponseBody())) {
            ScheduledFuture<?> heartbeat = null;
            try {
                heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                        () -> out.writeQuietly(OpenAiSseFormatter.heartbeat()),
                        config.getHeartbeatMillis(),
                        config.getHeartbeatMillis(),
                        TimeUnit.MILLISECONDS);
                backend.streamCompletions(request, chunkJson -> out.writeStrict(OpenAiSseFormatter.sseData(chunkJson)));
                out.writeStrict(OpenAiSseFormatter.sseDone());
            } catch (IllegalArgumentException e) {
                out.writeQuietly(
                        OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_REQUEST, null)));
            } catch (IOException e) {
                LOG.debug("client disconnected during stream", e);
            } catch (RuntimeException e) {
                LOG.warn("streaming completion failed", e);
                out.writeQuietly(
                        OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_SERVER, null)));
            } finally {
                if (heartbeat != null) {
                    heartbeat.cancel(false);
                }
            }
        }
    }

    private void handleEmbeddings(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                completeNonStreaming(exchange, request, backend::embeddings);
            }
        } finally {
            exchange.close();
        }
    }

    private void handleInfill(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                completeNonStreaming(exchange, request, backend::infill);
            }
        } finally {
            exchange.close();
        }
    }

    private void handleRerank(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request != null) {
                completeNonStreaming(exchange, request, backend::rerank);
            }
        } finally {
            exchange.close();
        }
    }

    /**
     * Run a non-streaming request through {@code producer} and write its JSON body, translating an
     * {@link IllegalArgumentException} to {@code 400} and any other failure to {@code 500}.
     */
    private void completeNonStreaming(HttpExchange exchange, JsonNode request, BodyProducer producer)
            throws IOException {
        final String body;
        try {
            body = producer.produce(request);
        } catch (IllegalArgumentException e) {
            sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, message(e));
            return;
        } catch (IOException | RuntimeException e) {
            LOG.warn("request failed", e);
            sendError(exchange, HTTP_SERVER_ERROR, ERROR_TYPE_SERVER, message(e));
            return;
        }
        sendJson(exchange, HTTP_OK, body);
    }

    private void streamChat(HttpExchange exchange, JsonNode request) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_SSE);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        try (ResponseStream out = new ResponseStream(exchange.getResponseBody())) {
            ScheduledFuture<?> heartbeat = null;
            try {
                heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                        () -> out.writeQuietly(OpenAiSseFormatter.heartbeat()),
                        config.getHeartbeatMillis(),
                        config.getHeartbeatMillis(),
                        TimeUnit.MILLISECONDS);
                backend.stream(
                        request,
                        chunkJson -> out.writeStrict(
                                OpenAiSseFormatter.sseData(OpenAiSseFormatter.ensureUsageCachedTokens(chunkJson))));
                out.writeStrict(OpenAiSseFormatter.sseDone());
            } catch (IllegalArgumentException e) {
                out.writeQuietly(
                        OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_REQUEST, null)));
            } catch (IOException e) {
                LOG.debug("client disconnected during stream", e);
            } catch (RuntimeException e) {
                LOG.warn("streaming chat completion failed", e);
                out.writeQuietly(
                        OpenAiSseFormatter.sseData(OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_SERVER, null)));
            } finally {
                // try-with-resources closes the stream (under its lock) after the heartbeat is cancelled,
                // so the close never races a still-in-flight heartbeat write.
                if (heartbeat != null) {
                    heartbeat.cancel(false);
                }
            }
        }
    }

    // ----- Ollama-native surface -----

    private void handleOllamaVersion(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            sendJson(exchange, HTTP_OK, OllamaApiSupport.versionJson());
        } finally {
            exchange.close();
        }
    }

    private void handleOllamaTags(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            sendJson(exchange, HTTP_OK, OllamaApiSupport.tagsJson(config.getModelId()));
        } finally {
            exchange.close();
        }
    }

    private void handleOllamaShow(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only POST is supported");
                return;
            }
            // The request body (optionally {"model":...}) is ignored: this server serves one model.
            int contextLength = config.getMaxInputTokens() + config.getMaxOutputTokens();
            sendJson(
                    exchange,
                    HTTP_OK,
                    OllamaApiSupport.showJson(config.getModelId(), contextLength, config.isSupportsVision()));
        } finally {
            exchange.close();
        }
    }

    private void handleOllamaChat(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            JsonNode openAiRequest = OllamaApiSupport.toOpenAiChatRequest(request);
            String model = request.path("model").asText(config.getModelId());
            if (OllamaApiSupport.isStreaming(request)) {
                streamOllamaChat(exchange, openAiRequest, model);
            } else {
                final String body;
                try {
                    body = backend.complete(openAiRequest);
                } catch (IllegalArgumentException e) {
                    sendJson(exchange, HTTP_BAD_REQUEST, ollamaError(message(e)));
                    return;
                } catch (IOException | RuntimeException e) {
                    LOG.warn("ollama chat failed", e);
                    sendJson(exchange, HTTP_SERVER_ERROR, ollamaError(message(e)));
                    return;
                }
                sendJson(exchange, HTTP_OK, OllamaApiSupport.toOllamaChatResponse(body, model));
            }
        } finally {
            exchange.close();
        }
    }

    /** Stream an Ollama {@code /api/chat} response as newline-delimited JSON, ending with a done line. */
    private void streamOllamaChat(HttpExchange exchange, JsonNode openAiRequest, String model) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_NDJSON);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        final ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        try (ResponseStream out = new ResponseStream(exchange.getResponseBody())) {
            try {
                backend.stream(openAiRequest, chunkJson -> {
                    accumulator.accept(chunkJson);
                    String line = OllamaApiSupport.toOllamaContentLine(chunkJson, model);
                    if (line != null) {
                        out.writeStrict(line);
                    }
                });
                out.writeStrict(OllamaApiSupport.toOllamaDoneLine(model, accumulator));
            } catch (IllegalArgumentException e) {
                out.writeQuietly(ollamaError(message(e)) + "\n");
            } catch (IOException e) {
                LOG.debug("ollama client disconnected during stream", e);
            } catch (RuntimeException e) {
                LOG.warn("ollama streaming chat failed", e);
                out.writeQuietly(ollamaError(message(e)) + "\n");
            }
        }
    }

    private void handleOllamaGenerate(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            String model = request.path("model").asText(config.getModelId());
            // Generation runs to completion first (there is no streaming raw-completion path), then the
            // text is wrapped — as a single NDJSON content line + done line when stream is requested.
            final String text;
            try {
                if (OllamaApiSupport.hasSuffix(request)) {
                    text = OllamaApiSupport.extractInfillContent(
                            backend.infill(OllamaApiSupport.toInfillRequest(request)));
                } else {
                    text = OllamaApiSupport.extractCompletionText(
                            backend.completions(OllamaApiSupport.toOpenAiCompletionRequest(request)));
                }
            } catch (IllegalArgumentException e) {
                sendJson(exchange, HTTP_BAD_REQUEST, ollamaError(message(e)));
                return;
            } catch (IOException | RuntimeException e) {
                LOG.warn("ollama generate failed", e);
                sendJson(exchange, HTTP_SERVER_ERROR, ollamaError(message(e)));
                return;
            }
            if (OllamaApiSupport.isStreaming(request)) {
                byte[] bytes =
                        OllamaApiSupport.toOllamaGenerateStream(text, model).getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_NDJSON);
                exchange.sendResponseHeaders(HTTP_OK, bytes.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(bytes);
                }
            } else {
                sendJson(exchange, HTTP_OK, OllamaApiSupport.toOllamaGenerateResponse(text, model));
            }
        } finally {
            exchange.close();
        }
    }

    private static String ollamaError(String message) {
        return OBJECT_MAPPER.createObjectNode().put("error", message).toString();
    }

    // ----- Anthropic Messages API -----

    private void handleAnthropicMessages(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            JsonNode openAiRequest = AnthropicApiSupport.toOpenAiChatRequest(request);
            String model = request.path("model").asText(config.getModelId());
            if (AnthropicApiSupport.isStreaming(request)) {
                streamAnthropic(exchange, openAiRequest, model);
            } else {
                final String body;
                try {
                    body = backend.complete(openAiRequest);
                } catch (IllegalArgumentException e) {
                    sendJson(exchange, HTTP_BAD_REQUEST, anthropicError(message(e)));
                    return;
                } catch (IOException | RuntimeException e) {
                    LOG.warn("anthropic messages failed", e);
                    sendJson(exchange, HTTP_SERVER_ERROR, anthropicError(message(e)));
                    return;
                }
                sendJson(exchange, HTTP_OK, AnthropicApiSupport.toAnthropicResponse(body, model));
            }
        } finally {
            exchange.close();
        }
    }

    /** Stream an Anthropic {@code /v1/messages} response as the Anthropic SSE event sequence. */
    private void streamAnthropic(HttpExchange exchange, JsonNode openAiRequest, String model) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_SSE);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        final AnthropicStreamTranslator translator =
                new AnthropicStreamTranslator("msg_" + Long.toHexString(System.nanoTime()), model);
        try (ResponseStream out = new ResponseStream(exchange.getResponseBody())) {
            ScheduledFuture<?> heartbeat = null;
            try {
                heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                        () -> out.writeQuietly(OpenAiSseFormatter.heartbeat()),
                        config.getHeartbeatMillis(),
                        config.getHeartbeatMillis(),
                        TimeUnit.MILLISECONDS);
                out.writeStrict(translator.begin());
                backend.stream(withUsageChunk(openAiRequest), chunkJson -> {
                    String events = translator.onChunk(chunkJson);
                    if (!events.isEmpty()) {
                        out.writeStrict(events);
                    }
                });
                out.writeStrict(translator.end());
            } catch (IllegalArgumentException e) {
                out.writeQuietly(AnthropicApiSupport.sseEvent("error", anthropicError(message(e))));
            } catch (IOException e) {
                LOG.debug("anthropic client disconnected during stream", e);
            } catch (RuntimeException e) {
                LOG.warn("anthropic streaming failed", e);
                out.writeQuietly(AnthropicApiSupport.sseEvent("error", anthropicError(message(e))));
            } finally {
                if (heartbeat != null) {
                    heartbeat.cancel(false);
                }
            }
        }
    }

    /** Ensure protocol translators receive the native stream's trailing usage chunk. */
    private static JsonNode withUsageChunk(JsonNode request) {
        ObjectNode copy = request.deepCopy();
        JsonNode existing = copy.path("stream_options");
        ObjectNode streamOptions = existing.isObject() ? (ObjectNode) existing : copy.putObject("stream_options");
        streamOptions.put("include_usage", true);
        return copy;
    }

    private static String anthropicError(String message) {
        ObjectNode root = OBJECT_MAPPER.createObjectNode();
        root.put("type", "error");
        ObjectNode error = root.putObject("error");
        error.put("type", "invalid_request_error");
        error.put("message", message);
        return root.toString();
    }

    // ----- OpenAI Responses API -----

    private void handleResponses(HttpExchange exchange) throws IOException {
        try {
            JsonNode request = requirePostJson(exchange);
            if (request == null) {
                return;
            }
            JsonNode openAiRequest = ResponsesApiSupport.toOpenAiChatRequest(request);
            String model = request.path("model").asText(config.getModelId());
            String responseId = "resp_" + Long.toHexString(System.nanoTime());
            if (ResponsesApiSupport.isStreaming(request)) {
                streamResponses(exchange, openAiRequest, model, responseId);
            } else {
                final String body;
                try {
                    body = backend.complete(openAiRequest);
                } catch (IllegalArgumentException e) {
                    sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, message(e));
                    return;
                } catch (IOException | RuntimeException e) {
                    LOG.warn("responses failed", e);
                    sendError(exchange, HTTP_SERVER_ERROR, ERROR_TYPE_SERVER, message(e));
                    return;
                }
                sendJson(exchange, HTTP_OK, ResponsesApiSupport.toResponsesResponse(body, model, responseId));
            }
        } finally {
            exchange.close();
        }
    }

    /** Stream a Responses {@code /v1/responses} reply as the Responses SSE event sequence. */
    private void streamResponses(HttpExchange exchange, JsonNode openAiRequest, String model, String responseId)
            throws IOException {
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_SSE);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(HTTP_OK, 0);
        final ResponsesStreamTranslator translator = new ResponsesStreamTranslator(model, responseId);
        try (ResponseStream out = new ResponseStream(exchange.getResponseBody())) {
            ScheduledFuture<?> heartbeat = null;
            try {
                heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                        () -> out.writeQuietly(OpenAiSseFormatter.heartbeat()),
                        config.getHeartbeatMillis(),
                        config.getHeartbeatMillis(),
                        TimeUnit.MILLISECONDS);
                out.writeStrict(translator.begin());
                backend.stream(withUsageChunk(openAiRequest), chunkJson -> {
                    String events = translator.onChunk(chunkJson);
                    if (!events.isEmpty()) {
                        out.writeStrict(events);
                    }
                });
                out.writeStrict(translator.end());
            } catch (IllegalArgumentException e) {
                out.writeQuietly("event: error\ndata: "
                        + OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_REQUEST, null) + "\n\n");
            } catch (IOException e) {
                LOG.debug("responses client disconnected during stream", e);
            } catch (RuntimeException e) {
                LOG.warn("responses streaming failed", e);
                out.writeQuietly("event: error\ndata: "
                        + OpenAiSseFormatter.errorJson(message(e), ERROR_TYPE_SERVER, null) + "\n\n");
            } finally {
                if (heartbeat != null) {
                    heartbeat.cancel(false);
                }
            }
        }
    }

    private void handleModels(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            if (!authorized(exchange)) {
                sendError(exchange, HTTP_UNAUTHORIZED, ERROR_TYPE_REQUEST, "Missing or invalid API key");
                return;
            }
            sendJson(exchange, HTTP_OK, OpenAiSseFormatter.modelsJson(config.getModelId(), config.getModelFtype()));
        } finally {
            exchange.close();
        }
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        try {
            // Liveness probe: deliberately unauthenticated so orchestrators can poll it without a key.
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            sendJson(exchange, HTTP_OK, HEALTH_BODY);
        } finally {
            exchange.close();
        }
    }

    private void handleMetrics(HttpExchange exchange) throws IOException {
        handleMetricsView(exchange, false);
    }

    private void handleSlots(HttpExchange exchange) throws IOException {
        handleMetricsView(exchange, true);
    }

    private void handleMetricsView(HttpExchange exchange, boolean slotsOnly) throws IOException {
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            if (!authorized(exchange)) {
                sendError(exchange, HTTP_UNAUTHORIZED, ERROR_TYPE_REQUEST, "Missing or invalid API key");
                return;
            }
            String metrics = backend.metrics();
            if (slotsOnly) {
                // path() yields a MissingNode when "slots" is absent, and MissingNode.toString()
                // is the empty string — which would answer 200 with a zero-length body, i.e. not
                // JSON at all. Fall back to an empty array so the route always answers in shape.
                JsonNode slots = OBJECT_MAPPER.readTree(metrics).path("slots");
                metrics = slots.isArray() ? slots.toString() : "[]";
            }
            sendJson(exchange, HTTP_OK, metrics);
        } catch (IOException | RuntimeException e) {
            LOG.warn("metrics request failed", e);
            sendError(exchange, HTTP_SERVER_ERROR, ERROR_TYPE_SERVER, message(e));
        } finally {
            exchange.close();
        }
    }

    private void handleProps(HttpExchange exchange) throws IOException {
        // Deliberately unauthenticated, like the Ollama discovery routes (/api/version, /api/tags,
        // /api/show): autocomplete/agent clients read context length + capabilities here before they
        // have (or to discover whether they need) a key. It exposes only public model metadata —
        // no inference, no secrets — so it intentionally bypasses authorized().
        try {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only GET is supported");
                return;
            }
            int contextLength = config.getMaxInputTokens() + config.getMaxOutputTokens();
            sendJson(
                    exchange,
                    HTTP_OK,
                    OpenAiSseFormatter.propsJson(config.getModelId(), contextLength, config.isSupportsVision()));
        } finally {
            exchange.close();
        }
    }

    private void handleNotFound(HttpExchange exchange) throws IOException {
        try {
            sendError(exchange, HTTP_NOT_FOUND, ERROR_TYPE_REQUEST, "Not found: " + exchange.getRequestURI());
        } finally {
            exchange.close();
        }
    }

    // ----- helpers -----

    /**
     * Shared preamble for the {@code POST} JSON routes: enforce the method, authentication and a JSON
     * object body, sending the matching error and returning {@code null} when any precondition fails.
     */
    private @Nullable JsonNode requirePostJson(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendError(exchange, HTTP_METHOD_NOT_ALLOWED, ERROR_TYPE_REQUEST, "Only POST is supported");
            return null;
        }
        if (!authorized(exchange)) {
            sendError(exchange, HTTP_UNAUTHORIZED, ERROR_TYPE_REQUEST, "Missing or invalid API key");
            return null;
        }
        String contentLength = exchange.getRequestHeaders().getFirst("Content-Length");
        if (contentLength != null) {
            try {
                if (Long.parseLong(contentLength.trim()) > config.getMaxRequestBodyBytes()) {
                    sendError(exchange, HTTP_PAYLOAD_TOO_LARGE, ERROR_TYPE_REQUEST, "Request body too large");
                    return null;
                }
            } catch (NumberFormatException ignored) {
                // Unparseable header — the bounded read below still caps the body.
            }
        }
        JsonNode request;
        try {
            request = readBody(exchange);
        } catch (RequestBodyTooLargeException e) {
            sendError(exchange, HTTP_PAYLOAD_TOO_LARGE, ERROR_TYPE_REQUEST, "Request body too large");
            return null;
        }
        if (request == null || !request.isObject()) {
            sendError(exchange, HTTP_BAD_REQUEST, ERROR_TYPE_REQUEST, "Request body must be a JSON object");
            return null;
        }
        return request;
    }

    private boolean authorized(HttpExchange exchange) {
        if (!config.isAuthenticationEnabled()) {
            return true;
        }
        String expected = config.getApiKey();
        if (expected == null) {
            return true;
        }
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.startsWith(BEARER_PREFIX)) {
            return false;
        }
        // Constant-time comparison so response timing does not leak how many leading bytes
        // of the bearer token matched.
        byte[] expectedBytes = expected.getBytes(StandardCharsets.UTF_8);
        byte[] presentedBytes = header.substring(BEARER_PREFIX.length()).getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expectedBytes, presentedBytes);
    }

    private @Nullable JsonNode readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = new BoundedInputStream(exchange.getRequestBody(), config.getMaxRequestBodyBytes())) {
            return OBJECT_MAPPER.readTree(is);
        } catch (JsonProcessingException e) {
            LOG.debug("malformed request body", e);
            return null;
        }
    }

    /** Signals that the request body exceeded the configured cap. */
    private static final class RequestBodyTooLargeException extends IOException {
        private static final long serialVersionUID = 1L;

        RequestBodyTooLargeException(long limit) {
            super("Request body exceeds " + limit + " bytes");
        }
    }

    /** Wraps a stream so reading past {@code limit} bytes throws instead of buffering unboundedly. */
    private static final class BoundedInputStream extends FilterInputStream {
        private final long limit;
        private long count;

        BoundedInputStream(InputStream in, long limit) {
            super(in);
            this.limit = limit;
        }

        @Override
        public int read() throws IOException {
            int b = super.read();
            if (b != -1 && ++count > limit) {
                throw new RequestBodyTooLargeException(limit);
            }
            return b;
        }

        @Override
        public int read(byte[] buffer, int off, int len) throws IOException {
            int n = super.read(buffer, off, len);
            if (n > 0) {
                count += n;
                if (count > limit) {
                    throw new RequestBodyTooLargeException(limit);
                }
            }
            return n;
        }
    }

    private void sendJson(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", CONTENT_TYPE_JSON);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange exchange, int status, String type, String message) throws IOException {
        sendJson(exchange, status, OpenAiSseFormatter.errorJson(message, type, null));
    }

    /**
     * Per-request, thread-safe wrapper over a streaming HTTP response body. Every write and the close are
     * serialized on a {@code private final} lock, so the generation thread and the heartbeat-timer task
     * never write to (or close) the same stream concurrently. The lock is owned by this per-request
     * instance rather than shared, so independent concurrent streams never serialize against each other.
     * It is {@link AutoCloseable} so callers drive it with try-with-resources, which closes the stream
     * (under the lock) on every exit path.
     */
    private static final class ResponseStream implements AutoCloseable {

        private final OutputStream os;
        private final Object lock = new Object();

        ResponseStream(OutputStream os) {
            this.os = os;
        }

        /** Write under the lock, propagating failures so a streaming generation can be cancelled. */
        void writeStrict(String text) throws IOException {
            synchronized (lock) {
                os.write(text.getBytes(StandardCharsets.UTF_8));
                os.flush();
            }
        }

        /** Write under the lock, swallowing failures (used for heartbeats and best-effort events). */
        void writeQuietly(String text) {
            synchronized (lock) {
                try {
                    os.write(text.getBytes(StandardCharsets.UTF_8));
                    os.flush();
                } catch (IOException e) {
                    LOG.trace("stream write failed (client likely disconnected)", e);
                }
            }
        }

        @Override
        public void close() {
            synchronized (lock) {
                try {
                    os.close();
                } catch (IOException e) {
                    LOG.trace("stream close failed", e);
                }
            }
        }
    }

    private static String message(Throwable t) {
        String m = t.getMessage();
        return m != null ? m : t.getClass().getSimpleName();
    }

    private static ThreadFactory namedFactory(String prefix) {
        AtomicInteger counter = new AtomicInteger();
        return runnable -> {
            Thread thread = new Thread(runnable, prefix + "-" + counter.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
    }

    /** Produces a non-streaming response body from a parsed request; may fail with {@link IOException}. */
    @FunctionalInterface
    private interface BodyProducer {
        String produce(JsonNode request) throws IOException;
    }

    // ----- standalone launcher -----

    /**
     * Command-line launcher: load a GGUF model and serve it over the OpenAI-compatible endpoint. This is
     * the {@code Main-Class} of the {@code -jar-with-dependencies} assembly.
     *
     * <p>Parsing, validation and the option list live in {@link OpenAiServerCli}; run with
     * {@code --help} for the full usage text. No {@code System.exit} is used (the {@code noSystemExit}
     * architecture rule forbids it): a usage error prints to stderr and returns.
     *
     * @param args command-line options
     * @throws IOException if the listening socket cannot be bound
     */
    public static void main(String[] args) throws IOException {
        if (OpenAiServerCli.isHelpRequested(args)) {
            System.out.println(OpenAiServerCli.usage());
            return;
        }

        final OpenAiServerCli.Options options;
        try {
            options = OpenAiServerCli.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            return;
        }

        // The server runs on daemon threads, so the main thread blocks until the JVM is asked to
        // shut down (Ctrl-C / SIGTERM); the try-with-resources then closes the server and model.
        // Two latches keep that shutdown graceful and race-free: the hook signals stopRequested and
        // then waits on cleanedUp, so the JVM — which blocks until shutdown hooks return — does not
        // halt until the close has actually run.
        final CountDownLatch stopRequested = new CountDownLatch(1);
        final CountDownLatch cleanedUp = new CountDownLatch(1);
        Runtime.getRuntime()
                .addShutdownHook(new Thread(
                        () -> {
                            stopRequested.countDown();
                            try {
                                cleanedUp.await();
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                        },
                        "jllama-openai-shutdown"));

        try (LlamaModel model = new LlamaModel(options.toModelParameters())) {
            OpenAiServerConfig config = options.toServerConfig(model.supportsVision(), model.getModelFtype());
            try (OpenAiCompatServer server = new OpenAiCompatServer(model, config)) {
                server.start();
                printReady(config, server.getPort());
                try {
                    stopRequested.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        } finally {
            cleanedUp.countDown();
        }
    }

    private static void printReady(OpenAiServerConfig config, int port) {
        String url = "http://" + config.getHost() + ":" + port + PATH_CHAT_COMPLETIONS;
        System.out.println();
        System.out.println("OpenAI-compatible endpoint ready: " + url);
        System.out.println("Add this to VS Code's chatLanguageModels.json (Chat: Manage Language Models):");
        System.out.println('[');
        System.out.println("  {");
        System.out.println("    \"name\": \"Local llama.cpp (java-llama.cpp)\",");
        System.out.println("    \"vendor\": \"customendpoint\",");
        System.out.println(
                "    \"apiKey\": \"" + (config.isAuthenticationEnabled() ? "<your key>" : "local-dummy-key") + "\",");
        System.out.println("    \"apiType\": \"chat-completions\",");
        System.out.println("    \"models\": [");
        System.out.println("      {");
        System.out.println("        \"id\": \"" + config.getModelId() + "\",");
        System.out.println("        \"name\": \"" + config.getModelId() + "\",");
        System.out.println("        \"url\": \"" + url + "\",");
        System.out.println("        \"toolCalling\": true,");
        System.out.println("        \"vision\": false,");
        System.out.println("        \"maxInputTokens\": " + config.getMaxInputTokens() + ",");
        System.out.println("        \"maxOutputTokens\": " + config.getMaxOutputTokens());
        System.out.println("      }");
        System.out.println("    ]");
        System.out.println("  }");
        System.out.println(']');
    }
}
