// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * End-to-end HTTP tests for {@link OpenAiCompatServer} driven over a real socket with a
 * {@link FakeBackend} — no native library and no model are loaded. Exercises routing, authentication,
 * the non-streaming and Server-Sent-Events paths, heartbeats, the completions/embeddings/health routes,
 * and error statuses.
 *
 * <p>HTTP request plumbing is inherited from {@link OpenAiServerTestSupport}.
 */
public class OpenAiCompatServerHttpTest extends OpenAiServerTestSupport {

    private static final String CHAT_BODY = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

    private static OpenAiServerConfig config() {
        return OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .modelId("test-model")
                .build();
    }

    @Test
    public void nonStreamingReturnsTheCompletionBody() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("chat.completion"));
            assertThat(response.body, containsString("hello"));
        }
    }

    @Test
    public void streamingReturnsSseChunksThenDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("data: "));
            assertThat(response.body, containsString("chat.completion.chunk"));
            assertThat(response.body, containsString("data: [DONE]"));
        }
    }

    @Test
    public void streamingCompletionsReturnsTextCompletionChunksThenDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"stream\":true,\"prompt\":\"hi\"}";
            Response response = post(server.getPort(), "/v1/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("data: "));
            assertThat(response.body, containsString("text_completion"));
            assertThat(response.body, containsString("data: [DONE]"));
        }
    }

    @Test
    public void streamingEmitsHeartbeatsDuringAGap() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .heartbeatMillis(50L)
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new SlowFakeBackend(), cfg).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString(":")); // SSE comment heartbeat
            assertThat(response.body, containsString("data: [DONE]"));
        }
    }

    @Test
    public void completionsRouteReturnsTextCompletionBody() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/completions", "{\"prompt\":\"hi\"}", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("text_completion"));
        }
    }

    @Test
    public void embeddingsRouteReturnsEmbeddingList() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/embeddings", "{\"input\":\"hi\"}", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("embedding"));
        }
    }

    @Test
    public void metricsAndSlotsExposeCacheCounters() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response metrics = get(server.getPort(), "/metrics", "");
            assertThat(metrics.code, is(200));
            assertThat(metrics.body, containsString("n_prompt_tokens_cache"));
            Response slots = get(server.getPort(), "/slots", "");
            assertThat(slots.code, is(200));
            assertThat(slots.body, containsString("\"id\":0"));
        }
    }

    @Test
    public void slotsAnswersAnEmptyArrayWhenTheBackendReportsNoSlotsKey() throws IOException {
        // Regression guard: GET /slots used to hand MissingNode.toString() straight to the client,
        // i.e. HTTP 200 with a zero-length body, whenever the metrics payload carried no "slots"
        // key — which is exactly what the native layer produced between llama.cpp b10408 and the
        // JNI merge that restored the object. The route must always answer with a JSON array.
        OpenAiBackend noSlots = new FakeBackend() {
            @Override
            public String metrics() {
                return "{\"idle\":1}";
            }
        };
        try (OpenAiCompatServer server = new OpenAiCompatServer(noSlots, config()).start()) {
            Response slots = get(server.getPort(), "/slots", "");
            assertThat(slots.code, is(200));
            assertThat(slots.body, is("[]"));
        }
    }

    @Test
    public void oversizedRequestBodyRejectedWith413() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .maxRequestBodyBytes(32)
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            String body =
                    "{\"messages\":[{\"role\":\"user\",\"content\":\"this body is well over thirty-two bytes\"}]}";
            assertThat(post(server.getPort(), "/v1/chat/completions", body, "").code, is(413));
        }
    }

    @Test
    public void infillRouteReturnsContent() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"input_prefix\":\"def add(a,b):\\n    return \",\"input_suffix\":\"\"}";
            Response response = post(server.getPort(), "/infill", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("content"));
        }
    }

    @Test
    public void getOnInfillReturns405() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/infill", "");
            assertThat(response.code, is(405));
        }
    }

    @Test
    public void barePathAliasesResolveToTheSameHandlers() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            int port = server.getPort();
            // Clients disagree on the /v1 prefix; the bare aliases must reach the same handlers.
            assertThat(get(port, "/models", "").code, is(200));
            assertThat(post(port, "/completions", "{\"prompt\":\"hi\"}", "").code, is(200));
            assertThat(post(port, "/embeddings", "{\"input\":\"hi\"}", "").code, is(200));
            assertThat(post(port, "/chat/completions", CHAT_BODY, "").code, is(200));
        }
    }

    @Test
    public void rerankRouteReturnsResults() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"query\":\"q\",\"documents\":[\"a\",\"b\"]}";
            Response response = post(server.getPort(), "/v1/rerank", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("relevance_score"));
        }
    }

    @Test
    public void ollamaVersionTagsAndShowRespond() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            int port = server.getPort();
            Response version = get(port, "/api/version", "");
            assertThat(version.code, is(200));
            assertThat(version.body, containsString("version"));
            Response tags = get(port, "/api/tags", "");
            assertThat(tags.code, is(200));
            assertThat(tags.body, containsString("test-model"));
            Response show = post(port, "/api/show", "{\"model\":\"test-model\"}", "");
            assertThat(show.code, is(200));
            assertThat(show.body, containsString("capabilities"));
        }
    }

    @Test
    public void ollamaChatNonStreamingReturnsDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":false,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/api/chat", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("hello")); // from FakeBackend.complete
            assertThat(response.body, containsString("\"done\":true"));
        }
    }

    @Test
    public void ollamaChatStreamingReturnsNdjsonEndingWithDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            // stream defaults to true for Ollama.
            String body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/api/chat", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("llo")); // streamed content delta
            assertThat(response.body, containsString("\"done\":true"));
        }
    }

    @Test
    public void anthropicMessagesNonStreamingReturnsMessage() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"max_tokens\":16,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/messages", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"type\":\"message\""));
            assertThat(response.body, containsString("hello")); // FakeBackend.complete text
            assertThat(response.body, containsString("\"cache_read_input_tokens\":8"));
        }
    }

    @Test
    public void anthropicMessagesStreamingEmitsEventSequence() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"max_tokens\":16,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/messages", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("event: message_start"));
            assertThat(response.body, containsString("event: content_block_delta"));
            assertThat(response.body, containsString("event: message_stop"));
            assertThat(response.body, containsString("\"cache_read_input_tokens\":8"));
        }
    }

    @Test
    public void responsesNonStreamingReturnsResponseObject() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"input\":\"hi\"}";
            Response response = post(server.getPort(), "/v1/responses", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"object\":\"response\""));
            assertThat(response.body, containsString("output_text"));
            assertThat(response.body, containsString("hello"));
            assertThat(response.body, containsString("\"cached_tokens\":8"));
        }
    }

    @Test
    public void responsesStreamingEmitsEventSequence() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"input\":\"hi\"}";
            Response response = post(server.getPort(), "/v1/responses", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("event: response.created"));
            assertThat(response.body, containsString("event: response.output_text.delta"));
            assertThat(response.body, containsString("event: response.completed"));
            assertThat(response.body, containsString("\"cached_tokens\":8"));
        }
    }

    @Test
    public void propsEndpointReportsContextLength() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/props", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("n_ctx"));
            assertThat(response.body, containsString("modalities"));
        }
    }

    @Test
    public void ollamaGenerateNonStreamingReturnsResponse() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"stream\":false}";
            Response response = post(server.getPort(), "/api/generate", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"response\":\"hello\"")); // FakeBackend.completions text
            assertThat(response.body, containsString("\"done\":true"));
        }
    }

    @Test
    public void ollamaGenerateWithSuffixUsesInfill() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"def f():\",\"suffix\":\"\",\"stream\":false}";
            Response response = post(server.getPort(), "/api/generate", body, "");
            assertThat(response.code, is(200));
            // FakeBackend.infill returns content " world".
            assertThat(response.body, containsString("world"));
        }
    }

    @Test
    public void ollamaGenerateStreamingReturnsNdjsonDone() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"hi\",\"stream\":true}";
            Response response = post(server.getPort(), "/api/generate", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"done\":true"));
        }
    }

    @Test
    public void healthEndpointReturnsOk() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/health", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"status\":\"ok\""));
        }
    }

    @Test
    public void modelsEndpointAdvertisesConfiguredModel() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/models", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("test-model"));
        }
    }

    @Test
    public void unknownPathReturns404() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/does-not-exist", "");
            assertThat(response.code, is(404));
        }
    }

    @Test
    public void missingMessagesReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", "{}", "");
            assertThat(response.code, is(400));
        }
    }

    @Test
    public void malformedJsonReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", "not json", "");
            assertThat(response.code, is(400));
        }
    }

    @Test
    public void getOnChatCompletionsReturns405() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/chat/completions", "");
            assertThat(response.code, is(405));
        }
    }

    @Test
    public void getOnEmbeddingsReturns405() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = get(server.getPort(), "/v1/embeddings", "");
            assertThat(response.code, is(405));
        }
    }

    @Test
    public void getOnNewPostRoutesReturns405() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            int port = server.getPort();
            // The protocol-shim POST routes go through the shared requirePostJson preamble.
            assertThat(get(port, "/v1/rerank", "").code, is(405));
            assertThat(get(port, "/v1/messages", "").code, is(405));
            assertThat(get(port, "/v1/responses", "").code, is(405));
            assertThat(get(port, "/api/chat", "").code, is(405));
            assertThat(get(port, "/api/generate", "").code, is(405));
        }
    }

    @Test
    public void optionsPreflightReturns204WithCorsHeaders() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = options(server.getPort(), "/v1/chat/completions");
            assertThat(response.code, is(204));
            assertThat(response.corsAllowOrigin, is("*"));
        }
    }

    @Test
    public void normalResponsesCarryCorsAllowOrigin() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(response.code, is(200));
            assertThat(response.corsAllowOrigin, is("*"));
        }
    }

    @Test
    public void authRequiredWhenApiKeyConfigured() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("secret")
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            int port = server.getPort();
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "").code, is(401));
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "Bearer wrong").code, is(401));
            assertThat(post(port, "/v1/chat/completions", CHAT_BODY, "Bearer secret").code, is(200));
        }
    }

    @Test
    public void healthEndpointIsUnauthenticated() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("secret")
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            // No Authorization header, yet /health must still answer 200 for orchestrator probes.
            assertThat(get(server.getPort(), "/health", "").code, is(200));
        }
    }

    @Test
    public void metricsAndSlotsRequireApiKeyWhenConfigured() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("secret")
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            int port = server.getPort();
            // /metrics and /slots expose slot state and token counters, so they must be gated.
            assertThat(get(port, "/metrics", "").code, is(401));
            assertThat(get(port, "/metrics", "Bearer wrong").code, is(401));
            assertThat(get(port, "/metrics", "Bearer secret").code, is(200));
            assertThat(get(port, "/slots", "").code, is(401));
            assertThat(get(port, "/slots", "Bearer wrong").code, is(401));
            assertThat(get(port, "/slots", "Bearer secret").code, is(200));
        }
    }

    /** Deterministic backend that returns canned OpenAI shapes for every operation. */
    static class FakeBackend implements OpenAiBackend {
        @Override
        public String metrics() {
            return "{\"idle\":1,\"slots\":[{\"id\":0,\"n_prompt_tokens_cache\":8}]}";
        }

        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
                    + "\"message\":{\"role\":\"assistant\",\"content\":\"hello\"}}],"
                    + "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,"
                    + "\"prompt_tokens_details\":{\"cached_tokens\":8}}}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
            sink.accept("{\"object\":\"chat.completion.chunk\","
                    + "\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}]}");
            if (request.path("stream_options").path("include_usage").asBoolean(false)) {
                sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[],"
                        + "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,"
                        + "\"prompt_tokens_details\":{\"cached_tokens\":8}}}");
            }
        }

        @Override
        public String completions(JsonNode request) {
            return "{\"object\":\"text_completion\",\"choices\":[{\"text\":\"hello\"}]}";
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"text_completion\",\"choices\":[{\"text\":\"he\",\"finish_reason\":null}]}");
            sink.accept("{\"object\":\"text_completion\",\"choices\":[{\"text\":\"llo\",\"finish_reason\":\"stop\"}]}");
        }

        @Override
        public String embeddings(JsonNode request) {
            return "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"embedding\":[0.1,0.2]}]}";
        }

        @Override
        public String infill(JsonNode request) {
            return "{\"content\":\" world\",\"stop\":true}";
        }

        @Override
        public String rerank(JsonNode request) {
            return "{\"object\":\"list\",\"results\":[{\"index\":0,\"relevance_score\":0.9}],"
                    + "\"data\":[{\"index\":0,\"relevance_score\":0.9}]}";
        }
    }

    /** Backend that stalls before emitting, so the server's heartbeat fires during the gap. */
    static final class SlowFakeBackend implements OpenAiBackend {
        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[]}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            try {
                Thread.sleep(300L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            sink.accept("{\"object\":\"chat.completion.chunk\","
                    + "\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}");
        }

        @Override
        public String completions(JsonNode request) {
            return "{\"object\":\"text_completion\",\"choices\":[]}";
        }

        @Override
        public String embeddings(JsonNode request) {
            return "{\"object\":\"list\",\"data\":[]}";
        }

        @Override
        public String infill(JsonNode request) {
            return "{\"content\":\"\"}";
        }

        @Override
        public String rerank(JsonNode request) {
            return "{\"object\":\"list\",\"results\":[],\"data\":[]}";
        }
    }
}
