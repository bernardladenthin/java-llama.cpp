// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
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

    private static final com.fasterxml.jackson.databind.ObjectMapper READER =
            new com.fasterxml.jackson.databind.ObjectMapper();

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
    public void ollamaChatStreamingDoneLineCarriesTheTokenCountsLikeTheBlockingAnswer() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            // The blocking /api/chat reports prompt_eval_count/eval_count; clients compute tokens/s from
            // the done line of the stream, which needs the native stream's usage chunk.
            String body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/api/chat", body, "");
            String[] lines = response.body.trim().split("\n");
            JsonNode done = new com.fasterxml.jackson.databind.ObjectMapper().readTree(lines[lines.length - 1]);
            assertThat(done.path("done").asBoolean(false), is(true));
            assertThat(done.path("prompt_eval_count").asInt(-1), is(12));
            assertThat(done.path("eval_count").asInt(-1), is(3));
        }
    }

    @Test
    public void ollamaChatDoneReasonIsLengthWhenCutOffAndStopForToolCalls() throws IOException {
        // Ollama reports "length" for an answer cut off by num_predict and "stop" otherwise, tool calls
        // included; the blocking path passed OpenAI's "tool_calls" through, the stream said "stop" always.
        String body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
        for (String[] c : new String[][] {{"length", "length"}, {"tool_calls", "stop"}, {"stop", "stop"}}) {
            try (OpenAiCompatServer server = new OpenAiCompatServer(new FinishReasonBackend(c[0]), config()).start()) {
                JsonNode blocking =
                        READER.readTree(post(server.getPort(), "/api/chat", body + ",\"stream\":false}", "").body);
                assertThat(c[0], blocking.path("done_reason").asText(), is(c[1]));
                String[] lines = post(server.getPort(), "/api/chat", body + "}", "")
                        .body
                        .trim()
                        .split("\n");
                JsonNode done = READER.readTree(lines[lines.length - 1]);
                assertThat(c[0], done.path("done").asBoolean(false), is(true));
                assertThat(c[0], done.path("done_reason").asText(), is(c[1]));
            }
        }
    }

    @Test
    public void ollamaGenerateDoneReasonAndCountsFollowTheCompletion() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FinishReasonBackend("length"), config()).start()) {
            // raw: the bare prompt goes to the completion endpoint
            String body = "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"raw\":true";
            JsonNode blocking =
                    READER.readTree(post(server.getPort(), "/api/generate", body + ",\"stream\":false}", "").body);
            assertThat(blocking.path("done_reason").asText(), is("length"));
            assertThat(blocking.path("prompt_eval_count").asInt(-1), is(5));
            assertThat(blocking.path("eval_count").asInt(-1), is(2));
            String[] lines = post(server.getPort(), "/api/generate", body + ",\"stream\":true}", "")
                    .body
                    .trim()
                    .split("\n");
            JsonNode done = READER.readTree(lines[lines.length - 1]);
            assertThat(done.path("done_reason").asText(), is("length"));
            assertThat(done.path("eval_count").asInt(-1), is(2));
        }
    }

    @Test
    public void ollamaGenerateWithSuffixReadsTheInfillStopType() throws IOException {
        // The native infill answer says why it stopped in stop_type: "limit" is Ollama's "length".
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FinishReasonBackend("length"), config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"def f():\",\"suffix\":\"\",\"stream\":false}";
            JsonNode blocking = READER.readTree(post(server.getPort(), "/api/generate", body, "").body);
            assertThat(blocking.path("done_reason").asText(), is("length"));
            assertThat(blocking.path("prompt_eval_count").asInt(-1), is(4));
            assertThat(blocking.path("eval_count").asInt(-1), is(1));
        }
    }

    @Test
    public void aSubPathOfARouteIsNotHandledByTheRoute() throws IOException {
        // The JDK server matches contexts by prefix, so /v1/messages/count_tokens (Claude Code counts
        // tokens there) ran a whole generation through the /v1/messages handler instead of a 404.
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            assertThat(post(server.getPort(), "/v1/messages/count_tokens", body, "").code, is(404));
        }
    }

    @Test
    public void upstreamsRerankingAliasStillAnswers() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"query\":\"q\",\"documents\":[\"a\"]}";
            assertThat(post(server.getPort(), "/v1/reranking", body, "").code, is(200));
            assertThat(post(server.getPort(), "/v1/chat/completions/", CHAT_BODY, "").code, is(200));
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
    public void responsesWithAnInputFileIsRejectedLikeUpstream() throws IOException {
        // Upstream answers an input_file part with 400 ("not supported"); dropping it silently let the
        // model answer as if the attached file had never been sent.
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"input\":[{\"type\":\"message\",\"role\":\"user\","
                    + "\"content\":[{\"type\":\"input_text\",\"text\":\"summarize\"},"
                    + "{\"type\":\"input_file\",\"file_id\":\"file-1\"}]}]}";
            Response response = post(server.getPort(), "/v1/responses", body, "");
            assertThat(response.code, is(400));
            assertThat(response.body, containsString("input_file"));
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
    public void propsEndpointReportsTheSlotCountOfTheConfiguration() throws IOException {
        // total_slots was a constant 1; upstream's /props reports n_parallel, and a client that
        // spreads its requests over the slots needs the real count.
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .modelId("test-model")
                .totalSlots(4)
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            Response response = get(server.getPort(), "/props", "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"total_slots\":4"));
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
    public void ollamaGenerateWithoutRawAppliesTheChatTemplateLikeOllama() throws IOException {
        // Ollama's /api/generate applies the model's chat template (with "system" as the system
        // message) unless "raw" is true; ours always completed the bare prompt, so an instruct model
        // continued the text instead of answering, and "system" was lost.
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"why?\",\"system\":\"be brief\","
                    + "\"images\":[\"AAAA\"],\"stream\":false}";
            JsonNode out = READER.readTree(post(server.getPort(), "/api/generate", body, "").body);

            assertThat(backend.called, is("complete"));
            JsonNode messages = backend.request.path("messages");
            assertThat(messages.size(), is(2));
            assertThat(messages.path(0).path("role").asText(), is("system"));
            assertThat(messages.path(0).path("content").asText(), is("be brief"));
            assertThat(messages.path(1).path("role").asText(), is("user"));
            assertThat(messages.path(1).path("content").path(0).path("text").asText(), is("why?"));
            assertThat(messages.path(1).path("content").path(1).path("type").asText(), is("image_url"));
            assertThat(out.path("response").asText(), is("because"));
            assertThat(out.path("thinking").asText(), is("hmm"));
            assertThat(out.path("eval_count").asInt(-1), is(3));
        }
    }

    @Test
    public void ollamaGenerateWithRawCompletesTheBarePrompt() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"raw\":true,\"stream\":false}";
            JsonNode out = READER.readTree(post(server.getPort(), "/api/generate", body, "").body);

            assertThat(backend.called, is("completions"));
            assertThat(backend.request.path("prompt").asText(), is("once upon"));
            assertThat(backend.request.has("messages"), is(false));
            assertThat(out.path("response").asText(), is("hello"));
        }
    }

    /**
     * Ollama refuses {@code "raw": true} next to a {@code system}, {@code template} or {@code context}
     * with a 400, before its load answer and before fill-in-the-middle. The server dropped them silently
     * and completed the bare prompt.
     */
    @Test
    public void ollamaGenerateWithRawAndASystemPromptIsRejectedLikeOllama() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String body : new String[] {
                "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"system\":\"x\",\"raw\":true,\"stream\":false}",
                "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"template\":\"t\",\"raw\":true}",
                "{\"model\":\"test-model\",\"prompt\":\"once upon\",\"context\":[1],\"raw\":true,\"stream\":false}",
                "{\"model\":\"test-model\",\"prompt\":\"def f():\",\"suffix\":\"\",\"system\":\"x\",\"raw\":true}",
                "{\"model\":\"test-model\",\"system\":\"x\",\"raw\":true}"
            }) {
                Response response = post(server.getPort(), "/api/generate", body, "");
                assertThat(body, response.code, is(400));
                assertThat(
                        body,
                        response.body,
                        is("{\"error\":\"raw mode does not support template, system, or context\"}"));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * Ollama answers an {@code /api/generate} without a prompt by loading the model, which is how its
     * clients preload one: {@code done} at once, empty text, {@code done_reason "load"}. Ours sent an
     * empty user turn through the chat template and generated an answer nobody asked for.
     */
    @Test
    public void ollamaGenerateWithoutAPromptOnlyReportsTheModelLoaded() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String body : new String[] {
                "{\"model\":\"test-model\"}",
                "{\"model\":\"test-model\",\"prompt\":\"\",\"stream\":false}",
                "{\"model\":\"test-model\",\"prompt\":\"\",\"raw\":true,\"stream\":true}"
            }) {
                Response response = post(server.getPort(), "/api/generate", body, "");
                assertThat(body, response.code, is(200));
                JsonNode out = READER.readTree(response.body);
                assertThat(body, out.path("done").asBoolean(false), is(true));
                assertThat(body, out.path("done_reason").asText(), is("load"));
                assertThat(body, out.path("response").asText(null), is(""));
                assertThat(body, out.path("model").asText(), is("test-model"));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * Ollama answers an {@code /api/chat} without messages by loading the model, which is how clients
     * preload one through the chat route; ours ran the empty conversation into the chat mapper, which
     * refused it with a 400. The answer is one object even when streaming was asked for (or defaulted).
     */
    @Test
    public void ollamaChatWithoutMessagesOnlyReportsTheModelLoaded() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String body : new String[] {
                "{\"model\":\"test-model\"}",
                "{\"model\":\"test-model\",\"messages\":[],\"stream\":false}",
                "{\"model\":\"test-model\",\"messages\":null,\"stream\":true}",
                "{\"model\":\"test-model\",\"messages\":[],\"keep_alive\":0}"
            }) {
                Response response = post(server.getPort(), "/api/chat", body, "");
                assertThat(body, response.code, is(200));
                JsonNode out = READER.readTree(response.body);
                assertThat(body, out.path("done").asBoolean(false), is(true));
                assertThat(body, out.path("done_reason").asText(), is("load"));
                assertThat(body, out.path("message").path("content").asText(null), is(""));
                assertThat(body, out.path("model").asText(), is("test-model"));
            }
            assertThat("nothing was generated", backend.called, is(""));
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

    // -------------------------------------------------------------------------
    // A request the NATIVE layer rejects (InvalidRequestException, thrown by the JNI layer where
    // llama.cpp's own server answers 400) is the client's fault on every surface; a plain
    // LlamaException stays a server error. Pins the fix for the "400 vs. 500" entry of TODO.md.
    // -------------------------------------------------------------------------

    @Test
    public void nativeRejectionOfAChatCompletionReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new RejectingBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(response.code, is(400));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
            assertThat(response.body, containsString(RejectingBackend.REASON_FRAGMENT));
        }
    }

    @Test
    public void nativeRejectionOfAnEmbeddingReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new RejectingBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/embeddings", "{\"input\":[]}", "");
            assertThat(response.code, is(400));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
        }
    }

    @Test
    public void nativeRejectionOfACompletionReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new RejectingBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/completions", "{\"prompt\":\"\"}", "");
            assertThat(response.code, is(400));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
        }
    }

    // Upstream reads the first result before it starts a stream, so a request refused at once is answered
    // with the error status, as without "stream" (F182). The response starts with the first
    // write -- a chunk, the protocol's opening events or a heartbeat -- and only a failure after that
    // travels as an event.

    /** The five streaming routes: path, body, and the error body's distinguishing parts at 400 and 500. */
    private static final String[][] STREAMING_ROUTES = {
        {
            "/v1/chat/completions",
            "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            "\"type\":\"invalid_request_error\"",
            "\"type\":\"server_error\""
        },
        {
            "/v1/completions",
            "{\"stream\":true,\"prompt\":\"hi\"}",
            "\"type\":\"invalid_request_error\"",
            "\"type\":\"server_error\""
        },
        {
            "/api/chat",
            "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            "{\"error\":",
            "{\"error\":"
        },
        {
            "/v1/messages",
            "{\"model\":\"test-model\",\"stream\":true,\"max_tokens\":8,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            "\"error\":{\"type\":\"invalid_request_error\"",
            "\"error\":{\"type\":\"api_error\""
        },
        {
            "/v1/responses",
            "{\"model\":\"test-model\",\"stream\":true,\"input\":\"hi\"}",
            "\"type\":\"invalid_request_error\"",
            "\"type\":\"server_error\""
        },
    };

    @Test
    public void aRejectionBeforeTheFirstChunkIsA400OnEveryStreamingRoute() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new RejectingBackend(), config()).start()) {
            for (String[] route : STREAMING_ROUTES) {
                Response response = post(server.getPort(), route[0], route[1], "");
                assertThat(route[0], response.code, is(400));
                assertThat(route[0], response.contentType, startsWith("application/json"));
                assertThat(route[0], response.body, containsString(route[2]));
                assertThat(route[0], response.body, containsString(RejectingBackend.REASON_FRAGMENT));
            }
        }
    }

    @Test
    public void aFailureBeforeTheFirstChunkIsA500OnEveryStreamingRoute() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new StreamFailingBackend(), config()).start()) {
            for (String[] route : STREAMING_ROUTES) {
                Response response = post(server.getPort(), route[0], route[1], "");
                assertThat(route[0], response.code, is(500));
                assertThat(route[0], response.body, containsString(route[3]));
                assertThat(route[0], response.body, containsString(StreamFailingBackend.REASON));
            }
        }
    }

    /**
     * {@link OpenAiBackend} lets a backend report a generation failure as an {@link IOException}. Since
     * the response starts with its first write, such a failure before it left the request without any
     * response: the route took it for a client that had gone away. It is a 500 with the route's error
     * body, as without {@code stream}.
     */
    @Test
    public void aBackendIOExceptionBeforeTheFirstChunkIsA500OnEveryStreamingRoute() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new IoFailingBackend(), config()).start()) {
            for (String[] route : STREAMING_ROUTES) {
                Response response = post(server.getPort(), route[0], route[1], "");
                assertThat(route[0], response.code, is(500));
                assertThat(route[0], response.body, containsString(route[3]));
                assertThat(route[0], response.body, containsString(IoFailingBackend.REASON));
            }
        }
    }

    @Test
    public void aRejectionAfterTheFirstChunkIsAnInvalidRequestEvent() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, 0L, new InvalidRequestException(RejectingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            // the stream had started, so the verdict travels as an SSE data event after the chunk
            assertThat(response.code, is(200));
            assertThat(response.body, containsString("\"content\":\"he\""));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
            assertThat(response.body, containsString(RejectingBackend.REASON_FRAGMENT));
        }
    }

    // A heartbeat starts the response too: it is what keeps a client waiting through a long prompt.
    @Test
    public void aRejectionAfterAHeartbeatIsAnEvent() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .heartbeatMillis(50L)
                .build();
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(0, 300L, new InvalidRequestException(RejectingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, cfg).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(200));
            assertThat(response.body, containsString(": ping"));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
        }
    }

    // A Responses client dispatches on the data's "type" and reads a stream failure from a
    // response.failed event (Codex: process_responses_event -> parse_failed_response, which reads
    // response.error.message); without one it reports only "stream closed before response.completed".

    @Test
    public void rejectedStreamedResponseEndsWithAResponseFailedEvent() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, 0L, new InvalidRequestException(RejectingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"input\":\"hi\"}";
            String events = post(server.getPort(), "/v1/responses", body, "").body;
            assertThat(events, containsString("event: response.failed"));
            String failed = events.substring(events.indexOf("event: response.failed"));
            assertThat(failed, containsString("\"type\":\"response.failed\""));
            assertThat(failed, containsString("\"status\":\"failed\""));
            assertThat(failed, containsString("\"code\":\"invalid_request_error\""));
            assertThat(failed, containsString(RejectingBackend.REASON_FRAGMENT));
        }
    }

    @Test
    public void failedStreamedResponseEndsWithAResponseFailedEvent() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, 0L, new LlamaException(StreamFailingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"input\":\"hi\"}";
            String events = post(server.getPort(), "/v1/responses", body, "").body;
            assertThat(events, containsString("event: response.failed"));
            String failed = events.substring(events.indexOf("event: response.failed"));
            assertThat(failed, containsString("\"status\":\"failed\""));
            assertThat(failed, containsString("\"code\":\"server_error\""));
            assertThat(failed, containsString(StreamFailingBackend.REASON));
        }
    }

    @Test
    public void nativeRejectionOnTheOllamaAndAnthropicSurfacesReturns400() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new RejectingBackend(), config()).start()) {
            String ollama = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                    + "\"stream\":false}";
            assertThat(post(server.getPort(), "/api/chat", ollama, "").code, is(400));
            String anthropic = "{\"model\":\"test-model\",\"max_tokens\":8,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            assertThat(post(server.getPort(), "/v1/messages", anthropic, "").code, is(400));
        }
    }

    /**
     * A non-object {@code messages} element made the Ollama chat translation throw a
     * {@link ClassCastException} outside the handler's error handling: the connection was closed
     * without a status line. It is a request the server refuses, so it is a 400, streamed or not.
     */
    @Test
    public void aNonObjectOllamaChatMessageIsA400NotADroppedConnection() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String body : new String[] {
                "{\"model\":\"test-model\",\"messages\":[\"hi\"],\"stream\":false}",
                "{\"model\":\"test-model\",\"messages\":[\"hi\"]}"
            }) {
                Response response = post(server.getPort(), "/api/chat", body, "");
                assertThat(body, response.code, is(400));
                assertThat(body, response.body, containsString("{\"error\":"));
                assertThat(body, response.body, containsString("messages[0]"));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    @Test
    public void aPlainLlamaExceptionStaysAServerError() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FailingBackend(), config()).start()) {
            Response response = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(response.code, is(500));
            assertThat(response.body, containsString("\"type\":\"server_error\""));
        }
    }

    // Anthropic's error types: invalid_request_error is the client's fault (400), api_error an
    // unexpected server error (500). A 500 that calls itself invalid_request_error contradicts its status.

    @Test
    public void aFailedAnthropicMessageIsAnApiError() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FailingBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"max_tokens\":8,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = post(server.getPort(), "/v1/messages", body, "");
            assertThat(response.code, is(500));
            assertThat(response.body, containsString("\"error\":{\"type\":\"api_error\""));
        }
    }

    @Test
    public void aFailedStreamedAnthropicMessageIsAnApiError() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, 0L, new LlamaException(StreamFailingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"max_tokens\":8,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            String events = post(server.getPort(), "/v1/messages", body, "").body;
            assertThat(events, containsString("event: error"));
            assertThat(
                    events.substring(events.indexOf("event: error")),
                    containsString("\"error\":{\"type\":\"api_error\""));
        }
    }

    @Test
    public void aRejectedStreamedAnthropicMessageStaysAnInvalidRequestError() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, 0L, new InvalidRequestException(RejectingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"stream\":true,\"max_tokens\":8,"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            String events = post(server.getPort(), "/v1/messages", body, "").body;
            assertThat(
                    events.substring(events.indexOf("event: error")),
                    containsString("\"error\":{\"type\":\"invalid_request_error\""));
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

    /**
     * A preflight must allow the headers the browser asks for. The server authenticates Anthropic clients
     * with {@code x-api-key}, and browser SDKs also send {@code anthropic-version} and {@code x-stainless-*}
     * headers; with only {@code Content-Type, Authorization} allowed, the browser refused to send such a
     * request at all. The preflight is written over a raw socket because {@code HttpURLConnection}
     * silently drops the restricted {@code Access-Control-Request-Headers} header.
     */
    @Test
    public void optionsPreflightAllowsTheRequestedHeaders() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start();
                Socket socket = new Socket("127.0.0.1", server.getPort())) {
            OutputStream out = socket.getOutputStream();
            out.write(("OPTIONS /v1/messages HTTP/1.1\r\n"
                            + "Host: 127.0.0.1\r\n"
                            + "Origin: https://app.example\r\n"
                            + "Access-Control-Request-Method: POST\r\n"
                            + "Access-Control-Request-Headers: content-type, x-api-key, anthropic-version, x-stainless-os\r\n"
                            + "Connection: close\r\n\r\n")
                    .getBytes(StandardCharsets.US_ASCII));
            out.flush();
            InputStream in = socket.getInputStream();
            String response =
                    new String(in.readAllBytes(), StandardCharsets.ISO_8859_1).toLowerCase(java.util.Locale.ROOT);

            assertThat(
                    response,
                    containsString(
                            "access-control-allow-headers: content-type, x-api-key, anthropic-version, x-stainless-os"));
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
    public void anthropicApiKeyHeaderAndLowercaseBearerSchemeAreAccepted() throws IOException {
        // The Anthropic Messages surface is authenticated the Anthropic way, with an x-api-key header
        // (what Anthropic SDKs send), and the Authorization scheme name is case-insensitive (RFC 7235).
        // Accepting only "Bearer <key>" answered 401 to every Anthropic client of a server with a key.
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .apiKey("secret")
                .build();
        String messages = "{\"model\":\"test-model\",\"max_tokens\":16,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), cfg).start()) {
            int port = server.getPort();
            assertThat(postWithHeader(port, "/v1/messages", messages, "x-api-key", "secret"), is(200));
            assertThat(postWithHeader(port, "/v1/messages", messages, "x-api-key", "wrong"), is(401));
            assertThat(
                    postWithHeader(port, "/v1/chat/completions", CHAT_BODY, "Authorization", "bearer secret"), is(200));
        }
    }

    private static int postWithHeader(int port, String path, String body, String header, String value)
            throws IOException {
        java.net.HttpURLConnection conn =
                (java.net.HttpURLConnection) new java.net.URL("http://127.0.0.1:" + port + path).openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty(header, value);
        try (java.io.OutputStream os = conn.getOutputStream()) {
            os.write(body.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
        int code = conn.getResponseCode();
        conn.disconnect();
        return code;
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

    /** Backend that records which operation a request reached and with which body. */
    static final class RecordingBackend extends FakeBackend {
        volatile String called = "";
        volatile JsonNode request = READER.createObjectNode();

        @Override
        public String complete(JsonNode request) {
            this.called = "complete";
            this.request = request;
            return "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"finish_reason\":\"stop\","
                    + "\"message\":{\"role\":\"assistant\",\"content\":\"because\",\"reasoning_content\":\"hmm\"}}],"
                    + "\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3}}";
        }

        @Override
        public String completions(JsonNode request) {
            this.called = "completions";
            this.request = request;
            return super.completions(request);
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            this.called = "stream";
            this.request = request;
            super.stream(request, sink);
        }
    }

    /** Backend whose answers end with the given OpenAI finish reason (and a matching infill stop type). */
    static final class FinishReasonBackend extends FakeBackend {
        private final String finishReason;

        FinishReasonBackend(String finishReason) {
            this.finishReason = finishReason;
        }

        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
                    + "\"message\":{\"role\":\"assistant\",\"content\":\"hel\"},\"finish_reason\":\""
                    + finishReason + "\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}");
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\""
                    + finishReason + "\"}]}");
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[],"
                    + "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}");
        }

        @Override
        public String completions(JsonNode request) {
            return "{\"object\":\"text_completion\",\"choices\":[{\"text\":\"hel\",\"finish_reason\":\"" + finishReason
                    + "\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}";
        }

        @Override
        public String infill(JsonNode request) {
            return "{\"content\":\" w\",\"stop\":true,\"stop_type\":\""
                    + ("length".equals(finishReason) ? "limit" : "eos")
                    + "\",\"tokens_evaluated\":4,\"tokens_predicted\":1}";
        }
    }

    /**
     * Backend whose every operation is rejected the way the JNI layer rejects a request llama.cpp's
     * validation refuses: an {@link InvalidRequestException} with the bare upstream reason.
     */
    static final class RejectingBackend extends FakeBackend {
        static final String REASON = "\"prompt\" must not be empty";
        /** The part of {@link #REASON} without the quotes, which the error JSON escapes. */
        static final String REASON_FRAGMENT = "must not be empty";

        @Override
        public String complete(JsonNode request) {
            throw new InvalidRequestException(REASON);
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) {
            throw new InvalidRequestException(REASON);
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) {
            throw new InvalidRequestException(REASON);
        }

        @Override
        public String completions(JsonNode request) {
            throw new InvalidRequestException(REASON);
        }

        @Override
        public String embeddings(JsonNode request) {
            throw new InvalidRequestException(REASON);
        }
    }

    /** Backend whose chat completion fails like an inference failure: a plain {@link LlamaException}. */
    static final class FailingBackend extends FakeBackend {
        @Override
        public String complete(JsonNode request) {
            throw new LlamaException("decoding failed");
        }
    }

    /** Backend whose streamed generation fails like an inference failure: a plain {@link LlamaException}. */
    static final class StreamFailingBackend extends FakeBackend {
        static final String REASON = "decoding failed";

        @Override
        public void stream(JsonNode request, ChunkSink sink) {
            throw new LlamaException(REASON);
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) {
            throw new LlamaException(REASON);
        }
    }

    /** Backend whose streamed generation fails with an {@link IOException}, which the backend contract allows. */
    static final class IoFailingBackend extends FakeBackend {
        static final String REASON = "disk full";

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            throw new IOException(REASON);
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
            throw new IOException(REASON);
        }
    }

    /** Backend whose streams emit {@code chunks} chunks, wait {@code delayMillis}, then fail with {@code failure}. */
    static final class ScriptedStreamBackend extends FakeBackend {
        private final int chunks;
        private final long delayMillis;
        private final RuntimeException failure;

        ScriptedStreamBackend(int chunks, long delayMillis, RuntimeException failure) {
            this.chunks = chunks;
            this.delayMillis = delayMillis;
            this.failure = failure;
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            for (int i = 0; i < chunks; i++) {
                sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
            }
            pause();
            throw failure;
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
            for (int i = 0; i < chunks; i++) {
                sink.accept("{\"object\":\"text_completion\",\"choices\":[{\"text\":\"he\",\"finish_reason\":null}]}");
            }
            pause();
            throw failure;
        }

        private void pause() {
            if (delayMillis > 0) {
                try {
                    Thread.sleep(delayMillis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
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
