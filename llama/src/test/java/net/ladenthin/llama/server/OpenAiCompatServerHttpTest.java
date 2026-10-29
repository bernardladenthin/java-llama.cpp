// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.Socket;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.json.ChatResponseParser;
import net.ladenthin.llama.json.ChatStreamAssembler;
import net.ladenthin.llama.value.ChatResponse;
import nl.altindag.log.LogCaptor;
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

    /**
     * The client half of the stream contract: what the server streams assembles into the answer it
     * gives the same request without {@code stream}, usage chunk included.
     */
    @Test
    public void streamingAssemblesIntoTheNonStreamingAnswer() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"stream\":true,\"stream_options\":{\"include_usage\":true},"
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response streamed = post(server.getPort(), "/v1/chat/completions", body, "");
            Response whole = post(server.getPort(), "/v1/chat/completions", CHAT_BODY, "");
            assertThat(streamed.code, is(200));
            assertThat(whole.code, is(200));

            ChatResponse fromStream = ChatStreamAssembler.parseStream(streamed.body);
            ChatResponse direct = new ChatResponseParser().parseResponse(whole.body);

            assertThat(fromStream.getFirstContent(), is("hello"));
            assertThat(fromStream.getFirstContent(), is(direct.getFirstContent()));
            assertThat(fromStream.getChoices().get(0).getFinishReason(), is("stop"));
            assertThat(
                    fromStream.getUsage().getPromptTokens(),
                    is(direct.getUsage().getPromptTokens()));
            assertThat(
                    fromStream.getUsage().getCompletionTokens(),
                    is(direct.getUsage().getCompletionTokens()));
            assertThat(
                    fromStream.getUsage().getCachedTokens(),
                    is(direct.getUsage().getCachedTokens()));
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

    // The backend emits only after the client has read a heartbeat, so the assertion follows an
    // observed ": ping" instead of a timer that had to fire inside a fixed sleep.
    @Test
    public void streamingEmitsHeartbeatsDuringAGap() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .heartbeatMillis(50L)
                .build();
        CountDownLatch release = new CountDownLatch(1);
        try (OpenAiCompatServer server = new OpenAiCompatServer(new SlowFakeBackend(release), cfg).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = postReleasingAfterAHeartbeat(server.getPort(), "/v1/chat/completions", body, release);
            assertThat(response.code, is(200));
            assertThat(response.body, startsWith(": ping"));
            assertThat(response.body, containsString("\"content\":\"done\""));
            assertThat(response.body, containsString("data: [DONE]"));
            // the heartbeats are comments, which a client skips
            assertThat(ChatStreamAssembler.parseStream(response.body).getFirstContent(), is("done"));
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
        // tokens there) ran a whole generation through the /v1/messages handler instead of a 404. That
        // route is served now; any other sub-path still is no part of its parent.
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"m\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            assertThat(post(server.getPort(), "/v1/messages/batches", body, "").code, is(404));
            assertThat(post(server.getPort(), "/v1/chat/completions/other", CHAT_BODY, "").code, is(404));
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * upstream counts the prompt tokens of a request without generating on three routes:
     * {@code /v1/messages/count_tokens} (Anthropic, {@code {"input_tokens": N}}),
     * {@code /v1/responses/input_tokens} and {@code /v1/chat/completions/input_tokens} (with
     * {@code "object":"response.input_tokens"}). All three answered 404 here (F112 kept them from running
     * a generation through the parent route). Each translates its body as its generating route does.
     */
    @Test
    public void theTokenCountRoutesAnswerWithTheCountOfTheTranslatedRequest() throws IOException {
        CountingBackend backend = new CountingBackend(42);
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String anthropic = "{\"model\":\"m\",\"system\":\"be brief\","
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            for (String path : new String[] {"/v1/messages/count_tokens", "/messages/count_tokens"}) {
                Response response = post(server.getPort(), path, anthropic, "");
                assertThat(path, response.code, is(200));
                assertThat(path, READER.readTree(response.body), is(READER.readTree("{\"input_tokens\":42}")));
                JsonNode messages = backend.request.path("messages");
                assertThat(path, messages.path(0).path("role").asText(), is("system"));
                assertThat(path, messages.path(1).path("content").asText(), is("hi"));
            }
            String responses = "{\"model\":\"m\",\"instructions\":\"be brief\",\"input\":\"hi\"}";
            for (String path : new String[] {"/v1/responses/input_tokens", "/responses/input_tokens"}) {
                Response response = post(server.getPort(), path, responses, "");
                assertThat(path, response.code, is(200));
                assertThat(
                        path,
                        READER.readTree(response.body),
                        is(READER.readTree("{\"input_tokens\":42,\"object\":\"response.input_tokens\"}")));
                assertThat(
                        path,
                        backend.request.path("messages").path(1).path("content").asText(),
                        is("hi"));
            }
            for (String path : new String[] {"/v1/chat/completions/input_tokens", "/chat/completions/input_tokens"}) {
                Response response = post(server.getPort(), path, CHAT_BODY, "");
                assertThat(path, response.code, is(200));
                assertThat(
                        path,
                        READER.readTree(response.body),
                        is(READER.readTree("{\"input_tokens\":42,\"object\":\"response.input_tokens\"}")));
                assertThat(path, backend.request, is(READER.readTree(CHAT_BODY)));
            }
            assertThat("nothing was generated", backend.generated, is(false));
        }
    }

    /**
     * A request the translation or the count refuses is the client's fault, a backend without a count
     * answers 501 (upstream's answer for a capability the server lacks), as do images, whose tokens the
     * count cannot see.
     */
    @Test
    public void theTokenCountRoutesAnswerRefusalsInTheirRoutesErrorShape() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new CountingBackend(1), config()).start()) {
            Response bad = post(server.getPort(), "/v1/messages/count_tokens", "{\"max_tokens\":\"x\"}", "");
            assertThat(bad.body, bad.code, is(400));
            assertThat(READER.readTree(bad.body).path("error").path("type").asText(), is("invalid_request_error"));
        }
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            Response anthropic = post(
                    server.getPort(),
                    "/v1/messages/count_tokens",
                    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                    "");
            assertThat(anthropic.body, anthropic.code, is(501));
            assertThat(READER.readTree(anthropic.body).path("type").asText(), is("error"));
            Response chat = post(server.getPort(), "/v1/chat/completions/input_tokens", CHAT_BODY, "");
            assertThat(chat.body, chat.code, is(501));
            assertThat(READER.readTree(chat.body).path("error").path("type").asText(), is("not_supported_error"));
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
     * The raw route dropped {@code format} and {@code images}: an unconstrained completion, and the
     * images never reached the model. They now arrive as llama.cpp's raw schema reads them.
     */
    @Test
    public void ollamaGenerateWithRawCarriesItsFormatAndImages() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"look [img-0] now\",\"raw\":true,"
                    + "\"format\":\"json\",\"images\":[\"AAAA\"],\"stream\":false}";
            Response response = post(server.getPort(), "/api/generate", body, "");

            assertThat(response.body, response.code, is(200));
            assertThat(backend.called, is("completions"));
            assertThat(backend.request.path("json_schema"), is(READER.readTree("{\"type\":\"object\"}")));
            assertThat(backend.request.path("prompt").path("prompt_string").asText(), is("look <m> now"));
            assertThat(backend.request.path("prompt").path("multimodal_data"), is(READER.readTree("[\"AAAA\"]")));
        }
    }

    /**
     * A backend without a media marker cannot take a raw prompt with images: a capability it lacks, which
     * upstream answers 501, here in the route's Ollama error shape. The marker's default threw
     * {@link UnsupportedOperationException} into the generic branch, a 500 (F267).
     */
    @Test
    public void ollamaGenerateWithRawImagesOnABackendWithoutAMediaMarkerIsNotImplemented() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(new FakeBackend(), config()).start()) {
            String body = "{\"model\":\"test-model\",\"prompt\":\"look [img-0] now\",\"raw\":true,"
                    + "\"images\":[\"AAAA\"],\"stream\":false}";
            Response response = post(server.getPort(), "/api/generate", body, "");

            assertThat(response.body, response.code, is(501));
            assertThat(
                    READER.readTree(response.body).path("error").asText(),
                    is("raw prompts with images are not supported by this backend"));
        }
    }

    /** A raw image no {@code [img-N]} tag places, or a format Ollama does not know, is the client's fault. */
    @Test
    public void ollamaGenerateWithAnUnplacedImageOrAnUnknownFormatIsABadRequest() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String[][] cases = {
                {
                    "{\"model\":\"test-model\",\"prompt\":\"look\",\"raw\":true,\"images\":[\"AAAA\"]}",
                    "image 0 is not placed in the raw prompt; put [img-0] where it goes"
                },
                {
                    "{\"model\":\"test-model\",\"prompt\":\"look\",\"raw\":true,\"format\":\"yaml\"}",
                    "invalid format: \"yaml\"; expected \"json\" or a JSON Schema object"
                },
                {
                    "{\"model\":\"test-model\",\"prompt\":\"look\",\"format\":\"yaml\",\"stream\":false}",
                    "invalid format: \"yaml\"; expected \"json\" or a JSON Schema object"
                }
            };
            for (String[] c : cases) {
                Response response = post(server.getPort(), "/api/generate", c[0], "");
                assertThat(c[0], response.code, is(400));
                assertThat(c[0], READER.readTree(response.body).path("error").asText(), is(c[1]));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * Without {@code raw}, Ollama renders the prompt with a request's {@code template} and continues the
     * token {@code context} of an earlier answer. Neither can be honoured here -- the model's own chat
     * template renders the prompt, and no answer hands out a context -- and both were dropped, the
     * request answered as an ordinary chat turn. They are refused; empty values mean "not set".
     */
    @Test
    public void ollamaGenerateWithATemplateOrAContextIsRefused() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String[][] cases = {
                {
                    "{\"model\":\"test-model\",\"prompt\":\"why?\",\"template\":\"{{ .Prompt }}\",\"stream\":false}",
                    OllamaApiSupport.TEMPLATE_UNSUPPORTED_MESSAGE + " (got 13 characters)"
                },
                {
                    "{\"model\":\"test-model\",\"prompt\":\"why?\",\"context\":[1,2,3],\"stream\":false}",
                    OllamaApiSupport.CONTEXT_UNSUPPORTED_MESSAGE + " (got 3 tokens)"
                },
                {
                    "{\"model\":\"test-model\",\"prompt\":\"def f(\",\"suffix\":\")\",\"template\":\"t\"}",
                    OllamaApiSupport.TEMPLATE_UNSUPPORTED_MESSAGE + " (got 1 characters)"
                }
            };
            for (String[] c : cases) {
                Response response = post(server.getPort(), "/api/generate", c[0], "");
                assertThat(c[0], response.code, is(400));
                assertThat(c[0], READER.readTree(response.body).path("error").asText(), is(c[1]));
            }
            assertThat("nothing was generated", backend.called, is(""));

            String empty = "{\"model\":\"test-model\",\"prompt\":\"why?\",\"template\":\"\",\"context\":[],"
                    + "\"stream\":false}";
            assertThat(post(server.getPort(), "/api/generate", empty, "").code, is(200));
            assertThat(backend.called, is("complete"));
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
     * A {@code prompt} that is present but not a string is a malformed body, which Ollama refuses with
     * a 400 when it binds the JSON. Ours read it as {@code ""} and answered 200 with the load answer.
     */
    @Test
    public void ollamaGenerateWithANonTextPromptIsABadRequest() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String body : new String[] {
                "{\"model\":\"test-model\",\"prompt\":[\"a\"]}",
                "{\"model\":\"test-model\",\"prompt\":{},\"suffix\":\"x\"}",
                "{\"model\":\"test-model\",\"prompt\":1,\"raw\":true,\"stream\":true}"
            }) {
                Response response = post(server.getPort(), "/api/generate", body, "");
                assertThat(body, response.code, is(400));
                assertThat(
                        body, READER.readTree(response.body).path("error").asText(), is("'prompt' must be a string"));
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

    /**
     * A backend {@link IOException} after the stream has started was taken for a client that had gone
     * away: no error event, and a DEBUG line only. A client that is still connected then saw the stream
     * end as if the answer were complete. It is the route's server-error event now, and a WARN naming
     * the exception; a write that failed because the client did go away stays a DEBUG line (see
     * {@code ResponseStreamTest}).
     */
    @Test
    public void aBackendIOExceptionAfterTheFirstChunkIsAnErrorEventOnEveryStreamingRoute() throws IOException {
        try (LogCaptor captor = LogCaptor.forClass(OpenAiCompatServer.class);
                OpenAiCompatServer server =
                        new OpenAiCompatServer(new IoFailingAfterFirstChunkBackend(), config()).start()) {
            for (String[] route : STREAMING_ROUTES) {
                captor.clearLogs();
                Response response = post(server.getPort(), route[0], route[1], "");
                assertThat(route[0], response.code, is(200));
                assertThat(route[0], response.body, containsString("he"));
                assertThat(route[0], response.body, containsString(route[3]));
                assertThat(route[0], response.body, containsString(IoFailingAfterFirstChunkBackend.REASON));
                assertThat(
                        route[0],
                        captor.getWarnLogs(),
                        hasItem(allOf(
                                containsString("IOException"),
                                containsString(IoFailingAfterFirstChunkBackend.REASON))));
            }
        }
    }

    @Test
    public void aRejectionAfterTheFirstChunkIsAnInvalidRequestEvent() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, new InvalidRequestException(RejectingBackend.REASON));
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
    // The backend fails only after the client has read a heartbeat, so the stream has provably
    // started when the rejection arrives.
    @Test
    public void aRejectionAfterAHeartbeatIsAnEvent() throws IOException {
        OpenAiServerConfig cfg = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .heartbeatMillis(50L)
                .build();
        CountDownLatch release = new CountDownLatch(1);
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(0, release, new InvalidRequestException(RejectingBackend.REASON));
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, cfg).start()) {
            String body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            Response response = postReleasingAfterAHeartbeat(server.getPort(), "/v1/chat/completions", body, release);
            assertThat(response.code, is(200));
            assertThat(response.body, startsWith(": ping"));
            assertThat(response.body, containsString("\"type\":\"invalid_request_error\""));
            assertThat(response.body, containsString(RejectingBackend.REASON_FRAGMENT));
        }
    }

    /**
     * POSTs {@code body} and reads the response line by line; {@code release} is counted down once a
     * heartbeat line has arrived, and the rest of the body is read to its end.
     *
     * @param port the server port
     * @param path the request path
     * @param body the JSON request body
     * @param release the latch the backend waits on
     * @return the status and the whole body, lines joined by {@code \n}
     * @throws IOException on transport failure
     * @throws AssertionError if a successful stream ends before a heartbeat arrived
     */
    private static Response postReleasingAfterAHeartbeat(int port, String path, String body, CountDownLatch release)
            throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL("http://127.0.0.1:" + port + path).openConnection();
        conn.setReadTimeout((int) TimeUnit.SECONDS.toMillis(BACKEND_RELEASE_TIMEOUT_SECONDS * 2));
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.getBytes(StandardCharsets.UTF_8));
        }
        // blocks until the first write of the server, which is the heartbeat: nothing else is written
        // before the backend is released
        int code = conn.getResponseCode();
        StringBuilder text = new StringBuilder();
        // an error status means the backend already gave up waiting: hand it to the caller's assertions
        InputStream stream = code < 400 ? conn.getInputStream() : conn.getErrorStream();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                text.append(line).append('\n');
                if (release.getCount() > 0 && line.startsWith(": ping")) {
                    release.countDown();
                }
            }
        }
        if (code < 400 && release.getCount() > 0) {
            throw new AssertionError("the stream ended without a heartbeat: " + text);
        }
        return new Response(code, text.toString(), null, conn.getContentType());
    }

    // A Responses client dispatches on the data's "type" and reads a stream failure from a
    // response.failed event (Codex: process_responses_event -> parse_failed_response, which reads
    // response.error.message); without one it reports only "stream closed before response.completed".

    @Test
    public void rejectedStreamedResponseEndsWithAResponseFailedEvent() throws IOException {
        ScriptedStreamBackend backend =
                new ScriptedStreamBackend(1, new InvalidRequestException(RejectingBackend.REASON));
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
        ScriptedStreamBackend backend = new ScriptedStreamBackend(1, new LlamaException(StreamFailingBackend.REASON));
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

    /**
     * A {@code max_tokens} that is not a number is refused by the Anthropic translation (F223); the
     * refusal must reach the client as Anthropic's 400, streamed or not, before any SSE starts and
     * without generating anything.
     */
    @Test
    public void aStringMaxTokensOnTheAnthropicRouteIsA400InvalidRequestError() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String stream : new String[] {"false", "true"}) {
                String body = "{\"model\":\"test-model\",\"max_tokens\":\"100\",\"stream\":" + stream
                        + ",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
                Response response = post(server.getPort(), "/v1/messages", body, "");
                assertThat(body, response.code, is(400));
                assertThat(body, response.body, containsString("\"type\":\"invalid_request_error\""));
                assertThat(body, response.body, containsString("max_tokens"));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * A {@code max_tokens} that is not a number is refused by the chat mapper, as upstream's schema
     * refuses it; the refusal reaches the client as 400, streamed or not, before any SSE starts.
     */
    @Test
    public void aStringMaxTokensOnTheChatRouteIsA400() throws IOException {
        MappingBackend backend = new MappingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String stream : new String[] {"false", "true"}) {
                String body = "{\"max_tokens\":\"100\",\"stream\":" + stream
                        + ",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
                Response response = post(server.getPort(), "/v1/chat/completions", body, "");
                assertThat(body, response.code, is(400));
                assertThat(body, response.body, containsString("\"type\":\"invalid_request_error\""));
                assertThat(
                        body, response.body, containsString("Field 'max_tokens': type must be number, but is string"));
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * A {@code tool_choice} the server cannot honour is refused on the Anthropic route, streamed or not,
     * before anything is generated: a type it does not know, and a tool name no tool carries (the chat
     * mapper's check, reached through the translated request).
     */
    @Test
    public void anUnsupportedAnthropicToolChoiceIsA400() throws IOException {
        MappingBackend backend = new MappingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String stream : new String[] {"false", "true"}) {
                String[][] cases = {
                    {"{\"type\":\"allowed_tools\"}", "tool_choice type 'allowed_tools' is not supported"},
                    {"{\"type\":\"tool\",\"name\":\"g\"}", "tool_choice names unknown tool 'g'"}
                };
                for (String[] c : cases) {
                    String body = "{\"model\":\"test-model\",\"max_tokens\":8,\"stream\":" + stream
                            + ",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                            + "\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}],"
                            + "\"tool_choice\":" + c[0] + "}";
                    Response response = post(server.getPort(), "/v1/messages", body, "");
                    assertThat(body, response.code, is(400));
                    assertThat(body, response.body, containsString("\"type\":\"invalid_request_error\""));
                    assertThat(body, response.body, containsString(c[1]));
                }
            }
            assertThat("nothing was generated", backend.called, is(""));
        }
    }

    /**
     * Upstream refuses {@code logprobs} on a streamed chat request with tools; the chat route answers
     * the same {@code 400} before the stream starts, and the blocking request still generates.
     */
    @Test
    public void logprobsWithToolsOnAStreamedChatIsA400LikeUpstream() throws IOException {
        MappingBackend backend = new MappingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            String body = "{\"stream\":true,\"logprobs\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                    + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}";
            Response response = post(server.getPort(), "/v1/chat/completions", body, "");
            assertThat(response.code, is(400));
            assertThat(response.body, containsString("logprobs is not supported with tools + stream"));
            assertThat("nothing was generated", backend.called, is(""));
            Response blocking = post(
                    server.getPort(), "/v1/chat/completions", body.replace("\"stream\":true", "\"stream\":false"), "");
            assertThat(blocking.code, is(200));
            assertThat(backend.called, is("complete"));
        }
    }

    /** An Anthropic image the server cannot fetch is a 400, streamed or not, before anything is generated. */
    @Test
    public void aFileImageSourceOnTheAnthropicRouteIsA400() throws IOException {
        RecordingBackend backend = new RecordingBackend();
        try (OpenAiCompatServer server = new OpenAiCompatServer(backend, config()).start()) {
            for (String stream : new String[] {"false", "true"}) {
                String body = "{\"model\":\"test-model\",\"max_tokens\":8,\"stream\":" + stream
                        + ",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what?\"},"
                        + "{\"type\":\"image\",\"source\":{\"type\":\"file\",\"file_id\":\"file_1\"}}]}]}";
                Response response = post(server.getPort(), "/v1/messages", body, "");
                assertThat(body, response.code, is(400));
                assertThat(body, response.body, containsString("\"type\":\"invalid_request_error\""));
                assertThat(body, response.body, containsString("image source type 'file' is not supported"));
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
        ScriptedStreamBackend backend = new ScriptedStreamBackend(1, new LlamaException(StreamFailingBackend.REASON));
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
                new ScriptedStreamBackend(1, new InvalidRequestException(RejectingBackend.REASON));
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
        public String mediaMarker() {
            return "<m>";
        }

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

    /** Backend that counts every request as {@code count} tokens and records it; generation is flagged. */
    static final class CountingBackend extends FakeBackend {
        private final int count;
        volatile JsonNode request = READER.createObjectNode();
        volatile boolean generated;

        CountingBackend(int count) {
            this.count = count;
        }

        @Override
        public int countTokens(JsonNode request) {
            this.request = request;
            return count;
        }

        @Override
        public String complete(JsonNode request) {
            generated = true;
            return super.complete(request);
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            generated = true;
            super.stream(request, sink);
        }
    }

    /**
     * Backend that maps every chat request with the real {@link OpenAiRequestMapper} before answering, as
     * {@link LlamaModelBackend} does, so a request the mapper refuses never reaches generation.
     */
    static final class MappingBackend extends FakeBackend {
        private final OpenAiRequestMapper mapper = new OpenAiRequestMapper();
        volatile String called = "";

        @Override
        public String complete(JsonNode request) {
            mapper.toInferenceParameters(request);
            this.called = "complete";
            return super.complete(request);
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            mapper.toInferenceParameters(request);
            this.called = "stream";
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

    /** Backend whose streams emit one chunk and then fail with an {@link IOException} of their own. */
    static final class IoFailingAfterFirstChunkBackend extends FakeBackend {
        static final String REASON = "generation failed";

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}");
            throw new IOException(REASON);
        }

        @Override
        public void streamCompletions(JsonNode request, ChunkSink sink) throws IOException {
            sink.accept("{\"object\":\"text_completion\",\"choices\":[{\"text\":\"he\",\"finish_reason\":null}]}");
            throw new IOException(REASON);
        }
    }

    /** How long a test backend waits for the client to release it before it gives up. */
    static final long BACKEND_RELEASE_TIMEOUT_SECONDS = 30L;

    /**
     * Waits until the test releases the backend; a release that never comes fails the request
     * instead of hanging the fork.
     */
    static void awaitRelease(CountDownLatch release) {
        try {
            if (!release.await(BACKEND_RELEASE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                throw new IllegalStateException("the client never read a heartbeat, so the backend was not released");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for the client", e);
        }
    }

    /**
     * Backend whose streams emit {@code chunks} chunks, wait until {@code release} is counted down,
     * then fail with {@code failure}.
     */
    static final class ScriptedStreamBackend extends FakeBackend {
        private final int chunks;
        private final CountDownLatch release;
        private final RuntimeException failure;

        ScriptedStreamBackend(int chunks, RuntimeException failure) {
            this(chunks, new CountDownLatch(0), failure);
        }

        ScriptedStreamBackend(int chunks, CountDownLatch release, RuntimeException failure) {
            this.chunks = chunks;
            this.release = release;
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
            awaitRelease(release);
        }
    }

    /** Backend that stalls until {@code release} is counted down, so the server's heartbeat fires during the gap. */
    static final class SlowFakeBackend implements OpenAiBackend {
        private final CountDownLatch release;

        SlowFakeBackend(CountDownLatch release) {
            this.release = release;
        }

        @Override
        public String complete(JsonNode request) {
            return "{\"object\":\"chat.completion\",\"choices\":[]}";
        }

        @Override
        public void stream(JsonNode request, ChunkSink sink) throws IOException {
            awaitRelease(release);
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
