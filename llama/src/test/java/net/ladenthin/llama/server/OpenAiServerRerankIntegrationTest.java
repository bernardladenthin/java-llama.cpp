// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.oneOf;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end integration test for the {@code POST /v1/rerank} route against a real model loaded in
 * reranking mode ({@code enableReranking()}), served over a real socket. Reuses the CI reranking model
 * (jina-reranker, {@link TestConstants#RERANKING_MODEL_PATH}). Self-skips when the model file is absent.
 * Assertions are structural (sorted {@code results}/{@code data} of {@code index}+{@code relevance_score})
 * and check the {@code top_n} cap; exact scores are model-dependent. The other cases pin what only the
 * real backend can show: a negative {@code top_n} refused before the model evaluates anything, the
 * {@code usage} of a {@code top_n: 0} request summed from the real {@code tokens_evaluated}, and the TEI
 * request answered as a bare array. HTTP plumbing is inherited from {@link OpenAiServerTestSupport}.
 */
public class OpenAiServerRerankIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String MODEL_ID = "rerank-local";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws IOException {
        Assumptions.assumeTrue(
                new File(TestConstants.RERANKING_MODEL_PATH).exists(),
                "Reranking model (jina-reranker) not found, skipping rerank server integration test");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.RERANKING_MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .enableReranking()
                .skipWarmup());
        server = new OpenAiCompatServer(
                        model,
                        OpenAiServerConfig.builder().port(0).modelId(MODEL_ID).build())
                .start();
        port = server.getPort();
    }

    @AfterAll
    public static void tearDown() {
        if (server != null) {
            server.close();
        }
        if (model != null) {
            model.close();
        }
    }

    @Test
    public void rerankReturnsScoredResultsCappedByTopN() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"query\":\"Machine learning is\","
                + "\"documents\":[\"A machine applies forces to perform an action.\","
                + "\"Machine learning is a field of artificial intelligence.\","
                + "\"Paris is the capital of France.\"],\"top_n\":2}";
        Response response = post(port, "/v1/rerank", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("object").asText(), is("list"));
        JsonNode results = json.path("results");
        assertThat(results.isArray(), is(true));
        assertThat(results.size(), greaterThan(0));
        assertThat(results.size(), lessThanOrEqualTo(2)); // top_n cap
        assertThat(results.get(0).path("index").isInt(), is(true));
        assertThat(results.get(0).path("relevance_score").isNumber(), is(true));
        // `data` is an alias of `results` for Continue (#6478).
        assertThat(json.path("data").size(), is(results.size()));
    }

    @Test
    public void negativeTopNIsRejectedBeforeAnythingIsEvaluated() throws IOException {
        // The status alone cannot show the order: toResponse reads top_n again after the native call, so
        // the route answers 400 either way. The prompt-token counter can: it stays put only when the
        // backend refuses the request before handleRerank evaluates the documents.
        long processedBefore = model.getMetricsTyped().getCumulativeProcessedPromptTokens();
        String body = "{\"model\":\"" + MODEL_ID + "\",\"query\":\"Machine learning is\","
                + "\"documents\":[\"Machine learning is a field of artificial intelligence.\"],\"top_n\":-1}";
        Response response = post(port, "/v1/rerank", body, "");
        assertThat(response.code, is(400));
        JsonNode error = MAPPER.readTree(response.body).path("error");
        assertThat(error.path("type").asText(), is("invalid_request_error"));
        assertThat(error.path("message").asText(), containsString("top_n"));
        assertThat(model.getMetricsTyped().getCumulativeProcessedPromptTokens(), is(processedBefore));
    }

    @Test
    public void topNZeroAnswersNoResultsButUsageCountsTheEvaluatedTokens() throws IOException {
        // upstream's format_response_rerank sums tokens_evaluated into usage before it keeps
        // min(top_n, n) entries, so top_n 0 still reports what was evaluated. The C++ test feeds a fake
        // result; this is the real server_task_result_rerank carrying tokens_evaluated into usage.
        String body = "{\"model\":\"" + MODEL_ID + "\",\"query\":\"Machine learning is\","
                + "\"documents\":[\"A machine applies forces to perform an action.\","
                + "\"Paris is the capital of France.\"],\"top_n\":0}";
        Response response = post(port, "/v1/rerank", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("results").size(), is(0));
        assertThat(json.path("data").size(), is(0));
        JsonNode usage = json.path("usage");
        assertThat(usage.path("prompt_tokens").asInt(-1), greaterThan(0));
        assertThat(
                usage.path("total_tokens").asInt(-1),
                is(usage.path("prompt_tokens").asInt(-1)));
    }

    @Test
    public void teiRequestIsAnsweredAsABareArray() throws IOException {
        // A request with "texts" is the Hugging Face text-embeddings-inference shape, which upstream
        // answers as a bare [{index, score[, text]}] array; return_text adds the document looked up by
        // index (server-common.cpp, format_response_rerank).
        String[] texts = {"A machine applies forces to perform an action.", "Paris is the capital of France."};
        String body = "{\"query\":\"Machine learning is\",\"texts\":[\"" + texts[0] + "\",\"" + texts[1]
                + "\"],\"return_text\":true}";
        Response response = post(port, "/v1/rerank", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.isArray(), is(true));
        assertThat(json.size(), is(2));
        JsonNode first = json.get(0);
        assertThat(first.has("index"), is(true));
        assertThat(first.has("score"), is(true));
        assertThat(first.has("relevance_score"), is(false));
        assertThat(first.path("text").asText(), is(oneOf(texts)));
        assertThat(first.path("text").asText(), is(texts[first.path("index").asInt()]));
    }
}
