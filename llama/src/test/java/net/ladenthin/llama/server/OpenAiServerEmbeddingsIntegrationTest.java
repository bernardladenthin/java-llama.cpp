// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.args.PoolingType;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end integration test for the {@code POST /v1/embeddings} route against a real model loaded in
 * embedding mode ({@code enableEmbedding()}), served over a real socket. Reuses the CI text model
 * (CodeLlama-7B, {@link TestConstants#MODEL_PATH}) — the same model {@code LlamaEmbeddingsTest} drives in
 * embedding mode. Self-skips when the model file is absent (e.g. a local checkout without models), so it
 * never breaks a model-free run. Assertions are structural (valid OpenAI embeddings shape) rather than
 * value-specific. HTTP plumbing is inherited from {@link OpenAiServerTestSupport}.
 */
public class OpenAiServerEmbeddingsIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String MODEL_ID = "embed-local";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws IOException {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(),
                "Text model (CodeLlama-7B) not found, skipping embeddings server integration test");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        // The OpenAI /v1/embeddings path (oaicompat=true) requires a real pooling type: llama.cpp rejects
        // pooling type NONE there ("pooling type 'none' is not OAI compatible"). CodeLlama's GGUF reports
        // pooling = -1 (NONE), so an explicit MEAN pooling is set here — MEAN/LAST both produce a single
        // pooled sentence vector for decoder-only models (see LlamaEmbeddingsTest). enableEmbedding()
        // alone (as the low-level LlamaModel#embed path uses) leaves pooling NONE and would 500 here.
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .enableEmbedding()
                .setPoolingType(PoolingType.MEAN));
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
    public void embeddingsReturnsAVector() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"input\":\"hello world\"}";
        Response response = post(port, "/v1/embeddings", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("object").asText(), is("list"));
        JsonNode first = json.path("data").get(0);
        assertThat(first.path("object").asText(), is("embedding"));
        assertThat(first.path("embedding").isArray(), is(true));
        assertThat(first.path("embedding").size(), greaterThan(0));
    }

    @Test
    public void anEmptyInputArrayIsAnswered400LikeUpstream() throws IOException {
        // llama.cpp's own server answers this with 400 since b11337 (#29060); the JNI layer throws the
        // typed InvalidRequestException for it, which the server maps to 400 rather than 500
        String body = "{\"model\":\"" + MODEL_ID + "\",\"input\":[]}";
        Response response = post(port, "/v1/embeddings", body, "");
        assertThat(response.code, is(400));
        JsonNode error = MAPPER.readTree(response.body).path("error");
        assertThat(error.path("type").asText(), is("invalid_request_error"));
        assertThat(error.path("message").asText(), containsString("must not be empty"));
    }

    @Test
    public void anUnknownEncodingFormatIsAnswered400() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"input\":\"hello\",\"encoding_format\":\"hex\"}";
        Response response = post(port, "/v1/embeddings", body, "");
        assertThat(response.code, is(400));
        assertThat(MAPPER.readTree(response.body).path("error").path("type").asText(), is("invalid_request_error"));
    }

    @Test
    public void embeddingsReachableWithoutV1Prefix() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"input\":\"alias check\"}";
        Response response = post(port, "/embeddings", body, "");
        assertThat(response.code, is(200));
        assertThat(
                MAPPER.readTree(response.body)
                        .path("data")
                        .get(0)
                        .path("embedding")
                        .isArray(),
                is(true));
    }
}
