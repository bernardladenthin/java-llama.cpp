// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

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
 * End-to-end integration test for the completion-family routes — {@code POST /v1/completions},
 * {@code POST /infill} (fill-in-the-middle) and the Ollama {@code POST /api/generate} (plain + FIM via a
 * {@code suffix}) — against a real model over a real socket. Reuses the CI text model (CodeLlama-7B,
 * {@link TestConstants#MODEL_PATH}), which is FIM-capable (see {@code LlamaModelTest#testGenerateInfill}).
 * Self-skips when the model file is absent. Assertions are structural (valid response envelopes) rather
 * than value-specific. HTTP plumbing is inherited from {@link OpenAiServerTestSupport}.
 */
public class OpenAiServerCompletionIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String MODEL_ID = "completion-local";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws IOException {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(),
                "Text model (CodeLlama-7B) not found, skipping completion server integration test");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers));
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
    public void completionsReturnsTextChoice() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":16,\"prompt\":\"def add(a, b):\\n    return\"}";
        Response response = post(port, "/v1/completions", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("object").asText(), is("text_completion"));
        assertThat(json.path("choices").get(0).path("text").isTextual(), is(true));
    }

    @Test
    public void aCompletionWithoutAPromptIsAnswered400LikeUpstream() throws IOException {
        // oaicompat_completion_params_parse refuses the body natively; the typed rejection reaches the
        // server as InvalidRequestException and is a 400, not the 500 it used to be
        Response response = post(port, "/v1/completions", "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":1}", "");
        assertThat(response.code, is(400));
        assertThat(MAPPER.readTree(response.body).path("error").path("type").asText(), is("invalid_request_error"));
    }

    @Test
    public void infillReturnsContent() throws IOException {
        String body = "{\"input_prefix\":\"def add(a, b):\\n    return \",\"input_suffix\":\"\\n\",\"n_predict\":16}";
        Response response = post(port, "/infill", body, "");
        assertThat(response.code, is(200));
        // The native infill response carries the generated middle under "content".
        assertThat(MAPPER.readTree(response.body).path("content").isTextual(), is(true));
    }

    @Test
    public void ollamaGenerateNonStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":false,"
                + "\"prompt\":\"def add(a, b):\\n    return\",\"options\":{\"num_predict\":16}}";
        Response response = post(port, "/api/generate", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("response").isTextual(), is(true));
        assertThat(json.path("done").asBoolean(), is(true));
    }

    @Test
    public void ollamaGenerateWithSuffixUsesInfill() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":false,"
                + "\"prompt\":\"def add(a, b):\\n    return \",\"suffix\":\"\\n\",\"options\":{\"num_predict\":16}}";
        Response response = post(port, "/api/generate", body, "");
        assertThat(response.code, is(200));
        assertThat(MAPPER.readTree(response.body).path("response").isTextual(), is(true));
    }
}
