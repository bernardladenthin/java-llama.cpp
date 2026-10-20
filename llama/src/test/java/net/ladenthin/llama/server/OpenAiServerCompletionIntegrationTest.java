// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end integration test for the completion-family routes — {@code POST /v1/completions},
 * {@code POST /infill} (fill-in-the-middle) and the Ollama {@code POST /api/generate} (chat template,
 * {@code raw} and FIM via a {@code suffix}) — against a real model over a real socket. Reuses the CI text model (CodeLlama-7B,
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
                // two slots, so that "n": 2 passes upstream's schema and reaches the engine's own refusal
                .setParallel(2)
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

    // The streamed /v1/completions went through the Java generator and built its own chunks: no usage,
    // no probabilities, and a prompt given as a token array refused while the same request without
    // "stream" was accepted. It now streams upstream's own text_completion chunks (F186).
    @Test
    public void aStreamedCompletionIsUpstreamsStream() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":true,\"max_tokens\":4,\"logprobs\":2,"
                + "\"prompt\":[72,105]}";
        Response response = post(port, "/v1/completions", body, "");
        assertThat(response.body, response.code, is(200));
        java.util.List<JsonNode> chunks = new java.util.ArrayList<>();
        for (String line : response.body.split("\n")) {
            if (line.startsWith("data: ") && !line.equals("data: [DONE]")) {
                chunks.add(MAPPER.readTree(line.substring("data: ".length())));
            }
        }
        assertThat(response.body, response.body.contains("data: [DONE]"), is(true));
        assertThat(response.body, chunks.size() >= 2, is(true));
        java.util.Set<String> ids = new java.util.HashSet<>();
        boolean probabilities = false;
        for (JsonNode chunk : chunks) {
            assertThat(chunk.path("object").asText(), is("text_completion"));
            ids.add(chunk.path("id").asText());
            probabilities |= chunk.path("choices")
                    .path(0)
                    .path("logprobs")
                    .path("content")
                    .isArray();
        }
        assertThat("one id per stream: " + ids, ids.size(), is(1));
        assertThat(ids.iterator().next().isEmpty(), is(false));
        assertThat("the chunks carry the probabilities asked for", probabilities, is(true));
        JsonNode last = chunks.get(chunks.size() - 1);
        assertThat(last.path("usage").path("prompt_tokens").asInt(), is(2));
        assertThat(last.path("usage").path("completion_tokens").asInt() > 0, is(true));
        assertThat(last.path("choices").path(0).path("finish_reason").isTextual(), is(true));
    }

    /**
     * Several completions of one prompt ({@code "n": 2}) are refused, not answered with one: upstream
     * creates the other completions as child tasks, which the engine does not. The streamed route lost
     * this refusal when it moved onto the native OAI path (F186) and got it back in the engine; the
     * blocking route forwards the body to the same engine.
     */
    @Test
    public void aCompletionAskingForSeveralChoicesIsAnswered400OnBothRoutes() throws IOException {
        String streamed =
                "{\"model\":\"" + MODEL_ID + "\",\"stream\":true,\"max_tokens\":2,\"n\":2," + "\"prompt\":\"Hi\"}";
        Response streamedResponse = post(port, "/v1/completions", streamed, "");
        assertThat(streamedResponse.body, streamedResponse.code, is(400));
        assertThat(streamedResponse.body, streamedResponse.body.contains("'n'"), is(true));

        String blocking = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":2,\"n\":2,\"prompt\":\"Hi\"}";
        Response blockingResponse = post(port, "/v1/completions", blocking, "");
        assertThat(blockingResponse.body, blockingResponse.code, is(400));
    }

    /** The same refusal for a caller of the model itself, on every native completion path. */
    @Test
    public void severalChoicesAreRefusedOnTheModelsOwnCompletionAndChatPaths() {
        assertThrows(
                InvalidRequestException.class,
                () -> model.handleCompletionsOai("{\"prompt\":\"Hi\",\"n\":2,\"max_tokens\":2}"));
        assertThrows(
                InvalidRequestException.class,
                () -> model.streamCompletionsOai("{\"prompt\":\"Hi\",\"n\":2,\"max_tokens\":2}", chunk -> {}));
        assertThrows(
                InvalidRequestException.class,
                () -> model.handleChatCompletions(
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"n\":2,\"max_tokens\":2}"));
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
    public void ollamaGenerateRawRoundTrip() throws IOException {
        // "raw": true sends the bare prompt to the completion endpoint; without it the request goes
        // through the chat template, as Ollama does (the test above).
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":false,\"raw\":true,"
                + "\"prompt\":\"def add(a, b):\\n    return\",\"options\":{\"num_predict\":16}}";
        Response response = post(port, "/api/generate", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("response").isTextual(), is(true));
        assertThat(json.path("done").asBoolean(), is(true));
        assertThat(json.path("eval_count").asInt(-1) > 0, is(true));
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
