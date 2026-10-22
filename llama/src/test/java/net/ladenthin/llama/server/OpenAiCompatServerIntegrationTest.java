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
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end integration test for {@link OpenAiCompatServer} against a real model served over a real
 * socket. Reuses the Qwen3-0.6B GGUF that the CI pipeline already downloads as the reasoning model
 * ({@link TestConstants#REASONING_MODEL_PATH}); it is instruct-tuned (has a chat template) and one of
 * llama.cpp's better tool-calling families, so no extra download is needed. Self-skips when the model
 * file is absent (e.g. a local checkout without models), so it never breaks a model-free run.
 *
 * <p>Assertions are deliberately structural (valid OpenAI shapes, stream terminates) rather than
 * content-specific — a 0.6B model's exact wording and whether it elects to call a tool are not
 * deterministic. The deterministic chunk/tool-call plumbing is covered by
 * {@link OpenAiCompatServerHttpTest} with a fake backend. HTTP request plumbing is inherited from
 * {@link OpenAiServerTestSupport}.
 */
public class OpenAiCompatServerIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String MODEL_ID = "qwen3-local";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws IOException {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model (Qwen3-0.6B) not found, skipping OpenAI server integration test");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setCtxSize(1024)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .setParallel(2));
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

    /**
     * The token-count routes count the prompt a generation of the same request evaluates: the chat
     * template rendered over the request, tokenized as the generation tokenizes it. upstream serves all
     * three; they answered 404 here.
     */
    @Test
    public void theTokenCountRoutesCountThePromptAGenerationEvaluates() throws IOException {
        String messages = "[{\"role\":\"user\",\"content\":\"Count the tokens of this sentence, please.\"}]";
        String chat =
                "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":1,\"temperature\":0,\"messages\":" + messages + "}";
        int promptTokens = MAPPER.readTree(post(port, "/v1/chat/completions", chat, "").body)
                .path("usage")
                .path("prompt_tokens")
                .asInt(-1);
        assertThat(promptTokens, greaterThan(0));

        Response chatCount = post(port, "/v1/chat/completions/input_tokens", chat, "");
        assertThat(chatCount.body, chatCount.code, is(200));
        assertThat(MAPPER.readTree(chatCount.body).path("input_tokens").asInt(-1), is(promptTokens));

        String anthropic =
                "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":1,\"temperature\":0,\"messages\":" + messages + "}";
        JsonNode usage =
                MAPPER.readTree(post(port, "/v1/messages", anthropic, "").body).path("usage");
        int anthropicPrompt = usage.path("input_tokens").asInt(-1)
                + usage.path("cache_read_input_tokens").asInt(0);
        Response anthropicCount = post(port, "/v1/messages/count_tokens", anthropic, "");
        assertThat(anthropicCount.body, anthropicCount.code, is(200));
        assertThat(MAPPER.readTree(anthropicCount.body).path("input_tokens").asInt(-1), is(anthropicPrompt));

        String responses = "{\"model\":\"" + MODEL_ID + "\",\"max_output_tokens\":1,\"temperature\":0,"
                + "\"input\":\"Count the tokens of this sentence, please.\"}";
        int responsesPrompt = MAPPER.readTree(post(port, "/v1/responses", responses, "").body)
                .path("usage")
                .path("input_tokens")
                .asInt(-1);
        Response responsesCount = post(port, "/v1/responses/input_tokens", responses, "");
        assertThat(responsesCount.body, responsesCount.code, is(200));
        assertThat(MAPPER.readTree(responsesCount.body).path("input_tokens").asInt(-1), is(responsesPrompt));
    }

    @Test
    public void nonStreamingChatReturnsValidCompletion() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":16,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}";
        Response response = post(port, "/v1/chat/completions", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("object").asText(), is("chat.completion"));
        assertThat(json.path("choices").size(), greaterThan(0));
        assertThat(json.path("choices").get(0).path("message").path("role").asText(), is("assistant"));
    }

    @Test
    public void streamingChatEmitsChunksAndDone() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":true,\"max_tokens\":16,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}";
        Response response = post(port, "/v1/chat/completions", body, "");
        assertThat(response.code, is(200));
        assertThat(response.body, containsString("chat.completion.chunk"));
        assertThat(response.body, containsString("data: [DONE]"));
    }

    @Test
    public void toolRequestRoundTripsThroughTheJinjaPath() throws IOException {
        // Forwards an OpenAI tools array. Jinja rendering (which is what lets the native parser
        // apply Qwen3's tool-aware template) comes from how the model was loaded, not from the
        // request -- upstream reads no "use_jinja" request key. We assert the request is accepted
        // and returns a structurally valid OpenAI message (content and/or tool_calls) — not that
        // this tiny model elects to call.
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":48,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                + "\"description\":\"Get the weather for a city\",\"parameters\":{\"type\":\"object\","
                + "\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}";
        Response response = post(port, "/v1/chat/completions", body, "");
        assertThat(response.code, is(200));
        JsonNode message = MAPPER.readTree(response.body).path("choices").get(0).path("message");
        assertThat(message.isObject(), is(true));
    }

    @Test
    public void modelsEndpointAdvertisesTheServedModel() throws IOException {
        Response response = get(port, "/v1/models", "");
        assertThat(response.code, is(200));
        assertThat(response.body, containsString(MODEL_ID));
    }

    // ----- alternative protocol surfaces (same Qwen3 model, structural assertions only) -----

    @Test
    public void ollamaChatNonStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":false,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],"
                + "\"options\":{\"num_predict\":16}}";
        Response response = post(port, "/api/chat", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("done").asBoolean(), is(true));
        assertThat(json.path("message").path("role").asText(), is("assistant"));
    }

    @Test
    public void ollamaChatStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],"
                + "\"options\":{\"num_predict\":16}}";
        Response response = post(port, "/api/chat", body, "");
        assertThat(response.code, is(200));
        // NDJSON terminates with a done line regardless of the model's wording.
        assertThat(response.body, containsString("\"done\":true"));
    }

    @Test
    public void ollamaDiscoveryEndpointsRespond() throws IOException {
        assertThat(get(port, "/api/version", "").code, is(200));
        assertThat(get(port, "/api/tags", "").body, containsString(MODEL_ID));
        Response show = post(port, "/api/show", "{\"model\":\"" + MODEL_ID + "\"}", "");
        assertThat(show.code, is(200));
        assertThat(show.body, containsString("capabilities"));
    }

    @Test
    public void anthropicMessagesNonStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":16,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}";
        Response response = post(port, "/v1/messages", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("type").asText(), is("message"));
        assertThat(json.path("role").asText(), is("assistant"));
    }

    /**
     * The Anthropic route asks for upstream's debug object only for the matched stop word; with the
     * projection it carries just that word, not the rendered prompt and the generation settings it
     * used to bring through JNI with every answer.
     */
    @Test
    public void theDebugObjectTheAnthropicRouteAsksForCarriesOnlyTheStopWord() throws IOException {
        JsonNode answer = MAPPER.readTree(model.handleChatCompletions("{\"max_tokens\":4,\"verbose\":true,"
                + "\"response_fields\":[\"stopping_word\"],"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}"));
        JsonNode debug = answer.path("__verbose");
        assertThat(answer.toString(), debug.isObject(), is(true));
        java.util.List<String> keys = new java.util.ArrayList<>();
        debug.fieldNames().forEachRemaining(keys::add);
        assertThat(keys, is(java.util.Collections.singletonList("stopping_word")));
    }

    /**
     * {@code response_fields} names nested values with a slash, as upstream's
     * {@code json_get_nested_values} splits them ({@code server-common.cpp}, b11639), and the projected
     * key is the whole path. A dotted path is one key the result does not have, which upstream drops
     * without an error.
     */
    @Test
    public void aNestedResponseFieldIsNamedWithASlashAndKeepsThePathAsItsKey() throws IOException {
        JsonNode debug = MAPPER.readTree(model.handleChatCompletions("{\"max_tokens\":4,\"verbose\":true,"
                        + "\"response_fields\":[\"timings/predicted_n\"],"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}"))
                .path("__verbose");
        assertThat(debug.toString(), debug.isObject(), is(true));
        java.util.List<String> keys = new java.util.ArrayList<>();
        debug.fieldNames().forEachRemaining(keys::add);
        assertThat(keys, is(java.util.Collections.singletonList("timings/predicted_n")));
        assertThat(debug.toString(), debug.path("timings/predicted_n").isIntegralNumber(), is(true));
        assertThat(debug.path("timings/predicted_n").asInt(), greaterThan(0));

        JsonNode dotted = MAPPER.readTree(model.handleChatCompletions("{\"max_tokens\":4,\"verbose\":true,"
                        + "\"response_fields\":[\"timings.predicted_n\"],"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}"))
                .path("__verbose");
        assertThat(dotted.toString(), dotted.isObject(), is(true));
        assertThat(dotted.toString(), dotted.size(), is(0));
    }

    @Test
    public void anthropicMessagesStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":true,\"max_tokens\":16,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}";
        Response response = post(port, "/v1/messages", body, "");
        assertThat(response.code, is(200));
        assertThat(response.body, containsString("event: message_start"));
        assertThat(response.body, containsString("event: message_stop"));
    }

    /**
     * An answer that cannot fit into {@code max_output_tokens} ends with finish_reason "length", which the
     * Responses API reports as {@code incomplete} with reason {@code max_output_tokens} (F229); it was
     * reported as {@code completed}.
     */
    @Test
    public void responsesNonStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_output_tokens\":4,"
                + "\"input\":\"Count from 1 to 100, one number per line.\"}";
        Response response = post(port, "/v1/responses", body, "");
        assertThat(response.code, is(200));
        JsonNode json = MAPPER.readTree(response.body);
        assertThat(json.path("object").asText(), is("response"));
        assertThat(json.path("status").asText(), is("incomplete"));
        assertThat(json.path("incomplete_details").path("reason").asText(), is("max_output_tokens"));
    }

    @Test
    public void responsesStreamingRoundTrip() throws IOException {
        String body = "{\"model\":\"" + MODEL_ID + "\",\"stream\":true,\"max_output_tokens\":4,"
                + "\"input\":\"Count from 1 to 100, one number per line.\"}";
        Response response = post(port, "/v1/responses", body, "");
        assertThat(response.code, is(200));
        assertThat(response.body, containsString("event: response.created"));
        assertThat(response.body, containsString("event: response.incomplete"));
        assertThat(response.body, containsString("\"reason\":\"max_output_tokens\""));
    }

    @Test
    public void propsEndpointReportsContextLength() throws IOException {
        Response response = get(port, "/props", "");
        assertThat(response.code, is(200));
        assertThat(response.body, containsString("n_ctx"));
    }
}
