// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
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
 * End-to-end tool-calling integration test for {@link OpenAiCompatServer}, driven over a real socket
 * against the Qwen2.5-1.5B-Instruct tool model — a stronger tool-calling family than the 0.6B reasoning
 * model used by {@link OpenAiCompatServerIntegrationTest}, so it actually emits tool calls. The model is
 * resolved from {@link TestConstants#PROP_TOOL_MODEL_PATH} (CI sets it; otherwise
 * {@link TestConstants#DEFAULT_TOOL_MODEL_PATH}) and the test self-skips when the GGUF is absent, so a
 * model-free checkout is never broken.
 *
 * <p>Where {@link OpenAiCompatServerIntegrationTest}'s tool test can only assert a structurally valid
 * message (the 0.6B model may not elect to call), these force a call via {@code tool_choice:"required"}
 * so the native grammar must emit one — letting us assert, deterministically, that the HTTP server
 * returns a well-formed OpenAI {@code tool_calls} array with {@code arguments} carried as a JSON
 * <em>string</em> (the agentic-client invariant, llama.cpp #20198), and that #244's
 * {@code parallel_tool_calls} flag travels HTTP &rarr; mapper &rarr; native without breaking the request.
 */
public class OpenAiServerToolCallingIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String MODEL_ID = "qwen25-tools";

    /** A trivial single-required-argument function; {@code tool_choice:"required"} forces a call. */
    private static final String TOOLS = "\"tools\":[{\"type\":\"function\",\"function\":{"
            + "\"name\":\"get_weather\",\"description\":\"Get the weather for a city\","
            + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
            + "\"required\":[\"city\"]}}}]";

    private static LlamaModel model;
    private static OpenAiCompatServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws IOException {
        String modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_TOOL_MODEL_PATH, TestConstants.DEFAULT_TOOL_MODEL_PATH);
        Assumptions.assumeTrue(
                new File(modelPath).exists(),
                "Tool-calling model (Qwen2.5-1.5B) not found, skipping server tool-calling test: " + modelPath);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(modelPath)
                .setCtxSize(4096)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .setParallel(1));
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
    public void requiredToolChoiceReturnsWellFormedToolCalls() throws IOException {
        // tool_choice=required forces a function call, so a capable model deterministically returns a
        // structurally valid OpenAI tool_calls array regardless of its exact wording.
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":64,\"tool_choice\":\"required\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],"
                + TOOLS + "}";
        Response response = post(port, "/v1/chat/completions", body, "");
        assertThat(response.code, is(200));
        JsonNode toolCalls = MAPPER.readTree(response.body)
                .path("choices")
                .get(0)
                .path("message")
                .path("tool_calls");
        assertThat(toolCalls.isArray(), is(true));
        assertThat(toolCalls.size(), greaterThanOrEqualTo(1));
        JsonNode function = toolCalls.get(0).path("function");
        assertThat(function.path("name").asText(), is("get_weather"));
        // arguments must be a JSON *string* (not an inlined object) — the agentic-client invariant.
        assertThat(function.path("arguments").isTextual(), is(true));
        assertThat(MAPPER.readTree(function.path("arguments").asText()).isObject(), is(true));
    }

    @Test
    public void parallelToolCallsFalseIsAcceptedEndToEnd() throws IOException {
        // parallel_tool_calls=false must flow HTTP -> OpenAiRequestMapper -> native without breaking the
        // request; tool_choice=required still yields a well-formed tool call.
        String body = "{\"model\":\"" + MODEL_ID + "\",\"max_tokens\":64,\"tool_choice\":\"required\","
                + "\"parallel_tool_calls\":false,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],"
                + TOOLS + "}";
        Response response = post(port, "/v1/chat/completions", body, "");
        assertThat(response.code, is(200));
        JsonNode toolCalls = MAPPER.readTree(response.body)
                .path("choices")
                .get(0)
                .path("message")
                .path("tool_calls");
        assertThat(toolCalls.isArray(), is(true));
        assertThat(toolCalls.size(), greaterThanOrEqualTo(1));
    }
}
