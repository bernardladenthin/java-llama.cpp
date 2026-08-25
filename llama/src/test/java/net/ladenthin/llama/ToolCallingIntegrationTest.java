// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.parameters.ChatRequest;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ToolCall;
import net.ladenthin.llama.value.ToolDefinition;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Real-model coverage for the blocking and streaming tool-call paths. */
public class ToolCallingIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final ToolDefinition TEST_TOOL = new ToolDefinition(
            "test",
            "",
            "{\"type\":\"object\",\"properties\":{\"success\":{\"type\":\"boolean\",\"const\":true}},"
                    + "\"required\":[\"success\"]}");

    private static LlamaModel model;

    @BeforeAll
    public static void loadModel() {
        String modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_TOOL_MODEL_PATH, TestConstants.DEFAULT_TOOL_MODEL_PATH);
        Assumptions.assumeTrue(new File(modelPath).exists(), "Tool-calling model missing: " + modelPath);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, 0);
        ModelParameters parameters = new ModelParameters()
                .setModel(modelPath)
                .setCtxSize(8192)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .enableJinja();
        if (gpuLayers == 0) {
            parameters.setDevices("none");
        }
        model = new LlamaModel(parameters);
    }

    @AfterAll
    public static void closeModel() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    public void requiredToolCallIsParsedFromBlockingResponse() throws IOException {
        ChatResponse response = model.chat(toolRequest());

        List<ToolCall> calls = response.getFirstMessage().orElseThrow().getToolCalls();
        assertThat(calls, hasSize(1));
        assertThat(calls.get(0).getName(), is("test"));
        assertThat(
                MAPPER.readTree(calls.get(0).getArgumentsJson()).path("success").asBoolean(), is(true));
    }

    @Test
    public void requiredToolCallIsParsedFromStreamingResponse() throws IOException {
        ChatRequest request = toolRequest();
        InferenceParameters params = request.applyCustomizer(InferenceParameters.empty()
                .withMessagesJson(request.buildMessagesJson())
                .withToolsJson(request.buildToolsJson().orElseThrow())
                .withToolChoice(request.getToolChoice().orElseThrow())
                .withParallelToolCalls(request.getParallelToolCalls().orElseThrow())
                .withUseChatTemplate(true));
        List<String> chunks = new ArrayList<String>();

        model.streamChatCompletion(params, chunks::add);

        StringBuilder name = new StringBuilder();
        StringBuilder arguments = new StringBuilder();
        for (String chunk : chunks) {
            JsonNode toolCalls =
                    MAPPER.readTree(chunk).path("choices").path(0).path("delta").path("tool_calls");
            if (!toolCalls.isArray()) {
                continue;
            }
            for (JsonNode call : toolCalls) {
                JsonNode function = call.path("function");
                if (function.path("name").isTextual()) {
                    name.append(function.path("name").asText());
                }
                if (function.path("arguments").isTextual()) {
                    arguments.append(function.path("arguments").asText());
                }
            }
        }

        assertThat(name.toString(), is("test"));
        assertThat(MAPPER.readTree(arguments.toString()).path("success").asBoolean(), is(true));
    }

    private static ChatRequest toolRequest() {
        return ChatRequest.empty()
                .appendMessage("system", "You are a coding assistant.")
                .appendMessage("user", "Write an example")
                .appendTool(TEST_TOOL)
                .withToolChoice("required")
                .withParallelToolCalls(Boolean.FALSE)
                .withInferenceCustomizer(params -> params.withNPredict(512)
                        .withTemperature(0.0f)
                        .withTopK(1)
                        .withTopP(1.0f));
    }
}
