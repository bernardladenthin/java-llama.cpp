// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Real-model coverage for the langchain4j tool-calling and JSON-mode bridges. Self-skips unless a
 * tool-capable GGUF is provided via {@code -Dnet.ladenthin.llama.langchain4j.tool.model} (the CI
 * job passes the same Qwen2.5-Instruct model the core {@code ToolCallingIntegrationTest} uses),
 * mirroring {@code JllamaChatModelIntegrationTest}'s self-skip pattern. The model is loaded with
 * jinja enabled — required for the upstream tool-call chat-template path.
 */
class JllamaToolCallingIntegrationTest {

    /** System property naming the tool-capable GGUF this test loads. */
    static final String PROP_TOOL_MODEL = "net.ladenthin.llama.langchain4j.tool.model";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static LlamaModel model;

    @BeforeAll
    static void loadModel() {
        Path resolved = TestModelPaths.fromProperty(PROP_TOOL_MODEL);
        Assumptions.assumeTrue(resolved != null, "tool model path property not set");
        Assumptions.assumeTrue(Files.exists(resolved), "model file not present: " + resolved);
        String path = resolved.toString();
        model = new LlamaModel(new ModelParameters()
                .setModel(path)
                .setCtxSize(8192)
                .setFit(false)
                .enableJinja());
    }

    @AfterAll
    static void closeModel() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    void requiredToolCallComesBackAsToolExecutionRequest() throws Exception {
        JllamaChatModel chat = new JllamaChatModel(model);

        ChatResponse response = chat.chat(ChatRequest.builder()
                .messages(UserMessage.from("Report success using the test tool."))
                .toolSpecifications(dev.langchain4j.agent.tool.ToolSpecification.builder()
                        .name("test")
                        .description("Report success")
                        .parameters(JsonObjectSchema.builder()
                                .addProperty("success", new JsonBooleanSchema())
                                .required("success")
                                .build())
                        .build())
                .toolChoice(ToolChoice.REQUIRED)
                .maxOutputTokens(512)
                .temperature(0.0)
                .build());

        assertThat(response.aiMessage().hasToolExecutionRequests(), is(true));
        ToolExecutionRequest request =
                response.aiMessage().toolExecutionRequests().get(0);
        assertThat(request.name(), is("test"));
        // The grammar enforces the required field, so the arguments must be JSON carrying it.
        JsonNode arguments = MAPPER.readTree(request.arguments());
        assertThat(arguments.has("success"), is(true));
        assertThat(response.finishReason(), is(FinishReason.TOOL_EXECUTION));
    }

    @Test
    void streamedRequiredToolCallComesBackAsToolExecutionRequest() throws Exception {
        JllamaStreamingChatModel streaming = new JllamaStreamingChatModel(model);
        java.util.concurrent.CompletableFuture<ChatResponse> done = new java.util.concurrent.CompletableFuture<>();

        streaming.chat(
                ChatRequest.builder()
                        .messages(UserMessage.from("Report success using the test tool."))
                        .toolSpecifications(dev.langchain4j.agent.tool.ToolSpecification.builder()
                                .name("test")
                                .description("Report success")
                                .parameters(JsonObjectSchema.builder()
                                        .addProperty("success", new JsonBooleanSchema())
                                        .required("success")
                                        .build())
                                .build())
                        .toolChoice(ToolChoice.REQUIRED)
                        .maxOutputTokens(512)
                        .temperature(0.0)
                        .build(),
                new dev.langchain4j.model.chat.response.StreamingChatResponseHandler() {
                    @Override
                    public void onPartialResponse(String partialResponse) {}

                    @Override
                    public void onCompleteResponse(ChatResponse completeResponse) {
                        done.complete(completeResponse);
                    }

                    @Override
                    public void onError(Throwable error) {
                        done.completeExceptionally(error);
                    }
                });

        ChatResponse response = done.get(120, java.util.concurrent.TimeUnit.SECONDS);
        assertThat(response.aiMessage().hasToolExecutionRequests(), is(true));
        ToolExecutionRequest request =
                response.aiMessage().toolExecutionRequests().get(0);
        assertThat(request.name(), is("test"));
        assertThat(MAPPER.readTree(request.arguments()).has("success"), is(true));
        assertThat(response.finishReason(), is(FinishReason.TOOL_EXECUTION));
    }

    @Test
    void jsonSchemaResponseFormatYieldsConformingJson() throws Exception {
        JllamaChatModel chat = new JllamaChatModel(model);

        ChatResponse response = chat.chat(ChatRequest.builder()
                .messages(UserMessage.from("The person is called Alice. Extract the person."))
                .responseFormat(ResponseFormat.builder()
                        .type(ResponseFormatType.JSON)
                        .jsonSchema(JsonSchema.builder()
                                .name("Person")
                                .rootElement(JsonObjectSchema.builder()
                                        .addProperty("name", new JsonStringSchema())
                                        .required("name")
                                        .build())
                                .build())
                        .build())
                .maxOutputTokens(256)
                .temperature(0.0)
                .build());

        // The native json_schema grammar constraint forces conforming output.
        JsonNode parsed = MAPPER.readTree(response.aiMessage().text());
        assertThat(parsed.path("name").isTextual(), is(true));
        assertThat(response.aiMessage(), is(notNullValue()));
    }
}
