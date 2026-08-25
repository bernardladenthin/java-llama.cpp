// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * End-to-end smoke test over a real model. Self-skips unless a GGUF is provided via
 * {@code -Dnet.ladenthin.llama.model.path=/abs/path/to/model.gguf} (and the native library is on
 * the path), mirroring the core project's model-gated tests, so a model-free checkout stays green.
 */
class JllamaChatModelIntegrationTest {

    private static Path modelPath() {
        Path resolved = TestModelPaths.fromProperty("net.ladenthin.llama.model.path");
        Assumptions.assumeTrue(resolved != null, "model path property not set");
        Assumptions.assumeTrue(Files.exists(resolved), "model file not present: " + resolved);
        return resolved;
    }

    @Test
    void chatReturnsAssistantText() {
        Path model = modelPath();
        try (LlamaModel llama = new LlamaModel(new ModelParameters().setModel(model.toString()))) {
            JllamaChatModel chat = new JllamaChatModel(llama);

            ChatResponse response =
                    chat.chat(
                            ChatRequest.builder()
                                    .messages(UserMessage.from("Reply with the single word: ok"))
                                    .maxOutputTokens(8)
                                    .build());

            assertThat(response.aiMessage(), is(notNullValue()));
            assertThat(response.aiMessage().text(), is(notNullValue()));
        }
    }

    @Test
    void streamingDeliversTokensThenCompletes() throws Exception {
        Path model = modelPath();
        try (LlamaModel llama = new LlamaModel(new ModelParameters().setModel(model.toString()))) {
            JllamaStreamingChatModel streaming = new JllamaStreamingChatModel(llama);
            StringBuilder streamed = new StringBuilder();
            CompletableFuture<ChatResponse> done = new CompletableFuture<>();

            streaming.chat(
                    ChatRequest.builder()
                            .messages(UserMessage.from("Reply with the single word: ok"))
                            .maxOutputTokens(8)
                            .build(),
                    new StreamingChatResponseHandler() {
                        @Override
                        public void onPartialResponse(String partial) {
                            streamed.append(partial);
                        }

                        @Override
                        public void onCompleteResponse(ChatResponse complete) {
                            done.complete(complete);
                        }

                        @Override
                        public void onError(Throwable error) {
                            done.completeExceptionally(error);
                        }
                    });

            ChatResponse complete = done.get(60, TimeUnit.SECONDS);

            // The CI model is Qwen3-0.6B, a reasoning model, and this request budgets 8 output
            // tokens: a run can legitimately spend all of them inside <think> and produce no
            // assistant text at all. StreamingChunkAssembler then routes every token to
            // onPartialThinking and leaves AiMessage.text() null (it only sets text when it
            // accumulated some), while `streamed` stays "" -- which is the "expected \"\" but was
            // null" this assertion used to fail with once the test actually ran.
            //
            // Assert the invariant that holds in both shapes -- the concatenated onPartialResponse
            // fragments are exactly the final text -- and separately that the stream delivered
            // something, so a stream that produces nothing at all still fails.
            String finalText = complete.aiMessage().text() == null
                    ? ""
                    : complete.aiMessage().text();
            assertThat(finalText, is(streamed.toString()));
            assertThat(
                    "stream delivered neither content nor reasoning tokens",
                    !streamed.toString().isEmpty() || complete.aiMessage().thinking() != null,
                    is(true));
        }
    }
}
