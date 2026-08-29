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

    /**
     * Generation budget for both tests, matching the core module's {@code ReasoningBudgetTest}
     * ({@code N_PREDICT = 1500}) for the same model.
     *
     * <p>Qwen3-0.6B is a reasoning model: it spends its first few hundred tokens inside
     * {@code <think>} and only then emits assistant content, so a budget that does not clear the
     * thinking block yields an <em>empty</em> answer rather than a short one. 320 was tried and is
     * right on the boundary — in one CI run (33109360197) the blocking test finished thinking at 267
     * tokens and passed while the streaming test consumed all 320 inside {@code <think>} and failed,
     * from the same prompt against the same model. The adapter exposes no reasoning-budget knob, so
     * the output budget is the only lever.
     *
     * <p>This is a cap, not a target: a normal run stops around 270–340 tokens, so raising it costs
     * nothing except in the pathological case it exists to absorb.
     */
    private static final int MAX_OUTPUT_TOKENS = 1500;

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
                                    // See MAX_OUTPUT_TOKENS: too small a budget yields an EMPTY
                                    // assistant text, which a bare notNullValue() would accept.
                                    .maxOutputTokens(MAX_OUTPUT_TOKENS)
                                    .build());

            assertThat(response.aiMessage(), is(notNullValue()));
            assertThat(response.aiMessage().text(), is(notNullValue()));
            assertThat("the model must produce assistant text, not only a thinking block",
                    response.aiMessage().text().trim().isEmpty(), is(false));
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
                            // See MAX_OUTPUT_TOKENS: the budget has to clear the thinking block, or
                            // the run produces no assistant content at all and the second assertion
                            // below fails on a healthy model.
                            .maxOutputTokens(MAX_OUTPUT_TOKENS)
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

            ChatResponse complete = done.get(180, TimeUnit.SECONDS);

            // Two independent assertions, both of which must hold.
            //
            // 1. The concatenated onPartialResponse fragments are exactly the final text. This is the
            //    streaming contract itself: a regression that misroutes content deltas into
            //    reasoning_content, or drops a fragment, breaks it.
            String finalText = complete.aiMessage().text() == null
                    ? ""
                    : complete.aiMessage().text();
            assertThat(finalText, is(streamed.toString()));

            // 2. Actual assistant CONTENT arrived -- not merely "content or thinking". With a budget
            //    that clears the thinking block this is the real signal; accepting thinking alone
            //    would let a content-routing regression pass unnoticed, which is what the earlier,
            //    8-token version of this test did.
            assertThat("stream delivered no assistant content", !streamed.toString().isEmpty(), is(true));
        }
    }
}
