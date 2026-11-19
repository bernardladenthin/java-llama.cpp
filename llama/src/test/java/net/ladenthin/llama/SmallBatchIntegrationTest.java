// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * A batch smaller than llama.cpp expected: one token ({@code setBatchSize(1)}), fewer tokens than
 * slots ({@code setParallel(4)} on a batch of two), and a batch too small for a draft next to the
 * sampled token. Each ended the JVM in llama.cpp -- the server's load probes the memory with two
 * tokens in one batch (patches/0032), a context of more sequences than batch tokens asserted while
 * it was created (patches/0034), and a draft that did not fit the batch asserted at the first
 * generation step (patches/0033) -- and until patches/0032 the engine refused a batch below two at
 * load instead. Every model here evaluates in ubatches of one token, so it answers exactly what a
 * model with a large batch answers in ubatches of one token; the C API tests (CApiBatchSize) hold
 * the same without a downloaded model.
 */
@ClaudeGenerated(
        purpose = "Pin that a batch of one token, a batch smaller than the slot count and a batch next to a "
                + "draft model load and answer through JNI what a model with a large batch answers.")
public class SmallBatchIntegrationTest {

    private static final String PROMPT = "def main():";

    private static final int N_PREDICT = 8;

    @BeforeAll
    static void needsTheDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping SmallBatchIntegrationTest");
    }

    private static ModelParameters parameters(int batch, int slots) {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(512)
                .setBatchSize(batch)
                .setUbatchSize(1)
                .setParallel(slots)
                .setGpuLayers(gpuLayers)
                .setFit(false);
    }

    private static InferenceParameters request() {
        return new InferenceParameters(PROMPT)
                .withNPredict(N_PREDICT)
                .withTemperature(0.0f)
                .withCachePrompt(false);
    }

    /** What the model answers to {@link #PROMPT} with a batch of 64 tokens and the given slots. */
    private static String reference(int slots) {
        try (LlamaModel model = new LlamaModel(parameters(64, slots))) {
            return model.complete(request());
        }
    }

    /** The answers of {@code slots} requests sent at once. */
    private static List<String> concurrentAnswers(LlamaModel model, int slots) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(slots);
        try {
            List<Future<String>> futures = new ArrayList<>();
            for (int i = 0; i < slots; i++) {
                futures.add(pool.submit(() -> model.complete(request())));
            }
            List<String> answers = new ArrayList<>();
            for (Future<String> future : futures) {
                answers.add(future.get(5, TimeUnit.MINUTES)); // a hang fails the test instead of the job
            }
            return answers;
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void aBatchOfOneTokenLoadsAndAnswers() {
        String expected = reference(1);
        assertFalse(expected.isEmpty());
        try (LlamaModel model = new LlamaModel(parameters(1, 1))) {
            assertEquals(expected, model.complete(request()));
        }
    }

    @Test
    void moreSlotsThanBatchTokensLoadAndAnswerTogether() throws Exception {
        String expected = reference(4);
        try (LlamaModel model = new LlamaModel(parameters(2, 4))) {
            for (String answer : concurrentAnswers(model, 4)) {
                assertEquals(expected, answer);
            }
        }
    }

    @Test
    void aDraftModelNextToABatchOfThreeTokensAnswers() {
        // the draft model is the model itself, the default draft three tokens: with the sampled token
        // four, one more than the batch holds
        String expected = reference(1);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        ModelParameters withDraft =
                parameters(3, 1).setModelDraft(TestConstants.DRAFT_MODEL_PATH).setGpuLayersDraft(gpuLayers);
        try (LlamaModel model = new LlamaModel(withDraft)) {
            assertEquals(expected, model.complete(request()));
        }
    }
}
