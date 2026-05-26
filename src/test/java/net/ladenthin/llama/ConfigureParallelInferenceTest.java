// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;

import org.junit.jupiter.api.AfterAll;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link LlamaModel#configureParallelInference(String)} covering:
 * <ul>
 *   <li>Valid n_threads configuration</li>
 *   <li>Valid n_threads_batch configuration</li>
 *   <li>Combined n_threads + n_threads_batch</li>
 *   <li>Valid slot_prompt_similarity with n_threads</li>
 *   <li>Empty config (no-op)</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Verify configureParallelInference for n_threads, n_threads_batch, combined configs, " +
                  "and empty/no-op configuration.",
        model = "claude-opus-4-6"
)
public class ConfigureParallelInferenceTest {

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping ConfigureParallelInferenceTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(128)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
        );
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    // -------------------------------------------------------------------------
    // Valid n_threads configuration
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureNThreads() {
        boolean result = model.configureParallelInference("{\"n_threads\":2}");
        assertTrue(result, "configureParallelInference with valid n_threads should succeed");
    }

    @Test
    public void testConfigureNThreadsOne() {
        boolean result = model.configureParallelInference("{\"n_threads\":1}");
        assertTrue(result, "configureParallelInference with n_threads=1 should succeed");
    }

    // -------------------------------------------------------------------------
    // Valid n_threads_batch configuration
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureNThreadsBatch() {
        boolean result = model.configureParallelInference("{\"n_threads_batch\":2}");
        assertTrue(result, "configureParallelInference with valid n_threads_batch should succeed");
    }

    @Test
    public void testConfigureNThreadsBatchOne() {
        boolean result = model.configureParallelInference("{\"n_threads_batch\":1}");
        assertTrue(result, "configureParallelInference with n_threads_batch=1 should succeed");
    }

    // -------------------------------------------------------------------------
    // Combined configuration
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureCombinedThreadsAndBatch() {
        boolean result = model.configureParallelInference(
                "{\"n_threads\":2,\"n_threads_batch\":4}");
        assertTrue(result, "Combined n_threads + n_threads_batch should succeed");
    }

    @Test
    public void testConfigureCombinedAllParams() {
        boolean result = model.configureParallelInference(
                "{\"slot_prompt_similarity\":0.5,\"n_threads\":2,\"n_threads_batch\":2}");
        assertTrue(result, "Combined all params should succeed");
    }

    // -------------------------------------------------------------------------
    // slot_prompt_similarity
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureSlotPromptSimilarityValid() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":0.8}");
        assertTrue(result, "Valid slot_prompt_similarity should succeed");
    }

    @Test
    public void testConfigureSlotPromptSimilarityZero() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":0.0}");
        assertTrue(result, "slot_prompt_similarity=0.0 should succeed");
    }

    @Test
    public void testConfigureSlotPromptSimilarityOne() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":1.0}");
        assertTrue(result, "slot_prompt_similarity=1.0 should succeed");
    }

    // -------------------------------------------------------------------------
    // Empty / no-op config
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureEmptyJson() {
        boolean result = model.configureParallelInference("{}");
        assertTrue(result, "Empty config should succeed (no-op)");
    }

    // -------------------------------------------------------------------------
    // Verify model still works after reconfiguration
    // -------------------------------------------------------------------------

    @Test
    public void testModelWorksAfterReconfiguration() {
        model.configureParallelInference("{\"n_threads\":2}");
        InferenceParameters params = new InferenceParameters("int main() {")
                .setNPredict(5)
                .setTemperature(0);
        String result = model.complete(params);
        assertNotNull(result, "Model should produce output after reconfiguration");
        assertFalse(result.isEmpty(), "Output should not be empty");
    }
}
