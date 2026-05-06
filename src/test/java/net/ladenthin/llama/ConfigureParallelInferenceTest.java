package net.ladenthin.llama;

import java.io.File;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

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

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping ConfigureParallelInferenceTest",
                new File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(128)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
        );
    }

    @AfterClass
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
        Assert.assertTrue("configureParallelInference with valid n_threads should succeed", result);
    }

    @Test
    public void testConfigureNThreadsOne() {
        boolean result = model.configureParallelInference("{\"n_threads\":1}");
        Assert.assertTrue("configureParallelInference with n_threads=1 should succeed", result);
    }

    // -------------------------------------------------------------------------
    // Valid n_threads_batch configuration
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureNThreadsBatch() {
        boolean result = model.configureParallelInference("{\"n_threads_batch\":2}");
        Assert.assertTrue("configureParallelInference with valid n_threads_batch should succeed", result);
    }

    @Test
    public void testConfigureNThreadsBatchOne() {
        boolean result = model.configureParallelInference("{\"n_threads_batch\":1}");
        Assert.assertTrue("configureParallelInference with n_threads_batch=1 should succeed", result);
    }

    // -------------------------------------------------------------------------
    // Combined configuration
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureCombinedThreadsAndBatch() {
        boolean result = model.configureParallelInference(
                "{\"n_threads\":2,\"n_threads_batch\":4}");
        Assert.assertTrue("Combined n_threads + n_threads_batch should succeed", result);
    }

    @Test
    public void testConfigureCombinedAllParams() {
        boolean result = model.configureParallelInference(
                "{\"slot_prompt_similarity\":0.5,\"n_threads\":2,\"n_threads_batch\":2}");
        Assert.assertTrue("Combined all params should succeed", result);
    }

    // -------------------------------------------------------------------------
    // slot_prompt_similarity
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureSlotPromptSimilarityValid() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":0.8}");
        Assert.assertTrue("Valid slot_prompt_similarity should succeed", result);
    }

    @Test
    public void testConfigureSlotPromptSimilarityZero() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":0.0}");
        Assert.assertTrue("slot_prompt_similarity=0.0 should succeed", result);
    }

    @Test
    public void testConfigureSlotPromptSimilarityOne() {
        boolean result = model.configureParallelInference("{\"slot_prompt_similarity\":1.0}");
        Assert.assertTrue("slot_prompt_similarity=1.0 should succeed", result);
    }

    // -------------------------------------------------------------------------
    // Empty / no-op config
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureEmptyJson() {
        boolean result = model.configureParallelInference("{}");
        Assert.assertTrue("Empty config should succeed (no-op)", result);
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
        Assert.assertNotNull("Model should produce output after reconfiguration", result);
        Assert.assertFalse("Output should not be empty", result.isEmpty());
    }
}
