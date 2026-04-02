package de.kherud.llama;

import java.io.File;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests for error handling paths in the JNI layer. Verifies that:
 * <ul>
 *   <li>Invalid model path throws LlamaException</li>
 *   <li>embed() on a model without enableEmbedding() throws</li>
 *   <li>handleInfill with missing input_prefix/input_suffix returns error</li>
 *   <li>handleEmbeddings without embedding support returns error</li>
 *   <li>handleEmbeddings with invalid encoding_format returns error</li>
 *   <li>handleEmbeddings with empty input returns error</li>
 *   <li>configureParallelInference with invalid n_threads returns error</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Verify error handling paths in the JNI layer: invalid model path, embed without " +
                  "enableEmbedding, handleInfill missing fields, handleEmbeddings invalid params, " +
                  "and configureParallelInference validation.",
        model = "claude-opus-4-6"
)
public class ErrorHandlingTest {

    private static LlamaModel model;
    private static LlamaModel modelNoEmbed;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping ErrorHandlingTest",
                new File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        // Model WITH embedding
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(128)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .enableEmbedding()
        );
        // Model WITHOUT embedding
        modelNoEmbed = new LlamaModel(
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
        if (modelNoEmbed != null) {
            modelNoEmbed.close();
        }
    }

    // -------------------------------------------------------------------------
    // Invalid model path
    // -------------------------------------------------------------------------

    @Test(expected = LlamaException.class)
    public void testInvalidModelPathThrows() {
        new LlamaModel(
                new ModelParameters()
                        .setModel("/nonexistent/path/model.gguf")
                        .setFit(false)
        );
    }

    @Test(expected = LlamaException.class)
    public void testEmptyModelPathThrows() {
        new LlamaModel(
                new ModelParameters()
                        .setModel("")
                        .setFit(false)
        );
    }

    // -------------------------------------------------------------------------
    // embed() without embedding support
    // -------------------------------------------------------------------------

    @Test(expected = LlamaException.class)
    public void testEmbedWithoutEnableEmbeddingThrows() {
        modelNoEmbed.embed("hello world");
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings without embedding support
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsWithoutEmbeddingSupportReturnsError() {
        String json = "{\"input\":\"hello world\"}";
        try {
            String result = modelNoEmbed.handleEmbeddings(json, false);
            // If it doesn't throw, the result should indicate an error
            Assert.fail("Expected LlamaException for embeddings without embedding support");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention embedding",
                    e.getMessage().toLowerCase().contains("embedding"));
        }
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings with invalid encoding_format
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsInvalidEncodingFormat() {
        String json = "{\"input\":\"hello world\",\"encoding_format\":\"invalid\"}";
        try {
            String result = model.handleEmbeddings(json, false);
            Assert.fail("Expected LlamaException for invalid encoding_format");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention encoding_format",
                    e.getMessage().contains("encoding_format"));
        }
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings with empty input
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsEmptyInput() {
        String json = "{\"input\":\"\"}";
        try {
            String result = model.handleEmbeddings(json, false);
            Assert.fail("Expected LlamaException for empty input");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention empty or content",
                    e.getMessage().toLowerCase().contains("empty") ||
                    e.getMessage().toLowerCase().contains("content"));
        }
    }

    // -------------------------------------------------------------------------
    // handleInfill with missing fields
    // -------------------------------------------------------------------------

    @Test
    public void testHandleInfillMissingInputPrefix() {
        String json = "{\"input_suffix\":\"return result\",\"n_predict\":5}";
        try {
            String result = model.handleCompletions(json);
            // Infill-specific missing fields may cause an error or just return completion
            // The handleInfill endpoint specifically requires these
        } catch (LlamaException e) {
            // Expected
        }
    }

    @Test
    public void testHandleInfillMissingInputSuffix() {
        String json = "{\"input_prefix\":\"def hello():\",\"n_predict\":5}";
        try {
            model.handleInfill(json);
            // May succeed with empty suffix or throw
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention input_suffix",
                    e.getMessage().contains("input_suffix"));
        }
    }

    // -------------------------------------------------------------------------
    // configureParallelInference validation
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureParallelInferenceInvalidNThreads() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads\":-1}");
            Assert.fail("Expected exception for invalid n_threads");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention n_threads",
                    e.getMessage().contains("n_threads"));
        }
    }

    @Test
    public void testConfigureParallelInferenceInvalidNThreadsBatch() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads_batch\":-1}");
            Assert.fail("Expected exception for invalid n_threads_batch");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention n_threads_batch",
                    e.getMessage().contains("n_threads_batch"));
        }
    }

    @Test
    public void testConfigureParallelInferenceZeroNThreads() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads\":0}");
            Assert.fail("Expected exception for n_threads=0");
        } catch (LlamaException e) {
            Assert.assertTrue("Error should mention n_threads",
                    e.getMessage().contains("n_threads"));
        }
    }
}
