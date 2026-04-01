package de.kherud.llama;

import de.kherud.llama.args.PoolingType;
import org.junit.After;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import java.io.File;

/**
 * Integration tests for {@link LlamaModel#embed(String)} across the pooling types that
 * are meaningful for decoder-only embedding models (e.g. CodeLlama).
 *
 * <p>Skipped pooling types and their reasons:
 * <ul>
 *   <li>{@link PoolingType#RANK} – requires a dedicated re-ranking model, not a plain LLM.</li>
 *   <li>{@link PoolingType#NONE} – instructs llama.cpp to return one embedding <em>per token</em>;
 *       {@link LlamaModel#embed(String)} returns only the first row, so the result would silently
 *       be the embedding of a single BOS token rather than a sentence-level vector.</li>
 * </ul>
 *
 * <p>All other types (UNSPECIFIED, MEAN, CLS, LAST) produce a single pooled vector whose
 * dimension equals the model's hidden size (4 096 for CodeLlama-7B).
 */
@ClaudeGenerated(
        purpose = "Verify that LlamaModel.embed() returns a correctly-sized float[] for every " +
                  "pooling type that is applicable to decoder-only embedding models, and that " +
                  "UNSPECIFIED (= model default) behaves the same way."
)
public class LlamaEmbeddingsTest {

    /** Expected embedding dimension for codellama-7b (hidden size = 4 096). */
    private static final int EXPECTED_DIM = 4096;

    private static final String TEXT = "This is a test sentence for embedding.";

    private LlamaModel model;

    @After
    public void tearDown() {
        if (model != null) {
            model.close();
            model = null;
        }
    }

    // -------------------------------------------------------------------------
    // Helper
    // -------------------------------------------------------------------------

    private LlamaModel openModel(PoolingType type) {
        Assume.assumeTrue(
                "Model file not found, skipping " + getClass().getSimpleName(),
                new File(TestConstants.MODEL_PATH).exists()
        );
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .enableEmbedding()
                        .setPoolingType(type)
        );
    }

    // -------------------------------------------------------------------------
    // One test per applicable pooling type
    // -------------------------------------------------------------------------

    /**
     * UNSPECIFIED leaves --pooling unset, so the model applies its built-in default (MEAN for
     * CodeLlama). The result must be a valid 4096-dimensional vector.
     */
    @Test
    public void testEmbedUnspecifiedPooling() {
        model = openModel(PoolingType.UNSPECIFIED);
        float[] embedding = model.embed(TEXT);
        Assert.assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.UNSPECIFIED);
    }

    /** MEAN pooling averages all token embeddings into a single vector. */
    @Test
    public void testEmbedMeanPooling() {
        model = openModel(PoolingType.MEAN);
        float[] embedding = model.embed(TEXT);
        Assert.assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.MEAN);
    }

    /**
     * CLS pooling uses the first (CLS / BOS) token's representation.
     * Decoder-only models do not have a dedicated CLS token, so this is equivalent to the BOS
     * token embedding — still produces a valid 4096-dimensional vector.
     */
    @Test
    public void testEmbedClsPooling() {
        model = openModel(PoolingType.CLS);
        float[] embedding = model.embed(TEXT);
        Assert.assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.CLS);
    }

    /** LAST pooling uses the last token's representation. */
    @Test
    public void testEmbedLastPooling() {
        model = openModel(PoolingType.LAST);
        float[] embedding = model.embed(TEXT);
        Assert.assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.LAST);
    }

    // -------------------------------------------------------------------------
    // Sanity: MEAN and UNSPECIFIED should be numerically close (model default = MEAN)
    // -------------------------------------------------------------------------

    /**
     * Because UNSPECIFIED lets CodeLlama fall back to its model-default pooling (MEAN), the
     * embeddings produced by UNSPECIFIED and MEAN must be identical element-wise.
     */
    @Test
    public void testUnspecifiedEquivalentToMeanForCodeLlama() {
        model = openModel(PoolingType.UNSPECIFIED);
        float[] unspecified = model.embed(TEXT);
        model.close();

        model = openModel(PoolingType.MEAN);
        float[] mean = model.embed(TEXT);

        Assert.assertEquals("dimension mismatch", unspecified.length, mean.length);
        for (int i = 0; i < unspecified.length; i++) {
            Assert.assertEquals("element " + i + " differs", unspecified[i], mean[i], 1e-6f);
        }
    }

    // -------------------------------------------------------------------------
    // Sanity: different pooling strategies should differ
    // -------------------------------------------------------------------------

    /**
     * CLS and LAST pool different tokens, so their outputs must not be identical for a
     * multi-token input.
     */
    @Test
    public void testClsAndLastPoolingDiffer() {
        model = openModel(PoolingType.CLS);
        float[] cls = model.embed(TEXT);
        model.close();

        model = openModel(PoolingType.LAST);
        float[] last = model.embed(TEXT);

        Assert.assertEquals(cls.length, last.length);
        boolean differ = false;
        for (int i = 0; i < cls.length; i++) {
            if (Math.abs(cls[i] - last[i]) > 1e-6f) {
                differ = true;
                break;
            }
        }
        Assert.assertTrue("CLS and LAST pooling must produce different vectors for multi-token input", differ);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static void assertEmbeddingValid(float[] embedding, PoolingType type) {
        for (int i = 0; i < embedding.length; i++) {
            Assert.assertFalse(
                    type + " embedding[" + i + "] is NaN",
                    Float.isNaN(embedding[i])
            );
            Assert.assertFalse(
                    type + " embedding[" + i + "] is infinite",
                    Float.isInfinite(embedding[i])
            );
        }
        // At least some values must be non-zero
        boolean hasNonZero = false;
        for (float v : embedding) {
            if (v != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        Assert.assertTrue(type + " embedding must not be all-zeros", hasNonZero);
    }
}
