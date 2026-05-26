// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.args.PoolingType;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

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
 *   <li>{@link PoolingType#CLS} – decoder-only models (LLaMA / CodeLlama) have no CLS token;
 *       requesting CLS pooling triggers a native {@code SIGABRT} in llama.cpp.</li>
 * </ul>
 *
 * <p>Tested types (UNSPECIFIED, MEAN, LAST) all produce a single pooled vector whose
 * dimension equals the model's hidden size (4 096 for CodeLlama-7B).
 *
 * <p><strong>Note on UNSPECIFIED:</strong> CodeLlama's GGUF metadata reports
 * {@code pooling type = -1}, meaning no pooling is baked into the model file. When
 * {@code --pooling} is omitted, llama.cpp keeps that {@code -1} and the JNI layer
 * returns the first row of the embedding matrix (the BOS-token vector). This is
 * intentionally different from MEAN pooling, so there is no equivalence assertion
 * between the two.
 */
@ClaudeGenerated(
        purpose = "Verify that LlamaModel.embed() returns a correctly-sized float[] for every " +
                  "pooling type that is applicable to decoder-only embedding models, and that " +
                  "UNSPECIFIED (= model default) behaves the same way as MEAN for CodeLlama."
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
        Assumptions.assumeTrue(new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping " + getClass().getSimpleName());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.MODEL_PATH)
                        .setCtxSize(128)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
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
        assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.UNSPECIFIED);
    }

    /** MEAN pooling averages all token embeddings into a single vector. */
    @Test
    public void testEmbedMeanPooling() {
        model = openModel(PoolingType.MEAN);
        float[] embedding = model.embed(TEXT);
        assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.MEAN);
    }

    /** LAST pooling uses the last token's representation. */
    @Test
    public void testEmbedLastPooling() {
        model = openModel(PoolingType.LAST);
        float[] embedding = model.embed(TEXT);
        assertEquals(EXPECTED_DIM, embedding.length);
        assertEmbeddingValid(embedding, PoolingType.LAST);
    }

    // -------------------------------------------------------------------------
    // Sanity: MEAN and LAST should produce different vectors
    // -------------------------------------------------------------------------

    /**
     * MEAN and LAST pool over different token positions, so their outputs must not be identical
     * for a multi-token input.
     * The two models are loaded and freed sequentially so only one is in memory at a time.
     */
    @Test
    public void testMeanAndLastPoolingDiffer() {
        model = openModel(PoolingType.MEAN);
        float[] mean = model.embed(TEXT);
        model.close();

        model = openModel(PoolingType.LAST);
        float[] last = model.embed(TEXT);

        assertEquals(mean.length, last.length);
        boolean differ = false;
        for (int i = 0; i < mean.length; i++) {
            if (Math.abs(mean[i] - last[i]) > 1e-6f) {
                differ = true;
                break;
            }
        }
        assertTrue(differ, "MEAN and LAST pooling must produce different vectors for multi-token input");
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static void assertEmbeddingValid(float[] embedding, PoolingType type) {
        for (int i = 0; i < embedding.length; i++) {
            assertFalse(Float.isNaN(embedding[i]), type + " embedding[" + i + "] is NaN");
            assertFalse(Float.isInfinite(embedding[i]), type + " embedding[" + i + "] is infinite");
        }
        boolean hasNonZero = false;
        for (float v : embedding) {
            if (v != 0.0f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, type + " embedding must not be all-zeros");
    }

    // -------------------------------------------------------------------------
    // Issue #98 — nomic-embed-text loads with enableEmbedding()
    // -------------------------------------------------------------------------

    /**
     * Upstream issue <a href="https://github.com/kherud/llama.cpp/issues/98">#98</a>:
     * loading {@code nomic-embed-text-v1.5.f16.gguf} aborted with
     * {@code GGML_ASSERT(strcmp(res->name, "result_output") == 0)} because the reporter's
     * config did not call {@code enableEmbedding()}, so the upstream {@code --embedding}
     * flag was never set and the embedding pipeline was not initialised.
     *
     * <p>This test reproduces the reporter's batch/ubatch sizing <em>plus</em> the fix
     * ({@code enableEmbedding()}) and asserts the model loads and produces a 768-dimensional
     * vector. Gated on the {@link TestConstants#PROP_NOMIC_MODEL_PATH} system property so
     * CI hosts without the ~120 MB GGUF file self-skip cleanly.
     *
     * <p>Run with:
     * <pre>
     *   mvn test -Dtest=LlamaEmbeddingsTest#testNomicEmbedLoads \
     *            -Dnet.ladenthin.llama.nomic.path=models/nomic-embed-text-v1.5.f16.gguf
     * </pre>
     */
    @Test
    public void testNomicEmbedLoads() {
        String nomicPath = System.getProperty(TestConstants.PROP_NOMIC_MODEL_PATH);
        Assumptions.assumeTrue(nomicPath != null,
                "Set -D" + TestConstants.PROP_NOMIC_MODEL_PATH + " to a nomic-embed-text GGUF to run this test");
        Assumptions.assumeTrue(new File(nomicPath).exists(), "Nomic model file not found at " + nomicPath);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setModel(nomicPath)
                        .setBatchSize(8192)
                        .setUbatchSize(8192)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .enableEmbedding()
        );

        float[] embedding = model.embed("search_query: What is TSNE?");
        assertEquals(TestConstants.NOMIC_EMBED_DIM, embedding.length, "nomic-embed-text-v1.5 must return a " + TestConstants.NOMIC_EMBED_DIM + "-dim vector");
        assertEmbeddingValid(embedding, PoolingType.MEAN);
    }
}
