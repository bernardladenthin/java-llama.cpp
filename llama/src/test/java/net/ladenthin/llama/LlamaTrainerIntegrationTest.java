// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import net.ladenthin.llama.args.Optimizer;
import net.ladenthin.llama.parameters.TrainingParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end fine-tuning smoke over a real model — the only test that exercises the
 * Java -> JNI -> native trainer round trip. Uses {@link TestConstants#DEFAULT_TRAIN_MODEL_PATH}
 * ({@code models/stories260K.gguf}, 1.19 MB, F32 -- in CI's model set, so it runs on every Java test
 * job) unless {@code -Dnet.ladenthin.llama.train.model=/path/to/model.gguf} names another; a local
 * build without that model skips cleanly.
 *
 * <p>The fixture must be <strong>F32</strong>: {@code llama_set_param} silently ignores any tensor
 * that is not {@code GGML_TYPE_F32}, so a quantized model trains nothing and still produces an
 * output file. The final assertion below is what makes that visible instead of green.
 */
class LlamaTrainerIntegrationTest {

    @Test
    void finetuneWritesAnOutputModel(@TempDir Path tmp) throws Exception {
        String modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_TRAIN_MODEL_PATH, TestConstants.DEFAULT_TRAIN_MODEL_PATH);
        Assumptions.assumeTrue(
                modelPath != null && !modelPath.isEmpty() && Files.exists(Paths.get(modelPath)),
                "model missing: " + modelPath + " (or set -D" + TestConstants.PROP_TRAIN_MODEL_PATH
                        + "=/path/to/f32.gguf)");

        StringBuilder corpus = new StringBuilder();
        for (int i = 0; i < 64; i++) {
            corpus.append("The quick brown fox jumps over the lazy dog. ");
        }

        Path output = tmp.resolve("finetuned.gguf");
        LlamaTrainer.finetune(TrainingParameters.builder()
                .modelPath(Paths.get(modelPath))
                .trainingText(corpus.toString())
                .outputPath(output)
                // Pin nCtx rather than inheriting the model's n_ctx_train. common_opt_dataset_init
                // computes ndata as (tokens - n_ctx - 1) / (n_ctx/2) on a size_t: a corpus shorter
                // than n_ctx + 1 wraps unsigned, and anything short of ~1.5x n_ctx yields ndata 0,
                // which trips GGML_ASSERT(ndata > 0) — a GGML_ABORT that kills the JVM rather than
                // failing the test. The corpus above is comfortably longer than 1.5 * 128, so this
                // keeps the test self-contained instead of depending on the fixture's metadata.
                .nCtx(128)
                // CPU-only on purpose. The default is -1 (offload every layer), and upstream's
                // training README says to build without additional backends for CPU training —
                // the backward ops this path needs (OUT_PROD, OPT_STEP_ADAMW) are not something
                // we have verified on Metal, which the three macOS jobs build with.
                .nGpuLayers(0)
                .epochs(1)
                .learningRate(1e-5f)
                .optimizer(Optimizer.ADAMW)
                .build());

        assertThat(Files.exists(output), is(true));
        assertThat(Files.size(output), greaterThan(0L));

        // The two assertions above pass on a run that trained nothing at all, which is a real
        // possibility rather than a hypothetical: llama_set_param silently skips every tensor
        // that is not GGML_TYPE_F32, so a quantized fixture loads, updates nothing meaningful,
        // and still writes a plausible GGUF. These two close that gap.
        long inputSize = Files.size(Paths.get(modelPath));
        assertThat(
                "the fine-tuned model should be about the size of its input, not a stub",
                Files.size(output),
                greaterThan(inputSize / 2));

        // Weights that actually moved make the bytes differ. This can only ever miss a failure
        // (a no-op run whose metadata happens to be rewritten), never invent one — a real
        // training run always changes the tensor data.
        assertThat(
                "output is byte-identical to the input: nothing was trained (is the fixture F32?)",
                Arrays.equals(Files.readAllBytes(output), Files.readAllBytes(Paths.get(modelPath))),
                is(false));
    }
}
