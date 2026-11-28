// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.LlamaLoader;
import net.ladenthin.llama.parameters.TrainingParameters;
import org.jspecify.annotations.Nullable;

/**
 * In-process fine-tuning entry point, wrapping llama.cpp's ggml-opt training path
 * ({@code llama_opt_init} / {@code llama_opt_epoch}) the same way the upstream
 * {@code examples/training/finetune.cpp} tool does. Loads its own model and context (independent of
 * {@link LlamaModel}), fine-tunes on a text corpus, and writes a new GGUF.
 *
 * <p>Configure a run with {@link TrainingParameters} and pass it to {@link #finetune(TrainingParameters)}.
 * Full-model fine-tuning is compute- and memory-intensive and blocks for the whole run; upstream
 * training support is itself experimental.
 *
 * <p>Only a model whose graph ggml can differentiate trains: ggml has no backward pass for the expert
 * matmul of a mixture of experts ({@code MUL_MAT_ID}), the scan of a recurrent model ({@code SSM_SCAN}),
 * {@code NORM} or the gated feed-forward {@code GEGLU} (Gemma 4's), among others, and its backward pass of
 * a matmul ignores a weight's scale (a GGUF with {@code <matrix>.scale} tensors, as FP8 and NVFP4
 * conversions carry). Such a model is refused with a {@link LlamaException} naming the op, before its
 * first training step and without writing an output.
 *
 * <p>What is trained: every F32 weight of the model except the token embedding, which llama.cpp leaves
 * out of the trained parameters ({@code llama_set_param}, marked FIXME upstream). The key and value
 * projections get a gradient only while one micro-batch is the whole context, which is the default
 * here (see {@link TrainingParameters}); a smaller {@code nUbatch} saves memory and leaves them as
 * they were.
 */
public final class LlamaTrainer {

    static {
        LlamaLoader.initialize();
    }

    private LlamaTrainer() {}

    /**
     * Run one fine-tuning job to completion.
     *
     * @param parameters the training configuration (model, corpus, output, optimizer, schedule, ...)
     * @throws IllegalArgumentException if {@code parameters} violate the documented bounds (see
     *     {@link TrainingParameters#toJson()}); checked before anything native runs
     * @throws NullPointerException if {@code parameters} was built without {@code modelPath} or
     *     {@code outputPath}
     * @throws LlamaException if the model cannot be loaded or training fails
     */
    public static void finetune(TrainingParameters parameters) {
        String error = finetuneNative(parameters.toJson());
        if (error != null && !error.isEmpty()) {
            throw new LlamaException(error);
        }
    }

    /**
     * Convenience fine-tune with inline text and otherwise-default settings.
     *
     * @param model the base GGUF model to fine-tune
     * @param trainingText the training corpus (tokenized in-process)
     * @param output the path the fine-tuned GGUF is written to
     * @param epochs number of passes over the corpus (at least 1)
     * @param learningRate the AdamW learning rate at the first epoch (e.g. {@code 1e-5f})
     * @throws IllegalArgumentException if {@code epochs} is below 1
     * @throws LlamaException if the model cannot be loaded or training fails
     */
    public static void finetune(Path model, String trainingText, Path output, int epochs, float learningRate) {
        finetune(TrainingParameters.builder()
                .modelPath(model)
                .trainingText(trainingText)
                .outputPath(output)
                .epochs(epochs)
                .learningRate(learningRate)
                .build());
    }

    // The native layer returns null on success and a non-empty error message on failure, so the
    // return is genuinely @Nullable — without this the module's @NullMarked default would treat it
    // as @NonNull and SpotBugs would flag the `error != null` guard as a redundant null check.
    private static native @Nullable String finetuneNative(String configJson);
}
