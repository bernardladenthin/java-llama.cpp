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
import net.ladenthin.llama.args.Optimizer;
import net.ladenthin.llama.parameters.TrainingParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end fine-tuning smoke over a real model. Self-skips unless a (small) GGUF is provided via
 * {@code -Dnet.ladenthin.llama.train.model=/abs/path/to/model.gguf}. Full-model fine-tuning is
 * compute- and memory-intensive, so this is opt-in and never runs in a default build.
 */
class LlamaTrainerIntegrationTest {

    @Test
    void finetuneWritesAnOutputModel(@TempDir Path tmp) throws Exception {
        String modelPath = TestConstants.resolveModelProperty("net.ladenthin.llama.train.model");
        Assumptions.assumeTrue(
                modelPath != null && !modelPath.isEmpty() && Files.exists(Paths.get(modelPath)),
                "set -Dnet.ladenthin.llama.train.model=/path/to/small.gguf to run the fine-tune smoke");

        StringBuilder corpus = new StringBuilder();
        for (int i = 0; i < 64; i++) {
            corpus.append("The quick brown fox jumps over the lazy dog. ");
        }

        Path output = tmp.resolve("finetuned.gguf");
        LlamaTrainer.finetune(TrainingParameters.builder()
                .modelPath(Paths.get(modelPath))
                .trainingText(corpus.toString())
                .outputPath(output)
                .epochs(1)
                .learningRate(1e-5f)
                .optimizer(Optimizer.ADAMW)
                .build());

        assertThat(Files.exists(output), is(true));
        assertThat(Files.size(output), greaterThan(0L));
    }
}
