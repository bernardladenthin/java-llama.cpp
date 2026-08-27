// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Paths;
import net.ladenthin.llama.args.Optimizer;
import org.junit.jupiter.api.Test;

/** Model-free tests for {@link TrainingParameters} builder defaults and JSON serialization. */
class TrainingParametersTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private JsonNode json(TrainingParameters parameters) throws Exception {
        return MAPPER.readTree(parameters.toJson());
    }

    @Test
    void defaultsSerializeWithExpectedValues() throws Exception {
        TrainingParameters parameters = TrainingParameters.builder()
                .modelPath(Paths.get("base.gguf"))
                .trainingText("hello world")
                .outputPath(Paths.get("tuned.gguf"))
                .build();

        JsonNode node = json(parameters);
        assertThat(node.get("model_path").asText(), is("base.gguf"));
        assertThat(node.get("training_text").asText(), is("hello world"));
        assertThat(node.get("output_path").asText(), is("tuned.gguf"));
        assertThat(node.get("epochs").asInt(), is(2));
        assertThat(node.get("optimizer").asInt(), is(0)); // ADAMW
        assertThat(node.get("learning_rate").floatValue(), is(1e-5f));
        assertThat(node.get("n_gpu_layers").asInt(), is(-1));
        assertThat(node.get("n_batch").asInt(), is(0));
        // training_file is omitted when only inline text is given.
        assertThat(node.has("training_file"), is(false));
    }

    @Test
    void customValuesSerialize() throws Exception {
        TrainingParameters parameters = TrainingParameters.builder()
                .modelPath(Paths.get("base.gguf"))
                .trainingFile(Paths.get("corpus.txt"))
                .outputPath(Paths.get("tuned.gguf"))
                .epochs(5)
                .learningRate(3e-4f)
                .optimizer(Optimizer.SGD)
                .nCtx(512)
                .nGpuLayers(0)
                .valSplit(0.1f)
                .nBatch(256)
                .nUbatch(64)
                .build();

        JsonNode node = json(parameters);
        assertThat(node.get("epochs").asInt(), is(5));
        assertThat(node.get("optimizer").asInt(), is(1)); // SGD
        assertThat(node.get("n_ctx").asInt(), is(512));
        assertThat(node.get("n_gpu_layers").asInt(), is(0));
        assertThat(node.get("n_batch").asInt(), is(256));
        assertThat(node.get("n_ubatch").asInt(), is(64));
        assertThat(node.get("training_file").asText(), is("corpus.txt"));
        assertThat(node.get("val_split").floatValue(), is(0.1f));
        // training_text is omitted when a corpus file is given.
        assertThat(node.has("training_text"), is(false));
    }

    /**
     * The LR-schedule and regularization keys. {@code train_engine.cpp} reads these with fallbacks
     * that are byte-identical to this class's own defaults, so a renamed key does not fail loudly --
     * it silently reverts the fine-tune to a schedule the caller never chose. The values below are
     * therefore deliberately non-default and mutually distinct, so both a key rename and a
     * copy-paste value swap fail this test.
     */
    @Test
    void learningScheduleKeysSerialize() throws Exception {
        TrainingParameters parameters = TrainingParameters.builder()
                .modelPath(Paths.get("base.gguf"))
                .trainingText("hello world")
                .outputPath(Paths.get("tuned.gguf"))
                .lrMin(1e-6f)
                .decayEpochs(3.0f)
                .weightDecay(0.01f)
                .valSplit(0.2f)
                .build();

        JsonNode node = json(parameters);
        assertThat(node.get("lr_min").floatValue(), is(1e-6f));
        assertThat(node.get("decay_epochs").floatValue(), is(3.0f));
        assertThat(node.get("weight_decay").floatValue(), is(0.01f));
        assertThat(node.get("val_split").floatValue(), is(0.2f));
    }
}
