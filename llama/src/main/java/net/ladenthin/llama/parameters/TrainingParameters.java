// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import net.ladenthin.llama.args.Optimizer;
import org.jspecify.annotations.Nullable;

/**
 * Immutable configuration for a {@link net.ladenthin.llama.LlamaTrainer} fine-tuning run.
 *
 * <p>Build with {@code builder()}; only {@code modelPath} and {@code outputPath} are required, and
 * exactly one of {@code trainingText} / {@code trainingFile} should be set. All other fields default
 * to values that mirror upstream llama.cpp's fine-tuning defaults. The configuration is serialized to
 * JSON via {@link #toJson()} and parsed by the native layer, the same way {@link ModelParameters} and
 * {@link InferenceParameters} cross the JNI boundary.
 */
@Builder
@Getter
@ToString
public final class TrainingParameters {

    // Jackson mapper for JSON serialization. Static field declared before the instance fields
    // to satisfy fb-contrib's IMC_IMMATURE_CLASS_WRONG_FIELD_ORDER (static members come first).
    private static final ObjectMapper MAPPER = new ObjectMapper();

    // Base GGUF model to fine-tune.
    private final Path modelPath;

    // Training corpus supplied inline; mutually exclusive with trainingFile.
    private final @Nullable String trainingText;

    // Training corpus read from a file by the native layer; mutually exclusive with trainingText.
    private final @Nullable Path trainingFile;

    // Destination path for the fine-tuned GGUF.
    private final Path outputPath;

    // Number of passes over the corpus (at least 1).
    @Builder.Default
    private final int epochs = 2;

    // Learning rate at the first epoch.
    @Builder.Default
    private final float learningRate = 1e-5f;

    // Minimum learning rate for decay, or -1 to disable decay.
    @Builder.Default
    private final float lrMin = -1f;

    // If > 0, decay the learning rate from learningRate to lrMin over this many epochs.
    @Builder.Default
    private final float decayEpochs = -1f;

    // Weight decay (0 disables it).
    @Builder.Default
    private final float weightDecay = 0f;

    // Optimizer algorithm.
    @Builder.Default
    private final Optimizer optimizer = Optimizer.ADAMW;

    // Context size in tokens, or 0 to use the model's trained context.
    @Builder.Default
    private final int nCtx = 0;

    // Layers to offload to the GPU, or -1 for automatic.
    @Builder.Default
    private final int nGpuLayers = -1;

    // Fraction of the corpus held out for validation.
    @Builder.Default
    private final float valSplit = 0.05f;

    // Logical batch size, or 0 to use the native default.
    @Builder.Default
    private final int nBatch = 0;

    // Physical (micro) batch size, or 0 to use the native default.
    @Builder.Default
    private final int nUbatch = 0;

    /**
     * Serialize this configuration to the JSON object the native fine-tuning layer expects.
     *
     * @return a compact JSON string
     */
    public String toJson() {
        ObjectNode node = MAPPER.createObjectNode();
        node.put(TrainingField.MODEL_PATH.getKey(), modelPath.toString());
        if (trainingText != null) {
            node.put(TrainingField.TRAINING_TEXT.getKey(), trainingText);
        }
        if (trainingFile != null) {
            node.put(TrainingField.TRAINING_FILE.getKey(), trainingFile.toString());
        }
        node.put(TrainingField.OUTPUT_PATH.getKey(), outputPath.toString());
        node.put(TrainingField.EPOCHS.getKey(), epochs);
        node.put(TrainingField.LEARNING_RATE.getKey(), learningRate);
        node.put(TrainingField.LR_MIN.getKey(), lrMin);
        node.put(TrainingField.DECAY_EPOCHS.getKey(), decayEpochs);
        node.put(TrainingField.WEIGHT_DECAY.getKey(), weightDecay);
        node.put(TrainingField.OPTIMIZER.getKey(), optimizer.getNativeValue());
        node.put(TrainingField.N_CTX.getKey(), nCtx);
        node.put(TrainingField.N_GPU_LAYERS.getKey(), nGpuLayers);
        node.put(TrainingField.VAL_SPLIT.getKey(), valSplit);
        node.put(TrainingField.N_BATCH.getKey(), nBatch);
        node.put(TrainingField.N_UBATCH.getKey(), nUbatch);
        return node.toString();
    }
}
