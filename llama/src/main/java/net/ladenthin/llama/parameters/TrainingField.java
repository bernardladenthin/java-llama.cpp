// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

/**
 * Every key {@link TrainingParameters} writes into the fine-tuning configuration JSON.
 *
 * <p>Unlike the CLI and request registries, both ends of this contract are ours:
 * {@code train_engine.cpp} reads the same names back. That does not make it safer — it reads them
 * with {@code j.value(key, default)}, which <em>silently falls back to the default</em> when a key
 * is missing, so a rename on one side turns a configured knob into its default with no error
 * anywhere. And nothing runnable covers it: {@code LlamaTrainerIntegrationTest} is gated on a
 * system property no CI job sets.
 *
 * <p>So the registry exists for the same reason the other two do, and
 * {@code src/test/cpp/test_wire_contracts.cpp} checks it against
 * {@code jllama_train::config_keys()} — the list the C++ parser itself is written against.
 *
 * @see RequestField
 */
enum TrainingField {

    /** Path to the base model to fine-tune. */
    MODEL_PATH("model_path"),

    /** Inline training corpus. */
    TRAINING_TEXT("training_text"),

    /** Path to a training corpus file. */
    TRAINING_FILE("training_file"),

    /** Path the fine-tuned model is written to. */
    OUTPUT_PATH("output_path"),

    /** Number of training epochs. */
    EPOCHS("epochs"),

    /** Initial learning rate. */
    LEARNING_RATE("learning_rate"),

    /** Floor the learning rate decays towards. */
    LR_MIN("lr_min"),

    /** Number of epochs over which the learning rate decays. */
    DECAY_EPOCHS("decay_epochs"),

    /** Weight-decay coefficient. */
    WEIGHT_DECAY("weight_decay"),

    /** Optimizer selector, as the native enum value. */
    OPTIMIZER("optimizer"),

    /** Training context length. */
    N_CTX("n_ctx"),

    /** Number of layers offloaded to the GPU. */
    N_GPU_LAYERS("n_gpu_layers"),

    /** Fraction of the corpus held out for validation. */
    VAL_SPLIT("val_split"),

    /** Logical batch size. */
    N_BATCH("n_batch"),

    /** Physical (micro) batch size. */
    N_UBATCH("n_ubatch");

    private final String key;

    TrainingField(String key) {
        this.key = key;
    }

    /**
     * Returns the JSON key this field is written under (e.g. {@code "model_path"}).
     *
     * @return the configuration key
     */
    String getKey() {
        return key;
    }
}
