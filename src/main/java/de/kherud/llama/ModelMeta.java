package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Model metadata returned by {@link LlamaModel#getModelMeta()}.
 * <p>
 * Typed getters cover all fields currently returned by the native {@code model_meta()}
 * function. The underlying {@link JsonNode} is also exposed via {@link #asJson()} so
 * that future fields added on the C++ side remain accessible without code changes.
 * </p>
 * <p>{@link #toString()} re-serializes to compact JSON and is suitable for
 * {@code assertEquals} in unit tests.</p>
 */
public final class ModelMeta {

    private final JsonNode node;

    ModelMeta(JsonNode node) {
        this.node = node;
    }

    /** Vocabulary type identifier (e.g. SPM = 2, BPE = 1). */
    public int getVocabType() {
        return node.path("vocab_type").asInt(0);
    }

    /** Total number of tokens in the model vocabulary. */
    public int getNVocab() {
        return node.path("n_vocab").asInt(0);
    }

    /** Context length the model was trained with. */
    public int getNCtxTrain() {
        return node.path("n_ctx_train").asInt(0);
    }

    /** Embedding dimension of the model. */
    public int getNEmbd() {
        return node.path("n_embd").asInt(0);
    }

    /** Total number of model parameters. */
    public long getNParams() {
        return node.path("n_params").asLong(0L);
    }

    /** Model file size in bytes. */
    public long getSize() {
        return node.path("size").asLong(0L);
    }

    /** Returns true if the model supports vision (image) input. */
    public boolean supportsVision() {
        return node.at("/modalities/vision").asBoolean(false);
    }

    /** Returns true if the model supports audio input. */
    public boolean supportsAudio() {
        return node.at("/modalities/audio").asBoolean(false);
    }

    /**
     * The model architecture string from GGUF {@code general.architecture} metadata
     * (e.g. {@code "llama"}, {@code "gemma3"}, {@code "mistral"}).
     * Returns an empty string if the field is absent in the GGUF file.
     */
    public String getArchitecture() {
        return node.path("architecture").asText("");
    }

    /**
     * The human-readable model name from GGUF {@code general.name} metadata.
     * Returns an empty string if the field is absent in the GGUF file.
     */
    public String getModelName() {
        return node.path("name").asText("");
    }

    /**
     * Returns the underlying {@link JsonNode} for direct access to any field,
     * including fields added in future llama.cpp versions.
     */
    public JsonNode asJson() {
        return node;
    }

    /** Re-serializes to compact JSON. Suitable for {@code assertEquals} in tests. */
    @Override
    public String toString() {
        return node.toString();
    }
}
