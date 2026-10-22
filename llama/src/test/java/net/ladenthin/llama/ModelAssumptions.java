// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.List;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.value.ModelMeta;
import net.ladenthin.llama.value.Pair;
import net.ladenthin.llama.value.SlotMetrics;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Assumptions;

/**
 * Skips a model-backed test whose assertion checks a property of CI's model, on a model without it.
 *
 * <p>Some tests assert what CI's model is rather than what the library does: CodeLlama-7B's embedding
 * dimension, the space its SentencePiece tokenizer puts before a prompt, the thinking block Qwen3's
 * chat template opens, a conversation short enough for the context the test sized. Against another
 * model, such as the random two-layer GGUF of a sandbox, they used to fail, which reads as a defect.
 * Each check here reads the property from the loaded model before the assertion, so such a run skips
 * with the reason; the assertion itself is unchanged, and against CI's models every check holds (the
 * evidence is in {@code BUGHUNT-CHECKED.md}, Brief JR-18), so nothing skips there.</p>
 */
@ClaudeGenerated(
        purpose = "Let the tests that assert a property of CI's models skip with a reason on another "
                + "model instead of failing, by reading that property from the loaded model.")
public final class ModelAssumptions {

    /** {@code LLAMA_VOCAB_TYPE_SPM} in {@code llama.h}: the SentencePiece tokenizer of LLaMA models. */
    static final int VOCAB_TYPE_SPM = 1;

    /** The GGUF key with which an SPM model can turn the space before a prompt off. */
    static final String KEY_ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix";

    /**
     * The fewest parameters of the language model a test that asserts what a vision model sees accepts.
     * CI's SmolVLM-500M-Instruct decodes with SmolLM2-360M, about 360 M parameters (the vision encoder
     * sits in the mmproj and is not counted); a random test model has a few hundred thousand.
     */
    static final long MIN_PARAMETERS_OF_A_TRAINED_VISION_MODEL = 100_000_000L;

    private ModelAssumptions() {}

    /**
     * Aborts the calling test unless the model's embedding dimension is {@code expected}.
     *
     * @param model the loaded model
     * @param expected the dimension the test asserts
     */
    public static void assumeEmbeddingDimension(LlamaModel model, int expected) {
        int actual = model.getModelMeta().getNEmbd();
        Assumptions.assumeTrue(
                actual == expected,
                "needs CI's CodeLlama-7B: the test asserts embedding dimension " + expected + ", this model has "
                        + actual);
    }

    /**
     * Aborts the calling test unless the model's tokenizer is SentencePiece with the space prefix on,
     * which is what puts a space before a decoded prompt.
     *
     * @param model the loaded model
     */
    public static void assumeSentencePieceSpacePrefix(LlamaModel model) {
        ModelMeta meta = model.getModelMeta();
        boolean spaceOff = "false".equals(meta.getMetadata(KEY_ADD_SPACE_PREFIX));
        Assumptions.assumeTrue(
                meta.getVocabType() == VOCAB_TYPE_SPM && !spaceOff,
                "needs CI's CodeLlama-7B: the test asserts the space a SentencePiece tokenizer puts before a "
                        + "prompt; this model has vocab type " + meta.getVocabType()
                        + (spaceOff ? " with " + KEY_ADD_SPACE_PREFIX + " false" : ""));
    }

    /**
     * Aborts the calling test unless the model is a trained vision-language model: a vision projector is
     * loaded and the language model has at least {@link #MIN_PARAMETERS_OF_A_TRAINED_VISION_MODEL}
     * parameters. A test that asserts what the model sees in an image (the colour of a picture) holds
     * only for such a model; a random model with a random projector answers noise whatever the image.
     *
     * @param model the loaded model
     */
    public static void assumeTrainedVisionModel(LlamaModel model) {
        ModelMeta meta = model.getModelMeta();
        long parameters = meta.getNParams();
        Assumptions.assumeTrue(
                meta.supportsVision() && parameters >= MIN_PARAMETERS_OF_A_TRAINED_VISION_MODEL,
                "needs CI's SmolVLM-500M: the test asserts what a trained vision-language model sees in an "
                        + "image; this model has " + parameters + " parameters"
                        + (meta.supportsVision() ? "" : " and no vision projector"));
    }

    /**
     * Aborts the calling test unless the model's chat template opens a thinking block.
     *
     * @param model the loaded model
     */
    public static void assumeThinkingTemplate(LlamaModel model) {
        Assumptions.assumeTrue(
                model.getModelMeta().getChatTemplate().contains("<think>"),
                "needs CI's Qwen3: the test asserts that the model thinks by default, and this model's chat "
                        + "template has no <think> block");
    }

    /**
     * Aborts the calling test unless a conversation fits the context of one slot of the model: the
     * messages rendered by the model's own chat template (with the generation prompt), tokenized by its
     * tokenizer, plus {@code answerTokens} for the answers still to come. A multi-turn test sizes its
     * context for CI's tokenizer; a byte-level tokenizer needs several times the tokens for the same
     * text, and a model with a short training context gets a smaller slot.
     *
     * <p>Pass the whole conversation the test will send, with the answers it will receive as empty
     * assistant messages, and the tokens those answers may take ({@code n_predict} each).</p>
     *
     * @param model the loaded model
     * @param systemMessage the system message the test sends, or {@code null}
     * @param messages the conversation, user and (empty) assistant turns
     * @param answerTokens the tokens reserved for the answers
     */
    public static void assumeConversationFits(
            LlamaModel model, @Nullable String systemMessage, List<Pair<String, String>> messages, int answerTokens) {
        String prompt = model.applyTemplate(new InferenceParameters("").withMessages(systemMessage, messages));
        int promptTokens = model.encode(prompt).length;
        int slotContext = slotContext(model);
        Assumptions.assumeTrue(
                promptTokens + answerTokens <= slotContext,
                "the conversation does not fit this model's slot context: " + promptTokens + " prompt tokens + "
                        + answerTokens + " for the answers > " + slotContext
                        + " (the test sizes its context for CI's model)");
    }

    private static int slotContext(LlamaModel model) {
        List<SlotMetrics> slots = model.getMetricsTyped().getSlotMetrics();
        if (slots.isEmpty()) {
            throw new IllegalStateException("the model reports no slots");
        }
        return slots.get(0).getContextSize();
    }
}
