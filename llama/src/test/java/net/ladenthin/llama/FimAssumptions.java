// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import org.junit.jupiter.api.Assumptions;

/**
 * Skips a fill-in-the-middle test on a model without FIM tokens.
 *
 * <p>The infill tests exercise the infill paths, not the model's capability, so a model that has
 * no FIM prefix, suffix or middle token (the CI model CodeLlama has all three, found by
 * upstream's token-text detection, {@code "▁<PRE>"} etc. in {@code llama-vocab.cpp}; a random test
 * model has none) is a reason to skip, not a failure. The GGUF metadata cannot decide it: a
 * CodeLlama GGUF names no {@code tokenizer.ggml.fim_*} key, upstream finds its tokens by their
 * text. So the native check itself is asked: an infill request without {@code input_prefix} is
 * refused for the missing field when the model has the tokens, and with exactly
 * {@link #NO_FIM_MESSAGE} when it has none (the engine checks the tokens first). Nothing is
 * generated either way. Any other answer is a failure, not a skip.</p>
 */
@ClaudeGenerated(
        purpose = "Let the infill tests skip themselves on a model without FIM tokens instead of "
                + "failing, through the native FIM check rather than the GGUF metadata.")
public final class FimAssumptions {

    /** The message the native infill handler refuses a model without FIM tokens with. */
    public static final String NO_FIM_MESSAGE = "Model does not support fill-in-the-middle infill";

    private FimAssumptions() {}

    /**
     * Aborts the calling test when {@code model} has no FIM tokens.
     *
     * @param model the loaded model the test infills with
     */
    public static void assumeFimTokens(LlamaModel model) {
        if (!hasFimTokens(model)) {
            Assumptions.abort("model has no fill-in-the-middle tokens: " + NO_FIM_MESSAGE);
        }
    }

    /**
     * Whether {@code model} has FIM tokens, asked through the native FIM check as described above.
     *
     * @param model the loaded model
     * @return {@code true} when the infill handler refuses the probe for its missing field,
     *         {@code false} when it refuses it with {@link #NO_FIM_MESSAGE}
     */
    public static boolean hasFimTokens(LlamaModel model) {
        try {
            model.handleInfill("{}");
        } catch (InvalidRequestException e) {
            if (e.getMessage() != null && e.getMessage().contains("input_prefix")) {
                return true; // the FIM check passed; the probe was refused for the missing field
            }
            throw e;
        } catch (LlamaException e) {
            if (NO_FIM_MESSAGE.equals(e.getMessage())) {
                return false;
            }
            throw e;
        }
        throw new AssertionError("an infill request without input_prefix was answered");
    }
}
