// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Sampling algorithm for {@code --samplers} (CLI) and the {@code "samplers"} JSON field.
 */
public enum Sampler implements CliArg {

    /** DRY (Don't Repeat Yourself) repetition penalty sampler. */
    DRY("dry"),
    /** Top-K sampling: keep only the K most likely tokens. */
    TOP_K("top_k"),
    /** Top-P (nucleus) sampling: keep tokens whose cumulative probability exceeds P. */
    TOP_P("top_p"),
    /** Typical-P sampling: keep tokens whose local typicality exceeds P. */
    TYP_P("typ_p"),
    /** Min-P sampling: remove tokens below a minimum probability threshold. */
    MIN_P("min_p"),
    /** Temperature scaling applied to logits before sampling. */
    TEMPERATURE("temperature"),
    /** XTC (eXclude Top Choices) sampler. */
    XTC("xtc"),
    /** Infill-specific sampler for fill-in-the-middle tasks. */
    INFILL("infill"),
    /** Repetition, frequency, and presence penalties sampler. */
    PENALTIES("penalties"),
    /** Top-n-sigma sampling: keep tokens within n standard deviations of the top logit. */
    TOP_N_SIGMA("top_n_sigma"),
    /**
     * Adaptive-p: selects a token near a target probability that adapts over the recent tokens, in
     * place of the final random draw. Only in the chain when named; its target and decay are the
     * request's {@code adaptive_target} / {@code adaptive_decay}.
     */
    ADAPTIVE_P("adaptive_p");

    private final String argValue;

    Sampler(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
