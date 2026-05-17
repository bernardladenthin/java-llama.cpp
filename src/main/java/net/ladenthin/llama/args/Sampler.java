// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Heurer
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
    PENALTIES("penalties");

    private final String argValue;

    Sampler(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
