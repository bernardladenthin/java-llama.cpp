// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * How speculative decoding samples the draft, for a draft model ({@code --spec-draft-model}) or a
 * model's own MTP heads.
 *
 * <p>The string constants are the exact values accepted by llama.cpp's {@code --spec-draft-sampling}
 * CLI argument (added in b11368, #27694), which sets {@code common_params_speculative::draft.probabilistic}.
 * The output distribution is the target model's in both modes; what changes is how many drafted tokens
 * it accepts at a non-zero temperature.</p>
 *
 * @see net.ladenthin.llama.parameters.ModelParameters#setDraftSampling(DraftSampling)
 */
public enum DraftSampling implements CliArg {

    /**
     * Draft the argmax token at every position and accept it while the target agrees.
     *
     * <p>CLI string: {@code "greedy"}. This is upstream's default, so passing it is equivalent to
     * omitting the flag.
     */
    GREEDY("greedy"),

    /**
     * Sample the draft and let the target verify it by rejection sampling, which accepts more drafted
     * tokens when the request samples at a temperature above zero. A grammar-constrained request is
     * supported as well.
     *
     * <p>CLI string: {@code "probabilistic"}.
     */
    PROBABILISTIC("probabilistic");

    /**
     * The CLI string passed to {@code --spec-draft-sampling} in llama.cpp's {@code common/arg.cpp}.
     */
    private final String argValue;

    DraftSampling(String value) {
        this.argValue = value;
    }

    /**
     * Returns the CLI string accepted by llama.cpp's {@code --spec-draft-sampling} argument.
     *
     * @return the mode string ({@code "greedy"} or {@code "probabilistic"})
     */
    @Override
    public String getArgValue() {
        return argValue;
    }
}
