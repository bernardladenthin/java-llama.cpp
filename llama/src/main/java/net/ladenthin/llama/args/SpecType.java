// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * A type of speculative decoding: where the drafted tokens come from that the model then verifies
 * in one batch.
 *
 * <p>The string constants are the exact names accepted by llama.cpp's {@code --spec-type} CLI
 * argument ({@code common_speculative_type} in {@code common/common.h}). Upstream's default is
 * {@link #NONE}: a draft model alone ({@code --spec-draft-model}) drafts nothing until a type asks
 * for it, which is why this library selects {@link #DRAFT_SIMPLE} for a draft model when no type is
 * set (see {@link net.ladenthin.llama.parameters.ModelParameters#setModelDraft(String)}).</p>
 *
 * @see net.ladenthin.llama.parameters.ModelParameters#setSpecTypes(SpecType, SpecType...)
 */
public enum SpecType implements CliArg {

    /** No speculative decoding. CLI string: {@code "none"}. */
    NONE("none"),

    /** A separate, smaller draft model predicts the next tokens. CLI string: {@code "draft-simple"}. */
    DRAFT_SIMPLE("draft-simple"),

    /** An EAGLE-3 draft head over the model's hidden states. CLI string: {@code "draft-eagle3"}. */
    DRAFT_EAGLE3("draft-eagle3"),

    /** The model's own multi-token-prediction heads. CLI string: {@code "draft-mtp"}. */
    DRAFT_MTP("draft-mtp"),

    /** A DFlash block-diffusion drafter. CLI string: {@code "draft-dflash"}. */
    DRAFT_DFLASH("draft-dflash"),

    /** A DSpark drafter (DFlash with a Markov head). CLI string: {@code "draft-dspark"}. */
    DRAFT_DSPARK("draft-dspark"),

    /** Drafts from n-grams of the current context. CLI string: {@code "ngram-simple"}. */
    NGRAM_SIMPLE("ngram-simple"),

    /** Drafts from an n-gram map with key lookup. CLI string: {@code "ngram-map-k"}. */
    NGRAM_MAP_K("ngram-map-k"),

    /** Drafts from an n-gram map with four values per key. CLI string: {@code "ngram-map-k4v"}. */
    NGRAM_MAP_K4V("ngram-map-k4v"),

    /** Drafts from a hashed n-gram table shared by the slots. CLI string: {@code "ngram-mod"}. */
    NGRAM_MOD("ngram-mod"),

    /** Drafts from an n-gram cache. CLI string: {@code "ngram-cache"}. */
    NGRAM_CACHE("ngram-cache");

    /** The CLI string passed to {@code --spec-type} in llama.cpp's {@code common/arg.cpp}. */
    private final String argValue;

    SpecType(String value) {
        this.argValue = value;
    }

    /**
     * Returns the CLI string accepted by llama.cpp's {@code --spec-type} argument.
     *
     * @return the type name, e.g. {@code "draft-simple"}
     */
    @Override
    public String getArgValue() {
        return argValue;
    }
}
