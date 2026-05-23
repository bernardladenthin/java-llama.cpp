// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Collections;
import java.util.List;

/**
 * Per-token log-probability entry from the native {@code completion_probabilities} array.
 * <p>
 * Populated when {@link InferenceParameters#setNProbs(int)} is &gt; 0. The native server
 * emits one of two equivalent shapes depending on whether post-sampling probabilities are
 * enabled:
 * </p>
 * <ul>
 *   <li>Post-sampling: {@code prob} field, range [0, 1]</li>
 *   <li>Pre-sampling: {@code logprob} field, range (-inf, 0]</li>
 * </ul>
 * <p>
 * Whichever was present in the JSON is stored verbatim in {@link #getLogprob()}; callers
 * inspecting the value should know which mode they configured.
 * </p>
 */
public final class TokenLogprob {

    private final String token;
    private final int tokenId;
    private final float logprob;
    private final List<TokenLogprob> topLogprobs;

    /**
     * Construct a token logprob entry.
     *
     * @param token       the rendered token text
     * @param tokenId     the integer vocab id of the token, or -1 when unknown
     * @param logprob     the raw probability or natural-log probability (see {@link #getLogprob()})
     * @param topLogprobs alternative tokens the sampler considered; may be {@code null} (treated as empty)
     */
    public TokenLogprob(String token, int tokenId, float logprob, List<TokenLogprob> topLogprobs) {
        this.token = token;
        this.tokenId = tokenId;
        this.logprob = logprob;
        this.topLogprobs = topLogprobs == null ? Collections.<TokenLogprob>emptyList() : topLogprobs;
    }

    /**
     * Token text.
     * @return the rendered token text from the native response
     */
    public String getToken() {
        return token;
    }

    /**
     * Token vocab id.
     * @return the integer vocab id, or {@code -1} when the native response did not include one
     */
    public int getTokenId() {
        return tokenId;
    }

    /**
     * Raw probability or log-probability value from the native response. The interpretation
     * depends on the post-sampling-probs setting: a value in [0, 1] is a probability;
     * a non-positive value is a natural-log probability.
     *
     * @return the raw {@code prob} or {@code logprob} value as emitted by the native server
     */
    public float getLogprob() {
        return logprob;
    }

    /**
     * Alternative tokens the sampler considered, populated from the native
     * {@code top_probs} / {@code top_logprobs} array.
     *
     * @return the alternatives list, never {@code null}; empty when the native response
     *         did not include alternatives
     */
    public List<TokenLogprob> getTopLogprobs() {
        return topLogprobs;
    }

    @Override
    public String toString() {
        return "TokenLogprob{token=" + token + ", id=" + tokenId
                + ", logprob=" + logprob + ", top=" + topLogprobs.size() + "}";
    }
}
