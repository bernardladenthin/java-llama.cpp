// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Collections;
import java.util.List;

/**
 * Typed result of {@link LlamaModel#completeWithStats(InferenceParameters)}.
 * <p>
 * Bundles the generated text with parsed {@link Usage}, {@link Timings},
 * per-token {@link TokenLogprob} entries (populated only when
 * {@link InferenceParameters#setNProbs(int)} &gt; 0), and the {@link StopReason}.
 * The raw native JSON is exposed via {@link #getRawJson()} as an escape hatch.
 * </p>
 */
public final class CompletionResult {

    private final String text;
    private final Usage usage;
    private final Timings timings;
    private final List<TokenLogprob> logprobs;
    private final StopReason stopReason;
    private final String rawJson;

    /**
     * Construct a completion result.
     *
     * @param text       the generated text
     * @param usage      parsed prompt/completion token counts
     * @param timings    parsed result timings
     * @param logprobs   typed per-token logprob entries (empty when {@code n_probs} was not requested)
     * @param stopReason the parsed stop reason
     * @param rawJson    the raw native JSON string
     */
    public CompletionResult(String text, Usage usage, Timings timings,
                            List<TokenLogprob> logprobs, StopReason stopReason, String rawJson) {
        this.text = text;
        this.usage = usage;
        this.timings = timings;
        this.logprobs = logprobs == null ? Collections.<TokenLogprob>emptyList() : logprobs;
        this.stopReason = stopReason;
        this.rawJson = rawJson;
    }

    /**
     * Generated text accessor.
     * @return the generated text string
     */
    public String getText() {
        return text;
    }

    /**
     * Token-count usage accessor.
     * @return parsed {@link Usage} (prompt + completion token counts)
     */
    public Usage getUsage() {
        return usage;
    }

    /**
     * Timings accessor.
     * @return parsed {@link Timings} for this completion
     */
    public Timings getTimings() {
        return timings;
    }

    /**
     * Per-token logprob entries.
     * @return list of {@link TokenLogprob}; empty when {@code n_probs} was not requested
     */
    public List<TokenLogprob> getLogprobs() {
        return logprobs;
    }

    /**
     * Stop reason accessor.
     * @return the {@link StopReason} parsed from {@code stop_type}
     */
    public StopReason getStopReason() {
        return stopReason;
    }

    /**
     * Raw JSON passthrough.
     * @return the native response JSON string
     */
    public String getRawJson() {
        return rawJson;
    }

    @Override
    public String toString() {
        return text;
    }
}
