// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * An output of the LLM providing access to the generated text and the associated probabilities. You have to configure
 * {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
 */
public final class LlamaOutput {

    /**
     * The last bit of generated text that is representable as text (i.e., cannot be individual utf-8 multibyte code
     * points).
     */
    @NotNull
    public final String text;

    /**
     * Token-to-probability map kept for backwards compatibility. Each entry's value is the
     * raw {@code prob} or {@code logprob} from the native response. For richer per-token
     * detail (token id and the {@code top_logprobs} alternatives), use {@link #logprobs}.
     * <p>
     * Note, that you have to configure {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
     */
    @NotNull
    public final Map<String, Float> probabilities;

    /**
     * Typed per-token logprob entries with token id and {@code top_logprobs} alternatives.
     * Empty when {@link InferenceParameters#setNProbs(int)} is not configured or the native
     * response did not include {@code completion_probabilities}.
     */
    @NotNull
    public final List<TokenLogprob> logprobs;

    /** Whether this is the final token of the generation. */
    public final boolean stop;

    /**
     * The reason generation stopped. {@link StopReason#NONE} on intermediate streaming tokens.
     * Only meaningful when {@link #stop} is {@code true}.
     */
    @NotNull
    public final StopReason stopReason;

    /**
     * Backwards-compatible constructor that leaves {@link #logprobs} empty.
     *
     * @param text          generated text fragment
     * @param probabilities token-to-probability map (may be empty)
     * @param stop          whether this is the final token
     * @param stopReason    the stop reason ({@link StopReason#NONE} on intermediate tokens)
     */
    public LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities, boolean stop, @NotNull StopReason stopReason) {
        this(text, probabilities, Collections.<TokenLogprob>emptyList(), stop, stopReason);
    }

    /**
     * Construct an output with typed per-token logprobs in addition to the flat probability map.
     *
     * @param text          generated text fragment
     * @param probabilities token-to-probability map (may be empty)
     * @param logprobs      typed per-token logprob entries (may be empty)
     * @param stop          whether this is the final token
     * @param stopReason    the stop reason ({@link StopReason#NONE} on intermediate tokens)
     */
    public LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities,
                       @NotNull List<TokenLogprob> logprobs, boolean stop, @NotNull StopReason stopReason) {
        this.text = text;
        this.probabilities = probabilities;
        this.logprobs = logprobs;
        this.stop = stop;
        this.stopReason = stopReason;
    }

    @Override
    public String toString() {
        return text;
    }
}
