package net.ladenthin.llama;

import org.jetbrains.annotations.NotNull;

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
     * Note, that you have to configure {@link InferenceParameters#setNProbs(int)} in order for probabilities to be returned.
     */
    @NotNull
    public final Map<String, Float> probabilities;

    /** Whether this is the final token of the generation. */
    public final boolean stop;

    /**
     * The reason generation stopped. {@link StopReason#NONE} on intermediate streaming tokens.
     * Only meaningful when {@link #stop} is {@code true}.
     */
    @NotNull
    public final StopReason stopReason;

    public LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities, boolean stop, @NotNull StopReason stopReason) {
        this.text = text;
        this.probabilities = probabilities;
        this.stop = stop;
        this.stopReason = stopReason;
    }

    @Override
    public String toString() {
        return text;
    }
}
