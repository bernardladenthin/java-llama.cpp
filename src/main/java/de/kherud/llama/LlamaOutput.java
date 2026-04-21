package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;
import de.kherud.llama.json.CompletionResponseParser;
import de.kherud.llama.json.RerankResponseParser;
import org.jetbrains.annotations.NotNull;

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

    /**
     * Parse a LlamaOutput from a JSON string returned by the native receiveCompletionJson method.
     * Delegates to {@link CompletionResponseParser#parse(String)}.
     */
    static LlamaOutput fromJson(String json) {
        return CompletionResponseParser.parse(json);
    }

    /**
     * Parse a LlamaOutput from a pre-parsed JsonNode.
     * Delegates to {@link CompletionResponseParser#parse(JsonNode)}.
     */
    static LlamaOutput fromJson(JsonNode node) {
        return CompletionResponseParser.parse(node);
    }

    /**
     * Parse rerank results from a JSON array string.
     * Delegates to {@link RerankResponseParser#parse(String)}.
     */
    static List<Pair<String, Float>> parseRerankResults(String json) {
        return RerankResponseParser.parse(json);
    }
}
