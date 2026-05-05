package net.ladenthin.llama.args;

/**
 * Pooling strategy applied to token embeddings when {@link net.ladenthin.llama.ModelParameters#enableEmbedding()}
 * is active.
 *
 * <p>The string constants stored in each enum constant are the exact values accepted by the
 * {@code --pooling} CLI argument in llama.cpp (see {@code common/arg.cpp}).  They map 1-to-1 to
 * the {@code llama_pooling_type} enum in {@code include/llama.h}.
 *
 * <p>{@link #UNSPECIFIED} is special: it has no valid CLI string and is therefore never forwarded
 * to the native layer.  Omitting {@code --pooling} entirely lets llama.cpp fall back to the
 * model's built-in default pooling strategy (equivalent to {@code LLAMA_POOLING_TYPE_UNSPECIFIED = -1}).
 *
 * @see <a href="https://github.com/ggerganov/llama.cpp/blob/b8609/common/arg.cpp">
 *      llama.cpp b8609 – common/arg.cpp: {@code --pooling} argument registration</a>
 * @see <a href="https://github.com/ggerganov/llama.cpp/blob/b8609/include/llama.h">
 *      llama.cpp b8609 – include/llama.h: {@code llama_pooling_type} enum</a>
 */
public enum PoolingType implements CliArg {

    /**
     * Use the model's built-in default pooling type.
     *
     * <p>Maps to {@code LLAMA_POOLING_TYPE_UNSPECIFIED = -1} in {@code include/llama.h}.
     * This value has no corresponding CLI string; passing it to
     * {@link net.ladenthin.llama.ModelParameters#setPoolingType(PoolingType)} intentionally
     * omits the {@code --pooling} flag so llama.cpp chooses the pooling strategy itself.
     */
    UNSPECIFIED("unspecified"),

    /**
     * No pooling – returns one embedding vector per input token.
     *
     * <p>CLI string: {@code "none"} — maps to {@code LLAMA_POOLING_TYPE_NONE = 0}.
     */
    NONE("none"),

    /**
     * Mean pooling – averages all token embeddings into a single vector.
     *
     * <p>CLI string: {@code "mean"} — maps to {@code LLAMA_POOLING_TYPE_MEAN = 1}.
     */
    MEAN("mean"),

    /**
     * CLS pooling – uses the representation of the first (CLS / BOS) token.
     *
     * <p>CLI string: {@code "cls"} — maps to {@code LLAMA_POOLING_TYPE_CLS = 2}.
     *
     * <p><strong>Note:</strong> decoder-only models (e.g. LLaMA / CodeLlama) have no dedicated
     * CLS token; requesting this pooling type for such models causes a native abort in llama.cpp.
     * Use only with encoder or encoder-decoder models that include a CLS token.
     */
    CLS("cls"),

    /**
     * Last-token pooling – uses the representation of the last token.
     *
     * <p>CLI string: {@code "last"} — maps to {@code LLAMA_POOLING_TYPE_LAST = 3}.
     */
    LAST("last"),

    /**
     * Rank pooling – used by re-ranking models to produce a relevance score.
     * Requires a model loaded with {@link net.ladenthin.llama.ModelParameters#enableReranking()};
     * not applicable to plain embedding models.
     *
     * <p>CLI string: {@code "rank"} — maps to {@code LLAMA_POOLING_TYPE_RANK = 4}.
     */
    RANK("rank");

    /**
     * The CLI string passed to {@code --pooling} in llama.cpp's {@code common/arg.cpp}.
     * {@link #UNSPECIFIED} carries {@code "unspecified"} as a sentinel but that string is
     * never forwarded to the native layer.
     */
    private final String argValue;

    PoolingType(String value) {
        this.argValue = value;
    }

    /**
     * Returns the CLI string accepted by llama.cpp's {@code --pooling} argument.
     *
     * @return the pooling type string (e.g. {@code "mean"}, {@code "cls"})
     */
    public String getArgValue() {
        return argValue;
    }
}
