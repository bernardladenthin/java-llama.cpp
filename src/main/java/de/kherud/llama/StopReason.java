package de.kherud.llama;

/**
 * The reason why token generation stopped for a {@link LlamaOutput}.
 *
 * <ul>
 *   <li>{@link #NONE} — generation has not stopped yet (intermediate streaming token);
 *       {@link #getStopType()} returns {@code null}.</li>
 *   <li>{@link #EOS} — the model produced the end-of-sequence token.</li>
 *   <li>{@link #STOP_STRING} — a caller-specified stop string was matched.</li>
 *   <li>{@link #MAX_TOKENS} — the token budget ({@code nPredict} or context limit) was exhausted;
 *       the response was truncated.</li>
 * </ul>
 */
public enum StopReason {

    /** No stop yet; the {@code "stop_type"} field is absent for intermediate tokens. */
    NONE(null),

    /** End-of-sequence token produced. Server {@code "stop_type"} value: {@code "eos"}. */
    EOS("eos"),

    /** A caller-supplied stop string was matched. Server {@code "stop_type"} value: {@code "word"}. */
    STOP_STRING("word"),

    /** Token budget exhausted. Server {@code "stop_type"} value: {@code "limit"}. */
    MAX_TOKENS("limit");

    private final String stopType;

    StopReason(String stopType) {
        this.stopType = stopType;
    }

    /**
     * Returns the {@code "stop_type"} string used by the native server for this constant,
     * or {@code null} for {@link #NONE} (intermediate tokens carry no stop-type field).
     *
     * @return the stop-type string, or {@code null} for {@link #NONE}
     */
    public String getStopType() {
        return stopType;
    }

    /**
     * Map a raw {@code "stop_type"} string from the native server to a {@link StopReason}.
     * Pass the already-extracted field value, e.g.
     * {@code node.path("stop_type").asText("")}.
     *
     * @param stopType the raw stop-type string, or {@code null} / empty for absent field
     * @return the corresponding {@link StopReason}, or {@link #NONE} if unrecognised
     */
    public static StopReason fromStopType(String stopType) {
        if (stopType == null) return NONE;
        switch (stopType) {
            case "eos":   return EOS;
            case "word":  return STOP_STRING;
            case "limit": return MAX_TOKENS;
            default:      return NONE;
        }
    }
}
