package de.kherud.llama;

/**
 * The reason why token generation stopped for a {@link LlamaOutput}.
 *
 * <ul>
 *   <li>{@link #NONE} — generation has not stopped yet (intermediate streaming token).</li>
 *   <li>{@link #EOS} — the model produced the end-of-sequence token.</li>
 *   <li>{@link #STOP_STRING} — a caller-specified stop string was matched.</li>
 *   <li>{@link #MAX_TOKENS} — the token budget ({@code nPredict} or context limit) was exhausted;
 *       the response was truncated.</li>
 * </ul>
 */
public enum StopReason {
    NONE,
    EOS,
    STOP_STRING,
    MAX_TOKENS;

    static StopReason fromJson(String json) {
        if (json.contains("\"stop_type\":\"eos\"")) return EOS;
        if (json.contains("\"stop_type\":\"word\"")) return STOP_STRING;
        if (json.contains("\"stop_type\":\"limit\"")) return MAX_TOKENS;
        return NONE;
    }
}
