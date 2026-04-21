package de.kherud.llama;

import com.fasterxml.jackson.databind.JsonNode;

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

    public static StopReason fromJson(JsonNode node) {
        switch (node.path("stop_type").asText("")) {
            case "eos":   return EOS;
            case "word":  return STOP_STRING;
            case "limit": return MAX_TOKENS;
            default:      return NONE;
        }
    }
}
