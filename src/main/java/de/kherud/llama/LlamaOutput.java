package de.kherud.llama;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
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

    LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities, boolean stop, @NotNull StopReason stopReason) {
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
     * The JSON has the structure: {"content": "...", "stop": true/false, ...}
     */
    static LlamaOutput fromJson(String json) {
        String content = getContentFromJson(json);
        boolean stop = json.contains("\"stop\":true");
        Map<String, Float> probabilities = parseProbabilities(json);
        StopReason stopReason = stop ? StopReason.fromJson(json) : StopReason.NONE;
        return new LlamaOutput(content, probabilities, stop, stopReason);
    }

    /**
     * Extract the "content" field from a JSON response string.
     */
    static String getContentFromJson(String json) {
        // Find "content":"..." or "content": "..."
        int keyIdx = json.indexOf("\"content\"");
        if (keyIdx < 0) {
            return "";
        }
        int colonIdx = json.indexOf(':', keyIdx + 9);
        if (colonIdx < 0) {
            return "";
        }
        int startQuote = json.indexOf('"', colonIdx + 1);
        if (startQuote < 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = startQuote + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case '"': sb.append('"'); i++; break;
                    case '\\': sb.append('\\'); i++; break;
                    case '/': sb.append('/'); i++; break;
                    case 'n': sb.append('\n'); i++; break;
                    case 'r': sb.append('\r'); i++; break;
                    case 't': sb.append('\t'); i++; break;
                    case 'b': sb.append('\b'); i++; break;
                    case 'f': sb.append('\f'); i++; break;
                    case 'u':
                        if (i + 5 < json.length()) {
                            String hex = json.substring(i + 2, i + 6);
                            sb.append((char) Integer.parseInt(hex, 16));
                            i += 5;
                        }
                        break;
                    default: sb.append('\\').append(next); i++; break;
                }
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Parse token probabilities from a JSON response. Returns an empty map if no probabilities are present.
     *
     * <p>The native server produces a {@code completion_probabilities} array where each element
     * represents one generated token:
     * <pre>{"token":"txt","bytes":[...],"id":N,"prob":F,"top_probs":[...]}</pre>
     * or with {@code "logprob"} instead of {@code "prob"} when post-sampling mode is off.
     * We map each outer {@code token → prob/logprob} value, ignoring the nested
     * {@code top_probs} / {@code top_logprobs} arrays.
     */
    private static Map<String, Float> parseProbabilities(String json) {
        int arrayStart = json.indexOf("\"completion_probabilities\"");
        if (arrayStart < 0) {
            return Collections.emptyMap();
        }
        int bracketOpen = json.indexOf('[', arrayStart + 26);
        if (bracketOpen < 0) {
            return Collections.emptyMap();
        }

        Map<String, Float> result = new HashMap<>();
        int idx = bracketOpen + 1;

        while (idx < json.length()) {
            // Skip whitespace and commas between array entries
            while (idx < json.length() && (json.charAt(idx) == ',' || Character.isWhitespace(json.charAt(idx)))) {
                idx++;
            }
            if (idx >= json.length() || json.charAt(idx) == ']') break;
            if (json.charAt(idx) != '{') { idx++; continue; }

            // Find the closing brace of this entry, respecting nested objects/arrays
            int entryStart = idx;
            int depth = 0;
            int entryEnd = idx;
            for (int i = idx; i < json.length(); i++) {
                char ch = json.charAt(i);
                if (ch == '{' || ch == '[') depth++;
                else if (ch == '}' || ch == ']') {
                    depth--;
                    if (depth == 0) { entryEnd = i + 1; break; }
                }
            }
            String entry = json.substring(entryStart, entryEnd);
            idx = entryEnd;

            // Extract outer "token" value (first occurrence = the generated token)
            int tokenKey = entry.indexOf("\"token\"");
            if (tokenKey < 0) continue;
            int colonT = entry.indexOf(':', tokenKey + 7);
            if (colonT < 0) continue;
            int sq = entry.indexOf('"', colonT + 1);
            if (sq < 0) continue;
            int eq = findEndQuote(entry, sq + 1);
            String token = unescapeJson(entry.substring(sq + 1, eq));

            // Find "prob" or "logprob" before "top_probs" / "top_logprobs"
            int topIdx = entry.indexOf("\"top_");
            int searchLimit = topIdx > 0 ? topIdx : entry.length();

            int probKey = entry.indexOf("\"prob\"");
            int logprobKey = entry.indexOf("\"logprob\"");

            int valueStart;
            if (probKey >= 0 && probKey < searchLimit) {
                valueStart = entry.indexOf(':', probKey + 6);
            } else if (logprobKey >= 0 && logprobKey < searchLimit) {
                valueStart = entry.indexOf(':', logprobKey + 9);
            } else {
                continue;
            }
            if (valueStart < 0 || valueStart >= searchLimit) continue;
            int vs = valueStart + 1;
            while (vs < entry.length() && entry.charAt(vs) == ' ') vs++;
            int ve = vs;
            while (ve < entry.length()) {
                char ch = entry.charAt(ve);
                if (Character.isDigit(ch) || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' || ch == '+') ve++;
                else break;
            }
            if (ve == vs) continue;
            result.put(token, Float.parseFloat(entry.substring(vs, ve)));
        }

        return result.isEmpty() ? Collections.emptyMap() : result;
    }

    /**
     * Parse rerank results from a JSON array string.
     * Expected format: [{"document": "...", "index": 0, "score": 0.95}, ...]
     */
    static List<Pair<String, Float>> parseRerankResults(String json) {
        List<Pair<String, Float>> results = new ArrayList<>();
        // Simple parser for the known JSON array structure
        int idx = 0;
        while ((idx = json.indexOf("\"document\"", idx)) >= 0) {
            // Extract document string
            int colonIdx = json.indexOf(':', idx + 10);
            int startQuote = json.indexOf('"', colonIdx + 1);
            int endQuote = findEndQuote(json, startQuote + 1);
            String document = unescapeJson(json.substring(startQuote + 1, endQuote));

            // Extract score
            int scoreIdx = json.indexOf("\"score\"", endQuote);
            if (scoreIdx < 0) break;
            int scoreColon = json.indexOf(':', scoreIdx + 7);
            int scoreStart = scoreColon + 1;
            // Skip whitespace
            while (scoreStart < json.length() && json.charAt(scoreStart) == ' ') scoreStart++;
            int scoreEnd = scoreStart;
            while (scoreEnd < json.length() && (Character.isDigit(json.charAt(scoreEnd)) || json.charAt(scoreEnd) == '.' || json.charAt(scoreEnd) == '-' || json.charAt(scoreEnd) == 'e' || json.charAt(scoreEnd) == 'E' || json.charAt(scoreEnd) == '+')) scoreEnd++;
            float score = Float.parseFloat(json.substring(scoreStart, scoreEnd));

            results.add(new Pair<>(document, score));
            idx = scoreEnd;
        }
        return results;
    }

    private static int findEndQuote(String s, int from) {
        for (int i = from; i < s.length(); i++) {
            if (s.charAt(i) == '\\') {
                i++; // skip escaped char
            } else if (s.charAt(i) == '"') {
                return i;
            }
        }
        return s.length();
    }

    private static String unescapeJson(String s) {
        if (!s.contains("\\")) return s;
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char next = s.charAt(i + 1);
                switch (next) {
                    case '"': sb.append('"'); i++; break;
                    case '\\': sb.append('\\'); i++; break;
                    case 'n': sb.append('\n'); i++; break;
                    case 'r': sb.append('\r'); i++; break;
                    case 't': sb.append('\t'); i++; break;
                    default: sb.append(c); break;
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
