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

    final boolean stop;

    LlamaOutput(@NotNull String text, @NotNull Map<String, Float> probabilities, boolean stop) {
        this.text = text;
        this.probabilities = probabilities;
        this.stop = stop;
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
        return new LlamaOutput(content, probabilities, stop);
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
     */
    private static Map<String, Float> parseProbabilities(String json) {
        if (!json.contains("\"completion_probabilities\"")) {
            return Collections.emptyMap();
        }
        // For now, return empty map. Full probability parsing can be added later if needed.
        // The probabilities data is available in the raw JSON for advanced users.
        return Collections.emptyMap();
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
