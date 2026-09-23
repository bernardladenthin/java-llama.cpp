// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.ArrayList;
import java.util.List;
import org.jspecify.annotations.Nullable;

/**
 * The text side of editing a file: what to match, what to write back, and what to say when it does
 * not match. No file access, no tools — so every rule here is testable on plain strings.
 *
 * <p><b>Line endings are the reason this class exists.</b> A model answers in LF, always; a file on
 * Windows is usually CRLF. Comparing the two directly never matches, which is why the naive
 * implementation silently fails on every Windows file. So the file is normalized to LF before
 * matching and written back in <em>its own</em> ending, and a byte-order mark is taken off first and
 * put back afterwards — it is invisible, and it sits exactly where the first match would be.
 *
 * <p><b>A failed edit is not a free retry.</b> Measured on SWE-agent trajectories: any edit attempt
 * eventually succeeds in 90.5 % of cases, but only 57.2 % once a single edit has failed — a failure
 * derails the rest of the run. That is why a miss does not answer "not found" but shows the closest
 * lines in the file, and an ambiguous match names the line numbers instead of asking for "more
 * context". aider's threshold is used for the closest-line search: similarity ≥ {@value #CANDIDATE_MIN_SIMILARITY}.
 *
 * <p><b>Several edits are all-or-nothing</b>, which deviates from every shipping agent — they apply
 * sequentially and leave a half-edited file behind. Here the whole batch works on one in-memory
 * string and is written only if every edit matched, because at that point atomicity costs nothing and
 * a half-applied batch is the state a model reasons about worst.
 */
public final class TextEdits {

    /** How similar a line must be to be worth showing after a failed match (aider's threshold). */
    public static final double CANDIDATE_MIN_SIMILARITY = 0.6;

    /** How many near misses are shown. */
    public static final int MAX_CANDIDATES = 3;

    /** The byte-order mark, as the character it decodes to. */
    private static final char BOM = '﻿';

    private TextEdits() {}

    /**
     * One replacement.
     *
     * @param oldString the text to find, in LF form
     * @param newString what replaces it
     * @param replaceAll whether every occurrence is replaced instead of requiring exactly one
     */
    public record Edit(String oldString, String newString, boolean replaceAll) {}

    /** Thrown when an edit cannot be applied; the message is what the model gets to read. */
    public static final class EditException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        /**
         * Create the exception.
         *
         * @param message the explanation shown to the model
         */
        public EditException(String message) {
            super(message);
        }
    }

    /**
     * The line ending a text uses.
     *
     * @param text the file content as read
     * @return {@code "\r\n"} when the first line ending is a carriage-return pair, else {@code "\n"}
     */
    public static String lineEnding(String text) {
        int newline = text.indexOf('\n');
        return newline > 0 && text.charAt(newline - 1) == '\r' ? "\r\n" : "\n";
    }

    /**
     * Strip a byte-order mark and convert every line ending to LF.
     *
     * @param text the file content as read
     * @return the normalized content
     */
    public static String normalize(String text) {
        String withoutBom = text.isEmpty() || text.charAt(0) != BOM ? text : text.substring(1);
        return withoutBom.replace("\r\n", "\n").replace("\r", "\n");
    }

    /**
     * Put the file's own line ending and byte-order mark back.
     *
     * @param normalized the edited content in LF form
     * @param original the content as it was read, for its ending and mark
     * @return the content to write
     */
    public static String denormalize(String normalized, String original) {
        String ending = lineEnding(original);
        String restored = "\n".equals(ending) ? normalized : normalized.replace("\n", ending);
        return !original.isEmpty() && original.charAt(0) == BOM ? BOM + restored : restored;
    }

    /**
     * Apply every edit to {@code original}, or none of them.
     *
     * @param original the file content as read
     * @param edits the edits, applied in order to the result of the previous one
     * @return the content to write back, with the original line ending and mark
     * @throws EditException when any edit does not match, naming what went wrong
     */
    public static String apply(String original, List<Edit> edits) {
        if (edits.isEmpty()) {
            throw new EditException("No edits were given.");
        }
        String content = normalize(original);
        for (int i = 0; i < edits.size(); i++) {
            Edit edit = edits.get(i);
            String where = edits.size() == 1 ? "" : " (edit " + (i + 1) + " of " + edits.size() + ")";
            content = applyOne(content, edit, where);
        }
        return denormalize(content, original);
    }

    private static String applyOne(String content, Edit edit, String where) {
        String oldString = normalize(edit.oldString());
        if (oldString.isEmpty()) {
            throw new EditException("old_string must not be empty" + where + ".");
        }
        if (oldString.equals(edit.newString())) {
            throw new EditException("old_string and new_string are identical" + where + ".");
        }
        List<Integer> lines = matchLines(content, oldString);
        if (lines.isEmpty()) {
            throw new EditException(notFoundMessage(content, oldString, where));
        }
        if (lines.size() > 1 && !edit.replaceAll()) {
            throw new EditException("old_string occurs " + lines.size() + " times" + where + ", on lines " + join(lines)
                    + ". Add surrounding lines to make it unique, or set replace_all.");
        }
        return edit.replaceAll()
                ? content.replace(oldString, edit.newString())
                : replaceFirst(content, oldString, edit.newString());
    }

    private static String replaceFirst(String content, String oldString, String newString) {
        int index = content.indexOf(oldString);
        return content.substring(0, index) + newString + content.substring(index + oldString.length());
    }

    /**
     * The 1-based line numbers where {@code oldString} starts.
     *
     * @param content the normalized content
     * @param oldString the normalized text to find
     * @return every match position, as line numbers
     */
    static List<Integer> matchLines(String content, String oldString) {
        List<Integer> lines = new ArrayList<>();
        int index = content.indexOf(oldString);
        while (index >= 0) {
            lines.add(lineOf(content, index));
            index = content.indexOf(oldString, index + 1);
        }
        return lines;
    }

    private static int lineOf(String content, int index) {
        int line = 1;
        for (int i = 0; i < index; i++) {
            if (content.charAt(i) == '\n') {
                line++;
            }
        }
        return line;
    }

    /**
     * The message for a miss: the closest lines in the file, so the next attempt can be corrected
     * rather than guessed.
     *
     * @param content the normalized content
     * @param oldString the normalized text that was not found
     * @param where which edit of the batch failed
     * @return the message
     */
    static String notFoundMessage(String content, String oldString, String where) {
        StringBuilder message = new StringBuilder("old_string was not found" + where + " — nothing was changed.");
        List<String> candidates = candidates(content, oldString);
        if (candidates.isEmpty()) {
            message.append(" No similar line exists; read the file again and copy the text from it"
                    + " (without the line numbers the read tool prints).");
        } else {
            message.append(" The closest lines in the file are:");
            for (String candidate : candidates) {
                message.append(System.lineSeparator()).append("  ").append(candidate);
            }
        }
        return message.toString();
    }

    /**
     * The lines most similar to the first line of {@code oldString}.
     *
     * @param content the normalized content
     * @param oldString the text that was not found
     * @return up to {@value #MAX_CANDIDATES} lines as {@code "  12: text"}, best first
     */
    static List<String> candidates(String content, String oldString) {
        String needle = oldString.lines().findFirst().orElse(oldString).strip();
        if (needle.isEmpty()) {
            return List.of();
        }
        record Candidate(int line, String text, double score) {}
        List<Candidate> scored = new ArrayList<>();
        String[] lines = content.split("\n", -1);
        for (int i = 0; i < lines.length; i++) {
            double score = similarity(needle, lines[i].strip());
            if (score >= CANDIDATE_MIN_SIMILARITY) {
                scored.add(new Candidate(i + 1, lines[i], score));
            }
        }
        return scored.stream()
                .sorted((a, b) -> Double.compare(b.score(), a.score()))
                .limit(MAX_CANDIDATES)
                .map(candidate -> candidate.line() + ": " + candidate.text())
                .toList();
    }

    /**
     * How similar two lines are, as 1 minus the edit distance over the longer length.
     *
     * @param a one line
     * @param b the other
     * @return a value between 0 and 1
     */
    static double similarity(String a, String b) {
        if (a.equals(b)) {
            return 1;
        }
        if (a.isEmpty() || b.isEmpty()) {
            return 0;
        }
        int distance = editDistance(a, b);
        return 1.0 - (double) distance / Math.max(a.length(), b.length());
    }

    private static int editDistance(String a, String b) {
        int[] previous = new int[b.length() + 1];
        int[] current = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            current[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int substitution = previous[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1);
                current[j] = Math.min(substitution, Math.min(previous[j] + 1, current[j - 1] + 1));
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[b.length()];
    }

    private static String join(List<Integer> lines) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < lines.size(); i++) {
            text.append(i == 0 ? "" : ", ").append(lines.get(i));
        }
        return text.toString();
    }

    /**
     * The lines around an edit, so the answer shows what the file now looks like instead of the whole
     * file or nothing at all.
     *
     * @param content the content after the edit, in LF form
     * @param oldString the text that was replaced, to locate the region
     * @param newString what it was replaced with
     * @param context how many lines above and below are included
     * @return the snippet as {@code "  12: text"} lines, or {@code null} when the region is gone
     */
    static @Nullable String snippet(String content, String oldString, String newString, int context) {
        int index = newString.isEmpty() ? content.indexOf(normalize(oldString)) : content.indexOf(newString);
        if (index < 0) {
            return null;
        }
        int match = lineOf(content, index);
        int start = Math.max(1, match - context);
        // the replacement may be several lines; the window ends after it, not after a fixed size
        int end = match + Math.max(0, (int) newString.lines().count() - 1) + context;
        String[] lines = content.split("\n", -1);
        StringBuilder text = new StringBuilder();
        for (int i = start; i <= Math.min(end, lines.length); i++) {
            text.append(i == start ? "" : System.lineSeparator())
                    .append("  ")
                    .append(i)
                    .append(": ")
                    .append(lines[i - 1]);
        }
        return text.toString();
    }
}
