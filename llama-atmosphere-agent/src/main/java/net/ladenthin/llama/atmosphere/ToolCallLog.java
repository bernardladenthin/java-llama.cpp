// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Every tool call of the session, in order, for {@code /calls}.
 *
 * <p>It answers one question the transcript cannot: <em>did that actually happen?</em> A model that
 * runs out of context, or simply drifts, starts describing work instead of doing it — reporting an
 * exit code, a created file, a passing test, all invented. The scrollback looks convincing, because
 * the prose is the same either way. This log only ever grows when a tool really ran, so an empty or
 * short list is the proof.
 *
 * <p>Kept small on purpose: one line per call, arguments and result cut hard. It is a receipt, not a
 * second transcript.
 */
public final class ToolCallLog {

    /** Characters kept of the arguments and of the result. */
    private static final int PREVIEW_CHARS = 120;

    private static final DateTimeFormatter TIME = DateTimeFormatter.ofPattern("HH:mm:ss");

    private final List<Entry> entries = new ArrayList<>();

    /**
     * One recorded call.
     *
     * @param time when it ran
     * @param turn the user turn it belonged to, counting from 1
     * @param name the tool
     * @param arguments the arguments, shortened
     * @param result what came back, shortened
     */
    public record Entry(LocalTime time, int turn, String name, String arguments, String result) {}

    /**
     * Record the calls of one finished turn.
     *
     * @param turn the turn number
     * @param rounds the calls, in order
     */
    public void add(int turn, List<ConsoleSession.ToolRound> rounds) {
        for (ConsoleSession.ToolRound round : rounds) {
            entries.add(new Entry(
                    LocalTime.now(), turn, round.name(), cut(round.argumentsJson()), cut(oneLine(round.result()))));
        }
    }

    /**
     * How many calls were made in this session.
     *
     * @return the count
     */
    public int size() {
        return entries.size();
    }

    /**
     * The log as the console shows it.
     *
     * @return one line per call, or a sentence saying there were none
     */
    public String render() {
        if (entries.isEmpty()) {
            return "No tool has been called in this session — everything so far was text only.";
        }
        StringBuilder text = new StringBuilder();
        for (Entry entry : entries) {
            text.append(entry.time().format(TIME))
                    .append("  turn ")
                    .append(entry.turn())
                    .append("  ")
                    .append(entry.name())
                    .append(" ")
                    .append(entry.arguments())
                    .append(System.lineSeparator())
                    .append("             ↳ ")
                    .append(entry.result())
                    .append(System.lineSeparator());
        }
        text.append(entries.size()).append(" calls");
        return text.toString();
    }

    private static String oneLine(String text) {
        return text.replace("\r\n", " ").replace('\n', ' ').strip();
    }

    private static String cut(String text) {
        return text.length() <= PREVIEW_CHARS ? text : text.substring(0, PREVIEW_CHARS) + " …";
    }
}
