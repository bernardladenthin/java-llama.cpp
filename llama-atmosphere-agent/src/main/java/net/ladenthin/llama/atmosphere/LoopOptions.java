// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.time.Duration;
import java.util.Locale;
import org.jspecify.annotations.Nullable;

/**
 * What {@code /loop} was asked to do: the task, and the limits that stop it.
 *
 * <p>Syntax: {@code /loop [--every <duration>] [--max <n>] [--check <command>] <task>}. The flags come
 * first, the rest of the line is the task verbatim — so a task may contain anything, including words
 * that look like flags, as long as they are not at the front.
 *
 * @param task what the agent should work on, restated unchanged every step
 * @param interval the pause between steps, or {@code null} to run them back to back
 * @param maxSteps the hard cap on steps
 * @param check a command that must succeed before {@code <<TASK_COMPLETE>>} is accepted, or
 *     {@code null} to trust the model
 */
public record LoopOptions(
        String task,
        @Nullable Duration interval,
        int maxSteps,
        @Nullable String check) {

    /** Steps before the loop gives up on its own. */
    public static final int DEFAULT_MAX_STEPS = 20;

    /**
     * Parse the argument of {@code /loop}.
     *
     * @param arguments everything after the command name
     * @return the parsed options
     * @throws IllegalArgumentException when a flag has no value, a duration or number is malformed, or
     *     no task is left
     */
    public static LoopOptions parse(String arguments) {
        String rest = arguments == null ? "" : arguments.trim();
        Duration interval = null;
        int maxSteps = DEFAULT_MAX_STEPS;
        String check = null;
        while (rest.startsWith("--")) {
            String[] flag = split(rest);
            switch (flag[0]) {
                case "--every" -> {
                    String[] value = split(flag[1]);
                    interval = parseDuration(require(value[0], "--every"));
                    rest = value[1];
                }
                case "--max" -> {
                    String[] value = split(flag[1]);
                    maxSteps = parseSteps(require(value[0], "--max"));
                    rest = value[1];
                }
                case "--check" -> {
                    String[] value = splitQuoted(flag[1]);
                    check = require(value[0], "--check");
                    rest = value[1];
                }
                default -> throw new IllegalArgumentException("Unknown /loop flag: " + flag[0]);
            }
        }
        if (rest.isEmpty()) {
            throw new IllegalArgumentException("Usage: /loop [--every 5m] [--max 20] [--check '<cmd>'] <task>");
        }
        return new LoopOptions(rest, interval, maxSteps, check);
    }

    /**
     * A duration written the way a human writes it: {@code 30s}, {@code 5m}, {@code 2h}, or a bare
     * number of minutes.
     *
     * @param text the duration
     * @return the parsed duration
     * @throws IllegalArgumentException when it is not a duration or is not positive
     */
    static Duration parseDuration(String text) {
        String value = text.trim().toLowerCase(Locale.ROOT);
        char unit = value.charAt(value.length() - 1);
        String number = Character.isDigit(unit) ? value : value.substring(0, value.length() - 1);
        long amount;
        try {
            amount = Long.parseLong(number);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Not a duration: " + text + " (try 30s, 5m or 2h)", e);
        }
        Duration duration =
                switch (Character.isDigit(unit) ? 'm' : unit) {
                    case 's' -> Duration.ofSeconds(amount);
                    case 'm' -> Duration.ofMinutes(amount);
                    case 'h' -> Duration.ofHours(amount);
                    default -> throw new IllegalArgumentException("Unknown time unit in " + text + " (use s, m or h)");
                };
        if (duration.isZero() || duration.isNegative()) {
            throw new IllegalArgumentException("The interval must be positive: " + text);
        }
        return duration;
    }

    private static int parseSteps(String text) {
        try {
            int steps = Integer.parseInt(text.trim());
            if (steps <= 0) {
                throw new IllegalArgumentException("--max must be positive: " + text);
            }
            return steps;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("--max expects a number, got: " + text, e);
        }
    }

    private static String require(String value, String flag) {
        if (value.isEmpty()) {
            throw new IllegalArgumentException("Missing value for " + flag);
        }
        return value;
    }

    /** Split off the first whitespace-separated word: {@code [word, rest]}. */
    private static String[] split(String text) {
        int space = text.indexOf(' ');
        return space < 0
                ? new String[] {text.trim(), ""}
                : new String[] {
                    text.substring(0, space).trim(), text.substring(space + 1).trim()
                };
    }

    /** Like {@link #split}, but a leading {@code '...'} or {@code "..."} keeps its spaces. */
    private static String[] splitQuoted(String text) {
        String trimmed = text.trim();
        if (trimmed.startsWith("'") || trimmed.startsWith("\"")) {
            char quote = trimmed.charAt(0);
            int end = trimmed.indexOf(quote, 1);
            if (end > 0) {
                return new String[] {
                    trimmed.substring(1, end), trimmed.substring(end + 1).trim()
                };
            }
        }
        return split(trimmed);
    }
}
