// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * The REPL's client-side commands: a line the agent answers itself instead of sending it to the model.
 *
 * <p>Dispatch is deliberately conservative. A line is a command only when it starts with {@code /}
 * <em>and</em> its first word names a known command; everything else — including an unknown
 * {@code /foo} — goes to the model verbatim. That is what makes {@code /usr/bin/env} or a line of
 * Markdown work with no escape syntax, at the price of a typo being answered by the model rather than
 * rejected. (aider rejects unknown commands outright and needs no escape because its REPL is
 * line-oriented; this agent is asked prose far more often than it is asked commands.)
 *
 * @param command the command
 * @param arguments the rest of the line, trimmed; empty when the line was just the command
 */
public record SlashCommands(Command command, String arguments) {

    /** The commands the REPL answers itself. */
    public enum Command {
        /** Print the command overview. */
        HELP("this overview", null, "/help", "/?", "/commands"),
        /** Drop the conversation history, and wipe the screen with it. */
        CLEAR("drop the history and start over", null, "/clear", "/reset", "/new"),
        /** Wipe the screen, keeping the conversation. */
        CLS("wipe the screen, keep the conversation", null, "/cls", "/clear-screen"),
        /** Write what was said, with the time, to a file in the workspace. */
        SAVE("write what was said, with timestamps, into the workspace", "file name", "/save", "/transcript"),
        /** Ask the last question again, without the answer that came back. */
        RETRY("ask the last question again, dropping the answer", null, "/retry", "/again"),
        /** Read a saved transcript back in as the conversation. */
        LOAD("read a saved transcript back in as the conversation", "file name", "/load", "/resume"),
        /** Summarize the history and continue with the summary; the argument steers the summary. */
        COMPACT("summarize the history and continue with the summary", "what to focus on", "/compact"),
        /** Keep working on one task until it is done; see {@link TaskLoop}. */
        LOOP(
                "keep working on a task until it is done (needs auto mode)",
                "[--every 5m] [--max 20] [--check '<cmd>'] <task>",
                "/loop"),
        /** Show or set the approval mode; the argument is {@code manual} or {@code auto}. */
        MODE("show or set the approval mode", "manual or auto", "/mode", "/approve"),
        /** Print endpoint, model, tools, approval mode and context usage. */
        STATUS("model, tools, mode and context use", null, "/status"),
        /** List the tools offered to the model. */
        TOOLS("the tools offered to the model", null, "/tools"),
        /** Show every tool call of this session — the receipt for what really happened. */
        CALLS("every tool call of this session, with its result", null, "/calls", "/log"),
        /** Leave the REPL. */
        EXIT("leave", null, "/exit", "/quit");

        private final String description;
        private final @Nullable String argumentHint;
        private final List<String> names;

        Command(String description, @Nullable String argumentHint, String... names) {
            this.description = description;
            this.argumentHint = argumentHint;
            this.names = List.of(names);
        }

        /**
         * One line saying what the command does, for front ends that list the commands (an editor's
         * {@code /} menu).
         *
         * @return the description
         */
        public String description() {
            return description;
        }

        /**
         * What the command's argument is, when it takes one.
         *
         * @return a short hint, or {@code null} for a command without an argument
         */
        public @Nullable String argumentHint() {
            return argumentHint;
        }

        /**
         * The names that select this command, the canonical one first.
         *
         * @return the names, each including the leading slash
         */
        public List<String> names() {
            return names;
        }

        /**
         * The canonical name.
         *
         * @return e.g. {@code "/help"}
         */
        public String canonicalName() {
            return names.get(0);
        }
    }

    /**
     * Parse one REPL line.
     *
     * @param line the raw line as typed
     * @return the command and its arguments, or empty when the line is a message for the model
     */
    public static Optional<SlashCommands> parse(@Nullable String line) {
        // A BOM at the start of piped input would otherwise hide the slash and send /help to the model.
        String trimmed = line == null ? "" : line.replace("﻿", "").trim();
        if (!trimmed.startsWith("/")) {
            return Optional.empty();
        }
        int space = indexOfWhitespace(trimmed);
        String name = (space < 0 ? trimmed : trimmed.substring(0, space)).toLowerCase(Locale.ROOT);
        String arguments = space < 0 ? "" : trimmed.substring(space + 1).trim();
        for (Command command : Command.values()) {
            if (command.names().contains(name)) {
                return Optional.of(new SlashCommands(command, arguments));
            }
        }
        return Optional.empty();
    }

    private static int indexOfWhitespace(String text) {
        for (int i = 0; i < text.length(); i++) {
            if (Character.isWhitespace(text.charAt(i))) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Whether an argument was given.
     *
     * @return {@code true} when the line carried more than the command name
     */
    public boolean hasArguments() {
        return !arguments.isEmpty();
    }
}
