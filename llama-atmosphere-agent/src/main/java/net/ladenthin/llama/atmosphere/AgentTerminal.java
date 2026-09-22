// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import org.jspecify.annotations.Nullable;

/**
 * Everything the agent needs from a console, so that a real terminal and a plain stream are
 * interchangeable.
 *
 * <p>Two implementations: {@link JLineTerminal} for an interactive session (line editing, history,
 * tab completion, a status line pinned to the bottom, single-key answers) and {@link PlainTerminal}
 * for everything else — one-shot runs, piped input, and the tests. The agent picks one at startup and
 * never branches again.
 *
 * <p>Output is line-oriented on purpose: a line is written once, complete, and is never touched
 * again. That is what lets the pinned status line coexist with streamed output without redrawing
 * anything the reader has already seen.
 */
public interface AgentTerminal extends AutoCloseable {

    /**
     * Write one completed line, above the prompt and the status line.
     *
     * @param text the line, without a terminator
     */
    void line(String text);

    /**
     * Read one line from the user.
     *
     * @param prompt the prompt to show, e.g. {@code "you> "}
     * @return the line, or {@code null} at end of input
     */
    @Nullable
    String readLine(String prompt);

    /**
     * Read a single answer, without waiting for Enter where the terminal allows it.
     *
     * @param prompt the question to show
     * @return the answer in lower case (a single key, or a whole line on a plain stream), or
     *     {@code null} at end of input
     */
    @Nullable
    String readKey(String prompt);

    /**
     * Set the status line kept at the bottom of the window.
     *
     * @param text the line; an empty string removes it
     */
    void status(String text);

    /**
     * The styles to use for this console.
     *
     * @return a colouring instance on a terminal, a plain one otherwise
     */
    Ansi ansi();

    /** Restore the terminal. */
    @Override
    void close();
}
