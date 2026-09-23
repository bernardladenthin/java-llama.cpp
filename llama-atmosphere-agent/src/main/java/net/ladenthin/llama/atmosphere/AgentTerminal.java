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
     * Set the block kept at the bottom of the window.
     *
     * <p>Two lines in practice: what the agent is doing right now, and the session's state. Keeping
     * both there at all times is what stops the block from changing height, which would make the
     * output above it jump on every update.
     *
     * @param lines the lines, top to bottom; an empty list removes the block
     */
    void status(java.util.List<String> lines);

    /**
     * Whether {@link #status} really pins the line to the bottom of the window.
     *
     * <p>{@code false} on a plain stream, where the caller has to print the status itself.
     *
     * @return {@code true} on a real terminal
     */
    boolean pinsStatus();

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
