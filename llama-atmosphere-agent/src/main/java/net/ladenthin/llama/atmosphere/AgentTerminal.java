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
     * Ask a question and read the answer.
     *
     * <p>The answer is a line, terminated with Enter, on both consoles. A real terminal could read a
     * single key instead, but only by opening a second reader on the keyboard, and the one reader it
     * has is busy offering the prompt that stays visible while the agent works — which is worth more
     * than saving an Enter on a question that is asked a few times a session.
     *
     * @param prompt the question to show
     * @return the answer in lower case, or {@code null} at end of input
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
     * Wipe the window, leaving the input line and the block at the bottom.
     *
     * <p>The scrollback of the terminal emulator is not touched — what was written stays where the
     * scrollbar can reach it. This only clears what is on screen, the way {@code clear} or Ctrl-L does.
     *
     * <p>A plain stream has no screen to clear, so the default does nothing.
     */
    default void clearScreen() {
        // nothing to wipe on a stream
    }

    /**
     * Whether the user has already typed a line that nobody has read yet.
     *
     * <p>This is what makes the prompt useful during a turn: a console that keeps reading while the
     * agent works can say so, and the turn is then cut short and the line answered instead of being
     * made to wait for an answer nobody wants any more. The line stays queued — the caller reads it
     * with {@link #readLine} as usual.
     *
     * <p>A console that reads only when asked has nothing pending by definition, which is the default.
     *
     * @return {@code true} when a line is waiting
     */
    default boolean hasPendingInput() {
        return false;
    }

    /**
     * Ask the terminal to run {@code action} when the user presses shift+tab at the prompt.
     *
     * <p>Only a real terminal can offer this: it needs to own the keyboard and see a key that is not
     * a line of text. It also only fires **while a line is being read** — during a turn nobody is
     * reading keys, so the mode is switched between turns, which is when it matters.
     *
     * <p>The default is to decline, which every non-interactive console does; the caller uses that
     * answer to decide whether to advertise the shortcut, and {@code /mode} remains either way.
     *
     * @param action what to run on the key; it must not block
     * @return {@code true} when the key was bound
     */
    default boolean onCycleMode(Runnable action) {
        return false;
    }

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
