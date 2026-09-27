// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import org.jline.terminal.Size;
import org.jline.terminal.impl.LineDisciplineTerminal;
import org.jline.utils.InfoCmp;
import org.jline.utils.ScreenTerminal;
import org.jline.utils.ScreenTerminalOutputStream;

/**
 * A terminal with a real screen behind it, for asserting what a person would see.
 *
 * <p>This exists because the stream-backed terminal the other tests use cannot answer the question
 * that matters. Escape sequences are what the console writes, not what it shows: a redraw landing on
 * the wrong column, a torn sequence whose {@code ESC} was lost and whose tail is printed as text, a
 * block drawn twice — all of that is a perfectly ordinary byte stream and only a defect once a
 * terminal has interpreted it. Several assertions in this package were written against bytes, passed,
 * and had to be deleted when the real console disagreed; one of them was green with the fix it was
 * written for switched off.
 *
 * <p>The screen is JLine's own {@link ScreenTerminal} — the same VT interpreter its internal tests use
 * — driven through {@link ScreenTerminalOutputStream}. Both are public API in the shipped jar, so no
 * copied code is involved. Input is fed in with {@link #type(String)} and the interpreted screen read
 * back with {@link #rows()}.
 *
 * <p>It is deliberately NOT {@code AutoCloseable}: {@code close()} is final in JLine's
 * {@code AbstractTerminal}, so a tolerant override is impossible, and the console under test closes the
 * terminal itself. A test therefore puts only the console in a try-with-resources and keeps the harness
 * in a plain local, or every test ends in an error about a terminal that is already closed.
 *
 * <p>The terminal type is {@code windows-vtp} on purpose: it is where every report in this class's
 * history came from, and it differs from {@code xterm} in ways that matter here (no
 * {@code eat_newline_glitch} in older JLine, no {@code scroll_reverse}, no {@code key_btab}).
 */
final class ScreenTerminalHarness extends LineDisciplineTerminal {

    private final ScreenTerminal screen;

    @SuppressWarnings("this-escape")
    ScreenTerminalHarness(String type, int columns, int rows) throws IOException {
        super("screen-harness", type, new ScreenTerminalOutputStream.DelegateOutputStream(), StandardCharsets.UTF_8);
        setSize(Size.of(columns, rows));
        boolean delayedWrap = getBooleanCapability(InfoCmp.Capability.eat_newline_glitch);
        screen = new ScreenTerminal(columns, rows, delayedWrap);
        OutputStream feedback = new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                // The terminal answers some queries (a cursor-position report) on its own input side.
                processInputByte(b);
            }
        };
        ((ScreenTerminalOutputStream.DelegateOutputStream) masterOutput)
                .setDelegate(new ScreenTerminalOutputStream(screen, StandardCharsets.UTF_8, feedback));
    }

    /**
     * Feed keystrokes in, as a person typing would.
     *
     * @param keystrokes what was typed
     * @throws IOException if the terminal is gone
     */
    void type(String keystrokes) throws IOException {
        processInputBytes(keystrokes.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Resize the window, screen included, and tell the application.
     *
     * <p>Both halves are needed: the screen has to change shape, and the size change has to be
     * announced the way a console announces it. A test that only calls {@code setSize} changes what
     * the application reads back without ever redrawing anything.
     *
     * @param columns the new width
     * @param rows the new height
     */
    void resize(int columns, int rows) {
        screen.setSize(Size.of(columns, rows));
        setSize(Size.of(columns, rows));
        raise(Signal.WINCH);
    }

    /**
     * The screen as a person would read it, one string per row, trailing blanks kept.
     *
     * @return the rows, top to bottom
     */
    String[] rows() {
        Size size = getSize();
        long[] dump = new long[size.getRows() * size.getColumns()];
        screen.dump(dump, 0, 0, size.getRows(), size.getColumns(), null);
        String[] out = new String[size.getRows()];
        for (int row = 0; row < size.getRows(); row++) {
            StringBuilder text = new StringBuilder();
            for (int column = 0; column < size.getColumns(); column++) {
                text.append((char) dump[column + size.getColumns() * row]);
            }
            out[row] = text.toString();
        }
        return out;
    }

    /**
     * The whole screen as one string, rows separated by newlines — for a failure message.
     *
     * @return the screen, ready to print
     */
    String describe() {
        StringBuilder text = new StringBuilder();
        String[] rows = rows();
        for (int row = 0; row < rows.length; row++) {
            text.append(String.format("%2d|%s|%n", row, rows[row]));
        }
        return text.toString();
    }

    /**
     * How many rows carry anything at all.
     *
     * @return the count of non-blank rows
     */
    int nonBlankRows() {
        int count = 0;
        for (String row : rows()) {
            if (!row.isBlank()) {
                count++;
            }
        }
        return count;
    }
}
