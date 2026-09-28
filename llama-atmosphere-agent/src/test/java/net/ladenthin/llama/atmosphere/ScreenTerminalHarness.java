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
 * <p>Not final: one test subclasses it to make {@code getSize()} throw once, which is how the size
 * poll's resilience is driven through the real thread rather than around it.
 *
 * <p>The terminal type is {@code windows-vtp} on purpose: it is where every report in this class's
 * history came from, and it differs from {@code xterm} in ways that matter here (no
 * {@code eat_newline_glitch} in older JLine, no {@code scroll_reverse}, no {@code key_btab}).
 */
class ScreenTerminalHarness extends LineDisciplineTerminal {

    private final ScreenTerminal screen;

    /**
     * The screen's OWN geometry, which is not necessarily what the application is told.
     *
     * <p>Every reader below used to ask {@code getSize()} for the shape of the dump, which is the
     * application's view — and a subclass that lies about it (to reproduce a console reporting a size it
     * has not applied) then made the harness misread its own screen: an undersized dump buffer threw
     * {@code ArrayIndexOutOfBoundsException} out of {@code ScreenTerminal.dump}. The screen is the
     * harness's own object, so its shape is the harness's own knowledge and is kept here.
     */
    private Size screenSize;

    @SuppressWarnings("this-escape")
    ScreenTerminalHarness(String type, int columns, int rows) throws IOException {
        super("screen-harness", type, new ScreenTerminalOutputStream.DelegateOutputStream(), StandardCharsets.UTF_8);
        setSize(Size.of(columns, rows));
        screenSize = Size.of(columns, rows);
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
        screenSize = Size.of(columns, rows);
        screen.setSize(Size.of(columns, rows));
        setSize(Size.of(columns, rows));
        raise(Signal.WINCH);
    }

    /**
     * Write straight onto the screen, behind the application's back.
     *
     * <p>What a console does to itself. Windows reflows its screen buffer when the window is widened —
     * rows it had marked as wrapped are joined again — and JLine is never told: its {@code Display} still
     * believes the rows it last wrote are on screen, so the next update computes an empty diff and emits
     * NOTHING. That is why the artefact survives every later redraw and why "ein paar Mal Enter" is what
     * repairs it. The bytes go into {@code masterOutput}, which is the screen's own input, so they reach
     * the screen exactly the way the console's own reflow does: without passing through anything that
     * keeps a model of it.
     *
     * @param ansi the sequence to interpret, e.g. a cursor address followed by text
     * @throws IOException if the screen is gone
     */
    void writeBehindTheApplicationsBack(String ansi) throws IOException {
        masterOutput.write(ansi.getBytes(StandardCharsets.UTF_8));
        masterOutput.flush();
    }

    /**
     * Change the SCREEN's shape without telling the application and without raising a signal.
     *
     * <p>The state a console is briefly in while it is being dragged, and the one shape of this defect
     * that cannot be arranged any other way: the screen already has its new width while
     * {@code getSize()} still reports the old one. A block rendered in that moment is padded to the
     * width the application was told, does not reach the right margin, the terminal does not wrap, and
     * the next row continues on the SAME screen line. A console that reflows its wrapped rows when it is
     * widened — which Windows does — arrives at the identical screen by a different route, so this models
     * both.
     *
     * @param columns the screen's real new width
     * @param rows the screen's real new height
     */
    void resizeScreenOnly(int columns, int rows) {
        screenSize = Size.of(columns, rows);
        screen.setSize(Size.of(columns, rows));
    }

    /**
     * Change the size and announce it from ANOTHER thread, the way a console does.
     *
     * <p>{@link #resize(int, int)} raises the signal on the caller's thread, so the reader's handler runs
     * serialised with whatever the test is doing — which is not how it happens. A real console delivers
     * the size change on its own input pump, concurrently with every other writer, and that
     * interleaving is where the reports live: the same test that is green with a serialised signal shows
     * the block drawn several times over when the signal arrives on its own thread.
     *
     * <p>Deliberately not joined: joining would serialise it again and defeat the point.
     *
     * @param columns the new width
     * @param rows the new height
     * @return the thread the signal was raised on, so a test can wait for it at the very end
     */
    Thread resizeAsynchronously(int columns, int rows) {
        screenSize = Size.of(columns, rows);
        screen.setSize(Size.of(columns, rows));
        setSize(Size.of(columns, rows));
        Thread pump = new Thread(() -> {
            try {
                raise(Signal.WINCH);
            } catch (RuntimeException e) {
                // A console's pump swallows this too -- and that is part of the defect, not of the test.
            }
        });
        pump.setDaemon(true);
        pump.start();
        return pump;
    }

    /**
     * The screen as a person would read it, one string per row, trailing blanks kept.
     *
     * @return the rows, top to bottom
     */
    String[] rows() {
        Size size = screenSize;
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
     * Which screen row the cursor is on.
     *
     * <p>The measurement that finally located the defect, and the one this class never asked for: every
     * earlier assertion here was about the CONTENT of the rows. A probe on the reporter's console showed
     * the console's own cursor two and three rows above where a three-row block expects it -- 32 and 31 in
     * a 38-row window where 34 is right -- while the content still looked plausible. A pinned region is
     * reserved in rows counted from the bottom, so the cursor drifting up by the block's own height is
     * the defect itself rather than a symptom of it.
     *
     * @return the cursor's row, counted from the top
     */
    int cursorRow() {
        Size size = screenSize;
        long[] dump = new long[size.getRows() * size.getColumns()];
        int[] cursor = new int[2];
        screen.dump(dump, cursor);
        return cursor[1];
    }

    /**
     * Which screen column the cursor is on.
     *
     * @return the cursor's column, counted from the left
     */
    int cursorColumn() {
        Size size = screenSize;
        long[] dump = new long[size.getRows() * size.getColumns()];
        int[] cursor = new int[2];
        screen.dump(dump, cursor);
        return cursor[0];
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
