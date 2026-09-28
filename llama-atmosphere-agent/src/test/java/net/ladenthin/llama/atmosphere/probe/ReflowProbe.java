// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere.probe;

import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

/**
 * Establishes, on a real console, what a resize does to lines that are already on the screen.
 *
 * <p>Two questions, and the second is the one an emulator cannot guess:
 *
 * <ol>
 *   <li><b>Does the console reflow?</b> A line longer than the window occupies two screen rows; widening the
 *       window may join it back into one. Each wide line here carries its number at the start and {@code =END}
 *       at the end, so a joined line shows both on ONE row.
 *   <li><b>Which edge keeps its content when joining frees rows?</b> This only has an answer when the screen is
 *       FULL, so the probe fills it first with numbered short lines. After widening, either the numbers at the
 *       top stay and everything below moves up, or the bottom row keeps what it had and older lines appear at
 *       the top. That difference decides whether a bar pinned to the bottom is carried away by a reflow, which
 *       is the defect being chased.
 * </ol>
 *
 * <p><b>The first version of this probe measured nothing</b>, and it is worth saying why: it built its wide
 * lines exactly as wide as the window, and a line of exactly {@code columns} characters does not wrap — it fills
 * the row and the cursor waits in the last cell. The lines here are deliberately {@code columns + 12} long.
 *
 * <p>No line reader, no status bar, nothing pinned: only plain writing, so what the screen does is the console's
 * own behaviour and nothing of this project's.
 *
 * <p><b>It lives in the test tree but is not a test</b>: it needs a real console, which a test does not have, and
 * its answer is read by a person. Surefire ignores it (the name does not match {@code *Test}). Run it with the
 * terminal library the agent uses:
 *
 * <pre>
 * mvn -q test-compile exec:java -Dexec.classpathScope=test  *     -Dexec.mainClass=net.ladenthin.llama.atmosphere.probe.ReflowProbe  *     "-Dllama.version=5.2.0" "-Djline.version=4.4.6-atmosphere"
 * </pre>
 *
 * <p>What its answers were used for, and the numbers they produced, is in
 * {@code docs/upstream-investigation-jline-status-windows-redraw.md}; {@code ReflowingScreenHarness} is the model
 * they anchor.
 */
public final class ReflowProbe {

    private ReflowProbe() {}

    /**
     * Run the probe.
     *
     * @param args ignored
     * @throws Exception if the terminal cannot be built
     */
    public static void main(String[] args) throws Exception {
        try (Terminal terminal = TerminalBuilder.builder().system(true).build()) {
            Size size = terminal.getSize();
            int columns = size.getColumns();
            int rows = size.getRows();

            // Fill the screen so the question "which edge keeps its content" has an answer at all. The FILL
            // lines are short, so they can never wrap: whatever happens to them is not reflow.
            int fill = Math.max(1, rows - 9);
            for (int line = 1; line <= fill; line++) {
                terminal.writer().println("FILL" + two(line) + " (short, cannot wrap)");
            }
            // Three lines LONGER than the window, so each must occupy two screen rows.
            for (int line = 1; line <= 3; line++) {
                terminal.writer().println(wide(line, columns));
            }
            terminal.writer().println("BOTTOM-MARKER (the last line printed)");
            terminal.writer().println();
            terminal.writer()
                    .println("Window: " + columns + " columns, " + rows + " rows, type " + terminal.getType()
                            + ". Each WIDE line is " + (columns + 12) + " characters.");
            terminal.writer().println("STEP 1: drag the window WIDER, then press Enter and copy the whole screen.");
            terminal.writer().flush();

            for (int round = 1; round <= 3; round++) {
                if (terminal.reader().read() < 0) {
                    return;
                }
                Size now = terminal.getSize();
                // The number that settles it, and it has to be asked BEFORE anything else is printed: the last
                // thing written went to the cursor, so the cursor's row says where the bottom of the content
                // sits in the WINDOW. A pasted screen cannot answer this -- Windows Terminal copies the whole
                // scrollback, not the visible part -- and a cursor-position report can, because this probe owns
                // the keyboard: there is no line reader here to compete with.
                int cursorRow = -1;
                try {
                    org.jline.terminal.Cursor cursor = terminal.getCursorPosition(codepoint -> {});
                    if (cursor != null) {
                        cursorRow = cursor.getY();
                    }
                } catch (RuntimeException e) {
                    cursorRow = -1;
                }
                terminal.writer()
                        .println("--- after step " + round + ": " + now.getColumns() + " columns, "
                                + now.getRows() + " rows, CURSOR ON ROW " + cursorRow + " of " + now.getRows()
                                + " (last row would be " + (now.getRows() - 1) + ")");
                if (round == 1) {
                    terminal.writer().println("STEP 2: drag it NARROWER than at the start, Enter, copy again.");
                } else if (round == 2) {
                    terminal.writer().println("STEP 3: drag it wider and narrower quickly, Enter, copy again.");
                }
                terminal.writer().flush();
            }
        }
    }

    /**
     * A line LONGER than the window, so the console has to wrap it.
     *
     * @param number which line this is
     * @param columns the window's width
     * @return the line, {@code columns + 12} characters long
     */
    private static String wide(int number, int columns) {
        String head = "WIDE" + number + "-START ";
        String tail = " WIDE" + number + "-=END";
        int filler = Math.max(1, columns + 12 - head.length() - tail.length());
        return head + "-".repeat(filler) + tail;
    }

    private static String two(int number) {
        return number < 10 ? "0" + number : Integer.toString(number);
    }
}
