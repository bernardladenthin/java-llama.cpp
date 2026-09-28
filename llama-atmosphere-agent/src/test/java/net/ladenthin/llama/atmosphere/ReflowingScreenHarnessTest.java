// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * The reflowing screen itself, under test.
 *
 * <p><b>Why this class comes first.</b> {@link ReflowingScreenHarness} exists to model one behaviour of a real
 * console, and a model is worth exactly as much as the evidence that it behaves like the thing it models. Every
 * assertion here is a statement about a console that can be checked by hand — {@code ReflowProbe} prints the
 * same pattern on a real one — so a case built on this harness rests on a measurement rather than on an
 * assumption. If the console disagrees with any of these, the harness is wrong and the cases above it mean
 * nothing.
 *
 * <p>The pattern is the probe's: a line printed exactly as wide as the window must occupy two screen rows, and
 * the two must be joined again when the window is widened. A line printed well inside the window is the
 * control — it cannot wrap, so nothing may happen to it.
 */
class ReflowingScreenHarnessTest {

    private static final int ROWS = 10;
    private static final int NARROW = 40;
    private static final int WIDE = 80;

    /**
     * A line WIDER than the window, so the console has to wrap it.
     *
     * <p>Wider, not equal: a line of exactly {@code columns} characters fills the row and stops there — with
     * delayed wrap the cursor waits in the last cell and nothing continues onto the next row. Measured, after
     * this test first asserted a wrap that never happened.
     *
     * @param number which line this is
     * @param columns the window's width
     * @return the line, {@code columns + 5} characters long
     */
    private String wide(int number, int columns) {
        String head = "W" + number + " ";
        String tail = " =END";
        return head + "-".repeat(columns + 5 - head.length() - tail.length()) + tail;
    }

    private int rowsCarrying(ReflowingScreenHarness terminal, String text) {
        int found = 0;
        for (String row : terminal.rows()) {
            if (row.contains(text)) {
                found++;
            }
        }
        return found;
    }

    @Test
    void aLineAsWideAsTheWindowTakesTwoRowsAndIsJoinedAgainWhenTheWindowGrows() throws Exception {
        ReflowingScreenHarness terminal = new ReflowingScreenHarness("windows-vtp", NARROW, ROWS);
        terminal.writeBehindTheApplicationsBack("\u001b[1;1H" + wide(1, NARROW));

        // At the narrow width the line has wrapped: its head and its tail are on DIFFERENT rows.
        assertThat("the head is on one row" + terminal.describe(), rowsCarrying(terminal, "W1 "), is(1));
        assertThat("the tail on another" + terminal.describe(), rowsCarrying(terminal, "=END"), is(1));
        assertThat(
                "and not on the same one" + terminal.describe(),
                rowsCarrying(terminal, "W1 ") == 1 && terminal.rows()[0].contains("=END"),
                is(false));

        terminal.resize(WIDE, ROWS);

        // Widened: the two are one logical line again, so head and tail share a row.
        String[] rows = terminal.rows();
        int carrying = 0;
        for (String row : rows) {
            if (row.contains("W1 ") && row.contains("=END")) {
                carrying++;
            }
        }
        assertThat("the line is joined again" + terminal.describe(), carrying, is(1));
    }

    @Test
    void aLineThatNeverWrappedIsLeftAloneByAResize() throws Exception {
        // The control. If this ever fails, the harness is moving content that a console would not touch, and
        // every case built on it is suspect.
        ReflowingScreenHarness terminal = new ReflowingScreenHarness("windows-vtp", WIDE, ROWS);
        terminal.writeBehindTheApplicationsBack("\u001b[3;1HSHORT1 =END\u001b[4;1HSHORT2 =END");

        terminal.resize(NARROW, ROWS);

        assertThat("SHORT1 is on exactly one row" + terminal.describe(), rowsCarrying(terminal, "SHORT1"), is(1));
        assertThat("SHORT2 is on exactly one row" + terminal.describe(), rowsCarrying(terminal, "SHORT2"), is(1));
        assertThat("and they are still whole" + terminal.describe(), rowsCarrying(terminal, "SHORT1 =END"), is(1));
    }

    @Test
    void joiningLinesFreesRowsAndWHICHedgeKeepsItsContentIsTheHarnessSquestion() throws Exception {
        // The consequence that decides every case built on this harness, and it is asserted BOTH ways on
        // purpose: joining wrapped lines makes the content need fewer rows, and where the freed rows appear
        // decides whether a pinned bar at the bottom stays put or is carried upwards. Which one a Windows
        // console does is what ReflowProbe asks it; until that answer is in, this test pins the harness's two
        // behaviours rather than a belief about the console.
        int bottom = bottomRowAfterReflow(ReflowingScreenHarness.Anchor.BOTTOM);
        int top = bottomRowAfterReflow(ReflowingScreenHarness.Anchor.TOP);
        System.out.println("ANCHOR bottom -> marker on row " + bottom + ", top -> row " + top);
        assertThat("with the bottom anchored the marker keeps the last row", bottom, is(ROWS - 1));
        assertThat("with the top anchored it moves up: " + top, top < ROWS - 1, is(true));
    }

    /**
     * Print three wrapping lines plus a marker on the last row, widen the window, and say where the marker went.
     *
     * @param anchor which edge keeps its content
     * @return the marker's row after the reflow
     */
    private int bottomRowAfterReflow(ReflowingScreenHarness.Anchor anchor) throws Exception {
        ReflowingScreenHarness terminal = new ReflowingScreenHarness("windows-vtp", NARROW, ROWS, anchor);
        StringBuilder painted = new StringBuilder();
        for (int line = 1; line <= 3; line++) {
            painted.append("[").append(line * 2 - 1).append(";1H").append(wide(line, NARROW));
        }
        painted.append("[").append(ROWS).append(";1HAT-THE-BOTTOM");
        terminal.writeBehindTheApplicationsBack(painted.toString());

        terminal.resize(WIDE, ROWS);

        return rowOf(terminal, "AT-THE-BOTTOM");
    }

    private int rowOf(ReflowingScreenHarness terminal, String text) {
        String[] rows = terminal.rows();
        for (int row = 0; row < rows.length; row++) {
            if (rows[row].contains(text)) {
                return row;
            }
        }
        return -1;
    }
}
