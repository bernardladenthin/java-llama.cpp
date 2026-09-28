// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import org.junit.jupiter.api.Test;

/**
 * What the harness itself does on a resize — checked before believing anything it reports about one.
 *
 * <p>Two cases in {@link ScreenUseCasesTest} report content vanishing after a series of size changes, and
 * a bisection ruled out every writer: with the console's own refresh disabled, and with both halves of
 * JLine's resize handling disabled, they stayed red. That leaves the screen model, so it is asked
 * directly: does text written before a resize survive it? If it does not, those two cases measure this
 * class rather than the console, and they are worth nothing until they are rewritten.
 */
class ScreenTerminalHarnessTest {

    @Test
    void textWrittenBeforeAResizeSurvivesIt() throws Exception {
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 60, 12);
        terminal.writer().println("a line written before the resize");
        terminal.writer().flush();
        assertThat(
                "the line is on screen to begin with" + System.lineSeparator() + terminal.describe(),
                terminal.describe().contains("before the resize"),
                is(true));

        terminal.resize(100, 12);
        Thread.sleep(100);

        assertThat(
                "and it is still there after the resize" + System.lineSeparator() + terminal.describe(),
                terminal.describe().contains("before the resize"),
                is(true));
    }

    @Test
    void textSurvivesSeveralResizesInBothDirections() throws Exception {
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, 12);
        terminal.writer().println("a line written before the resizes");
        terminal.writer().flush();

        for (int columns : new int[] {60, 100, 70, 100}) {
            terminal.resize(columns, 12);
            Thread.sleep(50);
        }

        assertThat(
                "the line survived them all" + System.lineSeparator() + terminal.describe(),
                terminal.describe().contains("before the resizes"),
                is(true));
    }
}
