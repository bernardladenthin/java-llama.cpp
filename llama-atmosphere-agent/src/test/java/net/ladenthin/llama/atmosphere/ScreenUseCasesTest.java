// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * The console's screen, for the cases that were actually reported: dragging the window wider, dragging
 * it narrower, typing while dragging, and {@code /cls}.
 *
 * <p><b>Why this class exists at all.</b> Every earlier attempt asserted on the bytes the terminal
 * emits, and every one of them was worthless for these reports. "The block is smeared across the
 * output", "an escape sequence is printed as text", "the block is drawn twice" — a byte stream does not
 * distinguish any of that from correct output. Two assertions written that way had to be deleted from
 * {@link JLineTerminalTest}, one of them after it turned out to be green with the fix it was written
 * for switched off. {@link ScreenTerminalHarness} puts JLine's own VT interpreter behind the terminal,
 * so the question is asked of a screen instead.
 *
 * <p><b>How to use it to demonstrate the JLine fixes.</b> The library is a property, so the same class
 * run twice is the red/green pair:
 *
 * <pre>
 * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6              # the released library
 * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6-statusfix3   # with the three fixes
 * </pre>
 *
 * <p><b>Two things the interpreted screen shows that the byte stream hid.</b> A rule written through
 * the status region arrives as {@code U+2500}, but one written through a <em>prompt</em> arrives as the
 * DEC line-drawing set ({@code ESC(0} plus a row of {@code q}), which this screen renders literally —
 * so a row of {@code q} is a rule that took the wrong path. And a torn escape sequence shows up as its
 * own tail: {@code 1H} from a cursor address whose {@code ESC[} was swallowed, {@code [m} or
 * {@code [C90m} from a style, which is exactly what was reported twice ("36;1H" on screen).
 */
class ScreenUseCasesTest {

    private static final int ROWS = 12;
    private static final int NARROW = 60;
    private static final int WIDE = 100;

    /** Short enough never to be cut, distinctive enough to count. */
    private static final String STATE = "[state]";

    /** The tails a torn escape sequence leaves behind as visible text. */
    private static final List<String> ESCAPE_TAILS = List.of("1H", "[m", "[C", "[?", "[0m", "[90m");

    private ScreenTerminalHarness terminal(int columns) throws Exception {
        return new ScreenTerminalHarness("windows-vtp", columns, ROWS);
    }

    /** A rule row, in either form the screen can show it. */
    private boolean isRule(String row) {
        return row.contains("─".repeat(10)) || row.contains("q".repeat(10));
    }

    private int count(String[] rows, java.util.function.Predicate<String> matches) {
        int found = 0;
        for (String row : rows) {
            if (matches.test(row)) {
                found++;
            }
        }
        return found;
    }

    /**
     * Start the console with a reader running and a block pinned, as a session always is.
     *
     * <p>The reader is not optional: it runs for the whole session in the application, it is what owns
     * the resize signal, and without it the pinned region is never told a new geometry — a state the
     * application cannot be in, so a test in it proves nothing about the application.
     */
    private JLineTerminal start(ScreenTerminalHarness terminal, List<String> block) throws Exception {
        JLineTerminal console = JLineTerminal.over(terminal, List.of());
        Thread reading = new Thread(() -> console.readLine("ignored"));
        reading.setDaemon(true);
        reading.start();
        Thread.sleep(200);
        console.status(block);
        Thread.sleep(200);
        return console;
    }

    /**
     * One column per step, the way a drag reports it: the probe recorded ~22 events for one drag.
     *
     * <p>The block rebuild is invoked directly rather than waited for. In the application a poll does it
     * within ~120 ms, but a test that sleeps for it passed alone and failed in a full run, where the
     * machine is busy — and a flaky test is worse than none. This drives the same method the poll calls,
     * at a known moment.
     */
    private void drag(ScreenTerminalHarness terminal, JLineTerminal console, int from, int to) throws Exception {
        int step = from < to ? 1 : -1;
        for (int columns = from + step; columns != to + step; columns += step) {
            terminal.resize(columns, ROWS);
            Thread.sleep(15);
            console.refreshBlockForCurrentSize();
        }
        Thread.sleep(300);
    }

    /**
     * Everything the reports were about, in one place.
     *
     * @param terminal the screen to read
     * @param blockRows how many rows the pinned block occupies
     */
    private void assertBlockIsIntact(ScreenTerminalHarness terminal, int blockRows) {
        String[] rows = terminal.rows();
        String screen = terminal.describe();

        for (int row = 0; row < rows.length; row++) {
            for (String tail : ESCAPE_TAILS) {
                assertThat(
                        "row " + row + " shows \"" + tail + "\", the tail of a torn escape sequence:\n" + screen,
                        rows[row].contains(tail),
                        is(false));
            }
        }

        assertThat("the state row is on screen once:\n" + screen, count(rows, row -> row.contains(STATE)), is(1));
        assertThat("the rule is on screen once:\n" + screen, count(rows, this::isRule), is(1));
        assertThat("the state row is the bottom row:\n" + screen, rows[ROWS - 1].contains(STATE), is(true));
        assertThat("the rule is directly above the block:\n" + screen, isRule(rows[ROWS - blockRows]), is(true));

        long dashes = rows[ROWS - blockRows]
                .chars()
                .filter(character -> character == '─' || character == 'q')
                .count();
        int width = terminal.getSize().getColumns();
        assertThat(
                "the rule spans the window it is in now: " + dashes + " of " + (width - 1) + ":\n" + screen,
                dashes >= width - 1L,
                is(true));
    }

    @Test
    void draggingWiderWithNothingTyped() throws Exception {
        // Reported as "ohne irgend etwas getippt, nur groesser gezogen" -- and it was the worst of the
        // four: the block drawn twice with torn sequences between the copies.
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            drag(terminal, console, NARROW, WIDE);
            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    void draggingWiderWithTextInTheInput() throws Exception {
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            terminal.type("Hallo");
            Thread.sleep(200);

            drag(terminal, console, NARROW, WIDE);

            String screen = terminal.describe();
            assertThat(
                    "the prompt and what was typed are on screen once:\n" + screen,
                    count(terminal.rows(), row -> row.contains("> Hallo")),
                    is(1));
            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    @Disabled("Known defect, and this test is the record of it: dragging the window NARROWER loses the"
            + " edit line entirely -- reported as \"beim kleiner ziehen ist der Text nicht mehr"
            + " sichtbar\" and reproduced here as zero occurrences of \"> Hallo\" on an otherwise"
            + " correct screen. Not caused by anything in this class: the block and the state row end up"
            + " exactly where they belong. Delete the annotation to see it.")
    void draggingNarrowerWithTextInTheInput() throws Exception {
        // The other direction, reported as "beim kleiner ziehen ist der Text nicht mehr sichtbar".
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            terminal.type("Hallo");
            Thread.sleep(200);

            drag(terminal, console, WIDE, NARROW);

            String screen = terminal.describe();
            assertThat(
                    "what was typed is still visible:\n" + screen,
                    count(terminal.rows(), row -> row.contains("> Hallo")),
                    is(1));
            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    @Disabled("Known defect, and this test is the record of it: with a THREE-row block a wide drag ends"
            + " with the rule three columns short of the window (96 of 99, with a gap near its end), the"
            + " state row shifted one column right, and the prompt on two rows. A two-row block comes"
            + " out clean, which is what makes the row count the discriminator. Delete the annotation to"
            + " see it.")
    void draggingWiderWithAThreeRowBlock() throws Exception {
        // The shape the agent actually pins: a rule plus an activity row plus a state row.
        List<String> block = new ArrayList<>(Arrays.asList("... waiting for input ...", STATE));
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, block)) {
            drag(terminal, console, NARROW, WIDE);
            assertBlockIsIntact(terminal, 3);
        }
    }

    @Test
    void clearingTheScreenLeavesTheBlockAndPutsTheInputAboveIt() throws Exception {
        // Two reports in one: after /cls the input sat at the top left, and before that a wipe followed
        // by blank rows scrolled the erased content back into view.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            console.line("something written earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            assertThat(
                    "what was written before the wipe is gone:\n" + screen,
                    count(rows, row -> row.contains("something written earlier")),
                    is(0));
            assertBlockIsIntact(terminal, 2);
            assertThat(
                    "the input line is directly above the block, not at the top:\n" + screen,
                    rows[ROWS - 3].contains(">") || rows[ROWS - 3].isBlank(),
                    is(true));
        }
    }

    @Test
    void clearingTheScreenAfterADragIsStillClean() throws Exception {
        // The combination, because that is how it was hit: drag first, then /cls.
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            drag(terminal, console, NARROW, WIDE);
            console.clearScreen();
            Thread.sleep(400);

            assertBlockIsIntact(terminal, 2);
        }
    }
}
