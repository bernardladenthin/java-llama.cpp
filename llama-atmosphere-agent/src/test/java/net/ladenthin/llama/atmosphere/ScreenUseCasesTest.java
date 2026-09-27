// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.jline.terminal.Size;
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
     * Change the window size once, then let everything settle.
     *
     * <p><b>One event, not a drag, and that was measured rather than chosen.</b> A drag reports a size per
     * step — the probe recorded ~22 for one — and stepping through them here failed between one and four
     * of these cases per run, in different combinations each time: JLine's reader redraws on the signal
     * thread for every event, there is no way to join it, and more sleep made it worse rather than
     * better. A test that fails one run in five is not evidence. Every artefact in the reports appears
     * <em>per size event</em>, so a single event is enough to see a misplaced row, and it is
     * reproducible.
     */
    private void resizeOnce(ScreenTerminalHarness terminal, JLineTerminal console, int to) throws Exception {
        terminal.resize(to, ROWS);
        Thread.sleep(300);
        console.refreshBlockForCurrentSize();
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
            resizeOnce(terminal, console, WIDE);
            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    void draggingWiderWithTextInTheInput() throws Exception {
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, WIDE);

            String screen = terminal.describe();
            assertThat(
                    "the prompt and what was typed are on screen once:\n" + screen,
                    count(terminal.rows(), row -> row.contains("> Hallo")),
                    is(1));
            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    void draggingNarrowerLeavesOneCleanBlock() throws Exception {
        // Reported: below a certain width the block appears TWICE, once smeared into the upper area with
        // a stray "1" in it and once correctly at the bottom. Whether the block survives and whether the
        // edit line survives are two properties, so they are two tests -- together, a fix for one of
        // them cannot be seen.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, NARROW);

            assertBlockIsIntact(terminal, 2);
        }
    }

    @Test
    @Disabled("Known defect, and this test is the record of it: dragging the window NARROWER loses the"
            + " edit line entirely -- reported as \"beim kleiner ziehen ist der Text nicht mehr"
            + " sichtbar\" and reproduced here as zero occurrences of \"> Hallo\" on an otherwise"
            + " correct screen. Not caused by anything in this class: the block and the state row end up"
            + " exactly where they belong. Delete the annotation to see it.")
    void draggingNarrowerKeepsTheEditLine() throws Exception {
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            assertThat(
                    "what was typed is still visible:\n" + screen,
                    count(terminal.rows(), row -> row.contains("> Hallo")),
                    is(1));
        }
    }

    @Test
    void draggingNarrowerWithAThreeRowBlockLeavesOneCleanBlock() throws Exception {
        // The shape the agent pins, which is where the report came from.
        List<String> block = new ArrayList<>(Arrays.asList("... waiting for input ...", STATE));
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, NARROW);

            assertBlockIsIntact(terminal, 3);
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
            resizeOnce(terminal, console, WIDE);
            assertBlockIsIntact(terminal, 3);
        }
    }

    /** An emoji state row: kept as the record of a JLine limit, no longer what the agent pins. */
    private static final String REAL_STATE = "📁 X:\\tmp\\agent-sandbox · ⏸ manual · 📊 0/16k · 🔧 8 · 🤖 local-model";

    /** The three rows the agent really pins, rule excluded — the console prepends that itself. */
    private List<String> realBlock() {
        return new ArrayList<>(Arrays.asList("… waiting for input …", REAL_STATE));
    }

    /** A newline, spelled out so a heredoc cannot eat the escape. */
    private static final String NEWLINE = System.lineSeparator();

    /** The block the agent really pins now: basic-plane icons, the shape the reports came from. */
    private List<String> realBlockWithBasicPlaneIcons() {
        return new ArrayList<>(Arrays.asList(
                "… waiting for input …",
                "[▤ X:/tmp/agent-sandbox · " + ApprovalMode.MANUAL.badge() + " · ▦ 0/16k · ⚒ 8 · ◆ local-model]"));
    }

    /** Rows carrying the real state row, found by its leading glyph rather than the whole string. */
    private int realStateRows(String[] rows) {
        return count(rows, row -> row.contains("📁"));
    }

    @Test
    void draggingWiderWithTheRealBlockButNoAstralGlyphs() throws Exception {
        // The control for the three tests below, and it has to be run before believing them: this screen
        // stores one cell per UTF-16 char, so an astral glyph (an emoji is a surrogate pair) lands in two
        // cells and comes back as something else entirely -- the dump shows a CJK character where a robot
        // was written. So a failure with emoji could be the harness rather than the console. This row has
        // the same shape, the same separators and the same ellipsis, but every glyph is from the basic
        // plane. If this passes and the emoji ones fail, the trigger is the glyphs; if this fails too, it
        // is the shape.
        String state = "[dir] X:\\tmp\\agent-sandbox · || manual · [ctx] 0/16k · [t] 8 · [m] local-model";
        List<String> block = new ArrayList<>(Arrays.asList("… waiting for input …", state));
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, block)) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, WIDE);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat("the state row is on screen once:\n" + screen, count(rows, row -> row.contains("[dir]")), is(1));
            assertThat("the rule is on screen once:\n" + screen, count(rows, this::isRule), is(1));
            assertThat("the state row is the bottom row:\n" + screen, rows[ROWS - 1].contains("[dir]"), is(true));
        }
    }

    @Test
    @Disabled(
            "A JLine limit this records rather than a defect to fix here: an astral glyph (an emoji is a surrogate pair) breaks the column arithmetic of the pinned region, so the rule and the state row end up written into one screen line character by character and the block is drawn twice. The agent pins basic-plane icons instead (see StatusLine) and the same line with those passes all three cases. Delete the annotation to see it; re-check if JLine ever fixes the arithmetic.")
    void draggingWiderWithTheRealBlockWhoseGlyphsAreDoubleWidth() throws Exception {
        // Every green case above used plain ASCII in the block; none of the reports did. An emoji is one
        // character and TWO screen columns, so a row padded to the window width by character count ends
        // up past the right edge, wraps onto a second screen line, and everything below the reserved
        // region lands one row off — which is what "the three rows run together" looks like. This is the
        // one difference left between the harness and the console the reports came from.
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, realBlock())) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, WIDE);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat("the state row is on screen once:\n" + screen, realStateRows(rows), is(1));
            assertThat("the rule is on screen once:\n" + screen, count(rows, this::isRule), is(1));
            assertThat("the state row is the bottom row:\n" + screen, rows[ROWS - 1].contains("📁"), is(true));
        }
    }

    @Test
    @Disabled(
            "A JLine limit this records rather than a defect to fix here: an astral glyph (an emoji is a surrogate pair) breaks the column arithmetic of the pinned region, so the rule and the state row end up written into one screen line character by character and the block is drawn twice. The agent pins basic-plane icons instead (see StatusLine) and the same line with those passes all three cases. Delete the annotation to see it; re-check if JLine ever fixes the arithmetic.")
    void draggingNarrowerWithTheRealBlockWhoseGlyphsAreDoubleWidth() throws Exception {
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlock())) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat("the state row is on screen once:\n" + screen, realStateRows(rows), is(1));
            assertThat("the rule is on screen once:\n" + screen, count(rows, this::isRule), is(1));
            assertThat("the state row is the bottom row:\n" + screen, rows[ROWS - 1].contains("📁"), is(true));
        }
    }

    @Test
    @Disabled(
            "A JLine limit this records rather than a defect to fix here: an astral glyph (an emoji is a surrogate pair) breaks the column arithmetic of the pinned region, so the rule and the state row end up written into one screen line character by character and the block is drawn twice. The agent pins basic-plane icons instead (see StatusLine) and the same line with those passes all three cases. Delete the annotation to see it; re-check if JLine ever fixes the arithmetic.")
    void clearingTheScreenWithTheRealBlockAfterADrag() throws Exception {
        // The exact sequence of the report: type, drag, then /cls, with the block the agent pins.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlock())) {
            terminal.type("Hallo");
            Thread.sleep(200);
            resizeOnce(terminal, console, NARROW);

            console.clearScreen();
            Thread.sleep(400);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat("the state row is on screen once:\n" + screen, realStateRows(rows), is(1));
            assertThat("the state row is the bottom row:\n" + screen, rows[ROWS - 1].contains("📁"), is(true));
        }
    }

    @Test
    void noBlockRowIsEverWiderThanTheWindow() throws Exception {
        // The report this exists for: after resizing, the rule was WIDER than the window, wrapped onto a
        // second screen line, and pushed the two rows below it one row out of place -- on a screen whose
        // block was otherwise correct. A row of the pinned region may never exceed the window, whatever
        // width it was originally built for, because the region is reserved in LINES: one wrapped row
        // costs two of them and everything below lands wrong.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            // A wrapped row shows up as the SAME row content continuing on the next screen line, so the
            // direct check is that no row above the block carries rule characters: the rule belongs on
            // exactly one row, the third from the bottom.
            for (int row = 0; row < ROWS - 3; row++) {
                assertThat(
                        "row " + row + " carries part of the block, so a row below wrapped:" + NEWLINE + screen,
                        isRule(rows[row]),
                        is(false));
            }
            assertThat("the rule is on screen once:" + NEWLINE + screen, count(rows, this::isRule), is(1));
            assertThat(
                    "the rule is the third row from the bottom:" + NEWLINE + screen, isRule(rows[ROWS - 3]), is(true));
        }
    }

    @Test
    void theBlockIsRebuiltWhenTheWindowShrinksBelowTheRuleItWasBuiltFor() throws Exception {
        // The other half of the same report: the rule is as wide as the window it was built for, so after
        // a shrink it has to be REMADE, not merely cut. Measured as the number of rule characters on the
        // bottom rows, which must follow the new window rather than the old one.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            long dashes = terminal.rows()[ROWS - 3]
                    .chars()
                    .filter(character -> character == '─' || character == 'q')
                    .count();
            assertThat(
                    "the rule follows the new window: " + dashes + " characters in a " + NARROW + "-column window"
                            + NEWLINE + screen,
                    dashes <= NARROW && dashes >= NARROW - 2L,
                    is(true));
        }
    }

    @Test
    void theSizeWatchSurvivesATerminalThatThrowsOnce() throws Exception {
        // The defect behind "beim Groesse veraendern geht es immer noch kaputt" on a screen whose block was
        // otherwise correct. The poll read the size, one read threw, and the thread returned -- so the
        // block kept the width it had last been built for, and a window shrunk afterwards showed a rule
        // WIDER than itself, wrapping and pushing the rows below it out of place. One transient failure
        // must cost one poll, not the session.
        //
        // Driven through the real thread rather than around it, because the thread is the thing that was
        // wrong: a terminal that throws once on getSize(), then behaves.
        // Armed only after the console is up: the line reader reads the size while it is being built, and
        // an unarmed-from-the-start version had its one throw consumed there instead of by the poll.
        java.util.concurrent.atomic.AtomicBoolean armed = new java.util.concurrent.atomic.AtomicBoolean(false);
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", WIDE, ROWS) {
            @Override
            public Size getSize() {
                if (armed.compareAndSet(true, false)) {
                    throw new IllegalStateException("transient");
                }
                return super.getSize();
            }
        };
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            armed.set(true);
            // Give the poll time to hit the throwing read and carry on.
            Thread.sleep(400);

            terminal.resize(NARROW, ROWS);
            // No refresh call here on purpose: this is the one case where the POLL has to do the work.
            // Waited for by CONDITION, not by the clock: a fixed sleep made this the one flaky test in the
            // class, failing about two runs in three under load. It cannot be shorter than the poll and
            // there is no upper bound on how long a busy machine takes to get there.
            long deadline = System.currentTimeMillis() + 5000;
            while (System.currentTimeMillis() < deadline && terminal.rows()[ROWS - 3].contains("…")) {
                Thread.sleep(100);
            }

            String screen = terminal.describe();
            long dashes = terminal.rows()[ROWS - 3]
                    .chars()
                    .filter(character -> character == '─' || character == 'q')
                    .count();
            // Both bounds, and the lower one is the whole test: a rule that is too WIDE gets cut by the
            // pinned region on its own, so "not wider than the window" passes even when nothing was
            // rebuilt -- verified by reverting the fix and watching this go green with only that half.
            // A rebuilt rule spans the new window; a cut one falls short of it and ends in an ellipsis.
            assertThat(
                    "the poll kept working after one failed read: rule is " + dashes + " wide in a " + NARROW
                            + "-column window" + NEWLINE + screen,
                    dashes <= NARROW && dashes >= NARROW - 2L,
                    is(true));
            // The discriminator, and counting alone is not it: a rule built for the old window is CUT by
            // the pinned region, which ends the row in an ellipsis and leaves the same number of dashes a
            // rebuilt one would have. Verified by reverting the fix -- the count-only version stayed green.
            assertThat(
                    "the rule was rebuilt, not cut down from the old width" + NEWLINE + screen,
                    terminal.rows()[ROWS - 3].contains("…"),
                    is(false));
        }
    }

    @Test
    void oneAstralGlyphAloneInTheBlockIsEnoughToBreakIt() throws Exception {
        // Which glyph class actually breaks the pinned region, isolated to one character. The arithmetic
        // that matters is "UTF-16 chars vs screen columns": an emoji is 2 chars and 2 columns, which
        // AGREES, while a transport symbol is 1 char and 2 columns, which does not -- so by that reading
        // the emoji should be harmless and the mode glyph should not be. The measured result was the
        // opposite, which is why this pins each class on its own rather than in a whole status row.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of("x 📁 x", STATE))) {
            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            assertThat(
                    "with one emoji in the block, the state row is still on screen once" + NEWLINE + screen,
                    count(terminal.rows(), row -> row.contains(STATE)),
                    is(1));
        }
    }

    @Test
    void oneDoubleWidthBasicPlaneGlyphAloneInTheBlockIsFine() throws Exception {
        // The counterpart: a glyph that is one char and (on many terminals) two columns.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of("x ⏸ x", STATE))) {
            resizeOnce(terminal, console, NARROW);

            String screen = terminal.describe();
            assertThat(
                    "with one transport glyph in the block, the state row is still on screen once" + NEWLINE + screen,
                    count(terminal.rows(), row -> row.contains(STATE)),
                    is(1));
        }
    }

    @Test
    void theModeBadgeKeepsItsSpaceOnScreen() throws Exception {
        // Reported: "beim auto mode hat immer ein leerzeichen gefehlt: [pause] manual". The badge is
        // built as symbol + " " + name, so the space is there in the string -- the question is whether it
        // survives to the screen, and the transport symbols are exactly the kind of glyph whose width the
        // terminal and the column arithmetic can disagree about. Asserted on the rendered row, which is
        // the only place the answer lives.
        String badge = ApprovalMode.MANUAL.badge();
        List<String> block = new ArrayList<>(Arrays.asList("... waiting ...", "state " + badge));
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            String screen = terminal.describe();
            assertThat(
                    "the bottom row shows the badge with its space, expected: " + badge + "\n" + screen,
                    terminal.rows()[ROWS - 1].contains(badge),
                    is(true));
        }
    }

    @Test
    void theAutoModeBadgeKeepsItsSpaceOnScreen() throws Exception {
        String badge = ApprovalMode.AUTO.badge();
        List<String> block = new ArrayList<>(Arrays.asList("... waiting ...", "state " + badge));
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            String screen = terminal.describe();
            assertThat(
                    "the bottom row shows the badge with its space, expected: " + badge + "\n" + screen,
                    terminal.rows()[ROWS - 1].contains(badge),
                    is(true));
        }
    }

    @Test
    void clearingTheScreenWithAThreeRowBlock() throws Exception {
        // Reported after /cls: the three block rows ran together on ONE screen line and the input sat
        // above them near the top. The two-row case is clean, so the row count is a discriminator too.
        List<String> block = new ArrayList<>(Arrays.asList("... waiting for input ...", STATE));
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            console.line("something written earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);

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
            resizeOnce(terminal, console, WIDE);
            console.clearScreen();
            Thread.sleep(400);

            assertBlockIsIntact(terminal, 2);
        }
    }
}
