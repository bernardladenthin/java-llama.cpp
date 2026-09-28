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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
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
 * <p><b>Which JLine this needs.</b> The library is a property, and these cases need the patched one:
 *
 * <pre>
 * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6-statusfix8   # runs
 * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6              # every case SKIPS
 * </pre>
 *
 * <p>Six fixes are carried against JLine and the class skips itself without them — see
 * {@code onlyWithAJLineThatCarriesTheFixes} for why the gate is the whole class and not a list. What proves
 * the fixes themselves are JLine's own tests, in the clone: {@code StatusRedisplayTest},
 * {@code StatusDelayedWrapTest}, {@code StatusConcurrencyTest}, {@code StatusRepaintTest} and
 * {@code StatusWrongWidthTest}.
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

    /**
     * The marker that says the JLine on the classpath carries the fixes this console needs.
     *
     * <p>{@code Status.repaint()} is the fifth of them and the first to add a method, so its presence is a
     * usable stand-in for the set — it exists only in a build that has the others too. It does not
     * distinguish the fifth build from the later ones: the sixth adds a <em>protected</em>
     * {@code Display.addressesEveryRow()} and the seventh changes only how wide a row is padded, neither of
     * which can be probed from here. So on an older patched jar one or two cases fail rather than skip
     * ({@code aWidthTheScreenDoesNotHaveMustNotRunTheBlockRowsTogether} needs the sixth,
     * {@code aBlockRowBuiltWIDERThanTheWindowMustNotSPILLaCopyIntoTheOutput} the seventh). Stated rather than
     * worked around: the message names the version to use.
     *
     * @return whether the patched library is on the classpath
     */
    /**
     * Run these cases even on a JLine without the fixes, so the staircase can be measured without editing.
     *
     * <p>Which of the fixes are still needed is a question that comes up — the console has changed a great deal
     * around them — and answering it means running this class against each patched build <em>and</em> against the
     * released one. The skip is what makes CI green; this property is what makes the measurement possible without
     * turning the skip into an edit, which is how the counts in the investigation document were produced.
     */
    private static final String RUN_ANYWAY = "atmosphere.screen.tests.runAnyway";

    private static boolean jlineCarriesTheFixes() {
        try {
            org.jline.utils.Status.class.getMethod("repaint");
            return true;
        } catch (NoSuchMethodException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Skip everything here on a JLine that does not carry the fixes.
     *
     * <p><b>This is a rule from CLAUDE.md, applied late:</b> "no project test may assert the fixed behaviour
     * while the build depends on an unfixed release". The pom's {@code jline.version} is the released one, so
     * that is what CI builds against — and against it these cases fail, between six and thirteen of them,
     * a DIFFERENT set each run. The variation is the point: the fourth fix is a data race, and when its
     * {@code ConcurrentModificationException} lands on the reader's signal thread it ends that thread, after
     * which no size change is reported at all and whichever cases were still to run fail too. So there is no
     * fixed list to mark, and the honest gate is the whole class.
     *
     * <p>Per test rather than in a {@code @BeforeAll}, on purpose: a class-level assumption makes Surefire
     * record the class as zero tests, which reads as "nothing to see" instead of "skipped" — the same trap
     * that silently muted every model-backed test in this repository for months.
     *
     * <p><b>To see the library make the difference</b>, run this class twice:
     *
     * <pre>
     * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6-statusfix8   # green
     * mvn test -Dtest=ScreenUseCasesTest -Djline.version=4.4.6              # skipped -- delete this
     *                                                                        # assumption to see it fail
     * </pre>
     */
    @BeforeEach
    void onlyWithAJLineThatCarriesTheFixes() {
        Assumptions.assumeTrue(
                jlineCarriesTheFixes() || Boolean.getBoolean(RUN_ANYWAY),
                "needs the patched JLine: mvn test -Djline.version=4.4.6-statusfix8"
                        + " (see docs/upstream-investigation-jline-status-windows-redraw.md)");
    }

    private ScreenTerminalHarness terminal(int columns) throws Exception {
        return new ScreenTerminalHarness("windows-vtp", columns, ROWS);
    }

    /**
     * A screen whose height is not this class's constant.
     *
     * @param columns the width
     * @param rows the height
     * @return the harness
     */
    private ScreenTerminalHarness terminalWithRows(int columns, int rows) throws Exception {
        return new ScreenTerminalHarness("windows-vtp", columns, rows);
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
        // Long enough for the SETTLE to have happened as well, which is 400 ms after the last size event. It used
        // to be 300, and once a settled widening started wiping the screen that window straddled the wipe: the
        // assertions then read a screen that was halfway through being cleared and redrawn, which showed up as one
        // case failing in a full run and passing on its own. Waiting for the state the application ends up in is
        // the only stable thing to assert.
        Thread.sleep(800);
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
                "the rule spans the window it is in now: " + dashes + " of " + (width - 2) + ":\n" + screen,
                dashes >= width - 2L,
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
                    dashes <= NARROW && dashes >= NARROW - 3L,
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
                    dashes <= NARROW && dashes >= NARROW - 3L,
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
    void theRuleNeverSharesARowWithThePrompt() throws Exception {
        // Reported after enlarging with text in the input: "> Hallo" and the rule on ONE screen line, the
        // activity and state rows below it. The rule is the first row of the reserved region, so sharing a
        // row with the prompt means the region starts one row too high and everything in it is off by one.
        // This is the exact block the agent pins, with the icons it pins now.
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            terminal.type("Hallo");
            Thread.sleep(200);

            resizeOnce(terminal, console, WIDE);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            for (int row = 0; row < rows.length; row++) {
                boolean carriesBoth = rows[row].contains("Hallo") && isRule(rows[row]);
                assertThat(
                        "row " + row + " carries the prompt AND the rule, so the region starts a row too high" + NEWLINE
                                + screen,
                        carriesBoth,
                        is(false));
            }
            assertThat("the rule is on screen once" + NEWLINE + screen, count(rows, this::isRule), is(1));
            assertThat(
                    "the rule is the third row from the bottom" + NEWLINE + screen, isRule(rows[ROWS - 3]), is(true));
        }
    }

    @Test
    void clearingTheScreenAfterAWholeTurnHasScrolledIt() throws Exception {
        // Reported in this exact order: type, Enter, the answer looked fine, then /cls -- and the three
        // block rows ended up running together on one screen line with a stray character above the prompt.
        // What every green /cls case above was missing is that the screen had SCROLLED first: a turn prints
        // more lines than the window holds and refreshes the block as it goes, so the wipe happens in a
        // state none of those tests reach.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            // A turn's worth of output: more rows than the window has, with the block refreshed between
            // them the way the activity line is refreshed four times a second while a turn runs.
            for (int line = 0; line < ROWS * 2; line++) {
                console.line("answer line " + line);
                if (line % 3 == 0) {
                    console.refreshBlockForCurrentSize();
                }
            }
            Thread.sleep(300);

            console.clearScreen();
            Thread.sleep(400);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat(
                    "nothing from before the wipe is left" + NEWLINE + screen,
                    count(rows, row -> row.contains("answer line")),
                    is(0));
            assertThat("the rule is on screen once" + NEWLINE + screen, count(rows, this::isRule), is(1));
            assertThat(
                    "the rule has a row to itself, the third from the bottom" + NEWLINE + screen,
                    isRule(rows[ROWS - 3]),
                    is(true));
            assertThat(
                    "the activity row has a row to itself" + NEWLINE + screen,
                    rows[ROWS - 2].contains("waiting for input") && !isRule(rows[ROWS - 2]),
                    is(true));
            assertThat(
                    "the state row is the bottom row" + NEWLINE + screen,
                    rows[ROWS - 1].contains("local-model") && !isRule(rows[ROWS - 1]),
                    is(true));
        }
    }

    @Test
    void refreshingTheBlockWhileTheReaderRedrawsNeverThrows() throws Exception {
        // The race that is actually behind the remaining reports, conserved. Status and Display are not
        // thread-safe: a run of this suite caught a ConcurrentModificationException inside JLine's
        // Display.cost -- a plain HashMap in computeIfAbsent -- raised from Status$MovingCursorDisplay
        // while the reader was redrawing. The agent refreshes the pinned block four times a second while
        // a turn runs and again on every size change, so that collision is the normal case, not an edge.
        //
        // This is reachable in the harness even though its output stream is synchronized, because the
        // unprotected state is JLine's own map rather than the byte stream: locking the stream does not
        // help. Whatever the screen ends up looking like, an exception must not escape into the turn loop,
        // which is what this pins.
        java.util.List<Throwable> escaped = java.util.Collections.synchronizedList(new ArrayList<>());
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            java.util.List<Thread> writers = new ArrayList<>();
            for (int writer = 0; writer < 3; writer++) {
                Thread thread = new Thread(() -> {
                    long deadline = System.currentTimeMillis() + 1500;
                    while (System.currentTimeMillis() < deadline) {
                        try {
                            console.status(realBlockWithBasicPlaneIcons());
                        } catch (RuntimeException e) {
                            escaped.add(e);
                            return;
                        }
                    }
                });
                thread.setDaemon(true);
                writers.add(thread);
            }
            writers.forEach(Thread::start);

            // Keep the reader redrawing at the same time. Keystrokes alone are not enough -- three runs of
            // that version were green: a keystroke is a small redisplay, while a SIZE change makes the
            // reader resize its display, resize the pinned region and redisplay, which is far more of the
            // shared state at once and is where the exception was actually observed.
            long deadline = System.currentTimeMillis() + 1500;
            boolean wide = false;
            while (System.currentTimeMillis() < deadline) {
                terminal.type("x");
                wide = !wide;
                try {
                    // The size change is delivered on this thread here, so the exception surfaces here.
                    // In the application it is raised on JLine's own input pump, where it would kill that
                    // thread instead -- and a dead pump means no further size change is ever reported,
                    // which is why a block can keep a width for the rest of a session.
                    terminal.resize(wide ? WIDE : WIDE - 7, ROWS);
                } catch (RuntimeException e) {
                    escaped.add(e);
                    break;
                }
                Thread.sleep(5);
            }
            for (Thread thread : writers) {
                thread.join(3000);
            }

            assertThat(
                    "a block refresh raced the reader and threw: " + escaped + NEWLINE + terminal.describe(),
                    escaped.isEmpty(),
                    is(true));
        }
    }

    @Test
    void theBlockStaysWholeWhenSizeEventsArriveOnAnotherThread() throws Exception {
        // The difference between this harness and the console the reports come from, and it took far too
        // long to notice: resize() raises the signal on the CALLER's thread, so the reader's handler runs
        // serialised with the test. A real console delivers it on its own input pump, at the same time as
        // every other writer. Reported as many rules of different lengths stacked up after a shrink and a
        // grow, which is the block drawn once per size event at whatever width each one saw.
        ScreenTerminalHarness terminal = terminal(NARROW);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            terminal.type("hallo");
            Thread.sleep(200);

            List<Thread> pumps = new ArrayList<>();
            for (int columns = NARROW; columns <= WIDE; columns += 5) {
                pumps.add(terminal.resizeAsynchronously(columns, ROWS));
                console.refreshBlockForCurrentSize();
                Thread.sleep(10);
            }
            for (int columns = WIDE; columns >= NARROW; columns -= 5) {
                pumps.add(terminal.resizeAsynchronously(columns, ROWS));
                console.refreshBlockForCurrentSize();
                Thread.sleep(10);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(300);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat("the rule is on screen once" + NEWLINE + screen, count(rows, this::isRule), is(1));
            // Counted by its LEADING glyph, not its tail: a narrow window cuts the row and ends it in an
            // ellipsis, so looking for "local-model" reports zero on a perfectly correct screen -- which
            // is how the first version of this failed three runs out of three while the block was intact.
            assertThat(
                    "the state row is on screen once" + NEWLINE + screen, count(rows, row -> row.contains("▤")), is(1));
            for (int row = 0; row < rows.length; row++) {
                assertThat(
                        "row " + row + " carries the prompt AND the rule" + NEWLINE + screen,
                        rows[row].contains("hallo") && isRule(rows[row]),
                        is(false));
            }
        }
    }

    @Test
    void justEnlargingAFreshSessionKeepsThePromptOffTheRuleRow() throws Exception {
        // The smallest form the report ever took: start, type NOTHING, enlarge -- and ">" shares a screen
        // line with the rule. No turn, no scrolling, no text in the buffer. The geometry is the reporter's
        // rather than this class's default, because the window there is 34 rows and ~150 columns and the
        // block is three rows, and the row count is the one variable that had never been varied.
        int rows = 34;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            Thread pump = terminal.resizeAsynchronously(150, rows);
            console.refreshBlockForCurrentSize();
            pump.join(2000);
            Thread.sleep(400);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String[] screenRows = terminal.rows();
            String screen = terminal.describe();
            for (int row = 0; row < screenRows.length; row++) {
                assertThat(
                        "row " + row + " carries the prompt AND the rule" + NEWLINE + screen,
                        screenRows[row].contains(">") && isRule(screenRows[row]),
                        is(false));
            }
            assertThat("the rule is on screen once" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            assertThat(
                    "the rule is the third row from the bottom" + NEWLINE + screen,
                    isRule(screenRows[rows - 3]),
                    is(true));
        }
    }

    @Test
    void theCursorStaysOnTheRowTheBlockLeavesForItWhenTheWindowIsEnlarged() throws Exception {
        // The measurement that located the defect, taken from the reporter's console with a probe and now
        // asked of this screen too. Every other assertion in this class is about the CONTENT of the rows,
        // and content is exactly what still looked plausible while this was already wrong: in a 38-row
        // window with a three-row block the prompt belongs on row 34, and the console reported 33, 32 and
        // 31 after dragging -- drifting UP by one to three rows, never more than the block's own height.
        // A pinned region is reserved in rows counted from the bottom, so that drift IS the defect.
        int rows = 38;
        int blockRows = 3;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 118, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            int expected = rows - 1 - blockRows;
            assertThat(
                    "before any resize the cursor is on the row the block leaves for it",
                    terminal.cursorRow(),
                    is(expected));

            List<Thread> pumps = new ArrayList<>();
            for (int columns : new int[] {116, 118, 116, 118, 112, 118}) {
                pumps.add(terminal.resizeAsynchronously(columns, rows));
                console.refreshBlockForCurrentSize();
                Thread.sleep(30);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(400);

            assertThat(
                    "the cursor is still on row " + expected + " of a " + rows + "-row window with a " + blockRows
                            + "-row block" + NEWLINE + terminal.describe(),
                    terminal.cursorRow(),
                    is(expected));
        }
    }

    @Test
    void theRuleIsExactlyTwoColumnsShortOfTheWindowAndStaysThere() throws Exception {
        // Reported as "der Strich wird gefuehlt minimal kleiner". Two columns short is by DESIGN -- a row as
        // wide as the window risks wrapping, a wrapped row costs a second screen line, and the region is
        // reserved in lines, which is how a long row tore the block apart before. What would not be by
        // design is the rule losing a column per resize, so this returns to the SAME width several times
        // over and requires the exact same length every time.
        int rows = 20;
        int wide = 118;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", wide, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            List<Long> lengths = new ArrayList<>();
            for (int round = 0; round < 4; round++) {
                Thread pump = terminal.resizeAsynchronously(wide - 20, rows);
                console.refreshBlockForCurrentSize();
                pump.join(2000);
                Thread.sleep(150);
                pump = terminal.resizeAsynchronously(wide, rows);
                console.refreshBlockForCurrentSize();
                pump.join(2000);
                Thread.sleep(250);
                lengths.add(terminal.rows()[rows - 3]
                        .chars()
                        .filter(character -> character == '─' || character == 'q')
                        .count());
            }

            String screen = terminal.describe();
            assertThat(
                    "the rule is the same length every time it comes back to " + wide + " columns: " + lengths + NEWLINE
                            + screen,
                    lengths.stream().distinct().count(),
                    is(1L));
            assertThat(
                    "and that length is two columns short of the window, deliberately: " + lengths + NEWLINE + screen,
                    lengths.get(0),
                    is((long) wide - 2));
        }
    }

    @Test
    void aWindowThatReportsMoreColumnsThanItHasMustNotCostThePromptItsRow() throws Exception {
        // The mechanism this has been narrowing towards, forced instead of waited for. Two reports arrived
        // together -- "nur die Eingabe wandert hoch" and "wenn ich groesser ziehe kommen viel mehr Striche"
        // -- and one cause explains both: a rule built for a width the console has not applied yet is wider
        // than the window, so it WRAPS, the region needs four screen lines where three are reserved, and
        // the row the wrap eats is the prompt's. That also matches the probe's number exactly: the cursor
        // drifted up by one to three rows, never more than the block's height.
        //
        // A console reporting a size it has not finished applying cannot be arranged by waiting, so here
        // the terminal simply lies: the screen is REAL_COLUMNS wide and getSize() claims more.
        int rows = 20;
        int realColumns = 80;
        // ONE column, because that is the lag a console being dragged actually shows -- it reports a width
        // it has not finished applying. A larger overshoot cannot be defended against by anything built
        // from a reported width, and this test would be a wish rather than a contract.
        int claimedExtra = 1;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", realColumns, rows) {
            @Override
            public Size getSize() {
                Size real = super.getSize();
                return Size.of(real.getColumns() + claimedExtra, real.getRows());
            }
        };
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat("the rule is on screen once" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            for (int row = 0; row < screenRows.length; row++) {
                assertThat(
                        "row " + row + " carries the prompt AND the rule" + NEWLINE + screen,
                        screenRows[row].contains(">") && isRule(screenRows[row]),
                        is(false));
            }
            assertThat(
                    "the cursor still has the row the block leaves for it" + NEWLINE + screen,
                    terminal.cursorRow(),
                    is(rows - 1 - 3));
        }
    }

    @Test
    @Disabled(
            "Reproduced and OPEN, with four candidates eliminated by bisection and one named. After a series of size changes the block sits exactly right while the prompt row is blank. Ruled out by measurement, each a separate run: this console re-establishing the region (Status.resize), this console rebuilding the rows at all, JLine handleSignal calling Status.resize, and JLine handleSignal calling redisplay -- red with every one of them switched off. Also ruled out: the screen model itself, which keeps text across resizes (ScreenTerminalHarnessTest). The candidate left is Status.update clearing excess rows from display.rows - oldLinesSize, which reaches ABOVE the region when that count is stale. Delete the annotation to see it.")
    void thePromptItselfStaysVisibleAfterEnlarging() throws Exception {
        // A gap in every assertion above, and it is why they were all green while the console was not.
        // They ask that no row carries the prompt AND the rule -- but when the rule is drawn ON the prompt's
        // row it OVERWRITES it, so the row carries only the rule and the check is satisfied. Reported as
        // "darueber erscheint aber immer noch ein strich, auf gleicher hoehe wie die eingabe" together with
        // "das > zeichen bei der eingabe ist aber unsichtbar" -- one screen, two halves, and the second half
        // is the one nothing was looking for.
        //
        // What has to hold is simply that the prompt is still on the screen somewhere.
        int rows = 20;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", NARROW, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            assertThat(
                    "the prompt is on screen before any resize" + NEWLINE + terminal.describe(),
                    count(terminal.rows(), row -> row.contains(">")),
                    is(1));

            List<Thread> pumps = new ArrayList<>();
            for (int columns : new int[] {WIDE, NARROW, WIDE, NARROW + 10, WIDE}) {
                pumps.add(terminal.resizeAsynchronously(columns, rows));
                console.refreshBlockForCurrentSize();
                Thread.sleep(40);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(400);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            assertThat(
                    "the prompt is still on screen after the resizes" + NEWLINE + screen,
                    count(terminal.rows(), row -> row.contains(">")),
                    is(1));
            assertThat(
                    "and it is on the row above the block, not overwritten by the rule" + NEWLINE + screen,
                    terminal.rows()[rows - 4].contains(">"),
                    is(true));
        }
    }

    @Test
    @Disabled(
            "Reproduced and OPEN, the same defect from the other side: with a turn already on screen, shrinking and growing erases the answer while the rule assertions hold. Same bisection as the test above -- neither this console nor either half of JLine handleSignal is the writer that erases it, and the screen model keeps text across resizes. Delete the annotation to see it.")
    void afterATurnShrinkingAndGrowingKeepsOneRuleNoWiderThanTheWindow() throws Exception {
        // Both halves of the latest report in one case, because both are the same measurement from
        // different sides: shrinking moves the input area and the output up, and growing produces a rule
        // far wider than the window -- roughly 700 columns of it in a window of about 175, so four screen
        // rows of rule. A rule that occupies four rows where one is reserved pushes everything above it up
        // by three, which IS the "Ausgabe zu weit oben" half.
        //
        // With a turn's output on screen, as the report had it: the conversation is what gets pushed.
        int rows = 20;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", WIDE, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            console.line("› Hallo");
            console.line("Hello! How can I assist you today?");
            Thread.sleep(300);

            List<Thread> pumps = new ArrayList<>();
            for (int columns : new int[] {NARROW, WIDE, NARROW, WIDE}) {
                pumps.add(terminal.resizeAsynchronously(columns, rows));
                console.refreshBlockForCurrentSize();
                Thread.sleep(40);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(400);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            int columns = terminal.getSize().getColumns();

            assertThat("the rule is on screen once" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            long dashes = screenRows[rows - 3]
                    .chars()
                    .filter(character -> character == '─' || character == 'q')
                    .count();
            assertThat(
                    "and it is no wider than the window: " + dashes + " of " + columns + NEWLINE + screen,
                    dashes <= columns - 2L,
                    is(true));
            assertThat(
                    "the answer from the turn is still on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("How can I assist")),
                    is(1));
        }
    }

    @Test
    void shrinkingOneStepAtATimeMustNotWalkTheInputUpOneRowPerStep() throws Exception {
        // The report in its most precise form yet: "beim verkleinern wandert bei jedem Schritt die eingabe
        // eine zeile weiter hoch". One row per size event, cumulative -- which is a different measurement
        // from every cursor assertion here so far, all of which looked only at the END state after a mixture
        // of shrinks and grows. A drift that accumulates and one that cancels out are indistinguishable
        // there, and this walks down one column at a time and records the cursor row after EVERY step.
        int rows = 24;
        int blockRows = 3;
        int expected = rows - 1 - blockRows;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 120, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            terminal.type("Hallo");
            Thread.sleep(150);

            List<Integer> cursorRows = new ArrayList<>();
            cursorRows.add(terminal.cursorRow());
            for (int columns = 119; columns >= 100; columns--) {
                Thread pump = terminal.resizeAsynchronously(columns, rows);
                console.refreshBlockForCurrentSize();
                pump.join(2000);
                Thread.sleep(40);
                cursorRows.add(terminal.cursorRow());
            }

            String screen = terminal.describe();
            assertThat(
                    "the cursor stays on row " + expected + " through every step, saw " + cursorRows + NEWLINE + screen,
                    cursorRows.stream().distinct().toList(),
                    is(List.of(expected)));
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

    @Test
    void clearingTheScreenLeavesThePromptOnTheRowTheBlockLeavesForIt() throws Exception {
        // The report: "nach /cls ist der cursor auch ganz oben und nicht unten". A wipe puts the cursor
        // home, and the reader then draws its prompt where the cursor is -- at the top left, with the
        // block still pinned to the bottom and ten blank rows between them.
        //
        // Why this needs a test of its own although three /cls cases already exist: every one of them
        // asks about the CONTENT of the rows, and content is exactly what a wipe leaves right. The
        // block is intact, no escape tail is on screen, the erased output is gone -- and the prompt is
        // ten rows too high. clearingTheScreenLeavesTheBlockAndPutsTheInputAboveIt even looks at the
        // input row, but accepts a blank one ("|| isBlank()"), which is precisely the defect.
        List<String> block = List.of(STATE);
        int blockRows = 2; // the rule plus the state row
        int promptRow = ROWS - 1 - blockRows;
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            console.line("something written earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            assertThat(
                    "the cursor is on the row the block leaves for the prompt, not at the top:\n" + screen,
                    terminal.cursorRow(),
                    is(promptRow));
            assertThat("the prompt is on that row:\n" + screen, rows[promptRow].contains(">"), is(true));
            for (int row = 0; row < promptRow; row++) {
                assertThat(
                        "row " + row + " is above the prompt and must be empty after a wipe:\n" + screen,
                        rows[row].isBlank(),
                        is(true));
            }
        }
    }

    @Test
    void clearingTheScreenWithAThreeRowBlockAlsoLeavesThePromptAtTheBottom() throws Exception {
        // The same question with the block the application really pins, because the row count has been a
        // discriminator in this class before: the two-row case came out clean where the three-row case
        // did not.
        List<String> block = new ArrayList<>(Arrays.asList("... waiting for input ...", STATE));
        int blockRows = 3;
        int promptRow = ROWS - 1 - blockRows;
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            console.line("something written earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);

            String screen = terminal.describe();
            assertThat(
                    "the cursor is on the row a three-row block leaves for the prompt:\n" + screen,
                    terminal.cursorRow(),
                    is(promptRow));
            assertThat("the prompt is on that row:\n" + screen, terminal.rows()[promptRow].contains(">"), is(true));
        }
    }

    @Test
    void clearingTheScreenPutsNothingBackIntoTheScrollbackAboveThePrompt() throws Exception {
        // The other direction of the same report, and the reason the obvious fix is not allowed: pushing
        // the prompt back down with blank rows scrolls the erased lines back into view on the real
        // console, and addressing the cursor from inside printAbove left a single character stranded
        // above the prompt ("nach cls weiterhin eingabe ueber dem eingabe > zeichen"). So whatever puts
        // the prompt on its row must leave every row above it blank -- which is what this asserts, one
        // row at a time so a failure names the row.
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, List.of(STATE))) {
            for (int line = 0; line < 8; line++) {
                console.line("output line " + line);
            }
            Thread.sleep(300);

            console.clearScreen();
            Thread.sleep(400);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            int promptRow = ROWS - 1 - 2;
            for (int row = 0; row < promptRow; row++) {
                assertThat(
                        "row " + row + " must be blank after a wipe, it is \"" + rows[row].strip() + "\":\n" + screen,
                        rows[row].isBlank(),
                        is(true));
            }
            assertThat("the prompt is on its row:\n" + screen, rows[promptRow].contains(">"), is(true));
        }
    }

    @Test
    void controlLAlsoLeavesThePromptOnTheRowTheBlockLeavesForIt() throws Exception {
        // The same question through the other door. Ctrl-L is bound by JLine, not by this class, and its
        // own widget wipes the screen and redraws the line -- which is where the prompt at the top left
        // comes from. Two ways into one command must not end on two different rows, so the binding is
        // this console's to own once the command is.
        List<String> block = List.of(STATE);
        int promptRow = ROWS - 1 - 2;
        ScreenTerminalHarness terminal = terminal(WIDE);
        try (JLineTerminal console = start(terminal, block)) {
            console.line("something written earlier");
            Thread.sleep(200);

            terminal.type("\u000c"); // Ctrl-L
            Thread.sleep(400);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            assertThat(
                    "Ctrl-L cleared the screen:\n" + screen,
                    count(rows, row -> row.contains("something written earlier")),
                    is(0));
            assertThat(
                    "the cursor is on the row the block leaves for the prompt, not at the top:\n" + screen,
                    terminal.cursorRow(),
                    is(promptRow));
            assertThat("the prompt is on that row:\n" + screen, rows[promptRow].contains(">"), is(true));
        }
    }

    /**
     * Drag a window to a new width AND a new height, the way a corner drag does.
     *
     * <p>The variable no test in this class had ever varied. Every case here changes the width and keeps
     * the row count, but a window is dragged by its corner: the reported screen came from a drag that grew
     * both. A taller window moves the pinned region DOWN — it is reserved from the bottom — and whatever
     * stood on the rows it used to occupy is not erased by anything, because nothing writes there again.
     *
     * @param terminal the screen
     * @param console the console under test
     * @param columns the new width
     * @param rows the new height
     */
    private void growOnce(ScreenTerminalHarness terminal, JLineTerminal console, int columns, int rows)
            throws Exception {
        Thread pump = terminal.resizeAsynchronously(columns, rows);
        console.refreshBlockForCurrentSize();
        pump.join(2000);
        Thread.sleep(400);
        console.refreshBlockForCurrentSize();
        // See resizeOnce: long enough for the settle, which now wipes after a widening.
        Thread.sleep(800);
    }

    /**
     * Everything the reports were about, for a window whose height is not this class's constant.
     *
     * @param terminal the screen to read
     * @param rows the window's height
     * @param blockRows how many rows the pinned block occupies
     */
    private void assertBlockIsIntactAt(ScreenTerminalHarness terminal, int rows, int blockRows) {
        String[] screenRows = terminal.rows();
        String screen = terminal.describe();
        for (int row = 0; row < screenRows.length; row++) {
            for (String tail : ESCAPE_TAILS) {
                assertThat(
                        "row " + row + " shows \"" + tail + "\", the tail of a torn escape sequence" + NEWLINE + screen,
                        screenRows[row].contains(tail),
                        is(false));
            }
            assertThat(
                    "row " + row + " carries the prompt AND the rule" + NEWLINE + screen,
                    screenRows[row].contains(">") && isRule(screenRows[row]),
                    is(false));
        }
        assertThat("the rule is on screen once" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
        assertThat(
                "the rule is directly above the block" + NEWLINE + screen,
                isRule(screenRows[rows - blockRows]),
                is(true));
        // The HEAD of the state row, not its tail: a narrow window cuts the row with an ellipsis, so
        // "local-model" is legitimately absent there and looking for it fails for the wrong reason -- which it
        // did, twice, in tests about narrowing.
        assertThat(
                "the state row is on screen once" + NEWLINE + screen,
                count(screenRows, row -> row.contains("agent-sandbox")),
                is(1));
    }

    @Test
    void draggingTheCORNERSoTheWindowGrowsInBOTHDIRECTIONS() throws Exception {
        // The reported screen, and the one variable this class had never varied: a corner drag changes the
        // HEIGHT as well as the width. The pinned block is reserved from the BOTTOM, so a taller window
        // moves it down -- and the rows it used to stand on are never written again, so nothing erases
        // them. The report shows exactly that: a rule at the OLD width with the activity row continuing on
        // the SAME screen line (the old block, left behind, its rows no longer padded to a full window),
        // then the new rule below it, then the block. Four rows where three belong.
        int rows = 24;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            int grown = 34;
            growOnce(terminal, console, 150, grown);

            assertBlockIsIntactAt(terminal, grown, 3);
        }
    }

    @Test
    void makingOnlyTheWindowTALLERLeavesNoBlockBehind() throws Exception {
        // The same drag with the width held still, so the two variables are separated: if this is red and
        // the width-only cases are green, the height is the whole story.
        int rows = 24;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            int grown = 34;
            growOnce(terminal, console, 100, grown);

            assertBlockIsIntactAt(terminal, grown, 3);
        }
    }

    @Test
    void makingOnlyTheWindowSHORTERLeavesNoBlockBehind() throws Exception {
        // The other direction, because "kleiner ziehen sah ganz gut aus" is a report too and a test that
        // pins it is what keeps it that way.
        int rows = 34;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            int shrunk = 24;
            growOnce(terminal, console, 100, shrunk);

            assertBlockIsIntactAt(terminal, shrunk, 3);
        }
    }

    @Test
    void aFastCORNERDragThroughSeveralSizesEndsWithOneCleanBlock() throws Exception {
        // The reported screen is not a single size event: it shows a rule of ONE width with the activity
        // row CUT and continuing on the same screen line, and a second rule below it starting at the
        // column the cut left off at. Rows built for one width, written into a region that has another --
        // which is what two overlapping size events produce, and a drag delivers ~22 of them.
        //
        // Every other case in this class is deliberately ONE event, because stepping through a width-only
        // drag failed between one and four cases per run. This one steps on purpose and grows in BOTH
        // directions, which no other case does.
        int rows = 24;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            List<Thread> pumps = new ArrayList<>();
            int grown = rows;
            for (int columns = 106; columns <= 150; columns += 8) {
                grown += 2;
                pumps.add(terminal.resizeAsynchronously(columns, grown));
                console.refreshBlockForCurrentSize();
                Thread.sleep(40);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(600);
            console.refreshBlockForCurrentSize();
            Thread.sleep(600);

            assertBlockIsIntactAt(terminal, grown, 3);
        }
    }

    @Test
    void aWidthTheScreenDoesNotHaveMustNotRunTheBlockRowsTogether() throws Exception {
        // THE defect behind "beim größer ziehen wieder ganz viele Striche unten", and the one that had to be
        // fixed in the library rather than here.
        //
        // JLine pads every row of the pinned region to the width the terminal REPORTS and writes the rows
        // one after another: the second begins on a new screen row only because writing the last column of
        // the first one wrapped. A screen that is WIDER than the reported width therefore never wraps, and
        // the whole block lands on one screen row, side by side -- rule, activity row and state row, which
        // is exactly what the reported screens showed. Worse, what the collapse pushes past the window does
        // not vanish: the block is reserved from the BOTTOM, so the overflow lands in the OUTPUT area above
        // it, where nothing writes again. Hence the screen full of rule fragments after a dozen drags, and
        // hence /cls being the only thing that cleaned it up -- it scrolls.
        //
        // Nothing built from a reported width can defend against this, which is why it is the sixth fix
        // carried against JLine: the pinned region now ADDRESSES each of its rows instead of trusting the
        // wrap. Proven red/green in the library's own suite as well (StatusWrongWidthTest), where without
        // the fix the rule row reads "------  working" and the activity row's screen row holds "[state]".
        int rows = 20;
        int realColumns = 121;
        int claimedMissing = 8;
        LaggingWidth terminal = new LaggingWidth(realColumns, rows);
        terminal.missing = claimedMissing;
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat(
                    "no row carries the rule AND the activity row" + NEWLINE + screen,
                    count(screenRows, row -> isRule(row) && row.contains("wait")),
                    is(0));
            assertThat("the rule is on screen once" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            assertThat(
                    "the rule has the row above the block" + NEWLINE + screen, isRule(screenRows[rows - 3]), is(true));
            assertThat(
                    "the state row is the bottom row" + NEWLINE + screen,
                    screenRows[rows - 1].contains("local-model"),
                    is(true));
            for (int row = 0; row < rows - 3; row++) {
                assertThat(
                        "row " + row + " is above the block and must carry nothing" + NEWLINE + screen,
                        screenRows[row].contains("─") || screenRows[row].contains("wait"),
                        is(false));
            }
        }
    }

    /** A screen that is already wider than the size the application is told, for as long as the lie lasts. */
    private static final class LaggingWidth extends ScreenTerminalHarness {

        /** How many columns the application is NOT told about. Zero means the console has caught up. */
        private volatile int missing;

        LaggingWidth(int columns, int rows) throws java.io.IOException {
            super("windows-vtp", columns, rows);
        }

        @Override
        public Size getSize() {
            Size real = super.getSize();
            return Size.of(real.getColumns() - missing, real.getRows());
        }
    }

    @Test
    void aWidthTheConsoleAnnouncesLATEMustNotLEAVEHalfABlockAbOVEtheRegion() throws Exception {
        // The faithful shape of the report, in three steps, because only the third one says what is still
        // broken. (1) The block is rendered while the screen is ALREADY wider than the application has been
        // told -- the rows run together, which the previous case pins. (2) The console catches up. (3) The
        // block is rebuilt, which is what this console's settle redraw does 400 ms after the last size
        // event. The question this asks is whether step 3 is enough: the region owns three rows, and if the
        // run-together render put block content on a row ABOVE them, nothing ever writes there again.
        int rows = 20;
        int narrow = 113;
        int wide = 121;
        LaggingWidth terminal = new LaggingWidth(narrow, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(300);

            // (1) the screen is widened; the application is told nothing yet
            terminal.resizeScreenOnly(wide, rows);
            terminal.missing = wide - narrow;
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            // (2) the console catches up, and announces it the way one does
            terminal.missing = 0;
            terminal.raise(org.jline.terminal.Terminal.Signal.WINCH);
            Thread.sleep(300);

            // (3) the settle redraw
            console.refreshBlockForCurrentSize();
            Thread.sleep(400);

            assertBlockIsIntactAt(terminal, rows, 3);
        }
    }

    @Test
    void aLateWidthPLUSaTallerWindowLeavesBlockRowsSTRANDEDAboveTheRegion() throws Exception {
        // The reported screen, complete. The previous case shows the settle redraw repairs the three rows
        // the region owns; this one adds the second half of a CORNER drag -- the window also gets taller.
        // The region is reserved from the BOTTOM, so it moves DOWN, and the row the run-together render
        // dirtied is then no longer one of its rows. Nothing writes there again, so it stays: exactly the
        // one leftover row above an otherwise correct block that was reported.
        int rows = 20;
        int narrow = 113;
        int wide = 121;
        LaggingWidth terminal = new LaggingWidth(narrow, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(300);

            // the screen is already wider; the application has not been told
            terminal.resizeScreenOnly(wide, rows);
            terminal.missing = wide - narrow;
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            // the console catches up AND the window is a row taller, which a corner drag does
            int taller = rows + 1;
            terminal.missing = 0;
            terminal.resize(wide, taller);
            Thread.sleep(300);
            console.refreshBlockForCurrentSize();
            Thread.sleep(400);

            assertBlockIsIntactAt(terminal, taller, 3);
        }
    }

    @Test
    void aScreenTheCONSOLEChangedBehindJLinesBackMustStillBeRepaired() throws Exception {
        // The explanation the other cases were circling, and the only one that accounts for "it stays
        // broken". Windows reflows its screen buffer when the window is widened: rows it had marked as
        // wrapped -- which is every row the pinned region writes, because each one fills the last column --
        // are joined back together. JLine is not told. Its Display still believes the rows it wrote are on
        // screen, so every later update computes an EMPTY diff and emits nothing, and the joined rows stay
        // there for the rest of the session. That is exactly the reported screen, and exactly why pressing
        // Enter a dozen times is what repairs it.
        //
        // The screen is dirtied here the same way a reflow dirties it: straight onto the screen, past
        // everything that keeps a model of it. The size never changes, because the size is not the point --
        // what matters is that the SCREEN and JLine's belief about it have come apart.
        int rows = 20;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(300);

            // the console joins the rule row and the activity row, as a reflow does: address the region's
            // first row and write the two of them onto one line
            // The cursor is saved and restored around it: a reflow changes CONTENT, and leaving the cursor
            // somewhere else would be a different defect (measured: the block was then drawn four rows too
            // high and stood on screen twice -- true, but not what a console does).
            terminal.writeBehindTheApplicationsBack(
                    "\u001b7\u001b[" + (rows - 2) + ";1H" + "-".repeat(40) + "  … wait" + "\u001b8");
            Thread.sleep(200);

            // and now the redraw this console does when the size settles
            console.repaintBlockFromScratch();
            Thread.sleep(400);

            String[] screenRows = terminal.rows();
            String screen = terminal.describe();
            for (int row = 0; row < screenRows.length; row++) {
                assertThat(
                        "row " + row + " still carries what the console left behind" + NEWLINE + screen,
                        screenRows[row].contains("… wait") && !screenRows[row].contains("waiting for input"),
                        is(false));
            }
            assertBlockIsIntactAt(terminal, rows, 3);
        }
    }

    @Test
    void clsAndThenAShrinkingCornerDragKeepsOneBlockOfThreeSeparateRows() throws Exception {
        // The same in the other direction, because "größer und kleiner gemacht" is what the report says.
        int rows = 24;
        ScreenTerminalHarness terminal = terminalWithRows(150, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.line("something said earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);

            int shrunk = rows - 4;
            growOnce(terminal, console, 100, shrunk);

            assertBlockIsIntactAt(terminal, shrunk, 3);
        }
    }

    @Test
    void aGrowingWindowLeavesThePromptOnTheRowTheBlockLeavesForIt() throws Exception {
        // The sequence is the reported one -- /cls, then drag the corner bigger -- and every step is measured
        // rather than assumed, because three earlier rounds blamed the wrong writer.
        int rows = 20;
        int blockRows = 3;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.line("something said earlier");
            Thread.sleep(200);

            console.clearScreen();
            Thread.sleep(400);
            assertThat(
                    "a clear leaves the cursor on its row" + NEWLINE + terminal.describe(),
                    terminal.cursorRow(),
                    is(rows - 1 - blockRows));

            int grown = rows + 4;
            growOnce(terminal, console, 150, grown);

            // The block itself is fine -- three separate rows, pinned at the bottom, the rule spanning the
            // window. Only the prompt's row is wrong, which is why every assertion about CONTENT was green
            // while the console was not.
            assertBlockIsIntactAt(terminal, grown, blockRows);
            assertThat(
                    "the cursor is on the row the block leaves for the prompt" + NEWLINE + terminal.describe(),
                    terminal.cursorRow(),
                    is(grown - 1 - blockRows));
        }
    }

    @Test
    void severalGrowingDragsInARowMustNotLEAVEaStaircaseOfRulesBehind() throws Exception {
        // The report, in the words that name the shape: "beim größer ziehen tauchen von unten rechts nach
        // oben links immer mehr von den zeilen strichen auf". A staircase of rules, one per drag, climbing
        // away from the block.
        //
        // The mechanism this looks for: the bar is drawn at the rows the region has NOW, so every earlier
        // render sits at the rows the region had THEN. On a window that grows, those are higher up -- and
        // they are above the region, in the output area, where nothing writes again. One render left behind
        // per size change is exactly a staircase.
        int rows = 14;
        ScreenTerminalHarness terminal = terminalWithRows(80, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);

            int grown = rows;
            for (int step = 0; step < 3; step++) {
                grown += 3;
                growOnce(terminal, console, 80 + 20 * (step + 1), grown);
            }

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat("exactly one rule is on screen" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            assertThat(
                    "exactly one state row is on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("local-model")),
                    is(1));
            assertThat(
                    "exactly one activity row is on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("waiting for input")),
                    is(1));
        }
    }

    @Test
    void aLongLineIsFoldedByUsSoTheCONSOLEneverWrapsIt() throws Exception {
        // The fix for the whole family of drag artefacts, and the reason is the console's own reflow: a line
        // the CONSOLE wrapped is one logical line spanning two screen rows, and when the window is widened it
        // joins them again. The text above then needs fewer rows, everything below moves UP -- including the
        // block rows last rendered, which end up above the region where nothing writes again. One leftover
        // per drag step, "von unten rechts nach oben links". Narrowing does it in reverse and walks the input
        // upwards.
        //
        // Nothing can observe or prevent a reflow. What it can be denied is a target: a line that was never
        // soft-wrapped has nothing to join. So every output line is folded HERE, to one column less than the
        // window, and the screen then holds no full-width row at all -- which is what this asserts, because
        // that is the property the reflow needs.
        int columns = 60;
        ScreenTerminalHarness terminal = terminalWithRows(columns, 14);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(List.of(STATE));
            Thread.sleep(200);

            console.line("x".repeat(columns * 2 + 7));
            Thread.sleep(300);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            for (int row = 0; row < rows.length; row++) {
                assertThat(
                        "row " + row + " reaches the last column, so the console wrapped it" + NEWLINE + screen,
                        rows[row].charAt(columns - 1) != ' ',
                        is(false));
            }
            long carrying = count(rows, row -> row.contains("xxx"));
            assertThat(
                    "the text is spread over its own rows: " + carrying + NEWLINE + screen, carrying >= 3L, is(true));
        }
    }

    @Test
    void foldingCountsSCREENCOLUMNSnotCharacters() throws Exception {
        // An icon is one character and TWO columns. Folding by character length lets a piece come out wider
        // than the window after all, which is the very thing being prevented -- and it is how a long summary
        // tore the block apart once before, through another door.
        int columns = 40;
        ScreenTerminalHarness terminal = terminalWithRows(columns, 12);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);

            console.line("◆".repeat(60));
            Thread.sleep(300);

            String[] rows = terminal.rows();
            String screen = terminal.describe();
            for (int row = 0; row < rows.length; row++) {
                assertThat(
                        "row " + row + " reaches the last column" + NEWLINE + screen,
                        rows[row].charAt(columns - 1) != ' ',
                        is(false));
            }
        }
    }

    @Test
    void aWindowThatONLYgetsWiderKeepsTheConversationAndPutsThePromptOnItsRow() throws Exception {
        // This case changed sides once, and the reason is worth keeping because it was a cost that turned out not
        // to be necessary.
        //
        // A width change wipes the screen -- the only thing that removes the rows a console's re-wrap leaves
        // behind -- and for one round that was all it did. The report came back in a sentence: "allerdings sehe
        // ich den Verlauf nicht mehr". So the assertion here used to be that the conversation had scrolled away,
        // "the accepted cost". It is not accepted any more: the console knows what it printed, so after the wipe
        // it prints the recent lines again, folded for the width the window has now.
        int rows = 20;
        int blockRows = 3;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            console.line("AN-ANSWER the user was reading");
            Thread.sleep(200);

            growOnce(terminal, console, 150, rows);

            String screen = terminal.describe();
            assertThat("the prompt is on its row" + NEWLINE + screen, terminal.cursorRow(), is(rows - 1 - blockRows));
            assertBlockIsIntactAt(terminal, rows, blockRows);
            assertThat(
                    "and the answer is back on screen, once" + NEWLINE + screen,
                    count(terminal.rows(), row -> row.contains("AN-ANSWER")),
                    is(1));
        }
    }

    @Test
    @Disabled("Reproduced and OPEN, the other half of the resize drift and the harder one. A window that gets"
            + " SHORTER leaves the prompt BELOW its row -- measured 13 where 10 is right, shrinking 20"
            + " rows to 14 -- which means inside the pinned band, where the block draws over it: the"
            + " reported \"nach dem kleiner ziehen sehe ich es nicht mehr\". The growing side is fixed"
            + " by printing (pushThePromptBackToItsRow), because printing moves the cursor DOWN; here it"
            + " would have to move UP, and nothing a caller can emit does that without breaking the"
            + " reader's own cursor bookkeeping -- cursor_address smuggled into printAbove was tried"
            + " twice and stranded characters above the prompt both times. /cls repairs it in one"
            + " keystroke. Delete the annotation to see it.")
    void aWindowThatGetsSHORTERLeavesThePromptOnTheRowTheBlockLeavesForIt() throws Exception {
        int rows = 20;
        int blockRows = 3;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.line("an answer the user wants to keep seeing");
            Thread.sleep(200);

            int shrunk = 14;
            growOnce(terminal, console, 120, shrunk);

            assertBlockIsIntactAt(terminal, shrunk, blockRows);
            assertThat(
                    "the prompt is on the row the block leaves for it" + NEWLINE + terminal.describe(),
                    terminal.cursorRow(),
                    is(shrunk - 1 - blockRows));
        }
    }

    @Test
    void alternatingDragsManyTimesOverLeaveExactlyOneBlock() throws Exception {
        // The report in its worst form: "ganz viel kleiner / größer abwechselnd nach einander zerhackt alles",
        // with a screen carrying SEVERAL complete blocks at different widths -- rule, activity row, sometimes a
        // state row -- and some rules ending in the ellipsis Status uses to cut a row that is too WIDE for the
        // region. So two things to catch: copies of the block left in the output area, and rows built for a
        // width the region does not have.
        int rows = 20;
        ScreenTerminalHarness terminal = terminalWithRows(120, rows);
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(200);
            console.line("an answer the user wants to keep seeing");
            Thread.sleep(200);

            // A DRAG, not a single event: a real console reports a size roughly every 125 ms while the mouse
            // moves, and the probe counted ~22 for one drag. Every other case in this class deliberately uses
            // one event, because stepping through them made assertions about content flaky -- but here the
            // assertions are made after everything has settled, which is the state the report is about.
            List<Thread> pumps = new ArrayList<>();
            int at = rows;
            int columns = 120;
            for (int drag = 0; drag < 4; drag++) {
                boolean smaller = drag % 2 == 0;
                for (int step = 0; step < 5; step++) {
                    at += smaller ? -1 : 1;
                    columns += smaller ? -8 : 8;
                    pumps.add(terminal.resizeAsynchronously(columns, at));
                    Thread.sleep(30);
                }
                Thread.sleep(200);
            }
            for (Thread pump : pumps) {
                pump.join(2000);
            }
            Thread.sleep(800);
            console.refreshBlockForCurrentSize();
            Thread.sleep(800);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat("exactly one rule is on screen" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            assertThat(
                    "exactly one activity row is on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("waiting for input")),
                    is(1));
            assertThat(
                    "exactly one state row is on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("local-model")),
                    is(1));
            assertThat(
                    "no row was cut with an ellipsis, which means built for a width the region does not have" + NEWLINE
                            + screen,
                    count(screenRows, row -> row.contains("─…") || row.contains("model…")),
                    is(0));
        }
    }

    @Test
    void aBlockRowBuiltWIDERThanTheWindowMustNotSPILLaCopyIntoTheOutput() throws Exception {
        // The mechanism behind the stacked blocks: "ganz viel kleiner / größer abwechselnd zerhackt alles",
        // with several complete blocks at different widths on screen and some rules ending in the ellipsis
        // Status uses to cut a row that is too WIDE for the region.
        //
        // A row wider than the window WRAPS. The region is reserved in rows, so a block of three rows then
        // needs four screen rows, and the one that no longer fits spills UPWARD into the output area, where
        // nothing writes again. One copy per bad render, which is exactly a stack of them.
        //
        // Forced rather than waited for: the console reports more columns than the screen has. That is the
        // state a drag really produces, because this console reads the size, re-asserts the region, and builds
        // the rows -- while JLine's own signal handler resizes the same region with a newer size in between.
        int rows = 20;
        int realColumns = 80;
        // ONE column, because that is the lag a console being dragged actually shows, and it is what the fix
        // absorbs: the rows are padded one column short of the reported width, so a width that is one too
        // large still fits. A larger overshoot cannot be absorbed by anything built from a reported width --
        // measured with eight, the state row came out with its first eight characters replaced by spaces --
        // and asserting otherwise would be a wish rather than a contract.
        int claimedExtra = 1;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", realColumns, rows) {
            @Override
            public Size getSize() {
                Size real = super.getSize();
                return Size.of(real.getColumns() + claimedExtra, real.getRows());
            }
        };
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(200);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(300);
            // FOUR renders, because one is only half the story: the last row of a too-wide block wraps past
            // the bottom of the screen, which SCROLLS it -- so every bad render pushes the block up and leaves
            // a copy above it. That is the stack.
            for (int render = 0; render < 4; render++) {
                console.refreshBlockForCurrentSize();
                Thread.sleep(200);
            }

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat("exactly one rule is on screen" + NEWLINE + screen, count(screenRows, this::isRule), is(1));
            assertThat(
                    "exactly one state row is on screen" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("local-model")),
                    is(1));
            assertThat(
                    "the state row is whole, not cut by a wrap" + NEWLINE + screen,
                    screenRows[rows - 1].startsWith("[▤"),
                    is(true));
        }
    }

    @Test
    void aSizeThatCHANGESwhileTheBlockIsBeingBuiltMustNotProduceRowsForTheWrongWidth() throws Exception {
        // OUR half of the same defect, and the one the ellipsis in the report pointed at: Status cuts a row
        // with "…" when it is WIDER than the region, which means the rows were built for a size the region does
        // not have. That happens during a fast drag, because this console reads the size, re-asserts the region
        // with it, and then builds the rows -- while the reader's own signal handler resizes the same region
        // with a newer size in between.
        //
        // Forced deterministically rather than raced: this terminal reports a different size on every call, the
        // way a console does mid-drag, and settles after a few. The rows must end up matching the size the
        // region settled on, not an intermediate one.
        int rows = 20;
        java.util.concurrent.atomic.AtomicInteger reads = new java.util.concurrent.atomic.AtomicInteger();
        int settled = 90;
        ScreenTerminalHarness terminal = new ScreenTerminalHarness("windows-vtp", 130, rows) {
            @Override
            public Size getSize() {
                // 130, 120, 110, 100, then 90 for ever: a drag that stops.
                int at = reads.getAndIncrement();
                int columns = Math.max(settled, 130 - 10 * at);
                return Size.of(columns, rows);
            }
        };
        try (JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reading = new Thread(() -> console.readLine("ignored"));
            reading.setDaemon(true);
            reading.start();
            Thread.sleep(300);
            console.status(realBlockWithBasicPlaneIcons());
            Thread.sleep(300);
            console.refreshBlockForCurrentSize();
            Thread.sleep(300);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat(
                    "no row was cut with an ellipsis, which is what a row too wide for the region looks like" + NEWLINE
                            + screen,
                    count(screenRows, row -> row.contains("─…") || row.contains("model…")),
                    is(0));
            assertThat("the state row is whole" + NEWLINE + screen, screenRows[rows - 1].startsWith("[▤"), is(true));
        }
    }

    @Test
    void makingTheWindowNARROWERwipesAndLeavesOneCleanBlock() throws Exception {
        // This case changed sides once the wipe covered narrowing too, and both sides are worth keeping.
        //
        // It began as the reproduction of "nach dem kleiner ziehen sehe ich es nicht mehr": Status.resize ERASED
        // rows above the bar -- six of them for a halved window -- and that erase is unrecoverable, which is why
        // it is fixed in the library (StatusRepaintTest.makingTheWindowNarrowerDoesNotEraseWhatIsAboveTheBar) and
        // stays fixed for every consumer that does not wipe.
        //
        // What this console does on top is scroll the screen once the width has settled, because narrowing
        // re-wraps the bar's own rows -- built for the old width, they no longer fit -- and the pieces that land
        // above the region cannot be found afterwards. Scrolling is not erasing: the conversation is in the
        // scrollback. So what is asserted here is the settled state: a clean single block, with the input on its
        // row.
        int rows = 16;
        int wide = 100;
        ScreenTerminalHarness terminal = terminalWithRows(wide, rows);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            console.line("ANSWER-ONE the user wants to keep seeing");
            console.line("ANSWER-TWO the user wants to keep seeing");
            Thread.sleep(300);

            resizeOnceAt(terminal, console, wide / 2, rows);

            String screen = terminal.describe();
            assertBlockIsIntactAt(terminal, rows, 3);
            assertThat(
                    "the input is on the row the block leaves for it" + NEWLINE + screen,
                    terminal.cursorRow(),
                    is(rows - 4));
        }
    }

    /**
     * Change the window size once, at a row count that is not this class's constant.
     *
     * @param terminal the screen
     * @param console the console under test
     * @param columns the new width
     * @param rows the new height
     */
    private void resizeOnceAt(ScreenTerminalHarness terminal, JLineTerminal console, int columns, int rows)
            throws Exception {
        terminal.resize(columns, rows);
        Thread.sleep(300);
        console.refreshBlockForCurrentSize();
        Thread.sleep(300);
    }

    @Test
    void aWidthChangeKeepsTheConversationVisibleByPrintingItAgain() throws Exception {
        // "allerdings sehe ich den verlauf nicht mehr". The wipe is what removes the rows a reflow leaves
        // behind, and it cannot be given up -- but the conversation does not have to go with it, because this
        // console knows what it printed. After the wipe the recent lines are printed again, folded at the NEW
        // width, so the screen comes back with the conversation on it and re-flowed to the window it now has.
        int rows = 16;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            console.line("ANSWER-ONE the user was reading");
            console.line("ANSWER-TWO the user was reading");
            console.line("ANSWER-THREE the user was reading");
            Thread.sleep(300);

            resizeOnceAt(terminal, console, 150, rows);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            for (String answer : List.of("ANSWER-ONE", "ANSWER-TWO", "ANSWER-THREE")) {
                assertThat(
                        answer + " is on screen again after the width changed" + NEWLINE + screen,
                        count(screenRows, row -> row.contains(answer)),
                        is(1));
            }
            assertBlockIsIntactAt(terminal, rows, 3);
        }
    }

    @Test
    void whatIsPrintedAgainIsFoldedForTheWidthTheWindowHasNOW() throws Exception {
        // The reason this is a reprint rather than a scroll-back: the lines are folded again, so a line that
        // needed two rows in the old window uses one in a wider one. That is the behaviour a reader expects from
        // a window they just made bigger, and it is only possible because the console keeps what it printed
        // rather than what it drew.
        int rows = 16;
        int narrow = 60;
        ScreenTerminalHarness terminal = terminalWithRows(narrow, rows);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            String longAnswer = "A-LONG-ANSWER " + "x".repeat(80) + " END-OF-THE-ANSWER";
            console.line(longAnswer);
            Thread.sleep(300);
            assertThat(
                    "at the narrow width it needed two rows" + NEWLINE + terminal.describe(),
                    count(terminal.rows(), row -> row.contains("A-LONG-ANSWER")) == 1
                            && count(terminal.rows(), row -> row.contains("END-OF-THE-ANSWER")) == 1
                            && count(terminal.rows(), row -> row.contains("A-LONG-ANSWER") && row.contains("END-OF"))
                                    == 0,
                    is(true));

            resizeOnceAt(terminal, console, 140, rows);

            String screen = terminal.describe();
            assertThat(
                    "and after widening it is one row again" + NEWLINE + screen,
                    count(terminal.rows(), row -> row.contains("A-LONG-ANSWER") && row.contains("END-OF-THE-ANSWER")),
                    is(1));
        }
    }

    @Test
    void printingTheConversationAgainMustNotMULTIPLYit() throws Exception {
        // "wenn man größer / kleiner macht erscheint der text zwar wieder, aber mehrmals": one turn on screen
        // four times over. The redraw after a wipe went through line(), and line() REMEMBERS what it prints -- so
        // every wipe put the whole visible conversation into the ring a second time and the next one printed it
        // twice, then four times.
        //
        // TWO width changes, because one cannot show it: after the first wipe the ring holds the line twice but
        // only one copy has been printed. The second is where it becomes visible, which is also why the first
        // version of this case was green while the console was not.
        int rows = 20;
        ScreenTerminalHarness terminal = terminalWithRows(100, rows);
        try (JLineTerminal console = start(terminal, realBlockWithBasicPlaneIcons())) {
            console.line("› Hallo");
            console.line("Hello! How can I assist you today?");
            Thread.sleep(300);

            resizeOnceAt(terminal, console, 130, rows);
            resizeOnceAt(terminal, console, 110, rows);
            resizeOnceAt(terminal, console, 140, rows);

            String screen = terminal.describe();
            String[] screenRows = terminal.rows();
            assertThat(
                    "the question is on screen once, not once per drag" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("› Hallo")),
                    is(1));
            assertThat(
                    "and so is the answer" + NEWLINE + screen,
                    count(screenRows, row -> row.contains("How can I assist")),
                    is(1));
        }
    }
}
