// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The resize reports, on a screen that <b>reflows</b> — which is the only kind that can show them.
 *
 * <p><b>Why this class exists next to {@code ScreenUseCasesTest}.</b> That one interprets JLine's screen, which
 * adjusts its buffer on a resize but never reflows, and consequently reported every drag case green while the
 * console was in pieces. {@link ReflowingScreenHarness} adds the one behaviour it lacks and is itself under test
 * ({@code ReflowingScreenHarnessTest}) against a pattern {@code ReflowProbe} prints on a real console, so what
 * is asserted here rests on a measurement rather than on an assumption.
 *
 * <p><b>The insight that made these cases possible.</b> This console folds every output line to one column less
 * than the window, so nothing it prints is ever soft-wrapped — at the width it was printed at. Making the window
 * <b>narrower</b> turns those same lines into wrapped ones, and widening then joins them again. That is why
 * enlarging alone looks fine while "ganz viel kleiner / größer abwechselnd zerhackt alles": the shrink
 * manufactures the wrapped lines that the next widening moves everything with.
 */
class ReflowResizeTest {

    private static final int ROWS = 16;
    private static final int WIDE = 100;
    private static final int NARROW = 50;

    private static boolean jlineCarriesTheFixes() {
        try {
            org.jline.utils.Status.class.getMethod("repaint");
            return true;
        } catch (NoSuchMethodException | RuntimeException e) {
            return false;
        }
    }

    @BeforeEach
    void onlyWithAJLineThatCarriesTheFixes() {
        Assumptions.assumeTrue(
                jlineCarriesTheFixes(), "needs the patched JLine: mvn test -Djline.version=4.4.6-statusfix8");
    }

    /** The block the application really pins, without astral glyphs. */
    private List<String> block() {
        return List.of("… waiting for input …", "[▤ X:/tmp/agent-sandbox · ⏸  manual · ▦ 0/16k · ⚒ 8 · ◆ local-model]");
    }

    private JLineTerminal start(ReflowingScreenHarness terminal) throws Exception {
        JLineTerminal console = JLineTerminal.over(terminal, List.of());
        Thread reading = new Thread(() -> console.readLine("ignored"));
        reading.setDaemon(true);
        reading.start();
        Thread.sleep(200);
        console.status(block());
        Thread.sleep(200);
        return console;
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

    private boolean isRule(String row) {
        return row.contains("─".repeat(10)) || row.contains("q".repeat(10));
    }

    private void resizeAndSettle(ReflowingScreenHarness terminal, JLineTerminal console, int columns, int rows)
            throws Exception {
        Thread pump = terminal.resizeAsynchronously(columns, rows);
        pump.join(2000);
        Thread.sleep(300);
        console.refreshBlockForCurrentSize();
        // Long enough for the settle too, which wipes after a widening.
        Thread.sleep(900);
    }

    @Test
    void shrinkingThenWideningLeavesExactlyOneBlock() throws Exception {
        // The reported sequence, and the one the non-reflowing screen cannot show. An answer is printed at the
        // wide width -- folded, so not wrapped. Narrowing wraps it after all. Widening joins it again, the text
        // above needs fewer rows, and everything below moves up: with it the rows the bar was last drawn on,
        // which are then above the region where nothing writes again.
        ReflowingScreenHarness terminal =
                new ReflowingScreenHarness("windows-vtp", WIDE, ROWS, ReflowingScreenHarness.Anchor.TOP);
        try (JLineTerminal console = start(terminal)) {
            console.line("I do not understand the request. Could you please clarify what you would like me to do?");
            Thread.sleep(200);

            resizeAndSettle(terminal, console, NARROW, ROWS);
            resizeAndSettle(terminal, console, WIDE, ROWS);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat(
                    "exactly one rule is on screen" + System.lineSeparator() + screen,
                    count(rows, this::isRule),
                    is(1));
            assertThat(
                    "exactly one activity row is on screen" + System.lineSeparator() + screen,
                    count(rows, row -> row.contains("waiting for input")),
                    is(1));
            assertThat(
                    "exactly one state row is on screen" + System.lineSeparator() + screen,
                    count(rows, row -> row.contains("local-model")),
                    is(1));
            assertThat(
                    "the rule is the third row from the bottom" + System.lineSeparator() + screen,
                    isRule(rows[ROWS - 3]),
                    is(true));
        }
    }

    @Test
    void alternatingSHRINKandGROWmanyTimesLeavesExactlyOneBlock() throws Exception {
        // "ganz viel kleiner / größer abwechselnd nach einander zerhackt alles", with /cls as the only repair.
        ReflowingScreenHarness terminal =
                new ReflowingScreenHarness("windows-vtp", WIDE, ROWS, ReflowingScreenHarness.Anchor.TOP);
        try (JLineTerminal console = start(terminal)) {
            console.line("I do not understand the request. Could you please clarify what you would like me to do?");
            Thread.sleep(200);

            for (int round = 0; round < 3; round++) {
                resizeAndSettle(terminal, console, NARROW, ROWS);
                resizeAndSettle(terminal, console, WIDE, ROWS);
            }

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat(
                    "exactly one rule is on screen" + System.lineSeparator() + screen,
                    count(rows, this::isRule),
                    is(1));
            assertThat(
                    "no row carries the rule AND the activity row" + System.lineSeparator() + screen,
                    count(rows, row -> isRule(row) && row.contains("wait")),
                    is(0));
            assertThat(
                    "exactly one state row is on screen" + System.lineSeparator() + screen,
                    count(rows, row -> row.contains("local-model")),
                    is(1));
        }
    }

    @Test
    void narrowingMustNotLEAVETheOldWIDERuleWrappedAcrossTheScreen() throws Exception {
        // The reported screen for a NARROWING: "beim kleiner ziehen wandert es nach oben mit ganz vielen
        // Zeilen", with one enormous run of dashes. A paste rejoins wrapped runs, so what is really there is the
        // OLD rule -- built for the old, much wider window -- re-wrapped by the console across several screen
        // rows. The bar then occupies more rows than the region reserves and everything above it is pushed up.
        //
        // The console must re-wrap it: the text no longer fits, so no wrap flag and no padding trick can prevent
        // it, and this console learns of the new size up to 120 ms later. So what is asserted is the state after
        // everything has settled: one rule, at the new width, on the row above the block, and nothing of the old
        // one left anywhere.
        int wide = 200;
        int narrow = 60;
        ReflowingScreenHarness terminal = new ReflowingScreenHarness("windows-vtp", wide, ROWS);
        try (JLineTerminal console = start(terminal)) {
            console.line("AN-ANSWER the user was reading");
            Thread.sleep(200);

            resizeAndSettle(terminal, console, narrow, ROWS);

            String screen = terminal.describe();
            String[] rows = terminal.rows();
            assertThat(
                    "exactly one rule is on screen" + System.lineSeparator() + screen,
                    count(rows, this::isRule),
                    is(1));
            assertThat(
                    "the rule is the row above the block" + System.lineSeparator() + screen,
                    isRule(rows[ROWS - 3]),
                    is(true));
            // "agent-sandbox", not "local-model": at 60 columns the state row is legitimately cut with
            // an ellipsis, so its tail is not on screen and looking for it would fail for the wrong reason.
            assertThat(
                    "exactly one state row is on screen" + System.lineSeparator() + screen,
                    count(rows, row -> row.contains("agent-sandbox")),
                    is(1));
            for (int row = 0; row < ROWS - 3; row++) {
                assertThat(
                        "row " + row + " is above the block and carries no piece of a rule" + System.lineSeparator()
                                + screen,
                        rows[row].contains("─"),
                        is(false));
            }
        }
    }
}
