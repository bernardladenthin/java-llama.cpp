// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.junit.jupiter.api.Test;

/**
 * What the screen would look like, asserted on the bytes the terminal emits.
 *
 * <p>A JLine terminal built over two streams renders exactly like one on a TTY — same escape
 * sequences, same line reader — so a drawing bug is visible in the output without a console. That is
 * worth having, because this class is the one place where a mistake is invisible to every other test
 * and obvious to whoever is using the agent.
 *
 * <p>The bug these tests were written for: the rule above the input used to be the first line of a
 * two-line prompt, while {@code ERASE_LINE_ON_FINISH} erases exactly <b>one</b> line — so every Enter
 * left a rule behind, and holding Enter drew a column of them. Drawing it as ordinary output instead
 * only moved the problem: then one rule stayed in the scrollback per turn and travelled up with it.
 * There is now exactly one rule, the first line of the pinned block, and these tests hold the console
 * to writing none at all.
 */
class JLineTerminalTest {

    /** As many columns as a narrow window, so a wrapped line would be obvious. */
    private static final Size SIZE = new Size(60, 10);

    /** What a cleared screen looks like on the wire: erase the whole display. */
    private static final String ERASE_DISPLAY = "\u001b[2J";

    private final ByteArrayOutputStream emitted = new ByteArrayOutputStream();

    private Terminal terminal(String keystrokes) throws Exception {
        return TerminalBuilder.builder()
                .streams(new ByteArrayInputStream(keystrokes.getBytes(StandardCharsets.UTF_8)), emitted)
                .type("xterm-256color")
                // The rule is drawn with U+2500. Both are needed: the writer encodes through the
                // stdout charset, which is not the one .encoding() sets, and without it every rule
                // arrives as a row of "?" and the assertions compare against bytes nobody wrote.
                .encoding(StandardCharsets.UTF_8)
                .stdoutEncoding(StandardCharsets.UTF_8)
                .size(SIZE)
                .provider("exec")
                .build();
    }

    private String screen() {
        return emitted.toString(StandardCharsets.UTF_8);
    }

    /**
     * How many times a full-width rule was written, in either of the two forms JLine uses.
     *
     * <p>Counting only one of them is how the first version of this test passed against the very bug
     * it was written for: a box character goes out as UTF-8 {@code U+2500} from {@code printAbove},
     * but inside a <em>prompt</em> JLine switches to the DEC line-drawing character set and sends
     * {@code ESC(0} + a row of {@code q} + {@code ESC(B}. A rule carried in the prompt is therefore
     * invisible to a search for {@code ─}.
     *
     * @return how many rules were written
     */
    private int rules() {
        int width = SIZE.getColumns() - 1;
        return occurrences("─".repeat(width)) + occurrences("\u001b(0" + "q".repeat(width));
    }

    private int occurrences(String needle) {
        int count = 0;
        for (int at = screen().indexOf(needle); at >= 0; at = screen().indexOf(needle, at + 1)) {
            count++;
        }
        return count;
    }

    @Test
    void pressingEnterSeveralTimesLeavesNoRuleInTheScrollback() throws Exception {
        try (Terminal terminal = terminal("\n\n\n\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            for (int i = 0; i < 4; i++) {
                assertThat("an empty line is still a line", console.readLine("ignored"), is(""));
            }

            assertThat("the only rule is the pinned one, and it is never written as output", rules(), is(0));
        }
    }

    @Test
    void whatWasTypedSurvivesAboveTheInputLine() throws Exception {
        try (Terminal terminal = terminal("hello\nworld\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            assertThat(console.readLine("ignored"), is("hello"));
            assertThat(console.readLine("ignored"), is("world"));

            // the input line itself is erased on Enter, so the echo is what keeps the transcript
            assertThat(screen(), containsString("hello"));
            assertThat(screen(), containsString("world"));
            assertThat("and still no rule travels along with it", rules(), is(0));
        }
    }

    @Test
    void anEmptyLineIsNotAPendingRequest() throws Exception {
        try (Terminal terminal = terminal("\n   \nreal\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.readLine("ignored"); // starts the reader; the rest queues up behind it

            waitFor(console::hasPendingInput);
            // Two blank lines are queued in front of it, and it is still the real one that counts:
            // otherwise holding Enter cancels one turn per keystroke and produces nothing.
            assertThat(console.hasPendingInput(), is(true));
            assertThat(console.readLine("ignored").isBlank(), is(true));
            assertThat(console.readLine("ignored"), is("real"));
            // End of input is pending too, and has to be: a session whose console has closed must
            // stop waiting rather than keep a turn running for nobody.
            assertThat(console.hasPendingInput(), is(true));
            assertThat("and it reads as no line at all", console.readLine("ignored"), is(nullValue()));
        }
    }

    /**
     * Wait for the reader thread to have caught up.
     *
     * @param condition what to wait for
     * @throws InterruptedException if interrupted while waiting
     */
    private void waitFor(java.util.function.BooleanSupplier condition) throws InterruptedException {
        for (int attempt = 0; attempt < 200 && !condition.getAsBoolean(); attempt++) {
            Thread.sleep(10);
        }
    }

    @Test
    void theScreenIsScrolledSoTheInputStartsAtTheBottom() throws Exception {
        // The reader draws its prompt where the cursor is, and only the block below is pinned to the
        // window. Without this the input floats after the output with the block far below it, and the
        // two only meet once enough output has scrolled the cursor down on its own.
        try (Terminal terminal = terminal("\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.readLine("ignored");

            long blankLines =
                    screen().chars().filter(character -> character == '\n').count();
            assertThat(
                    "the cursor is pushed to the last row before the first prompt",
                    blankLines >= SIZE.getRows() - 1,
                    is(true));
        }
    }

    @Test
    void clearingTheScreenWipesItAndLeavesTheReaderWorking() throws Exception {
        try (Terminal terminal = terminal("first\nsecond\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            assertThat(console.readLine("ignored"), is("first"));
            int before = screen().length();

            console.clearScreen();

            // The capability is terminfo source ("\E[H\E[2J"), so what must reach the screen is the
            // expanded form. Writing the capability as it comes prints it as text, which is what this
            // assertion caught the first time it ran.
            assertThat(
                    terminal.getStringCapability(org.jline.utils.InfoCmp.Capability.clear_screen), is(notNullValue()));
            assertThat("erase display reached the screen", screen().substring(before), containsString(ERASE_DISPLAY));
            assertThat("and the prompt still reads afterwards", console.readLine("ignored"), is("second"));
        }
    }

    @Test
    void controlLIsBoundToTheReadersOwnClearScreen() throws Exception {
        // 0x0C is Ctrl-L. It is bound by JLine itself, so /cls is the second way to do this rather
        // than the only one -- worth pinning, because a keymap option could silently take it away.
        try (Terminal terminal = terminal("\u000cstill here\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            assertThat(console.readLine("ignored"), is("still here"));
            assertThat("Ctrl-L cleared rather than being typed into the line", screen(), containsString(ERASE_DISPLAY));
        }
    }

    @Test
    void aMultiLineStringIsStillPrintedAsSeveralLines() throws Exception {
        try (Terminal terminal = terminal("\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.line("first" + System.lineSeparator() + "second");

            assertThat(screen(), containsString("first"));
            assertThat(screen(), containsString("second"));
        }
    }
}
