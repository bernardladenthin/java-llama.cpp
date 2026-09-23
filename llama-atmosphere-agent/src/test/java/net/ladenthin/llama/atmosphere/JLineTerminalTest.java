// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

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
 * left a rule behind, and holding Enter drew a column of them.
 */
class JLineTerminalTest {

    /** As many columns as a narrow window, so a wrapped line would be obvious. */
    private static final Size SIZE = new Size(60, 10);

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
        return occurrences("─".repeat(width)) + occurrences("(0" + "q".repeat(width));
    }

    private int occurrences(String needle) {
        int count = 0;
        for (int at = screen().indexOf(needle); at >= 0; at = screen().indexOf(needle, at + 1)) {
            count++;
        }
        return count;
    }

    @Test
    void pressingEnterOnAnEmptyLineSeveralTimesDrawsTheRuleOnlyOnce() throws Exception {
        try (Terminal terminal = terminal("\n\n\n\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            for (int i = 0; i < 4; i++) {
                console.separator();
                assertThat("an empty line is still a line", console.readLine("ignored"), is(""));
            }

            assertThat("nothing was printed in between, so nothing needs separating again", rules(), is(1));
        }
    }

    @Test
    void whatWasTypedSurvivesAboveTheInputAndTheNextTurnIsSeparatedAgain() throws Exception {
        try (Terminal terminal = terminal("hello\nworld\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.separator();
            assertThat(console.readLine("ignored"), is("hello"));
            console.separator();
            assertThat(console.readLine("ignored"), is("world"));

            // the input line itself is erased on Enter, so the echo is what keeps the transcript
            assertThat(screen(), containsString("hello"));
            assertThat(screen(), containsString("world"));
            assertThat("the echo printed something, so the next read is separated again", rules(), is(2));
        }
    }

    @Test
    void outputPrintedWhileTheAgentWorksSeparatesTheNextInputAgain() throws Exception {
        try (Terminal terminal = terminal("\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.separator();
            console.line("some output from a tool");
            console.separator();

            assertThat("the rule is no longer next to the input, so it is drawn again", rules(), is(2));
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
