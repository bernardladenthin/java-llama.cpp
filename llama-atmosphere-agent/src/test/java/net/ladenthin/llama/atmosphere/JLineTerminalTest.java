// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.utils.AttributedString;
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
        return terminal(new ByteArrayInputStream(keystrokes.getBytes(StandardCharsets.UTF_8)));
    }

    private Terminal terminal(java.io.InputStream keystrokes) throws Exception {
        return TerminalBuilder.builder()
                .streams(keystrokes, emitted)
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
    void clearingTheScreenLeavesTheReaderWorking() throws Exception {
        try (Terminal terminal = terminal("first\nsecond\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            assertThat(console.readLine("ignored"), is("first"));

            console.clearScreen();

            assertThat("and the prompt still reads afterwards", console.readLine("ignored"), is("second"));
        }
    }

    @Test
    void makingTheWindowNarrowerRedrawsTheBlockAtTheNewWidth() throws Exception {
        // This pins an assumption about JLine rather than logic of ours: it re-cuts the rows it holds
        // when the window shrinks, so nothing here has to. That is worth a test because the whole
        // bottom block depends on it -- a row wider than the window wraps onto a second screen line,
        // and the reserved region cannot survive that. An upgrade that changed it would show up here
        // instead of on somebody's screen.
        try (Terminal terminal = terminal("\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.status(List.of("state"));
            int before = screen().length();

            terminal.setSize(new Size(30, 10));
            console.status(List.of("state"));

            String afterResize = screen().substring(before);
            assertThat("the block was drawn again", afterResize.isEmpty(), is(false));
            assertThat(
                    "and never again at the width of the window that is gone",
                    afterResize.contains("─".repeat(SIZE.getColumns() - 1)),
                    is(false));
        }
    }

    @Test
    void aRowTooWideForTheNewWindowIsCutRatherThanWrapped() throws Exception {
        try (Terminal terminal = terminal("\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.status(List.of("x".repeat(50)));

            terminal.setSize(new Size(20, 10));
            console.status(List.of("x".repeat(50)));

            assertThat(
                    "the row that was rendered for the wide window is not reused", occurrences("x".repeat(50)), is(1));
            assertThat("it is cut for the narrow one", screen(), containsString("…"));
        }
    }

    @Test
    void everyResizeDrawsExactlyOnePrompt() throws Exception {
        // The reported artefact is a second, stale "> " left on screen after dragging the window.
        // This drives the path that redraws it -- a real size change plus the signal, with the reader
        // sitting in readLine as it does all session -- and pins that shrinking, growing and changing
        // the row count never produce a SECOND prompt. It holds for every size tried, which is what
        // says the remaining artefact is not in this path.
        //
        // At most one, not exactly one, and the difference is measured: on a JLine whose resize path
        // keeps its display model (the fix filed upstream for the duplication) widening the window
        // emits no prompt at all, because the terminal has already reflowed the line itself -- which
        // is the very reasoning JLine's own no-status-bar branch states. Requiring exactly one pinned
        // the repainting behaviour rather than the property, and went red against the fixed library
        // with "was <0L>". Zero is not the defect; two is.
        java.io.PipedOutputStream keys = new java.io.PipedOutputStream();
        try (Terminal terminal = terminal(new java.io.PipedInputStream(keys));
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            Thread reader = new Thread(() -> console.readLine("ignored"));
            reader.setDaemon(true);
            reader.start();
            Thread.sleep(200);
            console.status(List.of("state row"));

            int[][] sizes = {{30, 10}, {90, 10}, {45, 10}, {120, 24}};
            for (int[] size : sizes) {
                int before = screen().length();

                terminal.setSize(new Size(size[0], size[1]));
                terminal.raise(Terminal.Signal.WINCH);
                Thread.sleep(150);

                String drawn = screen().substring(before);
                long prompts =
                        drawn.chars().filter(character -> character == '>').count();
                assertThat(
                        "at most one prompt after resizing to " + size[0] + "x" + size[1] + ", drew " + prompts,
                        prompts <= 1L,
                        is(true));
            }
        }
    }

    @Test
    void aClearScrollsAWindowAndErasesNothing() throws Exception {
        // This test changed sides, and both sides are worth keeping on record because the wrong one was
        // shipped twice. It used to assert the opposite -- that a clear ERASES and scrolls nothing -- and
        // that was right for as long as a wipe was how the screen was cleared. Erasing has been given up:
        // clear_screen puts the cursor home, the reader draws its prompt where the cursor is, and the
        // prompt then sat at the top left while the block stayed pinned at the bottom ("nach /cls ist der
        // cursor auch ganz oben und nicht unten"). Everything tried to put it back on top of an erase was
        // reported as a new defect -- blank rows pulled the erased lines back into view, a cursor_address
        // inside printAbove's argument stranded a character above the prompt -- and the screen tests then
        // showed the erase is worse than it looks on its own: the reader redraws its prompt as a diff
        // against a belief the erase invalidates, so the measured result was no prompt on screen at all.
        //
        // Scrolling a window's worth of blank lines through printAbove does both halves at once and breaks
        // neither: the screen goes blank, what was written stays reachable with the scrollbar, the block is
        // never touched, and the cursor ends on its row because printing is what puts it there. Nothing is
        // erased, so nothing can be pulled back into view -- which is exactly why the two assertions below
        // are the pair they are.
        //
        // A pipe has no screen, so "the prompt is on its row afterwards" is not assertable here; that is
        // what ScreenUseCasesTest asserts, on an interpreted screen. What a pipe does show is how much was
        // scrolled and whether an erase was emitted at all.
        try (Terminal terminal = terminal("go\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            console.readLine("ignored");
            Thread.sleep(200);
            console.status(List.of("state row"));
            int before = screen().length();

            console.clearScreen();

            String drawn = screen().substring(before);
            long lineFeeds =
                    drawn.chars().filter(character -> character == '\n').count();
            assertThat(
                    "a clear scrolls a whole window: it emitted " + lineFeeds + " line feeds for a " + SIZE.getRows()
                            + "-row window",
                    lineFeeds >= SIZE.getRows(),
                    is(true));
            assertThat(
                    "and it erases nothing, so nothing can be pulled back into view",
                    drawn.contains(ERASE_DISPLAY),
                    is(false));
        }
    }

    @Test
    void resizingWithTextInTheInputDoesNotDrawThePromptBesideItself() throws Exception {
        // Reported: start, type Hallo without pressing Enter, drag the window -- and "> Hallo" appears
        // ten times side by side on one line. Each redraw lands NEXT to the previous one instead of
        // over it. The existing resize test missed the case because its input buffer was empty, so
        // there was nothing to redraw and one prompt per resize was the whole story.
        //
        // **This does not reproduce the report**, and it is kept for what it does cover: that an
        // application-side regression cannot start appending redraws. The harness raises a synthetic
        // WINCH, and Windows -- where all three resize reports come from -- has no SIGWINCH at all; the
        // size change arrives as a console event on a path this terminal never takes. Said here rather
        // than left to be inferred from a green run.
        java.io.PipedOutputStream keys = new java.io.PipedOutputStream();
        Terminal terminal = terminal(new java.io.PipedInputStream(keys));
        JLineTerminal console = JLineTerminal.over(terminal, List.of());
        try {
            Thread reader = new Thread(() -> console.readLine("ignored"));
            reader.setDaemon(true);
            reader.start();
            Thread.sleep(250);
            console.status(List.of("state row"));
            keys.write("Hallo".getBytes(StandardCharsets.UTF_8)); // typed, deliberately not submitted
            keys.flush();
            Thread.sleep(250);
            int before = screen().length();

            for (int resize = 0; resize < 4; resize++) {
                terminal.setSize(new Size(SIZE.getColumns() - 10 * (resize + 1), SIZE.getRows()));
                terminal.raise(Terminal.Signal.WINCH);
                Thread.sleep(150);
            }

            String drawn = screen().substring(before);
            assertThat(
                    "a redraw overwrites the input line rather than appending to it",
                    drawn.contains("Hallo> Hallo"),
                    is(false));
        } finally {
            console.close();
            terminal.close();
        }
    }

    @Test
    void leavingReleasesTheReservedRowsAndTheScrollRegion() throws Exception {
        // Reported: after /exit the block was still on screen, and resizing the window then reflowed it
        // into a mess. The pinned block is a *reserved scroll region* -- ESC[1;<n>r keeps the bottom
        // rows out of it -- so a session that ends without resetting that region leaves the terminal
        // restricted, and everything the shell prints afterwards, or any resize, is laid out inside a
        // window that no longer matches.
        Terminal terminal = terminal("go" + System.lineSeparator());
        JLineTerminal console = JLineTerminal.over(terminal, List.of());
        try {
            console.readLine("ignored");
            Thread.sleep(200);
            console.status(List.of("state row"));
            int before = screen().length();

            console.close();

            String drawn = screen().substring(before);
            assertThat(
                    "the reserved region is handed back, so the next program gets the whole window",
                    drawn.contains("\u001b[1;" + SIZE.getRows() + "r") || drawn.contains("\u001b[r"),
                    is(true));
        } finally {
            terminal.close();
        }
    }

    @Test
    void controlLClearsTheSameWayTheCommandDoes() throws Exception {
        // 0x0C is Ctrl-L. JLine binds it itself, so /cls is the second way to do this rather than the
        // only one -- and both must end with the prompt on the same row, which is why this console now
        // owns the binding. Two things are pinned here: the key is still a clear rather than a character
        // typed into the line (a keymap option could silently take that away), and it clears the way the
        // command does, by scrolling. Whether the PROMPT lands on its row is asserted where it is
        // visible, on the interpreted screen in ScreenUseCasesTest.
        try (Terminal terminal = terminal("\u000cstill here\n");
                JLineTerminal console = JLineTerminal.over(terminal, List.of())) {
            assertThat(console.readLine("ignored"), is("still here"));

            long lineFeeds =
                    screen().chars().filter(character -> character == '\n').count();
            assertThat(
                    "Ctrl-L scrolled rather than being typed into the line: " + lineFeeds + " line feeds",
                    lineFeeds >= SIZE.getRows(),
                    is(true));
            assertThat("and it erased nothing", screen().contains(ERASE_DISPLAY), is(false));
        }
    }

    @Test
    void aRowIsCutByScreenColumnsNotByCharacters() {
        // An icon takes two columns and one character. Cutting by character length lets the row come
        // out wider than the window, wrap onto a second screen line, and push everything below the
        // reserved region out of place -- the tearing that a long summary caused, through another door.
        String icons = "📁".repeat(20);

        String cut = JLineTerminal.fit(icons, 10);

        // What matters is the width on screen, not how many characters that took.
        assertThat("the row fits the window", new org.jline.utils.AttributedString(cut).columnLength() <= 10, is(true));
        assertThat(
                "a cut by characters would have kept nine icons, which is eighteen columns",
                cut.codePointCount(0, cut.length()) < 9,
                is(true));
        assertThat(cut.endsWith("…"), is(true));
        assertThat("plain text is untouched when it fits", JLineTerminal.fit("short", 10), is("short"));
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

    @Test
    void foldingLeavesAShortLineExactlyAsItWas() {
        assertThat(JLineTerminal.fold("kurz", 40), is(List.of("kurz")));
        assertThat(JLineTerminal.fold("", 40), is(List.of("")));
    }

    @Test
    void foldingBreaksOnColumnsAndNoPieceIsWiderThanTheWidth() {
        List<String> pieces = JLineTerminal.fold("x".repeat(95), 40);
        assertThat(pieces.size(), is(3));
        for (String piece : pieces) {
            assertThat(new AttributedString(piece).columnLength() <= 40, is(true));
        }
        assertThat(String.join("", pieces), is("x".repeat(95)));
    }

    @Test
    void foldingKeepsDoubleWidthGlyphsWhole() {
        // An icon is one character and two columns. A piece may therefore come out one column short of the
        // width rather than splitting the glyph in half -- what must never happen is a piece WIDER than the
        // window, which is the thing the console would wrap.
        List<String> pieces = JLineTerminal.fold("📊".repeat(30), 41);
        for (String piece : pieces) {
            assertThat(
                    "piece is " + new AttributedString(piece).columnLength() + " columns wide",
                    new AttributedString(piece).columnLength() <= 41,
                    is(true));
        }
        assertThat(String.join("", pieces), is("📊".repeat(30)));
    }

    @Test
    void foldingKeepsTheStylingACallerPutIn() {
        // The text arrives with ANSI already in it (bold echoes, coloured markdown). The escapes have zero
        // width, so they must not count towards the fold, and each piece has to carry its own styling or the
        // second one comes out plain.
        String bold = "\u001b[1m" + "y".repeat(90) + "\u001b[0m";
        List<String> pieces = JLineTerminal.fold(bold, 40);
        assertThat(pieces.size(), is(3));
        for (String piece : pieces) {
            assertThat("a piece carries its own styling: " + piece, piece.contains("\u001b["), is(true));
            assertThat(new AttributedString(piece).columnLength() <= 40 + 10, is(true));
        }
    }

    @Test
    void foldingKeepsUmlautsAndSharpS() {
        // Two bytes in UTF-8, one column on screen: a fold that counted bytes would cut them in half.
        String text = "Grüße über Straßen".repeat(6);
        List<String> pieces = JLineTerminal.fold(text, 30);
        assertThat(String.join("", pieces), is(text));
        for (String piece : pieces) {
            assertThat(new AttributedString(piece).columnLength() <= 30, is(true));
        }
    }
}
