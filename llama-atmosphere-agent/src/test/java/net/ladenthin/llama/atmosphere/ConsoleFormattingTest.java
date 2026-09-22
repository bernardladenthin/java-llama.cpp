// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The console cosmetics: colour decision, status line, and the streaming Markdown renderer. */
class ConsoleFormattingTest {

    // ----- Ansi -----

    private static Ansi detect(Map<String, String> env, boolean terminal) {
        return Ansi.detect(env::get, () -> terminal);
    }

    @Test
    void colourIsOnOnlyOnATerminal() {
        assertThat(detect(Map.of(), true).isEnabled(), is(true));
        assertThat(detect(Map.of(), false).isEnabled(), is(false));
    }

    @Test
    void theEnvironmentCanForceColourOnOrOff() {
        // NO_COLOR: "when present and not an empty string ... prevents the addition of ANSI color"
        assertThat(detect(Map.of("NO_COLOR", "1"), true).isEnabled(), is(false));
        assertThat(detect(Map.of("NO_COLOR", ""), true).isEnabled(), is(true));
        assertThat(detect(Map.of("TERM", "dumb"), true).isEnabled(), is(false));
        assertThat(detect(Map.of("CLICOLOR", "0"), true).isEnabled(), is(false));
        // forcing wins over "not a terminal", e.g. when piping into a pager
        assertThat(detect(Map.of("CLICOLOR_FORCE", "1"), false).isEnabled(), is(true));
        // ... but NO_COLOR is checked after CLICOLOR_FORCE, which is the documented precedence
        assertThat(detect(Map.of("CLICOLOR_FORCE", "1", "NO_COLOR", "1"), false).isEnabled(), is(true));
    }

    @Test
    void aPlainInstanceReturnsTheTextUnchanged() {
        assertThat(Ansi.PLAIN.bold("x") + Ansi.PLAIN.dim("y") + Ansi.PLAIN.red("z"), is("xyz"));
        assertThat(detect(Map.of(), true).bold("x"), containsString("\u001b["));
    }

    // ----- StatusLine -----

    @Test
    void theStatusLineShowsModeContextToolsAndModel() {
        String line = StatusLine.render(ApprovalMode.MANUAL, 1234, false, 16384, 9, "local-model");

        assertThat(line, is("[manual · ctx 1.2k/16k · 9 tools · local-model]"));
    }

    @Test
    void contextIsShownInThousandsAndWithoutASizeWhenItIsUnknown() {
        assertThat(StatusLine.context(812, false, 32768), is("ctx 812/33k"));
        assertThat(StatusLine.context(16000, false, 32768), is("ctx 16k/33k"));
        assertThat(StatusLine.context(0, false, StatusLine.UNKNOWN_CONTEXT), is("ctx 0"));
        assertThat(StatusLine.context(2500, false, StatusLine.UNKNOWN_CONTEXT), is("ctx 2.5k"));
    }

    @Test
    void anEstimatedCountIsMarkedWithATilde() {
        // llama.cpp reports usage only to clients that ask for it, and Atmosphere does not, so the
        // number normally comes from LocalAgent.estimateTokens -- the tilde says so.
        assertThat(StatusLine.context(2500, true, 16384), is("ctx ~2.5k/16k"));
        assertThat(
                LocalAgent.estimateTokens(
                        "0123456789", java.util.List.of(org.atmosphere.ai.llm.ChatMessage.user("0123456789"))),
                is(5L));
    }

    // ----- MarkdownConsole -----

    private static String render(String text, Ansi ansi) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        MarkdownConsole console = new MarkdownConsole(new PrintStream(buffer, true, StandardCharsets.UTF_8), ansi);
        // one character at a time: the renderer must not depend on where the stream splits
        for (int i = 0; i < text.length(); i++) {
            console.append(text.substring(i, i + 1));
        }
        console.flush();
        return buffer.toString(StandardCharsets.UTF_8);
    }

    @Test
    void withoutColourTheTextIsPassedThroughUnchanged() {
        String markdown = "# Title\n\nSome **bold** and `code`.\n- one\n- two\n";

        assertThat(
                render(markdown, Ansi.PLAIN),
                is("Title\n\nSome bold and code.\n• one\n• two\n".replace("\n", System.lineSeparator())));
    }

    @Test
    void headingsBulletsAndInlineSpansAreStyled() {
        Ansi ansi = detect(Map.of(), true);
        String out = render("## Heading\n- item with **bold**\ntext with `code` inside\n", ansi);

        assertThat(out, containsString(ansi.bold("Heading")));
        assertThat(out, containsString(ansi.cyan("•")));
        assertThat(out, containsString(ansi.bold("bold")));
        assertThat(out, containsString(ansi.cyan("code")));
        // the markers themselves are gone, that is the point of rendering
        assertThat(out, not(containsString("**")));
        assertThat(out, not(containsString("##")));
    }

    @Test
    void aFencedBlockIsStyledAsAWholeAndNotParsedInside() {
        Ansi ansi = detect(Map.of(), true);
        String out = render("```java\nint a = b * c; // **not bold**\n```\n", ansi);

        assertThat(out, containsString(ansi.cyan("int a = b * c; // **not bold**")));
    }

    @Test
    void anUnclosedMarkerIsLeftAsTyped() {
        // A half-streamed "**bold" must never swallow the rest of the line.
        assertThat(render("a **b\n", Ansi.PLAIN), is("a **b" + System.lineSeparator()));
        assertThat(render("2 * 3 * 4\n", Ansi.PLAIN), is("2 * 3 * 4" + System.lineSeparator()));
    }
}
