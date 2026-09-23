// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

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
    void theStatusLineShowsWorkspaceModeContextToolsAndModel() {
        String line = StatusLine.render(
                java.nio.file.Path.of("/tmp/ws"), ApprovalMode.MANUAL, 1234, false, 16384, 9, "local-model", false);

        // an icon, a space, its value -- the same shape for every part of the line
        assertThat(line, containsString("📁 "));
        assertThat(line, containsString("ws · ⏸ manual · 📊 1.2k/16k · 🔧 9 · 🤖 local-model]"));
    }

    @Test
    void shiftTabCyclesTheModeAndAPlainStreamDeclinesTheShortcut() {
        // the key can only be seen by a console that owns the keyboard; everything else keeps /mode
        assertThat(ApprovalMode.MANUAL.next(), is(ApprovalMode.AUTO));
        assertThat(ApprovalMode.AUTO.next(), is(ApprovalMode.MANUAL));
        assertThat(
                "a cycle, so it always returns to where it started",
                ApprovalMode.MANUAL.next().next(),
                is(ApprovalMode.MANUAL));

        AgentTerminal plain =
                new PlainTerminal(new java.io.PrintStream(new java.io.ByteArrayOutputStream()), null, Ansi.PLAIN);
        assertThat(
                plain.onCycleMode(() -> {
                    throw new AssertionError("must not run");
                }),
                is(false));
    }

    @Test
    void eachModeCarriesItsOwnSymbol() {
        // the glyph is what makes the mode findable at a glance; the word stays next to it
        assertThat(ApprovalMode.MANUAL.badge(), is("⏸ manual"));
        assertThat(ApprovalMode.AUTO.badge(), is("⏵⏵ auto"));
        assertThat(
                StatusLine.render(java.nio.file.Path.of("/tmp/ws"), ApprovalMode.AUTO, 0, true, 0, 1, "m", false),
                containsString("⏵⏵ auto"));
    }

    @Test
    void aRemoteEndpointIsMarkedDifferentlyFromAModelLoadedHere() {
        java.nio.file.Path ws = java.nio.file.Path.of("/tmp/ws");
        assertThat(StatusLine.render(ws, ApprovalMode.MANUAL, 0, true, 0, 1, "m", false), containsString("🤖 m"));
        assertThat(StatusLine.render(ws, ApprovalMode.MANUAL, 0, true, 0, 1, "m", true), containsString("🌐 m"));
    }

    @Test
    void aLongWorkspacePathIsShortenedToItsLastTwoSegments() {
        // the path is on every line of the session, so it must not push the rest off the screen
        java.nio.file.Path deep = java.nio.file.Path.of("/home/someone/projects/customer/service/backend/module");
        assertThat(StatusLine.shorten(deep), containsString("backend"));
        assertThat(StatusLine.shorten(deep), containsString("module"));
        assertThat(StatusLine.shorten(deep).startsWith("…"), is(true));
        assertThat(
                StatusLine.shorten(java.nio.file.Path.of("/tmp/ws")),
                is(java.nio.file.Path.of("/tmp/ws").toString()));
    }

    @Test
    void contextIsShownInThousandsAndWithoutASizeWhenItIsUnknown() {
        assertThat(StatusLine.context(812, false, 32768), is("812/33k"));
        assertThat(StatusLine.context(16000, false, 32768), is("16k/33k"));
        assertThat(StatusLine.context(0, false, StatusLine.UNKNOWN_CONTEXT), is("0"));
        assertThat(StatusLine.context(2500, false, StatusLine.UNKNOWN_CONTEXT), is("2.5k"));
    }

    @Test
    void anEstimatedCountIsMarkedWithATilde() {
        // llama.cpp reports usage only to clients that ask for it, and Atmosphere does not, so the
        // number normally comes from LocalAgent.estimateTokens -- the tilde says so.
        assertThat(StatusLine.context(2500, true, 16384), is("~2.5k/16k"));
        assertThat(
                LocalAgent.estimateTokens(
                        "0123456789", java.util.List.of(org.atmosphere.ai.llm.ChatMessage.user("0123456789"))),
                is(5L));
    }

    // ----- MarkdownConsole -----

    private static String render(String text, Ansi ansi) {
        StringBuilder buffer = new StringBuilder();
        MarkdownConsole console =
                new MarkdownConsole(line -> buffer.append(line).append(System.lineSeparator()), ansi);
        // one character at a time: the renderer must not depend on where the stream splits
        for (int i = 0; i < text.length(); i++) {
            console.append(text.substring(i, i + 1));
        }
        console.flush();
        return buffer.toString();
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

    // ----- the pinned block -----

    @Test
    void aStatusRowIsCutToTheWindowWidthBecauseAWrappedRowBreaksTheBlock() {
        // A wrapped row takes two screen lines while the reserved region is sized in lines, so
        // everything below it lands in the wrong place -- a long /compact summary tore the block apart.
        assertThat(JLineTerminal.fit("short", 20), is("short"));
        assertThat(JLineTerminal.fit("0123456789", 10), is("0123456789"));
        assertThat(JLineTerminal.fit("0123456789x", 10), is("012345678…"));
        assertThat(JLineTerminal.fit("0123456789x", 10).length(), is(10));
    }
}
