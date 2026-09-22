// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

/** The console used whenever there is no terminal: one-shot runs, piped input, and these tests. */
class PlainTerminalTest {

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    private PlainTerminal terminal(String typed) {
        return new PlainTerminal(
                new PrintStream(out, true, StandardCharsets.UTF_8),
                typed == null ? null : new BufferedReader(new StringReader(typed)),
                Ansi.PLAIN);
    }

    private String written() {
        return out.toString(StandardCharsets.UTF_8);
    }

    @Test
    void linesAreWrittenAndInputIsReadBackLineByLine() {
        PlainTerminal terminal = terminal("first" + System.lineSeparator() + "second" + System.lineSeparator());

        terminal.line("hello");

        assertThat(written(), containsString("hello"));
        assertThat(terminal.readLine("you> "), is("first"));
        assertThat(written(), containsString("you> "));
        assertThat(terminal.readKey("allow? "), is("second"));
        assertThat(terminal.readLine("you> "), is(nullValue()));
    }

    @Test
    void theStatusLineIsPrintedBeforeThePromptBecauseNothingCanBePinned() {
        PlainTerminal terminal = terminal("x" + System.lineSeparator());
        terminal.status("[manual · ctx 0/16k]");

        terminal.readLine("you> ");

        assertThat(written(), containsString("[manual · ctx 0/16k]"));
        assertThat(
                written().indexOf("[manual"),
                is(org.hamcrest.Matchers.lessThan(written().indexOf("you> "))));
    }

    @Test
    void anAnswerIsNormalisedSoTheCallerCanCompareItToOneCase() {
        assertThat(terminal("YES" + System.lineSeparator()).readKey("? "), is("yes"));
        assertThat(terminal("  A  " + System.lineSeparator()).readKey("? "), is("a"));
    }

    @Test
    void withoutInputEverythingReadsAsEndOfInput() {
        PlainTerminal terminal = terminal(null);

        assertThat(terminal.readLine("you> "), is(nullValue()));
        assertThat(terminal.readKey("? "), is(nullValue()));
    }

    @Test
    void everyCommandNameIsOfferedForCompletion() {
        List<String> names = LocalAgent.commandNames();

        for (SlashCommands.Command command : SlashCommands.Command.values()) {
            for (String name : command.names()) {
                assertThat(names, hasItem(name));
            }
        }
    }
}
