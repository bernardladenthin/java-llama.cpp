// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class SlashCommandsTest {

    @Test
    void aCommandIsRecognisedWithItsAliasesAndCase() {
        assertThat(SlashCommands.parse("/help").orElseThrow().command(), is(SlashCommands.Command.HELP));
        assertThat(SlashCommands.parse("/?").orElseThrow().command(), is(SlashCommands.Command.HELP));
        assertThat(SlashCommands.parse("  /HELP  ").orElseThrow().command(), is(SlashCommands.Command.HELP));
        assertThat(SlashCommands.parse("/quit").orElseThrow().command(), is(SlashCommands.Command.EXIT));
        assertThat(SlashCommands.parse("/reset").orElseThrow().command(), is(SlashCommands.Command.CLEAR));
        assertThat(SlashCommands.parse("/approve").orElseThrow().command(), is(SlashCommands.Command.MODE));
    }

    @Test
    void theRestOfTheLineIsTheArgument() {
        SlashCommands compact =
                SlashCommands.parse("/compact focus on the build errors").orElseThrow();

        assertThat(compact.command(), is(SlashCommands.Command.COMPACT));
        assertThat(compact.arguments(), is("focus on the build errors"));
        assertThat(compact.hasArguments(), is(true));
        assertThat(SlashCommands.parse("/compact").orElseThrow().hasArguments(), is(false));
        assertThat(SlashCommands.parse("/mode auto").orElseThrow().arguments(), is("auto"));
    }

    @Test
    void anythingElseIsAMessageForTheModel() {
        // An unknown command is NOT rejected: a line may legitimately start with a slash, and a user
        // who types /halp would rather get an answer than an error. The trade-off is deliberate.
        assertThat(SlashCommands.parse("/halp"), is(Optional.empty()));
        assertThat(SlashCommands.parse("/usr/bin/env is where?"), is(Optional.empty()));
        assertThat(SlashCommands.parse("what does /help do?"), is(Optional.empty()));
        assertThat(SlashCommands.parse(""), is(Optional.empty()));
        assertThat(SlashCommands.parse(null), is(Optional.empty()));
    }

    @Test
    void everyCommandIsDocumentedInTheHelpText() {
        String help = LocalAgent.prompt(LocalAgent.HELP_TEXT);

        for (SlashCommands.Command command : SlashCommands.Command.values()) {
            assertThat(help, containsString(command.canonicalName()));
        }
    }
}
