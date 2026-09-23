// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.atmosphere.ai.approval.ApprovalStrategy.ApprovalOutcome;
import org.atmosphere.ai.approval.PendingApproval;
import org.atmosphere.ai.approval.ToolApprovalPolicy;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.Test;

class ConsoleApprovalStrategyTest {

    @org.junit.jupiter.api.Test
    void everyOfferedToolIsEitherGatedOrDeclaredReadOnly() {
        // The gate is a list of names, so a tool that upstream adds or renames drops out of it and
        // then runs without asking -- in manual mode, silently. This is the check that turns that into
        // a red build: every tool the model is offered must be classified, one way or the other.
        java.util.List<String> offered =
                new java.util.ArrayList<>(WorkspaceTools.all(new WorkspaceTools.ReadTracker()).stream()
                        .map(ToolDefinition::name)
                        .toList());
        offered.add(ShellTool.TOOL_NAME);

        for (String tool : offered) {
            boolean gated = ConsoleApprovalStrategy.GATED_TOOLS.contains(tool);
            boolean readOnly = ConsoleApprovalStrategy.READ_ONLY_TOOLS.contains(tool);
            assertThat(
                    tool + " is in neither set: decide whether it has to ask before it runs",
                    gated || readOnly,
                    is(true));
            assertThat(tool + " cannot be both", gated && readOnly, is(false));
        }
        assertThat(
                "a set that names tools nobody offers is stale",
                offered.containsAll(ConsoleApprovalStrategy.GATED_TOOLS),
                is(true));
        assertThat(offered.containsAll(ConsoleApprovalStrategy.READ_ONLY_TOOLS), is(true));
    }

    private final ByteArrayOutputStream console = new ByteArrayOutputStream();
    private final TurnActivity activity = new TurnActivity();

    private PendingApproval approval() {
        return new PendingApproval(
                "id-1",
                ShellTool.TOOL_NAME,
                Map.of("command", "rm -rf build"),
                null,
                "console",
                Instant.now().plus(Duration.ofMinutes(5)));
    }

    private ApprovalOutcome ask(AtomicReference<ApprovalMode> mode, String typed) {
        BufferedReader reader = typed == null ? null : new BufferedReader(new StringReader(typed));
        AgentTerminal terminal =
                new PlainTerminal(new PrintStream(console, true, StandardCharsets.UTF_8), reader, Ansi.PLAIN);
        // The strategy never touches the session; Atmosphere passes it only so a UI can emit events.
        return new ConsoleApprovalStrategy(mode, terminal, typed != null, activity).awaitApproval(approval(), null);
    }

    private String consoleText() {
        return console.toString(StandardCharsets.UTF_8);
    }

    @Test
    void yesRunsTheToolOnceAndKeepsAsking() {
        AtomicReference<ApprovalMode> mode = new AtomicReference<>(ApprovalMode.MANUAL);

        assertThat(ask(mode, "y\n"), is(ApprovalOutcome.APPROVED));
        assertThat(mode.get(), is(ApprovalMode.MANUAL));
        assertThat(consoleText(), containsString(ShellTool.TOOL_NAME));
        assertThat(consoleText(), containsString("rm -rf build"));
    }

    @Test
    void anEmptyAnswerMeansYes() {
        assertThat(ask(new AtomicReference<>(ApprovalMode.MANUAL), "\n"), is(ApprovalOutcome.APPROVED));
    }

    @Test
    void noDeniesAndAtmosphereTellsTheModel() {
        // The cancellation text itself is Atmosphere's ("Action cancelled by user"), which is why this
        // side only has to return DENIED -- see ApprovalWireTest for the message reaching the model.
        assertThat(ask(new AtomicReference<>(ApprovalMode.MANUAL), "n\n"), is(ApprovalOutcome.DENIED));
    }

    @Test
    void autoApprovesAndStopsAskingForTheRestOfTheSession() {
        AtomicReference<ApprovalMode> mode = new AtomicReference<>(ApprovalMode.MANUAL);

        assertThat(ask(mode, "a\n"), is(ApprovalOutcome.APPROVED));
        assertThat(mode.get(), is(ApprovalMode.AUTO));
        // the next call must not read anything: an empty reader would otherwise mean "input closed"
        assertThat(ask(mode, ""), is(ApprovalOutcome.APPROVED));
    }

    @Test
    void anUnreadableAnswerIsAskedAgain() {
        assertThat(ask(new AtomicReference<>(ApprovalMode.MANUAL), "maybe\ny\n"), is(ApprovalOutcome.APPROVED));
        assertThat(consoleText(), containsString("please answer y, n or a"));
    }

    @Test
    void withoutAConsoleTheAnswerIsNo() {
        // One-shot mode: nobody can answer, so the safe outcome is a denial with a printed reason --
        // never a silent auto-approval, which would make an unattended run the most permissive one.
        assertThat(ask(new AtomicReference<>(ApprovalMode.MANUAL), null), is(ApprovalOutcome.DENIED));
        assertThat(consoleText(), containsString("--auto"));

        assertThat(ask(new AtomicReference<>(ApprovalMode.AUTO), null), is(ApprovalOutcome.APPROVED));
    }

    @Test
    void closedInputDeniesRatherThanBlocking() {
        assertThat(ask(new AtomicReference<>(ApprovalMode.MANUAL), ""), is(ApprovalOutcome.DENIED));
        assertThat(consoleText(), containsString("input closed"));
    }

    @Test
    void writingToolsAndTheShellAreGatedReadingToolsAreNot() {
        ToolApprovalPolicy policy = ConsoleApprovalStrategy.policy();

        for (String gated : new String[] {ShellTool.TOOL_NAME, "write_file", "edit_file", "delete", "rename"}) {
            assertThat(gated, policy.requiresApproval(stub(gated)), is(true));
        }
        for (String free : new String[] {"ls", "read_file", "glob", "grep"}) {
            assertThat(free, policy.requiresApproval(stub(free)), is(false));
        }
        assertThat(
                ConsoleApprovalStrategy.gated(java.util.List.of("ls", "write_file", ShellTool.TOOL_NAME)),
                is(java.util.List.of("write_file", ShellTool.TOOL_NAME)));
    }

    @Test
    void theSpinnerIsPausedWhileTheQuestionIsOpen() {
        // The turn runs on its own thread while the console thread redraws the status block four times
        // a second. A redraw arriving in the middle of a raw-mode key read writes escape sequences
        // across the question, so the prompt owns the terminal until it has an answer.
        assertThat(activity.isPaused(), is(false));

        ask(new AtomicReference<>(ApprovalMode.MANUAL), "y" + System.lineSeparator());

        assertThat("and hands it back afterwards", activity.isPaused(), is(false));
        assertThat(consoleText(), containsString("allow?"));
    }

    private static ToolDefinition stub(String name) {
        return ToolDefinition.builder(name, "test").executor(args -> "").build();
    }
}
