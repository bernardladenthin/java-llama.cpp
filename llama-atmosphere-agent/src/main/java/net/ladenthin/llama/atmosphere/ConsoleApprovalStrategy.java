// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalResolution;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.PendingApproval;
import org.atmosphere.ai.approval.ToolApprovalPolicy;
import org.jspecify.annotations.Nullable;

/**
 * Asks on the console before a tool that changes something runs: {@code [y]es / [n]o / [a]uto}.
 *
 * <p>Atmosphere does the gating itself — {@code ToolExecutionHelper} consults the
 * {@link ToolApprovalPolicy} and, when it says the call needs approval, blocks the tool loop on
 * {@link #awaitApprovalDetailed} before the executor runs. A {@link ApprovalResolution#deny() denial}
 * is handed to the model as the tool result {@code {"status":"cancelled","message":"Action cancelled
 * by user"}}, so it can replan instead of believing the command ran; a timeout becomes
 * {@code {"status":"timeout",…}}. Nothing of that is reimplemented here.
 *
 * <p>Answering {@code a} switches the whole session to {@link ApprovalMode#AUTO} and approves — the
 * remaining calls of the running turn included, because the mode is read per call. {@code /mode
 * manual} switches back.
 *
 * <p><b>Without a console the answer is "no".</b> In one-shot mode ({@code --prompt}) there is no one
 * to ask, so a gated call is denied and the reason is printed. That is deliberate (and what Claude
 * Code's non-interactive mode does): silently auto-approving would make an unattended run the most
 * permissive one. Pass {@code --auto} to run unattended.
 */
public final class ConsoleApprovalStrategy implements ApprovalStrategy {

    /**
     * The tools that ask before they run: the shell plus everything that writes. Reading (
     * {@code ls}, {@code read_file}, {@code glob}, {@code grep}) is never gated — it cannot change
     * the machine, and gating it would make the prompt so frequent that it stops being read.
     */
    public static final Set<String> GATED_TOOLS =
            Set.of(ShellTool.TOOL_NAME, "write_file", "edit_file", "delete", "rename");

    private static final int ARGUMENT_PREVIEW_CHARS = 300;

    private final AtomicReference<ApprovalMode> mode;
    private final @Nullable BufferedReader input;
    private final PrintStream out;
    private final Ansi ansi;

    /**
     * Create the strategy.
     *
     * @param mode the shared, mutable approval mode (also written by {@code /mode} and by an
     *     {@code [a]} answer)
     * @param input the console the user answers on, or {@code null} when nobody can be asked
     * @param out where the prompt is printed
     * @param ansi the styles for the question
     */
    public ConsoleApprovalStrategy(
            AtomicReference<ApprovalMode> mode, @Nullable BufferedReader input, PrintStream out, Ansi ansi) {
        this.mode = mode;
        this.input = input;
        this.out = out;
        this.ansi = ansi;
    }

    /**
     * The policy that decides which tools this strategy is asked about.
     *
     * @return a policy gating {@link #GATED_TOOLS}
     */
    public static ToolApprovalPolicy policy() {
        return ToolApprovalPolicy.custom(tool -> tool != null && GATED_TOOLS.contains(tool.name()));
    }

    /**
     * The gated tool names among {@code tools}, for the console.
     *
     * @param toolNames every offered tool name
     * @return the names that will ask before running
     */
    public static List<String> gated(List<String> toolNames) {
        return toolNames.stream().filter(GATED_TOOLS::contains).toList();
    }

    @Override
    public ApprovalOutcome awaitApproval(PendingApproval approval, StreamingSession session) {
        return awaitApprovalDetailed(approval, session).outcome();
    }

    @Override
    public ApprovalResolution awaitApprovalDetailed(PendingApproval approval, StreamingSession session) {
        if (mode.get() == ApprovalMode.AUTO) {
            return ApprovalResolution.approve();
        }
        if (input == null) {
            out.println();
            out.println(ansi.red("✗ " + approval.toolName() + " " + preview(approval)
                    + " — denied: no console to ask (run with --auto to allow tools unattended)"));
            out.flush();
            return ApprovalResolution.deny();
        }
        out.println();
        out.println(ansi.yellow("? " + approval.toolName()) + " " + ansi.dim(preview(approval)));
        while (true) {
            out.print(ansi.yellow("  allow? [y]es / [n]o / [a]uto (no more questions): "));
            out.flush();
            String answer = readLine();
            if (answer == null) {
                // stdin closed mid-turn: the same situation as having no console at all
                out.println();
                out.println("  denied (input closed)");
                out.flush();
                return ApprovalResolution.deny();
            }
            switch (answer.trim().toLowerCase(Locale.ROOT)) {
                case "y", "yes", "" -> {
                    return ApprovalResolution.approve();
                }
                case "n", "no" -> {
                    return ApprovalResolution.deny();
                }
                case "a", "auto" -> {
                    mode.set(ApprovalMode.AUTO);
                    out.println("  approval mode: auto (use /mode manual to ask again)");
                    out.flush();
                    return ApprovalResolution.approve();
                }
                default -> out.println("  please answer y, n or a");
            }
        }
    }

    private @Nullable String readLine() {
        try {
            return input == null ? null : input.readLine();
        } catch (IOException e) {
            return null;
        }
    }

    private static String preview(PendingApproval approval) {
        String arguments = String.valueOf(approval.arguments());
        return arguments.length() <= ARGUMENT_PREVIEW_CHARS
                ? arguments
                : arguments.substring(0, ARGUMENT_PREVIEW_CHARS) + "… (" + arguments.length() + " chars)";
    }
}
