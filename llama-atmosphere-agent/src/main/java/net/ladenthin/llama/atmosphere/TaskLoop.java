// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.regex.Pattern;
import org.jspecify.annotations.Nullable;

/**
 * {@code /loop}: keep working on one task, step by step, until it is done — the state lives in a file,
 * not in the conversation.
 *
 * <p>Every step sends the <b>same</b> message: the task verbatim plus an instruction to read
 * {@value #LOOP_FILE}, do one concrete step, and write down what happened. The conversation history is
 * <b>dropped between steps</b>, which is the point of the file: the context never grows, so the loop
 * can run for hours, and what the model knows is exactly what it wrote down. (This is the shape
 * Claude Code's own ralph-wiggum plugin uses — re-inject the original prompt and let the filesystem be
 * the memory.)
 *
 * <p><b>Why a text marker and not a "done" tool.</b> A small local model produces a well-formed tool
 * call far less reliably than a line of text — below 7B, malformed calls are the norm, and
 * mini-SWE-agent reaches its SWE-bench results with a plain-text sentinel and no tool-call API at all.
 * So the loop ends when a line is <em>exactly</em> {@value #SENTINEL}; a substring never counts, which
 * is what keeps "I will answer &lt;&lt;TASK_COMPLETE&gt;&gt; when I am done" from ending the run.
 *
 * <p><b>The model saying "done" is not proof.</b> With {@code --check <command>} the sentinel is only
 * accepted when that command succeeds; otherwise its output goes back into the next step. A 4B model
 * declares victory early, and this is the cheapest defence against it.
 *
 * <p>Limits, all of them enforced here rather than trusted to the model: a step cap
 * ({@link LoopOptions#DEFAULT_MAX_STEPS} by default), a wall-clock budget, and a stall detector — if
 * neither the file nor any tool call changed anything for {@value #STALL_LIMIT} steps in a row, the
 * loop stops instead of burning tokens on a model that repeats itself.
 */
public final class TaskLoop {

    /** The file the loop keeps its plan and its notes in, inside the workspace. */
    public static final String LOOP_FILE = "AGENT-LOOP.md";

    /** The line that ends the loop. Matched as a whole line, never as a substring. */
    public static final String SENTINEL = "<<TASK_COMPLETE>>";

    /** Steps without any change before the loop gives up. */
    public static final int STALL_LIMIT = 3;

    /** How long a loop may run before it stops on its own. */
    public static final Duration DEFAULT_BUDGET = Duration.ofHours(2);

    private static final Pattern SENTINEL_LINE = Pattern.compile("^\\s*" + Pattern.quote(SENTINEL) + "\\s*$");

    private TaskLoop() {}

    /**
     * Whether an answer ends the loop.
     *
     * @param answer the model's complete answer for one step
     * @return {@code true} when one of its lines is exactly the sentinel
     */
    public static boolean isComplete(String answer) {
        return answer.lines().anyMatch(line -> SENTINEL_LINE.matcher(line).matches());
    }

    /**
     * The message sent for every step.
     *
     * @param options the loop options
     * @return the prompt, with the task, the file name and the check command filled in
     */
    public static String stepPrompt(LoopOptions options) {
        String checkHint = options.check() == null
                ? ""
                : System.lineSeparator() + "Before you declare the task done, run this and make sure it succeeds: "
                        + options.check();
        return LocalAgent.prompt(LocalAgent.LOOP_PROMPT)
                .replace("{task}", options.task())
                .replace("{file}", LOOP_FILE)
                .replace("{check_hint}", checkHint);
    }

    /**
     * Create the loop file when it does not exist yet.
     *
     * @param workspace the workspace directory
     * @param task the task, written into the file
     * @return the path of the loop file
     */
    public static Path ensureLoopFile(Path workspace, String task) {
        Path file = workspace.resolve(LOOP_FILE);
        try {
            if (!Files.exists(file)) {
                Files.writeString(
                        file,
                        LocalAgent.prompt(LocalAgent.LOOP_FILE_TEMPLATE).replace("{task}", task)
                                + System.lineSeparator(),
                        StandardCharsets.UTF_8);
            }
            return file;
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create " + file, e);
        }
    }

    /**
     * A fingerprint of the loop file, to tell a step that changed something from one that did not.
     *
     * @param file the loop file
     * @return size and content hash, or {@code -1} when the file cannot be read
     */
    public static long fingerprint(Path file) {
        try {
            return Files.exists(file)
                    ? Files.readString(file, StandardCharsets.UTF_8).hashCode()
                    : -1;
        } catch (IOException e) {
            return -1;
        }
    }

    /**
     * Why a loop ended.
     *
     * @param reason the wording shown to the user
     * @param completed whether the model declared the task finished
     */
    public record Outcome(String reason, boolean completed) {}

    /**
     * Run the loop until it is done, stopped, or out of budget.
     *
     * @param runner the runner
     * @param fileSystem the workspace filesystem for the sessions
     * @param terminal the console
     * @param workspace the workspace directory
     * @param options what to work on and for how long
     * @param stopped polled between steps; {@code true} ends the loop (Ctrl-C)
     * @param budget the wall-clock limit
     * @param callLog records every tool call of every step, so /calls shows what the loop did
     * @return why it ended
     * @throws InterruptedException if interrupted while waiting for a step or an interval
     */
    public static Outcome run(
            AgentRunner runner,
            org.atmosphere.ai.fs.AgentFileSystem fileSystem,
            AgentTerminal terminal,
            Path workspace,
            LoopOptions options,
            java.util.function.BooleanSupplier stopped,
            Duration budget,
            ToolCallLog callLog)
            throws InterruptedException {
        Path file = ensureLoopFile(workspace, options.task());
        terminal.line("loop: " + options.task());
        terminal.line("loop: state in " + file + ", max " + options.maxSteps() + " steps, budget "
                + budget.toMinutes() + " min"
                + (options.interval() == null
                        ? ""
                        : ", every " + options.interval().toSeconds() + "s")
                + (options.check() == null ? "" : ", check: " + options.check()));

        long deadline = System.nanoTime() + budget.toNanos();
        long lastFingerprint = fingerprint(file);
        int stalled = 0;
        String extra = "";

        for (int step = 1; step <= options.maxSteps(); step++) {
            if (stopped.getAsBoolean()) {
                return new Outcome("stopped after " + (step - 1) + " steps", false);
            }
            if (System.nanoTime() > deadline) {
                return new Outcome(
                        "budget of " + budget.toMinutes() + " min used up after " + (step - 1) + " steps", false);
            }
            terminal.line(terminal.ansi().dim("── loop step " + step + "/" + options.maxSteps() + " ──"));
            // the status row is rendered on every redraw, and a lambda may not close over the counter
            String stepLabel = "loop step " + step + "/" + options.maxSteps() + " · " + options.task();

            // A fresh history every step: the file is the memory, so the context cannot grow.
            ConsoleSession session = LocalAgent.turn(
                    runner,
                    fileSystem,
                    stepPrompt(options) + extra,
                    new java.util.ArrayList<>(),
                    terminal,
                    callLog,
                    step,
                    ignored -> stepLabel,
                    new TurnActivity());
            extra = "";
            if (session.failure() != null) {
                return new Outcome("step " + step + " failed: " + session.failure(), false);
            }

            // The marker is checked BEFORE the stall detector: a step that only answers "done" changes
            // no file and calls no tool, so the other order would report "no progress" on the very
            // step that finished the task.
            if (isComplete(session.text())) {
                String failure = checkFailure(options, workspace);
                if (failure == null) {
                    return new Outcome("done after " + step + " steps", true);
                }
                terminal.line(terminal.ansi().yellow("loop: the check failed, continuing"));
                extra = System.lineSeparator() + "You answered " + SENTINEL + ", but the check ("
                        + options.check() + ") failed:" + System.lineSeparator() + failure
                        + System.lineSeparator() + "Fix that first.";
            }

            long fingerprint = fingerprint(file);
            boolean changed = fingerprint != lastFingerprint || session.toolCalls() > 0;
            lastFingerprint = fingerprint;
            stalled = changed ? 0 : stalled + 1;
            if (stalled >= STALL_LIMIT) {
                return new Outcome("no progress for " + STALL_LIMIT + " steps (nothing written, no tools used)", false);
            }

            if (options.interval() != null && step < options.maxSteps()) {
                Thread.sleep(options.interval().toMillis());
            }
        }
        return new Outcome("step limit of " + options.maxSteps() + " reached", false);
    }

    /**
     * Run the check command, if there is one.
     *
     * @param options the loop options
     * @param workspace the directory the command runs in
     * @return the command output when it failed, or {@code null} when it succeeded or there is none
     */
    private static @Nullable String checkFailure(LoopOptions options, Path workspace) {
        String command = options.check();
        if (command == null) {
            return null;
        }
        try {
            String result = ShellTool.run(workspace, command, Duration.ofMinutes(10), 4000);
            return result.startsWith("exit code: 0") ? null : result;
        } catch (IOException e) {
            return "the check could not be started: " + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "the check was interrupted";
        }
    }

    /**
     * The tool names a loop needs; used only for the console hint.
     *
     * @param toolNames the offered tools
     * @return {@code true} when the file tools that the loop file needs are present
     */
    public static boolean canKeepNotes(List<String> toolNames) {
        return toolNames.contains("read_file") && toolNames.contains("write_file");
    }
}
