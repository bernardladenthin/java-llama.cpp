// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.args.LogFormat;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.jspecify.annotations.Nullable;

/**
 * A local, general-purpose agent in the spirit of Claude Code / OpenCode, built from two parts that already
 * exist: <b>Atmosphere</b>'s built-in OpenAI-compatible agent runtime (streaming, tool loop, workspace file
 * tools) and <b>java-llama.cpp</b>'s OpenAI-compatible server.
 *
 * <p>Two ways to reach a model:
 *
 * <ul>
 *   <li>{@code --base-url http://127.0.0.1:8080/v1} — a server you started yourself (java-llama.cpp's fat
 *       jar {@code NativeServer} with {@code --jinja}, its {@code OpenAiCompatServer}, or upstream
 *       {@code llama-server}), so you keep full control over model parameters.
 *   <li>{@code --model model.gguf} — loads the GGUF in this JVM and serves it to the agent over a loopback
 *       {@code OpenAiCompatServer}: one process, one command.
 * </ul>
 *
 * <p>Every way of talking to it goes through the same {@link AgentSession}; this class is the entry point
 * and the two consoles — the plain line-oriented one ({@code --plain}) and the JLine one (the default at a
 * terminal).
 *
 * <p>Run from the source tree: {@code mvn -q compile exec:java -Dexec.args="--base-url ... --workspace
 * /path --allow-shell"}. Exit code 0 on a completed turn, 1 when the turn errored, 2 on bad usage.
 */
public final class LocalAgent {

    /** How often the activity line is refreshed while a turn runs. */
    private static final Duration ACTIVITY_INTERVAL = Duration.ofMillis(250);

    /** The spinner shown in the activity line. */
    private static final String ACTIVITY_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

    /** The first line of the block while nothing is running. */
    static final String IDLE_LINE = "… waiting for input …";

    /** The whimsical words the activity line picks from, one per turn. */
    static final String SPINNER_WORDS = "spinner-words.txt";

    /** A line that is exactly this opens and closes a multi-line message. */
    static final String BLOCK_FENCE = "\"\"\"";

    private LocalAgent() {}

    /**
     * Entry point.
     *
     * @param args see {@link AgentOptions#usage()}
     * @throws Exception on an unrecoverable setup failure (model load, socket bind)
     */
    public static void main(String[] args) throws Exception {
        AgentOptions options;
        try {
            options = AgentOptions.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.err.println(AgentOptions.usage());
            System.exit(2);
            return;
        }
        if (options.isHelp()) {
            System.out.println(AgentOptions.usage());
            return;
        }
        if (options.isWeb()) {
            System.exit(WebAgent.run(options, System.err));
            return;
        }
        if (options.isAcp()) {
            System.exit(AcpServer.run(options, System.err));
            return;
        }
        System.exit(run(
                options,
                System.in == null ? null : new InputStreamReader(System.in, StandardCharsets.UTF_8),
                System.out,
                System.err));
    }

    /**
     * Run the agent with parsed options on a console.
     *
     * @param options the options
     * @param input the interactive input (ignored in one-shot mode), or {@code null} for none
     * @param out the console the answer streams to
     * @param err diagnostics
     * @return the process exit code
     * @throws Exception on an unrecoverable setup failure
     */
    static int run(AgentOptions options, java.io.@Nullable Reader input, PrintStream out, PrintStream err)
            throws Exception {
        AgentTerminal terminal = null;
        try (ModelEndpoint endpoint = ModelEndpoint.open(options, err)) {
            AgentSession session = AgentSession.open(options, endpoint.baseUrl(), endpoint.contextSize());
            err.println("Endpoint " + endpoint.baseUrl() + " models=" + session.models() + " workspace="
                    + options.getWorkspace() + " tools=" + session.toolNames());

            boolean interactive = options.getPrompt() == null && input != null;
            BufferedReader reader = input == null ? null : new BufferedReader(input);
            terminal = usesFullTerminal(options, interactive) ? JLineTerminal.open(AgentSession.commandNames()) : null;
            if (terminal == null) {
                terminal = new PlainTerminal(out, reader, Ansi.detect());
            }
            if (endpoint.inProcess()) {
                captureNativeLog(terminal);
            }
            // One-shot runs have nobody at the keyboard, so the strategy gets no console and denies gated
            // calls unless --auto was passed (see ConsoleApprovalStrategy).
            TurnActivity activity = new TurnActivity();
            SessionFrontend frontend = new ConsoleFrontend(
                    terminal,
                    activity,
                    new ConsoleApprovalStrategy(session.modeReference(), terminal, interactive, activity));

            if (options.getPrompt() != null) {
                return session.send(options.getPrompt(), frontend).failure() == null ? 0 : 1;
            }
            if (reader == null) {
                err.println("No interactive input available; pass --prompt <text>.");
                return 2;
            }
            AgentTerminal console = terminal;
            boolean shortcut = console.onCycleMode(() -> {
                session.cycleMode();
                showStatus(console, session.status());
            });
            err.println("Interactive mode: type a request, /help for the commands."
                    + (shortcut ? " shift+tab switches the approval mode." : ""));
            err.println(describeTerminal(console));
            while (true) {
                showStatus(terminal, session.status());
                String line = terminal.readLine("you> ");
                if (line == null) {
                    return 0;
                }
                if (BLOCK_FENCE.equals(line.strip())) {
                    line = readBlock(terminal);
                    if (line == null) {
                        return 0;
                    }
                }
                if (session.submit(line, frontend) == AgentSession.Result.EXIT) {
                    return 0;
                }
            }
        } finally {
            if (terminal != null) {
                terminal.close();
            }
        }
    }

    /**
     * Show the session's state between turns.
     *
     * <p>Pinned to the bottom of the window on a real terminal; printed above the prompt on a plain stream,
     * where there is nothing to pin and a repeatedly refreshed line would just fill a piped log.
     */
    private static void showStatus(AgentTerminal terminal, String status) {
        if (terminal.pinsStatus()) {
            terminal.status(List.of(IDLE_LINE, status));
        } else {
            terminal.line(terminal.ansi().dim(status));
        }
    }

    /**
     * Route llama.cpp's own log through the console instead of letting it write to stderr.
     *
     * <p>**This is what destroys a pinned block, and nothing on the Java side can defend against it.** With
     * an in-process model the server logs to stderr, which is the same console; those writes go around the
     * line reader, so the terminal scrolls lines JLine never sees and its reserved region ends up somewhere
     * else than it believes. What that looks like: a warning printed into the middle of the input line
     * (`&gt; d1.19.029.542 W srv stop: cancel task`), and after a few of them the block is gone. Routing the
     * log through {@link AgentTerminal#line} puts it under the same lock as everything else, so it scrolls in
     * above the prompt like any other output.
     *
     * <p>Only for an in-process model: with {@code --base-url} the server is another process and its log is
     * its own business, and calling this would load the native library for nothing.
     *
     * @param terminal where the log lines go
     */
    private static void captureNativeLog(AgentTerminal terminal) {
        Ansi ansi = terminal.ansi();
        LlamaModel.setLogger(LogFormat.TEXT, (level, message) -> {
            String text = message == null ? "" : message.strip();
            if (!text.isEmpty()) {
                terminal.line(ansi.dim(text));
            }
        });
    }

    /**
     * Read a block of lines, the way a fenced code block is written.
     *
     * <p>A console reads a line at a time, and Enter sends it — which makes pasting a stack trace or a
     * function into the prompt impossible without it becoming several questions. A line that is exactly
     * {@value #BLOCK_FENCE} starts a block and the next one closes it; everything between is one message,
     * newlines and all.
     *
     * <p>Chosen over a key combination because it works in both consoles, survives a paste (the fence
     * arrives as part of the pasted text), and needs nothing from the terminal.
     *
     * @param terminal where the lines come from
     * @return the block, or {@code null} at end of input
     */
    static @Nullable String readBlock(AgentTerminal terminal) {
        StringBuilder block = new StringBuilder();
        while (true) {
            String line = terminal.readLine("... ");
            if (line == null) {
                // End of input inside a block: what was collected is still a question worth asking, but
                // there is nobody left to answer it, so the session ends as it would anyway.
                return null;
            }
            if (BLOCK_FENCE.equals(line.strip())) {
                return block.toString();
            }
            if (block.length() > 0) {
                block.append(System.lineSeparator());
            }
            block.append(line);
        }
    }

    /**
     * Whether to drive the cursor-controlling console rather than the line-oriented one.
     *
     * <p>Both consoles are kept, and this is the only place that decides between them. The rich one needs
     * someone at a terminal <em>and</em> permission to position the cursor; {@code --plain} withholds the
     * second even when the first is true, which is what a session that is piped, logged, recorded, or run
     * through something that only forwards lines needs. A run without an interactive input has no use for it
     * either way.
     *
     * @param options the parsed command line
     * @param interactive whether there is someone typing
     * @return {@code true} to try the full terminal
     */
    static boolean usesFullTerminal(AgentOptions options, boolean interactive) {
        return interactive && !options.isPlain();
    }

    /**
     * Run one turn on a console, outside a session: what a bare runner and the tests use.
     *
     * @param runner the runner
     * @param fileSystem the workspace filesystem the tools use
     * @param message the message
     * @param history the conversation, extended by the turn
     * @param terminal the console
     * @param callLog records the turn's tool calls
     * @param turnNumber the turn's number, for the log
     * @param stateLine the second row of the block while the turn runs
     * @param activity paused while an approval question is open
     * @return the finished turn
     * @throws InterruptedException if interrupted while waiting
     */
    static TurnRecorder turn(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            String message,
            List<ChatMessage> history,
            AgentTerminal terminal,
            ToolCallLog callLog,
            int turnNumber,
            Function<TurnRecorder, String> stateLine,
            TurnActivity activity)
            throws InterruptedException {
        return AgentSession.runTurn(
                runner,
                fileSystem,
                message,
                history,
                new ConsoleFrontend(terminal, activity, null),
                callLog,
                turnNumber,
                stateLine,
                handle -> {});
    }

    /**
     * Wait for a turn while the status line shows that something is happening.
     *
     * <p>A local model can think for a while before the first token arrives, and a silent console is
     * indistinguishable from a hung one. The line is rewritten in place (it is the pinned status area, not
     * the scrollback), so nothing the user has already read moves.
     *
     * @param session the running turn
     * @param terminal the console
     * @param stateLine the second row of the block, asked again on every redraw so the context figure moves
     *     while the turn runs rather than standing still until the next prompt
     * @param activity paused while an approval question is open
     * @param handle stops the running turn when the user types instead of waiting
     * @return how the turn ended
     * @throws InterruptedException if interrupted while waiting
     */
    static TurnEnd awaitWithActivity(
            TurnRecorder session,
            AgentTerminal terminal,
            Supplier<String> stateLine,
            TurnActivity activity,
            ExecutionHandle handle)
            throws InterruptedException {
        long start = System.nanoTime();
        int frame = 0;
        // one word per turn, not per frame: a word that changes ten times a second is noise
        String word = spinnerWord();
        while (!session.await(ACTIVITY_INTERVAL)) {
            long seconds = (System.nanoTime() - start) / 1_000_000_000L;
            if (seconds > AgentSession.TURN_TIMEOUT.toSeconds()) {
                return TurnEnd.TIMED_OUT;
            }
            if (activity.isPaused()) {
                continue; // an approval question is waiting for its answer
            }
            if (terminal.hasPendingInput()) {
                // Something was typed while the agent was working. Stop the turn rather than finish a
                // request that has been overtaken: the handle closes the stream the model is answering on.
                // The line itself stays queued and becomes the next message, so what the model produced so
                // far is kept and the new instruction follows it.
                handle.cancel();
                terminal.line(terminal.ansi().yellow("(interrupted — taking your message)"));
                terminal.status(List.of(IDLE_LINE, stateLine.get()));
                return TurnEnd.INTERRUPTED;
            }
            terminal.status(List.of(
                    activityLine(
                            ACTIVITY_FRAMES.charAt(frame++ % ACTIVITY_FRAMES.length()),
                            word,
                            seconds,
                            session.runningTool(),
                            session.runningSeconds(),
                            session.toolCalls()),
                    stateLine.get()));
        }
        terminal.status(List.of(IDLE_LINE, stateLine.get()));
        return TurnEnd.FINISHED;
    }

    /**
     * What the pinned line says while a turn is running.
     *
     * <p>Naming the running tool is the point: a build or a test run can take minutes, and "working…"
     * during a two-minute {@code mvn test} is indistinguishable from a hang. When no tool runs, the model is
     * generating, which is its own kind of waiting.
     *
     * @param frame the spinner character
     * @param word the word of this turn
     * @param seconds how long the whole turn has been running
     * @param runningTool the tool executing right now, or {@code null}
     * @param toolSeconds how long that tool has been running
     * @param toolCalls how many tools ran in this turn so far
     * @return the line
     */
    static String activityLine(
            char frame, String word, long seconds, @Nullable String runningTool, long toolSeconds, int toolCalls) {
        String inside = runningTool == null ? seconds + "s" : runningTool + " " + toolSeconds + "s of " + seconds + "s";
        return frame + " " + word + "… (" + inside + (toolCalls == 0 ? "" : " · " + toolCalls + " tool calls") + ")";
    }

    /**
     * A word for the activity line, drawn once per turn.
     *
     * <p>Our own list ({@value #SPINNER_WORDS}), not the one Claude Code ships: that one is extracted from a
     * proprietary binary, and the public collections of it are either unlicensed or CC BY-NC-SA — neither is
     * compatible with this project's MIT licence or with REUSE. Edit the resource to change them; no Java
     * involved.
     *
     * @return one word, or {@code "Thinking"} when the list cannot be read
     */
    static String spinnerWord() {
        List<String> words = Prompts.prompt(SPINNER_WORDS)
                .lines()
                .map(String::strip)
                .filter(word -> !word.isEmpty())
                .toList();
        return words.isEmpty()
                ? "Thinking"
                : words.get(java.util.concurrent.ThreadLocalRandom.current().nextInt(words.size()));
    }

    /**
     * What this terminal is, and which of the JLine fixes the library on the classpath actually has.
     *
     * <p>Printed at startup because its absence cost several rounds of testing. Six fixes are carried
     * against JLine as a locally installed jar, selected by {@code -Djline.version=…}; a report from a
     * console then says nothing about <em>which</em> library produced it, and two of those rounds chased a
     * defect that the jar in use did not even contain the fix for. The version string in the jar's manifest
     * is no help — the patched builds overlay classes into the released jar and keep its version — so the
     * fixes are probed directly: {@code Status.repaint()} is the fifth and {@code Display.addressesEveryRow}
     * the sixth, and each exists only in a build that has the ones before it.
     *
     * @param console the terminal in use
     * @return a line naming the terminal type and the fixes present
     */
    static String describeTerminal(AgentTerminal console) {
        String type = console.terminalType();
        String fixes;
        if (!console.pinsStatus()) {
            fixes = "not used in this mode";
        } else if (hasJLineMethod("org.jline.utils.Display", "addressesEveryRow")) {
            fixes = "patched (row addressing + repaint)";
        } else if (hasJLineMethod("org.jline.utils.Status", "repaint")) {
            fixes = "patched (repaint only -- the block can still collapse on a resize)";
        } else {
            fixes = "RELEASED -- the pinned block can smear on a resize, /cls repairs it";
        }
        return "terminal: " + type + ", JLine: " + fixes;
    }

    /**
     * Whether a JLine class declares a method, however visible.
     *
     * @param className the class to look in
     * @param method the method to look for
     * @return whether it is there
     */
    private static boolean hasJLineMethod(String className, String method) {
        try {
            Class.forName(className).getDeclaredMethod(method);
            return true;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return false;
        }
    }
}
