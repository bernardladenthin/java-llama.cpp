// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.tool.ToolDefinition;
import org.jspecify.annotations.Nullable;

/**
 * A local, general-purpose terminal agent in the spirit of Claude Code / OpenCode, built from two parts that
 * already exist: <b>Atmosphere</b>'s built-in OpenAI-compatible agent runtime (streaming, tool loop,
 * workspace file tools) and <b>java-llama.cpp</b>'s OpenAI-compatible server.
 *
 * <p>Two ways to reach a model:
 *
 * <ul>
 *   <li>{@code --base-url http://127.0.0.1:8080/v1} — a server you started yourself (java-llama.cpp's
 *       fat jar {@code NativeServer} with {@code --jinja}, its {@code OpenAiCompatServer}, or upstream
 *       {@code llama-server}), so you keep full control over model parameters.
 *   <li>{@code --model model.gguf} — loads the GGUF in this JVM and serves it to the agent over a
 *       loopback {@link OpenAiCompatServer}: one process, one command.
 * </ul>
 *
 * <p>Run from the source tree: {@code mvn -q compile exec:java -Dexec.args="--base-url ... --workspace
 * /path --allow-shell"}. Exit code 0 on a completed turn, 1 when the turn errored, 2 on bad usage.
 */
public final class LocalAgent {

    /** Wall-clock bound on one user turn, including every tool round. */
    private static final Duration TURN_TIMEOUT = Duration.ofMinutes(30);

    /** The default system prompt; placeholders {@code {workspace}} and {@code {shell_section}}. */
    static final String SYSTEM_PROMPT = "system-prompt.txt";

    /** The {@code {shell_section}} with {@code --allow-shell}; placeholder {@code {shell}}. */
    static final String SHELL_PROMPT = "system-prompt-shell.txt";

    /** The {@code {shell_section}} without {@code --allow-shell}. */
    static final String NO_SHELL_PROMPT = "system-prompt-no-shell.txt";

    /** The {@code /help} overview. */
    static final String HELP_TEXT = "help.txt";

    /** The per-step message of {@code /loop}; placeholders {@code {task}}, {@code {file}}, {@code {check_hint}}. */
    static final String LOOP_PROMPT = "loop-prompt.txt";

    /** The skeleton written to {@link TaskLoop#LOOP_FILE}; placeholder {@code {task}}. */
    static final String LOOP_FILE_TEMPLATE = "loop-file-template.md";

    /** The instructions {@code /compact} sends; placeholder {@code {focus}}. */
    static final String COMPACT_PROMPT = "compact-prompt.txt";

    /** The system prompt of the summarizing turn: no tools, no agent role, just condense. */
    static final String COMPACT_SYSTEM_PROMPT =
            "You summarize a conversation between a user and a coding assistant. Follow the user's"
                    + " instructions exactly and answer with the summary only.";

    /** How the user message of a compacted history begins; also how a repeat compaction is detected. */
    static final String SUMMARY_PREFIX = "Summary of the conversation so far:";

    /** How much of a tool result is kept in the history of later turns. */
    private static final int HISTORY_RESULT_CHARS = 400;

    /** How often the activity line is refreshed while a turn runs. */
    private static final Duration ACTIVITY_INTERVAL = Duration.ofMillis(250);

    /** The usual rule of thumb, used wherever a token count has to be guessed from text. */
    private static final int CHARS_PER_TOKEN = 4;

    /** The spinner shown in the activity line. */
    private static final String ACTIVITY_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

    /** The first line of the block while nothing is running. */
    static final String IDLE_LINE = "… waiting for input …";

    /** The whimsical words the activity line picks from, one per turn. */
    static final String SPINNER_WORDS = "spinner-words.txt";

    private static final Duration SHELL_TIMEOUT = Duration.ofSeconds(120);
    private static final int SHELL_MAX_OUTPUT_CHARS = 20_000;

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
        System.exit(run(
                options,
                System.in == null ? null : new InputStreamReader(System.in, StandardCharsets.UTF_8),
                System.out,
                System.err));
    }

    /**
     * Run the agent with parsed options.
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
        LlamaModel model = null;
        OpenAiCompatServer server = null;
        AgentTerminal terminal = null;
        String baseUrl = options.getBaseUrl();
        try {
            if (options.getModelPath() != null) {
                err.println("Loading " + options.getModelPath() + " (gpu layers: " + options.getGpuLayers() + ", ctx: "
                        + options.getCtxSize() + ") ...");
                model = new LlamaModel(modelParameters(options));
                server = new OpenAiCompatServer(
                                model,
                                OpenAiServerConfig.builder()
                                        .host("127.0.0.1")
                                        .port(0)
                                        .apiKey(options.getApiKey())
                                        .modelId(options.getModelId())
                                        .build())
                        .start();
                baseUrl = "http://127.0.0.1:" + server.getPort() + "/v1";
            }
            if (baseUrl == null) {
                throw new IllegalStateException("no endpoint");
            }
            AgentFileSystem fileSystem =
                    new WorkspaceAgentFileSystem(options.getWorkspace(), AgentFileSystem.Limits.defaults());
            // Our own read_file/edit_file/grep replace the framework's (see WorkspaceTools); the
            // read tracker is what lets an edit insist the file was read first.
            List<ToolDefinition> tools = new ArrayList<>(WorkspaceTools.all(new WorkspaceTools.ReadTracker()));
            if (options.isAllowShell()) {
                // Live output: a two-minute build has to show that it is doing something.
                AgentTerminal console = terminal;
                tools.add(ShellTool.definition(
                        options.getWorkspace(),
                        SHELL_TIMEOUT,
                        SHELL_MAX_OUTPUT_CHARS,
                        line -> console.line(console.ansi().dim("  │ " + line))));
            }
            AgentRunner runner = new AgentRunner(
                    baseUrl,
                    options.getApiKey(),
                    options.getModelId(),
                    tools,
                    systemPrompt(options),
                    options.getTemperature(),
                    options.getMaxTokens(),
                    options.getMaxToolRounds());
            err.println("Endpoint " + baseUrl + " models=" + runner.models() + " workspace=" + options.getWorkspace()
                    + " tools=" + runner.toolNames());

            List<ChatMessage> history = new ArrayList<>();
            ToolCallLog callLog = new ToolCallLog();
            AtomicReference<ApprovalMode> mode =
                    new AtomicReference<>(options.isAuto() ? ApprovalMode.AUTO : ApprovalMode.MANUAL);
            boolean interactive = options.getPrompt() == null && input != null;
            BufferedReader reader = input == null ? null : new BufferedReader(input);
            terminal = interactive ? JLineTerminal.open(commandNames()) : null;
            if (terminal == null) {
                terminal = new PlainTerminal(out, reader, Ansi.detect());
            }
            // One-shot runs have nobody at the keyboard, so the strategy gets no console and denies
            // gated calls unless --auto was passed (see ConsoleApprovalStrategy).
            TurnActivity activity = new TurnActivity();
            runner.approval(
                    new ConsoleApprovalStrategy(mode, terminal, interactive, activity),
                    ConsoleApprovalStrategy.policy());
            int contextSize = options.getModelPath() != null
                    ? options.getCtxSize()
                    : ServerProps.contextSize(baseUrl, options.getApiKey());

            if (options.getPrompt() != null) {
                return turn(
                                                runner,
                                                fileSystem,
                                                options.getPrompt(),
                                                history,
                                                terminal,
                                                callLog,
                                                1,
                                                ignored -> "",
                                                activity)
                                        .failure()
                                == null
                        ? 0
                        : 1;
            }
            if (reader == null) {
                err.println("No interactive input available; pass --prompt <text>.");
                return 2;
            }
            // Held rather than kept in locals so the shift+tab shortcut, which fires from inside the
            // line reader, can re-render the status row with the numbers the last turn left behind.
            java.util.concurrent.atomic.AtomicLong inputTokens = new java.util.concurrent.atomic.AtomicLong();
            java.util.concurrent.atomic.AtomicBoolean estimated = new java.util.concurrent.atomic.AtomicBoolean();
            AgentTerminal console = terminal;
            java.util.function.Supplier<String> idleStatus = () -> StatusLine.render(
                    options.getWorkspace(),
                    mode.get(),
                    inputTokens.get(),
                    estimated.get(),
                    contextSize,
                    tools.size(),
                    options.getModelId());
            boolean shortcut = console.onCycleMode(() -> {
                mode.set(mode.get().next());
                if (console.pinsStatus()) {
                    console.status(List.of(IDLE_LINE, idleStatus.get()));
                } else {
                    console.line(console.ansi().dim(idleStatus.get()));
                }
            });
            err.println("Interactive mode: type a request, /help for the commands."
                    + (shortcut ? " shift+tab switches the approval mode." : ""));
            int turnNumber = 0;
            String pendingNote = "";
            while (true) {
                // Pinned to the bottom of the window on a real terminal; printed above the prompt on a
                // plain stream, where there is nothing to pin and a repeatedly refreshed line would
                // just fill a piped log.
                String status = StatusLine.render(
                        options.getWorkspace(),
                        mode.get(),
                        inputTokens.get(),
                        estimated.get(),
                        contextSize,
                        tools.size(),
                        options.getModelId());
                if (terminal.pinsStatus()) {
                    terminal.status(List.of(IDLE_LINE, status));
                } else {
                    terminal.line(terminal.ansi().dim(status));
                }
                terminal.separator();
                String line = terminal.readLine("you> ");
                if (line == null) {
                    return 0;
                }
                if (line.trim().isEmpty()) {
                    continue;
                }
                Optional<SlashCommands> command = SlashCommands.parse(line);
                if (command.isPresent()) {
                    if (command.get().command() == SlashCommands.Command.EXIT) {
                        return 0;
                    }
                    inputTokens.set(handleCommand(
                            command.get(),
                            runner,
                            fileSystem,
                            history,
                            mode,
                            options,
                            contextSize,
                            inputTokens.get(),
                            estimated.get(),
                            terminal,
                            callLog));
                    continue;
                }
                turnNumber++;
                // Compact BEFORE the request that would overflow, not after it: afterwards the
                // oversized request has already been sent, which is the one thing to avoid.
                if (needsCompaction(
                        options, contextSize, estimateTokens(systemPrompt(options), history) + line.length() / 4)) {
                    terminal.line(terminal.ansi().yellow("(context nearly full — compacting first)"));
                    inputTokens.set(compact(runner, fileSystem, history, "", terminal));
                    estimated.set(true);
                }
                // What the request carries before the model has answered anything; the turn adds to it.
                long baseTokens = estimateTokens(systemPrompt(options), history) + line.length() / CHARS_PER_TOKEN;
                ConsoleSession completed = turn(
                        runner,
                        fileSystem,
                        pendingNote.isEmpty()
                                ? line
                                : pendingNote + System.lineSeparator() + System.lineSeparator() + line,
                        history,
                        terminal,
                        callLog,
                        turnNumber,
                        running -> StatusLine.render(
                                options.getWorkspace(),
                                mode.get(),
                                liveTokens(baseTokens, running),
                                running.inputTokens() == 0,
                                contextSize,
                                tools.size(),
                                options.getModelId()),
                        activity);
                pendingNote = toolNote(completed.rounds());
                estimated.set(completed.inputTokens() == 0);
                inputTokens.set(
                        estimated.get() ? estimateTokens(systemPrompt(options), history) : completed.inputTokens());
            }
        } finally {
            if (terminal != null) {
                terminal.close();
            }
            if (server != null) {
                server.close();
            }
            if (model != null) {
                model.close();
            }
        }
    }

    /**
     * Run {@code /loop}: work on one task step by step until it is done.
     *
     * <p>Two things are settled before the first step. The loop needs the {@link ApprovalMode#AUTO}
     * mode — a run that asks before every write is not a loop, it is a conversation — so a manual
     * session is asked once and left alone if the answer is no. And the history is <em>not</em> the
     * loop's memory: {@link TaskLoop} drops it every step and keeps the state in a file, so nothing of
     * the current conversation is used or changed here.
     *
     * @param runner the runner
     * @param fileSystem the workspace filesystem
     * @param terminal the console
     * @param options the agent options, for the workspace
     * @param mode the approval mode, possibly switched to auto here
     * @param arguments everything after {@code /loop}
     * @param callLog records the steps' tool calls for {@code /calls}
     * @throws InterruptedException if interrupted while a step runs
     */
    private static void loop(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            AgentTerminal terminal,
            AgentOptions options,
            AtomicReference<ApprovalMode> mode,
            String arguments,
            ToolCallLog callLog)
            throws InterruptedException {
        LoopOptions loopOptions;
        try {
            loopOptions = LoopOptions.parse(arguments);
        } catch (IllegalArgumentException e) {
            terminal.line(e.getMessage());
            return;
        }
        if (mode.get() != ApprovalMode.AUTO) {
            String answer =
                    terminal.readKey("a loop cannot stop at every question — switch to auto for it? [y]es / [n]o: ");
            if (answer == null || !(answer.startsWith("y") || answer.isEmpty())) {
                terminal.line("loop: cancelled (use /mode auto to allow it)");
                return;
            }
            mode.set(ApprovalMode.AUTO);
        }
        if (!TaskLoop.canKeepNotes(runner.toolNames())) {
            terminal.line("loop: needs the read_file and write_file tools to keep its notes");
            return;
        }
        TaskLoop.Outcome outcome = TaskLoop.run(
                runner,
                fileSystem,
                terminal,
                options.getWorkspace(),
                loopOptions,
                () -> false,
                TaskLoop.DEFAULT_BUDGET,
                callLog);
        terminal.line(
                outcome.completed()
                        ? terminal.ansi().green("loop: " + outcome.reason())
                        : terminal.ansi().yellow("loop: " + outcome.reason()));
    }

    static ConsoleSession turn(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            String message,
            List<ChatMessage> history,
            AgentTerminal terminal,
            ToolCallLog callLog,
            int turnNumber,
            java.util.function.Function<ConsoleSession, String> stateLine,
            TurnActivity activity)
            throws InterruptedException {
        ConsoleSession session = new ConsoleSession(terminal, fileSystem);
        // The turn runs on a thread of Atmosphere's, not this one: execute() blocks until the whole
        // turn including every tool round is done, which would leave nobody to drive the activity
        // line -- the block sat on "… waiting for input …" for entire turns until this changed.
        // start() adds the half that makes the always-present prompt worth having: the handle can
        // close the stream the model is answering on, so a request typed mid-turn takes effect now.
        org.atmosphere.ai.ExecutionHandle handle;
        try {
            handle = runner.start(message, history, session);
        } catch (RuntimeException e) {
            session.error(e);
            handle = org.atmosphere.ai.ExecutionHandle.completed();
        }
        TurnEnd end = awaitWithActivity(session, terminal, () -> stateLine.apply(session), activity, handle);
        history.add(ChatMessage.user(message));
        callLog.add(turnNumber, session.rounds());
        if (!session.text().isEmpty()) {
            history.add(ChatMessage.assistant(session.text()));
        }
        if (end == TurnEnd.TIMED_OUT) {
            session.error(new IllegalStateException("turn did not finish within " + TURN_TIMEOUT));
        }
        return session;
    }

    /**
     * Put this turn's tool calls in front of its answer, so the next turn can see they happened.
     *
     * <p><b>Why this matters more than it looks.</b> Without it the history holds the user's messages
     * and the model's prose, and nothing else — so from the third or fourth turn on, a small model
     * sees only its own paragraphs and no evidence that it ever used a tool. It then continues that
     * pattern: it <em>describes</em> creating a file and running a build, reports an exit code, and
     * writes nothing at all. That is not hypothetical; it happened on a real session, with the model
     * inventing test results and a jar that never existed.
     *
     * <p><b>Where it goes, and why not somewhere more obvious.</b> In front of the <em>next user
     * message</em>. The first attempt put it in front of the assistant's own answer, and the model
     * promptly copied it into its next reply — the user read "(tools I actually ran this turn: …)" as
     * the first line of an answer, because text attributed to the assistant is text a model imitates.
     * A system message mid-history would be cleaner still, but not every chat template accepts one:
     * Mistral's requires strict user/assistant alternation and Gemma has no system role at all.
     * Riding along with the next user message keeps the sequence template-safe everywhere.
     *
     * <p><b>Why a note rather than real {@code tool_calls} messages.</b> Atmosphere's
     * {@code AbstractAgentRuntime.assembleMessages} rebuilds every history entry as
     * {@code new ChatMessage(role, content)} — the tool-call array and the tool-call id are dropped on
     * the way out. Protocol-faithful replay is therefore impossible through the framework's history;
     * what survives is the content, so the evidence goes there. It is also cheaper: one line per call
     * instead of a message pair, with the result cut to {@value #HISTORY_RESULT_CHARS} characters.
     *
     * @param rounds the tool calls of the finished turn
     * @return the note, or an empty string when no tool ran
     */
    static String toolNote(List<ConsoleSession.ToolRound> rounds) {
        if (rounds.isEmpty()) {
            return "";
        }
        StringBuilder note = new StringBuilder("Record of the tools that actually ran in the previous turn."
                + " This is a log for your reference; do not repeat it and do not mention it.");
        for (ConsoleSession.ToolRound round : rounds) {
            String result = round.result().replace("\r\n", " ").replace('\n', ' ');
            note.append(System.lineSeparator())
                    .append("- ")
                    .append(round.name())
                    .append(" ")
                    .append(round.argumentsJson())
                    .append(" -> ")
                    .append(
                            result.length() <= HISTORY_RESULT_CHARS
                                    ? result
                                    : result.substring(0, HISTORY_RESULT_CHARS) + " …[cut]");
        }
        return note.toString();
    }

    /**
     * Run one REPL command.
     *
     * @param command the parsed command
     * @param runner the runner (used by {@code /compact})
     * @param fileSystem the workspace filesystem (used by {@code /compact}'s session)
     * @param history the conversation history, modified in place by {@code /clear} and {@code /compact}
     * @param mode the approval mode, modified in place by {@code /mode}
     * @param options the options, for the status output
     * @param contextSize the context window in tokens, or {@link StatusLine#UNKNOWN_CONTEXT}
     * @param inputTokens the input tokens of the last turn
     * @param estimated whether that number is an estimate
     * @param terminal the console
     * @param callLog every tool call of the session, for {@code /calls}
     * @return the input tokens to show from now on (unchanged, or the summary's after {@code /compact})
     * @throws InterruptedException if interrupted while a summary is generated
     */
    private static long handleCommand(
            SlashCommands command,
            AgentRunner runner,
            AgentFileSystem fileSystem,
            List<ChatMessage> history,
            AtomicReference<ApprovalMode> mode,
            AgentOptions options,
            int contextSize,
            long inputTokens,
            boolean estimated,
            AgentTerminal terminal,
            ToolCallLog callLog)
            throws InterruptedException {
        switch (command.command()) {
            case HELP -> prompt(HELP_TEXT).lines().forEach(terminal::line);
            case CLEAR -> {
                history.clear();
                terminal.line("(history cleared)");
            }
            case CALLS -> callLog.render().lines().forEach(terminal::line);
            case TOOLS -> {
                terminal.line("tools: " + String.join(", ", runner.toolNames()));
                terminal.line("asks before running (manual mode): "
                        + String.join(", ", ConsoleApprovalStrategy.gated(runner.toolNames())));
            }
            case MODE -> {
                if (command.hasArguments()) {
                    try {
                        mode.set(ApprovalMode.parse(command.arguments()));
                    } catch (IllegalArgumentException e) {
                        terminal.line(e.getMessage());
                        return inputTokens;
                    }
                }
                terminal.line("approval mode: " + mode.get().badge());
            }
            case STATUS -> {
                terminal.line(StatusLine.render(
                        options.getWorkspace(),
                        mode.get(),
                        inputTokens,
                        estimated,
                        contextSize,
                        runner.toolNames().size(),
                        options.getModelId()));
                terminal.line("workspace: " + options.getWorkspace());
                terminal.line("history: " + history.size() + " messages");
            }
            case COMPACT -> {
                return compact(runner, fileSystem, history, command.arguments(), terminal);
            }
            case LOOP -> loop(runner, fileSystem, terminal, options, mode, command.arguments(), callLog);
            case EXIT -> {
                // handled by the caller, which has to return from the loop
            }
        }
        return inputTokens;
    }

    /**
     * Wait for a turn while the status line shows that something is happening.
     *
     * <p>A local model can think for a while before the first token arrives, and a silent console is
     * indistinguishable from a hung one. The line is rewritten in place (it is the pinned status area,
     * not the scrollback), so nothing the user has already read moves.
     *
     * @param session the running turn
     * @param terminal the console
     * @param stateLine the second row of the block, asked again on every redraw so the context figure
     *     moves while the turn runs rather than standing still until the next prompt
     * @param activity paused while an approval question is open
     * @param handle stops the running turn when the user types instead of waiting
     * @return how the turn ended
     * @throws InterruptedException if interrupted while waiting
     */
    /** How a turn ended: on its own, because the user typed something, or because it ran too long. */
    enum TurnEnd {
        /** The model produced its final answer (or errored). */
        FINISHED,
        /** The user typed while it was working; the turn was cut short and that line is next. */
        INTERRUPTED,
        /** Nothing arrived within {@link #TURN_TIMEOUT}. */
        TIMED_OUT
    }

    static TurnEnd awaitWithActivity(
            ConsoleSession session,
            AgentTerminal terminal,
            java.util.function.Supplier<String> stateLine,
            TurnActivity activity,
            org.atmosphere.ai.ExecutionHandle handle)
            throws InterruptedException {
        long start = System.nanoTime();
        int frame = 0;
        // one word per turn, not per frame: a word that changes ten times a second is noise
        String word = spinnerWord();
        while (!session.await(ACTIVITY_INTERVAL)) {
            long seconds = (System.nanoTime() - start) / 1_000_000_000L;
            if (seconds > TURN_TIMEOUT.toSeconds()) {
                return TurnEnd.TIMED_OUT;
            }
            if (activity.isPaused()) {
                continue; // an approval question is waiting for its answer
            }
            if (terminal.hasPendingInput()) {
                // Something was typed while the agent was working. Stop the turn rather than finish a
                // request that has been overtaken: the handle closes the stream the model is answering
                // on. The line itself stays queued and becomes the next message, so what the model
                // produced so far is kept and the new instruction follows it.
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
     * <p>Naming the running tool is the point: a build or a test run can take minutes, and
     * "working…" during a two-minute {@code mvn test} is indistinguishable from a hang. When no tool
     * runs, the model is generating, which is its own kind of waiting.
     *
     * @param frame the spinner character
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
     * <p>Our own list ({@value #SPINNER_WORDS}), not the one Claude Code ships: that one is extracted
     * from a proprietary binary, and the public collections of it are either unlicensed or
     * CC BY-NC-SA — neither is compatible with this project's MIT licence or with REUSE. Edit the
     * resource to change them; no Java involved.
     *
     * @return one word, or {@code "Thinking"} when the list cannot be read
     */
    static String spinnerWord() {
        List<String> words = prompt(SPINNER_WORDS)
                .lines()
                .map(String::strip)
                .filter(word -> !word.isEmpty())
                .toList();
        return words.isEmpty()
                ? "Thinking"
                : words.get(java.util.concurrent.ThreadLocalRandom.current().nextInt(words.size()));
    }

    /**
     * Whether the history has to be summarized before the next request is sent.
     *
     * <p>The threshold is on the low side ({@link AgentOptions#DEFAULT_COMPACT_AT} %) because the
     * number it is compared against is usually an estimate, and because the model's reply has to fit
     * next to the prompt. With an unknown context size — a foreign endpoint whose {@code /props} says
     * nothing — nothing is decided at all rather than guessed.
     *
     * @param options the options, for the switch and the threshold
     * @param contextSize the context window, or {@link StatusLine#UNKNOWN_CONTEXT}
     * @param estimatedTokens what the next request is expected to carry
     * @return {@code true} when the history should be summarized first
     */
    static boolean needsCompaction(AgentOptions options, int contextSize, long estimatedTokens) {
        if (!options.isAutoCompact() || contextSize <= StatusLine.UNKNOWN_CONTEXT) {
            return false;
        }
        return estimatedTokens * 100 >= (long) contextSize * options.getCompactAt();
    }

    /**
     * Whether the history is the untouched result of a compaction.
     *
     * @param history the conversation
     * @return {@code true} when it is exactly the summary and its acknowledgement
     */
    static boolean isCompacted(List<ChatMessage> history) {
        return history.size() == 2
                && history.get(0).content() != null
                && history.get(0).content().startsWith(SUMMARY_PREFIX);
    }

    /**
     * A rough token count of what the next request will carry.
     *
     * <p>Used only for the status line, and only because llama.cpp sends its own count just to clients
     * that ask for it ({@code stream_options.include_usage}), which Atmosphere's client does not. Four
     * characters per token is the usual rule of thumb; the status line marks the number with a
     * {@code ~} so nobody reads it as exact.
     *
     * @param systemPrompt the system prompt sent with every request
     * @param history the conversation so far
     * @return the estimated token count
     */
    static long estimateTokens(String systemPrompt, List<ChatMessage> history) {
        long characters = systemPrompt.length();
        for (ChatMessage message : history) {
            characters += message.content() == null ? 0 : message.content().length();
        }
        return characters / CHARS_PER_TOKEN;
    }

    /**
     * The context figure while a turn is running.
     *
     * <p>The count shown before the turn started is not the count during it: every tool round appends
     * the call and its output to the conversation the next model call of the same turn is sent, so a
     * turn that reads three files and runs a build can add thousands of tokens before the prompt comes
     * back. The status row used to be rendered once and handed to the redraw loop as a fixed string, so
     * it stood still for the whole turn and only moved at the next {@code you>} — which is precisely
     * when it no longer matters.
     *
     * @param baseTokens what the request carried when it was sent
     * @param session the running turn
     * @return the server's own count once it reported one, else the base plus what the turn produced
     */
    static long liveTokens(long baseTokens, ConsoleSession session) {
        long reported = session.inputTokens();
        return reported > 0 ? reported : baseTokens + session.producedChars() / CHARS_PER_TOKEN;
    }

    /**
     * Summarize the history and continue from the summary.
     *
     * <p>The summary is generated by the same model with no tools, then <b>replaces</b> the history as
     * a {@code user} message plus a short assistant acknowledgement — aider's shape, and the one that
     * survives a strict chat template, because a conversation may not start with two assistant turns.
     * The tool rounds of a turn are not in the history to begin with (only the user text and the final
     * answer are), so what is condensed here is what the next turn would have replayed anyway.
     *
     * @param runner the runner
     * @param fileSystem the workspace filesystem for the summary session
     * @param history the history, replaced in place
     * @param focus optional extra instructions from {@code /compact <focus>}
     * @param terminal the console
     * @return the input tokens the summarizing call reported
     * @throws InterruptedException if interrupted while the summary is generated
     */
    private static long compact(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            List<ChatMessage> history,
            String focus,
            AgentTerminal terminal)
            throws InterruptedException {
        if (history.isEmpty()) {
            terminal.line("(nothing to compact)");
            return 0;
        }
        if (isCompacted(history)) {
            // After a compaction the history IS the summary plus its acknowledgement. Summarizing that
            // again returns the same text for another model call -- and re-sends a byte-identical
            // prompt, which is what makes llama.cpp log "need to evaluate at least 1 token".
            terminal.line("(the history is already a summary — nothing to compact)");
            return estimateTokens("", history);
        }
        String instructions = prompt(COMPACT_PROMPT)
                .replace("{focus}", focus.isEmpty() ? "" : System.lineSeparator() + "Focus on: " + focus);
        int before = history.size();
        terminal.line("(compacting " + before + " messages …)");
        ConsoleSession session = new ConsoleSession(terminal, fileSystem);
        Thread worker = new Thread(
                () -> {
                    try {
                        runner.runWithoutTools(instructions, List.copyOf(history), session, COMPACT_SYSTEM_PROMPT);
                    } catch (RuntimeException e) {
                        session.error(e);
                    }
                },
                "agent-compact");
        worker.setDaemon(true);
        worker.start();
        if (awaitWithActivity(
                                session,
                                terminal,
                                () -> "/compact",
                                new TurnActivity(),
                                org.atmosphere.ai.ExecutionHandle.completed())
                        != TurnEnd.FINISHED
                || session.text().isBlank()) {
            terminal.line("(compact failed; history kept)");
            return 0;
        }
        history.clear();
        history.add(ChatMessage.user(
                SUMMARY_PREFIX + System.lineSeparator() + session.text().strip()));
        history.add(ChatMessage.assistant("Understood, I will continue from that summary."));
        terminal.line("(compacted " + before + " messages into a summary of "
                + session.text().strip().length() + " characters)");
        return session.inputTokens();
    }

    /**
     * The native parameters for {@code --model}.
     *
     * <p>Visible for tests: the log threshold is the one knob whose effect is only observable on a
     * console, so the test pins the flags that leave here instead.
     *
     * @param options the parsed options
     * @return the parameters the in-process {@link LlamaModel} is loaded with
     */
    static ModelParameters modelParameters(AgentOptions options) {
        ModelParameters parameters = new ModelParameters()
                .setModel(options.getModelPath())
                .setCtxSize(options.getCtxSize())
                .setGpuLayers(options.getGpuLayers())
                .setFit(false)
                // Jinja rendering is what lets the native parser apply the model's tool-call template.
                .enableJinja();
        // llama.cpp logs to stderr, which shares the console with the streamed answer on stdout; the
        // default threshold keeps warnings and errors and drops the per-request INFO lines.
        if (options.isVerbose()) {
            parameters.setVerbose();
        } else {
            parameters.setLogVerbosity(options.getLogVerbosity());
        }
        if (options.getGpuLayers() == 0) {
            parameters.setDevices("none");
        }
        return parameters;
    }

    /**
     * Every command name and alias, for tab completion.
     *
     * @return the names, each with its leading slash
     */
    static List<String> commandNames() {
        return java.util.Arrays.stream(SlashCommands.Command.values())
                .flatMap(command -> command.names().stream())
                .toList();
    }

    /**
     * The default system prompt, or the {@code --system} override.
     *
     * <p>The default describes a general-purpose agent on this machine, not a coding agent confined to a
     * project: a small model reads a narrow role or tool description as a prohibition and then refuses
     * requests such as "list the docker images" even though {@code run_command} could do it. With
     * {@code --allow-shell} the prompt therefore states that any command line is allowed and that the
     * model should run a command rather than explain one; without it, the prompt says so honestly
     * instead of letting the model invent a limitation. The text itself is in the resources
     * {@value #SYSTEM_PROMPT}, {@value #SHELL_PROMPT} and {@value #NO_SHELL_PROMPT} (see {@link #prompt}).
     *
     * @param options the options
     * @return the system prompt
     */
    static String systemPrompt(AgentOptions options) {
        if (options.getSystemPrompt() != null) {
            return options.getSystemPrompt();
        }
        String shellSection = options.isAllowShell()
                ? prompt(SHELL_PROMPT).replace("{shell}", ShellTool.shellName())
                : prompt(NO_SHELL_PROMPT);
        return prompt(SYSTEM_PROMPT)
                .replace("{workspace}", options.getWorkspace().toString())
                .replace("{shell_section}", shellSection);
    }

    /**
     * A prompt text from the resources next to this class, trimmed.
     *
     * <p>The wording lives in {@code src/main/resources/net/ladenthin/llama/atmosphere/*.txt} so it can
     * be read and edited as text; {@code {placeholders}} are filled in by {@link #systemPrompt}.
     *
     * @param name the file name, e.g. {@value #SYSTEM_PROMPT}
     * @return the file content without leading or trailing whitespace
     * @throws IllegalStateException when the resource is missing from the jar
     */
    static String prompt(String name) {
        try (InputStream in = LocalAgent.class.getResourceAsStream(name)) {
            if (in == null) {
                throw new IllegalStateException("Prompt resource missing: " + name);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8).strip();
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read prompt resource " + name, e);
        }
    }
}
