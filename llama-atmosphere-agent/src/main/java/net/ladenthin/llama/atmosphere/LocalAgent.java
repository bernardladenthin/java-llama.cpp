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

    /** How often the activity line is refreshed while a turn runs. */
    private static final Duration ACTIVITY_INTERVAL = Duration.ofMillis(250);

    /** The spinner shown in the activity line. */
    private static final String ACTIVITY_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

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
                tools.add(ShellTool.definition(options.getWorkspace(), SHELL_TIMEOUT, SHELL_MAX_OUTPUT_CHARS));
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
            runner.approval(new ConsoleApprovalStrategy(mode, terminal, interactive), ConsoleApprovalStrategy.policy());
            int contextSize = options.getModelPath() != null
                    ? options.getCtxSize()
                    : ServerProps.contextSize(baseUrl, options.getApiKey());

            if (options.getPrompt() != null) {
                return turn(runner, fileSystem, options.getPrompt(), history, terminal)
                                        .failure()
                                == null
                        ? 0
                        : 1;
            }
            if (reader == null) {
                err.println("No interactive input available; pass --prompt <text>.");
                return 2;
            }
            err.println("Interactive mode: type a request, /help for the commands.");
            long inputTokens = 0;
            boolean estimated = false;
            while (true) {
                terminal.status(StatusLine.render(
                        options.getWorkspace(),
                        mode.get(),
                        inputTokens,
                        estimated,
                        contextSize,
                        tools.size(),
                        options.getModelId()));
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
                    inputTokens = handleCommand(
                            command.get(),
                            runner,
                            fileSystem,
                            history,
                            mode,
                            options,
                            contextSize,
                            inputTokens,
                            estimated,
                            terminal);
                    continue;
                }
                ConsoleSession completed = turn(runner, fileSystem, line, history, terminal);
                estimated = completed.inputTokens() == 0;
                inputTokens = estimated ? estimateTokens(systemPrompt(options), history) : completed.inputTokens();
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
     * @throws InterruptedException if interrupted while a step runs
     */
    private static void loop(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            AgentTerminal terminal,
            AgentOptions options,
            AtomicReference<ApprovalMode> mode,
            String arguments)
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
                TaskLoop.DEFAULT_BUDGET);
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
            AgentTerminal terminal)
            throws InterruptedException {
        ConsoleSession session = new ConsoleSession(terminal, fileSystem);
        runner.run(message, history, session);
        boolean finished = awaitWithActivity(session, terminal);
        history.add(ChatMessage.user(message));
        if (!session.text().isEmpty()) {
            history.add(ChatMessage.assistant(session.text()));
        }
        if (!finished) {
            session.error(new IllegalStateException("turn did not finish within " + TURN_TIMEOUT));
        }
        return session;
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
            AgentTerminal terminal)
            throws InterruptedException {
        switch (command.command()) {
            case HELP -> prompt(HELP_TEXT).lines().forEach(terminal::line);
            case CLEAR -> {
                history.clear();
                terminal.line("(history cleared)");
            }
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
                terminal.line("approval mode: " + mode.get().label());
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
            case LOOP -> loop(runner, fileSystem, terminal, options, mode, command.arguments());
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
     * @return {@code true} when the turn finished within {@link #TURN_TIMEOUT}
     * @throws InterruptedException if interrupted while waiting
     */
    private static boolean awaitWithActivity(ConsoleSession session, AgentTerminal terminal)
            throws InterruptedException {
        long start = System.nanoTime();
        int frame = 0;
        while (!session.await(ACTIVITY_INTERVAL)) {
            long seconds = (System.nanoTime() - start) / 1_000_000_000L;
            if (seconds > TURN_TIMEOUT.toSeconds()) {
                return false;
            }
            terminal.status(ACTIVITY_FRAMES.charAt(frame++ % ACTIVITY_FRAMES.length()) + " working… (" + seconds
                    + "s · " + session.toolCalls() + " tool calls)");
        }
        terminal.status("");
        return true;
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
        return characters / 4;
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
        String instructions = prompt(COMPACT_PROMPT)
                .replace("{focus}", focus.isEmpty() ? "" : System.lineSeparator() + "Focus on: " + focus);
        int before = history.size();
        terminal.line("(compacting " + before + " messages …)");
        ConsoleSession session = new ConsoleSession(terminal, fileSystem);
        runner.runWithoutTools(instructions, List.copyOf(history), session, COMPACT_SYSTEM_PROMPT);
        if (!awaitWithActivity(session, terminal) || session.text().isBlank()) {
            terminal.line("(compact failed; history kept)");
            return 0;
        }
        history.clear();
        history.add(ChatMessage.user("Summary of the conversation so far:" + System.lineSeparator()
                + session.text().strip()));
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
