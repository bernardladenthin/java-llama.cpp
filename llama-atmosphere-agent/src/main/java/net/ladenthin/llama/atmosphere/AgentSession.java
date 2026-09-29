// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Function;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.tool.ToolDefinition;
import org.jspecify.annotations.Nullable;

/**
 * One conversation with the agent, independent of how it is shown.
 *
 * <p>Everything a conversation <em>is</em> lives here: the history the model is sent, the approval mode,
 * the timestamped record ({@link Transcript}), the log of tool calls, the token figures, the tools and the
 * slash commands. What differs between a console, a browser tab and an editor is only how that is shown
 * and who answers questions — the {@link SessionFrontend} each call brings along. That split is what lets
 * the same conversation be driven from a terminal, over a local port, or by an IDE, without a copy of the
 * logic per front end.
 *
 * <p><b>One request at a time.</b> {@link #submit} holds a lock for the whole request, because a history
 * cannot take two turns at once. A front end that lets the user send something while a turn is running —
 * a browser, an editor — calls {@link #cancel()} first; the running request then ends early and the lock is
 * free for the next one.
 */
public final class AgentSession {

    /** Wall-clock bound on one user turn, including every tool round. */
    public static final Duration TURN_TIMEOUT = Duration.ofMinutes(30);

    /** The system prompt of the summarizing turn: no tools, no agent role, just condense. */
    static final String COMPACT_SYSTEM_PROMPT = "You summarize a conversation between a user and a coding assistant."
            + " Follow the user's instructions exactly and answer with the summary only.";

    /** How the user message of a compacted history begins; also how a repeat compaction is detected. */
    static final String SUMMARY_PREFIX = "Summary of the conversation so far:";

    /** How long a command may run when the model does not say. */
    static final Duration SHELL_TIMEOUT = Duration.ofSeconds(120);

    /** How much of a command's output the model gets. */
    static final int SHELL_MAX_OUTPUT_CHARS = 20_000;

    /** How much of a tool result is kept in the history of later turns. */
    private static final int HISTORY_RESULT_CHARS = 400;

    /** The usual rule of thumb, used wherever a token count has to be guessed from text. */
    private static final int CHARS_PER_TOKEN = 4;

    /** How often a quiet wait looks whether the turn was stopped. */
    private static final Duration POLL_INTERVAL = Duration.ofMillis(100);

    /** What {@link #submit} did with a line. */
    public enum Result {
        /** The line was handled; the session goes on. */
        CONTINUE,
        /** The line asked to end the session ({@code /exit}). */
        EXIT
    }

    private final AgentOptions options;
    private final AgentRunner runner;
    private final AgentFileSystem fileSystem;
    private final String systemPrompt;
    private final int contextSize;
    private final AtomicReference<@Nullable SessionFrontend> current;
    private final List<ChatMessage> history = new ArrayList<>();
    private final ToolCallLog callLog = new ToolCallLog();
    private final Transcript transcript;
    private final AtomicReference<ApprovalMode> mode;
    private final AtomicLong inputTokens = new AtomicLong();
    private final AtomicBoolean estimated = new AtomicBoolean();
    private final ReentrantLock busy = new ReentrantLock();
    private volatile boolean stopRequested;
    private volatile @Nullable ExecutionHandle running;
    private int turnNumber;
    private String pendingNote = "";
    private String lastMessage = "";

    /**
     * Wire a session to an endpoint: the tools, the runner and the approval gate.
     *
     * @param options the parsed command line
     * @param baseUrl the OpenAI-compatible endpoint the model is served on
     * @param contextSize the context window in tokens, or {@link StatusLine#UNKNOWN_CONTEXT}
     * @return the session
     */
    public static AgentSession open(AgentOptions options, String baseUrl, int contextSize) {
        // The front end of the request that is running, looked up when a command prints a line — the
        // tools are built before any front end exists, so they must not capture one.
        AtomicReference<@Nullable SessionFrontend> current = new AtomicReference<>();
        List<ToolDefinition> tools = tools(options, line -> {
            SessionFrontend frontend = current.get();
            if (frontend != null) {
                frontend.commandOutput(line);
            }
        });
        AgentRunner runner = new AgentRunner(
                baseUrl,
                options.getApiKey(),
                options.getModelId(),
                tools,
                Prompts.systemPrompt(options),
                options.getTemperature(),
                options.getMaxTokens(),
                options.getMaxToolRounds());
        return new AgentSession(options, runner, contextSize, current);
    }

    /**
     * The tools a session offers: the workspace file tools, plus {@code run_command} with
     * {@code --allow-shell}.
     *
     * @param options the parsed command line
     * @param commandOutput receives each line of a running command's output
     * @return the tool definitions
     */
    static List<ToolDefinition> tools(AgentOptions options, Consumer<String> commandOutput) {
        // Our own read_file/edit_file/grep replace the framework's (see WorkspaceTools); the read tracker is
        // what lets an edit insist the file was read first.
        List<ToolDefinition> tools = new ArrayList<>(WorkspaceTools.all(new WorkspaceTools.ReadTracker()));
        if (options.isAllowShell()) {
            // Live output: a two-minute build has to show that it is doing something.
            tools.add(
                    ShellTool.definition(options.getWorkspace(), SHELL_TIMEOUT, SHELL_MAX_OUTPUT_CHARS, commandOutput));
        }
        return tools;
    }

    AgentSession(
            AgentOptions options,
            AgentRunner runner,
            int contextSize,
            AtomicReference<@Nullable SessionFrontend> current) {
        this.options = options;
        this.runner = runner;
        this.fileSystem = new WorkspaceAgentFileSystem(options.getWorkspace(), AgentFileSystem.Limits.defaults());
        this.systemPrompt = Prompts.systemPrompt(options);
        this.contextSize = contextSize;
        this.current = current;
        this.transcript = new Transcript(options.getTranscript());
        this.mode = new AtomicReference<>(options.isAuto() ? ApprovalMode.AUTO : ApprovalMode.MANUAL);
        runner.approval(
                new ModeGatedApprovalStrategy(mode, () -> {
                    SessionFrontend frontend = current.get();
                    return frontend == null ? null : frontend.approvals();
                }),
                ConsoleApprovalStrategy.policy());
    }

    /**
     * Handle one line the user sent: a slash command, or a message for the model.
     *
     * <p>An unknown {@code /command} is a message, so a path such as {@code /usr/bin/java} needs no escape.
     *
     * @param line what was typed
     * @param frontend who shows the result and answers questions
     * @return whether the session goes on
     * @throws InterruptedException if interrupted while a turn runs
     */
    public Result submit(String line, SessionFrontend frontend) throws InterruptedException {
        if (line.isBlank()) {
            return Result.CONTINUE;
        }
        return withFrontend(frontend, () -> {
            Optional<SlashCommands> command = SlashCommands.parse(line);
            if (command.isEmpty()) {
                message(line, false, frontend);
                return Result.CONTINUE;
            }
            switch (command.get().command()) {
                case EXIT -> {
                    return Result.EXIT;
                }
                case RETRY -> {
                    if (lastMessage.isEmpty()) {
                        frontend.line("nothing to retry yet");
                    } else {
                        dropLastExchange(history, lastMessage);
                        transcript.add(Transcript.Kind.NOTE, "retrying: " + lastMessage);
                        message(lastMessage, true, frontend);
                    }
                }
                default -> command(command.get(), frontend);
            }
            return Result.CONTINUE;
        });
    }

    /**
     * Send one message to the model, as it is — no command parsing. What a one-shot run does.
     *
     * @param message the message
     * @param frontend who shows the turn
     * @return the finished turn
     * @throws InterruptedException if interrupted while the turn runs
     */
    public TurnRecorder send(String message, SessionFrontend frontend) throws InterruptedException {
        return withFrontend(frontend, () -> message(message, false, frontend));
    }

    /**
     * Stop whatever this session is doing: the running turn is cut short, a running {@code /loop} ends
     * after its step. Safe to call from any thread and when nothing is running.
     */
    public void cancel() {
        stopRequested = true;
        ExecutionHandle handle = running;
        if (handle != null) {
            handle.cancel();
        }
    }

    /**
     * Whether a request is being handled right now.
     *
     * @return {@code true} while {@link #submit} or {@link #send} runs
     */
    public boolean isBusy() {
        return busy.isLocked();
    }

    /**
     * Whether {@link #cancel()} was called since the current request started.
     *
     * @return {@code true} when the running request should end early
     */
    public boolean stopRequested() {
        return stopRequested;
    }

    /** One request, with its front end registered for the tools and the approval gate. */
    private <T> T withFrontend(SessionFrontend frontend, Request<T> request) throws InterruptedException {
        busy.lockInterruptibly();
        try {
            stopRequested = false;
            current.set(frontend);
            return request.run();
        } finally {
            current.set(null);
            busy.unlock();
        }
    }

    @FunctionalInterface
    private interface Request<T> {
        T run() throws InterruptedException;
    }

    private TurnRecorder message(String line, boolean retrying, SessionFrontend frontend) throws InterruptedException {
        // A retry repeats the question on purpose; the record already notes it, so it is not written twice.
        if (!retrying) {
            transcript.add(Transcript.Kind.USER, line);
        }
        lastMessage = line;
        turnNumber++;
        // Compact BEFORE the request that would overflow, not after it: afterwards the oversized request has
        // already been sent, which is the one thing to avoid.
        if (needsCompaction(options, contextSize, estimateTokens(systemPrompt, history) + line.length() / 4)) {
            frontend.line(frontend.ansi().yellow("(context nearly full — compacting first)"));
            inputTokens.set(compact("", frontend));
            estimated.set(true);
        }
        // What the request carries before the model has answered anything; the turn adds to it.
        long baseTokens = estimateTokens(systemPrompt, history) + line.length() / CHARS_PER_TOKEN;
        TurnRecorder completed = runTurn(
                runner,
                fileSystem,
                pendingNote.isEmpty() ? line : pendingNote + System.lineSeparator() + System.lineSeparator() + line,
                history,
                frontend,
                callLog,
                turnNumber,
                turn -> status(liveTokens(baseTokens, turn), turn.inputTokens() == 0),
                handle -> running = handle);
        running = null;
        for (TurnRecorder.ToolRound round : completed.rounds()) {
            transcript.add(Transcript.Kind.TOOL, round.name() + " " + round.argumentsJson() + " -> " + round.result());
        }
        transcript.add(Transcript.Kind.AGENT, completed.text());
        pendingNote = toolNote(completed.rounds());
        estimated.set(completed.inputTokens() == 0);
        inputTokens.set(estimated.get() ? estimateTokens(systemPrompt, history) : completed.inputTokens());
        return completed;
    }

    /**
     * Run one turn and add it to the history.
     *
     * @param runner the runner
     * @param fileSystem the workspace filesystem the tools use
     * @param message the message sent, including any tool note
     * @param history the conversation, extended by the turn
     * @param frontend who shows the turn and decides when to stop waiting
     * @param callLog records the turn's tool calls
     * @param turnNumber the turn's number, for the log
     * @param stateLine the session's state line while the turn runs
     * @param started receives the handle as soon as the turn is running, so it can be stopped from outside
     * @return the finished turn
     * @throws InterruptedException if interrupted while waiting
     */
    static TurnRecorder runTurn(
            AgentRunner runner,
            AgentFileSystem fileSystem,
            String message,
            List<ChatMessage> history,
            SessionFrontend frontend,
            ToolCallLog callLog,
            int turnNumber,
            Function<TurnRecorder, String> stateLine,
            Consumer<ExecutionHandle> started)
            throws InterruptedException {
        TurnRecorder turn = new TurnRecorder(fileSystem, frontend.renderer());
        // start() rather than run(): the turn runs on a thread of Atmosphere's, and the handle can close the
        // stream the model is answering on, so a stop takes effect now instead of after the tool loop.
        ExecutionHandle handle;
        try {
            handle = runner.start(message, history, turn);
        } catch (RuntimeException e) {
            turn.error(e);
            handle = ExecutionHandle.completed();
        }
        started.accept(handle);
        TurnEnd end = frontend.await(turn, handle, () -> stateLine.apply(turn));
        history.add(ChatMessage.user(message));
        callLog.add(turnNumber, turn.rounds());
        if (!turn.text().isEmpty()) {
            history.add(ChatMessage.assistant(turn.text()));
        }
        if (end == TurnEnd.TIMED_OUT) {
            turn.error(new IllegalStateException("turn did not finish within " + TURN_TIMEOUT));
        }
        return turn;
    }

    /**
     * Wait for a turn without showing anything: what a front end without a console does.
     *
     * @param turn the running turn
     * @param handle stops the turn
     * @param stopped polled while waiting; {@code true} cuts the turn short
     * @return how the turn ended
     * @throws InterruptedException if interrupted while waiting
     */
    public static TurnEnd awaitQuietly(TurnRecorder turn, ExecutionHandle handle, BooleanSupplier stopped)
            throws InterruptedException {
        long deadline = System.nanoTime() + TURN_TIMEOUT.toNanos();
        while (!turn.await(POLL_INTERVAL)) {
            if (stopped.getAsBoolean()) {
                handle.cancel();
                return TurnEnd.INTERRUPTED;
            }
            if (System.nanoTime() > deadline) {
                return TurnEnd.TIMED_OUT;
            }
        }
        return TurnEnd.FINISHED;
    }

    private void command(SlashCommands command, SessionFrontend frontend) throws InterruptedException {
        switch (command.command()) {
            case HELP -> Prompts.prompt(Prompts.HELP_TEXT).lines().forEach(frontend::line);
            case CLEAR -> {
                history.clear();
                // "Forget this session" has to mean the record too, or the word is not true.
                transcript.clear();
                // The session's own memory of the last turn goes with it: the tool note belongs to a turn that
                // no longer exists, and so does the message a /retry would repeat.
                pendingNote = "";
                lastMessage = "";
                inputTokens.set(0);
                // The screen goes with it: what is still on it is a conversation the model no longer has,
                // which reads as if it were still in play.
                frontend.clearScreen();
                frontend.line("(history cleared)");
            }
            case CLS -> frontend.clearScreen();
            case LOAD -> {
                if (!command.hasArguments()) {
                    frontend.line("say which file: /load <name>");
                } else {
                    inputTokens.set(load(command.arguments(), frontend));
                    estimated.set(true);
                }
            }
            case SAVE -> {
                try {
                    Path written = transcript.save(
                            options.getWorkspace(), command.hasArguments() ? command.arguments() : null);
                    frontend.line("transcript: " + transcript.size() + " entries -> " + written);
                } catch (IOException e) {
                    frontend.line("could not write the transcript: " + e.getMessage());
                }
            }
            case CALLS -> callLog.render().lines().forEach(frontend::line);
            case TOOLS -> {
                frontend.line("tools: " + String.join(", ", runner.toolNames()));
                frontend.line("asks before running (manual mode): "
                        + String.join(", ", ConsoleApprovalStrategy.gated(runner.toolNames())));
            }
            case MODE -> {
                if (command.hasArguments()) {
                    try {
                        mode.set(ApprovalMode.parse(command.arguments()));
                    } catch (IllegalArgumentException e) {
                        frontend.line(e.getMessage());
                        return;
                    }
                }
                frontend.line("approval mode: " + mode.get().badge());
            }
            case STATUS -> {
                frontend.line(status());
                frontend.line("workspace: " + options.getWorkspace());
                frontend.line("history: " + history.size() + " messages");
            }
            case COMPACT -> {
                // The record is deliberately untouched: compacting rewrites what the model is sent, not what
                // happened. Only the fact that it happened is worth a line.
                transcript.add(Transcript.Kind.NOTE, "compacted the conversation");
                inputTokens.set(compact(command.arguments(), frontend));
            }
            case LOOP -> loop(command.arguments(), frontend);
            case EXIT, RETRY -> {
                // handled by submit
            }
        }
    }

    /**
     * Run {@code /loop}: work on one task step by step until it is done.
     *
     * <p>Two things are settled before the first step. The loop needs the {@link ApprovalMode#AUTO} mode —
     * a run that asks before every write is not a loop, it is a conversation — so a manual session is asked
     * once, and left alone if the answer is no or nobody can answer. And the history is <em>not</em> the
     * loop's memory: {@link TaskLoop} drops it every step and keeps the state in a file.
     */
    private void loop(String arguments, SessionFrontend frontend) throws InterruptedException {
        LoopOptions loopOptions;
        try {
            loopOptions = LoopOptions.parse(arguments);
        } catch (IllegalArgumentException e) {
            frontend.line(e.getMessage());
            return;
        }
        if (mode.get() != ApprovalMode.AUTO) {
            String answer =
                    frontend.ask("a loop cannot stop at every question — switch to auto for it? [y]es / [n]o: ");
            if (answer == null || !(answer.startsWith("y") || answer.isEmpty())) {
                frontend.line("loop: cancelled (use /mode auto to allow it)");
                return;
            }
            mode.set(ApprovalMode.AUTO);
        }
        if (!TaskLoop.canKeepNotes(runner.toolNames())) {
            frontend.line("loop: needs the read_file and write_file tools to keep its notes");
            return;
        }
        TaskLoop.Outcome outcome = TaskLoop.run(
                (message, step, label) -> runTurn(
                        runner,
                        fileSystem,
                        message,
                        new ArrayList<>(),
                        frontend,
                        callLog,
                        step,
                        ignored -> label,
                        handle -> running = handle),
                frontend::line,
                frontend.ansi(),
                options.getWorkspace(),
                loopOptions,
                () -> stopRequested,
                TaskLoop.DEFAULT_BUDGET);
        running = null;
        frontend.line(
                outcome.completed()
                        ? frontend.ansi().green("loop: " + outcome.reason())
                        : frontend.ansi().yellow("loop: " + outcome.reason()));
    }

    /**
     * Read a saved transcript back in, as the conversation and as the record.
     *
     * <p>Only the questions and the answers become messages again. Tool calls and session notes are kept in
     * the record but <b>not</b> replayed to the model: a tool result out of its round is not something any
     * chat template has a place for, and inventing one would be worse than leaving the model to call the
     * tool again if it needs to.
     *
     * <p>The file is resolved against the workspace when it is not an absolute path, so {@code /load
     * session.txt} finds what {@code /save session.txt} wrote.
     */
    private long load(String name, SessionFrontend frontend) {
        Path file = Path.of(name.strip());
        if (!file.isAbsolute()) {
            file = options.getWorkspace().resolve(file);
        }
        List<Transcript.Entry> loaded;
        try {
            loaded = Transcript.parse(Files.readString(file, StandardCharsets.UTF_8));
        } catch (IOException e) {
            frontend.line("cannot read " + file + ": " + e.getMessage());
            return estimateTokens(systemPrompt, history);
        }
        if (loaded.isEmpty()) {
            frontend.line(file + " holds no transcript entries");
            return estimateTokens(systemPrompt, history);
        }
        history.clear();
        int messages = 0;
        for (Transcript.Entry entry : loaded) {
            switch (entry.kind()) {
                case USER -> {
                    history.add(ChatMessage.user(entry.text()));
                    messages++;
                }
                case AGENT -> {
                    history.add(ChatMessage.assistant(entry.text()));
                    messages++;
                }
                default -> {
                    // kept in the record, not replayed as a message
                }
            }
        }
        transcript.replaceWith(loaded);
        transcript.add(Transcript.Kind.NOTE, "loaded " + file);
        frontend.line("loaded " + loaded.size() + " entries from " + file + " (" + messages
                + " of them replayed to the model)");
        return estimateTokens(systemPrompt, history);
    }

    /**
     * Summarize the history and continue from the summary.
     *
     * <p>The summary is generated by the same model with no tools, then <b>replaces</b> the history as a
     * {@code user} message plus a short assistant acknowledgement — aider's shape, and the one that survives
     * a strict chat template, because a conversation may not start with two assistant turns.
     *
     * @return the input tokens the summarizing call reported
     */
    private long compact(String focus, SessionFrontend frontend) throws InterruptedException {
        if (history.isEmpty()) {
            frontend.line("(nothing to compact)");
            return 0;
        }
        if (isCompacted(history)) {
            // After a compaction the history IS the summary plus its acknowledgement. Summarizing that again
            // returns the same text for another model call -- and re-sends a byte-identical prompt, which is
            // what makes llama.cpp log "need to evaluate at least 1 token".
            frontend.line("(the history is already a summary — nothing to compact)");
            return estimateTokens("", history);
        }
        String instructions = Prompts.prompt(Prompts.COMPACT_PROMPT)
                .replace("{focus}", focus.isEmpty() ? "" : System.lineSeparator() + "Focus on: " + focus);
        int before = history.size();
        frontend.line("(compacting " + before + " messages …)");
        TurnRecorder summary = new TurnRecorder(fileSystem, frontend.renderer());
        List<ChatMessage> snapshot = List.copyOf(history);
        Thread worker = new Thread(
                () -> {
                    try {
                        runner.runWithoutTools(instructions, snapshot, summary, COMPACT_SYSTEM_PROMPT);
                    } catch (RuntimeException e) {
                        summary.error(e);
                    }
                },
                "agent-compact");
        worker.setDaemon(true);
        worker.start();
        if (frontend.await(summary, ExecutionHandle.completed(), () -> "/compact") != TurnEnd.FINISHED
                || summary.text().isBlank()) {
            frontend.line("(compact failed; history kept)");
            return 0;
        }
        history.clear();
        history.add(ChatMessage.user(
                SUMMARY_PREFIX + System.lineSeparator() + summary.text().strip()));
        history.add(ChatMessage.assistant("Understood, I will continue from that summary."));
        frontend.line("(compacted " + before + " messages into a summary of "
                + summary.text().strip().length() + " characters)");
        return summary.inputTokens();
    }

    /**
     * The session's state as one line: workspace, mode, context use, tools, model.
     *
     * @return the line
     */
    public String status() {
        return status(inputTokens.get(), estimated.get());
    }

    private String status(long tokens, boolean isEstimate) {
        return StatusLine.render(
                options.getWorkspace(),
                mode.get(),
                tokens,
                isEstimate,
                contextSize,
                runner.toolNames().size(),
                options.getModelId(),
                options.getModelPath() == null);
    }

    /**
     * The approval mode.
     *
     * @return the current mode
     */
    public ApprovalMode mode() {
        return mode.get();
    }

    /**
     * Set the approval mode, e.g. from a toggle in a front end.
     *
     * @param newMode the mode
     */
    public void mode(ApprovalMode newMode) {
        mode.set(newMode);
    }

    /**
     * Switch to the next approval mode, what shift+tab does on the console.
     *
     * @return the mode switched to
     */
    public ApprovalMode cycleMode() {
        return mode.updateAndGet(ApprovalMode::next);
    }

    /**
     * The shared, mutable mode itself, for a console strategy whose {@code [a]uto} answer switches it.
     *
     * @return the reference
     */
    AtomicReference<ApprovalMode> modeReference() {
        return mode;
    }

    /**
     * The tools offered to the model.
     *
     * @return the names in registration order
     */
    public List<String> toolNames() {
        return runner.toolNames();
    }

    /**
     * The model ids the endpoint advertises.
     *
     * @return the ids, or the configured one when the endpoint lists none
     */
    public List<String> models() {
        return runner.models();
    }

    /**
     * The workspace the tools work in.
     *
     * @return the directory
     */
    public Path workspace() {
        return options.getWorkspace();
    }

    /**
     * The model id sent in every request.
     *
     * @return the id
     */
    public String modelId() {
        return options.getModelId();
    }

    /**
     * The conversation the model is sent, as it stands.
     *
     * @return a copy of the history
     */
    public List<ChatMessage> history() {
        return List.copyOf(history);
    }

    /**
     * Every command name and alias, for completion and for front ends that list them.
     *
     * @return the names, each with its leading slash
     */
    public static List<String> commandNames() {
        return java.util.Arrays.stream(SlashCommands.Command.values())
                .flatMap(command -> command.names().stream())
                .toList();
    }

    /**
     * Take the last exchange out of the conversation, so a retry asks again instead of following on.
     *
     * <p>Leaving the failed answer in place would be the opposite of a retry: the model would see what it
     * said last time and, being a model, would say it again. The question is removed with it, because the
     * turn that follows adds it back.
     *
     * <p>Only a trailing exchange that really is the one being retried is touched — a history that was just
     * replaced by a summary, or one that never got an answer, is left alone.
     *
     * @param history the conversation, modified in place
     * @param message the question being asked again
     */
    static void dropLastExchange(List<ChatMessage> history, String message) {
        if (!history.isEmpty()
                && "assistant".equals(history.get(history.size() - 1).role())) {
            history.remove(history.size() - 1);
        }
        if (!history.isEmpty()) {
            ChatMessage last = history.get(history.size() - 1);
            if ("user".equals(last.role()) && message.equals(last.content())) {
                history.remove(history.size() - 1);
            }
        }
    }

    /**
     * Put a turn's tool calls in front of the next message, so the next turn can see they happened.
     *
     * <p><b>Why this matters more than it looks.</b> Without it the history holds the user's messages and
     * the model's prose, and nothing else — so from the third or fourth turn on, a small model sees only its
     * own paragraphs and no evidence that it ever used a tool. It then continues that pattern: it
     * <em>describes</em> creating a file and running a build, reports an exit code, and writes nothing at
     * all. That is not hypothetical; it happened on a real session, with the model inventing test results
     * and a jar that never existed.
     *
     * <p><b>Where it goes, and why not somewhere more obvious.</b> In front of the <em>next user
     * message</em>. The first attempt put it in front of the assistant's own answer, and the model promptly
     * copied it into its next reply, because text attributed to the assistant is text a model imitates. A
     * system message mid-history would be cleaner still, but not every chat template accepts one: Mistral's
     * requires strict user/assistant alternation and Gemma has no system role at all.
     *
     * <p><b>Why a note rather than real {@code tool_calls} messages.</b> Atmosphere's
     * {@code AbstractAgentRuntime.assembleMessages} rebuilds every history entry as
     * {@code new ChatMessage(role, content)} — the tool-call array and the tool-call id are dropped on the way
     * out. What survives is the content, so the evidence goes there, with each result cut to
     * {@value #HISTORY_RESULT_CHARS} characters.
     *
     * @param rounds the tool calls of the finished turn
     * @return the note, or an empty string when no tool ran
     */
    static String toolNote(List<TurnRecorder.ToolRound> rounds) {
        if (rounds.isEmpty()) {
            return "";
        }
        StringBuilder note = new StringBuilder("Record of the tools that actually ran in the previous turn."
                + " This is a log for your reference; do not repeat it and do not mention it.");
        for (TurnRecorder.ToolRound round : rounds) {
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
     * Whether the history has to be summarized before the next request is sent.
     *
     * <p>The threshold is on the low side ({@link AgentOptions#DEFAULT_COMPACT_AT} %) because the number it
     * is compared against is usually an estimate, and because the model's reply has to fit next to the
     * prompt. With an unknown context size nothing is decided at all rather than guessed.
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
     * <p>Used only for the status line, and only because llama.cpp sends its own count just to clients that
     * ask for it ({@code stream_options.include_usage}), which Atmosphere's client does not. Four characters
     * per token is the usual rule of thumb; the status line marks the number with a {@code ~}.
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
     * <p>Every tool round appends the call and its output to the conversation the next model call of the
     * same turn is sent, so a turn that reads three files and runs a build can add thousands of tokens before
     * the prompt comes back.
     *
     * @param baseTokens what the request carried when it was sent
     * @param turn the running turn
     * @return the server's own count once it reported one, else the base plus what the turn produced
     */
    static long liveTokens(long baseTokens, TurnRecorder turn) {
        long reported = turn.inputTokens();
        return reported > 0 ? reported : baseTokens + turn.producedChars() / CHARS_PER_TOKEN;
    }
}
