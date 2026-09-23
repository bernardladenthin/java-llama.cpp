// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.TokenUsage;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.jspecify.annotations.Nullable;

/**
 * A {@link StreamingSession} that prints one agent turn to a console: streamed text as it arrives,
 * one line per tool call and tool result, and the terminal state.
 *
 * <p>It also carries the {@link AgentFileSystem} the built-in file tools resolve at execution time:
 * Atmosphere passes {@link #injectables()} into every tool executor, and {@code FileSystemTools}
 * looks the filesystem up there — this is how the tools are confined to the workspace without any
 * framework wiring.
 */
public final class ConsoleSession implements StreamingSession {

    private static final int RESULT_PREVIEW_CHARS = 400;

    /** How much of a call's arguments the console shows; the model still gets them in full. */
    private static final int ARGUMENT_PREVIEW_CHARS = 200;

    /** How much of a single argument value survives, so one big one cannot hide the others. */
    private static final int ARGUMENT_VALUE_PREVIEW_CHARS = 80;

    private final AgentTerminal terminal;
    private final Ansi ansi;
    private final MarkdownConsole markdown;
    private final Map<Class<?>, Object> injectables;
    private final StringBuilder text = new StringBuilder();
    private final List<String> chunks = new CopyOnWriteArrayList<>();
    private final CountDownLatch done = new CountDownLatch(1);
    private volatile @Nullable Throwable failure;
    private volatile int toolCalls;
    private volatile long inputTokens;
    private final List<ToolRound> rounds = new CopyOnWriteArrayList<>();
    private volatile @Nullable String runningTool;
    private volatile long runningSince;

    /**
     * Create a session writing to {@code terminal}.
     *
     * @param terminal where streamed text and tool lines go
     * @param fileSystem the workspace-confined filesystem handed to the file tools
     */
    public ConsoleSession(AgentTerminal terminal, AgentFileSystem fileSystem) {
        this.terminal = terminal;
        this.ansi = terminal.ansi();
        this.markdown = new MarkdownConsole(terminal::line, ansi);
        this.injectables = Map.of(AgentFileSystem.class, fileSystem);
    }

    /**
     * One tool call of this turn, kept so the next turn can see that it happened.
     *
     * @param name the tool
     * @param argumentsJson the arguments as JSON
     * @param result what the tool returned, already shortened
     */
    public record ToolRound(String name, String argumentsJson, String result) {}

    /**
     * The tool calls of this turn, in order.
     *
     * @return the rounds, empty when the model only wrote text
     */
    public List<ToolRound> rounds() {
        return List.copyOf(rounds);
    }

    @Override
    public String sessionId() {
        return "console";
    }

    @Override
    public Map<Class<?>, Object> injectables() {
        return injectables;
    }

    @Override
    public void send(String chunk) {
        chunks.add(chunk);
        // The history keeps the raw text; only the console sees the rendered form.
        text.append(chunk);
        markdown.append(chunk);
    }

    @Override
    public void sendMetadata(String key, Object value) {
        // model id, tool-call argument deltas: not shown on the console
    }

    @Override
    public void usage(TokenUsage usage) {
        // The prompt of the last model call is what fills the context window -- the tokens generated
        // in that call are part of the next call's input. Several calls happen per turn (one per tool
        // round); the last one wins, which is the largest and the one the next turn continues from.
        if (usage != null && usage.input() > 0) {
            inputTokens = usage.input();
        }
    }

    /**
     * Everything this turn has produced so far, in characters: the streamed text plus every tool call
     * with its result.
     *
     * <p>All of it is in the prompt of the <em>next</em> model call of the same turn — a tool round
     * appends the call and its output to the conversation the server is sent. So this is what makes the
     * context grow while the turn runs, and the status line adds it to the count it showed before the
     * turn started instead of standing still until the next prompt.
     *
     * @return the character count
     */
    public long producedChars() {
        long chars = text.length();
        for (ToolRound round : rounds) {
            chars += round.argumentsJson().length() + round.result().length();
        }
        return chars;
    }

    /**
     * The input tokens of the last model call of this turn.
     *
     * @return the count, or {@code 0} when the endpoint reported no usage
     */
    public long inputTokens() {
        return inputTokens;
    }

    @Override
    public void progress(String message) {
        // "Connecting to built-in..." and friends: not shown on the console
    }

    @Override
    public void complete() {
        markdown.flush();
        done.countDown();
    }

    @Override
    public void complete(String summary) {
        if (summary != null && text.length() == 0) {
            send(summary);
        }
        complete();
    }

    @Override
    public void error(Throwable t) {
        failure = t;
        markdown.flush();
        terminal.line(ansi.red("[error] " + t));
        done.countDown();
    }

    @Override
    public boolean isClosed() {
        return done.getCount() == 0;
    }

    @Override
    public void emit(AiEvent event) {
        switch (event) {
            case AiEvent.ToolStart start -> {
                toolCalls++;
                runningTool = start.toolName();
                runningSince = System.nanoTime();
                rounds.add(new ToolRound(start.toolName(), String.valueOf(start.arguments()), ""));
                markdown.flush();
                terminal.line(ansi.green("●") + " " + ansi.bold(start.toolName()) + " "
                        + ansi.dim(describeArguments(start.arguments())));
            }
            case AiEvent.ToolResult result -> {
                runningTool = null;
                recordResult(String.valueOf(result.result()));
                terminal.line(ansi.dim("  ↳ " + preview(String.valueOf(result.result()))));
            }
            case AiEvent.ToolError error -> {
                runningTool = null;
                recordResult("error: " + error.error());
                terminal.line(ansi.red("  ↳ error: " + cut(String.valueOf(error.error()), RESULT_PREVIEW_CHARS)));
            }
            default -> StreamingSession.super.emit(event);
        }
    }

    /**
     * The tool that is executing right now, if any.
     *
     * @return the tool name, or {@code null} when the model is generating rather than running something
     */
    public @Nullable String runningTool() {
        return runningTool;
    }

    /**
     * How long the running tool has been running.
     *
     * @return the seconds since it started, or {@code 0} when nothing runs
     */
    public long runningSeconds() {
        return runningTool == null ? 0 : (System.nanoTime() - runningSince) / 1_000_000_000L;
    }

    /** Attach a result to the round that is still waiting for one. */
    private void recordResult(String result) {
        for (int i = rounds.size() - 1; i >= 0; i--) {
            ToolRound round = rounds.get(i);
            if (round.result().isEmpty()) {
                rounds.set(i, new ToolRound(round.name(), round.argumentsJson(), result));
                return;
            }
        }
    }

    private static String preview(String value) {
        return cut(value, RESULT_PREVIEW_CHARS);
    }

    /**
     * The arguments of a call, short enough for one console line.
     *
     * <p>Every value is cut <em>on its own</em> before the whole thing is. Cutting only the rendered
     * map would let one big argument push the others out of the line — a {@code write_file} call would
     * then show half of the file and not the name of the file, which is the one thing worth seeing.
     *
     * @param arguments what the model passed, usually a map
     * @return one line
     */
    private static String describeArguments(Object arguments) {
        if (!(arguments instanceof Map<?, ?> map)) {
            return cut(String.valueOf(arguments), ARGUMENT_PREVIEW_CHARS);
        }
        StringBuilder rendered = new StringBuilder("{");
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (rendered.length() > 1) {
                rendered.append(", ");
            }
            rendered.append(entry.getKey())
                    .append('=')
                    .append(cut(String.valueOf(entry.getValue()), ARGUMENT_VALUE_PREVIEW_CHARS));
        }
        return cut(rendered.append('}').toString(), ARGUMENT_PREVIEW_CHARS);
    }

    /**
     * Fold a value onto one line and cut it.
     *
     * <p>Both halves matter. The cut keeps a whole file out of the scrollback, and the folding keeps
     * the pinned block intact: that block is sized in <em>lines</em>, so a single printed "line"
     * carrying twenty newlines moves the screen twenty rows further than the terminal accounted for,
     * and the block ends up drawn across the output. A {@code write_file} call whose arguments contain
     * the file did exactly that.
     *
     * @param value the raw text
     * @param max how many characters survive
     * @return one line, with a note about what was left out
     */
    private static String cut(String value, int max) {
        String oneLine = value.replace("\r\n", " ").replace('\n', ' ').replace('\r', ' ');
        return oneLine.length() <= max ? oneLine : oneLine.substring(0, max) + " … (" + value.length() + " chars)";
    }

    /**
     * Block until the turn completed or errored.
     *
     * @param timeout how long to wait
     * @return {@code true} if the session terminated within the timeout
     * @throws InterruptedException if interrupted while waiting
     */
    public boolean await(Duration timeout) throws InterruptedException {
        return done.await(timeout.toMillis(), TimeUnit.MILLISECONDS);
    }

    /**
     * The streamed assistant text of this turn.
     *
     * @return the text so far
     */
    public String text() {
        return text.toString();
    }

    /**
     * Every streamed text chunk of this turn, in arrival order.
     *
     * @return the chunks as delivered by the server (one SSE {@code delta.content} each)
     */
    public List<String> chunks() {
        return List.copyOf(chunks);
    }

    /**
     * The terminal error, if the turn failed.
     *
     * @return the throwable passed to {@link #error}, or {@code null}
     */
    public @Nullable Throwable failure() {
        return failure;
    }

    /**
     * How many tool calls the model made in this turn.
     *
     * @return the count of {@code ToolStart} events
     */
    public int toolCalls() {
        return toolCalls;
    }
}
