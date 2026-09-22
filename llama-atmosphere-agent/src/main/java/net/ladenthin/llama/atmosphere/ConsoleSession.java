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
                markdown.flush();
                terminal.line(ansi.green("●") + " " + ansi.bold(start.toolName()) + " "
                        + ansi.dim(String.valueOf(start.arguments())));
            }
            case AiEvent.ToolResult result -> {
                terminal.line(ansi.dim("  ↳ " + preview(String.valueOf(result.result()))));
            }
            case AiEvent.ToolError error -> {
                terminal.line(ansi.red("  ↳ error: " + error.error()));
            }
            default -> StreamingSession.super.emit(event);
        }
    }

    private static String preview(String value) {
        String oneLine = value.replace("\r\n", "\n").replace('\n', ' ');
        return oneLine.length() <= RESULT_PREVIEW_CHARS
                ? oneLine
                : oneLine.substring(0, RESULT_PREVIEW_CHARS) + " … (" + value.length() + " chars)";
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
