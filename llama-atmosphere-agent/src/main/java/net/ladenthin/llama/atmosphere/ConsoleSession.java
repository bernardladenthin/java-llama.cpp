// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.PrintStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.StreamingSession;
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

    private final PrintStream out;
    private final Map<Class<?>, Object> injectables;
    private final StringBuilder text = new StringBuilder();
    private final List<String> chunks = new CopyOnWriteArrayList<>();
    private final CountDownLatch done = new CountDownLatch(1);
    private volatile @Nullable Throwable failure;
    private volatile int toolCalls;

    /**
     * Create a session printing to {@code out}.
     *
     * @param out where streamed text and tool lines go
     * @param fileSystem the workspace-confined filesystem handed to the file tools
     */
    public ConsoleSession(PrintStream out, AgentFileSystem fileSystem) {
        this.out = out;
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
        text.append(chunk);
        out.print(chunk);
        out.flush();
    }

    @Override
    public void sendMetadata(String key, Object value) {
        // token usage, model id, tool-call argument deltas: not shown on the console
    }

    @Override
    public void progress(String message) {
        // "Connecting to built-in..." and friends: not shown on the console
    }

    @Override
    public void complete() {
        out.println();
        out.flush();
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
        out.println();
        out.println("[error] " + t);
        out.flush();
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
                if (text.length() > 0 && text.charAt(text.length() - 1) != '\n') {
                    out.println();
                }
                out.println("⚙ " + start.toolName() + " " + start.arguments());
                out.flush();
            }
            case AiEvent.ToolResult result -> {
                out.println("↳ " + preview(String.valueOf(result.result())));
                out.flush();
            }
            case AiEvent.ToolError error -> {
                out.println("↳ error: " + error.error());
                out.flush();
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
