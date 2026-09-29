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
 * The {@link StreamingSession} one agent turn runs against: it keeps what the session needs to know
 * afterwards and forwards everything to the front end that shows it.
 *
 * <p>Kept here, whatever the front end: the streamed text (the history needs the raw text, never a
 * rendering of it), every tool call with its result (the note the next turn carries, and {@code /calls}),
 * the input-token count, which tool is running right now, and whether and how the turn ended.
 *
 * <p>Forwarded: every call, unchanged, to the {@code downstream} session — the console renderer, the
 * browser connection, or the editor protocol. That is the only thing that differs between front ends, which
 * is why a turn is the same code for all of them.
 *
 * <p>It also carries the {@link AgentFileSystem} the file tools resolve at execution time: Atmosphere passes
 * {@link #injectables()} into every tool executor, and the file tools look the filesystem up there — this is
 * how the tools are confined to the workspace without any framework wiring.
 */
public class TurnRecorder implements StreamingSession {

    /**
     * One tool call of a turn, kept so the next turn can see that it happened.
     *
     * @param name the tool
     * @param argumentsJson the arguments as JSON
     * @param result what the tool returned
     */
    public record ToolRound(String name, String argumentsJson, String result) {}

    private final StreamingSession downstream;
    private final Map<Class<?>, Object> injectables;
    private final StringBuilder text = new StringBuilder();
    private final List<String> chunks = new CopyOnWriteArrayList<>();
    private final CountDownLatch done = new CountDownLatch(1);
    private final List<ToolRound> rounds = new CopyOnWriteArrayList<>();
    private volatile @Nullable Throwable failure;
    private volatile int toolCalls;
    private volatile long inputTokens;
    private volatile @Nullable String runningTool;
    private volatile long runningSince;

    /**
     * Record a turn and forward it to {@code downstream}.
     *
     * @param fileSystem the workspace-confined filesystem handed to the file tools
     * @param downstream where every event is shown
     */
    public TurnRecorder(AgentFileSystem fileSystem, StreamingSession downstream) {
        this.downstream = downstream;
        this.injectables = Map.of(AgentFileSystem.class, fileSystem);
    }

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
        return downstream.sessionId();
    }

    @Override
    public Map<Class<?>, Object> injectables() {
        return injectables;
    }

    @Override
    public void send(String chunk) {
        chunks.add(chunk);
        text.append(chunk);
        downstream.send(chunk);
    }

    @Override
    public void sendMetadata(String key, Object value) {
        downstream.sendMetadata(key, value);
    }

    @Override
    public void usage(TokenUsage usage) {
        // The prompt of the last model call is what fills the context window -- the tokens generated
        // in that call are part of the next call's input. Several calls happen per turn (one per tool
        // round); the last one wins, which is the largest and the one the next turn continues from.
        if (usage != null && usage.input() > 0) {
            inputTokens = usage.input();
        }
        downstream.usage(usage);
    }

    @Override
    public void progress(String message) {
        downstream.progress(message);
    }

    @Override
    public void complete() {
        downstream.complete();
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
        downstream.error(t);
        done.countDown();
    }

    @Override
    public boolean isClosed() {
        return done.getCount() == 0;
    }

    @Override
    public boolean hasErrored() {
        return failure != null;
    }

    @Override
    public void emit(AiEvent event) {
        switch (event) {
            case AiEvent.TextDelta delta -> send(delta.text());
            case AiEvent.Complete complete -> {
                if (complete.summary() != null) {
                    complete(complete.summary());
                } else {
                    complete();
                }
            }
            case AiEvent.Error error -> error(new IllegalStateException(error.message()));
            case AiEvent.ToolStart start -> {
                toolCalls++;
                runningTool = start.toolName();
                runningSince = System.nanoTime();
                rounds.add(new ToolRound(start.toolName(), String.valueOf(start.arguments()), ""));
                downstream.emit(event);
            }
            case AiEvent.ToolResult result -> {
                runningTool = null;
                recordResult(String.valueOf(result.result()));
                downstream.emit(event);
            }
            case AiEvent.ToolError error -> {
                runningTool = null;
                recordResult("error: " + error.error());
                downstream.emit(event);
            }
            default -> downstream.emit(event);
        }
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

    /**
     * Everything this turn has produced so far, in characters: the streamed text plus every tool call
     * with its result.
     *
     * <p>All of it is in the prompt of the <em>next</em> model call of the same turn — a tool round
     * appends the call and its output to the conversation the server is sent. So this is what makes the
     * context grow while the turn runs.
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
