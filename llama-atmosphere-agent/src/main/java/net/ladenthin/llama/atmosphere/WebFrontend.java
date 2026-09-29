// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.function.Supplier;
import org.atmosphere.ai.AiStreamingSession;
import org.atmosphere.ai.DelegatingStreamingSession;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.jspecify.annotations.Nullable;

/**
 * A session shown in a browser, over the Atmosphere connection one message arrived on.
 *
 * <p>Everything the browser sees goes out on that connection in Atmosphere's AI streaming protocol, which
 * the console renders: text as it streams, a card per tool call, and — in manual mode — an approval card
 * with <b>Approve</b> / <b>Deny</b> buttons. Those buttons send {@code /__approval/<id>/approve} or
 * {@code …/deny}, which Atmosphere resolves against the connection's approval registry before any message
 * reaches the agent; {@link #approvals()} is simply Atmosphere's own strategy on that registry.
 *
 * <p>The connection outlives a turn — a request may run a compaction turn before the real one, or a
 * {@code /loop} of many — so the renderer handed to each turn swallows {@code complete()}: the endpoint
 * completes the connection once, when the whole request is done.
 */
final class WebFrontend implements SessionFrontend {

    private final StreamingSession connection;
    private final AgentSession agent;

    /**
     * Show a request on the connection it came in on.
     *
     * @param connection the Atmosphere session of the message
     * @param agent the conversation, for its stop request
     */
    WebFrontend(StreamingSession connection, AgentSession agent) {
        this.connection = connection;
        this.agent = agent;
    }

    @Override
    public Ansi ansi() {
        return Ansi.PLAIN;
    }

    @Override
    public void line(String text) {
        // Two trailing spaces are Markdown's hard line break: the console renders Markdown, and without them
        // the lines of /help or /status would run together into one paragraph.
        connection.send(text + "  \n");
    }

    @Override
    public StreamingSession renderer() {
        return new DelegatingStreamingSession(connection) {
            @Override
            public void complete() {
                // the endpoint completes the connection once the whole request is done
            }

            @Override
            public void complete(String summary) {
                // same: the recorder has already sent the summary as text
            }

            @Override
            public void error(Throwable t) {
                // An error frame would end the connection in the middle of a request; say it instead.
                connection.send("\n\n**error:** " + t.getMessage() + "  \n");
            }
        };
    }

    @Override
    public TurnEnd await(TurnRecorder turn, ExecutionHandle handle, Supplier<String> stateLine)
            throws InterruptedException {
        // A closed connection is a stop too: nobody is there to see the rest, and the tools would keep
        // working for nobody.
        return AgentSession.awaitQuietly(turn, handle, () -> agent.stopRequested() || connection.isClosed());
    }

    @Override
    public @Nullable ApprovalStrategy approvals() {
        return AiStreamingSession.unwrap(connection)
                .map(ai -> ApprovalStrategy.virtualThread(ai.approvalRegistry()))
                .orElse(null);
    }

    @Override
    public @Nullable String ask(String question) {
        // The console has no way to answer a free question in the middle of a request; the caller treats
        // this as "no" and says how to get what it wanted (e.g. /mode auto before /loop).
        return null;
    }

    @Override
    public void commandOutput(String line) {
        // Progress rather than text: the command's output must not become part of the answer, and the tool
        // card shows the full result when the command ends.
        connection.progress(line);
    }
}
