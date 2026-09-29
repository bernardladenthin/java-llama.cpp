// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.concurrent.atomic.AtomicReference;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.annotation.AiEndpoint;
import org.atmosphere.ai.annotation.Prompt;
import org.jspecify.annotations.Nullable;

/**
 * The Atmosphere endpoint the browser talks to: every message becomes one {@link AgentSession#submit}.
 *
 * <p>Atmosphere's own AI pipeline stays out of it: the endpoint declares no tools, no memory and no
 * interceptors, and never calls {@code session.stream()} — the agent's {@link AgentRunner} does the model
 * calls, so the conversation, the tool loop and the approval rules are exactly the console's. The
 * {@code timeout} of {@code -1} matters: Atmosphere's default interrupts a prompt after two minutes and
 * suspends the connection for as long, while one agent request (a build, a {@code /loop}) can take much
 * longer.
 *
 * <p>Atmosphere instantiates this class itself, so the session it serves is handed over through
 * {@link #SESSION}; that makes the browser front end one per JVM, which {@link WebServer} enforces.
 */
@AiEndpoint(path = WebServer.AGENT_PATH, timeout = -1)
public final class WebAgentEndpoint {

    /** The session every browser message goes to. Set by {@link WebServer} while it runs. */
    static final AtomicReference<@Nullable AgentSession> SESSION = new AtomicReference<>();

    /** What the stop button of a front end, or a user, sends to stop the running request. */
    static final String STOP = "/stop";

    /**
     * Handle one message from the browser.
     *
     * @param message what was typed
     * @param connection the connection it came in on
     * @throws InterruptedException if interrupted while the request runs
     */
    @Prompt
    public void onPrompt(String message, StreamingSession connection) throws InterruptedException {
        AgentSession agent = SESSION.get();
        try {
            if (agent == null) {
                connection.send("the agent is shutting down");
                return;
            }
            String line = message.strip();
            if (STOP.equals(line)) {
                agent.cancel();
                connection.send("(stopped)");
                return;
            }
            // A new message while a request is running replaces it, as typing does on the console: the
            // running one is cut short and this one waits for the session to be free.
            if (agent.isBusy()) {
                agent.cancel();
            }
            WebFrontend frontend = new WebFrontend(connection, agent);
            if (agent.submit(line, frontend) == AgentSession.Result.EXIT) {
                frontend.line("(the browser session stays open; stop the agent with Ctrl-C where it was started)");
            }
        } finally {
            connection.complete();
        }
    }
}
