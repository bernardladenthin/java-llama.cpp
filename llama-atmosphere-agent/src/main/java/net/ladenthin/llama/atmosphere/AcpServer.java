// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import com.agentclientprotocol.sdk.agent.AcpAgent;
import com.agentclientprotocol.sdk.agent.AcpSyncAgent;
import com.agentclientprotocol.sdk.agent.SyncPromptContext;
import com.agentclientprotocol.sdk.agent.transport.StdioAcpAgentTransport;
import com.agentclientprotocol.sdk.json.AcpJsonMapper;
import com.agentclientprotocol.sdk.spec.AcpSchema;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import org.jspecify.annotations.Nullable;

/**
 * {@code --acp}: the agent as an Agent Client Protocol agent on stdin/stdout, for editors — JetBrains IDEs
 * and Zed natively, VS Code through an ACP extension.
 *
 * <p>The editor starts this process and talks JSON-RPC over its standard streams. Each ACP session is one
 * {@link AgentSession} whose workspace is the directory the editor names ({@code cwd}), so the tools are
 * confined to the open project. Everything the consoles have is here in ACP's terms:
 *
 * <ul>
 *   <li>the approval modes as ACP session modes ({@code manual}, {@code auto}), switchable from the editor;
 *   <li>the slash commands as {@code available_commands_update}, which the editor offers on {@code /};
 *   <li>stopping a turn as {@code session/cancel}.
 * </ul>
 *
 * <p>Configuration in an editor, e.g. JetBrains' {@code ~/.jetbrains/acp.json}:
 *
 * <pre>{@code
 * { "agent_servers": { "Local llama": {
 *     "command": "java",
 *     "args": ["-jar", "/path/llama-atmosphere-agent.jar", "--acp", "--model", "/path/model.gguf"] } } }
 * }</pre>
 */
public final class AcpServer implements AutoCloseable {

    /** How long an editor may take to answer a permission question before it counts as a no. */
    static final Duration PERMISSION_TIMEOUT = Duration.ofMinutes(30);

    private final AcpSyncAgent agent;
    private final Map<String, AgentSession> sessions = new ConcurrentHashMap<>();

    private AcpServer(Function<Path, AgentSession> sessionFactory, InputStream in, OutputStream out, String version) {
        StdioAcpAgentTransport transport = new StdioAcpAgentTransport(AcpJsonMapper.createDefault(), in, out);
        AcpSyncAgent[] self = new AcpSyncAgent[1];
        this.agent = AcpAgent.sync(transport)
                .requestTimeout(PERMISSION_TIMEOUT)
                .initializeHandler(request -> new AcpSchema.InitializeResponse(
                        AcpSchema.LATEST_PROTOCOL_VERSION,
                        new AcpSchema.AgentCapabilities(),
                        List.of(),
                        new AcpSchema.Implementation("java-llama.cpp-agent", version, "java-llama.cpp agent"),
                        null))
                .newSessionHandler(request -> {
                    String id = UUID.randomUUID().toString();
                    AgentSession session = sessionFactory.apply(Path.of(request.cwd()));
                    sessions.put(id, session);
                    announceCommandsSoon(self[0], id);
                    return new AcpSchema.NewSessionResponse(id, modes(session.mode()), null);
                })
                .promptHandler(this::prompt)
                .setSessionModeHandler(request -> {
                    AgentSession session = session(request.sessionId());
                    session.mode(ApprovalMode.parse(request.modeId()));
                    return new AcpSchema.SetSessionModeResponse();
                })
                .cancelHandler(notification -> {
                    AgentSession session = sessions.get(notification.sessionId());
                    if (session != null) {
                        session.cancel();
                    }
                })
                .build();
        self[0] = agent;
    }

    /**
     * Speak ACP over the given streams.
     *
     * @param sessionFactory builds a session for the directory the editor names
     * @param in what the editor sends
     * @param out what the editor reads — nothing else may write to it
     * @param version the version reported to the editor
     * @return the running server
     */
    static AcpServer start(
            Function<Path, AgentSession> sessionFactory, InputStream in, OutputStream out, String version) {
        AcpServer server = new AcpServer(sessionFactory, in, out, version);
        server.agent.start();
        return server;
    }

    /**
     * Serve the agent to an editor on stdin/stdout until the editor closes the stream.
     *
     * @param options the parsed command line
     * @param err diagnostics; stdout belongs to the protocol
     * @return the exit code
     * @throws Exception when the model cannot be reached
     */
    static int run(AgentOptions options, PrintStream err) throws Exception {
        // stdout is the protocol from here on: a stray println would corrupt it, so make any that slips
        // through land on stderr instead.
        PrintStream protocol = System.out;
        System.setOut(err);
        try (ModelEndpoint endpoint = ModelEndpoint.open(options, err)) {
            err.println("ACP agent on stdin/stdout, model " + endpoint.baseUrl());
            try (AcpServer server = start(
                    workspace -> AgentSession.open(
                            options.withWorkspace(workspace), endpoint.baseUrl(), endpoint.contextSize()),
                    System.in,
                    protocol,
                    AcpServer.class.getPackage().getImplementationVersion() == null
                            ? "dev"
                            : AcpServer.class.getPackage().getImplementationVersion())) {
                server.agent.await();
            }
        }
        return 0;
    }

    private AcpSchema.PromptResponse prompt(AcpSchema.PromptRequest request, SyncPromptContext context) {
        AgentSession session = session(request.sessionId());
        ApprovalMode before = session.mode();
        AcpFrontend frontend = new AcpFrontend(context, request.sessionId(), session, session.workspace());
        try {
            if (session.submit(text(request.prompt()), frontend) == AgentSession.Result.EXIT) {
                frontend.line("(/exit does nothing here; close the session in the editor)");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new AcpSchema.PromptResponse(AcpSchema.StopReason.CANCELLED);
        }
        if (session.mode() != before) {
            context.sendUpdate(
                    request.sessionId(),
                    new AcpSchema.CurrentModeUpdate("current_mode_update", modeId(session.mode())));
        }
        return new AcpSchema.PromptResponse(
                session.stopRequested() ? AcpSchema.StopReason.CANCELLED : AcpSchema.StopReason.END_TURN);
    }

    private AgentSession session(String id) {
        AgentSession session = sessions.get(id);
        if (session == null) {
            throw new IllegalArgumentException("unknown session: " + id);
        }
        return session;
    }

    /**
     * The text of a prompt: its text blocks, and the text of any resource the editor attached.
     *
     * @param prompt the content blocks
     * @return one message
     */
    static String text(@Nullable List<AcpSchema.ContentBlock> prompt) {
        StringBuilder text = new StringBuilder();
        if (prompt == null) {
            return "";
        }
        for (AcpSchema.ContentBlock block : prompt) {
            String part = switch (block) {
                case AcpSchema.TextContent content -> content.text();
                case AcpSchema.Resource resource
                when resource.resource() instanceof AcpSchema.TextResourceContents contents ->
                    "Contents of " + contents.uri() + ":\n" + contents.text();
                case AcpSchema.ResourceLink link -> "(attached: " + link.uri() + ")";
                default -> null;
            };
            if (part != null) {
                if (text.length() > 0) {
                    text.append("\n\n");
                }
                text.append(part);
            }
        }
        return text.toString();
    }

    /**
     * The approval modes as ACP session modes.
     *
     * @param current the session's mode
     * @return the mode state
     */
    static AcpSchema.SessionModeState modes(ApprovalMode current) {
        return new AcpSchema.SessionModeState(
                modeId(current),
                List.of(
                        new AcpSchema.SessionMode(
                                modeId(ApprovalMode.MANUAL), "Ask", "Ask before commands and file changes"),
                        new AcpSchema.SessionMode(
                                modeId(ApprovalMode.AUTO), "Auto", "Run commands and change files without asking")));
    }

    /**
     * The ACP mode id of an approval mode — the same word {@code /mode} takes.
     *
     * @param mode the mode
     * @return {@code manual} or {@code auto}
     */
    static String modeId(ApprovalMode mode) {
        return mode == ApprovalMode.AUTO ? "auto" : "manual";
    }

    /**
     * The slash commands in ACP's form: names without the slash, as the editor lists them.
     *
     * @return one entry per command (aliases left out, the editor shows one name)
     */
    static List<AcpSchema.AvailableCommand> commands() {
        List<AcpSchema.AvailableCommand> commands = new ArrayList<>();
        for (SlashCommands.Command command : SlashCommands.Command.values()) {
            if (command == SlashCommands.Command.EXIT || command == SlashCommands.Command.CLS) {
                continue; // nothing to exit or clear in an editor
            }
            commands.add(new AcpSchema.AvailableCommand(
                    command.canonicalName().substring(1),
                    command.description(),
                    command.argumentHint() == null
                            ? null
                            : new AcpSchema.AvailableCommandInput(command.argumentHint())));
        }
        return commands;
    }

    /** Tell the editor which commands exist, once it knows the session the update is about. */
    private static void announceCommandsSoon(AcpSyncAgent agent, String sessionId) {
        Thread.startVirtualThread(() -> {
            try {
                // The session id reaches the editor with the response to session/new; an update that
                // arrives before it names a session the editor does not know yet.
                Thread.sleep(100);
                agent.sendSessionUpdate(
                        sessionId, new AcpSchema.AvailableCommandsUpdate("available_commands_update", commands()));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (RuntimeException e) {
                // the editor went away; nothing to announce to
            }
        });
    }

    @Override
    public void close() {
        agent.closeGracefully();
    }
}
