// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import com.agentclientprotocol.sdk.agent.SyncPromptContext;
import com.agentclientprotocol.sdk.spec.AcpSchema;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.atmosphere.ai.AiEvent;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalResolution;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.PendingApproval;
import org.jspecify.annotations.Nullable;

/**
 * A session shown in an editor, over the Agent Client Protocol: one instance per {@code session/prompt}.
 *
 * <p>What the editor gets, in ACP's own terms:
 *
 * <ul>
 *   <li>streamed text as {@code agent_message_chunk} updates — the answer and every command's output;
 *   <li>each tool call as a {@code tool_call} update with its kind, arguments and the file it touches, then
 *       a {@code tool_call_update} with the result, so the editor can show it and open the file;
 *   <li>an approval as {@code session/request_permission}: the editor's own dialog, with <em>allow</em>,
 *       <em>allow always</em> (switches the session to auto mode, like the console's {@code [a]}) and
 *       <em>reject</em>. Its answer is the only one that counts; a cancelled dialog is a no.
 * </ul>
 *
 * <p>Nothing is written to stdout except through the SDK: stdout <em>is</em> the protocol.
 */
final class AcpFrontend implements SessionFrontend {

    /** How much of a tool's output the editor gets in the finished card. */
    private static final int RESULT_CHARS = 4_000;

    /** How often a running command's output is pushed to its card. */
    private static final long OUTPUT_INTERVAL_NANOS = 500_000_000L;

    /** How many lines of a running command's output the card shows. */
    private static final int OUTPUT_TAIL_LINES = 40;

    private final SyncPromptContext context;
    private final String sessionId;
    private final AgentSession agent;
    private final Path workspace;
    private final AtomicInteger toolCounter = new AtomicInteger();
    private final Deque<String[]> openTools = new ArrayDeque<>();
    private final Deque<String> output = new ArrayDeque<>();
    private long lastOutputPush;

    /**
     * Show one prompt of an editor session.
     *
     * @param context the SDK's handle on the running prompt
     * @param sessionId the ACP session id
     * @param agent the conversation
     * @param workspace the session's working directory, for the files a tool call touches
     */
    AcpFrontend(SyncPromptContext context, String sessionId, AgentSession agent, Path workspace) {
        this.context = context;
        this.sessionId = sessionId;
        this.agent = agent;
        this.workspace = workspace;
    }

    @Override
    public Ansi ansi() {
        return Ansi.PLAIN;
    }

    @Override
    public void line(String text) {
        context.sendMessage(text + "\n");
    }

    @Override
    public StreamingSession renderer() {
        return new StreamingSession() {
            @Override
            public String sessionId() {
                return sessionId;
            }

            @Override
            public void send(String text) {
                context.sendMessage(text);
            }

            @Override
            public void sendMetadata(String key, Object value) {
                // model id, token deltas: nothing an editor shows
            }

            @Override
            public void progress(String message) {
                // "connecting…": nothing an editor shows
            }

            @Override
            public void complete() {
                // the prompt response ends the turn
            }

            @Override
            public void complete(String summary) {
                // same; the recorder has already sent the summary as text
            }

            @Override
            public void error(Throwable t) {
                context.sendMessage("\n\nerror: " + t.getMessage() + "\n");
            }

            @Override
            public boolean isClosed() {
                return false;
            }

            @Override
            public void emit(AiEvent event) {
                switch (event) {
                    case AiEvent.ToolStart start -> toolStarted(start.toolName(), start.arguments());
                    case AiEvent.ToolResult result ->
                        toolEnded(
                                result.toolName(), String.valueOf(result.result()), AcpSchema.ToolCallStatus.COMPLETED);
                    case AiEvent.ToolError error ->
                        toolEnded(error.toolName(), String.valueOf(error.error()), AcpSchema.ToolCallStatus.FAILED);
                    default -> {
                        // approvals are asked with session/request_permission, not shown as events
                    }
                }
            }
        };
    }

    private synchronized void toolStarted(String tool, @Nullable Map<String, Object> arguments) {
        String id = "call_" + toolCounter.incrementAndGet();
        openTools.addLast(new String[] {id, tool});
        output.clear();
        context.sendUpdate(
                sessionId,
                new AcpSchema.ToolCall(
                        "tool_call",
                        id,
                        title(tool, arguments),
                        kind(tool),
                        AcpSchema.ToolCallStatus.IN_PROGRESS,
                        null,
                        locations(arguments),
                        arguments,
                        null,
                        null));
    }

    private synchronized void toolEnded(String tool, String result, AcpSchema.ToolCallStatus status) {
        String id = takeOpen(tool);
        if (id == null) {
            return;
        }
        context.sendUpdate(
                sessionId,
                new AcpSchema.ToolCallUpdateNotification(
                        "tool_call_update", id, null, null, status, text(cut(result)), null, null, result, null));
    }

    private @Nullable String takeOpen(String tool) {
        for (String[] open : openTools) {
            if (open[1].equals(tool)) {
                openTools.remove(open);
                return open[0];
            }
        }
        return null;
    }

    private synchronized @Nullable String openId(String tool) {
        for (String[] open : openTools) {
            if (open[1].equals(tool)) {
                return open[0];
            }
        }
        return null;
    }

    @Override
    public TurnEnd await(TurnRecorder turn, ExecutionHandle handle, Supplier<String> stateLine)
            throws InterruptedException {
        return AgentSession.awaitQuietly(turn, handle, agent::stopRequested);
    }

    @Override
    public ApprovalStrategy approvals() {
        return new ApprovalStrategy() {
            @Override
            public ApprovalOutcome awaitApproval(PendingApproval approval, StreamingSession session) {
                return awaitApprovalDetailed(approval, session).outcome();
            }

            @Override
            public ApprovalResolution awaitApprovalDetailed(PendingApproval approval, StreamingSession session) {
                return askEditor(approval);
            }
        };
    }

    private ApprovalResolution askEditor(PendingApproval approval) {
        String id = openId(approval.toolName());
        AcpSchema.ToolCallUpdate call = new AcpSchema.ToolCallUpdate(
                id != null ? id : "call_" + toolCounter.incrementAndGet(),
                title(approval.toolName(), approval.arguments()),
                kind(approval.toolName()),
                AcpSchema.ToolCallStatus.PENDING,
                null,
                locations(approval.arguments()),
                approval.arguments(),
                null);
        AcpSchema.RequestPermissionResponse answer;
        try {
            answer = context.requestPermission(new AcpSchema.RequestPermissionRequest(
                    sessionId,
                    call,
                    List.of(
                            new AcpSchema.PermissionOption("allow", "Allow", AcpSchema.PermissionOptionKind.ALLOW_ONCE),
                            new AcpSchema.PermissionOption(
                                    "always",
                                    "Allow all (switch to auto mode)",
                                    AcpSchema.PermissionOptionKind.ALLOW_ALWAYS),
                            new AcpSchema.PermissionOption(
                                    "reject", "Reject", AcpSchema.PermissionOptionKind.REJECT_ONCE))));
        } catch (RuntimeException e) {
            // no answer (the editor went away, or the request timed out): the same as nobody to ask
            return ApprovalResolution.deny();
        }
        if (answer == null || !(answer.outcome() instanceof AcpSchema.PermissionSelected selected)) {
            return ApprovalResolution.deny();
        }
        return switch (selected.optionId()) {
            case "allow" -> ApprovalResolution.approve();
            case "always" -> {
                agent.mode(ApprovalMode.AUTO);
                context.sendUpdate(
                        sessionId,
                        new AcpSchema.CurrentModeUpdate("current_mode_update", AcpServer.modeId(ApprovalMode.AUTO)));
                yield ApprovalResolution.approve();
            }
            default -> ApprovalResolution.deny();
        };
    }

    @Override
    public @Nullable String ask(String question) {
        // No free question in ACP; the caller treats this as "no" and says what to do instead.
        return null;
    }

    @Override
    public synchronized void commandOutput(String line) {
        output.addLast(line);
        while (output.size() > OUTPUT_TAIL_LINES) {
            output.removeFirst();
        }
        long now = System.nanoTime();
        String id = openId(ShellTool.TOOL_NAME);
        if (id == null || now - lastOutputPush < OUTPUT_INTERVAL_NANOS) {
            return;
        }
        lastOutputPush = now;
        context.sendUpdate(
                sessionId,
                new AcpSchema.ToolCallUpdateNotification(
                        "tool_call_update",
                        id,
                        null,
                        null,
                        AcpSchema.ToolCallStatus.IN_PROGRESS,
                        text(String.join("\n", output)),
                        null,
                        null,
                        null,
                        null));
    }

    private static List<AcpSchema.ToolCallContent> text(String value) {
        return List.of(new AcpSchema.ToolCallContentBlock("content", new AcpSchema.TextContent(value)));
    }

    private static String cut(String value) {
        return value.length() <= RESULT_CHARS
                ? value
                : value.substring(0, RESULT_CHARS) + "\n… (" + value.length() + " chars)";
    }

    /**
     * A short title for a tool call: the tool and its most telling argument.
     *
     * @param tool the tool name
     * @param arguments the call's arguments
     * @return e.g. {@code run_command: mvn test}
     */
    static String title(String tool, @Nullable Map<String, Object> arguments) {
        if (arguments == null) {
            return tool;
        }
        for (String key : new String[] {"command", "file_path", "path", "pattern", "source"}) {
            Object value = arguments.get(key);
            if (value != null) {
                return tool + ": " + ConsoleRenderer.cut(String.valueOf(value), 120);
            }
        }
        return tool;
    }

    /**
     * What kind of tool this is, in ACP's terms, so the editor can pick an icon and a presentation.
     *
     * @param tool the tool name
     * @return the kind
     */
    static AcpSchema.ToolKind kind(String tool) {
        return switch (tool) {
            case "read_file", "ls" -> AcpSchema.ToolKind.READ;
            case "grep", "glob" -> AcpSchema.ToolKind.SEARCH;
            case "write_file", "edit_file" -> AcpSchema.ToolKind.EDIT;
            case "delete" -> AcpSchema.ToolKind.DELETE;
            case "rename" -> AcpSchema.ToolKind.MOVE;
            case ShellTool.TOOL_NAME -> AcpSchema.ToolKind.EXECUTE;
            default -> AcpSchema.ToolKind.OTHER;
        };
    }

    private @Nullable List<AcpSchema.ToolCallLocation> locations(@Nullable Map<String, Object> arguments) {
        if (arguments == null) {
            return null;
        }
        for (String key : new String[] {"file_path", "path"}) {
            Object value = arguments.get(key);
            if (value != null) {
                Path file = workspace.resolve(String.valueOf(value)).normalize();
                return List.of(new AcpSchema.ToolCallLocation(file.toString(), null));
            }
        }
        return null;
    }
}
