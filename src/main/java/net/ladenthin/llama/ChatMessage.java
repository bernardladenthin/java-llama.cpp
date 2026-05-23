// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Collections;
import java.util.List;

/**
 * A single message in a chat conversation: a role ({@code "user"}, {@code "assistant"},
 * {@code "system"}, or {@code "tool"}) and its textual content. Used by {@link Session}
 * to accumulate conversation turns and by {@link ChatRequest} / {@link ChatResponse}
 * for the typed chat API.
 * <p>
 * Tool-call turns have role {@code "assistant"}, possibly empty content, and a non-empty
 * {@link #getToolCalls()} list. Tool-result turns have role {@code "tool"}, the tool's
 * output as content, and {@link #getToolCallId()} pointing back at the originating call.
 * </p>
 */
public final class ChatMessage {

    private final String role;
    private final String content;
    private final String toolCallId;
    private final List<ToolCall> toolCalls;

    /**
     * Plain user/assistant/system message.
     *
     * @param role    the message role
     * @param content the message text
     */
    public ChatMessage(String role, String content) {
        this(role, content, null, Collections.<ToolCall>emptyList());
    }

    /**
     * Full constructor including tool-related fields.
     *
     * @param role       the message role
     * @param content    the message text (may be empty for assistant tool-call turns)
     * @param toolCallId for tool-result turns ({@code role="tool"}), the id of the originating call; {@code null} otherwise
     * @param toolCalls  for assistant tool-call turns, the list of calls; empty otherwise
     */
    public ChatMessage(String role, String content, String toolCallId, List<ToolCall> toolCalls) {
        this.role = role;
        this.content = content;
        this.toolCallId = toolCallId;
        this.toolCalls = toolCalls == null ? Collections.<ToolCall>emptyList() : toolCalls;
    }

    /**
     * Factory for a tool-result turn.
     *
     * @param toolCallId the id of the originating tool call
     * @param content    the tool's output as a string
     * @return a {@link ChatMessage} with role {@code "tool"}
     */
    public static ChatMessage toolResult(String toolCallId, String content) {
        return new ChatMessage("tool", content, toolCallId, Collections.<ToolCall>emptyList());
    }

    /**
     * Factory for an assistant turn that issues tool calls.
     *
     * @param content   optional reasoning text accompanying the tool calls (may be empty)
     * @param toolCalls the tool calls to issue
     * @return a {@link ChatMessage} with role {@code "assistant"}
     */
    public static ChatMessage assistantToolCalls(String content, List<ToolCall> toolCalls) {
        return new ChatMessage("assistant", content == null ? "" : content, null, toolCalls);
    }

    /**
     * Message role accessor.
     * @return the message role string
     */
    public String getRole() {
        return role;
    }

    /**
     * Message content accessor.
     * @return the message text content
     */
    public String getContent() {
        return content;
    }

    /**
     * Tool-call id for tool-result turns.
     * @return the originating tool call id, or {@code null} for non-tool messages
     */
    public String getToolCallId() {
        return toolCallId;
    }

    /**
     * Tool calls issued by an assistant turn.
     * @return the calls list, never {@code null}; empty when the message is not a tool-call turn
     */
    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    @Override
    public String toString() {
        if (!toolCalls.isEmpty()) return role + " (tool_calls=" + toolCalls.size() + "): " + content;
        if (toolCallId != null)   return role + " (tool_call_id=" + toolCallId + "): " + content;
        return role + ": " + content;
    }
}
