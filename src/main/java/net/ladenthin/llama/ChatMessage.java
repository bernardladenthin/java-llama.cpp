// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * A single message in a chat conversation: a role ({@code "user"}, {@code "assistant"},
 * or {@code "system"}) and its textual content. Used by {@link Session} to accumulate
 * conversation turns.
 */
public final class ChatMessage {

    private final String role;
    private final String content;

    /**
     * Construct a chat message.
     *
     * @param role    the message role: {@code "user"}, {@code "assistant"}, or {@code "system"}
     * @param content the message text
     */
    public ChatMessage(String role, String content) {
        this.role = role;
        this.content = content;
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

    @Override
    public String toString() {
        return role + ": " + content;
    }
}
