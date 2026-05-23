// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * One choice in a chat completion response: the assistant message and the finish reason.
 * Mirrors the OpenAI {@code choices[i]} object.
 */
public final class ChatChoice {

    private final int index;
    private final ChatMessage message;
    private final String finishReason;

    /**
     * Construct a chat choice.
     *
     * @param index        the index in the choices array
     * @param message      the assistant's message for this choice
     * @param finishReason the finish reason (e.g. {@code "stop"}, {@code "length"}, {@code "tool_calls"})
     */
    public ChatChoice(int index, ChatMessage message, String finishReason) {
        this.index = index;
        this.message = message;
        this.finishReason = finishReason;
    }

    /**
     * Choice index.
     * @return the integer index in the choices array
     */
    public int getIndex() {
        return index;
    }

    /**
     * Assistant message accessor.
     * @return the assistant's reply (may include tool_calls)
     */
    public ChatMessage getMessage() {
        return message;
    }

    /**
     * Finish reason accessor.
     * @return the OAI finish reason string, or {@code ""} if absent
     */
    public String getFinishReason() {
        return finishReason;
    }
}
