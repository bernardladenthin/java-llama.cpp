// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Collections;
import java.util.List;

/**
 * Typed result of {@link LlamaModel#chat(ChatRequest)} and
 * {@link LlamaModel#chatWithTools(ChatRequest, java.util.Map)}.
 * <p>
 * Bundles the OpenAI-style {@code id} and {@code choices} array with the per-completion
 * {@link Usage} and {@link Timings} parsed from the response, plus a passthrough to the
 * raw OAI JSON for fields not yet typed.
 * </p>
 */
public final class ChatResponse {

    private final String id;
    private final List<ChatChoice> choices;
    private final Usage usage;
    private final Timings timings;
    private final String rawJson;

    /**
     * Construct a chat response.
     *
     * @param id      the OAI id (may be empty)
     * @param choices the choices array (never {@code null})
     * @param usage   parsed usage counters (never {@code null})
     * @param timings parsed timings (never {@code null})
     * @param rawJson the raw OAI response JSON string
     */
    public ChatResponse(String id, List<ChatChoice> choices, Usage usage, Timings timings, String rawJson) {
        this.id = id;
        this.choices = choices == null ? Collections.<ChatChoice>emptyList() : choices;
        this.usage = usage;
        this.timings = timings;
        this.rawJson = rawJson;
    }

    /**
     * Response id accessor.
     * @return the OAI {@code id} field, or {@code ""} if absent
     */
    public String getId() {
        return id;
    }

    /**
     * Choices accessor.
     * @return the choices array; empty when the response carried no choices
     */
    public List<ChatChoice> getChoices() {
        return choices;
    }

    /**
     * Convenience accessor for the first assistant message.
     * @return the first choice's message, or {@code null} when there are no choices
     */
    public ChatMessage getFirstMessage() {
        return choices.isEmpty() ? null : choices.get(0).getMessage();
    }

    /**
     * Convenience accessor for the first assistant text content.
     * @return the first choice's message content, or {@code ""} when there are no choices
     */
    public String getFirstContent() {
        ChatMessage m = getFirstMessage();
        return m == null ? "" : m.getContent();
    }

    /**
     * Usage accessor.
     * @return parsed token-count usage for this completion
     */
    public Usage getUsage() {
        return usage;
    }

    /**
     * Timings accessor.
     * @return parsed timings for this completion
     */
    public Timings getTimings() {
        return timings;
    }

    /**
     * Raw JSON accessor.
     * @return the raw OAI response JSON string
     */
    public String getRawJson() {
        return rawJson;
    }
}
