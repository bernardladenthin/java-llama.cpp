// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * A single tool/function call issued by the assistant. Mirrors the OpenAI chat-completions
 * {@code tool_calls[i]} object: an id, a function name, and the arguments as a JSON string.
 * <p>
 * Arguments are surfaced verbatim as the JSON string the model emitted; callers parse them
 * with their preferred JSON library (or hand them to a {@link ToolHandler}).
 * </p>
 */
public final class ToolCall {

    private final String id;
    private final String name;
    private final String argumentsJson;

    /**
     * Construct a tool call.
     *
     * @param id            the OpenAI-style call id used to correlate the matching tool result
     * @param name          the function name
     * @param argumentsJson the function arguments as a JSON-encoded string
     */
    public ToolCall(String id, String name, String argumentsJson) {
        this.id = id;
        this.name = name;
        this.argumentsJson = argumentsJson;
    }

    /**
     * Call id accessor.
     * @return the OpenAI-style call id
     */
    public String getId() {
        return id;
    }

    /**
     * Function name accessor.
     * @return the function name
     */
    public String getName() {
        return name;
    }

    /**
     * Arguments accessor.
     * @return the arguments as a JSON-encoded string
     */
    public String getArgumentsJson() {
        return argumentsJson;
    }

    @Override
    public String toString() {
        return name + "(" + argumentsJson + ")[" + id + "]";
    }
}
