// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Invocation contract for a tool registered with
 * {@link LlamaModel#chatWithTools(ChatRequest, java.util.Map)}.
 * <p>
 * The handler receives the model-supplied arguments as a JSON string and returns the
 * tool's output as a JSON string (an unwrapped string literal also works). Exceptions
 * thrown by the handler are reported back to the model as a {@code {"error":"..."}}
 * tool result so the agent loop can continue rather than aborting the request.
 * </p>
 */
@FunctionalInterface
public interface ToolHandler {

    /**
     * Invoke the tool.
     *
     * @param argumentsJson the arguments emitted by the model, as a JSON-encoded string
     * @return the tool's output (any JSON-serializable string)
     * @throws Exception when the tool fails; reported back to the model as an error result
     */
    String invoke(String argumentsJson) throws Exception;
}
