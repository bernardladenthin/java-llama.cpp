// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Declaration of a tool/function the model is allowed to call. Mirrors the OpenAI
 * chat-completions {@code tools[i].function} object: a name, a human-readable description,
 * and a JSON Schema for the function's parameters.
 * <p>
 * The schema must be valid JSON Schema (object form); it is sent verbatim to the native
 * server and propagates into the chat template / grammar driver.
 * </p>
 */
public final class ToolDefinition {

    private final String name;
    private final String description;
    private final String parametersSchemaJson;

    /**
     * Construct a tool definition.
     *
     * @param name                 the function name
     * @param description          human-readable description shown to the model
     * @param parametersSchemaJson JSON Schema for the function parameters, as a JSON string
     */
    public ToolDefinition(String name, String description, String parametersSchemaJson) {
        this.name = name;
        this.description = description;
        this.parametersSchemaJson = parametersSchemaJson;
    }

    /**
     * Function name accessor.
     * @return the function name
     */
    public String getName() {
        return name;
    }

    /**
     * Description accessor.
     * @return the human-readable description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Parameters schema accessor.
     * @return the JSON Schema string for the function's parameters
     */
    public String getParametersSchemaJson() {
        return parametersSchemaJson;
    }
}
