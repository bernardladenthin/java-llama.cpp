// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import lombok.EqualsAndHashCode;
import lombok.ToString;

/**
 * Declaration of a tool/function the model is allowed to call. Mirrors the OpenAI
 * chat-completions {@code tools[i].function} object: a name, a human-readable description,
 * and a JSON Schema for the function's parameters.
 * <p>
 * The schema must be valid JSON Schema (object form); it is sent verbatim to the native
 * server and propagates into the chat template / grammar driver.
 * </p>
 * <p>
 * A <em>deferred</em> tool ({@link #isDeferLoading()}, llama.cpp b11940) is withheld from the
 * system prompt: its definition only reaches the model once a message names it with a
 * {@link ContentPart#toolReference(String) tool reference}, the way a tool-search tool reveals tools
 * one at a time instead of the whole catalogue up front. Templates without deferred-tool support
 * render it like any other tool.
 * </p>
 */
@ToString
@EqualsAndHashCode
public final class ToolDefinition {

    private final String name;
    private final String description;
    private final String parametersSchemaJson;
    private final boolean deferLoading;

    /**
     * Construct a tool definition that is in the system prompt from the start.
     *
     * @param name                 the function name
     * @param description          human-readable description shown to the model
     * @param parametersSchemaJson JSON Schema for the function parameters, as a JSON string
     */
    public ToolDefinition(String name, String description, String parametersSchemaJson) {
        this(name, description, parametersSchemaJson, false);
    }

    /**
     * Construct a tool definition, optionally deferred.
     *
     * @param name                 the function name
     * @param description          human-readable description shown to the model
     * @param parametersSchemaJson JSON Schema for the function parameters, as a JSON string
     * @param deferLoading         {@code true} to withhold the tool from the system prompt until a
     *                             {@link ContentPart#toolReference(String) tool reference} names it
     */
    public ToolDefinition(String name, String description, String parametersSchemaJson, boolean deferLoading) {
        this.name = name;
        this.description = description;
        this.parametersSchemaJson = parametersSchemaJson;
        this.deferLoading = deferLoading;
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

    /**
     * Whether the tool is deferred: sent as {@code "defer_loading": true} and withheld from the
     * system prompt until a {@link ContentPart#toolReference(String) tool reference} names it.
     *
     * @return {@code true} for a deferred tool
     */
    public boolean isDeferLoading() {
        return deferLoading;
    }
}
