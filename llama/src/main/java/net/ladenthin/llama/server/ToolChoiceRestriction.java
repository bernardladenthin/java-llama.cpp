// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * The tools a {@code tool_choice} object lets the model call, and whether it must call one: a named
 * function ({@code required}, that tool) or OpenAI's {@code allowed_tools} ({@code auto} or
 * {@code required}, a subset). Shared by the chat mapping and the Responses translation, which write the
 * same choice in two shapes ({@link Shape}).
 *
 * <p>llama.cpp reads both objects itself since b12731 (#31564), but only the formats its automatic
 * parser handles restrict the grammar to the named tools; the specialized parsers (Qwen3-Coder, Inkling,
 * the Harmony formats, ...) take every tool offered (measured at b12731 with the Qwen3-Coder format).
 * The translators therefore forward the allowed tools alone with the mode as a string, which restricts
 * the call for every format; the chat template then does not render the other tool definitions either,
 * which the model may not call anyway (OpenAI keeps them visible and constrains only the output). The
 * validation is upstream's: the mode is {@code auto} or {@code required}, every entry a function that
 * names exactly one supplied tool, a required set not empty, and an empty {@code auto} set is
 * {@code none}.
 */
@EqualsAndHashCode
@ToString
final class ToolChoiceRestriction {

    /** Where a request writes a function's name. */
    enum Shape {
        /** {@code /v1/chat/completions}: {@code {"type":"function","function":{"name":...}}}. */
        CHAT,
        /** {@code /v1/responses}: {@code {"type":"function","name":...}}. */
        RESPONSES
    }

    private final String mode;
    /** The names the choice allows, in the request's order; empty when it allows none. */
    private final Set<String> allowed;

    private ToolChoiceRestriction(String mode, Set<String> allowed) {
        this.mode = mode;
        this.allowed = Collections.unmodifiableSet(allowed);
    }

    /**
     * The restriction of a {@code tool_choice}, or {@code null} for an absent, {@code null} or string
     * one (the string goes on as it is; upstream's chat parser refuses one it does not know).
     *
     * @param toolChoice the request's {@code tool_choice}
     * @param shape where the request writes a function's name
     * @return the restriction, or {@code null}
     * @throws IllegalArgumentException for a value that is neither a string nor an object, a function
     *     without a name, a malformed {@code allowed_tools}, and an object of any other type
     */
    static @Nullable ToolChoiceRestriction parse(JsonNode toolChoice, Shape shape) {
        if (toolChoice.isMissingNode() || toolChoice.isNull() || toolChoice.isTextual()) {
            return null;
        }
        if (!toolChoice.isObject()) {
            throw new IllegalArgumentException("'tool_choice' must be a string or an object (got " + toolChoice + ")");
        }
        String type = toolChoice.path("type").asText("");
        if ("function".equals(type)) {
            JsonNode name = functionName(toolChoice, shape);
            if (!name.isTextual() || name.asText().isEmpty()) {
                throw new IllegalArgumentException(
                        "tool_choice of type 'function' needs the function's 'name' (got " + toolChoice + ")");
            }
            Set<String> named = new LinkedHashSet<>();
            named.add(name.asText());
            return new ToolChoiceRestriction("required", named);
        }
        if (!"allowed_tools".equals(type)) {
            throw new IllegalArgumentException("tool_choice type '" + type
                    + "' is not supported; use auto, none, required, a function or allowed_tools");
        }
        JsonNode allowedTools = shape == Shape.CHAT ? toolChoice.path("allowed_tools") : toolChoice;
        String mode = allowedTools.path("mode").asText("");
        JsonNode tools = allowedTools.path("tools");
        if (!allowedTools.path("mode").isTextual()
                || (!"auto".equals(mode) && !"required".equals(mode))
                || !tools.isArray()) {
            throw new IllegalArgumentException(
                    "allowed_tools requires mode auto or required and a tools array (got " + toolChoice + ")");
        }
        Set<String> names = new LinkedHashSet<>();
        for (JsonNode tool : tools) {
            if (!"function".equals(tool.path("type").asText(""))) {
                throw new IllegalArgumentException("allowed_tools entries must have type function (got " + tool + ")");
            }
            JsonNode name = functionName(tool, shape);
            if (!name.isTextual() || name.asText().isEmpty()) {
                throw new IllegalArgumentException(
                        "allowed_tools entries need the function's 'name' (got " + tool + ")");
            }
            names.add(name.asText());
        }
        if (names.isEmpty()) {
            if ("required".equals(mode)) {
                throw new IllegalArgumentException(
                        "Required allowed_tools must contain at least one function (got " + toolChoice + ")");
            }
            return new ToolChoiceRestriction("none", names);
        }
        return new ToolChoiceRestriction(mode, names);
    }

    private static JsonNode functionName(JsonNode function, Shape shape) {
        return shape == Shape.CHAT ? function.path("function").path("name") : function.path("name");
    }

    /**
     * The {@code tool_choice} string llama.cpp gets with the allowed tools.
     *
     * @return {@code required}, {@code auto} or {@code none}
     */
    String mode() {
        return mode;
    }

    /**
     * Whether a tool goes to the model: one the choice names, or every tool when it names none (an
     * empty {@code auto} set, which forbids every call).
     *
     * @param name the tool's name
     * @return {@code true} when the tool is forwarded
     */
    boolean forwards(String name) {
        return allowed.isEmpty() || allowed.contains(name);
    }

    /**
     * Check every name of the choice against the tools of the request: each must name exactly one.
     *
     * @param toolNames the names of the request's tools, in order
     * @throws IllegalArgumentException for a name no tool has, or more than one
     */
    void checkAgainst(List<String> toolNames) {
        for (String name : allowed) {
            int matching = Collections.frequency(toolNames, name);
            if (matching == 0) {
                throw new IllegalArgumentException("tool_choice names unknown tool '" + name + "'");
            }
            if (matching > 1) {
                throw new IllegalArgumentException(
                        "tool_choice names tool '" + name + "', which more than one supplied tool has");
            }
        }
    }
}
