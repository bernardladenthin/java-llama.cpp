// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

/**
 * Builder for a typed chat completion call.
 * <p>
 * Bundles the conversation messages, optional tool definitions, an optional
 * {@code tool_choice} hint, and an {@link InferenceParameters} customizer that gets
 * applied to the underlying request just before invocation. Built with the fluent
 * setters; consumed by {@link LlamaModel#chat(ChatRequest)} and
 * {@link LlamaModel#chatWithTools(ChatRequest, java.util.Map)}.
 * </p>
 */
public final class ChatRequest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final List<ChatMessage> messages = new ArrayList<ChatMessage>();
    private final List<ToolDefinition> tools = new ArrayList<ToolDefinition>();
    private String toolChoice;
    private int maxToolRounds = 8;
    private Consumer<InferenceParameters> paramsCustomizer;

    /** Construct an empty request; populate via the setters. */
    public ChatRequest() {
        // empty
    }

    /**
     * Append a message to the conversation.
     * @param message the message to append
     * @return this builder
     */
    public ChatRequest addMessage(ChatMessage message) {
        messages.add(message);
        return this;
    }

    /**
     * Convenience for adding a system/user/assistant turn.
     * @param role    the role
     * @param content the content
     * @return this builder
     */
    public ChatRequest addMessage(String role, String content) {
        messages.add(new ChatMessage(role, content));
        return this;
    }

    /**
     * Append a tool definition.
     * @param tool the tool definition to expose to the model
     * @return this builder
     */
    public ChatRequest addTool(ToolDefinition tool) {
        tools.add(tool);
        return this;
    }

    /**
     * Set the {@code tool_choice} hint: typically {@code "auto"}, {@code "none"}, or
     * {@code "required"}. Defaults to absent (server default applies).
     *
     * @param toolChoice the hint string, or {@code null} to clear
     * @return this builder
     */
    public ChatRequest setToolChoice(String toolChoice) {
        this.toolChoice = toolChoice;
        return this;
    }

    /**
     * Set the maximum number of agent-loop rounds for
     * {@link LlamaModel#chatWithTools(ChatRequest, java.util.Map)}. A round is one
     * model call followed by zero or more tool invocations. Default {@code 8}.
     *
     * @param maxToolRounds the round cap (must be positive)
     * @return this builder
     */
    public ChatRequest setMaxToolRounds(int maxToolRounds) {
        if (maxToolRounds <= 0) {
            throw new IllegalArgumentException("maxToolRounds must be > 0");
        }
        this.maxToolRounds = maxToolRounds;
        return this;
    }

    /**
     * Register a callback that customizes the {@link InferenceParameters} (e.g.
     * {@code setNPredict}, {@code setTemperature}) right before each request is sent.
     *
     * @param customizer the customizer; {@code null} clears any prior customizer
     * @return this builder
     */
    public ChatRequest setInferenceCustomizer(Consumer<InferenceParameters> customizer) {
        this.paramsCustomizer = customizer;
        return this;
    }

    /**
     * Messages accessor.
     * @return an unmodifiable view of the messages added so far
     */
    public List<ChatMessage> getMessages() {
        return Collections.unmodifiableList(messages);
    }

    /**
     * Tools accessor.
     * @return an unmodifiable view of the tool definitions added so far
     */
    public List<ToolDefinition> getTools() {
        return Collections.unmodifiableList(tools);
    }

    /**
     * Tool choice accessor.
     * @return the {@code tool_choice} hint, or {@code null} when unset
     */
    public String getToolChoice() {
        return toolChoice;
    }

    /**
     * Max rounds accessor.
     * @return the agent-loop round cap
     */
    public int getMaxToolRounds() {
        return maxToolRounds;
    }

    /**
     * Build the OAI-style {@code messages} array as a JSON string. Each entry carries
     * role and content; assistant tool-call turns add a {@code tool_calls} array; tool-
     * result turns add a {@code tool_call_id} field.
     *
     * @return the JSON array as a string
     */
    public String buildMessagesJson() {
        ArrayNode arr = MAPPER.createArrayNode();
        for (ChatMessage m : messages) {
            ObjectNode obj = MAPPER.createObjectNode();
            obj.put("role", m.getRole());
            obj.put("content", m.getContent() == null ? "" : m.getContent());
            if (m.getToolCallId() != null) {
                obj.put("tool_call_id", m.getToolCallId());
            }
            if (!m.getToolCalls().isEmpty()) {
                ArrayNode tc = MAPPER.createArrayNode();
                for (ToolCall call : m.getToolCalls()) {
                    ObjectNode entry = MAPPER.createObjectNode();
                    entry.put("id", call.getId());
                    entry.put("type", "function");
                    ObjectNode fn = MAPPER.createObjectNode();
                    fn.put("name", call.getName());
                    fn.put("arguments", call.getArgumentsJson() == null ? "" : call.getArgumentsJson());
                    entry.set("function", fn);
                    tc.add(entry);
                }
                obj.set("tool_calls", tc);
            }
            arr.add(obj);
        }
        return arr.toString();
    }

    /**
     * Build the OAI-style {@code tools} array as a JSON string. Returns {@code null}
     * when no tools were added.
     *
     * @return the JSON array as a string, or {@code null} when there are no tools
     */
    public String buildToolsJson() {
        if (tools.isEmpty()) return null;
        ArrayNode arr = MAPPER.createArrayNode();
        for (ToolDefinition t : tools) {
            ObjectNode entry = MAPPER.createObjectNode();
            entry.put("type", "function");
            ObjectNode fn = MAPPER.createObjectNode();
            fn.put("name", t.getName());
            if (t.getDescription() != null) fn.put("description", t.getDescription());
            try {
                fn.set("parameters", MAPPER.readTree(t.getParametersSchemaJson()));
            } catch (Exception e) {
                fn.put("parameters", t.getParametersSchemaJson());
            }
            entry.set("function", fn);
            arr.add(entry);
        }
        return arr.toString();
    }

    /**
     * Apply the optional customizer to an {@link InferenceParameters} instance.
     * Package-private; called by {@link LlamaModel}.
     *
     * @param params the parameters to mutate
     */
    void applyCustomizer(InferenceParameters params) {
        if (paramsCustomizer != null) {
            paramsCustomizer.accept(params);
        }
    }
}
