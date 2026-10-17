// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.UnaryOperator;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ToolDefinition;
import org.jspecify.annotations.Nullable;

/**
 * Immutable typed chat-completion request, populated through a functional
 * "wither / appender" API.
 *
 * <h2>Design</h2>
 *
 * <p>The request carries the conversation messages, optional tool definitions,
 * an optional {@code tool_choice} hint, and an {@link InferenceParameters}
 * customiser applied to the underlying request just before invocation. Because
 * {@link InferenceParameters} is itself immutable, the customiser is a
 * {@link UnaryOperator} that takes a parameter set and returns the transformed
 * one — callers chain {@code withX(...)} calls on the input and return the
 * resulting instance. The type is consumed by
 * {@link net.ladenthin.llama.LlamaModel#chat(ChatRequest)} and
 * {@link net.ladenthin.llama.LlamaModel#chatWithTools(ChatRequest, java.util.Map)}.
 *
 * <p>All instances are <b>immutable</b>: every field is {@code final} and the
 * stored lists are wrapped with {@link Collections#unmodifiableList(List)}.
 * Modification methods return a <b>new</b> {@code ChatRequest} instance with
 * the requested change applied; the original is untouched. This makes
 * {@code ChatRequest} safe to share across threads and gives it a meaningful
 * value-equality semantics (two requests with the same content compare
 * equal regardless of identity).
 *
 * <h2>Construction patterns</h2>
 *
 * <p>Use {@link #empty()} as the entry point, then chain {@code append*}
 * (for list fields) and {@code with*} (for scalar fields):
 *
 * <pre>{@code
 * ChatRequest req = ChatRequest.empty()
 *         .appendMessage("system", "be terse")
 *         .appendMessage("user", "two plus two?")
 *         .withMaxToolRounds(2)
 *         .withInferenceCustomizer(p -> p.withNPredict(8).withSeed(1));
 * }</pre>
 *
 * <p>Each call allocates a new {@code ChatRequest}. The cost is intentional:
 * the API is functional, so a caller can hold an intermediate request and
 * derive variants without worrying about hidden state changes.
 *
 * <h2>Equality</h2>
 *
 * <p>{@code @EqualsAndHashCode} compares messages, tools, {@code toolChoice},
 * {@code parallelToolCalls}, and {@code maxToolRounds} by value. The {@code paramsCustomizer}
 * {@link UnaryOperator} is <b>excluded</b> from equality: lambdas have
 * compiler-synthesised identity equality which is not value-shaped, so
 * including it would mean two structurally-identical requests with the same
 * customiser source code rarely compare equal — surprising for the typical
 * snapshot-testing and caching use cases. The customiser is also excluded
 * from {@link ToString} for the same reason (the rendered hash is noise).
 */
@ToString
@EqualsAndHashCode
public final class ChatRequest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // Reads a tool's parameter schema. FAIL_ON_TRAILING_TOKENS makes a string holding more than one
    // JSON value take the fallback below instead of being truncated to its first value without a
    // word -- the same reason JsonParameters reads with it.
    private static final ObjectReader SCHEMA_READER =
            MAPPER.reader().with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    private static final ParameterJsonSerializer MESSAGE_SERIALIZER = new ParameterJsonSerializer();

    /**
     * Default {@code maxToolRounds} when the caller does not override it via
     * {@link #withMaxToolRounds(int)}. Mirrors the prior mutable builder's default.
     */
    public static final int DEFAULT_MAX_TOOL_ROUNDS = 8;

    private static final ChatRequest EMPTY = new ChatRequest(
            Collections.<ChatMessage>emptyList(),
            Collections.<ToolDefinition>emptyList(),
            null,
            null,
            DEFAULT_MAX_TOOL_ROUNDS,
            null);

    private final List<ChatMessage> messages;
    private final List<ToolDefinition> tools;
    private final @Nullable String toolChoice;
    private final @Nullable Boolean parallelToolCalls;
    private final int maxToolRounds;

    // Lambda Consumer — toString is the implementation hash, not useful in logs;
    // equality is compiler-synthesised class identity, not value-shaped.
    @ToString.Exclude
    @EqualsAndHashCode.Exclude
    private final @Nullable UnaryOperator<InferenceParameters> paramsCustomizer;

    /**
     * All-args constructor. Private because callers should enter via {@link #empty()}
     * and derive variants via the {@code append*} / {@code with*} methods. Each
     * variant call routes through this same constructor with one field replaced.
     */
    private ChatRequest(
            List<ChatMessage> messages,
            List<ToolDefinition> tools,
            @Nullable String toolChoice,
            @Nullable Boolean parallelToolCalls,
            int maxToolRounds,
            @Nullable UnaryOperator<InferenceParameters> paramsCustomizer) {
        this.messages = messages;
        this.tools = tools;
        this.toolChoice = toolChoice;
        this.parallelToolCalls = parallelToolCalls;
        this.maxToolRounds = maxToolRounds;
        this.paramsCustomizer = paramsCustomizer;
    }

    /**
     * Returns the empty request — no messages, no tools, {@code toolChoice}
     * absent, {@code maxToolRounds} = {@value #DEFAULT_MAX_TOOL_ROUNDS}, no
     * customiser. Acts as the starting point for chained derivations.
     *
     * @return the empty request
     */
    public static ChatRequest empty() {
        return EMPTY;
    }

    // -----------------------------------------------------------------------
    // List appends — each returns a new request with one entry added.
    // -----------------------------------------------------------------------

    /**
     * Returns a new request with {@code message} appended to the conversation.
     *
     * @param message the message to append
     * @return a new request with the appended message; this request is unchanged
     */
    public ChatRequest appendMessage(ChatMessage message) {
        List<ChatMessage> next = new ArrayList<ChatMessage>(messages.size() + 1);
        next.addAll(messages);
        next.add(message);
        return new ChatRequest(
                Collections.unmodifiableList(next),
                tools,
                toolChoice,
                parallelToolCalls,
                maxToolRounds,
                paramsCustomizer);
    }

    /**
     * Convenience for {@link #appendMessage(ChatMessage)} that wraps a role +
     * content pair into a new {@link net.ladenthin.llama.value.ChatMessage} and appends it.
     *
     * @param role    the role (e.g. {@code "system"}, {@code "user"}, {@code "assistant"})
     * @param content the message content
     * @return a new request with the appended message; this request is unchanged
     */
    public ChatRequest appendMessage(String role, String content) {
        return appendMessage(new ChatMessage(role, content));
    }

    /**
     * Returns a new request with {@code tool} added to the tool registry.
     *
     * @param tool the tool to expose to the model
     * @return a new request with the appended tool; this request is unchanged
     */
    public ChatRequest appendTool(ToolDefinition tool) {
        List<ToolDefinition> next = new ArrayList<ToolDefinition>(tools.size() + 1);
        next.addAll(tools);
        next.add(tool);
        return new ChatRequest(
                messages,
                Collections.unmodifiableList(next),
                toolChoice,
                parallelToolCalls,
                maxToolRounds,
                paramsCustomizer);
    }

    // -----------------------------------------------------------------------
    // Scalar withers — each returns a new request with one field replaced.
    // -----------------------------------------------------------------------

    /**
     * Returns a new request with the {@code tool_choice} hint replaced.
     *
     * @param newToolChoice the hint string (typically {@code "auto"}, {@code "none"}, or
     *     {@code "required"}), or {@code null} to clear
     * @return a new request with the hint replaced; this request is unchanged
     */
    public ChatRequest withToolChoice(@Nullable String newToolChoice) {
        return new ChatRequest(messages, tools, newToolChoice, parallelToolCalls, maxToolRounds, paramsCustomizer);
    }

    /**
     * Returns a new request with the {@code parallel_tool_calls} hint replaced.
     *
     * @param newParallelToolCalls whether the model may emit multiple calls in one turn,
     *     or {@code null} to use the model/template default
     * @return a new request with the hint replaced; this request is unchanged
     */
    public ChatRequest withParallelToolCalls(@Nullable Boolean newParallelToolCalls) {
        return new ChatRequest(messages, tools, toolChoice, newParallelToolCalls, maxToolRounds, paramsCustomizer);
    }

    /**
     * Returns a new request with the agent-loop round cap replaced.
     *
     * @param newMaxToolRounds the new round cap (must be {@code > 0})
     * @return a new request with the cap replaced; this request is unchanged
     * @throws IllegalArgumentException if {@code newMaxToolRounds} is non-positive
     */
    public ChatRequest withMaxToolRounds(int newMaxToolRounds) {
        if (newMaxToolRounds <= 0) {
            throw new IllegalArgumentException("maxToolRounds must be > 0 but was " + newMaxToolRounds);
        }
        return new ChatRequest(messages, tools, toolChoice, parallelToolCalls, newMaxToolRounds, paramsCustomizer);
    }

    /**
     * Returns a new request with the inference-parameter customiser replaced.
     *
     * @param newCustomizer the customiser; {@code null} clears any prior customiser
     * @return a new request with the customiser replaced; this request is unchanged
     */
    public ChatRequest withInferenceCustomizer(@Nullable UnaryOperator<InferenceParameters> newCustomizer) {
        return new ChatRequest(messages, tools, toolChoice, parallelToolCalls, maxToolRounds, newCustomizer);
    }

    // -----------------------------------------------------------------------
    // Accessors.
    // -----------------------------------------------------------------------

    /**
     * Messages accessor.
     *
     * @return an unmodifiable view of the messages accumulated so far
     */
    public List<ChatMessage> getMessages() {
        return messages;
    }

    /**
     * Tools accessor.
     *
     * @return an unmodifiable view of the tool definitions accumulated so far
     */
    public List<ToolDefinition> getTools() {
        return tools;
    }

    /**
     * Tool-choice hint accessor.
     *
     * @return the {@code tool_choice} hint, or {@link Optional#empty()} when unset
     */
    public Optional<String> getToolChoice() {
        return Optional.ofNullable(toolChoice);
    }

    /**
     * Parallel-tool-call hint accessor.
     *
     * @return the {@code parallel_tool_calls} hint, or {@link Optional#empty()} when unset
     */
    public Optional<Boolean> getParallelToolCalls() {
        return Optional.ofNullable(parallelToolCalls);
    }

    /**
     * Agent-loop round cap accessor.
     *
     * @return the agent-loop round cap
     */
    public int getMaxToolRounds() {
        return maxToolRounds;
    }

    // -----------------------------------------------------------------------
    // JSON build helpers — read-only, do not mutate this request.
    // -----------------------------------------------------------------------

    /**
     * Build the OAI-style {@code messages} array as a JSON string. Each entry carries
     * role and content; assistant tool-call turns add a {@code tool_calls} array; tool-
     * result turns add a {@code tool_call_id} field.
     *
     * @return the JSON array as a string
     */
    public String buildMessagesJson() {
        return MESSAGE_SERIALIZER.buildMessages(messages).toString();
    }

    /**
     * Build the OAI-style {@code tools} array as a JSON string.
     *
     * @return the JSON array as a string, or {@link Optional#empty()} when no tools were added
     */
    public Optional<String> buildToolsJson() {
        if (tools.isEmpty()) return Optional.empty();
        ArrayNode arr = MAPPER.createArrayNode();
        for (ToolDefinition t : tools) {
            ObjectNode entry = MAPPER.createObjectNode();
            entry.put("type", "function");
            ObjectNode fn = MAPPER.createObjectNode();
            fn.put("name", t.getName());
            fn.put("description", t.getDescription());
            try {
                fn.set("parameters", SCHEMA_READER.readTree(t.getParametersSchemaJson()));
            } catch (IOException e) {
                fn.put("parameters", t.getParametersSchemaJson());
            }
            entry.set("function", fn);
            arr.add(entry);
        }
        return Optional.of(arr.toString());
    }

    /**
     * Apply the optional customiser to an {@link InferenceParameters} instance and
     * return the transformed result. Package-private; called by {@link net.ladenthin.llama.LlamaModel}.
     * When no customiser is set, returns {@code params} unchanged.
     *
     * @param params the parameters to transform
     * @return the (possibly new) parameters produced by the customiser, or {@code params} when no customiser is set
     */
    public InferenceParameters applyCustomizer(InferenceParameters params) {
        return paramsCustomizer == null ? params : paramsCustomizer.apply(params);
    }
}
