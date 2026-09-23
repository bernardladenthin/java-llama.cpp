// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.List;
import java.util.Map;
import org.atmosphere.ai.AgentExecutionContext;
import org.atmosphere.ai.AiConfig;
import org.atmosphere.ai.ExecutionHandle;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.approval.ApprovalStrategy;
import org.atmosphere.ai.approval.ToolApprovalPolicy;
import org.atmosphere.ai.llm.BuiltInAgentRuntime;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.llm.ToolLoopPolicies;
import org.atmosphere.ai.llm.ToolLoopPolicy;
import org.atmosphere.ai.tool.ToolDefinition;
import org.jspecify.annotations.Nullable;

/**
 * The minimal wiring between Atmosphere's built-in OpenAI-compatible agent runtime and an
 * OpenAI-compatible base URL — no Spring Boot, no servlet container, no {@code @Agent} scanning.
 *
 * <p>One instance is one configured endpoint plus one tool set. Each {@link #run} call is one user turn:
 * Atmosphere streams the model's answer into the session, executes every {@code tool_calls} round
 * through the registered {@link ToolDefinition} executors, re-submits the tool results, and completes
 * the session when the model produces a final answer (or the round cap is hit).
 *
 * <p>Atmosphere resolves its settings through a process-wide {@link AiConfig} singleton; constructing
 * a runner (re)configures it, so build one runner per endpoint and reuse it.
 */
public final class AgentRunner {

    private final BuiltInAgentRuntime runtime;
    private final String modelId;
    private final List<ToolDefinition> tools;
    private final String systemPrompt;
    private final int maxToolRounds;
    private RetryPolicy retryPolicy = RetryPolicy.DEFAULT;
    private @Nullable ApprovalStrategy approvalStrategy;
    private @Nullable ToolApprovalPolicy approvalPolicy;

    /**
     * Configure the runtime for one endpoint.
     *
     * @param baseUrl the OpenAI-compatible base URL, e.g. {@code http://127.0.0.1:8080/v1}
     * @param apiKey the bearer token (sent as {@code Authorization: Bearer ...})
     * @param modelId the model id carried in every request
     * @param tools the tools offered to the model on every turn
     * @param systemPrompt the system prompt
     * @param temperature the sampling temperature
     * @param maxTokens the {@code max_tokens} budget per model call
     * @param maxToolRounds the tool-round cap per turn
     */
    public AgentRunner(
            String baseUrl,
            String apiKey,
            String modelId,
            List<ToolDefinition> tools,
            String systemPrompt,
            double temperature,
            int maxTokens,
            int maxToolRounds) {
        // GenerationParams are read from system properties when the settings are built.
        System.setProperty(AiConfig.TEMPERATURE_PROPERTY, Double.toString(temperature));
        System.setProperty(AiConfig.MAX_TOKENS_PROPERTY, Integer.toString(maxTokens));
        // "local" mode: no provider auto-detection, and an explicit base URL always wins.
        AiConfig.LlmSettings settings = AiConfig.configure("local", modelId, apiKey, baseUrl);
        this.runtime = new BuiltInAgentRuntime();
        this.runtime.configure(settings);
        this.modelId = modelId;
        this.tools = List.copyOf(tools);
        this.systemPrompt = systemPrompt;
        this.maxToolRounds = maxToolRounds;
    }

    /**
     * Replace the HTTP retry policy (default: Atmosphere's, which retries 429/5xx and connection
     * failures). Tests use {@link RetryPolicy#NONE} to make a failing endpoint fail fast.
     *
     * @param retryPolicy the policy
     * @return this runner
     */
    public AgentRunner retryPolicy(RetryPolicy retryPolicy) {
        this.retryPolicy = retryPolicy;
        return this;
    }

    /**
     * Gate the tools {@code policy} selects behind {@code strategy}: Atmosphere's tool loop then blocks
     * on the strategy before such a tool runs, and turns a denial into a {@code cancelled} tool result
     * for the model on its own.
     *
     * <p>Without this, no tool is gated — Atmosphere's default policy honours a tool's own
     * {@code requiresApproval()}, and none of this agent's tools set it.
     *
     * @param strategy what asks the user, e.g. {@link ConsoleApprovalStrategy}
     * @param policy which tools it is asked about, e.g. {@link ConsoleApprovalStrategy#policy()}
     * @return this runner
     */
    public AgentRunner approval(ApprovalStrategy strategy, ToolApprovalPolicy policy) {
        this.approvalStrategy = strategy;
        this.approvalPolicy = policy;
        return this;
    }

    /**
     * The model ids the endpoint advertises on {@code GET /v1/models}, falling back to the configured
     * id when enumeration fails.
     *
     * @return the model ids
     */
    public List<String> models() {
        return runtime.models();
    }

    /**
     * The tool names offered on every turn.
     *
     * @return the names in registration order
     */
    public List<String> toolNames() {
        return tools.stream().map(ToolDefinition::name).toList();
    }

    /**
     * Run one user turn to completion. Returns when the session has been completed or errored.
     *
     * @param message the user message
     * @param history prior turns ({@code user}/{@code assistant} messages), replayed before the message
     * @param session receives streamed text, tool events and the terminal complete/error
     */
    public void run(String message, List<ChatMessage> history, StreamingSession session) {
        runtime.execute(context(message, history, tools, systemPrompt, session), session);
    }

    /**
     * Start one user turn and return at once, with a handle that can stop it.
     *
     * <p>This is the same turn {@link #run} performs, on Atmosphere's cancellation-aware entry point:
     * the turn runs on a virtual thread of the framework's, and {@link ExecutionHandle#cancel()}
     * closes the HTTP stream the model is answering on, which unblocks the read loop. That is what
     * lets a request typed while the agent is working take effect immediately instead of at the end
     * of a tool loop that may run for minutes.
     *
     * @param message the user message
     * @param history prior turns, replayed before the message
     * @param session receives streamed text, tool events and the terminal complete/error
     * @return the handle; the session's own completion stays the signal that the turn is over
     */
    public ExecutionHandle start(String message, List<ChatMessage> history, StreamingSession session) {
        return runtime.executeWithHandle(context(message, history, tools, systemPrompt, session), session);
    }

    /**
     * Run one turn with no tools at all and a system prompt of its own — what {@code /compact} needs:
     * a summary must not read files or run commands, it must only condense what is already there.
     *
     * @param message the user message
     * @param history prior turns, replayed before the message
     * @param session receives the streamed summary
     * @param systemPrompt the system prompt for this one turn
     */
    public void runWithoutTools(
            String message, List<ChatMessage> history, StreamingSession session, String systemPrompt) {
        runtime.execute(context(message, history, List.of(), systemPrompt, session), session);
    }

    private AgentExecutionContext context(
            String message,
            List<ChatMessage> history,
            List<ToolDefinition> tools,
            String systemPrompt,
            StreamingSession session) {
        AgentExecutionContext context = new AgentExecutionContext(
                message,
                systemPrompt,
                modelId,
                null,
                session.sessionId(),
                null,
                null,
                tools,
                null,
                null,
                List.of(),
                Map.of(),
                history,
                null,
                null);
        context = context.withRetryPolicy(retryPolicy);
        if (approvalStrategy != null && approvalPolicy != null) {
            context = context.withApprovalStrategy(approvalStrategy).withApprovalPolicy(approvalPolicy);
        }
        return ToolLoopPolicies.attach(context, ToolLoopPolicy.maxIterations(maxToolRounds));
    }
}
