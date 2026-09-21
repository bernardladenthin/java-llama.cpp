// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.List;
import java.util.Map;
import org.atmosphere.ai.AgentExecutionContext;
import org.atmosphere.ai.AiConfig;
import org.atmosphere.ai.RetryPolicy;
import org.atmosphere.ai.StreamingSession;
import org.atmosphere.ai.llm.BuiltInAgentRuntime;
import org.atmosphere.ai.llm.ChatMessage;
import org.atmosphere.ai.llm.ToolLoopPolicies;
import org.atmosphere.ai.llm.ToolLoopPolicy;
import org.atmosphere.ai.tool.ToolDefinition;

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
        context = ToolLoopPolicies.attach(context, ToolLoopPolicy.maxIterations(maxToolRounds));
        runtime.execute(context, session);
    }
}
