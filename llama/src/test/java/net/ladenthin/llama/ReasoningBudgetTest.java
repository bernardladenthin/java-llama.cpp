// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.util.Collections;
import net.ladenthin.llama.args.ReasoningFormat;
import net.ladenthin.llama.json.ChatResponseParser;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for thinking/reasoning mode using Qwen3-0.6B-Q4_K_M.
 *
 * <p>These tests require the Qwen3-0.6B-Q4_K_M model (downloaded by CI). The entire
 * class is skipped when the model file is absent, matching the pattern used by all
 * other model-dependent test classes.
 *
 * <h2>Confirmed behaviour (Qwen3-0.6B, llama.cpp b9151)</h2>
 * <ol>
 *   <li><b>Thinking is active by default.</b> Qwen3's built-in chat template injects
 *       {@code <think>} into the prompt before generation starts. No extra kwarg is
 *       required; the model reasons on every request.</li>
 *   <li><b>DEEPSEEK reasoning format correctly extracts thinking tokens.</b> Setting
 *       {@code --reasoning-format deepseek} at model load time causes the server to
 *       strip the {@code <think>…</think>} block from the response body and surface it
 *       in {@code reasoning_content}.</li>
 *   <li><b>{@code reasoning_budget_tokens} IS enforced per-request.</b> This was originally
 *       broken in {@code tools/server/server-common.cpp} ({@code oaicompat_chat_params_parse}):
 *       the reasoning-budget block wrote the model-level default into
 *       {@code llama_params["reasoning_budget_tokens"]} before the generic copy loop, which then
 *       skipped the per-request value because the key already existed, so the reasoning-budget
 *       sampler was never created. It is fixed by upstream PR #23116, carried here as
 *       {@code patches/0004-pr23116-server-per-request-reasoning-budget-tokens.patch} (drop the
 *       patch once a pinned {@code b<nnnn>} includes it). With the fix,
 *       {@code reasoning_budget_tokens=0} suppresses thinking. Parameter serialisation is covered
 *       by {@code InferenceParametersTest} and the C++ unit tests.</li>
 * </ol>
 */
@ClaudeGenerated(
        purpose = "Integration tests for Qwen3 thinking-mode extraction and per-request "
                + "reasoning_budget_tokens enforcement (fixed via patches/0004, upstream PR #23116): "
                + "budget=0 suppresses thinking.")
public class ReasoningBudgetTest {

    /**
     * Generous token budget: Qwen3-0.6B typically spends ~200 tokens thinking before
     * answering, but on slow/contended CI runners (e.g. 2-thread GitHub-hosted x86_64)
     * the model occasionally rambles past 500 tokens while still inside the
     * {@code <think>} block, leaving {@code content} empty and failing
     * {@link #testThinkingDefault_reasoningContentAndAnswerPresent}. 1500 leaves
     * comfortable headroom for thinking + a short answer across all tested platforms.
     */
    private static final int N_PREDICT = 1500;

    private static LlamaModel model;
    private final ChatResponseParser parser = new ChatResponseParser();

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping ReasoningBudgetTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setCtxSize(2048)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .setReasoningFormat(ReasoningFormat.DEEPSEEK)
                .enableLogTimestamps()
                .enableLogPrefix());
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Qwen3 enters thinking mode by default. With {@code reasoning_format=deepseek} set
     * at model level, the thinking tokens must appear in {@code reasoning_content} and
     * the final answer must appear in {@code content}.
     *
     * <p>{@code temperature=0} (greedy sampling) is used so the model deterministically
     * enters the {@code <think>} block on every platform, including Metal (macOS arm64)
     * where GPU floating-point arithmetic can produce slightly different logit
     * distributions and occasionally sample a non-thinking first token.
     */
    @Test
    public void testThinkingDefault_reasoningContentAndAnswerPresent() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .withTemperature(0.0f)
                .withNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);

        assertFalse(
                reasoningContent == null || reasoningContent.trim().isEmpty(),
                "reasoning_content should be non-empty (Qwen3 thinks by default)");
        assertFalse(
                content == null || content.trim().isEmpty(),
                "content must not be empty (model must produce an answer after thinking)");
    }

    /**
     * Per-request {@code reasoning_budget_tokens=0} suppresses thinking: the model emits an
     * empty {@code reasoning_content}.
     *
     * <p>The per-request budget is honored by upstream
     * <a href="https://github.com/ggml-org/llama.cpp/pull/23116">llama.cpp PR #23116</a>, carried
     * in this repo as {@code patches/0004-pr23116-server-per-request-reasoning-budget-tokens.patch}
     * until a pinned {@code b<nnnn>} includes it. Before that fix,
     * {@code oaicompat_chat_params_parse} ({@code tools/server/server-common.cpp}) wrote the
     * model-level default into {@code llama_params["reasoning_budget_tokens"]} before the generic
     * copy loop, so the per-request value was dropped and the reasoning-budget sampler was never
     * created. With the fix, {@code budget=0} forces the end-of-thinking sequence immediately.
     *
     * <p>{@code temperature=0} (greedy) keeps the first-token choice deterministic across
     * platforms (notably macOS Metal), so the result does not depend on sampling. Parameter
     * serialisation is covered separately by {@code InferenceParametersTest} and the C++ unit tests.
     */
    @Test
    public void testReasoningBudgetZero_suppressesThinking() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .withTemperature(0.0f)
                .withReasoningBudgetTokens(0)
                .withNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        assertNotNull(json, "Response JSON must not be null");

        String reasoningContent = parser.extractChoiceReasoningContent(json);
        assertTrue(
                reasoningContent == null || reasoningContent.trim().isEmpty(),
                "reasoning_content must be empty when reasoning_budget_tokens=0 suppresses thinking, " + "but was: "
                        + reasoningContent);
    }

    /**
     * A positive {@code reasoning_budget_tokens} value is accepted and the call completes
     * without error.
     *
     * <p>The assertion checks that the model produced a non-empty response — either in
     * {@code reasoning_content} or {@code content}. On slow or constrained hardware the
     * model may exhaust the token budget inside the thinking block and emit an empty
     * {@code content}; checking both fields makes the test robust to that behaviour.
     *
     * <p>The exact number of thinking tokens consumed is not asserted — it is hardware- and
     * sampling-dependent; {@link #testReasoningBudgetZero_suppressesThinking} covers the
     * deterministic {@code budget=0} suppression case.
     */
    @Test
    public void testReasoningBudgetPositive_parameterAccepted() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(
                        null, Collections.singletonList(new Pair<>("user", "Think step by step: what is 3 times 7?")))
                .withReasoningBudgetTokens(100)
                .withNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        assertNotNull(json, "Response JSON must not be null");

        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);
        boolean hasReasoning =
                reasoningContent != null && !reasoningContent.trim().isEmpty();
        boolean hasContent = content != null && !content.trim().isEmpty();
        assertTrue(
                hasReasoning || hasContent,
                "model must produce at least some output in reasoning_content or content, " + "but both were empty");
    }
}
