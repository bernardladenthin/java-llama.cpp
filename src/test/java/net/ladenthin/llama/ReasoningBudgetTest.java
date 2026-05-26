// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.util.Collections;

import net.ladenthin.llama.args.ReasoningFormat;
import net.ladenthin.llama.json.ChatResponseParser;
import org.junit.jupiter.api.AfterAll;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
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
 *   <li><b>{@code reasoning_budget_tokens} is NOT enforced for any model when set
 *       per-request.</b> The root cause is a bug in
 *       {@code tools/server/server-common.cpp}, function
 *       {@code oaicompat_chat_params_parse}: the reasoning-budget block writes
 *       the model-level default ({@code opt.reasoning_budget}, typically &#x2212;1)
 *       into {@code llama_params["reasoning_budget_tokens"]} before the generic
 *       copy loop runs. The copy loop then skips the per-request value from the
 *       request body because the key already exists
 *       ({@code !llama_params.contains(item.key())} is false). Result: the
 *       reasoning-budget sampler is never created (it requires
 *       {@code reasoning_budget_tokens &#x2265; 0}), and any per-request budget
 *       has no effect. Parameter serialisation itself is correct — see
 *       {@code InferenceParametersTest} and the C++ unit tests.</li>
 * </ol>
 */
@ClaudeGenerated(
        purpose = "Integration tests for Qwen3 thinking-mode extraction and reasoning_budget_tokens " +
                  "parameter acceptance. Documents the known llama.cpp limitation that budget " +
                  "enforcement does not work for prompt-injected thinking models."
)
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

    @BeforeClass
    public static void setup() {
        Assumptions.assumeTrue(new File(TestConstants.REASONING_MODEL_PATH).exists(), "Reasoning model not found, skipping ReasoningBudgetTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.REASONING_MODEL_PATH)
                        .setCtxSize(2048)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .setReasoningFormat(ReasoningFormat.DEEPSEEK)
                        .enableLogTimestamps().enableLogPrefix()
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Qwen3 enters thinking mode by default. With {@code reasoning_format=deepseek} set
     * at model level, the thinking tokens must appear in {@code reasoning_content} and
     * the final answer must appear in {@code content}.
     */
    @Test
    public void testThinkingDefault_reasoningContentAndAnswerPresent() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);

        assertFalse(reasoningContent == null || reasoningContent.trim().isEmpty(), "reasoning_content should be non-empty (Qwen3 thinks by default)");
        assertFalse(content == null || content.trim().isEmpty(), "content must not be empty (model must produce an answer after thinking)");
    }

    /**
     * {@code reasoning_budget_tokens=0} is accepted by the API and the response
     * completes without error, but the budget is NOT enforced.
     *
     * <p><b>Documents current (broken) behaviour.</b> The per-request value is
     * silently discarded by a bug in {@code tools/server/server-common.cpp}
     * ({@code oaicompat_chat_params_parse}): the reasoning-budget block writes the
     * model-level default (&#x2212;1) to {@code llama_params["reasoning_budget_tokens"]}
     * before the generic copy loop runs, and the copy loop then skips the user value
     * because the key already exists. The reasoning-budget sampler is therefore never
     * created, and {@code reasoning_content} remains non-empty.
     *
     * <p>This assertion will start <b>failing</b> once the llama.cpp bug is fixed —
     * that is the signal to remove this test and enable
     * {@link #testReasoningBudgetZero_expectedBehavior_suppressesThinking}.
     * Tracked in <a href="https://github.com/ggml-org/llama.cpp/pull/23116">llama.cpp PR #23116</a>.
     */
    @Test
    public void testReasoningBudgetZero_parameterAccepted_thinkingNotSuppressed() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setReasoningBudgetTokens(0)
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);

        assertNotNull(json, "Response JSON must not be null");

        String reasoningContent = parser.extractChoiceReasoningContent(json);
        assertFalse(reasoningContent == null || reasoningContent.trim().isEmpty(), "reasoning_content is expected to be present because the per-request " +
                "budget is not applied (llama.cpp server-common.cpp copy-loop bug). " +
                "If this assertion fails, the bug has been fixed — remove this test and " +
                "enable testReasoningBudgetZero_expectedBehavior_suppressesThinking.");
    }

    /**
     * Expected correct behaviour after the llama.cpp bug is fixed.
     *
     * <p><b>Bug:</b> In {@code tools/server/server-common.cpp},
     * {@code oaicompat_chat_params_parse} sets
     * {@code llama_params["reasoning_budget_tokens"]} to the model-level default
     * ({@code opt.reasoning_budget}, typically &#x2212;1) before the generic copy
     * loop runs. The copy loop then skips the per-request value from the request
     * body because the key already exists. Result: the sampler is never created
     * ({@code reasoning_budget_tokens &#x2265; 0} is required), and budget=0
     * has no effect.
     *
     * <p><b>Fix (server-common.cpp, reasoning budget block):</b>
     * Read {@code reasoning_budget_tokens} from the request body <em>before</em>
     * writing to {@code llama_params}:
     * <pre>
     * int reasoning_budget = opt.reasoning_budget;
     * if (body.contains("reasoning_budget_tokens")) {
     *     reasoning_budget = json_value(body, "reasoning_budget_tokens", reasoning_budget);
     * }
     * if (reasoning_budget == -1 &amp;&amp; body.contains("thinking_budget_tokens")) {
     *     reasoning_budget = json_value(body, "thinking_budget_tokens", -1);
     * }
     * </pre>
     *
     * <p>Once this fix is applied: remove {@code @Ignore}, confirm this test passes,
     * and remove
     * {@link #testReasoningBudgetZero_parameterAccepted_thinkingNotSuppressed}.
     * Tracked in <a href="https://github.com/ggml-org/llama.cpp/pull/23116">llama.cpp PR #23116</a>.
     */
    @Disabled("llama.cpp bug: per-request reasoning_budget_tokens is overwritten by model default " +
            "in oaicompat_chat_params_parse (server-common.cpp). " +
            "See Javadoc for exact fix location and code.")
    @Test
    public void testReasoningBudgetZero_expectedBehavior_suppressesThinking() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                .setReasoningBudgetTokens(0)
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        assertNotNull(json, "Response JSON must not be null");

        String reasoningContent = parser.extractChoiceReasoningContent(json);
        assertTrue(reasoningContent == null || reasoningContent.trim().isEmpty(), "reasoning_content should be empty when budget=0 suppresses thinking, " +
                "but was: " + reasoningContent);
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
     * <p>See {@link #testReasoningBudgetZero_parameterAccepted_thinkingNotSuppressed} for
     * the note on why the budget count itself is not asserted.
     */
    @Test
    public void testReasoningBudgetPositive_parameterAccepted() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, Collections.singletonList(
                        new Pair<>("user", "Think step by step: what is 3 times 7?")))
                .setReasoningBudgetTokens(100)
                .setNPredict(N_PREDICT);

        String json = model.chatComplete(params);
        assertNotNull(json, "Response JSON must not be null");

        String reasoningContent = parser.extractChoiceReasoningContent(json);
        String content = parser.extractChoiceContent(json);
        boolean hasReasoning = reasoningContent != null && !reasoningContent.trim().isEmpty();
        boolean hasContent   = content          != null && !content.trim().isEmpty();
        assertTrue(hasReasoning || hasContent, "model must produce at least some output in reasoning_content or content, " +
                "but both were empty");
    }
}
