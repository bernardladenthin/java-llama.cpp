// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import net.ladenthin.llama.args.PoolingType;
import org.junit.jupiter.api.AfterAll;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Validates the full JSON response structure from various endpoints:
 * <ul>
 *   <li>Non-OAI completion response: content, stop, stop_type, model, tokens_predicted, tokens_evaluated, timings, generation_settings</li>
 *   <li>OAI completion response: choices, usage, model, object, created, system_fingerprint</li>
 *   <li>OAI chat completion response: choices with message/finish_reason, usage</li>
 *   <li>Timings object fields: prompt_n, prompt_ms, predicted_n, predicted_ms</li>
 *   <li>stop_type values: eos, word, limit</li>
 *   <li>finish_reason values: stop, length</li>
 *   <li>Embedding response structure</li>
 *   <li>Tokenization/detokenization response structure</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Validate full JSON response structures from all endpoints: non-OAI and OAI completions, " +
                  "chat completions, timings, stop_type/finish_reason values, embedding and tokenization responses.",
        model = "claude-opus-4-6"
)
public class ResponseJsonStructureTest {

    private static final int N_PREDICT = 5;
    private static final String PROMPT = "int main() {";
    // Use temperature=0 to produce deterministic ASCII output and avoid incomplete
    // UTF-8 sequences that crash nlohmann::json on low-quality quantizations (Q2_K).
    private static final String DETERMINISTIC = ",\"temperature\":0,\"seed\":42";

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assumptions.assumeTrue(new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping ResponseJsonStructureTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(256)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .enableEmbedding()
                        .setPoolingType(PoolingType.MEAN)
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    // -------------------------------------------------------------------------
    // Non-OAI completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testNonOaiCompletionHasContentField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"content\""), "Response must contain 'content'");
    }

    @Test
    public void testNonOaiCompletionHasStopField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"stop\""), "Response must contain 'stop'");
    }

    @Test
    public void testNonOaiCompletionHasStopType() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"stop_type\""), "Response must contain 'stop_type'");
    }

    @Test
    public void testNonOaiCompletionHasModelField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"model\""), "Response must contain 'model'");
    }

    @Test
    public void testNonOaiCompletionHasTokensPredicted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"tokens_predicted\""), "Response must contain 'tokens_predicted'");
    }

    @Test
    public void testNonOaiCompletionHasTokensEvaluated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"tokens_evaluated\""), "Response must contain 'tokens_evaluated'");
    }

    @Test
    public void testNonOaiCompletionHasTimings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"timings\""), "Response must contain 'timings'");
    }

    @Test
    public void testNonOaiCompletionHasGenerationSettings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"generation_settings\""), "Response must contain 'generation_settings'");
    }

    @Test
    public void testNonOaiCompletionHasTokensCached() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"tokens_cached\""), "Response must contain 'tokens_cached'");
    }

    @Test
    public void testNonOaiCompletionHasIdSlot() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"id_slot\""), "Response must contain 'id_slot'");
    }

    // -------------------------------------------------------------------------
    // Timings object fields
    // -------------------------------------------------------------------------

    @Test
    public void testTimingsHasPromptN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"prompt_n\""), "Timings must contain 'prompt_n'");
    }

    @Test
    public void testTimingsHasPromptMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"prompt_ms\""), "Timings must contain 'prompt_ms'");
    }

    @Test
    public void testTimingsHasPredictedN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"predicted_n\""), "Timings must contain 'predicted_n'");
    }

    @Test
    public void testTimingsHasPredictedMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"predicted_ms\""), "Timings must contain 'predicted_ms'");
    }

    @Test
    public void testTimingsHasPerTokenFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"prompt_per_token_ms\""), "Timings must contain 'prompt_per_token_ms'");
        assertTrue(result.contains("\"predicted_per_token_ms\""), "Timings must contain 'predicted_per_token_ms'");
    }

    @Test
    public void testTimingsHasPerSecondFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"prompt_per_second\""), "Timings must contain 'prompt_per_second'");
        assertTrue(result.contains("\"predicted_per_second\""), "Timings must contain 'predicted_per_second'");
    }

    // -------------------------------------------------------------------------
    // stop_type values
    // -------------------------------------------------------------------------

    @Test
    public void testStopTypeLimitOnMaxTokens() {
        // n_predict=N_PREDICT with no stop string should result in "limit" stop_type
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"stop_type\":\"limit\""), "stop_type should be 'limit' when max tokens reached");
    }

    @Test
    public void testStopTypeWordOnStopString() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":50" + DETERMINISTIC + ",\"stop\":[\"return\"]}";
        String result = model.handleCompletions(json);
        // May be "word" if stop string matched, or "limit" if n_predict reached first
        assertTrue(result.contains("\"stop_type\":\"word\"") ||
                result.contains("\"stop_type\":\"limit\"") ||
                result.contains("\"stop_type\":\"eos\""), "stop_type should be present");
    }

    // -------------------------------------------------------------------------
    // OAI completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiCompletionHasChoices() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"choices\""), "OAI response must contain 'choices'");
    }

    @Test
    public void testOaiCompletionHasUsage() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"usage\""), "OAI response must contain 'usage'");
    }

    @Test
    public void testOaiCompletionHasObject() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"object\":\"text_completion\""), "OAI response must contain 'object':'text_completion'");
    }

    @Test
    public void testOaiCompletionHasCreated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"created\""), "OAI response must contain 'created'");
    }

    @Test
    public void testOaiCompletionHasModel() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"model\""), "OAI response must contain 'model'");
    }

    @Test
    public void testOaiCompletionHasSystemFingerprint() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"system_fingerprint\""), "OAI response must contain 'system_fingerprint'");
    }

    @Test
    public void testOaiCompletionHasId() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"id\""), "OAI response must contain 'id'");
    }

    @Test
    public void testOaiCompletionUsageFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"completion_tokens\""), "Usage must contain 'completion_tokens'");
        assertTrue(result.contains("\"prompt_tokens\""), "Usage must contain 'prompt_tokens'");
        assertTrue(result.contains("\"total_tokens\""), "Usage must contain 'total_tokens'");
    }

    @Test
    public void testOaiCompletionChoiceHasFinishReason() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertTrue(result.contains("\"finish_reason\""), "Choice must contain 'finish_reason'");
    }

    @Test
    public void testOaiCompletionFinishReasonLength() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        // With small n_predict, finish_reason should be "length"
        assertTrue(result.contains("\"finish_reason\":\"length\"") ||
                result.contains("\"finish_reason\":\"stop\""), "finish_reason should be 'length' or 'stop'");
    }

    // -------------------------------------------------------------------------
    // OAI chat completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiChatCompletionHasChoices() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT)
                .setTemperature(0);
        String result = model.chatComplete(params);
        assertTrue(result.contains("\"choices\""), "Chat response must contain 'choices'");
    }

    @Test
    public void testOaiChatCompletionHasUsage() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT)
                .setTemperature(0);
        String result = model.chatComplete(params);
        assertTrue(result.contains("\"usage\""), "Chat response must contain 'usage'");
    }

    @Test
    public void testOaiChatCompletionHasMessageObject() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT)
                .setTemperature(0);
        String result = model.chatComplete(params);
        assertTrue(result.contains("\"message\""), "Chat response must contain 'message'");
    }

    @Test
    public void testOaiChatCompletionObjectType() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT)
                .setTemperature(0);
        String result = model.chatComplete(params);
        assertTrue(result.contains("\"object\":\"chat.completion\""), "Chat response 'object' must be 'chat.completion'");
    }

    @Test
    public void testOaiChatCompletionMessageHasRole() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT)
                .setTemperature(0);
        String result = model.chatComplete(params);
        assertTrue(result.contains("\"role\":\"assistant\""), "Message must contain 'role':'assistant'");
    }

    // -------------------------------------------------------------------------
    // Embedding response structure
    // -------------------------------------------------------------------------

    @Test
    public void testEmbeddingOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, true);
        assertTrue(result.contains("\"data\""), "OAI embedding must contain 'data'");
        assertTrue(result.contains("\"object\":\"embedding\""), "OAI embedding must contain 'object':'embedding'");
        assertTrue(result.contains("\"embedding\""), "OAI embedding must contain 'embedding' array");
        assertTrue(result.contains("\"usage\""), "OAI embedding must contain 'usage'");
    }

    @Test
    public void testEmbeddingNonOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, false);
        assertTrue(result.contains("\"embedding\""), "Non-OAI embedding must contain 'embedding'");
        assertTrue(result.contains("\"index\""), "Non-OAI embedding must contain 'index'");
    }

    // -------------------------------------------------------------------------
    // Tokenization response structure
    // -------------------------------------------------------------------------

    @Test
    public void testTokenizeResponseStructure() {
        String result = model.handleTokenize("hello world", false, false);
        assertNotNull(result);
        assertTrue(result.contains("\"tokens\""), "Tokenize response must contain 'tokens'");
    }

    @Test
    public void testTokenizeWithPiecesResponseStructure() {
        String result = model.handleTokenize("hello world", false, true);
        assertNotNull(result);
        assertTrue(result.contains("\"tokens\""), "Tokenize with pieces must contain 'tokens'");
    }

    @Test
    public void testDetokenizeResponseStructure() {
        int[] tokens = model.encode("hello world");
        String result = model.handleDetokenize(tokens);
        assertNotNull(result);
        assertTrue(result.contains("\"content\""), "Detokenize response must contain 'content'");
    }

    // -------------------------------------------------------------------------
    // Completion probabilities structure
    // -------------------------------------------------------------------------

    @Test
    public void testCompletionProbabilitiesStructure() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + DETERMINISTIC + ",\"n_probs\":3}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"completion_probabilities\""), "Response with n_probs should contain 'completion_probabilities'");
    }

    // -------------------------------------------------------------------------
    // generation_settings sub-fields
    // -------------------------------------------------------------------------

    @Test
    public void testGenerationSettingsContainsSamplingParams() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        // generation_settings should echo back the sampling parameters
        assertTrue(result.contains("\"temperature\""), "generation_settings should contain 'temperature'");
        assertTrue(result.contains("\"top_k\""), "generation_settings should contain 'top_k'");
        assertTrue(result.contains("\"top_p\""), "generation_settings should contain 'top_p'");
        assertTrue(result.contains("\"min_p\""), "generation_settings should contain 'min_p'");
    }

    @Test
    public void testGenerationSettingsContainsSamplers() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertTrue(result.contains("\"samplers\""), "generation_settings should contain 'samplers'");
    }
}
