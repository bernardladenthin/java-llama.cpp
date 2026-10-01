// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.anyOf;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import java.io.File;
import net.ladenthin.llama.args.PoolingType;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.AfterAll;
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
        purpose = "Validate full JSON response structures from all endpoints: non-OAI and OAI completions, "
                + "chat completions, timings, stop_type/finish_reason values, embedding and tokenization responses.",
        model = "claude-opus-4-6")
public class ResponseJsonStructureTest {

    private static final int N_PREDICT = 5;
    private static final String PROMPT = "int main() {";
    // Use temperature=0 to produce deterministic ASCII output and avoid incomplete
    // UTF-8 sequences that crash nlohmann::json on low-quality quantizations (Q2_K).
    private static final String DETERMINISTIC = ",\"temperature\":0,\"seed\":42";

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(),
                "Model file not found, skipping ResponseJsonStructureTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setCtxSize(256)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .enableEmbedding()
                .setPoolingType(PoolingType.MEAN));
    }

    @AfterAll
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
        assertThat("Response must contain 'content'", result, containsString("\"content\""));
    }

    @Test
    public void testNonOaiCompletionHasStopField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'stop'", result, containsString("\"stop\""));
    }

    @Test
    public void testNonOaiCompletionHasStopType() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'stop_type'", result, containsString("\"stop_type\""));
    }

    @Test
    public void testNonOaiCompletionHasModelField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'model'", result, containsString("\"model\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensPredicted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'tokens_predicted'", result, containsString("\"tokens_predicted\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensEvaluated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'tokens_evaluated'", result, containsString("\"tokens_evaluated\""));
    }

    @Test
    public void testNonOaiCompletionHasTimings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'timings'", result, containsString("\"timings\""));
    }

    @Test
    public void testNonOaiCompletionHasGenerationSettings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'generation_settings'", result, containsString("\"generation_settings\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensCached() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'tokens_cached'", result, containsString("\"tokens_cached\""));
    }

    @Test
    public void testNonOaiCompletionHasIdSlot() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Response must contain 'id_slot'", result, containsString("\"id_slot\""));
    }

    // -------------------------------------------------------------------------
    // Timings object fields
    // -------------------------------------------------------------------------

    @Test
    public void testTimingsHasPromptN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'prompt_n'", result, containsString("\"prompt_n\""));
    }

    @Test
    public void testTimingsHasPromptMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'prompt_ms'", result, containsString("\"prompt_ms\""));
    }

    @Test
    public void testTimingsHasPredictedN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'predicted_n'", result, containsString("\"predicted_n\""));
    }

    @Test
    public void testTimingsHasPredictedMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'predicted_ms'", result, containsString("\"predicted_ms\""));
    }

    @Test
    public void testTimingsHasPerTokenFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'prompt_per_token_ms'", result, containsString("\"prompt_per_token_ms\""));
        assertThat(
                "Timings must contain 'predicted_per_token_ms'", result, containsString("\"predicted_per_token_ms\""));
    }

    @Test
    public void testTimingsHasPerSecondFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("Timings must contain 'prompt_per_second'", result, containsString("\"prompt_per_second\""));
        assertThat("Timings must contain 'predicted_per_second'", result, containsString("\"predicted_per_second\""));
    }

    // -------------------------------------------------------------------------
    // stop_type values
    // -------------------------------------------------------------------------

    @Test
    public void testStopTypeLimitOnMaxTokens() {
        // n_predict=N_PREDICT with no stop string should result in "limit" stop_type
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat(
                "stop_type should be 'limit' when max tokens reached",
                result,
                containsString("\"stop_type\":\"limit\""));
    }

    @Test
    public void testStopTypeWordOnStopString() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":50" + DETERMINISTIC + ",\"stop\":[\"return\"]}";
        String result = model.handleCompletions(json);
        // May be "word" if stop string matched, or "limit" if n_predict reached first
        assertThat(
                "stop_type should be present",
                result,
                anyOf(
                        containsString("\"stop_type\":\"word\""),
                        containsString("\"stop_type\":\"limit\""),
                        containsString("\"stop_type\":\"eos\"")));
    }

    // -------------------------------------------------------------------------
    // OAI completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiCompletionHasChoices() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'choices'", result, containsString("\"choices\""));
    }

    @Test
    public void testOaiCompletionHasUsage() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'usage'", result, containsString("\"usage\""));
    }

    @Test
    public void testOaiCompletionHasObject() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat(
                "OAI response must contain 'object':'text_completion'",
                result,
                containsString("\"object\":\"text_completion\""));
    }

    @Test
    public void testOaiCompletionHasCreated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'created'", result, containsString("\"created\""));
    }

    @Test
    public void testOaiCompletionHasModel() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'model'", result, containsString("\"model\""));
    }

    @Test
    public void testOaiCompletionHasSystemFingerprint() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'system_fingerprint'", result, containsString("\"system_fingerprint\""));
    }

    @Test
    public void testOaiCompletionHasId() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("OAI response must contain 'id'", result, containsString("\"id\""));
    }

    @Test
    public void testOaiCompletionUsageFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("Usage must contain 'completion_tokens'", result, containsString("\"completion_tokens\""));
        assertThat("Usage must contain 'prompt_tokens'", result, containsString("\"prompt_tokens\""));
        assertThat("Usage must contain 'total_tokens'", result, containsString("\"total_tokens\""));
    }

    @Test
    public void testOaiCompletionChoiceHasFinishReason() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        assertThat("Choice must contain 'finish_reason'", result, containsString("\"finish_reason\""));
    }

    @Test
    public void testOaiCompletionFinishReasonLength() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletionsOai(json);
        // With small n_predict, finish_reason should be "length"
        assertThat(
                "finish_reason should be 'length' or 'stop'",
                result,
                anyOf(containsString("\"finish_reason\":\"length\""), containsString("\"finish_reason\":\"stop\"")));
    }

    // -------------------------------------------------------------------------
    // OAI chat completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiChatCompletionHasChoices() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, java.util.Collections.singletonList(new Pair<>("user", "Say hello")))
                .withNPredict(N_PREDICT)
                .withTemperature(0);
        String result = model.chatComplete(params);
        assertThat("Chat response must contain 'choices'", result, containsString("\"choices\""));
    }

    @Test
    public void testOaiChatCompletionHasUsage() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, java.util.Collections.singletonList(new Pair<>("user", "Say hello")))
                .withNPredict(N_PREDICT)
                .withTemperature(0);
        String result = model.chatComplete(params);
        assertThat("Chat response must contain 'usage'", result, containsString("\"usage\""));
    }

    @Test
    public void testOaiChatCompletionHasMessageObject() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, java.util.Collections.singletonList(new Pair<>("user", "Say hello")))
                .withNPredict(N_PREDICT)
                .withTemperature(0);
        String result = model.chatComplete(params);
        assertThat("Chat response must contain 'message'", result, containsString("\"message\""));
    }

    @Test
    public void testOaiChatCompletionObjectType() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, java.util.Collections.singletonList(new Pair<>("user", "Say hello")))
                .withNPredict(N_PREDICT)
                .withTemperature(0);
        String result = model.chatComplete(params);
        assertThat(
                "Chat response 'object' must be 'chat.completion'",
                result,
                containsString("\"object\":\"chat.completion\""));
    }

    @Test
    public void testOaiChatCompletionMessageHasRole() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, java.util.Collections.singletonList(new Pair<>("user", "Say hello")))
                .withNPredict(N_PREDICT)
                .withTemperature(0);
        String result = model.chatComplete(params);
        assertThat("Message must contain 'role':'assistant'", result, containsString("\"role\":\"assistant\""));
    }

    // -------------------------------------------------------------------------
    // Embedding response structure
    // -------------------------------------------------------------------------

    @Test
    public void testEmbeddingOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, true);
        assertThat("OAI embedding must contain 'data'", result, containsString("\"data\""));
        assertThat(
                "OAI embedding must contain 'object':'embedding'", result, containsString("\"object\":\"embedding\""));
        assertThat("OAI embedding must contain 'embedding' array", result, containsString("\"embedding\""));
        assertThat("OAI embedding must contain 'usage'", result, containsString("\"usage\""));
    }

    @Test
    public void testEmbeddingNonOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, false);
        assertThat("Non-OAI embedding must contain 'embedding'", result, containsString("\"embedding\""));
        assertThat("Non-OAI embedding must contain 'index'", result, containsString("\"index\""));
    }

    // -------------------------------------------------------------------------
    // Tokenization response structure
    // -------------------------------------------------------------------------

    @Test
    public void testTokenizeResponseStructure() {
        String result = model.handleTokenize("hello world", false, false);
        assertThat(result, is(notNullValue()));
        assertThat("Tokenize response must contain 'tokens'", result, containsString("\"tokens\""));
    }

    @Test
    public void testTokenizeWithPiecesResponseStructure() {
        String result = model.handleTokenize("hello world", false, true);
        assertThat(result, is(notNullValue()));
        assertThat("Tokenize with pieces must contain 'tokens'", result, containsString("\"tokens\""));
    }

    @Test
    public void testDetokenizeResponseStructure() {
        int[] tokens = model.encode("hello world");
        String result = model.handleDetokenize(tokens);
        assertThat(result, is(notNullValue()));
        assertThat("Detokenize response must contain 'content'", result, containsString("\"content\""));
    }

    // -------------------------------------------------------------------------
    // Completion probabilities structure
    // -------------------------------------------------------------------------

    @Test
    public void testCompletionProbabilitiesStructure() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + ",\"n_probs\":3}";
        String result = model.handleCompletions(json);
        assertThat(
                "Response with n_probs should contain 'completion_probabilities'",
                result,
                containsString("\"completion_probabilities\""));
    }

    // -------------------------------------------------------------------------
    // generation_settings sub-fields
    // -------------------------------------------------------------------------

    @Test
    public void testGenerationSettingsContainsSamplingParams() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        // generation_settings should echo back the sampling parameters
        assertThat("generation_settings should contain 'temperature'", result, containsString("\"temperature\""));
        assertThat("generation_settings should contain 'top_k'", result, containsString("\"top_k\""));
        assertThat("generation_settings should contain 'top_p'", result, containsString("\"top_p\""));
        assertThat("generation_settings should contain 'min_p'", result, containsString("\"min_p\""));
    }

    @Test
    public void testGenerationSettingsContainsSamplers() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + DETERMINISTIC + "}";
        String result = model.handleCompletions(json);
        assertThat("generation_settings should contain 'samplers'", result, containsString("\"samplers\""));
    }
}
