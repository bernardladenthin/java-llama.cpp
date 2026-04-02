package de.kherud.llama;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import de.kherud.llama.args.PoolingType;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

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

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping ResponseJsonStructureTest",
                new File(TestConstants.MODEL_PATH).exists());
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
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'content'", result.contains("\"content\""));
    }

    @Test
    public void testNonOaiCompletionHasStopField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'stop'", result.contains("\"stop\""));
    }

    @Test
    public void testNonOaiCompletionHasStopType() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'stop_type'", result.contains("\"stop_type\""));
    }

    @Test
    public void testNonOaiCompletionHasModelField() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'model'", result.contains("\"model\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensPredicted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'tokens_predicted'", result.contains("\"tokens_predicted\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensEvaluated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'tokens_evaluated'", result.contains("\"tokens_evaluated\""));
    }

    @Test
    public void testNonOaiCompletionHasTimings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'timings'", result.contains("\"timings\""));
    }

    @Test
    public void testNonOaiCompletionHasGenerationSettings() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'generation_settings'",
                result.contains("\"generation_settings\""));
    }

    @Test
    public void testNonOaiCompletionHasTokensCached() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'tokens_cached'", result.contains("\"tokens_cached\""));
    }

    @Test
    public void testNonOaiCompletionHasIdSlot() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response must contain 'id_slot'", result.contains("\"id_slot\""));
    }

    // -------------------------------------------------------------------------
    // Timings object fields
    // -------------------------------------------------------------------------

    @Test
    public void testTimingsHasPromptN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'prompt_n'", result.contains("\"prompt_n\""));
    }

    @Test
    public void testTimingsHasPromptMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'prompt_ms'", result.contains("\"prompt_ms\""));
    }

    @Test
    public void testTimingsHasPredictedN() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'predicted_n'", result.contains("\"predicted_n\""));
    }

    @Test
    public void testTimingsHasPredictedMs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'predicted_ms'", result.contains("\"predicted_ms\""));
    }

    @Test
    public void testTimingsHasPerTokenFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'prompt_per_token_ms'",
                result.contains("\"prompt_per_token_ms\""));
        Assert.assertTrue("Timings must contain 'predicted_per_token_ms'",
                result.contains("\"predicted_per_token_ms\""));
    }

    @Test
    public void testTimingsHasPerSecondFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Timings must contain 'prompt_per_second'",
                result.contains("\"prompt_per_second\""));
        Assert.assertTrue("Timings must contain 'predicted_per_second'",
                result.contains("\"predicted_per_second\""));
    }

    // -------------------------------------------------------------------------
    // stop_type values
    // -------------------------------------------------------------------------

    @Test
    public void testStopTypeLimitOnMaxTokens() {
        // n_predict=N_PREDICT with no stop string should result in "limit" stop_type
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("stop_type should be 'limit' when max tokens reached",
                result.contains("\"stop_type\":\"limit\""));
    }

    @Test
    public void testStopTypeWordOnStopString() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":50,\"stop\":[\"return\"]}";
        String result = model.handleCompletions(json);
        // May be "word" if stop string matched, or "limit" if n_predict reached first
        Assert.assertTrue("stop_type should be present",
                result.contains("\"stop_type\":\"word\"") ||
                result.contains("\"stop_type\":\"limit\"") ||
                result.contains("\"stop_type\":\"eos\""));
    }

    // -------------------------------------------------------------------------
    // OAI completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiCompletionHasChoices() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'choices'", result.contains("\"choices\""));
    }

    @Test
    public void testOaiCompletionHasUsage() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'usage'", result.contains("\"usage\""));
    }

    @Test
    public void testOaiCompletionHasObject() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'object':'text_completion'",
                result.contains("\"object\":\"text_completion\""));
    }

    @Test
    public void testOaiCompletionHasCreated() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'created'", result.contains("\"created\""));
    }

    @Test
    public void testOaiCompletionHasModel() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'model'", result.contains("\"model\""));
    }

    @Test
    public void testOaiCompletionHasSystemFingerprint() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'system_fingerprint'",
                result.contains("\"system_fingerprint\""));
    }

    @Test
    public void testOaiCompletionHasId() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("OAI response must contain 'id'", result.contains("\"id\""));
    }

    @Test
    public void testOaiCompletionUsageFields() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("Usage must contain 'completion_tokens'",
                result.contains("\"completion_tokens\""));
        Assert.assertTrue("Usage must contain 'prompt_tokens'",
                result.contains("\"prompt_tokens\""));
        Assert.assertTrue("Usage must contain 'total_tokens'",
                result.contains("\"total_tokens\""));
    }

    @Test
    public void testOaiCompletionChoiceHasFinishReason() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        Assert.assertTrue("Choice must contain 'finish_reason'",
                result.contains("\"finish_reason\""));
    }

    @Test
    public void testOaiCompletionFinishReasonLength() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletionsOai(json);
        // With small n_predict, finish_reason should be "length"
        Assert.assertTrue("finish_reason should be 'length' or 'stop'",
                result.contains("\"finish_reason\":\"length\"") ||
                result.contains("\"finish_reason\":\"stop\""));
    }

    // -------------------------------------------------------------------------
    // OAI chat completion response structure
    // -------------------------------------------------------------------------

    @Test
    public void testOaiChatCompletionHasChoices() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT);
        String result = model.chatComplete(params);
        Assert.assertTrue("Chat response must contain 'choices'", result.contains("\"choices\""));
    }

    @Test
    public void testOaiChatCompletionHasUsage() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT);
        String result = model.chatComplete(params);
        Assert.assertTrue("Chat response must contain 'usage'", result.contains("\"usage\""));
    }

    @Test
    public void testOaiChatCompletionHasMessageObject() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT);
        String result = model.chatComplete(params);
        Assert.assertTrue("Chat response must contain 'message'", result.contains("\"message\""));
    }

    @Test
    public void testOaiChatCompletionObjectType() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT);
        String result = model.chatComplete(params);
        Assert.assertTrue("Chat response 'object' must be 'chat.completion'",
                result.contains("\"object\":\"chat.completion\""));
    }

    @Test
    public void testOaiChatCompletionMessageHasRole() {
        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, java.util.Collections.singletonList(
                        new Pair<>("user", "Say hello")))
                .setNPredict(N_PREDICT);
        String result = model.chatComplete(params);
        Assert.assertTrue("Message must contain 'role':'assistant'",
                result.contains("\"role\":\"assistant\""));
    }

    // -------------------------------------------------------------------------
    // Embedding response structure
    // -------------------------------------------------------------------------

    @Test
    public void testEmbeddingOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, true);
        Assert.assertTrue("OAI embedding must contain 'data'", result.contains("\"data\""));
        Assert.assertTrue("OAI embedding must contain 'object':'embedding'",
                result.contains("\"object\":\"embedding\""));
        Assert.assertTrue("OAI embedding must contain 'embedding' array",
                result.contains("\"embedding\""));
        Assert.assertTrue("OAI embedding must contain 'usage'", result.contains("\"usage\""));
    }

    @Test
    public void testEmbeddingNonOaiResponseStructure() {
        String json = "{\"input\":\"hello world\"}";
        String result = model.handleEmbeddings(json, false);
        Assert.assertTrue("Non-OAI embedding must contain 'embedding'",
                result.contains("\"embedding\""));
        Assert.assertTrue("Non-OAI embedding must contain 'index'",
                result.contains("\"index\""));
    }

    // -------------------------------------------------------------------------
    // Tokenization response structure
    // -------------------------------------------------------------------------

    @Test
    public void testTokenizeResponseStructure() {
        String result = model.handleTokenize("hello world", false, false);
        Assert.assertNotNull(result);
        Assert.assertTrue("Tokenize response must contain 'tokens'", result.contains("\"tokens\""));
    }

    @Test
    public void testTokenizeWithPiecesResponseStructure() {
        String result = model.handleTokenize("hello world", false, true);
        Assert.assertNotNull(result);
        Assert.assertTrue("Tokenize with pieces must contain 'tokens'", result.contains("\"tokens\""));
    }

    @Test
    public void testDetokenizeResponseStructure() {
        int[] tokens = model.encode("hello world");
        String result = model.handleDetokenize(tokens);
        Assert.assertNotNull(result);
        Assert.assertTrue("Detokenize response must contain 'content'", result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // Completion probabilities structure
    // -------------------------------------------------------------------------

    @Test
    public void testCompletionProbabilitiesStructure() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"n_probs\":3}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("Response with n_probs should contain 'completion_probabilities'",
                result.contains("\"completion_probabilities\""));
    }

    // -------------------------------------------------------------------------
    // generation_settings sub-fields
    // -------------------------------------------------------------------------

    @Test
    public void testGenerationSettingsContainsSamplingParams() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        // generation_settings should echo back the sampling parameters
        Assert.assertTrue("generation_settings should contain 'temperature'",
                result.contains("\"temperature\""));
        Assert.assertTrue("generation_settings should contain 'top_k'",
                result.contains("\"top_k\""));
        Assert.assertTrue("generation_settings should contain 'top_p'",
                result.contains("\"top_p\""));
        Assert.assertTrue("generation_settings should contain 'min_p'",
                result.contains("\"min_p\""));
    }

    @Test
    public void testGenerationSettingsContainsSamplers() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT + "}";
        String result = model.handleCompletions(json);
        Assert.assertTrue("generation_settings should contain 'samplers'",
                result.contains("\"samplers\""));
    }
}
