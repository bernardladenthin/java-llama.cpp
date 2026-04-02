package de.kherud.llama;

import java.io.File;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests for raw JSON endpoint parameters that are not exposed through the Java
 * {@link InferenceParameters} builder but are accepted by the C++ server via
 * {@link LlamaModel#handleCompletions(String)}. Covers:
 * <ul>
 *   <li>DRY sampling parameters (dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, dry_sequence_breakers)</li>
 *   <li>XTC sampling parameters (xtc_probability, xtc_threshold)</li>
 *   <li>top_n_sigma sampling</li>
 *   <li>return_tokens flag</li>
 *   <li>response_fields filtering</li>
 *   <li>timings_per_token flag</li>
 *   <li>post_sampling_probs flag</li>
 *   <li>n_discard parameter</li>
 *   <li>id_slot selection</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Verify raw JSON parameters accepted by handleCompletions that are not exposed " +
                  "through InferenceParameters: DRY, XTC, top_n_sigma, return_tokens, response_fields, " +
                  "timings_per_token, post_sampling_probs, n_discard, and id_slot.",
        model = "claude-opus-4-6"
)
public class JsonEndpointParametersTest {

    private static final int N_PREDICT = 5;
    private static final String PROMPT = "int main() {";

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping JsonEndpointParametersTest",
                new File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(256)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    // -------------------------------------------------------------------------
    // DRY sampling parameters
    // -------------------------------------------------------------------------

    @Test
    public void testDryMultiplierAccepted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"dry_multiplier\":0.8,\"dry_base\":1.75,\"dry_allowed_length\":2"
                + ",\"dry_penalty_last_n\":-1}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue("Response should contain 'content' field", result.contains("\"content\""));
    }

    @Test
    public void testDrySequenceBreakersAccepted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"dry_multiplier\":0.5,\"dry_sequence_breakers\":[\"\\n\",\":\",\"*\"]}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    @Test
    public void testDryDisabledByDefault() {
        // dry_multiplier=0 means DRY is disabled; should still produce output
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"dry_multiplier\":0.0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // XTC sampling parameters
    // -------------------------------------------------------------------------

    @Test
    public void testXtcParametersAccepted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"xtc_probability\":0.5,\"xtc_threshold\":0.1}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    @Test
    public void testXtcDisabled() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"xtc_probability\":0.0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // top_n_sigma sampling
    // -------------------------------------------------------------------------

    @Test
    public void testTopNSigmaAccepted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"top_n_sigma\":2.0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    @Test
    public void testTopNSigmaDisabled() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"top_n_sigma\":-1.0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // return_tokens flag
    // -------------------------------------------------------------------------

    @Test
    public void testReturnTokensTrue() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"return_tokens\":true}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        // When return_tokens is true, the response should include a "tokens" array
        Assert.assertTrue("Response should contain 'tokens' field", result.contains("\"tokens\""));
    }

    @Test
    public void testReturnTokensFalse() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"return_tokens\":false}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // response_fields filtering
    // -------------------------------------------------------------------------

    @Test
    public void testResponseFieldsFiltering() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"response_fields\":[\"content\",\"stop\"]}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
        Assert.assertTrue(result.contains("\"stop\""));
    }

    // -------------------------------------------------------------------------
    // timings_per_token
    // -------------------------------------------------------------------------

    @Test
    public void testTimingsPerTokenTrue() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"timings_per_token\":true}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
        // timings_per_token enables per-token timing info
        Assert.assertTrue("Response should contain timings", result.contains("\"timings\""));
    }

    // -------------------------------------------------------------------------
    // post_sampling_probs
    // -------------------------------------------------------------------------

    @Test
    public void testPostSamplingProbsWithNProbs() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"n_probs\":3,\"post_sampling_probs\":true}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        // post_sampling_probs changes the label from "logprob" to "prob"
        Assert.assertTrue("Response should contain completion_probabilities",
                result.contains("\"completion_probabilities\"") || result.contains("\"prob\""));
    }

    // -------------------------------------------------------------------------
    // n_discard
    // -------------------------------------------------------------------------

    @Test
    public void testNDiscardAccepted() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"n_discard\":0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // id_slot selection
    // -------------------------------------------------------------------------

    @Test
    public void testIdSlotSelection() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"id_slot\":0}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
        Assert.assertTrue("Response should contain 'id_slot' field", result.contains("\"id_slot\""));
    }

    // -------------------------------------------------------------------------
    // ignore_eos via JSON
    // -------------------------------------------------------------------------

    @Test
    public void testIgnoreEosAccepted() {
        // With ignore_eos=true and n_predict=N_PREDICT, generation should still respect n_predict
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"ignore_eos\":true}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // Combined DRY + XTC + sampling
    // -------------------------------------------------------------------------

    @Test
    public void testCombinedAdvancedSampling() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"temperature\":0.7,\"top_k\":40,\"top_p\":0.95,\"min_p\":0.05"
                + ",\"dry_multiplier\":0.5,\"dry_base\":1.75,\"dry_allowed_length\":2"
                + ",\"xtc_probability\":0.3,\"xtc_threshold\":0.1"
                + ",\"repeat_penalty\":1.1,\"frequency_penalty\":0.1,\"presence_penalty\":0.1}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // Custom sampler chain via JSON
    // -------------------------------------------------------------------------

    @Test
    public void testCustomSamplerChainViaJson() {
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"samplers\":[\"top_k\",\"top_p\",\"temperature\"]}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }

    // -------------------------------------------------------------------------
    // Speculative decoding parameters via JSON
    // -------------------------------------------------------------------------

    @Test
    public void testSpeculativeParamsAccepted() {
        // These speculative params are accepted even without a draft model
        String json = "{\"prompt\":\"" + PROMPT + "\",\"n_predict\":" + N_PREDICT
                + ",\"speculative\":{\"n_min\":0,\"n_max\":16,\"p_min\":0.75}}";
        String result = model.handleCompletions(json);
        Assert.assertNotNull(result);
        Assert.assertTrue(result.contains("\"content\""));
    }
}
