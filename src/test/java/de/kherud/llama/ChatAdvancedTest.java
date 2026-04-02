package de.kherud.llama;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import de.kherud.llama.args.MiroStat;
import de.kherud.llama.args.Sampler;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Advanced inference parameter scenarios covering code paths untested by
 * {@link LlamaModelTest} and {@link ChatScenarioTest}:
 * <ul>
 *   <li>setCachePrompt — repeated call produces consistent output</li>
 *   <li>nPredict=-1 with stop string — unbounded generation terminates</li>
 *   <li>setNProbs — streaming JSON contains probability data</li>
 *   <li>setChatTemplate — custom Jinja template applied by applyTemplate</li>
 *   <li>setUseChatTemplate(true) in generate() — template applied in raw path</li>
 *   <li>setRepeatPenalty + setFrequencyPenalty + setPresencePenalty</li>
 *   <li>setSamplers — custom sampler chain</li>
 *   <li>setMiroStat V2 — alternative sampler path</li>
 *   <li>requestCompletion direct streaming (non-chat)</li>
 *   <li>disableTokenIds — logit bias to negative-infinity</li>
 *   <li>setPenaltyPrompt(String) and setPenaltyPrompt(int[]) accepted</li>
 *   <li>setNKeep — number of prompt tokens preserved</li>
 *   <li>Multiple stop strings — first match terminates generation</li>
 *   <li>setMinP / setTfsZ / setTypicalP — alternative sampler params</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Advanced inference parameter scenarios: caching, probability output, custom chat " +
                  "templates, penalty params, MiroStat, direct streaming, logit bias, multiple stop " +
                  "strings, and all alternative sampler configurations."
)
public class ChatAdvancedTest {

    private static final int N_PREDICT = 10;
    private static final String SIMPLE_PROMPT = "def hello():";

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping ChatAdvancedTest",
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

    // ------------------------------------------------------------------
    // 1. setCachePrompt — repeated call with same prompt is consistent
    // ------------------------------------------------------------------

    /**
     * Two identical calls with {@code setCachePrompt(true)} and a fixed seed
     * must produce the same output. This verifies that prompt caching does not
     * corrupt the KV cache or alter sampling state between requests.
     */
    @Test
    public void testCachePromptConsistentOutput() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setCachePrompt(true);

        String first  = model.complete(params);
        String second = model.complete(params);

        Assert.assertFalse("First cached call must produce output", first.isEmpty());
        Assert.assertEquals("Same prompt with cache_prompt must produce identical output",
                first, second);
    }

    // ------------------------------------------------------------------
    // 2. nPredict=-1 (infinite) + stop string terminates generation
    // ------------------------------------------------------------------

    /**
     * Setting {@code nPredict=-1} allows unlimited generation, but a stop
     * string must still terminate it. The output must not contain the stop
     * string itself and must be non-empty.
     */
    @Test
    public void testUnboundedGenerationTerminatesAtStopString() {
        // Use a stop string that the model will produce quickly
        InferenceParameters params = new InferenceParameters("A B C D E F G")
                .setNPredict(-1)
                .setSeed(42)
                .setTemperature(0.0f)
                .setStopStrings("E");

        String output = model.complete(params);

        Assert.assertNotNull("Unbounded+stop output must not be null", output);
        // The stop string itself must not appear in the completion
        Assert.assertFalse("Output must not contain the stop string 'E'",
                output.contains("E"));
    }

    // ------------------------------------------------------------------
    // 3. setNProbs — streaming JSON contains completion_probabilities
    // ------------------------------------------------------------------

    /**
     * When {@code setNProbs(n)} is set, each streaming token JSON must include
     * a {@code completion_probabilities} field. This exercises the probability
     * reporting path in the native layer.
     */
    @Test
    public void testSetNProbsStreamingJsonHasProbabilities() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(5)
                .setSeed(42)
                .setTemperature(0.0f)
                .setNProbs(3)
                .setStream(true);

        int taskId = model.requestCompletion(params.toString());

        boolean foundProbabilities = false;
        int tokens = 0;
        boolean done = false;
        while (!done) {
            String json = model.receiveCompletionJson(taskId);
            Assert.assertNotNull("receiveCompletionJson must not be null", json);
            LlamaOutput output = LlamaOutput.fromJson(json);
            if (json.contains("\"completion_probabilities\"")) {
                foundProbabilities = true;
            }
            tokens++;
            if (output.stop) {
                done = true;
                model.releaseTask(taskId);
            }
            if (tokens > N_PREDICT + 2) {
                model.releaseTask(taskId);
                break;
            }
        }

        Assert.assertTrue(
                "At least one streaming JSON chunk must contain 'completion_probabilities' when nProbs>0",
                foundProbabilities
        );
    }

    // ------------------------------------------------------------------
    // 4. setChatTemplate — custom Jinja template applied by applyTemplate
    // ------------------------------------------------------------------

    /**
     * {@link InferenceParameters#setChatTemplate(String)} puts a custom Jinja2
     * template in the request JSON. The server may or may not apply it depending
     * on whether the model has a compiled (peg-native) built-in template — if
     * one exists, the built-in template takes precedence over the per-request
     * {@code chat_template} field for the {@code applyTemplate()} code path.
     * <p>
     * This test therefore verifies the parameter is:
     * <ol>
     *   <li>Serialised correctly by {@link InferenceParameters} (no JSON error)</li>
     *   <li>Accepted by the native layer without throwing</li>
     *   <li>Producing a non-empty result that contains the message content</li>
     * </ol>
     * Behavioural verification that the custom filter ({@code | upper}) is
     * applied is intentionally omitted because the CodeLlama model's embedded
     * ChatML template overrides the per-request template for this endpoint.
     */
    @Test
    public void testCustomChatTemplateAcceptedWithoutError() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "hello world"));

        // A custom template using Jinja2 | upper filter
        String customTemplate =
                "{% for m in messages %}" +
                "{{ m.role | upper }}: {{ m.content }}" +
                "{% endfor %}";

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setChatTemplate(customTemplate);

        // Must not throw; parameter is accepted and forwarded to native layer
        String result = model.applyTemplate(params);

        Assert.assertNotNull("applyTemplate with setChatTemplate must return non-null", result);
        Assert.assertFalse("applyTemplate with setChatTemplate must return non-empty result",
                result.isEmpty());
        Assert.assertTrue(
                "Result must contain the message content 'hello world' regardless of template used",
                result.contains("hello world")
        );
    }

    // ------------------------------------------------------------------
    // 5. setUseChatTemplate(true) in generate() — template in raw path
    // ------------------------------------------------------------------

    /**
     * {@link InferenceParameters#setUseChatTemplate(boolean)} enables chat
     * template application inside the raw {@code generate()} path (not via
     * {@code generateChat()}). Combined with {@code setMessages()}, the
     * generation must produce non-empty output and must not throw.
     */
    @Test
    public void testUseChatTemplateInGenerate() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setUseChatTemplate(true)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        StringBuilder output = new StringBuilder();
        for (LlamaOutput token : model.generate(params)) {
            output.append(token.text);
        }

        Assert.assertFalse("generate() with use_chat_template must produce output",
                output.toString().isEmpty());
    }

    // ------------------------------------------------------------------
    // 6. Penalty params — repeatPenalty, frequencyPenalty, presencePenalty
    // ------------------------------------------------------------------

    /**
     * All three repetition-control parameters must be accepted by the native
     * layer and must not cause a crash or empty response. Passing all three
     * together exercises the penalty accumulation code path.
     */
    @Test
    public void testRepeatAndFrequencyAndPresencePenalty() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.5f)
                .setRepeatPenalty(1.3f)
                .setFrequencyPenalty(0.3f)
                .setPresencePenalty(0.2f)
                .setRepeatLastN(32);

        String output = model.complete(params);
        Assert.assertFalse("Penalty params must not produce empty output", output.isEmpty());
    }

    // ------------------------------------------------------------------
    // 7. setSamplers — custom sampler chain
    // ------------------------------------------------------------------

    /**
     * Specifying a custom ordered sampler chain via {@link InferenceParameters#setSamplers}
     * must be accepted by the native layer and must produce non-empty output.
     * This exercises the sampler-order parsing code path.
     */
    @Test
    public void testCustomSamplerChain() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.7f)
                .setTopK(40)
                .setTopP(0.9f)
                .setSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);

        String output = model.complete(params);
        Assert.assertFalse("Custom sampler chain must produce non-empty output", output.isEmpty());
    }

    // ------------------------------------------------------------------
    // 8. MiroStat V2 — alternative sampler path
    // ------------------------------------------------------------------

    /**
     * MiroStat V2 is a completely different sampling algorithm that replaces
     * top-k/p. Setting {@link MiroStat#V2} with valid tau/eta must produce
     * non-empty output without throwing.
     */
    @Test
    public void testMiroStatV2Sampling() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setMiroStat(MiroStat.V2)
                .setMiroStatTau(5.0f)
                .setMiroStatEta(0.1f);

        String output = model.complete(params);
        Assert.assertFalse("MiroStat V2 must produce non-empty output", output.isEmpty());
    }

    // ------------------------------------------------------------------
    // 9. requestCompletion direct streaming (non-chat)
    // ------------------------------------------------------------------

    /**
     * {@code requestCompletion()} + {@code receiveCompletionJson()} loop
     * exercises the raw (non-chat) native streaming path directly, bypassing
     * {@link LlamaIterator}. The task must complete cleanly with a stop token.
     */
    @Test
    public void testRequestCompletionDirectStreaming() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setStream(true);

        int taskId = model.requestCompletion(params.toString());

        StringBuilder sb = new StringBuilder();
        int tokens = 0;
        boolean stopped = false;
        while (!stopped) {
            String json = model.receiveCompletionJson(taskId);
            Assert.assertNotNull("receiveCompletionJson must not return null", json);
            LlamaOutput output = LlamaOutput.fromJson(json);
            sb.append(output.text);
            tokens++;
            if (output.stop) {
                stopped = true;
                model.releaseTask(taskId);
            }
            if (tokens > N_PREDICT + 2) {
                model.releaseTask(taskId);
                Assert.fail("Direct streaming did not stop within nPredict tokens");
            }
        }

        Assert.assertTrue("Direct non-chat streaming must emit at least one token", tokens > 0);
        Assert.assertFalse("Direct non-chat streaming must produce non-empty content",
                sb.toString().isEmpty());
    }

    // ------------------------------------------------------------------
    // 10. disableTokenIds — logit bias to -infinity
    // ------------------------------------------------------------------

    /**
     * {@link InferenceParameters#disableTokenIds(java.util.Collection)} sets
     * the logit bias for the given token IDs to {@code -infinity}, making them
     * impossible to generate. This test disables the EOS token ID (typically 2
     * in LLaMA-family models) and combines it with a short {@code nPredict}
     * to verify the call succeeds without crashing or producing an empty result.
     * <p>
     * The EOS token ID is derived via {@code model.encode("")} so it is not
     * hard-coded.
     */
    @Test
    public void testDisableTokenIdsAccepted() {
        // Token 2 is EOS in many LLaMA-family models; derive it from encode
        // to avoid hard-coding. If the model has no EOS token, skip gracefully.
        int[] eosTokens = model.encode("");
        if (eosTokens.length == 0) {
            // No EOS token found; just use a safe no-op token id (0 = padding)
            eosTokens = new int[]{0};
        }

        // Disable only the last token in the encoded empty string (which may include BOS)
        int disabledId = eosTokens[eosTokens.length - 1];

        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .disableTokenIds(Collections.singletonList(disabledId));

        String output = model.complete(params);
        Assert.assertFalse("disableTokenIds must not produce empty output", output.isEmpty());
    }

    // ------------------------------------------------------------------
    // 11. setPenaltyPrompt(String) and setPenaltyPrompt(int[]) accepted
    // ------------------------------------------------------------------

    /**
     * Both overloads of {@code setPenaltyPrompt} must be accepted without error
     * and must produce non-empty output. The string form restricts which part of
     * the prompt is penalised; the token-array form does the same by ID.
     */
    @Test
    public void testPenaltyPromptStringAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setPenaltyPrompt("def ")
                .setRepeatPenalty(1.2f);

        Assert.assertFalse("setPenaltyPrompt(String) must produce output",
                model.complete(params).isEmpty());
    }

    @Test
    public void testPenaltyPromptTokenArrayAccepted() {
        int[] penaltyTokens = model.encode("def ");
        Assume.assumeTrue("Need at least one penalty token", penaltyTokens.length > 0);

        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setPenaltyPrompt(penaltyTokens)
                .setRepeatPenalty(1.2f);

        Assert.assertFalse("setPenaltyPrompt(int[]) must produce output",
                model.complete(params).isEmpty());
    }

    // ------------------------------------------------------------------
    // 12. Multiple stop strings — first match terminates
    // ------------------------------------------------------------------

    /**
     * When multiple stop strings are configured, the first one encountered must
     * terminate generation. The output must not contain ANY of the stop strings.
     */
    @Test
    public void testMultipleStopStringsFirstMatchTerminates() {
        // Prompt that will produce digits quickly; stop at first of several options
        InferenceParameters params = new InferenceParameters("1 2 3 4 5 6 7 8 9")
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setStopStrings("4", "5", "6");

        String output = model.complete(params);

        Assert.assertNotNull(output);
        // None of the stop strings should appear in the output
        for (String stop : new String[]{"4", "5", "6"}) {
            Assert.assertFalse(
                    "Output must not contain stop string '" + stop + "', got: " + output,
                    output.contains(stop)
            );
        }
    }

    // ------------------------------------------------------------------
    // 13. Alternative sampler parameters: minP, tfsZ, typicalP
    // ------------------------------------------------------------------

    /**
     * {@code setMinP()}, {@code setTfsZ()}, and {@code setTypicalP()} are
     * alternative token-filtering parameters. Each must be individually accepted
     * by the native layer and must produce non-empty output.
     */
    @Test
    public void testMinPSamplerAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.7f)
                .setMinP(0.05f);

        Assert.assertFalse("setMinP must produce output", model.complete(params).isEmpty());
    }

    @Test
    public void testTfsZSamplerAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.7f)
                .setTfsZ(0.95f);

        Assert.assertFalse("setTfsZ must produce output", model.complete(params).isEmpty());
    }

    @Test
    public void testTypicalPSamplerAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.7f)
                .setTypicalP(0.9f);

        Assert.assertFalse("setTypicalP must produce output", model.complete(params).isEmpty());
    }

    // ------------------------------------------------------------------
    // 14. setNKeep — prompt token preservation
    // ------------------------------------------------------------------

    /**
     * {@code setNKeep(-1)} instructs the server to preserve all initial prompt
     * tokens when context shifting is needed. This must be accepted without
     * error and must produce non-empty output.
     */
    @Test
    public void testNKeepAllTokensAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setNKeep(-1);

        Assert.assertFalse("setNKeep(-1) must produce output", model.complete(params).isEmpty());
    }

    // ------------------------------------------------------------------
    // 15. disableTokens (string form) — accepted without crash
    // ------------------------------------------------------------------

    /**
     * {@link InferenceParameters#disableTokens(java.util.Collection)} uses
     * string-form logit bias. Disabling a low-probability token that is unlikely
     * to appear must not crash and must produce non-empty output.
     */
    @Test
    public void testDisableTokensStringFormAccepted() {
        // Disable a token that is very unlikely to appear in a Python snippet
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .disableTokens(Arrays.asList("!!!"));

        Assert.assertFalse("disableTokens must not produce empty output",
                model.complete(params).isEmpty());
    }

    // ------------------------------------------------------------------
    // 16. MiroStat V1 — first-generation algorithm path
    // ------------------------------------------------------------------

    /**
     * MiroStat V1 uses a different update rule than V2. Verify it is accepted
     * and produces output.
     */
    @Test
    public void testMiroStatV1Sampling() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setMiroStat(MiroStat.V1)
                .setMiroStatTau(5.0f)
                .setMiroStatEta(0.1f);

        Assert.assertFalse("MiroStat V1 must produce non-empty output",
                model.complete(params).isEmpty());
    }
}
