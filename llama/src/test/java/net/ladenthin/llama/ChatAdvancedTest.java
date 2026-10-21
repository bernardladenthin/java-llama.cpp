// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.Sampler;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.json.CompletionResponseParser;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.CompletionResult;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Advanced inference parameter scenarios covering code paths untested by
 * {@link LlamaModelTest} and {@link ChatScenarioTest}:
 * <ul>
 *   <li>setCachePrompt — repeated call produces consistent output</li>
 *   <li>nPredict=-1 with stop string — unbounded generation terminates</li>
 *   <li>setNProbs — streaming JSON contains probability data</li>
 *   <li>setChatTemplate — custom Jinja template applied by applyTemplate</li>
 *   <li>withMessages in generate() — the model's chat template applied in the raw path</li>
 *   <li>setRepeatPenalty + setFrequencyPenalty + setPresencePenalty</li>
 *   <li>setSamplers — custom sampler chain</li>
 *   <li>setMiroStat V2 — alternative sampler path</li>
 *   <li>requestCompletion direct streaming (non-chat)</li>
 *   <li>disableTokenIds — logit bias to negative-infinity</li>
 *   <li>setNKeep — number of prompt tokens preserved</li>
 *   <li>Multiple stop strings — first match terminates generation</li>
 *   <li>setMinP / setTypicalP — alternative sampler params</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Advanced inference parameter scenarios: caching, probability output, custom chat "
                + "templates, penalty params, MiroStat, direct streaming, logit bias, multiple stop "
                + "strings, and all alternative sampler configurations.")
public class ChatAdvancedTest {

    private static final int N_PREDICT = 10;
    private final CompletionResponseParser completionParser = new CompletionResponseParser();
    private static final String SIMPLE_PROMPT = "def hello():";

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping ChatAdvancedTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setCtxSize(256)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    @AfterAll
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
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withCachePrompt(true);

        String first = model.complete(params);
        String second = model.complete(params);

        assertFalse(first.isEmpty(), "First cached call must produce output");
        assertEquals(first, second, "Same prompt with cache_prompt must produce identical output");
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
                .withNPredict(-1)
                .withSeed(42)
                .withTemperature(0.0f)
                .withStopStrings("E");

        String output = model.complete(params);

        assertNotNull(output, "Unbounded+stop output must not be null");
        // The stop string itself must not appear in the completion
        assertFalse(output.contains("E"), "Output must not contain the stop string 'E'");
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
                .withNPredict(5)
                .withSeed(42)
                .withTemperature(0.0f)
                .withNProbs(3)
                .withStream(true);

        int taskId = model.requestCompletion(params.toJson());

        boolean foundProbabilities = false;
        int tokens = 0;
        boolean done = false;
        while (!done) {
            String json = model.receiveCompletionJson(taskId);
            assertNotNull(json, "receiveCompletionJson must not be null");
            LlamaOutput output = completionParser.parse(json);
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

        assertTrue(
                foundProbabilities,
                "At least one streaming JSON chunk must contain 'completion_probabilities' when nProbs>0");
    }

    // ------------------------------------------------------------------
    // 4. Messages in the raw generate() path
    // ------------------------------------------------------------------

    /**
     * {@code withMessages()} drives the raw {@code generate()} path (not
     * {@code generateChat()}): the server applies the model's chat template to the
     * message array, so the generation must produce non-empty output and must not throw.
     *
     * <p>This test used to also pass {@code withUseChatTemplate(true)}. That was never
     * read by the server — jinja is a launch-time option, and no {@code "use_jinja"}
     * request key has existed at any pin this project has carried — so the test measured
     * {@code withMessages()} alone the whole time. It says so now.
     */
    @Test
    public void testMessagesInGenerate() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));

        InferenceParameters params = new InferenceParameters("")
                .withMessages(null, messages)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f);

        StringBuilder output = new StringBuilder();
        for (LlamaOutput token : model.generate(params)) {
            output.append(token.text);
        }

        assertFalse(output.toString().isEmpty(), "generate() with messages must produce output");
    }

    /**
     * On the raw completion path the messages become the prompt the chat template renders from them
     * ({@link LlamaModel#applyTemplate(InferenceParameters)}), not an ignored field next to an empty
     * prompt: the request evaluates exactly the tokens a completion of that rendered prompt does. The
     * prompt token count is the full prompt whether or not the slot's cache served it.
     */
    @Test
    public void testMessagesInCompleteWithStatsEvaluateTheTemplatedPrompt() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));
        InferenceParameters chat = InferenceParameters.empty().withMessages(null, messages);
        String rendered = model.applyTemplate(chat);
        assertTrue(rendered.contains("Write one word."), rendered);
        long expected = model.completeWithStats(new InferenceParameters(rendered).withNPredict(1))
                .getUsage()
                .getPromptTokens();

        CompletionResult result = model.completeWithStats(chat.withNPredict(1));

        assertEquals(expected, result.getUsage().getPromptTokens(), result.getRawJson());
    }

    /** A prompt beside the messages is refused: one of the two would have to be dropped. */
    @Test
    public void testMessagesAndAPromptInCompleteAreAnInvalidRequest() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));
        InferenceParameters request = new InferenceParameters("def hello():")
                .withMessages(null, messages)
                .withNPredict(1);

        InvalidRequestException e = assertThrows(InvalidRequestException.class, () -> model.complete(request));

        assertTrue(e.getMessage().contains("'prompt'"), "Error should name the prompt but was: " + e.getMessage());
    }

    /**
     * The chat fields the raw completion paths cannot apply are refused beside the messages, by name:
     * the grammar of a response format and the tool-call grammar of the tool fields belong to the chat
     * paths, and {@code complete()}/{@code generate()} used to drop them without a word.
     */
    @Test
    public void testChatOnlyFieldsBesideMessagesInCompleteAndGenerateAreAnInvalidRequest() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));
        String tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                + "\"parameters\":{\"type\":\"object\"}}}]";
        List<Pair<String, InferenceParameters>> requests = Arrays.asList(
                new Pair<>("response_format", chatRequest(messages).withResponseFormat("{\"type\":\"json_object\"}")),
                new Pair<>("tools", chatRequest(messages).withToolsJson(tools)),
                new Pair<>("tool_choice", chatRequest(messages).withToolChoice("required")),
                new Pair<>("parallel_tool_calls", chatRequest(messages).withParallelToolCalls(true)));

        for (Pair<String, InferenceParameters> request : requests) {
            String field = "'" + request.getKey() + "'";
            InvalidRequestException completed =
                    assertThrows(InvalidRequestException.class, () -> model.complete(request.getValue()), field);
            assertTrue(
                    completed.getMessage().contains(field),
                    "complete() names " + field + ": " + completed.getMessage());
            InvalidRequestException generated = assertThrows(
                    InvalidRequestException.class,
                    () -> {
                        for (LlamaOutput ignored : model.generate(request.getValue())) {
                            // drained: the request must not run
                        }
                    },
                    field);
            assertTrue(
                    generated.getMessage().contains(field),
                    "generate() names " + field + ": " + generated.getMessage());
        }
    }

    private static InferenceParameters chatRequest(List<Pair<String, String>> messages) {
        return InferenceParameters.empty().withMessages(null, messages).withNPredict(1);
    }

    /**
     * A ChatML template with a thinking block that it opens in the generation prompt: llama.cpp derives
     * the thinking tags {@code <think>}/{@code </think>} from it, which the reasoning budget acts on.
     */
    private static final String THINKING_TEMPLATE = "{%- for message in messages -%}"
            + "{{- '<|im_start|>' + message.role + '\\n' -}}"
            + "{%- if message.reasoning_content -%}"
            + "{{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' -}}"
            + "{%- endif -%}"
            + "{{- message.content + '<|im_end|>\\n' -}}"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n' -}}{%- endif -%}";

    /**
     * Beside the messages, {@code complete()} and {@code generate()} keep the reasoning budget of a
     * template with thinking tags, as the chat paths do: with a budget of one token, which the
     * {@code "\n"} after the {@code <think>} of the generation prompt spends, the end tag is forced from
     * the first generated token on. Both paths used to render the messages and drop the tags, so the
     * budget did nothing there.
     *
     * <p>The reasoning model is used because its tokenizer gives a tag the same tokens inside the
     * prompt and on its own, which is what the budget sampler matches; a SentencePiece tokenizer puts a
     * space before a tag tokenized on its own, and the budget then never starts, on the chat paths too.
     * The tokens are compared first, then the text: the template's tags are preserved tokens of its chat
     * session, so the end tag reads as text in the content even where the vocabulary holds it as one
     * special token (Qwen3, CI's reasoning model). Before the raw paths took the session's preserved
     * tokens over, such an end tag was rendered as an empty piece there; with the local random model,
     * whose byte tokenizer has no special tags, the text check passes either way, and the C++ case
     * {@code CApiReasoningBudget.ARawCompletionWithMessagesRendersTheTemplatesSpecialTagsAsText} pins
     * it on a vocabulary with special tags.
     */
    @Test
    public void testReasoningBudgetBesideMessagesInCompleteAndGenerateForcesTheEndTag() throws Exception {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping the reasoning-budget case");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel thinking = new LlamaModel(new ModelParameters()
                .setCtxSize(256)
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .setChatTemplate(THINKING_TEMPLATE))) {
            InferenceParameters request = InferenceParameters.empty()
                    .withMessages(null, Collections.singletonList(new Pair<>("user", "What is 2+2?")))
                    .withReasoningBudgetTokens(1)
                    .withNPredict(N_PREDICT)
                    .withTemperature(0.0f)
                    .withReturnTokens(true);
            int[] endTag = thinking.encode("</think>");

            CompletionResult completed = thinking.completeWithStats(request);

            JsonNode tokens =
                    new ObjectMapper().readTree(completed.getRawJson()).path("tokens");
            assertTrue(tokens.size() > 0, completed.getRawJson());
            assertEquals(endTag[0], tokens.get(0).asInt(), completed.getRawJson());
            assertTrue(completed.getText().startsWith("</think>"), completed.getRawJson());
            StringBuilder generated = new StringBuilder();
            for (LlamaOutput output : thinking.generate(request)) {
                generated.append(output.text);
            }
            assertEquals(completed.getText(), generated.toString());
        }
    }

    /**
     * Without messages, {@code complete()} and {@code generate()} take the budget's tags from the
     * request: once the model writes the start tag, a budget of zero forces the first end tag. The
     * test is independent of the model: a logit bias makes the model write one token over and over,
     * and that token's text is the start tag, so the end tag follows it each time.
     *
     * <p>No {@code reasoning_budget_message} is set on purpose: upstream's completion schema sets the
     * sequence it forces only in the handler of that field, so a request naming the tags without it
     * armed a budget that forced nothing. The chat paths always set it (to the server's default);
     * the engine now does the same for a raw request.</p>
     */
    @Test
    public void testReasoningBudgetTagsWithoutMessagesForceTheEndTag() {
        Assumptions.assumeTrue(
                new File(TestConstants.REASONING_MODEL_PATH).exists(),
                "Reasoning model not found, skipping the reasoning-budget case");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel thinking = new LlamaModel(new ModelParameters()
                .setCtxSize(256)
                .setModel(TestConstants.REASONING_MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {
            int startToken = thinking.encode("Hello")[0];
            String startTag = thinking.decode(startToken);
            assertArrayEquals(new int[] {startToken}, thinking.encode(startTag), startTag);
            InferenceParameters request = new InferenceParameters("Say something.")
                    .withTokenIdBias(Collections.singletonMap(startToken, 100.0f))
                    .withReasoningBudgetTokens(0)
                    .withReasoningBudgetStartTag(startTag)
                    .withReasoningBudgetEndTags("END")
                    .withIgnoreEos(true)
                    .withNPredict(N_PREDICT)
                    .withTemperature(0.0f);

            String completed = thinking.complete(request);

            assertTrue(completed.startsWith(startTag + "END"), completed);
            StringBuilder generated = new StringBuilder();
            for (LlamaOutput output : thinking.generate(request)) {
                generated.append(output.text);
            }
            assertEquals(completed, generated.toString());
            // Without the tags the bias alone repeats the start tag, so the end tag above is the budget's.
            String unbounded = thinking.complete(new InferenceParameters("Say something.")
                    .withTokenIdBias(Collections.singletonMap(startToken, 100.0f))
                    .withIgnoreEos(true)
                    .withNPredict(N_PREDICT)
                    .withTemperature(0.0f));
            assertFalse(unbounded.contains("END"), unbounded);
        }
    }

    // ------------------------------------------------------------------
    // 5. Penalty params — repeatPenalty, frequencyPenalty, presencePenalty
    // ------------------------------------------------------------------

    /**
     * All three repetition-control parameters must be accepted by the native
     * layer and must not cause a crash or empty response. Passing all three
     * together exercises the penalty accumulation code path.
     */
    @Test
    public void testRepeatAndFrequencyAndPresencePenalty() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.5f)
                .withRepeatPenalty(1.3f)
                .withFrequencyPenalty(0.3f)
                .withPresencePenalty(0.2f)
                .withRepeatLastN(32);

        String output = model.complete(params);
        assertFalse(output.isEmpty(), "Penalty params must not produce empty output");
    }

    // ------------------------------------------------------------------
    // 6. setSamplers — custom sampler chain
    // ------------------------------------------------------------------

    /**
     * Specifying a custom ordered sampler chain via {@link InferenceParameters#setSamplers}
     * must be accepted by the native layer and must produce non-empty output.
     * This exercises the sampler-order parsing code path.
     */
    @Test
    public void testCustomSamplerChain() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.7f)
                .withTopK(40)
                .withTopP(0.9f)
                .withSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);

        String output = model.complete(params);
        assertFalse(output.isEmpty(), "Custom sampler chain must produce non-empty output");
    }

    // ------------------------------------------------------------------
    // 7. MiroStat V2 — alternative sampler path
    // ------------------------------------------------------------------

    /**
     * MiroStat V2 is a completely different sampling algorithm that replaces
     * top-k/p. Setting {@link MiroStat#V2} with valid tau/eta must produce
     * non-empty output without throwing.
     */
    @Test
    public void testMiroStatV2Sampling() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withMiroStat(MiroStat.V2)
                .withMiroStatTau(5.0f)
                .withMiroStatEta(0.1f);

        String output = model.complete(params);
        assertFalse(output.isEmpty(), "MiroStat V2 must produce non-empty output");
    }

    // ------------------------------------------------------------------
    // 8. requestCompletion direct streaming (non-chat)
    // ------------------------------------------------------------------

    /**
     * {@code requestCompletion()} + {@code receiveCompletionJson()} loop
     * exercises the raw (non-chat) native streaming path directly, bypassing
     * {@link LlamaIterator}. The task must complete cleanly with a stop token.
     */
    @Test
    public void testRequestCompletionDirectStreaming() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withStream(true);

        int taskId = model.requestCompletion(params.toJson());

        StringBuilder sb = new StringBuilder();
        int tokens = 0;
        boolean stopped = false;
        while (!stopped) {
            String json = model.receiveCompletionJson(taskId);
            assertNotNull(json, "receiveCompletionJson must not return null");
            LlamaOutput output = completionParser.parse(json);
            sb.append(output.text);
            tokens++;
            if (output.stop) {
                stopped = true;
                model.releaseTask(taskId);
            }
            if (tokens > N_PREDICT + 2) {
                model.releaseTask(taskId);
                fail("Direct streaming did not stop within nPredict tokens");
            }
        }

        assertTrue(tokens > 0, "Direct non-chat streaming must emit at least one token");
        assertFalse(sb.toString().isEmpty(), "Direct non-chat streaming must produce non-empty content");
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
            eosTokens = new int[] {0};
        }

        // Disable only the last token in the encoded empty string (which may include BOS)
        int disabledId = eosTokens[eosTokens.length - 1];

        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withDisabledTokenIds(Collections.singletonList(disabledId));

        String output = model.complete(params);
        assertFalse(output.isEmpty(), "disableTokenIds must not produce empty output");
    }

    // ------------------------------------------------------------------
    // 11. Multiple stop strings — first match terminates
    // ------------------------------------------------------------------

    /**
     * When multiple stop strings are configured, the first one encountered must
     * terminate generation. The output must not contain ANY of the stop strings.
     */
    @Test
    public void testMultipleStopStringsFirstMatchTerminates() {
        // Prompt that will produce digits quickly; stop at first of several options
        InferenceParameters params = new InferenceParameters("1 2 3 4 5 6 7 8 9")
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withStopStrings("4", "5", "6");

        String output = model.complete(params);

        assertNotNull(output);
        // None of the stop strings should appear in the output
        for (String stop : new String[] {"4", "5", "6"}) {
            assertFalse(output.contains(stop), "Output must not contain stop string '" + stop + "', got: " + output);
        }
    }

    // ------------------------------------------------------------------
    // 12. Alternative sampler parameters: minP, typicalP
    // ------------------------------------------------------------------

    /**
     * {@code setMinP()} and {@code setTypicalP()} are alternative token-filtering
     * parameters. Each must be individually accepted
     * by the native layer and must produce non-empty output.
     */
    @Test
    public void testMinPSamplerAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.7f)
                .withMinP(0.05f);

        assertFalse(model.complete(params).isEmpty(), "setMinP must produce output");
    }

    @Test
    public void testTypicalPSamplerAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.7f)
                .withTypicalP(0.9f);

        assertFalse(model.complete(params).isEmpty(), "setTypicalP must produce output");
    }

    // ------------------------------------------------------------------
    // 13. setNKeep — prompt token preservation
    // ------------------------------------------------------------------

    /**
     * {@code setNKeep(-1)} instructs the server to preserve all initial prompt
     * tokens when context shifting is needed. This must be accepted without
     * error and must produce non-empty output.
     */
    @Test
    public void testNKeepAllTokensAccepted() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withNKeep(-1);

        assertFalse(model.complete(params).isEmpty(), "setNKeep(-1) must produce output");
    }

    // ------------------------------------------------------------------
    // 14. disableTokens (string form) — accepted without crash
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
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withTemperature(0.0f)
                .withDisabledTokens(Arrays.asList("!!!"));

        assertFalse(model.complete(params).isEmpty(), "disableTokens must not produce empty output");
    }

    // ------------------------------------------------------------------
    // 15. MiroStat V1 — first-generation algorithm path
    // ------------------------------------------------------------------

    /**
     * MiroStat V1 uses a different update rule than V2. Verify it is accepted
     * and produces output.
     */
    @Test
    public void testMiroStatV1Sampling() {
        InferenceParameters params = new InferenceParameters(SIMPLE_PROMPT)
                .withNPredict(N_PREDICT)
                .withSeed(42)
                .withMiroStat(MiroStat.V1)
                .withMiroStatTau(5.0f)
                .withMiroStatEta(0.1f);

        assertFalse(model.complete(params).isEmpty(), "MiroStat V1 must produce non-empty output");
    }
}
