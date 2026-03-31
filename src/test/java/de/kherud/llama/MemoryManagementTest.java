package de.kherud.llama;

import java.io.File;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Exercises complex C++ memory-management logic inside {@code server.hpp} that is not reached
 * by any other test in the suite.
 *
 * <h2>Context shifting (server.hpp ~line 2951)</h2>
 * When {@code n_past + 1 >= n_ctx} the server discards the oldest {@code n_discard} KV-cache
 * positions by calling {@code llama_memory_seq_rm} / {@code llama_memory_seq_add} and rebuilds
 * {@code cache_tokens} in-place.  A small context window (32 tokens) and
 * {@code setIgnoreEos(true)} guarantee the limit is crossed during a single {@code complete()}
 * call.
 *
 * <h2>Prompt-cache prefix reuse (server.hpp ~line 3162)</h2>
 * With {@code cache_prompt=true} successive calls compute a common prefix via
 * {@code cache_tokens.get_common_prefix(prompt_tokens)}.  Tokens that are already in the KV
 * cache are skipped; the rest are evaluated fresh.  If this shifting is broken, logits diverge
 * and the output changes or an assertion fires inside the C++ layer.
 */
@ClaudeGenerated(
        purpose = "Verify context-shifting KV-cache management (llama_memory_seq_rm/add, " +
                  "cache_tokens rebuild) and prompt-cache prefix-reuse logic in server.hpp.")
public class MemoryManagementTest {

    /**
     * A short prompt whose token length we can reason about (~10 tokens for CodeLlama BPE).
     * Kept intentionally brief so that, together with nPredict=25 and ctxSize=32, the context
     * window is reliably exceeded and the shift path is triggered.
     */
    private static final String SHORT_PROMPT = "def add(a, b): return";

    /**
     * A longer prompt (~25 tokens) used as the stable cached prefix for cache-reuse tests.
     */
    private static final String CACHE_PREFIX_PROMPT =
            "def remove_non_ascii(s: str) -> str:\n    \"\"\"Remove all non-ASCII characters.\"\"\"";

    /**
     * Extension of {@link #CACHE_PREFIX_PROMPT}.  The second call shares the full prefix with
     * the first call, so {@code get_common_prefix()} returns the prefix length and reuses those
     * KV-cache positions via {@code llama_memory_seq_add}.
     */
    private static final String CACHE_EXTENDED_PROMPT =
            CACHE_PREFIX_PROMPT + "\n    result = ";

    /** Shared model used for prompt-cache tests (ctxSize=128 is ample). */
    private static LlamaModel model;

    /**
     * Separate model instance with a deliberately tiny context window (32 tokens).
     * SHORT_PROMPT (~10 tokens) + nPredict=25 totals ~35 tokens, which exceeds the window
     * and forces the context-shift path in server.hpp.
     */
    private static LlamaModel smallCtxModel;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue(
                "Model file not found, skipping MemoryManagementTest",
                new File(TestConstants.MODEL_PATH).exists());

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        model = new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.MODEL_PATH)
                        .setCtxSize(128)
                        .setGpuLayers(gpuLayers)
                        .setFit(false));

        // ctxSize=32 makes a context shift unavoidable: ~10-token prompt + 25 predicted tokens
        // totals ~35 positions, guaranteed to exceed the 32-token window.
        smallCtxModel = new LlamaModel(
                new ModelParameters()
                        .setModel(TestConstants.MODEL_PATH)
                        .setCtxSize(32)
                        .setGpuLayers(gpuLayers)
                        .setFit(false));
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
        if (smallCtxModel != null) {
            smallCtxModel.close();
        }
    }

    // ------------------------------------------------------------------
    // Context-shifting tests
    // ------------------------------------------------------------------

    /**
     * Forces the KV-cache context shift by generating more tokens than {@code ctxSize} allows.
     *
     * <p>C++ path exercised (server.hpp ~2951–2993):
     * <pre>
     *   llama_memory_seq_rm(ctx_mem, slot.id, n_keep, n_keep + n_discard)
     *   llama_memory_seq_add(ctx_mem, slot.id, n_keep + n_discard, slot.n_past, -n_discard)
     *   new_tokens.resize(slot.cache_tokens.size() - n_discard)
     *   slot.cache_tokens.clear(); slot.cache_tokens.insert(new_tokens)
     *   slot.n_past -= n_discard
     * </pre>
     * If the logic is broken the call throws, hangs, or returns an empty string.
     */
    @Test
    public void testContextShiftingAllowsContinuedGeneration() {
        InferenceParameters params = new InferenceParameters(SHORT_PROMPT)
                .setNPredict(25)
                .setIgnoreEos(true)   // prevent early stop so the shift is reliably triggered
                .setSeed(42);

        String output = smallCtxModel.complete(params);

        Assert.assertNotNull("Output must not be null after a context shift", output);
        Assert.assertFalse("Output must not be empty after a context shift", output.isEmpty());
    }

    /**
     * After a context-shift-triggering call the model's internal state must still be consistent
     * enough to handle a subsequent, unrelated generation request without error.
     *
     * <p>This tests that {@code slot.n_past}, {@code cache_tokens}, and the KV-cache sequence
     * positions are all coherent after the shift so that the next {@code launch_slot_with_task}
     * can reset the slot cleanly.
     */
    @Test
    public void testContextShiftFollowedByFreshGeneration() {
        // First call: triggers context shift
        InferenceParameters shiftParams = new InferenceParameters(SHORT_PROMPT)
                .setNPredict(25)
                .setIgnoreEos(true)
                .setSeed(1);
        smallCtxModel.complete(shiftParams);

        // Second call: independent generation on the same model after the shift
        InferenceParameters freshParams = new InferenceParameters("x = ")
                .setNPredict(5)
                .setSeed(2);
        String output = smallCtxModel.complete(freshParams);

        Assert.assertNotNull(output);
        Assert.assertFalse(
                "Model must still generate output after a prior context shift",
                output.isEmpty());
    }

    // ------------------------------------------------------------------
    // Prompt-cache determinism tests
    // ------------------------------------------------------------------

    /**
     * With {@code cache_prompt=true} and a fixed seed the second identical call must produce
     * exactly the same output as the first.
     *
     * <p>The first call populates {@code slot.cache_tokens}; the second call computes
     * {@code get_common_prefix(prompt_tokens)} which returns the full prompt length, skipping
     * all prompt re-evaluation and jumping straight to sampling.  Any divergence between the
     * cached and non-cached code paths would produce different tokens.
     */
    @Test
    public void testPromptCacheGivesDeterministicOutput() {
        InferenceParameters params = new InferenceParameters(CACHE_PREFIX_PROMPT)
                .setCachePrompt(true)
                .setNPredict(10)
                .setTemperature(0f)  // greedy decoding: fully deterministic
                .setSeed(42);

        String first  = model.complete(params);
        String second = model.complete(params);

        Assert.assertFalse("First cached-prompt call must produce output", first.isEmpty());
        Assert.assertFalse("Second cached-prompt call must produce output", second.isEmpty());
        Assert.assertEquals(
                "Both calls share the same prompt and seed; cache_prompt=true must not change output",
                first, second);
    }

    /**
     * Baseline: disabling prompt caching must also be deterministic with a fixed seed.
     * This rules out the possibility that the equality in
     * {@link #testPromptCacheGivesDeterministicOutput} is incidentally caused by something
     * other than the cache path.
     */
    @Test
    public void testNoCachePromptAlsoDeterministic() {
        InferenceParameters params = new InferenceParameters(CACHE_PREFIX_PROMPT)
                .setCachePrompt(false)
                .setNPredict(10)
                .setTemperature(0f)
                .setSeed(42);

        String first  = model.complete(params);
        String second = model.complete(params);

        Assert.assertFalse(first.isEmpty());
        Assert.assertEquals(
                "Without cache_prompt, repeated calls with the same seed must still be deterministic",
                first, second);
    }

    // ------------------------------------------------------------------
    // Prompt-cache prefix-reuse tests
    // ------------------------------------------------------------------

    /**
     * Exercises the partial-prefix reuse path: the second call's prompt is a strict extension
     * of the first, so {@code get_common_prefix()} returns the prefix length and those KV-cache
     * positions are reused via {@code llama_memory_seq_add} while the suffix tokens are
     * evaluated fresh.
     *
     * <p>A crash, hang, or empty output indicates that the KV-cache shifting during chunk reuse
     * (server.hpp ~3166–3218) is broken.
     */
    @Test
    public void testPromptCachePrefixReuseSucceeds() {
        // Warm the cache with the prefix prompt
        InferenceParameters warmup = new InferenceParameters(CACHE_PREFIX_PROMPT)
                .setCachePrompt(true)
                .setNPredict(5)
                .setSeed(1);
        model.complete(warmup);

        // Extend the prompt; the prefix is now in the KV cache and must be reused
        InferenceParameters extended = new InferenceParameters(CACHE_EXTENDED_PROMPT)
                .setCachePrompt(true)
                .setNPredict(10)
                .setSeed(2);
        String output = model.complete(extended);

        Assert.assertNotNull(output);
        Assert.assertFalse(
                "Generation with a cached prefix and a new suffix must produce output",
                output.isEmpty());
    }

    /**
     * Three consecutive calls with the same prompt, seed, and {@code cache_prompt=true} must all
     * produce identical output.  The third call reuses the cache path that was already exercised
     * by the second call, verifying that the cache state remains stable across multiple reuses.
     */
    @Test
    public void testPromptCacheStableAcrossMultipleCalls() {
        InferenceParameters params = new InferenceParameters(SHORT_PROMPT)
                .setCachePrompt(true)
                .setNPredict(8)
                .setTemperature(0f)
                .setSeed(77);

        String first  = model.complete(params);
        String second = model.complete(params);
        String third  = model.complete(params);

        Assert.assertFalse("First call must produce output", first.isEmpty());
        Assert.assertEquals("Second call must match first",  first, second);
        Assert.assertEquals("Third call must match first",   first, third);
    }
}
