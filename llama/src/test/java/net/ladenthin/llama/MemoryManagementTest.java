// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.CompletionResult;
import net.ladenthin.llama.value.Usage;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Exercises complex C++ memory-management logic inside {@code server.hpp} that is not reached
 * by any other test in the suite.
 *
 * <h2>Context shifting ({@code pre_decode()} in server-context.cpp)</h2>
 * With {@code --context-shift} ({@link ModelParameters#enableContextShift()}; upstream's default is
 * off), when a slot's tokens reach its context the server discards the oldest {@code n_discard} KV-cache
 * positions by calling {@code seq_rm} / {@code seq_add} and rebuilds the slot's tokens in-place. The
 * smallest context window (256 tokens), a 300-token answer and {@code setIgnoreEos(true)} guarantee
 * the limit is crossed during a single call; the shift tests assert that every requested token was
 * generated, which without the shift ends at the full context ({@code truncated}).
 *
 * <h2>Prompt-cache prefix reuse (server.hpp ~line 3162)</h2>
 * With {@code cache_prompt=true} successive calls compute a common prefix via
 * {@code cache_tokens.get_common_prefix(prompt_tokens)}.  Tokens that are already in the KV
 * cache are skipped; the rest are evaluated fresh.  If this shifting is broken, logits diverge
 * and the output changes or an assertion fires inside the C++ layer.
 */
@ClaudeGenerated(
        purpose = "Verify context-shifting KV-cache management (llama_memory_seq_rm/add, "
                + "cache_tokens rebuild) and prompt-cache prefix-reuse logic in server.hpp.")
public class MemoryManagementTest {

    /**
     * A short prompt whose token length we can reason about (~10 tokens for CodeLlama BPE); with
     * {@link #SHIFT_N_PREDICT} generated tokens it exceeds {@link #SMALL_CTX_SIZE}.
     */
    private static final String SHORT_PROMPT = "def add(a, b): return";

    /**
     * The context size of {@link #smallCtxModel}, and the smallest there is: llama.cpp pads
     * {@code n_ctx} (and a slot's share of it) to a multiple of 256 ({@code llama-context.cpp:377},
     * b11639), so the 32 this test used to ask for was a context of 256, which its 25 tokens never
     * filled.
     */
    private static final int SMALL_CTX_SIZE = 256;

    /** The tokens the shift tests request; with the prompt they exceed {@link #SMALL_CTX_SIZE}. */
    private static final int SHIFT_N_PREDICT = 300;

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
    private static final String CACHE_EXTENDED_PROMPT = CACHE_PREFIX_PROMPT + "\n    result = ";

    /** Shared model used for prompt-cache tests (ctxSize=128 is ample). */
    private static LlamaModel model;

    /**
     * Separate model instance with the smallest context window ({@link #SMALL_CTX_SIZE}) and one slot.
     * SHORT_PROMPT + {@link #SHIFT_N_PREDICT} exceeds the window, which forces the context-shift path
     * ({@code pre_decode()} in server-context.cpp). It is the small draft model, because the shift tests
     * generate a few hundred tokens each.
     */
    private static LlamaModel smallCtxModel;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping MemoryManagementTest");

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false));

        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model file not found, skipping MemoryManagementTest");
        // The prompt and SHIFT_N_PREDICT tokens exceed the one slot's context, so a context shift is
        // unavoidable -- once it is enabled: upstream's default is off (common.h, ctx_shift = false at
        // b11639), and without it the slot stops when its context is full.
        smallCtxModel = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(SMALL_CTX_SIZE)
                .setParallel(1)
                .enableContextShift()
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    @AfterAll
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
     * <p>C++ path exercised (the shift in {@code pre_decode()}, server-context.cpp; shown in the
     * names of the former server.hpp):
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
                .withNPredict(SHIFT_N_PREDICT)
                .withIgnoreEos(true) // prevent early stop so the shift is reliably triggered
                .withSeed(42);

        CompletionResult result = smallCtxModel.completeWithStats(params);

        assertGeneratedPastTheContext(result);
        assertFalse(result.getText().isEmpty(), "Output must not be empty after a context shift");
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
                .withNPredict(SHIFT_N_PREDICT)
                .withIgnoreEos(true)
                .withSeed(1);
        assertGeneratedPastTheContext(smallCtxModel.completeWithStats(shiftParams));

        // Second call: independent generation on the same model after the shift
        InferenceParameters freshParams =
                new InferenceParameters("x = ").withNPredict(5).withSeed(2);
        String output = smallCtxModel.complete(freshParams);

        assertNotNull(output);
        assertFalse(output.isEmpty(), "Model must still generate output after a prior context shift");
    }

    // ------------------------------------------------------------------
    // Prompt-cache determinism tests
    // ------------------------------------------------------------------

    /**
     * With {@code cache_prompt=true} and greedy decoding, repeated identical calls over the cached
     * prefix must produce the same output.
     *
     * <p>Each call after the first computes {@code get_common_prefix(prompt_tokens)} over the slot's
     * tokens, takes every prompt token but the last from the KV cache and re-evaluates that one in a
     * batch of its own, so that the request has logits (upstream {@code [TAG_PROMPT_LOGITS]}). The
     * second and the third call therefore compute the same thing over the same cache and are compared;
     * the first is not: it evaluated the prompt in one batch, whose logits the one-token re-evaluation
     * does not reproduce bit for bit, so a greedy near tie can flip between the two (see
     * {@code ChatAdvancedTest#testCachePromptConsistentOutput}). Any divergence between two reuses of
     * the cache would produce different tokens.
     */
    @Test
    public void testPromptCacheGivesDeterministicOutput() {
        InferenceParameters params = new InferenceParameters(CACHE_PREFIX_PROMPT)
                .withCachePrompt(true)
                .withNPredict(10)
                .withTemperature(0f) // greedy decoding: fully deterministic
                .withSeed(42);

        CompletionResult first = model.completeWithStats(params);
        CompletionResult second = model.completeWithStats(params);
        CompletionResult third = model.completeWithStats(params);

        assertFalse(first.getText().isEmpty(), "First cached-prompt call must produce output");
        assertFalse(second.getText().isEmpty(), "Second cached-prompt call must produce output");
        assertCachedPrefixReused(second);
        assertCachedPrefixReused(third);
        assertEquals(
                second.getText(),
                third.getText(),
                "Both calls share the same prompt, seed and cached prefix; cache_prompt=true must not change output");
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
                .withCachePrompt(false)
                .withNPredict(10)
                .withTemperature(0f)
                .withSeed(42);

        String first = model.complete(params);
        String second = model.complete(params);

        assertFalse(first.isEmpty());
        assertEquals(
                first, second, "Without cache_prompt, repeated calls with the same seed must still be deterministic");
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
                .withCachePrompt(true)
                .withNPredict(5)
                .withSeed(1);
        model.complete(warmup);

        // Extend the prompt; the prefix is now in the KV cache and must be reused
        InferenceParameters extended = new InferenceParameters(CACHE_EXTENDED_PROMPT)
                .withCachePrompt(true)
                .withNPredict(10)
                .withSeed(2);
        String output = model.complete(extended);

        assertNotNull(output);
        assertFalse(output.isEmpty(), "Generation with a cached prefix and a new suffix must produce output");
    }

    /**
     * Three consecutive calls with the same prompt, seed and {@code cache_prompt=true}: the second and
     * the third both reuse the cached prefix (every prompt token but the last, which upstream
     * re-evaluates) and must produce identical output, which verifies that the cache state stays
     * stable across repeated reuses. The first call is not compared with them on purpose: it
     * evaluated the prompt in one batch, and the one-token re-evaluation does not reproduce those
     * logits bit for bit, so a greedy near tie can flip between the two paths (see
     * {@code ChatAdvancedTest#testCachePromptConsistentOutput}).
     */
    @Test
    public void testPromptCacheStableAcrossMultipleCalls() {
        InferenceParameters params = new InferenceParameters(SHORT_PROMPT)
                .withCachePrompt(true)
                .withNPredict(8)
                .withTemperature(0f)
                .withSeed(77);

        CompletionResult first = model.completeWithStats(params);
        CompletionResult second = model.completeWithStats(params);
        CompletionResult third = model.completeWithStats(params);

        assertFalse(first.getText().isEmpty(), "First call must produce output");
        assertCachedPrefixReused(second);
        assertCachedPrefixReused(third);
        assertEquals(
                second.getText(),
                third.getText(),
                "Third call must match the second: both generate over the same cached prefix");
    }

    // ------------------------------------------------------------------
    // Edge case 1: context shift with n_keep > 0
    // ------------------------------------------------------------------

    /**
     * Exercises the context-shift path with an explicit {@code n_keep > 0}, which takes a
     * completely different index range in the cache-token copy-down loop (server.hpp ~2968–2990).
     *
     * <p>With the default {@code n_keep = 0}, the preserved region is just the BOS token and
     * the loop copies from position {@code n_discard} downward.  When {@code n_keep = 5} the
     * C++ computes:
     * <pre>
     *   int n_keep_eff = slot.params.n_keep + add_bos_token;   // = 5 + 1 = 6
     *   int n_left     = slot.n_past - n_keep_eff;
     *   int n_discard  = n_left / 2;
     *
     *   llama_memory_seq_rm (..., n_keep_eff, n_keep_eff + n_discard);
     *   llama_memory_seq_add(..., n_keep_eff + n_discard, n_past, -n_discard);
     *
     *   // copy-down starts at n_keep_eff, not 0 — tokens[0..n_keep_eff-1] are frozen
     *   for (i = n_keep_eff + n_discard; i &lt; size; i++)
     *       new_tokens[i - n_discard] = new_tokens[i];
     * </pre>
     * None of the existing tests set {@code n_keep > 0}, so the frozen-region arithmetic has
     * never been reached from Java.  A bug here (wrong index, off-by-one in the KV-cache range,
     * double-free of the kept region) would cause a crash, an assertion abort, or garbage output.
     */
    @Test
    public void testContextShiftWithNKeepPreservesGeneration() {
        // n_keep=5 asks the shift to freeze the first 5 prompt tokens as a "system prefix".
        // With SMALL_CTX_SIZE and SHIFT_N_PREDICT the window is reliably exceeded, so the shift fires
        // with the non-trivial n_keep_eff = 5 + add_bos_token path.
        InferenceParameters params = new InferenceParameters(SHORT_PROMPT)
                .withNKeep(5)
                .withNPredict(SHIFT_N_PREDICT)
                .withIgnoreEos(true)
                .withSeed(42);

        CompletionResult result = smallCtxModel.completeWithStats(params);

        assertGeneratedPastTheContext(result);
        assertFalse(result.getText().isEmpty(), "Output must not be empty after a context shift with n_keep > 0");
    }

    // ------------------------------------------------------------------
    // Edge case 2: prompt-cache complete miss (disjoint second prompt)
    // ------------------------------------------------------------------

    /**
     * Exercises the {@code cache_prompt=true} <em>complete-miss</em> path: after populating the
     * KV cache with prompt A, a second call with a completely unrelated prompt B makes
     * {@code get_common_prefix()} return at most the BOS token both prompts start with.
     *
     * <p>The relevant branch ({@code update_slots()} in server-context.cpp):
     * <pre>
     *   n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
     *   // → 1 (the BOS token) on a model that adds one, else 0: the first tokens of A and B differ
     * </pre>
     * At this point the slot still holds all of prompt A's tokens and the KV memory still contains
     * the evaluated key/value vectors for A. The server removes everything after {@code n_past} and
     * evaluates the rest of prompt B in one batch, overwriting those positions. If the stale A data
     * leaks into B's evaluation (wrong sequence ID, wrong position offset, residual KV entries beyond
     * the new prompt length), generation produces wrong tokens or aborts.
     *
     * <p>The test also cross-checks output correctness: a fresh model that answers the same prompt
     * must produce the identical result, confirming that the stale-cache state had no observable
     * effect on the logits. The fresh model gets the same warm-up first, so both answers are computed
     * the same way -- the reused BOS entry and the rest of the prompt in one batch, which the timings
     * of both prove; a model without the warm-up would evaluate the whole prompt in one batch, and that
     * does not reproduce the logits bit for bit, so a greedy near tie could flip (see
     * {@code ChatAdvancedTest#testCachePromptConsistentOutput}). The warm-up takes nothing from the
     * cache ({@code cache_prompt=false}), so on the shared model, whose slot may still hold an earlier
     * test's prompt with the same prefix, it too is evaluated from scratch in one batch.
     */
    @Test
    public void testPromptCacheCompleteMissAfterWarmup() {
        // Step 1: warm the cache with a distinct prompt so the slot's tokens are fully populated.
        InferenceParameters warmup = new InferenceParameters(CACHE_PREFIX_PROMPT)
                .withCachePrompt(false)
                .withNPredict(5)
                .withSeed(1);
        model.complete(warmup);

        // Step 2: call with a completely disjoint prompt.
        // "x = " shares no token with CACHE_PREFIX_PROMPT ("def remove_non_ascii…") but the BOS, so
        // get_common_prefix() returns at most 1 and the stale KV data for CACHE_PREFIX_PROMPT must
        // be silently discarded / overwritten.
        final String disjointPrompt = "x = ";
        InferenceParameters missParams = new InferenceParameters(disjointPrompt)
                .withCachePrompt(true)
                .withNPredict(8)
                .withTemperature(0f)
                .withSeed(99);
        CompletionResult afterMiss = model.completeWithStats(missParams);

        assertNotNull(afterMiss.getText());
        assertFalse(afterMiss.getText().isEmpty(), "Cache-miss call must still produce output");
        assertPromptCacheMiss(afterMiss);

        // Step 3: baseline — a fresh model with the same warm-up, on the same disjoint prompt.
        // Output must be identical, proving that the stale A cache had no effect on B's logits.
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel freshModel = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {
            freshModel.complete(warmup);
            CompletionResult fresh = freshModel.completeWithStats(missParams);

            assertPromptCacheMiss(fresh);
            assertEquals(
                    fresh.getTimings().getCacheN(),
                    afterMiss.getTimings().getCacheN(),
                    "both answers must take the same tokens from the cache: " + fresh.getRawJson() + " / "
                            + afterMiss.getRawJson());
            assertEquals(
                    fresh.getTimings().getPromptN(),
                    afterMiss.getTimings().getPromptN(),
                    "both answers must evaluate the same tokens: " + fresh.getRawJson() + " / "
                            + afterMiss.getRawJson());
            assertEquals(
                    fresh.getText(),
                    afterMiss.getText(),
                    "A cache-miss call must produce the same output as a fresh model on the same prompt");
        }
    }

    // ------------------------------------------------------------------
    // Open/close lifecycle regression tests
    // ------------------------------------------------------------------

    /**
     * Upstream issue <a href="https://github.com/kherud/llama.cpp/issues/102">#102</a>:
     * repeatedly constructing and {@code close()}-ing {@link LlamaModel} eventually OOMs
     * because the native destructor leaked. The fix is in
     * {@code Java_net_ladenthin_llama_LlamaModel_delete} (jllama.cpp): it now drains
     * {@code readers}, signals {@code server.terminate()} twice, joins the worker, and
     * deletes the context — releasing every owned resource.
     *
     * <p>This test runs the original reporter's loop with a higher iteration count and
     * asserts {@code VmRSS} growth stays within a generous tolerance. On non-Linux hosts
     * the {@code VmRSS} reader returns 0, in which case the test degenerates to a
     * "loop completes without crash" smoke check, which is still a valuable regression
     * guard against use-after-free in the destructor path.
     */
    @Test
    public void testOpenCloseLoopDoesNotLeak() {
        ModelParameters params = new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(1024)
                .setThreads(4)
                .setKeep(-1)
                .setGpuLayers(0)
                .setFit(false);

        long baseline = currentVmRssKb();
        for (int i = 0; i < 20; i++) {
            try (LlamaModel m = new LlamaModel(params)) {
                // intentionally no work: we only exercise construct + destruct
            }
        }
        System.gc();
        long after = currentVmRssKb();

        if (baseline > 0 && after > 0) {
            long deltaKb = after - baseline;
            assertTrue(
                    deltaKb < 200_000L,
                    "VmRSS grew by " + deltaKb + " kB across 20 open/close iterations "
                            + "(baseline=" + baseline + " kB, after=" + after + " kB); "
                            + "indicates a native-side leak in LlamaModel.close()");
        }
    }

    /**
     * Upstream issue <a href="https://github.com/kherud/llama.cpp/issues/80">#80</a>:
     * on 3.4.1, opening a model and immediately calling {@code close()} (without any
     * generation) segfaulted in {@code std::_Rb_tree::_M_erase} during destruction of a
     * half-initialised {@code server_context}. The fix waits on {@code worker_ready},
     * drains {@code readers} under lock, and double-calls {@code server.terminate()} —
     * see {@code jllama.cpp:929-940}.
     *
     * <p>If the regression returns the JVM exits with a non-zero status mid-test and JUnit
     * reports a crash; a successful run of all 20 iterations is the green signal.
     */
    @Test
    public void testOpenCloseWithoutGeneration() {
        ModelParameters params = new ModelParameters()
                .setModel(TestConstants.MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(0)
                .setFit(false);

        for (int i = 0; i < 20; i++) {
            try (LlamaModel m = new LlamaModel(params)) {
                // no generation, no embedding — only construct + immediate destruct
            }
        }
    }

    /**
     * A request that repeats the slot's prompt reuses it: every prompt token but the last comes from
     * the cache, and only the last is evaluated, in a batch of its own (upstream
     * {@code [TAG_PROMPT_LOGITS]}). Two results that pass this computed the same thing over the same
     * cache, which is what makes comparing their texts sound.
     */
    private static void assertCachedPrefixReused(CompletionResult result) {
        assertEquals(
                result.getUsage().getPromptTokens() - 1,
                result.getTimings().getCacheN(),
                "every prompt token but the last must come from the cache: " + result.getRawJson());
        assertEquals(
                1,
                result.getTimings().getPromptN(),
                "only the last prompt token must be evaluated: " + result.getRawJson());
    }

    /**
     * A prompt disjoint from the slot's reuses at most the BOS token both start with, and evaluates
     * every other prompt token.
     */
    private static void assertPromptCacheMiss(CompletionResult result) {
        int cacheN = result.getTimings().getCacheN();
        assertTrue(
                cacheN <= 1, "a disjoint prompt can share only the BOS token with the cache: " + result.getRawJson());
        assertEquals(
                result.getUsage().getPromptTokens() - cacheN,
                result.getTimings().getPromptN(),
                "every prompt token not taken from the cache must be evaluated: " + result.getRawJson());
    }

    /**
     * The shift tests are only worth something if the shift happened: every requested token was
     * generated although prompt and answer together do not fit into {@link #SMALL_CTX_SIZE}. Without
     * {@code --context-shift} the slot stops when its context is full ({@code truncated}, stop type
     * {@code limit}) and the answer is shorter than {@link #SHIFT_N_PREDICT}.
     */
    private static void assertGeneratedPastTheContext(CompletionResult result) {
        Usage usage = result.getUsage();
        assertEquals(
                SHIFT_N_PREDICT,
                usage.getCompletionTokens(),
                "every requested token must be generated, which needs a context shift: " + result.getRawJson());
        assertTrue(
                usage.getPromptTokens() + usage.getCompletionTokens() > SMALL_CTX_SIZE,
                "prompt and answer must exceed the context, or no shift was needed: " + usage);
    }

    /**
     * Reads {@code VmRSS} from {@code /proc/self/status} in kilobytes. Returns 0 on
     * non-Linux hosts (file absent) or on any read error — callers must treat 0 as
     * "not available" rather than "no memory used".
     */
    private static long currentVmRssKb() {
        Path status = Paths.get("/proc/self/status");
        if (!Files.exists(status)) {
            return 0L;
        }
        try {
            for (String line : Files.readAllLines(status)) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("\\D+", ""));
                }
            }
        } catch (IOException | NumberFormatException ignored) {
            // fall through
        }
        return 0L;
    }
}
