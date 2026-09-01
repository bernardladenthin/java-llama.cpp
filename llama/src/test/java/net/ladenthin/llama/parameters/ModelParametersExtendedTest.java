// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.hamcrest.Matchers.startsWith;

import java.util.HashMap;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.*;
import net.ladenthin.llama.args.CacheType;
import net.ladenthin.llama.args.FlashAttn;
import net.ladenthin.llama.args.GpuSplitMode;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.NumaStrategy;
import net.ladenthin.llama.args.RopeScalingType;
import org.junit.jupiter.api.Test;

/**
 * Extended tests for {@link ModelParameters} covering CLI argument serialization
 * for all setter methods not already tested in {@link ModelParametersTest}.
 */
@ClaudeGenerated(
        purpose = "Verify CLI argument serialization for all ModelParameters setters not covered by "
                + "ModelParametersTest: context/batch sizing, threading, sampling scalars, XTC, DRY, "
                + "RoPE, YaRN, KV cache, GPU, memory, parallel inference, flag-only toggles, "
                + "speculative decoding, logging, model loading, grammar, chat templates, and advanced options.",
        model = "claude-opus-4-6")
public class ModelParametersExtendedTest {

    // -------------------------------------------------------------------------
    // Context / Batch / Predict / Keep
    // -------------------------------------------------------------------------

    @Test
    public void testSetCtxSize() {
        ModelParameters p = new ModelParameters().setCtxSize(2048);
        assertThat(p.parameters.get("--ctx-size"), is("2048"));
    }

    @Test
    public void testSetCtxSizeZeroUsesModelDefault() {
        ModelParameters p = new ModelParameters().setCtxSize(0);
        assertThat(p.parameters.get("--ctx-size"), is("0"));
    }

    @Test
    public void testSetBatchSize() {
        ModelParameters p = new ModelParameters().setBatchSize(512);
        assertThat(p.parameters.get("--batch-size"), is("512"));
    }

    @Test
    public void testSetUbatchSize() {
        ModelParameters p = new ModelParameters().setUbatchSize(256);
        assertThat(p.parameters.get("--ubatch-size"), is("256"));
    }

    @Test
    public void testSetPredict() {
        ModelParameters p = new ModelParameters().setPredict(100);
        assertThat(p.parameters.get("--predict"), is("100"));
    }

    @Test
    public void testSetPredictInfinity() {
        ModelParameters p = new ModelParameters().setPredict(-1);
        assertThat(p.parameters.get("--predict"), is("-1"));
    }

    @Test
    public void testSetPredictFillContext() {
        ModelParameters p = new ModelParameters().setPredict(-2);
        assertThat(p.parameters.get("--predict"), is("-2"));
    }

    @Test
    public void testSetKeep() {
        ModelParameters p = new ModelParameters().setKeep(64);
        assertThat(p.parameters.get("--keep"), is("64"));
    }

    @Test
    public void testSetKeepAll() {
        ModelParameters p = new ModelParameters().setKeep(-1);
        assertThat(p.parameters.get("--keep"), is("-1"));
    }

    // -------------------------------------------------------------------------
    // Threading
    // -------------------------------------------------------------------------

    @Test
    public void testSetThreads() {
        ModelParameters p = new ModelParameters().setThreads(8);
        assertThat(p.parameters.get("--threads"), is("8"));
    }

    @Test
    public void testSetThreadsBatch() {
        ModelParameters p = new ModelParameters().setThreadsBatch(4);
        assertThat(p.parameters.get("--threads-batch"), is("4"));
    }

    // -------------------------------------------------------------------------
    // CPU Affinity
    // -------------------------------------------------------------------------

    @Test
    public void testSetCpuMask() {
        ModelParameters p = new ModelParameters().setCpuMask("ff");
        assertThat(p.parameters.get("--cpu-mask"), is("ff"));
    }

    @Test
    public void testSetCpuRange() {
        ModelParameters p = new ModelParameters().setCpuRange("0-3");
        assertThat(p.parameters.get("--cpu-range"), is("0-3"));
    }

    @Test
    public void testSetCpuStrict() {
        ModelParameters p = new ModelParameters().setCpuStrict(1);
        assertThat(p.parameters.get("--cpu-strict"), is("1"));
    }

    @Test
    public void testSetPoll() {
        ModelParameters p = new ModelParameters().setPoll(50);
        assertThat(p.parameters.get("--poll"), is("50"));
    }

    @Test
    public void testSetCpuMaskBatch() {
        ModelParameters p = new ModelParameters().setCpuMaskBatch("0f");
        assertThat(p.parameters.get("--cpu-mask-batch"), is("0f"));
    }

    @Test
    public void testSetCpuRangeBatch() {
        ModelParameters p = new ModelParameters().setCpuRangeBatch("4-7");
        assertThat(p.parameters.get("--cpu-range-batch"), is("4-7"));
    }

    @Test
    public void testSetCpuStrictBatch() {
        ModelParameters p = new ModelParameters().setCpuStrictBatch(0);
        assertThat(p.parameters.get("--cpu-strict-batch"), is("0"));
    }

    @Test
    public void testSetPollBatch() {
        ModelParameters p = new ModelParameters().setPollBatch(100);
        assertThat(p.parameters.get("--poll-batch"), is("100"));
    }

    // -------------------------------------------------------------------------
    // Sampling scalars
    // -------------------------------------------------------------------------

    @Test
    public void testSetTemp() {
        ModelParameters p = new ModelParameters().setTemp(0.7f);
        assertThat(p.parameters.get("--temp"), is("0.7"));
    }

    @Test
    public void testSetTopK() {
        ModelParameters p = new ModelParameters().setTopK(50);
        assertThat(p.parameters.get("--top-k"), is("50"));
    }

    @Test
    public void testSetTopKDisabled() {
        ModelParameters p = new ModelParameters().setTopK(0);
        assertThat(p.parameters.get("--top-k"), is("0"));
    }

    @Test
    public void testSetTopP() {
        ModelParameters p = new ModelParameters().setTopP(0.9f);
        assertThat(p.parameters.get("--top-p"), is("0.9"));
    }

    @Test
    public void testSetMinP() {
        ModelParameters p = new ModelParameters().setMinP(0.1f);
        assertThat(p.parameters.get("--min-p"), is("0.1"));
    }

    @Test
    public void testSetTypical() {
        ModelParameters p = new ModelParameters().setTypical(0.95f);
        assertThat(p.parameters.get("--typical"), is("0.95"));
    }

    @Test
    public void testSetRepeatPenalty() {
        ModelParameters p = new ModelParameters().setRepeatPenalty(1.1f);
        assertThat(p.parameters.get("--repeat-penalty"), is("1.1"));
    }

    @Test
    public void testSetPresencePenalty() {
        ModelParameters p = new ModelParameters().setPresencePenalty(0.5f);
        assertThat(p.parameters.get("--presence-penalty"), is("0.5"));
    }

    @Test
    public void testSetFrequencyPenalty() {
        ModelParameters p = new ModelParameters().setFrequencyPenalty(0.3f);
        assertThat(p.parameters.get("--frequency-penalty"), is("0.3"));
    }

    @Test
    public void testSetMirostatLR() {
        ModelParameters p = new ModelParameters().setMirostatLR(0.2f);
        assertThat(p.parameters.get("--mirostat-lr"), is("0.2"));
    }

    @Test
    public void testSetMirostatEnt() {
        ModelParameters p = new ModelParameters().setMirostatEnt(4.0f);
        assertThat(p.parameters.get("--mirostat-ent"), is("4.0"));
    }

    @Test
    public void testSetDynatempRange() {
        ModelParameters p = new ModelParameters().setDynatempRange(0.5f);
        assertThat(p.parameters.get("--dynatemp-range"), is("0.5"));
    }

    @Test
    public void testSetDynatempExponent() {
        ModelParameters p = new ModelParameters().setDynatempExponent(2.0f);
        assertThat(p.parameters.get("--dynatemp-exp"), is("2.0"));
    }

    // -------------------------------------------------------------------------
    // XTC sampling
    // -------------------------------------------------------------------------

    @Test
    public void testSetXtcProbability() {
        ModelParameters p = new ModelParameters().setXtcProbability(0.5f);
        assertThat(p.parameters.get("--xtc-probability"), is("0.5"));
    }

    @Test
    public void testSetXtcProbabilityDisabled() {
        ModelParameters p = new ModelParameters().setXtcProbability(0.0f);
        assertThat(p.parameters.get("--xtc-probability"), is("0.0"));
    }

    @Test
    public void testSetXtcThreshold() {
        ModelParameters p = new ModelParameters().setXtcThreshold(0.2f);
        assertThat(p.parameters.get("--xtc-threshold"), is("0.2"));
    }

    @Test
    public void testSetXtcThresholdDisabled() {
        ModelParameters p = new ModelParameters().setXtcThreshold(1.0f);
        assertThat(p.parameters.get("--xtc-threshold"), is("1.0"));
    }

    // -------------------------------------------------------------------------
    // DRY sampling
    // -------------------------------------------------------------------------

    @Test
    public void testSetDryMultiplier() {
        ModelParameters p = new ModelParameters().setDryMultiplier(0.8f);
        assertThat(p.parameters.get("--dry-multiplier"), is("0.8"));
    }

    @Test
    public void testSetDryMultiplierDisabled() {
        ModelParameters p = new ModelParameters().setDryMultiplier(0.0f);
        assertThat(p.parameters.get("--dry-multiplier"), is("0.0"));
    }

    @Test
    public void testSetDryBase() {
        ModelParameters p = new ModelParameters().setDryBase(2.0f);
        assertThat(p.parameters.get("--dry-base"), is("2.0"));
    }

    @Test
    public void testSetDryAllowedLength() {
        ModelParameters p = new ModelParameters().setDryAllowedLength(3);
        assertThat(p.parameters.get("--dry-allowed-length"), is("3"));
    }

    @Test
    public void testSetDrySequenceBreaker() {
        ModelParameters p = new ModelParameters().setDrySequenceBreaker("\\n");
        assertThat(p.parameters.get("--dry-sequence-breaker"), is("\\n"));
    }

    // -------------------------------------------------------------------------
    // RoPE parameters
    // -------------------------------------------------------------------------

    @Test
    public void testSetRopeScale() {
        ModelParameters p = new ModelParameters().setRopeScale(2.0f);
        assertThat(p.parameters.get("--rope-scale"), is("2.0"));
    }

    @Test
    public void testSetRopeFreqBase() {
        ModelParameters p = new ModelParameters().setRopeFreqBase(10000.0f);
        assertThat(p.parameters.get("--rope-freq-base"), is("10000.0"));
    }

    @Test
    public void testSetRopeFreqScale() {
        ModelParameters p = new ModelParameters().setRopeFreqScale(0.5f);
        assertThat(p.parameters.get("--rope-freq-scale"), is("0.5"));
    }

    // -------------------------------------------------------------------------
    // YaRN parameters
    // -------------------------------------------------------------------------

    @Test
    public void testSetYarnOrigCtx() {
        ModelParameters p = new ModelParameters().setYarnOrigCtx(4096);
        assertThat(p.parameters.get("--yarn-orig-ctx"), is("4096"));
    }

    @Test
    public void testSetYarnExtFactor() {
        ModelParameters p = new ModelParameters().setYarnExtFactor(0.5f);
        assertThat(p.parameters.get("--yarn-ext-factor"), is("0.5"));
    }

    @Test
    public void testSetYarnAttnFactor() {
        ModelParameters p = new ModelParameters().setYarnAttnFactor(1.5f);
        assertThat(p.parameters.get("--yarn-attn-factor"), is("1.5"));
    }

    @Test
    public void testSetYarnBetaSlow() {
        ModelParameters p = new ModelParameters().setYarnBetaSlow(2.0f);
        assertThat(p.parameters.get("--yarn-beta-slow"), is("2.0"));
    }

    @Test
    public void testSetYarnBetaFast() {
        ModelParameters p = new ModelParameters().setYarnBetaFast(16.0f);
        assertThat(p.parameters.get("--yarn-beta-fast"), is("16.0"));
    }

    // -------------------------------------------------------------------------
    // Group attention
    // -------------------------------------------------------------------------

    @Test
    public void testSetGrpAttnN() {
        ModelParameters p = new ModelParameters().setGrpAttnN(4);
        assertThat(p.parameters.get("--grp-attn-n"), is("4"));
    }

    @Test
    public void testSetGrpAttnW() {
        ModelParameters p = new ModelParameters().setGrpAttnW(1024);
        assertThat(p.parameters.get("--grp-attn-w"), is("1024"));
    }

    // -------------------------------------------------------------------------
    // KV cache
    // -------------------------------------------------------------------------

    @Test
    public void testSetCacheTypeKAllValues() {
        for (CacheType ct : CacheType.values()) {
            ModelParameters p = new ModelParameters().setCacheTypeK(ct);
            assertThat(p.parameters.get("--cache-type-k"), is(ct.name().toLowerCase()));
        }
    }

    @Test
    public void testSetCacheTypeVAllValues() {
        for (CacheType ct : CacheType.values()) {
            ModelParameters p = new ModelParameters().setCacheTypeV(ct);
            assertThat(p.parameters.get("--cache-type-v"), is(ct.name().toLowerCase()));
        }
    }

    @Test
    public void testSetDefragThold() {
        ModelParameters p = new ModelParameters().setDefragThold(0.2f);
        assertThat(p.parameters.get("--defrag-thold"), is("0.2"));
    }

    @Test
    public void testSetDefragTholdDisabled() {
        ModelParameters p = new ModelParameters().setDefragThold(-1.0f);
        assertThat(p.parameters.get("--defrag-thold"), is("-1.0"));
    }

    @Test
    public void testDisableKvOffload() {
        ModelParameters p = new ModelParameters().disableKvOffload();
        assertThat(p.parameters, hasKey("--no-kv-offload"));
        assertThat(p.parameters.get("--no-kv-offload"), is(nullValue()));
    }

    @Test
    public void testEnableDumpKvCache() {
        ModelParameters p = new ModelParameters().enableDumpKvCache();
        assertThat(p.parameters, hasKey("--dump-kv-cache"));
        assertThat(p.parameters.get("--dump-kv-cache"), is(nullValue()));
    }

    @Test
    public void testSetKvUnifiedTrue() {
        ModelParameters p = new ModelParameters().setKvUnified(true);
        assertThat(p.parameters, hasKey("--kv-unified"));
        assertThat(p.parameters.get("--kv-unified"), is(nullValue()));
        assertThat(p.parameters, not(hasKey("--no-kv-unified")));
    }

    @Test
    public void testSetKvUnifiedFalse() {
        ModelParameters p = new ModelParameters().setKvUnified(false);
        assertThat(p.parameters, hasKey("--no-kv-unified"));
        assertThat(p.parameters.get("--no-kv-unified"), is(nullValue()));
        assertThat(p.parameters, not(hasKey("--kv-unified")));
    }

    @Test
    public void testSetKvUnifiedFlipFromTrueToFalse() {
        ModelParameters p = new ModelParameters().setKvUnified(true).setKvUnified(false);
        assertThat(p.parameters, hasKey("--no-kv-unified"));
        assertThat(p.parameters, not(hasKey("--kv-unified")));
    }

    @Test
    public void testSetKvUnifiedFlipFromFalseToTrue() {
        ModelParameters p = new ModelParameters().setKvUnified(false).setKvUnified(true);
        assertThat(p.parameters, hasKey("--kv-unified"));
        assertThat(p.parameters, not(hasKey("--no-kv-unified")));
    }

    @Test
    public void testSetCacheRamMib() {
        ModelParameters p = new ModelParameters().setCacheRamMib(4096);
        assertThat(p.parameters.get("--cache-ram"), is("4096"));
    }

    @Test
    public void testSetCacheRamMibUnlimited() {
        ModelParameters p = new ModelParameters().setCacheRamMib(-1);
        assertThat(p.parameters.get("--cache-ram"), is("-1"));
    }

    @Test
    public void testSetCacheRamMibDisabled() {
        ModelParameters p = new ModelParameters().setCacheRamMib(0);
        assertThat(p.parameters.get("--cache-ram"), is("0"));
    }

    @Test
    public void testSetClearIdleTrue() {
        ModelParameters p = new ModelParameters().setClearIdle(true);
        assertThat(p.parameters, hasKey("--cache-idle-slots"));
        assertThat(p.parameters.get("--cache-idle-slots"), is(nullValue()));
        assertThat(p.parameters, not(hasKey("--no-cache-idle-slots")));
    }

    @Test
    public void testSetClearIdleFalse() {
        ModelParameters p = new ModelParameters().setClearIdle(false);
        assertThat(p.parameters, hasKey("--no-cache-idle-slots"));
        assertThat(p.parameters.get("--no-cache-idle-slots"), is(nullValue()));
        assertThat(p.parameters, not(hasKey("--cache-idle-slots")));
    }

    @Test
    public void testSetClearIdleFlipFromTrueToFalse() {
        ModelParameters p = new ModelParameters().setClearIdle(true).setClearIdle(false);
        assertThat(p.parameters, hasKey("--no-cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--cache-idle-slots")));
    }

    @Test
    public void testSetClearIdleFlipFromFalseToTrue() {
        ModelParameters p = new ModelParameters().setClearIdle(false).setClearIdle(true);
        assertThat(p.parameters, hasKey("--cache-idle-slots"));
        assertThat(p.parameters, not(hasKey("--no-cache-idle-slots")));
    }

    @Test
    public void testKvUnifiedCacheRamClearIdleChaining() {
        // All three features wired together as they would be in production use
        ModelParameters p =
                new ModelParameters().setKvUnified(true).setCacheRamMib(8192).setClearIdle(true);
        assertThat(p.parameters, hasKey("--kv-unified"));
        assertThat(p.parameters.get("--cache-ram"), is("8192"));
        assertThat(p.parameters, hasKey("--cache-idle-slots"));
        // Opposite flags must be absent
        assertThat(p.parameters, not(hasKey("--no-kv-unified")));
        assertThat(p.parameters, not(hasKey("--no-cache-idle-slots")));
    }

    @Test
    public void testSetKvUnifiedReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setKvUnified(true), is(sameInstance(p)));
    }

    @Test
    public void testSetCacheRamMibReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setCacheRamMib(4096), is(sameInstance(p)));
    }

    @Test
    public void testSetClearIdleReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setClearIdle(true), is(sameInstance(p)));
    }

    // -------------------------------------------------------------------------
    // GPU / Split mode
    // -------------------------------------------------------------------------

    @Test
    public void testSetGpuLayers() {
        ModelParameters p = new ModelParameters().setGpuLayers(32);
        assertThat(p.parameters.get("--gpu-layers"), is("32"));
    }

    @Test
    public void testSetSplitModeAllValues() {
        for (GpuSplitMode mode : GpuSplitMode.values()) {
            ModelParameters p = new ModelParameters().setSplitMode(mode);
            assertThat(p.parameters.get("--split-mode"), is(mode.name().toLowerCase()));
        }
    }

    @Test
    public void testSetTensorSplit() {
        ModelParameters p = new ModelParameters().setTensorSplit("0.5,0.5");
        assertThat(p.parameters.get("--tensor-split"), is("0.5,0.5"));
    }

    @Test
    public void testSetMainGpu() {
        ModelParameters p = new ModelParameters().setMainGpu(1);
        assertThat(p.parameters.get("--main-gpu"), is("1"));
    }

    @Test
    public void testSetDevices() {
        ModelParameters p = new ModelParameters().setDevices("cuda0,cuda1");
        assertThat(p.parameters.get("--device"), is("cuda0,cuda1"));
    }

    // -------------------------------------------------------------------------
    // Memory management
    // -------------------------------------------------------------------------

    @Test
    public void testEnableMlock() {
        ModelParameters p = new ModelParameters().enableMlock();
        assertThat(p.parameters, hasKey("--mlock"));
        assertThat(p.parameters.get("--mlock"), is(nullValue()));
    }

    @Test
    public void testDisableMmap() {
        ModelParameters p = new ModelParameters().disableMmap();
        assertThat(p.parameters, hasKey("--no-mmap"));
        assertThat(p.parameters.get("--no-mmap"), is(nullValue()));
    }

    @Test
    public void testSetNumaAllValues() {
        for (NumaStrategy ns : NumaStrategy.values()) {
            ModelParameters p = new ModelParameters().setNuma(ns);
            assertThat(p.parameters.get("--numa"), is(ns.name().toLowerCase()));
        }
    }

    // -------------------------------------------------------------------------
    // Parallel / continuous batching
    // -------------------------------------------------------------------------

    @Test
    public void testSetParallel() {
        ModelParameters p = new ModelParameters().setParallel(4);
        assertThat(p.parameters.get("--parallel"), is("4"));
    }

    @Test
    public void testEnableContBatching() {
        ModelParameters p = new ModelParameters().enableContBatching();
        assertThat(p.parameters, hasKey("--cont-batching"));
        assertThat(p.parameters.get("--cont-batching"), is(nullValue()));
    }

    @Test
    public void testDisableContBatching() {
        ModelParameters p = new ModelParameters().disableContBatching();
        assertThat(p.parameters, hasKey("--no-cont-batching"));
        assertThat(p.parameters.get("--no-cont-batching"), is(nullValue()));
    }

    // -------------------------------------------------------------------------
    // Flag-only toggles
    // -------------------------------------------------------------------------

    @Test
    public void testDisableContextShift() {
        ModelParameters p = new ModelParameters().disableContextShift();
        assertThat(p.parameters, hasKey("--no-context-shift"));
        assertThat(p.parameters.get("--no-context-shift"), is(nullValue()));
    }

    @Test
    public void testSetFlashAttnRendersEveryMode() {
        // Every mode the enum declares must reach argv as its own CLI string; a mode added later is
        // then covered without touching this test.
        for (FlashAttn mode : FlashAttn.values()) {
            ModelParameters p = new ModelParameters().setFlashAttn(mode);
            assertThat(p.parameters, hasKey("--flash-attn"));
            assertThat(p.parameters.get("--flash-attn"), is(mode.getArgValue()));
        }
    }

    @Test
    public void testSetFlashAttnPutsTheValueAfterTheKeyInArgv() {
        // The point of the whole change: the value must be its own argv token immediately after the
        // key. Anything else and llama.cpp consumes whatever follows as the mode.
        String[] arr = new ModelParameters().setFlashAttn(FlashAttn.ON).toArray();
        int key = -1;
        for (int i = 0; i < arr.length; i++) {
            if ("--flash-attn".equals(arr[i])) {
                key = i;
                break;
            }
        }
        assertThat("--flash-attn missing from argv", key, is(not(-1)));
        assertThat("--flash-attn is the last token, so its mandatory value is missing", key < arr.length - 1, is(true));
        assertThat(arr[key + 1], is("on"));
    }

    @Test
    public void testEnableSwaFull() {
        ModelParameters p = new ModelParameters().enableSwaFull();
        assertThat(p.parameters, hasKey("--swa-full"));
        assertThat(p.parameters.get("--swa-full"), is(nullValue()));
    }

    @Test
    public void testSwaFullNotEnabledByDefault() {
        assertThat(new ModelParameters().parameters, not(hasKey("--swa-full")));
    }

    @Test
    public void testDisablePerf() {
        ModelParameters p = new ModelParameters().disablePerf();
        assertThat(p.parameters, hasKey("--no-perf"));
        assertThat(p.parameters.get("--no-perf"), is(nullValue()));
    }

    @Test
    public void testEnableEscape() {
        ModelParameters p = new ModelParameters().enableEscape();
        assertThat(p.parameters, hasKey("--escape"));
        assertThat(p.parameters.get("--escape"), is(nullValue()));
    }

    @Test
    public void testDisableEscape() {
        ModelParameters p = new ModelParameters().disableEscape();
        assertThat(p.parameters, hasKey("--no-escape"));
        assertThat(p.parameters.get("--no-escape"), is(nullValue()));
    }

    @Test
    public void testEnableSpecial() {
        ModelParameters p = new ModelParameters().enableSpecial();
        assertThat(p.parameters, hasKey("--special"));
        assertThat(p.parameters.get("--special"), is(nullValue()));
    }

    @Test
    public void testSkipWarmup() {
        ModelParameters p = new ModelParameters().skipWarmup();
        assertThat(p.parameters, hasKey("--no-warmup"));
        assertThat(p.parameters.get("--no-warmup"), is(nullValue()));
    }

    @Test
    public void testSetSpmInfill() {
        ModelParameters p = new ModelParameters().setSpmInfill();
        assertThat(p.parameters, hasKey("--spm-infill"));
        assertThat(p.parameters.get("--spm-infill"), is(nullValue()));
    }

    @Test
    public void testIgnoreEos() {
        ModelParameters p = new ModelParameters().ignoreEos();
        assertThat(p.parameters, hasKey("--ignore-eos"));
        assertThat(p.parameters.get("--ignore-eos"), is(nullValue()));
    }

    @Test
    public void testEnableCheckTensors() {
        ModelParameters p = new ModelParameters().enableCheckTensors();
        assertThat(p.parameters, hasKey("--check-tensors"));
        assertThat(p.parameters.get("--check-tensors"), is(nullValue()));
    }

    @Test
    public void testEnableEmbedding() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        assertThat(p.parameters, hasKey("--embedding"));
        assertThat(p.parameters.get("--embedding"), is(nullValue()));
    }

    @Test
    public void testEnableReranking() {
        ModelParameters p = new ModelParameters().enableReranking();
        assertThat(p.parameters, hasKey("--reranking"));
        assertThat(p.parameters.get("--reranking"), is(nullValue()));
    }

    @Test
    public void testSetVocabOnly() {
        ModelParameters p = new ModelParameters().setVocabOnly();
        assertThat(p.parameters, hasKey("--vocab-only"));
        assertThat(p.parameters.get("--vocab-only"), is(nullValue()));
    }

    @Test
    public void testEnableJinja() {
        ModelParameters p = new ModelParameters().enableJinja();
        assertThat(p.parameters, hasKey("--jinja"));
        assertThat(p.parameters.get("--jinja"), is(nullValue()));
    }

    @Test
    public void testSetLoraInitWithoutApply() {
        ModelParameters p = new ModelParameters().setLoraInitWithoutApply();
        assertThat(p.parameters, hasKey("--lora-init-without-apply"));
        assertThat(p.parameters.get("--lora-init-without-apply"), is(nullValue()));
    }

    // -------------------------------------------------------------------------
    // Seed / Logit bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetSeed() {
        ModelParameters p = new ModelParameters().setSeed(42);
        assertThat(p.parameters.get("--seed"), is("42"));
    }

    @Test
    public void testSetSeedRandom() {
        ModelParameters p = new ModelParameters().setSeed(-1);
        assertThat(p.parameters.get("--seed"), is("-1"));
    }

    @Test
    public void testSetLogitBias() {
        ModelParameters p = new ModelParameters().setLogitBias("1+0.5");
        assertThat(p.parameters.get("--logit-bias"), is("1+0.5"));
    }

    // -------------------------------------------------------------------------
    // Grammar / JSON schema
    // -------------------------------------------------------------------------

    @Test
    public void testSetGrammar() {
        ModelParameters p = new ModelParameters().setGrammar("root ::= \"hello\"");
        assertThat(p.parameters.get("--grammar"), is("root ::= \"hello\""));
    }

    @Test
    public void testSetGrammarFile() {
        ModelParameters p = new ModelParameters().setGrammarFile("grammar.gbnf");
        assertThat(p.parameters.get("--grammar-file"), is("grammar.gbnf"));
    }

    @Test
    public void testSetJsonSchema() {
        ModelParameters p = new ModelParameters().setJsonSchema("{\"type\":\"object\"}");
        assertThat(p.parameters.get("--json-schema"), is("{\"type\":\"object\"}"));
    }

    // -------------------------------------------------------------------------
    // Chat template
    // -------------------------------------------------------------------------

    @Test
    public void testSetChatTemplate() {
        ModelParameters p =
                new ModelParameters().setChatTemplate("{% for msg in messages %}{{ msg.content }}{% endfor %}");
        assertThat(p.parameters.get("--chat-template"), is("{% for msg in messages %}{{ msg.content }}{% endfor %}"));
    }

    @Test
    public void testSetChatTemplateKwargs() {
        Map<String, String> kwargs = new HashMap<>();
        kwargs.put("enable_thinking", "true");
        ModelParameters p = new ModelParameters().setChatTemplateKwargs(kwargs);
        String val = p.parameters.get("--chat-template-kwargs");
        assertThat(val, is(notNullValue()));
        assertThat(val, containsString("\"enable_thinking\":true"));
    }

    @Test
    public void testSetChatTemplateKwargsMultiple() {
        Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        kwargs.put("key1", "\"val1\"");
        kwargs.put("key2", "42");
        ModelParameters p = new ModelParameters().setChatTemplateKwargs(kwargs);
        String val = p.parameters.get("--chat-template-kwargs");
        assertThat(val, is(notNullValue()));
        assertThat(val, startsWith("{"));
        assertThat(val, endsWith("}"));
        assertThat(val, containsString("\"key1\":\"val1\""));
        assertThat(val, containsString("\"key2\":42"));
    }

    // -------------------------------------------------------------------------
    // Model loading
    // -------------------------------------------------------------------------

    @Test
    public void testSetModel() {
        ModelParameters p = new ModelParameters().setModel("/path/to/model.gguf");
        assertThat(p.parameters.get("--model"), is("/path/to/model.gguf"));
    }

    @Test
    public void testSetModelUrl() {
        ModelParameters p = new ModelParameters().setModelUrl("https://example.com/model.gguf");
        assertThat(p.parameters.get("--model-url"), is("https://example.com/model.gguf"));
    }

    @Test
    public void testSetHfRepo() {
        ModelParameters p = new ModelParameters().setHfRepo("meta-llama/Llama-2-7b");
        assertThat(p.parameters.get("--hf-repo"), is("meta-llama/Llama-2-7b"));
    }

    @Test
    public void testSetHfFile() {
        ModelParameters p = new ModelParameters().setHfFile("model-q4.gguf");
        assertThat(p.parameters.get("--hf-file"), is("model-q4.gguf"));
    }

    @Test
    public void testSetHfToken() {
        ModelParameters p = new ModelParameters().setHfToken("hf_abc123");
        assertThat(p.parameters.get("--hf-token"), is("hf_abc123"));
    }

    @Test
    public void testSetHfRepoV() {
        ModelParameters p = new ModelParameters().setHfRepoV("org/vocoder");
        assertThat(p.parameters.get("--hf-repo-v"), is("org/vocoder"));
    }

    @Test
    public void testSetHfFileV() {
        ModelParameters p = new ModelParameters().setHfFileV("vocoder.gguf");
        assertThat(p.parameters.get("--hf-file-v"), is("vocoder.gguf"));
    }

    // -------------------------------------------------------------------------
    // Slot / cache reuse
    // -------------------------------------------------------------------------

    @Test
    public void testSetCacheReuse() {
        ModelParameters p = new ModelParameters().setCacheReuse(128);
        assertThat(p.parameters.get("--cache-reuse"), is("128"));
    }

    @Test
    public void testSetSlotSavePath() {
        ModelParameters p = new ModelParameters().setSlotSavePath("/tmp/slots");
        assertThat(p.parameters.get("--slot-save-path"), is("/tmp/slots"));
    }

    @Test
    public void testSetSlotPromptSimilarity() {
        ModelParameters p = new ModelParameters().setSlotPromptSimilarity(0.8f);
        assertThat(p.parameters.get("--slot-prompt-similarity"), is("0.8"));
    }

    @Test
    public void testSetCtxCheckpoints() {
        ModelParameters p = new ModelParameters().setCtxCheckpoints(8);
        assertThat(p.parameters.get("--ctx-checkpoints"), is("8"));
    }

    @Test
    public void testSetCheckpointMinStep() {
        ModelParameters p = new ModelParameters().setCheckpointMinStep(0);
        assertThat(p.parameters.get("--checkpoint-min-step"), is("0"));
    }

    // -------------------------------------------------------------------------
    // Override KV
    // -------------------------------------------------------------------------

    @Test
    public void testSetOverrideKv() {
        ModelParameters p = new ModelParameters().setOverrideKv("tokenizer.ggml.pre=spm");
        assertThat(p.parameters.get("--override-kv"), is("tokenizer.ggml.pre=spm"));
    }

    // -------------------------------------------------------------------------
    // LoRA / Control vector (non-scaled variants)
    // -------------------------------------------------------------------------

    @Test
    public void testAddLoraAdapter() {
        ModelParameters p = new ModelParameters().addLoraAdapter("adapter.bin");
        assertThat(p.parameters.get("--lora"), is("adapter.bin"));
    }

    @Test
    public void testAddControlVector() {
        ModelParameters p = new ModelParameters().addControlVector("vec.bin");
        assertThat(p.parameters.get("--control-vector"), is("vec.bin"));
    }

    // -------------------------------------------------------------------------
    // Speculative decoding
    // -------------------------------------------------------------------------

    @Test
    public void testSetModelDraft() {
        ModelParameters p = new ModelParameters().setModelDraft("/path/to/draft.gguf");
        assertThat(p.parameters.get("--spec-draft-model"), is("/path/to/draft.gguf"));
    }

    @Test
    public void testSetDeviceDraft() {
        ModelParameters p = new ModelParameters().setDeviceDraft("cuda0");
        assertThat(p.parameters.get("--spec-draft-device"), is("cuda0"));
    }

    @Test
    public void testSetGpuLayersDraft() {
        ModelParameters p = new ModelParameters().setGpuLayersDraft(16);
        assertThat(p.parameters.get("--spec-draft-ngl"), is("16"));
    }

    @Test
    public void testSetDraftMax() {
        // Regression: --draft-max was REMOVED in b9016 and now throws std::invalid_argument
        // at model load. Must use --spec-draft-n-max.
        ModelParameters p = new ModelParameters().setDraftMax(8);
        assertThat(p.parameters.get("--spec-draft-n-max"), is("8"));
        assertThat("--draft-max throws on b9016+; must not appear in args", p.parameters, not(hasKey("--draft-max")));
    }

    @Test
    public void testSetDraftMin() {
        // Regression: --draft-min was REMOVED in b9016 and now throws std::invalid_argument
        // at model load. Must use --spec-draft-n-min.
        ModelParameters p = new ModelParameters().setDraftMin(2);
        assertThat(p.parameters.get("--spec-draft-n-min"), is("2"));
        assertThat("--draft-min throws on b9016+; must not appear in args", p.parameters, not(hasKey("--draft-min")));
    }

    @Test
    public void testSetDraftPMin() {
        ModelParameters p = new ModelParameters().setDraftPMin(0.5f);
        assertThat(p.parameters.get("--spec-draft-p-min"), is("0.5"));
    }

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    @Test
    public void testDisableLog() {
        ModelParameters p = new ModelParameters().disableLog();
        assertThat(p.parameters, hasKey("--log-disable"));
        assertThat(p.parameters.get("--log-disable"), is(nullValue()));
    }

    @Test
    public void testSetLogFile() {
        ModelParameters p = new ModelParameters().setLogFile("/tmp/llama.log");
        assertThat(p.parameters.get("--log-file"), is("/tmp/llama.log"));
    }

    @Test
    public void testSetVerbose() {
        ModelParameters p = new ModelParameters().setVerbose();
        assertThat(p.parameters, hasKey("--verbose"));
        assertThat(p.parameters.get("--verbose"), is(nullValue()));
    }

    @Test
    public void testSetLogVerbosity() {
        ModelParameters p = new ModelParameters().setLogVerbosity(3);
        assertThat(p.parameters.get("--log-verbosity"), is("3"));
    }

    @Test
    public void testEnableLogPrefix() {
        ModelParameters p = new ModelParameters().enableLogPrefix();
        assertThat(p.parameters, hasKey("--log-prefix"));
        assertThat(p.parameters.get("--log-prefix"), is(nullValue()));
    }

    @Test
    public void testEnableLogTimestamps() {
        ModelParameters p = new ModelParameters().enableLogTimestamps();
        assertThat(p.parameters, hasKey("--log-timestamps"));
        assertThat(p.parameters.get("--log-timestamps"), is(nullValue()));
    }

    // -------------------------------------------------------------------------
    // setFit variations
    // -------------------------------------------------------------------------

    @Test
    public void testSetFitTrue() {
        ModelParameters p = new ModelParameters().setFit(true);
        assertThat(p.parameters.get("--fit"), is(ModelParameters.FIT_ON));
    }

    @Test
    public void testSetFitFalse() {
        ModelParameters p = new ModelParameters().setFit(false);
        assertThat(p.parameters.get("--fit"), is(ModelParameters.FIT_OFF));
    }

    // -------------------------------------------------------------------------
    // RoPE scaling type — all values
    // -------------------------------------------------------------------------

    @Test
    public void testSetRopeScalingAllValues() {
        for (RopeScalingType type : RopeScalingType.values()) {
            ModelParameters p = new ModelParameters().setRopeScaling(type);
            assertThat(p.parameters.get("--rope-scaling"), is(type.getArgValue()));
        }
    }

    // -------------------------------------------------------------------------
    // MiroStat — all values
    // -------------------------------------------------------------------------

    @Test
    public void testSetMirostatAllValues() {
        for (MiroStat m : MiroStat.values()) {
            ModelParameters p = new ModelParameters().setMirostat(m);
            // Assert against the enum's CLI arg-value contract (what setMirostat
            // actually writes), not Enum.ordinal() (Error Prone EnumOrdinal).
            assertThat(p.parameters.get("--mirostat"), is(m.getArgValue()));
        }
    }

    // -------------------------------------------------------------------------
    // Chaining — extended chaining across categories
    // -------------------------------------------------------------------------

    @Test
    public void testExtendedChainingReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertThat(p.setCtxSize(2048), is(sameInstance(p)));
        assertThat(p.setBatchSize(512), is(sameInstance(p)));
        assertThat(p.setTemp(0.7f), is(sameInstance(p)));
        assertThat(p.setTopK(50), is(sameInstance(p)));
        assertThat(p.setDryMultiplier(0.5f), is(sameInstance(p)));
        assertThat(p.setXtcProbability(0.3f), is(sameInstance(p)));
        assertThat(p.setRopeScale(2.0f), is(sameInstance(p)));
        assertThat(p.setGpuLayers(32), is(sameInstance(p)));
        assertThat(p.setFlashAttn(FlashAttn.AUTO), is(sameInstance(p)));
        assertThat(p.disableContextShift(), is(sameInstance(p)));
        assertThat(p.setModelDraft("/draft.gguf"), is(sameInstance(p)));
        assertThat(p.disableLog(), is(sameInstance(p)));
    }

    // -------------------------------------------------------------------------
    // toArray — complex parameter combination
    // -------------------------------------------------------------------------

    @Test
    public void testToArrayComplexCombination() {
        ModelParameters p = new ModelParameters()
                .setModel("model.gguf")
                .setCtxSize(2048)
                .enableEmbedding()
                .setFlashAttn(FlashAttn.ON);
        String[] arr = p.toArray();
        // argv[0]="" + --fit + on + --model + model.gguf + --ctx-size + 2048 + --embedding
        //   + --flash-attn + on = 10.
        // This used to call the removed enableFlashAttn() and assert 9, which pinned the valueless
        // emission as the expected argv shape -- the very defect setFlashAttn exists to fix.
        assertThat(arr, arrayWithSize(10));
        assertThat(arr[0], is(""));
    }

    // -------------------------------------------------------------------------
    // isUnset — extended
    // -------------------------------------------------------------------------

    @Test
    public void testIsDefaultForCtxSize() {
        ModelParameters p = new ModelParameters();
        assertThat(p.isUnset("ctx-size"), is(true));
        p.setCtxSize(2048);
        assertThat(p.isUnset("ctx-size"), is(false));
    }

    @Test
    public void testIsDefaultForFlagOnly() {
        // Uses swa-full, not flash-attn: the latter is no longer a valueless flag, which is the whole
        // point of the FlashAttn enum.
        ModelParameters p = new ModelParameters();
        assertThat(p.isUnset("swa-full"), is(true));
        p.enableSwaFull();
        assertThat(p.isUnset("swa-full"), is(false));
    }
}
