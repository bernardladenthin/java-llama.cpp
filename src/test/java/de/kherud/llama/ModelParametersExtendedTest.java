package de.kherud.llama;

import de.kherud.llama.args.*;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Extended tests for {@link ModelParameters} covering CLI argument serialization
 * for all setter methods not already tested in {@link ModelParametersTest}.
 */
@ClaudeGenerated(
        purpose = "Verify CLI argument serialization for all ModelParameters setters not covered by " +
                  "ModelParametersTest: context/batch sizing, threading, sampling scalars, XTC, DRY, " +
                  "RoPE, YaRN, KV cache, GPU, memory, parallel inference, flag-only toggles, " +
                  "speculative decoding, logging, model loading, grammar, chat templates, and advanced options.",
        model = "claude-opus-4-6"
)
public class ModelParametersExtendedTest {

    // -------------------------------------------------------------------------
    // Context / Batch / Predict / Keep
    // -------------------------------------------------------------------------

    @Test
    public void testSetCtxSize() {
        ModelParameters p = new ModelParameters().setCtxSize(2048);
        assertEquals("2048", p.parameters.get("--ctx-size"));
    }

    @Test
    public void testSetCtxSizeZeroUsesModelDefault() {
        ModelParameters p = new ModelParameters().setCtxSize(0);
        assertEquals("0", p.parameters.get("--ctx-size"));
    }

    @Test
    public void testSetBatchSize() {
        ModelParameters p = new ModelParameters().setBatchSize(512);
        assertEquals("512", p.parameters.get("--batch-size"));
    }

    @Test
    public void testSetUbatchSize() {
        ModelParameters p = new ModelParameters().setUbatchSize(256);
        assertEquals("256", p.parameters.get("--ubatch-size"));
    }

    @Test
    public void testSetPredict() {
        ModelParameters p = new ModelParameters().setPredict(100);
        assertEquals("100", p.parameters.get("--predict"));
    }

    @Test
    public void testSetPredictInfinity() {
        ModelParameters p = new ModelParameters().setPredict(-1);
        assertEquals("-1", p.parameters.get("--predict"));
    }

    @Test
    public void testSetPredictFillContext() {
        ModelParameters p = new ModelParameters().setPredict(-2);
        assertEquals("-2", p.parameters.get("--predict"));
    }

    @Test
    public void testSetKeep() {
        ModelParameters p = new ModelParameters().setKeep(64);
        assertEquals("64", p.parameters.get("--keep"));
    }

    @Test
    public void testSetKeepAll() {
        ModelParameters p = new ModelParameters().setKeep(-1);
        assertEquals("-1", p.parameters.get("--keep"));
    }

    // -------------------------------------------------------------------------
    // Threading
    // -------------------------------------------------------------------------

    @Test
    public void testSetThreads() {
        ModelParameters p = new ModelParameters().setThreads(8);
        assertEquals("8", p.parameters.get("--threads"));
    }

    @Test
    public void testSetThreadsBatch() {
        ModelParameters p = new ModelParameters().setThreadsBatch(4);
        assertEquals("4", p.parameters.get("--threads-batch"));
    }

    // -------------------------------------------------------------------------
    // CPU Affinity
    // -------------------------------------------------------------------------

    @Test
    public void testSetCpuMask() {
        ModelParameters p = new ModelParameters().setCpuMask("ff");
        assertEquals("ff", p.parameters.get("--cpu-mask"));
    }

    @Test
    public void testSetCpuRange() {
        ModelParameters p = new ModelParameters().setCpuRange("0-3");
        assertEquals("0-3", p.parameters.get("--cpu-range"));
    }

    @Test
    public void testSetCpuStrict() {
        ModelParameters p = new ModelParameters().setCpuStrict(1);
        assertEquals("1", p.parameters.get("--cpu-strict"));
    }

    @Test
    public void testSetPoll() {
        ModelParameters p = new ModelParameters().setPoll(50);
        assertEquals("50", p.parameters.get("--poll"));
    }

    @Test
    public void testSetCpuMaskBatch() {
        ModelParameters p = new ModelParameters().setCpuMaskBatch("0f");
        assertEquals("0f", p.parameters.get("--cpu-mask-batch"));
    }

    @Test
    public void testSetCpuRangeBatch() {
        ModelParameters p = new ModelParameters().setCpuRangeBatch("4-7");
        assertEquals("4-7", p.parameters.get("--cpu-range-batch"));
    }

    @Test
    public void testSetCpuStrictBatch() {
        ModelParameters p = new ModelParameters().setCpuStrictBatch(0);
        assertEquals("0", p.parameters.get("--cpu-strict-batch"));
    }

    @Test
    public void testSetPollBatch() {
        ModelParameters p = new ModelParameters().setPollBatch(100);
        assertEquals("100", p.parameters.get("--poll-batch"));
    }

    // -------------------------------------------------------------------------
    // Sampling scalars
    // -------------------------------------------------------------------------

    @Test
    public void testSetTemp() {
        ModelParameters p = new ModelParameters().setTemp(0.7f);
        assertEquals("0.7", p.parameters.get("--temp"));
    }

    @Test
    public void testSetTopK() {
        ModelParameters p = new ModelParameters().setTopK(50);
        assertEquals("50", p.parameters.get("--top-k"));
    }

    @Test
    public void testSetTopKDisabled() {
        ModelParameters p = new ModelParameters().setTopK(0);
        assertEquals("0", p.parameters.get("--top-k"));
    }

    @Test
    public void testSetTopP() {
        ModelParameters p = new ModelParameters().setTopP(0.9f);
        assertEquals("0.9", p.parameters.get("--top-p"));
    }

    @Test
    public void testSetMinP() {
        ModelParameters p = new ModelParameters().setMinP(0.1f);
        assertEquals("0.1", p.parameters.get("--min-p"));
    }

    @Test
    public void testSetTypical() {
        ModelParameters p = new ModelParameters().setTypical(0.95f);
        assertEquals("0.95", p.parameters.get("--typical"));
    }

    @Test
    public void testSetRepeatPenalty() {
        ModelParameters p = new ModelParameters().setRepeatPenalty(1.1f);
        assertEquals("1.1", p.parameters.get("--repeat-penalty"));
    }

    @Test
    public void testSetPresencePenalty() {
        ModelParameters p = new ModelParameters().setPresencePenalty(0.5f);
        assertEquals("0.5", p.parameters.get("--presence-penalty"));
    }

    @Test
    public void testSetFrequencyPenalty() {
        ModelParameters p = new ModelParameters().setFrequencyPenalty(0.3f);
        assertEquals("0.3", p.parameters.get("--frequency-penalty"));
    }

    @Test
    public void testSetMirostatLR() {
        ModelParameters p = new ModelParameters().setMirostatLR(0.2f);
        assertEquals("0.2", p.parameters.get("--mirostat-lr"));
    }

    @Test
    public void testSetMirostatEnt() {
        ModelParameters p = new ModelParameters().setMirostatEnt(4.0f);
        assertEquals("4.0", p.parameters.get("--mirostat-ent"));
    }

    @Test
    public void testSetDynatempRange() {
        ModelParameters p = new ModelParameters().setDynatempRange(0.5f);
        assertEquals("0.5", p.parameters.get("--dynatemp-range"));
    }

    @Test
    public void testSetDynatempExponent() {
        ModelParameters p = new ModelParameters().setDynatempExponent(2.0f);
        assertEquals("2.0", p.parameters.get("--dynatemp-exp"));
    }

    // -------------------------------------------------------------------------
    // XTC sampling
    // -------------------------------------------------------------------------

    @Test
    public void testSetXtcProbability() {
        ModelParameters p = new ModelParameters().setXtcProbability(0.5f);
        assertEquals("0.5", p.parameters.get("--xtc-probability"));
    }

    @Test
    public void testSetXtcProbabilityDisabled() {
        ModelParameters p = new ModelParameters().setXtcProbability(0.0f);
        assertEquals("0.0", p.parameters.get("--xtc-probability"));
    }

    @Test
    public void testSetXtcThreshold() {
        ModelParameters p = new ModelParameters().setXtcThreshold(0.2f);
        assertEquals("0.2", p.parameters.get("--xtc-threshold"));
    }

    @Test
    public void testSetXtcThresholdDisabled() {
        ModelParameters p = new ModelParameters().setXtcThreshold(1.0f);
        assertEquals("1.0", p.parameters.get("--xtc-threshold"));
    }

    // -------------------------------------------------------------------------
    // DRY sampling
    // -------------------------------------------------------------------------

    @Test
    public void testSetDryMultiplier() {
        ModelParameters p = new ModelParameters().setDryMultiplier(0.8f);
        assertEquals("0.8", p.parameters.get("--dry-multiplier"));
    }

    @Test
    public void testSetDryMultiplierDisabled() {
        ModelParameters p = new ModelParameters().setDryMultiplier(0.0f);
        assertEquals("0.0", p.parameters.get("--dry-multiplier"));
    }

    @Test
    public void testSetDryBase() {
        ModelParameters p = new ModelParameters().setDryBase(2.0f);
        assertEquals("2.0", p.parameters.get("--dry-base"));
    }

    @Test
    public void testSetDryAllowedLength() {
        ModelParameters p = new ModelParameters().setDryAllowedLength(3);
        assertEquals("3", p.parameters.get("--dry-allowed-length"));
    }

    @Test
    public void testSetDrySequenceBreaker() {
        ModelParameters p = new ModelParameters().setDrySequenceBreaker("\\n");
        assertEquals("\\n", p.parameters.get("--dry-sequence-breaker"));
    }

    // -------------------------------------------------------------------------
    // RoPE parameters
    // -------------------------------------------------------------------------

    @Test
    public void testSetRopeScale() {
        ModelParameters p = new ModelParameters().setRopeScale(2.0f);
        assertEquals("2.0", p.parameters.get("--rope-scale"));
    }

    @Test
    public void testSetRopeFreqBase() {
        ModelParameters p = new ModelParameters().setRopeFreqBase(10000.0f);
        assertEquals("10000.0", p.parameters.get("--rope-freq-base"));
    }

    @Test
    public void testSetRopeFreqScale() {
        ModelParameters p = new ModelParameters().setRopeFreqScale(0.5f);
        assertEquals("0.5", p.parameters.get("--rope-freq-scale"));
    }

    // -------------------------------------------------------------------------
    // YaRN parameters
    // -------------------------------------------------------------------------

    @Test
    public void testSetYarnOrigCtx() {
        ModelParameters p = new ModelParameters().setYarnOrigCtx(4096);
        assertEquals("4096", p.parameters.get("--yarn-orig-ctx"));
    }

    @Test
    public void testSetYarnExtFactor() {
        ModelParameters p = new ModelParameters().setYarnExtFactor(0.5f);
        assertEquals("0.5", p.parameters.get("--yarn-ext-factor"));
    }

    @Test
    public void testSetYarnAttnFactor() {
        ModelParameters p = new ModelParameters().setYarnAttnFactor(1.5f);
        assertEquals("1.5", p.parameters.get("--yarn-attn-factor"));
    }

    @Test
    public void testSetYarnBetaSlow() {
        ModelParameters p = new ModelParameters().setYarnBetaSlow(2.0f);
        assertEquals("2.0", p.parameters.get("--yarn-beta-slow"));
    }

    @Test
    public void testSetYarnBetaFast() {
        ModelParameters p = new ModelParameters().setYarnBetaFast(16.0f);
        assertEquals("16.0", p.parameters.get("--yarn-beta-fast"));
    }

    // -------------------------------------------------------------------------
    // Group attention
    // -------------------------------------------------------------------------

    @Test
    public void testSetGrpAttnN() {
        ModelParameters p = new ModelParameters().setGrpAttnN(4);
        assertEquals("4", p.parameters.get("--grp-attn-n"));
    }

    @Test
    public void testSetGrpAttnW() {
        ModelParameters p = new ModelParameters().setGrpAttnW(1024);
        assertEquals("1024", p.parameters.get("--grp-attn-w"));
    }

    // -------------------------------------------------------------------------
    // KV cache
    // -------------------------------------------------------------------------

    @Test
    public void testSetCacheTypeKAllValues() {
        for (CacheType ct : CacheType.values()) {
            ModelParameters p = new ModelParameters().setCacheTypeK(ct);
            assertEquals(ct.name().toLowerCase(), p.parameters.get("--cache-type-k"));
        }
    }

    @Test
    public void testSetCacheTypeVAllValues() {
        for (CacheType ct : CacheType.values()) {
            ModelParameters p = new ModelParameters().setCacheTypeV(ct);
            assertEquals(ct.name().toLowerCase(), p.parameters.get("--cache-type-v"));
        }
    }

    @Test
    public void testSetDefragThold() {
        ModelParameters p = new ModelParameters().setDefragThold(0.2f);
        assertEquals("0.2", p.parameters.get("--defrag-thold"));
    }

    @Test
    public void testSetDefragTholdDisabled() {
        ModelParameters p = new ModelParameters().setDefragThold(-1.0f);
        assertEquals("-1.0", p.parameters.get("--defrag-thold"));
    }

    @Test
    public void testDisableKvOffload() {
        ModelParameters p = new ModelParameters().disableKvOffload();
        assertTrue(p.parameters.containsKey("--no-kv-offload"));
        assertNull(p.parameters.get("--no-kv-offload"));
    }

    @Test
    public void testEnableDumpKvCache() {
        ModelParameters p = new ModelParameters().enableDumpKvCache();
        assertTrue(p.parameters.containsKey("--dump-kv-cache"));
        assertNull(p.parameters.get("--dump-kv-cache"));
    }

    // -------------------------------------------------------------------------
    // GPU / Split mode
    // -------------------------------------------------------------------------

    @Test
    public void testSetGpuLayers() {
        ModelParameters p = new ModelParameters().setGpuLayers(32);
        assertEquals("32", p.parameters.get("--gpu-layers"));
    }

    @Test
    public void testSetSplitModeAllValues() {
        for (GpuSplitMode mode : GpuSplitMode.values()) {
            ModelParameters p = new ModelParameters().setSplitMode(mode);
            assertEquals(mode.name().toLowerCase(), p.parameters.get("--split-mode"));
        }
    }

    @Test
    public void testSetTensorSplit() {
        ModelParameters p = new ModelParameters().setTensorSplit("0.5,0.5");
        assertEquals("0.5,0.5", p.parameters.get("--tensor-split"));
    }

    @Test
    public void testSetMainGpu() {
        ModelParameters p = new ModelParameters().setMainGpu(1);
        assertEquals("1", p.parameters.get("--main-gpu"));
    }

    @Test
    public void testSetDevices() {
        ModelParameters p = new ModelParameters().setDevices("cuda0,cuda1");
        assertEquals("cuda0,cuda1", p.parameters.get("--device"));
    }

    // -------------------------------------------------------------------------
    // Memory management
    // -------------------------------------------------------------------------

    @Test
    public void testEnableMlock() {
        ModelParameters p = new ModelParameters().enableMlock();
        assertTrue(p.parameters.containsKey("--mlock"));
        assertNull(p.parameters.get("--mlock"));
    }

    @Test
    public void testDisableMmap() {
        ModelParameters p = new ModelParameters().disableMmap();
        assertTrue(p.parameters.containsKey("--no-mmap"));
        assertNull(p.parameters.get("--no-mmap"));
    }

    @Test
    public void testSetNumaAllValues() {
        for (NumaStrategy ns : NumaStrategy.values()) {
            ModelParameters p = new ModelParameters().setNuma(ns);
            assertEquals(ns.name().toLowerCase(), p.parameters.get("--numa"));
        }
    }

    // -------------------------------------------------------------------------
    // Parallel / continuous batching
    // -------------------------------------------------------------------------

    @Test
    public void testSetParallel() {
        ModelParameters p = new ModelParameters().setParallel(4);
        assertEquals("4", p.parameters.get("--parallel"));
    }

    @Test
    public void testEnableContBatching() {
        ModelParameters p = new ModelParameters().enableContBatching();
        assertTrue(p.parameters.containsKey("--cont-batching"));
        assertNull(p.parameters.get("--cont-batching"));
    }

    @Test
    public void testDisableContBatching() {
        ModelParameters p = new ModelParameters().disableContBatching();
        assertTrue(p.parameters.containsKey("--no-cont-batching"));
        assertNull(p.parameters.get("--no-cont-batching"));
    }

    // -------------------------------------------------------------------------
    // Flag-only toggles
    // -------------------------------------------------------------------------

    @Test
    public void testDisableContextShift() {
        ModelParameters p = new ModelParameters().disableContextShift();
        assertTrue(p.parameters.containsKey("--no-context-shift"));
        assertNull(p.parameters.get("--no-context-shift"));
    }

    @Test
    public void testEnableFlashAttn() {
        ModelParameters p = new ModelParameters().enableFlashAttn();
        assertTrue(p.parameters.containsKey("--flash-attn"));
        assertNull(p.parameters.get("--flash-attn"));
    }

    @Test
    public void testDisablePerf() {
        ModelParameters p = new ModelParameters().disablePerf();
        assertTrue(p.parameters.containsKey("--no-perf"));
        assertNull(p.parameters.get("--no-perf"));
    }

    @Test
    public void testEnableEscape() {
        ModelParameters p = new ModelParameters().enableEscape();
        assertTrue(p.parameters.containsKey("--escape"));
        assertNull(p.parameters.get("--escape"));
    }

    @Test
    public void testDisableEscape() {
        ModelParameters p = new ModelParameters().disableEscape();
        assertTrue(p.parameters.containsKey("--no-escape"));
        assertNull(p.parameters.get("--no-escape"));
    }

    @Test
    public void testEnableSpecial() {
        ModelParameters p = new ModelParameters().enableSpecial();
        assertTrue(p.parameters.containsKey("--special"));
        assertNull(p.parameters.get("--special"));
    }

    @Test
    public void testSkipWarmup() {
        ModelParameters p = new ModelParameters().skipWarmup();
        assertTrue(p.parameters.containsKey("--no-warmup"));
        assertNull(p.parameters.get("--no-warmup"));
    }

    @Test
    public void testSetSpmInfill() {
        ModelParameters p = new ModelParameters().setSpmInfill();
        assertTrue(p.parameters.containsKey("--spm-infill"));
        assertNull(p.parameters.get("--spm-infill"));
    }

    @Test
    public void testIgnoreEos() {
        ModelParameters p = new ModelParameters().ignoreEos();
        assertTrue(p.parameters.containsKey("--ignore-eos"));
        assertNull(p.parameters.get("--ignore-eos"));
    }

    @Test
    public void testEnableCheckTensors() {
        ModelParameters p = new ModelParameters().enableCheckTensors();
        assertTrue(p.parameters.containsKey("--check-tensors"));
        assertNull(p.parameters.get("--check-tensors"));
    }

    @Test
    public void testEnableEmbedding() {
        ModelParameters p = new ModelParameters().enableEmbedding();
        assertTrue(p.parameters.containsKey("--embedding"));
        assertNull(p.parameters.get("--embedding"));
    }

    @Test
    public void testEnableReranking() {
        ModelParameters p = new ModelParameters().enableReranking();
        assertTrue(p.parameters.containsKey("--reranking"));
        assertNull(p.parameters.get("--reranking"));
    }

    @Test
    public void testSetVocabOnly() {
        ModelParameters p = new ModelParameters().setVocabOnly();
        assertTrue(p.parameters.containsKey("--vocab-only"));
        assertNull(p.parameters.get("--vocab-only"));
    }

    @Test
    public void testEnableJinja() {
        ModelParameters p = new ModelParameters().enableJinja();
        assertTrue(p.parameters.containsKey("--jinja"));
        assertNull(p.parameters.get("--jinja"));
    }

    @Test
    public void testSetLoraInitWithoutApply() {
        ModelParameters p = new ModelParameters().setLoraInitWithoutApply();
        assertTrue(p.parameters.containsKey("--lora-init-without-apply"));
        assertNull(p.parameters.get("--lora-init-without-apply"));
    }

    // -------------------------------------------------------------------------
    // Seed / Logit bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetSeed() {
        ModelParameters p = new ModelParameters().setSeed(42);
        assertEquals("42", p.parameters.get("--seed"));
    }

    @Test
    public void testSetSeedRandom() {
        ModelParameters p = new ModelParameters().setSeed(-1);
        assertEquals("-1", p.parameters.get("--seed"));
    }

    @Test
    public void testSetLogitBias() {
        ModelParameters p = new ModelParameters().setLogitBias("1+0.5");
        assertEquals("1+0.5", p.parameters.get("--logit-bias"));
    }

    // -------------------------------------------------------------------------
    // Grammar / JSON schema
    // -------------------------------------------------------------------------

    @Test
    public void testSetGrammar() {
        ModelParameters p = new ModelParameters().setGrammar("root ::= \"hello\"");
        assertEquals("root ::= \"hello\"", p.parameters.get("--grammar"));
    }

    @Test
    public void testSetGrammarFile() {
        ModelParameters p = new ModelParameters().setGrammarFile("grammar.gbnf");
        assertEquals("grammar.gbnf", p.parameters.get("--grammar-file"));
    }

    @Test
    public void testSetJsonSchema() {
        ModelParameters p = new ModelParameters().setJsonSchema("{\"type\":\"object\"}");
        assertEquals("{\"type\":\"object\"}", p.parameters.get("--json-schema"));
    }

    // -------------------------------------------------------------------------
    // Chat template
    // -------------------------------------------------------------------------

    @Test
    public void testSetChatTemplate() {
        ModelParameters p = new ModelParameters().setChatTemplate("{% for msg in messages %}{{ msg.content }}{% endfor %}");
        assertEquals("{% for msg in messages %}{{ msg.content }}{% endfor %}", p.parameters.get("--chat-template"));
    }

    @Test
    public void testSetChatTemplateKwargs() {
        Map<String, String> kwargs = new HashMap<>();
        kwargs.put("enable_thinking", "true");
        ModelParameters p = new ModelParameters().setChatTemplateKwargs(kwargs);
        String val = p.parameters.get("--chat-template-kwargs");
        assertNotNull(val);
        assertTrue(val.contains("\"enable_thinking\":true"));
    }

    @Test
    public void testSetChatTemplateKwargsMultiple() {
        Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        kwargs.put("key1", "\"val1\"");
        kwargs.put("key2", "42");
        ModelParameters p = new ModelParameters().setChatTemplateKwargs(kwargs);
        String val = p.parameters.get("--chat-template-kwargs");
        assertNotNull(val);
        assertTrue(val.startsWith("{"));
        assertTrue(val.endsWith("}"));
        assertTrue(val.contains("\"key1\":\"val1\""));
        assertTrue(val.contains("\"key2\":42"));
    }

    // -------------------------------------------------------------------------
    // Model loading
    // -------------------------------------------------------------------------

    @Test
    public void testSetModel() {
        ModelParameters p = new ModelParameters().setModel("/path/to/model.gguf");
        assertEquals("/path/to/model.gguf", p.parameters.get("--model"));
    }

    @Test
    public void testSetModelUrl() {
        ModelParameters p = new ModelParameters().setModelUrl("https://example.com/model.gguf");
        assertEquals("https://example.com/model.gguf", p.parameters.get("--model-url"));
    }

    @Test
    public void testSetHfRepo() {
        ModelParameters p = new ModelParameters().setHfRepo("meta-llama/Llama-2-7b");
        assertEquals("meta-llama/Llama-2-7b", p.parameters.get("--hf-repo"));
    }

    @Test
    public void testSetHfFile() {
        ModelParameters p = new ModelParameters().setHfFile("model-q4.gguf");
        assertEquals("model-q4.gguf", p.parameters.get("--hf-file"));
    }

    @Test
    public void testSetHfToken() {
        ModelParameters p = new ModelParameters().setHfToken("hf_abc123");
        assertEquals("hf_abc123", p.parameters.get("--hf-token"));
    }

    @Test
    public void testSetHfRepoV() {
        ModelParameters p = new ModelParameters().setHfRepoV("org/vocoder");
        assertEquals("org/vocoder", p.parameters.get("--hf-repo-v"));
    }

    @Test
    public void testSetHfFileV() {
        ModelParameters p = new ModelParameters().setHfFileV("vocoder.gguf");
        assertEquals("vocoder.gguf", p.parameters.get("--hf-file-v"));
    }

    // -------------------------------------------------------------------------
    // Slot / cache reuse
    // -------------------------------------------------------------------------

    @Test
    public void testSetCacheReuse() {
        ModelParameters p = new ModelParameters().setCacheReuse(128);
        assertEquals("128", p.parameters.get("--cache-reuse"));
    }

    @Test
    public void testSetSlotSavePath() {
        ModelParameters p = new ModelParameters().setSlotSavePath("/tmp/slots");
        assertEquals("/tmp/slots", p.parameters.get("--slot-save-path"));
    }

    @Test
    public void testSetSlotPromptSimilarity() {
        ModelParameters p = new ModelParameters().setSlotPromptSimilarity(0.8f);
        assertEquals("0.8", p.parameters.get("--slot-prompt-similarity"));
    }

    // -------------------------------------------------------------------------
    // Override KV
    // -------------------------------------------------------------------------

    @Test
    public void testSetOverrideKv() {
        ModelParameters p = new ModelParameters().setOverrideKv("tokenizer.ggml.pre=spm");
        assertEquals("tokenizer.ggml.pre=spm", p.parameters.get("--override-kv"));
    }

    // -------------------------------------------------------------------------
    // LoRA / Control vector (non-scaled variants)
    // -------------------------------------------------------------------------

    @Test
    public void testAddLoraAdapter() {
        ModelParameters p = new ModelParameters().addLoraAdapter("adapter.bin");
        assertEquals("adapter.bin", p.parameters.get("--lora"));
    }

    @Test
    public void testAddControlVector() {
        ModelParameters p = new ModelParameters().addControlVector("vec.bin");
        assertEquals("vec.bin", p.parameters.get("--control-vector"));
    }

    // -------------------------------------------------------------------------
    // Speculative decoding
    // -------------------------------------------------------------------------

    @Test
    public void testSetModelDraft() {
        ModelParameters p = new ModelParameters().setModelDraft("/path/to/draft.gguf");
        assertEquals("/path/to/draft.gguf", p.parameters.get("--model-draft"));
    }

    @Test
    public void testSetCtxSizeDraft() {
        ModelParameters p = new ModelParameters().setCtxSizeDraft(512);
        assertEquals("512", p.parameters.get("--ctx-size-draft"));
    }

    @Test
    public void testSetDeviceDraft() {
        ModelParameters p = new ModelParameters().setDeviceDraft("cuda0");
        assertEquals("cuda0", p.parameters.get("--device-draft"));
    }

    @Test
    public void testSetGpuLayersDraft() {
        ModelParameters p = new ModelParameters().setGpuLayersDraft(16);
        assertEquals("16", p.parameters.get("--gpu-layers-draft"));
    }

    @Test
    public void testSetDraftMax() {
        ModelParameters p = new ModelParameters().setDraftMax(8);
        assertEquals("8", p.parameters.get("--draft-max"));
    }

    @Test
    public void testSetDraftMin() {
        ModelParameters p = new ModelParameters().setDraftMin(2);
        assertEquals("2", p.parameters.get("--draft-min"));
    }

    @Test
    public void testSetDraftPMin() {
        ModelParameters p = new ModelParameters().setDraftPMin(0.5f);
        assertEquals("0.5", p.parameters.get("--draft-p-min"));
    }

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    @Test
    public void testDisableLog() {
        ModelParameters p = new ModelParameters().disableLog();
        assertTrue(p.parameters.containsKey("--log-disable"));
        assertNull(p.parameters.get("--log-disable"));
    }

    @Test
    public void testSetLogFile() {
        ModelParameters p = new ModelParameters().setLogFile("/tmp/llama.log");
        assertEquals("/tmp/llama.log", p.parameters.get("--log-file"));
    }

    @Test
    public void testSetVerbose() {
        ModelParameters p = new ModelParameters().setVerbose();
        assertTrue(p.parameters.containsKey("--verbose"));
        assertNull(p.parameters.get("--verbose"));
    }

    @Test
    public void testSetLogVerbosity() {
        ModelParameters p = new ModelParameters().setLogVerbosity(3);
        assertEquals("3", p.parameters.get("--log-verbosity"));
    }

    @Test
    public void testEnableLogPrefix() {
        ModelParameters p = new ModelParameters().enableLogPrefix();
        assertTrue(p.parameters.containsKey("--log-prefix"));
        assertNull(p.parameters.get("--log-prefix"));
    }

    @Test
    public void testEnableLogTimestamps() {
        ModelParameters p = new ModelParameters().enableLogTimestamps();
        assertTrue(p.parameters.containsKey("--log-timestamps"));
        assertNull(p.parameters.get("--log-timestamps"));
    }

    // -------------------------------------------------------------------------
    // setFit variations
    // -------------------------------------------------------------------------

    @Test
    public void testSetFitTrue() {
        ModelParameters p = new ModelParameters().setFit(true);
        assertEquals(ModelParameters.FIT_ON, p.parameters.get("--fit"));
    }

    @Test
    public void testSetFitFalse() {
        ModelParameters p = new ModelParameters().setFit(false);
        assertEquals(ModelParameters.FIT_OFF, p.parameters.get("--fit"));
    }

    // -------------------------------------------------------------------------
    // RoPE scaling type — all values
    // -------------------------------------------------------------------------

    @Test
    public void testSetRopeScalingAllValues() {
        for (RopeScalingType type : RopeScalingType.values()) {
            ModelParameters p = new ModelParameters().setRopeScaling(type);
            assertEquals(type.getArgValue(), p.parameters.get("--rope-scaling"));
        }
    }

    // -------------------------------------------------------------------------
    // MiroStat — all values
    // -------------------------------------------------------------------------

    @Test
    public void testSetMirostatAllValues() {
        for (MiroStat m : MiroStat.values()) {
            ModelParameters p = new ModelParameters().setMirostat(m);
            assertEquals(String.valueOf(m.ordinal()), p.parameters.get("--mirostat"));
        }
    }

    // -------------------------------------------------------------------------
    // Chaining — extended chaining across categories
    // -------------------------------------------------------------------------

    @Test
    public void testExtendedChainingReturnsSameInstance() {
        ModelParameters p = new ModelParameters();
        assertSame(p, p.setCtxSize(2048));
        assertSame(p, p.setBatchSize(512));
        assertSame(p, p.setTemp(0.7f));
        assertSame(p, p.setTopK(50));
        assertSame(p, p.setDryMultiplier(0.5f));
        assertSame(p, p.setXtcProbability(0.3f));
        assertSame(p, p.setRopeScale(2.0f));
        assertSame(p, p.setGpuLayers(32));
        assertSame(p, p.enableFlashAttn());
        assertSame(p, p.disableContextShift());
        assertSame(p, p.setModelDraft("/draft.gguf"));
        assertSame(p, p.disableLog());
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
                .enableFlashAttn();
        String[] arr = p.toArray();
        // argv[0]="" + --fit + on + --model + model.gguf + --ctx-size + 2048 + --embedding + --flash-attn = 9
        assertEquals(9, arr.length);
        assertEquals("", arr[0]);
    }

    // -------------------------------------------------------------------------
    // isDefault — extended
    // -------------------------------------------------------------------------

    @Test
    public void testIsDefaultForCtxSize() {
        ModelParameters p = new ModelParameters();
        assertTrue(p.isDefault("ctx-size"));
        p.setCtxSize(2048);
        assertFalse(p.isDefault("ctx-size"));
    }

    @Test
    public void testIsDefaultForFlagOnly() {
        ModelParameters p = new ModelParameters();
        assertTrue(p.isDefault("flash-attn"));
        p.enableFlashAttn();
        assertFalse(p.isDefault("flash-attn"));
    }
}
