package de.kherud.llama;

import de.kherud.llama.args.*;

/***
 * Parameters used for initializing a {@link LlamaModel}.
 */
@SuppressWarnings("unused")
public final class ModelParameters extends CliParameters {

    private static final String ARG_FIT = "--fit";
    static final String ARG_POOLING = "--pooling";
    public static final String FIT_ON = "on";
    public static final String FIT_OFF = "off";
    /** Mirrors the llama.cpp default: {@code fit_params = true}. */
    public static final String DEFAULT_FIT_VALUE = FIT_ON;

    public ModelParameters() {
        parameters.put(ARG_FIT, DEFAULT_FIT_VALUE);
    }

    static String fitValue(boolean fit) {
        return fit ? FIT_ON : FIT_OFF;
    }

    /**
     * Whether to adjust unset arguments to fit in device memory (default: {@value #DEFAULT_FIT_VALUE}).
     *
     * @param fit whether to adjust unset arguments to fit in device memory
     * @return this builder
     */
    public ModelParameters setFit(boolean fit) {
        parameters.put(ARG_FIT, fitValue(fit));
        return this;
    }

    /**
     * Set the number of threads to use during generation (default: -1).
     *
     * @param nThreads the number of threads to use during generation
     * @return this builder
     */
    public ModelParameters setThreads(int nThreads) {
        parameters.put("--threads", String.valueOf(nThreads));
        return this;
    }

    /**
     * Set the number of threads to use during batch and prompt processing (default: same as --threads).
     *
     * @param nThreads the number of threads for batch and prompt processing
     * @return this builder
     */
    public ModelParameters setThreadsBatch(int nThreads) {
        parameters.put("--threads-batch", String.valueOf(nThreads));
        return this;
    }

    /**
     * Set the CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: "").
     *
     * @param mask the CPU affinity mask as an arbitrarily long hex string
     * @return this builder
     */
    public ModelParameters setCpuMask(String mask) {
        parameters.put("--cpu-mask", mask);
        return this;
    }

    /**
     * Set the range of CPUs for affinity. Complements --cpu-mask.
     *
     * @param range the range of CPUs for affinity
     * @return this builder
     */
    public ModelParameters setCpuRange(String range) {
        parameters.put("--cpu-range", range);
        return this;
    }

    /**
     * Use strict CPU placement (default: 0).
     *
     * @param strictCpu 1 to enable strict CPU placement, 0 to disable
     * @return this builder
     */
    public ModelParameters setCpuStrict(int strictCpu) {
        parameters.put("--cpu-strict", String.valueOf(strictCpu));
        return this;
    }

    /**
     * Set process/thread priority: 0-normal, 1-medium, 2-high, 3-realtime (default: 0).
     *
     * @param priority process/thread priority (0=normal, 1=medium, 2=high, 3=realtime)
     * @return this builder
     */
    public ModelParameters setPriority(int priority) {
        if (priority < 0 || priority > 3) {
            throw new IllegalArgumentException("Invalid value for priority");
        }
        parameters.put("--prio", String.valueOf(priority));
        return this;
    }

    /**
     * Set the polling level to wait for work (0 - no polling, default: 0).
     *
     * @param poll the polling level (0 = no polling)
     * @return this builder
     */
    public ModelParameters setPoll(int poll) {
        parameters.put("--poll", String.valueOf(poll));
        return this;
    }

    /**
     * Set the CPU affinity mask for batch processing: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask).
     *
     * @param mask the CPU affinity mask for batch processing
     * @return this builder
     */
    public ModelParameters setCpuMaskBatch(String mask) {
        parameters.put("--cpu-mask-batch", mask);
        return this;
    }

    /**
     * Set the ranges of CPUs for batch affinity. Complements --cpu-mask-batch.
     *
     * @param range the ranges of CPUs for batch affinity
     * @return this builder
     */
    public ModelParameters setCpuRangeBatch(String range) {
        parameters.put("--cpu-range-batch", range);
        return this;
    }

    /**
     * Use strict CPU placement for batch processing (default: same as --cpu-strict).
     *
     * @param strictCpuBatch 1 to enable strict CPU placement for batch processing, 0 to disable
     * @return this builder
     */
    public ModelParameters setCpuStrictBatch(int strictCpuBatch) {
        parameters.put("--cpu-strict-batch", String.valueOf(strictCpuBatch));
        return this;
    }

    /**
     * Set process/thread priority for batch processing: 0-normal, 1-medium, 2-high, 3-realtime (default: 0).
     *
     * @param priorityBatch batch process/thread priority (0=normal, 1=medium, 2=high, 3=realtime)
     * @return this builder
     */
    public ModelParameters setPriorityBatch(int priorityBatch) {
        if (priorityBatch < 0 || priorityBatch > 3) {
            throw new IllegalArgumentException("Invalid value for priority batch");
        }
        parameters.put("--prio-batch", String.valueOf(priorityBatch));
        return this;
    }

    /**
     * Set the polling level for batch processing (default: same as --poll).
     *
     * @param pollBatch the polling level for batch processing
     * @return this builder
     */
    public ModelParameters setPollBatch(int pollBatch) {
        parameters.put("--poll-batch", String.valueOf(pollBatch));
        return this;
    }

    /**
     * Set the size of the prompt context (default: 0, 0 = loaded from model).
     *
     * @param ctxSize the prompt context size in tokens (0 = loaded from model)
     * @return this builder
     */
    public ModelParameters setCtxSize(int ctxSize) {
        parameters.put("--ctx-size", String.valueOf(ctxSize));
        return this;
    }

    /**
     * Set the number of tokens to predict (default: -1 = infinity, -2 = until context filled).
     *
     * @param nPredict the maximum number of tokens to predict (-1 = infinity, -2 = until context filled)
     * @return this builder
     */
    public ModelParameters setPredict(int nPredict) {
        parameters.put("--predict", String.valueOf(nPredict));
        return this;
    }

    /**
     * Set the logical maximum batch size (default: 0).
     *
     * @param batchSize the logical maximum batch size
     * @return this builder
     */
    public ModelParameters setBatchSize(int batchSize) {
        parameters.put("--batch-size", String.valueOf(batchSize));
        return this;
    }

    /**
     * Set the physical maximum batch size (default: 0).
     *
     * @param ubatchSize the physical maximum batch size
     * @return this builder
     */
    public ModelParameters setUbatchSize(int ubatchSize) {
        parameters.put("--ubatch-size", String.valueOf(ubatchSize));
        return this;
    }

    /**
     * Set the number of tokens to keep from the initial prompt (default: -1 = all).
     *
     * @param keep the number of tokens to keep from the initial prompt (-1 = all)
     * @return this builder
     */
    public ModelParameters setKeep(int keep) {
        parameters.put("--keep", String.valueOf(keep));
        return this;
    }

    /**
     * Disable context shift on infinite text generation (default: enabled).
     *
     * @return this builder
     */
    public ModelParameters disableContextShift() {
        return setFlag(ModelFlag.NO_CONTEXT_SHIFT);
    }

    /**
     * Enable Flash Attention (default: disabled).
     *
     * @return this builder
     */
    public ModelParameters enableFlashAttn() {
        return setFlag(ModelFlag.FLASH_ATTN);
    }

    /**
     * Disable internal libllama performance timings (default: false).
     *
     * @return this builder
     */
    public ModelParameters disablePerf() {
        return setFlag(ModelFlag.NO_PERF);
    }

    /**
     * Process escape sequences (default: true).
     *
     * @return this builder
     */
    public ModelParameters enableEscape() {
        return setFlag(ModelFlag.ESCAPE);
    }

    /**
     * Do not process escape sequences (default: false).
     *
     * @return this builder
     */
    public ModelParameters disableEscape() {
        return setFlag(ModelFlag.NO_ESCAPE);
    }

    /**
     * Enable special tokens output (default: true).
     *
     * @return this builder
     */
    public ModelParameters enableSpecial() {
        return setFlag(ModelFlag.SPECIAL);
    }

    /**
     * Skip warming up the model with an empty run (default: false).
     *
     * @return this builder
     */
    public ModelParameters skipWarmup() {
        return setFlag(ModelFlag.NO_WARMUP);
    }

    /**
     * Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.
     * (default: disabled)
     *
     * @return this builder
     */
    public ModelParameters setSpmInfill() {
        return setFlag(ModelFlag.SPM_INFILL);
    }

    /**
     * Set samplers that will be used for generation in the order, separated by ';' (default: all).
     *
     * @param samplers the samplers to use, separated by semicolons
     * @return this builder
     */
    public ModelParameters setSamplers(Sampler... samplers) {
        if (samplers.length > 0) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < samplers.length; i++) {
                builder.append(samplers[i].getArgValue());
                if (i < samplers.length - 1) {
                    builder.append(";");
                }
            }
            parameters.put("--samplers", builder.toString());
        }
        return this;
    }

    /**
     * Set RNG seed (default: -1, use random seed).
     *
     * @param seed the RNG seed (-1 = random)
     * @return this builder
     */
    public ModelParameters setSeed(long seed) {
        parameters.put("--seed", String.valueOf(seed));
        return this;
    }

    /**
     * Ignore end of stream token and continue generating (implies --logit-bias EOS-inf).
     *
     * @return this builder
     */
    public ModelParameters ignoreEos() {
        return setFlag(ModelFlag.IGNORE_EOS);
    }

    /**
     * Set temperature for sampling (default: 0.8).
     *
     * @param temp the sampling temperature
     * @return this builder
     */
    public ModelParameters setTemp(float temp) {
        parameters.put("--temp", String.valueOf(temp));
        return this;
    }

    /**
     * Set top-k sampling (default: 40, 0 = disabled).
     *
     * @param topK the top-k value (0 = disabled)
     * @return this builder
     */
    public ModelParameters setTopK(int topK) {
        parameters.put("--top-k", String.valueOf(topK));
        return this;
    }

    /**
     * Set top-p sampling (default: 0.95, 1.0 = disabled).
     *
     * @param topP the top-p value (1.0 = disabled)
     * @return this builder
     */
    public ModelParameters setTopP(float topP) {
        parameters.put("--top-p", String.valueOf(topP));
        return this;
    }

    /**
     * Set min-p sampling (default: 0.05, 0.0 = disabled).
     *
     * @param minP the min-p value (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setMinP(float minP) {
        parameters.put("--min-p", String.valueOf(minP));
        return this;
    }

    /**
     * Set xtc probability (default: 0.0, 0.0 = disabled).
     *
     * @param xtcProbability the XTC probability (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setXtcProbability(float xtcProbability) {
        parameters.put("--xtc-probability", String.valueOf(xtcProbability));
        return this;
    }

    /**
     * Set xtc threshold (default: 0.1, 1.0 = disabled).
     *
     * @param xtcThreshold the XTC threshold (1.0 = disabled)
     * @return this builder
     */
    public ModelParameters setXtcThreshold(float xtcThreshold) {
        parameters.put("--xtc-threshold", String.valueOf(xtcThreshold));
        return this;
    }

    /**
     * Set locally typical sampling parameter p (default: 1.0, 1.0 = disabled).
     *
     * @param typP the locally typical sampling parameter p (1.0 = disabled)
     * @return this builder
     */
    public ModelParameters setTypical(float typP) {
        parameters.put("--typical", String.valueOf(typP));
        return this;
    }

    /**
     * Set last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size).
     *
     * @param repeatLastN the number of last tokens to consider for penalties (0 = disabled, -1 = ctx_size)
     * @return this builder
     */
    public ModelParameters setRepeatLastN(int repeatLastN) {
        if (repeatLastN < -1) {
            throw new RuntimeException("Invalid repeat-last-n value");
        }
        parameters.put("--repeat-last-n", String.valueOf(repeatLastN));
        return this;
    }

    /**
     * Set penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled).
     *
     * @param repeatPenalty the repeat penalty (1.0 = disabled)
     * @return this builder
     */
    public ModelParameters setRepeatPenalty(float repeatPenalty) {
        parameters.put("--repeat-penalty", String.valueOf(repeatPenalty));
        return this;
    }

    /**
     * Set repeat alpha presence penalty (default: 0.0, 0.0 = disabled).
     *
     * @param presencePenalty the alpha presence penalty (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setPresencePenalty(float presencePenalty) {
        parameters.put("--presence-penalty", String.valueOf(presencePenalty));
        return this;
    }

    /**
     * Set repeat alpha frequency penalty (default: 0.0, 0.0 = disabled).
     *
     * @param frequencyPenalty the alpha frequency penalty (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setFrequencyPenalty(float frequencyPenalty) {
        parameters.put("--frequency-penalty", String.valueOf(frequencyPenalty));
        return this;
    }

    /**
     * Set DRY sampling multiplier (default: 0.0, 0.0 = disabled).
     *
     * @param dryMultiplier the DRY sampling multiplier (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setDryMultiplier(float dryMultiplier) {
        parameters.put("--dry-multiplier", String.valueOf(dryMultiplier));
        return this;
    }

    /**
     * Set DRY sampling base value (default: 1.75).
     *
     * @param dryBase the DRY sampling base value
     * @return this builder
     */
    public ModelParameters setDryBase(float dryBase) {
        parameters.put("--dry-base", String.valueOf(dryBase));
        return this;
    }

    /**
     * Set allowed length for DRY sampling (default: 2).
     *
     * @param dryAllowedLength the allowed length for DRY sampling
     * @return this builder
     */
    public ModelParameters setDryAllowedLength(int dryAllowedLength) {
        parameters.put("--dry-allowed-length", String.valueOf(dryAllowedLength));
        return this;
    }

    /**
     * Set DRY penalty for the last n tokens (default: -1, 0 = disable, -1 = context size).
     *
     * @param dryPenaltyLastN the DRY penalty window (-1 = context size, 0 = disabled)
     * @return this builder
     */
    public ModelParameters setDryPenaltyLastN(int dryPenaltyLastN) {
        if (dryPenaltyLastN < -1) {
            throw new RuntimeException("Invalid dry-penalty-last-n value");
        }
        parameters.put("--dry-penalty-last-n", String.valueOf(dryPenaltyLastN));
        return this;
    }

    /**
     * Add sequence breaker for DRY sampling, clearing out default breakers (default: none).
     *
     * @param drySequenceBreaker the sequence breaker string for DRY sampling
     * @return this builder
     */
    public ModelParameters setDrySequenceBreaker(String drySequenceBreaker) {
        parameters.put("--dry-sequence-breaker", drySequenceBreaker);
        return this;
    }

    /**
     * Set dynamic temperature range (default: 0.0, 0.0 = disabled).
     *
     * @param dynatempRange the dynamic temperature range (0.0 = disabled)
     * @return this builder
     */
    public ModelParameters setDynatempRange(float dynatempRange) {
        parameters.put("--dynatemp-range", String.valueOf(dynatempRange));
        return this;
    }

    /**
     * Set dynamic temperature exponent (default: 1.0).
     *
     * @param dynatempExponent the dynamic temperature exponent
     * @return this builder
     */
    public ModelParameters setDynatempExponent(float dynatempExponent) {
        parameters.put("--dynatemp-exp", String.valueOf(dynatempExponent));
        return this;
    }

    /**
     * Use Mirostat sampling (default: PLACEHOLDER, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
     *
     * @param mirostat the Mirostat sampling mode
     * @return this builder
     */
    public ModelParameters setMirostat(MiroStat mirostat) {
        parameters.put("--mirostat", mirostat.getArgValue());
        return this;
    }

    /**
     * Set Mirostat learning rate, parameter eta (default: 0.1).
     *
     * @param mirostatLR the Mirostat learning rate (eta)
     * @return this builder
     */
    public ModelParameters setMirostatLR(float mirostatLR) {
        parameters.put("--mirostat-lr", String.valueOf(mirostatLR));
        return this;
    }

    /**
     * Set Mirostat target entropy, parameter tau (default: 5.0).
     *
     * @param mirostatEnt the Mirostat target entropy (tau)
     * @return this builder
     */
    public ModelParameters setMirostatEnt(float mirostatEnt) {
        parameters.put("--mirostat-ent", String.valueOf(mirostatEnt));
        return this;
    }

    /**
     * Modify the likelihood of token appearing in the completion.
     *
     * @param tokenIdAndBias token id and bias value as a formatted string
     * @return this builder
     */
    public ModelParameters setLogitBias(String tokenIdAndBias) {
        parameters.put("--logit-bias", tokenIdAndBias);
        return this;
    }

    /**
     * Set BNF-like grammar to constrain generations (default: empty).
     *
     * @param grammar the BNF-like grammar string
     * @return this builder
     */
    public ModelParameters setGrammar(String grammar) {
        parameters.put("--grammar", grammar);
        return this;
    }

    /**
     * Specify the file to read grammar from.
     *
     * @param fileName the path to a file containing the grammar
     * @return this builder
     */
    public ModelParameters setGrammarFile(String fileName) {
        parameters.put("--grammar-file", fileName);
        return this;
    }

    /**
     * Specify the JSON schema to constrain generations (default: empty).
     *
     * @param schema the JSON schema string
     * @return this builder
     */
    public ModelParameters setJsonSchema(String schema) {
        parameters.put("--json-schema", schema);
        return this;
    }

    /**
     * Set pooling type for embeddings (default: model default if unspecified).
     * When {@link de.kherud.llama.args.PoolingType#UNSPECIFIED} is passed, the parameter is not forwarded
     * to the native layer, leaving the model to use its built-in default pooling strategy.
     *
     * @param type the pooling type for embeddings
     * @return this builder
     */
    public ModelParameters setPoolingType(PoolingType type) {
        if (type != PoolingType.UNSPECIFIED) {
            // Don't set if unspecified, as it will use the model's default pooling type
            parameters.put(ARG_POOLING, type.getArgValue());
        }
        return this;
    }

    /**
     * Set RoPE frequency scaling method (default: linear unless specified by the model).
     *
     * @param type the RoPE frequency scaling method
     * @return this builder
     */
    public ModelParameters setRopeScaling(RopeScalingType type) {
        parameters.put("--rope-scaling", type.getArgValue());
        return this;
    }

    /**
     * Set RoPE context scaling factor, expands context by a factor of N.
     *
     * @param ropeScale the RoPE context scaling factor
     * @return this builder
     */
    public ModelParameters setRopeScale(float ropeScale) {
        parameters.put("--rope-scale", String.valueOf(ropeScale));
        return this;
    }

    /**
     * Set RoPE base frequency, used by NTK-aware scaling (default: loaded from model).
     *
     * @param ropeFreqBase the RoPE base frequency
     * @return this builder
     */
    public ModelParameters setRopeFreqBase(float ropeFreqBase) {
        parameters.put("--rope-freq-base", String.valueOf(ropeFreqBase));
        return this;
    }

    /**
     * Set RoPE frequency scaling factor, expands context by a factor of 1/N.
     *
     * @param ropeFreqScale the RoPE frequency scaling factor
     * @return this builder
     */
    public ModelParameters setRopeFreqScale(float ropeFreqScale) {
        parameters.put("--rope-freq-scale", String.valueOf(ropeFreqScale));
        return this;
    }

    /**
     * Set YaRN: original context size of model (default: model training context size).
     *
     * @param yarnOrigCtx the original context size of the model
     * @return this builder
     */
    public ModelParameters setYarnOrigCtx(int yarnOrigCtx) {
        parameters.put("--yarn-orig-ctx", String.valueOf(yarnOrigCtx));
        return this;
    }

    /**
     * Set YaRN: extrapolation mix factor (default: 0.0 = full interpolation).
     *
     * @param yarnExtFactor the YaRN extrapolation mix factor (0.0 = full interpolation)
     * @return this builder
     */
    public ModelParameters setYarnExtFactor(float yarnExtFactor) {
        parameters.put("--yarn-ext-factor", String.valueOf(yarnExtFactor));
        return this;
    }

    /**
     * Set YaRN: scale sqrt(t) or attention magnitude (default: 1.0).
     *
     * @param yarnAttnFactor the YaRN attention scale factor
     * @return this builder
     */
    public ModelParameters setYarnAttnFactor(float yarnAttnFactor) {
        parameters.put("--yarn-attn-factor", String.valueOf(yarnAttnFactor));
        return this;
    }

    /**
     * Set YaRN: high correction dim or alpha (default: 1.0).
     *
     * @param yarnBetaSlow the YaRN high correction dimension (alpha)
     * @return this builder
     */
    public ModelParameters setYarnBetaSlow(float yarnBetaSlow) {
        parameters.put("--yarn-beta-slow", String.valueOf(yarnBetaSlow));
        return this;
    }

    /**
     * Set YaRN: low correction dim or beta (default: 32.0).
     *
     * @param yarnBetaFast the YaRN low correction dimension (beta)
     * @return this builder
     */
    public ModelParameters setYarnBetaFast(float yarnBetaFast) {
        parameters.put("--yarn-beta-fast", String.valueOf(yarnBetaFast));
        return this;
    }

    /**
     * Set group-attention factor (default: 1).
     *
     * @param grpAttnN the group-attention factor
     * @return this builder
     */
    public ModelParameters setGrpAttnN(int grpAttnN) {
        parameters.put("--grp-attn-n", String.valueOf(grpAttnN));
        return this;
    }

    /**
     * Set group-attention width (default: 512).
     *
     * @param grpAttnW the group-attention width
     * @return this builder
     */
    public ModelParameters setGrpAttnW(int grpAttnW) {
        parameters.put("--grp-attn-w", String.valueOf(grpAttnW));
        return this;
    }

    /**
     * Enable verbose printing of the KV cache.
     *
     * @return this builder
     */
    public ModelParameters enableDumpKvCache() {
        return setFlag(ModelFlag.DUMP_KV_CACHE);
    }

    /**
     * Disable KV offload.
     *
     * @return this builder
     */
    public ModelParameters disableKvOffload() {
        return setFlag(ModelFlag.NO_KV_OFFLOAD);
    }

    /**
     * Set KV cache data type for K (allowed values: F16).
     *
     * @param type the KV cache data type for K
     * @return this builder
     */
    public ModelParameters setCacheTypeK(CacheType type) {
        parameters.put("--cache-type-k", type.getArgValue());
        return this;
    }

    /**
     * Set KV cache data type for V (allowed values: F16).
     *
     * @param type the KV cache data type for V
     * @return this builder
     */
    public ModelParameters setCacheTypeV(CacheType type) {
        parameters.put("--cache-type-v", type.getArgValue());
        return this;
    }

    /**
     * Set KV cache defragmentation threshold (default: 0.1, &lt; 0 - disabled).
     *
     * @param defragThold the KV cache defragmentation threshold (negative = disabled)
     * @return this builder
     */
    public ModelParameters setDefragThold(float defragThold) {
        parameters.put("--defrag-thold", String.valueOf(defragThold));
        return this;
    }

    /**
     * Set the number of parallel sequences to decode (default: 1).
     *
     * @param nParallel the number of parallel sequences to decode
     * @return this builder
     */
    public ModelParameters setParallel(int nParallel) {
        parameters.put("--parallel", String.valueOf(nParallel));
        return this;
    }

    /**
     * Enable continuous batching (a.k.a dynamic batching) (default: disabled).
     *
     * @return this builder
     */
    public ModelParameters enableContBatching() {
        return setFlag(ModelFlag.CONT_BATCHING);
    }

    /**
     * Disable continuous batching.
     *
     * @return this builder
     */
    public ModelParameters disableContBatching() {
        return setFlag(ModelFlag.NO_CONT_BATCHING);
    }

    /**
     * Force system to keep model in RAM rather than swapping or compressing.
     *
     * @return this builder
     */
    public ModelParameters enableMlock() {
        return setFlag(ModelFlag.MLOCK);
    }

    /**
     * Do not memory-map model (slower load but may reduce pageouts if not using mlock).
     *
     * @return this builder
     */
    public ModelParameters disableMmap() {
        return setFlag(ModelFlag.NO_MMAP);
    }

    /**
     * Set NUMA optimization type for system.
     *
     * @param numaStrategy the NUMA optimization strategy
     * @return this builder
     */
    public ModelParameters setNuma(NumaStrategy numaStrategy) {
        parameters.put("--numa", numaStrategy.getArgValue());
        return this;
    }

    /**
     * Set comma-separated list of devices to use for offloading &lt;dev1,dev2,..&gt; (none = don't offload).
     *
     * @param devices comma-separated list of device names to use for offloading
     * @return this builder
     */
    public ModelParameters setDevices(String devices) {
        parameters.put("--device", devices);
        return this;
    }

    /**
     * Set the number of layers to store in VRAM.
     *
     * @param gpuLayers the number of model layers to store in VRAM
     * @return this builder
     */
    public ModelParameters setGpuLayers(int gpuLayers) {
        parameters.put("--gpu-layers", String.valueOf(gpuLayers));
        return this;
    }

    /**
     * Set how to split the model across multiple GPUs (none, layer, row).
     *
     * @param splitMode how to split the model across multiple GPUs
     * @return this builder
     */
    public ModelParameters setSplitMode(GpuSplitMode splitMode) {
        parameters.put("--split-mode", splitMode.getArgValue());
        return this;
    }

    /**
     * Set fraction of the model to offload to each GPU, comma-separated list of proportions N0,N1,N2,....
     *
     * @param tensorSplit comma-separated proportions for offloading to each GPU
     * @return this builder
     */
    public ModelParameters setTensorSplit(String tensorSplit) {
        parameters.put("--tensor-split", tensorSplit);
        return this;
    }

    /**
     * Set the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row).
     *
     * @param mainGpu the index of the primary GPU
     * @return this builder
     */
    public ModelParameters setMainGpu(int mainGpu) {
        parameters.put("--main-gpu", String.valueOf(mainGpu));
        return this;
    }

    /**
     * Enable checking model tensor data for invalid values.
     *
     * @return this builder
     */
    public ModelParameters enableCheckTensors() {
        return setFlag(ModelFlag.CHECK_TENSORS);
    }

    /**
     * Override model metadata by key. This option can be specified multiple times.
     *
     * @param keyValue the key-value metadata override string
     * @return this builder
     */
    public ModelParameters setOverrideKv(String keyValue) {
        parameters.put("--override-kv", keyValue);
        return this;
    }

    /**
     * Add a LoRA adapter (can be repeated to use multiple adapters).
     *
     * @param fname the file path of the LoRA adapter
     * @return this builder
     */
    public ModelParameters addLoraAdapter(String fname) {
        parameters.put("--lora", fname);
        return this;
    }

    /**
     * Add a LoRA adapter with user-defined scaling (can be repeated to use multiple adapters).
     *
     * @param fname the file path of the LoRA adapter
     * @param scale the user-defined scaling factor
     * @return this builder
     */
    public ModelParameters addLoraScaledAdapter(String fname, float scale) {
        parameters.put("--lora-scaled", fname + "," + scale);
        return this;
    }

    /**
     * Add a control vector (this argument can be repeated to add multiple control vectors).
     *
     * @param fname the file path of the control vector
     * @return this builder
     */
    public ModelParameters addControlVector(String fname) {
        parameters.put("--control-vector", fname);
        return this;
    }

    /**
     * Add a control vector with user-defined scaling (can be repeated to add multiple scaled control vectors).
     *
     * @param fname the file path of the control vector
     * @param scale the user-defined scaling factor
     * @return this builder
     */
    public ModelParameters addControlVectorScaled(String fname, float scale) {
        parameters.put("--control-vector-scaled", fname + "," + scale);
        return this;
    }

    /**
     * Set the layer range to apply the control vector(s) to (start and end inclusive).
     *
     * @param start the first layer to apply the control vector to (inclusive)
     * @param end the last layer to apply the control vector to (inclusive)
     * @return this builder
     */
    public ModelParameters setControlVectorLayerRange(int start, int end) {
        parameters.put("--control-vector-layer-range", start + "," + end);
        return this;
    }

    /**
     * Set the model path from which to load the base model.
     *
     * @param model the file path of the base model
     * @return this builder
     */
    public ModelParameters setModel(String model) {
        parameters.put("--model", model);
        return this;
    }

    /**
     * Set the model download URL (default: unused).
     *
     * @param modelUrl the URL from which to download the model
     * @return this builder
     */
    public ModelParameters setModelUrl(String modelUrl) {
        parameters.put("--model-url", modelUrl);
        return this;
    }

    /**
     * Set the Hugging Face model repository (default: unused).
     *
     * @param hfRepo the Hugging Face repository identifier
     * @return this builder
     */
    public ModelParameters setHfRepo(String hfRepo) {
        parameters.put("--hf-repo", hfRepo);
        return this;
    }

    /**
     * Set the Hugging Face model file (default: unused).
     *
     * @param hfFile the model file within the Hugging Face repository
     * @return this builder
     */
    public ModelParameters setHfFile(String hfFile) {
        parameters.put("--hf-file", hfFile);
        return this;
    }

    /**
     * Set the Hugging Face model repository for the vocoder model (default: unused).
     *
     * @param hfRepoV the Hugging Face repository for the vocoder model
     * @return this builder
     */
    public ModelParameters setHfRepoV(String hfRepoV) {
        parameters.put("--hf-repo-v", hfRepoV);
        return this;
    }

    /**
     * Set the Hugging Face model file for the vocoder model (default: unused).
     *
     * @param hfFileV the vocoder model file within the Hugging Face repository
     * @return this builder
     */
    public ModelParameters setHfFileV(String hfFileV) {
        parameters.put("--hf-file-v", hfFileV);
        return this;
    }

    /**
     * Set the Hugging Face access token (default: value from HF_TOKEN environment variable).
     *
     * @param hfToken the Hugging Face API access token
     * @return this builder
     */
    public ModelParameters setHfToken(String hfToken) {
        parameters.put("--hf-token", hfToken);
        return this;
    }

    /**
     * Enable embedding use case; use only with dedicated embedding models.
     *
     * @return this builder
     */
    public ModelParameters enableEmbedding() {
        return setFlag(ModelFlag.EMBEDDING);
    }

    /**
     * Enable reranking endpoint on server.
     *
     * @return this builder
     */
    public ModelParameters enableReranking() {
        return setFlag(ModelFlag.RERANKING);
    }

    /**
     * Set minimum chunk size to attempt reusing from the cache via KV shifting.
     *
     * @param cacheReuse the minimum chunk size to attempt reusing from KV cache
     * @return this builder
     */
    public ModelParameters setCacheReuse(int cacheReuse) {
        parameters.put("--cache-reuse", String.valueOf(cacheReuse));
        return this;
    }

    /**
     * Set the path to save the slot kv cache.
     *
     * @param slotSavePath the directory path for saving slot KV cache
     * @return this builder
     */
    public ModelParameters setSlotSavePath(String slotSavePath) {
        parameters.put("--slot-save-path", slotSavePath);
        return this;
    }

    /**
     * Set custom jinja chat template.
     *
     * @param chatTemplate the custom Jinja chat template string
     * @return this builder
     */
    public ModelParameters setChatTemplate(String chatTemplate) {
        parameters.put("--chat-template", chatTemplate);
        return this;
    }

    /**
     * Set custom Jinja template variables that are injected into the chat template context.
     * Requires {@link #enableJinja()} to be set. Values are JSON-serialized strings that
     * get parsed and merged into the template's extra_context.
     * <p>
     * Example:
     * <pre>{@code
     * Map<String, String> kwargs = new HashMap<>();
     * kwargs.put("enable_thinking", "true");
     * kwargs.put("custom_var", "\"my_value\"");
     * params.setChatTemplateKwargs(kwargs);
     * }</pre>
     *
     * @param kwargs map of template variable names to JSON-serialized values
     * @return this builder
     */
    public ModelParameters setChatTemplateKwargs(java.util.Map<String, String> kwargs) {
        parameters.put("--chat-template-kwargs", JsonParameters.mapToJsonObject(kwargs));
        return this;
    }

    /**
     * Set how much the prompt of a request must match the prompt of a slot in order to use that slot.
     *
     * @param similarity the minimum similarity threshold for slot reuse
     * @return this builder
     */
    public ModelParameters setSlotPromptSimilarity(float similarity) {
        parameters.put("--slot-prompt-similarity", String.valueOf(similarity));
        return this;
    }

    /**
     * Load LoRA adapters without applying them (apply later via POST /lora-adapters).
     *
     * @return this builder
     */
    public ModelParameters setLoraInitWithoutApply() {
        return setFlag(ModelFlag.LORA_INIT_WITHOUT_APPLY);
    }

    /**
     * Disable logging.
     *
     * @return this builder
     */
    public ModelParameters disableLog() {
        return setFlag(ModelFlag.LOG_DISABLE);
    }

    /**
     * Set the log file path.
     *
     * @param logFile the path to the log file
     * @return this builder
     */
    public ModelParameters setLogFile(String logFile) {
        parameters.put("--log-file", logFile);
        return this;
    }

    /**
     * Set verbosity level to infinity (log all messages, useful for debugging).
     *
     * @return this builder
     */
    public ModelParameters setVerbose() {
        return setFlag(ModelFlag.VERBOSE);
    }

    /**
     * Set the verbosity threshold (messages with a higher verbosity will be ignored).
     *
     * @param verbosity the verbosity threshold level
     * @return this builder
     */
    public ModelParameters setLogVerbosity(int verbosity) {
        parameters.put("--log-verbosity", String.valueOf(verbosity));
        return this;
    }

    /**
     * Enable prefix in log messages.
     *
     * @return this builder
     */
    public ModelParameters enableLogPrefix() {
        return setFlag(ModelFlag.LOG_PREFIX);
    }

    /**
     * Enable timestamps in log messages.
     *
     * @return this builder
     */
    public ModelParameters enableLogTimestamps() {
        return setFlag(ModelFlag.LOG_TIMESTAMPS);
    }

    /**
     * Set the number of tokens to draft for speculative decoding.
     *
     * @param draftMax the number of tokens to draft for speculative decoding
     * @return this builder
     */
    public ModelParameters setDraftMax(int draftMax) {
        parameters.put("--draft-max", String.valueOf(draftMax));
        return this;
    }

    /**
     * Set the minimum number of draft tokens to use for speculative decoding.
     *
     * @param draftMin the minimum number of draft tokens for speculative decoding
     * @return this builder
     */
    public ModelParameters setDraftMin(int draftMin) {
        parameters.put("--draft-min", String.valueOf(draftMin));
        return this;
    }

    /**
     * Set the minimum speculative decoding probability for greedy decoding.
     *
     * @param draftPMin the minimum speculative decoding probability for greedy decoding
     * @return this builder
     */
    public ModelParameters setDraftPMin(float draftPMin) {
        parameters.put("--draft-p-min", String.valueOf(draftPMin));
        return this;
    }

    /**
     * Set the size of the prompt context for the draft model.
     *
     * @param ctxSizeDraft the prompt context size for the draft model
     * @return this builder
     */
    public ModelParameters setCtxSizeDraft(int ctxSizeDraft) {
        parameters.put("--ctx-size-draft", String.valueOf(ctxSizeDraft));
        return this;
    }

    /**
     * Set the comma-separated list of devices to use for offloading the draft model.
     *
     * @param deviceDraft comma-separated list of devices for offloading the draft model
     * @return this builder
     */
    public ModelParameters setDeviceDraft(String deviceDraft) {
        parameters.put("--device-draft", deviceDraft);
        return this;
    }

    /**
     * Set the number of layers to store in VRAM for the draft model.
     *
     * @param gpuLayersDraft the number of draft model layers to store in VRAM
     * @return this builder
     */
    public ModelParameters setGpuLayersDraft(int gpuLayersDraft) {
        parameters.put("--gpu-layers-draft", String.valueOf(gpuLayersDraft));
        return this;
    }

    /**
     * Set the draft model for speculative decoding.
     *
     * @param modelDraft the file path of the draft model
     * @return this builder
     */
    public ModelParameters setModelDraft(String modelDraft) {
        parameters.put("--model-draft", modelDraft);
        return this;
    }

    /**
     * Enable jinja for templating
     *
     * @return this builder
     */
    public ModelParameters enableJinja() {
        return setFlag(ModelFlag.JINJA);
    }

    /**
     * Only load the vocabulary for tokenization, no weights (default: false).
     * A model loaded with this option can only be used for {@link LlamaModel#encode(String)}
     * and {@link LlamaModel#decode(int[])}. Inference, embedding, and reranking will not work.
     *
     * @return this builder
     */
    public ModelParameters setVocabOnly() {
        return setFlag(ModelFlag.VOCAB_ONLY);
    }

    /**
     * Enable or disable a single unified KV buffer shared across all sequences.
     * <p>
     * When enabled (the default when the number of slots is set to auto), all parallel slots
     * share one contiguous KV buffer which enables features such as cross-slot cache reuse
     * ({@link #setCacheRamMib}) and idle-slot eviction ({@link #setClearIdle}).
     *
     * @param kvUnified {@code true} to enable unified KV, {@code false} to disable
     * @return this builder
     */
    public ModelParameters setKvUnified(boolean kvUnified) {
        setFlag(kvUnified ? ModelFlag.KV_UNIFIED : ModelFlag.NO_KV_UNIFIED);
        clearFlag(kvUnified ? ModelFlag.NO_KV_UNIFIED : ModelFlag.KV_UNIFIED);
        return this;
    }

    /**
     * Set the maximum RAM cache size in MiB used to store saved slot KV state.
     * <p>
     * Requires {@link #setKvUnified(boolean) unified KV} to be enabled.
     * Set to {@code -1} for no limit, {@code 0} to disable (default: 8192 MiB).
     * Together with {@link #setClearIdle} this allows idle slots to be evicted
     * from GPU/CPU memory and restored quickly on the next matching request.
     *
     * @param cacheRamMib maximum cache size in MiB, or {@code -1} for unlimited
     * @return this builder
     */
    public ModelParameters setCacheRamMib(int cacheRamMib) {
        parameters.put("--cache-ram", String.valueOf(cacheRamMib));
        return this;
    }

    /**
     * Enable or disable saving and clearing idle slots when a new task starts.
     * <p>
     * When enabled (the default), idle slots have their KV state saved to the
     * RAM cache ({@link #setCacheRamMib}) and are then cleared, freeing GPU/CPU
     * memory for the active request.  The saved state is transparently restored
     * on the next request that shares the same prompt prefix, so cache-hit
     * latency is preserved.
     * <p>
     * Requires {@link #setKvUnified(boolean) unified KV} and a non-zero
     * {@link #setCacheRamMib RAM cache}.  If either dependency is absent the
     * server logs a warning and silently disables the feature.
     *
     * @param clearIdle {@code true} to enable idle-slot eviction (default), {@code false} to disable
     * @return this builder
     */
    public ModelParameters setClearIdle(boolean clearIdle) {
        setFlag(clearIdle ? ModelFlag.CLEAR_IDLE : ModelFlag.NO_CLEAR_IDLE);
        clearFlag(clearIdle ? ModelFlag.NO_CLEAR_IDLE : ModelFlag.CLEAR_IDLE);
        return this;
    }

    /**
     * Enable the given flag, adding it to the active parameter set.
     * Equivalent to calling the specific named method (e.g. {@link #enableFlashAttn()}
     * for {@link ModelFlag#FLASH_ATTN}).
     *
     * @param flag the flag to enable
     * @return this builder
     */
    public ModelParameters setFlag(ModelFlag flag) {
        parameters.put(flag.getCliFlag(), null);
        return this;
    }

    /**
     * Remove the given flag from the active parameter set.
     *
     * @param flag the flag to remove
     * @return this builder
     */
    public ModelParameters clearFlag(ModelFlag flag) {
        parameters.remove(flag.getCliFlag());
        return this;
    }

    /**
     * Returns whether the given parameter key has not been explicitly set.
     *
     * @param key the parameter key without the {@code --} prefix
     * @return {@code true} if the key is absent from the configured parameters
     */
    public boolean isDefault(String key) {
        return !parameters.containsKey("--" + key);
    }

}
