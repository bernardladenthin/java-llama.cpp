// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Every value-taking CLI option {@link net.ladenthin.llama.parameters.ModelParameters} can emit.
 *
 * <p>This enum is the single declaration of those names. The builder setters reference a constant
 * instead of writing a string, and the base class accepts nothing else, so a name that is not
 * declared here cannot reach the argv that {@code LlamaModel.loadModel()} hands to
 * {@code common_params_parse} — the emitted set is closed by construction rather than by review.
 *
 * <p>{@link ModelFlag} is the same registry for the options that take no value. Both are read at
 * configure time by {@code cmake/extract-java-cli-flags.cmake} and checked against the real server
 * option table by {@code src/test/cpp/test_model_flags.cpp}, which runs on every platform.
 *
 * @see CliContract
 */
public enum ModelOption {

    /** CLI option {@code --batch-size}. */
    BATCH_SIZE("--batch-size"),

    /** CLI option {@code --cache-ram}. */
    CACHE_RAM("--cache-ram"),

    /** CLI option {@code --cache-reuse}. */
    CACHE_REUSE("--cache-reuse"),

    /** CLI option {@code --cache-type-k}. */
    CACHE_TYPE_K("--cache-type-k"),

    /** CLI option {@code --cache-type-v}. */
    CACHE_TYPE_V("--cache-type-v"),

    /** CLI option {@code --chat-template}. */
    CHAT_TEMPLATE("--chat-template"),

    /** CLI option {@code --chat-template-kwargs}. */
    CHAT_TEMPLATE_KWARGS("--chat-template-kwargs"),

    /** CLI option {@code --checkpoint-min-step}. */
    CHECKPOINT_MIN_STEP("--checkpoint-min-step"),

    /** CLI option {@code --control-vector}. */
    CONTROL_VECTOR("--control-vector"),

    /** CLI option {@code --control-vector-layer-range}. */
    CONTROL_VECTOR_LAYER_RANGE("--control-vector-layer-range"),

    /** CLI option {@code --control-vector-scaled}. */
    CONTROL_VECTOR_SCALED("--control-vector-scaled"),

    /** CLI option {@code --cpu-mask}. */
    CPU_MASK("--cpu-mask"),

    /** CLI option {@code --cpu-mask-batch}. */
    CPU_MASK_BATCH("--cpu-mask-batch"),

    /** CLI option {@code --cpu-range}. */
    CPU_RANGE("--cpu-range"),

    /** CLI option {@code --cpu-range-batch}. */
    CPU_RANGE_BATCH("--cpu-range-batch"),

    /** CLI option {@code --cpu-strict}. */
    CPU_STRICT("--cpu-strict"),

    /** CLI option {@code --cpu-strict-batch}. */
    CPU_STRICT_BATCH("--cpu-strict-batch"),

    /** CLI option {@code --ctx-checkpoints}. */
    CTX_CHECKPOINTS("--ctx-checkpoints"),

    /** CLI option {@code --ctx-size}. */
    CTX_SIZE("--ctx-size"),

    /** CLI option {@code --defrag-thold}. */
    DEFRAG_THOLD("--defrag-thold"),

    /** CLI option {@code --device}. */
    DEVICE("--device"),

    /** CLI option {@code --dry-allowed-length}. */
    DRY_ALLOWED_LENGTH("--dry-allowed-length"),

    /** CLI option {@code --dry-base}. */
    DRY_BASE("--dry-base"),

    /** CLI option {@code --dry-multiplier}. */
    DRY_MULTIPLIER("--dry-multiplier"),

    /** CLI option {@code --dry-penalty-last-n}. */
    DRY_PENALTY_LAST_N("--dry-penalty-last-n"),

    /** CLI option {@code --dry-sequence-breaker}. */
    DRY_SEQUENCE_BREAKER("--dry-sequence-breaker"),

    /** CLI option {@code --dynatemp-exp}. */
    DYNATEMP_EXP("--dynatemp-exp"),

    /** CLI option {@code --dynatemp-range}. */
    DYNATEMP_RANGE("--dynatemp-range"),

    /** CLI option {@code --fit}. */
    FIT("--fit"),

    /** CLI option {@code --flash-attn}. */
    FLASH_ATTN("--flash-attn"),

    /** CLI option {@code --frequency-penalty}. */
    FREQUENCY_PENALTY("--frequency-penalty"),

    /** CLI option {@code --gpu-layers}. */
    GPU_LAYERS("--gpu-layers"),

    /** CLI option {@code --grammar}. */
    GRAMMAR("--grammar"),

    /** CLI option {@code --grammar-file}. */
    GRAMMAR_FILE("--grammar-file"),

    /** CLI option {@code --hf-file}. */
    HF_FILE("--hf-file"),

    /** CLI option {@code --hf-repo}. */
    HF_REPO("--hf-repo"),

    /** CLI option {@code --hf-token}. */
    HF_TOKEN("--hf-token"),

    /** CLI option {@code --json-schema}. */
    JSON_SCHEMA("--json-schema"),

    /** CLI option {@code --keep}. */
    KEEP("--keep"),

    /** CLI option {@code --kv-unified-per-slot}. */
    KV_UNIFIED_PER_SLOT("--kv-unified-per-slot"),

    /** CLI option {@code --lazy-mode}. */
    LAZY_MODE("--lazy-mode"),

    /** CLI option {@code --load-mode}. */
    LOAD_MODE("--load-mode"),

    /** CLI option {@code --log-file}. */
    LOG_FILE("--log-file"),

    /** CLI option {@code --log-verbosity}. */
    LOG_VERBOSITY("--log-verbosity"),

    /** CLI option {@code --logit-bias}. */
    LOGIT_BIAS("--logit-bias"),

    /** CLI option {@code --lora}. */
    LORA("--lora"),

    /** CLI option {@code --lora-scaled}. */
    LORA_SCALED("--lora-scaled"),

    /** CLI option {@code --main-gpu}. */
    MAIN_GPU("--main-gpu"),

    /** CLI option {@code --min-p}. */
    MIN_P("--min-p"),

    /** CLI option {@code --mirostat}. */
    MIROSTAT("--mirostat"),

    /** CLI option {@code --mirostat-ent}. */
    MIROSTAT_ENT("--mirostat-ent"),

    /** CLI option {@code --mirostat-lr}. */
    MIROSTAT_LR("--mirostat-lr"),

    /** CLI option {@code --mmproj}. */
    MMPROJ("--mmproj"),

    /** CLI option {@code --mmproj-device}. */
    MMPROJ_DEVICE("--mmproj-device"),

    /** CLI option {@code --mmproj-url}. */
    MMPROJ_URL("--mmproj-url"),

    /** CLI option {@code --model}. */
    MODEL("--model"),

    /** CLI option {@code --model-url}. */
    MODEL_URL("--model-url"),

    /** CLI option {@code --n-cpu-ffn}. */
    N_CPU_FFN("--n-cpu-ffn"),

    /** CLI option {@code --n-cpu-moe}. */
    N_CPU_MOE("--n-cpu-moe"),

    /** CLI option {@code --numa}. */
    NUMA("--numa"),

    /** CLI option {@code --override-kv}. */
    OVERRIDE_KV("--override-kv"),

    /** CLI option {@code --parallel}. */
    PARALLEL("--parallel"),

    /** CLI option {@code --poll}. */
    POLL("--poll"),

    /** CLI option {@code --poll-batch}. */
    POLL_BATCH("--poll-batch"),

    /** CLI option {@code --pooling}. */
    POOLING("--pooling"),

    /** CLI option {@code --predict}. */
    PREDICT("--predict"),

    /** CLI option {@code --presence-penalty}. */
    PRESENCE_PENALTY("--presence-penalty"),

    /** CLI option {@code --prio}. */
    PRIO("--prio"),

    /** CLI option {@code --prio-batch}. */
    PRIO_BATCH("--prio-batch"),

    /** CLI option {@code --reasoning-budget}. */
    REASONING_BUDGET("--reasoning-budget"),

    /** CLI option {@code --reasoning-format}. */
    REASONING_FORMAT("--reasoning-format"),

    /** CLI option {@code --repeat-last-n}. */
    REPEAT_LAST_N("--repeat-last-n"),

    /** CLI option {@code --repeat-penalty}. */
    REPEAT_PENALTY("--repeat-penalty"),

    /** CLI option {@code --rope-freq-base}. */
    ROPE_FREQ_BASE("--rope-freq-base"),

    /** CLI option {@code --rope-freq-scale}. */
    ROPE_FREQ_SCALE("--rope-freq-scale"),

    /** CLI option {@code --rope-scale}. */
    ROPE_SCALE("--rope-scale"),

    /** CLI option {@code --rope-scaling}. */
    ROPE_SCALING("--rope-scaling"),

    /** CLI option {@code --samplers}. */
    SAMPLERS("--samplers"),

    /** CLI option {@code --seed}. */
    SEED("--seed"),

    /** CLI option {@code --sleep-idle-seconds}. */
    SLEEP_IDLE_SECONDS("--sleep-idle-seconds"),

    /** CLI option {@code --slot-prompt-similarity}. */
    SLOT_PROMPT_SIMILARITY("--slot-prompt-similarity"),

    /** CLI option {@code --slot-save-path}. */
    SLOT_SAVE_PATH("--slot-save-path"),

    /** CLI option {@code --spec-draft-device}. */
    SPEC_DRAFT_DEVICE("--spec-draft-device"),

    /** CLI option {@code --spec-draft-model}. */
    SPEC_DRAFT_MODEL("--spec-draft-model"),

    /** CLI option {@code --spec-draft-n-max}. */
    SPEC_DRAFT_N_MAX("--spec-draft-n-max"),

    /** CLI option {@code --spec-draft-n-min}. */
    SPEC_DRAFT_N_MIN("--spec-draft-n-min"),

    /** CLI option {@code --spec-draft-ngl}. */
    SPEC_DRAFT_NGL("--spec-draft-ngl"),

    /** CLI option {@code --spec-draft-p-min}. */
    SPEC_DRAFT_P_MIN("--spec-draft-p-min"),

    /** CLI option {@code --split-mode}. */
    SPLIT_MODE("--split-mode"),

    /** CLI option {@code --temp}. */
    TEMP("--temp"),

    /** CLI option {@code --tensor-split}. */
    TENSOR_SPLIT("--tensor-split"),

    /** CLI option {@code --threads}. */
    THREADS("--threads"),

    /** CLI option {@code --threads-batch}. */
    THREADS_BATCH("--threads-batch"),

    /** CLI option {@code --top-k}. */
    TOP_K("--top-k"),

    /** CLI option {@code --top-p}. */
    TOP_P("--top-p"),

    /** CLI option {@code --typical}. */
    TYPICAL("--typical"),

    /** CLI option {@code --ubatch-size}. */
    UBATCH_SIZE("--ubatch-size"),

    /** CLI option {@code --video-ffmpeg-dir}. */
    VIDEO_FFMPEG_DIR("--video-ffmpeg-dir"),

    /** CLI option {@code --video-fps}. */
    VIDEO_FPS("--video-fps"),

    /** CLI option {@code --video-timestamp-interval}. */
    VIDEO_TIMESTAMP_INTERVAL("--video-timestamp-interval"),

    /** CLI option {@code --xtc-probability}. */
    XTC_PROBABILITY("--xtc-probability"),

    /** CLI option {@code --xtc-threshold}. */
    XTC_THRESHOLD("--xtc-threshold"),

    /** CLI option {@code --yarn-attn-factor}. */
    YARN_ATTN_FACTOR("--yarn-attn-factor"),

    /** CLI option {@code --yarn-beta-fast}. */
    YARN_BETA_FAST("--yarn-beta-fast"),

    /** CLI option {@code --yarn-beta-slow}. */
    YARN_BETA_SLOW("--yarn-beta-slow"),

    /** CLI option {@code --yarn-ext-factor}. */
    YARN_EXT_FACTOR("--yarn-ext-factor"),

    /** CLI option {@code --yarn-orig-ctx}. */
    YARN_ORIG_CTX("--yarn-orig-ctx");

    private final String cliOption;
    private final CliContract contract;

    ModelOption(String cliOption) {
        this(cliOption, CliContract.SERVER_PARSER);
    }

    ModelOption(String cliOption, CliContract contract) {
        this.cliOption = cliOption;
        this.contract = contract;
    }

    /**
     * Returns the CLI argument string for this option (e.g. {@code "--ctx-size"}).
     *
     * @return the CLI option string
     */
    public String getCliOption() {
        return cliOption;
    }

    /**
     * Returns the contract this option is required to satisfy.
     *
     * @return the contract kind
     */
    public CliContract getContract() {
        return contract;
    }
}
