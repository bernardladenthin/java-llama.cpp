package de.kherud.llama.args;

/**
 * Boolean CLI flags for {@link de.kherud.llama.ModelParameters}.
 *
 * <p>Each constant maps to a single CLI argument that takes no value — its presence
 * alone enables the behaviour. Pass to
 * {@link de.kherud.llama.ModelParameters#setFlag(ModelFlag)} /
 * {@link de.kherud.llama.ModelParameters#clearFlag(ModelFlag)} for programmatic control,
 * or use the named convenience methods (e.g. {@link de.kherud.llama.ModelParameters#enableFlashAttn()}).
 */
public enum ModelFlag {

    /** Disable context shift on infinite text generation. */
    NO_CONTEXT_SHIFT("--no-context-shift"),

    /** Enable Flash Attention. */
    FLASH_ATTN("--flash-attn"),

    /** Disable internal libllama performance timings. */
    NO_PERF("--no-perf"),

    /** Process escape sequences (e.g. {@code \\n}, {@code \\t}). */
    ESCAPE("--escape"),

    /** Do not process escape sequences. */
    NO_ESCAPE("--no-escape"),

    /** Enable special tokens in output. */
    SPECIAL("--special"),

    /** Skip warming up the model with an empty run. */
    NO_WARMUP("--no-warmup"),

    /** Use Suffix/Prefix/Middle infill pattern instead of Prefix/Suffix/Middle. */
    SPM_INFILL("--spm-infill"),

    /** Ignore end-of-stream token and continue generating. */
    IGNORE_EOS("--ignore-eos"),

    /** Enable verbose printing of the KV cache. */
    DUMP_KV_CACHE("--dump-kv-cache"),

    /** Disable KV offload. */
    NO_KV_OFFLOAD("--no-kv-offload"),

    /** Enable continuous (dynamic) batching. */
    CONT_BATCHING("--cont-batching"),

    /** Disable continuous batching. */
    NO_CONT_BATCHING("--no-cont-batching"),

    /** Force system to keep model in RAM rather than swapping or compressing. */
    MLOCK("--mlock"),

    /** Do not memory-map model (slower load but may reduce pageouts if not using mlock). */
    NO_MMAP("--no-mmap"),

    /** Enable checking model tensor data for invalid values. */
    CHECK_TENSORS("--check-tensors"),

    /** Enable embedding use case; use only with dedicated embedding models. */
    EMBEDDING("--embedding"),

    /** Enable reranking endpoint on server. */
    RERANKING("--reranking"),

    /** Load LoRA adapters without applying them (apply later via POST /lora-adapters). */
    LORA_INIT_WITHOUT_APPLY("--lora-init-without-apply"),

    /** Disable logging. */
    LOG_DISABLE("--log-disable"),

    /** Set verbosity level to infinity (log all messages). */
    VERBOSE("--verbose"),

    /** Enable prefix in log messages. */
    LOG_PREFIX("--log-prefix"),

    /** Enable timestamps in log messages. */
    LOG_TIMESTAMPS("--log-timestamps"),

    /** Enable Jinja templating for chat templates. */
    JINJA("--jinja"),

    /** Only load the vocabulary for tokenization; no weights are loaded. */
    VOCAB_ONLY("--vocab-only"),

    /** Enable a single unified KV buffer shared across all sequences. */
    KV_UNIFIED("--kv-unified"),

    /** Disable the unified KV buffer. */
    NO_KV_UNIFIED("--no-kv-unified"),

    /** Enable saving and clearing idle slots when a new task starts. */
    CLEAR_IDLE("--clear-idle"),

    /** Disable saving and clearing idle slots. */
    NO_CLEAR_IDLE("--no-clear-idle");

    private final String cliFlag;

    ModelFlag(String cliFlag) {
        this.cliFlag = cliFlag;
    }

    /**
     * Returns the CLI argument string for this flag (e.g. {@code "--flash-attn"}).
     *
     * @return the CLI flag string
     */
    public String getCliFlag() {
        return cliFlag;
    }
}
