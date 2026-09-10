// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

/**
 * Every key {@link InferenceParameters} can write into a request body.
 *
 * <p>This enum is the single declaration of those names, and {@link JsonParameters} accepts nothing
 * else, so a key that is not declared here cannot reach the wire. It is deliberately not public:
 * the point is a closed set, not an escape hatch — a new key is added by declaring a constant with
 * the contract it satisfies, which is what makes it visible to the guard.
 *
 * <p>Read at configure time by {@code cmake/extract-java-wire-names.cmake} and checked against
 * llama.cpp's own request-schema field table by the C++ test suite.
 *
 * @see RequestContract
 */
enum RequestField {

    /** Request key {@code cache_prompt}. */
    CACHE_PROMPT("cache_prompt"),

    /** Request key {@code chat_template_kwargs}, consumed by the OAI/task layer. */
    CHAT_TEMPLATE_KWARGS("chat_template_kwargs", RequestContract.OAI_LAYER),

    /** Request key {@code continue_final_message}. */
    CONTINUE_FINAL_MESSAGE("continue_final_message"),

    /** Request key {@code dry_allowed_length}. */
    DRY_ALLOWED_LENGTH("dry_allowed_length"),

    /** Request key {@code dry_base}. */
    DRY_BASE("dry_base"),

    /** Request key {@code dry_multiplier}. */
    DRY_MULTIPLIER("dry_multiplier"),

    /** Request key {@code dry_penalty_last_n}. */
    DRY_PENALTY_LAST_N("dry_penalty_last_n"),

    /** Request key {@code dry_sequence_breakers}. */
    DRY_SEQUENCE_BREAKERS("dry_sequence_breakers"),

    /** Request key {@code dynatemp_exponent}. */
    DYNATEMP_EXPONENT("dynatemp_exponent"),

    /** Request key {@code dynatemp_range}. */
    DYNATEMP_RANGE("dynatemp_range"),

    /** Request key {@code frequency_penalty}. */
    FREQUENCY_PENALTY("frequency_penalty"),

    /** Request key {@code grammar}. */
    GRAMMAR("grammar"),

    /** Request key {@code id_slot}, consumed by the OAI/task layer. */
    ID_SLOT("id_slot", RequestContract.OAI_LAYER),

    /** Request key {@code ignore_eos}. */
    IGNORE_EOS("ignore_eos"),

    /** Request key {@code input_prefix}, consumed by the OAI/task layer. */
    INPUT_PREFIX("input_prefix", RequestContract.OAI_LAYER),

    /** Request key {@code input_suffix}, consumed by the OAI/task layer. */
    INPUT_SUFFIX("input_suffix", RequestContract.OAI_LAYER),

    /** Request key {@code json_schema}. */
    JSON_SCHEMA("json_schema"),

    /** Request key {@code logit_bias}. */
    LOGIT_BIAS("logit_bias"),

    /** Request key {@code messages}, consumed by the OAI/task layer. */
    MESSAGES("messages", RequestContract.OAI_LAYER),

    /** Request key {@code min_keep}. */
    MIN_KEEP("min_keep"),

    /** Request key {@code min_p}. */
    MIN_P("min_p"),

    /** Request key {@code mirostat}. */
    MIROSTAT("mirostat"),

    /** Request key {@code mirostat_eta}. */
    MIROSTAT_ETA("mirostat_eta"),

    /** Request key {@code mirostat_tau}. */
    MIROSTAT_TAU("mirostat_tau"),

    /** Request key {@code n_cache_reuse}. */
    N_CACHE_REUSE("n_cache_reuse"),

    /** Request key {@code n_discard}. */
    N_DISCARD("n_discard"),

    /** Request key {@code n_indent}. */
    N_INDENT("n_indent"),

    /** Request key {@code n_keep}. */
    N_KEEP("n_keep"),

    /** Request key {@code n_predict}. */
    N_PREDICT("n_predict"),

    /** Request key {@code n_probs}. */
    N_PROBS("n_probs"),

    /** Request key {@code parallel_tool_calls}, consumed by the OAI/task layer. */
    PARALLEL_TOOL_CALLS("parallel_tool_calls", RequestContract.OAI_LAYER),

    /** Request key {@code post_sampling_probs}. */
    POST_SAMPLING_PROBS("post_sampling_probs"),

    /** Request key {@code presence_penalty}. */
    PRESENCE_PENALTY("presence_penalty"),

    /** Request key {@code prompt}, consumed by the OAI/task layer. */
    PROMPT("prompt", RequestContract.OAI_LAYER),

    /** Request key {@code reasoning_budget_tokens}. */
    REASONING_BUDGET_TOKENS("reasoning_budget_tokens"),

    /** Request key {@code reasoning_format}. */
    REASONING_FORMAT("reasoning_format"),

    /** Request key {@code repeat_last_n}. */
    REPEAT_LAST_N("repeat_last_n"),

    /** Request key {@code repeat_penalty}. */
    REPEAT_PENALTY("repeat_penalty"),

    /** Request key {@code response_format}, consumed by the OAI/task layer. */
    RESPONSE_FORMAT("response_format", RequestContract.OAI_LAYER),

    /** Request key {@code return_tokens}. */
    RETURN_TOKENS("return_tokens"),

    /** Request key {@code samplers}. */
    SAMPLERS("samplers"),

    /** Request key {@code seed}. */
    SEED("seed"),

    /** Request key {@code sse_ping_interval}. */
    SSE_PING_INTERVAL("sse_ping_interval"),

    /** Request key {@code stop}. */
    STOP("stop"),

    /** Request key {@code stream}. */
    STREAM("stream"),

    /** Request key {@code stream_options}. */
    STREAM_OPTIONS("stream_options"),

    /** Request key {@code t_max_predict_ms}. */
    T_MAX_PREDICT_MS("t_max_predict_ms"),

    /** Request key {@code temperature}. */
    TEMPERATURE("temperature"),

    /** Request key {@code timings_per_token}. */
    TIMINGS_PER_TOKEN("timings_per_token"),

    /** Request key {@code tool_choice}, consumed by the OAI/task layer. */
    TOOL_CHOICE("tool_choice", RequestContract.OAI_LAYER),

    /** Request key {@code tools}, consumed by the OAI/task layer. */
    TOOLS("tools", RequestContract.OAI_LAYER),

    /** Request key {@code top_k}. */
    TOP_K("top_k"),

    /** Request key {@code top_n_sigma}. */
    TOP_N_SIGMA("top_n_sigma"),

    /** Request key {@code top_p}. */
    TOP_P("top_p"),

    /** Request key {@code typical_p}. */
    TYPICAL_P("typical_p"),

    /** Request key {@code xtc_probability}. */
    XTC_PROBABILITY("xtc_probability"),

    /** Request key {@code xtc_threshold}. */
    XTC_THRESHOLD("xtc_threshold");

    private final String key;
    private final RequestContract contract;

    RequestField(String key) {
        this(key, RequestContract.SCHEMA);
    }

    RequestField(String key, RequestContract contract) {
        this.key = key;
        this.contract = contract;
    }

    /**
     * Returns the JSON key this field is written under (e.g. {@code "top_k"}).
     *
     * @return the request key
     */
    String getKey() {
        return key;
    }

    /**
     * Returns the contract this key is required to satisfy.
     *
     * @return the contract kind
     */
    RequestContract getContract() {
        return contract;
    }
}
