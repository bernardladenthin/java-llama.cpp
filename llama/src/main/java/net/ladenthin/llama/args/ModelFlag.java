// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Boolean CLI flags for {@link net.ladenthin.llama.parameters.ModelParameters}.
 *
 * <p>Each constant maps to a single CLI argument that takes no value — its presence
 * alone enables the behaviour. Pass to
 * {@link net.ladenthin.llama.parameters.ModelParameters#setFlag(ModelFlag)} /
 * {@link net.ladenthin.llama.parameters.ModelParameters#clearFlag(ModelFlag)} for programmatic control,
 * or use the named convenience methods (e.g. {@link net.ladenthin.llama.parameters.ModelParameters#enableSwaFull()}).
 *
 * <p>{@code --flash-attn} is deliberately NOT here. It looks like a flag and was modelled as one, but
 * llama.cpp has required a mandatory {@code on|off|auto} value since b10273 — emitting the key alone
 * makes the parser consume the next argv token. Listing it would leave that broken argv reachable
 * through {@code setFlag}. Use
 * {@link net.ladenthin.llama.parameters.ModelParameters#setFlashAttn(net.ladenthin.llama.args.FlashAttn)}.</p>
 */
public enum ModelFlag {

    /** Disable context shift on infinite text generation. */
    NO_CONTEXT_SHIFT("--no-context-shift"),

    /** Keep the full-size sliding-window-attention (SWA) KV cache, enabling cross-request
     *  prompt-prefix reuse (pairs with --cache-reuse) at ~2x the SWA-layer KV RAM. Default off.
     *  Env: LLAMA_ARG_SWA_FULL. */
    SWA_FULL("--swa-full"),

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
    CLEAR_IDLE("--cache-idle-slots"),

    /** Disable saving and clearing idle slots. */
    NO_CLEAR_IDLE("--no-cache-idle-slots"),

    /** Automatically detect and load the mmproj vision projection model. */
    MMPROJ_AUTO("--mmproj-auto"),

    /** Disable automatic multimodal-projector detection. */
    NO_MMPROJ_AUTO("--no-mmproj-auto"),

    /** Offload the mmproj vision projection model to the GPU. */
    MMPROJ_OFFLOAD("--mmproj-offload"),

    /** Keep the multimodal projector on the CPU. */
    NO_MMPROJ_OFFLOAD("--no-mmproj-offload"),

    /**
     * Run fully offline — never make an outbound network request to download a model. Useful for
     * air-gapped or pre-staged-model deployments where any outbound call is itself a failure mode.
     *
     * <p>Maps to the upstream {@code --offline} flag ({@code common_params::offline}), which the
     * model-download pipeline honors by skipping all download tasks: a model already present on
     * disk (or in the Hugging Face cache) loads normally, while a missing one fails instead of
     * being fetched. When a local model path is configured via
     * {@link net.ladenthin.llama.parameters.ModelParameters#setModel(String)} and that file does
     * not exist, the loader reports a typed
     * {@link net.ladenthin.llama.exception.ModelUnavailableException} so callers can distinguish an
     * air-gapped miss from a genuine misconfiguration.</p>
     */
    OFFLINE("--offline");

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
