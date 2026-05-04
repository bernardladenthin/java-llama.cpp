package de.kherud.llama.args;

/**
 * Controls how reasoning/thinking tokens produced by models like DeepSeek-R1 and QwQ are
 * extracted and returned in the response.
 *
 * <p>Passed as {@code "reasoning_format"} in inference requests. Only meaningful when the model
 * uses a thinking tag (e.g. {@code <think>...</think>}) and chat-template rendering is active
 * ({@link de.kherud.llama.InferenceParameters#setUseChatTemplate(boolean)}).
 */
public enum ReasoningFormat implements CliArg {

    /**
     * Reasoning tokens are left in-line; no extraction is performed.
     */
    NONE("none"),

    /**
     * Automatically detect the reasoning format from the model's chat template.
     * Equivalent to {@link #DEEPSEEK} in most cases.
     */
    AUTO("auto"),

    /**
     * Extract thinking-tag content into a separate {@code reasoning_content} field,
     * including in streaming deltas.
     */
    DEEPSEEK("deepseek"),

    /**
     * Legacy DeepSeek format: extract thinking content into {@code reasoning_content} in
     * non-streaming mode; leave inline in {@code <think>} tags during streaming.
     */
    DEEPSEEK_LEGACY("deepseek-legacy");

    private final String argValue;

    ReasoningFormat(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
