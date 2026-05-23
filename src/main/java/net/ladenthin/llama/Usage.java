// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

/**
 * Token-usage counters, modeled after the OpenAI / Llama Stack {@code usage} block.
 * <p>
 * Used by {@link ServerMetrics} to expose cumulative server-wide token totals and
 * (in a future {@code ChatResponse}) per-completion counts.
 * </p>
 */
public final class Usage {

    private final long promptTokens;
    private final long completionTokens;

    /**
     * Construct a usage record.
     *
     * @param promptTokens     number of prompt tokens
     * @param completionTokens number of completion tokens
     */
    public Usage(long promptTokens, long completionTokens) {
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
    }

    /**
     * Prompt-side token count.
     * @return number of prompt tokens
     */
    public long getPromptTokens() {
        return promptTokens;
    }

    /**
     * Completion-side token count.
     * @return number of completion tokens
     */
    public long getCompletionTokens() {
        return completionTokens;
    }

    /**
     * Convenience sum of the prompt and completion counts.
     * @return sum of prompt and completion tokens
     */
    public long getTotalTokens() {
        return promptTokens + completionTokens;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Usage)) return false;
        Usage u = (Usage) o;
        return promptTokens == u.promptTokens && completionTokens == u.completionTokens;
    }

    @Override
    public int hashCode() {
        return (int) (promptTokens * 31 + completionTokens);
    }

    @Override
    public String toString() {
        return "Usage{promptTokens=" + promptTokens
                + ", completionTokens=" + completionTokens
                + ", totalTokens=" + getTotalTokens() + "}";
    }
}
