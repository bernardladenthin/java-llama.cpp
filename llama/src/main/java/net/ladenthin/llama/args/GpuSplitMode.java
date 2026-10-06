// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * GPU tensor split mode for {@code --split-mode}.
 */
public enum GpuSplitMode implements CliArg {

    /** No split; use a single GPU. */
    NONE("none"),
    /** Split by transformer layer across GPUs. */
    LAYER("layer"),
    /** Split by tensor row across GPUs. */
    ROW("row"),
    /**
     * Split weights and KV cache across GPUs and run them in parallel (tensor parallelism). Upstream
     * marks it EXPERIMENTAL; since llama.cpp b11450 (#26610) it works across RPC servers as well, which
     * then reduce their partial results with each other directly.
     */
    TENSOR("tensor");

    private final String argValue;

    GpuSplitMode(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
