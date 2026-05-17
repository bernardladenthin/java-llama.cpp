// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Heurer
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
    ROW("row");

    private final String argValue;

    GpuSplitMode(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
