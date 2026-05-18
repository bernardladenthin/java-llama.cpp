// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * RoPE (Rotary Position Embedding) scaling type for {@code --rope-scaling}.
 */
public enum RopeScalingType implements CliArg {

    /** No scaling type specified; use the model default. */
    UNSPECIFIED("unspecified"),
    /** No RoPE scaling applied. */
    NONE("none"),
    /** Linear RoPE scaling. */
    LINEAR("linear"),
    /** YaRN (Yet Another RoPE extensioN) scaling. */
    YARN2("yarn"),
    /** LongRoPE scaling for extended context. */
    LONGROPE("longrope"),
    /** Maximum value sentinel. */
    MAX_VALUE("maxvalue");

    private final String argValue;

    RopeScalingType(String value) {
        this.argValue = value;
    }

    public String getArgValue() {
        return argValue;
    }
}