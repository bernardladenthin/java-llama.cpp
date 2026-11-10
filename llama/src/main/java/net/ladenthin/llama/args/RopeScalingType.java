// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * RoPE (Rotary Position Embedding) scaling type for {@code --rope-scaling}: the three values
 * llama.cpp's argument accepts. The model's own choice is not a constant; it applies when
 * {@link net.ladenthin.llama.parameters.ModelParameters#setRopeScaling(RopeScalingType)} is not called.
 */
public enum RopeScalingType implements CliArg {

    /** No RoPE scaling applied. */
    NONE("none"),
    /** Linear RoPE scaling. */
    LINEAR("linear"),
    /** YaRN (Yet Another RoPE extensioN) scaling. */
    YARN("yarn");

    private final String argValue;

    RopeScalingType(String value) {
        this.argValue = value;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
