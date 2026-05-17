// SPDX-FileCopyrightText: 2023-2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Konstantin Heurer
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * KV cache quantization type for {@code --cache-type-k} and {@code --cache-type-v}.
 */
public enum CacheType implements CliArg {

    /** 32-bit float. */
    F32("f32"),
    /** 16-bit float. */
    F16("f16"),
    /** 16-bit brain float. */
    BF16("bf16"),
    /** 8-bit quantization, scheme 0. */
    Q8_0("q8_0"),
    /** 4-bit quantization, scheme 0. */
    Q4_0("q4_0"),
    /** 4-bit quantization, scheme 1. */
    Q4_1("q4_1"),
    /** 4-bit non-linear importance quantization. */
    IQ4_NL("iq4_nl"),
    /** 5-bit quantization, scheme 0. */
    Q5_0("q5_0"),
    /** 5-bit quantization, scheme 1. */
    Q5_1("q5_1");

    private final String argValue;

    CacheType(String argValue) {
        this.argValue = argValue;
    }

    @Override
    public String getArgValue() {
        return argValue;
    }
}
