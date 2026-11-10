// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

/**
 * Target file type (quantization scheme) for {@link net.ladenthin.llama.LlamaQuantizer}.
 *
 * <p>Each constant maps 1-to-1 to a {@code llama_ftype} enumerator in {@code include/llama.h}
 * (llama.cpp b9870); the stored integer is the exact native enum value passed to
 * {@code llama_model_quantize}. Only the file types the quantizer produces are constants: NVFP4
 * (39) and OCP FP8 E4M3 (42) name models written by converters, and {@code llama_model_quantize}
 * refuses both as a target ("invalid output file type"). Seven more are missing because they
 * need an importance matrix, which this library cannot pass ("this quantization requires an
 * imatrix!"): IQ2_XXS (19), IQ2_XS (20), Q2_K_S (21), IQ1_S (24), IQ2_S (28), IQ2_M (29) and
 * IQ1_M (31). Enumerators that upstream removed or
 * commented out are not represented either.
 */
public enum QuantizationType {

    /** All tensors kept as F32 — {@code LLAMA_FTYPE_ALL_F32 = 0}. */
    ALL_F32(0),
    /** Mostly F16, except 1d tensors — {@code LLAMA_FTYPE_MOSTLY_F16 = 1}. */
    F16(1),
    /** Mostly Q4_0 — {@code LLAMA_FTYPE_MOSTLY_Q4_0 = 2}. */
    Q4_0(2),
    /** Mostly Q4_1 — {@code LLAMA_FTYPE_MOSTLY_Q4_1 = 3}. */
    Q4_1(3),
    /** Mostly Q8_0 — {@code LLAMA_FTYPE_MOSTLY_Q8_0 = 7}. */
    Q8_0(7),
    /** Mostly Q5_0 — {@code LLAMA_FTYPE_MOSTLY_Q5_0 = 8}. */
    Q5_0(8),
    /** Mostly Q5_1 — {@code LLAMA_FTYPE_MOSTLY_Q5_1 = 9}. */
    Q5_1(9),
    /** Mostly Q2_K — {@code LLAMA_FTYPE_MOSTLY_Q2_K = 10}. */
    Q2_K(10),
    /** Mostly Q3_K, small — {@code LLAMA_FTYPE_MOSTLY_Q3_K_S = 11}. */
    Q3_K_S(11),
    /** Mostly Q3_K, medium — {@code LLAMA_FTYPE_MOSTLY_Q3_K_M = 12}. */
    Q3_K_M(12),
    /** Mostly Q3_K, large — {@code LLAMA_FTYPE_MOSTLY_Q3_K_L = 13}. */
    Q3_K_L(13),
    /** Mostly Q4_K, small — {@code LLAMA_FTYPE_MOSTLY_Q4_K_S = 14}. */
    Q4_K_S(14),
    /** Mostly Q4_K, medium — {@code LLAMA_FTYPE_MOSTLY_Q4_K_M = 15}. */
    Q4_K_M(15),
    /** Mostly Q5_K, small — {@code LLAMA_FTYPE_MOSTLY_Q5_K_S = 16}. */
    Q5_K_S(16),
    /** Mostly Q5_K, medium — {@code LLAMA_FTYPE_MOSTLY_Q5_K_M = 17}. */
    Q5_K_M(17),
    /** Mostly Q6_K — {@code LLAMA_FTYPE_MOSTLY_Q6_K = 18}. */
    Q6_K(18),
    /** Mostly IQ3_XS — {@code LLAMA_FTYPE_MOSTLY_IQ3_XS = 22}. */
    IQ3_XS(22),
    /** Mostly IQ3_XXS — {@code LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23}. */
    IQ3_XXS(23),
    /** Mostly IQ4_NL — {@code LLAMA_FTYPE_MOSTLY_IQ4_NL = 25}. */
    IQ4_NL(25),
    /** Mostly IQ3_S — {@code LLAMA_FTYPE_MOSTLY_IQ3_S = 26}. */
    IQ3_S(26),
    /** Mostly IQ3_M — {@code LLAMA_FTYPE_MOSTLY_IQ3_M = 27}. */
    IQ3_M(27),
    /** Mostly IQ4_XS — {@code LLAMA_FTYPE_MOSTLY_IQ4_XS = 30}. */
    IQ4_XS(30),
    /** Mostly BF16 — {@code LLAMA_FTYPE_MOSTLY_BF16 = 32}. */
    BF16(32),
    /** Mostly TQ1_0 — {@code LLAMA_FTYPE_MOSTLY_TQ1_0 = 36}. */
    TQ1_0(36),
    /** Mostly TQ2_0 — {@code LLAMA_FTYPE_MOSTLY_TQ2_0 = 37}. */
    TQ2_0(37),
    /** Mostly MXFP4 (MoE) — {@code LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38}. */
    MXFP4_MOE(38),
    /** Mostly Q1_0 — {@code LLAMA_FTYPE_MOSTLY_Q1_0 = 40}. */
    Q1_0(40),
    /** Mostly Q2_0 — {@code LLAMA_FTYPE_MOSTLY_Q2_0 = 41}. */
    Q2_0(41);

    private final int ftypeValue;

    QuantizationType(int ftypeValue) {
        this.ftypeValue = ftypeValue;
    }

    /**
     * The native {@code llama_ftype} enum value this constant maps to.
     *
     * @return the integer passed to {@code llama_model_quantize}
     */
    public int getFtypeValue() {
        return ftypeValue;
    }
}
