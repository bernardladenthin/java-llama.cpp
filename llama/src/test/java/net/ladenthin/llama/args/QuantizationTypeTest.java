// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Stream;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@ClaudeGenerated(
        purpose = "Pin every QuantizationType constant to its exact native llama_ftype enum value "
                + "(llama.cpp b9870 include/llama.h) — a wrong value would silently quantize to a "
                + "different scheme.")
public class QuantizationTypeTest {

    static Stream<Arguments> expectedFtypeValues() {
        return Stream.of(
                Arguments.of(QuantizationType.ALL_F32, 0),
                Arguments.of(QuantizationType.F16, 1),
                Arguments.of(QuantizationType.Q4_0, 2),
                Arguments.of(QuantizationType.Q4_1, 3),
                Arguments.of(QuantizationType.Q8_0, 7),
                Arguments.of(QuantizationType.Q5_0, 8),
                Arguments.of(QuantizationType.Q5_1, 9),
                Arguments.of(QuantizationType.Q2_K, 10),
                Arguments.of(QuantizationType.Q3_K_S, 11),
                Arguments.of(QuantizationType.Q3_K_M, 12),
                Arguments.of(QuantizationType.Q3_K_L, 13),
                Arguments.of(QuantizationType.Q4_K_S, 14),
                Arguments.of(QuantizationType.Q4_K_M, 15),
                Arguments.of(QuantizationType.Q5_K_S, 16),
                Arguments.of(QuantizationType.Q5_K_M, 17),
                Arguments.of(QuantizationType.Q6_K, 18),
                Arguments.of(QuantizationType.IQ2_XXS, 19),
                Arguments.of(QuantizationType.IQ2_XS, 20),
                Arguments.of(QuantizationType.Q2_K_S, 21),
                Arguments.of(QuantizationType.IQ3_XS, 22),
                Arguments.of(QuantizationType.IQ3_XXS, 23),
                Arguments.of(QuantizationType.IQ1_S, 24),
                Arguments.of(QuantizationType.IQ4_NL, 25),
                Arguments.of(QuantizationType.IQ3_S, 26),
                Arguments.of(QuantizationType.IQ3_M, 27),
                Arguments.of(QuantizationType.IQ2_S, 28),
                Arguments.of(QuantizationType.IQ2_M, 29),
                Arguments.of(QuantizationType.IQ4_XS, 30),
                Arguments.of(QuantizationType.IQ1_M, 31),
                Arguments.of(QuantizationType.BF16, 32),
                Arguments.of(QuantizationType.TQ1_0, 36),
                Arguments.of(QuantizationType.TQ2_0, 37),
                Arguments.of(QuantizationType.MXFP4_MOE, 38),
                Arguments.of(QuantizationType.NVFP4, 39),
                Arguments.of(QuantizationType.Q1_0, 40),
                Arguments.of(QuantizationType.Q2_0, 41));
    }

    @ParameterizedTest
    @MethodSource("expectedFtypeValues")
    public void ftypeValueMatchesNativeEnum(QuantizationType type, int expected) {
        assertThat(type.getFtypeValue(), is(expected));
    }

    /** Every constant must be covered by the mapping table above. */
    @Test
    public void mappingTableCoversAllConstants() {
        assertThat((int) expectedFtypeValues().count(), is(QuantizationType.values().length));
    }

    /** llama_ftype values are unique; two constants sharing a value would be a copy-paste bug. */
    @Test
    public void ftypeValuesAreUnique() {
        Set<Integer> seen = new HashSet<>();
        for (QuantizationType type : QuantizationType.values()) {
            assertThat("duplicate ftype value " + type.getFtypeValue(), seen.add(type.getFtypeValue()), is(true));
        }
    }
}
