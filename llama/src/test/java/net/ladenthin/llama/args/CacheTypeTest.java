// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class CacheTypeTest extends AbstractCliArgEnumTest<CacheType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {CacheType.F32, "f32", 10},
            {CacheType.F16, "f16", 10},
            {CacheType.BF16, "bf16", 10},
            {CacheType.F8_E4M3, "f8_e4m3", 10},
            {CacheType.Q8_0, "q8_0", 10},
            {CacheType.Q4_0, "q4_0", 10},
            {CacheType.Q4_1, "q4_1", 10},
            {CacheType.IQ4_NL, "iq4_nl", 10},
            {CacheType.Q5_0, "q5_0", 10},
            {CacheType.Q5_1, "q5_1", 10},
        });
    }
}
