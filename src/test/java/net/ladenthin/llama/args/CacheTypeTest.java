// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class CacheTypeTest extends AbstractCliArgEnumTest<CacheType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {CacheType.F32,    "f32",    9},
            {CacheType.F16,    "f16",    9},
            {CacheType.BF16,   "bf16",   9},
            {CacheType.Q8_0,   "q8_0",   9},
            {CacheType.Q4_0,   "q4_0",   9},
            {CacheType.Q4_1,   "q4_1",   9},
            {CacheType.IQ4_NL, "iq4_nl", 9},
            {CacheType.Q5_0,   "q5_0",   9},
            {CacheType.Q5_1,   "q5_1",   9},
        });
    }

}
