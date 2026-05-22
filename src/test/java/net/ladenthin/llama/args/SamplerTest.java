// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
public class SamplerTest extends AbstractCliArgEnumTest<Sampler> {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {Sampler.DRY,         "dry",         9},
            {Sampler.TOP_K,       "top_k",       9},
            {Sampler.TOP_P,       "top_p",       9},
            {Sampler.TYP_P,       "typ_p",       9},
            {Sampler.MIN_P,       "min_p",       9},
            {Sampler.TEMPERATURE, "temperature", 9},
            {Sampler.XTC,         "xtc",         9},
            {Sampler.INFILL,      "infill",      9},
            {Sampler.PENALTIES,   "penalties",   9},
        });
    }

    public SamplerTest(Sampler value, String expectedArgValue, int expectedEnumCount) {
        super(value, expectedArgValue, expectedEnumCount);
    }
}
