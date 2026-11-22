// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class SamplerTest extends AbstractCliArgEnumTest<Sampler> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {Sampler.DRY, "dry", 11},
            {Sampler.TOP_K, "top_k", 11},
            {Sampler.TOP_P, "top_p", 11},
            {Sampler.TYP_P, "typ_p", 11},
            {Sampler.MIN_P, "min_p", 11},
            {Sampler.TEMPERATURE, "temperature", 11},
            {Sampler.XTC, "xtc", 11},
            {Sampler.INFILL, "infill", 11},
            {Sampler.PENALTIES, "penalties", 11},
            {Sampler.TOP_N_SIGMA, "top_n_sigma", 11},
            {Sampler.ADAPTIVE_P, "adaptive_p", 11},
        });
    }
}
