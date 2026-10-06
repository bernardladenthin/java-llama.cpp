// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class DraftSamplingTest extends AbstractCliArgEnumTest<DraftSampling> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {DraftSampling.GREEDY, "greedy", 2},
            {DraftSampling.PROBABILISTIC, "probabilistic", 2},
        });
    }
}
