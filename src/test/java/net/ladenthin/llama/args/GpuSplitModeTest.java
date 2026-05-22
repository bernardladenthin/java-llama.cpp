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
public class GpuSplitModeTest extends AbstractCliArgEnumTest<GpuSplitMode> {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {GpuSplitMode.NONE,  "none",  3},
            {GpuSplitMode.LAYER, "layer", 3},
            {GpuSplitMode.ROW,   "row",   3},
        });
    }

    public GpuSplitModeTest(GpuSplitMode value, String expectedArgValue, int expectedEnumCount) {
        super(value, expectedArgValue, expectedEnumCount);
    }
}
