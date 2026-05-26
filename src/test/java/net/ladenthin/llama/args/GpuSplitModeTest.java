// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class GpuSplitModeTest extends AbstractCliArgEnumTest<GpuSplitMode> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {GpuSplitMode.NONE,  "none",  3},
            {GpuSplitMode.LAYER, "layer", 3},
            {GpuSplitMode.ROW,   "row",   3},
        });
    }

}
