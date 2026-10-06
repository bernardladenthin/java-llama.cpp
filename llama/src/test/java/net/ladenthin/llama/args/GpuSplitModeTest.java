// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class GpuSplitModeTest extends AbstractCliArgEnumTest<GpuSplitMode> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {GpuSplitMode.NONE, "none", 4},
            {GpuSplitMode.LAYER, "layer", 4},
            {GpuSplitMode.ROW, "row", 4},
            {GpuSplitMode.TENSOR, "tensor", 4},
        });
    }
}
