// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class NumaStrategyTest extends AbstractCliArgEnumTest<NumaStrategy> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {NumaStrategy.DISTRIBUTE, "distribute", 3},
            {NumaStrategy.ISOLATE,    "isolate",    3},
            {NumaStrategy.NUMACTL,    "numactl",    3},
        });
    }

}
