// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class PoolingTypeTest extends AbstractCliArgEnumTest<PoolingType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {PoolingType.UNSPECIFIED, "unspecified", 6},
            {PoolingType.NONE,        "none",        6},
            {PoolingType.MEAN,        "mean",        6},
            {PoolingType.CLS,         "cls",         6},
            {PoolingType.LAST,        "last",        6},
            {PoolingType.RANK,        "rank",        6},
        });
    }

}
