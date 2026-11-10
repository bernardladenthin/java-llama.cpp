// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class PoolingTypeTest extends AbstractCliArgEnumTest<PoolingType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {PoolingType.NONE, "none", 5},
            {PoolingType.MEAN, "mean", 5},
            {PoolingType.CLS, "cls", 5},
            {PoolingType.LAST, "last", 5},
            {PoolingType.RANK, "rank", 5},
        });
    }
}
