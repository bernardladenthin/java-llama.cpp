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
public class PoolingTypeTest extends AbstractCliArgEnumTest<PoolingType> {

    @Parameterized.Parameters(name = "{0} -> {1}")
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

    public PoolingTypeTest(PoolingType value, String expectedArgValue, int expectedEnumCount) {
        super(value, expectedArgValue, expectedEnumCount);
    }
}
