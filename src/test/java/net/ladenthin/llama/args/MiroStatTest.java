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
public class MiroStatTest extends AbstractCliArgEnumTest<MiroStat> {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {MiroStat.DISABLED, "0", 3},
            {MiroStat.V1,       "1", 3},
            {MiroStat.V2,       "2", 3},
        });
    }

    public MiroStatTest(MiroStat value, String expectedArgValue, int expectedEnumCount) {
        super(value, expectedArgValue, expectedEnumCount);
    }
}
