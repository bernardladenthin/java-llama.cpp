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
public class RopeScalingTypeTest extends AbstractCliArgEnumTest<RopeScalingType> {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {RopeScalingType.UNSPECIFIED, "unspecified", 6},
            {RopeScalingType.NONE,        "none",        6},
            {RopeScalingType.LINEAR,      "linear",      6},
            {RopeScalingType.YARN2,       "yarn",        6},
            {RopeScalingType.LONGROPE,    "longrope",    6},
            {RopeScalingType.MAX_VALUE,   "maxvalue",    6},
        });
    }

    public RopeScalingTypeTest(RopeScalingType value, String expectedArgValue, int expectedEnumCount) {
        super(value, expectedArgValue, expectedEnumCount);
    }
}
