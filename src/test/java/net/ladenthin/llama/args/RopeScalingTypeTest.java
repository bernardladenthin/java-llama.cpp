// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class RopeScalingTypeTest extends AbstractCliArgEnumTest<RopeScalingType> {

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

}
