// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class RopeScalingTypeTest extends AbstractCliArgEnumTest<RopeScalingType> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {RopeScalingType.NONE, "none", 3},
            {RopeScalingType.LINEAR, "linear", 3},
            {RopeScalingType.YARN, "yarn", 3},
        });
    }
}
