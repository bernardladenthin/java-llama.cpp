// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class LazyModeTest extends AbstractCliArgEnumTest<LazyMode> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {LazyMode.OFF, "off", 3},
            {LazyMode.AUTO, "auto", 3},
            {LazyMode.ON, "on", 3},
        });
    }
}
