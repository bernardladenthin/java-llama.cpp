// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class TensorReadLazyModeTest extends AbstractCliArgEnumTest<TensorReadLazyMode> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {TensorReadLazyMode.OFF, "off", 3},
            {TensorReadLazyMode.AUTO, "auto", 3},
            {TensorReadLazyMode.ON, "on", 3},
        });
    }
}
