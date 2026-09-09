// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class LoadModeTest extends AbstractCliArgEnumTest<LoadMode> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {LoadMode.AUTO, "auto", 6},
            {LoadMode.NONE, "none", 6},
            {LoadMode.MMAP, "mmap", 6},
            {LoadMode.MLOCK, "mlock", 6},
            {LoadMode.MMAP_MLOCK, "mmap+mlock", 6},
            {LoadMode.DIRECT_IO, "dio", 6},
        });
    }
}
