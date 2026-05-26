// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class MiroStatTest extends AbstractCliArgEnumTest<MiroStat> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {MiroStat.DISABLED, "0", 3},
            {MiroStat.V1,       "1", 3},
            {MiroStat.V2,       "2", 3},
        });
    }

}
