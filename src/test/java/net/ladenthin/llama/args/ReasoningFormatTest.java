// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

public class ReasoningFormatTest extends AbstractCliArgEnumTest<ReasoningFormat> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {ReasoningFormat.NONE,            "none",            4},
            {ReasoningFormat.AUTO,            "auto",            4},
            {ReasoningFormat.DEEPSEEK,        "deepseek",        4},
            {ReasoningFormat.DEEPSEEK_LEGACY, "deepseek-legacy", 4},
        });
    }

}
