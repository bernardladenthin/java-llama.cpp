// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import java.util.Arrays;
import java.util.Collection;

/**
 * The three {@code --flash-attn} values, pinned to the exact tokens llama.cpp's parser accepts.
 *
 * <p>These strings are a wire contract, not labels: since llama.cpp b10273 the option takes a
 * mandatory value, so a wrong or empty token is not a cosmetic defect — the parser consumes the
 * following argv entry and the model load fails naming a flag the caller never set.</p>
 */
public class FlashAttnTest extends AbstractCliArgEnumTest<FlashAttn> {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {FlashAttn.ON, "on", 3},
            {FlashAttn.OFF, "off", 3},
            {FlashAttn.AUTO, "auto", 3},
        });
    }
}
