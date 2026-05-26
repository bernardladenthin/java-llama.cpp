// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.assertNotEquals;

@ClaudeGenerated(
        purpose = "Verify Usage records prompt/completion totals correctly and derives totalTokens."
)
public class UsageTest {

    @Test
    public void totalTokensIsSum() {
        Usage u = new Usage(10, 7);
        assertEquals(10, u.getPromptTokens());
        assertEquals(7, u.getCompletionTokens());
        assertEquals(17, u.getTotalTokens());
    }

    @Test
    public void zeroIsZero() {
        Usage u = new Usage(0, 0);
        assertEquals(0, u.getTotalTokens());
    }

    @Test
    public void equalsAndHashCode() {
        assertEquals(new Usage(3, 4), new Usage(3, 4));
        assertEquals(new Usage(3, 4).hashCode(), new Usage(3, 4).hashCode());
        assertNotEquals(new Usage(3, 4), new Usage(4, 3));
    }
}
