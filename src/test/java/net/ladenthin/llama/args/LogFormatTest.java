// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@ClaudeGenerated(
        purpose = "Verify LogFormat enum values and count.",
        model = "claude-opus-4-6"
)
public class LogFormatTest {

    @Test
    public void testEnumCount() {
        assertEquals(2, LogFormat.values().length);
    }

    @Test
    public void testJson() {
        assertEquals("JSON", LogFormat.JSON.name());
    }

    @Test
    public void testText() {
        assertEquals("TEXT", LogFormat.TEXT.name());
    }

    @Test
    public void testValueOf() {
        assertSame(LogFormat.valueOf("JSON"), LogFormat.JSON);
        assertSame(LogFormat.valueOf("TEXT"), LogFormat.TEXT);
    }
}
