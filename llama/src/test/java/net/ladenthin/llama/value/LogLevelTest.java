// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.junit.jupiter.api.Assertions.*;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify LogLevel enum values, count, and ordinal order matching llama.cpp native log levels.",
        model = "claude-opus-4-6")
public class LogLevelTest {

    @Test
    public void testEnumCount() {
        assertEquals(4, LogLevel.values().length);
    }

    @Test
    public void testDebug() {
        assertEquals("DEBUG", LogLevel.DEBUG.name());
    }

    @Test
    public void testInfo() {
        assertEquals("INFO", LogLevel.INFO.name());
    }

    @Test
    public void testWarn() {
        assertEquals("WARN", LogLevel.WARN.name());
    }

    @Test
    public void testError() {
        assertEquals("ERROR", LogLevel.ERROR.name());
    }

    @Test
    public void testDeclarationOrder() {
        // Declared from least to most severe; the order is part of the contract
        // (mirrors llama.cpp's native log-level severity). values() returns the
        // constants in declaration order, so this pins the full order without
        // depending on Enum.ordinal() (Error Prone EnumOrdinal).
        assertArrayEquals(
                new LogLevel[] {LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR}, LogLevel.values());
    }

    @Test
    public void testValueOf() {
        assertSame(LogLevel.valueOf("DEBUG"), LogLevel.DEBUG);
        assertSame(LogLevel.valueOf("INFO"), LogLevel.INFO);
        assertSame(LogLevel.valueOf("WARN"), LogLevel.WARN);
        assertSame(LogLevel.valueOf("ERROR"), LogLevel.ERROR);
    }
}
