package net.ladenthin.llama;

import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify LogLevel enum values, count, and ordinal order matching llama.cpp native log levels.",
        model = "claude-opus-4-6"
)
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
    public void testOrdinalOrder() {
        // Log levels must be ordered from least to most severe
        assertTrue(LogLevel.DEBUG.ordinal() < LogLevel.INFO.ordinal());
        assertTrue(LogLevel.INFO.ordinal() < LogLevel.WARN.ordinal());
        assertTrue(LogLevel.WARN.ordinal() < LogLevel.ERROR.ordinal());
    }

    @Test
    public void testValueOf() {
        assertSame(LogLevel.DEBUG, LogLevel.valueOf("DEBUG"));
        assertSame(LogLevel.INFO, LogLevel.valueOf("INFO"));
        assertSame(LogLevel.WARN, LogLevel.valueOf("WARN"));
        assertSame(LogLevel.ERROR, LogLevel.valueOf("ERROR"));
    }
}
