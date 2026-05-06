package net.ladenthin.llama.args;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

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
        assertSame(LogFormat.JSON, LogFormat.valueOf("JSON"));
        assertSame(LogFormat.TEXT, LogFormat.valueOf("TEXT"));
    }
}
