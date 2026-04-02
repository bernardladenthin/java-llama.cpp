package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify MiroStat enum values and count.",
        model = "claude-opus-4-6"
)
public class MiroStatTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, MiroStat.values().length);
    }

    @Test
    public void testDisabledOrdinal() {
        assertEquals(0, MiroStat.DISABLED.ordinal());
    }

    @Test
    public void testV1Ordinal() {
        assertEquals(1, MiroStat.V1.ordinal());
    }

    @Test
    public void testV2Ordinal() {
        assertEquals(2, MiroStat.V2.ordinal());
    }

    @Test
    public void testValueOf() {
        assertEquals(MiroStat.DISABLED, MiroStat.valueOf("DISABLED"));
        assertEquals(MiroStat.V1, MiroStat.valueOf("V1"));
        assertEquals(MiroStat.V2, MiroStat.valueOf("V2"));
    }
}
