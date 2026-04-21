package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class MiroStatTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, MiroStat.values().length);
    }

    @Test
    public void testDisabled() {
        assertEquals("0", MiroStat.DISABLED.getArgValue());
    }

    @Test
    public void testV1() {
        assertEquals("1", MiroStat.V1.getArgValue());
    }

    @Test
    public void testV2() {
        assertEquals("2", MiroStat.V2.getArgValue());
    }

    @Test
    public void testAllValuesHaveNonEmptyArgValue() {
        for (MiroStat m : MiroStat.values()) {
            assertNotNull(m.getArgValue());
            assertFalse("MiroStat " + m + " has empty argValue", m.getArgValue().isEmpty());
        }
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(MiroStat.DISABLED instanceof CliArg);
    }
}
