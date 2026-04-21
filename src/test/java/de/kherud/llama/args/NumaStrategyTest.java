package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class NumaStrategyTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, NumaStrategy.values().length);
    }

    @Test
    public void testDistribute() {
        assertEquals("distribute", NumaStrategy.DISTRIBUTE.getArgValue());
    }

    @Test
    public void testIsolate() {
        assertEquals("isolate", NumaStrategy.ISOLATE.getArgValue());
    }

    @Test
    public void testNumactl() {
        assertEquals("numactl", NumaStrategy.NUMACTL.getArgValue());
    }

    @Test
    public void testAllValuesHaveNonEmptyArgValue() {
        for (NumaStrategy ns : NumaStrategy.values()) {
            assertNotNull(ns.getArgValue());
            assertFalse("NumaStrategy " + ns + " has empty argValue", ns.getArgValue().isEmpty());
        }
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(NumaStrategy.DISTRIBUTE instanceof CliArg);
    }
}
