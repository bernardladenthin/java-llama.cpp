package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class GpuSplitModeTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, GpuSplitMode.values().length);
    }

    @Test
    public void testNone() {
        assertEquals("none", GpuSplitMode.NONE.getArgValue());
    }

    @Test
    public void testLayer() {
        assertEquals("layer", GpuSplitMode.LAYER.getArgValue());
    }

    @Test
    public void testRow() {
        assertEquals("row", GpuSplitMode.ROW.getArgValue());
    }

    @Test
    public void testAllValuesHaveNonEmptyArgValue() {
        for (GpuSplitMode mode : GpuSplitMode.values()) {
            assertNotNull(mode.getArgValue());
            assertFalse("GpuSplitMode " + mode + " has empty argValue", mode.getArgValue().isEmpty());
        }
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(GpuSplitMode.LAYER instanceof CliArg);
    }
}
