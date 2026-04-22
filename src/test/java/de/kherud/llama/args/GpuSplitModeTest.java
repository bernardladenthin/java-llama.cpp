package de.kherud.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class GpuSplitModeTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {GpuSplitMode.NONE,  "none"},
            {GpuSplitMode.LAYER, "layer"},
            {GpuSplitMode.ROW,   "row"},
        });
    }

    private final GpuSplitMode gpuSplitMode;
    private final String expectedArgValue;

    public GpuSplitModeTest(GpuSplitMode gpuSplitMode, String expectedArgValue) {
        this.gpuSplitMode = gpuSplitMode;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, gpuSplitMode.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(3, GpuSplitMode.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(gpuSplitMode instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(gpuSplitMode.getArgValue());
        assertFalse(gpuSplitMode.getArgValue().isEmpty());
    }
}
