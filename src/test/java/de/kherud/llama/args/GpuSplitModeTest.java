package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify GpuSplitMode enum values, count, and lowercase name convention used by ModelParameters.",
        model = "claude-opus-4-6"
)
public class GpuSplitModeTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, GpuSplitMode.values().length);
    }

    @Test
    public void testNone() {
        assertEquals("none", GpuSplitMode.NONE.name().toLowerCase());
    }

    @Test
    public void testLayer() {
        assertEquals("layer", GpuSplitMode.LAYER.name().toLowerCase());
    }

    @Test
    public void testRow() {
        assertEquals("row", GpuSplitMode.ROW.name().toLowerCase());
    }

    @Test
    public void testAllValuesHaveNonEmptyLowercaseName() {
        for (GpuSplitMode mode : GpuSplitMode.values()) {
            String lower = mode.name().toLowerCase();
            assertNotNull(lower);
            assertFalse("GpuSplitMode " + mode + " has empty lowercase name", lower.isEmpty());
        }
    }
}
