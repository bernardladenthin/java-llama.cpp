package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify NumaStrategy enum values, count, and lowercase name convention used by ModelParameters.",
        model = "claude-opus-4-6"
)
public class NumaStrategyTest {

    @Test
    public void testEnumCount() {
        assertEquals(3, NumaStrategy.values().length);
    }

    @Test
    public void testDistribute() {
        assertEquals("distribute", NumaStrategy.DISTRIBUTE.name().toLowerCase());
    }

    @Test
    public void testIsolate() {
        assertEquals("isolate", NumaStrategy.ISOLATE.name().toLowerCase());
    }

    @Test
    public void testNumactl() {
        assertEquals("numactl", NumaStrategy.NUMACTL.name().toLowerCase());
    }

    @Test
    public void testAllValuesHaveNonEmptyLowercaseName() {
        for (NumaStrategy ns : NumaStrategy.values()) {
            String lower = ns.name().toLowerCase();
            assertNotNull(lower);
            assertFalse("NumaStrategy " + ns + " has empty lowercase name", lower.isEmpty());
        }
    }
}
