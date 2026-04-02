package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify CacheType enum values, count, and lowercase name convention used by ModelParameters.",
        model = "claude-opus-4-6"
)
public class CacheTypeTest {

    @Test
    public void testEnumCount() {
        assertEquals(9, CacheType.values().length);
    }

    @Test
    public void testF32() {
        assertEquals("f32", CacheType.F32.name().toLowerCase());
    }

    @Test
    public void testF16() {
        assertEquals("f16", CacheType.F16.name().toLowerCase());
    }

    @Test
    public void testBF16() {
        assertEquals("bf16", CacheType.BF16.name().toLowerCase());
    }

    @Test
    public void testQ8_0() {
        assertEquals("q8_0", CacheType.Q8_0.name().toLowerCase());
    }

    @Test
    public void testQ4_0() {
        assertEquals("q4_0", CacheType.Q4_0.name().toLowerCase());
    }

    @Test
    public void testQ4_1() {
        assertEquals("q4_1", CacheType.Q4_1.name().toLowerCase());
    }

    @Test
    public void testIQ4_NL() {
        assertEquals("iq4_nl", CacheType.IQ4_NL.name().toLowerCase());
    }

    @Test
    public void testQ5_0() {
        assertEquals("q5_0", CacheType.Q5_0.name().toLowerCase());
    }

    @Test
    public void testQ5_1() {
        assertEquals("q5_1", CacheType.Q5_1.name().toLowerCase());
    }

    @Test
    public void testAllValuesHaveNonEmptyLowercaseName() {
        for (CacheType ct : CacheType.values()) {
            String lower = ct.name().toLowerCase();
            assertNotNull(lower);
            assertFalse("CacheType " + ct + " has empty lowercase name", lower.isEmpty());
        }
    }
}
