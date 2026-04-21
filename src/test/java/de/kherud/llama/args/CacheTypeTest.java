package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class CacheTypeTest {

    @Test
    public void testEnumCount() {
        assertEquals(9, CacheType.values().length);
    }

    @Test
    public void testF32() {
        assertEquals("f32", CacheType.F32.getArgValue());
    }

    @Test
    public void testF16() {
        assertEquals("f16", CacheType.F16.getArgValue());
    }

    @Test
    public void testBF16() {
        assertEquals("bf16", CacheType.BF16.getArgValue());
    }

    @Test
    public void testQ8_0() {
        assertEquals("q8_0", CacheType.Q8_0.getArgValue());
    }

    @Test
    public void testQ4_0() {
        assertEquals("q4_0", CacheType.Q4_0.getArgValue());
    }

    @Test
    public void testQ4_1() {
        assertEquals("q4_1", CacheType.Q4_1.getArgValue());
    }

    @Test
    public void testIQ4_NL() {
        assertEquals("iq4_nl", CacheType.IQ4_NL.getArgValue());
    }

    @Test
    public void testQ5_0() {
        assertEquals("q5_0", CacheType.Q5_0.getArgValue());
    }

    @Test
    public void testQ5_1() {
        assertEquals("q5_1", CacheType.Q5_1.getArgValue());
    }

    @Test
    public void testAllValuesHaveNonEmptyArgValue() {
        for (CacheType ct : CacheType.values()) {
            assertNotNull(ct.getArgValue());
            assertFalse("CacheType " + ct + " has empty argValue", ct.getArgValue().isEmpty());
        }
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(CacheType.F16 instanceof CliArg);
    }
}
