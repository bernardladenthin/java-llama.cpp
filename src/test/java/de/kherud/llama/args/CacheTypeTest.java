package de.kherud.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class CacheTypeTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {CacheType.F32,    "f32"},
            {CacheType.F16,    "f16"},
            {CacheType.BF16,   "bf16"},
            {CacheType.Q8_0,   "q8_0"},
            {CacheType.Q4_0,   "q4_0"},
            {CacheType.Q4_1,   "q4_1"},
            {CacheType.IQ4_NL, "iq4_nl"},
            {CacheType.Q5_0,   "q5_0"},
            {CacheType.Q5_1,   "q5_1"},
        });
    }

    private final CacheType cacheType;
    private final String expectedArgValue;

    public CacheTypeTest(CacheType cacheType, String expectedArgValue) {
        this.cacheType = cacheType;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, cacheType.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants — tested separately from the per-value check
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(9, CacheType.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(cacheType instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(cacheType.getArgValue());
        assertFalse(cacheType.getArgValue().isEmpty());
    }
}
