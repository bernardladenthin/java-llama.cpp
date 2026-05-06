package net.ladenthin.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class SamplerTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {Sampler.DRY,         "dry"},
            {Sampler.TOP_K,       "top_k"},
            {Sampler.TOP_P,       "top_p"},
            {Sampler.TYP_P,       "typ_p"},
            {Sampler.MIN_P,       "min_p"},
            {Sampler.TEMPERATURE, "temperature"},
            {Sampler.XTC,         "xtc"},
            {Sampler.INFILL,      "infill"},
            {Sampler.PENALTIES,   "penalties"},
        });
    }

    private final Sampler sampler;
    private final String expectedArgValue;

    public SamplerTest(Sampler sampler, String expectedArgValue) {
        this.sampler = sampler;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, sampler.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(9, Sampler.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(sampler instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(sampler.getArgValue());
        assertFalse(sampler.getArgValue().isEmpty());
    }
}
