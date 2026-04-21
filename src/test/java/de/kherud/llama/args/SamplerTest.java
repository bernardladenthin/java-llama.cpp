package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class SamplerTest {

    @Test
    public void testEnumCount() {
        assertEquals(9, Sampler.values().length);
    }

    @Test
    public void testDry() {
        assertEquals("dry", Sampler.DRY.getArgValue());
    }

    @Test
    public void testTopK() {
        assertEquals("top_k", Sampler.TOP_K.getArgValue());
    }

    @Test
    public void testTopP() {
        assertEquals("top_p", Sampler.TOP_P.getArgValue());
    }

    @Test
    public void testTypP() {
        assertEquals("typ_p", Sampler.TYP_P.getArgValue());
    }

    @Test
    public void testMinP() {
        assertEquals("min_p", Sampler.MIN_P.getArgValue());
    }

    @Test
    public void testTemperature() {
        assertEquals("temperature", Sampler.TEMPERATURE.getArgValue());
    }

    @Test
    public void testXtc() {
        assertEquals("xtc", Sampler.XTC.getArgValue());
    }

    @Test
    public void testInfill() {
        assertEquals("infill", Sampler.INFILL.getArgValue());
    }

    @Test
    public void testPenalties() {
        assertEquals("penalties", Sampler.PENALTIES.getArgValue());
    }

    @Test
    public void testAllValuesHaveNonEmptyArgValue() {
        for (Sampler s : Sampler.values()) {
            assertNotNull(s.getArgValue());
            assertFalse("Sampler " + s + " has empty argValue", s.getArgValue().isEmpty());
        }
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(Sampler.TOP_K instanceof CliArg);
    }
}
