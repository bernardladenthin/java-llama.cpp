package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify Sampler enum values, count, and lowercase name convention used by " +
                  "ModelParameters.setSamplers() semicolon-separated serialization.",
        model = "claude-opus-4-6"
)
public class SamplerTest {

    @Test
    public void testEnumCount() {
        assertEquals(9, Sampler.values().length);
    }

    @Test
    public void testDry() {
        assertEquals("dry", Sampler.DRY.name().toLowerCase());
    }

    @Test
    public void testTopK() {
        assertEquals("top_k", Sampler.TOP_K.name().toLowerCase());
    }

    @Test
    public void testTopP() {
        assertEquals("top_p", Sampler.TOP_P.name().toLowerCase());
    }

    @Test
    public void testTypP() {
        assertEquals("typ_p", Sampler.TYP_P.name().toLowerCase());
    }

    @Test
    public void testMinP() {
        assertEquals("min_p", Sampler.MIN_P.name().toLowerCase());
    }

    @Test
    public void testTemperature() {
        assertEquals("temperature", Sampler.TEMPERATURE.name().toLowerCase());
    }

    @Test
    public void testXtc() {
        assertEquals("xtc", Sampler.XTC.name().toLowerCase());
    }

    @Test
    public void testInfill() {
        assertEquals("infill", Sampler.INFILL.name().toLowerCase());
    }

    @Test
    public void testPenalties() {
        assertEquals("penalties", Sampler.PENALTIES.name().toLowerCase());
    }

    @Test
    public void testAllValuesHaveNonEmptyLowercaseName() {
        for (Sampler s : Sampler.values()) {
            String lower = s.name().toLowerCase();
            assertNotNull(lower);
            assertFalse("Sampler " + s + " has empty lowercase name", lower.isEmpty());
        }
    }
}
