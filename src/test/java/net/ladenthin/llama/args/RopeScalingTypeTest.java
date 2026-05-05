package net.ladenthin.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class RopeScalingTypeTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {RopeScalingType.UNSPECIFIED, "unspecified"},
            {RopeScalingType.NONE,        "none"},
            {RopeScalingType.LINEAR,      "linear"},
            {RopeScalingType.YARN2,       "yarn"},
            {RopeScalingType.LONGROPE,    "longrope"},
            {RopeScalingType.MAX_VALUE,   "maxvalue"},
        });
    }

    private final RopeScalingType ropeScalingType;
    private final String expectedArgValue;

    public RopeScalingTypeTest(RopeScalingType ropeScalingType, String expectedArgValue) {
        this.ropeScalingType = ropeScalingType;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, ropeScalingType.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(6, RopeScalingType.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(ropeScalingType instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(ropeScalingType.getArgValue());
        assertFalse(ropeScalingType.getArgValue().isEmpty());
    }
}
