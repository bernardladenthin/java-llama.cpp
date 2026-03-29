package de.kherud.llama.args;

import de.kherud.llama.ClaudeGenerated;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify that every RopeScalingType enum constant returns the exact CLI argument " +
                  "string expected by llama.cpp (e.g. YARN2 -> \"yarn\", LONGROPE -> \"longrope\") " +
                  "via getArgValue(), and that the enum has the expected number of constants."
)
public class RopeScalingTypeTest {

	@Test
	public void testUnspecifiedArgValue() {
		assertEquals("unspecified", RopeScalingType.UNSPECIFIED.getArgValue());
	}

	@Test
	public void testNoneArgValue() {
		assertEquals("none", RopeScalingType.NONE.getArgValue());
	}

	@Test
	public void testLinearArgValue() {
		assertEquals("linear", RopeScalingType.LINEAR.getArgValue());
	}

	@Test
	public void testYarn2ArgValue() {
		assertEquals("yarn", RopeScalingType.YARN2.getArgValue());
	}

	@Test
	public void testLongRopeArgValue() {
		assertEquals("longrope", RopeScalingType.LONGROPE.getArgValue());
	}

	@Test
	public void testMaxValueArgValue() {
		assertEquals("maxvalue", RopeScalingType.MAX_VALUE.getArgValue());
	}

	@Test
	public void testAllValuesHaveArgValue() {
		for (RopeScalingType type : RopeScalingType.values()) {
			assertNotNull("getArgValue() should not be null for " + type, type.getArgValue());
			assertFalse("getArgValue() should not be empty for " + type, type.getArgValue().isEmpty());
		}
	}

	@Test
	public void testEnumCount() {
		assertEquals(6, RopeScalingType.values().length);
	}
}
