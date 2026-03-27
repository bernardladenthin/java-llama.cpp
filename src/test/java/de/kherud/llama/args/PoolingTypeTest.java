package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class PoolingTypeTest {

	@Test
	public void testUnspecifiedArgValue() {
		assertEquals("unspecified", PoolingType.UNSPECIFIED.getArgValue());
	}

	@Test
	public void testNoneArgValue() {
		assertEquals("none", PoolingType.NONE.getArgValue());
	}

	@Test
	public void testMeanArgValue() {
		assertEquals("mean", PoolingType.MEAN.getArgValue());
	}

	@Test
	public void testClsArgValue() {
		assertEquals("cls", PoolingType.CLS.getArgValue());
	}

	@Test
	public void testLastArgValue() {
		assertEquals("last", PoolingType.LAST.getArgValue());
	}

	@Test
	public void testRankArgValue() {
		assertEquals("rank", PoolingType.RANK.getArgValue());
	}

	@Test
	public void testAllValuesHaveArgValue() {
		for (PoolingType type : PoolingType.values()) {
			assertNotNull("getArgValue() should not be null for " + type, type.getArgValue());
			assertFalse("getArgValue() should not be empty for " + type, type.getArgValue().isEmpty());
		}
	}

	@Test
	public void testEnumCount() {
		assertEquals(6, PoolingType.values().length);
	}
}
