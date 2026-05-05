package net.ladenthin.llama;

import org.junit.Test;

import static org.junit.Assert.*;

public class PairTest {

	@Test
	public void testGetKey() {
		Pair<String, Integer> pair = new Pair<>("key1", 42);
		assertEquals("key1", pair.getKey());
	}

	@Test
	public void testGetValue() {
		Pair<String, Integer> pair = new Pair<>("key1", 42);
		assertEquals(Integer.valueOf(42), pair.getValue());
	}

	@Test
	public void testGetKeyWithNullValue() {
		Pair<String, Integer> pair = new Pair<>("key1", null);
		assertEquals("key1", pair.getKey());
		assertNull(pair.getValue());
	}

	@Test
	public void testGetKeyWithNullKey() {
		Pair<String, Integer> pair = new Pair<>(null, 42);
		assertNull(pair.getKey());
		assertEquals(Integer.valueOf(42), pair.getValue());
	}

	@Test
	public void testEqualsWithSamePair() {
		Pair<String, Integer> pair1 = new Pair<>("key", 123);
		Pair<String, Integer> pair2 = new Pair<>("key", 123);
		assertEquals(pair1, pair2);
	}

	@Test
	public void testEqualsWithDifferentKey() {
		Pair<String, Integer> pair1 = new Pair<>("key1", 123);
		Pair<String, Integer> pair2 = new Pair<>("key2", 123);
		assertNotEquals(pair1, pair2);
	}

	@Test
	public void testEqualsWithDifferentValue() {
		Pair<String, Integer> pair1 = new Pair<>("key", 123);
		Pair<String, Integer> pair2 = new Pair<>("key", 456);
		assertNotEquals(pair1, pair2);
	}

	@Test
	public void testEqualsWithNull() {
		Pair<String, Integer> pair = new Pair<>("key", 123);
		assertNotEquals(pair, null);
	}

	@Test
	public void testEqualsWithDifferentClass() {
		Pair<String, Integer> pair = new Pair<>("key", 123);
		assertNotEquals(pair, "not a pair");
	}

	@Test
	public void testEqualsSameInstance() {
		Pair<String, Integer> pair = new Pair<>("key", 123);
		assertEquals(pair, pair);
	}

	@Test
	public void testEqualsWithBothNullKeyAndValue() {
		Pair<String, Integer> pair1 = new Pair<>(null, null);
		Pair<String, Integer> pair2 = new Pair<>(null, null);
		assertEquals(pair1, pair2);
	}

	@Test
	public void testHashCodeSamePair() {
		Pair<String, Integer> pair1 = new Pair<>("key", 123);
		Pair<String, Integer> pair2 = new Pair<>("key", 123);
		assertEquals(pair1.hashCode(), pair2.hashCode());
	}

	@Test
	public void testHashCodeDifferentPairs() {
		Pair<String, Integer> pair1 = new Pair<>("key1", 123);
		Pair<String, Integer> pair2 = new Pair<>("key2", 456);
		// Different pairs may have different hash codes (not guaranteed, but likely)
		// We mostly check that hashCode() doesn't throw
		assertNotNull(pair1.hashCode());
		assertNotNull(pair2.hashCode());
	}

	@Test
	public void testHashCodeWithNull() {
		Pair<String, Integer> pair = new Pair<>(null, null);
		// Should not throw when hashing null values
		assertNotNull(pair.hashCode());
	}

	@Test
	public void testToString() {
		Pair<String, Integer> pair = new Pair<>("testKey", 42);
		String result = pair.toString();
		assertTrue(result.contains("Pair"));
		assertTrue(result.contains("testKey"));
		assertTrue(result.contains("42"));
	}

	@Test
	public void testToStringWithNull() {
		Pair<String, Integer> pair = new Pair<>(null, 42);
		String result = pair.toString();
		assertTrue(result.contains("Pair"));
		assertTrue(result.contains("null"));
	}

	@Test
	public void testPairWithDifferentTypes() {
		Pair<Integer, Double> pair = new Pair<>(10, 3.14);
		assertEquals(Integer.valueOf(10), pair.getKey());
		assertEquals(Double.valueOf(3.14), pair.getValue());
	}

	@Test
	public void testPairWithComplexTypes() {
		Pair<String[], Integer[]> pair = new Pair<>(
			new String[]{"a", "b"},
			new Integer[]{1, 2}
		);
		assertArrayEquals(new String[]{"a", "b"}, pair.getKey());
		assertArrayEquals(new Integer[]{1, 2}, pair.getValue());
	}
}
