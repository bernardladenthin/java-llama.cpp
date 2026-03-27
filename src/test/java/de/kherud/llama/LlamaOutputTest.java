package de.kherud.llama;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import static org.junit.Assert.*;

public class LlamaOutputTest {

	@Test
	public void testTextFromBytes() {
		byte[] bytes = "hello".getBytes(StandardCharsets.UTF_8);
		LlamaOutput output = new LlamaOutput(bytes, Collections.emptyMap(), false);
		assertEquals("hello", output.text);
	}

	@Test
	public void testEmptyText() {
		LlamaOutput output = new LlamaOutput(new byte[0], Collections.emptyMap(), false);
		assertEquals("", output.text);
	}

	@Test
	public void testUtf8MultibyteText() {
		String original = "héllo wörld";
		byte[] bytes = original.getBytes(StandardCharsets.UTF_8);
		LlamaOutput output = new LlamaOutput(bytes, Collections.emptyMap(), false);
		assertEquals(original, output.text);
	}

	@Test
	public void testProbabilitiesStored() {
		Map<String, Float> probs = new HashMap<>();
		probs.put("hello", 0.9f);
		probs.put("world", 0.1f);
		LlamaOutput output = new LlamaOutput(new byte[0], probs, false);
		assertEquals(2, output.probabilities.size());
		assertEquals(0.9f, output.probabilities.get("hello"), 0.0001f);
		assertEquals(0.1f, output.probabilities.get("world"), 0.0001f);
	}

	@Test
	public void testEmptyProbabilities() {
		LlamaOutput output = new LlamaOutput(new byte[0], Collections.emptyMap(), false);
		assertTrue(output.probabilities.isEmpty());
	}

	@Test
	public void testStopFlagFalse() {
		LlamaOutput output = new LlamaOutput(new byte[0], Collections.emptyMap(), false);
		assertFalse(output.stop);
	}

	@Test
	public void testStopFlagTrue() {
		LlamaOutput output = new LlamaOutput(new byte[0], Collections.emptyMap(), true);
		assertTrue(output.stop);
	}

	@Test
	public void testToStringReturnsText() {
		byte[] bytes = "generated text".getBytes(StandardCharsets.UTF_8);
		LlamaOutput output = new LlamaOutput(bytes, Collections.emptyMap(), false);
		assertEquals("generated text", output.toString());
	}

	@Test
	public void testToStringEmptyText() {
		LlamaOutput output = new LlamaOutput(new byte[0], Collections.emptyMap(), false);
		assertEquals("", output.toString());
	}
}
