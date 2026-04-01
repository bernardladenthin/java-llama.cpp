package de.kherud.llama;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify that LlamaOutput correctly stores text, the probability map and stop flag " +
                  "unchanged, and that toString() delegates to the text field."
)
public class LlamaOutputTest {

	@Test
	public void testTextFromString() {
		LlamaOutput output = new LlamaOutput("hello", Collections.emptyMap(), false);
		assertEquals("hello", output.text);
	}

	@Test
	public void testEmptyText() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false);
		assertEquals("", output.text);
	}

	@Test
	public void testUtf8MultibyteText() {
		String original = "héllo wörld";
		LlamaOutput output = new LlamaOutput(original, Collections.emptyMap(), false);
		assertEquals(original, output.text);
	}

	@Test
	public void testProbabilitiesStored() {
		Map<String, Float> probs = new HashMap<>();
		probs.put("hello", 0.9f);
		probs.put("world", 0.1f);
		LlamaOutput output = new LlamaOutput("", probs, false);
		assertEquals(2, output.probabilities.size());
		assertEquals(0.9f, output.probabilities.get("hello"), 0.0001f);
		assertEquals(0.1f, output.probabilities.get("world"), 0.0001f);
	}

	@Test
	public void testEmptyProbabilities() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false);
		assertTrue(output.probabilities.isEmpty());
	}

	@Test
	public void testStopFlagFalse() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false);
		assertFalse(output.stop);
	}

	@Test
	public void testStopFlagTrue() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), true);
		assertTrue(output.stop);
	}

	@Test
	public void testToStringReturnsText() {
		LlamaOutput output = new LlamaOutput("generated text", Collections.emptyMap(), false);
		assertEquals("generated text", output.toString());
	}

	@Test
	public void testToStringEmptyText() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false);
		assertEquals("", output.toString());
	}

	@Test
	public void testFromJson() {
		String json = "{\"content\":\"hello world\",\"stop\":true}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertEquals("hello world", output.text);
		assertTrue(output.stop);
	}

	@Test
	public void testFromJsonWithEscapes() {
		String json = "{\"content\":\"line1\\nline2\\t\\\"quoted\\\"\",\"stop\":false}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertEquals("line1\nline2\t\"quoted\"", output.text);
		assertFalse(output.stop);
	}

	@Test
	public void testGetContentFromJsonEmpty() {
		String json = "{\"content\":\"\",\"stop\":true}";
		assertEquals("", LlamaOutput.getContentFromJson(json));
	}
}
