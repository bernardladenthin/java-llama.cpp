package de.kherud.llama;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify that LlamaOutput correctly stores text, the probability map, stop flag, " +
                  "and stopReason, and that toString() delegates to the text field."
)
public class LlamaOutputTest {

	@Test
	public void testTextFromString() {
		LlamaOutput output = new LlamaOutput("hello", Collections.emptyMap(), false, StopReason.NONE);
		assertEquals("hello", output.text);
	}

	@Test
	public void testEmptyText() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false, StopReason.NONE);
		assertEquals("", output.text);
	}

	@Test
	public void testUtf8MultibyteText() {
		String original = "héllo wörld";
		LlamaOutput output = new LlamaOutput(original, Collections.emptyMap(), false, StopReason.NONE);
		assertEquals(original, output.text);
	}

	@Test
	public void testProbabilitiesStored() {
		Map<String, Float> probs = new HashMap<>();
		probs.put("hello", 0.9f);
		probs.put("world", 0.1f);
		LlamaOutput output = new LlamaOutput("", probs, false, StopReason.NONE);
		assertEquals(2, output.probabilities.size());
		assertEquals(0.9f, output.probabilities.get("hello"), 0.0001f);
		assertEquals(0.1f, output.probabilities.get("world"), 0.0001f);
	}

	@Test
	public void testEmptyProbabilities() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false, StopReason.NONE);
		assertTrue(output.probabilities.isEmpty());
	}

	@Test
	public void testStopFlagFalse() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false, StopReason.NONE);
		assertFalse(output.stop);
	}

	@Test
	public void testStopFlagTrue() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), true, StopReason.EOS);
		assertTrue(output.stop);
	}

	@Test
	public void testToStringReturnsText() {
		LlamaOutput output = new LlamaOutput("generated text", Collections.emptyMap(), false, StopReason.NONE);
		assertEquals("generated text", output.toString());
	}

	@Test
	public void testToStringEmptyText() {
		LlamaOutput output = new LlamaOutput("", Collections.emptyMap(), false, StopReason.NONE);
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

	// --- StopReason tests ---

	@Test
	public void testStopReasonNoneOnIntermediateToken() {
		LlamaOutput output = new LlamaOutput("token", Collections.emptyMap(), false, StopReason.NONE);
		assertEquals(StopReason.NONE, output.stopReason);
	}

	@Test
	public void testStopReasonFromJsonEos() {
		String json = "{\"content\":\"done\",\"stop\":true,\"stop_type\":\"eos\"}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertTrue(output.stop);
		assertEquals(StopReason.EOS, output.stopReason);
	}

	@Test
	public void testStopReasonFromJsonWord() {
		String json = "{\"content\":\"done\",\"stop\":true,\"stop_type\":\"word\",\"stopping_word\":\"END\"}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertTrue(output.stop);
		assertEquals(StopReason.STOP_STRING, output.stopReason);
	}

	@Test
	public void testStopReasonFromJsonLimit() {
		String json = "{\"content\":\"truncated\",\"stop\":true,\"stop_type\":\"limit\",\"truncated\":true}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertTrue(output.stop);
		assertEquals(StopReason.MAX_TOKENS, output.stopReason);
	}

	@Test
	public void testStopReasonNoneWhenStopFalse() {
		String json = "{\"content\":\"partial\",\"stop\":false,\"stop_type\":\"eos\"}";
		LlamaOutput output = LlamaOutput.fromJson(json);
		assertFalse(output.stop);
		// stopReason is NONE for non-final tokens regardless of stop_type
		assertEquals(StopReason.NONE, output.stopReason);
	}
}
