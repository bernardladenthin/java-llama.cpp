package de.kherud.llama;

import java.nio.charset.StandardCharsets;
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

	// -----------------------------------------------------------------------
	// fromBytes() — the fast path that bypasses JSON serialization
	// -----------------------------------------------------------------------

	/** Helper: build the byte[] that receiveCompletionBytes returns. */
	private static byte[] makeResultBytes(boolean stop, String content) {
		byte[] contentBytes = content.getBytes(StandardCharsets.UTF_8);
		byte[] result = new byte[1 + contentBytes.length];
		result[0] = stop ? (byte) 1 : (byte) 0;
		System.arraycopy(contentBytes, 0, result, 1, contentBytes.length);
		return result;
	}

	@Test
	public void testFromBytesNonStop() {
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(false, "hello"));
		assertEquals("hello", output.text);
		assertFalse(output.stop);
		assertTrue(output.probabilities.isEmpty());
	}

	@Test
	public void testFromBytesStop() {
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(true, "world"));
		assertEquals("world", output.text);
		assertTrue(output.stop);
	}

	@Test
	public void testFromBytesEmptyContent() {
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(true, ""));
		assertEquals("", output.text);
		assertTrue(output.stop);
	}

	@Test
	public void testFromBytesOnlyStopByte() {
		// Array of length 1 — only the stop flag, no content bytes
		byte[] bytes = { 1 };
		LlamaOutput output = LlamaOutput.fromBytes(bytes);
		assertEquals("", output.text);
		assertTrue(output.stop);
	}

	@Test
	public void testFromBytesMultibyteUtf8TwoByteSeq() {
		// 2-byte UTF-8: ü = U+00FC (0xC3 0xBC)
		String original = "über";
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(false, original));
		assertEquals(original, output.text);
	}

	@Test
	public void testFromBytesMultibyteUtf8ThreeByteSeq() {
		// 3-byte UTF-8: CJK ideographs
		String original = "日本語";
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(false, original));
		assertEquals(original, output.text);
	}

	@Test
	public void testFromBytesMixedAsciiAndMultibyte() {
		String original = "résumé 日本語 über";
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(false, original));
		assertEquals(original, output.text);
	}

	@Test
	public void testFromBytesWithEscapeCharacters() {
		// Content with newlines, tabs, quotes — should pass through unchanged
		// (unlike fromJson, no escaping is needed because we send raw UTF-8 bytes)
		String original = "line1\nline2\t\"quoted\"\\backslash";
		LlamaOutput output = LlamaOutput.fromBytes(makeResultBytes(false, original));
		assertEquals(original, output.text);
	}

	@Test
	public void testFromBytesStopFlagIsZeroForNonStop() {
		byte[] bytes = makeResultBytes(false, "token");
		assertEquals(0, bytes[0]);
	}

	@Test
	public void testFromBytesStopFlagIsOneForStop() {
		byte[] bytes = makeResultBytes(true, "token");
		assertEquals(1, bytes[0]);
	}

	/**
	 * Regression: fromBytes must produce the same text as fromJson for the
	 * content field, confirming that the fast path is equivalent to the JSON path
	 * for typical ASCII content.
	 */
	@Test
	public void testFromBytesEquivalentToFromJsonForAscii() {
		String content = "hello world";
		LlamaOutput fromJson = LlamaOutput.fromJson("{\"content\":\"hello world\",\"stop\":false}");
		LlamaOutput fromBytes = LlamaOutput.fromBytes(makeResultBytes(false, content));
		assertEquals(fromJson.text, fromBytes.text);
		assertEquals(fromJson.stop, fromBytes.stop);
	}
}
