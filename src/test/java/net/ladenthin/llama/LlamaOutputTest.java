package net.ladenthin.llama;

import net.ladenthin.llama.json.CompletionResponseParser;
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

	private final CompletionResponseParser parser = new CompletionResponseParser();

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
		LlamaOutput output = parser.parse(json);
		assertEquals("hello world", output.text);
		assertTrue(output.stop);
	}

	@Test
	public void testFromJsonWithEscapes() {
		String json = "{\"content\":\"line1\\nline2\\t\\\"quoted\\\"\",\"stop\":false}";
		LlamaOutput output = parser.parse(json);
		assertEquals("line1\nline2\t\"quoted\"", output.text);
		assertFalse(output.stop);
	}

	@Test
	public void testFromJsonWithUnicodeEscape() {
		String json = "{\"content\":\"caf\\u00e9\",\"stop\":false}";
		LlamaOutput output = parser.parse(json);
		assertEquals("café", output.text);
		assertFalse(output.stop);
	}

	@Test
	public void testFromJsonMalformedReturnsEmptyNonStop() {
		LlamaOutput output = parser.parse("{not valid json");
		assertEquals("", output.text);
		assertFalse(output.stop);
		assertEquals(StopReason.NONE, output.stopReason);
		assertTrue(output.probabilities.isEmpty());
	}

	@Test
	public void testGetContentFromJsonEmpty() {
		String json = "{\"content\":\"\",\"stop\":true}";
		assertEquals("", parser.parse(json).text);
	}

	// --- parseProbabilities tests ---

	@Test
	public void testProbabilitiesAbsentWhenNoProbsKey() {
		String json = "{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"}";
		LlamaOutput output = parser.parse(json);
		assertTrue("No completion_probabilities key → empty map", output.probabilities.isEmpty());
	}

	@Test
	public void testProbabilitiesParsedPostSampling() {
		// post_sampling_probs=true → "prob" key
		String json = "{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"," +
				"\"completion_probabilities\":[" +
				"{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"prob\":0.82," +
				"\"top_probs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"prob\":0.1}]}," +
				"{\"token\":\" world\",\"bytes\":[32,119],\"id\":1917,\"prob\":0.65," +
				"\"top_probs\":[{\"token\":\" World\",\"bytes\":[32,87],\"id\":2304,\"prob\":0.2}]}" +
				"]}";
		LlamaOutput output = parser.parse(json);
		assertEquals(2, output.probabilities.size());
		assertEquals(0.82f, output.probabilities.get("Hello"), 0.001f);
		assertEquals(0.65f, output.probabilities.get(" world"), 0.001f);
	}

	@Test
	public void testProbabilitiesParsedPreSampling() {
		// post_sampling_probs=false → "logprob" key
		String json = "{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"," +
				"\"completion_probabilities\":[" +
				"{\"token\":\"Hello\",\"bytes\":[72],\"id\":15043,\"logprob\":-0.2," +
				"\"top_logprobs\":[{\"token\":\"Hi\",\"bytes\":[72],\"id\":9932,\"logprob\":-2.3}]}" +
				"]}";
		LlamaOutput output = parser.parse(json);
		assertEquals(1, output.probabilities.size());
		assertEquals(-0.2f, output.probabilities.get("Hello"), 0.001f);
	}

	@Test
	public void testProbabilitiesTokenWithEscapedChars() {
		String json = "{\"content\":\"hi\",\"stop\":true,\"stop_type\":\"eos\"," +
				"\"completion_probabilities\":[" +
				"{\"token\":\"say \\\"yes\\\"\",\"bytes\":[],\"id\":1,\"prob\":0.5," +
				"\"top_probs\":[]}" +
				"]}";
		LlamaOutput output = parser.parse(json);
		assertEquals(1, output.probabilities.size());
		assertEquals(0.5f, output.probabilities.get("say \"yes\""), 0.001f);
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
		LlamaOutput output = parser.parse(json);
		assertTrue(output.stop);
		assertEquals(StopReason.EOS, output.stopReason);
	}

	@Test
	public void testStopReasonFromJsonWord() {
		String json = "{\"content\":\"done\",\"stop\":true,\"stop_type\":\"word\",\"stopping_word\":\"END\"}";
		LlamaOutput output = parser.parse(json);
		assertTrue(output.stop);
		assertEquals(StopReason.STOP_STRING, output.stopReason);
	}

	@Test
	public void testStopReasonFromJsonLimit() {
		String json = "{\"content\":\"truncated\",\"stop\":true,\"stop_type\":\"limit\",\"truncated\":true}";
		LlamaOutput output = parser.parse(json);
		assertTrue(output.stop);
		assertEquals(StopReason.MAX_TOKENS, output.stopReason);
	}

	@Test
	public void testStopReasonNoneWhenStopFalse() {
		String json = "{\"content\":\"partial\",\"stop\":false,\"stop_type\":\"eos\"}";
		LlamaOutput output = parser.parse(json);
		assertFalse(output.stop);
		// stopReason is NONE for non-final tokens regardless of stop_type
		assertEquals(StopReason.NONE, output.stopReason);
	}
}
