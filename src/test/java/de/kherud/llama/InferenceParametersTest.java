package de.kherud.llama;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import de.kherud.llama.args.MiroStat;
import de.kherud.llama.args.Sampler;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify that every InferenceParameters setter correctly stores its value in the " +
                  "internal JSON parameter map, that the toJsonString helper properly escapes all " +
                  "special characters (backslash, double-quote, newline, tab, CR, '</' sequence), " +
                  "that collection-based setters (logit bias, disable tokens, stop strings, samplers) " +
                  "produce correctly formatted JSON arrays, and that setMessages enforces the " +
                  "'user'/'assistant'-only role contract."
)
public class InferenceParametersTest {

	// -------------------------------------------------------------------------
	// Constructor / prompt
	// -------------------------------------------------------------------------

	@Test
	public void testConstructorSetsPrompt() {
		InferenceParameters params = new InferenceParameters("hello");
		assertTrue(params.parameters.containsKey("prompt"));
		assertEquals("\"hello\"", params.parameters.get("prompt"));
	}

	@Test
	public void testConstructorWithEmptyPrompt() {
		InferenceParameters params = new InferenceParameters("");
		assertEquals("\"\"", params.parameters.get("prompt"));
	}

	@Test
	public void testSetPromptOverrides() {
		InferenceParameters params = new InferenceParameters("first");
		params.setPrompt("second");
		assertEquals("\"second\"", params.parameters.get("prompt"));
	}

	// -------------------------------------------------------------------------
	// Basic scalar setters
	// -------------------------------------------------------------------------

	@Test
	public void testSetNPredict() {
		InferenceParameters params = new InferenceParameters("").setNPredict(42);
		assertEquals("42", params.parameters.get("n_predict"));
	}

	@Test
	public void testSetTemperature() {
		InferenceParameters params = new InferenceParameters("").setTemperature(0.5f);
		assertEquals("0.5", params.parameters.get("temperature"));
	}

	@Test
	public void testSetTopK() {
		InferenceParameters params = new InferenceParameters("").setTopK(10);
		assertEquals("10", params.parameters.get("top_k"));
	}

	@Test
	public void testSetTopP() {
		InferenceParameters params = new InferenceParameters("").setTopP(0.9f);
		assertEquals("0.9", params.parameters.get("top_p"));
	}

	@Test
	public void testSetMinP() {
		InferenceParameters params = new InferenceParameters("").setMinP(0.1f);
		assertEquals("0.1", params.parameters.get("min_p"));
	}

	@Test
	public void testSetTfsZ() {
		InferenceParameters params = new InferenceParameters("").setTfsZ(1.0f);
		assertEquals("1.0", params.parameters.get("tfs_z"));
	}

	@Test
	public void testSetTypicalP() {
		InferenceParameters params = new InferenceParameters("").setTypicalP(0.8f);
		assertEquals("0.8", params.parameters.get("typical_p"));
	}

	@Test
	public void testSetRepeatLastN() {
		InferenceParameters params = new InferenceParameters("").setRepeatLastN(64);
		assertEquals("64", params.parameters.get("repeat_last_n"));
	}

	@Test
	public void testSetRepeatPenalty() {
		InferenceParameters params = new InferenceParameters("").setRepeatPenalty(1.1f);
		assertEquals("1.1", params.parameters.get("repeat_penalty"));
	}

	@Test
	public void testSetFrequencyPenalty() {
		InferenceParameters params = new InferenceParameters("").setFrequencyPenalty(0.2f);
		assertEquals("0.2", params.parameters.get("frequency_penalty"));
	}

	@Test
	public void testSetPresencePenalty() {
		InferenceParameters params = new InferenceParameters("").setPresencePenalty(0.3f);
		assertEquals("0.3", params.parameters.get("presence_penalty"));
	}

	@Test
	public void testSetSeed() {
		InferenceParameters params = new InferenceParameters("").setSeed(1234);
		assertEquals("1234", params.parameters.get("seed"));
	}

	@Test
	public void testSetNProbs() {
		InferenceParameters params = new InferenceParameters("").setNProbs(5);
		assertEquals("5", params.parameters.get("n_probs"));
	}

	@Test
	public void testSetMinKeep() {
		InferenceParameters params = new InferenceParameters("").setMinKeep(2);
		assertEquals("2", params.parameters.get("min_keep"));
	}

	@Test
	public void testSetNKeep() {
		InferenceParameters params = new InferenceParameters("").setNKeep(-1);
		assertEquals("-1", params.parameters.get("n_keep"));
	}

	@Test
	public void testSetCachePrompt() {
		InferenceParameters params = new InferenceParameters("").setCachePrompt(true);
		assertEquals("true", params.parameters.get("cache_prompt"));
	}

	@Test
	public void testSetIgnoreEos() {
		InferenceParameters params = new InferenceParameters("").setIgnoreEos(true);
		assertEquals("true", params.parameters.get("ignore_eos"));
	}

	@Test
	public void testSetPenalizeNl() {
		InferenceParameters params = new InferenceParameters("").setPenalizeNl(false);
		assertEquals("false", params.parameters.get("penalize_nl"));
	}

	@Test
	public void testSetDynamicTemperatureRange() {
		InferenceParameters params = new InferenceParameters("").setDynamicTemperatureRange(0.5f);
		assertEquals("0.5", params.parameters.get("dynatemp_range"));
	}

	@Test
	public void testSetDynamicTemperatureExponent() {
		InferenceParameters params = new InferenceParameters("").setDynamicTemperatureExponent(2.0f);
		assertEquals("2.0", params.parameters.get("dynatemp_exponent"));
	}

	// -------------------------------------------------------------------------
	// String setters (JSON-escaped)
	// -------------------------------------------------------------------------

	@Test
	public void testSetInputPrefix() {
		InferenceParameters params = new InferenceParameters("").setInputPrefix("prefix");
		assertEquals("\"prefix\"", params.parameters.get("input_prefix"));
	}

	@Test
	public void testSetInputSuffix() {
		InferenceParameters params = new InferenceParameters("").setInputSuffix("suffix");
		assertEquals("\"suffix\"", params.parameters.get("input_suffix"));
	}

	@Test
	public void testSetGrammar() {
		InferenceParameters params = new InferenceParameters("").setGrammar("root ::= \"a\"");
		assertEquals("\"root ::= \\\"a\\\"\"", params.parameters.get("grammar"));
	}

	@Test
	public void testSetPenaltyPromptString() {
		InferenceParameters params = new InferenceParameters("").setPenaltyPrompt("Hello!");
		assertEquals("\"Hello!\"", params.parameters.get("penalty_prompt"));
	}

	@Test
	public void testSetUseChatTemplate() {
		InferenceParameters params = new InferenceParameters("").setUseChatTemplate(true);
		assertEquals("true", params.parameters.get("use_jinja"));
	}

	@Test
	public void testSetChatTemplate() {
		InferenceParameters params = new InferenceParameters("").setChatTemplate("{{messages}}");
		assertEquals("\"{{messages}}\"", params.parameters.get("chat_template"));
	}

	@Test
	public void testSetChatTemplateKwargs() {
		java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
		kwargs.put("enable_thinking", "true");
		kwargs.put("max_tokens", "1024");
		InferenceParameters params = new InferenceParameters("").setChatTemplateKwargs(kwargs);
		String value = params.parameters.get("chat_template_kwargs");
		assertNotNull(value);
		assertTrue(value.contains("\"enable_thinking\":true"));
		assertTrue(value.contains("\"max_tokens\":1024"));
	}

	@Test
	public void testSetChatTemplateKwargsEmpty() {
		java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
		InferenceParameters params = new InferenceParameters("").setChatTemplateKwargs(kwargs);
		assertEquals("{}", params.parameters.get("chat_template_kwargs"));
	}

	// -------------------------------------------------------------------------
	// MiroStat
	// -------------------------------------------------------------------------

	@Test
	public void testSetMiroStatDisabled() {
		InferenceParameters params = new InferenceParameters("").setMiroStat(MiroStat.DISABLED);
		assertEquals("0", params.parameters.get("mirostat"));
	}

	@Test
	public void testSetMiroStatV1() {
		InferenceParameters params = new InferenceParameters("").setMiroStat(MiroStat.V1);
		assertEquals("1", params.parameters.get("mirostat"));
	}

	@Test
	public void testSetMiroStatV2() {
		InferenceParameters params = new InferenceParameters("").setMiroStat(MiroStat.V2);
		assertEquals("2", params.parameters.get("mirostat"));
	}

	@Test
	public void testSetMiroStatTau() {
		InferenceParameters params = new InferenceParameters("").setMiroStatTau(5.0f);
		assertEquals("5.0", params.parameters.get("mirostat_tau"));
	}

	@Test
	public void testSetMiroStatEta() {
		InferenceParameters params = new InferenceParameters("").setMiroStatEta(0.1f);
		assertEquals("0.1", params.parameters.get("mirostat_eta"));
	}

	// -------------------------------------------------------------------------
	// Stop strings
	// -------------------------------------------------------------------------

	@Test
	public void testSetStopStringsSingle() {
		InferenceParameters params = new InferenceParameters("").setStopStrings("stop");
		assertEquals("[\"stop\"]", params.parameters.get("stop"));
	}

	@Test
	public void testSetStopStringsMultiple() {
		InferenceParameters params = new InferenceParameters("").setStopStrings("stop1", "stop2");
		assertEquals("[\"stop1\", \"stop2\"]", params.parameters.get("stop"));
	}

	@Test
	public void testSetStopStringsEmpty() {
		InferenceParameters params = new InferenceParameters("");
		params.setStopStrings();
		assertFalse(params.parameters.containsKey("stop"));
	}

	// -------------------------------------------------------------------------
	// Samplers
	// -------------------------------------------------------------------------

	@Test
	public void testSetSamplersSingle() {
		InferenceParameters params = new InferenceParameters("").setSamplers(Sampler.TOP_K);
		assertEquals("[\"top_k\"]", params.parameters.get("samplers"));
	}

	@Test
	public void testSetSamplersMultiple() {
		InferenceParameters params = new InferenceParameters("").setSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
		assertEquals("[\"top_k\", \"top_p\", \"temperature\"]", params.parameters.get("samplers"));
	}

	@Test
	public void testSetSamplersMinP() {
		InferenceParameters params = new InferenceParameters("").setSamplers(Sampler.MIN_P);
		assertEquals("[\"min_p\"]", params.parameters.get("samplers"));
	}

	@Test
	public void testSetSamplersEmpty() {
		InferenceParameters params = new InferenceParameters("");
		params.setSamplers();
		assertFalse(params.parameters.containsKey("samplers"));
	}

	// -------------------------------------------------------------------------
	// Token ID bias
	// -------------------------------------------------------------------------

	@Test
	public void testSetTokenIdBias() {
		Map<Integer, Float> bias = Collections.singletonMap(15043, 1.0f);
		InferenceParameters params = new InferenceParameters("").setTokenIdBias(bias);
		String value = params.parameters.get("logit_bias");
		assertNotNull(value);
		assertTrue(value.contains("15043"));
		assertTrue(value.contains("1.0"));
	}

	@Test
	public void testSetTokenIdBiasEmpty() {
		InferenceParameters params = new InferenceParameters("").setTokenIdBias(Collections.emptyMap());
		assertFalse(params.parameters.containsKey("logit_bias"));
	}

	// -------------------------------------------------------------------------
	// Token string bias
	// -------------------------------------------------------------------------

	@Test
	public void testSetTokenBias() {
		Map<String, Float> bias = Collections.singletonMap(" Hello", 1.0f);
		InferenceParameters params = new InferenceParameters("").setTokenBias(bias);
		String value = params.parameters.get("logit_bias");
		assertNotNull(value);
		assertTrue(value.contains("Hello"));
		assertTrue(value.contains("1.0"));
	}

	@Test
	public void testSetTokenBiasEmpty() {
		InferenceParameters params = new InferenceParameters("").setTokenBias(Collections.emptyMap());
		assertFalse(params.parameters.containsKey("logit_bias"));
	}

	// -------------------------------------------------------------------------
	// Disable tokens
	// -------------------------------------------------------------------------

	@Test
	public void testDisableTokenIds() {
		InferenceParameters params = new InferenceParameters("").disableTokenIds(Arrays.asList(1, 2, 3));
		String value = params.parameters.get("logit_bias");
		assertNotNull(value);
		assertTrue(value.contains("false"));
		assertTrue(value.contains("1"));
	}

	@Test
	public void testDisableTokenIdsEmpty() {
		InferenceParameters params = new InferenceParameters("").disableTokenIds(Collections.emptyList());
		assertFalse(params.parameters.containsKey("logit_bias"));
	}

	@Test
	public void testDisableTokens() {
		InferenceParameters params = new InferenceParameters("").disableTokens(Arrays.asList("bad", "word"));
		String value = params.parameters.get("logit_bias");
		assertNotNull(value);
		assertTrue(value.contains("false"));
		assertTrue(value.contains("bad"));
	}

	@Test
	public void testDisableTokensEmpty() {
		InferenceParameters params = new InferenceParameters("").disableTokens(Collections.emptyList());
		assertFalse(params.parameters.containsKey("logit_bias"));
	}

	// -------------------------------------------------------------------------
	// Penalty prompt with token ids
	// -------------------------------------------------------------------------

	@Test
	public void testSetPenaltyPromptTokenIds() {
		InferenceParameters params = new InferenceParameters("").setPenaltyPrompt(new int[]{1, 2, 3});
		assertEquals("[1, 2, 3]", params.parameters.get("penalty_prompt"));
	}

	@Test
	public void testSetPenaltyPromptTokenIdsEmpty() {
		InferenceParameters params = new InferenceParameters("");
		params.setPenaltyPrompt(new int[]{});
		assertFalse(params.parameters.containsKey("penalty_prompt"));
	}

	// -------------------------------------------------------------------------
	// setMessages
	// -------------------------------------------------------------------------

	@Test
	public void testSetMessagesWithSystemAndUserMessages() {
		List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hi"));
		InferenceParameters params = new InferenceParameters("").setMessages("System msg", messages);
		String value = params.parameters.get("messages");
		assertNotNull(value);
		assertTrue(value.contains("system"));
		assertTrue(value.contains("System msg"));
		assertTrue(value.contains("user"));
		assertTrue(value.contains("Hi"));
	}

	@Test
	public void testSetMessagesWithAssistantRole() {
		List<Pair<String, String>> messages = Arrays.asList(
				new Pair<>("user", "Hello"),
				new Pair<>("assistant", "Hi there")
		);
		InferenceParameters params = new InferenceParameters("").setMessages(null, messages);
		String value = params.parameters.get("messages");
		assertNotNull(value);
		assertTrue(value.contains("assistant"));
		assertTrue(value.contains("Hi there"));
	}

	@Test
	public void testSetMessagesNoSystemMessage() {
		List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
		InferenceParameters params = new InferenceParameters("").setMessages(null, messages);
		String value = params.parameters.get("messages");
		assertNotNull(value);
		assertFalse(value.contains("system"));
		assertTrue(value.contains("user"));
	}

	@Test
	public void testSetMessagesEmptySystemMessage() {
		List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
		InferenceParameters params = new InferenceParameters("").setMessages("", messages);
		String value = params.parameters.get("messages");
		assertFalse(value.contains("system"));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetMessagesInvalidRole() {
		List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("system", "Bad"));
		new InferenceParameters("").setMessages(null, messages);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetMessagesInvalidRoleOther() {
		List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("admin", "Hack"));
		new InferenceParameters("").setMessages(null, messages);
	}

	// -------------------------------------------------------------------------
	// toString (JSON output)
	// -------------------------------------------------------------------------

	@Test
	public void testToStringContainsPrompt() {
		InferenceParameters params = new InferenceParameters("test prompt");
		String json = params.toString();
		assertTrue(json.startsWith("{"));
		assertTrue(json.endsWith("}"));
		assertTrue(json.contains("\"prompt\""));
		assertTrue(json.contains("\"test prompt\""));
	}

	@Test
	public void testToStringWithMultipleParams() {
		InferenceParameters params = new InferenceParameters("p")
				.setTemperature(0.7f)
				.setTopK(20);
		String json = params.toString();
		assertTrue(json.contains("\"temperature\""));
		assertTrue(json.contains("\"top_k\""));
	}

	// -------------------------------------------------------------------------
	// toJsonString special character escaping
	// -------------------------------------------------------------------------

	@Test
	public void testToJsonStringEscapesBackslash() {
		InferenceParameters params = new InferenceParameters("path\\to\\file");
		assertEquals("\"path\\\\to\\\\file\"", params.parameters.get("prompt"));
	}

	@Test
	public void testToJsonStringEscapesDoubleQuote() {
		InferenceParameters params = new InferenceParameters("say \"hi\"");
		assertEquals("\"say \\\"hi\\\"\"", params.parameters.get("prompt"));
	}

	@Test
	public void testToJsonStringEscapesNewline() {
		InferenceParameters params = new InferenceParameters("line1\nline2");
		assertEquals("\"line1\\nline2\"", params.parameters.get("prompt"));
	}

	@Test
	public void testToJsonStringEscapesTab() {
		InferenceParameters params = new InferenceParameters("col1\tcol2");
		assertEquals("\"col1\\tcol2\"", params.parameters.get("prompt"));
	}

	@Test
	public void testToJsonStringEscapesCarriageReturn() {
		InferenceParameters params = new InferenceParameters("a\rb");
		assertEquals("\"a\\rb\"", params.parameters.get("prompt"));
	}

	@Test
	public void testToJsonStringNull() {
		// toJsonString(null) returns null — only used internally but verify via grammar
		InferenceParameters params = new InferenceParameters("");
		params.setGrammar(null);
		assertNull(params.parameters.get("grammar"));
	}

	@Test
	public void testToJsonStringEscapesSlashAfterLt() {
		// '</' should become '<\/'
		InferenceParameters params = new InferenceParameters("</script>");
		String value = params.parameters.get("prompt");
		assertTrue(value.contains("<\\/"));
	}

	// -------------------------------------------------------------------------
	// Builder chaining returns same instance
	// -------------------------------------------------------------------------

	@Test
	public void testBuilderChainingReturnsSameInstance() {
		InferenceParameters params = new InferenceParameters("");
		assertSame(params, params.setTemperature(0.5f));
		assertSame(params, params.setTopK(10));
		assertSame(params, params.setNPredict(5));
	}

	// -------------------------------------------------------------------------
	// Stream (package-private)
	// -------------------------------------------------------------------------

	@Test
	public void testSetStreamTrue() {
		InferenceParameters params = new InferenceParameters("").setStream(true);
		assertEquals("true", params.parameters.get("stream"));
	}

	@Test
	public void testSetStreamFalse() {
		InferenceParameters params = new InferenceParameters("").setStream(false);
		assertEquals("false", params.parameters.get("stream"));
	}

	// -------------------------------------------------------------------------
	// Multiple logit bias entries (ordering independent check)
	// -------------------------------------------------------------------------

	@Test
	public void testSetTokenIdBiasMultiple() {
		Map<Integer, Float> bias = new HashMap<>();
		bias.put(1, 0.5f);
		bias.put(2, -1.0f);
		InferenceParameters params = new InferenceParameters("").setTokenIdBias(bias);
		String value = params.parameters.get("logit_bias");
		assertNotNull(value);
		assertTrue(value.startsWith("["));
		assertTrue(value.endsWith("]"));
		assertTrue(value.contains("1"));
		assertTrue(value.contains("2"));
	}
}
