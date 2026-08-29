// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.hasKey;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.args.ContinuationMode;
import net.ladenthin.llama.args.MiroStat;
import net.ladenthin.llama.args.ReasoningFormat;
import net.ladenthin.llama.args.Sampler;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify that every InferenceParameters wither correctly stores its value in the "
                + "internal JSON parameter map, that the toJsonString helper properly escapes all "
                + "special characters (backslash, double-quote, newline, tab, CR, '</' sequence), "
                + "that collection-based withers (logit bias, disable tokens, stop strings, samplers) "
                + "produce correctly formatted JSON arrays, and that withMessages enforces the "
                + "'user'/'assistant'-only role contract.")
public class InferenceParametersTest {

    // -------------------------------------------------------------------------
    // Constructor / prompt
    // -------------------------------------------------------------------------

    @Test
    public void testConstructorSetsPrompt() {
        InferenceParameters params = new InferenceParameters("hello");
        assertThat(params.parameters, hasKey("prompt"));
        assertThat(params.parameters.get("prompt"), is("\"hello\""));
    }

    @Test
    public void testConstructorWithEmptyPrompt() {
        InferenceParameters params = new InferenceParameters("");
        assertThat(params.parameters.get("prompt"), is("\"\""));
    }

    @Test
    public void testSetPromptOverrides() {
        InferenceParameters params = new InferenceParameters("first");
        params = params.withPrompt("second");
        assertThat(params.parameters.get("prompt"), is("\"second\""));
    }

    // -------------------------------------------------------------------------
    // Basic scalar setters
    // -------------------------------------------------------------------------

    @Test
    public void testSetNPredict() {
        InferenceParameters params = new InferenceParameters("").withNPredict(42);
        assertThat(params.parameters.get("n_predict"), is("42"));
    }

    @Test
    public void testSetSsePingInterval() {
        InferenceParameters params = new InferenceParameters("").withSsePingInterval(1);
        assertThat(params.parameters.get("sse_ping_interval"), is("1"));
        // -1 disables pings and must be accepted (the schema's hard lower bound is -1).
        assertThat(
                InferenceParameters.empty().withSsePingInterval(-1).parameters.get("sse_ping_interval"), is("-1"));
    }

    @Test
    public void testAdditionalCompletionScalarsFromB9864Audit() {
        // Plain scalars honored by eval_llama_cmpl_schema but previously not surfaced as withers.
        assertThat(
                InferenceParameters.empty().withXtcProbability(0.5f).parameters.get("xtc_probability"), is("0.5"));
        assertThat(InferenceParameters.empty().withXtcThreshold(0.1f).parameters.get("xtc_threshold"), is("0.1"));
        assertThat(InferenceParameters.empty().withNDiscard(64).parameters.get("n_discard"), is("64"));
        assertThat(InferenceParameters.empty().withNIndent(4).parameters.get("n_indent"), is("4"));
        assertThat(
                InferenceParameters.empty().withTMaxPredictMs(2000).parameters.get("t_max_predict_ms"), is("2000"));
        assertThat(
                InferenceParameters.empty()
                        .withPostSamplingProbs(true)
                        .parameters
                        .get("post_sampling_probs"),
                is("true"));
        assertThat(
                InferenceParameters.empty().withTimingsPerToken(true).parameters.get("timings_per_token"), is("true"));
        assertThat(InferenceParameters.empty().withReturnTokens(true).parameters.get("return_tokens"), is("true"));
    }

    @Test
    public void testSetCacheReuse() {
        InferenceParameters params = InferenceParameters.empty().withCacheReuse(256);
        assertThat(params.parameters.get("n_cache_reuse"), is("256"));
        assertThrows(IllegalArgumentException.class, () -> params.withCacheReuse(-1));
    }

    @Test
    public void testSetSlotId() {
        InferenceParameters params = InferenceParameters.empty().withSlotId(2);
        assertThat(params.parameters.get("id_slot"), is("2"));
        assertThrows(IllegalArgumentException.class, () -> params.withSlotId(-1));
    }

    @Test
    public void testNaNFloatRejected() {
        // NaN/Infinity would serialize to "NaN"/"Infinity" — invalid JSON the native parser rejects.
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withTemperature(Float.NaN));
    }

    @Test
    public void testInfiniteFloatRejected() {
        assertThrows(
                IllegalArgumentException.class, () -> new InferenceParameters("").withTopP(Float.POSITIVE_INFINITY));
    }

    @Test
    public void testSetParallelToolCalls() {
        InferenceParameters params = new InferenceParameters("").withParallelToolCalls(false);
        assertThat(params.parameters.get("parallel_tool_calls"), is("false"));
    }

    @Test
    public void testSetTemperature() {
        InferenceParameters params = new InferenceParameters("").withTemperature(0.5f);
        assertThat(params.parameters.get("temperature"), is("0.5"));
    }

    @Test
    public void testSetTopK() {
        InferenceParameters params = new InferenceParameters("").withTopK(10);
        assertThat(params.parameters.get("top_k"), is("10"));
    }

    @Test
    public void testSetTopP() {
        InferenceParameters params = new InferenceParameters("").withTopP(0.9f);
        assertThat(params.parameters.get("top_p"), is("0.9"));
    }

    @Test
    public void testSetMinP() {
        InferenceParameters params = new InferenceParameters("").withMinP(0.1f);
        assertThat(params.parameters.get("min_p"), is("0.1"));
    }

    @Test
    public void testSetTfsZ() {
        InferenceParameters params = new InferenceParameters("").withTfsZ(1.0f);
        assertThat(params.parameters.get("tfs_z"), is("1.0"));
    }

    @Test
    public void testSetTypicalP() {
        InferenceParameters params = new InferenceParameters("").withTypicalP(0.8f);
        assertThat(params.parameters.get("typical_p"), is("0.8"));
    }

    @Test
    public void testSetRepeatLastN() {
        InferenceParameters params = new InferenceParameters("").withRepeatLastN(64);
        assertThat(params.parameters.get("repeat_last_n"), is("64"));
    }

    @Test
    public void testSetRepeatPenalty() {
        InferenceParameters params = new InferenceParameters("").withRepeatPenalty(1.1f);
        assertThat(params.parameters.get("repeat_penalty"), is("1.1"));
    }

    @Test
    public void testSetFrequencyPenalty() {
        InferenceParameters params = new InferenceParameters("").withFrequencyPenalty(0.2f);
        assertThat(params.parameters.get("frequency_penalty"), is("0.2"));
    }

    @Test
    public void testSetPresencePenalty() {
        InferenceParameters params = new InferenceParameters("").withPresencePenalty(0.3f);
        assertThat(params.parameters.get("presence_penalty"), is("0.3"));
    }

    @Test
    public void testSetSeed() {
        InferenceParameters params = new InferenceParameters("").withSeed(1234);
        assertThat(params.parameters.get("seed"), is("1234"));
    }

    @Test
    public void testSetNProbs() {
        InferenceParameters params = new InferenceParameters("").withNProbs(5);
        assertThat(params.parameters.get("n_probs"), is("5"));
    }

    @Test
    public void testSetMinKeep() {
        InferenceParameters params = new InferenceParameters("").withMinKeep(2);
        assertThat(params.parameters.get("min_keep"), is("2"));
    }

    @Test
    public void testSetNKeep() {
        InferenceParameters params = new InferenceParameters("").withNKeep(-1);
        assertThat(params.parameters.get("n_keep"), is("-1"));
    }

    @Test
    public void testSetCachePrompt() {
        InferenceParameters params = new InferenceParameters("").withCachePrompt(true);
        assertThat(params.parameters.get("cache_prompt"), is("true"));
    }

    @Test
    public void testSetIgnoreEos() {
        InferenceParameters params = new InferenceParameters("").withIgnoreEos(true);
        assertThat(params.parameters.get("ignore_eos"), is("true"));
    }

    @Test
    public void testSetPenalizeNl() {
        InferenceParameters params = new InferenceParameters("").withPenalizeNl(false);
        assertThat(params.parameters.get("penalize_nl"), is("false"));
    }

    @Test
    public void testSetDynamicTemperatureRange() {
        InferenceParameters params = new InferenceParameters("").withDynamicTemperatureRange(0.5f);
        assertThat(params.parameters.get("dynatemp_range"), is("0.5"));
    }

    @Test
    public void testSetDynamicTemperatureExponent() {
        InferenceParameters params = new InferenceParameters("").withDynamicTemperatureExponent(2.0f);
        assertThat(params.parameters.get("dynatemp_exponent"), is("2.0"));
    }

    // -------------------------------------------------------------------------
    // String setters (JSON-escaped)
    // -------------------------------------------------------------------------

    @Test
    public void testSetInputPrefix() {
        InferenceParameters params = new InferenceParameters("").withInputPrefix("prefix");
        assertThat(params.parameters.get("input_prefix"), is("\"prefix\""));
    }

    @Test
    public void testSetInputSuffix() {
        InferenceParameters params = new InferenceParameters("").withInputSuffix("suffix");
        assertThat(params.parameters.get("input_suffix"), is("\"suffix\""));
    }

    @Test
    public void testSetGrammar() {
        InferenceParameters params = new InferenceParameters("").withGrammar("root ::= \"a\"");
        assertThat(params.parameters.get("grammar"), is("\"root ::= \\\"a\\\"\""));
    }

    @Test
    public void testSetJsonSchemaStoresVerbatim() {
        String schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}";
        InferenceParameters params = new InferenceParameters("").withJsonSchema(schema);
        assertThat(params.parameters.get("json_schema"), is(schema));
        assertThat(params.toString(), containsString("\"json_schema\": " + schema));
    }

    @Test
    public void testSetPenaltyPromptString() {
        InferenceParameters params = new InferenceParameters("").withPenaltyPrompt("Hello!");
        assertThat(params.parameters.get("penalty_prompt"), is("\"Hello!\""));
    }

    @Test
    public void testSetUseChatTemplate() {
        InferenceParameters params = new InferenceParameters("").withUseChatTemplate(true);
        assertThat(params.parameters.get("use_jinja"), is("true"));
    }

    @Test
    public void testSetChatTemplate() {
        InferenceParameters params = new InferenceParameters("").withChatTemplate("{{messages}}");
        assertThat(params.parameters.get("chat_template"), is("\"{{messages}}\""));
    }

    @Test
    public void testSetChatTemplateKwargs() {
        java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        kwargs.put("enable_thinking", "true");
        kwargs.put("max_tokens", "1024");
        InferenceParameters params = new InferenceParameters("").withChatTemplateKwargs(kwargs);
        String value = params.parameters.get("chat_template_kwargs");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("\"enable_thinking\":true"));
        assertThat(value, containsString("\"max_tokens\":1024"));
    }

    @Test
    public void testSetChatTemplateKwargsEmpty() {
        java.util.Map<String, String> kwargs = new java.util.LinkedHashMap<>();
        InferenceParameters params = new InferenceParameters("").withChatTemplateKwargs(kwargs);
        assertThat(params.parameters.get("chat_template_kwargs"), is("{}"));
    }

    // -------------------------------------------------------------------------
    // setTopNSigma
    // -------------------------------------------------------------------------

    @Test
    public void testSetTopNSigmaEnabled() {
        InferenceParameters params = new InferenceParameters("").withTopNSigma(2.0f);
        assertThat(params.parameters.get("top_n_sigma"), is("2.0"));
    }

    @Test
    public void testSetTopNSigmaDisabled() {
        InferenceParameters params = new InferenceParameters("").withTopNSigma(-1.0f);
        assertThat(params.parameters.get("top_n_sigma"), is("-1.0"));
    }

    // -------------------------------------------------------------------------
    // ReasoningFormat / ReasoningBudgetTokens
    // -------------------------------------------------------------------------

    @Test
    public void testSetReasoningFormatNone() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.NONE);
        assertThat(params.parameters.get("reasoning_format"), is("\"none\""));
    }

    @Test
    public void testSetReasoningFormatAuto() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.AUTO);
        assertThat(params.parameters.get("reasoning_format"), is("\"auto\""));
    }

    @Test
    public void testSetReasoningFormatDeepseek() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.DEEPSEEK);
        assertThat(params.parameters.get("reasoning_format"), is("\"deepseek\""));
    }

    @Test
    public void testSetReasoningFormatDeepseekLegacy() {
        InferenceParameters params = new InferenceParameters("").withReasoningFormat(ReasoningFormat.DEEPSEEK_LEGACY);
        assertThat(params.parameters.get("reasoning_format"), is("\"deepseek-legacy\""));
    }

    @Test
    public void testSetReasoningBudgetTokensPositive() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(512);
        assertThat(params.parameters.get("reasoning_budget_tokens"), is("512"));
    }

    @Test
    public void testSetReasoningBudgetTokensZero() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(0);
        assertThat(params.parameters.get("reasoning_budget_tokens"), is("0"));
    }

    @Test
    public void testSetReasoningBudgetTokensDisabled() {
        InferenceParameters params = new InferenceParameters("").withReasoningBudgetTokens(-1);
        assertThat(params.parameters.get("reasoning_budget_tokens"), is("-1"));
    }

    @Test
    public void testSetContinueFinalMessageTrue() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(true);
        assertThat(params.parameters.get("continue_final_message"), is("true"));
    }

    @Test
    public void testSetContinueFinalMessageFalse() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(false);
        assertThat(params.parameters.get("continue_final_message"), is("false"));
    }

    @Test
    public void testSetContinueFinalMessageReasoningContent() {
        InferenceParameters params =
                new InferenceParameters("").withContinueFinalMessage(ContinuationMode.REASONING_CONTENT);
        assertThat(params.parameters.get("continue_final_message"), is("\"reasoning_content\""));
    }

    @Test
    public void testSetContinueFinalMessageContent() {
        InferenceParameters params = new InferenceParameters("").withContinueFinalMessage(ContinuationMode.CONTENT);
        assertThat(params.parameters.get("continue_final_message"), is("\"content\""));
    }

    // -------------------------------------------------------------------------
    // MiroStat
    // -------------------------------------------------------------------------

    @Test
    public void testSetMiroStatDisabled() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.DISABLED);
        assertThat(params.parameters.get("mirostat"), is("0"));
    }

    @Test
    public void testSetMiroStatV1() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.V1);
        assertThat(params.parameters.get("mirostat"), is("1"));
    }

    @Test
    public void testSetMiroStatV2() {
        InferenceParameters params = new InferenceParameters("").withMiroStat(MiroStat.V2);
        assertThat(params.parameters.get("mirostat"), is("2"));
    }

    @Test
    public void testSetMiroStatTau() {
        InferenceParameters params = new InferenceParameters("").withMiroStatTau(5.0f);
        assertThat(params.parameters.get("mirostat_tau"), is("5.0"));
    }

    @Test
    public void testSetMiroStatEta() {
        InferenceParameters params = new InferenceParameters("").withMiroStatEta(0.1f);
        assertThat(params.parameters.get("mirostat_eta"), is("0.1"));
    }

    // -------------------------------------------------------------------------
    // Stop strings
    // -------------------------------------------------------------------------

    @Test
    public void testSetStopStringsSingle() {
        InferenceParameters params = new InferenceParameters("").withStopStrings("stop");
        assertThat(params.parameters.get("stop"), is("[\"stop\"]"));
    }

    @Test
    public void testSetStopStringsMultiple() {
        InferenceParameters params = new InferenceParameters("").withStopStrings("stop1", "stop2");
        assertThat(params.parameters.get("stop"), is("[\"stop1\",\"stop2\"]"));
    }

    @Test
    public void testSetStopStringsEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withStopStrings();
        assertThat(params.parameters, not(hasKey("stop")));
    }

    // -------------------------------------------------------------------------
    // Samplers
    // -------------------------------------------------------------------------

    @Test
    public void testSetSamplersSingle() {
        InferenceParameters params = new InferenceParameters("").withSamplers(Sampler.TOP_K);
        assertThat(params.parameters.get("samplers"), is("[\"top_k\"]"));
    }

    @Test
    public void testSetSamplersMultiple() {
        InferenceParameters params =
                new InferenceParameters("").withSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.TEMPERATURE);
        assertThat(params.parameters.get("samplers"), is("[\"top_k\",\"top_p\",\"temperature\"]"));
    }

    @Test
    public void testSetSamplersMinP() {
        InferenceParameters params = new InferenceParameters("").withSamplers(Sampler.MIN_P);
        assertThat(params.parameters.get("samplers"), is("[\"min_p\"]"));
    }

    @Test
    public void testSetSamplersEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withSamplers();
        assertThat(params.parameters, not(hasKey("samplers")));
    }

    // -------------------------------------------------------------------------
    // Token ID bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenIdBias() {
        Map<Integer, Float> bias = Collections.singletonMap(15043, 1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(bias);
        String value = params.parameters.get("logit_bias");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("15043"));
        assertThat(value, containsString("1.0"));
    }

    @Test
    public void testSetTokenIdBiasEmpty() {
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(Collections.emptyMap());
        assertThat(params.parameters, not(hasKey("logit_bias")));
    }

    // -------------------------------------------------------------------------
    // Token string bias
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenBias() {
        Map<String, Float> bias = Collections.singletonMap(" Hello", 1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenBias(bias);
        String value = params.parameters.get("logit_bias");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("Hello"));
        assertThat(value, containsString("1.0"));
    }

    @Test
    public void testSetTokenBiasEmpty() {
        InferenceParameters params = new InferenceParameters("").withTokenBias(Collections.emptyMap());
        assertThat(params.parameters, not(hasKey("logit_bias")));
    }

    // -------------------------------------------------------------------------
    // Disable tokens
    // -------------------------------------------------------------------------

    @Test
    public void testDisableTokenIds() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokenIds(Arrays.asList(1, 2, 3));
        String value = params.parameters.get("logit_bias");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("false"));
        assertThat(value, containsString("1"));
    }

    @Test
    public void testDisableTokenIdsEmpty() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokenIds(Collections.emptyList());
        assertThat(params.parameters, not(hasKey("logit_bias")));
    }

    @Test
    public void testDisableTokens() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokens(Arrays.asList("bad", "word"));
        String value = params.parameters.get("logit_bias");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("false"));
        assertThat(value, containsString("bad"));
    }

    @Test
    public void testDisableTokensEmpty() {
        InferenceParameters params = new InferenceParameters("").withDisabledTokens(Collections.emptyList());
        assertThat(params.parameters, not(hasKey("logit_bias")));
    }

    // -------------------------------------------------------------------------
    // Penalty prompt with token ids
    // -------------------------------------------------------------------------

    @Test
    public void testSetPenaltyPromptTokenIds() {
        InferenceParameters params = new InferenceParameters("").withPenaltyPrompt(new int[] {1, 2, 3});
        assertThat(params.parameters.get("penalty_prompt"), is("[1,2,3]"));
    }

    @Test
    public void testSetPenaltyPromptTokenIdsEmpty() {
        InferenceParameters params = new InferenceParameters("");
        params = params.withPenaltyPrompt(new int[] {});
        assertThat(params.parameters, not(hasKey("penalty_prompt")));
    }

    // -------------------------------------------------------------------------
    // setMessages
    // -------------------------------------------------------------------------

    @Test
    public void testSetMessagesWithSystemAndUserMessages() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hi"));
        InferenceParameters params = new InferenceParameters("").withMessages("System msg", messages);
        String value = params.parameters.get("messages");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("system"));
        assertThat(value, containsString("System msg"));
        assertThat(value, containsString("user"));
        assertThat(value, containsString("Hi"));
    }

    @Test
    public void testSetMessagesWithAssistantRole() {
        List<Pair<String, String>> messages =
                Arrays.asList(new Pair<>("user", "Hello"), new Pair<>("assistant", "Hi there"));
        InferenceParameters params = new InferenceParameters("").withMessages(null, messages);
        String value = params.parameters.get("messages");
        assertThat(value, is(notNullValue()));
        assertThat(value, containsString("assistant"));
        assertThat(value, containsString("Hi there"));
    }

    @Test
    public void testSetMessagesNoSystemMessage() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
        InferenceParameters params = new InferenceParameters("").withMessages(null, messages);
        String value = params.parameters.get("messages");
        assertThat(value, is(notNullValue()));
        assertThat(value, not(containsString("system")));
        assertThat(value, containsString("user"));
    }

    @Test
    public void testSetMessagesEmptySystemMessage() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("user", "Hello"));
        InferenceParameters params = new InferenceParameters("").withMessages("", messages);
        String value = params.parameters.get("messages");
        assertThat(value, not(containsString("system")));
    }

    @Test
    public void testSetMessagesInvalidRole() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("system", "Bad"));
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withMessages(null, messages));
    }

    @Test
    public void testSetMessagesInvalidRoleOther() {
        List<Pair<String, String>> messages = Collections.singletonList(new Pair<>("admin", "Hack"));
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withMessages(null, messages));
    }

    // -------------------------------------------------------------------------
    // toString (JSON output)
    // -------------------------------------------------------------------------

    @Test
    public void testToStringContainsPrompt() {
        InferenceParameters params = new InferenceParameters("test prompt");
        String json = params.toString();
        assertThat(json, startsWith("{"));
        assertThat(json, endsWith("}"));
        assertThat(json, containsString("\"prompt\""));
        assertThat(json, containsString("\"test prompt\""));
    }

    @Test
    public void testToStringWithMultipleParams() {
        InferenceParameters params =
                new InferenceParameters("p").withTemperature(0.7f).withTopK(20);
        String json = params.toString();
        assertThat(json, containsString("\"temperature\""));
        assertThat(json, containsString("\"top_k\""));
    }

    // -------------------------------------------------------------------------
    // toJsonString special character escaping
    // -------------------------------------------------------------------------

    @Test
    public void testToJsonStringEscapesBackslash() {
        InferenceParameters params = new InferenceParameters("path\\to\\file");
        assertThat(params.parameters.get("prompt"), is("\"path\\\\to\\\\file\""));
    }

    @Test
    public void testToJsonStringEscapesDoubleQuote() {
        InferenceParameters params = new InferenceParameters("say \"hi\"");
        assertThat(params.parameters.get("prompt"), is("\"say \\\"hi\\\"\""));
    }

    @Test
    public void testToJsonStringEscapesNewline() {
        InferenceParameters params = new InferenceParameters("line1\nline2");
        assertThat(params.parameters.get("prompt"), is("\"line1\\nline2\""));
    }

    @Test
    public void testToJsonStringEscapesTab() {
        InferenceParameters params = new InferenceParameters("col1\tcol2");
        assertThat(params.parameters.get("prompt"), is("\"col1\\tcol2\""));
    }

    @Test
    public void testToJsonStringEscapesCarriageReturn() {
        InferenceParameters params = new InferenceParameters("a\rb");
        assertThat(params.parameters.get("prompt"), is("\"a\\rb\""));
    }

    @Test
    public void testToJsonStringNull() {
        // toJsonString(null) returns null — only used internally but verify via grammar
        InferenceParameters params = new InferenceParameters("");
        params = params.withGrammar(null);
        assertThat(params.parameters.get("grammar"), is(nullValue()));
    }

    @Test
    public void testToJsonStringSlashNotEscaped() {
        // Jackson does not escape '/' — forward slashes are passed through verbatim
        InferenceParameters params = new InferenceParameters("</script>");
        String value = params.parameters.get("prompt");
        assertThat(value, containsString("</script>"));
        assertThat(value, not(containsString("<\\/")));
    }

    // -------------------------------------------------------------------------
    // Builder chaining returns a new instance (immutable wither semantics)
    // -------------------------------------------------------------------------

    @Test
    public void testBuilderChainingReturnsNewInstance() {
        InferenceParameters params = new InferenceParameters("");
        assertThat(params.withTemperature(0.5f), is(not(sameInstance(params))));
        assertThat(params.withTopK(10), is(not(sameInstance(params))));
        assertThat(params.withNPredict(5), is(not(sameInstance(params))));
    }

    // -------------------------------------------------------------------------
    // Stream (package-private)
    // -------------------------------------------------------------------------

    @Test
    public void testSetStreamTrue() {
        InferenceParameters params = new InferenceParameters("").withStream(true);
        assertThat(params.parameters.get("stream"), is("true"));
    }

    @Test
    public void testSetStreamFalse() {
        InferenceParameters params = new InferenceParameters("").withStream(false);
        assertThat(params.parameters.get("stream"), is("false"));
    }

    // -------------------------------------------------------------------------
    // Multiple logit bias entries (ordering independent check)
    // -------------------------------------------------------------------------

    @Test
    public void testSetTokenIdBiasMultiple() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(1, 0.5f);
        bias.put(2, -1.0f);
        InferenceParameters params = new InferenceParameters("").withTokenIdBias(bias);
        String value = params.parameters.get("logit_bias");
        assertThat(value, is(notNullValue()));
        assertThat(value, startsWith("["));
        assertThat(value, endsWith("]"));
        assertThat(value, containsString("1"));
        assertThat(value, containsString("2"));
    }

    // -------------------------------------------------------------------------
    // DRY (Don't Repeat Yourself) sampling
    // -------------------------------------------------------------------------

    @Test
    public void testSetDryMultiplier() {
        InferenceParameters params = new InferenceParameters("").withDryMultiplier(0.8f);
        assertThat(params.parameters.get("dry_multiplier"), is("0.8"));
    }

    @Test
    public void testSetDryBase() {
        InferenceParameters params = new InferenceParameters("").withDryBase(1.75f);
        assertThat(params.parameters.get("dry_base"), is("1.75"));
    }

    @Test
    public void testSetDryAllowedLength() {
        InferenceParameters params = new InferenceParameters("").withDryAllowedLength(2);
        assertThat(params.parameters.get("dry_allowed_length"), is("2"));
    }

    @Test
    public void testSetDryPenaltyLastN() {
        InferenceParameters params = new InferenceParameters("").withDryPenaltyLastN(512);
        assertThat(params.parameters.get("dry_penalty_last_n"), is("512"));
    }

    @Test
    public void testSetDryPenaltyLastNRejectsMinusOne() {
        // llama.cpp b10273 set the request schema's hard limits to [0, INT32_MAX]; -1 no longer means
        // "the whole context", it makes the server reject the request.
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withDryPenaltyLastN(-1));
    }

    @Test
    public void testWithRepeatLastNRejectsMinusOne() {
        assertThrows(IllegalArgumentException.class, () -> new InferenceParameters("").withRepeatLastN(-1));
    }

    @Test
    public void testWithRepeatLastNAcceptsZeroAndPositive() {
        assertThat(new InferenceParameters("").withRepeatLastN(0).parameters.get("repeat_last_n"), is("0"));
        assertThat(new InferenceParameters("").withRepeatLastN(64).parameters.get("repeat_last_n"), is("64"));
    }

    @Test
    public void testSetDryPenaltyLastNDisabled() {
        InferenceParameters params = new InferenceParameters("").withDryPenaltyLastN(0);
        assertThat(params.parameters.get("dry_penalty_last_n"), is("0"));
    }

    @Test
    public void testSetDryPenaltyLastNBelowMinusOneRejected() {
        InferenceParameters params = new InferenceParameters("");
        assertThrows(IllegalArgumentException.class, () -> params.withDryPenaltyLastN(-2));
    }

    @Test
    public void testSetDrySequenceBreakersSingle() {
        InferenceParameters params = new InferenceParameters("").withDrySequenceBreakers("\n");
        assertThat(params.parameters.get("dry_sequence_breakers"), is("[\"\\n\"]"));
    }

    @Test
    public void testSetDrySequenceBreakersMultiple() {
        InferenceParameters params = new InferenceParameters("").withDrySequenceBreakers("\n", ":", "\"", "*");
        assertThat(params.parameters.get("dry_sequence_breakers"), is("[\"\\n\",\":\",\"\\\"\",\"*\"]"));
    }

    @Test
    public void testSetDrySequenceBreakersEmpty() {
        InferenceParameters params = new InferenceParameters("");
        InferenceParameters result = params.withDrySequenceBreakers();
        assertThat(params.parameters, not(hasKey("dry_sequence_breakers")));
        assertThat(result, is(sameInstance(params)));
    }

    @Test
    public void testDryDefaultsEmitNothing() {
        InferenceParameters params = new InferenceParameters("prompt");
        assertThat(params.parameters, not(hasKey("dry_multiplier")));
        assertThat(params.parameters, not(hasKey("dry_base")));
        assertThat(params.parameters, not(hasKey("dry_allowed_length")));
        assertThat(params.parameters, not(hasKey("dry_penalty_last_n")));
        assertThat(params.parameters, not(hasKey("dry_sequence_breakers")));
    }

    @Test
    public void testDryWithersReturnNewInstance() {
        InferenceParameters params = new InferenceParameters("");
        assertThat(params.withDryMultiplier(0.8f), is(not(sameInstance(params))));
        assertThat(params.withDryBase(1.75f), is(not(sameInstance(params))));
        assertThat(params.withDryAllowedLength(2), is(not(sameInstance(params))));
        assertThat(params.withDryPenaltyLastN(512), is(not(sameInstance(params))));
        assertThat(params.withDrySequenceBreakers("\n"), is(not(sameInstance(params))));
    }
}
