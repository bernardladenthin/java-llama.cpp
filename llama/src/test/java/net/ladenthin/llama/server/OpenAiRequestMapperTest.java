// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import net.ladenthin.llama.parameters.InferenceParameters;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OpenAiRequestMapper}. Pure mapping — no model or native library.
 *
 * <p>Assertions parse {@link InferenceParameters#toString()} (the JSON sent to native) and check the
 * field names the binding actually reads.
 */
public class OpenAiRequestMapperTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final OpenAiRequestMapper mapper = new OpenAiRequestMapper();

    private JsonNode mapAndSerialize(String requestJson) throws IOException {
        InferenceParameters params = mapper.toInferenceParameters(MAPPER.readTree(requestJson));
        return MAPPER.readTree(params.toJson());
    }

    @Test
    public void logitBiasAndNProbsAreForwarded() throws IOException {
        // Upstream reads logit_bias (OpenAI's object form included) and n_probs from the request; the
        // mapper dropped them, so a client's banned token was generated anyway.
        JsonNode chat = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"logit_bias\":{\"15043\":-100,\"2\":false},\"n_probs\":3}");
        assertThat(chat.path("logit_bias").path("15043").asInt(), is(-100));
        assertThat(chat.path("logit_bias").path("2").asBoolean(true), is(false));
        assertThat(chat.path("n_probs").asInt(), is(3));

        JsonNode pairs =
                mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"logit_bias\":[[7,1.5]]}");
        assertThat(pairs.path("logit_bias").path(0).path(1).asDouble(), closeTo(1.5, 1e-9));
    }

    @Test
    public void verboseIsForwarded() throws IOException {
        // upstream's "verbose" adds __verbose to the answer, which carries the matched stop word (F183)
        JsonNode chat = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"verbose\":true}");
        assertThat(chat.path("verbose").asBoolean(false), is(true));
        JsonNode without = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        assertThat(without.has("verbose"), is(false));
    }

    @Test
    public void moreThanOneChoiceIsRefusedBecauseOneAnswerComesBack() throws IOException {
        // Upstream reads "n" as n_cmpl; the chat paths return one answer, so n > 1 silently gave one.
        assertThrows(
                IllegalArgumentException.class,
                () -> mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":2}"));
        assertDoesNotThrow(() -> mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":1}"));
    }

    @Test
    public void chatTemplateKwargsAreForwarded() throws IOException {
        // Qwen3 clients switch thinking off with chat_template_kwargs; upstream's chat parser merges the
        // request's kwargs into the template variables.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
        assertThat(out.path("chat_template_kwargs").path("enable_thinking").isBoolean(), is(true));
        assertThat(out.path("chat_template_kwargs").path("enable_thinking").asBoolean(), is(false));
    }

    @Test
    public void reasoningEffortNoneSwitchesThinkingOffLikeUpstream() throws IOException {
        JsonNode out = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"none\"}");
        assertThat(out.path("chat_template_kwargs").path("enable_thinking").isBoolean(), is(true));
        assertThat(out.path("chat_template_kwargs").path("enable_thinking").asBoolean(), is(false));
        assertThat(out.path("chat_template_kwargs").has("reasoning_effort"), is(false));
    }

    @Test
    public void reasoningEffortLevelBecomesATemplateVariableLikeUpstream() throws IOException {
        JsonNode out =
                mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"low\"}");
        assertThat(out.path("chat_template_kwargs").path("reasoning_effort").asText(), is("low"));
    }

    @Test
    public void reasoningBudgetIsForwardedUnderEitherUpstreamName() throws IOException {
        JsonNode thinking = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking_budget_tokens\":512}");
        assertThat(thinking.path("reasoning_budget_tokens").asInt(), is(512));
        JsonNode reasoning = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_budget_tokens\":256}");
        assertThat(reasoning.path("reasoning_budget_tokens").asInt(), is(256));
    }

    @Test
    public void anthropicAndResponsesReasoningControlsReachTheInferenceParameters() throws IOException {
        // The translation layers produce these fields (F75, F76); they only take effect if the chat
        // mapper carries them on into the request the native side parses.
        InferenceParameters anthropic = mapper.toInferenceParameters(AnthropicApiSupport.toOpenAiChatRequest(
                MAPPER.readTree("{\"max_tokens\":64,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024},"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")));
        assertThat(
                MAPPER.readTree(anthropic.toJson())
                        .path("reasoning_budget_tokens")
                        .asInt(),
                is(1024));
        InferenceParameters responses = mapper.toInferenceParameters(ResponsesApiSupport.toOpenAiChatRequest(
                MAPPER.readTree("{\"input\":\"hi\",\"reasoning\":{\"effort\":\"none\"}}")));
        assertThat(
                MAPPER.readTree(responses.toJson())
                        .path("chat_template_kwargs")
                        .path("enable_thinking")
                        .asBoolean(true),
                is(false));
    }

    @Test
    public void messagesForwardedVerbatim() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        assertThat(out.path("messages").isArray(), is(true));
        assertThat(out.path("messages").get(0).path("role").asText(), is("user"));
        assertThat(out.path("messages").get(0).path("content").asText(), is("hi"));
    }

    @Test
    public void multimodalContentPartsForwardedVerbatim() throws IOException {
        String request = "{\"messages\":[{\"role\":\"user\",\"content\":["
                + "{\"type\":\"text\",\"text\":\"What color?\"},"
                + "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}}]}]}";
        JsonNode expectedMessages = MAPPER.readTree(request).path("messages");

        JsonNode out = mapAndSerialize(request);

        assertThat(out.path("messages"), is(expectedMessages));
    }

    @Test
    public void toolMessageHistoryRoundTripsVerbatim() throws IOException {
        // A full agent-loop history: assistant tool_calls + a role:"tool" result with tool_call_id.
        String request = "{\"messages\":["
                + "{\"role\":\"user\",\"content\":\"weather?\"},"
                + "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]},"
                + "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"sunny\"}]}";
        JsonNode out = mapAndSerialize(request);
        JsonNode messages = out.path("messages");
        assertThat(messages.size(), is(3));
        assertThat(messages.get(1).path("tool_calls").get(0).path("id").asText(), is("c1"));
        assertThat(messages.get(2).path("role").asText(), is("tool"));
        assertThat(messages.get(2).path("tool_call_id").asText(), is("c1"));
    }

    @Test
    public void missingMessagesThrows() throws IOException {
        JsonNode request = MAPPER.readTree("{\"temperature\":0.5}");
        assertThrows(IllegalArgumentException.class, () -> mapper.toInferenceParameters(request));
    }

    @Test
    public void emptyMessagesThrows() throws IOException {
        JsonNode request = MAPPER.readTree("{\"messages\":[]}");
        assertThrows(IllegalArgumentException.class, () -> mapper.toInferenceParameters(request));
    }

    @Test
    public void samplingFieldsMapped() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"temperature\":0.7,\"top_p\":0.9,\"top_k\":40,\"seed\":42,\"max_tokens\":128}");
        assertThat(out.path("temperature").asDouble(), is(closeTo(0.7, 1e-4)));
        assertThat(out.path("top_p").asDouble(), is(closeTo(0.9, 1e-4)));
        assertThat(out.path("top_k").asInt(), is(40));
        assertThat(out.path("seed").asInt(), is(42));
        assertThat(out.path("n_predict").asInt(), is(128));
    }

    // Upstream's oaicompat_chat_params_parse (server-common.cpp, b11639) turns logprobs=true into
    // n_probs = top_logprobs, 20 when top_logprobs is absent; the native layer then reports them.

    @Test
    public void logprobsWithTopLogprobsBecomesNProbs() throws IOException {
        JsonNode out = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," + "\"logprobs\":true,\"top_logprobs\":3}");
        assertThat(out.path("n_probs").asInt(-1), is(3));
    }

    @Test
    public void logprobsWithoutTopLogprobsDefaultsToTwentyLikeUpstream() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"logprobs\":true}");
        assertThat(out.path("n_probs").asInt(-1), is(20));
    }

    @Test
    public void logprobsFalseLeavesNProbsUnset() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"logprobs\":false}");
        assertThat(out.has("n_probs"), is(false));
    }

    @Test
    public void maxCompletionTokensPreferredOverMaxTokens() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"max_tokens\":50,\"max_completion_tokens\":200}");
        assertThat(out.path("n_predict").asInt(), is(200));
    }

    @Test
    public void toolsEnableChatTemplateAndForwardChoice() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\"}}],"
                + "\"tool_choice\":\"auto\"}");
        assertThat(out.path("tools").isArray(), is(true));
        assertThat(out.path("tools").get(0).path("function").path("name").asText(), is("read_file"));
        assertThat(out.path("tool_choice").asText(), is("auto"));
        // No "use_jinja" is sent: jinja is a load-time option (ModelParameters.enableJinja() /
        // --jinja), and upstream never reads a "use_jinja" key from a request body -- it would be
        // discarded silently. Native tool-call parsing depends on how the model was loaded.
        assertThat(out.has("use_jinja"), is(false));
    }

    @Test
    public void toolChoiceNamingAFunctionRequiresAToolCall() throws IOException {
        // {"type":"function","function":{"name":...}} asks for that tool; llama.cpp's chat parser reads
        // tool_choice only as a string and fell back to "auto". "required" is the closest it allows, as
        // for Anthropic's {"type":"tool"}.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}],"
                + "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}");
        assertThat(out.path("tool_choice").asText(), is("required"));
    }

    @Test
    public void parallelToolCallsForwarded() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}}],"
                + "\"parallel_tool_calls\":false}");
        assertThat(out.path("parallel_tool_calls").asBoolean(), is(false));
    }

    @Test
    public void stopAsSingleStringMapped() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":\"END\"}");
        assertThat(out.path("stop").isArray(), is(true));
        assertThat(out.path("stop").get(0).asText(), is("END"));
    }

    @Test
    public void stopAsArrayMapped() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":[\"A\",\"B\"]}");
        assertThat(out.path("stop").size(), is(2));
    }

    @Test
    public void streamOptionsForwardedVerbatim() throws IOException {
        // include_usage must reach the native layer so the trailing usage chunk is emitted.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"stream_options\":{\"include_usage\":true}}");
        assertThat(out.path("stream_options").path("include_usage").asBoolean(), is(true));
    }

    @Test
    public void responseFormatForwardedVerbatim() throws IOException {
        // Structured outputs: json_object / json_schema must reach the native grammar constraint.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"response_format\":{\"type\":\"json_object\"}}");
        assertThat(out.path("response_format").path("type").asText(), is("json_object"));
    }

    @Test
    public void cachePromptDefaultedTrue() throws IOException {
        // The mapper defaults cache_prompt=true so the slot KV prefix is reused across IDE turns.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        assertThat(out.path("cache_prompt").asBoolean(), is(true));
    }

    @Test
    public void cacheAndSlotExtensionsForwarded() throws IOException {
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"cache_prompt\":false,\"n_cache_reuse\":128,\"id_slot\":2}");
        assertThat(out.path("cache_prompt").asBoolean(), is(false));
        assertThat(out.path("n_cache_reuse").asInt(), is(128));
        assertThat(out.path("id_slot").asInt(), is(2));
    }

    @Test
    public void slotMinusOneMeansAnySlotLikeUpstream() throws IOException {
        // upstream reads id_slot with -1 as its default ("any slot"); a client echoing that default must
        // not get a 400 here, which the non-streamed /v1/completions (forwarded verbatim) never gives.
        JsonNode request = MAPPER.readTree("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"id_slot\":-1}");
        InferenceParameters params = assertDoesNotThrow(() -> mapper.toInferenceParameters(request));
        assertThat(MAPPER.readTree(params.toJson()).path("id_slot").asInt(-1), is(-1));
    }

    @Test
    public void zeroTokenCapOnlyWarmsTheCacheLikeUpstream() throws IOException {
        // upstream: "When 0, no tokens will be generated but the prompt is evaluated into the cache". Dropping
        // a 0 made the mapped routes generate without a cap, unlike the verbatim /v1/completions.
        JsonNode out = mapAndSerialize("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":0}");
        assertThat(out.path("n_predict").asInt(-1), is(0));
    }

    @Test
    public void llamaCppSamplingExtensionsReachTheMappedRequest() throws IOException {
        // Upstream's schema reads these fields from a chat body too; the mapped chat routes must not
        // answer the same request with other sampling.
        JsonNode out = MAPPER.readTree(
                mapper.toInferenceParameters(MAPPER.readTree("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                                + "\"min_p\":0.05,\"typical_p\":0.9,\"repeat_penalty\":1.1,\"repeat_last_n\":32,"
                                + "\"top_n_sigma\":1.5,\"xtc_probability\":0.3,\"xtc_threshold\":0.2,"
                                + "\"dynatemp_range\":0.4,\"dynatemp_exponent\":1.2,\"min_keep\":2,\"n_keep\":8,"
                                + "\"ignore_eos\":true,\"grammar\":\"root ::= \\\"a\\\"\"}"))
                        .toJson());
        assertThat(out.path("min_p").asDouble(), is(closeTo(0.05, 1e-4)));
        assertThat(out.path("typical_p").asDouble(), is(closeTo(0.9, 1e-4)));
        assertThat(out.path("repeat_penalty").asDouble(), is(closeTo(1.1, 1e-4)));
        assertThat(out.path("repeat_last_n").asInt(), is(32));
        assertThat(out.path("top_n_sigma").asDouble(), is(closeTo(1.5, 1e-4)));
        assertThat(out.path("xtc_probability").asDouble(), is(closeTo(0.3, 1e-4)));
        assertThat(out.path("xtc_threshold").asDouble(), is(closeTo(0.2, 1e-4)));
        assertThat(out.path("dynatemp_range").asDouble(), is(closeTo(0.4, 1e-4)));
        assertThat(out.path("dynatemp_exponent").asDouble(), is(closeTo(1.2, 1e-4)));
        assertThat(out.path("min_keep").asInt(), is(2));
        assertThat(out.path("n_keep").asInt(), is(8));
        assertThat(out.path("ignore_eos").asBoolean(false), is(true));
        assertThat(out.path("grammar").asText(), is("root ::= \"a\""));
    }

    @Test
    public void nPredictWinsOverMaxTokensLikeUpstream() throws IOException {
        // upstream reads n_predict with max_completion_tokens and max_tokens as aliases, n_predict first.
        JsonNode out = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50,\"n_predict\":7}");
        assertThat(out.path("n_predict").asInt(), is(7));
    }

    @Test
    public void jsonSchemaReachesTheMappedRequestUnlessAGrammarIsGiven() throws IOException {
        JsonNode schema = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"json_schema\":{\"type\":\"object\"}}");
        assertThat(schema.path("json_schema").path("type").asText(), is("object"));
        // upstream applies json_schema only without a grammar.
        JsonNode both = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"grammar\":\"root ::= \\\"a\\\"\",\"json_schema\":{\"type\":\"object\"}}");
        assertThat(both.has("json_schema"), is(false));
        assertThat(both.path("grammar").asText(), is("root ::= \"a\""));
    }

    @Test
    public void unknownFieldsIgnored() throws IOException {
        JsonNode out = mapAndSerialize(
                // "n" is not unknown: more than one answer is refused (moreThanOneChoiceIsRefused...)
                "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," + "\"some_future_field\":true}");
        assertThat(out.path("messages").isArray(), is(true));
        assertThat(out.has("some_future_field"), is(false));
    }
}
