// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
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
        return MAPPER.readTree(params.toString());
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
    public void unknownFieldsIgnored() throws IOException {
        JsonNode out = mapAndSerialize(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," + "\"some_future_field\":true,\"n\":3}");
        assertThat(out.path("messages").isArray(), is(true));
        assertThat(out.has("some_future_field"), is(false));
    }
}
