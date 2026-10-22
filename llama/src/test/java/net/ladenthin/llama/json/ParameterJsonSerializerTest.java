// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.args.Sampler;
import net.ladenthin.llama.parameters.ParameterJsonSerializer;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.Pair;
import net.ladenthin.llama.value.ToolCall;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Unit tests for {@link ParameterJsonSerializer}.
 * No JVM native library or model file needed — plain Java values only.
 */
public class ParameterJsonSerializerTest {

    private final ParameterJsonSerializer serializer = new ParameterJsonSerializer();

    // ------------------------------------------------------------------
    // toJsonString
    // ------------------------------------------------------------------

    @Test
    public void testToJsonString_simple() {
        assertThat(serializer.toJsonString("hello"), is("\"hello\""));
    }

    @Test
    public void testToJsonString_null() {
        assertThat(serializer.toJsonString(null), is("null"));
    }

    @Test
    public void testToJsonString_emptyString() {
        assertThat(serializer.toJsonString(""), is("\"\""));
    }

    @Test
    public void testToJsonString_newline() {
        assertThat(serializer.toJsonString("line1\nline2"), is("\"line1\\nline2\""));
    }

    @Test
    public void testToJsonString_tab() {
        assertThat(serializer.toJsonString("a\tb"), is("\"a\\tb\""));
    }

    @Test
    public void testToJsonString_quote() {
        assertThat(serializer.toJsonString("say \"hi\""), is("\"say \\\"hi\\\"\""));
    }

    @Test
    public void testToJsonString_backslash() {
        assertThat(serializer.toJsonString("path\\file"), is("\"path\\\\file\""));
    }

    @Test
    public void testToJsonString_unicode() {
        assertThat(serializer.toJsonString("café"), is("\"café\""));
    }

    // ------------------------------------------------------------------
    // buildMessages
    // ------------------------------------------------------------------

    /**
     * Upstream parses every {@code function.arguments} of the history with {@code json::parse}
     * (templates with object arguments, Qwen and Llama 3 among them), so an empty string failed the
     * whole follow-up request with 400 "Failed to parse tool call arguments as JSON" (measured at
     * b11639); {@code "{}"} goes through. An empty string can come from a client or from an
     * accumulator that saw no argument delta.
     */
    @ParameterizedTest
    @ValueSource(strings = {"", " ", "\n"})
    public void testBuildMessages_emptyToolCallArguments_areWrittenAsAnEmptyObject(String arguments) {
        ArrayNode arr = serializer.buildMessages(Collections.singletonList(ChatMessage.assistantToolCalls(
                "", Collections.singletonList(new ToolCall("call_1", "get_time", arguments)))));
        JsonNode function = arr.get(0).get("tool_calls").get(0).get("function");
        assertThat(function.get("arguments").asText(), is("{}"));
    }

    @Test
    public void testBuildMessages_toolCallArguments_arePassedOnVerbatim() {
        ArrayNode arr = serializer.buildMessages(Collections.singletonList(ChatMessage.assistantToolCalls(
                "", Collections.singletonList(new ToolCall("call_1", "get_weather", "{\"city\":\"Berlin\"}")))));
        JsonNode function = arr.get(0).get("tool_calls").get(0).get("function");
        assertThat(function.get("arguments").asText(), is("{\"city\":\"Berlin\"}"));
    }

    @Test
    public void testBuildMessages_withSystemMessage() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hello"));
        ArrayNode arr = serializer.buildMessages("You are helpful.", msgs);
        assertThat(arr.size(), is(2));
        assertThat(arr.get(0).path("role").asText(), is("system"));
        assertThat(arr.get(0).path("content").asText(), is("You are helpful."));
        assertThat(arr.get(1).path("role").asText(), is("user"));
        assertThat(arr.get(1).path("content").asText(), is("Hello"));
    }

    @Test
    public void testBuildMessages_withoutSystemMessage() {
        List<Pair<String, String>> msgs =
                Arrays.asList(new Pair<>("user", "Hi"), new Pair<>("assistant", "Hello there"));
        ArrayNode arr = serializer.buildMessages(null, msgs);
        assertThat(arr.size(), is(2));
        assertThat(arr.get(0).path("role").asText(), is("user"));
        assertThat(arr.get(1).path("role").asText(), is("assistant"));
    }

    @Test
    public void testBuildMessages_emptySystemMessage_skipped() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hi"));
        ArrayNode arr = serializer.buildMessages("", msgs);
        assertThat(arr.size(), is(1));
        assertThat(arr.get(0).path("role").asText(), is("user"));
    }

    @Test
    public void testBuildMessages_specialCharsInContent() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "line1\nline2\t\"quoted\""));
        ArrayNode arr = serializer.buildMessages(null, msgs);
        assertThat(arr.get(0).path("content").asText(), is("line1\nline2\t\"quoted\""));
    }

    @Test
    public void testBuildMessages_invalidRole_throws() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("system", "oops"));
        assertThrows(IllegalArgumentException.class, () -> serializer.buildMessages(null, msgs));
    }

    @Test
    public void testBuildMessages_roundtripsAsJson() throws Exception {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hello"));
        ArrayNode arr = serializer.buildMessages("Sys", msgs);
        String json = arr.toString();
        JsonNode parsed = serializer.OBJECT_MAPPER.readTree(json);
        assertThat(parsed.get(0).path("role").asText(), is("system"));
        assertThat(parsed.get(0).path("content").asText(), is("Sys"));
        assertThat(parsed.get(1).path("role").asText(), is("user"));
        assertThat(parsed.get(1).path("content").asText(), is("Hello"));
    }

    // ------------------------------------------------------------------
    // buildStopStrings
    // ------------------------------------------------------------------

    @Test
    public void testBuildStopStrings_single() {
        ArrayNode arr = serializer.buildStopStrings("<|endoftext|>");
        assertThat(arr.size(), is(1));
        assertThat(arr.get(0).asText(), is("<|endoftext|>"));
    }

    @Test
    public void testBuildStopStrings_multiple() {
        ArrayNode arr = serializer.buildStopStrings("stop1", "stop2", "stop3");
        assertThat(arr.size(), is(3));
        assertThat(arr.get(0).asText(), is("stop1"));
        assertThat(arr.get(2).asText(), is("stop3"));
    }

    @Test
    public void testBuildStopStrings_withSpecialChars() {
        ArrayNode arr = serializer.buildStopStrings("line\nnewline", "tab\there");
        assertThat(arr.get(0).asText(), is("line\nnewline"));
        assertThat(arr.get(1).asText(), is("tab\there"));
    }

    @Test
    public void testBuildStopStrings_roundtripsAsJson() throws Exception {
        ArrayNode arr = serializer.buildStopStrings("a", "b");
        JsonNode parsed = serializer.OBJECT_MAPPER.readTree(arr.toString());
        assertThat(parsed.isArray(), is(true));
        assertThat(parsed.get(0).asText(), is("a"));
    }

    // ------------------------------------------------------------------
    // buildSamplers
    // ------------------------------------------------------------------

    @Test
    public void testBuildSamplers_allTypes() {
        ArrayNode arr = serializer.buildSamplers(Sampler.TOP_K, Sampler.TOP_P, Sampler.MIN_P, Sampler.TEMPERATURE);
        assertThat(arr.size(), is(4));
        assertThat(arr.get(0).asText(), is("top_k"));
        assertThat(arr.get(1).asText(), is("top_p"));
        assertThat(arr.get(2).asText(), is("min_p"));
        assertThat(arr.get(3).asText(), is("temperature"));
    }

    @Test
    public void testBuildSamplers_single() {
        ArrayNode arr = serializer.buildSamplers(Sampler.TEMPERATURE);
        assertThat(arr.size(), is(1));
        assertThat(arr.get(0).asText(), is("temperature"));
    }

    // ------------------------------------------------------------------
    // buildTokenIdBiasArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildTokenIdBiasArray_structure() {
        Map<Integer, Float> biases = new LinkedHashMap<>();
        biases.put(15043, 1.0f);
        biases.put(50256, -0.5f);
        ArrayNode arr = serializer.buildTokenIdBiasArray(biases);
        assertThat(arr.size(), is(2));
        assertThat(arr.get(0).get(0).asInt(), is(15043));
        assertEquals(1.0, arr.get(0).get(1).asDouble(), 0.001);
        assertThat(arr.get(1).get(0).asInt(), is(50256));
        assertEquals(-0.5, arr.get(1).get(1).asDouble(), 0.001);
    }

    @Test
    public void testBuildTokenIdBiasArray_negativeInfinityBansTheToken() {
        // JSON has no -Infinity: Jackson wrote the string "-Infinity", which llama.cpp's logit_bias
        // parser skips (only a number or false is a bias), so the ban silently did nothing. false is
        // llama.cpp's spelling of -inf.
        ArrayNode ids = serializer.buildTokenIdBiasArray(Collections.singletonMap(15043, Float.NEGATIVE_INFINITY));
        assertThat(ids.get(0).get(1).isBoolean(), is(true));
        assertThat(ids.get(0).get(1).asBoolean(), is(false));
        ArrayNode strings =
                serializer.buildTokenStringBiasArray(Collections.singletonMap("Hello", Float.NEGATIVE_INFINITY));
        assertThat(strings.get(0).get(1).isBoolean(), is(true));
        assertThat(strings.get(0).get(1).asBoolean(), is(false));
    }

    @Test
    public void testBuildTokenIdBiasArray_nanOrPositiveInfinityIsRejected() {
        // Neither has a JSON spelling llama.cpp accepts; silently dropping the entry hides the mistake.
        assertThrows(
                IllegalArgumentException.class,
                () -> serializer.buildTokenIdBiasArray(Collections.singletonMap(1, Float.NaN)));
        assertThrows(
                IllegalArgumentException.class,
                () -> serializer.buildTokenStringBiasArray(Collections.singletonMap("a", Float.POSITIVE_INFINITY)));
    }

    @Test
    public void testBuildTokenIdBiasArray_empty() {
        ArrayNode arr = serializer.buildTokenIdBiasArray(Collections.emptyMap());
        assertThat(arr.size(), is(0));
    }

    // ------------------------------------------------------------------
    // buildTokenStringBiasArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildTokenStringBiasArray_structure() {
        Map<String, Float> biases = new LinkedHashMap<>();
        biases.put("Hello", 1.0f);
        biases.put(" world", -0.5f);
        ArrayNode arr = serializer.buildTokenStringBiasArray(biases);
        assertThat(arr.size(), is(2));
        assertThat(arr.get(0).get(0).asText(), is("Hello"));
        assertEquals(1.0, arr.get(0).get(1).asDouble(), 0.001);
        assertThat(arr.get(1).get(0).asText(), is(" world"));
    }

    @Test
    public void testBuildTokenStringBiasArray_specialCharsInKey() {
        Map<String, Float> biases = new LinkedHashMap<>();
        biases.put("line\nnewline", 2.0f);
        ArrayNode arr = serializer.buildTokenStringBiasArray(biases);
        assertThat(arr.get(0).get(0).asText(), is("line\nnewline"));
    }

    // ------------------------------------------------------------------
    // buildDisableTokenIdArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildDisableTokenIdArray_structure() {
        ArrayNode arr = serializer.buildDisableTokenIdArray(Arrays.asList(100, 200, 300));
        assertThat(arr.size(), is(3));
        for (int i = 0; i < arr.size(); i++) {
            assertThat(arr.get(i).get(1).asBoolean(), is(false));
        }
        assertThat(arr.get(0).get(0).asInt(), is(100));
    }

    @Test
    public void testBuildDisableTokenIdArray_empty() {
        ArrayNode arr = serializer.buildDisableTokenIdArray(Collections.emptyList());
        assertThat(arr.size(), is(0));
    }

    // ------------------------------------------------------------------
    // buildDisableTokenStringArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildDisableTokenStringArray_structure() {
        ArrayNode arr = serializer.buildDisableTokenStringArray(Arrays.asList("foo", "bar"));
        assertThat(arr.size(), is(2));
        assertThat(arr.get(0).get(0).asText(), is("foo"));
        assertThat(arr.get(0).get(1).asBoolean(), is(false));
        assertThat(arr.get(1).get(0).asText(), is("bar"));
    }

    // ------------------------------------------------------------------
    // buildRawValueObject
    // ------------------------------------------------------------------

    @Test
    public void testBuildRawValueObject_booleanValue() {
        Map<String, String> map = Collections.singletonMap("enable_thinking", "true");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertThat(node.path("enable_thinking").isBoolean(), is(true));
        assertThat(node.path("enable_thinking").asBoolean(), is(true));
    }

    @Test
    public void testBuildRawValueObject_numberValue() {
        Map<String, String> map = Collections.singletonMap("temperature", "0.7");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertEquals(node.path("temperature").asDouble(), 0.001, 0.7);
    }

    @Test
    public void testBuildRawValueObject_stringValue() {
        Map<String, String> map = Collections.singletonMap("mode", "\"fast\"");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertThat(node.path("mode").asText(), is("fast"));
    }

    @Test
    public void testBuildRawValueObject_invalidJsonFallsBackToString() {
        Map<String, String> map = Collections.singletonMap("key", "not-valid-json{{{");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertThat(node.path("key").asText(), is("not-valid-json{{{"));
    }

    /**
     * A value holding more than one JSON value is no JSON value either. The plain readTree parsed the
     * first and dropped the rest without a word: "true false" became true, "1, 2" became 1.
     */
    @Test
    public void testBuildRawValueObject_trailingTokensFallBackToString() {
        Map<String, String> map = new LinkedHashMap<>();
        map.put("two", "true false");
        map.put("list", "1, 2");
        map.put("object", "{\"a\":1} {\"b\":2}");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertThat(
                node.toString(),
                is("{\"two\":\"true false\",\"list\":\"1, 2\",\"object\":\"{\\\"a\\\":1} {\\\"b\\\":2}\"}"));
    }

    /** Jackson parses an empty or blank text to MissingNode, which would be written as null. */
    @Test
    public void testBuildRawValueObject_emptyOrBlankValueFallsBackToString() {
        Map<String, String> map = new LinkedHashMap<>();
        map.put("empty", "");
        map.put("blank", "   ");
        ObjectNode node = serializer.buildRawValueObject(map);
        assertThat(node.toString(), is("{\"empty\":\"\",\"blank\":\"   \"}"));
    }

    @Test
    public void testBuildRawValueObject_roundtripsAsJson() throws Exception {
        Map<String, String> map = new LinkedHashMap<>();
        map.put("flag", "true");
        map.put("count", "3");
        ObjectNode node = serializer.buildRawValueObject(map);
        JsonNode parsed = serializer.OBJECT_MAPPER.readTree(node.toString());
        assertThat(parsed.path("flag").asBoolean(), is(true));
        assertThat(parsed.path("count").asInt(), is(3));
    }
}
