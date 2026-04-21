package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.kherud.llama.Pair;
import de.kherud.llama.args.Sampler;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link ParameterJsonSerializer}.
 * No JVM native library or model file needed — plain Java values only.
 */
public class ParameterJsonSerializerTest {

    // ------------------------------------------------------------------
    // toJsonString
    // ------------------------------------------------------------------

    @Test
    public void testToJsonString_simple() {
        assertEquals("\"hello\"", ParameterJsonSerializer.toJsonString("hello"));
    }

    @Test
    public void testToJsonString_null() {
        assertEquals("null", ParameterJsonSerializer.toJsonString(null));
    }

    @Test
    public void testToJsonString_emptyString() {
        assertEquals("\"\"", ParameterJsonSerializer.toJsonString(""));
    }

    @Test
    public void testToJsonString_newline() {
        assertEquals("\"line1\\nline2\"", ParameterJsonSerializer.toJsonString("line1\nline2"));
    }

    @Test
    public void testToJsonString_tab() {
        assertEquals("\"a\\tb\"", ParameterJsonSerializer.toJsonString("a\tb"));
    }

    @Test
    public void testToJsonString_quote() {
        assertEquals("\"say \\\"hi\\\"\"", ParameterJsonSerializer.toJsonString("say \"hi\""));
    }

    @Test
    public void testToJsonString_backslash() {
        assertEquals("\"path\\\\file\"", ParameterJsonSerializer.toJsonString("path\\file"));
    }

    @Test
    public void testToJsonString_unicode() {
        assertEquals("\"café\"", ParameterJsonSerializer.toJsonString("café"));
    }

    // ------------------------------------------------------------------
    // buildMessages
    // ------------------------------------------------------------------

    @Test
    public void testBuildMessages_withSystemMessage() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hello"));
        ArrayNode arr = ParameterJsonSerializer.buildMessages("You are helpful.", msgs);
        assertEquals(2, arr.size());
        assertEquals("system", arr.get(0).path("role").asText());
        assertEquals("You are helpful.", arr.get(0).path("content").asText());
        assertEquals("user", arr.get(1).path("role").asText());
        assertEquals("Hello", arr.get(1).path("content").asText());
    }

    @Test
    public void testBuildMessages_withoutSystemMessage() {
        List<Pair<String, String>> msgs = Arrays.asList(
                new Pair<>("user", "Hi"),
                new Pair<>("assistant", "Hello there")
        );
        ArrayNode arr = ParameterJsonSerializer.buildMessages(null, msgs);
        assertEquals(2, arr.size());
        assertEquals("user", arr.get(0).path("role").asText());
        assertEquals("assistant", arr.get(1).path("role").asText());
    }

    @Test
    public void testBuildMessages_emptySystemMessage_skipped() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hi"));
        ArrayNode arr = ParameterJsonSerializer.buildMessages("", msgs);
        assertEquals(1, arr.size());
        assertEquals("user", arr.get(0).path("role").asText());
    }

    @Test
    public void testBuildMessages_specialCharsInContent() {
        List<Pair<String, String>> msgs = Collections.singletonList(
                new Pair<>("user", "line1\nline2\t\"quoted\""));
        ArrayNode arr = ParameterJsonSerializer.buildMessages(null, msgs);
        assertEquals("line1\nline2\t\"quoted\"", arr.get(0).path("content").asText());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBuildMessages_invalidRole_throws() {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("system", "oops"));
        ParameterJsonSerializer.buildMessages(null, msgs);
    }

    @Test
    public void testBuildMessages_roundtripsAsJson() throws Exception {
        List<Pair<String, String>> msgs = Collections.singletonList(new Pair<>("user", "Hello"));
        ArrayNode arr = ParameterJsonSerializer.buildMessages("Sys", msgs);
        String json = arr.toString();
        JsonNode parsed = ParameterJsonSerializer.OBJECT_MAPPER.readTree(json);
        assertEquals("system", parsed.get(0).path("role").asText());
        assertEquals("Sys", parsed.get(0).path("content").asText());
        assertEquals("user", parsed.get(1).path("role").asText());
        assertEquals("Hello", parsed.get(1).path("content").asText());
    }

    // ------------------------------------------------------------------
    // buildStopStrings
    // ------------------------------------------------------------------

    @Test
    public void testBuildStopStrings_single() {
        ArrayNode arr = ParameterJsonSerializer.buildStopStrings("<|endoftext|>");
        assertEquals(1, arr.size());
        assertEquals("<|endoftext|>", arr.get(0).asText());
    }

    @Test
    public void testBuildStopStrings_multiple() {
        ArrayNode arr = ParameterJsonSerializer.buildStopStrings("stop1", "stop2", "stop3");
        assertEquals(3, arr.size());
        assertEquals("stop1", arr.get(0).asText());
        assertEquals("stop3", arr.get(2).asText());
    }

    @Test
    public void testBuildStopStrings_withSpecialChars() {
        ArrayNode arr = ParameterJsonSerializer.buildStopStrings("line\nnewline", "tab\there");
        assertEquals("line\nnewline", arr.get(0).asText());
        assertEquals("tab\there", arr.get(1).asText());
    }

    @Test
    public void testBuildStopStrings_roundtripsAsJson() throws Exception {
        ArrayNode arr = ParameterJsonSerializer.buildStopStrings("a", "b");
        JsonNode parsed = ParameterJsonSerializer.OBJECT_MAPPER.readTree(arr.toString());
        assertTrue(parsed.isArray());
        assertEquals("a", parsed.get(0).asText());
    }

    // ------------------------------------------------------------------
    // buildSamplers
    // ------------------------------------------------------------------

    @Test
    public void testBuildSamplers_allTypes() {
        ArrayNode arr = ParameterJsonSerializer.buildSamplers(
                Sampler.TOP_K, Sampler.TOP_P, Sampler.MIN_P, Sampler.TEMPERATURE);
        assertEquals(4, arr.size());
        assertEquals("top_k", arr.get(0).asText());
        assertEquals("top_p", arr.get(1).asText());
        assertEquals("min_p", arr.get(2).asText());
        assertEquals("temperature", arr.get(3).asText());
    }

    @Test
    public void testBuildSamplers_single() {
        ArrayNode arr = ParameterJsonSerializer.buildSamplers(Sampler.TEMPERATURE);
        assertEquals(1, arr.size());
        assertEquals("temperature", arr.get(0).asText());
    }

    // ------------------------------------------------------------------
    // buildIntArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildIntArray_values() {
        ArrayNode arr = ParameterJsonSerializer.buildIntArray(new int[]{1, 2, 3});
        assertEquals(3, arr.size());
        assertEquals(1, arr.get(0).asInt());
        assertEquals(3, arr.get(2).asInt());
    }

    @Test
    public void testBuildIntArray_empty() {
        ArrayNode arr = ParameterJsonSerializer.buildIntArray(new int[]{});
        assertEquals(0, arr.size());
    }

    @Test
    public void testBuildIntArray_roundtripsAsJson() throws Exception {
        ArrayNode arr = ParameterJsonSerializer.buildIntArray(new int[]{10, 20});
        JsonNode parsed = ParameterJsonSerializer.OBJECT_MAPPER.readTree(arr.toString());
        assertTrue(parsed.isArray());
        assertEquals(10, parsed.get(0).asInt());
    }

    // ------------------------------------------------------------------
    // buildTokenIdBiasArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildTokenIdBiasArray_structure() {
        Map<Integer, Float> biases = new LinkedHashMap<>();
        biases.put(15043, 1.0f);
        biases.put(50256, -0.5f);
        ArrayNode arr = ParameterJsonSerializer.buildTokenIdBiasArray(biases);
        assertEquals(2, arr.size());
        assertEquals(15043, arr.get(0).get(0).asInt());
        assertEquals(1.0, arr.get(0).get(1).asDouble(), 0.001);
        assertEquals(50256, arr.get(1).get(0).asInt());
        assertEquals(-0.5, arr.get(1).get(1).asDouble(), 0.001);
    }

    @Test
    public void testBuildTokenIdBiasArray_empty() {
        ArrayNode arr = ParameterJsonSerializer.buildTokenIdBiasArray(Collections.emptyMap());
        assertEquals(0, arr.size());
    }

    // ------------------------------------------------------------------
    // buildTokenStringBiasArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildTokenStringBiasArray_structure() {
        Map<String, Float> biases = new LinkedHashMap<>();
        biases.put("Hello", 1.0f);
        biases.put(" world", -0.5f);
        ArrayNode arr = ParameterJsonSerializer.buildTokenStringBiasArray(biases);
        assertEquals(2, arr.size());
        assertEquals("Hello", arr.get(0).get(0).asText());
        assertEquals(1.0, arr.get(0).get(1).asDouble(), 0.001);
        assertEquals(" world", arr.get(1).get(0).asText());
    }

    @Test
    public void testBuildTokenStringBiasArray_specialCharsInKey() {
        Map<String, Float> biases = new LinkedHashMap<>();
        biases.put("line\nnewline", 2.0f);
        ArrayNode arr = ParameterJsonSerializer.buildTokenStringBiasArray(biases);
        assertEquals("line\nnewline", arr.get(0).get(0).asText());
    }

    // ------------------------------------------------------------------
    // buildDisableTokenIdArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildDisableTokenIdArray_structure() {
        ArrayNode arr = ParameterJsonSerializer.buildDisableTokenIdArray(Arrays.asList(100, 200, 300));
        assertEquals(3, arr.size());
        for (int i = 0; i < arr.size(); i++) {
            assertFalse(arr.get(i).get(1).asBoolean());
        }
        assertEquals(100, arr.get(0).get(0).asInt());
    }

    @Test
    public void testBuildDisableTokenIdArray_empty() {
        ArrayNode arr = ParameterJsonSerializer.buildDisableTokenIdArray(Collections.emptyList());
        assertEquals(0, arr.size());
    }

    // ------------------------------------------------------------------
    // buildDisableTokenStringArray
    // ------------------------------------------------------------------

    @Test
    public void testBuildDisableTokenStringArray_structure() {
        ArrayNode arr = ParameterJsonSerializer.buildDisableTokenStringArray(Arrays.asList("foo", "bar"));
        assertEquals(2, arr.size());
        assertEquals("foo", arr.get(0).get(0).asText());
        assertFalse(arr.get(0).get(1).asBoolean());
        assertEquals("bar", arr.get(1).get(0).asText());
    }

    // ------------------------------------------------------------------
    // buildRawValueObject
    // ------------------------------------------------------------------

    @Test
    public void testBuildRawValueObject_booleanValue() {
        Map<String, String> map = Collections.singletonMap("enable_thinking", "true");
        ObjectNode node = ParameterJsonSerializer.buildRawValueObject(map);
        assertTrue(node.path("enable_thinking").isBoolean());
        assertTrue(node.path("enable_thinking").asBoolean());
    }

    @Test
    public void testBuildRawValueObject_numberValue() {
        Map<String, String> map = Collections.singletonMap("temperature", "0.7");
        ObjectNode node = ParameterJsonSerializer.buildRawValueObject(map);
        assertEquals(0.7, node.path("temperature").asDouble(), 0.001);
    }

    @Test
    public void testBuildRawValueObject_stringValue() {
        Map<String, String> map = Collections.singletonMap("mode", "\"fast\"");
        ObjectNode node = ParameterJsonSerializer.buildRawValueObject(map);
        assertEquals("fast", node.path("mode").asText());
    }

    @Test
    public void testBuildRawValueObject_invalidJsonFallsBackToString() {
        Map<String, String> map = Collections.singletonMap("key", "not-valid-json{{{");
        ObjectNode node = ParameterJsonSerializer.buildRawValueObject(map);
        assertEquals("not-valid-json{{{", node.path("key").asText());
    }

    @Test
    public void testBuildRawValueObject_roundtripsAsJson() throws Exception {
        Map<String, String> map = new LinkedHashMap<>();
        map.put("flag", "true");
        map.put("count", "3");
        ObjectNode node = ParameterJsonSerializer.buildRawValueObject(map);
        JsonNode parsed = ParameterJsonSerializer.OBJECT_MAPPER.readTree(node.toString());
        assertTrue(parsed.path("flag").asBoolean());
        assertEquals(3, parsed.path("count").asInt());
    }
}
