package de.kherud.llama.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.kherud.llama.Pair;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link RerankResponseParser}.
 * No JVM native library or model file needed — JSON string literals only.
 */
public class RerankResponseParserTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final RerankResponseParser parser = new RerankResponseParser();

    // ------------------------------------------------------------------
    // parse(String)
    // ------------------------------------------------------------------

    @Test
    public void testParseString_singleEntry() {
        String json = "[{\"document\":\"The quick brown fox\",\"index\":0,\"score\":0.92}]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertEquals(1, result.size());
        assertEquals("The quick brown fox", result.get(0).getKey());
        assertEquals(0.92f, result.get(0).getValue(), 0.001f);
    }

    @Test
    public void testParseString_multipleEntries() {
        String json = "[" +
                "{\"document\":\"First\",\"index\":0,\"score\":0.9}," +
                "{\"document\":\"Second\",\"index\":1,\"score\":0.5}," +
                "{\"document\":\"Third\",\"index\":2,\"score\":0.1}" +
                "]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertEquals(3, result.size());
        assertEquals("First",  result.get(0).getKey());
        assertEquals("Second", result.get(1).getKey());
        assertEquals("Third",  result.get(2).getKey());
        assertEquals(0.9f, result.get(0).getValue(), 0.001f);
        assertEquals(0.5f, result.get(1).getValue(), 0.001f);
        assertEquals(0.1f, result.get(2).getValue(), 0.001f);
    }

    @Test
    public void testParseString_emptyArray() {
        List<Pair<String, Float>> result = parser.parse("[]");
        assertTrue(result.isEmpty());
    }

    @Test
    public void testParseString_malformed() {
        List<Pair<String, Float>> result = parser.parse("{not json");
        assertTrue(result.isEmpty());
    }

    @Test
    public void testParseString_notAnArray() {
        List<Pair<String, Float>> result = parser.parse("{\"document\":\"x\",\"score\":0.5}");
        assertTrue(result.isEmpty());
    }

    @Test
    public void testParseString_documentWithSpecialChars() {
        String json = "[{\"document\":\"line1\\nline2\\t\\\"quoted\\\"\",\"index\":0,\"score\":0.75}]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertEquals(1, result.size());
        assertEquals("line1\nline2\t\"quoted\"", result.get(0).getKey());
    }

    @Test
    public void testParseString_scoreZero() {
        String json = "[{\"document\":\"irrelevant\",\"index\":0,\"score\":0.0}]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertEquals(1, result.size());
        assertEquals(0.0f, result.get(0).getValue(), 0.001f);
    }

    // ------------------------------------------------------------------
    // parse(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testParseNode_preservesOrder() throws Exception {
        String json = "[" +
                "{\"document\":\"A\",\"index\":0,\"score\":0.8}," +
                "{\"document\":\"B\",\"index\":1,\"score\":0.3}" +
                "]";
        JsonNode arr = MAPPER.readTree(json);
        List<Pair<String, Float>> result = parser.parse(arr);
        assertEquals(2, result.size());
        assertEquals("A", result.get(0).getKey());
        assertEquals("B", result.get(1).getKey());
    }

    @Test
    public void testParseNode_notArray() throws Exception {
        JsonNode obj = MAPPER.readTree("{\"document\":\"x\",\"score\":0.5}");
        assertTrue(parser.parse(obj).isEmpty());
    }

    @Test
    public void testParseNode_missingScore_defaultsToZero() throws Exception {
        JsonNode arr = MAPPER.readTree("[{\"document\":\"doc\",\"index\":0}]");
        List<Pair<String, Float>> result = parser.parse(arr);
        assertEquals(1, result.size());
        assertEquals(0.0f, result.get(0).getValue(), 0.001f);
    }

    @Test
    public void testParseNode_missingDocument_defaultsToEmpty() throws Exception {
        JsonNode arr = MAPPER.readTree("[{\"index\":0,\"score\":0.5}]");
        List<Pair<String, Float>> result = parser.parse(arr);
        assertEquals(1, result.size());
        assertEquals("", result.get(0).getKey());
    }
}
