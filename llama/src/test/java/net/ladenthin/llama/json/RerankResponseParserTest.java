// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Test;

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
        assertThat(result, hasSize(1));
        assertThat(result.get(0).getKey(), is("The quick brown fox"));
        assertEquals(0.92f, result.get(0).getValue(), 0.001f);
    }

    @Test
    public void testParseString_multipleEntries() {
        String json = "[" + "{\"document\":\"First\",\"index\":0,\"score\":0.9},"
                + "{\"document\":\"Second\",\"index\":1,\"score\":0.5},"
                + "{\"document\":\"Third\",\"index\":2,\"score\":0.1}"
                + "]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertThat(result, hasSize(3));
        assertThat(result.get(0).getKey(), is("First"));
        assertThat(result.get(1).getKey(), is("Second"));
        assertThat(result.get(2).getKey(), is("Third"));
        assertEquals(0.9f, result.get(0).getValue(), 0.001f);
        assertEquals(0.5f, result.get(1).getValue(), 0.001f);
        assertEquals(0.1f, result.get(2).getValue(), 0.001f);
    }

    @Test
    public void testParseString_emptyArray() {
        List<Pair<String, Float>> result = parser.parse("[]");
        assertThat(result, is(empty()));
    }

    @Test
    public void testParseString_malformed() {
        List<Pair<String, Float>> result = parser.parse("{not json");
        assertThat(result, is(empty()));
    }

    @Test
    public void testParseString_notAnArray() {
        List<Pair<String, Float>> result = parser.parse("{\"document\":\"x\",\"score\":0.5}");
        assertThat(result, is(empty()));
    }

    @Test
    public void testParseString_documentWithSpecialChars() {
        String json = "[{\"document\":\"line1\\nline2\\t\\\"quoted\\\"\",\"index\":0,\"score\":0.75}]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertThat(result, hasSize(1));
        assertThat(result.get(0).getKey(), is("line1\nline2\t\"quoted\""));
    }

    @Test
    public void testParseString_scoreZero() {
        String json = "[{\"document\":\"irrelevant\",\"index\":0,\"score\":0.0}]";
        List<Pair<String, Float>> result = parser.parse(json);
        assertThat(result, hasSize(1));
        assertEquals(0.0f, result.get(0).getValue(), 0.001f);
    }

    // ------------------------------------------------------------------
    // parse(JsonNode)
    // ------------------------------------------------------------------

    @Test
    public void testParseNode_preservesOrder() throws Exception {
        String json = "[" + "{\"document\":\"A\",\"index\":0,\"score\":0.8},"
                + "{\"document\":\"B\",\"index\":1,\"score\":0.3}"
                + "]";
        JsonNode arr = MAPPER.readTree(json);
        List<Pair<String, Float>> result = parser.parse(arr);
        assertThat(result, hasSize(2));
        assertThat(result.get(0).getKey(), is("A"));
        assertThat(result.get(1).getKey(), is("B"));
    }

    @Test
    public void testParseNode_notArray() throws Exception {
        JsonNode obj = MAPPER.readTree("{\"document\":\"x\",\"score\":0.5}");
        assertThat(parser.parse(obj), is(empty()));
    }

    @Test
    public void testParseNode_notArray_returnsMutableEmptyList() throws Exception {
        // The non-array branch returns a MUTABLE empty list (matches the non-empty path,
        // for Error Prone MixedMutabilityReturnType). Mutating it must succeed — which also
        // kills the EmptyObjectReturnVals mutant that would return an immutable emptyList().
        JsonNode obj = MAPPER.readTree("{\"document\":\"x\",\"score\":0.5}");
        List<Pair<String, Float>> result = parser.parse(obj);
        assertThat(result, is(empty()));
        result.add(new Pair<>("added", 1.0f));
        assertThat(result, hasSize(1));
    }

    @Test
    public void testParseNode_missingScore_defaultsToZero() throws Exception {
        JsonNode arr = MAPPER.readTree("[{\"document\":\"doc\",\"index\":0}]");
        List<Pair<String, Float>> result = parser.parse(arr);
        assertThat(result, hasSize(1));
        assertEquals(0.0f, result.get(0).getValue(), 0.001f);
    }

    @Test
    public void testParseNode_missingDocument_defaultsToEmpty() throws Exception {
        JsonNode arr = MAPPER.readTree("[{\"index\":0,\"score\":0.5}]");
        List<Pair<String, Float>> result = parser.parse(arr);
        assertThat(result, hasSize(1));
        assertThat(result.get(0).getKey(), is(""));
    }
}
