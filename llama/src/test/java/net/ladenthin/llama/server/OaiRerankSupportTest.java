// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OaiRerankSupport}: request parsing and the native-array to OpenAI-rerank
 * reshape (sorting, {@code top_n}, the {@code results}/{@code data} alias). Pure — no model.
 */
public class OaiRerankSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    @Test
    public void readQueryReturnsText() throws IOException {
        assertThat(OaiRerankSupport.readQuery(read("{\"query\":\"hello\"}")), is("hello"));
    }

    @Test
    public void readQueryThrowsWhenMissing() throws IOException {
        JsonNode request = read("{\"documents\":[\"a\"]}");
        assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readQuery(request));
    }

    @Test
    public void readDocumentsAcceptsStrings() throws IOException {
        String[] docs = OaiRerankSupport.readDocuments(read("{\"documents\":[\"a\",\"b\"]}"));
        assertThat(docs.length, is(2));
        assertThat(docs[0], is("a"));
        assertThat(docs[1], is("b"));
    }

    @Test
    public void readDocumentsAcceptsTextObjects() throws IOException {
        String[] docs = OaiRerankSupport.readDocuments(read("{\"documents\":[{\"text\":\"x\"},{\"text\":\"y\"}]}"));
        assertThat(docs.length, is(2));
        assertThat(docs[0], is("x"));
        assertThat(docs[1], is("y"));
    }

    @Test
    public void readDocumentsThrowsWhenEmptyOrMissing() throws IOException {
        JsonNode empty = read("{\"documents\":[]}");
        assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(empty));
        JsonNode missing = read("{\"query\":\"q\"}");
        assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(missing));
    }

    @Test
    public void readDocumentsThrowsOnUnsupportedEntry() throws IOException {
        JsonNode request = read("{\"documents\":[123]}");
        assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(request));
    }

    @Test
    public void readQueryAcceptsATextObjectLikeUpstream() throws IOException {
        // llama.cpp b11820 (#29625) reads the query like a document: a string or {"text": ...}.
        assertThat(OaiRerankSupport.readQuery(read("{\"query\":{\"text\":\"hello\"}}")), is("hello"));
        assertThat(OaiRerankSupport.readQuery(read("{\"query\":{\"text\":[\"hel\",\"lo\"]}}")), is("hello"));
    }

    @Test
    public void readDocumentsConcatenatesATextArrayLikeUpstream() throws IOException {
        String[] docs = OaiRerankSupport.readDocuments(read("{\"documents\":[{\"text\":[\"a\",\"b\"]},\"c\"]}"));
        assertThat(docs.length, is(2));
        assertThat(docs[0], is("ab"));
        assertThat(docs[1], is("c"));
    }

    @Test
    public void readDocumentsRefusesATextArrayWithANonString() throws IOException {
        JsonNode request = read("{\"documents\":[{\"text\":[\"a\",1]}]}");
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(request));
        assertThat(e.getMessage(), containsString("'text' array elements must be strings"));
    }

    @Test
    public void anImageInADocumentIsRefusedRatherThanDropped() throws IOException {
        // Upstream ranks {"text", "image"} items through the projector since b11820; this transport ranks
        // text only, so ranking the text alone would answer another question than the one asked.
        JsonNode request = read("{\"documents\":[\"a\",{\"text\":\"x\",\"image\":\"data:image/png;base64,AA==\"}]}");
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(request));
        assertThat(e.getMessage(), containsString("'image'"));
        assertThat(e.getMessage(), containsString("NativeServer"));
    }

    @Test
    public void anImageInTheQueryIsRefusedRatherThanDropped() throws IOException {
        JsonNode request = read("{\"query\":{\"text\":\"q\",\"image\":[\"u\"]},\"documents\":[\"a\"]}");
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readQuery(request));
        assertThat(e.getMessage(), containsString("'image'"));
        assertThat(e.getMessage(), containsString("NativeServer"));
    }

    @Test
    public void videoIsRefusedLikeUpstream() throws IOException {
        for (String key : new String[] {"video", "fps", "max_frames"}) {
            JsonNode request = read("{\"documents\":[{\"text\":\"x\",\"" + key + "\":1}]}");
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(request));
            assertThat(key, e.getMessage(), containsString("video/fps/max_frames are not supported"));
        }
    }

    @Test
    public void readTopNReturnsValueOrNegativeOne() throws IOException {
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":3}")), is(3));
        assertThat(OaiRerankSupport.readTopN(read("{}")), is(-1));
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":0}")), is(0));
    }

    @Test
    public void negativeTopNIsAnInvalidRequest() throws IOException {
        // Upstream resizes its result vector to min(top_n, n), which for a negative top_n is a huge
        // size_t: the request fails inside the server rather than as the caller's fault.
        JsonNode request = read("{\"top_n\":-1}");
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readTopN(request));
        assertThat(e.getMessage(), containsString("top_n"));
    }

    @Test
    public void readTopNConvertsAFloatLikeUpstream() throws IOException {
        // Upstream reads top_n with nlohmann's get<int>(), which static_casts a JSON float: 2.0 and 2.7
        // keep 2 there. Ours treated any float as absent and kept every result.
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":2.0}")), is(2));
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":2.7}")), is(2));
    }

    @Test
    public void negativeFloatTopNIsAnInvalidRequest() throws IOException {
        // The float spelling of a negative top_n reached the native call as "keep all" before.
        JsonNode request = read("{\"top_n\":-1.5}");
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readTopN(request));
        assertThat(e.getMessage(), containsString("top_n"));
    }

    @Test
    public void readTopNKeepsAllForANonNumberOrAValueOutsideTheIntRange() throws IOException {
        // Deliberate: a string or boolean is no number (upstream: a type error, i.e. its default = all;
        // it would convert true to 1), and a value outside the int range keeps all rather than wrapping
        // as upstream's static_cast does.
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":\"2\"}")), is(-1));
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":true}")), is(-1));
        assertThat(OaiRerankSupport.readTopN(read("{\"top_n\":4294967296}")), is(-1));
    }

    @Test
    public void topNZeroReturnsNoResultsLikeUpstream() throws IOException {
        // format_response_rerank keeps min(top_n, n) entries, so 0 keeps none; ours kept all.
        String nativeJson = "[{\"index\":0,\"score\":0.2},{\"index\":1,\"score\":0.9}]";
        assertThat(
                read(OaiRerankSupport.toOaiResponse(nativeJson, "", 0))
                        .path("results")
                        .size(),
                is(0));
        assertThat(read(OaiRerankSupport.toTeiResponse(nativeJson, 0, false)).size(), is(0));
    }

    @Test
    public void toOaiResponseCarriesTheUsageOfTheEvaluatedTokens() throws IOException {
        // Upstream's answer always has usage, prompt_tokens and total_tokens both the sum of the
        // tokens evaluated per document; ours had none.
        String nativeJson = "[{\"index\":0,\"score\":0.2,\"tokens_evaluated\":5},"
                + "{\"index\":1,\"score\":0.9,\"tokens_evaluated\":7}]";
        JsonNode usage =
                read(OaiRerankSupport.toOaiResponse(nativeJson, "", -1)).path("usage");
        assertThat(usage.path("prompt_tokens").asInt(-1), is(12));
        assertThat(usage.path("total_tokens").asInt(-1), is(12));
    }

    @Test
    public void toOaiResponseSortsByScoreDescendingWithRelevanceScoreAndDataAlias() throws IOException {
        String nativeJson =
                "[{\"document\":\"a\",\"index\":0,\"score\":0.2}," + "{\"document\":\"b\",\"index\":1,\"score\":0.9}]";
        JsonNode out = read(OaiRerankSupport.toOaiResponse(nativeJson, "rr", -1));
        assertThat(out.path("object").asText(), is("list"));
        assertThat(out.path("model").asText(), is("rr"));
        // Highest score first; score is renamed to relevance_score; index preserved.
        assertThat(out.path("results").get(0).path("index").asInt(), is(1));
        assertThat(out.path("results").get(0).path("relevance_score").asDouble(), is(0.9));
        assertThat(out.path("results").get(1).path("index").asInt(), is(0));
        // data is an alias of results (Continue #6478).
        assertThat(out.path("data").get(0).path("index").asInt(), is(1));
    }

    @Test
    public void toOaiResponseAppliesTopN() throws IOException {
        String nativeJson = "[{\"index\":0,\"score\":0.2},{\"index\":1,\"score\":0.9},{\"index\":2,\"score\":0.5}]";
        JsonNode out = read(OaiRerankSupport.toOaiResponse(nativeJson, "", 2));
        assertThat(out.path("results").size(), is(2));
        assertThat(out.path("results").get(0).path("index").asInt(), is(1)); // 0.9
        assertThat(out.path("results").get(1).path("index").asInt(), is(2)); // 0.5
        // Empty model id is omitted.
        assertThat(out.has("model"), is(false));
    }

    @Test
    public void readDocumentsAcceptsTheTeiTexts() throws IOException {
        // Hugging Face text-embeddings-inference sends "texts"; upstream reads it when "documents" is absent.
        JsonNode request = read("{\"query\":\"q\",\"texts\":[\"a\",\"b\"]}");
        String[] documents = assertDoesNotThrow(() -> OaiRerankSupport.readDocuments(request));
        assertThat(documents.length, is(2));
        assertThat(documents[1], is("b"));
    }

    @Test
    public void readDocumentsFallsBackToTextsWhenDocumentsIsNull() throws IOException {
        // upstream's json_value treats an explicit null as absent and serves the request from "texts";
        // ours selected the null and refused the request.
        JsonNode request = read("{\"query\":\"q\",\"documents\":null,\"texts\":[\"a\"]}");
        String[] documents = assertDoesNotThrow(() -> OaiRerankSupport.readDocuments(request));
        assertThat(documents.length, is(1));
        assertThat(documents[0], is("a"));
    }

    @Test
    public void readDocumentsThrowsWhenDocumentsIsNullWithoutTexts() throws IOException {
        JsonNode request = read("{\"query\":\"q\",\"documents\":null}");
        assertThrows(IllegalArgumentException.class, () -> OaiRerankSupport.readDocuments(request));
    }

    @Test
    public void teiRequestGetsTheTeiArrayWithScoreLikeUpstream() throws IOException {
        // upstream answers a request with "texts" as a bare array of {index, score}, best first.
        JsonNode out = read(OaiRerankSupport.toResponse(
                read("{\"query\":\"q\",\"texts\":[\"a\",\"b\"]}"),
                "[{\"document\":\"a\",\"index\":0,\"score\":0.1},{\"document\":\"b\",\"index\":1,\"score\":0.9}]",
                "m"));
        assertThat(out.isArray(), is(true));
        assertThat(out.path(0).path("index").asInt(), is(1));
        assertThat(out.path(0).path("score").asDouble(), is(0.9));
    }

    @Test
    public void teiReturnTextAddsTheDocumentText() throws IOException {
        JsonNode out = read(OaiRerankSupport.toResponse(
                read("{\"query\":\"q\",\"texts\":[\"a\",\"b\"],\"return_text\":true}"),
                "[{\"document\":\"a\",\"index\":0,\"score\":0.1},{\"document\":\"b\",\"index\":1,\"score\":0.9}]",
                "m"));
        assertThat(out.path(0).path("text").asText(), is("b"));
    }

    @Test
    public void documentsRequestGetsTheOpenAiObjectThroughToResponse() throws IOException {
        // A request with "documents" (no "texts") is answered as the OpenAI/Jina object, not as the
        // bare TEI array; the model id is echoed.
        JsonNode out = read(OaiRerankSupport.toResponse(
                read("{\"query\":\"q\",\"documents\":[\"a\",\"b\"]}"),
                "[{\"document\":\"a\",\"index\":0,\"score\":0.1},{\"document\":\"b\",\"index\":1,\"score\":0.9}]",
                "m"));
        assertThat(out.isObject(), is(true));
        assertThat(out.path("object").asText(), is("list"));
        assertThat(out.path("results").size(), is(2));
        assertThat(out.path("model").asText(), is("m"));
    }

    @Test
    public void teiRequestHonoursTopNAndOmitsTextByDefault() throws IOException {
        // upstream's format_response_rerank (server-common.cpp, b11639) resizes to min(top_n, n) before
        // its is_tei_format early return, and reads return_text with the default false, so the TEI
        // answer honours top_n and carries no text unless asked to.
        JsonNode out = read(OaiRerankSupport.toResponse(
                read("{\"query\":\"q\",\"texts\":[\"a\",\"b\"],\"top_n\":1}"),
                "[{\"document\":\"a\",\"index\":0,\"score\":0.1},{\"document\":\"b\",\"index\":1,\"score\":0.9}]",
                "m"));
        assertThat(out.size(), is(1));
        assertThat(out.path(0).path("index").asInt(), is(1));
        assertThat(out.path(0).has("text"), is(false));
    }

    @Test
    public void toOaiResponseRefusesANativeBodyThatIsNotAJsonArray() {
        // An unparsable body used to become 200 with empty results, indistinguishable from "nothing
        // matched". An empty body is no exception to Jackson (it reads a MissingNode) and an object is
        // valid JSON, so both are checked as well as the parse failure.
        for (String body : new String[] {"not json", "", "{}"}) {
            IOException e = assertThrows(IOException.class, () -> OaiRerankSupport.toOaiResponse(body, "m", -1), body);
            assertThat(e.getMessage(), containsString("native rerank response"));
            assertThat(e.getMessage(), containsString("'" + body + "'"));
        }
    }

    @Test
    public void toTeiResponseRefusesANativeBodyThatIsNotAJsonArray() {
        for (String body : new String[] {"not json", "", "{}"}) {
            IOException e =
                    assertThrows(IOException.class, () -> OaiRerankSupport.toTeiResponse(body, -1, false), body);
            assertThat(e.getMessage(), containsString("native rerank response"));
            assertThat(e.getMessage(), containsString("'" + body + "'"));
        }
    }
}
