// Tests for utils.hpp — focused on APIs changed in llama.cpp b4916 → b8576
//
// Covered:
//   - server_grammar_trigger  (new JSON wrapper replacing template to_json/from_json)
//   - raw_buffer / base64_decode  (return type changed from std::string to raw_buffer)
//   - gen_tool_call_id()  (new helper added in b8576)
//   - format_response_rerank()  (top_n parameter added)
//   - server_tokens  (major new type: wraps llama_tokens + optional mtmd chunk map)
//   - json_value / json_is_array_* helpers  (utility coverage)
//   - validate_utf8 / is_valid_utf8  (pure-logic helpers)

#include <gtest/gtest.h>

// Pull in all utils.hpp definitions.  No JNI headers needed.
#include "utils.hpp"

// ============================================================
// server_grammar_trigger
//   New in b8576: replaces direct to_json / from_json templates
//   on common_grammar_trigger with a thin named wrapper struct.
// ============================================================

TEST(ServerGrammarTrigger, DefaultConstruct) {
    server_grammar_trigger sgt;
    // Must compile and not crash — value is zero-initialised by common_grammar_trigger
    (void)sgt;
    SUCCEED();
}

TEST(ServerGrammarTrigger, ConstructFromTrigger) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
    t.value = "tool_call";

    server_grammar_trigger sgt(t);
    EXPECT_EQ(sgt.value.type,  COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
    EXPECT_EQ(sgt.value.value, "tool_call");
}

TEST(ServerGrammarTrigger, WordType_RoundTrip) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
    t.value = "```json";

    json j = server_grammar_trigger(t).to_json();

    EXPECT_TRUE(j.contains("type"));
    EXPECT_TRUE(j.contains("value"));
    EXPECT_FALSE(j.contains("token")); // "token" field is TOKEN-type only

    EXPECT_EQ(j.at("type").get<int>(),         static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_WORD));
    EXPECT_EQ(j.at("value").get<std::string>(), "```json");

    server_grammar_trigger restored(j);
    EXPECT_EQ(restored.value.type,  COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
    EXPECT_EQ(restored.value.value, "```json");
}

TEST(ServerGrammarTrigger, PatternType_RoundTrip) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN;
    t.value = "^\\{";

    json j = server_grammar_trigger(t).to_json();

    EXPECT_EQ(j.at("type").get<int>(),         static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN));
    EXPECT_EQ(j.at("value").get<std::string>(), "^\\{");
    EXPECT_FALSE(j.contains("token"));

    server_grammar_trigger restored(j);
    EXPECT_EQ(restored.value.type,  COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN);
    EXPECT_EQ(restored.value.value, "^\\{");
}

TEST(ServerGrammarTrigger, PatternFullType_RoundTrip) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
    t.value = ".*<tool_call>.*";

    json j = server_grammar_trigger(t).to_json();

    EXPECT_EQ(j.at("type").get<int>(),         static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL));
    EXPECT_EQ(j.at("value").get<std::string>(), ".*<tool_call>.*");
    EXPECT_FALSE(j.contains("token"));

    server_grammar_trigger restored(j);
    EXPECT_EQ(restored.value.type,  COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL);
    EXPECT_EQ(restored.value.value, ".*<tool_call>.*");
}

TEST(ServerGrammarTrigger, TokenType_IncludesTokenField) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
    t.value = "<tool>";
    t.token = 12345;

    json j = server_grammar_trigger(t).to_json();

    EXPECT_TRUE(j.contains("token")); // only TOKEN type serialises the token id
    EXPECT_EQ(j.at("token").get<int>(), 12345);
    EXPECT_EQ(j.at("type").get<int>(),  static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN));

    server_grammar_trigger restored(j);
    EXPECT_EQ(restored.value.type,  COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN);
    EXPECT_EQ(restored.value.token, 12345);
    EXPECT_EQ(restored.value.value, "<tool>");
}

TEST(ServerGrammarTrigger, TypeField_IsIntInJson) {
    common_grammar_trigger t;
    t.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
    t.value = "x";

    json j = server_grammar_trigger(t).to_json();
    EXPECT_TRUE(j.at("type").is_number_integer());
}

// ============================================================
// raw_buffer / base64_decode
//   Return type changed from std::string to raw_buffer
//   (= std::vector<uint8_t>) in b8576.
// ============================================================

TEST(Base64Decode, ReturnType_IsRawBuffer) {
    // Compile-time assertion: the return type must be raw_buffer
    static_assert(
        std::is_same<decltype(base64_decode(std::string{})), raw_buffer>::value,
        "base64_decode must return raw_buffer (std::vector<uint8_t>)");
    SUCCEED();
}

TEST(Base64Decode, RawBufferIsVectorOfUint8) {
    static_assert(
        std::is_same<raw_buffer, std::vector<uint8_t>>::value,
        "raw_buffer must be std::vector<uint8_t>");
    SUCCEED();
}

TEST(Base64Decode, DecodesHello) {
    // "Hello" → "SGVsbG8="
    raw_buffer r = base64_decode("SGVsbG8=");
    ASSERT_EQ(r.size(), 5u);
    EXPECT_EQ(r[0], static_cast<uint8_t>('H'));
    EXPECT_EQ(r[1], static_cast<uint8_t>('e'));
    EXPECT_EQ(r[2], static_cast<uint8_t>('l'));
    EXPECT_EQ(r[3], static_cast<uint8_t>('l'));
    EXPECT_EQ(r[4], static_cast<uint8_t>('o'));
}

TEST(Base64Decode, DecodesEmptyString) {
    raw_buffer r = base64_decode("");
    EXPECT_TRUE(r.empty());
}

TEST(Base64Decode, DecodesThreeBytes_NoFinalPadding) {
    // "ABC" → "QUJD"
    raw_buffer r = base64_decode("QUJD");
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0], static_cast<uint8_t>('A'));
    EXPECT_EQ(r[1], static_cast<uint8_t>('B'));
    EXPECT_EQ(r[2], static_cast<uint8_t>('C'));
}

TEST(Base64Decode, DecodesTwoBytes_OnePadChar) {
    // "Ma" → "TWE="
    raw_buffer r = base64_decode("TWE=");
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0], static_cast<uint8_t>('M'));
    EXPECT_EQ(r[1], static_cast<uint8_t>('a'));
}

TEST(Base64Decode, DecodesBinaryData) {
    // 0x00 0xFF 0x80 → "AP+A" — exercises non-ASCII byte values
    raw_buffer r = base64_decode("AP+A");
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0], 0x00u);
    EXPECT_EQ(r[1], 0xFFu);
    EXPECT_EQ(r[2], 0x80u);
}

// ============================================================
// gen_tool_call_id
//   New helper added in b8576 (previously only gen_chatcmplid
//   existed; tool call IDs were not generated separately).
// ============================================================

TEST(GenToolCallId, NonEmpty) {
    EXPECT_FALSE(gen_tool_call_id().empty());
}

TEST(GenToolCallId, Length_Is32) {
    // random_string() always produces exactly 32 characters
    EXPECT_EQ(gen_tool_call_id().size(), 32u);
}

TEST(GenToolCallId, ContainsOnlyAlphanumeric) {
    const std::string id = gen_tool_call_id();
    for (char c : id) {
        EXPECT_TRUE(std::isalnum(static_cast<unsigned char>(c)))
            << "Non-alphanumeric character: '" << c << "'";
    }
}

TEST(GenToolCallId, TwoCallsProduceDifferentValues) {
    // Collision probability with 62^32 possible values is negligible
    EXPECT_NE(gen_tool_call_id(), gen_tool_call_id());
}

TEST(GenToolCallId, DifferentFromChatCmplId) {
    const std::string cmpl_id = gen_chatcmplid();
    EXPECT_EQ(cmpl_id.substr(0, 9), std::string("chatcmpl-")); // guard — it has prefix
    // gen_tool_call_id has NO "chatcmpl-" prefix
    const std::string tool_id = gen_tool_call_id();
    EXPECT_EQ(tool_id.find("chatcmpl-"), std::string::npos);
}

// ============================================================
// format_response_rerank
//   top_n parameter added in b8576; unified TEI + Jina format.
// ============================================================

namespace {

json make_rank(int index, double score, int tokens_evaluated = 10) {
    return json{{"index", index}, {"score", score}, {"tokens_evaluated", tokens_evaluated}};
}

} // namespace

TEST(FormatResponseRerank, JinaFormat_WrapperStructure) {
    json request = {{"model", "my-reranker"}};
    json ranks   = json::array({make_rank(0, 0.5), make_rank(1, 0.9)});
    std::vector<std::string> texts = {"doc0", "doc1"};

    json res = format_response_rerank(request, ranks, /*is_tei=*/false, texts, /*top_n=*/2);

    EXPECT_EQ(res.at("model").get<std::string>(),  "my-reranker");
    EXPECT_EQ(res.at("object").get<std::string>(), "list");
    EXPECT_TRUE(res.contains("usage"));
    EXPECT_TRUE(res.contains("results"));
    EXPECT_TRUE(res.at("results").is_array());
}

TEST(FormatResponseRerank, JinaFormat_UsesRelevanceScoreLabel) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.7)});
    std::vector<std::string> texts = {"doc"};

    json res = format_response_rerank(request, ranks, false, texts, 1);

    EXPECT_TRUE(res.at("results")[0].contains("relevance_score"));
    EXPECT_FALSE(res.at("results")[0].contains("score"));
}

TEST(FormatResponseRerank, JinaFormat_SortedDescendingByScore) {
    json request = json::object();
    // ranks arrive in arbitrary order
    json ranks = json::array({make_rank(0, 0.3), make_rank(1, 0.9), make_rank(2, 0.1)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, ranks, false, texts, 3);

    auto &results = res.at("results");
    EXPECT_EQ(results[0].at("index").get<int>(), 1); // highest: 0.9
    EXPECT_EQ(results[1].at("index").get<int>(), 0); // middle:  0.3
    EXPECT_EQ(results[2].at("index").get<int>(), 2); // lowest:  0.1
}

TEST(FormatResponseRerank, TopN_LimitsResultCount) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.5), make_rank(1, 0.9), make_rank(2, 0.1)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, ranks, false, texts, /*top_n=*/1);

    EXPECT_EQ(res.at("results").size(), 1u);
    // The single returned result must be the highest-scoring one
    EXPECT_EQ(res.at("results")[0].at("index").get<int>(), 1);
}

TEST(FormatResponseRerank, TopN_Two_KeepsTopTwo) {
    json request = json::object();
    json ranks   = json::array({
        make_rank(0, 0.1), make_rank(1, 0.9), make_rank(2, 0.5), make_rank(3, 0.7)});
    std::vector<std::string> texts = {"a", "b", "c", "d"};

    json res = format_response_rerank(request, ranks, false, texts, 2);

    EXPECT_EQ(res.at("results").size(), 2u);
    EXPECT_EQ(res.at("results")[0].at("index").get<int>(), 1); // 0.9
    EXPECT_EQ(res.at("results")[1].at("index").get<int>(), 3); // 0.7
}

TEST(FormatResponseRerank, TopN_LargerThanCount_ReturnsAll) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8), make_rank(1, 0.2)});
    std::vector<std::string> texts = {"x", "y"};

    json res = format_response_rerank(request, ranks, false, texts, /*top_n=*/100);

    EXPECT_EQ(res.at("results").size(), 2u);
}

TEST(FormatResponseRerank, TokenCounting_Accumulated) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.5, 15), make_rank(1, 0.9, 25)});
    std::vector<std::string> texts = {"a", "b"};

    json res = format_response_rerank(request, ranks, false, texts, 2);

    EXPECT_EQ(res.at("usage").at("prompt_tokens").get<int>(), 40); // 15 + 25
    EXPECT_EQ(res.at("usage").at("total_tokens").get<int>(),  40);
}

TEST(FormatResponseRerank, TeiFormat_ReturnsArrayDirectly) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8), make_rank(1, 0.3)});
    std::vector<std::string> texts = {"x", "y"};

    json res = format_response_rerank(request, ranks, /*is_tei=*/true, texts, 2);

    EXPECT_TRUE(res.is_array()); // no outer wrapper object
    EXPECT_EQ(res.size(), 2u);
}

TEST(FormatResponseRerank, TeiFormat_UsesScoreLabel) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8)});
    std::vector<std::string> texts = {"doc"};

    json res = format_response_rerank(request, ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_TRUE(res[0].contains("score"));
    EXPECT_FALSE(res[0].contains("relevance_score"));
}

TEST(FormatResponseRerank, TeiFormat_ReturnText_IncludesDocumentText) {
    json request = {{"return_text", true}};
    json ranks   = json::array({make_rank(0, 0.9)});
    std::vector<std::string> texts = {"my document content"};

    json res = format_response_rerank(request, ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_TRUE(res[0].contains("text"));
    EXPECT_EQ(res[0].at("text").get<std::string>(), "my document content");
}

TEST(FormatResponseRerank, TeiFormat_NoReturnText_NoTextField) {
    json request = {{"return_text", false}};
    json ranks   = json::array({make_rank(0, 0.9)});
    std::vector<std::string> texts = {"doc"};

    json res = format_response_rerank(request, ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_FALSE(res[0].contains("text"));
}

TEST(FormatResponseRerank, TeiFormat_SortedDescendingByScore) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.1), make_rank(1, 0.9), make_rank(2, 0.5)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, ranks, true, texts, 3);

    ASSERT_TRUE(res.is_array());
    EXPECT_EQ(res[0].at("index").get<int>(), 1); // 0.9
    EXPECT_EQ(res[1].at("index").get<int>(), 2); // 0.5
    EXPECT_EQ(res[2].at("index").get<int>(), 0); // 0.1
}

// ============================================================
// server_tokens
//   Major new type in b8576.  Tests cover the non-mtmd path
//   (has_mtmd = false) which is what the Java bindings use for
//   all text-only inference.
// ============================================================

TEST(ServerTokens, DefaultConstruct_EmptyAndNoMtmd) {
    server_tokens st;
    EXPECT_TRUE(st.empty());
    EXPECT_EQ(st.size(), 0u);
    EXPECT_FALSE(st.has_mtmd);
}

TEST(ServerTokens, ConstructFromLlamaTokens_CopiesTokens) {
    llama_tokens toks = {1, 2, 3, 4, 5};
    server_tokens st(toks, /*has_mtmd=*/false);

    EXPECT_EQ(st.size(), 5u);
    EXPECT_FALSE(st.empty());
    EXPECT_FALSE(st.has_mtmd);
}

TEST(ServerTokens, IndexOperator_ReadsCorrectValue) {
    llama_tokens toks = {10, 20, 30};
    server_tokens st(toks, false);

    EXPECT_EQ(st[0], 10);
    EXPECT_EQ(st[1], 20);
    EXPECT_EQ(st[2], 30);
}

TEST(ServerTokens, ConstIndexOperator) {
    llama_tokens toks = {7, 8};
    server_tokens st(toks, false);
    const server_tokens &cst = st;

    EXPECT_EQ(cst[0], 7);
    EXPECT_EQ(cst[1], 8);
}

TEST(ServerTokens, PushBack_ValidToken_GrowsSize) {
    server_tokens st;
    st.push_back(42);
    st.push_back(99);

    EXPECT_EQ(st.size(), 2u);
    EXPECT_EQ(st[0], 42);
    EXPECT_EQ(st[1], 99);
}

TEST(ServerTokens, PushBack_NullToken_Throws) {
    server_tokens st;
    EXPECT_THROW(st.push_back(LLAMA_TOKEN_NULL), std::runtime_error);
    // Size must not change after the throw
    EXPECT_EQ(st.size(), 0u);
}

TEST(ServerTokens, Insert_AppendsAllTokens) {
    llama_tokens initial = {1, 2};
    server_tokens st(initial, false);

    llama_tokens extra = {3, 4, 5};
    st.insert(extra);

    EXPECT_EQ(st.size(), 5u);
    EXPECT_EQ(st[2], 3);
    EXPECT_EQ(st[3], 4);
    EXPECT_EQ(st[4], 5);
}

TEST(ServerTokens, Insert_IntoEmpty_Works) {
    server_tokens st;
    llama_tokens toks = {10, 20};
    st.insert(toks);

    EXPECT_EQ(st.size(), 2u);
    EXPECT_EQ(st[0], 10);
}

TEST(ServerTokens, GetTextTokens_ReturnsSameTokens) {
    llama_tokens toks = {7, 8, 9};
    server_tokens st(toks, false);

    const llama_tokens &text = st.get_text_tokens();
    ASSERT_EQ(text.size(), 3u);
    EXPECT_EQ(text[0], 7);
    EXPECT_EQ(text[1], 8);
    EXPECT_EQ(text[2], 9);
}

TEST(ServerTokens, SetToken_UpdatesSpecificPosition) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);

    st.set_token(1, 99);

    EXPECT_EQ(st[0], 1);
    EXPECT_EQ(st[1], 99);
    EXPECT_EQ(st[2], 3);
}

TEST(ServerTokens, KeepFirst_TruncatesToN) {
    llama_tokens toks = {1, 2, 3, 4, 5};
    server_tokens st(toks, false);

    st.keep_first(3);

    EXPECT_EQ(st.size(), 3u);
    EXPECT_EQ(st[0], 1);
    EXPECT_EQ(st[2], 3);
}

TEST(ServerTokens, KeepFirst_Zero_EmptiesTokens) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);

    st.keep_first(0);

    EXPECT_TRUE(st.empty());
}

TEST(ServerTokens, KeepFirst_FullSize_NoChange) {
    llama_tokens toks = {10, 20, 30};
    server_tokens st(toks, false);

    st.keep_first(3);

    EXPECT_EQ(st.size(), 3u);
    EXPECT_EQ(st[2], 30);
}

TEST(ServerTokens, GetCommonPrefix_IdenticalSequences_ReturnsFullLength) {
    llama_tokens t1 = {1, 2, 3, 4};
    llama_tokens t2 = {1, 2, 3, 4};
    server_tokens a(t1, false);
    server_tokens b(t2, false);

    EXPECT_EQ(a.get_common_prefix(b), 4u);
}

TEST(ServerTokens, GetCommonPrefix_DivergesAtIndex2) {
    llama_tokens t1 = {1, 2, 3};
    llama_tokens t2 = {1, 2, 9};
    server_tokens a(t1, false);
    server_tokens b(t2, false);

    EXPECT_EQ(a.get_common_prefix(b), 2u);
}

TEST(ServerTokens, GetCommonPrefix_NothingInCommon) {
    llama_tokens t1 = {1, 2, 3};
    llama_tokens t2 = {9, 8, 7};
    server_tokens a(t1, false);
    server_tokens b(t2, false);

    EXPECT_EQ(a.get_common_prefix(b), 0u);
}

TEST(ServerTokens, GetCommonPrefix_BoundedByShortestSequence) {
    llama_tokens t1 = {1, 2, 3, 4, 5};
    llama_tokens t2 = {1, 2, 3};
    server_tokens a(t1, false);
    server_tokens b(t2, false);

    EXPECT_EQ(a.get_common_prefix(b), 3u);
}

TEST(ServerTokens, GetCommonPrefix_BothEmpty_ReturnsZero) {
    server_tokens a;
    server_tokens b;

    EXPECT_EQ(a.get_common_prefix(b), 0u);
}

TEST(ServerTokens, Clear_RemovesAllTokens) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);

    st.clear();

    EXPECT_TRUE(st.empty());
    EXPECT_EQ(st.size(), 0u);
}

TEST(ServerTokens, MoveConstruct_TransfersOwnership) {
    llama_tokens toks = {1, 2, 3};
    server_tokens original(toks, false);

    server_tokens moved(std::move(original));

    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved[0], 1);
    EXPECT_EQ(moved[2], 3);
}

TEST(ServerTokens, MoveAssign_TransfersOwnership) {
    llama_tokens toks = {10, 20};
    server_tokens a(toks, false);
    server_tokens b;

    b = std::move(a);

    EXPECT_EQ(b.size(), 2u);
    EXPECT_EQ(b[0], 10);
    EXPECT_EQ(b[1], 20);
}

TEST(ServerTokens, CopyIsDeleted) {
    // Compile-time assertion: copying must be disabled to prevent
    // accidental shallow copies of the chunk map.
    static_assert(!std::is_copy_constructible<server_tokens>::value,
                  "server_tokens must not be copy-constructible");
    static_assert(!std::is_copy_assignable<server_tokens>::value,
                  "server_tokens must not be copy-assignable");
    SUCCEED();
}

TEST(ServerTokens, MoveIsAllowed) {
    static_assert(std::is_move_constructible<server_tokens>::value,
                  "server_tokens must be move-constructible");
    static_assert(std::is_move_assignable<server_tokens>::value,
                  "server_tokens must be move-assignable");
    SUCCEED();
}

TEST(ServerTokens, Str_ContainsTokensLabel) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);

    const std::string s = st.str();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("tokens"), std::string::npos);
}

// ============================================================
// json_value utility
// ============================================================

TEST(JsonValue, MissingKey_ReturnsDefault) {
    const json body = json::object();
    EXPECT_EQ(json_value(body, "missing", 42), 42);
}

TEST(JsonValue, NullValue_ReturnsDefault) {
    const json body = {{"key", nullptr}};
    EXPECT_EQ(json_value(body, "key", 99), 99);
}

TEST(JsonValue, PresentKey_ReturnsValue) {
    const json body = {{"temperature", 0.8}};
    EXPECT_DOUBLE_EQ(json_value(body, "temperature", 1.0), 0.8);
}

TEST(JsonValue, StringValue) {
    const json body = {{"model", "llama3"}};
    EXPECT_EQ(json_value(body, "model", std::string("default")), std::string("llama3"));
}

TEST(JsonValue, BoolValue) {
    const json body = {{"stream", true}};
    EXPECT_EQ(json_value(body, "stream", false), true);
}

// ============================================================
// json_is_array_of_numbers / json_is_array_of_mixed
// ============================================================

TEST(JsonArrayChecks, ArrayOfIntegers_IsNumbers) {
    EXPECT_TRUE(json_is_array_of_numbers(json{1, 2, 3}));
}

TEST(JsonArrayChecks, EmptyArray_IsNumbers) {
    EXPECT_TRUE(json_is_array_of_numbers(json::array()));
}

TEST(JsonArrayChecks, ArrayWithString_NotNumbers) {
    EXPECT_FALSE(json_is_array_of_numbers(json{1, "hello", 3}));
}

TEST(JsonArrayChecks, NonArray_NotNumbers) {
    EXPECT_FALSE(json_is_array_of_numbers(json("just a string")));
    EXPECT_FALSE(json_is_array_of_numbers(json(42)));
}

TEST(JsonArrayChecks, MixedNumbersAndStrings_IsMixed) {
    EXPECT_TRUE(json_is_array_of_mixed_numbers_strings(json{1, "hello", 3}));
}

TEST(JsonArrayChecks, OnlyNumbers_NotMixed) {
    EXPECT_FALSE(json_is_array_of_mixed_numbers_strings(json{1, 2, 3}));
}

TEST(JsonArrayChecks, OnlyStrings_NotMixed) {
    EXPECT_FALSE(json_is_array_of_mixed_numbers_strings(json{"a", "b"}));
}

TEST(JsonArrayChecks, EmptyArray_NotMixed) {
    EXPECT_FALSE(json_is_array_of_mixed_numbers_strings(json::array()));
}

// ============================================================
// validate_utf8 — pure logic, no llama.cpp deps
// ============================================================

TEST(ValidateUtf8, AsciiOnly_ReturnsFullLength) {
    const std::string s = "hello";
    EXPECT_EQ(validate_utf8(s), s.size());
}

TEST(ValidateUtf8, EmptyString_ReturnsZero) {
    EXPECT_EQ(validate_utf8(""), 0u);
}

TEST(ValidateUtf8, ValidTwoByteSequence_FullLength) {
    // "é" = 0xC3 0xA9
    const std::string s = "\xC3\xA9";
    EXPECT_EQ(validate_utf8(s), 2u);
}

TEST(ValidateUtf8, TruncatedTwoByte_ReturnsShorter) {
    // Only the lead byte of a 2-byte sequence — cut off
    const std::string s = "ab\xC3";
    EXPECT_LT(validate_utf8(s), s.size());
}

TEST(ValidateUtf8, ValidThreeByteSequence_FullLength) {
    // "€" = 0xE2 0x82 0xAC
    const std::string s = "\xE2\x82\xAC";
    EXPECT_EQ(validate_utf8(s), 3u);
}

// ============================================================
// is_valid_utf8 — pure logic, no llama.cpp deps
// ============================================================

TEST(IsValidUtf8, PlainAscii_Valid) {
    EXPECT_TRUE(is_valid_utf8("Hello, World!"));
}

TEST(IsValidUtf8, EmptyString_Valid) {
    EXPECT_TRUE(is_valid_utf8(""));
}

TEST(IsValidUtf8, TwoByteChar_Valid) {
    EXPECT_TRUE(is_valid_utf8("\xC3\xA9")); // é
}

TEST(IsValidUtf8, ThreeByteChar_Valid) {
    EXPECT_TRUE(is_valid_utf8("\xE2\x82\xAC")); // €
}

TEST(IsValidUtf8, FourByteChar_Valid) {
    // 😀 = 0xF0 0x9F 0x98 0x80
    EXPECT_TRUE(is_valid_utf8("\xF0\x9F\x98\x80"));
}

TEST(IsValidUtf8, InvalidLeadByte_Invalid) {
    EXPECT_FALSE(is_valid_utf8("\xFF\xFF"));
}

TEST(IsValidUtf8, TruncatedTwoByte_Invalid) {
    EXPECT_FALSE(is_valid_utf8("\xC3")); // missing continuation byte
}

TEST(IsValidUtf8, TruncatedThreeByte_Invalid) {
    EXPECT_FALSE(is_valid_utf8("\xE2\x82")); // missing final byte
}
