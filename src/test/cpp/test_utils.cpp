// Tests for utils.hpp — focused on APIs changed in llama.cpp b4916 → b8576
//
// Covered:
//   - server_grammar_trigger  (new JSON wrapper replacing template to_json/from_json)
//   - gen_tool_call_id()  (new helper added in b8576)
//   - format_response_rerank()  (top_n parameter added)
//   - server_tokens  (major new type: wraps llama_tokens + optional mtmd chunk map)
//   - json_value / json_is_array_* helpers  (utility coverage)
//   - validate_utf8 / is_valid_utf8  (pure-logic helpers)
//   - json_get_nested_values  (path-based JSON extractor)
//   - oaicompat_completion_params_parse  (OAI /completions param validation)
//   - format_embeddings_response_oaicompat  (OAI embedding response formatter)
//   - format_tokenizer_response / format_detokenized_response / format_logit_bias
//   - safe_json_to_str  (lossy JSON→string with bad-char replacement)
//   - token_piece_value  (native /tokenize wire format)

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

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, /*is_tei=*/false, texts, /*top_n=*/2);

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

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, 1);

    EXPECT_TRUE(res.at("results")[0].contains("relevance_score"));
    EXPECT_FALSE(res.at("results")[0].contains("score"));
}

TEST(FormatResponseRerank, JinaFormat_SortedDescendingByScore) {
    json request = json::object();
    // ranks arrive in arbitrary order
    json ranks = json::array({make_rank(0, 0.3), make_rank(1, 0.9), make_rank(2, 0.1)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, 3);

    auto &results = res.at("results");
    EXPECT_EQ(results[0].at("index").get<int>(), 1); // highest: 0.9
    EXPECT_EQ(results[1].at("index").get<int>(), 0); // middle:  0.3
    EXPECT_EQ(results[2].at("index").get<int>(), 2); // lowest:  0.1
}

TEST(FormatResponseRerank, TopN_LimitsResultCount) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.5), make_rank(1, 0.9), make_rank(2, 0.1)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, /*top_n=*/1);

    EXPECT_EQ(res.at("results").size(), 1u);
    // The single returned result must be the highest-scoring one
    EXPECT_EQ(res.at("results")[0].at("index").get<int>(), 1);
}

TEST(FormatResponseRerank, TopN_Two_KeepsTopTwo) {
    json request = json::object();
    json ranks   = json::array({
        make_rank(0, 0.1), make_rank(1, 0.9), make_rank(2, 0.5), make_rank(3, 0.7)});
    std::vector<std::string> texts = {"a", "b", "c", "d"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, 2);

    EXPECT_EQ(res.at("results").size(), 2u);
    EXPECT_EQ(res.at("results")[0].at("index").get<int>(), 1); // 0.9
    EXPECT_EQ(res.at("results")[1].at("index").get<int>(), 3); // 0.7
}

TEST(FormatResponseRerank, TopN_LargerThanCount_ReturnsAll) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8), make_rank(1, 0.2)});
    std::vector<std::string> texts = {"x", "y"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, /*top_n=*/100);

    EXPECT_EQ(res.at("results").size(), 2u);
}

TEST(FormatResponseRerank, TopN_Zero_ReturnsEmptyResults) {
    // top_n=0 must truncate to zero elements, not crash or return all
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.9), make_rank(1, 0.5)});
    std::vector<std::string> texts = {"a", "b"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, /*top_n=*/0);

    ASSERT_TRUE(res.at("results").is_array());
    EXPECT_TRUE(res.at("results").empty());
}

TEST(FormatResponseRerank, TokenCounting_Accumulated) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.5, 15), make_rank(1, 0.9, 25)});
    std::vector<std::string> texts = {"a", "b"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, false, texts, 2);

    EXPECT_EQ(res.at("usage").at("prompt_tokens").get<int>(), 40); // 15 + 25
    EXPECT_EQ(res.at("usage").at("total_tokens").get<int>(),  40);
}

TEST(FormatResponseRerank, TeiFormat_ReturnsArrayDirectly) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8), make_rank(1, 0.3)});
    std::vector<std::string> texts = {"x", "y"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, /*is_tei=*/true, texts, 2);

    EXPECT_TRUE(res.is_array()); // no outer wrapper object
    EXPECT_EQ(res.size(), 2u);
}

TEST(FormatResponseRerank, TeiFormat_UsesScoreLabel) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.8)});
    std::vector<std::string> texts = {"doc"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_TRUE(res[0].contains("score"));
    EXPECT_FALSE(res[0].contains("relevance_score"));
}

TEST(FormatResponseRerank, TeiFormat_ReturnText_IncludesDocumentText) {
    json request = {{"return_text", true}};
    json ranks   = json::array({make_rank(0, 0.9)});
    std::vector<std::string> texts = {"my document content"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_TRUE(res[0].contains("text"));
    EXPECT_EQ(res[0].at("text").get<std::string>(), "my document content");
}

TEST(FormatResponseRerank, TeiFormat_NoReturnText_NoTextField) {
    json request = {{"return_text", false}};
    json ranks   = json::array({make_rank(0, 0.9)});
    std::vector<std::string> texts = {"doc"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, true, texts, 1);

    ASSERT_TRUE(res.is_array());
    EXPECT_FALSE(res[0].contains("text"));
}

TEST(FormatResponseRerank, TeiFormat_SortedDescendingByScore) {
    json request = json::object();
    json ranks   = json::array({make_rank(0, 0.1), make_rank(1, 0.9), make_rank(2, 0.5)});
    std::vector<std::string> texts = {"a", "b", "c"};

    json res = format_response_rerank(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), ranks, true, texts, 3);

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

// pos_next / size_up_to_pos — text-only path (has_mtmd=false).
// In the non-multimodal path, positions are 1-to-1 with token indices.

TEST(ServerTokens, PosNext_DefaultAll_ReturnsSize) {
    llama_tokens toks = {10, 20, 30};
    server_tokens st(toks, false);
    // pos_next(-1) == total positions == tokens.size()
    EXPECT_EQ(st.pos_next(-1), static_cast<llama_pos>(3));
}

TEST(ServerTokens, PosNext_ExactN_ReturnsN) {
    llama_tokens toks = {1, 2, 3, 4, 5};
    server_tokens st(toks, false);
    EXPECT_EQ(st.pos_next(2), static_cast<llama_pos>(2));
    EXPECT_EQ(st.pos_next(5), static_cast<llama_pos>(5));
}

TEST(ServerTokens, PosNext_EmptyTokens_ReturnsZero) {
    server_tokens st;
    EXPECT_EQ(st.pos_next(-1), static_cast<llama_pos>(0));
}

TEST(ServerTokens, SizeUpToPos_LessThanSize_ReturnsPos) {
    llama_tokens toks = {1, 2, 3, 4};
    server_tokens st(toks, false);
    // max_pos < tokens.size() → clamp to max_pos
    EXPECT_EQ(st.size_up_to_pos(2), 2u);
}

TEST(ServerTokens, SizeUpToPos_BeyondSize_ReturnsSize) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);
    EXPECT_EQ(st.size_up_to_pos(100), 3u);
}

TEST(ServerTokens, SizeUpToPos_Zero_ReturnsZero) {
    llama_tokens toks = {1, 2, 3};
    server_tokens st(toks, false);
    EXPECT_EQ(st.size_up_to_pos(0), 0u);
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

// json_is_array_and_contains_numbers
//   Returns true when the input is an array that has at least one integer
//   element; returns false for a string-only array, an empty array, or a
//   non-array value.

TEST(JsonArrayChecks, ArrayWithNumber_ContainsNumbers) {
    EXPECT_TRUE(json_is_array_and_contains_numbers(json{1, "hello"}));
}

TEST(JsonArrayChecks, ArrayOnlyStrings_NotContainsNumbers) {
    EXPECT_FALSE(json_is_array_and_contains_numbers(json{"a", "b"}));
}

TEST(JsonArrayChecks, EmptyArray_NotContainsNumbers) {
    EXPECT_FALSE(json_is_array_and_contains_numbers(json::array()));
}

TEST(JsonArrayChecks, NonArray_NotContainsNumbers) {
    EXPECT_FALSE(json_is_array_and_contains_numbers(json(42)));
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

TEST(ValidateUtf8, ValidFourByteSequence_FullLength) {
    // 😀 = 0xF0 0x9F 0x98 0x80
    const std::string s = "\xF0\x9F\x98\x80";
    EXPECT_EQ(validate_utf8(s), 4u);
}

TEST(ValidateUtf8, TruncatedFourByte_ReturnsShorter) {
    // Lead byte 0xF0 + two continuation bytes — missing the last
    const std::string s = "\xF0\x9F\x98";
    EXPECT_LT(validate_utf8(s), s.size());
}

TEST(ValidateUtf8, MixedAsciiAndMultiByte_ReturnsFullLength) {
    // "aé" = 0x61 0xC3 0xA9 — all valid
    const std::string s = "a\xC3\xA9";
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

TEST(IsValidUtf8, TruncatedFourByte_Invalid) {
    EXPECT_FALSE(is_valid_utf8("\xF0\x9F\x98")); // missing last continuation
}

TEST(IsValidUtf8, MixedAsciiAndMultiByte_Valid) {
    EXPECT_TRUE(is_valid_utf8("Hello \xC3\xA9!")); // "Hello é!"
}

// ============================================================
// json_get_nested_values
//   Pure recursive path extractor; paths delimited by '/'.
// ============================================================

TEST(JsonGetNestedValues, SimpleKey_ExtractsValue) {
    const json js = {{"a", 42}, {"b", "hello"}};
    const json res = json_get_nested_values({"a"}, js);
    ASSERT_TRUE(res.contains("a"));
    EXPECT_EQ(res.at("a").get<int>(), 42);
    EXPECT_FALSE(res.contains("b")); // only requested key
}

TEST(JsonGetNestedValues, NestedPath_ExtractsDeepValue) {
    const json js = {{"outer", {{"inner", 7}}}};
    const json res = json_get_nested_values({"outer/inner"}, js);
    ASSERT_TRUE(res.contains("outer/inner"));
    EXPECT_EQ(res.at("outer/inner").get<int>(), 7);
}

TEST(JsonGetNestedValues, MissingPath_Skipped) {
    const json js = {{"x", 1}};
    const json res = json_get_nested_values({"x", "y/z"}, js);
    EXPECT_TRUE(res.contains("x"));
    EXPECT_FALSE(res.contains("y/z"));
}

TEST(JsonGetNestedValues, MultiplePaths_AllExtracted) {
    const json js = {{"a", 1}, {"b", {{"c", 2}}}};
    const json res = json_get_nested_values({"a", "b/c"}, js);
    EXPECT_EQ(res.at("a").get<int>(), 1);
    EXPECT_EQ(res.at("b/c").get<int>(), 2);
}

TEST(JsonGetNestedValues, EmptyPaths_ReturnsEmptyObject) {
    const json js = {{"a", 1}};
    const json res = json_get_nested_values({}, js);
    EXPECT_TRUE(res.is_object());
    EXPECT_TRUE(res.empty());
}

// ============================================================
// oaicompat_completion_params_parse
//   Model-free JSON parameter validation for /completions.
// ============================================================

TEST(OaicompatCompletionParams, MissingPrompt_Throws) {
    const json body = {{"max_tokens", 50}};
    EXPECT_THROW(oaicompat_completion_params_parse(body), std::runtime_error);
}

TEST(OaicompatCompletionParams, StopString_NormalizedToArray) {
    const json body = {{"prompt", "hello"}, {"stop", "STOP"}};
    const json res = oaicompat_completion_params_parse(body);
    ASSERT_TRUE(res.contains("stop"));
    ASSERT_TRUE(res.at("stop").is_array());
    EXPECT_EQ(res.at("stop").size(), 1u);
    EXPECT_EQ(res.at("stop")[0].get<std::string>(), "STOP");
}

TEST(OaicompatCompletionParams, StopArray_PassedThrough) {
    const json body = {{"prompt", "hi"}, {"stop", json::array({"A", "B"})}};
    const json res = oaicompat_completion_params_parse(body);
    ASSERT_TRUE(res.at("stop").is_array());
    EXPECT_EQ(res.at("stop").size(), 2u);
}

TEST(OaicompatCompletionParams, NNotOne_PassedThrough) {
    // upstream oaicompat_completion_params_parse no longer rejects n > 1;
    // the value is forwarded to llama_params like any other field
    const json body = {{"prompt", "hi"}, {"n", 3}};
    EXPECT_NO_THROW(oaicompat_completion_params_parse(body));
}

TEST(OaicompatCompletionParams, NEqualsOne_OK) {
    const json body = {{"prompt", "hi"}, {"n", 1}};
    EXPECT_NO_THROW(oaicompat_completion_params_parse(body));
}

TEST(OaicompatCompletionParams, EchoTrue_Throws) {
    const json body = {{"prompt", "hi"}, {"echo", true}};
    EXPECT_THROW(oaicompat_completion_params_parse(body), std::runtime_error);
}

TEST(OaicompatCompletionParams, UnsupportedParam_BestOf_Throws) {
    const json body = {{"prompt", "hi"}, {"best_of", 3}};
    EXPECT_THROW(oaicompat_completion_params_parse(body), std::runtime_error);
}

TEST(OaicompatCompletionParams, UnsupportedParam_Suffix_Throws) {
    const json body = {{"prompt", "hi"}, {"suffix", "end"}};
    EXPECT_THROW(oaicompat_completion_params_parse(body), std::runtime_error);
}

TEST(OaicompatCompletionParams, ValidParams_PassedThrough) {
    const json body = {{"prompt", "hello"}, {"max_tokens", 64}, {"temperature", 0.8}};
    const json res = oaicompat_completion_params_parse(body);
    EXPECT_EQ(res.at("prompt").get<std::string>(), "hello");
    EXPECT_EQ(res.at("max_tokens").get<int>(), 64);
    EXPECT_DOUBLE_EQ(res.at("temperature").get<double>(), 0.8);
}

TEST(OaicompatCompletionParams, NPredictOverridesByMaxTokens) {
    // When both max_tokens and n_predict are in body, n_predict wins (special-cased copy)
    const json body = {{"prompt", "hi"}, {"max_tokens", 10}, {"n_predict", 99}};
    const json res = oaicompat_completion_params_parse(body);
    // n_predict should be present at its given value
    EXPECT_EQ(res.at("n_predict").get<int>(), 99);
}

// ============================================================
// format_embeddings_response_oaicompat
//   Pure JSON response formatter — no model required.
// ============================================================

namespace {
json make_embedding_elem(const std::vector<float> &vec, int tokens = 4) {
    return json{{"embedding", vec}, {"tokens_evaluated", tokens}};
}
} // namespace

TEST(FormatEmbeddingsResponse, SingleEmbedding_Fields) {
    const json request = {{"model", "test-model"}};
    const json embeddings = json::array({make_embedding_elem({0.1f, 0.2f, 0.3f})});
    const json res = format_embeddings_response_oaicompat(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), embeddings);
    EXPECT_EQ(res.at("object").get<std::string>(), "list");
    EXPECT_EQ(res.at("model").get<std::string>(), "test-model");
    EXPECT_EQ(res.at("data").size(), 1u);
    EXPECT_EQ(res.at("data")[0].at("index").get<int>(), 0);
    EXPECT_EQ(res.at("data")[0].at("object").get<std::string>(), "embedding");
}

TEST(FormatEmbeddingsResponse, TokensAccumulated) {
    const json request = {};
    const json embeddings = json::array({make_embedding_elem({1.0f}, 3), make_embedding_elem({2.0f}, 7)});
    const json res = format_embeddings_response_oaicompat(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), embeddings);
    EXPECT_EQ(res.at("usage").at("prompt_tokens").get<int>(), 10);
    EXPECT_EQ(res.at("usage").at("total_tokens").get<int>(), 10);
}

TEST(FormatEmbeddingsResponse, MultipleEmbeddings_IndicesIncrement) {
    const json request = {};
    const json embeddings = json::array({
        make_embedding_elem({0.1f}),
        make_embedding_elem({0.2f}),
        make_embedding_elem({0.3f}),
    });
    const json res = format_embeddings_response_oaicompat(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), embeddings);
    EXPECT_EQ(res.at("data").size(), 3u);
    EXPECT_EQ(res.at("data")[0].at("index").get<int>(), 0);
    EXPECT_EQ(res.at("data")[1].at("index").get<int>(), 1);
    EXPECT_EQ(res.at("data")[2].at("index").get<int>(), 2);
}

TEST(FormatEmbeddingsResponse, Base64Format_EncodingFormatField) {
    const json request = {};
    const json embeddings = json::array({make_embedding_elem({1.0f, 0.0f})});
    const json res = format_embeddings_response_oaicompat(request, json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), embeddings, /*use_base64=*/true);
    const json &elem = res.at("data")[0];
    EXPECT_TRUE(elem.contains("encoding_format"));
    EXPECT_EQ(elem.at("encoding_format").get<std::string>(), "base64");
    // embedding value should be a string (base64), not an array
    EXPECT_TRUE(elem.at("embedding").is_string());
}

// ============================================================
// format_tokenizer_response / format_detokenized_response /
// format_logit_bias
//   Tiny response formatters — pure data wrappers.
// ============================================================

TEST(FormatTokenizerResponse, WrapsInTokensKey) {
    const json tokens = json::array({1, 2, 3});
    const json res = format_tokenizer_response(tokens);
    ASSERT_TRUE(res.contains("tokens"));
    EXPECT_EQ(res.at("tokens"), tokens);
}

TEST(FormatDetokenizedResponse, WrapsInContentKey) {
    const json res = format_detokenized_response("hello world");
    ASSERT_TRUE(res.contains("content"));
    EXPECT_EQ(res.at("content").get<std::string>(), "hello world");
}

TEST(FormatDetokenizedResponse, EmptyString) {
    const json res = format_detokenized_response("");
    EXPECT_EQ(res.at("content").get<std::string>(), "");
}

TEST(FormatLogitBias, EmptyVector_ReturnsEmptyArray) {
    const json res = format_logit_bias({});
    EXPECT_TRUE(res.is_array());
    EXPECT_TRUE(res.empty());
}

TEST(FormatLogitBias, SingleEntry_CorrectFields) {
    llama_logit_bias lb;
    lb.token = 42;
    lb.bias  = -1.5f;
    const json res = format_logit_bias({lb});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].at("token").get<int>(), 42);
    EXPECT_FLOAT_EQ(res[0].at("bias").get<float>(), -1.5f);
}

TEST(FormatLogitBias, MultipleEntries) {
    llama_logit_bias a; a.token = 1; a.bias = 0.5f;
    llama_logit_bias b; b.token = 2; b.bias = -2.0f;
    const json res = format_logit_bias({a, b});
    EXPECT_EQ(res.size(), 2u);
}

// ============================================================
// safe_json_to_str
//   Converts JSON to compact string, replacing un-serialisable
//   values (e.g. invalid UTF-8) instead of throwing.
// ============================================================

TEST(SafeJsonToStr, NormalJson_ProducesCompactString) {
    const json j = {{"key", "value"}, {"n", 3}};
    const std::string s = safe_json_to_str(j);
    EXPECT_FALSE(s.empty());
    // compact — no newlines
    EXPECT_EQ(s.find('\n'), std::string::npos);
}

TEST(SafeJsonToStr, EmptyObject_ProducesEmptyBraces) {
    EXPECT_EQ(safe_json_to_str(json::object()), "{}");
}

TEST(SafeJsonToStr, ArrayValue_Roundtrips) {
    const json j = json::array({1, 2, 3});
    EXPECT_EQ(safe_json_to_str(j), "[1,2,3]");
}

TEST(SafeJsonToStr, InvalidUtf8InString_DoesNotThrow) {
    // nlohmann json with error_handler_t::replace should not throw for bad bytes
    const json j = json{{"bad", "\xFF\xFE invalid"}};
    EXPECT_NO_THROW(safe_json_to_str(j));
}

// ============================================================
// oaicompat_chat_params_parse — early validation throws
//   These all fire BEFORE common_chat_templates_apply is called,
//   so opt.tmpls can remain nullptr safely.
// ============================================================

namespace {
// Minimal helper: build body + options + out_files for early-throw tests
std::vector<raw_buffer> g_out_files;

json make_chat_body_with_messages(const json &messages_override = json::array({
    {{"role", "user"}, {"content", "hello"}}
})) {
    return json{{"messages", messages_override}};
}

server_chat_params make_no_jinja_opts() {
    server_chat_params opt;
    opt.use_jinja = false;
    // tmpls: shared_ptr default-constructs to nullptr — no explicit set needed
    return opt;
}
} // namespace

TEST(OaicompatChatParams, MissingMessages_Throws) {
    json body = {{"model", "x"}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, MessagesNotArray_Throws) {
    json body = {{"messages", "not-an-array"}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, NonAssistantMissingContent_Throws) {
    // user message with no "content" field
    json body = {{"messages", json::array({{{"role", "user"}}})}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, AssistantMissingBothContentAndToolCalls_Throws) {
    // assistant message must have content OR tool_calls
    json body = {{"messages", json::array({{{"role", "assistant"}}})}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, ToolsWithoutJinja_Throws) {
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
        {"tools", json::array({{{"type", "function"}}})}
    };
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, NonAutoToolChoiceWithoutJinja_Throws) {
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
        {"tool_choice", "none"}
    };
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, GrammarAndJsonSchema_Throws) {
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
        {"grammar", "root ::= [a-z]+"},
        {"json_schema", {{"type", "object"}}}
    };
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, InvalidResponseFormatType_Throws) {
    json body = {
        {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
        {"response_format", {{"type", "invalid_type"}}}
    };
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, ContentPartTypeUnsupported_Throws) {
    json body = {{"messages", json::array({{
        {"role", "user"},
        {"content", json::array({{{"type", "video_url"}, {"url", "x"}}})}
    }})}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, ImageUrlWithoutAllowImage_Throws) {
    json body = {{"messages", json::array({{
        {"role", "user"},
        {"content", json::array({{
            {"type", "image_url"},
            {"image_url", {{"url", "data:image/png;base64,abc"}}}
        }})}
    }})}};
    server_chat_params opt = make_no_jinja_opts();
    opt.allow_image = false;
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

TEST(OaicompatChatParams, ContentNotStringOrArray_Throws) {
    // content is an integer — not allowed
    json body = {{"messages", json::array({{{"role", "user"}, {"content", 42}}})}};
    server_chat_params opt = make_no_jinja_opts();
    std::vector<raw_buffer> files;
    EXPECT_THROW(oaicompat_chat_params_parse(body, opt, files), std::exception);
}

// ============================================================
// are_lora_equal / parse_lora_request
//   Pure data-structure helpers; no model needed.
// ============================================================

namespace {
common_adapter_lora_info make_lora(float scale, struct llama_adapter_lora *ptr = nullptr) {
    common_adapter_lora_info info;
    info.scale = scale;
    info.ptr   = ptr;
    return info;
}
} // namespace

TEST(AreLoraEqual, BothEmpty_AreEqual) {
    EXPECT_TRUE(are_lora_equal({}, {}));
}

TEST(AreLoraEqual, DifferentSizes_NotEqual) {
    EXPECT_FALSE(are_lora_equal({make_lora(1.0f)}, {}));
}

TEST(AreLoraEqual, SameScaleNullPtr_AreEqual) {
    EXPECT_TRUE(are_lora_equal({make_lora(0.5f)}, {make_lora(0.5f)}));
}

TEST(AreLoraEqual, DifferentScale_NotEqual) {
    EXPECT_FALSE(are_lora_equal({make_lora(0.5f)}, {make_lora(1.0f)}));
}

TEST(AreLoraEqual, DifferentPtr_NotEqual) {
    int dummy = 0;
    auto *p = reinterpret_cast<struct llama_adapter_lora *>(&dummy);
    EXPECT_FALSE(are_lora_equal({make_lora(1.0f, p)}, {make_lora(1.0f, nullptr)}));
}

TEST(AreLoraEqual, PathDifference_Ignored) {
    common_adapter_lora_info a = make_lora(1.0f);
    common_adapter_lora_info b = make_lora(1.0f);
    a.path = "model_a.gguf";
    b.path = "model_b.gguf";
    // path is explicitly not checked in are_lora_equal
    EXPECT_TRUE(are_lora_equal({a}, {b}));
}

TEST(ParseLoraRequest, EmptyData_ClearsAllScales) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.8f), make_lora(0.6f)};
    const auto result = parse_lora_request(base, json::array());
    ASSERT_EQ(result.size(), 2u);
    EXPECT_FLOAT_EQ(result[0].scale, 0.0f);
    EXPECT_FLOAT_EQ(result[1].scale, 0.0f);
}

TEST(ParseLoraRequest, ValidId_SetsScale) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.0f), make_lora(0.0f)};
    const json data = json::array({{{"id", 1}, {"scale", 0.75f}}});
    const auto result = parse_lora_request(base, data);
    EXPECT_FLOAT_EQ(result[0].scale, 0.0f); // untouched
    EXPECT_FLOAT_EQ(result[1].scale, 0.75f);
}

TEST(ParseLoraRequest, InvalidId_Throws) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.0f)};
    const json data = json::array({{{"id", 5}, {"scale", 1.0f}}});
    EXPECT_THROW(parse_lora_request(base, data), std::runtime_error);
}

TEST(ParseLoraRequest, NegativeId_Throws) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.0f)};
    const json data = json::array({{{"id", -1}, {"scale", 1.0f}}});
    EXPECT_THROW(parse_lora_request(base, data), std::runtime_error);
}

TEST(ParseLoraRequest, MultipleIds_AllSet) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.0f), make_lora(0.0f), make_lora(0.0f)};
    const json data = json::array({
        {{"id", 0}, {"scale", 0.3f}},
        {{"id", 2}, {"scale", 0.9f}}
    });
    const auto result = parse_lora_request(base, data);
    EXPECT_FLOAT_EQ(result[0].scale, 0.3f);
    EXPECT_FLOAT_EQ(result[1].scale, 0.0f); // not set
    EXPECT_FLOAT_EQ(result[2].scale, 0.9f);
}

TEST(ParseLoraRequest, DoesNotModifyOriginalBase) {
    std::vector<common_adapter_lora_info> base = {make_lora(0.8f)};
    const json data = json::array({{{"id", 0}, {"scale", 0.2f}}});
    parse_lora_request(base, data);
    // original must be unchanged
    EXPECT_FLOAT_EQ(base[0].scale, 0.8f);
}

// ============================================================
// StripFlagFromArgv
//   Helper used by loadModel to remove --vocab-only from argv
//   before passing to common_params_parse, which does not know
//   about this flag.
// ============================================================

// Build a mutable char* array from string literals for use as argv.
// Lifetime is tied to the vector of strings passed in.
static std::vector<char *> make_argv(std::vector<std::string> &strings) {
    std::vector<char *> ptrs;
    ptrs.reserve(strings.size());
    for (auto &s : strings) {
        ptrs.push_back(s.data());
    }
    return ptrs;
}

TEST(StripFlagFromArgv, FlagAbsent_NothingRemoved) {
    std::vector<std::string> s = {"prog", "--model", "foo.gguf"};
    auto argv = make_argv(s);
    bool found = true; // pre-set to true so we verify it gets cleared
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_FALSE(found);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_STREQ(out[0], "prog");
    EXPECT_STREQ(out[1], "--model");
    EXPECT_STREQ(out[2], "foo.gguf");
}

TEST(StripFlagFromArgv, FlagAtStart_Removed) {
    std::vector<std::string> s = {"--vocab-only", "--model", "foo.gguf"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_STREQ(out[0], "--model");
    EXPECT_STREQ(out[1], "foo.gguf");
}

TEST(StripFlagFromArgv, FlagAtEnd_Removed) {
    std::vector<std::string> s = {"prog", "--model", "foo.gguf", "--vocab-only"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_STREQ(out[2], "foo.gguf");
}

TEST(StripFlagFromArgv, FlagInMiddle_OrderPreserved) {
    std::vector<std::string> s = {"prog", "--ctx-size", "128", "--vocab-only", "--model", "m.gguf"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    ASSERT_EQ(out.size(), 5u);
    EXPECT_STREQ(out[0], "prog");
    EXPECT_STREQ(out[1], "--ctx-size");
    EXPECT_STREQ(out[2], "128");
    EXPECT_STREQ(out[3], "--model");
    EXPECT_STREQ(out[4], "m.gguf");
}

TEST(StripFlagFromArgv, FlagAppearsMultipleTimes_AllRemoved) {
    std::vector<std::string> s = {"--vocab-only", "--model", "m.gguf", "--vocab-only"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_STREQ(out[0], "--model");
    EXPECT_STREQ(out[1], "m.gguf");
}

TEST(StripFlagFromArgv, FlagOnly_ResultEmpty) {
    std::vector<std::string> s = {"--vocab-only"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    EXPECT_TRUE(out.empty());
}

TEST(StripFlagFromArgv, EmptyArgv_NoCrash) {
    bool found = true;
    auto out = strip_flag_from_argv(nullptr, 0, "--vocab-only", &found);
    EXPECT_FALSE(found);
    EXPECT_TRUE(out.empty());
}

TEST(StripFlagFromArgv, PartialMatchNotStripped) {
    // "--vocab-onlyX" must not be treated as "--vocab-only"
    std::vector<std::string> s = {"prog", "--vocab-onlyX"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_FALSE(found);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_STREQ(out[1], "--vocab-onlyX");
}

TEST(StripFlagFromArgv, OtherFlagsUnchanged) {
    std::vector<std::string> s = {"prog", "--embedding", "--vocab-only", "--jinja"};
    auto argv = make_argv(s);
    bool found = false;
    auto out = strip_flag_from_argv(argv.data(), static_cast<int>(argv.size()), "--vocab-only", &found);
    EXPECT_TRUE(found);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_STREQ(out[0], "prog");
    EXPECT_STREQ(out[1], "--embedding");
    EXPECT_STREQ(out[2], "--jinja");
}

// ============================================================
// token_piece_value
//   Used in handleTokenize to build the "piece" field.
//   Valid UTF-8 → JSON string; invalid UTF-8 → JSON byte array.
// ============================================================

TEST(TokenPieceValue, ValidAscii_ReturnsString) {
    const json j = token_piece_value("hello");
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "hello");
}

TEST(TokenPieceValue, ValidMultiByte_ReturnsString) {
    // "é" = 0xC3 0xA9 — valid two-byte UTF-8
    const json j = token_piece_value("\xC3\xA9");
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "\xC3\xA9");
}

TEST(TokenPieceValue, InvalidUtf8_ReturnsByteArray) {
    // 0xFF is never valid in UTF-8
    const json j = token_piece_value("\xFF");
    EXPECT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 1u);
    EXPECT_EQ(j[0].get<int>(), 0xFF);
}

TEST(TokenPieceValue, TruncatedMultiByte_ReturnsByteArray) {
    // Lead byte 0xC3 without continuation — invalid
    const json j = token_piece_value("\xC3");
    EXPECT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 1u);
    EXPECT_EQ(j[0].get<int>(), 0xC3);
}

TEST(TokenPieceValue, EmptyString_ReturnsEmptyString) {
    const json j = token_piece_value("");
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "");
}

TEST(TokenPieceValue, ValidThreeByteChar_ReturnsString) {
    // "€" = 0xE2 0x82 0xAC
    const json j = token_piece_value("\xE2\x82\xAC");
    EXPECT_TRUE(j.is_string());
}
