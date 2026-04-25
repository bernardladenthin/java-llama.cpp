// Tests for json_helpers.hpp.
//
// Every function in json_helpers.hpp is pure JSON transformation with no JNI
// and no llama state.  Tests for functions that only take nlohmann::json
// arguments need zero setup.  Tests for functions that take
// server_task_result_ptr use lightweight fake result objects defined below;
// they need upstream server headers for the type definitions but never load a model.
//
// Covered functions:
//   get_result_error_message
//   results_to_json
//   rerank_results_to_json
//   build_embeddings_response_json
//   extract_first_embedding_row
//   parse_encoding_format
//   extract_embedding_prompt
//   is_infill_request
//   parse_slot_prompt_similarity
//   parse_positive_int_config

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "json_helpers.hpp"

// ============================================================
// Minimal fake result types
// ============================================================

namespace {

// Error result — reuses the real server_task_result_error so that
// to_json() → format_error_response() → {"message": msg, ...} matches the
// exact JSON key that get_result_error_message reads.
static server_task_result_ptr make_error(int id_, const std::string &msg) {
    auto r      = std::make_unique<server_task_result_error>();
    r->id       = id_;
    r->err_msg  = msg;
    r->err_type = ERROR_TYPE_SERVER;
    return r;
}

// Generic success result: to_json() returns {"content": msg}.
struct fake_ok_result : server_task_result {
    std::string msg;
    explicit fake_ok_result(int id_, std::string m) : msg(std::move(m)) { id = id_; }
    json to_json() override { return {{"content", msg}}; }
};

static server_task_result_ptr make_ok(int id_, const std::string &msg = "ok") {
    return std::make_unique<fake_ok_result>(id_, msg);
}

// Embedding result: to_json() returns the shape expected by
// format_embeddings_response_oaicompat.
struct fake_embedding_result : server_task_result {
    std::vector<float> vec;
    int tokens_evaluated;
    explicit fake_embedding_result(int id_, std::vector<float> v, int tok = 4)
        : vec(std::move(v)), tokens_evaluated(tok) { id = id_; }
    json to_json() override {
        return {{"embedding", vec}, {"tokens_evaluated", tokens_evaluated}};
    }
};

static server_task_result_ptr make_embedding(int id_,
                                              std::vector<float> v = {0.1f, 0.2f, 0.3f}) {
    return std::make_unique<fake_embedding_result>(id_, std::move(v));
}

} // namespace

// ============================================================
// get_result_error_message
// ============================================================

TEST(GetResultErrorMessage, ErrorResult_ReturnsMessageString) {
    auto r = make_error(1, "something went wrong");
    EXPECT_EQ(get_result_error_message(r), "something went wrong");
}

TEST(GetResultErrorMessage, DifferentMessage_ReturnsCorrectString) {
    auto r = make_error(2, "out of memory");
    EXPECT_EQ(get_result_error_message(r), "out of memory");
}

// ============================================================
// results_to_json
// ============================================================

TEST(ResultsToJson, SingleResult_ReturnsObjectDirectly) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_ok(1, "only"));

    json out = results_to_json(results);

    EXPECT_TRUE(out.is_object());
    EXPECT_EQ(out.value("content", ""), "only");
}

TEST(ResultsToJson, MultipleResults_ReturnsArray) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_ok(1, "a"));
    results.push_back(make_ok(2, "b"));

    json out = results_to_json(results);

    EXPECT_TRUE(out.is_array());
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].value("content", ""), "a");
    EXPECT_EQ(out[1].value("content", ""), "b");
}

TEST(ResultsToJson, EmptyVector_ReturnsEmptyArray) {
    std::vector<server_task_result_ptr> results;
    json out = results_to_json(results);
    EXPECT_TRUE(out.is_array());
    EXPECT_TRUE(out.empty());
}

// results_to_json has no special error-result handling: a single error result
// is returned as an object directly (not wrapped in an array), exactly like a
// success result. This matters because jllama.cpp callers must inspect the
// object for "error" / "message" without expecting an array wrapper.
TEST(ResultsToJson, SingleErrorResult_ReturnsObjectDirectly) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_error(1, "task failed"));

    json out = results_to_json(results);

    EXPECT_TRUE(out.is_object());
    EXPECT_TRUE(out.contains("message"));
    EXPECT_EQ(out.value("message", ""), "task failed");
}

// ============================================================
// rerank_results_to_json
// ============================================================

namespace {
struct fake_rerank_result : server_task_result {
    int index; float score;
    fake_rerank_result(int id_, int idx, float sc) : index(idx), score(sc) { id = id_; }
    json to_json() override { return {{"index", index}, {"score", score}}; }
};
static server_task_result_ptr make_rerank(int id_, int idx, float sc) {
    return std::make_unique<fake_rerank_result>(id_, idx, sc);
}
} // namespace

TEST(RerankResultsToJson, TwoResults_CorrectShape) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, 0, 0.9f));
    results.push_back(make_rerank(2, 1, 0.4f));
    std::vector<std::string> docs = {"doc A", "doc B"};

    json out = rerank_results_to_json(results, docs);

    ASSERT_TRUE(out.is_array());
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].value("document", ""), "doc A");
    EXPECT_EQ(out[0].value("index", -1), 0);
    EXPECT_FLOAT_EQ(out[0].value("score", 0.0f), 0.9f);
    EXPECT_EQ(out[1].value("document", ""), "doc B");
}

TEST(RerankResultsToJson, EmptyResults_ReturnsEmptyArray) {
    std::vector<server_task_result_ptr> results;
    json out = rerank_results_to_json(results, {});
    EXPECT_TRUE(out.is_array());
    EXPECT_TRUE(out.empty());
}

TEST(RerankResultsToJson, SingleResult_CorrectShape) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, 0, 0.75f));
    std::vector<std::string> docs = {"only doc"};

    json out = rerank_results_to_json(results, docs);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].value("document", ""), "only doc");
    EXPECT_EQ(out[0].value("index", -1), 0);
    EXPECT_FLOAT_EQ(out[0].value("score", 0.0f), 0.75f);
}

TEST(RerankResultsToJson, IndexLookup_UsesResultIndexNotPosition) {
    // Result at position 0 has index=1 — must look up documents[1], not documents[0].
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, 1, 0.5f));
    std::vector<std::string> docs = {"doc zero", "doc one"};

    json out = rerank_results_to_json(results, docs);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].value("document", ""), "doc one");
    EXPECT_EQ(out[0].value("index", -1), 1);
}

// rerank_results_to_json preserves the order in which results were passed in.
// Unlike the upstream OAI helper (format_response_rerank) which sorts by score,
// this function is intentionally order-preserving so the Java caller can decide
// on sorting.  A score inversion in the output is the regression signal.
TEST(RerankResultsToJson, PreservesInputOrder) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, 0, 0.3f)); // low score first
    results.push_back(make_rerank(2, 1, 0.9f)); // high score second
    results.push_back(make_rerank(3, 2, 0.6f));
    std::vector<std::string> docs = {"doc 0", "doc 1", "doc 2"};

    json out = rerank_results_to_json(results, docs);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0].value("score", 0.0f), 0.3f); // order unchanged
    EXPECT_FLOAT_EQ(out[1].value("score", 0.0f), 0.9f);
    EXPECT_FLOAT_EQ(out[2].value("score", 0.0f), 0.6f);
}

// ============================================================
// build_embeddings_response_json
// ============================================================

TEST(BuildEmbeddingsResponseJson, NonOai_SingleResult_ReturnsBareArray) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.1f, 0.2f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_NONE, false);

    ASSERT_TRUE(out.is_array());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].contains("embedding"));
}

TEST(BuildEmbeddingsResponseJson, NonOai_MultipleResults_AllInArray) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.1f}));
    results.push_back(make_embedding(2, {0.2f}));
    results.push_back(make_embedding(3, {0.3f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_NONE, false);

    ASSERT_TRUE(out.is_array());
    EXPECT_EQ(out.size(), 3u);
}

TEST(BuildEmbeddingsResponseJson, OaiFloat_WrapsWithOaiStructure) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.5f, 0.6f, 0.7f}));
    json body = {{"model", "text-embedding-ada-002"}};

    json out = build_embeddings_response_json(results, body,
                                               TASK_RESPONSE_TYPE_OAI_EMBD, false);

    EXPECT_TRUE(out.is_object());
    EXPECT_EQ(out.value("object", ""), "list");
    EXPECT_TRUE(out.contains("data"));
    EXPECT_TRUE(out.contains("usage"));
    EXPECT_EQ(out.value("model", ""), "text-embedding-ada-002");
    ASSERT_TRUE(out["data"].is_array());
    ASSERT_EQ(out["data"].size(), 1u);
    EXPECT_EQ(out["data"][0].value("object", ""), "embedding");
}

TEST(BuildEmbeddingsResponseJson, OaiBase64_EmbeddingEncodedAsString) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {1.0f, 2.0f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_OAI_EMBD, /*use_base64=*/true);

    ASSERT_TRUE(out["data"].is_array());
    EXPECT_TRUE(out["data"][0]["embedding"].is_string())
        << "base64 embedding must be serialised as a string";
}

TEST(BuildEmbeddingsResponseJson, OaiUsage_TokensSummedAcrossResults) {
    std::vector<server_task_result_ptr> results;
    results.push_back(std::make_unique<fake_embedding_result>(1, std::vector<float>{0.1f}, 3));
    results.push_back(std::make_unique<fake_embedding_result>(2, std::vector<float>{0.2f}, 5));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_OAI_EMBD, false);

    EXPECT_EQ(out["usage"].value("prompt_tokens", 0), 8)
        << "usage.prompt_tokens must be sum of tokens_evaluated across all results";
}

// Only TASK_RESPONSE_TYPE_OAI_EMBD wraps in OAI compat structure.
// Other OAI types (OAI_CMPL, OAI_CHAT) must return the bare array —
// the function must branch exclusively on OAI_EMBD, not "any OAI type".
TEST(BuildEmbeddingsResponseJson, OaiCmpl_TreatsAsNonOai) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.1f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_OAI_CMPL, false);

    EXPECT_TRUE(out.is_array());
    EXPECT_FALSE(out.contains("object")); // OAI wrapper has "object":"list"
}

TEST(BuildEmbeddingsResponseJson, OaiChat_TreatsAsNonOai) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.1f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_OAI_CHAT, false);

    EXPECT_TRUE(out.is_array());
    EXPECT_FALSE(out.contains("object"));
}

// use_base64=true is ignored entirely on the non-OAI path.
// In jllama.cpp, parse_encoding_format() and the res_type decision are
// independent: "content" key forces TASK_RESPONSE_TYPE_NONE even when the
// caller also passed encoding_format=base64.  The response must still be a
// bare float array in that case, not a base64-encoded string.
TEST(BuildEmbeddingsResponseJson, NonOai_Base64FlagIgnored_ReturnsFloatArray) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.5f, 0.6f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_NONE, /*use_base64=*/true);

    ASSERT_TRUE(out.is_array());
    // The embedding value must be a JSON number array, not a base64 string.
    EXPECT_TRUE(out[0]["embedding"].is_array())
        << "use_base64 must be ignored when res_type is not OAI_EMBD";
}

// OAI compat wrapper reads "model" from the request body and falls back to
// DEFAULT_OAICOMPAT_MODEL when the field is absent.  The model field must
// always be present and non-empty in the response.
TEST(BuildEmbeddingsResponseJson, OaiFloat_ModelAbsent_UsesDefault) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_embedding(1, {0.1f}));

    json out = build_embeddings_response_json(results, json::object(),
                                               TASK_RESPONSE_TYPE_OAI_EMBD, false);

    EXPECT_TRUE(out.is_object());
    ASSERT_TRUE(out.contains("model"));
    EXPECT_FALSE(out.value("model", "").empty())
        << "model must fall back to DEFAULT_OAICOMPAT_MODEL when absent from body";
}

// ============================================================
// extract_first_embedding_row
// ============================================================

TEST(ExtractFirstEmbeddingRow, SingleRow_ReturnsRow) {
    json j = {{"embedding", {{0.1f, 0.2f, 0.3f}}}};
    auto row = extract_first_embedding_row(j);
    ASSERT_EQ(row.size(), 3u);
    EXPECT_FLOAT_EQ(row[0], 0.1f);
    EXPECT_FLOAT_EQ(row[1], 0.2f);
    EXPECT_FLOAT_EQ(row[2], 0.3f);
}

TEST(ExtractFirstEmbeddingRow, MultipleRows_ReturnsFirstRowOnly) {
    json j = {{"embedding", {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}}};
    auto row = extract_first_embedding_row(j);
    ASSERT_EQ(row.size(), 2u);
    EXPECT_FLOAT_EQ(row[0], 1.0f);
    EXPECT_FLOAT_EQ(row[1], 2.0f);
}

TEST(ExtractFirstEmbeddingRow, MissingEmbeddingKey_ThrowsJsonException) {
    json j = {{"other_key", "value"}};
    EXPECT_THROW((void)extract_first_embedding_row(j), nlohmann::json::exception);
}

TEST(ExtractFirstEmbeddingRow, EmptyOuterArray_ThrowsRuntimeError) {
    json j = {{"embedding", json::array()}};
    EXPECT_THROW((void)extract_first_embedding_row(j), std::runtime_error);
}

TEST(ExtractFirstEmbeddingRow, EmptyInnerArray_ThrowsRuntimeError) {
    json j = {{"embedding", {json::array()}}};
    EXPECT_THROW((void)extract_first_embedding_row(j), std::runtime_error);
}

TEST(ExtractFirstEmbeddingRow, LargeRow_AllValuesPreserved) {
    std::vector<float> vals(128);
    for (int i = 0; i < 128; ++i) vals[i] = static_cast<float>(i) * 0.01f;
    json j = {{"embedding", {vals}}};
    auto row = extract_first_embedding_row(j);
    ASSERT_EQ(row.size(), 128u);
    for (int i = 0; i < 128; ++i) {
        EXPECT_FLOAT_EQ(row[i], static_cast<float>(i) * 0.01f);
    }
}

// ============================================================
// parse_encoding_format
// ============================================================

TEST(ParseEncodingFormat, FieldAbsent_ReturnsFalse) {
    EXPECT_FALSE(parse_encoding_format({{"model", "x"}}));
}

TEST(ParseEncodingFormat, ExplicitFloat_ReturnsFalse) {
    EXPECT_FALSE(parse_encoding_format({{"encoding_format", "float"}}));
}

TEST(ParseEncodingFormat, Base64_ReturnsTrue) {
    EXPECT_TRUE(parse_encoding_format({{"encoding_format", "base64"}}));
}

TEST(ParseEncodingFormat, UnknownFormat_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_encoding_format({{"encoding_format", "binary"}}),
                 std::invalid_argument);
}

TEST(ParseEncodingFormat, EmptyString_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_encoding_format({{"encoding_format", ""}}),
                 std::invalid_argument);
}

TEST(ParseEncodingFormat, ErrorMessage_MentionsBothValidOptions) {
    try {
        (void)parse_encoding_format({{"encoding_format", "hex"}});
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("float"),  std::string::npos);
        EXPECT_NE(msg.find("base64"), std::string::npos);
    }
}

// ============================================================
// extract_embedding_prompt
// ============================================================

TEST(ExtractEmbeddingPrompt, InputKey_ReturnsValueAndDoesNotSetFlag) {
    bool flag = true; // pre-set to verify it gets cleared
    json prompt = extract_embedding_prompt({{"input", "hello world"}}, flag);
    EXPECT_EQ(prompt, "hello world");
    EXPECT_FALSE(flag);
}

TEST(ExtractEmbeddingPrompt, ContentKey_ReturnsValueAndSetsFlag) {
    bool flag = false;
    json prompt = extract_embedding_prompt({{"content", "some text"}}, flag);
    EXPECT_EQ(prompt, "some text");
    EXPECT_TRUE(flag);
}

TEST(ExtractEmbeddingPrompt, InputTakesPriorityOverContent) {
    bool flag = false;
    json prompt = extract_embedding_prompt(
        {{"input", "from input"}, {"content", "from content"}}, flag);
    EXPECT_EQ(prompt, "from input");
    EXPECT_FALSE(flag);
}

TEST(ExtractEmbeddingPrompt, NeitherKey_ThrowsInvalidArgument) {
    bool flag = false;
    EXPECT_THROW((void)extract_embedding_prompt({{"model", "x"}}, flag),
                 std::invalid_argument);
}

TEST(ExtractEmbeddingPrompt, EmptyBody_ThrowsInvalidArgument) {
    bool flag = false;
    EXPECT_THROW((void)extract_embedding_prompt(json::object(), flag),
                 std::invalid_argument);
}

TEST(ExtractEmbeddingPrompt, ArrayPrompt_ReturnedAsIs) {
    bool flag = false;
    json prompt = extract_embedding_prompt(
        {{"input", {"sentence one", "sentence two"}}}, flag);
    ASSERT_TRUE(prompt.is_array());
    ASSERT_EQ(prompt.size(), 2u);
    EXPECT_EQ(prompt[0], "sentence one");
    EXPECT_EQ(prompt[1], "sentence two");
    EXPECT_FALSE(flag);
}

// ============================================================
// is_infill_request
// ============================================================

TEST(IsInfillRequest, HasInputPrefix_ReturnsTrue) {
    EXPECT_TRUE(is_infill_request({{"input_prefix", "def f():"}}));
}

TEST(IsInfillRequest, HasInputSuffix_ReturnsTrue) {
    EXPECT_TRUE(is_infill_request({{"input_suffix", "return 1"}}));
}

TEST(IsInfillRequest, HasBoth_ReturnsTrue) {
    EXPECT_TRUE(is_infill_request(
        {{"input_prefix", "def f():"}, {"input_suffix", "return 1"}}));
}

TEST(IsInfillRequest, HasNeither_ReturnsFalse) {
    EXPECT_FALSE(is_infill_request({{"prompt", "hello"}}));
}

TEST(IsInfillRequest, EmptyBody_ReturnsFalse) {
    EXPECT_FALSE(is_infill_request(json::object()));
}

// ============================================================
// parse_slot_prompt_similarity
// ============================================================

TEST(ParseSlotPromptSimilarity, FieldAbsent_ReturnsEmpty) {
    EXPECT_FALSE(parse_slot_prompt_similarity({{"other", 1}}).has_value());
}

TEST(ParseSlotPromptSimilarity, Zero_ReturnsZero) {
    auto v = parse_slot_prompt_similarity({{"slot_prompt_similarity", 0.0f}});
    ASSERT_TRUE(v.has_value());
    EXPECT_FLOAT_EQ(*v, 0.0f);
}

TEST(ParseSlotPromptSimilarity, Half_ReturnsHalf) {
    auto v = parse_slot_prompt_similarity({{"slot_prompt_similarity", 0.5f}});
    ASSERT_TRUE(v.has_value());
    EXPECT_FLOAT_EQ(*v, 0.5f);
}

TEST(ParseSlotPromptSimilarity, One_ReturnsOne) {
    auto v = parse_slot_prompt_similarity({{"slot_prompt_similarity", 1.0f}});
    ASSERT_TRUE(v.has_value());
    EXPECT_FLOAT_EQ(*v, 1.0f);
}

TEST(ParseSlotPromptSimilarity, TooLow_ThrowsInvalidArgument) {
    EXPECT_THROW(
        (void)parse_slot_prompt_similarity({{"slot_prompt_similarity", -0.1f}}),
        std::invalid_argument);
}

TEST(ParseSlotPromptSimilarity, TooHigh_ThrowsInvalidArgument) {
    EXPECT_THROW(
        (void)parse_slot_prompt_similarity({{"slot_prompt_similarity", 1.1f}}),
        std::invalid_argument);
}

// ============================================================
// parse_positive_int_config
// ============================================================

TEST(ParsePositiveIntConfig, FieldAbsent_ReturnsEmpty) {
    EXPECT_FALSE(parse_positive_int_config({{"other", 1}}, "n_threads").has_value());
}

TEST(ParsePositiveIntConfig, ValidOne_ReturnsOne) {
    auto v = parse_positive_int_config({{"n_threads", 1}}, "n_threads");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 1);
}

TEST(ParsePositiveIntConfig, ValidLarge_ReturnsValue) {
    auto v = parse_positive_int_config({{"n_threads", 128}}, "n_threads");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 128);
}

TEST(ParsePositiveIntConfig, Zero_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_positive_int_config({{"n_threads", 0}}, "n_threads"),
                 std::invalid_argument);
}

TEST(ParsePositiveIntConfig, Negative_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_positive_int_config({{"n_threads", -4}}, "n_threads"),
                 std::invalid_argument);
}

TEST(ParsePositiveIntConfig, ErrorMessage_ContainsKeyName) {
    try {
        (void)parse_positive_int_config({{"n_threads_batch", 0}}, "n_threads_batch");
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("n_threads_batch"), std::string::npos);
    }
}
