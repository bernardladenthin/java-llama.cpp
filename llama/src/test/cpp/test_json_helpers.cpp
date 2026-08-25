// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

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
//   parse_encoding_format
//   extract_embedding_prompt
//   is_infill_request
//   parse_slot_prompt_similarity
//   parse_positive_int_config
//   wrap_stream_chunk

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
    auto r = std::make_unique<server_task_result_error>();
    r->id = id_;
    r->err_msg = msg;
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
        : vec(std::move(v)), tokens_evaluated(tok) {
        id = id_;
    }
    json to_json() override { return {{"embedding", vec}, {"tokens_evaluated", tokens_evaluated}}; }
};

static server_task_result_ptr make_embedding(int id_, std::vector<float> v = {0.1f, 0.2f, 0.3f}) {
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

// make_error uses the real server_task_result_error; verify is_error() is true.
TEST(GetResultErrorMessage, RealErrorType_IsErrorTrue) {
    auto r = make_error(3, "x");
    EXPECT_TRUE(r->is_error());
}

// Success results must NOT be flagged as errors.
TEST(GetResultErrorMessage, SuccessResult_IsErrorFalse) {
    auto r = make_ok(4);
    EXPECT_FALSE(r->is_error());
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
    int index;
    float score;
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

// A rerank result whose to_json() omits the "index" field entirely. The real
// upstream formatter always emits it, but a malformed/future result may not, and
// rerank_results_to_json must reject it rather than throw json::type_error.
struct fake_rerank_result_no_index : server_task_result {
    float score;
    explicit fake_rerank_result_no_index(int id_, float sc) : score(sc) { id = id_; }
    json to_json() override { return {{"score", score}}; }
};

static server_task_result_ptr make_rerank_no_index(int id_, float sc) {
    return std::make_unique<fake_rerank_result_no_index>(id_, sc);
}

TEST(RerankResultsToJson, MissingIndex_ThrowsInvalidArgument) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank_no_index(1, 0.5f));
    std::vector<std::string> docs = {"doc A"};

    EXPECT_THROW((void)rerank_results_to_json(results, docs), std::invalid_argument);
}

TEST(RerankResultsToJson, IndexAboveRange_ThrowsInvalidArgument) {
    // index 5 with only 2 documents is out of bounds — must throw, not OOB-index.
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, 5, 0.5f));
    std::vector<std::string> docs = {"doc zero", "doc one"};

    EXPECT_THROW((void)rerank_results_to_json(results, docs), std::invalid_argument);
}

TEST(RerankResultsToJson, NegativeIndex_ThrowsInvalidArgument) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_rerank(1, -1, 0.5f));
    std::vector<std::string> docs = {"only doc"};

    EXPECT_THROW((void)rerank_results_to_json(results, docs), std::invalid_argument);
}

// ============================================================
// parse_encoding_format
// ============================================================

TEST(ParseEncodingFormat, FieldAbsent_ReturnsFalse) { EXPECT_FALSE(parse_encoding_format({{"model", "x"}})); }

TEST(ParseEncodingFormat, ExplicitFloat_ReturnsFalse) {
    EXPECT_FALSE(parse_encoding_format({{"encoding_format", "float"}}));
}

TEST(ParseEncodingFormat, Base64_ReturnsTrue) { EXPECT_TRUE(parse_encoding_format({{"encoding_format", "base64"}})); }

TEST(ParseEncodingFormat, UnknownFormat_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_encoding_format({{"encoding_format", "binary"}}), std::invalid_argument);
}

TEST(ParseEncodingFormat, EmptyString_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_encoding_format({{"encoding_format", ""}}), std::invalid_argument);
}

TEST(ParseEncodingFormat, ErrorMessage_MentionsBothValidOptions) {
    try {
        (void)parse_encoding_format({{"encoding_format", "hex"}});
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("float"), std::string::npos);
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
    json prompt = extract_embedding_prompt({{"input", "from input"}, {"content", "from content"}}, flag);
    EXPECT_EQ(prompt, "from input");
    EXPECT_FALSE(flag);
}

TEST(ExtractEmbeddingPrompt, NeitherKey_ThrowsInvalidArgument) {
    bool flag = false;
    EXPECT_THROW((void)extract_embedding_prompt({{"model", "x"}}, flag), std::invalid_argument);
}

TEST(ExtractEmbeddingPrompt, EmptyBody_ThrowsInvalidArgument) {
    bool flag = false;
    EXPECT_THROW((void)extract_embedding_prompt(json::object(), flag), std::invalid_argument);
}

TEST(ExtractEmbeddingPrompt, ArrayPrompt_ReturnedAsIs) {
    bool flag = false;
    json prompt = extract_embedding_prompt({{"input", json::array({"sentence one", "sentence two"})}}, flag);
    ASSERT_TRUE(prompt.is_array());
    ASSERT_EQ(prompt.size(), 2u);
    EXPECT_EQ(prompt[0], "sentence one");
    EXPECT_EQ(prompt[1], "sentence two");
    EXPECT_FALSE(flag);
}

// ============================================================
// common_json enum trap (llama.cpp b10585, upstream #27511)
//
//   The upstream `json` alias is `common_json`, whose value constructors cover
//   bool / integral / floating-point / string / container — but the integral one
//   is `std::is_integral`-gated, and an *enum* is not integral.  An unscoped enum
//   therefore binds to `common_json_value(bool)` and silently serialises as
//   true/false.  Project code must cast enum values to `int` before putting them
//   in JSON; `jllama.cpp`'s two "vocab_type" emit sites do exactly that, and
//   `ModelMeta.getVocabType()` reads the result with Jackson's `asInt(0)`.
// ============================================================

TEST(CommonJsonEnumTrap, UncastEnumBecomesBoolean) {
    // documents the trap this guard exists for (tripwire: if upstream ever adds an
    // enum constructor, this flips and the cast convention can be revisited)
    const json j = json::object({{"vocab_type", LLAMA_VOCAB_TYPE_WPM}});
    EXPECT_TRUE(j.at("vocab_type").is_boolean());
}

TEST(CommonJsonEnumTrap, ExplicitIntCastKeepsTheNumericValue) {
    const json j = json::object({{"vocab_type", static_cast<int>(LLAMA_VOCAB_TYPE_WPM)}});
    EXPECT_TRUE(j.at("vocab_type").is_number_integer());
    EXPECT_EQ(j.at("vocab_type").get<int>(), static_cast<int>(LLAMA_VOCAB_TYPE_WPM));
}

// ============================================================
// is_infill_request
// ============================================================

TEST(IsInfillRequest, HasInputPrefix_ReturnsTrue) { EXPECT_TRUE(is_infill_request({{"input_prefix", "def f():"}})); }

TEST(IsInfillRequest, HasInputSuffix_ReturnsTrue) { EXPECT_TRUE(is_infill_request({{"input_suffix", "return 1"}})); }

TEST(IsInfillRequest, HasBoth_ReturnsTrue) {
    EXPECT_TRUE(is_infill_request({{"input_prefix", "def f():"}, {"input_suffix", "return 1"}}));
}

TEST(IsInfillRequest, HasNeither_ReturnsFalse) { EXPECT_FALSE(is_infill_request({{"prompt", "hello"}})); }

TEST(IsInfillRequest, EmptyBody_ReturnsFalse) { EXPECT_FALSE(is_infill_request(json::object())); }

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
    EXPECT_THROW((void)parse_slot_prompt_similarity({{"slot_prompt_similarity", -0.1f}}), std::invalid_argument);
}

TEST(ParseSlotPromptSimilarity, TooHigh_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_slot_prompt_similarity({{"slot_prompt_similarity", 1.1f}}), std::invalid_argument);
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
    EXPECT_THROW((void)parse_positive_int_config({{"n_threads", 0}}, "n_threads"), std::invalid_argument);
}

TEST(ParsePositiveIntConfig, Negative_ThrowsInvalidArgument) {
    EXPECT_THROW((void)parse_positive_int_config({{"n_threads", -4}}, "n_threads"), std::invalid_argument);
}

TEST(ParsePositiveIntConfig, ErrorMessage_ContainsKeyName) {
    try {
        (void)parse_positive_int_config({{"n_threads_batch", 0}}, "n_threads_batch");
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("n_threads_batch"), std::string::npos);
    }
}

// ============================================================
// wrap_stream_chunk
// ============================================================

TEST(WrapStreamChunk, ObjectPayload_NotStopped) {
    json chunk = {{"object", "chat.completion.chunk"}, {"choices", json::array({{{"delta", {{"content", "hi"}}}}})}};
    json out = wrap_stream_chunk(chunk, false);
    EXPECT_FALSE(out.at("stop").get<bool>());
    ASSERT_TRUE(out.at("data").is_object());
    EXPECT_EQ(out.at("data").at("object").get<std::string>(), "chat.completion.chunk");
}

TEST(WrapStreamChunk, ArrayPayload_Stopped) {
    json final_chunks =
        json::array({{{"choices", json::array({{{"finish_reason", "stop"}, {"delta", json::object()}}})}},
                     {{"usage", {{"completion_tokens", 3}}}}});
    json out = wrap_stream_chunk(final_chunks, true);
    EXPECT_TRUE(out.at("stop").get<bool>());
    ASSERT_TRUE(out.at("data").is_array());
    EXPECT_EQ(out.at("data").size(), 2u);
}

TEST(WrapStreamChunk, StopFlagPropagates) {
    EXPECT_TRUE(wrap_stream_chunk(json::object(), true).at("stop").get<bool>());
    EXPECT_FALSE(wrap_stream_chunk(json::object(), false).at("stop").get<bool>());
}

TEST(WrapStreamChunk, NullPayload_DataIsNull) {
    json out = wrap_stream_chunk(json(), false);
    EXPECT_TRUE(out.at("data").is_null());
    EXPECT_FALSE(out.at("stop").get<bool>());
}

TEST(WrapStreamChunk, ExactlyTwoKeys) {
    json out = wrap_stream_chunk(json::object(), false);
    EXPECT_EQ(out.size(), 2u);
    EXPECT_TRUE(out.contains("data"));
    EXPECT_TRUE(out.contains("stop"));
}
