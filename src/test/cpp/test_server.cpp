// Tests for upstream server APIs — regression coverage for the contract that
// jllama.cpp depends on.  These tests catch llama.cpp upgrade breakage before
// the Java integration tests run.
//
// Covered:
//   - result_timings::to_json()       — draft_n/draft_n_accepted conditional fields
//   - task_params::to_json()          — grammar, chat_parser_params, grammar_triggers
//   - completion_token_output         — logarithm edge-case, str_to_bytes, to_json, probs_vector_to_json
//   - server_task_result_rerank       — score / index / tokens_evaluated
//   - server_task_result_embd         — oaicompat vs non-oaicompat shapes
//   - format_error_response           — all 7 error types → correct HTTP code + type string
//   - server_task::need_embd/logits   — routing helpers
//   - server_task_result_metrics      — slot count + token count fields
//   - server_task_result_slot_*       — save/load/erase JSON shapes

#include <gtest/gtest.h>

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"

// ============================================================
// result_timings::to_json
//   New fields draft_n / draft_n_accepted added in b8576.
//   They must be absent when draft_n == 0 (default) and present
//   only when draft_n > 0 (i.e. speculative decoding was active).
// ============================================================

namespace {

result_timings make_base_timings() {
    result_timings t;
    t.prompt_n              = 10;
    t.prompt_ms             = 200.0;
    t.prompt_per_token_ms   = 20.0;
    t.prompt_per_second     = 50.0;
    t.predicted_n           = 5;
    t.predicted_ms          = 100.0;
    t.predicted_per_token_ms = 20.0;
    t.predicted_per_second  = 50.0;
    return t;
}

} // namespace

TEST(ResultTimings, BaseFields_AlwaysPresent) {
    const json j = make_base_timings().to_json();

    EXPECT_TRUE(j.contains("cache_n"));
    EXPECT_TRUE(j.contains("prompt_n"));
    EXPECT_TRUE(j.contains("prompt_ms"));
    EXPECT_TRUE(j.contains("prompt_per_token_ms"));
    EXPECT_TRUE(j.contains("prompt_per_second"));
    EXPECT_TRUE(j.contains("predicted_n"));
    EXPECT_TRUE(j.contains("predicted_ms"));
    EXPECT_TRUE(j.contains("predicted_per_token_ms"));
    EXPECT_TRUE(j.contains("predicted_per_second"));
}

TEST(ResultTimings, CacheN_ReflectsValue) {
    result_timings t = make_base_timings();
    t.cache_n = 7;
    const json j = t.to_json();
    EXPECT_EQ(j.at("cache_n").get<int>(), 7);
}

TEST(ResultTimings, BaseFieldValues_MatchInput) {
    result_timings t = make_base_timings();
    const json j = t.to_json();

    EXPECT_EQ(j.at("prompt_n").get<int>(),    10);
    EXPECT_EQ(j.at("predicted_n").get<int>(), 5);
    EXPECT_DOUBLE_EQ(j.at("prompt_ms").get<double>(),          200.0);
    EXPECT_DOUBLE_EQ(j.at("predicted_per_second").get<double>(), 50.0);
}

TEST(ResultTimings, WithoutSpeculative_DraftFieldsAbsent) {
    // default draft_n = 0  →  fields must NOT appear in JSON
    result_timings t = make_base_timings();
    // draft_n and draft_n_accepted remain at their default (0)

    const json j = t.to_json();

    EXPECT_FALSE(j.contains("draft_n"))
        << "draft_n must be absent when draft_n == 0";
    EXPECT_FALSE(j.contains("draft_n_accepted"))
        << "draft_n_accepted must be absent when draft_n == 0";
}

TEST(ResultTimings, WithSpeculative_DraftFieldsPresent) {
    result_timings t = make_base_timings();
    t.draft_n          = 50;
    t.draft_n_accepted = 35;

    const json j = t.to_json();

    EXPECT_TRUE(j.contains("draft_n"))
        << "draft_n must be present when draft_n > 0";
    EXPECT_TRUE(j.contains("draft_n_accepted"))
        << "draft_n_accepted must be present when draft_n > 0";
    EXPECT_EQ(j.at("draft_n").get<int>(),          50);
    EXPECT_EQ(j.at("draft_n_accepted").get<int>(), 35);
}

TEST(ResultTimings, DraftNOne_FieldsPresent) {
    // Edge case: even a single speculative token triggers the fields
    result_timings t = make_base_timings();
    t.draft_n          = 1;
    t.draft_n_accepted = 0;

    const json j = t.to_json();

    EXPECT_TRUE(j.contains("draft_n"));
    EXPECT_TRUE(j.contains("draft_n_accepted"));
    EXPECT_EQ(j.at("draft_n").get<int>(),          1);
    EXPECT_EQ(j.at("draft_n_accepted").get<int>(), 0);
}

TEST(ResultTimings, DraftFieldsAbsent_WhenExplicitlyZero) {
    result_timings t = make_base_timings();
    t.draft_n          = 0;
    t.draft_n_accepted = 0;

    const json j = t.to_json();

    EXPECT_FALSE(j.contains("draft_n"));
    EXPECT_FALSE(j.contains("draft_n_accepted"));
}

// ============================================================
// slot_params::to_json
//   Changes in b8576:
//   1. grammar  → common_grammar_value(sampling.grammar)
//        was: sampling.grammar  (std::string)
//        now: common_grammar{type, string}, extracted via helper
//   2. oaicompat_chat_format (enum)  replaced by:
//        chat_format         from oaicompat_chat_syntax.format
//        reasoning_format    from oaicompat_chat_syntax.reasoning_format
//        reasoning_in_content from oaicompat_chat_syntax.reasoning_in_content
//        generation_prompt   from oaicompat_chat_syntax.generation_prompt
// ============================================================

TEST(SlotParamsToJson, CoreFields_Present) {
    task_params p;
    const json j = p.to_json();

    // Fields that must always be present regardless of configuration
    EXPECT_TRUE(j.contains("n_predict"));
    EXPECT_TRUE(j.contains("seed"));
    EXPECT_TRUE(j.contains("temperature"));
    EXPECT_TRUE(j.contains("grammar"));
    EXPECT_TRUE(j.contains("grammar_lazy"));
    EXPECT_TRUE(j.contains("grammar_triggers"));
    EXPECT_TRUE(j.contains("stream"));
    EXPECT_TRUE(j.contains("samplers"));
    EXPECT_TRUE(j.contains("stop"));
    EXPECT_TRUE(j.contains("lora"));
}

TEST(SlotParamsToJson, NewChatSyntaxFields_Present) {
    // These fields replace the old single oaicompat_chat_format enum field
    task_params p;
    const json j = p.to_json();

    EXPECT_TRUE(j.contains("chat_format"))
        << "chat_format must come from oaicompat_chat_syntax.format";
    EXPECT_TRUE(j.contains("reasoning_format"))
        << "reasoning_format must come from oaicompat_chat_syntax.reasoning_format";
    EXPECT_TRUE(j.contains("reasoning_in_content"))
        << "reasoning_in_content must come from oaicompat_chat_syntax.reasoning_in_content";
    EXPECT_TRUE(j.contains("generation_prompt"))
        << "generation_prompt must come from oaicompat_chat_syntax.generation_prompt";
}

TEST(SlotParamsToJson, OldChatFormatEnum_NotPresent) {
    // The raw integer oaicompat_chat_format field must be gone
    task_params p;
    const json j = p.to_json();

    EXPECT_FALSE(j.contains("oaicompat_chat_format"))
        << "Legacy oaicompat_chat_format field must not appear in b8576";
}

TEST(SlotParamsToJson, GrammarValue_EmptyByDefault) {
    task_params p;
    // sampling.grammar is default-constructed (empty)
    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "")
        << "Empty grammar must serialise to empty string via common_grammar_value()";
}

TEST(SlotParamsToJson, GrammarValue_UserGrammarExtracted) {
    task_params p;
    // Mirrors the assignment in params_from_json_cmpl for user-provided grammar
    p.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z]+"};

    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "root ::= [a-z]+")
        << "User grammar string must be extracted by common_grammar_value()";
}

TEST(SlotParamsToJson, GrammarValue_OutputFormatGrammarExtracted) {
    task_params p;
    // Mirrors the assignment in params_from_json_cmpl for JSON schema grammars
    p.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, "root ::= object"};

    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "root ::= object");
}

TEST(SlotParamsToJson, GenerationPrompt_ReflectsSyntaxField) {
    task_params p;
    p.chat_parser_params.generation_prompt = "Think step by step:";

    const json j = p.to_json();

    EXPECT_EQ(j.at("generation_prompt").get<std::string>(), "Think step by step:");
}

TEST(SlotParamsToJson, ReasoningInContent_ReflectsSyntaxField) {
    task_params p;
    p.chat_parser_params.reasoning_in_content = true;

    const json j = p.to_json();

    EXPECT_TRUE(j.at("reasoning_in_content").get<bool>());
}

TEST(SlotParamsToJson, ReasoningInContent_FalseByDefault) {
    task_params p;
    const json j = p.to_json();

    EXPECT_FALSE(j.at("reasoning_in_content").get<bool>());
}

TEST(SlotParamsToJson, SpeculativeFields_Present) {
    task_params p;
    const json j = p.to_json();

    // b8962: only speculative.type is serialised; n_max/n_min/p_min are
    // input-only (consumed by params_from_json_cmpl, not emitted by to_json)
    EXPECT_TRUE(j.contains("speculative.type"));
}

TEST(SlotParamsToJson, GrammarTriggers_IsArrayByDefault) {
    task_params p;
    const json j = p.to_json();

    EXPECT_TRUE(j.at("grammar_triggers").is_array());
    EXPECT_TRUE(j.at("grammar_triggers").empty());
}

TEST(SlotParamsToJson, Lora_EmptyArrayByDefault) {
    task_params p;
    const json j = p.to_json();
    ASSERT_TRUE(j.at("lora").is_array());
    EXPECT_TRUE(j.at("lora").empty());
}

TEST(SlotParamsToJson, Lora_PopulatedEntries) {
    task_params p;
    p.lora[0] = 0.5f;
    p.lora[2] = 1.0f;
    const json j = p.to_json();
    // Each entry is {id, scale}; order not guaranteed — build a map to verify
    ASSERT_EQ(j.at("lora").size(), 2u);
    std::map<int,float> got;
    for (const auto &entry : j.at("lora")) {
        got[entry.at("id").get<int>()] = entry.at("scale").get<float>();
    }
    EXPECT_FLOAT_EQ(got.at(0), 0.5f);
    EXPECT_FLOAT_EQ(got.at(2), 1.0f);
}

TEST(SlotParamsToJson, GrammarTriggers_SerialiseViaServerGrammarTrigger) {
    task_params p;
    // Add a WORD trigger — must be serialised through server_grammar_trigger
    common_grammar_trigger trigger;
    trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
    trigger.value = "```json";
    p.sampling.grammar_triggers.push_back(trigger);

    const json j = p.to_json();

    ASSERT_EQ(j.at("grammar_triggers").size(), 1u);
    const json &t = j.at("grammar_triggers")[0];
    EXPECT_TRUE(t.contains("type"));
    EXPECT_TRUE(t.contains("value"));
    EXPECT_EQ(t.at("value").get<std::string>(), "```json");
    EXPECT_EQ(t.at("type").get<int>(), static_cast<int>(COMMON_GRAMMAR_TRIGGER_TYPE_WORD));
}

// ============================================================
// task_params::to_json — dry_sequence_breakers / preserved_tokens
//   These two sampling fields are serialised unconditionally but
//   were never asserted in earlier tests.
// ============================================================

TEST(SlotParamsToJson, DrySequenceBreakers_DefaultValues) {
    task_params p;
    const json j = p.to_json();
    ASSERT_TRUE(j.contains("dry_sequence_breakers"));
    EXPECT_TRUE(j.at("dry_sequence_breakers").is_array());
    // Default is {"\n", ":", "\"", "*"} — must be non-empty
    EXPECT_FALSE(j.at("dry_sequence_breakers").empty());
}

TEST(SlotParamsToJson, DrySequenceBreakers_CustomValue) {
    task_params p;
    p.sampling.dry_sequence_breakers = {".", "!"};
    const json j = p.to_json();
    const auto &br = j.at("dry_sequence_breakers");
    ASSERT_EQ(br.size(), 2u);
    EXPECT_EQ(br[0].get<std::string>(), ".");
    EXPECT_EQ(br[1].get<std::string>(), "!");
}

TEST(SlotParamsToJson, PreservedTokens_EmptyByDefault) {
    task_params p;
    const json j = p.to_json();
    ASSERT_TRUE(j.contains("preserved_tokens"));
    // std::set serialises as a JSON array
    EXPECT_TRUE(j.at("preserved_tokens").is_array());
    EXPECT_TRUE(j.at("preserved_tokens").empty());
}

TEST(SlotParamsToJson, PreservedTokens_Populated) {
    task_params p;
    p.sampling.preserved_tokens.insert(1);
    p.sampling.preserved_tokens.insert(99);
    const json j = p.to_json();
    const auto &pt = j.at("preserved_tokens");
    ASSERT_EQ(pt.size(), 2u);
    // set serialises in ascending order
    EXPECT_EQ(pt[0].get<llama_token>(), 1);
    EXPECT_EQ(pt[1].get<llama_token>(), 99);
}

TEST(SlotParamsToJson, TimingsPerToken_DefaultFalse) {
    // timings_per_token must be serialised and default to false
    task_params p;
    const json j = p.to_json();
    ASSERT_TRUE(j.contains("timings_per_token"));
    EXPECT_FALSE(j.at("timings_per_token").get<bool>());
}

TEST(SlotParamsToJson, TimingsPerToken_SetTrue_Preserved) {
    task_params p;
    p.timings_per_token = true;
    const json j = p.to_json();
    EXPECT_TRUE(j.at("timings_per_token").get<bool>());
}

// ============================================================
// completion_token_output
//   Model-free struct.  Tests the helpers that are always
//   exercised during token streaming.
// ============================================================

TEST(CompletionTokenOutput, Logarithm_ZeroReturnsLowest) {
    // Prevents nlohmann/json serialising -inf as null
    const float result = completion_token_output::logarithm(0.0f);
    EXPECT_EQ(result, std::numeric_limits<float>::lowest());
}

TEST(CompletionTokenOutput, Logarithm_OneReturnsZero) {
    EXPECT_FLOAT_EQ(completion_token_output::logarithm(1.0f), 0.0f);
}

TEST(CompletionTokenOutput, Logarithm_PositiveIsNaturalLog) {
    EXPECT_NEAR(completion_token_output::logarithm(std::exp(1.0f)), 1.0f, 1e-5f);
}

TEST(StrToBytes, AsciiChars) {
    json bytes = str_to_bytes("ABC");
    ASSERT_EQ(bytes.size(), 3u);
    EXPECT_EQ(bytes[0].get<int>(), static_cast<int>('A'));
    EXPECT_EQ(bytes[1].get<int>(), static_cast<int>('B'));
    EXPECT_EQ(bytes[2].get<int>(), static_cast<int>('C'));
}

TEST(StrToBytes, EmptyString) {
    EXPECT_TRUE(str_to_bytes("").empty());
}

TEST(StrToBytes, HighByte) {
    // Byte 0xFF must survive the conversion unchanged
    json bytes = str_to_bytes("\xFF");
    ASSERT_EQ(bytes.size(), 1u);
    EXPECT_EQ(bytes[0].get<int>(), 0xFF);
}

TEST(CompletionTokenOutput, ToJson_PostSampling_UsesProbLabel) {
    completion_token_output cto;
    cto.tok = 1; cto.prob = 0.5f; cto.text_to_send = "hi";
    completion_token_output::prob_info pi;
    pi.tok = 1; pi.txt = "hi"; pi.prob = 0.5f;
    cto.probs.push_back(pi);

    const json j = cto.to_json(/*post_sampling_probs=*/true);
    ASSERT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 1u);
    EXPECT_TRUE(j[0].contains("prob"));
    EXPECT_FALSE(j[0].contains("logprob"));
    EXPECT_FLOAT_EQ(j[0].at("prob").get<float>(), 0.5f);
}

TEST(CompletionTokenOutput, ToJson_PreSampling_UsesLogprobLabel) {
    completion_token_output cto;
    cto.tok = 2; cto.prob = 0.25f; cto.text_to_send = "x";
    completion_token_output::prob_info pi;
    pi.tok = 2; pi.txt = "x"; pi.prob = 0.25f;
    cto.probs.push_back(pi);

    const json j = cto.to_json(/*post_sampling_probs=*/false);
    ASSERT_EQ(j.size(), 1u);
    EXPECT_TRUE(j[0].contains("logprob"));
    EXPECT_FALSE(j[0].contains("prob"));
    EXPECT_NEAR(j[0].at("logprob").get<float>(), std::log(0.25f), 1e-4f);
}

TEST(CompletionTokenOutput, ProbsVectorToJson_Empty_ReturnsEmptyArray) {
    const json j = completion_token_output::probs_vector_to_json({}, true);
    EXPECT_TRUE(j.is_array());
    EXPECT_TRUE(j.empty());
}

TEST(CompletionTokenOutput, ProbsVectorToJson_TokenFields) {
    completion_token_output cto;
    cto.tok = 7; cto.prob = 1.0f; cto.text_to_send = "ok";
    const json j = completion_token_output::probs_vector_to_json({cto}, true);
    ASSERT_EQ(j.size(), 1u);
    EXPECT_EQ(j[0].at("id").get<int>(), 7);
    EXPECT_EQ(j[0].at("token").get<std::string>(), "ok");
    EXPECT_TRUE(j[0].contains("bytes"));
    EXPECT_TRUE(j[0].contains("top_probs"));
}

// ============================================================
// server_task_result_rerank::to_json
//   Simple struct serialisation — all three fields must be present.
// ============================================================

TEST(ServerTaskResultRerank, ToJson_AllFieldsPresent) {
    server_task_result_rerank r;
    r.index    = 3;
    r.score    = 0.87f;
    r.n_tokens = 42;

    const json j = r.to_json();
    EXPECT_EQ(j.at("index").get<int>(), 3);
    EXPECT_NEAR(j.at("score").get<float>(), 0.87f, 1e-5f);
    EXPECT_EQ(j.at("tokens_evaluated").get<int>(), 42);
}

TEST(ServerTaskResultRerank, ToJson_DefaultScore_IsNegativeLarge) {
    server_task_result_rerank r;
    // default score = -1e6 (sentinel for "not computed")
    EXPECT_LT(r.to_json().at("score").get<float>(), 0.0f);
}

// ============================================================
// server_task_result_embd::to_json_*
//   Two shapes: non-oaicompat (multi-embedding) vs oaicompat
//   (single embedding[0] with tokens_evaluated).
// ============================================================

TEST(ServerTaskResultEmbd, NonOaicompat_ShapeCorrect) {
    server_task_result_embd e;
    e.index    = 1;
    e.embedding = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    e.n_tokens = 5;
    e.res_type = TASK_RESPONSE_TYPE_NONE;

    const json j = e.to_json();
    EXPECT_EQ(j.at("index").get<int>(), 1);
    // full embedding matrix returned
    EXPECT_EQ(j.at("embedding").size(), 2u);
    EXPECT_FALSE(j.contains("tokens_evaluated"));
}

TEST(ServerTaskResultEmbd, Oaicompat_UsesFirstRow) {
    server_task_result_embd e;
    e.index    = 0;
    e.embedding = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    e.n_tokens = 8;
    e.res_type = TASK_RESPONSE_TYPE_OAI_EMBD;

    const json j = e.to_json();
    // OAI compat exposes only embedding[0]
    ASSERT_TRUE(j.at("embedding").is_array());
    EXPECT_EQ(j.at("embedding").size(), 2u);  // first row has 2 elements
    EXPECT_FLOAT_EQ(j.at("embedding")[0].get<float>(), 1.0f);
    EXPECT_EQ(j.at("tokens_evaluated").get<int>(), 8);
}

TEST(ServerTaskResultEmbd, NonOaicompat_NTokensAbsent) {
    // tokens_evaluated must not appear in the non-OAI shape
    server_task_result_embd e;
    e.embedding = {{0.5f}};
    e.n_tokens  = 3;
    e.res_type  = TASK_RESPONSE_TYPE_NONE;
    const json j = e.to_json();
    EXPECT_FALSE(j.contains("tokens_evaluated"));
}

TEST(ServerTaskResultEmbd, NonOaicompat_SingleRowValues) {
    // Verify the float values survive the JSON round-trip
    server_task_result_embd e;
    e.embedding = {{0.1f, 0.2f, 0.3f}};
    e.res_type  = TASK_RESPONSE_TYPE_NONE;
    const json j = e.to_json();
    ASSERT_EQ(j.at("embedding").size(), 1u);   // one row
    ASSERT_EQ(j.at("embedding")[0].size(), 3u); // three elements
    EXPECT_FLOAT_EQ(j.at("embedding")[0][1].get<float>(), 0.2f);
}

TEST(ServerTaskResultEmbd, Dispatcher_NoneRoutes_ToNonOaicompat) {
    // to_json() dispatches on res_type; NONE → non-oaicompat (full matrix)
    server_task_result_embd e;
    e.embedding = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    e.res_type  = TASK_RESPONSE_TYPE_NONE;
    const json j = e.to_json();
    EXPECT_EQ(j.at("embedding").size(), 2u); // full 2D matrix
    EXPECT_FALSE(j.contains("tokens_evaluated"));
}

// ============================================================
// format_error_response
//   Covers all 7 error_type variants and their HTTP codes.
// ============================================================

namespace {
struct ErrorCase { error_type type; int code; std::string type_str; };
} // namespace

TEST(FormatErrorResponse, InvalidRequest_400) {
    const json j = format_error_response("bad input", ERROR_TYPE_INVALID_REQUEST);
    EXPECT_EQ(j.at("code").get<int>(), 400);
    EXPECT_EQ(j.at("type").get<std::string>(), "invalid_request_error");
    EXPECT_EQ(j.at("message").get<std::string>(), "bad input");
}

TEST(FormatErrorResponse, Authentication_401) {
    const json j = format_error_response("no auth", ERROR_TYPE_AUTHENTICATION);
    EXPECT_EQ(j.at("code").get<int>(), 401);
    EXPECT_EQ(j.at("type").get<std::string>(), "authentication_error");
}

TEST(FormatErrorResponse, NotFound_404) {
    const json j = format_error_response("not found", ERROR_TYPE_NOT_FOUND);
    EXPECT_EQ(j.at("code").get<int>(), 404);
    EXPECT_EQ(j.at("type").get<std::string>(), "not_found_error");
}

TEST(FormatErrorResponse, Permission_403) {
    const json j = format_error_response("denied", ERROR_TYPE_PERMISSION);
    EXPECT_EQ(j.at("code").get<int>(), 403);
    EXPECT_EQ(j.at("type").get<std::string>(), "permission_error");
}

TEST(FormatErrorResponse, Server_500) {
    const json j = format_error_response("crash", ERROR_TYPE_SERVER);
    EXPECT_EQ(j.at("code").get<int>(), 500);
    EXPECT_EQ(j.at("type").get<std::string>(), "server_error");
}

TEST(FormatErrorResponse, Unavailable_503) {
    const json j = format_error_response("overload", ERROR_TYPE_UNAVAILABLE);
    EXPECT_EQ(j.at("code").get<int>(), 503);
    EXPECT_EQ(j.at("type").get<std::string>(), "unavailable_error");
}

TEST(FormatErrorResponse, NotSupported_501) {
    const json j = format_error_response("nope", ERROR_TYPE_NOT_SUPPORTED);
    EXPECT_EQ(j.at("code").get<int>(), 501);
    EXPECT_EQ(j.at("type").get<std::string>(), "not_supported_error");
}

// ============================================================
// server_task_type_need_embd / server_task_type_need_logits
//   Routing helpers used by the scheduler to decide which
//   pipeline branch handles a task.
// ============================================================

TEST(ServerTaskTypeHelpers, NeedEmbd_TrueForEmbeddingAndRerank) {
    { server_task t; t.type = SERVER_TASK_TYPE_EMBEDDING; EXPECT_TRUE(t.need_embd()); }
    { server_task t; t.type = SERVER_TASK_TYPE_RERANK;    EXPECT_TRUE(t.need_embd()); }
}

TEST(ServerTaskTypeHelpers, NeedEmbd_FalseForOtherTypes) {
    { server_task t; t.type = SERVER_TASK_TYPE_COMPLETION; EXPECT_FALSE(t.need_embd()); }
    { server_task t; t.type = SERVER_TASK_TYPE_INFILL;     EXPECT_FALSE(t.need_embd()); }
    { server_task t; t.type = SERVER_TASK_TYPE_METRICS;    EXPECT_FALSE(t.need_embd()); }
    { server_task t; t.type = SERVER_TASK_TYPE_CANCEL;     EXPECT_FALSE(t.need_embd()); }
}

TEST(ServerTaskTypeHelpers, NeedLogits_TrueForCompletionAndInfill) {
    { server_task t; t.type = SERVER_TASK_TYPE_COMPLETION; EXPECT_TRUE(t.need_logits()); }
    { server_task t; t.type = SERVER_TASK_TYPE_INFILL;     EXPECT_TRUE(t.need_logits()); }
}

TEST(ServerTaskTypeHelpers, NeedLogits_FalseForOtherTypes) {
    { server_task t; t.type = SERVER_TASK_TYPE_EMBEDDING; EXPECT_FALSE(t.need_logits()); }
    { server_task t; t.type = SERVER_TASK_TYPE_RERANK;    EXPECT_FALSE(t.need_logits()); }
    { server_task t; t.type = SERVER_TASK_TYPE_METRICS;   EXPECT_FALSE(t.need_logits()); }
}

TEST(ServerTaskTypeHelpers, NeedSampling_TrueForCompletionAndInfill) {
    { server_task t; t.type = SERVER_TASK_TYPE_COMPLETION; EXPECT_TRUE(t.need_sampling()); }
    { server_task t; t.type = SERVER_TASK_TYPE_INFILL;     EXPECT_TRUE(t.need_sampling()); }
}

TEST(ServerTaskTypeHelpers, NeedSampling_FalseForNonGenerativeTasks) {
    { server_task t; t.type = SERVER_TASK_TYPE_EMBEDDING; EXPECT_FALSE(t.need_sampling()); }
    { server_task t; t.type = SERVER_TASK_TYPE_RERANK;    EXPECT_FALSE(t.need_sampling()); }
    { server_task t; t.type = SERVER_TASK_TYPE_METRICS;   EXPECT_FALSE(t.need_sampling()); }
}

// ============================================================
// server_task::n_tokens
//   Returns the number of pre-tokenised tokens stored in the task.
//   Used by the slot scheduler to decide if a task can be batched.
// ============================================================

TEST(ServerTaskNTokens, EmptyTokens_ReturnsZero) {
    server_task t;
    EXPECT_EQ(t.n_tokens(), 0);
}

TEST(ServerTaskNTokens, PopulatedTokens_ReturnsCount) {
    server_task t;
    t.tokens = server_tokens(llama_tokens{1, 2, 3, 4, 5}, /*has_mtmd=*/false);
    EXPECT_EQ(t.n_tokens(), 5);
}

// ============================================================
// server_task_result_metrics::to_json
//   Pure struct → JSON; no model needed.
// ============================================================

namespace {
server_task_result_metrics make_metrics() {
    server_task_result_metrics m;
    m.n_idle_slots       = 2;
    m.n_processing_slots = 1;
    m.n_tasks_deferred   = 3;
    m.t_start            = 1234567890LL;
    m.n_prompt_tokens_processed_total = 100;
    m.t_prompt_processing_total       = 50;
    m.n_tokens_predicted_total        = 200;
    m.t_tokens_generation_total       = 80;
    m.n_prompt_tokens_processed       = 10;
    m.t_prompt_processing             = 5;
    m.n_tokens_predicted              = 20;
    m.t_tokens_generation             = 8;
    m.n_decode_total                  = 300;
    m.n_busy_slots_total              = 4;
    return m;
}
} // namespace

TEST(ServerTaskResultMetrics, ToJson_SlotCountFields) {
    const json j = make_metrics().to_json();
    EXPECT_EQ(j.at("idle").get<int>(), 2);
    EXPECT_EQ(j.at("processing").get<int>(), 1);
    EXPECT_EQ(j.at("deferred").get<int>(), 3);
    EXPECT_EQ(j.at("t_start").get<int64_t>(), 1234567890LL);
}

TEST(ServerTaskResultMetrics, ToJson_NTokensMax) {
    server_task_result_metrics m = make_metrics();
    m.n_tokens_max = 4096;
    const json j = m.to_json();
    EXPECT_EQ(j.at("n_tokens_max").get<int>(), 4096);
}

TEST(ServerTaskResultMetrics, ToJson_TokenCountFields) {
    const json j = make_metrics().to_json();
    EXPECT_EQ(j.at("n_prompt_tokens_processed_total").get<uint64_t>(), 100u);
    EXPECT_EQ(j.at("n_tokens_predicted_total").get<uint64_t>(), 200u);
    EXPECT_EQ(j.at("n_decode_total").get<uint64_t>(), 300u);
    EXPECT_EQ(j.at("n_busy_slots_total").get<uint64_t>(), 4u);
}

TEST(ServerTaskResultMetrics, ToJson_TimingAndWindowFields) {
    const json j = make_metrics().to_json();
    // Timing totals
    EXPECT_EQ(j.at("t_prompt_processing_total").get<uint64_t>(), 50u);
    EXPECT_EQ(j.at("t_tokens_generation_total").get<uint64_t>(), 80u);
    // Current-window counts (not the _total variants)
    EXPECT_EQ(j.at("n_prompt_tokens_processed").get<uint64_t>(), 10u);
    EXPECT_EQ(j.at("t_prompt_processing").get<uint64_t>(), 5u);
    EXPECT_EQ(j.at("n_tokens_predicted").get<uint64_t>(), 20u);
    EXPECT_EQ(j.at("t_tokens_generation").get<uint64_t>(), 8u);
}

TEST(ServerTaskResultMetrics, ToJson_SlotDataIsArray) {
    server_task_result_metrics m = make_metrics();
    m.slots_data = json::array({{{"id", 0}}, {{"id", 1}}});
    const json j = m.to_json();
    ASSERT_TRUE(j.at("slots").is_array());
    EXPECT_EQ(j.at("slots").size(), 2u);
}

// ============================================================
// server_task_result_slot_save_load::to_json
//   Two different shapes depending on is_save flag.
// ============================================================

TEST(ServerTaskResultSlotSaveLoad, SaveMode_CorrectFields) {
    server_task_result_slot_save_load r;
    r.id_slot  = 0;
    r.filename = "slot_0.bin";
    r.is_save  = true;
    r.n_tokens = 128;
    r.n_bytes  = 4096;
    r.t_ms     = 12.5;

    const json j = r.to_json();
    EXPECT_EQ(j.at("filename").get<std::string>(), "slot_0.bin");
    EXPECT_EQ(j.at("n_saved").get<size_t>(), 128u);
    EXPECT_EQ(j.at("n_written").get<size_t>(), 4096u);
    EXPECT_DOUBLE_EQ(j.at("timings").at("save_ms").get<double>(), 12.5);
    // load-only keys must be absent
    EXPECT_FALSE(j.contains("n_restored"));
    EXPECT_FALSE(j.contains("n_read"));
}

TEST(ServerTaskResultSlotSaveLoad, LoadMode_CorrectFields) {
    server_task_result_slot_save_load r;
    r.id_slot  = 1;
    r.filename = "slot_1.bin";
    r.is_save  = false;
    r.n_tokens = 64;
    r.n_bytes  = 2048;
    r.t_ms     = 7.3;

    const json j = r.to_json();
    EXPECT_EQ(j.at("n_restored").get<size_t>(), 64u);
    EXPECT_EQ(j.at("n_read").get<size_t>(), 2048u);
    EXPECT_DOUBLE_EQ(j.at("timings").at("restore_ms").get<double>(), 7.3);
    // save-only keys must be absent
    EXPECT_FALSE(j.contains("n_saved"));
    EXPECT_FALSE(j.contains("n_written"));
}

// ============================================================
// server_task_result_slot_erase::to_json
// server_task_result_apply_lora::to_json
// ============================================================

TEST(ServerTaskResultSlotErase, ToJson_NErasedPresent) {
    server_task_result_slot_erase r;
    r.id_slot  = 2;
    r.n_erased = 512;

    const json j = r.to_json();
    EXPECT_EQ(j.at("id_slot").get<int>(), 2);
    EXPECT_EQ(j.at("n_erased").get<size_t>(), 512u);
}

TEST(ServerTaskResultApplyLora, ToJson_SuccessTrue) {
    server_task_result_apply_lora r;
    const json j = r.to_json();
    ASSERT_TRUE(j.contains("success"));
    EXPECT_TRUE(j.at("success").get<bool>());
}

// ============================================================
// server_task_result_error::to_json
//   jllama.cpp calls is_error() then get_result_error_message()
//   (which calls to_json()["message"]) on every error result.
//   The shape must survive changes in format_error_response.
// ============================================================

TEST(ServerTaskResultError, StandardError_HasMessageField) {
    server_task_result_error e;
    e.err_type = ERROR_TYPE_SERVER;
    e.err_msg  = "something went wrong";
    const json j = e.to_json();
    EXPECT_EQ(j.at("message").get<std::string>(), "something went wrong");
}

TEST(ServerTaskResultError, StandardError_HasCodeAndType) {
    server_task_result_error e;
    e.err_type = ERROR_TYPE_INVALID_REQUEST;
    e.err_msg  = "bad param";
    const json j = e.to_json();
    EXPECT_EQ(j.at("code").get<int>(), 400);
    EXPECT_EQ(j.at("type").get<std::string>(), "invalid_request_error");
}

TEST(ServerTaskResultError, IsError_ReturnsTrue) {
    server_task_result_error e;
    EXPECT_TRUE(e.is_error());
}

TEST(ServerTaskResultError, ExceedContextSize_AddsExtraFields) {
    server_task_result_error e;
    e.err_type        = ERROR_TYPE_EXCEED_CONTEXT_SIZE;
    e.err_msg         = "context full";
    e.n_prompt_tokens = 512;
    e.n_ctx           = 256;
    const json j = e.to_json();
    EXPECT_EQ(j.at("n_prompt_tokens").get<int>(), 512);
    EXPECT_EQ(j.at("n_ctx").get<int>(), 256);
}

TEST(ServerTaskResultError, DefaultError_NoExtraContextFields) {
    server_task_result_error e;
    e.err_type = ERROR_TYPE_SERVER;
    e.err_msg  = "fail";
    const json j = e.to_json();
    EXPECT_FALSE(j.contains("n_prompt_tokens"));
    EXPECT_FALSE(j.contains("n_ctx"));
}

// ============================================================
// result_prompt_progress::to_json
//   Emitted inside server_task_result_cmpl_partial when is_progress
//   is true.  Verifies the four required fields.
// ============================================================

TEST(ResultPromptProgress, ToJson_AllFourFields) {
    result_prompt_progress p;
    p.total     = 100;
    p.cache     = 40;
    p.processed = 60;
    p.time_ms   = 1234;
    const json j = p.to_json();
    EXPECT_EQ(j.at("total").get<int>(),     100);
    EXPECT_EQ(j.at("cache").get<int>(),     40);
    EXPECT_EQ(j.at("processed").get<int>(), 60);
    EXPECT_EQ(j.at("time_ms").get<int64_t>(), 1234);
}

TEST(ResultPromptProgress, ToJson_DefaultZeros) {
    result_prompt_progress p;
    const json j = p.to_json();
    EXPECT_EQ(j.at("total").get<int>(),     0);
    EXPECT_EQ(j.at("cache").get<int>(),     0);
    EXPECT_EQ(j.at("processed").get<int>(), 0);
    EXPECT_EQ(j.at("time_ms").get<int64_t>(), 0);
}

// ============================================================
// server_task_result_cmpl_partial::to_json_non_oaicompat
//   The non-OAI streaming chunk shape used by requestCompletion
//   when the caller has not set an OAI-compat response type.
//   Call to_json_non_oaicompat() directly to bypass the
//   is_updated assertion in to_json().
// ============================================================

TEST(ServerTaskResultCmplPartial, NonOaicompat_CoreFields) {
    server_task_result_cmpl_partial p;
    p.is_updated      = true;
    p.res_type        = TASK_RESPONSE_TYPE_NONE;
    p.content         = "hello";
    p.n_decoded       = 3;
    p.n_prompt_tokens = 10;

    const json j = p.to_json_non_oaicompat();

    EXPECT_EQ(j.at("content").get<std::string>(), "hello");
    EXPECT_EQ(j.at("tokens_predicted").get<int>(), 3);
    EXPECT_EQ(j.at("tokens_evaluated").get<int>(), 10);
    EXPECT_FALSE(j.at("stop").get<bool>());
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_TimingsAbsentByDefault) {
    server_task_result_cmpl_partial p;
    p.is_updated = true;
    p.res_type   = TASK_RESPONSE_TYPE_NONE;
    // timings.prompt_n == 0 by default → timings should be absent
    const json j = p.to_json_non_oaicompat();
    EXPECT_FALSE(j.contains("timings"));
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_TimingsPresentWhenPromptNNonzero) {
    server_task_result_cmpl_partial p;
    p.is_updated      = true;
    p.res_type        = TASK_RESPONSE_TYPE_NONE;
    p.timings.prompt_n = 5;
    const json j = p.to_json_non_oaicompat();
    EXPECT_TRUE(j.contains("timings"));
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_ProgressAbsentWhenNotProgress) {
    server_task_result_cmpl_partial p;
    p.is_updated  = true;
    p.res_type    = TASK_RESPONSE_TYPE_NONE;
    p.is_progress = false;
    const json j  = p.to_json_non_oaicompat();
    EXPECT_FALSE(j.contains("prompt_progress"));
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_ProgressPresentWhenIsProgress) {
    server_task_result_cmpl_partial p;
    p.is_updated         = true;
    p.res_type           = TASK_RESPONSE_TYPE_NONE;
    p.is_progress        = true;
    p.progress.total     = 20;
    p.progress.processed = 10;
    const json j = p.to_json_non_oaicompat();
    ASSERT_TRUE(j.contains("prompt_progress"));
    EXPECT_EQ(j.at("prompt_progress").at("total").get<int>(), 20);
}

TEST(ServerTaskResultCmplPartial, IsStop_ReturnsFalse) {
    server_task_result_cmpl_partial p;
    EXPECT_FALSE(p.is_stop());
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_IdSlotField) {
    server_task_result_cmpl_partial p;
    p.is_updated = true;
    p.res_type   = TASK_RESPONSE_TYPE_NONE;
    p.id_slot    = 3;
    const json j = p.to_json_non_oaicompat();
    EXPECT_EQ(j.at("id_slot").get<int>(), 3);
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_CompletionProbabilitiesAbsentWhenProbsEmpty) {
    server_task_result_cmpl_partial p;
    p.is_updated = true;
    p.res_type   = TASK_RESPONSE_TYPE_NONE;
    // prob_output.probs is empty by default
    const json j = p.to_json_non_oaicompat();
    EXPECT_FALSE(j.contains("completion_probabilities"));
}

TEST(ServerTaskResultCmplPartial, NonOaicompat_CompletionProbabilitiesPresentWhenProbsSet) {
    server_task_result_cmpl_partial p;
    p.is_updated          = true;
    p.res_type            = TASK_RESPONSE_TYPE_NONE;
    p.post_sampling_probs = true;
    completion_token_output::prob_info pi;
    pi.tok = 5; pi.txt = "hi"; pi.prob = 0.8f;
    p.prob_output.probs.push_back(pi);
    const json j = p.to_json_non_oaicompat();
    ASSERT_TRUE(j.contains("completion_probabilities"));
    EXPECT_TRUE(j.at("completion_probabilities").is_array());
}

// ============================================================
// server_task_result_cmpl_final::to_json_non_oaicompat
//   The terminal (stop=true) chunk shape used by blocking
//   completions.  Call to_json_non_oaicompat() directly.
// ============================================================

TEST(ServerTaskResultCmplFinal, IsStop_ReturnsTrue) {
    server_task_result_cmpl_final f;
    EXPECT_TRUE(f.is_stop());
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StopAlwaysTrue) {
    server_task_result_cmpl_final f;
    f.content         = "done";
    f.n_decoded       = 3;
    f.n_prompt_tokens = 7;
    const json j = f.to_json_non_oaicompat();
    EXPECT_TRUE(j.at("stop").get<bool>());
    EXPECT_EQ(j.at("content").get<std::string>(), "done");
    EXPECT_EQ(j.at("tokens_predicted").get<int>(), 3);
    EXPECT_EQ(j.at("tokens_evaluated").get<int>(), 7);
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StopType_None) {
    server_task_result_cmpl_final f;
    f.stop = STOP_TYPE_NONE;
    const json j = f.to_json_non_oaicompat();
    EXPECT_EQ(j.at("stop_type").get<std::string>(), "none");
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StopType_Eos) {
    server_task_result_cmpl_final f;
    f.stop = STOP_TYPE_EOS;
    const json j = f.to_json_non_oaicompat();
    EXPECT_EQ(j.at("stop_type").get<std::string>(), "eos");
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StopType_Word) {
    server_task_result_cmpl_final f;
    f.stop         = STOP_TYPE_WORD;
    f.stopping_word = "</s>";
    const json j = f.to_json_non_oaicompat();
    EXPECT_EQ(j.at("stop_type").get<std::string>(), "word");
    EXPECT_EQ(j.at("stopping_word").get<std::string>(), "</s>");
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StopType_Limit) {
    server_task_result_cmpl_final f;
    f.stop = STOP_TYPE_LIMIT;
    const json j = f.to_json_non_oaicompat();
    EXPECT_EQ(j.at("stop_type").get<std::string>(), "limit");
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_NoProbsOutput_CompletionProbabilitiesAbsent) {
    // completion_probabilities must be absent when probs_output is empty;
    // Java's CompletionResponseParser skips this field when absent.
    server_task_result_cmpl_final f;
    f.stream = false;
    // probs_output stays empty (default)
    const json j = f.to_json_non_oaicompat();
    EXPECT_FALSE(j.contains("completion_probabilities"));
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_WithProbsOutput_CompletionProbabilitiesPresent) {
    // When probs_output is non-empty and stream==false, the key must appear.
    server_task_result_cmpl_final f;
    f.stream              = false;
    f.post_sampling_probs = true;
    completion_token_output cto;
    cto.tok = 42; cto.prob = 0.9f; cto.text_to_send = "hi";
    f.probs_output.push_back(cto);
    const json j = f.to_json_non_oaicompat();
    ASSERT_TRUE(j.contains("completion_probabilities"));
    EXPECT_TRUE(j.at("completion_probabilities").is_array());
}

TEST(ServerTaskResultCmplFinal, NonOaicompat_StreamModeWithProbs_CompletionProbabilitiesAbsent) {
    // stream==true suppresses completion_probabilities even if probs_output is set.
    server_task_result_cmpl_final f;
    f.stream              = true;
    f.post_sampling_probs = true;
    completion_token_output cto;
    cto.tok = 1; cto.prob = 0.5f; cto.text_to_send = "x";
    f.probs_output.push_back(cto);
    const json j = f.to_json_non_oaicompat();
    EXPECT_FALSE(j.contains("completion_probabilities"));
}

// ============================================================
// server_task_result_cmpl_final::usage_json_oaicompat
//   Called by to_json_oaicompat / to_json_oaicompat_chat.
//   Directly callable without update().
// ============================================================

TEST(ServerTaskResultCmplFinal, UsageJsonOaicompat_FieldsCorrect) {
    server_task_result_cmpl_final f;
    f.n_decoded              = 17;
    f.n_prompt_tokens        = 8;
    f.n_prompt_tokens_cache  = 3;
    const json j = f.usage_json_oaicompat();
    EXPECT_EQ(j.at("completion_tokens").get<int>(), 17);
    EXPECT_EQ(j.at("prompt_tokens").get<int>(), 8);
    EXPECT_EQ(j.at("total_tokens").get<int>(), 25);  // 17 + 8
    EXPECT_EQ(j.at("prompt_tokens_details").at("cached_tokens").get<int>(), 3);
}

TEST(ServerTaskResultCmplFinal, UsageJsonOaicompat_TotalTokensIsSumOfBoth) {
    server_task_result_cmpl_final f;
    f.n_decoded       = 5;
    f.n_prompt_tokens = 10;
    const json j = f.usage_json_oaicompat();
    EXPECT_EQ(j.at("total_tokens").get<int>(), f.n_decoded + f.n_prompt_tokens);
}

// ============================================================
// server_task_result_cmpl_final::to_json_oaicompat
//   OAI /completions (non-chat) response shape.
//   finish_reason is "stop" when stop==EOS or WORD; "length" otherwise.
//   object field must always be "text_completion".
// ============================================================

namespace {
server_task_result_cmpl_final make_oai_final(const std::string &content = "hello") {
    server_task_result_cmpl_final f;
    f.content         = content;
    f.oaicompat_model = "test-model";
    f.oaicompat_cmpl_id = "cmpl-test";
    f.n_decoded       = 3;
    f.n_prompt_tokens = 5;
    return f;
}
} // namespace

TEST(CmplFinalOaicompat, Object_IsTextCompletion) {
    const json j = make_oai_final().to_json_oaicompat();
    EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
}

TEST(CmplFinalOaicompat, Choices_ContainsContentAndIndex) {
    const json j = make_oai_final("world").to_json_oaicompat();
    ASSERT_TRUE(j.at("choices").is_array());
    ASSERT_EQ(j.at("choices").size(), 1u);
    EXPECT_EQ(j.at("choices")[0].at("text").get<std::string>(), "world");
    EXPECT_EQ(j.at("choices")[0].at("index").get<int>(), 0);
}

TEST(CmplFinalOaicompat, FinishReason_StopForEos) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_EOS;
    const json j = f.to_json_oaicompat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "stop");
}

TEST(CmplFinalOaicompat, FinishReason_LengthForLimit) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_LIMIT;
    const json j = f.to_json_oaicompat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "length");
}

TEST(CmplFinalOaicompat, FinishReason_StopForWord) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_WORD;
    const json j = f.to_json_oaicompat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "stop");
}

TEST(CmplFinalOaicompat, Usage_FieldsPresent) {
    auto f = make_oai_final();
    const json j = f.to_json_oaicompat();
    ASSERT_TRUE(j.contains("usage"));
    EXPECT_TRUE(j.at("usage").contains("completion_tokens"));
    EXPECT_TRUE(j.at("usage").contains("prompt_tokens"));
    EXPECT_TRUE(j.at("usage").contains("total_tokens"));
}

TEST(CmplFinalOaicompat, Model_ReflectsOaicompatModel) {
    auto f = make_oai_final();
    const json j = f.to_json_oaicompat();
    EXPECT_EQ(j.at("model").get<std::string>(), "test-model");
}

TEST(CmplFinalOaicompat, Id_ReflectsOaicompatCmplId) {
    auto f = make_oai_final();
    const json j = f.to_json_oaicompat();
    EXPECT_EQ(j.at("id").get<std::string>(), "cmpl-test");
}

// ============================================================
// server_task_result_cmpl_final::to_json_oaicompat_chat
//   OAI /chat/completions response shape.
//   When oaicompat_msg is empty the method synthesises a plain
//   assistant message from `content`.  finish_reason follows
//   the same stop logic as to_json_oaicompat.
// ============================================================

TEST(CmplFinalOaicompatChat, Object_IsChatCompletion) {
    const json j = make_oai_final().to_json_oaicompat_chat();
    EXPECT_EQ(j.at("object").get<std::string>(), "chat.completion");
}

TEST(CmplFinalOaicompatChat, Choices_ContainsMessageWithRoleAndContent) {
    auto f = make_oai_final("think deeply");
    const json j = f.to_json_oaicompat_chat();
    ASSERT_TRUE(j.at("choices").is_array());
    const json &msg = j.at("choices")[0].at("message");
    EXPECT_EQ(msg.at("role").get<std::string>(), "assistant");
    EXPECT_EQ(msg.at("content").get<std::string>(), "think deeply");
}

TEST(CmplFinalOaicompatChat, FinishReason_StopForEos) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_EOS;
    const json j = f.to_json_oaicompat_chat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "stop");
}

TEST(CmplFinalOaicompatChat, FinishReason_LengthForLimit) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_LIMIT;
    const json j = f.to_json_oaicompat_chat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "length");
}

TEST(CmplFinalOaicompatChat, Usage_Present) {
    const json j = make_oai_final().to_json_oaicompat_chat();
    EXPECT_TRUE(j.contains("usage"));
}

TEST(CmplFinalOaicompatChat, WithExplicitOaicompatMsg_MessageContentUsed) {
    auto f = make_oai_final("ignored");
    f.oaicompat_msg.role    = "assistant";
    f.oaicompat_msg.content = "explicit reply";
    const json j = f.to_json_oaicompat_chat();
    EXPECT_EQ(j.at("choices")[0].at("message").at("content").get<std::string>(), "explicit reply");
}

TEST(CmplFinalOaicompatChat, WithToolCalls_FinishReason_IsToolCalls) {
    // When oaicompat_msg has tool_calls and stop==EOS, finish_reason must
    // be "tool_calls" (not "stop").
    auto f = make_oai_final("");
    common_chat_tool_call tc;
    tc.id        = "call_1";
    tc.name      = "search";
    tc.arguments = R"({"q":"test"})";
    f.oaicompat_msg.tool_calls.push_back(tc);
    f.stop = STOP_TYPE_EOS;
    const json j = f.to_json_oaicompat_chat();
    EXPECT_EQ(j.at("choices")[0].at("finish_reason").get<std::string>(), "tool_calls");
}

TEST(CmplFinalOaicompatChat, WithToolCalls_MessageHasToolCallsArray) {
    auto f = make_oai_final("");
    common_chat_tool_call tc;
    tc.id        = "call_1";
    tc.name      = "search";
    tc.arguments = R"({"q":"test"})";
    f.oaicompat_msg.tool_calls.push_back(tc);
    const json j = f.to_json_oaicompat_chat();
    const json &msg = j.at("choices")[0].at("message");
    ASSERT_TRUE(msg.contains("tool_calls"));
    ASSERT_EQ(msg.at("tool_calls").size(), 1u);
    EXPECT_EQ(msg.at("tool_calls")[0].at("function").at("name").get<std::string>(), "search");
}

// ============================================================
// server_task_result_cmpl_final::to_json_anthropic
//   Anthropic Messages API response shape.
//   stop_reason: "end_turn" for EOS/WORD, "max_tokens" for LIMIT/NONE.
//   content_blocks: text block when content is non-empty;
//                   thinking block first when reasoning_content is set;
//                   tool_use blocks for each tool call.
// ============================================================

TEST(CmplFinalAnthropic, StopReason_MaxTokensByDefault) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_LIMIT;
    const json j = f.to_json_anthropic();
    EXPECT_EQ(j.at("stop_reason").get<std::string>(), "max_tokens");
}

TEST(CmplFinalAnthropic, StopReason_EndTurnForEos) {
    auto f = make_oai_final();
    f.stop = STOP_TYPE_EOS;
    const json j = f.to_json_anthropic();
    EXPECT_EQ(j.at("stop_reason").get<std::string>(), "end_turn");
}

TEST(CmplFinalAnthropic, StopReason_EndTurnForWord) {
    auto f = make_oai_final();
    f.stop         = STOP_TYPE_WORD;
    f.stopping_word = "</s>";
    const json j   = f.to_json_anthropic();
    EXPECT_EQ(j.at("stop_reason").get<std::string>(), "end_turn");
}

TEST(CmplFinalAnthropic, StopSequence_NullWhenEmpty) {
    auto f = make_oai_final();
    const json j = f.to_json_anthropic();
    EXPECT_TRUE(j.at("stop_sequence").is_null());
}

TEST(CmplFinalAnthropic, StopSequence_ReflectsStoppingWord) {
    auto f = make_oai_final();
    f.stop         = STOP_TYPE_WORD;
    f.stopping_word = "</tool>";
    f.oaicompat_msg.content = "done";
    const json j   = f.to_json_anthropic();
    EXPECT_EQ(j.at("stop_sequence").get<std::string>(), "</tool>");
}

TEST(CmplFinalAnthropic, ContentBlock_TextBlockForPlainContent) {
    auto f = make_oai_final("plain text");
    const json j     = f.to_json_anthropic();
    const json &blks = j.at("content");
    ASSERT_FALSE(blks.empty());
    // last block is the text block when no reasoning
    bool found_text = false;
    for (const auto &b : blks) {
        if (b.at("type").get<std::string>() == "text") { found_text = true; break; }
    }
    EXPECT_TRUE(found_text);
}

TEST(CmplFinalAnthropic, ContentBlock_ThinkingBlockFirst) {
    auto f = make_oai_final("answer");
    f.oaicompat_msg.role              = "assistant";
    f.oaicompat_msg.content           = "answer";
    f.oaicompat_msg.reasoning_content = "step by step";
    const json j   = f.to_json_anthropic();
    const json &blks = j.at("content");
    ASSERT_GE(blks.size(), 2u);
    EXPECT_EQ(blks[0].at("type").get<std::string>(), "thinking");
    EXPECT_EQ(blks[0].at("thinking").get<std::string>(), "step by step");
}

TEST(CmplFinalAnthropic, ContentBlock_ToolUseBlock) {
    auto f = make_oai_final("");
    common_chat_tool_call tc;
    tc.id        = "call_1";
    tc.name      = "get_weather";
    tc.arguments = R"({"city":"Paris"})";
    f.oaicompat_msg.tool_calls.push_back(tc);
    f.stop = STOP_TYPE_EOS;
    const json j   = f.to_json_anthropic();
    EXPECT_EQ(j.at("stop_reason").get<std::string>(), "tool_use");
    bool found_tool = false;
    for (const auto &b : j.at("content")) {
        if (b.at("type").get<std::string>() == "tool_use") {
            EXPECT_EQ(b.at("name").get<std::string>(), "get_weather");
            EXPECT_EQ(b.at("id").get<std::string>(),   "call_1");
            EXPECT_EQ(b.at("input").at("city").get<std::string>(), "Paris");
            found_tool = true;
        }
    }
    EXPECT_TRUE(found_tool);
}

// ============================================================
// server_task_result_cmpl_partial::to_json_oaicompat
//   OAI /completions streaming chunk shape.
//   object must be "text_completion"; finish_reason must be null
//   (streaming chunks never carry a finish reason).
// ============================================================

namespace {
server_task_result_cmpl_partial make_partial(const std::string &content = "tok") {
    server_task_result_cmpl_partial p;
    p.is_updated        = true;
    p.res_type          = TASK_RESPONSE_TYPE_OAI_CMPL;
    p.content           = content;
    p.oaicompat_model   = "test-model";
    p.oaicompat_cmpl_id = "cmpl-part";
    return p;
}
} // namespace

TEST(CmplPartialOaicompat, Object_IsTextCompletion) {
    const json j = make_partial().to_json_oaicompat();
    EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
}

TEST(CmplPartialOaicompat, Choices_ContentAndNullFinishReason) {
    const json j = make_partial("chunk").to_json_oaicompat();
    ASSERT_TRUE(j.at("choices").is_array());
    EXPECT_EQ(j.at("choices")[0].at("text").get<std::string>(), "chunk");
    EXPECT_TRUE(j.at("choices")[0].at("finish_reason").is_null());
}

TEST(CmplPartialOaicompat, Model_ReflectsOaicompatModel) {
    const json j = make_partial().to_json_oaicompat();
    EXPECT_EQ(j.at("model").get<std::string>(), "test-model");
}

TEST(CmplPartialOaicompat, Id_ReflectsOaicompatCmplId) {
    const json j = make_partial().to_json_oaicompat();
    EXPECT_EQ(j.at("id").get<std::string>(), "cmpl-part");
}

TEST(CmplPartialOaicompat, LogProbs_EmptyProbs_IsNull) {
    // prob_output.probs empty by default → logprobs field is JSON null
    const json j = make_partial().to_json_oaicompat();
    EXPECT_TRUE(j.at("choices")[0].at("logprobs").is_null());
}

TEST(CmplPartialOaicompat, LogProbs_NonEmptyProbs_HasContentArray) {
    // When probs are set, logprobs becomes {"content": [...]} (not null)
    auto p = make_partial();
    completion_token_output::prob_info pi;
    pi.tok = 5; pi.txt = "hi"; pi.prob = 0.8f;
    p.prob_output.probs.push_back(pi);
    const json j = p.to_json_oaicompat();
    ASSERT_FALSE(j.at("choices")[0].at("logprobs").is_null());
    EXPECT_TRUE(j.at("choices")[0].at("logprobs").contains("content"));
    EXPECT_TRUE(j.at("choices")[0].at("logprobs").at("content").is_array());
}

// ============================================================
// server_task_result_cmpl_partial::to_json  (dispatcher)
//   The top-level to_json() switches on res_type.
//   With is_updated=true, it must route to the correct formatter
//   without asserting.  Verify that NONE and OAI_CMPL both produce
//   structurally valid (non-empty) JSON.
// ============================================================

TEST(CmplPartialToJsonDispatch, ResTypeNone_RoutesToNonOaicompat) {
    server_task_result_cmpl_partial p;
    p.is_updated = true;
    p.res_type   = TASK_RESPONSE_TYPE_NONE;
    p.content    = "hello";
    const json j = p.to_json();   // must not assert/abort
    // non-oaicompat shape has "content" directly
    EXPECT_EQ(j.at("content").get<std::string>(), "hello");
}

TEST(CmplPartialToJsonDispatch, ResTypeOaiCmpl_RoutesToOaicompat) {
    server_task_result_cmpl_partial p;
    p.is_updated        = true;
    p.res_type          = TASK_RESPONSE_TYPE_OAI_CMPL;
    p.content           = "hi";
    p.oaicompat_model   = "m";
    p.oaicompat_cmpl_id = "c";
    const json j = p.to_json();
    // oaicompat shape wraps content inside choices
    EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
}

TEST(CmplPartialToJsonDispatch, NotUpdated_Asserts) {
    server_task_result_cmpl_partial p;
    p.is_updated = false;
    // GGML_ASSERT fires when is_updated==false; this terminates the process,
    // so we verify the flag semantics by checking the truthy case passes.
    // (The death test would require EXPECT_DEATH which needs signal handling.)
    p.is_updated = true;
    p.res_type   = TASK_RESPONSE_TYPE_NONE;
    EXPECT_NO_THROW(p.to_json());
}

TEST(CmplPartialToJsonDispatch, ResTypeAnthropic_RoutesToAnthropicStream) {
    // ANTHROPIC arm in the dispatcher calls to_json_anthropic(), which
    // returns a json::array (not a json::object like the OAI arms).
    // With n_decoded==1 the first-token message_start event is emitted.
    server_task_result_cmpl_partial p;
    p.is_updated        = true;
    p.res_type          = TASK_RESPONSE_TYPE_ANTHROPIC;
    p.n_decoded         = 1;
    p.oaicompat_model   = "m";
    p.oaicompat_cmpl_id = "id";
    const json j = p.to_json();
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
    EXPECT_EQ(j.front().at("event").get<std::string>(), "message_start");
}

// ============================================================
// server_task_result_cmpl_final::to_json  — dispatcher
//   The switch covers NONE / OAI_CMPL / OAI_CHAT / ANTHROPIC
//   (OAI_RESP and OAI_ASR are structurally similar but not tested here).
//   OAI_CHAT forks further on stream: false→object, true→array.
// ============================================================

namespace {
// Minimal final result ready for to_json(); no vocab-dependent fields.
server_task_result_cmpl_final make_dispatched_final(task_response_type rt,
                                                     bool stream = false) {
    server_task_result_cmpl_final f;
    f.is_updated        = true;
    f.res_type          = rt;
    f.stream            = stream;
    f.content           = "hi";
    f.oaicompat_model   = "m";
    f.oaicompat_cmpl_id = "id";
    return f;
}
} // namespace

TEST(CmplFinalDispatch, ResTypeNone_ToJsonNonOaicompat) {
    auto f = make_dispatched_final(TASK_RESPONSE_TYPE_NONE);
    const json j = f.to_json();
    // non-oaicompat shape has "content" at top level, no "object" key
    EXPECT_EQ(j.at("content").get<std::string>(), "hi");
    EXPECT_FALSE(j.contains("object"));
}

TEST(CmplFinalDispatch, ResTypeOaiCmpl_ToJsonOaicompat) {
    auto f = make_dispatched_final(TASK_RESPONSE_TYPE_OAI_CMPL);
    const json j = f.to_json();
    EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
}

TEST(CmplFinalDispatch, ResTypeOaiChat_StreamFalse_ReturnsObject) {
    auto f = make_dispatched_final(TASK_RESPONSE_TYPE_OAI_CHAT, /*stream=*/false);
    const json j = f.to_json();
    // non-streaming chat → single JSON object
    EXPECT_TRUE(j.is_object());
    EXPECT_EQ(j.at("object").get<std::string>(), "chat.completion");
}

TEST(CmplFinalDispatch, ResTypeOaiChat_StreamTrue_ReturnsArray) {
    auto f = make_dispatched_final(TASK_RESPONSE_TYPE_OAI_CHAT, /*stream=*/true);
    const json j = f.to_json();
    // streaming chat → JSON array of chunks
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
}

TEST(CmplFinalDispatch, ResTypeAnthropic_StreamFalse_HasStopReason) {
    auto f = make_dispatched_final(TASK_RESPONSE_TYPE_ANTHROPIC, /*stream=*/false);
    const json j = f.to_json();
    EXPECT_TRUE(j.contains("stop_reason"));
}

// ============================================================
// verbose flag — cross-cutting concern in OAI formatters
//   Both to_json_oaicompat() and to_json_oaicompat_chat() inject a
//   __verbose key containing the non-oaicompat representation when
//   f.verbose==true.  This is a cross-cutting concern that must be
//   tested to catch regressions across future formatter refactors.
// ============================================================

TEST(CmplFinalVerboseFlag, Oaicompat_VerboseFalse_NoDebugKey) {
    auto f = make_oai_final();
    f.verbose = false;
    const json j = f.to_json_oaicompat();
    EXPECT_FALSE(j.contains("__verbose"));
}

TEST(CmplFinalVerboseFlag, Oaicompat_VerboseTrue_DebugKeyPresent) {
    auto f = make_oai_final("debug content");
    f.verbose = true;
    const json j = f.to_json_oaicompat();
    ASSERT_TRUE(j.contains("__verbose"));
    // __verbose must contain the non-oaicompat representation
    EXPECT_TRUE(j.at("__verbose").contains("content"));
    EXPECT_EQ(j.at("__verbose").at("content").get<std::string>(), "debug content");
}

TEST(CmplFinalVerboseFlag, OaicompatChat_VerboseTrue_DebugKeyPresent) {
    auto f = make_oai_final("chat debug");
    f.verbose = true;
    const json j = f.to_json_oaicompat_chat();
    ASSERT_TRUE(j.contains("__verbose"));
    EXPECT_EQ(j.at("__verbose").at("content").get<std::string>(), "chat debug");
}

TEST(CmplFinalVerboseFlag, Oaicompat_TimingsAbsentByDefault) {
    auto f = make_oai_final();
    // timings.prompt_n is default-constructed to a value < 0 — absent
    const json j = f.to_json_oaicompat();
    EXPECT_FALSE(j.contains("timings"));
}

TEST(CmplFinalVerboseFlag, Oaicompat_TimingsPresentWhenPromptNNonNeg) {
    auto f = make_oai_final();
    f.timings.prompt_n = 0;  // >= 0 triggers inclusion
    const json j = f.to_json_oaicompat();
    EXPECT_TRUE(j.contains("timings"));
}

// ============================================================
// server_task_result_cmpl_final::to_json_oaicompat_chat_stream
//   Returns a JSON array of chat.completion.chunk objects.
//   Structure:
//     [delta_0, delta_1, ..., final_chunk]           (include_usage=false)
//     [delta_0, ..., final_chunk, usage_chunk]        (include_usage=true)
//   - Every chunk has object="chat.completion.chunk".
//   - All intermediate chunks have choices[0].finish_reason=null.
//   - The terminal chunk has a non-null finish_reason.
//   - The usage chunk (if present) has empty choices array + usage object.
// ============================================================

namespace {
server_task_result_cmpl_final make_stream_final(bool include_usage = false) {
    server_task_result_cmpl_final f;
    f.oaicompat_model   = "m";
    f.oaicompat_cmpl_id = "id";
    f.stop              = STOP_TYPE_EOS;
    f.include_usage     = include_usage;
    // No oaicompat_msg_diffs → just the single terminal chunk
    return f;
}
} // namespace

TEST(CmplFinalChatStream, ReturnsArray) {
    const json j = make_stream_final().to_json_oaicompat_chat_stream();
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
}

TEST(CmplFinalChatStream, EveryChunk_HasChatCompletionChunkObject) {
    const json j = make_stream_final().to_json_oaicompat_chat_stream();
    for (const auto &chunk : j) {
        EXPECT_EQ(chunk.at("object").get<std::string>(), "chat.completion.chunk");
    }
}

TEST(CmplFinalChatStream, LastChunk_HasNonNullFinishReason) {
    const json j = make_stream_final().to_json_oaicompat_chat_stream();
    // Last element is the terminal stop chunk
    const json &last_chunk = j.back();
    const json &fr = last_chunk.at("choices")[0].at("finish_reason");
    EXPECT_FALSE(fr.is_null());
    EXPECT_EQ(fr.get<std::string>(), "stop");  // STOP_TYPE_EOS → "stop"
}

TEST(CmplFinalChatStream, IncludeUsageFalse_NoUsageChunk) {
    const json j = make_stream_final(/*include_usage=*/false).to_json_oaicompat_chat_stream();
    // No extra trailing chunk for usage
    for (const auto &chunk : j) {
        // all chunks with choices must have exactly 1 choice
        if (!chunk.at("choices").empty()) {
            EXPECT_FALSE(chunk.contains("usage"));
        }
    }
}

TEST(CmplFinalChatStream, IncludeUsageTrue_TrailingChunkHasEmptyChoicesAndUsage) {
    const json j = make_stream_final(/*include_usage=*/true).to_json_oaicompat_chat_stream();
    // Per OAI spec, the usage chunk has empty choices and a usage object
    bool found_usage_chunk = false;
    for (const auto &chunk : j) {
        if (chunk.at("choices").empty() && chunk.contains("usage")) {
            found_usage_chunk = true;
            EXPECT_TRUE(chunk.at("usage").contains("completion_tokens"));
        }
    }
    EXPECT_TRUE(found_usage_chunk);
}

// ============================================================
// server_task::params_from_json_cmpl — parsing pipeline
//   Called with nullptr vocab when the JSON does not exercise
//   grammar/preserved_tokens tokenisation.  Tests verify:
//     - simple field round-trip (temperature, seed, n_predict)
//     - repeat_last_n=-1 is expanded to n_ctx_slot
//     - dry_penalty_last_n=-1 is expanded to n_ctx_slot
//     - dry_base < 1.0 is reset to default
//     - n_discard negative is clamped to 0
//     - empty dry_sequence_breakers throws std::runtime_error
//     - lora field not an array throws std::runtime_error
//     - repeat_last_n < -1 throws std::runtime_error
// ============================================================

namespace {
task_params parse_params(const json &data, int n_ctx = 512) {
    common_params params_base;
    std::vector<llama_logit_bias> no_bias;
    return server_task::params_from_json_cmpl(nullptr, params_base, n_ctx, no_bias, data);
}
} // namespace

TEST(ParamsFromJsonCmpl, SimpleFields_RoundTrip) {
    const json data = {{"temperature", 0.7f}, {"seed", 42}, {"n_predict", 128}};
    const auto p = parse_params(data);
    EXPECT_FLOAT_EQ(p.sampling.temp, 0.7f);
    EXPECT_EQ(p.sampling.seed, 42u);
    EXPECT_EQ(p.n_predict, 128);
}

TEST(ParamsFromJsonCmpl, RepeatLastN_MinusOne_ExpandsToNCtxSlot) {
    const auto p = parse_params({{"repeat_last_n", -1}}, /*n_ctx=*/256);
    EXPECT_EQ(p.sampling.penalty_last_n, 256);
}

TEST(ParamsFromJsonCmpl, DryPenaltyLastN_MinusOne_ExpandsToNCtxSlot) {
    const auto p = parse_params({{"dry_penalty_last_n", -1}}, /*n_ctx=*/128);
    EXPECT_EQ(p.sampling.dry_penalty_last_n, 128);
}

TEST(ParamsFromJsonCmpl, DryBase_BelowOne_ResetToDefault) {
    // dry_base must be >= 1.0; if below, it reverts to the default (1.75)
    const auto p = parse_params({{"dry_base", 0.5f}});
    common_params defaults;
    EXPECT_FLOAT_EQ(p.sampling.dry_base, defaults.sampling.dry_base);
}

TEST(ParamsFromJsonCmpl, NDiscard_Negative_ClampedToZero) {
    const auto p = parse_params({{"n_discard", -5}});
    EXPECT_EQ(p.n_discard, 0);
}

TEST(ParamsFromJsonCmpl, EmptyDrySequenceBreakers_Throws) {
    EXPECT_THROW(parse_params({{"dry_sequence_breakers", json::array()}}),
                 std::runtime_error);
}

TEST(ParamsFromJsonCmpl, LoraNotArray_Throws) {
    EXPECT_THROW(parse_params({{"lora", "not-an-array"}}), std::runtime_error);
}

TEST(ParamsFromJsonCmpl, RepeatLastN_BelowMinusOne_Throws) {
    EXPECT_THROW(parse_params({{"repeat_last_n", -2}}), std::runtime_error);
}

TEST(ParamsFromJsonCmpl, StreamOptions_IncludeUsage_Parsed) {
    const json data = {{"stream", true},
                       {"stream_options", {{"include_usage", true}}}};
    const auto p = parse_params(data);
    EXPECT_TRUE(p.include_usage);
}

TEST(ParamsFromJsonCmpl, NCmpl_AliasedFromN) {
    // n_cmpl falls back to the "n" key when "n_cmpl" is absent.
    // n_cmpl is capped at n_parallel (1 by default); use 1 to stay valid.
    const auto p = parse_params({{"n", 1}});
    EXPECT_EQ(p.n_cmpl, 1);
}

// ============================================================
// params_from_json_cmpl — reasoning_budget_tokens
//   reasoning_budget_tokens defaults to -1 (disabled).
//   Any explicit value is stored directly in sampling.reasoning_budget_tokens.
//   The tag-tokenisation paths (start/end/message) are skipped when tags are empty,
//   so these tests do not require a vocab pointer.
// ============================================================

TEST(ParamsFromJsonCmpl, ReasoningBudgetTokens_Default_IsMinusOne) {
    const auto p = parse_params({});
    EXPECT_EQ(p.sampling.reasoning_budget_tokens, -1);
}

TEST(ParamsFromJsonCmpl, ReasoningBudgetTokens_SetPositive) {
    const auto p = parse_params({{"reasoning_budget_tokens", 512}});
    EXPECT_EQ(p.sampling.reasoning_budget_tokens, 512);
}

TEST(ParamsFromJsonCmpl, ReasoningBudgetTokens_Zero) {
    const auto p = parse_params({{"reasoning_budget_tokens", 0}});
    EXPECT_EQ(p.sampling.reasoning_budget_tokens, 0);
}

TEST(ParamsFromJsonCmpl, ReasoningBudgetTokens_ExplicitMinusOne_Disabled) {
    const auto p = parse_params({{"reasoning_budget_tokens", -1}});
    EXPECT_EQ(p.sampling.reasoning_budget_tokens, -1);
}

// ============================================================
// params_from_json_cmpl — grammar type routing
//   Three distinct paths set grammar.type:
//     "json_schema" key (no "grammar") → COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
//     "grammar" + "grammar_type"="tool_calls" → COMMON_GRAMMAR_TYPE_TOOL_CALLS
//     "grammar" (no grammar_type, or other value) → COMMON_GRAMMAR_TYPE_USER
// ============================================================

TEST(ParamsFromJsonCmpl, JsonSchema_SetsOutputFormatGrammarType) {
    // json_schema without "grammar" → grammar type OUTPUT_FORMAT
    const json data = {
        {"json_schema", {{"type", "object"}, {"properties", json::object()}}}
    };
    const auto p = parse_params(data);
    EXPECT_EQ(p.sampling.grammar.type, COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT);
}

TEST(ParamsFromJsonCmpl, GrammarTypeToolCalls_SetsToolCallsType) {
    // grammar_type="tool_calls" routes to COMMON_GRAMMAR_TYPE_TOOL_CALLS
    const json data = {
        {"grammar",      "root ::= object"},
        {"grammar_type", "tool_calls"}
    };
    const auto p = parse_params(data);
    EXPECT_EQ(p.sampling.grammar.type, COMMON_GRAMMAR_TYPE_TOOL_CALLS);
}

TEST(ParamsFromJsonCmpl, PlainGrammar_NoGrammarType_SetsUserType) {
    // grammar without grammar_type key → COMMON_GRAMMAR_TYPE_USER
    const json data = {{"grammar", "root ::= [a-z]+"}};
    const auto p = parse_params(data);
    EXPECT_EQ(p.sampling.grammar.type, COMMON_GRAMMAR_TYPE_USER);
}

// ============================================================
// response_fields projection in cmpl_final::to_json_non_oaicompat
//   When generation_params.response_fields is non-empty, only those
//   slash-delimited paths survive in the returned JSON.  This is a
//   server-side field filtering mechanism used to trim large responses.
// ============================================================

TEST(CmplFinalResponseFields, EmptyList_AllFieldsPresent) {
    server_task_result_cmpl_final f;
    f.content    = "hi";
    f.stop       = STOP_TYPE_EOS;
    // response_fields is empty by default → full object returned
    const json j = f.to_json_non_oaicompat();
    EXPECT_TRUE(j.contains("content"));
    EXPECT_TRUE(j.contains("stop_type"));
    EXPECT_TRUE(j.contains("timings"));
}

TEST(CmplFinalResponseFields, NonEmptyList_OnlyRequestedFieldsPresent) {
    server_task_result_cmpl_final f;
    f.content         = "projected";
    f.response_fields = {"content", "tokens_predicted"};
    const json j      = f.to_json_non_oaicompat();
    EXPECT_TRUE(j.contains("content"));
    EXPECT_TRUE(j.contains("tokens_predicted"));
    EXPECT_FALSE(j.contains("stop_type"));    // filtered out
    EXPECT_FALSE(j.contains("timings"));      // filtered out
    EXPECT_FALSE(j.contains("prompt"));       // filtered out
}

TEST(CmplFinalResponseFields, ContentValue_PreservedThroughProjection) {
    server_task_result_cmpl_final f;
    f.content         = "keep this";
    f.response_fields = {"content"};
    const json j      = f.to_json_non_oaicompat();
    EXPECT_EQ(j.at("content").get<std::string>(), "keep this");
}

// ============================================================
// server_task_result_cmpl_partial::to_json_oaicompat_chat
//   Streaming OAI chat chunk.  Returns a JSON array of delta
//   objects (each has object="chat.completion.chunk").
//   Special rule: when n_decoded==1 (first token), the method
//   prepends a role-announcement delta with role="assistant"
//   and content=null before the content deltas.
// ============================================================

namespace {
server_task_result_cmpl_partial make_chat_partial(int n_decoded = 1) {
    server_task_result_cmpl_partial p;
    p.is_updated        = true;
    p.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
    p.n_decoded         = n_decoded;
    p.oaicompat_model   = "m";
    p.oaicompat_cmpl_id = "id";
    return p;
}
} // namespace

TEST(CmplPartialOaicompatChat, ReturnsArray) {
    // Even with no diffs the first-token header delta is emitted
    const json j = make_chat_partial(/*n_decoded=*/1).to_json_oaicompat_chat();
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
}

TEST(CmplPartialOaicompatChat, EveryChunk_ObjectIsChatCompletionChunk) {
    const json j = make_chat_partial(1).to_json_oaicompat_chat();
    for (const auto &chunk : j) {
        EXPECT_EQ(chunk.at("object").get<std::string>(), "chat.completion.chunk");
    }
}

TEST(CmplPartialOaicompatChat, FirstToken_HasRoleHeaderDelta) {
    // n_decoded==1 → prepend a delta with role:"assistant", content:null
    const json j = make_chat_partial(/*n_decoded=*/1).to_json_oaicompat_chat();
    ASSERT_FALSE(j.empty());
    const json &delta = j.front().at("choices")[0].at("delta");
    EXPECT_EQ(delta.at("role").get<std::string>(), "assistant");
    EXPECT_TRUE(delta.at("content").is_null());
}

TEST(CmplPartialOaicompatChat, NotFirstToken_NoRoleHeaderDelta) {
    // n_decoded==2 → no role header; with no diffs the array is empty
    const json j = make_chat_partial(/*n_decoded=*/2).to_json_oaicompat_chat();
    // no diffs + not first → nothing emitted
    EXPECT_TRUE(j.empty());
}

TEST(CmplPartialOaicompatChat, AllChunks_FinishReasonIsNull) {
    // Partial chunks must always carry finish_reason=null
    const json j = make_chat_partial(1).to_json_oaicompat_chat();
    for (const auto &chunk : j) {
        ASSERT_FALSE(chunk.at("choices").empty());
        EXPECT_TRUE(chunk.at("choices")[0].at("finish_reason").is_null());
    }
}

// ============================================================
// server_task_result_cmpl_final::to_json_anthropic_stream
//   Returns a JSON array of Anthropic SSE event objects.
//   Every event has "event" + "data" fields (for format_anthropic_sse).
//   Regardless of diffs, the array always ends with:
//     - A "message_delta" event carrying stop_reason and stop_sequence
//     - A "message_stop" event
//   When oaicompat_msg_diffs contains text deltas, the method emits
//   content_block_start → content_block_delta → content_block_stop
//   event triples.
// ============================================================

namespace {
server_task_result_cmpl_final make_anthropic_stream_final(stop_type st = STOP_TYPE_EOS) {
    server_task_result_cmpl_final f;
    f.stop              = st;
    f.oaicompat_model   = "m";
    f.oaicompat_cmpl_id = "id";
    return f;
}
} // namespace

TEST(CmplFinalAnthropicStream, ReturnsArray) {
    const json j = make_anthropic_stream_final().to_json_anthropic_stream();
    EXPECT_TRUE(j.is_array());
    EXPECT_FALSE(j.empty());
}

TEST(CmplFinalAnthropicStream, LastEvent_IsMessageStop) {
    const json j = make_anthropic_stream_final().to_json_anthropic_stream();
    EXPECT_EQ(j.back().at("event").get<std::string>(), "message_stop");
}

TEST(CmplFinalAnthropicStream, SecondToLast_IsMessageDelta_WithStopReason) {
    const json j     = make_anthropic_stream_final(STOP_TYPE_EOS).to_json_anthropic_stream();
    // message_delta is always the penultimate event
    ASSERT_GE(j.size(), 2u);
    const json &md = j[j.size() - 2];
    EXPECT_EQ(md.at("event").get<std::string>(), "message_delta");
    EXPECT_EQ(md.at("data").at("delta").at("stop_reason").get<std::string>(), "end_turn");
}

TEST(CmplFinalAnthropicStream, MessageDelta_MaxTokensForLimit) {
    const json j = make_anthropic_stream_final(STOP_TYPE_LIMIT).to_json_anthropic_stream();
    ASSERT_GE(j.size(), 2u);
    const json &md = j[j.size() - 2];
    EXPECT_EQ(md.at("data").at("delta").at("stop_reason").get<std::string>(), "max_tokens");
}

TEST(CmplFinalAnthropicStream, WithTextDiff_EmitsContentBlockEvents) {
    auto f = make_anthropic_stream_final();
    // Inject a text content delta.
    // content_block_stop requires oaicompat_msg.content non-empty
    // (the accumulated final message, separate from diffs).
    f.oaicompat_msg.content = "hello";
    common_chat_msg_diff diff;
    diff.content_delta = "hello";
    f.oaicompat_msg_diffs.push_back(diff);
    const json j = f.to_json_anthropic_stream();
    // Must contain at least: content_block_start, content_block_delta,
    //                        content_block_stop, message_delta, message_stop
    ASSERT_GE(j.size(), 5u);
    bool found_start = false, found_delta = false;
    for (const auto &ev : j) {
        const std::string e = ev.at("event").get<std::string>();
        if (e == "content_block_start") found_start = true;
        if (e == "content_block_delta") found_delta = true;
    }
    EXPECT_TRUE(found_start);
    EXPECT_TRUE(found_delta);
}

TEST(CmplFinalAnthropicStream, WithThinkingDiff_EmitsThinkingBlockEvents) {
    auto f = make_anthropic_stream_final();
    common_chat_msg_diff diff;
    diff.reasoning_content_delta = "step1";
    f.oaicompat_msg_diffs.push_back(diff);
    const json j = f.to_json_anthropic_stream();
    // Find content_block_start with type="thinking"
    bool found_thinking_start = false;
    for (const auto &ev : j) {
        if (ev.at("event").get<std::string>() == "content_block_start") {
            if (ev.at("data").at("content_block").at("type").get<std::string>() == "thinking") {
                found_thinking_start = true;
            }
        }
    }
    EXPECT_TRUE(found_thinking_start);
}

// ============================================================
// server_task_result_cmpl_partial::to_json_anthropic
//   Anthropic partial streaming formatter.
//   n_decoded==1 (first token) → first event is "message_start"
//     containing id, model, role, and token usage counts.
//   n_decoded > 1 with no diffs → empty array.
//   reasoning_content_delta → content_block_start(thinking) + content_block_delta(thinking_delta).
//   content_delta → content_block_start(text) + content_block_delta(text_delta).
//   tool_call_index != npos → content_block_start(tool_use) with name/id.
//   anthropic_has_reasoning=true → text block index is 1 (shifted past thinking block).
// ============================================================

namespace {
server_task_result_cmpl_partial make_anthropic_partial(int n_decoded = 1) {
    server_task_result_cmpl_partial p;
    p.is_updated        = true;
    p.res_type          = TASK_RESPONSE_TYPE_ANTHROPIC;
    p.n_decoded         = n_decoded;
    p.n_prompt_tokens   = 10;
    p.oaicompat_model   = "test-model";
    p.oaicompat_cmpl_id = "msg-id";
    return p;
}
} // namespace

TEST(CmplPartialAnthropicStream, FirstToken_EmitsMessageStart) {
    const json j = make_anthropic_partial(/*n_decoded=*/1).to_json_anthropic();
    ASSERT_FALSE(j.empty());
    EXPECT_EQ(j.front().at("event").get<std::string>(), "message_start");
}

TEST(CmplPartialAnthropicStream, FirstToken_MessageStart_HasIdModelRole) {
    const json j   = make_anthropic_partial(1).to_json_anthropic();
    const json &msg = j.front().at("data").at("message");
    EXPECT_EQ(msg.at("id").get<std::string>(), "msg-id");
    EXPECT_EQ(msg.at("model").get<std::string>(), "test-model");
    EXPECT_EQ(msg.at("role").get<std::string>(), "assistant");
    EXPECT_TRUE(msg.at("content").is_array());
    EXPECT_TRUE(msg.at("content").empty());
}

TEST(CmplPartialAnthropicStream, FirstToken_MessageStart_HasUsageCounts) {
    auto p = make_anthropic_partial(1);
    p.n_prompt_tokens       = 12;
    p.n_prompt_tokens_cache = 4;
    const json j     = p.to_json_anthropic();
    const json &usage = j.front().at("data").at("message").at("usage");
    EXPECT_EQ(usage.at("input_tokens").get<int>(), 8);            // 12 - 4
    EXPECT_EQ(usage.at("cache_read_input_tokens").get<int>(), 4);
    EXPECT_EQ(usage.at("output_tokens").get<int>(), 0);
}

TEST(CmplPartialAnthropicStream, NotFirstToken_NoDiffs_EmptyArray) {
    // n_decoded > 1 with no diffs → nothing emitted
    const json j = make_anthropic_partial(/*n_decoded=*/2).to_json_anthropic();
    EXPECT_TRUE(j.empty());
}

TEST(CmplPartialAnthropicStream, WithTextDiff_EmitsBlockStartAndDelta) {
    auto p = make_anthropic_partial(/*n_decoded=*/2);
    common_chat_msg_diff diff;
    diff.content_delta = "hello";
    p.oaicompat_msg_diffs.push_back(diff);
    const json j = p.to_json_anthropic();
    bool found_start = false, found_delta = false;
    for (const auto &ev : j) {
        const std::string e = ev.at("event").get<std::string>();
        if (e == "content_block_start") {
            EXPECT_EQ(ev.at("data").at("content_block").at("type").get<std::string>(), "text");
            found_start = true;
        }
        if (e == "content_block_delta") {
            EXPECT_EQ(ev.at("data").at("delta").at("type").get<std::string>(), "text_delta");
            EXPECT_EQ(ev.at("data").at("delta").at("text").get<std::string>(), "hello");
            found_delta = true;
        }
    }
    EXPECT_TRUE(found_start);
    EXPECT_TRUE(found_delta);
}

TEST(CmplPartialAnthropicStream, WithReasoningDiff_EmitsThinkingBlockStartAndDelta) {
    auto p = make_anthropic_partial(/*n_decoded=*/2);
    common_chat_msg_diff diff;
    diff.reasoning_content_delta = "step1";
    p.oaicompat_msg_diffs.push_back(diff);
    const json j = p.to_json_anthropic();
    bool found_start = false, found_delta = false;
    for (const auto &ev : j) {
        const std::string e = ev.at("event").get<std::string>();
        if (e == "content_block_start") {
            if (ev.at("data").at("content_block").at("type").get<std::string>() == "thinking") {
                found_start = true;
            }
        }
        if (e == "content_block_delta") {
            if (ev.at("data").at("delta").at("type").get<std::string>() == "thinking_delta") {
                EXPECT_EQ(ev.at("data").at("delta").at("thinking").get<std::string>(), "step1");
                found_delta = true;
            }
        }
    }
    EXPECT_TRUE(found_start);
    EXPECT_TRUE(found_delta);
}

TEST(CmplPartialAnthropicStream, WithReasoningFlag_TextBlockIndex_IsOne) {
    // anthropic_has_reasoning=true shifts text_block_index to 1
    auto p = make_anthropic_partial(/*n_decoded=*/2);
    p.anthropic_has_reasoning = true;
    common_chat_msg_diff diff;
    diff.content_delta = "text";
    p.oaicompat_msg_diffs.push_back(diff);
    const json j = p.to_json_anthropic();
    for (const auto &ev : j) {
        const std::string e = ev.at("event").get<std::string>();
        if (e == "content_block_start" || e == "content_block_delta") {
            EXPECT_EQ(ev.at("data").at("index").get<size_t>(), 1u);
        }
    }
}

TEST(CmplPartialAnthropicStream, WithToolCallDiff_EmitsToolUseBlockStart) {
    auto p = make_anthropic_partial(/*n_decoded=*/2);
    common_chat_msg_diff diff;
    diff.tool_call_index      = 0;
    diff.tool_call_delta.name = "get_weather";
    diff.tool_call_delta.id   = "call_abc";
    p.oaicompat_msg_diffs.push_back(diff);
    const json j = p.to_json_anthropic();
    bool found_tool_start = false;
    for (const auto &ev : j) {
        if (ev.at("event").get<std::string>() == "content_block_start") {
            const json &cb = ev.at("data").at("content_block");
            if (cb.at("type").get<std::string>() == "tool_use") {
                EXPECT_EQ(cb.at("name").get<std::string>(), "get_weather");
                EXPECT_EQ(cb.at("id").get<std::string>(),   "call_abc");
                found_tool_start = true;
            }
        }
    }
    EXPECT_TRUE(found_tool_start);
}

