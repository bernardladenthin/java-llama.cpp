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

    EXPECT_TRUE(j.contains("speculative.n_max"));
    EXPECT_TRUE(j.contains("speculative.n_min"));
    EXPECT_TRUE(j.contains("speculative.p_min"));
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

