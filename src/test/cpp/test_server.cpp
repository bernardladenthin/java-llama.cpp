// Tests for server.hpp — focused on APIs changed in llama.cpp b4916 → b8576
//
// server.hpp includes utils.hpp transitively, so all utils types are available.
//
// Covered:
//   - result_timings::to_json()
//       draft_n / draft_n_accepted fields added (conditional on draft_n > 0)
//   - slot_params::to_json()
//       grammar field now uses common_grammar_value()
//       oaicompat_chat_syntax fields replace oaicompat_chat_format:
//         chat_format / reasoning_format / reasoning_in_content / generation_prompt
//   - completion_token_output  (logarithm edge-case, str_to_bytes, to_json, probs_vector_to_json)
//   - server_task_result_rerank::to_json  (score / index / tokens_evaluated)
//   - server_task_result_embd::to_json_*  (oaicompat vs non-oaicompat shapes)
//   - format_error_response  (all 7 error types → correct HTTP code + type string)
//   - server_task_type_need_embd / need_logits  (routing helpers)

#include <gtest/gtest.h>

// server.hpp includes utils.hpp; no JNI headers required.
#include "server.hpp"

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

    EXPECT_TRUE(j.contains("prompt_n"));
    EXPECT_TRUE(j.contains("prompt_ms"));
    EXPECT_TRUE(j.contains("prompt_per_token_ms"));
    EXPECT_TRUE(j.contains("prompt_per_second"));
    EXPECT_TRUE(j.contains("predicted_n"));
    EXPECT_TRUE(j.contains("predicted_ms"));
    EXPECT_TRUE(j.contains("predicted_per_token_ms"));
    EXPECT_TRUE(j.contains("predicted_per_second"));
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
    slot_params p;
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
    slot_params p;
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
    slot_params p;
    const json j = p.to_json();

    EXPECT_FALSE(j.contains("oaicompat_chat_format"))
        << "Legacy oaicompat_chat_format field must not appear in b8576";
}

TEST(SlotParamsToJson, GrammarValue_EmptyByDefault) {
    slot_params p;
    // sampling.grammar is default-constructed (empty)
    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "")
        << "Empty grammar must serialise to empty string via common_grammar_value()";
}

TEST(SlotParamsToJson, GrammarValue_UserGrammarExtracted) {
    slot_params p;
    // Mirrors the assignment in params_from_json_cmpl for user-provided grammar
    p.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z]+"};

    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "root ::= [a-z]+")
        << "User grammar string must be extracted by common_grammar_value()";
}

TEST(SlotParamsToJson, GrammarValue_OutputFormatGrammarExtracted) {
    slot_params p;
    // Mirrors the assignment in params_from_json_cmpl for JSON schema grammars
    p.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, "root ::= object"};

    const json j = p.to_json();

    EXPECT_EQ(j.at("grammar").get<std::string>(), "root ::= object");
}

TEST(SlotParamsToJson, GenerationPrompt_ReflectsSyntaxField) {
    slot_params p;
    p.oaicompat_chat_syntax.generation_prompt = "Think step by step:";

    const json j = p.to_json();

    EXPECT_EQ(j.at("generation_prompt").get<std::string>(), "Think step by step:");
}

TEST(SlotParamsToJson, ReasoningInContent_ReflectsSyntaxField) {
    slot_params p;
    p.oaicompat_chat_syntax.reasoning_in_content = true;

    const json j = p.to_json();

    EXPECT_TRUE(j.at("reasoning_in_content").get<bool>());
}

TEST(SlotParamsToJson, ReasoningInContent_FalseByDefault) {
    slot_params p;
    const json j = p.to_json();

    EXPECT_FALSE(j.at("reasoning_in_content").get<bool>());
}

TEST(SlotParamsToJson, SpeculativeFields_Present) {
    slot_params p;
    const json j = p.to_json();

    EXPECT_TRUE(j.contains("speculative.n_max"));
    EXPECT_TRUE(j.contains("speculative.n_min"));
    EXPECT_TRUE(j.contains("speculative.p_min"));
}

TEST(SlotParamsToJson, GrammarTriggers_IsArrayByDefault) {
    slot_params p;
    const json j = p.to_json();

    EXPECT_TRUE(j.at("grammar_triggers").is_array());
    EXPECT_TRUE(j.at("grammar_triggers").empty());
}

TEST(SlotParamsToJson, GrammarTriggers_SerialiseViaServerGrammarTrigger) {
    slot_params p;
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

TEST(CompletionTokenOutput, StrToBytes_AsciiChars) {
    auto bytes = completion_token_output::str_to_bytes("ABC");
    ASSERT_EQ(bytes.size(), 3u);
    EXPECT_EQ(bytes[0], static_cast<unsigned char>('A'));
    EXPECT_EQ(bytes[1], static_cast<unsigned char>('B'));
    EXPECT_EQ(bytes[2], static_cast<unsigned char>('C'));
}

TEST(CompletionTokenOutput, StrToBytes_EmptyString) {
    EXPECT_TRUE(completion_token_output::str_to_bytes("").empty());
}

TEST(CompletionTokenOutput, StrToBytes_HighBytes) {
    // Byte 0xFF must survive the conversion unchanged
    auto bytes = completion_token_output::str_to_bytes("\xFF");
    ASSERT_EQ(bytes.size(), 1u);
    EXPECT_EQ(bytes[0], static_cast<unsigned char>(0xFF));
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
    e.oaicompat = OAICOMPAT_TYPE_NONE;

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
    e.oaicompat = OAICOMPAT_TYPE_EMBEDDING;

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
    EXPECT_TRUE(server_task_type_need_embd(SERVER_TASK_TYPE_EMBEDDING));
    EXPECT_TRUE(server_task_type_need_embd(SERVER_TASK_TYPE_RERANK));
}

TEST(ServerTaskTypeHelpers, NeedEmbd_FalseForOtherTypes) {
    EXPECT_FALSE(server_task_type_need_embd(SERVER_TASK_TYPE_COMPLETION));
    EXPECT_FALSE(server_task_type_need_embd(SERVER_TASK_TYPE_INFILL));
    EXPECT_FALSE(server_task_type_need_embd(SERVER_TASK_TYPE_METRICS));
    EXPECT_FALSE(server_task_type_need_embd(SERVER_TASK_TYPE_CANCEL));
}

TEST(ServerTaskTypeHelpers, NeedLogits_TrueForCompletionAndInfill) {
    EXPECT_TRUE(server_task_type_need_logits(SERVER_TASK_TYPE_COMPLETION));
    EXPECT_TRUE(server_task_type_need_logits(SERVER_TASK_TYPE_INFILL));
}

TEST(ServerTaskTypeHelpers, NeedLogits_FalseForOtherTypes) {
    EXPECT_FALSE(server_task_type_need_logits(SERVER_TASK_TYPE_EMBEDDING));
    EXPECT_FALSE(server_task_type_need_logits(SERVER_TASK_TYPE_RERANK));
    EXPECT_FALSE(server_task_type_need_logits(SERVER_TASK_TYPE_METRICS));
}
