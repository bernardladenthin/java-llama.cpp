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
