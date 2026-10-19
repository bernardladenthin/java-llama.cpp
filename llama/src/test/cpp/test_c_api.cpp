// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The C API (src/main/cpp/jllama_api.h) driven without a JVM, end to end: a model is loaded, a chat
// completion answered, a streamed one received chunk by chunk, the log sink fed, the error contract
// kept -- over a tiny random llama-architecture GGUF this file writes itself with the public gguf
// API, so the test is hermetic (no model download, no network) and runs in every C++ Tests job.
// The output of a random model is noise; what is asserted is the contract: the shapes llama.cpp's
// server produces, the status codes, the envelope of the stream, the lifecycle of a task.
//
// This is the same code the Java_* entry points run (they call these functions), so a behaviour
// pinned here is the behaviour LlamaModel has.

#include "jllama_api.h"
#include "tiny_llama_gguf.hpp"

#include "build-info.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;

using namespace jllama_test;

struct Owned {
    char *s = nullptr;
    ~Owned() { jllama_string_free(s); }
    std::string str() const { return s != nullptr ? std::string(s) : std::string(); }
};

class CApi : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf();
        ASSERT_FALSE(path_.empty());
        const std::vector<std::string> args = {"jllama", "-m", path_,         "-c", "64",
                                               "-ngl",   "0",  "--no-warmup", "-t", "2"};
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        const jllama_status status =
            jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model_);
        ASSERT_EQ(status, JLLAMA_STATUS_OK) << jllama_last_error();
        ASSERT_NE(model_, nullptr);
    }
    static void TearDownTestSuite() {
        jllama_model_free(model_);
        model_ = nullptr;
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    static std::string path_;
    static jllama_model *model_;
};
std::string CApi::path_;
jllama_model *CApi::model_ = nullptr;

const char *CHAT_REQUEST = R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0})";

} // namespace

TEST(CApiStatic, VersionAndBuildInfoAreTheBuildsOwn) {
    const std::string version = jllama_api_version();
    EXPECT_TRUE(std::regex_match(version, std::regex("[0-9]+\\.[0-9]+\\.[0-9]+(-SNAPSHOT)?"))) << version;
    const std::string build = jllama_build_info();
    EXPECT_EQ(build.rfind("b", 0), 0u) << build;
    EXPECT_STREQ(jllama_build_info(), llama_build_info());
    EXPECT_STREQ(jllama_last_error(), "");
}

TEST(CApiStatic, NullArgumentsAreRefusedWithAMessage) {
    EXPECT_EQ(jllama_model_load(0, nullptr, nullptr, nullptr, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
    jllama_model *out = reinterpret_cast<jllama_model *>(1);
    const char *argv[] = {"jllama"};
    EXPECT_EQ(jllama_model_load(0, argv, nullptr, nullptr, &out), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(out, nullptr);
    char *response = nullptr;
    EXPECT_EQ(jllama_chat_completions(nullptr, CHAT_REQUEST, &response), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(response, nullptr);
    jllama_model_free(nullptr);  // ignored
    jllama_string_free(nullptr); // ignored
}

TEST(CApiStatic, AModelThatDoesNotExistFailsToLoad) {
    jllama_model *model = nullptr;
    const char *argv[] = {"jllama", "-m", "/nonexistent/model.gguf", "--no-warmup"};
    EXPECT_EQ(jllama_model_load(4, argv, nullptr, nullptr, &model), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(model, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
}

TEST_F(CApi, LoadProgressIsReportedAndAFalseReturnAbortsTheLoad) {
    struct Seen {
        std::atomic<int> calls{0};
        float last = -1.0f;
    } seen;
    jllama_model *second = nullptr;
    const std::vector<std::string> args = {"jllama", "-m", path_, "-c", "64", "-ngl", "0", "--no-warmup"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    auto progress = [](float p, void *ud) -> bool {
        auto *s = static_cast<Seen *>(ud);
        s->calls++;
        s->last = p;
        return true;
    };
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), progress, &seen, &second), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_GT(seen.calls.load(), 0);
    EXPECT_FLOAT_EQ(seen.last, 1.0f);
    jllama_model_free(second);

    auto abort_at_once = [](float, void *) -> bool { return false; };
    jllama_model *third = nullptr;
    EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), abort_at_once, nullptr, &third),
              JLLAMA_STATUS_ERROR);
    EXPECT_EQ(third, nullptr);
}

TEST_F(CApi, ChatCompletionReturnsAnOpenAiChatCompletionObject) {
    Owned response;
    ASSERT_EQ(jllama_chat_completions(model_, CHAT_REQUEST, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(response.s, nullptr);
    const json j = json::parse(response.str());
    EXPECT_EQ(j.at("object").get<std::string>(), "chat.completion");
    ASSERT_TRUE(j.at("choices").is_array());
    ASSERT_EQ(j.at("choices").size(), 1u);
    EXPECT_TRUE(j.at("choices")[0].at("message").contains("content"));
    EXPECT_TRUE(j.at("usage").at("completion_tokens").is_number());
    EXPECT_LE(j.at("usage").at("completion_tokens").get<int>(), 4);
    EXPECT_STREQ(jllama_last_error(), "");
}

TEST_F(CApi, AnInvalidRequestIsTheCallersFault) {
    Owned response;
    EXPECT_EQ(jllama_chat_completions(model_, "{not json", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(response.s, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
    // A well-formed body the server refuses: messages must be an array.
    EXPECT_EQ(jllama_chat_completions(model_, R"({"messages":"no"})", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(response.s, nullptr);
}

TEST_F(CApi, StreamingDeliversChunksUntilStopAndReleasesTheTask) {
    const char *request =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0,"stream":true})";
    int task = 0;
    ASSERT_EQ(jllama_chat_completions_stream_request(model_, request, &task), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GE(task, 0); // the server numbers its tasks from 0
    int chunks = 0;
    bool stop = false;
    for (int i = 0; i < 20 && !stop; ++i) {
        Owned envelope;
        ASSERT_EQ(jllama_chat_completions_stream_receive(model_, task, &envelope.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(envelope.str());
        EXPECT_EQ(j.at("stop").get<bool>(), stop);
        const json &data = j.at("data");
        if (data.is_array()) {
            for (const json &chunk : data) {
                EXPECT_EQ(chunk.at("object").get<std::string>(), "chat.completion.chunk");
            }
        } else {
            EXPECT_EQ(data.at("object").get<std::string>(), "chat.completion.chunk");
        }
        ++chunks;
    }
    EXPECT_TRUE(stop);
    EXPECT_GE(chunks, 2); // at least one delta and the final chunk array
    // The task was released on stop: receiving again is "Task not found", as is an unknown id.
    Owned again;
    bool s = false;
    EXPECT_EQ(jllama_chat_completions_stream_receive(model_, task, &again.s, &s), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), "Task not found");
    EXPECT_EQ(jllama_task_release(model_, task), JLLAMA_STATUS_OK); // unknown ids are ignored
}

TEST_F(CApi, AReleasedStreamingTaskIsGone) {
    const char *request =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0,"stream":true})";
    int task = 0;
    ASSERT_EQ(jllama_chat_completions_stream_request(model_, request, &task), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_task_release(model_, task), JLLAMA_STATUS_OK);
    Owned envelope;
    bool stop = false;
    EXPECT_EQ(jllama_chat_completions_stream_receive(model_, task, &envelope.s, &stop), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), "Task not found");
}

// ---------------------------------------------------------------------------
// The rest of LlamaModel through the C API (B.3). The model is random, so what is asserted is the
// shape and the contract of every answer, never its content.
// ---------------------------------------------------------------------------

TEST_F(CApi, ModelMetaDescribesTheModel) {
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json j = json::parse(meta.str());
    EXPECT_EQ(j.at("n_vocab").get<int>(), N_VOCAB);
    EXPECT_EQ(j.at("n_ctx_train").get<int>(), N_CTX_TRAIN);
    EXPECT_EQ(j.at("n_embd").get<int>(), N_EMBD);
    EXPECT_EQ(j.at("architecture").get<std::string>(), "llama");
    EXPECT_EQ(j.at("special_tokens").at("bos").get<int>(), '<');
    EXPECT_EQ(j.at("special_tokens").at("eos").get<int>(), '>');
    EXPECT_TRUE(j.at("modalities").at("vision").is_boolean());
    EXPECT_TRUE(j.at("input_modalities").is_array());
    EXPECT_EQ(j.at("metadata").at("general.architecture").get<std::string>(), "llama");
    EXPECT_EQ(jllama_model_meta(model_, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, RawCompletionsReturnTheResultObject) {
    Owned response;
    ASSERT_EQ(jllama_completions(model_, R"({"prompt":"Hi","n_predict":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json j = json::parse(response.str());
    EXPECT_TRUE(j.at("content").is_string());
    EXPECT_LE(j.at("tokens_predicted").get<int>(), 2);
    EXPECT_TRUE(j.at("stop").get<bool>());
    EXPECT_EQ(jllama_completions(model_, "{not json", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
}

TEST_F(CApi, OaiCompletionsReturnATextCompletionObject) {
    Owned response;
    ASSERT_EQ(jllama_completions_oai(model_, R"({"prompt":"Hi","max_tokens":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json j = json::parse(response.str());
    EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
    ASSERT_EQ(j.at("choices").size(), 1u);
    EXPECT_TRUE(j.at("choices")[0].at("text").is_string());
    // The OAI layer refuses a body without a prompt before anything reaches the model.
    EXPECT_EQ(jllama_completions_oai(model_, R"({"max_tokens":2})", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
}

// Upstream gives every completion an id (gen_chatcmplid) and names the model it ran on
// (model_name: the alias, else the file name); the engine set neither, so every OpenAI answer and
// chunk carried "id":"" and "model":"" (F185).
TEST_F(CApi, OpenAiAnswersCarryAnIdAndTheModelName) {
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string name = json::parse(meta.str()).at("name").get<std::string>();
    ASSERT_FALSE(name.empty());

    Owned completion;
    ASSERT_EQ(jllama_completions_oai(model_, R"({"prompt":"Hi","max_tokens":2})", &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json c = json::parse(completion.str());
    EXPECT_EQ(c.at("id").get<std::string>().rfind("chatcmpl-", 0), 0u) << c.dump();
    EXPECT_EQ(c.at("model").get<std::string>(), name);

    Owned chat;
    ASSERT_EQ(
        jllama_chat_completions(model_, R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":2})", &chat.s),
        JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json m = json::parse(chat.str());
    EXPECT_EQ(m.at("id").get<std::string>().rfind("chatcmpl-", 0), 0u) << m.dump();
    EXPECT_EQ(m.at("model").get<std::string>(), name);
    EXPECT_NE(m.at("id"), c.at("id"));

    // every chunk of one stream carries the same id
    int task = 0;
    ASSERT_EQ(jllama_chat_completions_stream_request(
                  model_, R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"stream":true})", &task),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::set<std::string> ids;
    bool stop = false;
    for (int i = 0; i < 20 && !stop; ++i) {
        Owned envelope;
        ASSERT_EQ(jllama_chat_completions_stream_receive(model_, task, &envelope.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json data = json::parse(envelope.str()).at("data");
        for (const json &chunk : data.is_array() ? data : json::array({data})) {
            ids.insert(chunk.at("id").get<std::string>());
            EXPECT_EQ(chunk.at("model").get<std::string>(), name);
        }
    }
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids.begin()->rfind("chatcmpl-", 0), 0u);
}

// Upstream answers several prompts in one request with one choice each; a task here carries one
// prompt, and every prompt after the first was dropped without a word -- two prompts, one answer
// (F184). They are refused instead; a token array is one prompt and stays accepted.
TEST_F(CApi, SeveralPromptsInOneRequestAreRefusedNotDropped) {
    Owned response;
    EXPECT_EQ(jllama_completions_oai(model_, R"({"prompt":["Hi","Ho"],"max_tokens":2})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("2 prompts"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(jllama_completions(model_, R"({"prompt":["Hi","Ho"],"n_predict":2})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    int task = -1;
    EXPECT_EQ(jllama_completions_stream_request(model_, R"({"prompt":["Hi","Ho"],"n_predict":2,"stream":true})", &task),
              JLLAMA_STATUS_INVALID_REQUEST);

    ASSERT_EQ(jllama_completions_oai(model_, R"({"prompt":[72,105],"max_tokens":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(response.str()).at("choices").size(), 1u);
    ASSERT_EQ(jllama_completions_oai(model_, R"({"prompt":["Hi"],"max_tokens":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
}

// The streamed /v1/completions: the body parsed as the blocking one, upstream's text_completion
// chunks, one id, the last with usage (TODO (14), F186).
TEST_F(CApi, OaiCompletionsStreamDeliversTextCompletionChunksEndingWithUsage) {
    int task = -1;
    ASSERT_EQ(
        jllama_completions_oai_stream_request(model_, R"({"prompt":[72,105],"max_tokens":3,"logprobs":1})", &task),
        JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::vector<json> chunks;
    bool stop = false;
    for (int i = 0; i < 20 && !stop; ++i) {
        Owned envelope;
        ASSERT_EQ(jllama_chat_completions_stream_receive(model_, task, &envelope.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json data = json::parse(envelope.str()).at("data");
        for (const json &chunk : data.is_array() ? data : json::array({data})) {
            chunks.push_back(chunk);
        }
    }
    EXPECT_TRUE(stop);
    ASSERT_GE(chunks.size(), 2u);
    std::set<std::string> ids;
    for (const json &chunk : chunks) {
        EXPECT_EQ(chunk.at("object").get<std::string>(), "text_completion") << chunk.dump();
        ids.insert(chunk.at("id").get<std::string>());
    }
    EXPECT_EQ(ids.size(), 1u);
    EXPECT_TRUE(chunks.front().at("choices")[0].at("logprobs").at("content").is_array()) << chunks.front().dump();
    const json &last = chunks.back();
    EXPECT_EQ(last.at("usage").at("prompt_tokens").get<int>(), 2) << last.dump();
    EXPECT_TRUE(last.at("choices")[0].at("finish_reason").is_string());

    // refused before anything runs, as the blocking request is
    EXPECT_EQ(jllama_completions_oai_stream_request(model_, R"({"max_tokens":2})", &task),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(jllama_completions_oai_stream_request(model_, R"({"prompt":["Hi","Ho"]})", &task),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(jllama_completions_oai_stream_request(model_, nullptr, &task), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, RawStreamingDeliversResultsWithAStopKeyAndReleasesTheTask) {
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(
                  model_, R"({"prompt":"Hi","n_predict":3,"temperature":0,"stream":true})", &task),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_GE(task, 0);
    bool stop = false;
    int results = 0;
    for (int i = 0; i < 20 && !stop; ++i) {
        Owned payload;
        ASSERT_EQ(jllama_completions_stream_receive(model_, task, &payload.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(payload.str());
        EXPECT_EQ(j.at("stop").get<bool>(), stop);
        EXPECT_TRUE(j.at("content").is_string());
        ++results;
    }
    EXPECT_TRUE(stop);
    EXPECT_GE(results, 2); // the partial results and the final one
    Owned again;
    bool s = false;
    EXPECT_EQ(jllama_completions_stream_receive(model_, task, &again.s, &s), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), "Task not found");
}

TEST_F(CApi, ChatCompletionsRequestStreamsRawResultsOverTheChatTemplate) {
    int task = -1;
    ASSERT_EQ(jllama_chat_completions_request(
                  model_,
                  R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":2,"temperature":0,"stream":true})",
                  &task),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    bool stop = false;
    for (int i = 0; i < 20 && !stop; ++i) {
        Owned payload;
        ASSERT_EQ(jllama_completions_stream_receive(model_, task, &payload.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(payload.str());
        EXPECT_FALSE(j.contains("object")); // raw results, not chat.completion.chunk objects
    }
    EXPECT_TRUE(stop);
    EXPECT_EQ(jllama_chat_completions_request(model_, R"({"messages":"no"})", &task), JLLAMA_STATUS_INVALID_REQUEST);
}

TEST_F(CApi, ApplyTemplateRendersThePrompt) {
    Owned prompt;
    ASSERT_EQ(jllama_apply_template(model_, R"({"messages":[{"role":"user","content":"Hi there"}]})", &prompt.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(prompt.str().find("Hi there"), std::string::npos);
    EXPECT_EQ(jllama_apply_template(model_, R"({"messages":"no"})", &prompt.s), JLLAMA_STATUS_INVALID_REQUEST);
}

TEST_F(CApi, TokenizeAndDetokenizeRoundTrip) {
    int *tokens = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tokenize(model_, "Hello", false, &tokens, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(tokens, nullptr);
    EXPECT_EQ(count, 5u); // one byte token per byte
    Owned text;
    size_t length = 0;
    ASSERT_EQ(jllama_detokenize(model_, tokens, count, &text.s, &length), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(std::string(text.s, length), "Hello");
    jllama_tokens_free(tokens);

    Owned with_pieces;
    ASSERT_EQ(jllama_tokenize_json(model_, "Hi", false, true, &with_pieces.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json tj = json::parse(with_pieces.str());
    ASSERT_EQ(tj.at("tokens").size(), 2u);
    EXPECT_EQ(tj.at("tokens")[0].at("id").get<int>(), 'H');
    EXPECT_EQ(tj.at("tokens")[0].at("piece").get<std::string>(), "H");
    const int ids[] = {'H', 'i'};
    Owned detok;
    ASSERT_EQ(jllama_detokenize_json(model_, ids, 2, &detok.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(detok.str()).at("content").get<std::string>(), "Hi");
    EXPECT_EQ(jllama_tokenize(model_, nullptr, false, &tokens, &count), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, MetricsCarryTheCountersAndTheSlots) {
    Owned metrics;
    ASSERT_EQ(jllama_metrics(model_, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json j = json::parse(metrics.str());
    EXPECT_TRUE(j.at("idle").is_number());
    EXPECT_TRUE(j.at("n_prompt_tokens_processed_total").is_number());
    ASSERT_TRUE(j.at("slots").is_array());
    EXPECT_EQ(j.at("slots").size(), 1u);
}

TEST_F(CApi, SlotActionsAndLoraAdapters) {
    Owned response;
    ASSERT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_ERASE, 0, nullptr, &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_TRUE(json::parse(response.str()).is_object());
    Owned refused;
    EXPECT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_SAVE, 0, "", &refused.s), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), "Filename is required for slot save");
    EXPECT_EQ(jllama_slot_action(model_, static_cast<jllama_slot_op>(7), 0, nullptr, &refused.s),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "Invalid slot action");

    Owned adapters;
    ASSERT_EQ(jllama_lora_adapters(model_, &adapters.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(json::parse(adapters.str()).is_array());
    Owned set;
    ASSERT_EQ(jllama_set_lora_adapters(model_, "[]", &set.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NO_THROW(json::parse(set.str())); // upstream's own result object for the set task
    EXPECT_EQ(jllama_set_lora_adapters(model_, "{}", &set.s), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(jllama_set_lora_adapters(model_, "[", &set.s), JLLAMA_STATUS_INVALID_REQUEST);
}

TEST_F(CApi, ConfigureTunesTheThreadsAndRefusesNonsense) {
    EXPECT_EQ(jllama_configure(model_, R"({"n_threads":1,"n_threads_batch":1})"), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({"slot_prompt_similarity":0.5})"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({"slot_prompt_similarity":2.0})"), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(jllama_configure(model_, "{not json"), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(jllama_configure(model_, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, OperationsTheModelWasNotLoadedForAreServerErrors) {
    // Loaded without --embedding / --reranking, and a model without FIM tokens or decisions.
    float *values = nullptr;
    size_t count = 0;
    EXPECT_EQ(jllama_embed(model_, "Hi", &values, &count), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(values, nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("embedding support"), std::string::npos);
    Owned response;
    EXPECT_EQ(jllama_embeddings(model_, R"({"content":"Hi"})", false, &response.s), JLLAMA_STATUS_ERROR);
    const char *docs[] = {"a", "b"};
    EXPECT_EQ(jllama_rerank(model_, "q", 2, docs, &response.s), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("reranking"), std::string::npos);
    EXPECT_EQ(jllama_infill(model_, R"({"input_prefix":"a","input_suffix":"b"})", &response.s), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("fill-in-the-middle"), std::string::npos);
    EXPECT_NE(jllama_systemone(model_, R"({"state":"x","questions":[]})", &response.s), JLLAMA_STATUS_OK);
    EXPECT_EQ(response.s, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
}

TEST(CApiStatic, JsonSchemaToGrammarNeedsNoModel) {
    Owned grammar;
    ASSERT_EQ(jllama_json_schema_to_grammar(R"({"type":"object","properties":{"a":{"type":"string"}}})", &grammar.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(grammar.str().find("root"), std::string::npos);
    EXPECT_EQ(jllama_json_schema_to_grammar("{bad", &grammar.s), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(grammar.s, nullptr);
    EXPECT_EQ(jllama_json_schema_to_grammar(nullptr, &grammar.s), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, QuantizeWritesAModelThatLoads) {
    const std::string out = path_ + ".q8_0.gguf";
    ASSERT_EQ(jllama_quantize(path_.c_str(), out.c_str(), /*LLAMA_FTYPE_MOSTLY_Q8_0*/ 7, 1, false), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::error_code ec;
    EXPECT_GT(std::filesystem::file_size(out, ec), 0u);
    const std::vector<std::string> args = {"jllama", "-m", out, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *quantized = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &quantized),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_model_free(quantized);
    std::filesystem::remove(out, ec);
    EXPECT_EQ(jllama_quantize("/nonexistent/in.gguf", out.c_str(), 7, 1, false), JLLAMA_STATUS_ERROR);
}

// ---------------------------------------------------------------------------
// The other engines through the C API: the embedded server, the RPC server, text to speech and
// the trainer.
// ---------------------------------------------------------------------------

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiServer, TheEmbeddedServerStartsInRouterModeAndStops) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-models-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string models_dir = dir.string();
    const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", "0", "--models-dir", models_dir.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(7, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(server, nullptr);
    EXPECT_TRUE(jllama_server_is_running(server));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_FALSE(jllama_server_is_running(nullptr));
    EXPECT_EQ(jllama_server_stop(nullptr), JLLAMA_STATUS_OK);
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);

    EXPECT_EQ(jllama_server_start(0, argv, &server), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(jllama_server_start(7, argv, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(jllama_server_set_worker_command("java -cp app.jar net.ladenthin.llama.server.NativeServer"),
              JLLAMA_STATUS_OK);
    const char *set = std::getenv("LLAMA_SERVER_WORKER_CMD");
    ASSERT_NE(set, nullptr);
    EXPECT_STREQ(set, "java -cp app.jar net.ladenthin.llama.server.NativeServer");
    EXPECT_EQ(jllama_server_set_worker_command(""), JLLAMA_STATUS_OK);
    const char *cleared = std::getenv("LLAMA_SERVER_WORKER_CMD");
    EXPECT_TRUE(cleared == nullptr || cleared[0] == '\0');
    EXPECT_EQ(jllama_server_set_worker_command(nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
}

TEST_F(CApi, TheAttachedServerServesTheModelAndLeavesItUsable) {
    const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", "0"};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start_attached(model_, 5, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(server, nullptr);
    EXPECT_TRUE(jllama_server_is_running(server));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // The model answers while the frontend is attached to it.
    Owned during;
    EXPECT_EQ(jllama_chat_completions(model_, CHAT_REQUEST, &during.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    // And after the server is gone: stopping the frontend never stops the model's own loop.
    Owned after;
    EXPECT_EQ(jllama_chat_completions(model_, CHAT_REQUEST, &after.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_server_start_attached(nullptr, 5, argv, &server), JLLAMA_STATUS_INVALID_ARGUMENT);
}
#endif // JLLAMA_HAS_NATIVE_SERVER

TEST(CApiRpc, TheRpcServerServesTheCpuOverLoopbackAndStops) {
    char **names = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_rpc_server_devices("CPU", &names, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(count, 1u);
    EXPECT_STREQ(names[0], "CPU");
    jllama_strings_free(names, count);
    EXPECT_EQ(jllama_rpc_server_devices("no-such-device", &names, &count), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(names, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
    EXPECT_EQ(jllama_rpc_server_devices(nullptr, &names, &count), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_FALSE(jllama_rpc_server_listening());

    // The first free port of a small range (a fixed port would collide with parallel jobs); the
    // server returns at once when it cannot bind, which `listening` tells apart.
    bool served = false;
    for (int attempt = 0; attempt < 50 && !served; ++attempt) {
        const int port =
            42000 + static_cast<int>(
                        (std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + attempt * 97) % 20000);
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto status = std::make_shared<jllama_status>(JLLAMA_STATUS_OK);
        std::thread worker([port, done, status]() {
            *status = jllama_rpc_server_serve("127.0.0.1", port, 2, nullptr, "CPU");
            done->store(true);
        });
        for (int i = 0; i < 500 && !jllama_rpc_server_listening() && !done->load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (jllama_rpc_server_listening()) {
            served = true;
            EXPECT_EQ(jllama_rpc_server_stop(), JLLAMA_STATUS_OK);
            worker.join();
            EXPECT_EQ(*status, JLLAMA_STATUS_OK) << jllama_last_error();
        } else {
            worker.join();
        }
    }
    EXPECT_TRUE(served);
    EXPECT_FALSE(jllama_rpc_server_listening());
    EXPECT_EQ(jllama_rpc_server_serve("127.0.0.1", 1, 2, nullptr, "no-such-device"), JLLAMA_STATUS_ERROR);
}

TEST(CApiTts, TextToSpeechRefusesMissingModelsAndNullHandles) {
    jllama_tts_model *tts = nullptr;
    EXPECT_EQ(jllama_tts_load("/nonexistent/backbone.gguf", "/nonexistent/mmproj.gguf", 0, 1, &tts),
              JLLAMA_STATUS_ERROR);
    EXPECT_EQ(tts, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
    EXPECT_EQ(jllama_tts_load(nullptr, "x", 0, 1, &tts), JLLAMA_STATUS_INVALID_ARGUMENT);
    unsigned char *wav = nullptr;
    size_t length = 0;
    EXPECT_EQ(jllama_tts_synthesize(nullptr, "Hi", nullptr, nullptr, 10, 1, 1, &wav, &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    jllama_tts_free(nullptr);
    jllama_bytes_free(nullptr);
}

TEST_F(CApi, FinetuneTrainsTheRandomModelAndRefusesABadConfig) {
    EXPECT_EQ(jllama_finetune("{not json"), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("invalid training config"), std::string::npos);
    EXPECT_EQ(jllama_finetune(nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(jllama_finetune(R"({"model_path":"/nonexistent.gguf","training_text":"abc","output_path":"/dev/null"})"),
              JLLAMA_STATUS_ERROR);

    const std::string out = path_ + ".finetuned.gguf";
    std::string corpus;
    for (int i = 0; i < 16; ++i) {
        corpus += "the quick brown fox jumps over the lazy dog. ";
    }
    const json config = {{"model_path", path_}, {"training_text", corpus}, {"output_path", out},
                         {"epochs", 1},         {"learning_rate", 1e-4},   {"n_ctx", 64},
                         {"n_gpu_layers", 0},   {"val_split", 0.0}};
    ASSERT_EQ(jllama_finetune(config.dump().c_str()), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    EXPECT_GT(std::filesystem::file_size(out, ec), 0u);
    std::filesystem::remove(out, ec);
}

TEST_F(CApi, TheLogCallbackSeesTheServersLinesAndNullFlushesSynchronously) {
    struct Sink {
        std::mutex m;
        std::vector<std::string> lines;
        std::vector<int> levels;
    } sink;
    auto cb = [](int level, const char *text, void *ud) {
        auto *s = static_cast<Sink *>(ud);
        std::lock_guard<std::mutex> lk(s->m);
        s->lines.emplace_back(text);
        s->levels.push_back(level);
    };
    ASSERT_EQ(jllama_set_log_callback(cb, &sink), JLLAMA_STATUS_OK);
    Owned response;
    ASSERT_EQ(jllama_chat_completions(model_, CHAT_REQUEST, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    // Null drains the queue through the callback before returning: everything the request logged
    // is in `sink` now, and `sink` may be destroyed afterwards.
    ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK);
    std::lock_guard<std::mutex> lk(sink.m);
    EXPECT_FALSE(sink.lines.empty());
    bool server_line = false;
    for (const std::string &line : sink.lines) {
        if (line.find("slot") != std::string::npos || line.find("srv") != std::string::npos) {
            server_line = true;
        }
    }
    EXPECT_TRUE(server_line) << "the server's slot/srv INFO lines reach the sink";
    for (int level : sink.levels) {
        EXPECT_GE(level, JLLAMA_LOG_LEVEL_NONE);
        EXPECT_LE(level, JLLAMA_LOG_LEVEL_CONT);
    }
}
