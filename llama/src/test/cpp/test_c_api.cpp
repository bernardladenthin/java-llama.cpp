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
#include "rpc_support.hpp"
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
#include <condition_variable>
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

const char *SLOT_SAVE_PATH_MISSING = "Saving or restoring a slot needs a slot directory: load the model with "
                                     "--slot-save-path <dir> (ModelParameters.setSlotSavePath)";

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

// The same for several completions of one prompt: upstream's schema accepts "n" (n_cmpl) up to the
// slot count and answers each in a child task the server creates; the engine posts one task and
// creates none, so "n": 2 came back as one choice without a word. Needs two slots: with one, the
// schema itself refuses "n": 2. Covers the blocking, the streamed and the chat path, which all go
// through populate_completion_task.
TEST_F(CApi, SeveralCompletionsInOneRequestAreRefusedNotDropped) {
    const std::vector<std::string> args = {"jllama", "-m",   path_, "-c",          "128", "-np",
                                           "2",      "-ngl", "0",   "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *two_slots = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &two_slots),
              JLLAMA_STATUS_OK)
        << jllama_last_error();

    Owned response;
    EXPECT_EQ(jllama_completions_oai(two_slots, R"({"prompt":"Hi","n":2,"max_tokens":2})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("'n'"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(jllama_completions(two_slots, R"({"prompt":"Hi","n_cmpl":2,"n_predict":2})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    int task = -1;
    EXPECT_EQ(jllama_completions_oai_stream_request(two_slots, R"({"prompt":"Hi","n":2,"max_tokens":2})", &task),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(jllama_chat_completions(
                  two_slots, R"({"messages":[{"role":"user","content":"Hi"}],"n":2,"max_tokens":2})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);

    ASSERT_EQ(jllama_completions_oai(two_slots, R"({"prompt":"Hi","n":1,"max_tokens":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(response.str()).at("choices").size(), 1u);
    jllama_model_free(two_slots);
}

// The streamed /v1/completions: the body parsed as the blocking one, upstream's text_completion
// chunks, one id, the last with usage (F186).
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

TEST(CApiSlotFiles, ASlotFileIsNamedInsideTheSlotSaveDirectory) {
    // Upstream's slots route takes a file name, validates it (fs_validate_filename: no directory, no
    // "..") and resolves it under --slot-save-path. The C API used the name as the path, so the file
    // landed relative to the working directory and the slot directory had no effect.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-files");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-slots-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    const std::vector<std::string> args = {
        "jllama", "-m", path, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2", "--slot-save-path", dir.string()};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned completion;
    ASSERT_EQ(jllama_completions(model, R"({"prompt":"Hello","n_predict":2,"temperature":0})", &completion.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();

    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "slot0.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_TRUE(std::filesystem::exists(dir / "slot0.bin")) << saved.str();
    EXPECT_FALSE(std::filesystem::exists("slot0.bin"));
    EXPECT_EQ(json::parse(saved.str()).at("filename").get<std::string>(), "slot0.bin") << saved.str();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "slot0.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(restored.str()).at("id_slot").get<int>(), 0) << restored.str();

    // A name with a directory, a parent reference or nothing at all is the caller's error, as upstream's
    // 400 "Invalid filename"; nothing is written outside the slot directory.
    const std::string absolute = (dir / "absolute.bin").string();
    for (const std::string &name : {std::string("../escaped.bin"), std::string("sub/slot.bin"), absolute,
                                    std::string("..\\escaped.bin"), std::string()}) {
        Owned refused;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &refused.s),
                  JLLAMA_STATUS_INVALID_REQUEST)
            << name;
        EXPECT_EQ(std::string(jllama_last_error()), "Invalid slot file name '" + name +
                                                        "': name a file inside the --slot-save-path directory, "
                                                        "without a directory part")
            << name;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, name.c_str(), &refused.s),
                  JLLAMA_STATUS_INVALID_REQUEST)
            << name;
    }
    EXPECT_FALSE(std::filesystem::exists(dir.parent_path() / "escaped.bin"));
    EXPECT_FALSE(std::filesystem::exists(absolute));
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove("slot0.bin", ignored); // only written by the defect this test pins
}

TEST_F(CApi, SlotActionsAndLoraAdapters) {
    Owned response;
    ASSERT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_ERASE, 0, nullptr, &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_TRUE(json::parse(response.str()).is_object());
    // The model was loaded without --slot-save-path: saving and restoring are refused as upstream's
    // slots route refuses them (501), before the file name is looked at; erasing needs no file.
    Owned refused;
    EXPECT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_SAVE, 0, "slot0.bin", &refused.s), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), SLOT_SAVE_PATH_MISSING);
    EXPECT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_RESTORE, 0, "", &refused.s), JLLAMA_STATUS_ERROR);
    EXPECT_STREQ(jllama_last_error(), SLOT_SAVE_PATH_MISSING);
    EXPECT_FALSE(std::filesystem::exists("slot0.bin"));
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
    // The infill fields on the completion paths are refused the same way instead of being dropped.
    EXPECT_EQ(jllama_completions(model_, R"({"input_prefix":"a","input_suffix":"b"})", &response.s),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("fill-in-the-middle"), std::string::npos);
    int task = -1;
    EXPECT_EQ(jllama_completions_stream_request(model_, R"({"input_prefix":"a","input_suffix":"b"})", &task),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("fill-in-the-middle"), std::string::npos);
    EXPECT_NE(jllama_systemone(model_, R"({"state":"x","questions":[]})", &response.s), JLLAMA_STATUS_OK);
    EXPECT_EQ(response.s, nullptr);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
}

namespace {

// Receives a raw streaming task until its stop and returns the last result, the one with the totals.
json receive_to_stop(jllama_model *model, int task) {
    bool stop = false;
    json last;
    for (int i = 0; i < 50 && !stop; ++i) {
        Owned payload;
        if (jllama_completions_stream_receive(model, task, &payload.s, &stop) != JLLAMA_STATUS_OK) {
            ADD_FAILURE() << jllama_last_error();
            return json();
        }
        last = json::parse(payload.str());
    }
    EXPECT_TRUE(stop);
    return last;
}

// The same random model with fill-in-the-middle tokens.
class CApiFim : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf("jllama-c-api-fim", true);
        ASSERT_FALSE(path_.empty());
        const std::vector<std::string> args = {"jllama", "-m", path_,         "-c", "128",
                                               "-ngl",   "0",  "--no-warmup", "-t", "2"};
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model_),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
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
std::string CApiFim::path_;
jllama_model *CApiFim::model_ = nullptr;

const char *INFILL_REQUEST = R"({"input_prefix":"abcdefghij","input_suffix":"klmnopqrst","n_predict":1,)"
                             R"("temperature":0})";

} // namespace

TEST_F(CApiFim, ARawCompletionWithInfillFieldsEvaluatesTheInfillPrompt) {
    // The raw completion paths (LlamaModel.complete/generate stream, handleCompletions blocks) take
    // input_prefix/input_suffix as an infill request; they used to post the empty prompt instead and
    // drop both fields without a word (tokens_evaluated 0).
    Owned infill;
    ASSERT_EQ(jllama_infill(model_, INFILL_REQUEST, &infill.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const int expected = json::parse(infill.str()).at("tokens_evaluated").get<int>();
    EXPECT_GE(expected, 23); // ten prefix and ten suffix bytes around the three FIM tokens

    Owned completion;
    ASSERT_EQ(jllama_completions(model_, INFILL_REQUEST, &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(completion.str()).at("tokens_evaluated").get<int>(), expected);

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model_, INFILL_REQUEST, &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model_, task).at("tokens_evaluated").get<int>(), expected);
}

TEST_F(CApiFim, ARawCompletionWithOnlyOneInfillFieldIsRefused) {
    Owned response;
    int task = -1;
    EXPECT_EQ(jllama_completions(model_, R"({"input_prefix":"abc","n_predict":1})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("input_suffix"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(jllama_completions_stream_request(model_, R"({"input_suffix":"abc","n_predict":1})", &task),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("input_prefix"), std::string::npos) << jllama_last_error();
}

TEST_F(CApi, ARawCompletionWithMessagesCompletesThePromptTheChatTemplateRenders) {
    // The raw completion paths render "messages" through the chat template into the prompt, as
    // jllama_apply_template does; they used to ignore the field and complete the prompt next to it.
    const char *messages = R"([{"role":"user","content":"Hi there"}])";
    Owned prompt;
    ASSERT_EQ(jllama_apply_template(model_, (std::string(R"({"messages":)") + messages + "}").c_str(), &prompt.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned reference;
    ASSERT_EQ(jllama_completions(model_,
                                 json{{"prompt", prompt.str()}, {"n_predict", 1}, {"temperature", 0}}.dump().c_str(),
                                 &reference.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int expected = json::parse(reference.str()).at("tokens_evaluated").get<int>();
    EXPECT_GT(expected, 8); // "Hi there" and the template around it

    const std::string request = std::string(R"({"messages":)") + messages + R"(,"n_predict":1,"temperature":0})";
    Owned completion;
    ASSERT_EQ(jllama_completions(model_, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(completion.str()).at("tokens_evaluated").get<int>(), expected);
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model_, request.c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model_, task).at("tokens_evaluated").get<int>(), expected);
    // An empty prompt beside the messages is no prompt (InferenceParameters("") writes one).
    const std::string with_empty_prompt =
        std::string(R"({"prompt":"","messages":)") + messages + R"(,"n_predict":1,"temperature":0})";
    ASSERT_EQ(jllama_completions(model_, with_empty_prompt.c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(completion.str()).at("tokens_evaluated").get<int>(), expected);
}

TEST_F(CApi, ARawCompletionWithMessagesAndAPromptOrInfillFieldsIsRefused) {
    Owned response;
    int task = -1;
    EXPECT_EQ(jllama_completions(model_, R"({"messages":[{"role":"user","content":"Hi"}],"prompt":"x"})", &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("'prompt'"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(
        jllama_completions_stream_request(
            model_, R"({"messages":[{"role":"user","content":"Hi"}],"input_prefix":"a","input_suffix":"b"})", &task),
        JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("input_prefix"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(jllama_completions(model_, R"({"messages":"no"})", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
}

TEST_F(CApi, ARawCompletionWithMessagesRefusesTheChatOnlyFields) {
    // The raw completion paths keep only the prompt the chat template renders from "messages". The
    // grammar the chat parser derives from response_format, the tool-call grammar of tools,
    // tool_choice and parallel_tool_calls, top_logprobs and the thinking_budget_tokens alias used to
    // be dropped without a word; they are refused, each by name.
    const std::vector<std::pair<std::string, json>> fields = {
        {"response_format", json{{"type", "json_object"}}},
        {"tools",
         json::array({json{{"type", "function"},
                           {"function", json{{"name", "get_weather"}, {"parameters", json{{"type", "object"}}}}}}})},
        {"tool_choice", "required"},
        {"parallel_tool_calls", true},
        {"top_logprobs", 2},
        {"thinking_budget_tokens", 8},
    };
    for (const auto &field : fields) {
        json request = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                        {"n_predict", 1},
                        {"temperature", 0}};
        request[field.first] = field.second;
        if (field.first == "top_logprobs") {
            request["logprobs"] = true; // without it the chat parser refuses top_logprobs itself
        }
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST)
            << field.first;
        EXPECT_NE(std::string(jllama_last_error()).find("'" + field.first + "'"), std::string::npos)
            << jllama_last_error();
        int task = -1;
        EXPECT_EQ(jllama_completions_stream_request(model_, request.dump().c_str(), &task),
                  JLLAMA_STATUS_INVALID_REQUEST)
            << field.first;
        EXPECT_NE(std::string(jllama_last_error()).find("'" + field.first + "'"), std::string::npos)
            << jllama_last_error();
        if (task >= 0) {
            jllama_task_release(model_, task);
        }
    }
    // A null value is no field, and the fields the raw path reads or the template renders stay allowed.
    const json allowed = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                          {"response_format", nullptr},
                          {"chat_template_kwargs", json::object()},
                          {"grammar", "root ::= \"a\""},
                          {"n_predict", 1},
                          {"temperature", 0}};
    Owned response;
    EXPECT_EQ(jllama_completions(model_, allowed.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST(CApiTemplateStops, ARawCompletionWithMessagesStopsAtTheStopWordsOfTheChatTemplate) {
    // A chat template can add stop strings of its own (upstream's Laguna workaround adds
    // "</assistant>", which the model may spell out as text); the chat paths stop at them. The raw
    // completion paths rendered the template and dropped them, so a raw completion with "messages"
    // ran past the end of the answer. The grammar forces the stop string; the end-of-generation token
    // is a byte that never occurs in it.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-stops", false, 4);
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{#- laguna_glm_thinking -#}"
                             "{%- for message in messages -%}"
                             "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
                             "{%- endfor -%}"
                             "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();

    const json request = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                          {"grammar", "root ::= \"</assistant>\""},
                          {"n_predict", 20},
                          {"temperature", 0}};
    Owned completion;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json blocking = json::parse(completion.str());
    EXPECT_EQ(blocking.at("stop_type").get<std::string>(), "word") << blocking.dump();
    EXPECT_EQ(blocking.at("stopping_word").get<std::string>(), "</assistant>") << blocking.dump();
    EXPECT_EQ(blocking.at("content").get<std::string>(), "") << blocking.dump();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, request.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json streamed = receive_to_stop(model, task);
    EXPECT_EQ(streamed.at("stop_type").get<std::string>(), "word") << streamed.dump();
    EXPECT_EQ(streamed.at("stopping_word").get<std::string>(), "</assistant>") << streamed.dump();

    // The stops of the request do not replace the template's: both are in force.
    json with_stop = request;
    with_stop["stop"] = json::array({"zzz"});
    Owned other_stop;
    ASSERT_EQ(jllama_completions(model, with_stop.dump().c_str(), &other_stop.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(other_stop.str()).at("stopping_word").get<std::string>(), "</assistant>") << other_stop.str();
    with_stop["stop"] = "</assist";
    Owned own_stop;
    ASSERT_EQ(jllama_completions(model, with_stop.dump().c_str(), &own_stop.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(own_stop.str()).at("stopping_word").get<std::string>(), "</assist") << own_stop.str();

    // The chat path stops there too: the template's stop string is what both paths share.
    Owned chat;
    ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &chat.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json choice = json::parse(chat.str()).at("choices").at(0);
    EXPECT_EQ(choice.at("finish_reason").get<std::string>(), "stop") << chat.str();
    EXPECT_EQ(choice.at("message").value("content", std::string()), "") << chat.str();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

namespace {

// A ChatML template with a thinking block: it renders an assistant's reasoning_content between
// "<think>" and "</think>" and opens the block in the generation prompt, the shape of the reasoning
// templates that think by default. llama.cpp's template analysis derives the thinking tags
// "<think>"/"</think>" from it, which the chat parser turns into the reasoning budget's tags.
const char *THINKING_TEMPLATE =
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\\n' -}}"
    "{%- if message.reasoning_content -%}"
    "{{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' -}}"
    "{%- endif -%}"
    "{{- message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n<think>\\n' -}}{%- endif -%}";

jllama_model *load_with_chat_template(const std::string &path, const std::string &tmpl) {
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    return model;
}

json budget_request(int budget) {
    json request = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                    {"ignore_eos", true},
                    {"n_predict", 12},
                    {"temperature", 0}};
    if (budget >= 0) {
        request["reasoning_budget_tokens"] = budget;
    }
    return request;
}

} // namespace

TEST(CApiReasoningBudget, ARawCompletionWithMessagesKeepsTheReasoningBudgetOfTheChatTemplate) {
    // The chat paths turn reasoning_budget_tokens into a budget over the thinking tags the template
    // declares, and the sampler forces the end tag once the budget is spent. The raw completion paths
    // rendered the messages and dropped the tags, so the same budget did nothing there. The random
    // model gets an end-of-generation byte of its own, so the '>' of the end tag can be generated.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-budget", false, 4);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_with_chat_template(path, THINKING_TEMPLATE);
    ASSERT_NE(model, nullptr);
    const json request = budget_request(1);

    // The reference: on the chat path the budget closes the thinking block the generation prompt
    // opened (its "\n" after "<think>" spends the one token), and the chat parser takes the end tag
    // off the answer. The raw paths complete the same tokens without the parser: the forced end tag,
    // then the chat path's answer.
    Owned chat;
    ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &chat.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string answer =
        json::parse(chat.str()).at("choices").at(0).at("message").value("content", std::string());
    EXPECT_EQ(answer.find("</think>"), std::string::npos) << chat.str();

    Owned completion;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const std::string blocking = json::parse(completion.str()).at("content").get<std::string>();
    EXPECT_EQ(blocking.rfind("</think>", 0), 0u) << completion.str();
    EXPECT_GE(blocking.size(), answer.size()) << completion.str();
    EXPECT_EQ(blocking.substr(blocking.size() - std::min(blocking.size(), answer.size())), answer)
        << completion.str() << "\n"
        << chat.str();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, request.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::string streamed;
    bool stop = false;
    for (int i = 0; i < 50 && !stop; ++i) {
        Owned chunk;
        ASSERT_EQ(jllama_completions_stream_receive(model, task, &chunk.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        streamed += json::parse(chunk.str()).value("content", std::string());
    }
    EXPECT_TRUE(stop);
    EXPECT_EQ(streamed, blocking);

    // An output-format grammar beside the budget cannot take the generation prompt the budget is
    // primed with: the request runs under the grammar, and the budget waits for a start tag.
    json with_schema = request;
    with_schema["json_schema"] = json{{"type", "string"}};
    with_schema["n_predict"] = 4;
    Owned constrained;
    ASSERT_EQ(jllama_completions(model, with_schema.dump().c_str(), &constrained.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(constrained.str()).at("content").get<std::string>().rfind("\"", 0), 0u) << constrained.str();

    // Without a budget the random model does not close the block on its own: the end tag above is
    // the budget's doing.
    Owned unbounded;
    ASSERT_EQ(jllama_completions(model, budget_request(-1).dump().c_str(), &unbounded.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(unbounded.str()).at("content").get<std::string>().find("</think>"), std::string::npos)
        << unbounded.str();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiReasoningBudget, ARawCompletionWithMessagesRendersTheTemplatesSpecialTagsAsText) {
    // A vocabulary that holds the thinking tags as control tokens, as reasoning models' vocabularies
    // do: a control token is rendered as an empty piece unless the request preserves it. The chat
    // parser's session preserves the template's tags (the chat path renders them and the parser
    // takes them out of the answer); the raw paths took the budget from the parser but not the
    // preserved tokens, so the forced end tag was an invisible empty piece in the raw content.
    const std::string path =
        write_tiny_llama_gguf("jllama-c-api-budget-special", false, 4, {"<think>", "</think>", "<tool>"});
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_with_chat_template(path, THINKING_TEMPLATE);
    ASSERT_NE(model, nullptr);
    const json request = budget_request(1);

    Owned chat;
    ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &chat.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(
        json::parse(chat.str()).at("choices").at(0).at("message").value("content", std::string()).find("</think>"),
        std::string::npos)
        << chat.str();

    Owned completion;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const std::string blocking = json::parse(completion.str()).at("content").get<std::string>();
    EXPECT_EQ(blocking.rfind("</think>", 0), 0u) << completion.str();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, request.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::string streamed;
    bool stop = false;
    for (int i = 0; i < 50 && !stop; ++i) {
        Owned chunk;
        ASSERT_EQ(jllama_completions_stream_receive(model, task, &chunk.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        streamed += json::parse(chunk.str()).value("content", std::string());
    }
    EXPECT_TRUE(stop);
    EXPECT_EQ(streamed, blocking);

    // The request's own preserved tokens stay in force beside the template's: a third control token
    // the template does not name, made the greedy choice after the forced end tag by a logit bias, is
    // rendered as text when the request preserves it, and the template's end tag still is.
    json with_own = request;
    with_own["logit_bias"] = json::array({json::array({N_VOCAB + 2, 100})});
    Owned unpreserved;
    ASSERT_EQ(jllama_completions(model, with_own.dump().c_str(), &unpreserved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(unpreserved.str()).at("content").get<std::string>(), "</think>") << unpreserved.str();
    with_own["preserved_tokens"] = json::array({"<tool>"});
    Owned own;
    ASSERT_EQ(jllama_completions(model, with_own.dump().c_str(), &own.s), JLLAMA_STATUS_OK) << jllama_last_error();
    std::string expected = "</think>";
    for (int i = 1; i < 12; ++i) {
        expected += "<tool>";
    }
    EXPECT_EQ(json::parse(own.str()).at("content").get<std::string>(), expected) << own.str();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiReasoningBudget, ARawCompletionWithoutMessagesForcesTheEndTagOfItsOwnTags) {
    // A raw completion names the budget's tags itself. Upstream's completion schema sets the sequence
    // the budget forces only in the handler of reasoning_budget_message, so tags without a message
    // armed a budget that forced nothing; the chat paths always set the message (the server's
    // default, empty unless --reasoning-budget-message names one). A logit bias makes the model write
    // the start tag (a control token, preserved so it reads as text); a budget of zero then forces
    // the end tag after each one.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-budget-raw", false, 4, {"<think>", "</think>"});
    ASSERT_FALSE(path.empty());
    const std::vector<std::string> args = {"jllama", "-m", path, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    json request = {{"prompt", "Hi"},
                    {"n_predict", 6},
                    {"temperature", 0},
                    {"ignore_eos", true},
                    {"logit_bias", json::array({json::array({N_VOCAB, 100})})},
                    {"preserved_tokens", json::array({"<think>", "</think>"})},
                    {"reasoning_budget_tokens", 0},
                    {"reasoning_budget_start_tag", "<think>"},
                    {"reasoning_budget_end_tags", json::array({"</think>"})}};
    Owned without_message;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &without_message.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(without_message.str()).at("content").get<std::string>(),
              "<think></think><think></think><think></think>")
        << without_message.str();

    // A message is forced before the end tag, as on the chat paths.
    request["reasoning_budget_message"] = "!";
    request["n_predict"] = 3;
    Owned with_message;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &with_message.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(with_message.str()).at("content").get<std::string>(), "<think>!</think>")
        << with_message.str();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST_F(CApi, ARawCompletionWithMessagesAndATemplateWithoutThinkingTagsIgnoresTheBudget) {
    // A template that declares no thinking tags gives the budget nothing to act on, on the chat path
    // and on the raw paths alike: the request runs unchanged, without an error.
    Owned without;
    ASSERT_EQ(jllama_completions(model_, budget_request(-1).dump().c_str(), &without.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned with;
    ASSERT_EQ(jllama_completions(model_, budget_request(1).dump().c_str(), &with.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(with.str()).at("content"), json::parse(without.str()).at("content")) << with.str();
    EXPECT_EQ(json::parse(with.str()).at("content").get<std::string>().find("</think>"), std::string::npos)
        << with.str();
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

namespace {

// A port of the range the serve test above uses (a fixed port would collide with parallel jobs).
int rpc_test_port(int attempt) {
    return 42000 + static_cast<int>(
                       (std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + attempt * 97) % 20000);
}

// jllama_rpc_server_serve() on a thread of its own, with a way to wait for it to return.
class rpc_serve_run {
  public:
    explicit rpc_serve_run(int port)
        : thread_([this, port]() {
              const jllama_status status = jllama_rpc_server_serve("127.0.0.1", port, 2, nullptr, "CPU");
              std::lock_guard<std::mutex> lock(mutex_);
              status_ = status;
              done_ = true;
              ended_.notify_all();
          }) {}
    rpc_serve_run(const rpc_serve_run &) = delete;
    rpc_serve_run &operator=(const rpc_serve_run &) = delete;
    // A serve that is still running (the test failed) is stopped once it listens, so the thread ends.
    ~rpc_serve_run() {
        while (!returned_within(std::chrono::milliseconds(100))) {
            jllama_rpc_server_stop();
        }
        thread_.join();
    }

    bool returned_within(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return ended_.wait_for(lock, timeout, [this]() { return done_; });
    }
    jllama_status status() {
        std::lock_guard<std::mutex> lock(mutex_);
        return status_;
    }

  private:
    std::mutex mutex_;
    std::condition_variable ended_;
    bool done_ = false;
    jllama_status status_ = JLLAMA_STATUS_ERROR;
    std::thread thread_; // last: it reads the members above
};

std::atomic<jllama_status> g_hook_stop_status{JLLAMA_STATUS_ERROR};

// Holds the serving thread in the seam until released, so a stop from another thread lands while
// serve runs and before ggml binds.
struct held_serve {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
};
held_serve g_held;

// Resets the seam when a test ends, whatever its outcome.
struct serve_hook_scope {
    explicit serve_hook_scope(void (*hook)()) { jllama::rpc::set_serve_hook_for_testing(hook); }
    ~serve_hook_scope() { jllama::rpc::set_serve_hook_for_testing(nullptr); }
    serve_hook_scope(const serve_hook_scope &) = delete;
    serve_hook_scope &operator=(const serve_hook_scope &) = delete;
};

} // namespace

// ggml clears its stop flag when it binds the socket (patches/0015), so a stop issued before the
// bind used to be lost and the server went on listening. A stop from the serving thread itself
// (the seam stands for anything that runs there before the bind) must end the serve.
TEST(CApiRpc, AStopBeforeTheBindFromTheServingThreadEndsTheServe) {
    serve_hook_scope hook([]() { g_hook_stop_status.store(jllama_rpc_server_stop()); });
    rpc_serve_run run(rpc_test_port(0));

    EXPECT_TRUE(run.returned_within(std::chrono::seconds(10))) << "serve went on after a stop issued before the bind";
    EXPECT_EQ(g_hook_stop_status.load(), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(run.status(), JLLAMA_STATUS_OK);
    EXPECT_FALSE(jllama_rpc_server_listening());
}

// The same from another thread: the stop returns only once serve has returned, also when it was
// issued before the bind.
TEST(CApiRpc, AStopBeforeTheBindFromAnotherThreadReturnsOnlyOnceTheServeHasEnded) {
    g_held.entered = false;
    g_held.released = false;
    serve_hook_scope hook([]() {
        std::unique_lock<std::mutex> lock(g_held.mutex);
        g_held.entered = true;
        g_held.changed.notify_all();
        g_held.changed.wait(lock, []() { return g_held.released; });
    });
    rpc_serve_run run(rpc_test_port(1));
    {
        std::unique_lock<std::mutex> lock(g_held.mutex);
        ASSERT_TRUE(g_held.changed.wait_for(lock, std::chrono::seconds(10), []() { return g_held.entered; }));
    }

    std::mutex stop_mutex;
    std::condition_variable stop_changed;
    bool stop_returned = false;
    jllama_status stop_status = JLLAMA_STATUS_ERROR;
    std::thread stopper([&]() {
        const jllama_status status = jllama_rpc_server_stop();
        std::lock_guard<std::mutex> lock(stop_mutex);
        stop_status = status;
        stop_returned = true;
        stop_changed.notify_all();
    });
    {
        std::unique_lock<std::mutex> lock(stop_mutex);
        EXPECT_FALSE(stop_changed.wait_for(lock, std::chrono::milliseconds(300), [&]() { return stop_returned; }))
            << "the stop returned while serve had not even started the server";
    }
    {
        std::lock_guard<std::mutex> lock(g_held.mutex);
        g_held.released = true;
        g_held.changed.notify_all();
    }

    EXPECT_TRUE(run.returned_within(std::chrono::seconds(10))) << "serve went on after a stop issued before the bind";
    stopper.join();
    EXPECT_EQ(stop_status, JLLAMA_STATUS_OK);
    EXPECT_EQ(run.status(), JLLAMA_STATUS_OK);
    EXPECT_FALSE(jllama_rpc_server_listening());
}

// A stop while nothing serves stays a no-op: it must not end the next serve as well.
TEST(CApiRpc, AStopWhileNoServeRunsDoesNotStopTheNextOne) {
    ASSERT_EQ(jllama_rpc_server_stop(), JLLAMA_STATUS_OK);
    bool served = false;
    for (int attempt = 0; attempt < 50 && !served; ++attempt) {
        rpc_serve_run run(rpc_test_port(attempt + 2));
        for (int i = 0;
             i < 500 && !jllama_rpc_server_listening() && !run.returned_within(std::chrono::milliseconds(10)); ++i) {
        }
        if (jllama_rpc_server_listening()) {
            served = true;
            EXPECT_EQ(jllama_rpc_server_stop(), JLLAMA_STATUS_OK);
            EXPECT_TRUE(run.returned_within(std::chrono::seconds(10)));
            EXPECT_EQ(run.status(), JLLAMA_STATUS_OK) << jllama_last_error();
        }
    }
    EXPECT_TRUE(served);
    EXPECT_FALSE(jllama_rpc_server_listening());
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
