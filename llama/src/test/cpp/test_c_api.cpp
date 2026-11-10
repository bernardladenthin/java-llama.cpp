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
#include "jllama_java_enum_values.h"
#include "rpc_support.hpp"
#include "tiny_llama_gguf.hpp"

#include "build-info.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

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

TEST_F(CApi, ModelMetaReportsTheCapabilities) {
    // From upstream's server meta, which takes them from mtmd_helper_get_capabilities() (b11901): a
    // model without a projector chats (with or without a template of its own) and neither generates
    // audio nor transcribes it in real time.
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json j = json::parse(meta.str());
    ASSERT_TRUE(j.contains("capabilities")) << meta.str();
    const json caps = j.at("capabilities");
    EXPECT_TRUE(caps.at("chat").get<bool>());
    EXPECT_FALSE(caps.at("audio_gen").get<bool>());
    EXPECT_FALSE(caps.at("audio_stream").get<bool>());
    EXPECT_EQ(caps.size(), 3u) << caps.dump();
    // and no predefined voices: the key is there and empty, as the list of a model without them
    ASSERT_TRUE(j.contains("voices")) << meta.str();
    EXPECT_EQ(j.at("voices").size(), 0u);
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

namespace {

// The same random model loaded with --vocab-only: the tokenizer and nothing else, no server context,
// no inference loop. Every operation that needs the server is refused with one message (F266); before,
// they reached the never-loaded server_context and crashed the process (SIGSEGV in
// llama_vocab::token_bos()) or waited forever for a result no loop would produce (metrics).
class CApiVocabOnly : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf("jllama-c-api-vocab-only");
        ASSERT_FALSE(path_.empty());
        const std::vector<std::string> args = {"jllama", "-m", path_, "--vocab-only"};
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
std::string CApiVocabOnly::path_;
jllama_model *CApiVocabOnly::model_ = nullptr;

// One operation that needs the server, called on a vocab-only model; returns its status.
struct vocab_only_call {
    const char *name;
    jllama_status (*call)(jllama_model *model);
};

std::ostream &operator<<(std::ostream &os, const vocab_only_call &c) { return os << c.name; }

jllama_status call_returning_string(jllama_status status, char *s) {
    jllama_string_free(s);
    return status;
}

const vocab_only_call VOCAB_ONLY_REFUSED[] = {
    {"apply_template",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_apply_template(m, CHAT_REQUEST, &s), s);
     }},
    {"chat_completions",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_chat_completions(m, CHAT_REQUEST, &s), s);
     }},
    {"chat_completions_stream_request",
     [](jllama_model *m) {
         int task = -1;
         return jllama_chat_completions_stream_request(m, CHAT_REQUEST, &task);
     }},
    {"chat_completions_request",
     [](jllama_model *m) {
         int task = -1;
         return jllama_chat_completions_request(m, CHAT_REQUEST, &task);
     }},
    {"chat_completions_stream_receive",
     [](jllama_model *m) {
         char *s = nullptr;
         bool stop = false;
         return call_returning_string(jllama_chat_completions_stream_receive(m, 0, &s, &stop), s);
     }},
    {"completions",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_completions(m, R"({"prompt":"Hi","n_predict":2})", &s), s);
     }},
    {"completions_oai",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_completions_oai(m, R"({"prompt":"Hi","max_tokens":2})", &s), s);
     }},
    {"completions_stream_request",
     [](jllama_model *m) {
         int task = -1;
         return jllama_completions_stream_request(m, R"({"prompt":"Hi","n_predict":2})", &task);
     }},
    {"completions_oai_stream_request",
     [](jllama_model *m) {
         int task = -1;
         return jllama_completions_oai_stream_request(m, R"({"prompt":"Hi","max_tokens":2})", &task);
     }},
    {"completions_stream_receive",
     [](jllama_model *m) {
         char *s = nullptr;
         bool stop = false;
         return call_returning_string(jllama_completions_stream_receive(m, 0, &s, &stop), s);
     }},
    {"task_release", [](jllama_model *m) { return jllama_task_release(m, 0); }},
    {"infill",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_infill(m, R"({"input_prefix":"a","input_suffix":"b"})", &s), s);
     }},
    {"embed",
     [](jllama_model *m) {
         float *values = nullptr;
         size_t count = 0;
         const jllama_status status = jllama_embed(m, "Hi", &values, &count);
         jllama_floats_free(values);
         return status;
     }},
    {"embeddings",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_embeddings(m, R"({"content":"Hi"})", false, &s), s);
     }},
    {"rerank",
     [](jllama_model *m) {
         const char *docs[] = {"a", "b"};
         char *s = nullptr;
         return call_returning_string(jllama_rerank(m, "q", 2, docs, &s), s);
     }},
    {"metrics",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_metrics(m, &s), s);
     }},
    {"slot_action",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_slot_action(m, JLLAMA_SLOT_ERASE, 0, nullptr, &s), s);
     }},
    {"lora_adapters",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_lora_adapters(m, &s), s);
     }},
    {"set_lora_adapters",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_set_lora_adapters(m, "[]", &s), s);
     }},
    {"configure", [](jllama_model *m) { return jllama_configure(m, R"({"n_threads":2})"); }},
    {"systemone",
     [](jllama_model *m) {
         char *s = nullptr;
         return call_returning_string(jllama_systemone(m, R"({"state":"x","questions":[]})", &s), s);
     }},
#ifdef JLLAMA_HAS_NATIVE_SERVER
    {"server_start_attached",
     [](jllama_model *m) {
         const char *argv[] = {"jllama", "--port", "0"};
         jllama_server *server = nullptr;
         const jllama_status status = jllama_server_start_attached(m, 3, argv, &server);
         if (server != nullptr) {
             jllama_server_stop(server);
         }
         return status;
     }},
#endif
};

class CApiVocabOnlyRefuses : public CApiVocabOnly, public ::testing::WithParamInterface<vocab_only_call> {};

} // namespace

TEST_F(CApiVocabOnly, TheTokenizerOperationsAnswer) {
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("n_vocab").get<int>(), N_VOCAB);
    // Without a server context the model can do nothing but tokenize, so the meta claims no capability.
    EXPECT_FALSE(json::parse(meta.str()).contains("capabilities")) << meta.str();
    int *tokens = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tokenize(model_, "Hello", false, &tokens, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(count, 5u);
    Owned text;
    size_t length = 0;
    ASSERT_EQ(jllama_detokenize(model_, tokens, count, &text.s, &length), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(std::string(text.s, length), "Hello");
    jllama_tokens_free(tokens);
    Owned with_pieces;
    ASSERT_EQ(jllama_tokenize_json(model_, "Hi", false, true, &with_pieces.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(with_pieces.str()).at("tokens")[1].at("piece").get<std::string>(), "i");
    const int ids[] = {'H', 'i'};
    Owned detok;
    ASSERT_EQ(jllama_detokenize_json(model_, ids, 2, &detok.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(detok.str()).at("content").get<std::string>(), "Hi");
}

TEST_P(CApiVocabOnlyRefuses, AnOperationThatNeedsTheServerIsRefusedWithTheVocabOnlyMessage) {
    const vocab_only_call &c = GetParam();
    EXPECT_EQ(c.call(model_), JLLAMA_STATUS_ERROR) << c.name;
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find(std::string("jllama_") + c.name + " is not available in vocab-only mode"), std::string::npos)
        << error;
    EXPECT_NE(error.find("--vocab-only"), std::string::npos) << error;
}

INSTANTIATE_TEST_SUITE_P(EveryServerOperation, CApiVocabOnlyRefuses, ::testing::ValuesIn(VOCAB_ONLY_REFUSED),
                         [](const ::testing::TestParamInfo<vocab_only_call> &info) {
                             return std::string(info.param.name);
                         });

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

namespace {

// The same random model with a random vision projector (--mmproj): a multimodal server.
class CApiVision : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf("jllama-c-api-vision");
        mmproj_ = write_tiny_mmproj_gguf();
        ASSERT_FALSE(path_.empty());
        ASSERT_FALSE(mmproj_.empty());
        const std::vector<std::string> args = {"jllama", "-m",   path_, "--mmproj",    mmproj_, "-c",
                                               "128",    "-ngl", "0",   "--no-warmup", "-t",    "2"};
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
        std::filesystem::remove(mmproj_, ignored);
    }
    // The server's media marker, which a raw multimodal prompt places where each image goes.
    static std::string media_marker() {
        Owned meta;
        if (jllama_model_meta(model_, &meta.s) != JLLAMA_STATUS_OK) {
            ADD_FAILURE() << jllama_last_error();
            return std::string();
        }
        return json::parse(meta.str()).value("media_marker", std::string());
    }
    static std::string path_;
    static std::string mmproj_;
    static jllama_model *model_;
};
std::string CApiVision::path_;
std::string CApiVision::mmproj_;
jllama_model *CApiVision::model_ = nullptr;

// A raw request whose prompt is upstream's multimodal prompt object.
std::string multimodal_prompt_request(const std::string &prompt_string, int n_images, const char *length_field) {
    json images = json::array();
    for (int i = 0; i < n_images; ++i) {
        images.push_back(TINY_PNG_BASE64);
    }
    return json{{"prompt", {{"prompt_string", prompt_string}, {"multimodal_data", images}}},
                {length_field, 2},
                {"temperature", 0}}
        .dump();
}

} // namespace

TEST_F(CApiVision, ModelMetaNamesTheMediaMarker) {
    // The marker is random per process (upstream's get_media_marker); a raw multimodal prompt cannot
    // be written without it, and upstream publishes it in /props.
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json j = json::parse(meta.str());
    EXPECT_TRUE(j.at("modalities").at("vision").get<bool>());
    ASSERT_TRUE(j.contains("media_marker")) << meta.str();
    EXPECT_FALSE(j.at("media_marker").get<std::string>().empty());
}

TEST_F(CApiVision, ModelMetaReportsChatButNoAudioCapabilities) {
    // A vision projector reads images; it neither generates speech nor turns the model into a
    // TTS-only one, so chat stays on (the capabilities are computed with the projector loaded).
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model_, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json caps = json::parse(meta.str()).at("capabilities");
    EXPECT_TRUE(caps.at("chat").get<bool>());
    EXPECT_FALSE(caps.at("audio_gen").get<bool>());
    EXPECT_FALSE(caps.at("audio_stream").get<bool>());
}

TEST_F(CApiVision, ARawCompletionTakesUpstreamsMultimodalPromptObject) {
    // upstream's raw schema: "prompt": {"prompt_string": "...<marker>...", "multimodal_data": ["<base64>"]}.
    // The raw paths tokenized it without the projector and refused it as if the model had none.
    const std::string marker = media_marker();
    ASSERT_FALSE(marker.empty());
    Owned text_only;
    ASSERT_EQ(jllama_completions(model_, R"({"prompt":"look  now","n_predict":1,"temperature":0})", &text_only.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int text_tokens = json::parse(text_only.str()).at("tokens_evaluated").get<int>();

    const std::string request = multimodal_prompt_request("look " + marker + " now", 1, "n_predict");
    Owned completion;
    ASSERT_EQ(jllama_completions(model_, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    // four 16x16 patches of the 32x32 image, one embedding each, between the text tokens
    EXPECT_EQ(json::parse(completion.str()).at("tokens_evaluated").get<int>(), text_tokens + 4) << completion.str();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model_, request.c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model_, task).at("tokens_evaluated").get<int>(), text_tokens + 4);

    Owned oai;
    ASSERT_EQ(jllama_completions_oai(
                  model_, multimodal_prompt_request("look " + marker + " now", 1, "max_tokens").c_str(), &oai.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(oai.str()).at("usage").at("prompt_tokens").get<int>(), text_tokens + 4) << oai.str();
}

TEST_F(CApiVision, AMultimodalPromptWhoseMarkersDoNotMatchItsImagesIsRefused) {
    const std::string marker = media_marker();
    ASSERT_FALSE(marker.empty());
    Owned response;
    const json bad_media = {{"prompt", {{"prompt_string", marker}, {"multimodal_data", json::array({"not base64!"})}}}};
    const json no_media = {{"prompt", {{"prompt_string", marker}, {"multimodal_data", json::array()}}}};
    for (const std::string &request :
         {multimodal_prompt_request("no marker", 1, "n_predict"),
          multimodal_prompt_request(marker + " and " + marker, 1, "n_predict"), bad_media.dump(), no_media.dump(),
          // the image last leaves no text token for the logits: the server aborted the process on it
          multimodal_prompt_request("look " + marker, 1, "n_predict")}) {
        EXPECT_EQ(jllama_completions(model_, request.c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST) << request;
    }
}

TEST_F(CApi, AMultimodalPromptObjectOnAModelWithoutAProjectorIsTheCallersFault) {
    Owned response;
    const std::string request = multimodal_prompt_request("look <__media__> now", 1, "n_predict");
    EXPECT_EQ(jllama_completions(model_, request.c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("does not support multimodal"), std::string::npos)
        << jllama_last_error();
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

// The request with the prompt cache off, so it evaluates its whole prompt in one batch. With the cache
// a request after one with the same prompt takes every prompt token but the last from the slot and
// re-evaluates that one in a batch of its own (upstream [TAG_PROMPT_LOGITS]), whose logits are not
// bit-identical to the batched ones, so a greedy near tie can flip between the two answers (F272).
json without_prompt_cache(json request) {
    request["cache_prompt"] = false;
    return request;
}

// The result took nothing from the prompt cache and evaluated all prompt_tokens of its prompt.
void expect_whole_prompt_evaluated(const json &result, int prompt_tokens) {
    EXPECT_EQ(result.at("timings").at("cache_n").get<int>(), 0) << result.dump();
    EXPECT_EQ(result.at("timings").at("prompt_n").get<int>(), prompt_tokens) << result.dump();
}

} // namespace

TEST(CApiReasoningBudget, ARawCompletionWithMessagesKeepsTheReasoningBudgetOfTheChatTemplate) {
    // The chat paths turn reasoning_budget_tokens into a budget over the thinking tags the template
    // declares, and the sampler forces the end tag once the budget is spent. The raw completion paths
    // rendered the messages and dropped the tags, so the same budget did nothing there. The random
    // model gets an end-of-generation byte of its own, so the '>' of the end tag can be generated.
    // Every request below that is compared with another runs without the prompt cache and evaluates
    // its whole prompt in one batch, which its timings prove: with the cache the raw request after the
    // chat request re-evaluated only the last prompt token, which does not reproduce the chat path's
    // batched logits bit for bit (F272).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-budget", false, 4);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_with_chat_template(path, THINKING_TEMPLATE);
    ASSERT_NE(model, nullptr);
    const json request = without_prompt_cache(budget_request(1));

    // The reference: on the chat path the budget closes the thinking block the generation prompt
    // opened (its "\n" after "<think>" spends the one token), and the chat parser takes the end tag
    // off the answer. The raw paths complete the same tokens without the parser: the forced end tag,
    // then the chat path's answer.
    Owned chat;
    ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &chat.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json chat_result = json::parse(chat.str());
    expect_whole_prompt_evaluated(chat_result, chat_result.at("usage").at("prompt_tokens").get<int>());
    const std::string answer = chat_result.at("choices").at(0).at("message").value("content", std::string());
    EXPECT_EQ(answer.find("</think>"), std::string::npos) << chat.str();

    Owned completion;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json raw_result = json::parse(completion.str());
    expect_whole_prompt_evaluated(raw_result, raw_result.at("tokens_evaluated").get<int>());
    const std::string blocking = raw_result.at("content").get<std::string>();
    EXPECT_EQ(blocking.rfind("</think>", 0), 0u) << completion.str();
    EXPECT_GE(blocking.size(), answer.size()) << completion.str();
    EXPECT_EQ(blocking.substr(blocking.size() - std::min(blocking.size(), answer.size())), answer)
        << completion.str() << "\n"
        << chat.str();

    // The stream runs without the prompt cache too, so it is computed like the blocking request.
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, request.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::string streamed;
    json last_chunk;
    bool stop = false;
    for (int i = 0; i < 50 && !stop; ++i) {
        Owned chunk;
        ASSERT_EQ(jllama_completions_stream_receive(model, task, &chunk.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        last_chunk = json::parse(chunk.str());
        streamed += last_chunk.value("content", std::string());
    }
    EXPECT_TRUE(stop);
    expect_whole_prompt_evaluated(last_chunk, last_chunk.at("tokens_evaluated").get<int>());
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
    // and on the raw paths alike: the request runs unchanged, without an error. Both requests run
    // without the prompt cache and evaluate their whole prompt in one batch, which their timings
    // prove; with the cache the second took all but the last prompt token from the slot, and the
    // first did so too whenever the test before it left the same prompt there, so the two answers
    // agreed only by the order the tests run in (F272).
    Owned without;
    ASSERT_EQ(jllama_completions(model_, without_prompt_cache(budget_request(-1)).dump().c_str(), &without.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned with;
    ASSERT_EQ(jllama_completions(model_, without_prompt_cache(budget_request(1)).dump().c_str(), &with.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json without_result = json::parse(without.str());
    const json with_result = json::parse(with.str());
    expect_whole_prompt_evaluated(without_result, without_result.at("tokens_evaluated").get<int>());
    expect_whole_prompt_evaluated(with_result, with_result.at("tokens_evaluated").get<int>());
    EXPECT_EQ(with_result.at("tokens_evaluated").get<int>(), without_result.at("tokens_evaluated").get<int>())
        << with.str();
    EXPECT_EQ(with_result.at("content"), without_result.at("content")) << with.str();
    EXPECT_EQ(with_result.at("content").get<std::string>().find("</think>"), std::string::npos) << with.str();
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

// patches/0022's switch through the C API: it reaches the CPU backend's entry points, round-trips,
// and refuses a null out-pointer. What the two paths compute is test_moe_repack.cpp's business.
TEST(CApiStatic, CpuMoeBatchedSwitchRoundTrips) {
    bool initial = false;
    ASSERT_EQ(jllama_get_cpu_moe_batched(&initial), JLLAMA_STATUS_OK) << jllama_last_error();
    bool now = true;
    ASSERT_EQ(jllama_set_cpu_moe_batched(false), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(jllama_get_cpu_moe_batched(&now), JLLAMA_STATUS_OK);
    EXPECT_FALSE(now);
    ASSERT_EQ(jllama_set_cpu_moe_batched(true), JLLAMA_STATUS_OK);
    ASSERT_EQ(jllama_get_cpu_moe_batched(&now), JLLAMA_STATUS_OK);
    EXPECT_TRUE(now);
    EXPECT_EQ(jllama_get_cpu_moe_batched(nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::strlen(jllama_last_error()), 0u);
    EXPECT_EQ(jllama_set_cpu_moe_batched(initial), JLLAMA_STATUS_OK);
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
    // Q8_0 needs no importance matrix, so the message must not send the caller looking for one
    EXPECT_EQ(std::string(jllama_last_error()).find("importance matrix"), std::string::npos) << jllama_last_error();
}

TEST_F(CApi, TheQuantizerRefusesTheFileTypesItDoesNotProduce) {
    // llama_ftype names every file type a GGUF can declare, llama_model_quantize produces most of
    // them: NVFP4 (39) and OCP FP8 E4M3 (42, llama.cpp b12000) are written by converters only and
    // refused as a target ("invalid output file type"), which is why QuantizationType has a constant
    // for neither. Should upstream start to quantize to one of them, this goes red and the enum can
    // offer it.
    for (const int ftype : {39, 42}) {
        EXPECT_EQ(std::find(JLLAMA_JAVA_FTYPE_VALUES, JLLAMA_JAVA_FTYPE_VALUES + JLLAMA_JAVA_FTYPE_COUNT, ftype),
                  JLLAMA_JAVA_FTYPE_VALUES + JLLAMA_JAVA_FTYPE_COUNT)
            << "QuantizationType has a constant for the refused file type " << ftype;
        const std::string out = path_ + ".ftype" + std::to_string(ftype) + ".gguf";
        EXPECT_EQ(jllama_quantize(path_.c_str(), out.c_str(), ftype, 1, false), JLLAMA_STATUS_ERROR) << ftype;
        EXPECT_NE(std::strlen(jllama_last_error()), 0u) << ftype;
        std::error_code ec;
        EXPECT_FALSE(std::filesystem::exists(out, ec)) << "a refused quantization left a file behind";
        std::filesystem::remove(out, ec);
    }
}

namespace {

// Loads a model with the arguments after the common ones, or returns null with the error in jllama_last_error().
jllama_model *load_tiny(const std::string &model_path, const std::vector<std::string> &extra) {
    std::vector<std::string> args = {"jllama", "-m", model_path, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model);
    return model;
}

// The greedy answer to one prompt, evaluated whole (a fresh model has nothing cached).
std::string greedy_answer(jllama_model *model, int n_predict = 8) {
    Owned completion;
    const std::string request =
        R"({"prompt":"Hello there","temperature":0,"n_predict":)" + std::to_string(n_predict) + "}";
    EXPECT_EQ(jllama_completions(model, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    return completion.s != nullptr ? json::parse(completion.str()).at("content").get<std::string>() : std::string();
}

} // namespace

TEST(CApiShard, AnExtraShardInAnotherDirectoryCompletesTheModel) {
    // --shard (llama.cpp b12233, ModelParameters.addShard): an extra GGUF with tensors of the model,
    // wherever it lies. The generator writes the same weights twice, once whole and once with the last
    // layer in a shard of another directory; loaded with the shard, the split model must answer what
    // the whole one answers, and without it the load must fail -- else the shard would prove nothing.
    const std::string whole = write_tiny_llama_gguf("jllama-c-api-whole");
    std::string shard;
    const std::string split = write_tiny_llama_gguf("jllama-c-api-split", false, '>', {}, &shard);
    ASSERT_FALSE(whole.empty());
    ASSERT_FALSE(split.empty());
    ASSERT_TRUE(std::filesystem::exists(shard));
    ASSERT_NE(std::filesystem::path(shard).parent_path(), std::filesystem::path(split).parent_path());

    jllama_model *reference = load_tiny(whole, {});
    ASSERT_NE(reference, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(reference);
    jllama_model_free(reference);
    ASSERT_FALSE(expected.empty());

    EXPECT_EQ(load_tiny(split, {}), nullptr) << "the model without its shard lacks a layer and must not load";

    jllama_model *sharded = load_tiny(split, {"--shard", shard});
    ASSERT_NE(sharded, nullptr) << jllama_last_error();
    EXPECT_EQ(greedy_answer(sharded), expected);
    jllama_model_free(sharded);

    // upstream's alias, and a shard that does not exist is a failed load, not a model without a layer
    jllama_model *aliased = load_tiny(split, {"--ple", shard});
    EXPECT_NE(aliased, nullptr) << jllama_last_error();
    jllama_model_free(aliased);
    EXPECT_EQ(load_tiny(split, {"--shard", shard + ".missing"}), nullptr);

    std::error_code ignored;
    std::filesystem::remove(whole, ignored);
    std::filesystem::remove(split, ignored);
    std::filesystem::remove_all(std::filesystem::path(shard).parent_path(), ignored);
}

namespace {

// The tokens the model's draft has proposed so far, from the metrics.
int drafted_tokens(jllama_model *model) {
    Owned metrics;
    EXPECT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
    return metrics.s != nullptr ? json::parse(metrics.str()).at("n_draft_tokens_total").get<int>() : -1;
}

} // namespace

TEST(CApiDraft, ADraftModelIsUsedWithoutASpeculativeTypeAndNotWithNone) {
    // Upstream's speculative types default to none, and a plain draft GGUF infers none, so
    // --spec-draft-model alone loaded a second model that drafted nothing: 0 tokens here, against
    // 35 of 48 with --spec-type draft-simple. The engine now uses such a draft model as
    // draft-simple; an explicit type stays what the caller named, "none" included.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-draft");
    ASSERT_FALSE(path.empty());
    const std::vector<std::string> draft = {"--spec-draft-model", path, "--spec-draft-ngl", "0"};
    struct row {
        const char *what;
        std::vector<std::string> type;
        bool drafts;
    };
    for (const row &r :
         {row{"no --spec-type", {}, true}, row{"--spec-type draft-simple", {"--spec-type", "draft-simple"}, true},
          row{"--spec-type none", {"--spec-type", "none"}, false}}) {
        SCOPED_TRACE(r.what);
        std::vector<std::string> extra = draft;
        extra.insert(extra.end(), r.type.begin(), r.type.end());
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(model, 48).empty());
        if (r.drafts) {
            // the draft is the model itself, so what it proposes greedily is what the model samples
            EXPECT_GT(drafted_tokens(model), 0);
        } else {
            EXPECT_EQ(drafted_tokens(model), 0);
        }
        jllama_model_free(model);
    }
    // without a draft model nothing is selected: the default stays upstream's
    jllama_model *plain = load_tiny(path, {});
    ASSERT_NE(plain, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(plain, 48).empty());
    EXPECT_EQ(drafted_tokens(plain), 0);
    jllama_model_free(plain);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiMoeCache, TheCpuShareOfTheMoeCacheIsAcceptedAndRangeChecked) {
    // --moe-cache-cpu-share (llama.cpp b12191, ModelParameters.setMoeCacheCpuShare). Without a cache
    // the share has no effect, so the model loads and answers with it on any host -- which is what
    // shows the option reaches upstream's parser and the context parameters; a share outside 0..1 is
    // refused by upstream, as the Java setter refuses it before. The cache itself (--moe-cache-mib)
    // is not loaded here: it needs a GPU backend and fails the load without one.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-moe-cache");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--moe-cache-cpu-share", "0.25"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model).empty());
    jllama_model_free(model);
    for (const char *share : {"1.5", "-0.1"}) {
        EXPECT_EQ(load_tiny(path, {"--moe-cache-cpu-share", share}), nullptr) << share;
        EXPECT_NE(std::strlen(jllama_last_error()), 0u) << share;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiSlotFiles, ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile) {
    // Since llama.cpp b12131/b12132 upstream builds its paths from plain std::string, which Windows
    // reads in the process's active code page, where it used u8path(); its own executables are UTF-8
    // there through an app manifest, a library inside java.exe is not. The slot file itself is written
    // through ggml_fopen (UTF-8), but its appendix -- the state of the draft context -- through a
    // std::ofstream over that string. So with a draft model and a directory outside ASCII the appendix
    // has to land in the same file, and the round trip has to work, on every platform. This is the
    // measurement TODO.md asks for: red on Windows means the appendix went to a mis-decoded path.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-utf8");
    ASSERT_FALSE(path.empty());
    const std::string leaf = "jllama-slots-\xC3\xA4-\xE6\x97\xA5\xE6\x9C\xAC-" + std::to_string(std::random_device{}());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / std::filesystem::u8path(leaf);
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    // the directory as UTF-8, which is what the C API takes on every platform
    const auto dir_u8 = dir.u8string();
    const std::string dir_arg(dir_u8.begin(), dir_u8.end());

#ifdef _WIN32
    // Measured on all three Windows jobs of Publish run 42783058488: llama.cpp checks the option
    // with std::filesystem::is_directory over the plain string (fs_is_directory, common/arg.cpp),
    // which Windows reads in the process's ANSI code page, so a UTF-8 path outside ASCII is "not a
    // directory" and the load is refused before any slot file is written. Pinned as it is (README,
    // "Slot files"): a load that works here means upstream reads the path as UTF-8 now, and this
    // branch goes. A process whose code page is UTF-8 takes the path of the other platforms.
    if (GetACP() != CP_UTF8) {
        EXPECT_EQ(load_tiny(path, {"--slot-save-path", dir_arg}), nullptr)
            << "loads now: upstream reads the path as UTF-8, run the round trip here too";
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
        std::filesystem::remove(path, ignored);
        return;
    }
#endif
    for (const bool with_draft : {false, true}) {
        SCOPED_TRACE(with_draft ? "with a draft model" : "without a draft model");
        std::vector<std::string> extra = {"--slot-save-path", dir_arg};
        if (with_draft) {
            extra.insert(extra.end(),
                         {"--spec-draft-model", path, "--spec-draft-ngl", "0", "--spec-type", "draft-simple"});
        }
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(model, 48).empty());

        const std::string name = with_draft ? "draft.bin" : "plain.bin";
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json saved_json = json::parse(saved.str());
        std::error_code ec;
        const auto on_disk = std::filesystem::file_size(dir / name, ec);
        ASSERT_FALSE(ec) << "no file under the directory's real name: " << ec.message();
        // the server reports what it wrote, appendix included: all of it has to be in this one file
        EXPECT_EQ(on_disk, saved_json.at("n_written").get<size_t>()) << saved.str();

        Owned restored;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, name.c_str(), &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json restored_json = json::parse(restored.str());
        EXPECT_EQ(restored_json.at("n_restored").get<int>(), saved_json.at("n_saved").get<int>()) << restored.str();
        EXPECT_EQ(restored_json.at("n_read").get<size_t>(), on_disk) << restored.str();
        jllama_model_free(model);
    }
    // the draft's file carries the appendix this test is about: without it the case above proves nothing
    std::error_code size_ec;
    EXPECT_GT(std::filesystem::file_size(dir / "draft.bin", size_ec),
              std::filesystem::file_size(dir / "plain.bin", size_ec));
    // nothing but the two slot files, and no sibling directory under a mis-decoded name
    size_t entries = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        (void)entry;
        ++entries;
    }
    EXPECT_EQ(entries, 2u);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
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
        // Below every default ephemeral range (Linux from 32768, Windows and macOS from 49152).
        const int port =
            20000 + static_cast<int>(
                        (std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + attempt * 97) % 12000);
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
// Below every default ephemeral range (Linux from 32768, Windows and macOS from 49152): a port out
// of 42000-62000 was once taken by an outgoing connection on the aarch64 runner, and the serve failed
// to bind ("Failed to create server socket", run 40195444689).
int rpc_test_port(int attempt) {
    return 20000 + static_cast<int>(
                       (std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + attempt * 97) % 12000);
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

namespace {

// The second serve of the test below, started from the stop seam and held in the serve seam.
held_serve g_held_next;
std::unique_ptr<rpc_serve_run> g_next_run;
std::atomic<int> g_serve_hook_calls{0};

void hold_in(held_serve &held) {
    std::unique_lock<std::mutex> lock(held.mutex);
    held.entered = true;
    held.changed.notify_all();
    held.changed.wait(lock, [&held]() { return held.released; });
}

bool wait_entered(held_serve &held) {
    std::unique_lock<std::mutex> lock(held.mutex);
    return held.changed.wait_for(lock, std::chrono::seconds(10), [&held]() { return held.entered; });
}

void release(held_serve &held) {
    std::lock_guard<std::mutex> lock(held.mutex);
    held.released = true;
    held.changed.notify_all();
}

struct stop_hook_scope {
    explicit stop_hook_scope(void (*hook)()) { jllama::rpc::set_stop_hook_for_testing(hook); }
    ~stop_hook_scope() { jllama::rpc::set_stop_hook_for_testing(nullptr); }
    stop_hook_scope(const stop_hook_scope &) = delete;
    stop_hook_scope &operator=(const stop_hook_scope &) = delete;
};

} // namespace

// A stop is for the serve in progress when it is called, not for one started afterwards. Here the
// next serve starts while the stop still waits for the first (the stop seam runs once the request
// is recorded): the stop must return as soon as the first serve has ended, and the second must come
// up and listen. Before, the stop repeated ggml's stop until no serve ran at all, so a serve that
// took over right after the first had ended was ended too -- it returned without binding (the
// request was still set, it is reset only when no serve runs), or lost its listener once bound.
TEST(CApiRpc, AStopDoesNotReachTheServeStartedAfterIt) {
    g_held.entered = g_held.released = false;
    g_held_next.entered = g_held_next.released = false;
    g_serve_hook_calls.store(0);
    g_next_run.reset();
    serve_hook_scope serve_hook([]() { hold_in(g_serve_hook_calls.fetch_add(1) == 0 ? g_held : g_held_next); });
    stop_hook_scope stop_hook([]() {
        // Once: the stop of the second serve below must not start a third.
        jllama::rpc::set_stop_hook_for_testing(nullptr);
        // The stop is recorded: start the next serve, let it enter its seam, then end the first.
        g_next_run = std::make_unique<rpc_serve_run>(rpc_test_port(11));
        EXPECT_TRUE(wait_entered(g_held_next)) << "the second serve did not start";
        release(g_held);
    });
    rpc_serve_run first(rpc_test_port(10));
    ASSERT_TRUE(wait_entered(g_held)) << "the first serve did not start";

    const jllama_status stop_status = jllama_rpc_server_stop();
    EXPECT_EQ(stop_status, JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(first.returned_within(std::chrono::seconds(10))) << "the first serve went on after the stop";
    EXPECT_EQ(first.status(), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(g_next_run, nullptr);

    // The second serve was not meant: released, it binds and listens.
    release(g_held_next);
    for (int i = 0;
         i < 500 && !jllama_rpc_server_listening() && !g_next_run->returned_within(std::chrono::milliseconds(10));
         ++i) {
    }
    EXPECT_TRUE(jllama_rpc_server_listening()) << "the stop of the first serve ended the second one";
    EXPECT_EQ(jllama_rpc_server_stop(), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(g_next_run->returned_within(std::chrono::seconds(10)));
    EXPECT_EQ(g_next_run->status(), JLLAMA_STATUS_OK) << jllama_last_error();
    g_next_run.reset();
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
    // The voice variant (6.1.0) checks the same before it touches anything; the old function is it with
    // a null voice.
    EXPECT_EQ(jllama_tts_synthesize_voice(nullptr, "Hi", nullptr, "en", "male", 10, 1, 1, &wav, &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    int not_a_tts = 0;
    EXPECT_EQ(jllama_tts_synthesize_voice(reinterpret_cast<jllama_tts_model *>(&not_a_tts), nullptr, nullptr, nullptr,
                                          nullptr, 10, 1, 1, &wav, &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(wav, nullptr);
    // The temperature variant (b12233) is the one the other two end in.
    EXPECT_EQ(jllama_tts_synthesize_temperature(nullptr, "Hi", nullptr, nullptr, "alba", 10, 1, 1, 0.7f, &wav, &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    EXPECT_EQ(jllama_tts_synthesize_temperature(reinterpret_cast<jllama_tts_model *>(&not_a_tts), nullptr, nullptr,
                                                nullptr, nullptr, 10, 1, 1, -1.0f, &wav, &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(wav, nullptr);
    bool supported = true;
    EXPECT_EQ(jllama_tts_supports_speaker_reference(nullptr, &supported), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    EXPECT_TRUE(supported) << "a refused call must not write the out-parameter";
    // The out-parameter is checked before the handle is used, so any non-null pointer reaches it.
    int not_a_model = 0;
    EXPECT_EQ(jllama_tts_supports_speaker_reference(reinterpret_cast<const jllama_tts_model *>(&not_a_model), nullptr),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "supported must not be null");
    // The predefined voices (b12061): the same refusals, and a refused call leaves the out-parameters alone.
    char **voices = reinterpret_cast<char **>(&not_a_model);
    size_t n_voices = 7;
    EXPECT_EQ(jllama_tts_voices(nullptr, &voices, &n_voices), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    EXPECT_EQ(n_voices, 7u) << "a refused call must not write the out-parameters";
    EXPECT_EQ(jllama_tts_voices(reinterpret_cast<const jllama_tts_model *>(&not_a_model), nullptr, &n_voices),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "names and count must not be null");
    EXPECT_EQ(jllama_tts_voices(reinterpret_cast<const jllama_tts_model *>(&not_a_model), &voices, nullptr),
              JLLAMA_STATUS_INVALID_ARGUMENT);
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

// ---------------------------------------------------------------------------------------------------
// Every value a Java enum can put on the wire, against a real model. test_model_flags.cpp proves the
// parser accepts each; accepting a value and being able to run with it are two things (a KV cache
// type parses and then fails the context), so each is loaded and used here. The model is the wide
// one: quantized types work in blocks of up to 256 values (IQ1_M), and a 32-wide model cannot hold one.
// ---------------------------------------------------------------------------------------------------

class CApiEnumSweep : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf(model());
        ASSERT_FALSE(path_.empty());
        imatrix_ = write_tiny_imatrix_gguf(model());
        ASSERT_FALSE(imatrix_.empty());
    }
    static void TearDownTestSuite() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
        std::filesystem::remove(imatrix_, ignored);
    }
    // The file types llama_model_quantize refuses without an importance matrix; in Java,
    // QuantizationType.requiresImportanceMatrix() (QuantizationTypeTest holds it to the same nine).
    // one layer: the IQ types search their codebooks per block, which is slow, and a layer is enough
    static tiny_model_spec model() {
        tiny_model_spec spec = tiny_model_spec::wide(256);
        spec.n_layer = 1;
        return spec;
    }
    static bool needs_imatrix(int ftype) {
        static const std::set<int> types = {19, 20, 21, 22, 23, 24, 28, 29, 31};
        return types.count(ftype) != 0;
    }
    static std::string path_;
    static std::string imatrix_;
};
std::string CApiEnumSweep::path_;
std::string CApiEnumSweep::imatrix_;

TEST_F(CApiEnumSweep, EveryQuantizationTypeOfTheJavaEnumGivesAModelThatLoadsAndAnswers) {
    // QuantizationType is the Java side's list of quantizer targets; CMake reads its constants into
    // JLLAMA_JAVA_FTYPE_*. Each must be a file type llama_model_quantize writes, and what it writes
    // must load -- a constant for a type it refuses is a call that can only fail (NVFP4 was one).
    // Every type is written twice: without an importance matrix, where exactly the nine that need
    // one are refused, and with one, where all of them work. The width matters: at 128 columns the
    // IQ3 types fell back to another type and seemed to need none, at 256 they are really written.
    ASSERT_GT(JLLAMA_JAVA_FTYPE_COUNT, 30) << "the extractor read implausibly few QuantizationType constants";
    int refused_without = 0;
    for (int i = 0; i < JLLAMA_JAVA_FTYPE_COUNT; ++i) {
        const std::string name = JLLAMA_JAVA_FTYPE_NAMES[i];
        const int ftype = JLLAMA_JAVA_FTYPE_VALUES[i];
        for (const bool with_imatrix : {false, true}) {
            SCOPED_TRACE("QuantizationType." + name + (with_imatrix ? " with" : " without") + " an importance matrix");
            const std::string out = path_ + ".java-ftype-" + name + (with_imatrix ? "-im" : "") + ".gguf";
            const jllama_status quantized = jllama_quantize_imatrix(path_.c_str(), out.c_str(), ftype, 1, false,
                                                                    with_imatrix ? imatrix_.c_str() : nullptr);
            std::error_code ec;
            if (!with_imatrix && needs_imatrix(ftype)) {
                EXPECT_EQ(quantized, JLLAMA_STATUS_ERROR);
                EXPECT_NE(std::string(jllama_last_error()).find("importance matrix"), std::string::npos)
                    << jllama_last_error();
                ++refused_without;
                std::filesystem::remove(out, ec);
                continue;
            }
            EXPECT_EQ(quantized, JLLAMA_STATUS_OK) << jllama_last_error();
            if (quantized != JLLAMA_STATUS_OK) {
                continue;
            }
            jllama_model *model = load_tiny(out, {});
            EXPECT_NE(model, nullptr) << jllama_last_error();
            if (model != nullptr) {
                Owned completion;
                EXPECT_EQ(jllama_completions(model, R"({"prompt":"Hello there","temperature":0,"n_predict":4})",
                                             &completion.s),
                          JLLAMA_STATUS_OK)
                    << jllama_last_error();
                jllama_model_free(model);
            }
            std::filesystem::remove(out, ec);
        }
    }
    EXPECT_EQ(refused_without, 9) << "QuantizationType lacks a type that needs an importance matrix";
}

TEST_F(CApiEnumSweep, AnImportanceMatrixChangesTheWeightsAndIsRecorded) {
    // The matrix must reach the quantizer, not only be accepted: Q4_K_M, which works without one,
    // writes other bytes with it, and the output names the file the way llama-quantize does.
    const std::string plain = path_ + ".q4km.gguf";
    const std::string weighted = path_ + ".q4km-im.gguf";
    ASSERT_EQ(jllama_quantize(path_.c_str(), plain.c_str(), 15, 1, false), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(jllama_quantize_imatrix(path_.c_str(), weighted.c_str(), 15, 1, false, imatrix_.c_str()),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const auto bytes = [](const std::string &file) {
        std::ifstream in(file, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    EXPECT_NE(bytes(plain), bytes(weighted));

    gguf_init_params no_data = {true, nullptr};
    gguf_context *g = gguf_init_from_file(weighted.c_str(), no_data);
    ASSERT_NE(g, nullptr);
    const int64_t dataset_key = gguf_find_key(g, "quantize.imatrix.dataset");
    EXPECT_GE(gguf_find_key(g, "quantize.imatrix.file"), 0);
    EXPECT_GE(gguf_find_key(g, "quantize.imatrix.entries_count"), 0);
    EXPECT_GE(gguf_find_key(g, "quantize.imatrix.chunks_count"), 0);
    EXPECT_GE(dataset_key, 0);
    if (dataset_key >= 0) {
        EXPECT_STREQ(gguf_get_val_str(g, dataset_key), "random.txt");
    }
    gguf_free(g);
    gguf_context *unweighted = gguf_init_from_file(plain.c_str(), no_data);
    ASSERT_NE(unweighted, nullptr);
    EXPECT_LT(gguf_find_key(unweighted, "quantize.imatrix.file"), 0);
    gguf_free(unweighted);
    std::error_code ec;
    std::filesystem::remove(plain, ec);
    std::filesystem::remove(weighted, ec);
}

TEST_F(CApiEnumSweep, AnImportanceMatrixThatCannotBeUsedIsRefusedByName) {
    const std::string out = path_ + ".bad-imatrix.gguf";
    // a missing file, and a GGUF that is no importance matrix (the model itself)
    for (const std::string &bad : {std::string("/nonexistent/imatrix.gguf"), path_}) {
        EXPECT_EQ(jllama_quantize_imatrix(path_.c_str(), out.c_str(), 15, 1, false, bad.c_str()), JLLAMA_STATUS_ERROR);
        EXPECT_NE(std::string(jllama_last_error()).find("importance matrix"), std::string::npos) << jllama_last_error();
        std::error_code ec;
        EXPECT_FALSE(std::filesystem::exists(out, ec));
        std::filesystem::remove(out, ec);
    }
    // an empty path is no matrix, as null is
    EXPECT_EQ(jllama_quantize_imatrix(path_.c_str(), out.c_str(), 15, 1, false, ""), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::error_code ec;
    std::filesystem::remove(out, ec);
}

TEST_F(CApiEnumSweep, EveryCliEnumValueLoadsTheModelAndIsUsable) {
    // The values that cannot work on this model, each with its reason; they must keep failing, so a
    // reason that no longer holds is noticed. Everything else loads and answers a request.
    const std::map<std::string, std::string> expected_load_failure = {
        // needs a model with MTP layers ("context type MTP requested but model doesn't contain MTP layers")
        {"--spec-type draft-mtp", "no MTP layers in the model"},
        // rank pooling scores with a classification head, which a plain decoder lacks (the engine
        // refuses the load; CApiBert covers rank pooling on a model that has one)
        {"--pooling rank", "no classification head"},
    };
    // Fails or loads with the machine: tensor split needs a GPU device, which most test runners lack,
    // and row split a device with split buffers, which Metal is not ("device MTL0 does not support
    // split buffers", both macOS jobs of Publish run 42783058488).
    const std::set<std::string> machine_dependent = {"--split-mode tensor", "--split-mode row"};
    int used = 0;
    for (int i = 0; i < JLLAMA_JAVA_ENUM_COUNT; ++i) {
        const std::string option = JLLAMA_JAVA_ENUM_OPTIONS[i];
        const std::string value = JLLAMA_JAVA_ENUM_VALUES[i];
        if (option == "--numa") {
            continue; // sets the CPU affinity of the whole test process
        }
        SCOPED_TRACE(option + " " + value);
        std::vector<std::string> extra = {option, value, "--fit", "off"};
        if (option == "--pooling") {
            extra.push_back("--embedding");
        }
        jllama_model *model = load_tiny(path_, extra);
        if (expected_load_failure.count(option + " " + value) != 0) {
            EXPECT_EQ(model, nullptr) << "loads now: take it out of the exceptions";
            jllama_model_free(model);
            continue;
        }
        if (machine_dependent.count(option + " " + value) == 0) {
            EXPECT_NE(model, nullptr) << jllama_last_error();
        }
        if (model == nullptr) {
            continue;
        }
        Owned response;
        if (option == "--pooling") {
            ASSERT_EQ(jllama_embeddings(model, R"({"content":"Hello there"})", false, &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            const json rows = json::parse(response.str()).at(0).at("embedding");
            // without pooling one vector per token, with any pooling one for the text
            if (value == "none") {
                EXPECT_GT(rows.size(), 1u);
            } else {
                EXPECT_EQ(rows.size(), 1u);
            }
        } else {
            EXPECT_EQ(
                jllama_completions(model, R"({"prompt":"Hello there","temperature":0,"n_predict":4})", &response.s),
                JLLAMA_STATUS_OK)
                << jllama_last_error();
        }
        jllama_model_free(model);
        ++used;
    }
    EXPECT_GT(used, 60) << "the sweep ran over implausibly few values";
}

TEST(CApiPooling, AModelThatDeclaresItsPoolingTypeNeedsNoOption) {
    // Why there is no PoolingType constant for "unspecified": without --pooling the model's own
    // <arch>.pooling_type applies. Two models with the same weights, one declaring mean pooling:
    // it answers without the option exactly what the other answers with --pooling mean.
    tiny_model_spec declared;
    declared.prefix = "jllama-pooling-mean";
    declared.pooling_type = 1; // LLAMA_POOLING_TYPE_MEAN
    const std::string with_type = write_tiny_llama_gguf(declared);
    const std::string plain = write_tiny_llama_gguf("jllama-pooling-plain");
    ASSERT_FALSE(with_type.empty());
    ASSERT_FALSE(plain.empty());
    const auto embedding = [](const std::string &path, const std::vector<std::string> &extra) {
        jllama_model *model = load_tiny(path, extra);
        EXPECT_NE(model, nullptr) << jllama_last_error();
        Owned response;
        if (model != nullptr) {
            EXPECT_EQ(jllama_embeddings(model, R"({"content":"Hello there"})", false, &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            jllama_model_free(model);
        }
        return response.s != nullptr ? json::parse(response.str()).at(0).at("embedding").dump() : std::string();
    };
    const std::string by_model = embedding(with_type, {"--embedding"});
    const std::string by_option = embedding(plain, {"--embedding", "--pooling", "mean"});
    const std::string by_other_option = embedding(plain, {"--embedding", "--pooling", "last"});
    EXPECT_FALSE(by_model.empty());
    EXPECT_EQ(by_model, by_option);
    EXPECT_NE(by_option, by_other_option) << "mean and last pooling agree: the comparison above proves nothing";
    std::error_code ec;
    std::filesystem::remove(with_type, ec);
    std::filesystem::remove(plain, ec);
}

// ---------------------------------------------------------------------------------------------------
// A second architecture: the random BERT of tiny_llama_gguf.hpp. An encoder is what cls pooling and
// reranking are for, and the decoder above can only pretend at both.
// ---------------------------------------------------------------------------------------------------

namespace {

// The embedding rows of one text, each as its JSON text; empty when the call failed. Every row
// must hold n_embd finite numbers.
std::vector<std::string> embedding_rows(jllama_model *model, const std::string &text, size_t n_embd) {
    Owned response;
    const std::string request = json{{"content", text}}.dump();
    EXPECT_EQ(jllama_embeddings(model, request.c_str(), false, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    std::vector<std::string> rows;
    if (response.s != nullptr) {
        const json all = json::parse(response.str()).at(0).at("embedding");
        for (size_t i = 0; i < all.size(); ++i) {
            const json row = all.at(i);
            EXPECT_EQ(row.size(), n_embd);
            for (size_t j = 0; j < row.size(); ++j) {
                EXPECT_TRUE(std::isfinite(row.at(j).get<float>()));
            }
            rows.push_back(row.dump());
        }
    }
    return rows;
}

} // namespace

TEST(CApiBert, AnEncoderEmbedsWithEveryPoolingTypeAndItsOwnByDefault) {
    const std::string path = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */);
    ASSERT_FALSE(path.empty());
    const std::string text = "hello there world"; // [CLS] hello there world [SEP]
    std::map<std::string, std::vector<std::string>> by_pooling;
    for (const char *pooling : {"", "none", "mean", "cls", "last"}) {
        SCOPED_TRACE(std::string("--pooling ") + pooling);
        std::vector<std::string> extra = {"--embedding"};
        if (*pooling != 0) {
            extra.insert(extra.end(), {"--pooling", pooling});
        }
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        by_pooling[pooling] = embedding_rows(model, text, 32);
        // the same text twice is the same vector: an encoder has no cache to differ over
        EXPECT_EQ(by_pooling[pooling], embedding_rows(model, text, 32));
        jllama_model_free(model);
    }
    EXPECT_EQ(by_pooling["none"].size(), 5u) << "one row per token, the two special tokens included";
    for (const char *pooled : {"", "mean", "cls", "last"}) {
        EXPECT_EQ(by_pooling[pooled].size(), 1u) << pooled;
    }
    EXPECT_EQ(by_pooling[""], by_pooling["cls"]) << "without --pooling the model's own type (cls) applies";
    EXPECT_NE(by_pooling["cls"], by_pooling["mean"]);
    EXPECT_NE(by_pooling["cls"], by_pooling["last"]);
    EXPECT_NE(by_pooling["mean"], by_pooling["last"]);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiBert, ARerankerScoresEveryDocumentByItselfAndTheQuery) {
    const std::string path = write_tiny_bert_gguf(true, 4 /* LLAMA_POOLING_TYPE_RANK */);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--reranking"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const auto scores = [&](const std::string &query, const std::vector<const char *> &documents) {
        Owned response;
        EXPECT_EQ(
            jllama_rerank(model, query.c_str(), static_cast<int>(documents.size()), documents.data(), &response.s),
            JLLAMA_STATUS_OK)
            << jllama_last_error();
        std::map<std::string, float> by_document;
        if (response.s != nullptr) {
            const json results = json::parse(response.str());
            EXPECT_EQ(results.size(), documents.size());
            std::set<int> indices;
            for (size_t i = 0; i < results.size(); ++i) {
                const json result = results.at(i);
                const int index = result.at("index").get<int>();
                indices.insert(index);
                EXPECT_EQ(result.at("document").get<std::string>(), documents.at(static_cast<size_t>(index)));
                EXPECT_GT(result.at("tokens_evaluated").get<int>(), 0);
                const float score = result.at("score").get<float>();
                EXPECT_TRUE(std::isfinite(score));
                by_document[documents.at(static_cast<size_t>(index))] = score;
            }
            EXPECT_EQ(indices.size(), documents.size()) << "every document is answered once";
        }
        return by_document;
    };
    const std::string query = "what is the capital of france ?";
    const auto forward = scores(query, {"paris is the capital of france", "the cat sat on the mat", "berlin"});
    const auto backward = scores(query, {"berlin", "the cat sat on the mat", "paris is the capital of france"});
    // a document's score depends on the query and the document, not on its neighbours or its place
    EXPECT_EQ(forward, backward);
    const auto alone = scores(query, {"berlin"});
    EXPECT_EQ(alone.at("berlin"), forward.at("berlin"));
    EXPECT_NE(forward.at("berlin"), forward.at("the cat sat on the mat"));
    // and it does depend on the query
    EXPECT_NE(scores("the dog", {"berlin"}).at("berlin"), alone.at("berlin"));

    // a reranking context computes no logits, so a completion is refused, not answered with noise
    Owned completion;
    EXPECT_EQ(jllama_completions(model, R"({"prompt":"hello","n_predict":2})", &completion.s), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("logits"), std::string::npos) << jllama_last_error();
    // and the model still reranks afterwards
    EXPECT_EQ(scores(query, {"berlin"}).at("berlin"), alone.at("berlin"));
    jllama_model_free(model);

    // The same encoder without the classification head is no reranker: llama.cpp would answer the
    // first value of the pooled embedding as the score, so the load is refused by name -- with
    // --reranking and with rank pooling asked for directly.
    const std::string encoder = write_tiny_bert_gguf(false, 2);
    ASSERT_FALSE(encoder.empty());
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"--reranking"}, std::vector<std::string>{"--embedding", "--pooling", "rank"}}) {
        jllama_model *headless = load_tiny(encoder, extra);
        EXPECT_EQ(headless, nullptr) << "a model without a classification head loaded as a reranker";
        EXPECT_NE(std::string(jllama_last_error()).find("classification head"), std::string::npos)
            << jllama_last_error();
        jllama_model_free(headless);
    }
    // and it still loads for what it is
    jllama_model *embedder = load_tiny(encoder, {"--embedding"});
    EXPECT_NE(embedder, nullptr) << jllama_last_error();
    jllama_model_free(embedder);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(encoder, ec);
}

// ---------------------------------------------------------------------------------------------------
// A mixture of experts (tiny_model_spec::n_expert): the decoder whose feed-forward is a router over
// expert matrices -- the MUL_MAT_ID path, which test_moe_repack.cpp checks as an operation and
// nothing checked through a model.
// ---------------------------------------------------------------------------------------------------

namespace {

tiny_model_spec moe_spec() {
    tiny_model_spec spec = tiny_model_spec::wide(64, "jllama-moe");
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    return spec;
}

} // namespace

TEST(CApiMoe, AMixtureOfExpertsAnswersTheSameOnBothRepackPathsAndInEveryQuantization) {
    const std::string path = write_tiny_llama_gguf(moe_spec());
    ASSERT_FALSE(path.empty());
    bool batched_before = true;
    ASSERT_EQ(jllama_get_cpu_moe_batched(&batched_before), JLLAMA_STATUS_OK) << jllama_last_error();

    // the experts change the answer: the same shape without them is another model
    tiny_model_spec dense = tiny_model_spec::wide(64, "jllama-moe-dense");
    const std::string dense_path = write_tiny_llama_gguf(dense);
    jllama_model *dense_model = load_tiny(dense_path, {});
    ASSERT_NE(dense_model, nullptr) << jllama_last_error();
    const std::string dense_answer = greedy_answer(dense_model, 16);
    jllama_model_free(dense_model);

    // F32, and the types the CPU backend repacks for MUL_MAT_ID where the CPU supports it
    // (patches/0022: Q4_0, Q4_K, IQ4_NL; 2 = Q4_0, 15 = Q4_K_M, 25 = IQ4_NL)
    for (const int ftype : {0, 2, 15, 25}) {
        SCOPED_TRACE("file type " + std::to_string(ftype));
        std::string model_path = path;
        if (ftype != 0) {
            model_path = path + ".ftype" + std::to_string(ftype) + ".gguf";
            ASSERT_EQ(jllama_quantize(path.c_str(), model_path.c_str(), ftype, 1, false), JLLAMA_STATUS_OK)
                << jllama_last_error();
        }
        std::map<bool, std::string> answers;
        for (const bool batched : {true, false}) {
            ASSERT_EQ(jllama_set_cpu_moe_batched(batched), JLLAMA_STATUS_OK) << jllama_last_error();
            jllama_model *model = load_tiny(model_path, {});
            ASSERT_NE(model, nullptr) << jllama_last_error();
            answers[batched] = greedy_answer(model, 16);
            jllama_model_free(model);
        }
        EXPECT_FALSE(answers[true].empty());
        EXPECT_FALSE(answers[false].empty());
        // The batched path and upstream's token-by-token path are the same computation for F32 and
        // Q4_0; for Q4_K and IQ4_NL the two kernels round differently (measured for Q4_K with real
        // weights: KLD 0.0075), so a greedy near tie may fall the other way and only F32 and Q4_0
        // are held to equality.
        if (ftype == 0 || ftype == 2) {
            EXPECT_EQ(answers[true], answers[false]);
        }
        if (ftype == 0) {
            EXPECT_NE(answers[true], dense_answer) << "the experts have no effect on the answer";
        } else {
            std::error_code ec;
            std::filesystem::remove(model_path, ec);
        }
    }
    EXPECT_EQ(jllama_set_cpu_moe_batched(batched_before), JLLAMA_STATUS_OK);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(dense_path, ec);
}

// ---------------------------------------------------------------------------------------------------
// jllama_imatrix: the importance matrix computed here, not written by the fixture.
// ---------------------------------------------------------------------------------------------------

namespace {

// A calibration text of about n_tokens byte tokens: words over the printable ASCII the model knows.
std::string write_calibration_text(const std::string &prefix, size_t n_tokens) {
    const std::string path = (std::filesystem::temp_directory_path() / (prefix + "-calibration.txt")).string();
    std::ofstream out(path, std::ios::binary);
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> letter('a', 'z');
    for (size_t i = 0; i < n_tokens; ++i) {
        out << (i % 6 == 5 ? ' ' : static_cast<char>(letter(rng)));
    }
    return path;
}

// The entries of an importance matrix file: tensor name -> number of values.
std::map<std::string, int64_t> imatrix_entries(const std::string &path) {
    std::map<std::string, int64_t> entries;
    gguf_init_params no_data = {true, nullptr};
    gguf_context *g = gguf_init_from_file(path.c_str(), no_data);
    EXPECT_NE(g, nullptr);
    if (g != nullptr) {
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            entries[gguf_get_tensor_name(g, i)] = static_cast<int64_t>(gguf_get_tensor_size(g, i) / sizeof(float));
        }
        gguf_free(g);
    }
    return entries;
}

} // namespace

TEST(CApiImatrix, AComputedImportanceMatrixCoversEveryLayerMatrixAndQuantizesTheTypesThatNeedOne) {
    tiny_model_spec spec = tiny_model_spec::wide(256, "jllama-imatrix-model");
    spec.n_layer = 1;
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text = write_calibration_text("jllama-imatrix", 200);
    const std::string imatrix = model + ".imatrix.gguf";

    int chunks = -1;
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), imatrix.c_str(), 32, 0, 2, 0, nullptr, nullptr, &chunks),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(chunks, 6) << "200 tokens and a BOS in chunks of 32";

    // one .in_sum2 with a value per input column and one .counts per weight matrix of the layer
    const auto entries = imatrix_entries(imatrix);
    for (const char *weight : {"attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"}) {
        const std::string name = std::string("blk.0.") + weight + ".weight";
        ASSERT_EQ(entries.count(name + ".in_sum2"), 1u) << name;
        EXPECT_EQ(entries.at(name + ".in_sum2"), spec.n_embd) << name;
        EXPECT_EQ(entries.at(name + ".counts"), 1) << name;
    }
    ASSERT_EQ(entries.count("blk.0.ffn_down.weight.in_sum2"), 1u);
    EXPECT_EQ(entries.at("blk.0.ffn_down.weight.in_sum2"), spec.n_ff);
    EXPECT_EQ(entries.count("output.weight.in_sum2"), 0u) << "the output layer is not part of the matrix";

    // progress: one call per chunk on this thread with the fraction done, the last one 1
    std::vector<float> fractions;
    const auto record = [](float fraction, void *user_data) {
        static_cast<std::vector<float> *>(user_data)->push_back(fraction);
        return true;
    };
    const std::string reported = model + ".imatrix-progress.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), reported.c_str(), 32, 0, 2, 0, record, &fractions, &chunks),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(fractions.size(), 6u);
    EXPECT_FLOAT_EQ(fractions.front(), 1.0f / 6.0f);
    EXPECT_FLOAT_EQ(fractions.back(), 1.0f);
    // the same input gives the same file, with or without somebody watching
    const auto file_bytes = [](const std::string &file) {
        std::ifstream in(file, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    EXPECT_EQ(file_bytes(reported), file_bytes(imatrix));

    // and a callback that says no cancels: an error naming it, and no file
    int calls = 0;
    const auto cancel = [](float, void *user_data) { return ++*static_cast<int *>(user_data) < 2; };
    const std::string cancelled = model + ".imatrix-cancelled.gguf";
    EXPECT_EQ(jllama_imatrix(model.c_str(), text.c_str(), cancelled.c_str(), 32, 0, 2, 0, cancel, &calls, nullptr),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("cancelled after chunk 2 of 6"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(calls, 2);
    std::error_code no_file;
    EXPECT_FALSE(std::filesystem::exists(cancelled, no_file));
    std::filesystem::remove(reported, no_file);

    // max_chunks limits the work, and the file says how many chunks it holds
    const std::string limited = model + ".imatrix-2.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), limited.c_str(), 32, 2, 2, 0, nullptr, nullptr, &chunks),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(chunks, 2);

    // what it is for: the types that cannot be written without a matrix, written and loaded
    // (Q2_K_S and IQ1_M: the two whose tables are cheap to build; the sweep above writes all nine)
    for (const int ftype : {21, 31}) {
        SCOPED_TRACE("file type " + std::to_string(ftype));
        const std::string out = model + ".im-ftype" + std::to_string(ftype) + ".gguf";
        ASSERT_EQ(jllama_quantize_imatrix(model.c_str(), out.c_str(), ftype, 1, false, imatrix.c_str()),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model *loaded = load_tiny(out, {});
        EXPECT_NE(loaded, nullptr) << jllama_last_error();
        if (loaded != nullptr) {
            EXPECT_FALSE(greedy_answer(loaded, 4).empty());
            jllama_model_free(loaded);
        }
        std::error_code ec;
        std::filesystem::remove(out, ec);
    }
    std::error_code ec;
    for (const std::string &file : {model, text, imatrix, limited}) {
        std::filesystem::remove(file, ec);
    }
}

TEST(CApiImatrix, AMixtureOfExpertsGetsACountPerExpert) {
    const tiny_model_spec spec = moe_spec();
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text = write_calibration_text("jllama-imatrix-moe", 200);
    const std::string imatrix = model + ".imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), imatrix.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const auto entries = imatrix_entries(imatrix);
    for (int layer = 0; layer < spec.n_layer; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        ASSERT_EQ(entries.count(p + "ffn_up_exps.weight.in_sum2"), 1u);
        EXPECT_EQ(entries.at(p + "ffn_up_exps.weight.in_sum2"), int64_t{spec.n_embd} * spec.n_expert);
        EXPECT_EQ(entries.at(p + "ffn_up_exps.weight.counts"), spec.n_expert);
        EXPECT_EQ(entries.at(p + "ffn_down_exps.weight.in_sum2"), int64_t{spec.n_ff} * spec.n_expert);
    }
    // and the quantizer takes it: a type that needs a matrix, over the expert tensors
    const std::string out = model + ".iq2.gguf";
    ASSERT_EQ(jllama_quantize_imatrix(model.c_str(), out.c_str(), 20, 1, false, imatrix.c_str()), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_model *loaded = load_tiny(out, {});
    EXPECT_NE(loaded, nullptr) << jllama_last_error();
    jllama_model_free(loaded);
    std::error_code ec;
    for (const std::string &file : {model, text, imatrix, out}) {
        std::filesystem::remove(file, ec);
    }
}

namespace {

// The values of every tensor of an importance matrix file, and its chunk count and datasets.
struct imatrix_file {
    std::map<std::string, std::vector<float>> tensors;
    uint32_t chunk_count = 0;
    uint32_t chunk_size = 0;
    size_t n_datasets = 0;
};

imatrix_file read_imatrix(const std::string &path) {
    imatrix_file file;
    ggml_context *ctx = nullptr;
    gguf_init_params with_data = {false, &ctx};
    gguf_context *g = gguf_init_from_file(path.c_str(), with_data);
    EXPECT_NE(g, nullptr) << path;
    if (g == nullptr) {
        return file;
    }
    // by the file's tensor list: the context also holds one tensor with the whole data section
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const ggml_tensor *t = ggml_get_tensor(ctx, gguf_get_tensor_name(g, i));
        const float *data = static_cast<const float *>(t->data);
        file.tensors[t->name].assign(data, data + ggml_nelements(t));
    }
    file.chunk_count = gguf_get_val_u32(g, gguf_find_key(g, "imatrix.chunk_count"));
    file.chunk_size = gguf_get_val_u32(g, gguf_find_key(g, "imatrix.chunk_size"));
    file.n_datasets = gguf_get_arr_n(g, gguf_find_key(g, "imatrix.datasets"));
    gguf_free(g);
    ggml_free(ctx);
    return file;
}

} // namespace

TEST(CApiImatrix, MatricesOfOneModelAddUpAndTheLegacyFormatRoundTrips) {
    tiny_model_spec spec = tiny_model_spec::wide(256, "jllama-imatrix-merge");
    spec.n_layer = 1;
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text_a = write_calibration_text("jllama-imatrix-merge-a", 200);
    const std::string text_b = write_calibration_text("jllama-imatrix-merge-b", 400);
    const std::string a = model + ".a.imatrix.gguf";
    const std::string b = model + ".b.imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text_a.c_str(), a.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(jllama_imatrix(model.c_str(), text_b.c_str(), b.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();

    // two matrices: every sum and every count is the sum of the two, the chunks and datasets add up
    const std::string merged = model + ".merged.imatrix.gguf";
    const char *both[] = {a.c_str(), b.c_str()};
    int chunks = -1;
    ASSERT_EQ(jllama_imatrix_merge(both, 2, merged.c_str(), false, &chunks), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(chunks, 18);
    const imatrix_file fa = read_imatrix(a);
    const imatrix_file fb = read_imatrix(b);
    const imatrix_file fm = read_imatrix(merged);
    EXPECT_EQ(fm.chunk_count, 18u);
    EXPECT_EQ(fm.chunk_size, 32u);
    EXPECT_EQ(fm.n_datasets, 2u);
    ASSERT_EQ(fm.tensors.size(), fa.tensors.size());
    ASSERT_FALSE(fm.tensors.empty());
    for (const auto &[name, values] : fm.tensors) {
        const std::vector<float> &va = fa.tensors.at(name);
        const std::vector<float> &vb = fb.tensors.at(name);
        ASSERT_EQ(values.size(), va.size()) << name;
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_FLOAT_EQ(values[i], va[i] + vb[i]) << name << "[" << i << "]";
        }
    }

    // one matrix into the legacy format and back: no GGUF in between, the same mean per column after
    const std::string legacy = model + ".a.imatrix.dat";
    const char *only_a[] = {a.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_a, 1, legacy.c_str(), true, &chunks), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(chunks, 6);
    {
        std::ifstream in(legacy, std::ios::binary);
        char magic[4] = {};
        in.read(magic, 4);
        EXPECT_NE(std::string(magic, 4), "GGUF");
    }
    const std::string back = model + ".a.back.imatrix.gguf";
    const char *only_legacy[] = {legacy.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_legacy, 1, back.c_str(), false, &chunks), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(chunks, 6);
    const imatrix_file fback = read_imatrix(back);
    for (const char *weight : {"attn_q", "ffn_down"}) {
        const std::string name = std::string("blk.0.") + weight + ".weight";
        const std::vector<float> &sums = fa.tensors.at(name + ".in_sum2");
        const std::vector<float> &sums_back = fback.tensors.at(name + ".in_sum2");
        const float count = fa.tensors.at(name + ".counts").at(0);
        const float count_back = fback.tensors.at(name + ".counts").at(0);
        ASSERT_EQ(sums.size(), sums_back.size());
        ASSERT_GT(count, 0.0f);
        ASSERT_GT(count_back, 0.0f);
        for (size_t i = 0; i < sums.size(); ++i) {
            EXPECT_NEAR(sums_back[i] / count_back, sums[i] / count, 1e-4f * std::abs(sums[i] / count) + 1e-9f)
                << name << "[" << i << "]";
        }
    }
    // the quantizer reads the legacy file as it reads the GGUF one
    const std::string quantized = model + ".legacy.iq2.gguf";
    ASSERT_EQ(jllama_quantize_imatrix(model.c_str(), quantized.c_str(), 20, 1, false, legacy.c_str()), JLLAMA_STATUS_OK)
        << jllama_last_error();
    // a legacy file next to a GGUF one takes the other's chunk size
    const std::string mixed = model + ".mixed.imatrix.gguf";
    const char *legacy_and_b[] = {legacy.c_str(), b.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(legacy_and_b, 2, mixed.c_str(), false, &chunks), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(chunks, 18);
    EXPECT_EQ(read_imatrix(mixed).chunk_size, 32u);

    // what does not fit is refused by name, with nothing written
    const std::string refused = model + ".refused.imatrix.gguf";
    EXPECT_EQ(jllama_imatrix_merge(nullptr, 0, refused.c_str(), false, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(jllama_imatrix_merge(both, 2, nullptr, false, nullptr), JLLAMA_STATUS_INVALID_ARGUMENT);
    const std::string missing = model + ".missing.imatrix.gguf";
    const char *with_missing[] = {a.c_str(), missing.c_str()};
    EXPECT_EQ(jllama_imatrix_merge(with_missing, 2, refused.c_str(), false, nullptr), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find(missing), std::string::npos) << jllama_last_error();
    const std::string other_size = model + ".c.imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text_b.c_str(), other_size.c_str(), 64, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const char *two_sizes[] = {a.c_str(), other_size.c_str()};
    EXPECT_EQ(jllama_imatrix_merge(two_sizes, 2, refused.c_str(), false, nullptr), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("different chunk sizes"), std::string::npos) << jllama_last_error();
    tiny_model_spec narrow = tiny_model_spec::wide(128, "jllama-imatrix-merge-other");
    narrow.n_layer = 1;
    const std::string other_model = write_tiny_llama_gguf(narrow);
    const std::string other = other_model + ".imatrix.gguf";
    ASSERT_EQ(
        jllama_imatrix(other_model.c_str(), text_a.c_str(), other.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
        JLLAMA_STATUS_OK)
        << jllama_last_error();
    const char *two_models[] = {a.c_str(), other.c_str()};
    EXPECT_EQ(jllama_imatrix_merge(two_models, 2, refused.c_str(), false, nullptr), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("not of the same model"), std::string::npos) << jllama_last_error();
    EXPECT_FALSE(std::filesystem::exists(refused));

    std::error_code ec;
    for (const std::string &file :
         {model, text_a, text_b, a, b, merged, legacy, back, quantized, mixed, other_size, other_model, other}) {
        std::filesystem::remove(file, ec);
    }
}

TEST(CApiImatrix, WhatCannotBeComputedIsRefusedByName) {
    const std::string model = write_tiny_llama_gguf("jllama-imatrix-refused");
    const std::string text = write_calibration_text("jllama-imatrix-refused", 40);
    const std::string out = model + ".imatrix.gguf";
    const auto refused = [&](const char *model_path, const char *text_path, int chunk_size, const char *what) {
        EXPECT_EQ(jllama_imatrix(model_path, text_path, out.c_str(), chunk_size, 0, 1, 0, nullptr, nullptr, nullptr),
                  JLLAMA_STATUS_ERROR)
            << what;
        EXPECT_NE(std::string(jllama_last_error()).find(what), std::string::npos) << jllama_last_error();
        std::error_code ec;
        EXPECT_FALSE(std::filesystem::exists(out, ec)) << what;
    };
    refused("/nonexistent/model.gguf", text.c_str(), 32, "could not load the model");
    refused(model.c_str(), "/nonexistent/text.txt", 32, "calibration text");
    refused(model.c_str(), text.c_str(), 8, "at least 16");
    refused(model.c_str(), text.c_str(), 64, "fewer than one chunk");
    EXPECT_EQ(jllama_imatrix(nullptr, text.c_str(), out.c_str(), 32, 0, 1, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    std::error_code ec;
    std::filesystem::remove(model, ec);
    std::filesystem::remove(text, ec);
}

// ---------------------------------------------------------------------------------------------------
// Multi-token prediction: tiny_model_spec::mtp() is the smallest architecture llama.cpp has an MTP
// graph for (hy_v3, dense), a second decoder architecture and the model --spec-type draft-mtp needs.
// ---------------------------------------------------------------------------------------------------

TEST(CApiMtp, AModelWithAnMtpBlockDraftsWithItAndAnswersTheSameWithout) {
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::mtp());
    ASSERT_FALSE(path.empty());
    std::map<std::string, std::string> answers;
    std::map<std::string, int> drafted;
    for (const char *type : {"", "none", "draft-mtp"}) {
        SCOPED_TRACE(std::string("--spec-type ") + type);
        std::vector<std::string> extra;
        if (*type != 0) {
            extra = {"--spec-type", type};
        }
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        answers[type] = greedy_answer(model, 48);
        drafted[type] = drafted_tokens(model);
        jllama_model_free(model);
    }
    EXPECT_FALSE(answers[""].empty());
    // drafts are checked by the model itself, so they change the speed and never the answer
    EXPECT_EQ(answers["draft-mtp"], answers[""]);
    EXPECT_EQ(answers["none"], answers[""]);
    EXPECT_GT(drafted["draft-mtp"], 0) << "the MTP block drafted nothing";
    EXPECT_EQ(drafted["none"], 0);
    // an MTP block alone is not used: like a draft model, upstream wants the type named
    EXPECT_EQ(drafted[""], 0);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiMtp, ASlotOfAModelDraftingWithItsMtpBlockIsSavedAndRestoredWhole) {
    // llama.cpp b12091 (#30141): a slot file carries the state of the speculative decoding next to
    // the prompt -- for MTP the hidden state the block drafts from. The round trip has to give back
    // what was written, and the model has to go on drafting and answering as before.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::mtp("jllama-mtp-slot"));
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-mtp-slots-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    std::map<std::string, size_t> written;
    for (const char *type : {"none", "draft-mtp"}) {
        SCOPED_TRACE(std::string("--spec-type ") + type);
        jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string(), "--spec-type", type});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        const std::string before = greedy_answer(model, 48);
        EXPECT_FALSE(before.empty());

        const std::string name = std::string(type) + ".bin";
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json saved_json = json::parse(saved.str());
        std::error_code ec;
        written[type] = std::filesystem::file_size(dir / name, ec);
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_EQ(written[type], saved_json.at("n_written").get<size_t>()) << saved.str();

        Owned restored;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, name.c_str(), &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json restored_json = json::parse(restored.str());
        EXPECT_EQ(restored_json.at("n_restored").get<int>(), saved_json.at("n_saved").get<int>()) << restored.str();
        EXPECT_EQ(restored_json.at("n_read").get<size_t>(), written[type]) << restored.str();
        // the restored slot answers the same request the same way
        EXPECT_EQ(greedy_answer(model, 48), before);
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

// ---------------------------------------------------------------------------------------------------
// The other vocabularies (tiny_model_spec::tokenizer): a BPE whose merges apply, and SentencePiece.
// ---------------------------------------------------------------------------------------------------

namespace {

std::vector<int> token_ids(jllama_model *model, const std::string &text, bool add_special = false) {
    int *tokens = nullptr;
    size_t count = 0;
    EXPECT_EQ(jllama_tokenize(model, text.c_str(), add_special, &tokens, &count), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::vector<int> ids(tokens, tokens + count);
    jllama_tokens_free(tokens);
    return ids;
}

std::string detokenized(jllama_model *model, const std::vector<int> &ids) {
    Owned text;
    size_t length = 0;
    EXPECT_EQ(jllama_detokenize(model, ids.data(), ids.size(), &text.s, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    return text.s != nullptr ? std::string(text.s, length) : std::string();
}

} // namespace

TEST(CApiTokenizers, ABpeVocabularyAppliesItsMergesAndTheByteVocabularyHasNone) {
    tiny_model_spec merged;
    merged.prefix = "jllama-bpe-merges";
    merged.tokenizer = tiny_model_spec::tokenizer_kind::BPE_MERGES;
    const std::string merged_path = write_tiny_llama_gguf(merged);
    const std::string bytes_path = write_tiny_llama_gguf("jllama-bpe-bytes");
    jllama_model *with_merges = load_tiny(merged_path, {});
    jllama_model *bytes = load_tiny(bytes_path, {});
    ASSERT_NE(with_merges, nullptr) << jllama_last_error();
    ASSERT_NE(bytes, nullptr) << jllama_last_error();

    const std::string text = "hello world";
    // "hello" and " world" are the fourth and the ninth merged token after the 256 bytes
    EXPECT_EQ(token_ids(with_merges, text), (std::vector<int>{256 + 3, 256 + 8}));
    EXPECT_EQ(token_ids(bytes, text).size(), text.size());
    // a word the merges only partly cover: "hell" merges, the rest stays bytes
    EXPECT_EQ(token_ids(with_merges, "hells"), (std::vector<int>{256 + 2, 's'}));
    for (jllama_model *model : {with_merges, bytes}) {
        for (const std::string &round_trip : {text, std::string("hells"), std::string("Hello, World!")}) {
            EXPECT_EQ(detokenized(model, token_ids(model, round_trip)), round_trip);
        }
        // the model runs over its vocabulary: a merged token has an embedding row of its own
        EXPECT_FALSE(greedy_answer(model, 4).empty());
    }
    jllama_model_free(with_merges);
    jllama_model_free(bytes);
    std::error_code ec;
    std::filesystem::remove(merged_path, ec);
    std::filesystem::remove(bytes_path, ec);
}

TEST(CApiTokenizers, ASentencePieceVocabularyTokenizesWordsAndRoundTrips) {
    tiny_model_spec spec;
    spec.prefix = "jllama-spm";
    spec.tokenizer = tiny_model_spec::tokenizer_kind::SENTENCEPIECE;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    // <unk> <s> </s>, the space piece, 94 printable characters, then the pieces of the two words
    const int hello = 3 + 1 + 94 + 4;
    const int world = hello + 5;
    EXPECT_EQ(token_ids(model, "hello world"), (std::vector<int>{hello, world}));
    // with the special tokens the text starts with <s>
    const std::vector<int> with_special = token_ids(model, "hello world", true);
    ASSERT_FALSE(with_special.empty());
    EXPECT_EQ(with_special.front(), 1);
    // a word the vocabulary has no piece for falls apart into the space piece and characters
    EXPECT_EQ(token_ids(model, "cat"), (std::vector<int>{3, 4 + ('c' - '!'), 4 + ('a' - '!'), 4 + ('t' - '!')}));
    // The round trip is the text behind a space: SentencePiece puts its space piece in front of the
    // first word, and llama.cpp takes it off again only behind a <s>, which tokens without the
    // special ones do not have. A byte or BPE vocabulary gives the text back as it was (above).
    for (const std::string &round_trip : {std::string("hello world"), std::string("cat"), std::string("hello, cat!")}) {
        EXPECT_EQ(detokenized(model, token_ids(model, round_trip)), " " + round_trip);
    }

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json m = json::parse(meta.str());
    EXPECT_EQ(m.at("n_vocab").get<int>(), 3 + 1 + 94 + 10);
    EXPECT_EQ(m.at("vocab_type").get<int>(), 1) << "LLAMA_VOCAB_TYPE_SPM";

    Owned completion;
    EXPECT_EQ(jllama_completions(model, R"({"prompt":"hello world","temperature":0,"n_predict":4})", &completion.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiTokenizers, ASentencePieceVocabularyWithBytePiecesSpellsOutWhatItHasNoPieceFor) {
    tiny_model_spec plain;
    plain.prefix = "jllama-spm-plain";
    plain.tokenizer = tiny_model_spec::tokenizer_kind::SENTENCEPIECE;
    tiny_model_spec bytes;
    bytes.prefix = "jllama-spm-bytes";
    bytes.tokenizer = tiny_model_spec::tokenizer_kind::SENTENCEPIECE_BYTES;
    const std::string plain_path = write_tiny_llama_gguf(plain);
    const std::string bytes_path = write_tiny_llama_gguf(bytes);
    jllama_model *without = load_tiny(plain_path, {});
    jllama_model *with = load_tiny(bytes_path, {});
    ASSERT_NE(without, nullptr) << jllama_last_error();
    ASSERT_NE(with, nullptr) << jllama_last_error();

    // <unk> <s> </s>, the space piece, 94 characters, 10 word pieces, then <0x00> .. <0xFF>
    const int space = 3;
    const int first_byte = 3 + 1 + 94 + 10;
    const std::string e_acute = "\xC3\xA9"; // U+00E9, no piece of its own
    // Without byte pieces llama.cpp has nothing to fall back to: the tokenization fails (its lookup of
    // the byte piece throws, and the message is the map's, "unordered_map::at"). An error, not <unk>
    // and not a crash; a converted SentencePiece model always has the byte pieces.
    {
        int *tokens = nullptr;
        size_t count = 0;
        EXPECT_NE(jllama_tokenize(without, e_acute.c_str(), false, &tokens, &count), JLLAMA_STATUS_OK);
        EXPECT_EQ(tokens, nullptr);
        EXPECT_FALSE(std::string(jllama_last_error()).empty());
    }
    // with them it is its two UTF-8 bytes, and comes back
    EXPECT_EQ(token_ids(with, e_acute), (std::vector<int>{space, first_byte + 0xC3, first_byte + 0xA9}));
    EXPECT_EQ(detokenized(with, token_ids(with, e_acute)), " " + e_acute);
    // a known word is still its piece
    EXPECT_EQ(token_ids(with, "hello"), (std::vector<int>{3 + 1 + 94 + 4}));

    jllama_model_free(without);
    jllama_model_free(with);
    std::error_code ec;
    std::filesystem::remove(plain_path, ec);
    std::filesystem::remove(bytes_path, ec);
}

TEST(CApiTokenizers, AUnigramVocabularyTakesTheBestScoringPiecesAndEndsWithItsEos) {
    tiny_model_spec spec;
    spec.prefix = "jllama-unigram";
    spec.tokenizer = tiny_model_spec::tokenizer_kind::UNIGRAM;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    // <pad> </s> <unk>, the space piece, 94 characters, then the pieces of the two words
    const int hello = 3 + 1 + 94 + 4;
    const int world = hello + 5;
    EXPECT_EQ(token_ids(model, "hello world"), (std::vector<int>{hello, world}));
    // T5 has no BOS; with the special tokens the text ends with </s>
    EXPECT_EQ(token_ids(model, "hello world", true), (std::vector<int>{hello, world, 1}));
    // a word without a piece is the space piece and its characters
    EXPECT_EQ(token_ids(model, "cat"), (std::vector<int>{3, 4 + ('c' - '!'), 4 + ('a' - '!'), 4 + ('t' - '!')}));
    // a character outside the vocabulary is <unk>
    const std::vector<int> unknown = token_ids(model, "\xC3\xA9");
    EXPECT_NE(std::find(unknown.begin(), unknown.end(), 2), unknown.end());
    // the round trip comes back behind the space the prefix put in front, as for SentencePiece
    EXPECT_EQ(detokenized(model, token_ids(model, "hello world")), " hello world");

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("vocab_type").get<int>(), 4) << "LLAMA_VOCAB_TYPE_UGM";
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiTokenizers, AnRwkvVocabularyMatchesTheLongestTokenGreedily) {
    tiny_model_spec spec;
    spec.prefix = "jllama-rwkv-vocab";
    spec.tokenizer = tiny_model_spec::tokenizer_kind::RWKV;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    // 256 byte tokens, then "hello" and " world"
    EXPECT_EQ(token_ids(model, "hello world"), (std::vector<int>{256, 257}));
    // the longest match wins and what is left is bytes: "hell" has no token, "o" is one byte
    EXPECT_EQ(token_ids(model, "hell"), (std::vector<int>{'h', 'e', 'l', 'l'}));
    EXPECT_EQ(token_ids(model, "helloo"), (std::vector<int>{256, 'o'}));
    // escaped tokens are the bytes they stand for: a backslash, a newline, a byte above 0x7F
    EXPECT_EQ(token_ids(model, "\\\n\xC3\xA9"), (std::vector<int>{'\\', '\n', 0xC3, 0xA9}));
    for (const std::string &text : {std::string("hello world"), std::string("a\\b\nc"), std::string("\xC3\xA9")}) {
        EXPECT_EQ(detokenized(model, token_ids(model, text)), text);
    }

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("vocab_type").get<int>(), 5) << "LLAMA_VOCAB_TYPE_RWKV";
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------------------------------
// Structured output on the chat path. A random model writes noise, so an answer that is exactly one
// of two JSON strings is the grammar's doing -- which makes these checks of what constrains what.
// ---------------------------------------------------------------------------------------------------

namespace {

// The content of a chat completion over the fixture model; empty when the call failed.
std::string chat_content(jllama_model *model, const std::string &extra_fields) {
    Owned response;
    const std::string request =
        R"({"messages":[{"role":"user","content":"Answer."}],"temperature":0,"max_tokens":24,"cache_prompt":false)" +
        extra_fields + "}";
    EXPECT_EQ(jllama_chat_completions(model, request.c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    if (response.s == nullptr) {
        return std::string();
    }
    const json message = json::parse(response.str()).at("choices").at(0).at("message");
    return message.contains("content") && !message.at("content").is_null() ? message.at("content").get<std::string>()
                                                                           : std::string();
}

bool is_yes_or_no(const std::string &content) { return content == "\"yes\"" || content == "\"no\""; }

const char *const YES_NO_NESTED =
    R"(,"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"enum":["yes","no"]}}})";
const char *const YES_NO_TOP_LEVEL = R"(,"response_format":{"type":"json_schema","schema":{"enum":["yes","no"]}})";
const char *const ONE_TOOL =
    R"(,"tools":[{"type":"function","function":{"name":"lookup","description":"Look something up.",)"
    R"("parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}}])";

} // namespace

// The model of these tests ends its answers with a byte of its own (4). The shared fixture's end
// token is '>', which the chat template's generation prompt ("<|im_start|>assistant") contains, and
// llama.cpp feeds that prompt to the grammar first: an end token inside an unfinished grammar is
// GGML_ABORT("fatal error") in llama_grammar_accept_impl, the whole process. No real model has an
// end token that the text of its own template tokenizes to, so this is the fixture's trap.
class CApiStructured : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        tiny_model_spec spec;
        spec.prefix = "jllama-structured";
        spec.eos = 4;
        spec.eos_is_control = true; // or the end byte is part of the answer, which is then no JSON
        path_ = write_tiny_llama_gguf(spec);
        ASSERT_FALSE(path_.empty());
        model_ = load_tiny(path_, {});
        ASSERT_NE(model_, nullptr) << jllama_last_error();
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
std::string CApiStructured::path_;
jllama_model *CApiStructured::model_ = nullptr;

TEST_F(CApiStructured, AResponseFormatConstrainsTheAnswerInBothSpellingsOfTheSchema) {
    // unconstrained, the random model does not happen to write one of the two strings
    EXPECT_FALSE(is_yes_or_no(chat_content(model_, "")));
    EXPECT_TRUE(is_yes_or_no(chat_content(model_, YES_NO_NESTED)));
    // the schema directly under response_format, which llama.cpp reads since b12229
    EXPECT_TRUE(is_yes_or_no(chat_content(model_, YES_NO_TOP_LEVEL)));
}

TEST_F(CApiStructured, AResponseFormatNextToToolsStillConstrainsTheWholeAnswer) {
    // llama.cpp b12242 (#31956): with tools the grammar used to be lazy -- in force only once the
    // model began a tool call -- so a request with both got free text. The answer is held to the
    // schema now whether or not tools are offered.
    const std::string content = chat_content(model_, std::string(YES_NO_NESTED) + ONE_TOOL);
    EXPECT_TRUE(is_yes_or_no(content)) << content;
}
