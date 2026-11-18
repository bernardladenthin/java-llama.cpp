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
#include "tts_params.hpp"
#include "tts_wav.hpp"

#ifdef JLLAMA_HAS_NATIVE_SERVER
#include <cpp-httplib/httplib.h>
#endif

#include "build-info.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#ifndef NOMINMAX // or windows.h's min/max macros break std::min / std::max below
#define NOMINMAX
#endif
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

TEST_F(CApi, AGrammarTriggerOfAnUnknownTypeIsTheCallersFault) {
    // Up to b12387 common_sampler_init() asserted on a trigger type it did not know, which ended the
    // process -- for this library the JVM. Upstream #31478 (b12408) throws instead, and the server
    // reports it as an invalid request when the slot starts.
    const json request = {
        {"prompt", "Hi"},       {"n_predict", 2},
        {"temperature", 0},     {"grammar", "root ::= \"a\""},
        {"grammar_lazy", true}, {"grammar_triggers", json::array({json{{"type", 99}, {"value", "a"}}})}};
    Owned response;
    EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("grammar trigger type"), std::string::npos) << jllama_last_error();
    // The model is unharmed and answers the next request.
    ASSERT_EQ(jllama_completions(model_, R"({"prompt":"Hi","n_predict":2,"temperature":0})", &response.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
}

TEST_F(CApi, IgnoreEosNextToAGrammarIsTheCallersFault) {
    // Since b12489 (upstream #32377) a grammar next to a ban of the end-of-generation tokens -- which is
    // what ignore_eos is, a logit bias of -inf on every EOG token -- is refused when the sampler starts:
    // the grammar could reach a state where only the end may follow, and the sampler crashed there. The
    // server answers it as an invalid request, so this library throws InvalidRequestException.
    const auto refused = [this](const json &request) {
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST)
            << request.dump();
        EXPECT_NE(std::string(jllama_last_error()).find("EOG tokens cannot be banned"), std::string::npos)
            << jllama_last_error();
    };
    const json base = {{"prompt", "Hi"}, {"n_predict", 2}, {"temperature", 0}};
    json grammar = base;
    grammar["grammar"] = "root ::= \"a\"";
    grammar["ignore_eos"] = true;
    refused(grammar);
    json schema = base;
    schema["json_schema"] = {{"type", "object"}, {"properties", {{"a", {{"type", "string"}}}}}};
    schema["ignore_eos"] = true;
    refused(schema);
    // the same ban written out as a logit bias: false is -inf
    json bias = base;
    bias["grammar"] = "root ::= \"a\"";
    bias["logit_bias"] = json::array({json::array({">", false})});
    refused(bias);
    // each half alone still answers
    Owned response;
    json only_grammar = base;
    only_grammar["grammar"] = "root ::= \"a\"";
    EXPECT_EQ(jllama_completions(model_, only_grammar.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    json only_ignore = base;
    only_ignore["ignore_eos"] = true;
    EXPECT_EQ(jllama_completions(model_, only_ignore.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    // and so does a negative bias that is not -inf
    json soft = bias;
    soft["logit_bias"] = json::array({json::array({">", -5.0})});
    EXPECT_EQ(jllama_completions(model_, soft.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApi, AModelLoadedWithIgnoreEosAndAGrammarIsRefused) {
    // The same combination as server defaults fails the argument parse (b12489, common/arg.cpp); either
    // one alone loads.
    for (const char *constraint : {"--grammar", "--json-schema"}) {
        const std::string value = std::string(constraint) == "--grammar" ? "root ::= \"a\"" : R"({"type":"object"})";
        const std::vector<std::string> args = {"jllama", "-m", path_,         "-c",       "128", "-ngl",        "0",
                                               "-t",     "2",  "--no-warmup", constraint, value, "--ignore-eos"};
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        jllama_model *model = nullptr;
        EXPECT_NE(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
                  JLLAMA_STATUS_OK)
            << constraint;
        EXPECT_EQ(model, nullptr);
        // the refusal carries upstream's reason (patches/0031; before it, stderr had it alone)
        EXPECT_NE(std::string(jllama_last_error()).find("Failed to parse model parameters"), std::string::npos)
            << jllama_last_error();
        EXPECT_NE(std::string(jllama_last_error()).find("--ignore-eos is not supported with --grammar"),
                  std::string::npos)
            << jllama_last_error();
        jllama_model_free(model);
    }
}

TEST_F(CApi, AnEmptyStopStringIsIgnored) {
    // Upstream #32093 (b12416..b12431): an empty stop string matched at once, so a request with
    // "stop": "" or [""] ended before its first token. It is dropped now, and the answer is the one
    // without it.
    const json base = {
        {"prompt", "Hello"}, {"n_predict", 6}, {"temperature", 0}, {"ignore_eos", true}, {"cache_prompt", false}};
    Owned plain;
    ASSERT_EQ(jllama_completions(model_, base.dump().c_str(), &plain.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json expected = json::parse(plain.str());
    ASSERT_EQ(expected.at("stop_type").get<std::string>(), "limit") << plain.str();
    for (const json &stop : {json(""), json::array({""}), json::array({"", ""})}) {
        json request = base;
        request["stop"] = stop;
        Owned response;
        ASSERT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(response.str());
        EXPECT_EQ(j.at("stop_type").get<std::string>(), "limit") << stop.dump();
        EXPECT_EQ(j.at("tokens_predicted").get<int>(), 6) << stop.dump();
        EXPECT_EQ(j.at("content"), expected.at("content")) << stop.dump();
    }
}

TEST(CApiLoadModes, EveryLoadModeReadsTheSameWeights) {
    // A load mode decides how the file is read, never what is read. b12233..b12283 (#26014) rewrote
    // the direct-I/O reader on Windows (aligned chunks, the file pointer moved by hand): an offset off
    // by a chunk loads other bytes and answers without an error. The tensors of this GGUF are aligned
    // to 32 bytes, not to a sector, so the unaligned path is the one taken.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-load-modes");
    ASSERT_FALSE(path.empty());
    const json request = {{"prompt", "Hello"}, {"n_predict", 6},     {"temperature", 0},
                          {"n_probs", 3},      {"ignore_eos", true}, {"cache_prompt", false}};
    std::vector<json> answers;
    for (const char *mode : {"none", "mmap", "dio"}) {
        const std::vector<std::string> args = {"jllama", "-m",          path, "-c", "128",         "-ngl",
                                               "0",      "--no-warmup", "-t", "2",  "--load-mode", mode};
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        jllama_model *model = nullptr;
        ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
                  JLLAMA_STATUS_OK)
            << mode << ": " << jllama_last_error();
        Owned response;
        ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(response.str());
        answers.push_back({{"content", j.at("content")}, {"probs", j.at("completion_probabilities")}});
        jllama_model_free(model);
    }
    EXPECT_EQ(answers[1], answers[0]) << "mmap";
    EXPECT_EQ(answers[2], answers[0]) << "dio";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
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

TEST(CApiTemplate, AChatTemplateMayChainComparisonsAndUnpackArguments) {
    // Jinja since b12309 (upstream #32072): `a < b < c` is one chained comparison, and a call may
    // unpack a list or a dict into its arguments. A template that uses either failed to parse
    // before, which fails the model load.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-jinja");
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{%- macro tag(role, text, close=true) -%}"
                             "<{{ role }}>{{ text }}{{ '</' ~ role ~ '>' if close else '' }}"
                             "{%- endmacro -%}"
                             "{%- for message in messages -%}"
                             "{{ tag(*[message.role, message.content]) }}"
                             "{%- endfor -%}"
                             "[{{ 'few' if 0 < messages|length < 3 else 'many' }}]"
                             "{%- if add_generation_prompt -%}{{ tag('assistant', **{'text': '', 'close': false}) }}"
                             "{%- endif -%}";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();

    Owned one;
    EXPECT_EQ(jllama_apply_template(model, R"({"messages":[{"role":"user","content":"Hi"}]})", &one.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(one.str(), "<user>Hi</user>[few]<assistant>");

    Owned three;
    EXPECT_EQ(jllama_apply_template(model,
                                    R"({"messages":[{"role":"user","content":"a"},)"
                                    R"({"role":"assistant","content":"b"},{"role":"user","content":"c"}]})",
                                    &three.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(three.str(), "<user>a</user><assistant>b</assistant><user>c</user>[many]<assistant>");

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiTemplate, AChatTemplateMayRenderJsonWithSortedKeys) {
    // Jinja since b12440 (upstream #32366): tojson(sort_keys=true) orders the keys of every object, at
    // every level, as Python's json.dumps does; before, it threw "not implemented", which a template
    // rendering the tools that way turned into a failed load. Without the flag the order is the input's.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sortkeys");
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{%- for message in messages -%}"
                             "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
                             "{%- endfor -%}"
                             "{%- if tools is defined and tools -%}"
                             "[{{ tools[0] | tojson(sort_keys=true) }}|{{ tools[0] | tojson }}]"
                             "{%- endif -%}"
                             "{%- if add_generation_prompt -%}<assistant>{%- endif -%}";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();

    Owned rendered;
    EXPECT_EQ(jllama_apply_template(model,
                                    R"({"messages":[{"role":"user","content":"Hi"}],)"
                                    R"("tools":[{"type":"function","function":{"name":"f","description":"d",)"
                                    R"("parameters":{"type":"object","properties":{}}}}]})",
                                    &rendered.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const std::string text = rendered.str();
    const size_t open = text.find('[');
    const size_t bar = text.find('|', open);
    ASSERT_NE(open, std::string::npos) << text;
    ASSERT_NE(bar, std::string::npos) << text;
    const std::string sorted = text.substr(open + 1, bar - open - 1);
    const std::string unsorted = text.substr(bar + 1, text.find(']', bar) - bar - 1);
    // sorted: "function" before "type", and inside it "description" < "name" < "parameters"
    EXPECT_LT(sorted.find("\"function\""), sorted.find("\"type\": \"function\"")) << sorted;
    EXPECT_LT(sorted.find("\"description\""), sorted.find("\"name\"")) << sorted;
    EXPECT_LT(sorted.find("\"name\""), sorted.find("\"parameters\"")) << sorted;
    EXPECT_LT(sorted.find("\"properties\""), sorted.find("\"type\": \"object\"")) << sorted;
    // the same object without the flag keeps "type" first, so the flag is what sorted it
    EXPECT_LT(unsorted.find("\"type\""), unsorted.find("\"function\"")) << unsorted;
    EXPECT_NE(sorted, unsorted);

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiTemplate, AnIntegerOutsideItsRangeSaturatesInAChatTemplate) {
    // Jinja since b12480 (upstream #32397, float_to_int): a float made an integer, and an integer
    // product, sum or remainder computed through a double, saturate at the int64 limits and make NaN
    // zero. Before, the plain cast was undefined behaviour outside the range (on x86-64 it gave
    // INT64_MIN for both directions).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-jinja-int");
    ASSERT_FALSE(path.empty());
    const std::string tmpl =
        "{%- for message in messages -%}{{ message.content }}{%- endfor -%}"
        "[{{ 100000000000000000000000.0 | int }}|{{ -100000000000000000000000.0 | int }}|{{ 9223372036854775807 * 2 }}"
        "|{{ [9223372036854775807, 9223372036854775807] | sum }}|{{ 2.5 | int }}]";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned rendered;
    EXPECT_EQ(jllama_apply_template(model, R"({"messages":[{"role":"user","content":"x"}]})", &rendered.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(rendered.str(), "x[9223372036854775807|-9223372036854775808|9223372036854775807|9223372036854775807|2]");
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiTemplate, AChatTemplateMayUseTheJinjaOfB12431) {
    // Jinja since b12431 (upstream #32352, #32396): floor division, comparison of strings, a unary
    // minus after a keyword, sum with a start value, and a caller that outlives its call block. A
    // template that used any of them failed to parse or rendered something else, which fails the load.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-jinja-b12431");
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{%- set ns = namespace(f=none) -%}"
                             "{%- macro keep() -%}{%- set ns.f = caller -%}{%- endmacro -%}"
                             "{%- macro outer(p) -%}{%- call keep() -%}[{{ p }}]{%- endcall -%}{%- endmacro -%}"
                             "{%- for message in messages -%}{{ message.content }}{%- endfor -%}"
                             "|{{ 7 // 2 }},{{ -7 // 2 }},{{ 'VBD' > 'VB' }},{{ 1 if false else -1 }}"
                             ",{{ [1, 2]|sum(start=10) }}|{{ outer('a') }}{{ outer('b') }}{{ ns.f() }}";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned rendered;
    EXPECT_EQ(jllama_apply_template(model, R"({"messages":[{"role":"user","content":"x"}]})", &rendered.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(rendered.str(), "x|3,-4,True,-1,13|[b]") << "outer() renders nothing, the kept caller the last block";
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiTemplate, AReasoningEffortTheTemplateRejectsIsTheCallersFaultAndNamesTheLevels) {
    // Since b12431 (upstream #32332) the template probe keeps the reasoning_effort levels a template
    // raises on, and the chat parser refuses such a level with an invalid_argument naming the levels
    // it supports; a level the template renders still renders.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-effort");
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{%- if reasoning_effort is defined and reasoning_effort == 'max' -%}"
                             "{{ raise_exception('max is not supported') }}"
                             "{%- endif -%}"
                             "{%- for message in messages -%}"
                             "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
                             "{%- endfor -%}"
                             "[{{ reasoning_effort if reasoning_effort is defined else 'none' }}]"
                             "{%- if add_generation_prompt -%}<assistant>{%- endif -%}";
    const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                           tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();

    Owned response;
    EXPECT_EQ(jllama_chat_completions(model,
                                      R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"max",)"
                                      R"("max_tokens":2,"temperature":0})",
                                      &response.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find("supported levels"), std::string::npos) << error;
    EXPECT_NE(error.find("high"), std::string::npos) << error;
    EXPECT_EQ(error.find("max,"), std::string::npos) << error;

    Owned high;
    EXPECT_EQ(jllama_chat_completions(model,
                                      R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"high",)"
                                      R"("max_tokens":2,"temperature":0})",
                                      &high.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
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

TEST_F(CApi, AnEditedPromptTakesTheSharedPrefixFromTheCacheAndEvaluatesTheRest) {
    // A request whose prompt shares a prefix with the one the slot holds keeps that prefix and
    // removes the rest of the slot's sequence from the KV cache (llama_memory_seq_rm from the end of
    // the prefix on), then evaluates the tokens after it. That removal is a scan of every cell before
    // b12440 and a walk over the sequence's own positions since (#32380); either way exactly the
    // tokens after the prefix are evaluated, and the slot answers what a fresh slot answers.
    // Pinned to one slot, so the slot selection cannot pick another one.
    const std::string base = "The quick brown fox ";
    const std::string first = base + "jumps over the dog";
    const std::string second = base + "naps in the sun";
    const auto tokens_of = [](const std::string &text) {
        int *ids = nullptr;
        size_t count = 0;
        EXPECT_EQ(jllama_tokenize(model_, text.c_str(), true, &ids, &count), JLLAMA_STATUS_OK) << jllama_last_error();
        std::vector<int> out(ids, ids + count);
        jllama_tokens_free(ids);
        return out;
    };
    const std::vector<int> first_tokens = tokens_of(first);
    const std::vector<int> second_tokens = tokens_of(second);
    size_t shared = 0;
    while (shared < first_tokens.size() && shared < second_tokens.size() &&
           first_tokens[shared] == second_tokens[shared]) {
        ++shared;
    }
    ASSERT_GE(shared, base.size()) << "the two prompts share at least the bytes of the base";
    ASSERT_LT(shared, first_tokens.size());
    ASSERT_LT(shared, second_tokens.size());

    const auto complete = [](const std::string &prompt, bool cache) {
        json request = {{"prompt", prompt}, {"n_predict", 2}, {"temperature", 0}, {"id_slot", 0}};
        if (!cache) {
            request = without_prompt_cache(request);
        }
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return json::parse(response.str());
    };
    const auto expect_prefix_reused = [](const json &result, size_t cached, size_t prompt_tokens) {
        EXPECT_EQ(result.at("timings").at("cache_n").get<size_t>(), cached) << result.dump();
        EXPECT_EQ(result.at("timings").at("prompt_n").get<size_t>(), prompt_tokens - cached) << result.dump();
    };

    // A cold slot evaluates the whole prompt; the answer of the second prompt is the reference.
    const json reference = complete(second, false);
    expect_whole_prompt_evaluated(reference, static_cast<int>(second_tokens.size()));
    const json cold = complete(first, false);
    expect_whole_prompt_evaluated(cold, static_cast<int>(first_tokens.size()));
    // A request without the cache still leaves its prompt in the slot: the same prompt again takes
    // all of it but the last token, which is evaluated again for its logits.
    const json warm_first = complete(first, true);
    expect_prefix_reused(warm_first, first_tokens.size() - 1, first_tokens.size());

    // The slot holds the first prompt: the second takes the shared prefix and evaluates its tail.
    const json edited = complete(second, true);
    expect_prefix_reused(edited, shared, second_tokens.size());
    // And back: the tail of the second prompt goes, the first prompt's tail is evaluated again.
    const json back = complete(first, true);
    expect_prefix_reused(back, shared, first_tokens.size());
    // The second prompt again over the same prefix: what the slot answers does not depend on what
    // was removed before. Compared with the third request over the cached prefix, not the first
    // (F272): both evaluate the same tail in one batch after the same cached prefix.
    const json edited_again = complete(second, true);
    expect_prefix_reused(edited_again, shared, second_tokens.size());
    EXPECT_EQ(edited_again.at("content"), edited.at("content")) << edited.dump() << "\n" << edited_again.dump();
}

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
    // b12489 (#32377) refuses a grammar next to ignore_eos; the grammar decides where the answer ends
    with_schema.erase("ignore_eos");
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

TEST(CApiStatic, AnAllOfOverNumbersIsOneRangeAndAnEmptyOneIsRefused) {
    // Upstream #31834 (b12233..b12283): allOf over number or integer schemas is merged into one range.
    // Before, the merge took them for objects and the grammar asked for "{...}", which no number is.
    Owned grammar;
    ASSERT_EQ(jllama_json_schema_to_grammar(
                  R"({"allOf":[{"type":"integer","minimum":1},{"type":"integer","maximum":5}]})", &grammar.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(grammar.str().find("\"{\""), std::string::npos) << grammar.str();
    EXPECT_NE(grammar.str().find("[1-5]"), std::string::npos) << grammar.str();
    EXPECT_EQ(jllama_json_schema_to_grammar(
                  R"({"allOf":[{"type":"integer","minimum":6},{"type":"integer","maximum":5}]})", &grammar.s),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("empty numeric range"), std::string::npos) << jllama_last_error();
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

namespace {

// What a model answers greedily, with the probabilities of every step: two models that compute the
// same logits bit for bit answer the same here, and every other difference shows in the numbers.
json scored_answer(const std::string &path) {
    jllama_model *model = load_tiny(path, {});
    EXPECT_NE(model, nullptr) << jllama_last_error();
    if (model == nullptr) {
        return json();
    }
    const json request = {{"prompt", "Hello there"}, {"n_predict", 6},       {"temperature", 0}, {"n_probs", 3},
                          {"ignore_eos", true},      {"cache_prompt", false}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    json answer;
    if (response.s != nullptr) {
        const json j = json::parse(response.str());
        answer = {{"content", j.at("content")}, {"probs", j.at("completion_probabilities")}};
    }
    jllama_model_free(model);
    return answer;
}

// The same model written three ways: without scales, with a <matrix>.scale of 2 next to every weight
// matrix that has one, and with the weights doubled instead. A factor of 2 is exact in floating
// point, so (2W)x and 2(Wx) are the same bits: the scaled model must answer exactly like the folded
// one, and differently from the plain one -- else the scales were not read at all.
void expect_weight_scales_applied(tiny_model_spec spec) {
    const std::string prefix = spec.prefix;
    spec.prefix = prefix + "-plain";
    const std::string plain = write_tiny_llama_gguf(spec);
    spec.prefix = prefix + "-scaled";
    spec.weight_scale = 2.0f;
    const std::string scaled = write_tiny_llama_gguf(spec);
    spec.prefix = prefix + "-folded";
    spec.weight_scale_folded = true;
    const std::string folded = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(plain.empty());
    ASSERT_FALSE(scaled.empty());
    ASSERT_FALSE(folded.empty());

    const json with_scales = scored_answer(scaled);
    ASSERT_FALSE(with_scales.is_null());
    EXPECT_EQ(with_scales, scored_answer(folded)) << "a .scale tensor was not applied like the folded weights";
    EXPECT_NE(with_scales, scored_answer(plain)) << "the scales changed nothing";
    std::error_code ignored;
    for (const std::string &path : {plain, scaled, folded}) {
        std::filesystem::remove(path, ignored);
    }
}

} // namespace

TEST(CApiWeightScales, AScaleTensorIsTheSameModelAsTheScaledWeights) {
    // The per-tensor weight scales of NVFP4 and FP8 checkpoints: llama.cpp loads a <matrix>.scale for
    // every architecture in one generic pass and multiplies the product of its matrix by it -- but
    // only where an architecture's graph passes the scale on, which is decided per call site. The
    // plain decoder (attention, feed-forward, output), the decoder with experts (one scale per expert)
    // and the recurrent model (its in and out projections).
    tiny_model_spec dense;
    dense.prefix = "jllama-c-api-scales-dense";
    expect_weight_scales_applied(dense);
    tiny_model_spec moe;
    moe.prefix = "jllama-c-api-scales-moe";
    moe.n_expert = 4;
    moe.n_expert_used = 2;
    expect_weight_scales_applied(moe);
    expect_weight_scales_applied(tiny_model_spec::recurrent("jllama-c-api-scales-mamba"));
}

TEST(CApiWeightScales, QwenNextScalesEveryProjectionOfItsDeltaNet) {
    // Qwen3-Next's delta net multiplied its output projection without ssm_out.scale until b12493
    // (#32393), so an NVFP4 or FP8 model computed every recurrent layer with the wrong magnitude and
    // loaded without a word. The generator's qwen3next has a delta net and an attention layer, both
    // with experts and a shared expert, and every matrix of them carries a scale.
    expect_weight_scales_applied(tiny_model_spec::qwen3next("jllama-c-api-scales-qwen3next"));
}

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
    // there through an app manifest, a library inside java.exe is not. Two sites stood between a UTF-8
    // slot directory and a working round trip on a code page like 1252: the option's own check
    // (fs_is_directory, "not a directory" at load; patches/0024) and the draft/checkpoint appendix,
    // appended through std::ofstream over the narrow string ("incomplete draft state" at save;
    // patches/0025). So load, save and restore have to work on every code page, with and without a
    // draft model, and the appendix has to be in the same file.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-utf8");
    ASSERT_FALSE(path.empty());
    const std::string leaf = "jllama-slots-\xC3\xA4-\xE6\x97\xA5\xE6\x9C\xAC-" + std::to_string(std::random_device{}());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / std::filesystem::u8path(leaf);
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    // the directory as UTF-8, which is what the C API takes on every platform
    const auto dir_u8 = dir.u8string();
    const std::string dir_arg(dir_u8.begin(), dir_u8.end());

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
        EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
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

TEST(CApiSlotFiles, ASlotFileNameWithNonAsciiCharactersIsOneFileThatRestoresWhole) {
    // The name half of the test above, in an ASCII directory, where fs_is_directory is no obstacle.
    // The slot file went through ggml_fopen under its real name, and the draft appendix went through a
    // std::ofstream into a second file whose name was mis-decoded in the ANSI code page (measured on
    // code page 1252: 9348 bytes under the real name, 9216 under the mangled one). The save reported
    // the bytes of both files as n_written, the restore read only the first, without the draft state,
    // and both said ok (patches/0025). So: exactly one file, and the restore reads what the save wrote.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-utf8-name");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-ascii-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    const std::string dir_arg = dir.string(); // ASCII, the same in every encoding

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

        const std::string name = std::string("f-\xC3\xA4-\xE6\x97\xA5-") + (with_draft ? "draft.bin" : "plain.bin");
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json saved_json = json::parse(saved.str());
        std::vector<std::string> files;
        for (const auto &entry : std::filesystem::directory_iterator(dir)) {
            const auto u8 = entry.path().filename().u8string();
            files.emplace_back(u8.begin(), u8.end());
        }
        ASSERT_EQ(files.size(), 1u) << "the slot file and its appendix must be one file";
        EXPECT_EQ(files[0], name);
        std::error_code ec;
        EXPECT_EQ(std::filesystem::file_size(dir / std::filesystem::u8path(name), ec),
                  saved_json.at("n_written").get<size_t>())
            << saved.str();

        Owned restored;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, name.c_str(), &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json restored_json = json::parse(restored.str());
        EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
        EXPECT_EQ(restored_json.at("n_restored").get<int>(), saved_json.at("n_saved").get<int>()) << restored.str();
        jllama_model_free(model);
        std::filesystem::remove(dir / std::filesystem::u8path(name), ec);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

// Standard base64, for the media of a multimodal prompt.
std::string base64_of(const std::vector<uint8_t> &bytes) {
    static const char *const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t b0 = bytes[i];
        const uint32_t b1 = i + 1 < bytes.size() ? bytes[i + 1] : 0;
        const uint32_t b2 = i + 2 < bytes.size() ? bytes[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += i + 1 < bytes.size() ? alphabet[(v >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[v & 63] : '=';
    }
    return out;
}

// And back, for a media file a test writes to disk.
std::vector<uint8_t> base64_bytes(const std::string &text) {
    std::vector<uint8_t> out;
    uint32_t acc = 0;
    int bits = 0;
    for (const char c : text) {
        const char *const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const char *at = c == '=' ? nullptr : std::strchr(alphabet, c);
        if (at == nullptr) {
            break;
        }
        acc = (acc << 6) | static_cast<uint32_t>(at - alphabet);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

TEST(CApiAudio, AClipOfThirtySecondsIsOneChunkAndOneSampleMoreIsTwo) {
    // Upstream #30403 (b12309..b12387): the chunk count is the samples over a chunk's hop, rounded up.
    // Before, a clip just short of or exactly 30 s (480000 samples at 16 kHz) got a second chunk of
    // padding alone, 375 embeddings of silence more in the prompt.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-audio-chunks";
    spec.n_ctx_train = 1024;
    const std::string path = write_tiny_llama_gguf(spec);
    const std::string mmproj = write_tiny_audio_mmproj_gguf();
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(mmproj.empty());
    jllama_model *model = load_tiny(path, {"--mmproj", mmproj, "-c", "1024", "-b", "1024", "-ub", "1024"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string marker = json::parse(meta.str()).value("media_marker", std::string());
    ASSERT_FALSE(marker.empty());
    Owned text_only;
    ASSERT_EQ(jllama_completions(model, R"({"prompt":"hear  now","n_predict":1,"temperature":0})", &text_only.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int text_tokens = json::parse(text_only.str()).at("tokens_evaluated").get<int>();
    const auto evaluated = [&](size_t samples) {
        std::vector<float> tone(samples);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
        }
        const json request = {
            {"prompt",
             {{"prompt_string", "hear " + marker + " now"},
              {"multimodal_data", json::array({base64_of(jllama_tts::pcm_to_wav16_bytes(tone, 16000))})}}},
            {"n_predict", 1},
            {"temperature", 0},
            {"cache_prompt", false}};
        Owned completion;
        if (jllama_completions(model, request.dump().c_str(), &completion.s) != JLLAMA_STATUS_OK) {
            ADD_FAILURE() << jllama_last_error();
            return -1;
        }
        return json::parse(completion.str()).at("tokens_evaluated").get<int>();
    };
    EXPECT_EQ(evaluated(479999), text_tokens + 375);
    EXPECT_EQ(evaluated(480000), text_tokens + 375);
    EXPECT_EQ(evaluated(480001), text_tokens + 750);
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(mmproj, ignored);
}

namespace {
// The embeddings a 30 s tone adds to a raw prompt, for a projector of the given encoder width and
// stack factor; -1 when the request failed.
int embeddings_of_a_chunk(int a_embd, int stack) {
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-audio-stack";
    spec.n_ctx_train = 1024;
    const std::string path = write_tiny_llama_gguf(spec);
    const std::string mmproj = write_tiny_audio_mmproj_gguf("jllama-mmproj-audio-stack", a_embd, stack);
    jllama_model *model = load_tiny(path, {"--mmproj", mmproj, "-c", "1024", "-b", "1024", "-ub", "1024"});
    if (model == nullptr) {
        return -1;
    }
    Owned meta;
    jllama_model_meta(model, &meta.s);
    const std::string marker = json::parse(meta.str()).value("media_marker", std::string());
    const auto evaluated = [&](bool with_clip) {
        std::vector<float> tone(480000);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
        }
        json request = {{"n_predict", 1}, {"temperature", 0}, {"cache_prompt", false}};
        if (with_clip) {
            request["prompt"] = {
                {"prompt_string", "hear " + marker + " now"},
                {"multimodal_data", json::array({base64_of(jllama_tts::pcm_to_wav16_bytes(tone, 16000))})}};
        } else {
            request["prompt"] = "hear  now";
        }
        Owned completion;
        if (jllama_completions(model, request.dump().c_str(), &completion.s) != JLLAMA_STATUS_OK) {
            return -1;
        }
        return json::parse(completion.str()).at("tokens_evaluated").get<int>();
    };
    const int text = evaluated(false);
    const int clip = evaluated(true);
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(mmproj, ignored);
    return text < 0 || clip < 0 ? -1 : clip - text;
}
} // namespace

TEST(CApiAudio, AStackFactorThatDoesNotDivideTheFramesKeepsUpstreamsCount) {
    // Ultravox v0.5 (CI's audio pair) stacks 8 of the 1500 encoder frames of a 30 s chunk: measured
    // with the real pair, every chunk is 187 prompt tokens (a 2.6 s and a 35 s clip: 37, 224, 411).
    // Python's StackAudioFrames pads 1500 to 1504 and gives 188; llama.cpp counts 187 and its graph
    // keeps 187, dropping the last 4 frames (80 ms), because GGML_PAD rounds a stride that is no power
    // of two (1280 * 8) by bit mask. An encoder 20 wide has the same property (160) and the same count.
    EXPECT_EQ(embeddings_of_a_chunk(20, 8), 187);
    // a stack factor that divides 1500 agrees in every shape
    EXPECT_EQ(embeddings_of_a_chunk(20, 4), 375);
    EXPECT_EQ(embeddings_of_a_chunk(16, 4), 375);
}

TEST(CApiAudio, AStackStrideThatIsAPowerOfTwoAbortsTheProcess) {
    // The other half of the defect above: with 16 * 8 = 128 GGML_PAD pads a whole row, the graph gives
    // 188 embeddings, llama.cpp expects 187 and calls GGML_ABORT -- the end of a JVM, for an Ultravox
    // projector over a Whisper encoder 512 or 1024 wide. Not filed upstream. This test turns red once
    // llama.cpp counts and builds the same rows; then the count above says which of the two it chose.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_DEATH(embeddings_of_a_chunk(16, 8), "Invalid number of output tokens");
}

TEST(CApiAudio, AModelWithAnAudioProjectorHearsAClipInAPrompt) {
    // Audio input (llama.cpp discussion #13759): no audio model is in CI's set, so the Java test of
    // it skips itself everywhere. The generator's Whisper-style projector makes the random model
    // hear: the meta says so, and a clip in upstream's multimodal prompt object becomes embeddings
    // between the text tokens, on the raw, streamed and OpenAI paths.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-audio";
    spec.n_ctx_train = 512; // a clip is one 30 s chunk of 375 embeddings
    const std::string path = write_tiny_llama_gguf(spec);
    const std::string mmproj = write_tiny_audio_mmproj_gguf();
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(mmproj.empty());
    jllama_model *model = load_tiny(path, {"--mmproj", mmproj, "-c", "512"});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json j = json::parse(meta.str());
    EXPECT_TRUE(j.at("modalities").at("audio").get<bool>()) << meta.str();
    EXPECT_FALSE(j.at("modalities").at("vision").get<bool>()) << meta.str();
    const std::string marker = j.value("media_marker", std::string());
    ASSERT_FALSE(marker.empty());

    // half a second of a 440 Hz tone at 16 kHz, as a WAV file
    std::vector<float> tone(8000);
    for (size_t i = 0; i < tone.size(); ++i) {
        tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
    }
    const std::string clip = base64_of(jllama_tts::pcm_to_wav16_bytes(tone, 16000));

    Owned text_only;
    ASSERT_EQ(jllama_completions(model, R"({"prompt":"hear  now","n_predict":1,"temperature":0})", &text_only.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int text_tokens = json::parse(text_only.str()).at("tokens_evaluated").get<int>();

    const auto request = [&](const char *length_field) {
        return json{
            {"prompt", {{"prompt_string", "hear " + marker + " now"}, {"multimodal_data", json::array({clip})}}},
            {length_field, 2},
            {"temperature", 0}}
            .dump();
    };
    Owned completion;
    ASSERT_EQ(jllama_completions(model, request("n_predict").c_str(), &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int with_clip = json::parse(completion.str()).at("tokens_evaluated").get<int>();
    // 3000 mel frames, halved by the second convolution and stacked by four
    EXPECT_EQ(with_clip, text_tokens + 375) << completion.str();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, request("n_predict").c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model, task).at("tokens_evaluated").get<int>(), with_clip);

    Owned oai;
    ASSERT_EQ(jllama_completions_oai(model, request("max_tokens").c_str(), &oai.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(oai.str()).at("usage").at("prompt_tokens").get<int>(), with_clip) << oai.str();

    // a chat message with an input_audio part goes the same way
    const json chat = {
        {"messages",
         json::array({{{"role", "user"},
                       {"content", json::array({{{"type", "text"}, {"text", "What is this?"}},
                                                {{"type", "input_audio"},
                                                 {"input_audio", {{"data", clip}, {"format", "wav"}}}}})}}})},
        {"max_tokens", 2},
        {"temperature", 0}};
    Owned answer;
    ASSERT_EQ(jllama_chat_completions(model, chat.dump().c_str(), &answer.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(json::parse(answer.str()).at("usage").at("prompt_tokens").get<int>(), 375) << answer.str();

    // an image is not something this projector reads
    const json image = {
        {"prompt", {{"prompt_string", "look " + marker + " now"}, {"multimodal_data", json::array({TINY_PNG_BASE64})}}},
        {"n_predict", 1}};
    Owned refused;
    EXPECT_NE(jllama_completions(model, image.dump().c_str(), &refused.s), JLLAMA_STATUS_OK);

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(mmproj, ignored);
}

TEST(CApiFp8Cache, AModelWithCacheScalesRunsAnEightBitFloatCache) {
    // The 8-bit float KV cache (--cache-type-k/-v f8_e4m3, b12283; on CUDA since b12289) stores K and
    // V divided by a per-layer scale the model may carry and multiplies it back in the attention. The
    // enum sweep runs the type on a model without scales; this is the model with them. It has to
    // answer with the cache on both sides and on one, with flash attention on and off, the same again
    // from a saved and restored slot, and exactly like the model without scales under an F16 cache,
    // which never reads them.
    tiny_model_spec scaled;
    scaled.prefix = "jllama-c-api-fp8-scaled";
    scaled.cache_scales = true;
    scaled.k_cache_scale = 0.5f;
    scaled.v_cache_scale = 0.25f;
    const std::string with_scales = write_tiny_llama_gguf(scaled);
    tiny_model_spec plain_spec;
    plain_spec.prefix = "jllama-c-api-fp8-plain";
    const std::string plain = write_tiny_llama_gguf(plain_spec);
    ASSERT_FALSE(with_scales.empty());
    ASSERT_FALSE(plain.empty());

    const auto answer_of = [](const std::string &path, const std::vector<std::string> &args) {
        jllama_model *model = load_tiny(path, args);
        EXPECT_NE(model, nullptr) << jllama_last_error();
        if (model == nullptr) {
            return std::string();
        }
        const std::string answer = greedy_answer(model, 16);
        jllama_model_free(model);
        return answer;
    };

    const std::string f16_scaled = answer_of(with_scales, {"--cache-type-k", "f16", "--cache-type-v", "f16"});
    EXPECT_FALSE(f16_scaled.empty());
    EXPECT_EQ(f16_scaled, answer_of(plain, {"--cache-type-k", "f16", "--cache-type-v", "f16"}))
        << "an F16 cache read the cache scales";

    for (const char *flash : {"on", "off"}) {
        SCOPED_TRACE(flash);
        EXPECT_FALSE(
            answer_of(with_scales, {"--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--flash-attn", flash})
                .empty());
        EXPECT_FALSE(answer_of(with_scales, {"--cache-type-k", "f8_e4m3", "--flash-attn", flash}).empty());
        EXPECT_FALSE(answer_of(with_scales, {"--cache-type-v", "f8_e4m3", "--flash-attn", flash}).empty());
    }

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-fp8-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *model = load_tiny(
        with_scales, {"--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3", "--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model, 16).empty());
    const std::string second = greedy_answer(model, 16);
    const std::string third = greedy_answer(model, 16);
    EXPECT_EQ(second, third); // both over the cached prompt
    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "fp8.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "fp8.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(restored.str()).at("n_read").get<size_t>(),
              json::parse(saved.str()).at("n_written").get<size_t>())
        << restored.str();
    EXPECT_EQ(greedy_answer(model, 16), third);
    jllama_model_free(model);

    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove(with_scales, ignored);
    std::filesystem::remove(plain, ignored);
}

TEST(CApiFp8Cache, ACacheScaleOfZeroIsRefusedOnlyByTheEightBitFloatCache) {
    // llama.cpp refuses a context whose 8-bit float cache would divide by zero; the same file is a
    // usable model for every other cache type, which does not read the scale.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-fp8-zero";
    spec.cache_scales = true;
    spec.k_cache_scale = 0.0f;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());

    jllama_model *refused = load_tiny(path, {"--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3"});
    EXPECT_EQ(refused, nullptr) << "a cache scale of zero was accepted";
    jllama_model_free(refused);

    jllama_model *loaded = load_tiny(path, {});
    ASSERT_NE(loaded, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(loaded).empty());
    jllama_model_free(loaded);

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiRecurrent, ARecurrentModelAnswersTheSameFromACachedPromptAndARestoredSlot) {
    // The second kind of memory llama.cpp has: Mamba keeps no rows per token but one state per
    // layer, which every token overwrites. A prefix cannot be cut out of that, so the server works
    // with checkpoints, and a slot file holds the state. A random Mamba has to answer, answer the
    // same again (the second request restores the state the prompt left), and answer the same from
    // a slot saved and restored.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::recurrent("jllama-c-api-mamba"));
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-mamba-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("architecture").get<std::string>(), "mamba") << meta.str();

    const std::string first = greedy_answer(model, 16);
    EXPECT_FALSE(first.empty());
    const std::string second = greedy_answer(model, 16);
    const std::string third = greedy_answer(model, 16);
    EXPECT_EQ(second, third); // both over the cached prompt

    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "mamba.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json saved_json = json::parse(saved.str());
    EXPECT_GT(saved_json.at("n_written").get<size_t>(), 0u);
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "mamba.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json restored_json = json::parse(restored.str());
    EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
    EXPECT_EQ(restored_json.at("n_restored").get<int>(), saved_json.at("n_saved").get<int>()) << restored.str();
    EXPECT_EQ(greedy_answer(model, 16), third);

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(CApiZaya1, AZaya1ModelAnswersAndRestoresItsHybridState) {
    // ZAYA1 (b12517, #32071) keeps two kinds of memory per layer: KV rows for its attention, and a
    // recurrent state -- the queries and keys of the last positions for its convolution and the
    // previous token's value for its delayed value head. The server treats it as a hybrid model
    // (checkpoints, a slot file with both). A random ZAYA1 has to load, answer, answer the same over
    // the cached prompt, and answer the same from a slot saved, erased and restored.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::zaya1("jllama-c-api-zaya1"));
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-zaya1-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("architecture").get<std::string>(), "zaya1") << meta.str();

    const std::string first = greedy_answer(model, 16);
    EXPECT_FALSE(first.empty());
    const std::string second = greedy_answer(model, 16);
    const std::string third = greedy_answer(model, 16);
    EXPECT_EQ(second, third); // both over the cached prompt

    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "zaya1.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json saved_json = json::parse(saved.str());
    EXPECT_GT(saved_json.at("n_written").get<size_t>(), 0u);
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "zaya1.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json restored_json = json::parse(restored.str());
    EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
    EXPECT_EQ(greedy_answer(model, 16), third);

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(CApiZaya1, ThePromptInOneBatchAndOneTokenAtATimeGiveTheSameAnswer) {
    // Evaluated whole, the convolution and the delayed value read the previous position from the
    // batch itself; one token at a time (-ub 1) they read it from the recurrent state the step before
    // left. Both are the same computation, so the answer and the probabilities of its first token
    // agree up to rounding -- a state written or read at the wrong place does not.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::zaya1("jllama-c-api-zaya1-ubatch"));
    ASSERT_FALSE(path.empty());
    const json request = {{"prompt", "Yellow paint, how are you"},
                          {"n_predict", 6},
                          {"temperature", 0},
                          {"n_probs", 3},
                          {"ignore_eos", true},
                          {"cache_prompt", false}};
    std::vector<json> answers;
    for (const std::vector<std::string> extra : {std::vector<std::string>{}, std::vector<std::string>{"-ub", "1"}}) {
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned response;
        ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        answers.push_back(json::parse(response.str()));
        jllama_model_free(model);
    }
    EXPECT_EQ(answers[0].at("content"), answers[1].at("content"));
    const json &batched = answers[0].at("completion_probabilities").at(0).at("top_logprobs");
    const json &stepped = answers[1].at("completion_probabilities").at(0).at("top_logprobs");
    ASSERT_EQ(batched.size(), stepped.size());
    for (size_t i = 0; i < batched.size(); ++i) {
        EXPECT_EQ(batched[i].at("id"), stepped[i].at("id")) << i;
        EXPECT_NEAR(batched[i].at("logprob").get<double>(), stepped[i].at("logprob").get<double>(), 1e-3) << i;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiZaya1, TheQuantizerKeepsTheRouterAndTheGroupedConvolutionInF32) {
    // b12521 (#32071): ZAYA1's top-1 router (ffn_router_*) and its grouped convolution kernels
    // (cca_conv_grp) stay F32 when the model is quantized, every other matrix takes the type asked
    // for. On the wide model every one of them fits a Q8_0 block, so it is the name that decides; the
    // quantized model loads and answers.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::zaya1_wide("jllama-c-api-zaya1-quant"));
    ASSERT_FALSE(path.empty());
    const std::string out = path + ".q8_0.gguf";
    ASSERT_EQ(jllama_quantize(path.c_str(), out.c_str(), /*LLAMA_FTYPE_MOSTLY_Q8_0*/ 7, 1, false), JLLAMA_STATUS_OK)
        << jllama_last_error();
    gguf_init_params params = {true, nullptr};
    gguf_context *g = gguf_init_from_file(out.c_str(), params);
    ASSERT_NE(g, nullptr);
    const auto type_of = [&](const std::string &name) {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        EXPECT_GE(id, 0) << name;
        return id >= 0 ? gguf_get_tensor_type(g, id) : GGML_TYPE_COUNT;
    };
    for (const char *name : {"blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.1.ffn_gate_up_exps.weight",
                             "blk.1.ffn_down_exps.weight"}) {
        EXPECT_EQ(type_of(name), GGML_TYPE_Q8_0) << name;
    }
    for (const char *name : {"blk.0.cca_conv_grp.weight", "blk.1.cca_conv_grp.weight", "blk.0.ffn_router_fc1.weight",
                             "blk.0.ffn_router_fc2.weight", "blk.1.ffn_router_out.weight"}) {
        EXPECT_EQ(type_of(name), GGML_TYPE_F32) << name;
    }
    gguf_free(g);
    jllama_model *model = load_tiny(out, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model, 8).empty());
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(out, ignored);
}

TEST(CApiZaya1, TwoSlotsAtOnceAnswerAsEachAlone) {
    // Two requests on two slots are decoded in one batch, each sequence with its own recurrent
    // state (the convolution's last positions, the delayed value) next to its own KV rows. A state
    // read from or written to the other sequence's row changes the probabilities of an answer. The
    // random model's text is the same for both prompts, so it is the probabilities that are compared.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::zaya1("jllama-c-api-zaya1-slots"));
    ASSERT_FALSE(path.empty());
    // the same last tokens: what tells the answers apart has to come through the attention and the state
    const std::vector<std::string> prompts = {"Hello there, how are you", "Yellow paint, how are you"};
    const auto request = [](const std::string &prompt) {
        return json{{"prompt", prompt}, {"n_predict", 4},     {"temperature", 0},
                    {"n_probs", 3},     {"ignore_eos", true}, {"cache_prompt", false}}
            .dump();
    };
    // the log-probabilities of the top tokens of every step
    const auto logprobs = [](const Owned &response) {
        std::vector<double> out;
        const json steps = json::parse(response.str()).at("completion_probabilities");
        for (size_t s = 0; s < steps.size(); ++s) {
            const json &top = steps[s].at("top_logprobs");
            for (size_t t = 0; t < top.size(); ++t) {
                out.push_back(top[t].at("logprob").get<double>());
            }
        }
        return out;
    };
    std::vector<std::vector<double>> alone;
    {
        jllama_model *model = load_tiny(path, {"-c", "128"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        for (const std::string &prompt : prompts) {
            Owned response;
            ASSERT_EQ(jllama_completions(model, request(prompt).c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            alone.push_back(logprobs(response));
        }
        jllama_model_free(model);
    }
    ASSERT_EQ(alone[0].size(), 12u); // 4 steps of 3
    ASSERT_EQ(alone[1].size(), 12u);
    double apart = 0.0;
    for (size_t i = 0; i < alone[0].size(); ++i) {
        apart = std::max(apart, std::fabs(alone[0][i] - alone[1][i]));
    }
    EXPECT_GT(apart, 1e-3) << "the two prompts are answered with other probabilities";

    jllama_model *model = load_tiny(path, {"-c", "128", "-np", "2"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::vector<Owned> together(prompts.size());
    std::vector<jllama_status> status(prompts.size(), JLLAMA_STATUS_ERROR);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < prompts.size(); ++i) {
        threads.emplace_back(
            [&, i]() { status[i] = jllama_completions(model, request(prompts[i]).c_str(), &together[i].s); });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (size_t p = 0; p < prompts.size(); ++p) {
        ASSERT_EQ(status[p], JLLAMA_STATUS_OK) << p;
        const std::vector<double> got = logprobs(together[p]);
        ASSERT_EQ(got.size(), alone[p].size()) << p;
        for (size_t i = 0; i < got.size(); ++i) {
            EXPECT_NEAR(got[i], alone[p][i], 1e-3) << "prompt " << p << ", value " << i;
        }
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiThreads, ALongPromptsEmbeddingLookupGivesTheSameAnswerOnOneAndOnFourThreads) {
    // Since b12517 (#32514) the CPU backend splits a GET_ROWS of at least 262144 elements over every
    // thread (it was one). The token embedding lookup of 2100 tokens of width 128 in one batch is
    // 268800 elements: on four threads each takes a share of the rows, on one thread one takes them
    // all, and the logits must be the same.
    tiny_model_spec spec = tiny_model_spec::wide(128, "jllama-c-api-get-rows");
    spec.n_ctx_train = 4096;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    std::string prompt;
    for (int i = 0; prompt.size() < 2100; ++i) {
        prompt += static_cast<char>('a' + (i * 7) % 26);
        if (i % 9 == 8) {
            prompt += ' ';
        }
    }
    const json request = {{"prompt", prompt}, {"n_predict", 2},     {"temperature", 0},
                          {"n_probs", 3},     {"ignore_eos", true}, {"cache_prompt", false}};
    std::vector<json> answers;
    for (const char *threads : {"1", "4"}) {
        const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "4096", "-b",    "4096", "-ub",
                                               "4096",   "-ngl", "0",  "--no-warmup", "-t",   threads, "-tb",  threads};
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        jllama_model *model = nullptr;
        ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned response;
        ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json j = json::parse(response.str());
        EXPECT_GE(j.at("timings").at("prompt_n").get<int>(), 2048) << "one batch over the threshold";
        answers.push_back(j);
        jllama_model_free(model);
    }
    EXPECT_EQ(answers[0].at("content"), answers[1].at("content"));
    const json &one = answers[0].at("completion_probabilities").at(0).at("top_logprobs");
    const json &four = answers[1].at("completion_probabilities").at(0).at("top_logprobs");
    ASSERT_EQ(one.size(), four.size());
    for (size_t i = 0; i < one.size(); ++i) {
        EXPECT_EQ(one[i].at("id"), four[i].at("id")) << i;
        EXPECT_NEAR(one[i].at("logprob").get<double>(), four[i].at("logprob").get<double>(), 1e-4) << i;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiBatchSize, ABatchSizeBelowTwoIsRefusedInsteadOfEndingTheProcess) {
    // The server's load evaluates two tokens in one batch to learn whether the memory can drop part
    // of a sequence (common_context_can_seq_rm). With --batch-size 1 llama.cpp aborted the process in
    // common_batch::get_sub_batch -- a JVM, for setBatchSize(1). The load is refused by name instead,
    // the process goes on, and a batch of 2 loads and answers.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-batch-size");
    ASSERT_FALSE(path.empty());
    for (const char *size : {"1", "0"}) {
        EXPECT_EQ(load_tiny(path, {"-b", size}), nullptr) << size;
        EXPECT_NE(std::string(jllama_last_error()).find("--batch-size must be at least 2"), std::string::npos)
            << jllama_last_error();
    }
    jllama_model *model = load_tiny(path, {"-b", "2", "-ub", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model, 4).empty());
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiParseErrors, AChatTemplateThatDoesNotParseIsRefusedWithItsErrorAndTheProcessGoesOn) {
    // A --chat-template is verified after the arguments are read, and upstream's parse called
    // exit(1) for that refusal (std::runtime_error, not std::invalid_argument): setChatTemplate()
    // with a broken template ended the JVM. patches/0031 answers false for it like for any other
    // refusal and keeps the reason, which the load names. The next load in the process works.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-bad-template");
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_tiny(path, {"--jinja", "--chat-template", "{% if %}"}), nullptr);
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find("Failed to parse model parameters"), std::string::npos) << error;
    EXPECT_NE(error.find("the supplied chat template is not supported"), std::string::npos) << error;
    EXPECT_EQ(error.find("usage:"), std::string::npos) << "the usage text stays out: " << error;
    jllama_model *model = load_tiny(path, {"--jinja", "--chat-template", "{{ messages[0].content }}"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiParseErrors, AnArgumentTheParserRefusesIsNamedWithoutTheUsageText) {
    // an option's handler refusing its value (here a number the option cannot take) is wrapped by
    // upstream into "error while handling argument ..." plus the option's usage; the load keeps the
    // first part
    const std::string path = write_tiny_llama_gguf("jllama-c-api-bad-argument");
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(load_tiny(path, {"--ctx-size", "not-a-number"}), nullptr);
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find("Failed to parse model parameters: error while handling argument \"--ctx-size\""),
              std::string::npos)
        << error;
    EXPECT_EQ(error.find("usage:"), std::string::npos) << error;
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

// The request of upstream's own /v1/systemone test: one question of each type.
const char *const SYSTEMONE_REQUEST = R"({
  "state": "I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"}
  }
})";

double sum_of(const json &probabilities) {
    double sum = 0.0;
    for (const auto &entry : probabilities.items()) {
        sum += entry.value().get<double>();
    }
    return sum;
}

TEST(CApiSystemOne, ADecisionModelAnswersEveryQuestionType) {
    // /v1/systemone (jllama_systemone, LlamaModel.handleSystemOne) on a model that declares a
    // decision type. The Java test of this path runs in CI on upstream's tinylaya-for-testing
    // (decision type "laya"); the generator's "openjev" model is the plain decoder with the type and a
    // "systemone" template, and its answers are read at the label tokens. A random model decides
    // nothing: what is asserted is the shape upstream documents -- a distribution per question that
    // sums to one, the choice its most probable option, a score inside its scale.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::decision("jllama-c-api-decision"));
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"-c", "1024"});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    Owned response;
    ASSERT_EQ(jllama_systemone(model, SYSTEMONE_REQUEST, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json body = json::parse(response.str());
    const json &answers = body.at("answers");
    ASSERT_EQ(answers.size(), 3u) << response.str();

    const json &route = answers.at("route");
    EXPECT_EQ(route.at("type").get<std::string>(), "choice");
    ASSERT_EQ(route.at("probabilities").size(), 3u) << response.str();
    EXPECT_NEAR(sum_of(route.at("probabilities")), 1.0, 1e-4);
    const std::string choice = route.at("choice").get<std::string>();
    for (const auto &option : route.at("probabilities").items()) {
        EXPECT_LE(option.value().get<double>(), route.at("probabilities").at(choice).get<double>()) << option.key();
    }

    const json &urgency = answers.at("urgency");
    EXPECT_EQ(urgency.at("type").get<std::string>(), "score");
    ASSERT_EQ(urgency.at("probabilities").size(), 4u) << response.str();
    EXPECT_NEAR(sum_of(urgency.at("probabilities")), 1.0, 1e-4);
    EXPECT_GE(urgency.at("score").get<double>(), 0.0);
    EXPECT_LE(urgency.at("score").get<double>(), 3.0);
    EXPECT_EQ(urgency.at("legend").at("3").get<std::string>(), "right now");

    const json &angry = answers.at("angry");
    EXPECT_EQ(angry.at("type").get<std::string>(), "noul");
    EXPECT_GE(angry.at("noul").get<double>(), 0.0);
    EXPECT_LE(angry.at("noul").get<double>(), 1.0);

    EXPECT_GT(body.at("usage").at("input_tokens").get<int>(), 0);
    EXPECT_EQ(body.at("usage").at("output_tokens").get<int>(), 0);

    // the same question twice is the same answer: nothing is sampled
    Owned again;
    ASSERT_EQ(jllama_systemone(model, SYSTEMONE_REQUEST, &again.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(again.str()).at("answers").at("route").at("choice").get<std::string>(), choice);

    // what the request gets wrong is the caller's fault
    for (const char *bad :
         {R"({"state":"x","questions":{"q":{"type":"bogus","instructions":"?"}}})",
          R"({"questions":{"q":{"type":"noul","instructions":"?"}}})", R"({"state":"x","questions":{}})"}) {
        SCOPED_TRACE(bad);
        Owned refused;
        EXPECT_EQ(jllama_systemone(model, bad, &refused.s), JLLAMA_STATUS_INVALID_REQUEST) << jllama_last_error();
    }

    // a decision model does not generate text, and says where to go instead
    Owned completion;
    EXPECT_NE(jllama_completions(model, R"({"prompt":"Hello","n_predict":4})", &completion.s), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("/v1/systemone"), std::string::npos) << jllama_last_error();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiSlotFiles, ASlidingWindowModelsSlotFileCarriesItsContextCheckpoints) {
    // The checkpoint half of the slot file's appendix (patches/0025 shares its two sites with the
    // draft half). The server keeps context checkpoints only for a model that cannot roll its memory
    // back -- a sliding attention window, a recurrent state -- so the plain decoder never wrote one.
    // The generator's sliding-window model does: the save appends them, under a non-ASCII file name
    // they stay in the one file, and the restore reads them back.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::swa(16, "jllama-c-api-slot-swa"));
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-swa-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));

    struct Lines {
        std::mutex mutex;
        std::vector<std::string> text;
        int count(const std::string &needle) {
            std::lock_guard<std::mutex> lock(mutex);
            int n = 0;
            for (const std::string &line : text) {
                n += line.find(needle) != std::string::npos ? 1 : 0;
            }
            return n;
        }
    } lines;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *user) {
                      auto *sink = static_cast<Lines *>(user);
                      std::lock_guard<std::mutex> lock(sink->mutex);
                      sink->text.emplace_back(text);
                  },
                  &lines),
              JLLAMA_STATUS_OK);
    // An ASSERT below returns early, and the callback must not outlive `lines`: left installed, the
    // next test's first log line went to this frame and ended the run (exit code 3 on Windows), so
    // every later test went unreported.
    struct LogCallbackReset {
        ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); }
    } log_callback_reset;

    jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    // longer than the window, so the window has moved on by the time the slot is saved
    EXPECT_FALSE(greedy_answer(model, 48).empty());

    const std::string name = "swa-\xC3\xA4-\xE6\x97\xA5.bin";
    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json saved_json = json::parse(saved.str());
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        const auto u8 = entry.path().filename().u8string();
        files.emplace_back(u8.begin(), u8.end());
    }
    ASSERT_EQ(files.size(), 1u) << "the slot file and its checkpoints must be one file";
    EXPECT_EQ(files[0], name);
    std::error_code ec;
    EXPECT_EQ(std::filesystem::file_size(dir / std::filesystem::u8path(name), ec),
              saved_json.at("n_written").get<size_t>())
        << saved.str();

    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, name.c_str(), &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json restored_json = json::parse(restored.str());
    EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
    EXPECT_FALSE(greedy_answer(model, 8).empty()); // the restored slot, checkpoints and all, still answers
    jllama_model_free(model);
    ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK); // flushes

    // without these the test would pass on a model that wrote no checkpoint at all
    EXPECT_EQ(lines.count("context checkpoint(s) ("), 1) << "the save appended no checkpoints";
    EXPECT_EQ(lines.count("context checkpoint(s) from"), 1) << "the restore read no checkpoints";

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
TEST_F(CApi, TheAttachedServerKeepsTheConnectionAfterAStreamedResponse) {
    // Upstream #30492 (b12408): the stream callback returned `has_next`, which is false after the last
    // chunk, and cpp-httplib then closed the connection -- every streamed response cost the client a
    // new connection. A client that streams and asks again must do both over one socket.
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    probe_listener.join();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", port_text.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start_attached(model_, 5, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();

    httplib::Client client("127.0.0.1", port);
    client.set_keep_alive(true);
    client.set_read_timeout(std::chrono::seconds(30));
    std::atomic<int> sockets{0};
    client.set_socket_options([&sockets](auto) { sockets.fetch_add(1); });
    httplib::Result health;
    for (int i = 0; i < 100 && !(health && health->status == 200); i++) {
        health = client.Get("/health");
        if (!(health && health->status == 200)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ASSERT_TRUE(health && health->status == 200);
    const int before = sockets.load();

    const auto streamed = client.Post("/completion", R"({"prompt":"Hi","n_predict":4,"temperature":0,"stream":true})",
                                      "application/json");
    ASSERT_TRUE(streamed) << httplib::to_string(streamed.error());
    EXPECT_EQ(streamed->status, 200);
    EXPECT_NE(streamed->body.find("data: "), std::string::npos) << streamed->body;
    EXPECT_NE(streamed->get_header_value("Connection"), "close");
    const auto again = client.Get("/health");
    ASSERT_TRUE(again) << httplib::to_string(again.error());
    EXPECT_EQ(again->status, 200);
    EXPECT_EQ(sockets.load(), before) << "the streamed response closed the connection";

    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
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

// jllama_tts_options (b12431): the defaults leave every choice to the engine or the model, and
// struct_size is the library's own sizeof, which the FFM layout of llama-ffm is held to as well.
TEST(CApiTts, TheOptionsStartAtTheDefaultsOfTheEngine) {
    jllama_tts_options options;
    std::memset(&options, 0x7f, sizeof(options));
    jllama_tts_options_init(&options);
    const jllama_tts::synthesis_options engine;
    EXPECT_EQ(options.struct_size, sizeof(jllama_tts_options));
    EXPECT_EQ(sizeof(jllama_tts_options), 88u) << "llama-ffm's TTS_OPTIONS lays out 88 bytes (b12494's raw_prompt)";
    EXPECT_EQ(offsetof(jllama_tts_options, speaker_reference_text), 64u)
        << "b12453 appended the transcript behind the 64 bytes of 6.1.0's header";
    EXPECT_EQ(offsetof(jllama_tts_options, cfg_scale), 72u) << "b12459 appended guidance and solver steps";
    EXPECT_EQ(offsetof(jllama_tts_options, flow_steps), 76u);
    EXPECT_EQ(options.speaker_reference_path, nullptr);
    EXPECT_EQ(options.lang, nullptr);
    EXPECT_EQ(options.voice, nullptr);
    EXPECT_EQ(options.max_frames, engine.max_frames);
    EXPECT_EQ(options.top_k, engine.top_k);
    EXPECT_EQ(options.seed, engine.seed);
    EXPECT_FLOAT_EQ(options.temperature, engine.temperature);
    EXPECT_FLOAT_EQ(options.top_p, 1.0f);
    EXPECT_FLOAT_EQ(options.repeat_penalty, 1.0f);
    EXPECT_FLOAT_EQ(options.speed, 1.0f);
    EXPECT_FLOAT_EQ(options.loudness, 0.0f);
    EXPECT_EQ(options.speaker_reference_text, nullptr);
    EXPECT_FLOAT_EQ(options.cfg_scale, engine.cfg_scale);
    EXPECT_EQ(options.flow_steps, engine.flow_steps);
}

// What is checked before the model is touched, so a pointer that is no engine never reaches it: a
// struct_size that does not hold the field is the caller's mistake, a value no model can take the
// request's fault -- the out-parameters stay untouched either way.
TEST(CApiTts, TheOptionsAreCheckedBeforeTheModel) {
    int not_a_tts = 0;
    auto *tts = reinterpret_cast<jllama_tts_model *>(&not_a_tts);
    unsigned char sentinel = 0;
    unsigned char *wav = &sentinel;
    size_t length = 7;
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.struct_size = 4;
    EXPECT_EQ(jllama_tts_synthesize_options(tts, "Hi", &options, &wav, &length), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(jllama_last_error()).find("struct_size"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(wav, &sentinel);
    EXPECT_EQ(length, 7u);

    const auto expect_refused = [&](const jllama_tts_options &o, const char *name) {
        unsigned char *out = nullptr;
        size_t n = 0;
        EXPECT_EQ(jllama_tts_synthesize_options(tts, "Hi", &o, &out, &n), JLLAMA_STATUS_INVALID_REQUEST) << name;
        EXPECT_NE(std::string(jllama_last_error()).find(name), std::string::npos) << jllama_last_error();
        EXPECT_EQ(out, nullptr);
    };
    jllama_tts_options_init(&options);
    options.speed = 0.0f;
    expect_refused(options, "speed");
    jllama_tts_options_init(&options);
    options.loudness = 3.0f;
    expect_refused(options, "loudness");
    jllama_tts_options_init(&options);
    options.top_p = 2.0f;
    expect_refused(options, "top_p");
    jllama_tts_options_init(&options);
    options.repeat_penalty = -1.0f;
    expect_refused(options, "repeat_penalty");
    jllama_tts_options_init(&options);
    options.cfg_scale = -1.0f;
    expect_refused(options, "cfg_scale");
    jllama_tts_options_init(&options);
    options.flow_steps = 2000;
    expect_refused(options, "flow_steps");
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
    // The options variant (b12431) is the one every other ends in.
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    EXPECT_EQ(jllama_tts_synthesize_options(nullptr, "Hi", &options, &wav, &length), JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(jllama_last_error(), "TextToSpeech handle is null");
    EXPECT_EQ(jllama_tts_synthesize_options(reinterpret_cast<jllama_tts_model *>(&not_a_tts), "Hi", &options, nullptr,
                                            &length),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    jllama_tts_options_init(nullptr); // ignored
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

// A random Pocket TTS pair (tiny_llama_gguf.hpp), in the layout of CI's real english_2026-04 pair:
// the T5 Unigram vocabulary with its three control tokens, the flow net, the end-of-speech head and
// the Mimi decoder. What makes its lengths exact is the end-of-speech head: with a bias of +100 it
// fires at the first frame, and llama.cpp then speaks 5 more frames for a chunk of up to 6 tokens and
// 3 for a longer one; with -100 it never fires and a chunk ends at its budget of
// ceil((tokens / 3 + 2) * 12.5) frames. A frame is 1920 samples (12.5 per second at 24 kHz).
class CApiPocketTts : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        backbone() = jllama_test::write_tiny_pockettts_gguf();
        never() = jllama_test::write_tiny_pockettts_mmproj_gguf("jllama-pockettts-never", -100.0f);
        at_once() = jllama_test::write_tiny_pockettts_mmproj_gguf("jllama-pockettts-at-once", 100.0f);
        no_voices() = jllama_test::write_tiny_pockettts_mmproj_gguf("jllama-pockettts-no-voices", 100.0f, {});
        ASSERT_FALSE(backbone().empty());
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), never().c_str(), 0, 2, &budget_tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), at_once().c_str(), 0, 2, &eos_tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(budget_tts());
        jllama_tts_free(eos_tts());
        budget_tts() = nullptr;
        eos_tts() = nullptr;
        for (const std::string &p : {backbone(), never(), at_once(), no_voices()}) {
            std::remove(p.c_str());
        }
    }
    static std::string &backbone() {
        static std::string p;
        return p;
    }
    static std::string &never() {
        static std::string p;
        return p;
    }
    static std::string &at_once() {
        static std::string p;
        return p;
    }
    static std::string &no_voices() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&budget_tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_tts_model *&eos_tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_status speak(jllama_tts_model *tts, const char *text, std::vector<unsigned char> *wav,
                               const char *voice = nullptr, int seed = 7, float temperature = -1.0f) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 0;
        options.seed = seed;
        options.voice = voice;
        if (temperature >= 0.0f) {
            options.temperature = temperature;
        }
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts, text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    static uint32_t le32(const std::vector<unsigned char> &b, size_t at) {
        return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
    }
    // the frames of a 16-bit mono WAV, -1 when its samples are no whole number of frames
    static long frames_of(const std::vector<unsigned char> &wav) {
        const size_t samples = (wav.size() - 44) / 2;
        return samples % 1920 == 0 ? static_cast<long>(samples / 1920) : -1;
    }
};

TEST_F(CApiPocketTts, ItsPredefinedVoicesAreListedAndItClonesNone) {
    char **names = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tts_voices(budget_tts(), &names, &count), JLLAMA_STATUS_OK);
    ASSERT_EQ(count, 2u);
    EXPECT_STREQ(names[0], "alba");
    EXPECT_STREQ(names[1], "marius");
    jllama_strings_free(names, count);
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(budget_tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported) << "the english pack has no speaker encoder";
}

TEST_F(CApiPocketTts, ASynthesisIsAWavOfWholeFramesAt24kHz) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak(eos_tts(), "Hi.", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(std::string(wav.begin(), wav.begin() + 4), "RIFF");
    EXPECT_EQ(le32(wav, 24), 24000u);
    EXPECT_EQ(wav[22] | wav[23] << 8, 1) << "mono";
    EXPECT_EQ(frames_of(wav), 5) << "end of speech at the first frame, 5 frames after it for a short chunk";
    std::vector<unsigned char> again;
    ASSERT_EQ(speak(eos_tts(), "Hi.", &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same audio";
}

TEST_F(CApiPocketTts, ALongerChunkGetsThreeFramesAfterTheEndOfSpeech) {
    std::vector<unsigned char> wav;
    // "Hello." is 7 tokens already (the space piece, H, e, l, l, o, .): the pieces are lower case
    ASSERT_EQ(speak(eos_tts(), "Hello.", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(frames_of(wav), 3) << "more than 6 tokens: 3 frames after the end of speech";
}

TEST_F(CApiPocketTts, WithoutTheEndOfSpeechAChunkEndsAtItsBudget) {
    std::vector<unsigned char> short_text;
    std::vector<unsigned char> long_text;
    ASSERT_EQ(speak(budget_tts(), "Hi.", &short_text), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "hello world hello world hello world.", &long_text), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const long a = frames_of(short_text);
    const long b = frames_of(long_text);
    // "Hi." is 4 tokens (the space piece, H, i, .): ceil((4 / 3 + 2) * 12.5) = 42
    EXPECT_EQ(a, 42);
    EXPECT_GT(b, a) << "the budget grows with the text";
}

TEST_F(CApiPocketTts, ALongTextIsSpokenInChunksOfFiftyTokens) {
    // 60 sentences of two tokens each: the reference splits after the sentence ends
    // into chunks of at most 50 tokens, each re-prompted on the voice and ended by its own end of
    // speech -- 3 frames for a chunk of more than 6 tokens
    std::string text;
    for (int i = 0; i < 60; ++i) {
        text += "a.";
    }
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak(eos_tts(), text.c_str(), &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    // 121 tokens (the space piece, "A", then 60 times "." and "a" pairs): sentences merged greedily
    // up to 50 tokens give chunks of 49, 50 and 22, three ends of speech of 3 frames each
    EXPECT_EQ(frames_of(wav), 9);
}

TEST_F(CApiPocketTts, AVoiceChangesTheAudioAndTheFirstIsTheDefault) {
    std::vector<unsigned char> own;
    std::vector<unsigned char> alba;
    std::vector<unsigned char> marius;
    ASSERT_EQ(speak(budget_tts(), "Hi.", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi.", &alba, "alba"), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi.", &marius, "marius"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(own, alba) << "without a voice the first predefined one speaks";
    EXPECT_NE(alba, marius);
    EXPECT_EQ(speak(budget_tts(), "Hi.", nullptr, "nobody"), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("marius"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(budget_tts(), "Hi.", nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiPocketTts, TheSeedAndTheFlowTemperatureChangeTheAudio) {
    std::vector<unsigned char> seven;
    std::vector<unsigned char> eight;
    std::vector<unsigned char> cold;
    std::vector<unsigned char> cold_other_seed;
    ASSERT_EQ(speak(budget_tts(), "Hi.", &seven, nullptr, 7), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi.", &eight, nullptr, 8), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(seven, eight) << "the flow net starts from noise of the seed";
    ASSERT_EQ(speak(budget_tts(), "Hi.", &cold, nullptr, 7, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi.", &cold_other_seed, nullptr, 8, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(cold, seven) << "a temperature of 0 scales the noise away";
    EXPECT_EQ(cold, cold_other_seed) << "without noise the seed does not matter";
}

TEST_F(CApiPocketTts, APackWithoutVoicesSpeaksWithoutOne) {
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone().c_str(), no_voices().c_str(), 0, 2, &tts), JLLAMA_STATUS_OK)
        << jllama_last_error();
    char **names = nullptr;
    size_t count = 7;
    ASSERT_EQ(jllama_tts_voices(tts, &names, &count), JLLAMA_STATUS_OK);
    EXPECT_EQ(count, 0u);
    jllama_strings_free(names, count);
    std::vector<unsigned char> wav;
    EXPECT_EQ(speak(tts, "Hi.", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(frames_of(wav), 5);
    // without predefined voices a voice reaches only a "tts" template, which this pack has none of:
    // the name is not refused but left unused, as upstream's /tts, its realtime route and llama-tts do
    std::vector<unsigned char> named;
    EXPECT_EQ(speak(tts, "Hi.", &named, "alba"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(named, wav);
    jllama_tts_free(tts);
}

// A random Qwen3-TTS pair (tiny_llama_gguf.hpp), in the layout of CI's real 12Hz-1.7B-Base pair: the
// backbone samples the first code of a frame from its codec head, the code predictor the other three,
// code2wav turns them into 1920 samples a frame at 24 kHz. Without a speaker encoder it clones no voice.
class CApiQwen3Tts : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        backbone() = jllama_test::write_tiny_qwen3tts_gguf();
        mmproj() = jllama_test::write_tiny_qwen3tts_mmproj_gguf();
        ASSERT_FALSE(backbone().empty());
        ASSERT_FALSE(mmproj().empty());
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), mmproj().c_str(), 0, 2, &tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(tts());
        tts() = nullptr;
        std::remove(backbone().c_str());
        std::remove(mmproj().c_str());
    }
    static std::string &backbone() {
        static std::string p;
        return p;
    }
    static std::string &mmproj() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_status speak(const char *text, std::vector<unsigned char> *wav, int max_frames = 10,
                               unsigned seed = 7, const char *lang = nullptr) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = max_frames;
        options.seed = seed;
        options.lang = lang;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    static uint32_t le32(const std::vector<unsigned char> &b, size_t at) {
        return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
    }
    static long frames_of(const std::vector<unsigned char> &wav) {
        const size_t samples = (wav.size() - 44) / 2;
        return samples % 1920 == 0 ? static_cast<long>(samples / 1920) : -1;
    }
};

TEST_F(CApiQwen3Tts, ASynthesisIsAWavOfWholeFramesAt24kHz) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak("hello world", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(le32(wav, 24), 24000u);
    EXPECT_EQ(frames_of(wav), 10) << wav.size();
    EXPECT_EQ(wav[22] | wav[23] << 8, 1) << "mono";
    std::vector<unsigned char> again;
    ASSERT_EQ(speak("hello world", &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same audio";
    std::vector<unsigned char> other;
    ASSERT_EQ(speak("hello world", &other, 10, 8), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(wav, other) << "the backbone and the code predictor sample";
}

TEST_F(CApiQwen3Tts, Code2wavDecodesAcrossItsSeventyTwoFrameWindow) {
    // code2wav decodes 72 frames at a time and carries its convolution and attention state into the
    // next window: 72 frames are one window, 73 and 150 cross it, and the first 72 frames of a longer
    // synthesis are the same samples as a synthesis that stops there
    std::vector<unsigned char> one;
    std::vector<unsigned char> two;
    std::vector<unsigned char> three;
    ASSERT_EQ(speak("hello world", &one, 72), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &two, 73), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &three, 150), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(frames_of(one), 72);
    EXPECT_EQ(frames_of(two), 73);
    EXPECT_EQ(frames_of(three), 150);
    ASSERT_GE(three.size(), one.size());
    EXPECT_TRUE(std::equal(one.begin() + 44, one.end(), three.begin() + 44))
        << "the window boundary must not change the samples before it";
}

TEST_F(CApiQwen3Tts, TheLanguageIsATokenOfThePromptAndAnUnknownOneFails) {
    std::vector<unsigned char> own;
    std::vector<unsigned char> english;
    std::vector<unsigned char> en;
    std::vector<unsigned char> german;
    ASSERT_EQ(speak("hello world", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &english, 10, 7, "english"), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &en, 10, 7, "en"), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &german, 10, 7, "de"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(own, english) << "without a language this library sends \"english\"";
    EXPECT_EQ(en, english) << "llama.cpp maps the code to the name";
    EXPECT_NE(german, english) << "another <|codec_language_*|> token in the prompt";
    // a language the vocabulary has no token for fails in set_input; llama.cpp's /tts answers a
    // server error too (ERROR_TYPE_SERVER), not the request's fault
    EXPECT_EQ(speak("hello world", nullptr, 10, 7, "klingon"), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(speak("hello world", nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiQwen3Tts, WithTopKOneTheSeedDoesNotMatter) {
    // top_k reaches the backbone sampler and the code predictor: 1 makes both greedy
    const auto greedy = [](unsigned seed, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 10;
        options.seed = seed;
        options.top_k = 1;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "hello world", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    std::vector<unsigned char> a;
    std::vector<unsigned char> b;
    ASSERT_EQ(greedy(7, &a), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(greedy(8, &b), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(a, b);
    std::vector<unsigned char> sampled;
    ASSERT_EQ(speak("hello world", &sampled), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(a, sampled);
}

TEST_F(CApiQwen3Tts, ItClonesNoVoiceAndATranscriptAloneChangesNothing) {
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported) << "the mmproj has no speaker encoder";
    char **names = nullptr;
    size_t count = 7;
    ASSERT_EQ(jllama_tts_voices(tts(), &names, &count), JLLAMA_STATUS_OK);
    EXPECT_EQ(count, 0u);
    jllama_strings_free(names, count);
    // the transcript of b12491 (#32438) continues a reference; without one it is not used
    std::vector<unsigned char> own;
    ASSERT_EQ(speak("hello world", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 10;
    options.seed = 7;
    options.speaker_reference_text = "what the reference says";
    unsigned char *bytes = nullptr;
    size_t length = 0;
    ASSERT_EQ(jllama_tts_synthesize_options(tts(), "hello world", &options, &bytes, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + length), own);
    jllama_bytes_free(bytes);
}

// The same backbone with the speaker encoder in its mmproj, as CI's real pair: it clones a voice from a
// clip. The clip becomes one row of the prompt (the speaker embedding), so a reference changes the
// audio of a seed, and two references change it differently.
TEST(CApiQwen3TtsClone, AReferenceClipClonesTheVoice) {
    const std::string backbone = jllama_test::write_tiny_qwen3tts_gguf("jllama-qwen3tts-clone");
    const std::string mmproj = jllama_test::write_tiny_qwen3tts_mmproj_gguf("jllama-qwen3tts-clone-mmproj", true);
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    bool supported = false;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts, &supported), JLLAMA_STATUS_OK);
    EXPECT_TRUE(supported);
    const auto clip = [](float hz) {
        std::vector<float> tone(24000);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.4f * std::sin(2.0f * 3.14159265f * hz * static_cast<float>(i) / 24000.0f);
        }
        const std::string path = (std::filesystem::temp_directory_path() /
                                  ("jllama-qwen3tts-ref-" + std::to_string(static_cast<int>(hz)) + ".wav"))
                                     .string();
        const std::vector<uint8_t> wav = jllama_tts::pcm_to_wav16_bytes(tone, 24000);
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(wav.data()), wav.size());
        return path;
    };
    const std::string low = clip(220.0f);
    const std::string high = clip(880.0f);
    const auto speak = [&](const char *reference, std::vector<unsigned char> *wav, const char *transcript = nullptr) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        // greedy over 30 frames: a random backbone of two layers moves its sampling distribution only a
        // little for one more prompt row (measured: the row reaches the prompt, 22 rows instead of 21,
        // and the seeded draws of top-k 4 stayed the same), while an argmax over 30 frames of four codes
        // flips as soon as the hidden state moves
        options.max_frames = 30;
        options.top_k = 1;
        options.seed = 7;
        options.speaker_reference_path = reference;
        options.speaker_reference_text = transcript;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts, "hello world", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    std::vector<unsigned char> own;
    std::vector<unsigned char> cloned_low;
    std::vector<unsigned char> cloned_low_again;
    std::vector<unsigned char> cloned_high;
    ASSERT_EQ(speak(nullptr, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(low.c_str(), &cloned_low), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(low.c_str(), &cloned_low_again), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(high.c_str(), &cloned_high), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(cloned_low, own) << "the speaker embedding is a row of the prompt";
    EXPECT_EQ(cloned_low, cloned_low_again);
    EXPECT_NE(cloned_low, cloned_high) << "another clip, another speaker embedding";
    // the transcript (b12491, #32438) needs the reference encoder this mmproj lacks: llama.cpp warns and
    // clones from the speaker embedding alone
    std::vector<unsigned char> with_transcript;
    ASSERT_EQ(speak(low.c_str(), &with_transcript, "what the clip says"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(with_transcript, cloned_low);
    // a file that is no audio is the request's fault, and the model answers the next one
    EXPECT_EQ(speak(backbone.c_str(), nullptr), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("speaker reference"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(nullptr, nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tts_free(tts);
    for (const std::string &p : {backbone, mmproj, low, high}) {
        std::remove(p.c_str());
    }
}

// A random LFM3-Audio pair (tiny_llama_gguf.hpp), in the layout of CI's real 2B pair: the "tts"
// template renders the prompt, the backbone's hidden state seeds three MaskGIT passes of the
// depthformer over the 12 codebooks, the codec decoder turns each frame into six STFT frames and the
// host ISTFT into 1920 samples at 24 kHz. A frame with the end-of-audio code in any codebook ends it.
class CApiLfm3Tts : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        backbone() = jllama_test::write_tiny_lfm3_gguf();
        mmproj() = jllama_test::write_tiny_lfm3_mmproj_gguf();
        ASSERT_FALSE(backbone().empty());
        ASSERT_FALSE(mmproj().empty());
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), mmproj().c_str(), 0, 2, &tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(tts());
        tts() = nullptr;
        std::remove(backbone().c_str());
        std::remove(mmproj().c_str());
    }
    static std::string &backbone() {
        static std::string p;
        return p;
    }
    static std::string &mmproj() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_status speak(const char *text, std::vector<unsigned char> *wav, int max_frames = 10,
                               unsigned seed = 7, const char *lang = nullptr, const char *voice = nullptr,
                               float temperature = -1.0f) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = max_frames;
        options.seed = seed;
        options.lang = lang;
        options.voice = voice;
        if (temperature >= 0.0f) {
            options.temperature = temperature;
        }
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    static uint32_t le32(const std::vector<unsigned char> &b, size_t at) {
        return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
    }
};

TEST_F(CApiLfm3Tts, AFrameIs1920SamplesAt24kHzAcrossTheDecoderWindow) {
    // greedy: the end-of-audio row has the logit 0, which never beats 16 random codes, so the frame cap
    // decides the length; 31 frames cross the codec decoder's 30-frame window
    for (int frames : {1, 2, 10, 31}) {
        std::vector<unsigned char> wav;
        ASSERT_EQ(speak("hello world", &wav, frames, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(le32(wav, 24), 24000u);
        EXPECT_EQ(wav[22] | wav[23] << 8, 1) << "mono";
        EXPECT_EQ((wav.size() - 44) / 2, static_cast<size_t>(frames) * 1920u) << frames;
    }
    // the decoder carries its window and the ISTFT its overlap: a frame's samples are final once the
    // next one is decoded, so 29 frames of a 30-frame and a 31-frame synthesis are the same
    std::vector<unsigned char> thirty;
    std::vector<unsigned char> thirty_one;
    ASSERT_EQ(speak("hello world", &thirty, 30, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &thirty_one, 31, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const size_t settled = 44 + 29u * 1920u * 2u;
    // (after the header, which carries the length)
    EXPECT_TRUE(std::equal(thirty.begin() + 44, thirty.begin() + settled, thirty_one.begin() + 44));
}

TEST_F(CApiLfm3Tts, TheSeedAndTheTemperatureDecideTheCodes) {
    std::vector<unsigned char> greedy_seven;
    std::vector<unsigned char> greedy_eight;
    ASSERT_EQ(speak("hello world", &greedy_seven, 10, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(speak("hello world", &greedy_eight, 10, 8, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK)
        << jllama_last_error();
    // the seed also shuffles the order MaskGIT reveals the codebooks in, which greedy codes depend on
    std::vector<unsigned char> warm_seven;
    std::vector<unsigned char> warm_seven_again;
    ASSERT_EQ(speak("hello world", &warm_seven, 10, 7, nullptr, nullptr, 0.3f), JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(speak("hello world", &warm_seven_again, 10, 7, nullptr, nullptr, 0.3f), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(warm_seven, warm_seven_again) << "the same seed gives the same audio";
    EXPECT_NE(warm_seven, greedy_seven) << "a temperature samples";
}

TEST_F(CApiLfm3Tts, TheTemplateNamesTheLanguageAndTheVoice) {
    // the model has the two voices of its template as predefined voices (llama.cpp hardcodes them):
    // no voice is the first, female; male renders another system turn; another name is refused
    char **names = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tts_voices(tts(), &names, &count), JLLAMA_STATUS_OK);
    ASSERT_EQ(count, 2u);
    EXPECT_STREQ(names[0], "female");
    EXPECT_STREQ(names[1], "male");
    jllama_strings_free(names, count);
    std::vector<unsigned char> own;
    std::vector<unsigned char> female;
    std::vector<unsigned char> male;
    std::vector<unsigned char> german;
    ASSERT_EQ(speak("hello world", &own, 30, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &female, 30, 7, nullptr, "female", 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &male, 30, 7, nullptr, "male", 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("hello world", &german, 30, 7, "de", nullptr, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(own, female);
    EXPECT_NE(male, female);
    EXPECT_NE(german, own) << "another language in the system turn";
    EXPECT_EQ(speak("hello world", nullptr, 10, 7, nullptr, "robot"), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("male"), std::string::npos) << jllama_last_error();
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported);
}

TEST_F(CApiLfm3Tts, ARawPromptSkipsTheTemplate) {
    // b12494's raw_prompt: the text goes to the model as it is. The prompt the template renders, given
    // raw, is the same synthesis; the bare text ends in text instead of <|audio_start|> and cannot speak
    const auto speak_raw = [](const char *text, int raw, size_t struct_size, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 10;
        options.seed = 7;
        options.temperature = 0.0f;
        options.raw_prompt = raw;
        options.struct_size = struct_size;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    std::vector<unsigned char> templated;
    ASSERT_EQ(speak("hello world", &templated, 10, 7, nullptr, nullptr, 0.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string rendered = "<|startoftext|><|im_start|>system\nSpeak English with a female voice.<|im_end|>\n"
                                 "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n<|audio_start|>";
    std::vector<unsigned char> raw;
    ASSERT_EQ(speak_raw(rendered.c_str(), 1, sizeof(jllama_tts_options), &raw), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(raw, templated) << "the rendered prompt given raw is the same synthesis";
    EXPECT_EQ(speak_raw("hello world", 1, sizeof(jllama_tts_options), nullptr), JLLAMA_STATUS_ERROR)
        << "the bare text does not end in <|audio_start|>";
    // a caller of the header before b12494 declares 80 bytes: the field behind them is not read
    std::vector<unsigned char> older;
    ASSERT_EQ(speak_raw("hello world", 1, offsetof(jllama_tts_options, raw_prompt), &older), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(older, templated);
}

TEST(CApiLfm3TtsEnd, TheEndOfAudioAtTheFirstFrameLeavesNoAudio) {
    const std::string backbone = jllama_test::write_tiny_lfm3_gguf("jllama-lfm3-end");
    const std::string mmproj = jllama_test::write_tiny_lfm3_mmproj_gguf("jllama-lfm3-end-mmproj", 100.0f);
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 10;
    options.temperature = 0.0f;
    unsigned char *bytes = nullptr;
    size_t length = 0;
    // an end of audio is no frame: the first one leaves nothing to return, which the engine reports
    EXPECT_EQ(jllama_tts_synthesize_options(tts, "hello world", &options, &bytes, &length), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("no audio"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(bytes, nullptr);
    jllama_tts_free(tts);
    std::remove(backbone.c_str());
    std::remove(mmproj.c_str());
}

// The audio input half of the LFM3-Audio projector (lfm3a): the same random pair, its mmproj with the
// encoder as well (write_tiny_lfm3_mmproj_gguf's audio_encoder), loaded as a model that hears. The
// encoder downsamples the log-mel frames 8x in time, so a clip costs ((((frames + 1) / 2) + 1) / 2 + 1)
// / 2 embeddings, its frames being the 400-sample windows of a 160-sample hop over the clip padded on
// the right to a whole hop (mtmd_audio_preprocessor_lfm3a, no centre padding).
class CApiLfm3Audio : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        backbone() = jllama_test::write_tiny_lfm3_gguf("jllama-lfm3-hear", true, true);
        // greedy audio codes, and an end of audio that never wins: an audio answer runs to its token limit
        mmproj() = jllama_test::write_tiny_lfm3_mmproj_gguf("jllama-lfm3-hear-mmproj", -10.0f, true, 0.0f);
        ASSERT_FALSE(backbone().empty());
        ASSERT_FALSE(mmproj().empty());
        // a realtime session samples with the model's seed, random by default
        model() =
            load_tiny(backbone(), {"--mmproj", mmproj(), "-c", "1024", "-b", "1024", "-ub", "1024", "--seed", "5"});
        ASSERT_NE(model(), nullptr) << jllama_last_error();
        Owned meta;
        ASSERT_EQ(jllama_model_meta(model(), &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
        marker() = json::parse(meta.str()).value("media_marker", std::string());
    }
    static void TearDownTestSuite() {
        jllama_model_free(model());
        model() = nullptr;
        std::remove(backbone().c_str());
        std::remove(mmproj().c_str());
    }
    static std::string &backbone() {
        static std::string p;
        return p;
    }
    static std::string &mmproj() {
        static std::string p;
        return p;
    }
    static std::string &marker() {
        static std::string m;
        return m;
    }
    static jllama_model *&model() {
        static jllama_model *m = nullptr;
        return m;
    }
    static std::string wav_of(size_t samples, float frequency) {
        std::vector<float> tone(samples);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * frequency * static_cast<float>(i) / 16000.0f);
        }
        return base64_of(jllama_tts::pcm_to_wav16_bytes(tone, 16000));
    }
    // the prompt with one clip, greedy, with the top probabilities of the first token
    static json hear(const std::string &wav, int n_predict = 1) {
        const json request = {
            {"prompt", {{"prompt_string", "hear " + marker() + " now"}, {"multimodal_data", json::array({wav})}}},
            {"n_predict", n_predict},
            {"temperature", 0},
            {"n_probs", 5},
            {"cache_prompt", false}};
        Owned response;
        EXPECT_EQ(jllama_completions(model(), request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()) : json::object();
    }
    static int embeddings_of(size_t samples) {
        const size_t win = 400, hop = 160;
        const size_t padded = samples <= win ? win : samples + (hop - (samples - win) % hop) % hop;
        const int frames = static_cast<int>((padded - win) / hop + 1);
        return ((((frames + 1) / 2) + 1) / 2 + 1) / 2;
    }
};

TEST_F(CApiLfm3Audio, TheModelReportsThatItHears) {
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model(), &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json m = json::parse(meta.str());
    EXPECT_TRUE(m.at("modalities").value("audio", false)) << meta.str();
    EXPECT_FALSE(m.at("modalities").value("vision", true)) << meta.str();
    EXPECT_FALSE(marker().empty());
}

TEST_F(CApiLfm3Audio, AClipCostsItsFramesDownsampledEightTimes) {
    Owned text_only;
    ASSERT_EQ(jllama_completions(model(), R"({"prompt":"hear  now","n_predict":1,"temperature":0})", &text_only.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int text_tokens = json::parse(text_only.str()).at("tokens_evaluated").get<int>();
    // 0.025 s is one window, 0.5 s 49 frames, 3 s 299 frames: 38 positions, more than the sliding
    // window of the attention layer, and the conv and window states of the stream are crossed
    for (const size_t samples : {size_t{400}, size_t{8000}, size_t{16000}, size_t{48000}}) {
        const json answer = hear(wav_of(samples, 440.0f));
        EXPECT_EQ(answer.value("tokens_evaluated", -1), text_tokens + embeddings_of(samples))
            << samples << " samples: " << answer.dump();
    }
    EXPECT_EQ(embeddings_of(8000), 7);
    EXPECT_EQ(embeddings_of(48000), 38);
}

TEST_F(CApiLfm3Audio, TheSameClipIsHeardTheSameAndAnotherDifferently) {
    const json a = hear(wav_of(16000, 440.0f));
    const json again = hear(wav_of(16000, 440.0f));
    const json other = hear(wav_of(16000, 1500.0f));
    ASSERT_TRUE(a.contains("completion_probabilities")) << a.dump();
    const json probs = a.at("completion_probabilities");
    EXPECT_EQ(again.at("completion_probabilities"), probs);
    EXPECT_NE(other.at("completion_probabilities"), probs)
        << "the encoder's output reaches the backbone, so another tone gives other probabilities";
}

TEST_F(CApiLfm3Audio, AnImageIsRefusedByAProjectorThatOnlyHears) {
    const json request = {{"prompt",
                           {{"prompt_string", "see " + marker() + " now"},
                            {"multimodal_data", json::array({jllama_test::TINY_PNG_BASE64})}}},
                          {"n_predict", 1}};
    Owned response;
    EXPECT_NE(jllama_completions(model(), request.dump().c_str(), &response.s), JLLAMA_STATUS_OK);
    // and the model hears the next clip
    EXPECT_GT(hear(wav_of(8000, 440.0f)).value("tokens_evaluated", -1), 0);
}

TEST_F(CApiLfm3Audio, TheEncoderInTheSameFileLeavesTheSpeechAsItWas) {
    // the real mmproj holds both halves; the speech pipeline reads the generator's tensors only, and
    // the generator writes them before the encoder's, from the same random sequence
    const std::string speak_only =
        jllama_test::write_tiny_lfm3_mmproj_gguf("jllama-lfm3-hear-speak", -10.0f, false, 0.0f);
    const auto speak = [&](const std::string &projector, std::vector<unsigned char> *wav) {
        jllama_tts_model *tts = nullptr;
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), projector.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 10;
        options.seed = 7;
        options.temperature = 0.0f;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        ASSERT_EQ(jllama_tts_synthesize_options(tts, "hello world", &options, &bytes, &length), JLLAMA_STATUS_OK)
            << jllama_last_error();
        wav->assign(bytes, bytes + length);
        jllama_bytes_free(bytes);
        jllama_tts_free(tts);
    };
    std::vector<unsigned char> both;
    std::vector<unsigned char> alone;
    speak(mmproj(), &both);
    speak(speak_only, &alone);
    EXPECT_FALSE(both.empty());
    EXPECT_EQ(both, alone);
    std::remove(speak_only.c_str());
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
namespace {
// the attached server of a model on a free loopback port, up once /health answers
struct AttachedServer {
    jllama_server *server = nullptr;
    int port = 0;
    explicit AttachedServer(jllama_model *model) {
        httplib::Server probe;
        port = probe.bind_to_any_port("127.0.0.1");
        std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
        probe.wait_until_ready();
        probe.stop();
        probe_listener.join();
        const std::string port_text = std::to_string(port);
        const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", port_text.c_str()};
        EXPECT_EQ(jllama_server_start_attached(model, 5, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
        httplib::Client client("127.0.0.1", port);
        for (int i = 0; i < 100; i++) {
            const auto health = client.Get("/health");
            if (health && health->status == 200) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ~AttachedServer() {
        if (server != nullptr) {
            EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
        }
    }
};
} // namespace

TEST_F(CApiLfm3Audio, TheTranscriptionRouteAsksTheLfm3AsrPromptInTheLanguage) {
    // /v1/audio/transcriptions builds a chat request from an LFM3 template's ASR preset: the system turn
    // "Perform ASR in German." and the clip as the user turn (common_chat_get_asr_prompt). The route's
    // text is therefore the chat answer to exactly those messages.
    std::vector<float> tone(16000);
    for (size_t i = 0; i < tone.size(); ++i) {
        tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
    }
    const std::vector<uint8_t> wav = jllama_tts::pcm_to_wav16_bytes(tone, 16000);
    std::string transcript;
    {
        AttachedServer attached(model());
        ASSERT_NE(attached.server, nullptr);
        httplib::Client client("127.0.0.1", attached.port);
        client.set_read_timeout(std::chrono::seconds(60));
        const httplib::UploadFormDataItems items = {
            {"file", std::string(wav.begin(), wav.end()), "clip.wav", "audio/wav"},
            {"language", "de", "", ""},
            {"temperature", "0", "", ""},
            {"max_tokens", "6", "", ""},
        };
        const auto response = client.Post("/v1/audio/transcriptions", items);
        ASSERT_TRUE(response) << httplib::to_string(response.error());
        ASSERT_EQ(response->status, 200) << response->body;
        transcript = json::parse(response->body).at("text").get<std::string>();
        EXPECT_FALSE(transcript.empty()) << response->body;
        // the route refuses another format than json, as upstream does
        const httplib::UploadFormDataItems srt = {
            {"file", std::string(wav.begin(), wav.end()), "clip.wav", "audio/wav"}, {"response_format", "srt", "", ""}};
        const auto refused = client.Post("/v1/audio/transcriptions", srt);
        ASSERT_TRUE(refused) << httplib::to_string(refused.error());
        EXPECT_EQ(refused->status, 400) << refused->body;
    }
    const json chat = {
        {"messages",
         json::array({{{"role", "system"}, {"content", "Perform ASR in German."}},
                      {{"role", "user"},
                       {"content", json::array({{{"type", "input_audio"},
                                                 {"input_audio", {{"data", base64_of(wav)}, {"format", "wav"}}}}})}}})},
        {"temperature", 0},
        {"max_tokens", 6},
        {"cache_prompt", false}};
    Owned answer;
    ASSERT_EQ(jllama_chat_completions(model(), chat.dump().c_str(), &answer.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(answer.str()).at("choices").at(0).at("message").at("content").get<std::string>(), transcript)
        << answer.str();
}

TEST_F(CApiLfm3Audio, LiveTranscriptionHearsTheStreamedAudioAndCompletesTheItem) {
    // /v1/realtime?intent=transcription with the "asr" template: audio appended as 16-bit PCM, the
    // buffer committed as final, and the item completed with the text of its deltas
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    httplib::ws::WebSocketClient ws("ws://127.0.0.1:" + std::to_string(attached.port) +
                                    "/v1/realtime?intent=transcription");
    ws.set_read_timeout(std::chrono::seconds(60));
    ASSERT_TRUE(ws.connect());
    std::string message;
    ASSERT_EQ(ws.read(message), httplib::ws::ReadResult::Text);
    const json created = json::parse(message);
    ASSERT_EQ(created.value("type", std::string()), "session.created") << message;
    EXPECT_EQ(created.at("session").at("audio").at("input").at("format").at("rate").get<int>(), 16000) << message;

    std::string pcm;
    for (int i = 0; i < 16000; ++i) {
        const auto v = static_cast<int16_t>(16000.0f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f));
        pcm.append(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    ASSERT_TRUE(ws.send(
        json{{"type", "input_audio_buffer.append"}, {"audio", base64_of(std::vector<uint8_t>(pcm.begin(), pcm.end()))}}
            .dump()));
    ASSERT_TRUE(ws.send(json{{"type", "input_audio_buffer.commit"}, {"final", true}}.dump()));

    std::string deltas;
    json completed;
    while (completed.is_null()) {
        message.clear();
        const auto read = ws.read(message);
        ASSERT_EQ(read, httplib::ws::ReadResult::Text) << "deltas so far: " << deltas;
        const json event = json::parse(message);
        const std::string type = event.value("type", std::string());
        ASSERT_NE(type, "error") << message;
        if (type == "conversation.item.input_audio_transcription.delta") {
            deltas += event.at("delta").get<std::string>();
        } else if (type == "conversation.item.input_audio_transcription.completed") {
            completed = event;
        }
    }
    EXPECT_FALSE(deltas.empty()) << "the model answered the audio in text";
    EXPECT_EQ(completed.at("transcript").get<std::string>(), deltas) << completed.dump();
    ws.close();
}
TEST_F(CApiLfm3Audio, ALiveTranscriptionPromptGoesInBeforeTheAudio) {
    // a realtime session switched to the type "transcription" (session.update; ?intent=transcription
    // with an "asr" template takes the slot's own live transcription instead) builds its instruction
    // as /v1/audio/transcriptions does (b12511, #32507): the LFM3 preset has no user text, so
    // session.audio.input.transcription.prompt is the user text, and it goes in before the audio.
    // The session reports it, and the same clip is answered differently with it than without it;
    // without a prompt the answer is the same every time, so the difference is the prompt's.
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    std::string pcm;
    for (int i = 0; i < 16000; ++i) {
        const auto v = static_cast<int16_t>(16000.0f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f));
        pcm.append(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    const std::string audio = base64_of(std::vector<uint8_t>(pcm.begin(), pcm.end()));
    const auto transcribe = [&](const std::string &prompt, std::string *deltas) {
        httplib::ws::WebSocketClient ws("ws://127.0.0.1:" + std::to_string(attached.port) + "/v1/realtime");
        ws.set_read_timeout(std::chrono::seconds(60));
        ASSERT_TRUE(ws.connect());
        std::string message;
        ASSERT_EQ(ws.read(message), httplib::ws::ReadResult::Text);
        json transcription = json::object();
        if (!prompt.empty()) {
            transcription["prompt"] = prompt;
        }
        ASSERT_TRUE(ws.send(json{{"type", "session.update"},
                                 {"session",
                                  {{"type", "transcription"},
                                   {"max_output_tokens", 8},
                                   {"audio", {{"input", {{"transcription", transcription}}}}}}}}
                                .dump()));
        json updated;
        while (updated.is_null()) {
            message.clear();
            ASSERT_EQ(ws.read(message), httplib::ws::ReadResult::Text);
            const json event = json::parse(message);
            ASSERT_NE(event.value("type", std::string()), "error") << message;
            if (event.value("type", std::string()) == "session.updated") {
                updated = event;
            }
        }
        EXPECT_EQ(updated.at("session").at("audio").at("input").at("transcription").value("prompt", std::string("?")),
                  prompt)
            << updated.dump();
        ASSERT_TRUE(ws.send(json{{"type", "input_audio_buffer.append"}, {"audio", audio}}.dump()));
        ASSERT_TRUE(ws.send(json{{"type", "input_audio_buffer.commit"}}.dump()));
        json completed;
        while (completed.is_null()) {
            message.clear();
            ASSERT_EQ(ws.read(message), httplib::ws::ReadResult::Text) << "deltas so far: " << *deltas;
            const json event = json::parse(message);
            const std::string type = event.value("type", std::string());
            ASSERT_NE(type, "error") << message;
            if (type == "conversation.item.input_audio_transcription.delta") {
                *deltas += event.at("delta").get<std::string>();
            } else if (type == "conversation.item.input_audio_transcription.completed") {
                completed = event;
            }
        }
        ws.close();
    };
    std::string plain;
    std::string again;
    std::string prompted;
    transcribe("", &plain);
    transcribe("", &again);
    transcribe("Names that occur in the recording: Ladenthin, Gerganov, Lindqvist, Okonkwo.", &prompted);
    EXPECT_FALSE(plain.empty());
    EXPECT_EQ(plain, again) << "without a prompt the answer does not change";
    EXPECT_NE(prompted, plain) << "the prompt reached the model";
}
namespace {
// a /v1/realtime session over the attached server: events in, events out, errors kept
struct RealtimeSession {
    httplib::ws::WebSocketClient ws;
    explicit RealtimeSession(int port, const std::string &query = "")
        : ws("ws://127.0.0.1:" + std::to_string(port) + "/v1/realtime" + query) {
        ws.set_read_timeout(std::chrono::seconds(60));
    }
    bool open() { return static_cast<bool>(ws.connect()); }
    bool send(const json &event) { return ws.send(event.dump()); }
    // the next event, or a null json when the socket closed or timed out
    json next() {
        std::string message;
        if (ws.read(message) != httplib::ws::ReadResult::Text) {
            return json();
        }
        return json::parse(message);
    }
    // the next event of a type; an error event in between is returned instead
    json await(const std::string &type) {
        for (;;) {
            json event = next();
            if (event.is_null() || event.value("type", std::string()) == type ||
                event.value("type", std::string()) == "error") {
                return event;
            }
        }
    }
    void user_text(const std::string &text) {
        send({{"type", "conversation.item.create"},
              {"item",
               {{"type", "message"},
                {"role", "user"},
                {"content", json::array({{{"type", "input_text"}, {"text", text}}})}}}});
    }
    void response_create(const std::string &modality, int max_tokens) {
        send({{"type", "response.create"},
              {"response", {{"output_modalities", json::array({modality})}, {"max_output_tokens", max_tokens}}}});
    }
    struct response {
        std::string text;      // the text deltas
        std::string text_done; // response.output_text.done
        size_t audio_bytes = 0;
        json done;
        std::vector<std::string> types;
    };
    response collect() {
        response r;
        for (;;) {
            json event = next();
            if (event.is_null()) {
                return r;
            }
            const std::string type = event.value("type", std::string());
            r.types.push_back(type);
            if (type == "response.output_text.delta") {
                r.text += event.at("delta").get<std::string>();
            } else if (type == "response.output_text.done") {
                r.text_done = event.value("text", std::string());
            } else if (type == "response.output_audio.delta") {
                r.audio_bytes += base64_bytes(event.at("delta").get<std::string>()).size();
            } else if (type == "response.done" || type == "error") {
                r.done = event;
                return r;
            }
        }
    }
};
} // namespace

TEST_F(CApiLfm3Audio, ARealtimeSessionReportsItsFormatsAndAnswersATextMessageInText) {
    // /v1/realtime without intent=transcription: the model hears and speaks, so the server holds a
    // conversation (server_realtime_conversation)
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    RealtimeSession session(attached.port);
    ASSERT_TRUE(session.open());
    const json created = session.await("session.created");
    ASSERT_EQ(created.value("type", std::string()), "session.created") << created.dump();
    EXPECT_EQ(created.at("session").value("type", std::string()), "realtime") << created.dump();
    const json audio = created.at("session").at("audio");
    EXPECT_EQ(audio.at("input").at("format").at("rate").get<int>(), 16000) << created.dump();
    EXPECT_EQ(audio.at("output").at("format").at("rate").get<int>(), 24000) << created.dump();

    session.user_text("hello there");
    session.response_create("text", 6);
    const auto r = session.collect();
    ASSERT_EQ(r.done.value("type", std::string()), "response.done") << r.done.dump();
    const json response = r.done.at("response");
    EXPECT_EQ(response.value("status", std::string()), "completed") << r.done.dump();
    const int tokens = response.at("usage").at("output_tokens").get<int>();
    EXPECT_GT(tokens, 0);
    EXPECT_LE(tokens, 6) << "max_output_tokens";
    EXPECT_FALSE(r.text.empty());
    EXPECT_EQ(r.text_done, r.text) << "output_text.done carries the text of its deltas";
    EXPECT_EQ(r.audio_bytes, 0u) << "a text answer";
    // a second turn continues the conversation
    session.user_text("and again");
    session.response_create("text", 4);
    const auto second = session.collect();
    EXPECT_EQ(second.done.value("type", std::string()), "response.done") << second.done.dump();
    EXPECT_FALSE(second.text.empty());
}

TEST_F(CApiLfm3Audio, ARealtimeAudioResponseStreamsPcm16) {
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    RealtimeSession session(attached.port);
    ASSERT_TRUE(session.open());
    session.await("session.created");
    session.user_text("hello there");
    session.response_create("audio", 6);
    const auto r = session.collect();
    ASSERT_EQ(r.done.value("type", std::string()), "response.done") << r.done.dump();
    const json response = r.done.at("response");
    EXPECT_EQ(response.value("status", std::string()), "completed") << r.done.dump();
    // the pipeline opens the answer with <|audio_start|>; the generated projector never ends the audio, so
    // the answer is its six frames of 1920 samples at 24 kHz, as PCM16, and no text
    EXPECT_EQ(response.at("usage").at("output_tokens").get<int>(), 6) << r.done.dump();
    EXPECT_EQ(r.audio_bytes, 6u * 1920 * 2);
    EXPECT_TRUE(r.text.empty()) << r.text;
    const json content = response.at("output").at(0).at("content");
    ASSERT_EQ(content.size(), 1u) << content.dump();
    EXPECT_EQ(content.at(0).value("type", std::string()), "output_audio") << content.dump();
}

TEST_F(CApiLfm3Audio, ARealtimeSessionAnswersSpokenInput) {
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    RealtimeSession session(attached.port);
    ASSERT_TRUE(session.open());
    session.await("session.created");
    std::string pcm;
    for (int i = 0; i < 8000; ++i) {
        const auto v = static_cast<int16_t>(16000.0f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f));
        pcm.append(reinterpret_cast<const char *>(&v), sizeof(v));
    }
    ASSERT_TRUE(session.send(
        {{"type", "input_audio_buffer.append"}, {"audio", base64_of(std::vector<uint8_t>(pcm.begin(), pcm.end()))}}));
    ASSERT_TRUE(session.send({{"type", "input_audio_buffer.commit"}}));
    const json committed = session.await("input_audio_buffer.committed");
    ASSERT_EQ(committed.value("type", std::string()), "input_audio_buffer.committed") << committed.dump();
    session.response_create("text", 4);
    const auto r = session.collect();
    ASSERT_EQ(r.done.value("type", std::string()), "response.done") << r.done.dump();
    EXPECT_EQ(r.done.at("response").value("status", std::string()), "completed") << r.done.dump();
    EXPECT_FALSE(r.text.empty());
}

TEST_F(CApiLfm3Audio, ARealtimeSessionRefusesAResponseWithoutATurnAndUnknownEvents) {
    AttachedServer attached(model());
    ASSERT_NE(attached.server, nullptr);
    RealtimeSession session(attached.port);
    ASSERT_TRUE(session.open());
    session.await("session.created");
    session.response_create("text", 4);
    json error = session.await("error");
    // upstream's realtime server names the reason in the error's type, its code is null
    EXPECT_EQ(error.at("error").value("type", std::string()), "conversation_empty") << error.dump();
    session.send({{"type", "no.such.event"}});
    error = session.await("error");
    EXPECT_NE(error.at("error").value("message", std::string()).find("unknown event type"), std::string::npos)
        << error.dump();
    session.user_text("hello");
    session.send({{"type", "response.create"}, {"response", {{"output_modalities", json::array({"video"})}}}});
    error = session.await("error");
    EXPECT_NE(error.at("error").value("message", std::string()).find("unknown modality"), std::string::npos)
        << error.dump();
    // and the session still answers
    session.response_create("text", 2);
    EXPECT_EQ(session.collect().done.value("type", std::string()), "response.done");
}

#endif // JLLAMA_HAS_NATIVE_SERVER

// tiny-voca (b12497, #32427), from the schema of ggml-org's real pair: not autoregressive -- the
// encoder takes one sentence, the duration head gives every token its frames, two stacks and a
// source-filter synthesizer turn them into 256 samples a frame at 24 kHz. The generator's duration head
// gives every token the same frames (write_tiny_voca_mmproj_gguf), so a length is exact: the prompt's
// phonemes plus <bos>, the full stop the pipeline adds when the text has no final mark, and <eos>.
namespace {
struct TinyVocaPair {
    std::string backbone;
    std::string mmproj;
    jllama_tts_model *tts = nullptr;
    TinyVocaPair(const std::string &tts_template, int frames, float presence, int g2p_symbol) {
        backbone = jllama_test::write_tiny_voca_gguf("jllama-tiny-voca", tts_template);
        mmproj = jllama_test::write_tiny_voca_mmproj_gguf("jllama-tiny-voca-mmproj", frames, presence, g2p_symbol);
        EXPECT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    ~TinyVocaPair() {
        jllama_tts_free(tts);
        std::remove(backbone.c_str());
        std::remove(mmproj.c_str());
    }
    jllama_status speak(const char *text, std::vector<unsigned char> *wav, float speed = 1.0f,
                        const char *voice = nullptr, unsigned seed = 7, int raw_prompt = 0) const {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.seed = seed;
        options.voice = voice;
        options.speed = speed;
        options.raw_prompt = raw_prompt;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts, text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
};
size_t wav_samples(const std::vector<unsigned char> &wav) { return wav.size() < 44 ? 0 : (wav.size() - 44) / 2; }
uint32_t wav_rate(const std::vector<unsigned char> &wav) {
    return wav.size() < 28 ? 0 : wav[24] | (wav[25] << 8) | (wav[26] << 16) | (static_cast<uint32_t>(wav[27]) << 24);
}
const char *const TINY_VOCA_HELLO = "h \xC9\x99 l o\xCA\x8A"; // h ə l oʊ, four phonemes
} // namespace

TEST(CApiTinyVoca, ItsVoicesAreListedTheDefaultFirstAndItClonesNone) {
    TinyVocaPair pair("", 3, 1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    char **voices = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tts_voices(pair.tts, &voices, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    std::vector<std::string> names(voices, voices + count);
    jllama_strings_free(voices, count);
    EXPECT_EQ(names, (std::vector<std::string>{"M2", "F2"})) << "clip.gen.audio.default_voice goes first";
    bool clones = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(pair.tts, &clones), JLLAMA_STATUS_OK);
    EXPECT_FALSE(clones);
}

TEST(CApiTinyVoca, EveryTokenLastsItsFramesOf256SamplesAt24kHz) {
    TinyVocaPair pair("", 3, 1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> wav;
    // the synthesis reports no frame: it writes the whole sentence from the prompt (the engine's "no
    // audio" check reads the output only since then; it refused every tiny-voca synthesis before)
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_rate(wav), 24000u);
    EXPECT_EQ(wav_samples(wav), 7u * 3 * 256) << "<bos> h ə l oʊ . <eos>";
    // a final mark of its own: no full stop is added
    ASSERT_EQ(pair.speak("h \xC9\x99 l o\xCA\x8A ?", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 7u * 3 * 256) << "<bos> h ə l oʊ ? <eos>";
    // the same seed speaks the same, another seed draws other noise
    std::vector<unsigned char> again;
    std::vector<unsigned char> other;
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &wav), JLLAMA_STATUS_OK);
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &again), JLLAMA_STATUS_OK);
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &other, 1.0f, nullptr, 8), JLLAMA_STATUS_OK);
    EXPECT_EQ(again, wav);
    EXPECT_NE(other, wav);
}

TEST(CApiTinyVoca, TheSpeedDividesTheFramesOfEveryToken) {
    TinyVocaPair pair("", 4, 1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> wav;
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 7u * 4 * 256);
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &wav, 2.0f), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 7u * 2 * 256);
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &wav, 0.5f), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 7u * 8 * 256);
    // llama.cpp takes 0.25 to 4; above it the synthesis fails, the next one speaks
    EXPECT_NE(pair.speak(TINY_VOCA_HELLO, &wav, 5.0f), JLLAMA_STATUS_OK);
    EXPECT_EQ(pair.speak(TINY_VOCA_HELLO, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST(CApiTinyVoca, APauseTokenLastsOnlyWhenTheDurationHeadKeepsIt) {
    // presence < 0: punctuation, <bos> and <eos> (class 0) get no frame, phonemes keep theirs
    TinyVocaPair pair("", 3, -1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> wav;
    ASSERT_EQ(pair.speak("h \xC9\x99 , l o\xCA\x8A", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 4u * 3 * 256) << "four phonemes, the comma and the specials silent";
    // a symbol the model does not know is <unk>, which counts as a phoneme
    ASSERT_EQ(pair.speak("h \xC9\x99 l o\xCA\x8A xyz", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 5u * 3 * 256);
    // only punctuation: no frame at all, which the synthesis refuses
    EXPECT_NE(pair.speak(", .", &wav), JLLAMA_STATUS_OK);
}

TEST(CApiTinyVoca, EachSentenceEndedByEosIsOneSegment) {
    TinyVocaPair pair("", 3, 1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> wav;
    // two sentences the template would have split: <bos> h ə <eos> and <bos> l oʊ <eos>, no full stop added
    ASSERT_EQ(pair.speak("h \xC9\x99 <eos> l o\xCA\x8A <eos>", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 8u * 3 * 256);
    // a sentence of more frames than llama.cpp takes in one utterance (3000) is refused
    TinyVocaPair slow("", 400, 1.0f, 0);
    ASSERT_NE(slow.tts, nullptr);
    // 400 is the most frames llama.cpp gives a token: 8 tokens are 3200 frames
    EXPECT_NE(slow.speak("h \xC9\x99 l o\xCA\x8A l", &wav), JLLAMA_STATUS_OK) << "8 tokens of 400 frames";
    EXPECT_EQ(slow.speak("h <eos> \xC9\x99 <eos>", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(wav), 6u * 400 * 256) << "two sentences of three tokens, each below the limit";
}

TEST(CApiTinyVoca, AVoiceChangesTheAudioAndAnUnknownOneIsRefused) {
    TinyVocaPair pair("", 3, 1.0f, 0);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> m2;
    std::vector<unsigned char> f2;
    std::vector<unsigned char> none;
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &none), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &m2, 1.0f, "M2"), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &f2, 1.0f, "F2"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(none, m2) << "no voice is the default voice";
    EXPECT_NE(f2, m2);
    EXPECT_EQ(pair.speak(TINY_VOCA_HELLO, &m2, 1.0f, "robot"), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("F2"), std::string::npos) << jllama_last_error();
}

TEST(CApiTinyVoca, TheTemplateAsksTheProjectorsLetterToSoundModel) {
    // the G2P of the generated projector answers AE1 for every word; the template speaks "h ə l oʊ" when
    // tts_g2p() does so and "b æ d" (three phonemes) otherwise -- which is what an engine that does not
    // hand the template mtmd_gen_audio_g2p (as upstream's /tts does since b12497) would speak
    const std::string tmpl =
        "{%- if tts_g2p('hello') == 'AE1' -%}h \xC9\x99 l o\xCA\x8A{%- else -%}b \xC3\xA6 d{%- endif -%}";
    TinyVocaPair pair(tmpl, 3, 1.0f, 5);
    ASSERT_NE(pair.tts, nullptr);
    std::vector<unsigned char> templated;
    std::vector<unsigned char> raw;
    ASSERT_EQ(pair.speak("anything", &templated), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(pair.speak(TINY_VOCA_HELLO, &raw, 1.0f, nullptr, 7, 1), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(templated), 7u * 3 * 256) << "four phonemes, not the three of the fallback";
    EXPECT_EQ(templated, raw);
    // without its letter-to-sound model the same template takes the fallback
    TinyVocaPair without(tmpl, 3, 1.0f, 0);
    ASSERT_NE(without.tts, nullptr);
    ASSERT_EQ(without.speak("anything", &templated), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav_samples(templated), 6u * 3 * 256) << "<bos> b æ d . <eos>";
}

TEST(CApiLfm3TtsEnd, WithoutItsTtsTemplateTheModelCannotSpeak) {
    // the pipeline starts in audio only after a prompt ending in <|audio_start|>, which only the
    // "tts" template writes; the bare text ends in text, and llama.cpp refuses it in set_input
    const std::string backbone = jllama_test::write_tiny_lfm3_gguf("jllama-lfm3-no-template", false);
    const std::string mmproj = jllama_test::write_tiny_lfm3_mmproj_gguf("jllama-lfm3-no-template-mmproj");
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 10;
    unsigned char *bytes = nullptr;
    size_t length = 0;
    EXPECT_EQ(jllama_tts_synthesize_options(tts, "hello world", &options, &bytes, &length), JLLAMA_STATUS_ERROR);
    jllama_tts_free(tts);
    std::remove(backbone.c_str());
    std::remove(mmproj.c_str());
}

// A random ZONOS3 pair (tiny_llama_gguf.hpp), the one speech pipeline with delivery control
// (llama.cpp b12431, #32020): every synthesis runs the real pipeline -- the condition tokens of the
// prompt, the codebook heads, the ZAC codec, the iSTFT -- on noise weights, so what is asserted is the
// contract, not the sound. Its delivery is the part no other model in the C++ suite reaches: a speed
// and a loudness are spoken (another pipeline answers 2 from set_input), and they change the prompt,
// so the same seed gives other audio.
class CApiZonos3 : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path() = jllama_test::write_tiny_zonos3_gguf();
        mmproj_path() = jllama_test::write_tiny_zonos3_mmproj_gguf();
        ASSERT_FALSE(model_path().empty());
        ASSERT_FALSE(mmproj_path().empty());
        ASSERT_EQ(jllama_tts_load(model_path().c_str(), mmproj_path().c_str(), 0, 2, &tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(tts());
        tts() = nullptr;
        std::remove(model_path().c_str());
        std::remove(mmproj_path().c_str());
    }
    static std::string &model_path() {
        static std::string p;
        return p;
    }
    static std::string &mmproj_path() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    // A synthesis of 40 frames with a fixed seed; the status, and the WAV in *wav. Fewer frames give no
    // audio: the ten codebooks are delayed by one row each behind three rows of codec silence, so the
    // codec has a frame to decode only once a code row has passed through all of them.
    static jllama_status speak(float speed, float loudness, std::vector<unsigned char> *wav, int max_frames = 40) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = max_frames;
        options.seed = 7;
        options.speed = speed;
        options.loudness = loudness;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "Hello.", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    static uint32_t le32(const std::vector<unsigned char> &b, size_t at) {
        return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
    }
};

TEST_F(CApiZonos3, ItsVoiceIsListedAndItClonesNone) {
    char **names = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tts_voices(tts(), &names, &count), JLLAMA_STATUS_OK);
    ASSERT_EQ(count, 1u);
    EXPECT_STREQ(names[0], "test");
    jllama_strings_free(names, count);
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported) << "the mmproj has no speaker encoder";
    // its voice speaks, by name and as the default; another name is the request's fault
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 40;
    unsigned char *wav = nullptr;
    size_t length = 0;
    options.voice = "test";
    EXPECT_EQ(jllama_tts_synthesize_options(tts(), "Hello.", &options, &wav, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_bytes_free(wav);
    wav = nullptr;
    options.voice = "nobody";
    EXPECT_EQ(jllama_tts_synthesize_options(tts(), "Hello.", &options, &wav, &length), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("test"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(wav, nullptr);
}

TEST_F(CApiZonos3, ASynthesisIsAWavAtTheCodecsRate) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak(1.0f, 0.0f, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(std::string(wav.begin(), wav.begin() + 4), "RIFF");
    EXPECT_EQ(std::string(wav.begin() + 8, wav.begin() + 12), "WAVE");
    EXPECT_EQ(le32(wav, 24), 44100u) << "ZONOS3's ZAC codec resamples to 44.1 kHz";
    EXPECT_EQ(wav[22] | wav[23] << 8, 1)
        << "ZONOS3 is mono (b12459: the header follows mtmd_gen_audio_info::n_channels)";
    std::vector<unsigned char> again;
    ASSERT_EQ(speak(1.0f, 0.0f, &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same audio";
}

TEST_F(CApiZonos3, ASpeedAndALoudnessAreSpokenAndChangeTheAudio) {
    std::vector<unsigned char> own;
    std::vector<unsigned char> faster;
    std::vector<unsigned char> quieter;
    ASSERT_EQ(speak(1.0f, 0.0f, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(2.0f, 0.0f, &faster), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(1.0f, -28.0f, &quieter), JLLAMA_STATUS_OK) << jllama_last_error();
    // other condition tokens in the prompt, so another hidden state and other codes for one seed
    EXPECT_NE(faster, own);
    EXPECT_NE(quieter, own);
    EXPECT_NE(faster, quieter);
    // the buckets: two speeds in one speaking-rate bucket are one prompt
    std::vector<unsigned char> same_bucket;
    ASSERT_EQ(speak(2.01f, 0.0f, &same_bucket), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(same_bucket, faster) << "2.0 and 2.01 times 15 bytes per second share bucket 27";
}

// b12453 appended the reference's transcript (Fish S2) to jllama_tts_options. A caller of the older
// header declares 64 bytes, and the library must not read past them: the pointer behind is not one,
// and reading it would end the process. A model that does not use a transcript (ZONOS3) speaks the
// same with one as without.
TEST_F(CApiZonos3, AnOlderHeadersOptionsAndATranscriptTheModelDoesNotUseLeaveTheAudioAsIs) {
    std::vector<unsigned char> own;
    ASSERT_EQ(speak(1.0f, 0.0f, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    const auto synthesize = [](const jllama_tts_options &options, std::vector<unsigned char> *wav) {
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "Hello.", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 40;
    options.seed = 7;
    options.struct_size = offsetof(jllama_tts_options, speaker_reference_text);
    options.speaker_reference_text = reinterpret_cast<const char *>(static_cast<uintptr_t>(1));
    std::vector<unsigned char> older;
    ASSERT_EQ(synthesize(options, &older), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(older, own);

    options.struct_size = sizeof(jllama_tts_options);
    options.speaker_reference_text = "This is what the reference says.";
    std::vector<unsigned char> transcribed;
    ASSERT_EQ(synthesize(options, &transcribed), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(transcribed, own);
}

TEST_F(CApiZonos3, ADeliveryNoModelCanTakeIsRefusedAndTheModelAnswersTheNext) {
    EXPECT_EQ(speak(0.0f, 0.0f, nullptr), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("speed"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(1.0f, 3.0f, nullptr), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("loudness"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(1.0f, 0.0f, nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

// b12459: a frame cap of 0 is the model's own (gen_audio::max_frames()), which is 512 for ZONOS3 as for
// most pipelines: the same seed then gives exactly the audio of an explicit cap of 512. Before the bump
// the engine put 512 in itself; a stereo song model (YuE2) names none and stops by itself.
TEST_F(CApiZonos3, AFrameCapOfZeroIsTheModelsOwn) {
    std::vector<unsigned char> own;
    std::vector<unsigned char> explicit_cap;
    ASSERT_EQ(speak(1.0f, 0.0f, &own, 0), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(1.0f, 0.0f, &explicit_cap, 512), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(own, explicit_cap);
    std::vector<unsigned char> short_cap;
    ASSERT_EQ(speak(1.0f, 0.0f, &short_cap, 40), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GE(own.size(), short_cap.size()) << "a cap of 40 cannot speak longer than the model's own";
}

// b12459 (#32369): guidance and solver steps belong to YuE2's flow-matching pipeline. ZONOS3 has
// neither, so llama.cpp refuses any value but the model's own, and the engine names the fields
// rather than the delivery; the model answers the next request.
TEST_F(CApiZonos3, GuidanceAndSolverStepsAreRefusedByAModelWithoutThem) {
    const auto speak_with = [](float cfg_scale, int flow_steps) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 40;
        options.seed = 7;
        options.cfg_scale = cfg_scale;
        options.flow_steps = flow_steps;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "Hello.", &options, &bytes, &length);
        jllama_bytes_free(bytes);
        return status;
    };
    EXPECT_EQ(speak_with(1.5f, 0), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("cfg_scale or flow_steps"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(speak_with(0.0f, 8), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("cfg_scale or flow_steps"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(speak_with(0.0f, 0), JLLAMA_STATUS_OK) << jllama_last_error();
}

// A random MiniMax Music 3 pair (tiny_llama_gguf.hpp, llama.cpp b12480, #32435): the first pipeline
// in the suite whose backbone runs with classifier-free guidance (a second, unconditioned sequence
// next to the prompt), whose audio comes out of flow matching (an ODE over the latents, solved in
// cfg_scale and flow_steps) and whose WAV is stereo. On noise weights, so what is asserted is the
// contract, not the music.
class CApiMusic3 : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path() = jllama_test::write_tiny_music3_gguf();
        mmproj_path() = jllama_test::write_tiny_music3_mmproj_gguf();
        ASSERT_FALSE(model_path().empty());
        ASSERT_FALSE(mmproj_path().empty());
        ASSERT_EQ(jllama_tts_load(model_path().c_str(), mmproj_path().c_str(), 0, 2, &tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(tts());
        tts() = nullptr;
        std::remove(model_path().c_str());
        std::remove(mmproj_path().c_str());
    }
    static std::string &model_path() {
        static std::string p;
        return p;
    }
    static std::string &mmproj_path() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_status sing(const char *text, float cfg_scale, int flow_steps, std::vector<unsigned char> *wav,
                              int max_frames = 12) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = max_frames;
        options.seed = 11;
        options.cfg_scale = cfg_scale;
        options.flow_steps = flow_steps;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    static uint32_t le32(const std::vector<unsigned char> &b, size_t at) {
        return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
    }
};

TEST_F(CApiMusic3, ASongIsAStereoWavAt44100Hz) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(sing("la la la", 0.0f, 2, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(std::string(wav.begin(), wav.begin() + 4), "RIFF");
    EXPECT_EQ(std::string(wav.begin() + 8, wav.begin() + 12), "WAVE");
    EXPECT_EQ(le32(wav, 24), 44100u) << "the DAC vocoder's rate";
    EXPECT_EQ(wav[22] | wav[23] << 8, 2) << "MiniMax Music 3 is stereo (mtmd_gen_audio_info::n_channels)";
    // 16-bit stereo, and every latent frame is 512 samples per channel
    const size_t samples = (wav.size() - 44) / 4;
    EXPECT_EQ(samples % 512, 0u) << samples;
    EXPECT_GT(samples, 0u);
    std::vector<unsigned char> again;
    ASSERT_EQ(sing("la la la", 0.0f, 2, &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same song";
    std::vector<unsigned char> other;
    ASSERT_EQ(sing("na na na", 0.0f, 2, &other), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(wav, other) << "another text is another prompt";
}

TEST_F(CApiMusic3, GuidanceAndSolverStepsAreTakenAndChangeTheSong) {
    // b12459 passes cfg_scale and flow_steps through; ZONOS3 refuses them, this model takes both.
    std::vector<unsigned char> own;
    ASSERT_EQ(sing("la la la", 0.0f, 2, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    std::vector<unsigned char> stronger;
    ASSERT_EQ(sing("la la la", 3.0f, 2, &stronger), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(own, stronger) << "a guidance scale of 3 is not the model's own 1.5";
    // a scale of 1 runs no unconditioned sequence at all
    std::vector<unsigned char> unguided;
    ASSERT_EQ(sing("la la la", 1.0f, 2, &unguided), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(own, unguided);
    std::vector<unsigned char> more_steps;
    ASSERT_EQ(sing("la la la", 0.0f, 5, &more_steps), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(own, more_steps) << "the ODE solved in 5 steps ends elsewhere than in 2";
    EXPECT_EQ(own.size(), more_steps.size()) << "the steps change the latents, not their number";
}

TEST_F(CApiMusic3, MoreFramesAreALongerSongAndALongSongIsDecodedInWindows) {
    std::vector<unsigned char> shorter;
    std::vector<unsigned char> longer;
    ASSERT_EQ(sing("la la la", 1.0f, 1, &shorter, 12), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(sing("la la la", 1.0f, 1, &longer, 24), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(longer.size(), shorter.size());
    // past 200 frames the flow matching runs in windows of 200 every 100 frames, each cropped where
    // it overlaps its neighbour; the song stays whole latent frames and keeps growing with the frames
    std::vector<unsigned char> windowed;
    ASSERT_EQ(sing("la la la", 1.0f, 1, &windowed, 260), JLLAMA_STATUS_OK) << jllama_last_error();
    // 260 condition rows (the frame before them only moves the backbone past <|audio_start|>), in two
    // windows: [0, 200) is 689 latent frames and loses its last 258, [100, 260) is 551 and loses its
    // first 86, so 431 + 465 = 896 latent frames of 512 samples. One window over all 260 rows would be
    // 895 (260 * 44100 / 24000 * 960 / 512), so the length tells the windows apart.
    const size_t samples = (windowed.size() - 44) / 4;
    EXPECT_EQ(samples, 896u * 512u) << samples;
    EXPECT_GT(windowed.size(), 10 * longer.size()) << windowed.size() << " vs " << longer.size();
    std::vector<unsigned char> again;
    ASSERT_EQ(sing("la la la", 1.0f, 1, &again, 260), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(windowed, again) << "the windows start from the tail of the one before, the same each time";
}

// The tts template of a speech model may look words up in dictionaries the model carries (llama.cpp
// b12480, #32397: tts_dict(name, key), for Kokoro's pronunciation lexicon; the arrays are read
// through the new llama_model_meta_arr_n / llama_model_meta_arr_str), and rewrite the text with the
// normalizer's builtins (b12233..b12283, #32053: tts_norm_sub, tts_norm_scan). The random ZONOS3 with a
// template that speaks the dictionary's entry for a known text and the text with every number replaced
// by N otherwise: a
// looked-up word gives exactly the audio of its entry, an unknown one its own. A dictionary with
// fewer values than keys is ignored, so a template that names it fails the synthesis by its name.
// tts_g2p() (b12497, #32427) asks the projector's letter-to-sound model, which only tiny-voca has.
class CApiTtsDict : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        const std::string tmpl = "{%- set text = messages[0].content -%}"
                                 "{%- if text == 'broken' -%}{{ tts_dict('broken', text) }}"
                                 "{%- elif text == 'g2p' -%}{%- set p = tts_g2p('hello') -%}{{ p if p else 'Hello.' }}"
                                 "{%- else -%}{%- set entry = tts_dict('lex', text) -%}"
                                 "{{ entry.say if entry else tts_norm_sub(text, '%d+', 'N') }}{%- endif -%}";
        const std::vector<jllama_test::tiny_tts_dict> dicts = {
            {"lex", {"Hi.", "foo"}, {R"({"say":"Hello."})", R"({"say":"Goodbye."})"}},
            {"broken", {"a", "b"}, {R"({"say":"x"})"}},
        };
        model_path() = jllama_test::write_tiny_zonos3_gguf("jllama-zonos3-dict", tmpl, dicts);
        mmproj_path() = jllama_test::write_tiny_zonos3_mmproj_gguf();
        ASSERT_FALSE(model_path().empty());
        ASSERT_FALSE(mmproj_path().empty());
        ASSERT_EQ(jllama_tts_load(model_path().c_str(), mmproj_path().c_str(), 0, 2, &tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void TearDownTestSuite() {
        jllama_tts_free(tts());
        tts() = nullptr;
        std::remove(model_path().c_str());
        std::remove(mmproj_path().c_str());
    }
    static std::string &model_path() {
        static std::string p;
        return p;
    }
    static std::string &mmproj_path() {
        static std::string p;
        return p;
    }
    static jllama_tts_model *&tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_status speak(const char *text, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 40;
        options.seed = 7;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
};

TEST_F(CApiTtsDict, AWordOfTheDictionaryIsSpokenAsItsEntry) {
    std::vector<unsigned char> looked_up;
    std::vector<unsigned char> entry;
    std::vector<unsigned char> other;
    ASSERT_EQ(speak("Hi.", &looked_up), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("Hello.", &entry), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("Goodbye.", &other), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(looked_up, entry) << "'Hi.' is in the dictionary, its entry says 'Hello.'";
    EXPECT_NE(entry, other) << "two texts give two prompts, so the comparison above can fail";
    std::vector<unsigned char> foo;
    ASSERT_EQ(speak("foo", &foo), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(foo, other) << "the second key, found by the binary search over the sorted keys";
}

TEST_F(CApiTtsDict, AWordOutsideTheDictionaryIsSpokenAsItIs) {
    std::vector<unsigned char> unknown;
    std::vector<unsigned char> entry;
    ASSERT_EQ(speak("Hello.", &entry), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("Hi", &unknown), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(unknown, entry) << "'Hi' is no key (the key is 'Hi.'), so tts_dict answers none";
}

TEST_F(CApiTtsDict, TheNormalizerOfTheTemplateRewritesTheText) {
    std::vector<unsigned char> digits;
    std::vector<unsigned char> replaced;
    std::vector<unsigned char> other;
    ASSERT_EQ(speak("7 cats", &digits), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("N cats", &replaced), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("M cats", &other), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(digits, replaced) << "tts_norm_sub(text, '%d+', 'N') turned 7 into N";
    EXPECT_NE(replaced, other) << "two texts give two prompts, so the comparison above can fail";
}

TEST_F(CApiTtsDict, AModelWithoutALetterToSoundModelAnswersNoneToTheTemplatesG2p) {
    // the engine hands the template mtmd_gen_audio_g2p (b12497), as /tts does; ZONOS3's projector has
    // no letter-to-sound model, so tts_g2p() answers none and the template speaks its fallback
    std::vector<unsigned char> asked;
    std::vector<unsigned char> fallback;
    ASSERT_EQ(speak("g2p", &asked), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("Hello.", &fallback), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(asked, fallback);
}

TEST_F(CApiTtsDict, AMalformedDictionaryIsIgnoredAndATemplateNamingItFails) {
    EXPECT_NE(speak("broken", nullptr), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("unknown dictionary 'broken'"), std::string::npos)
        << jllama_last_error();
    // the model answers the next request
    EXPECT_EQ(speak("Hi.", nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiTtsDict, TheDictionaryArraysAreKeptAndNoOtherArray) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model *model = llama_model_load_from_file(model_path().c_str(), params);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(llama_model_meta_arr_n(model, "tokenizer.tts_dicts"), 2);
    EXPECT_STREQ(llama_model_meta_arr_str(model, "tokenizer.tts_dicts", 0), "lex");
    EXPECT_STREQ(llama_model_meta_arr_str(model, "tokenizer.tts_dicts", 1), "broken");
    EXPECT_EQ(llama_model_meta_arr_str(model, "tokenizer.tts_dicts", 2), nullptr);
    EXPECT_EQ(llama_model_meta_arr_str(model, "tokenizer.tts_dicts", -1), nullptr);
    EXPECT_EQ(llama_model_meta_arr_n(model, "tokenizer.tts_dict.lex.keys"), 2);
    EXPECT_STREQ(llama_model_meta_arr_str(model, "tokenizer.tts_dict.lex.values", 1), R"({"say":"Goodbye."})");
    EXPECT_EQ(llama_model_meta_arr_n(model, "tokenizer.tts_dict.broken.values"), 1);
    // only the tts_dict arrays are kept: the vocabulary is an array the model has, and still -1
    EXPECT_EQ(llama_model_meta_arr_n(model, "tokenizer.ggml.tokens"), -1);
    EXPECT_EQ(llama_model_meta_arr_n(model, "tokenizer.tts_dict.none.keys"), -1);
    llama_model_free(model);
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
    // the file's name, not its path: the output is published, the path names this machine's directories
    const int64_t file_key = gguf_find_key(g, "quantize.imatrix.file");
    ASSERT_GE(file_key, 0);
    const std::string recorded = gguf_get_val_str(g, file_key);
    EXPECT_EQ(recorded, imatrix_.substr(imatrix_.find_last_of("/\\") + 1));
    EXPECT_EQ(recorded.find_first_of("/\\"), std::string::npos) << recorded;
    EXPECT_NE(imatrix_.find_first_of("/\\"), std::string::npos) << "the fixture's matrix has no directory part";
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

namespace {

// Video input is llama.cpp's own: the server hands a file that is neither an image nor audio to
// the `ffmpeg` and `ffprobe` programs it finds on PATH and feeds the frames to the vision projector.
// These cases therefore need the two programs. Where they are missing the cases are skipped --
// unless JLLAMA_REQUIRE_FFMPEG is set, as in the CI job that installs them, where a skip would hide
// that nothing ran.
bool ffmpeg_is_installed() {
#if defined(_WIN32)
    const char *quiet = " >NUL 2>&1";
#else
    const char *quiet = " >/dev/null 2>&1";
#endif
    return std::system((std::string("ffmpeg -version") + quiet).c_str()) == 0 &&
           std::system((std::string("ffprobe -version") + quiet).c_str()) == 0;
}

// A clip ffmpeg writes itself: its test picture, `seconds` long at `fps` frames per second.
std::vector<uint8_t> ffmpeg_test_clip(int seconds, int fps, const char *extension = "mp4") {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() /
        ("jllama-c-api-clip-" + std::to_string(std::random_device{}()) + "." + extension);
#if defined(_WIN32)
    const char *quiet = " >NUL 2>&1";
#else
    const char *quiet = " >/dev/null 2>&1";
#endif
    const std::string command =
        "ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc=duration=" + std::to_string(seconds) +
        ":size=64x64:rate=" + std::to_string(fps) + " -pix_fmt yuv420p \"" + file.string() + "\"" + quiet;
    std::vector<uint8_t> bytes;
    if (std::system(command.c_str()) == 0) {
        std::ifstream in(file, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    return bytes;
}

std::string model_media_marker(jllama_model *model) {
    Owned meta;
    if (jllama_model_meta(model, &meta.s) != JLLAMA_STATUS_OK) {
        ADD_FAILURE() << jllama_last_error();
        return std::string();
    }
    return json::parse(meta.str()).value("media_marker", std::string());
}

// A raw request whose prompt carries one media file between two words.
std::string media_prompt_request(jllama_model *model, const std::vector<uint8_t> &media) {
    return json{{"prompt",
                 {{"prompt_string", "look " + model_media_marker(model) + " now"},
                  {"multimodal_data", json::array({base64_of(media)})}}},
                {"n_predict", 1},
                {"temperature", 0},
                {"cache_prompt", false}}
        .dump();
}

int tokens_evaluated(jllama_model *model, const std::string &request) {
    Owned completion;
    if (jllama_completions(model, request.c_str(), &completion.s) != JLLAMA_STATUS_OK) {
        ADD_FAILURE() << jllama_last_error();
        return -1;
    }
    return json::parse(completion.str()).at("tokens_evaluated").get<int>();
}

TEST(CApiMediaUrls, AnImageUrlOnAPrivateAddressIsRefusedUnlessTheModelAllowsIt) {
    // Upstream #32115 (b12309..b12387): the server fetches an http(s) media URL only from a public
    // address -- loopback, RFC 1918, link-local and CGNAT are refused, every redirect checked again --
    // unless it was started with --media-allow-private-urls. A loopback server serves the image here.
    httplib::Server images;
    const std::vector<uint8_t> png = base64_bytes(TINY_PNG_BASE64);
    images.Get("/tiny.png", [&png](const httplib::Request &, httplib::Response &res) {
        res.set_content(std::string(png.begin(), png.end()), "image/png");
    });
    const int port = images.bind_to_any_port("127.0.0.1");
    std::thread listener([&images]() { images.listen_after_bind(); });
    images.wait_until_ready();

    const std::string path = write_tiny_llama_gguf("jllama-c-api-media-urls");
    const std::string mmproj = write_tiny_mmproj_gguf("jllama-c-api-media-urls-mmproj");
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(mmproj.empty());
    const auto load = [&](bool allow) {
        std::vector<std::string> args = {"jllama", "-m",   path, "--mmproj",    mmproj, "-c",
                                         "128",    "-ngl", "0",  "--no-warmup", "-t",   "2"};
        if (allow) {
            args.push_back("--media-allow-private-urls");
        }
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        jllama_model *model = nullptr;
        EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        return model;
    };
    const auto chat = [](const std::string &url) {
        return json{{"messages", json::array({json{{"role", "user"},
                                                   {"content", json::array({json{{"type", "text"}, {"text", "look"}},
                                                                            json{{"type", "image_url"},
                                                                                 {"image_url", {{"url", url}}}}})}}})},
                    {"max_tokens", 1},
                    {"temperature", 0},
                    {"cache_prompt", false}}
            .dump();
    };
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/tiny.png";
    const std::string data_url = std::string("data:image/png;base64,") + TINY_PNG_BASE64;

    jllama_model *strict = load(false);
    ASSERT_NE(strict, nullptr);
    Owned refused;
    EXPECT_EQ(jllama_chat_completions(strict, chat(url).c_str(), &refused.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("not a public address"), std::string::npos) << jllama_last_error();
    // the same image inline is no URL and answers
    Owned inline_image;
    ASSERT_EQ(jllama_chat_completions(strict, chat(data_url).c_str(), &inline_image.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int expected = json::parse(inline_image.str()).at("usage").at("prompt_tokens").get<int>();
    jllama_model_free(strict);

    jllama_model *open = load(true);
    ASSERT_NE(open, nullptr);
    Owned fetched;
    ASSERT_EQ(jllama_chat_completions(open, chat(url).c_str(), &fetched.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(fetched.str()).at("usage").at("prompt_tokens").get<int>(), expected)
        << "the fetched image is the inline one";
    jllama_model_free(open);

    images.stop();
    listener.join();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(mmproj, ignored);
}

class CApiVideo : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        path_ = write_tiny_llama_gguf("jllama-c-api-video");
        mmproj_ = write_tiny_mmproj_gguf("jllama-c-api-video-mmproj");
        ASSERT_FALSE(path_.empty());
        ASSERT_FALSE(mmproj_.empty());
        ffmpeg_ = ffmpeg_is_installed();
    }
    static void TearDownTestSuite() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
        std::filesystem::remove(mmproj_, ignored);
    }
    // The random model with the random vision projector.
    static jllama_model *load(const std::vector<std::string> &extra = {}) {
        std::vector<std::string> args = {"--mmproj", mmproj_, "-c", "1024"};
        args.insert(args.end(), extra.begin(), extra.end());
        return load_tiny(path_, args);
    }
    static std::string path_;
    static std::string mmproj_;
    static bool ffmpeg_;
};
std::string CApiVideo::path_;
std::string CApiVideo::mmproj_;
bool CApiVideo::ffmpeg_ = false;

#define REQUIRE_FFMPEG()                                                                                               \
    do {                                                                                                               \
        if (!ffmpeg_) {                                                                                                \
            if (std::getenv("JLLAMA_REQUIRE_FFMPEG") != nullptr) {                                                     \
                FAIL() << "JLLAMA_REQUIRE_FFMPEG is set, and ffmpeg / ffprobe are not on PATH";                        \
            }                                                                                                          \
            GTEST_SKIP() << "ffmpeg / ffprobe are not on PATH";                                                        \
        }                                                                                                              \
    } while (false)

} // namespace

TEST_F(CApiVideo, TheModelMetaSaysWhetherTheModelTakesVideo) {
    // a desktop build decodes video with ffmpeg, so every model with a vision projector takes it;
    // the meta does not say whether the program is installed -- a request finds that out
    jllama_model *with_projector = load();
    ASSERT_NE(with_projector, nullptr) << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(with_projector, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(json::parse(meta.str()).at("modalities").at("video").get<bool>()) << meta.str();
    jllama_model_free(with_projector);

    jllama_model *without = load_tiny(path_, {});
    ASSERT_NE(without, nullptr) << jllama_last_error();
    Owned plain;
    ASSERT_EQ(jllama_model_meta(without, &plain.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_FALSE(json::parse(plain.str()).at("modalities").at("video").get<bool>()) << plain.str();
    const json request = {
        {"messages",
         json::array(
             {{{"role", "user"},
               {"content", json::array({json{{"type", "input_video"}, {"input_video", {{"data", "AAAA"}}}}})}}})}};
    Owned response;
    EXPECT_NE(jllama_chat_completions(without, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("video input is not supported"), std::string::npos)
        << jllama_last_error();
    jllama_model_free(without);
}

TEST_F(CApiVideo, AVideoReachesTheModelAsItsFrames) {
    REQUIRE_FFMPEG();
    jllama_model *model = load();
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const int text_tokens = tokens_evaluated(model, R"({"prompt":"look  now","n_predict":1,"temperature":0})");
    const int image_tokens = tokens_evaluated(model, media_prompt_request(model, base64_bytes(TINY_PNG_BASE64)));
    ASSERT_EQ(image_tokens, text_tokens + 4);

    // two seconds at four frames per second, the rate the server samples at by default: eight
    // frames, four patch embeddings each, plus the text mtmd puts around a video
    const std::vector<uint8_t> clip = ffmpeg_test_clip(2, 4);
    ASSERT_FALSE(clip.empty()) << "ffmpeg wrote no clip";
    const int video_tokens = tokens_evaluated(model, media_prompt_request(model, clip));
    EXPECT_GE(video_tokens, text_tokens + 8 * 4);
    EXPECT_LT(video_tokens, text_tokens + 8 * 4 + 64) << "more than eight frames and a few words";

    // twice as long: eight more frames
    const std::vector<uint8_t> longer = ffmpeg_test_clip(4, 4);
    ASSERT_FALSE(longer.empty());
    const int longer_tokens = tokens_evaluated(model, media_prompt_request(model, longer));
    EXPECT_GE(longer_tokens, video_tokens + 8 * 4);
    EXPECT_LT(longer_tokens, video_tokens + 8 * 4 + 32);

    // the same clip, the same tokens: streamed, and on the OpenAI completion path
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, media_prompt_request(model, clip).c_str(), &task),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model, task).at("tokens_evaluated").get<int>(), video_tokens);

    // another container and codec (Matroska)
    const std::vector<uint8_t> mkv = ffmpeg_test_clip(2, 4, "mkv");
    ASSERT_FALSE(mkv.empty());
    EXPECT_EQ(tokens_evaluated(model, media_prompt_request(model, mkv)), video_tokens);
    jllama_model_free(model);
}

TEST_F(CApiVideo, AChatMessageCarriesAVideoUnderBothNamesUpstreamReads) {
    REQUIRE_FFMPEG();
    jllama_model *model = load();
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::string clip = base64_of(ffmpeg_test_clip(2, 4));
    ASSERT_FALSE(clip.empty());
    const auto prompt_tokens = [&](const json &content) {
        const json request = {{"messages", json::array({{{"role", "user"}, {"content", content}}})},
                              {"max_tokens", 2},
                              {"temperature", 0},
                              {"cache_prompt", false}};
        Owned response;
        if (jllama_chat_completions(model, request.dump().c_str(), &response.s) != JLLAMA_STATUS_OK) {
            ADD_FAILURE() << jllama_last_error();
            return -1;
        }
        return json::parse(response.str()).at("usage").at("prompt_tokens").get<int>();
    };
    const json text = {{"type", "text"}, {"text", "what happens"}};
    const int text_only = prompt_tokens(json::array({text}));
    ASSERT_GT(text_only, 0);
    // llama.cpp's own part, with the bytes as base64 or as a data URL
    const int as_data =
        prompt_tokens(json::array({text, json{{"type", "input_video"}, {"input_video", {{"data", clip}}}}}));
    EXPECT_GE(as_data, text_only + 8 * 4);
    EXPECT_EQ(prompt_tokens(json::array(
                  {text, json{{"type", "input_video"}, {"input_video", {{"url", "data:video/mp4;base64," + clip}}}}})),
              as_data);
    // the OpenAI-style alias
    EXPECT_EQ(prompt_tokens(json::array(
                  {text, json{{"type", "video_url"}, {"video_url", {{"url", "data:video/mp4;base64," + clip}}}}})),
              as_data);
    // a video next to an image: both are counted
    EXPECT_EQ(prompt_tokens(json::array(
                  {text, json{{"type", "input_video"}, {"input_video", {{"data", clip}}}},
                   json{{"type", "image_url"},
                        {"image_url", {{"url", std::string("data:image/png;base64,") + TINY_PNG_BASE64}}}}})),
              as_data + 4);
    jllama_model_free(model);
}

TEST_F(CApiVideo, TheFrameRateOptionDecidesHowManyFramesAreTaken) {
    REQUIRE_FFMPEG();
    const std::vector<uint8_t> clip = ffmpeg_test_clip(2, 4);
    ASSERT_FALSE(clip.empty());
    std::map<std::string, int> tokens;
    for (const char *fps : {"4", "2", "1"}) {
        // timestamps off, so the difference is frames only
        jllama_model *model = load({"--video-fps", fps, "--video-timestamp-interval", "0"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        tokens[fps] = tokens_evaluated(model, media_prompt_request(model, clip));
        jllama_model_free(model);
    }
    // 8, 4 and 2 frames of four embeddings
    EXPECT_EQ(tokens["4"] - tokens["2"], 4 * 4) << tokens["4"] << " / " << tokens["2"];
    EXPECT_EQ(tokens["2"] - tokens["1"], 2 * 4) << tokens["2"] << " / " << tokens["1"];
}

TEST_F(CApiVideo, AFileThatIsNoMediaIsTheCallersFault) {
    // also without ffmpeg: neither an image, nor audio, nor a video anything decodes
    jllama_model *model = load();
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::vector<uint8_t> garbage = {'n', 'o', 't', ' ', 'a', ' ', 'v', 'i', 'd', 'e', 'o'};
    Owned response;
    EXPECT_EQ(jllama_completions(model, media_prompt_request(model, garbage).c_str(), &response.s),
              JLLAMA_STATUS_INVALID_REQUEST)
        << jllama_last_error();
    // and the model still answers
    EXPECT_GT(tokens_evaluated(model, media_prompt_request(model, base64_bytes(TINY_PNG_BASE64))), 4);
    jllama_model_free(model);
}

TEST_F(CApiVideo, WithoutTheFfmpegProgramsAVideoIsRefusedAndTheModelGoesOn) {
    REQUIRE_FFMPEG();
    // --video-ffmpeg-dir names where the two programs are; a directory without them is a machine
    // without ffmpeg, which must answer the request with an error and leave the process alone
    const std::filesystem::path empty =
        std::filesystem::temp_directory_path() / ("jllama-no-ffmpeg-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(empty);
    jllama_model *model = load({"--video-ffmpeg-dir", empty.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::vector<uint8_t> clip = ffmpeg_test_clip(2, 4);
    ASSERT_FALSE(clip.empty());
    Owned response;
    EXPECT_EQ(jllama_completions(model, media_prompt_request(model, clip).c_str(), &response.s),
              JLLAMA_STATUS_INVALID_REQUEST)
        << jllama_last_error();
    EXPECT_GT(tokens_evaluated(model, media_prompt_request(model, base64_bytes(TINY_PNG_BASE64))), 4);
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove_all(empty, ignored);
}

namespace {

// A video "file" only the test's decoder reads: "JLV1", then the frame count, width, height and
// frames per second, one byte each. Neither stb_image nor the audio sniffing takes it, so mtmd
// hands it to the video decoder, as it does with an mp4.
std::vector<uint8_t> fake_video(uint8_t n_frames, uint8_t width, uint8_t height, uint8_t fps) {
    return {'J', 'L', 'V', '1', n_frames, width, height, fps};
}

// What the test's decoder was asked, across every stream it opened.
struct FakeVideoDecoder {
    std::atomic<int> opened{0};
    std::atomic<int> started{0};
    std::atomic<int> frames_read{0};
    std::atomic<int> frames_skipped{0};
    std::atomic<int> closed{0};
    std::atomic<int> released{0};
    std::atomic<int> bad_frame_sizes{0};
    std::atomic<bool> picks_fps{false};
    std::atomic<bool> fail_start{false};
    std::atomic<float> fps_asked{-1.0f};

    struct Stream {
        FakeVideoDecoder *owner;
        int n_frames;
        int next = 0;
        uint32_t width;
        uint32_t height;
    };

    jllama_video_decoder c_decoder() {
        jllama_video_decoder d = {};
        d.user_data = this;
        d.open = [](void *user_data, const char *, const unsigned char *data, size_t size,
                    jllama_video_info *info) -> void * {
            auto *self = static_cast<FakeVideoDecoder *>(user_data);
            if (data == nullptr || size != 8 || std::memcmp(data, "JLV1", 4) != 0) {
                return nullptr;
            }
            self->opened++;
            info->width = data[5];
            info->height = data[6];
            info->fps = static_cast<float>(data[7]);
            info->n_frames = data[4];
            info->duration = static_cast<float>(data[4]) / static_cast<float>(data[7]);
            return new Stream{self, data[4], 0, data[5], data[6]};
        };
        d.start = [](void *state, unsigned int width, unsigned int height, float fps, bool *fps_picked) -> bool {
            auto *stream = static_cast<Stream *>(state);
            stream->owner->started++;
            stream->owner->fps_asked = fps;
            stream->width = width;
            stream->height = height;
            *fps_picked = stream->owner->picks_fps.load();
            return !stream->owner->fail_start.load();
        };
        d.read_frame = [](void *state, unsigned char *rgb, unsigned int width, unsigned int height) -> bool {
            auto *stream = static_cast<Stream *>(state);
            if (stream->next >= stream->n_frames) {
                return false;
            }
            if (width != stream->width || height != stream->height) {
                stream->owner->bad_frame_sizes++;
            }
            if (rgb == nullptr) {
                stream->owner->frames_skipped++;
            } else {
                // a colour per frame, so no two frames are the same picture
                std::memset(rgb, 40 + 20 * stream->next, static_cast<size_t>(width) * height * 3);
                stream->owner->frames_read++;
            }
            stream->next++;
            return true;
        };
        d.close = [](void *state) {
            auto *stream = static_cast<Stream *>(state);
            stream->owner->closed++;
            delete stream;
        };
        d.release = [](void *user_data) { static_cast<FakeVideoDecoder *>(user_data)->released++; };
        return d;
    }
};

// The random model with the random vision projector, loaded with `decoder` (null: none).
jllama_model *load_tiny_vision(const std::string &path, const std::string &mmproj, const jllama_video_decoder *decoder,
                               const std::vector<std::string> &extra = {}) {
    std::vector<std::string> args = {"jllama", "-m",   path, "--mmproj",    mmproj, "-c",
                                     "512",    "-ngl", "0",  "--no-warmup", "-t",   "2"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    if (jllama_model_load_video(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, decoder, &model) !=
        JLLAMA_STATUS_OK) {
        return nullptr;
    }
    return model;
}

// The fixture of CApiVideo, for the cases that decode through a decoder of the application.
class CApiVideoDecoder : public CApiVideo {};

} // namespace

TEST_F(CApiVideoDecoder, AModelLoadedWithADecoderTakesVideoThroughIt) {
    // llama.cpp b12387: an application may hand in the frames of a video itself
    // (server_context::set_video_decoder) -- from the platform's own media API on Android, where no
    // ffmpeg program can be run. The frames reach the projector as images: four patch embeddings
    // each, between the text of the prompt and its timestamps. No ffmpeg is involved here.
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    jllama_model *model = load_tiny_vision(path_, mmproj_, &c_decoder);
    ASSERT_NE(model, nullptr) << jllama_last_error();
    {
        Owned meta;
        ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_TRUE(json::parse(meta.str()).at("modalities").at("video").get<bool>()) << meta.str();
    }
    const int text_tokens = tokens_evaluated(model, R"({"prompt":"look  now","n_predict":1,"temperature":0})");
    const int image_tokens = tokens_evaluated(model, media_prompt_request(model, base64_bytes(TINY_PNG_BASE64)));
    EXPECT_EQ(image_tokens, text_tokens + 4);
    EXPECT_EQ(decoder.opened.load(), 0) << "an image went to the video decoder";

    // 8 frames at 4 per second, the rate the model samples at by default: every frame is read
    const int video_tokens = tokens_evaluated(model, media_prompt_request(model, fake_video(8, 32, 32, 4)));
    EXPECT_EQ(decoder.opened.load(), 1);
    EXPECT_EQ(decoder.started.load(), 1);
    EXPECT_EQ(decoder.frames_read.load(), 8);
    EXPECT_EQ(decoder.closed.load(), 1);
    EXPECT_EQ(decoder.bad_frame_sizes.load(), 0);
    EXPECT_GE(video_tokens, text_tokens + 8 * 4) << "fewer tokens than eight frames have embeddings";

    // twice the frames, twice the embeddings (and one more timestamp at most)
    const int longer_tokens = tokens_evaluated(model, media_prompt_request(model, fake_video(16, 32, 32, 4)));
    EXPECT_EQ(decoder.frames_read.load(), 8 + 16);
    EXPECT_GE(longer_tokens, video_tokens + 8 * 4);
    EXPECT_EQ(decoder.closed.load(), 2);

    // the same video in a chat message, under both names upstream reads, and streamed
    const std::string video_base64 = base64_of(fake_video(8, 32, 32, 4));
    for (const char *type : {"input_video", "video_url"}) {
        SCOPED_TRACE(type);
        const int before = decoder.frames_read.load();
        const json part = std::string(type) == "input_video"
                              ? json{{"type", type}, {type, {{"data", video_base64}}}}
                              : json{{"type", type}, {type, {{"url", "data:video/mp4;base64," + video_base64}}}};
        const json request = {
            {"messages",
             json::array({{{"role", "user"},
                           {"content", json::array({json{{"type", "text"}, {"text", "what happens"}}, part})}}})},
            {"max_tokens", 2},
            {"temperature", 0}};
        Owned response;
        ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_GE(json::parse(response.str()).at("usage").at("prompt_tokens").get<int>(), 8 * 4) << response.str();
        EXPECT_EQ(decoder.frames_read.load(), before + 8);
    }
    {
        const int before = decoder.frames_read.load();
        int task = -1;
        ASSERT_EQ(jllama_completions_stream_request(
                      model, media_prompt_request(model, fake_video(8, 32, 32, 4)).c_str(), &task),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(receive_to_stop(model, task).at("tokens_evaluated").get<int>(), video_tokens);
        EXPECT_EQ(decoder.frames_read.load(), before + 8);
    }
    EXPECT_EQ(decoder.opened.load(), decoder.closed.load());
    EXPECT_EQ(decoder.released.load(), 0) << "released while the model is loaded";
    jllama_model_free(model);
    EXPECT_EQ(decoder.released.load(), 1);
}

TEST_F(CApiVideoDecoder, TheFrameRateOptionDecidesWhichFramesAreRead) {
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    // --video-fps 2 over a 4 fps video: every second frame is read, the others are skipped unread
    jllama_model *model = load_tiny_vision(path_, mmproj_, &c_decoder, {"--video-fps", "2"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const int half = tokens_evaluated(model, media_prompt_request(model, fake_video(8, 32, 32, 4)));
    EXPECT_EQ(decoder.frames_read.load(), 4);
    EXPECT_GE(decoder.frames_skipped.load(), 3);
    EXPECT_GE(half, 4 * 4);

    // a decoder that resamples itself is asked for the rate and then for every frame it delivers
    decoder.picks_fps = true;
    decoder.frames_read = 0;
    decoder.frames_skipped = 0;
    EXPECT_GT(tokens_evaluated(model, media_prompt_request(model, fake_video(8, 32, 32, 4))), half);
    EXPECT_FLOAT_EQ(decoder.fps_asked.load(), 2.0f);
    EXPECT_EQ(decoder.frames_read.load(), 8);
    EXPECT_EQ(decoder.frames_skipped.load(), 0);
    jllama_model_free(model);
    EXPECT_EQ(decoder.released.load(), 1);
}

TEST_F(CApiVideoDecoder, AVideoTheDecoderRefusesIsTheCallersFault) {
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    jllama_model *model = load_tiny_vision(path_, mmproj_, &c_decoder);
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned response;
    // bytes that are neither an image, nor audio, nor a video the decoder reads
    const std::vector<uint8_t> garbage = {'n', 'o', 't', ' ', 'a', ' ', 'v', 'i', 'd', 'e', 'o'};
    EXPECT_EQ(jllama_completions(model, media_prompt_request(model, garbage).c_str(), &response.s),
              JLLAMA_STATUS_INVALID_REQUEST)
        << jllama_last_error();
    EXPECT_EQ(decoder.opened.load(), 0);
    // a decoder that cannot start
    decoder.fail_start = true;
    EXPECT_EQ(jllama_completions(model, media_prompt_request(model, fake_video(8, 32, 32, 4)).c_str(), &response.s),
              JLLAMA_STATUS_INVALID_REQUEST)
        << jllama_last_error();
    EXPECT_EQ(decoder.opened.load(), 1);
    EXPECT_EQ(decoder.closed.load(), 1) << "a stream that failed to start was not closed";
    // and the model still answers
    decoder.fail_start = false;
    EXPECT_GT(tokens_evaluated(model, media_prompt_request(model, fake_video(4, 32, 32, 4))), 4 * 4);
    jllama_model_free(model);
}

TEST_F(CApiVideoDecoder, TheDecoderIsReleasedOnceAlsoWhenTheLoadFails) {
    FakeVideoDecoder decoder;
    jllama_video_decoder c_decoder = decoder.c_decoder();
    jllama_model *model = nullptr;
    const char *missing[] = {"jllama", "-m", "/nonexistent/jllama-video.gguf"};
    EXPECT_NE(jllama_model_load_video(3, missing, nullptr, nullptr, &c_decoder, &model), JLLAMA_STATUS_OK);
    EXPECT_EQ(model, nullptr);
    EXPECT_EQ(decoder.released.load(), 1);

    EXPECT_EQ(jllama_model_load_video(3, missing, nullptr, nullptr, &c_decoder, nullptr),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(decoder.released.load(), 2);
    EXPECT_EQ(jllama_model_load_video(0, nullptr, nullptr, nullptr, &c_decoder, &model),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(decoder.released.load(), 3);

    // a decoder without its required functions is refused by name, and released
    c_decoder.read_frame = nullptr;
    const std::vector<std::string> args = {"jllama", "-m", path_, "--mmproj", mmproj_, "-ngl", "0"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    EXPECT_EQ(jllama_model_load_video(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &c_decoder, &model),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("open, read_frame and close"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(decoder.released.load(), 4);
}

TEST_F(CApiVideoDecoder, ADecoderDoesNotGiveAModelWithoutAProjectorVideo) {
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    const std::vector<std::string> args = {"jllama", "-m", path_, "-c", "128", "-ngl", "0", "--no-warmup"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load_video(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &c_decoder, &model),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_FALSE(json::parse(meta.str()).at("modalities").at("video").get<bool>());
    const json request = {
        {"messages",
         json::array(
             {{{"role", "user"},
               {"content", json::array({json{{"type", "input_video"},
                                             {"input_video", {{"data", base64_of(fake_video(4, 32, 32, 4))}}}}})}}})}};
    Owned response;
    EXPECT_NE(jllama_chat_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("video input is not supported"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(decoder.opened.load(), 0);
    jllama_model_free(model);
    EXPECT_EQ(decoder.released.load(), 1);
}

TEST_F(CApiVideoDecoder, AModelThatSleptDecodesWithTheSameDecoder) {
    // waking from the idle sleep runs load_model() again; the decoder has to be there for it
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    struct Lines {
        std::mutex mutex;
        int sleeps = 0;
    } lines;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *user) {
                      auto *sink = static_cast<Lines *>(user);
                      std::lock_guard<std::mutex> lock(sink->mutex);
                      sink->sleeps += std::string(text).find("server is entering sleeping state") != std::string::npos;
                  },
                  &lines),
              JLLAMA_STATUS_OK);
    struct LogCallbackReset {
        ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); }
    } log_callback_reset;
    jllama_model *model = load_tiny_vision(path_, mmproj_, &c_decoder, {"--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const int awake = tokens_evaluated(model, media_prompt_request(model, fake_video(8, 32, 32, 4)));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    const auto slept = [&lines]() {
        std::lock_guard<std::mutex> lock(lines.mutex);
        return lines.sleeps > 0;
    };
    while (!slept() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(slept()) << "the model did not sleep";
    EXPECT_EQ(tokens_evaluated(model, media_prompt_request(model, fake_video(8, 32, 32, 4))), awake);
    EXPECT_EQ(decoder.frames_read.load(), 16);
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(json::parse(meta.str()).at("modalities").at("video").get<bool>());
    jllama_model_free(model);
    EXPECT_EQ(decoder.released.load(), 1);
}

TEST(CApiSleep, AModelWithAnIdleTimeoutSleepsAgainAfterEveryCall) {
    // llama.cpp b12387 (#32133): server_queue::wait_until_no_sleep() counts its caller as in flight,
    // and the idle timer does not fire while the count is positive. The engine wakes the server
    // through that function before every operation, so a count it did not give back kept a model
    // loaded with --sleep-idle-seconds awake for good after its first request -- the weights stayed
    // in memory, and nothing said so. Each call below must be followed by a sleep, and the call
    // after a sleep must wake the model and answer.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sleep");
    ASSERT_FALSE(path.empty());
    struct Lines {
        std::mutex mutex;
        std::vector<std::string> text;
        int count(const std::string &needle) {
            std::lock_guard<std::mutex> lock(mutex);
            int n = 0;
            for (const std::string &line : text) {
                n += line.find(needle) != std::string::npos ? 1 : 0;
            }
            return n;
        }
    } lines;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *user) {
                      auto *sink = static_cast<Lines *>(user);
                      std::lock_guard<std::mutex> lock(sink->mutex);
                      sink->text.emplace_back(text);
                  },
                  &lines),
              JLLAMA_STATUS_OK);
    struct LogCallbackReset {
        ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); }
    } log_callback_reset;

    jllama_model *model = load_tiny(path, {"--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const auto sleeps = [&lines](int n) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (lines.count("server is entering sleeping state") < n && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return lines.count("server is entering sleeping state") >= n;
    };
    // a completion (wake_server for the metadata, then wake_and_post), then metadata alone, then
    // tokenization: the three shapes of a call that wakes the server
    EXPECT_FALSE(greedy_answer(model, 4).empty());
    EXPECT_TRUE(sleeps(1)) << "the model did not sleep after a completion";
    EXPECT_FALSE(greedy_answer(model, 4).empty()) << "the sleeping model did not wake for a completion";
    EXPECT_TRUE(sleeps(2)) << "the model did not sleep after the completion that woke it";
    Owned meta;
    EXPECT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(sleeps(3)) << "the model did not sleep after a metadata call";
    int *tokens = nullptr;
    size_t n_tokens = 0;
    EXPECT_EQ(jllama_tokenize(model, "hello", false, &tokens, &n_tokens), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tokens_free(tokens);
    EXPECT_TRUE(sleeps(4)) << "the model did not sleep after a tokenization";
    jllama_model_free(model);
    ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
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

TEST(CApiBert, ARerankerWhoseClassificationHeadIsInAShardLoadsAndScoresLikeTheWholeModel) {
    // The refusal of a model without a classification head reads the tensor lists, and since --shard
    // (b12233) the head may be in a shard: the model file alone then looks like a plain encoder. So
    // the shards are read too -- a reranker split that way has to load and score like the whole
    // model, and an encoder stays refused however many shards it comes in.
    const std::string query = "what is the capital of france ?";
    const std::vector<const char *> documents = {"paris is the capital of france", "the cat sat on the mat", "berlin"};
    const auto scores = [&](jllama_model *model) {
        Owned response;
        EXPECT_EQ(
            jllama_rerank(model, query.c_str(), static_cast<int>(documents.size()), documents.data(), &response.s),
            JLLAMA_STATUS_OK)
            << jllama_last_error();
        std::map<std::string, float> by_document;
        if (response.s != nullptr) {
            const json results = json::parse(response.str());
            for (size_t i = 0; i < results.size(); ++i) {
                const json result = results.at(i);
                by_document[result.at("document").get<std::string>()] = result.at("score").get<float>();
            }
        }
        return by_document;
    };
    const std::string whole = write_tiny_bert_gguf(true, 4 /* LLAMA_POOLING_TYPE_RANK */);
    std::string head_shard;
    const std::string split = write_tiny_bert_gguf(true, 4, "jllama-bert-split", &head_shard);
    ASSERT_FALSE(whole.empty());
    ASSERT_FALSE(split.empty());
    ASSERT_FALSE(head_shard.empty());

    jllama_model *whole_model = load_tiny(whole, {"--reranking"});
    ASSERT_NE(whole_model, nullptr) << jllama_last_error();
    const auto expected = scores(whole_model);
    jllama_model_free(whole_model);
    ASSERT_EQ(expected.size(), documents.size());

    jllama_model *split_model = load_tiny(split, {"--reranking", "--shard", head_shard});
    ASSERT_NE(split_model, nullptr) << "a reranker with its head in a shard was refused: " << jllama_last_error();
    EXPECT_EQ(scores(split_model), expected);
    jllama_model_free(split_model);

    // an encoder without the head, last layer in a shard: no head in either file
    std::string plain_shard;
    const std::string encoder = write_tiny_bert_gguf(false, 2, "jllama-bert-split-encoder", &plain_shard);
    ASSERT_FALSE(encoder.empty());
    jllama_model *headless = load_tiny(encoder, {"--reranking", "--shard", plain_shard});
    EXPECT_EQ(headless, nullptr) << "an encoder without a classification head loaded as a reranker";
    EXPECT_NE(std::string(jllama_last_error()).find("classification head"), std::string::npos) << jllama_last_error();
    jllama_model_free(headless);
    // and with its shard it still loads for what it is
    jllama_model *embedder = load_tiny(encoder, {"--embedding", "--shard", plain_shard});
    EXPECT_NE(embedder, nullptr) << jllama_last_error();
    jllama_model_free(embedder);

    std::error_code ec;
    for (const std::string &f : {whole, split, encoder}) {
        std::filesystem::remove(f, ec);
    }
    for (const std::string &f : {head_shard, plain_shard}) {
        std::filesystem::remove_all(std::filesystem::path(f).parent_path(), ec);
    }
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

namespace {

// The two places a non-ASCII path takes in the tests below: a directory with an ASCII file name, and
// an ASCII directory with a file name outside ASCII. Both hold an umlaut (in cp1252) and a CJK
// character (in no single-byte code page). The C API takes the path as UTF-8 on every platform.
struct non_ascii_place {
    const char *what;
    std::filesystem::path dir;
    std::string stem;
};

std::vector<non_ascii_place> non_ascii_places(const std::string &prefix) {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / (prefix + "-" + std::to_string(std::random_device{}()));
    const std::filesystem::path in_dir = base / std::filesystem::u8path("d-\xC3\xA4-\xE6\x97\xA5");
    std::filesystem::create_directories(in_dir);
    return {{"a non-ASCII directory", in_dir, "f"}, {"a non-ASCII file name", base, "f-\xC3\xA4-\xE6\x97\xA5"}};
}

std::filesystem::path in_place(const non_ascii_place &p, const std::string &suffix) {
    return p.dir / std::filesystem::u8path(p.stem + suffix);
}

std::string utf8(const std::filesystem::path &p) {
    const auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

// A LoRA adapter for write_tiny_llama_gguf()'s model (rank 4 on attn_k and attn_v).
std::string write_tiny_lora_for_paths() {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("jllama-c-api-utf8-lora-" + std::to_string(std::random_device{}()) + ".gguf"))
                                 .string();
    std::mt19937 rng(7);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", 4.0f);
    ggml_init_params ip = {4u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char *name : {"attn_k", "attn_v"}) {
            add_random_tensor(g, ctx, rng, p + name + ".weight.lora_a", {N_EMBD, 4}, 1.0f);
            add_random_tensor(g, ctx, rng, p + name + ".weight.lora_b", {4, N_HEAD_KV * HEAD_DIM}, 1.0f);
        }
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

} // namespace

TEST(CApiNonAsciiPaths, TheModelFilesAreReadUnderANonAsciiPath) {
    // -m, --shard and --lora are opened through ggml_fopen, and --mmproj through gguf and a wide
    // std::ifstream. All of them read the path as UTF-8 on Windows: measured to work on code page
    // 1252, where the narrow-string sites since b12131 fail. This test holds them there.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-utf8-model");
    std::string shard;
    const std::string split = write_tiny_llama_gguf("jllama-c-api-utf8-split", false, '>', {}, &shard);
    const std::string mmproj = write_tiny_mmproj_gguf("jllama-c-api-utf8-mmproj");
    const std::string lora = write_tiny_lora_for_paths();
    ASSERT_FALSE(model.empty());
    ASSERT_FALSE(split.empty());
    ASSERT_FALSE(mmproj.empty());
    ASSERT_FALSE(lora.empty());
    for (const auto &place : non_ascii_places("jllama-c-api-utf8-models")) {
        SCOPED_TRACE(place.what);
        const auto m = in_place(place, ".gguf");
        const auto s = in_place(place, ".shard.gguf");
        const auto v = in_place(place, ".mmproj.gguf");
        const auto a = in_place(place, ".lora.gguf");
        std::filesystem::copy_file(model, m);
        std::filesystem::copy_file(shard, s);
        std::filesystem::copy_file(mmproj, v);
        std::filesystem::copy_file(lora, a);
        struct row {
            const char *option;
            std::string model;
            std::vector<std::string> extra;
        };
        for (const row &r :
             {row{"-m", utf8(m), {}}, row{"--shard", split, {"--shard", utf8(s)}},
              row{"--mmproj", model, {"-c", "128", "--mmproj", utf8(v)}}, row{"--lora", model, {"--lora", utf8(a)}}}) {
            SCOPED_TRACE(r.option);
            jllama_model *loaded = load_tiny(r.model, r.extra);
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
            EXPECT_FALSE(greedy_answer(loaded).empty());
            jllama_model_free(loaded);
        }
        std::error_code ignored;
        for (const auto &f : {m, s, v, a}) {
            std::filesystem::remove(f, ignored);
        }
        std::filesystem::remove_all(place.dir, ignored);
    }
    std::error_code ignored;
    for (const auto &f : {model, split, shard, mmproj, lora}) {
        std::filesystem::remove(f, ignored);
    }
}

TEST(CApiNonAsciiPaths, TheMediaDirectoryIsCheckedAsUtf8) {
    // --media-path goes through the same directory check as --slot-save-path (fs_is_directory,
    // patches/0024): a directory outside ASCII has to be accepted on every code page, and one that
    // does not exist refused (the option's own message goes to the log, the load reports the parse).
    // And a file below it has to be read from there: a "file://" URL of a request went through a
    // narrow std::ifstream (patches/0029), so the directory was accepted and nothing in it found.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-utf8-media");
    const std::string mmproj = write_tiny_mmproj_gguf();
    ASSERT_FALSE(model.empty());
    ASSERT_FALSE(mmproj.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        std::filesystem::u8path("jllama-media-\xC3\xA4-\xE6\x97\xA5-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    {
        std::ofstream png(dir / std::filesystem::u8path("b\xC3\xA4r.png"), std::ios::binary);
        const std::vector<uint8_t> bytes = base64_bytes(TINY_PNG_BASE64);
        png.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(png.good());
    }
    jllama_model *loaded = load_tiny(model, {"--media-path", utf8(dir), "--mmproj", mmproj});
    ASSERT_NE(loaded, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(loaded).empty());
    const auto look_at = [](const std::string &url) {
        return json{
            {"messages",
             json::array({{{"role", "user"},
                           {"content", json::array({{{"type", "text"}, {"text", "What is this?"}},
                                                    {{"type", "image_url"}, {"image_url", {{"url", url}}}}})}}})},
            {"max_tokens", 1},
            {"temperature", 0}}
            .dump();
    };
    Owned seen;
    ASSERT_EQ(jllama_chat_completions(loaded, look_at("file://b\xC3\xA4r.png").c_str(), &seen.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    // The image has to be in the prompt, not only accepted: the same message without it is shorter by
    // at least the projector's four patch embeddings.
    const std::string text_only =
        json{{"messages", json::array({{{"role", "user"},
                                        {"content", json::array({{{"type", "text"}, {"text", "What is this?"}}})}}})},
             {"max_tokens", 1},
             {"temperature", 0}}
            .dump();
    Owned unseen;
    ASSERT_EQ(jllama_chat_completions(loaded, text_only.c_str(), &unseen.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GE(json::parse(seen.str()).at("usage").at("prompt_tokens").get<int>(),
              json::parse(unseen.str()).at("usage").at("prompt_tokens").get<int>() + 4)
        << seen.str() << unseen.str();
    Owned missing;
    EXPECT_EQ(jllama_chat_completions(loaded, look_at("file://missing.png").c_str(), &missing.s),
              JLLAMA_STATUS_INVALID_REQUEST)
        << "a file that is not in the media directory was read from somewhere";
    jllama_model_free(loaded);

    jllama_model *refused = load_tiny(model, {"--media-path", utf8(dir / "missing")});
    EXPECT_EQ(refused, nullptr) << "a media directory that does not exist was accepted";
    jllama_model_free(refused);

    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove(model, ignored);
    std::filesystem::remove(mmproj, ignored);
}

// Sets (or, with nullptr, removes) an environment variable from UTF-8 text, the way a shell or a
// parent process hands one to the JVM: UTF-16 on Windows, bytes elsewhere.
void set_env_utf8(const char *name, const char *value) {
#ifdef _WIN32
    const std::wstring wname(name, name + std::strlen(name));
    const std::wstring wvalue = value != nullptr ? std::filesystem::u8path(value).wstring() : std::wstring();
    _wputenv_s(wname.c_str(), wvalue.c_str()); // an empty value removes the variable
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

TEST(CApiNonAsciiPaths, AnOptionFromTheEnvironmentIsReadAsUtf8) {
    // llama.cpp takes every option from an environment variable too (LLAMA_ARG_SLOT_SAVE_PATH), and
    // read it with std::getenv(): the active code page on Windows, where the handlers and the file
    // sites behind them expect UTF-8 (patches/0030). The variable is applied before the arguments,
    // so a path outside ASCII in it failed the load whatever the arguments said.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-utf8-env");
    ASSERT_FALSE(model.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        std::filesystem::u8path("jllama-env-\xC3\xA4-\xE6\x97\xA5-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));

    set_env_utf8("LLAMA_ARG_SLOT_SAVE_PATH", utf8(dir).c_str());
    jllama_model *loaded = load_tiny(model, {});
    set_env_utf8("LLAMA_ARG_SLOT_SAVE_PATH", nullptr);
    ASSERT_NE(loaded, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(loaded).empty());
    Owned saved;
    ASSERT_EQ(jllama_slot_action(loaded, JLLAMA_SLOT_SAVE, 0, "env.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::error_code ec;
    EXPECT_GT(std::filesystem::file_size(dir / "env.bin", ec), 0u) << "the slot file is not in the directory named";
    jllama_model_free(loaded);

    // and the variable is gone again: the next load has no slot directory
    jllama_model *plain = load_tiny(model, {});
    ASSERT_NE(plain, nullptr) << jllama_last_error();
    Owned refused;
    EXPECT_EQ(jllama_slot_action(plain, JLLAMA_SLOT_SAVE, 0, "env.bin", &refused.s), JLLAMA_STATUS_ERROR);
    jllama_model_free(plain);

    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiNonAsciiPaths, TheLogFileIsWrittenUnderItsNonAsciiName) {
    // --log-file went through the C runtime's narrow fopen (patches/0027): on Windows the file was
    // created under the path as the active code page reads it. Freeing the model closes the file
    // (CApiLogFile).
    const std::string model = write_tiny_llama_gguf("jllama-c-api-utf8-log");
    ASSERT_FALSE(model.empty());
    for (const auto &place : non_ascii_places("jllama-c-api-utf8-log")) {
        SCOPED_TRACE(place.what);
        const auto log = in_place(place, ".log");
        jllama_model *loaded = load_tiny(model, {"--log-file", utf8(log)});
        ASSERT_NE(loaded, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(loaded).empty());
        jllama_model_free(loaded); // closes the file the load opened
        std::error_code ec;
        EXPECT_TRUE(std::filesystem::exists(log, ec)) << "no log file under the name that was asked for";
        EXPECT_GT(std::filesystem::file_size(log, ec), 0u) << ec.message();
        std::filesystem::remove(log, ec);
        std::filesystem::remove_all(place.dir, ec);
    }
    std::error_code ignored;
    std::filesystem::remove(model, ignored);
}

namespace {
// The size of a log file once common_log has written what it queued: swapping the log callback
// pauses common_log's worker, which prints every queued entry first.
std::uintmax_t drained_size(const std::filesystem::path &file) {
    jllama_set_log_callback(nullptr, nullptr);
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    return ec ? 0 : size;
}

std::filesystem::path log_file_path(const std::string &name) {
    return std::filesystem::temp_directory_path() / (name + ".log");
}
} // namespace

TEST(CApiLogFile, AFreedModelClosesTheLogFileItsLoadOpened) {
    // --log-file sets the file of the process-wide common_log, and nothing closed it again: every
    // later model of the process wrote into it, and Windows could not delete it while the JVM ran.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file");
    ASSERT_FALSE(model.empty());
    const auto log = log_file_path("jllama-c-api-log-file");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    jllama_model *first = load_tiny(model, {"--log-file", log.string()});
    ASSERT_NE(first, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(first).empty());
    jllama_model_free(first);
    const auto written = drained_size(log);
    EXPECT_GT(written, 0u) << "the model logged nothing into its file";

    jllama_model *second = load_tiny(model, {});
    ASSERT_NE(second, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(second).empty());
    jllama_model_free(second);
    EXPECT_EQ(drained_size(log), written) << "a model without --log-file wrote into the file of a freed one";
    EXPECT_TRUE(std::filesystem::remove(log, ec)) << ec.message();
    std::filesystem::remove(model, ec);
}

TEST(CApiLogFile, AFailedLoadClosesTheLogFileItOpened) {
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-failed");
    ASSERT_FALSE(model.empty());
    const auto log = log_file_path("jllama-c-api-log-file-failed");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    const auto missing = std::filesystem::temp_directory_path() / "jllama-c-api-log-file-no-such-model.gguf";
    EXPECT_EQ(load_tiny(missing.string(), {"--log-file", log.string()}), nullptr);
    const auto written = drained_size(log);

    jllama_model *other = load_tiny(model, {});
    ASSERT_NE(other, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(other).empty());
    jllama_model_free(other);
    EXPECT_EQ(drained_size(log), written) << "the file of a failed load stayed open";
    std::filesystem::remove(log, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiLogFile, ALaterLogFileTakesTheFileAndItsOwnershipOver) {
    // common_log has one file: a second --log-file replaces the first, and the model that opened
    // the current one closes it. Freeing the first model must not close the second's file.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-two");
    ASSERT_FALSE(model.empty());
    const auto log_a = log_file_path("jllama-c-api-log-file-a");
    const auto log_b = log_file_path("jllama-c-api-log-file-b");
    std::error_code ec;
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(log_b, ec);
    jllama_model *a = load_tiny(model, {"--log-file", log_a.string()});
    ASSERT_NE(a, nullptr) << jllama_last_error();
    jllama_model *b = load_tiny(model, {"--log-file", log_b.string()});
    ASSERT_NE(b, nullptr) << jllama_last_error();
    jllama_model_free(a);
    const auto a_written = drained_size(log_a);
    const auto b_before = drained_size(log_b);

    EXPECT_FALSE(greedy_answer(b).empty());
    EXPECT_GT(drained_size(log_b), b_before) << "freeing the first model closed the second's file";
    EXPECT_EQ(drained_size(log_a), a_written);

    jllama_model_free(b);
    const auto b_written = drained_size(log_b);
    jllama_model *c = load_tiny(model, {});
    ASSERT_NE(c, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(c).empty());
    jllama_model_free(c);
    EXPECT_EQ(drained_size(log_b), b_written) << "the second file stayed open after its model was freed";
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(log_b, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiNonAsciiPaths, TheImatrixQuantizerAndTrainerReadAndWriteUnderANonAsciiPath) {
    // The paths of the file tools that go through ggml_fopen or gguf: the model each one reads, the
    // importance matrix that jllama_imatrix writes as GGUF and the quantizer reads, and the model the
    // trainer writes. Measured to work on code page 1252. The calibration text, the trainer's
    // training_file and the legacy imatrix output are this library's own streams, which failed there
    // until they read the path as UTF-8, and so did the quantizer's output, a narrow std::ofstream
    // in llama.cpp (patches/0028). See docs/handover/local-agent-report-windows-non-ascii-paths.md.
    tiny_model_spec spec = tiny_model_spec::wide(256, "jllama-c-api-utf8-tools");
    spec.n_layer = 1;
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text = write_calibration_text("jllama-c-api-utf8-tools", 200);
    std::string corpus;
    for (int i = 0; i < 16; ++i) {
        corpus += "the quick brown fox jumps over the lazy dog. ";
    }
    for (const auto &place : non_ascii_places("jllama-c-api-utf8-tools")) {
        SCOPED_TRACE(place.what);
        const auto in = in_place(place, ".gguf");
        const auto imatrix = in_place(place, ".imatrix.gguf");
        const auto trained = in_place(place, ".finetuned.gguf");
        const auto text_here = in_place(place, ".txt");
        const auto corpus_here = in_place(place, ".corpus.txt");
        const auto legacy = in_place(place, ".imatrix.dat");
        std::filesystem::copy_file(model, in);
        std::filesystem::copy_file(text, text_here);
        {
            std::ofstream out(corpus_here, std::ios::binary);
            out << corpus;
        }
        ASSERT_EQ(jllama_imatrix(utf8(in).c_str(), utf8(text_here).c_str(), utf8(imatrix).c_str(), 32, 0, 2, 0, nullptr,
                                 nullptr, nullptr),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        std::error_code ec;
        EXPECT_GT(std::filesystem::file_size(imatrix, ec), 0u) << ec.message();

        // the legacy format is written by this library, under the path's real name
        const std::string imatrix_u8 = utf8(imatrix);
        const char *one[] = {imatrix_u8.c_str()};
        int chunks = 0;
        ASSERT_EQ(jllama_imatrix_merge(one, 1, utf8(legacy).c_str(), true, &chunks), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_GT(std::filesystem::file_size(legacy, ec), 0u) << ec.message();

        // ... and read back from there (patches/0026: upstream's legacy reader took the narrow path)
        const std::string legacy_u8 = utf8(legacy);
        const char *one_legacy[] = {legacy_u8.c_str()};
        const std::string back = model + ".utf8-legacy-back.imatrix.gguf";
        int chunks_back = 0;
        ASSERT_EQ(jllama_imatrix_merge(one_legacy, 1, back.c_str(), false, &chunks_back), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(chunks_back, chunks);
        const std::string from_legacy = model + ".utf8-legacy-in.gguf";
        ASSERT_EQ(jllama_quantize_imatrix(model.c_str(), from_legacy.c_str(), 15, 1, false, legacy_u8.c_str()),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_GT(std::filesystem::file_size(from_legacy, ec), 0u) << ec.message();
        std::filesystem::remove(back, ec);
        std::filesystem::remove(from_legacy, ec);

        // the quantizer's output too (patches/0028: a narrow std::ofstream in llama.cpp before)
        const auto quantized = in_place(place, ".q4.gguf");
        ASSERT_EQ(
            jllama_quantize_imatrix(utf8(in).c_str(), utf8(quantized).c_str(), 15, 1, false, utf8(imatrix).c_str()),
            JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_GT(std::filesystem::file_size(quantized, ec), 0u) << ec.message();
        std::filesystem::remove(quantized, ec);

        const json config = {{"model_path", utf8(in)},
                             {"training_file", utf8(corpus_here)},
                             {"output_path", utf8(trained)},
                             {"epochs", 1},
                             {"learning_rate", 1e-4},
                             {"n_ctx", 64},
                             {"n_gpu_layers", 0},
                             {"val_split", 0.0}};
        ASSERT_EQ(jllama_finetune(config.dump().c_str()), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_GT(std::filesystem::file_size(trained, ec), 0u) << ec.message();
        for (const auto &f : {in, imatrix, trained, text_here, corpus_here, legacy}) {
            std::filesystem::remove(f, ec);
        }
        std::filesystem::remove_all(place.dir, ec);
    }
    std::error_code ignored;
    std::filesystem::remove(model, ignored);
    std::filesystem::remove(text, ignored);
}
