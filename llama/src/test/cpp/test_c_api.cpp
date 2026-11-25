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
#include "numeric_locale.hpp"

#ifdef JLLAMA_HAS_NATIVE_SERVER
#include <cpp-httplib/httplib.h>
#endif

#include "build-info.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <random>
#include <regex>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#endif
#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

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

// Every line the library logs while it lives (the log callback is process-wide), as one text.
// stop() ends the capture and returns once the queued lines have reached it.
class LogCapture {
  public:
    LogCapture() {
        EXPECT_EQ(jllama_set_log_callback(
                      [](int, const char *text, void *user) {
                          auto *self = static_cast<LogCapture *>(user);
                          std::lock_guard<std::mutex> lock(self->mutex_);
                          self->text_ += text;
                      },
                      this),
                  JLLAMA_STATUS_OK);
    }
    ~LogCapture() { stop(); }
    LogCapture(const LogCapture &) = delete;
    LogCapture &operator=(const LogCapture &) = delete;

    void stop() {
        if (active_) {
            jllama_set_log_callback(nullptr, nullptr);
            active_ = false;
        }
    }
    std::string text() {
        std::lock_guard<std::mutex> lock(mutex_);
        return text_;
    }
    int count(const std::string &needle) {
        std::lock_guard<std::mutex> lock(mutex_);
        int n = 0;
        for (size_t at = text_.find(needle); at != std::string::npos; at = text_.find(needle, at + needle.size())) {
            ++n;
        }
        return n;
    }
    // Waits until `needle` was logged at least `n` times; false after `timeout`.
    bool wait_for(const std::string &needle, int n, std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (count(needle) < n) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return true;
    }

  private:
    std::mutex mutex_;
    std::string text_;
    bool active_ = true;
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

TEST_F(CApi, TheAdaptivePSamplerAnswersWhenTheChainNamesItAndItsDecayIsRangeChecked) {
    // InferenceParameters.withSamplers(..., Sampler.ADAPTIVE_P) with withAdaptiveTarget / Decay:
    // the sampler replaces the final random draw, so with a seed it answers the same twice; a decay
    // outside upstream's hard limits (0..0.99) is the request's fault.
    const auto answer = [](const json &extra) {
        json request = {{"prompt", "Hello there"}, {"n_predict", 12}, {"seed", 7}, {"cache_prompt", false}};
        request.update(extra);
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()).at("content").get<std::string>() : std::string();
    };
    const json adaptive = {
        {"samplers", json::array({"top_k", "adaptive_p"})}, {"adaptive_target", 0.3}, {"adaptive_decay", 0.9}};
    const std::string once = answer(adaptive);
    EXPECT_FALSE(once.empty());
    EXPECT_EQ(answer(adaptive), once) << "the adaptive-p draw follows the seed";
    // another target is accepted too (what it picks depends on the random model)
    json other_target = adaptive;
    other_target["adaptive_target"] = 0.9;
    EXPECT_FALSE(answer(other_target).empty());

    for (const double decay : {-0.1, 1.5}) {
        json request = {{"prompt", "Hello there"},
                        {"n_predict", 2},
                        {"samplers", json::array({"adaptive_p"})},
                        {"adaptive_decay", decay}};
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_INVALID_REQUEST)
            << decay;
        EXPECT_NE(std::string(jllama_last_error()).find("adaptive_decay"), std::string::npos) << jllama_last_error();
    }
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

namespace {

// A request the server refuses only once a slot takes it (a prompt longer than the context, a
// grammar that does not parse): every path reports it as the request's fault with the server's
// reason, a stream on its first receive, after which the task is gone; and the model answers the
// next request. raw_extra makes the bad completion request, chat_extra the bad chat request.
void expect_refused_on_every_path(jllama_model *model, const json &raw_extra, const json &chat_extra,
                                  const std::string &reason) {
    const auto refused = [&reason](jllama_status status, const char *path) {
        EXPECT_EQ(status, JLLAMA_STATUS_INVALID_REQUEST) << path << ": " << jllama_last_error();
        EXPECT_NE(std::string(jllama_last_error()).find(reason), std::string::npos)
            << path << ": " << jllama_last_error();
    };
    json raw = {{"n_predict", 2}, {"temperature", 0}};
    raw.update(raw_extra);
    json chat = {{"max_tokens", 2}, {"temperature", 0}};
    chat.update(chat_extra);

    Owned response;
    refused(jllama_completions(model, raw.dump().c_str(), &response.s), "blocking raw completion");
    EXPECT_EQ(response.s, nullptr);
    json oai = raw;
    oai.erase("n_predict");
    oai["max_tokens"] = 2;
    refused(jllama_completions_oai(model, oai.dump().c_str(), &response.s), "blocking OAI completion");
    refused(jllama_chat_completions(model, chat.dump().c_str(), &response.s), "blocking chat completion");

    // the streams: the request is accepted, the slot refuses it, and the first receive says so
    const auto stream_refused = [&](int task, bool raw_receive, const char *path) {
        ASSERT_GE(task, 0) << path;
        Owned envelope;
        bool stop = false;
        refused(raw_receive ? jllama_completions_stream_receive(model, task, &envelope.s, &stop)
                            : jllama_chat_completions_stream_receive(model, task, &envelope.s, &stop),
                path);
        EXPECT_EQ(envelope.s, nullptr) << path;
        Owned again;
        EXPECT_EQ(raw_receive ? jllama_completions_stream_receive(model, task, &again.s, &stop)
                              : jllama_chat_completions_stream_receive(model, task, &again.s, &stop),
                  JLLAMA_STATUS_ERROR)
            << path;
        EXPECT_STREQ(jllama_last_error(), "Task not found") << path;
    };
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, raw.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    stream_refused(task, true, "raw stream");
    task = -1;
    ASSERT_EQ(jllama_completions_oai_stream_request(model, oai.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    stream_refused(task, false, "OAI stream");
    task = -1;
    ASSERT_EQ(jllama_chat_completions_stream_request(model, chat.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    stream_refused(task, false, "chat stream");
    task = -1;
    ASSERT_EQ(jllama_chat_completions_request(model, chat.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    stream_refused(task, true, "chat request with raw results");

    Owned next;
    ASSERT_EQ(jllama_completions(model, R"({"prompt":"Hi","n_predict":2,"temperature":0})", &next.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_TRUE(json::parse(next.str()).at("stop").get<bool>()) << next.str();
}

} // namespace

TEST_F(CApi, APromptLongerThanTheContextIsTheCallersFaultOnEveryRequestPath) {
    // 200 byte tokens against a context of 64: the server refuses the task when a slot takes it
    // (ERROR_TYPE_EXCEED_CONTEXT_SIZE, upstream's HTTP 400), on the blocking paths and on the
    // streams alike, and the slot is free for the next request.
    const std::string long_text(200, 'a');
    expect_refused_on_every_path(model_, {{"prompt", long_text}},
                                 {{"messages", json::array({json{{"role", "user"}, {"content", long_text}}})}},
                                 "exceeds the available context size");
}

TEST_F(CApi, AGrammarThatDoesNotParseIsTheCallersFaultOnEveryRequestPath) {
    // The grammar is parsed when the slot builds its sampler (common_sampler_init); a grammar that
    // does not parse is an invalid request there ("Failed to initialize samplers"), not a crash.
    const json grammar = {{"grammar", "root ::= ("}};
    json raw = grammar;
    raw["prompt"] = "Hi";
    json chat = grammar;
    chat["messages"] = json::array({json{{"role", "user"}, {"content", "Hi"}}});
    expect_refused_on_every_path(model_, raw, chat, "failed to parse grammar");
}

TEST_F(CApi, AGrammarWhoseRepetitionEndsBeforeItBeginsIsRefusedWithItsReason) {
    // b12604 (#32551): "a"{3,2} is refused while the grammar is parsed ("incorrect number of
    // repetitions"). Before it, the number of optional repetitions, max - min, wrapped around and the
    // parser added rules until the memory ran out. The parser prints its reason to stderr only, so the
    // test reads it there: an allocation failure is a parse error too and would be refused the same way.
    const json grammar = {{"grammar", "root ::= \"a\"{3,2}"}};
    json raw = grammar;
    raw["prompt"] = "Hi";
    json chat = grammar;
    chat["messages"] = json::array({json{{"role", "user"}, {"content", "Hi"}}});
    testing::internal::CaptureStderr();
    expect_refused_on_every_path(model_, raw, chat, "failed to parse grammar");
    const std::string printed = testing::internal::GetCapturedStderr();
    EXPECT_NE(printed.find("error parsing grammar: incorrect number of repetitions"), std::string::npos) << printed;
    // a bound that is in order still constrains the answer
    Owned answer;
    ASSERT_EQ(jllama_completions(model_,
                                 R"({"prompt":"Hi","n_predict":8,"temperature":0,"grammar":"root ::= \"a\"{2,3}"})",
                                 &answer.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    // two or three a, then the end of generation, whose piece is the '>' of this byte vocabulary
    const std::string content = json::parse(answer.str()).at("content").get<std::string>();
    EXPECT_TRUE(std::regex_match(content, std::regex("a{2,3}>?"))) << answer.str();
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

TEST_F(CApi, ABlockingCallAnswersAStreamRequestWhole) {
    // "stream": true in the body of a blocking call made the server send the answer in chunks, which
    // the blocking wait collects by their index -- every chunk has index 0, and the second one ended
    // the process (GGML_ASSERT "duplicate result received" in server_response_reader::wait_for_all).
    // A blocking call answers the whole result whatever the body says.
    Owned raw;
    ASSERT_EQ(jllama_completions(model_, R"({"prompt":"Hi","n_predict":4,"temperature":0,"stream":true})", &raw.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_TRUE(json::parse(raw.str()).at("stop").get<bool>()) << raw.str();
    Owned chat;
    ASSERT_EQ(jllama_chat_completions(
                  model_,
                  R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0,"stream":true})",
                  &chat.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(chat.str()).at("object").get<std::string>(), "chat.completion") << chat.str();
    Owned oai;
    ASSERT_EQ(jllama_completions_oai(model_, R"({"prompt":"Hi","max_tokens":4,"temperature":0,"stream":true})", &oai.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(oai.str()).at("object").get<std::string>(), "text_completion") << oai.str();
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

TEST_F(CApi, ATokenIdOutsideTheVocabularyIsTheCallersFaultAndTheModelGoesOn) {
    // An id the vocabulary does not have is the caller's mistake. In a prompt the server refuses it
    // when the slot takes the task ("Prompt contains invalid tokens", upstream's 400). Detokenizing
    // one read past the vocabulary's table (llama_vocab::token_get_attr, an at() that throws
    // std::out_of_range), which surfaced as a server error with the C++ library's message; it is
    // refused by name, before llama.cpp is asked.
    for (const int id : {N_VOCAB, 99999, -5}) {
        SCOPED_TRACE(id);
        const int ids[] = {'H', id};
        Owned text;
        size_t length = 0;
        EXPECT_EQ(jllama_detokenize(model_, ids, 2, &text.s, &length), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find(std::to_string(id)), std::string::npos) << jllama_last_error();
        EXPECT_EQ(text.s, nullptr);
        Owned detokenized;
        EXPECT_EQ(jllama_detokenize_json(model_, ids, 2, &detokenized.s), JLLAMA_STATUS_INVALID_REQUEST);

        const json prompt = json::array({'H', id});
        Owned response;
        EXPECT_EQ(jllama_completions(model_, json{{"prompt", prompt}, {"n_predict", 2}}.dump().c_str(), &response.s),
                  JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("invalid tokens"), std::string::npos) << jllama_last_error();
        EXPECT_EQ(
            jllama_completions_oai(model_, json{{"prompt", prompt}, {"max_tokens", 2}}.dump().c_str(), &response.s),
            JLLAMA_STATUS_INVALID_REQUEST);
        int task = -1;
        ASSERT_EQ(
            jllama_completions_stream_request(model_, json{{"prompt", prompt}, {"n_predict", 2}}.dump().c_str(), &task),
            JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned payload;
        bool stop = false;
        EXPECT_EQ(jllama_completions_stream_receive(model_, task, &payload.s, &stop), JLLAMA_STATUS_INVALID_REQUEST);
    }
    // the model and its vocabulary are unharmed
    const int last[] = {'H', N_VOCAB - 1};
    Owned text;
    size_t length = 0;
    EXPECT_EQ(jllama_detokenize(model_, last, 2, &text.s, &length), JLLAMA_STATUS_OK) << jllama_last_error();
    Owned next;
    ASSERT_EQ(jllama_completions(model_, R"({"prompt":[72,105],"n_predict":2,"temperature":0})", &next.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
}

TEST_F(CApi, ARequestMaySampleOnTheBackendAndAnswersAsOnTheCpu) {
    // "backend_sampling" (upstream's per-request field, --backend-sampling for the server) moves the
    // sampler chain into the graph where the backend supports it; greedily both places take the argmax
    // of the same logits, so the answer is the one sampled on the CPU. The echoed flag and the equal
    // answer would hold just as well if the field were ignored, so each answer also counts the tokens
    // the backend sampled: common_sampler_sample() logs every one it takes from the graph
    // ("Backend sampler selected token", a debug line, verbosity 5) instead of running the CPU chain.
    struct VerbosityReset {
        const int previous = common_log_get_verbosity_thold();
        ~VerbosityReset() { common_log_set_verbosity_thold(previous); }
    } verbosity_reset;
    const auto answer = [](bool backend, int *backend_sampled) {
        const json request = {{"prompt", "Hello there"},     {"n_predict", 8},     {"temperature", 0},
                              {"backend_sampling", backend}, {"ignore_eos", true}, {"cache_prompt", false}};
        Owned response;
        {
            LogCapture capture;
            common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
            EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            capture.stop();
            *backend_sampled = capture.count("Backend sampler selected token");
        }
        const json j = response.s != nullptr ? json::parse(response.str()) : json::object();
        EXPECT_EQ(j.at("timings").at("cache_n").get<int>(), 0) << j.dump();
        EXPECT_EQ(j.at("generation_settings").at("backend_sampling").get<bool>(), backend) << j.dump();
        return j.at("content").get<std::string>();
    };
    int sampled_on_cpu = -1;
    const std::string on_cpu = answer(false, &sampled_on_cpu);
    EXPECT_EQ(on_cpu.size(), 8u);
    EXPECT_EQ(sampled_on_cpu, 0) << "a request without backend sampling sampled on the backend";
    int sampled_on_backend = -1;
    EXPECT_EQ(answer(true, &sampled_on_backend), on_cpu);
    EXPECT_EQ(sampled_on_backend, 8) << "the backend did not sample every token";
}

TEST_F(CApi, ConfigureIgnoresAKeyItDoesNotKnow) {
    // jllama_configure reads n_threads, n_threads_batch and slot_prompt_similarity and nothing else: a
    // key it does not know -- a misspelling among them -- is accepted and changes nothing, as an unknown
    // field of a request is. A known key with a value it cannot take is refused.
    EXPECT_EQ(jllama_configure(model_, R"({"n_thread":1})"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({"unknown":{"nested":true}})"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({})"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({"n_threads":0})"), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(jllama_configure(model_, R"({"n_threads":"two"})"), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(jllama_configure(model_, R"([1,2])"), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(jllama_configure(model_, R"({"n_threads":2})"), JLLAMA_STATUS_OK) << jllama_last_error();
    Owned next;
    EXPECT_EQ(jllama_completions(model_, R"({"prompt":"Hi","n_predict":2,"temperature":0})", &next.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
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

TEST(CApiSlotFiles, ASlotFileCutInsideItsStateIsRefusedAndTheModelGoesOn) {
    // llama_file's reads past the end of a file are no error on POSIX (they stop short and count the
    // whole length), so llama_state_seq_load_file read a cut slot file as a state larger than the file
    // and asserted -- the process ended on the restore of a file the caller names (patches/0035; on
    // Windows the read throws and the restore failed already). A cut is now an invalid slot file, the
    // slot is left empty, and the whole file still restores: every cut of the first 1024 bytes (the
    // header, the token list and the cells' positions and sequences, where a short read leaves a
    // value unread) and every 97th byte after them. A cut inside the header was refused by upstream
    // already (the token count exceeds the file). Measured once over every byte of the first 4096 and
    // every 61st after them, 4205 cuts of a 10.7 KB file: all refused, none aborted.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-cut");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-slot-cut-" + std::to_string(std::random_device{}()));
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
    const char *request =
        R"({"prompt":"Hello there, a prompt of some tokens","n_predict":4,"temperature":0,"cache_prompt":true})";
    const auto complete = [&]() {
        Owned completion;
        EXPECT_EQ(jllama_completions(model, request, &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return completion.s != nullptr ? json::parse(completion.str()) : json::object();
    };
    complete();
    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "whole.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const uintmax_t size = std::filesystem::file_size(dir / "whole.bin");
    ASSERT_GT(size, 64u);
    std::vector<uintmax_t> cuts;
    for (uintmax_t cut = 0; cut < size; cut += cut < 1024 ? 1 : 97) {
        cuts.push_back(cut);
    }
    cuts.push_back(size - 1);
    for (const uintmax_t cut : cuts) {
        if (::testing::Test::HasFailure()) {
            break; // the first cut that is not refused is the finding; the rest would repeat it
        }
        SCOPED_TRACE("cut at " + std::to_string(cut) + " of " + std::to_string(size));
        std::error_code ec;
        std::filesystem::copy_file(dir / "whole.bin", dir / "cut.bin",
                                   std::filesystem::copy_options::overwrite_existing, ec);
        ASSERT_FALSE(ec) << ec.message();
        std::filesystem::resize_file(dir / "cut.bin", cut, ec);
        ASSERT_FALSE(ec) << ec.message();
        Owned refused;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "cut.bin", &refused.s),
                  JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("invalid slot save file"), std::string::npos)
            << jllama_last_error();
        const json after = complete();
        EXPECT_EQ(after.at("timings").at("cache_n").get<int>(), 0) << "the cut state stayed: " << after.dump();
    }
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "whole.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json whole = complete();
    EXPECT_LE(whole.at("timings").at("prompt_n").get<int>(), 1) << whole.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
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
    EXPECT_EQ(json::parse(set.str()), json({{"success", true}})) << set.str(); // upstream's result of the set task
    // a body that is no array is the caller's mistake, as upstream's route answers it (400)
    EXPECT_EQ(jllama_set_lora_adapters(model_, "{}", &set.s), JLLAMA_STATUS_INVALID_REQUEST);
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

TEST_F(CApiVision, ARawCompletionWhoseMessagesCarryAnImageEvaluatesItsPatches) {
    // A raw completion with "messages" renders them through the chat template, and the media of
    // array-form content come back from the chat parser as files: the image reaches the projector,
    // as on the chat path, blocking and streamed. The reference is the prompt the template renders
    // for the same messages (jllama_apply_template, with the media marker where the image goes)
    // completed without the marker: every byte is one token here, so the image adds exactly its
    // embeddings. The chat path over the same messages, which has always handled images, agrees.
    const std::string marker = media_marker();
    ASSERT_FALSE(marker.empty());
    const json image_part = {{"type", "image_url"},
                             {"image_url", {{"url", std::string("data:image/png;base64,") + TINY_PNG_BASE64}}}};
    const json messages = json::array(
        {json{{"role", "user"}, {"content", json::array({json{{"type", "text"}, {"text", "look"}}, image_part})}}});
    Owned rendered;
    ASSERT_EQ(jllama_apply_template(model_, json{{"messages", messages}}.dump().c_str(), &rendered.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    std::string text_only = rendered.str();
    const size_t at = text_only.find(marker);
    ASSERT_NE(at, std::string::npos) << text_only;
    text_only.erase(at, marker.size());
    ASSERT_EQ(text_only.find(marker), std::string::npos) << "one image, one marker: " << rendered.str();
    Owned reference;
    ASSERT_EQ(
        jllama_completions(
            model_,
            json{{"prompt", text_only}, {"n_predict", 1}, {"temperature", 0}, {"cache_prompt", false}}.dump().c_str(),
            &reference.s),
        JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int text_tokens = json::parse(reference.str()).at("tokens_evaluated").get<int>();

    const json with_image = {{"messages", messages}, {"n_predict", 1}, {"temperature", 0}, {"cache_prompt", false}};
    Owned blocking;
    ASSERT_EQ(jllama_completions(model_, with_image.dump().c_str(), &blocking.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int with_image_tokens = json::parse(blocking.str()).at("tokens_evaluated").get<int>();
    // four 16x16 patches of the 32x32 image, one embedding each, where the template put the marker
    EXPECT_EQ(with_image_tokens, text_tokens + 4) << blocking.str();

    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model_, with_image.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(receive_to_stop(model_, task).at("tokens_evaluated").get<int>(), with_image_tokens);

    json chat = with_image;
    chat.erase("n_predict");
    chat["max_tokens"] = 1;
    Owned chat_answer;
    ASSERT_EQ(jllama_chat_completions(model_, chat.dump().c_str(), &chat_answer.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(chat_answer.str()).at("usage").at("prompt_tokens").get<int>(), with_image_tokens)
        << chat_answer.str();
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

TEST(CApiTemplate, AToolCallWhoseArgumentsWereCutOffIsTheCallersFaultAndTheHistoryWithoutItIsRendered) {
    // b12663 (#32738) makes the WebUI drop a tool call that Stop cut off mid-stream, with the tool result
    // that answers it, from the next request: for a template that reads a call's arguments as an object,
    // the server parses every call's arguments as JSON and refuses the whole request when one does not
    // parse (and an empty string is no JSON either, which is why the WebUI sends a call without
    // arguments as "{}"). This holds the server's half: such a history is the caller's fault on the
    // template and the chat path, the model answers the next request, and the history the WebUI sends
    // instead is rendered.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-cut-tool-call");
    ASSERT_FALSE(path.empty());
    // reading `arguments.arg` makes the template one with object arguments (common/jinja/caps.cpp)
    const std::string tmpl = "{%- for message in messages -%}"
                             "{%- if message.tool_calls -%}"
                             "{%- for tc in message.tool_calls -%}<call {{ tc.function.name }}:"
                             "{{ tc.function.arguments.arg }}>{%- endfor -%}"
                             "{%- else -%}<{{ message.role }}>{{ message.content }}{%- endif -%}"
                             "{%- endfor -%}"
                             "{%- if tools -%}[{{ tools[0].function.name }}]{%- endif -%}"
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

    const auto history = [](const char *arguments, bool with_call) {
        json messages = json::array({json{{"role", "user"}, {"content", "Hi"}}});
        if (with_call) {
            messages.push_back(json{
                {"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", json::array({json{{"type", "function"},
                                                 {"id", "c1"},
                                                 {"function", json{{"name", "lookup"}, {"arguments", arguments}}}}})}});
            messages.push_back(json{{"role", "tool"}, {"tool_call_id", "c1"}, {"content", "found"}});
        } else {
            messages.push_back(json{{"role", "assistant"}, {"content", ""}});
        }
        messages.push_back(json{{"role", "user"}, {"content", "next"}});
        return json{
            {"messages", messages},
            {"tools",
             json::array({json{{"type", "function"},
                               {"function", json{{"name", "lookup"}, {"parameters", json{{"type", "object"}}}}}}})},
            {"n_predict", 1},
            {"temperature", 0}};
    };

    Owned whole;
    ASSERT_EQ(jllama_apply_template(model, history(R"({"arg":"x"})", true).dump().c_str(), &whole.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(whole.str().find("<call lookup:x>"), std::string::npos) << whole.str();

    for (const char *cut : {R"({"arg":"x)", R"({"arg)", ""}) {
        SCOPED_TRACE(std::string("arguments ") + cut);
        const std::string request = history(cut, true).dump();
        Owned prompt;
        EXPECT_EQ(jllama_apply_template(model, request.c_str(), &prompt.s), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("Failed to parse tool call arguments as JSON"),
                  std::string::npos)
            << jllama_last_error();
        Owned answer;
        EXPECT_EQ(jllama_chat_completions(model, request.c_str(), &answer.s), JLLAMA_STATUS_INVALID_REQUEST);
    }

    // what the WebUI sends instead: the call and its result dropped, or a call without arguments as "{}"
    Owned dropped;
    ASSERT_EQ(jllama_apply_template(model, history("", false).dump().c_str(), &dropped.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(dropped.str(), "<user>Hi<assistant><user>next[lookup]<assistant>");
    // and the model answers (without tools: a random model's answer is no tool call)
    json plain = history("", false);
    plain.erase("tools");
    Owned answer;
    ASSERT_EQ(jllama_chat_completions(model, plain.dump().c_str(), &answer.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(answer.str()).value("object", std::string()), "chat.completion") << answer.str();
    Owned empty;
    EXPECT_EQ(jllama_apply_template(model, history("{}", true).dump().c_str(), &empty.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(empty.str().find("<call lookup:>"), std::string::npos) << empty.str();

    jllama_model_free(model);
    std::remove(path.c_str());
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

TEST(CApiTemplate, AReasoningEffortOfNoneReachesATemplateThatRendersIt) {
    // Since b12624 (#32763) "none" is a reasoning_effort level of its own: a template that renders it
    // lists it among its levels and is rendered with it, where the chat parser dropped the level and
    // only turned thinking off. A template that raises on "none" does not list it, and a request with
    // it is still answered, without the level.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-effort-none");
    ASSERT_FALSE(path.empty());
    const auto load = [&](const std::string &tmpl) {
        const std::vector<std::string> args = {"jllama", "-m",   path, "-c",          "128", "--chat-template",
                                               tmpl,     "-ngl", "0",  "--no-warmup", "-t",  "2"};
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
    const auto levels = [](jllama_model *model) {
        Owned meta;
        EXPECT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return json::parse(meta.str()).at("chat_template_caps").at("reasoning_effort_levels");
    };
    const auto render = [](jllama_model *model, const char *effort) {
        const json request = {{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
                              {"reasoning_effort", effort}};
        Owned prompt;
        EXPECT_EQ(jllama_apply_template(model, request.dump().c_str(), &prompt.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return prompt.s != nullptr ? prompt.str() : std::string();
    };
    const std::string body = "{%- for message in messages -%}"
                             "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
                             "{%- endfor -%}"
                             "[{{ reasoning_effort if reasoning_effort is defined else 'unset' }}]"
                             "{%- if add_generation_prompt -%}<assistant>{%- endif -%}";

    jllama_model *renders = load(body);
    ASSERT_NE(renders, nullptr);
    const json renders_levels = levels(renders);
    ASSERT_TRUE(renders_levels.is_array()) << renders_levels.dump();
    EXPECT_EQ(renders_levels.at(0), "none") << renders_levels.dump();
    EXPECT_NE(render(renders, "none").find("[none]"), std::string::npos);
    EXPECT_NE(render(renders, "high").find("[high]"), std::string::npos);
    jllama_model_free(renders);

    jllama_model *raises = load("{%- if reasoning_effort is defined and reasoning_effort == 'none' -%}"
                                "{{ raise_exception('none is not supported') }}{%- endif -%}" +
                                body);
    ASSERT_NE(raises, nullptr);
    const json raises_levels = levels(raises);
    EXPECT_EQ(std::find(raises_levels.begin(), raises_levels.end(), json("none")), raises_levels.end())
        << raises_levels.dump();
    EXPECT_NE(render(raises, "none").find("[unset]"), std::string::npos);
    jllama_model_free(raises);
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

// patches/0042: a schema whose bounds admit no value is refused while it is read, naming the bound. The
// converter wrote such bounds into the grammar as they came: a repetition {2,1} the grammar parser refuses
// since b12604 without the reason, one item for minItems 2 / maxItems 1, the empty string for minLength 3
// / maxLength 0, and the character range [6-5] for an integer from 6 to 5. Equal bounds and an open upper
// bound stay what they were, and a number's bounds, which the grammar does not carry, stay unchecked.
TEST(CApiStatic, ASchemaWhoseBoundsAdmitNoValueIsRefused) {
    const struct {
        const char *schema;
        const char *reason;
    } refused[] = {
        {R"({"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":2})", "minItems"},
        {R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":1})", "minItems"},
        {R"({"type":"array","minItems":1,"maxItems":0})", "minItems"},
        {R"({"type":"string","minLength":5,"maxLength":2})", "minLength"},
        {R"({"type":"string","minLength":3,"maxLength":0})", "minLength"},
        {R"({"type":"integer","minimum":6,"maximum":5})", "no integer"},
        {R"({"type":"integer","minimum":1.5,"maximum":1.6})", "no integer"},
        {R"({"type":"integer","exclusiveMinimum":5,"exclusiveMaximum":6})", "no integer"},
        {R"({"type":"object","properties":{"a":{"type":"string","minLength":2,"maxLength":1}}})", "/properties/a"},
    };
    for (const auto &c : refused) {
        Owned grammar;
        EXPECT_EQ(jllama_json_schema_to_grammar(c.schema, &grammar.s), JLLAMA_STATUS_ERROR) << c.schema;
        EXPECT_EQ(grammar.s, nullptr) << c.schema;
        EXPECT_NE(std::string(jllama_last_error()).find(c.reason), std::string::npos) << jllama_last_error();
    }
    const struct {
        const char *schema;
        const char *rule;
    } kept[] = {
        {R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2})", "{1,1}"},
        {R"({"type":"array","items":{"type":"integer"},"minItems":0,"maxItems":0})", R"("[" space  space "]")"},
        {R"({"type":"array","items":{"type":"integer"},"minItems":3})", "{2,}"},
        {R"({"type":"string","minLength":3,"maxLength":3})", "char{3,3}"},
        {R"({"type":"integer","minimum":5,"maximum":5})", R"(("5"))"},
        {R"({"type":"number","minimum":6,"maximum":5})", "integral-part"},
    };
    for (const auto &c : kept) {
        Owned grammar;
        ASSERT_EQ(jllama_json_schema_to_grammar(c.schema, &grammar.s), JLLAMA_STATUS_OK)
            << c.schema << ": " << jllama_last_error();
        EXPECT_NE(grammar.str().find(c.rule), std::string::npos) << c.schema << "\n" << grammar.str();
    }
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

// The prompts of upstream's prompt-cache tests (tools/server/tests/unit/test_completion.py, b12686),
// shortened for the random model's byte tokens: LONG starts with the text of SHORT, which covers more of it
// than the default --slot-prompt-similarity (0.1), and OTHER shares nothing with either. SHORT ends with
// one byte LONG does not have there, as upstream's SHORT does in tokens: with a BPE vocabulary its trailing
// space is a token of its own, which LONG merges into the next word. A slot holding SHORT therefore keeps
// all but its last token of LONG; one holding an exact token prefix would keep all its tokens, and the
// cache would rightly not be preferred (find_better takes an entry only for more kept and more matched).
const std::string PROMPT_CACHE_HEAD = "The capital of France is Paris. ";
const std::string PROMPT_CACHE_TAIL = "The river Seine runs slowly through the middle of the old stone city. ";
std::string repeated(const std::string &text, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) {
        out += text;
    }
    return out;
}
const std::string PROMPT_CACHE_LONG = PROMPT_CACHE_HEAD + repeated(PROMPT_CACHE_TAIL, 12);
const std::string PROMPT_CACHE_SHORT = PROMPT_CACHE_HEAD + repeated(PROMPT_CACHE_TAIL, 3) + "|";
const std::string PROMPT_CACHE_OTHER = repeated("Whiskers the cat climbed the tall oak tree behind the red barn. ", 6);

// The random model with a training context the long prompt fits: the server caps a slot's context at it.
std::string write_prompt_cache_model(const std::string &prefix) {
    tiny_model_spec spec;
    spec.prefix = prefix;
    spec.n_ctx_train = 2048;
    return write_tiny_llama_gguf(spec);
}

// Two slots and a host-memory prompt cache, as upstream's prompt_cache_server().
jllama_model *load_prompt_cache_model(const std::string &path) {
    return load_tiny(path, {"-c", "2048", "-np", "2", "-b", "512", "--cache-ram", "64"});
}

// prompt_n and cache_n of a prompt evaluated into the given slot, nothing generated.
std::pair<int, int> evaluate_into_slot(jllama_model *model, const std::string &prompt, int id_slot) {
    const json request = {
        {"prompt", prompt}, {"cache_prompt", true}, {"n_predict", 0}, {"temperature", 0}, {"id_slot", id_slot}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    if (response.s == nullptr) {
        return {-1, -1};
    }
    const json timings = json::parse(response.str()).at("timings");
    return {timings.at("prompt_n").get<int>(), timings.at("cache_n").get<int>()};
}

} // namespace

TEST(CApiPromptCache, AnEmptySlotNamedByIdTakesItsPromptFromTheHostMemoryCache) {
    // A request that names an empty slot passed the slot selection without asking the host-memory
    // cache: the selection decided whether the cache is updated only from the outgoing state, and an
    // empty slot has none to save. So the prompt was evaluated whole although the cache held its state
    // (upstream #28992, b12686, issues #28139 and #28276).
    const std::string path = write_prompt_cache_model("jllama-c-api-prompt-cache-empty");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_prompt_cache_model(path);
    ASSERT_NE(model, nullptr) << jllama_last_error();

    const auto cold = evaluate_into_slot(model, PROMPT_CACHE_LONG, 0);
    ASSERT_GT(cold.first, 100) << "the long prompt is evaluated whole the first time";
    // another prompt in slot 0 pushes the long prompt's state into the cache
    evaluate_into_slot(model, PROMPT_CACHE_OTHER, 0);
    // slot 1 is empty: the cache holds the exact state for the long prompt and is used
    const auto pinned = evaluate_into_slot(model, PROMPT_CACHE_LONG, 1);
    EXPECT_EQ(pinned.second, cold.first - pinned.first) << "prompt_n " << pinned.first;
    EXPECT_LT(pinned.first, cold.first) << "the long prompt was evaluated whole again";

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiPromptCache, ASlotHoldingAShorterPrefixTakesTheLongerStateFromTheHostMemoryCache) {
    // A slot that holds a prefix of the prompt was reused as it was, and the cache, which held the
    // state of the whole prompt, was not asked (#28992, b12686).
    const std::string path = write_prompt_cache_model("jllama-c-api-prompt-cache-prefix");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_prompt_cache_model(path);
    ASSERT_NE(model, nullptr) << jllama_last_error();

    // warm slot 1 before the long state reaches the cache, or this request would take that entry itself
    const auto shorter = evaluate_into_slot(model, PROMPT_CACHE_SHORT, 1);
    const auto cold = evaluate_into_slot(model, PROMPT_CACHE_LONG, 0);
    ASSERT_GT(cold.first, shorter.first);
    // another prompt in slot 0 pushes the long state into the cache
    evaluate_into_slot(model, PROMPT_CACHE_OTHER, 0);
    // slot 1 holds a prefix of the long prompt; the cache holds all of it and is the better start
    const auto reuse = evaluate_into_slot(model, PROMPT_CACHE_LONG, 1);
    EXPECT_EQ(reuse.second, cold.first - reuse.first) << "prompt_n " << reuse.first;
    EXPECT_LT(reuse.first, cold.first - shorter.first) << "only the slot's own prefix was reused";

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiPromptCache, AUnifiedCacheRestoresAPromptAfterClearingTheIdleSlots) {
    // With a unified KV cache (-kvu) all slots share one pool of cells. A prompt whose state the host
    // memory cache holds could not be restored when another idle slot still held the cells it needs;
    // the cache entry was kept, but the prompt was evaluated whole. Since b12699 (#30997, issue #30961)
    // the server saves and clears the other idle slots and retries the restore once. Upstream's
    // test_restore_under_cache_pressure, with the random model's byte tokens: two prompts that do not
    // fit the 512 cells together, taken in turns, each reusing what the cache holds of it.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-prompt-cache-unified";
    spec.n_ctx_train = 512;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    for (const char *n_slots : {"2", "3"}) {
        SCOPED_TRACE(std::string("slots ") + n_slots);
        jllama_model *model = load_tiny(path, {"-c", "512", "-np", n_slots, "-kvu", "--cache-ram", "100"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        const auto tokens = [](int token, size_t n) { return std::vector<int>(n, token); };
        const auto plus = [](std::vector<int> prompt, int token, size_t n) {
            prompt.insert(prompt.end(), n, token);
            return prompt;
        };
        const int a = 'H', b = 'd';
        const std::vector<int> prompt_a = tokens(a, 350), prompt_b = tokens(b, 300);
        const std::vector<std::pair<std::vector<int>, int>> steps = {
            {prompt_a, 0},
            {plus(prompt_a, a, 10), 350},
            {prompt_b, 0},
            {plus(prompt_a, a, 20), 360},
            {plus(prompt_b, b, 10), 300},
            {plus(prompt_a, a, 30), 370},
        };
        for (size_t i = 0; i < steps.size(); ++i) {
            const json request = {
                {"prompt", steps[i].first}, {"n_predict", 1}, {"cache_prompt", true}, {"temperature", 0}};
            Owned response;
            ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            const json timings = json::parse(response.str()).at("timings");
            EXPECT_GE(timings.at("cache_n").get<int>(), steps[i].second) << "step " << i << ": " << timings.dump();
            EXPECT_LE(timings.at("prompt_n").get<int>(), static_cast<int>(steps[i].first.size()) - steps[i].second)
                << "step " << i << ": " << timings.dump();
        }
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiTemplateStops, ARawCompletionWithMessagesKeepsTheCliStopsBesideTheTemplatesOwn) {
    // The completion schema falls back to the CLI's stops (-r, --reverse-prompt) when a request names
    // no effective stop; the raw paths write the template's stops into "stop", which would hide the
    // CLI's -- so add_template_stops falls back to them first. Without "stop", with "stop": [""] and
    // with "stop": "" (an empty word is dropped, upstream #32093), both the CLI's word and the
    // template's are in force. The string form was kept as a word, which hid the CLI's stop.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-stops-cli", false, 4);
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{#- laguna_glm_thinking -#}"
                             "{%- for message in messages -%}"
                             "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
                             "{%- endfor -%}"
                             "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    jllama_model *model = load_tiny(path, {"-c", "128", "--chat-template", tmpl, "-r", "zzz"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const json messages = json::array({json{{"role", "user"}, {"content", "Hi"}}});
    for (const json &stop : {json(), json::array({""}), json("")}) {
        SCOPED_TRACE(stop.dump());
        for (const char *forced : {"zzz", "</assistant>"}) {
            SCOPED_TRACE(forced);
            json request = {{"messages", messages},
                            {"grammar", std::string("root ::= \"") + forced + "\""},
                            {"n_predict", 20},
                            {"temperature", 0}};
            if (!stop.is_null()) {
                request["stop"] = stop;
            }
            Owned completion;
            ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            const json blocking = json::parse(completion.str());
            EXPECT_EQ(blocking.at("stop_type").get<std::string>(), "word") << blocking.dump();
            EXPECT_EQ(blocking.at("stopping_word").get<std::string>(), forced) << blocking.dump();
            EXPECT_EQ(blocking.at("generation_settings").at("stop"), json::array({"zzz", "</assistant>"}))
                << blocking.dump();
            int task = -1;
            ASSERT_EQ(jllama_completions_stream_request(model, request.dump().c_str(), &task), JLLAMA_STATUS_OK)
                << jllama_last_error();
            EXPECT_EQ(receive_to_stop(model, task).at("stopping_word").get<std::string>(), forced);
        }
    }
    // a word of the request replaces the CLI's, as the schema does, and the template's stays
    const json own = {{"messages", messages},
                      {"grammar", "root ::= \"zzz</assistant>\""},
                      {"stop", json::array({"", "qqq"})},
                      {"n_predict", 20},
                      {"temperature", 0}};
    Owned completion;
    ASSERT_EQ(jllama_completions(model, own.dump().c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json answer = json::parse(completion.str());
    EXPECT_EQ(answer.at("generation_settings").at("stop"), json::array({"qqq", "</assistant>"})) << answer.dump();
    EXPECT_EQ(answer.at("stopping_word").get<std::string>(), "</assistant>") << answer.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiFimRepo, TheInfillPromptCarriesTheExtraFilesAndTheOrderOfTheFimTokens) {
    // The infill prompt is upstream's format_prompt_infill: the extra files (input_extra) before the
    // FIM prefix, each after a separator -- the FIM_SEP token and its file name when the model has one,
    // else the "--- snippet ---" line -- and with FIM_REP a "myproject" line first; --spm-infill puts
    // the suffix before the prefix. Every byte is one token in this vocabulary, so the counts are exact.
    tiny_model_spec plain_spec;
    plain_spec.prefix = "jllama-c-api-fim-extra";
    plain_spec.fim = true;
    tiny_model_spec repo_spec = plain_spec;
    repo_spec.prefix = "jllama-c-api-fim-repo";
    repo_spec.fim_repo = true;
    const std::string plain_path = write_tiny_llama_gguf(plain_spec);
    const std::string repo_path = write_tiny_llama_gguf(repo_spec);
    ASSERT_FALSE(plain_path.empty());
    ASSERT_FALSE(repo_path.empty());
    const json base = {{"input_prefix", "abcdefghij"},
                       {"input_suffix", "klmnopqrst"},
                       {"n_predict", 1},
                       {"temperature", 0},
                       {"cache_prompt", false}};
    json with_extra = base;
    with_extra["input_extra"] = json::array({json{{"filename", "a.c"}, {"text", "xyz"}}});
    const auto infill = [](jllama_model *model, const json &request) {
        Owned response;
        EXPECT_EQ(jllama_infill(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()) : json::object();
    };
    const auto tokens = [](const json &result) { return result.at("tokens_evaluated").get<int>(); };
    const std::string prefix = std::string("\x01") + "abcdefghij";
    const std::string suffix = std::string("\x02") + "klmnopqrst";

    jllama_model *plain = load_tiny(plain_path, {"-c", "512", "-b", "64"});
    ASSERT_NE(plain, nullptr) << jllama_last_error();
    const json plain_base = infill(plain, base);
    const int n_base = tokens(plain_base);
    EXPECT_GE(n_base, 23); // ten prefix and ten suffix bytes around the three FIM tokens
    const std::string prompt = plain_base.at("prompt").get<std::string>();
    EXPECT_LT(prompt.find(prefix), prompt.find(suffix)) << "prefix first without --spm-infill";
    const json plain_extra = infill(plain, with_extra);
    // the 19 bytes of "\n\n--- snippet ---\n\n" and the file's three
    EXPECT_EQ(tokens(plain_extra), n_base + 19 + 3) << plain_extra.dump();
    EXPECT_NE(plain_extra.at("prompt").get<std::string>().find("--- snippet ---\n\nxyz"), std::string::npos)
        << plain_extra.dump();
    jllama_model_free(plain);

    jllama_model *spm = load_tiny(plain_path, {"-c", "512", "-b", "64", "--spm-infill"});
    ASSERT_NE(spm, nullptr) << jllama_last_error();
    const json spm_base = infill(spm, base);
    EXPECT_EQ(tokens(spm_base), n_base);
    const std::string spm_prompt = spm_base.at("prompt").get<std::string>();
    EXPECT_LT(spm_prompt.find(suffix), spm_prompt.find(prefix)) << "suffix first with --spm-infill";
    jllama_model_free(spm);

    jllama_model *repo = load_tiny(repo_path, {"-c", "512", "-b", "64"});
    ASSERT_NE(repo, nullptr) << jllama_last_error();
    // FIM_REP "myproject\n" and the closing FIM_SEP "filename\n", around no extra file
    const json repo_base = infill(repo, base);
    EXPECT_EQ(tokens(repo_base), n_base + 1 + 10 + 1 + 9) << repo_base.dump();
    // and the file after its own FIM_SEP "a.c\n"
    const json repo_extra = infill(repo, with_extra);
    EXPECT_EQ(tokens(repo_extra), tokens(repo_base) + 1 + 4 + 3) << repo_extra.dump();
    EXPECT_NE(repo_extra.at("prompt").get<std::string>().find(std::string("\x04") + "myproject\n\x05" + "a.c\nxyz\x05" +
                                                              "filename\n"),
              std::string::npos)
        << repo_extra.dump();
    jllama_model_free(repo);
    std::error_code ignored;
    std::filesystem::remove(plain_path, ignored);
    std::filesystem::remove(repo_path, ignored);
}

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

TEST(CApiWeightScales, QwenNextScalesItsLegacyDeltaNetLayoutToo) {
    // The legacy layout of Qwen3-Next's delta net (one ssm_in for queries, keys, values and the gate)
    // multiplied ssm_in without the ssm_in.scale the generic pass loads, until build_lora_mm applied a
    // weight's scale itself (b12622, #32669).
    tiny_model_spec spec = tiny_model_spec::qwen3next("jllama-c-api-scales-qwen3next-legacy");
    spec.qwen3next_legacy = true;
    expect_weight_scales_applied(spec);
}

TEST(CApiWeightScales, Zaya1ScalesItsExperts) {
    // llama.cpp has loaded ffn_gate_up_exps.scale since b12577, but ZAYA1's graph passed no expert
    // scale on, so a scaled ZAYA1 computed its experts unscaled. Since b12622 (#32669) build_lora_mm
    // and build_lora_mm_id apply the scale a weight has, whatever the architecture's graph passes. The
    // generator's ZAYA1 writes a scale per expert next to both expert matrices.
    expect_weight_scales_applied(tiny_model_spec::zaya1("jllama-c-api-scales-zaya1"));
}

TEST(CApiWeightScales, EveryDecoderOfTheGeneratorAppliesItsScales) {
    // Since b12622 (#32669) the scale of a weight is applied by build_lora_mm itself, so an
    // architecture no longer has to pass it on: Gemma 4 (with its KV-shared layers and the output tied
    // to the token embedding, which carries no scale -- llama.cpp refuses one there), Domyn Edge,
    // Cohere2 (sliding window, tied output) and Qwen3-VL.
    expect_weight_scales_applied(tiny_model_spec::gemma4(8, "jllama-c-api-scales-gemma4"));
    expect_weight_scales_applied(tiny_model_spec::domyn_edge(8, "jllama-c-api-scales-domyn"));
    expect_weight_scales_applied(tiny_model_spec::swa(16, "jllama-c-api-scales-cohere2"));
    expect_weight_scales_applied(tiny_model_spec::qwen3vl("jllama-c-api-scales-qwen3vl"));
}

TEST(CApiWeightScales, AScaledModelWithAnMtpBlockLoadsWithoutMtpDrafting) {
    // patches/0043. Without --spec-type draft-mtp llama.cpp skips the weights of an MTP block, and the
    // generic pass that loads the scales loads one only next to a weight it has: the block's scales
    // counted as tensors of the file nobody loaded, and the model did not load at all ("wrong number
    // of tensors; expected 62, got 57") -- a model of the 18 architectures that skip an MTP block
    // this way (hy_v3, DeepSeek, GLM, Qwen3.5, Qwen3-Next, ...) whose block carries scales, as an
    // NVFP4 or FP8 conversion of the whole checkpoint does, unless it drafted with that block. The
    // generator's HY-V3 carries a scale next to every matrix, the MTP block's projection included;
    // loaded without MTP drafting it answers like the folded model.
    expect_weight_scales_applied(tiny_model_spec::mtp("jllama-c-api-scales-hy-v3"));
}

TEST(CApiWeightScales, AnMtpBlockDraftsWithTheScaleOfItsProjection) {
    // nextn.eh_proj.scale, which the generic pass loads for an MTP block: with --spec-type draft-mtp
    // the scaled and the folded model draft the same tokens and accept the same of them -- the drafts
    // come from the MTP block alone, the answer from the model either way.
    tiny_model_spec spec = tiny_model_spec::mtp("jllama-c-api-scales-mtp-scaled");
    spec.weight_scale = 2.0f;
    const std::string scaled = write_tiny_llama_gguf(spec);
    spec.prefix = "jllama-c-api-scales-mtp-folded";
    spec.weight_scale_folded = true;
    const std::string folded = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(scaled.empty());
    ASSERT_FALSE(folded.empty());
    const auto drafting = [](const std::string &path) {
        jllama_model *model = load_tiny(path, {"--spec-type", "draft-mtp"});
        EXPECT_NE(model, nullptr) << jllama_last_error();
        if (model == nullptr) {
            return json();
        }
        const std::string answer = greedy_answer(model, 48);
        Owned metrics;
        EXPECT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
        const json m = metrics.s != nullptr ? json::parse(metrics.str()) : json::object();
        jllama_model_free(model);
        return json{{"answer", answer},
                    {"drafted", m.value("n_draft_tokens_total", -1)},
                    {"accepted", m.value("n_draft_accepted_total", -1)}};
    };
    const json with_scale = drafting(scaled);
    ASSERT_FALSE(with_scale.is_null());
    EXPECT_GT(with_scale.at("drafted").get<int>(), 0) << with_scale.dump();
    EXPECT_EQ(with_scale, drafting(folded));
    std::error_code ignored;
    std::filesystem::remove(scaled, ignored);
    std::filesystem::remove(folded, ignored);
}

namespace {

// Overwrites the 4 bytes that follow `marker` + `skip` bytes in the file at `path`.
bool overwrite_after(const std::string &path, const std::string &marker, size_t skip, int32_t value) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t at = bytes.find(marker);
    if (at == std::string::npos) {
        return false;
    }
    f.clear();
    f.seekp(static_cast<std::streamoff>(at + marker.size() + skip));
    f.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return static_cast<bool>(f);
}

// A GGUF string as the file stores it: its length as a u64, then the bytes. gguf writes a file in the
// byte order of the host that writes it (a big-endian GGUF on s390x), and so does overwrite_after.
std::string gguf_string(const std::string &s) {
    std::string out(8, '\0');
    const uint64_t n = s.size();
    std::memcpy(&out[0], &n, sizeof(n));
    return out + s;
}

} // namespace

TEST(CApiStatic, AGgufWithATypeIdOutsideItsEnumIsRefusedAndTheProcessGoesOn) {
    // A tensor type and a key/value type are read from the file as plain integers. Until b12636
    // (#32641) the reader cast them to ggml_type / gguf_type before checking the range -- a value
    // outside the enum, undefined behaviour in C++. A file that carries one is refused by name.
    const std::string good = write_tiny_llama_gguf("jllama-c-api-gguf-types-good");
    ASSERT_FALSE(good.empty());
    std::ifstream in(good, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const auto copy = [&](const char *name) {
        const std::string path = good + "." + name + ".gguf";
        std::ofstream(path, std::ios::binary) << bytes;
        return path;
    };

    // the tensor info: name, n_dims (u32 2), two u64 dims, then the type
    const std::string tensor_type = copy("tensor-type");
    ASSERT_TRUE(overwrite_after(tensor_type, gguf_string("token_embd.weight"), 4 + 2 * 8, 1000));
    // the key/value: key, then the value type
    const std::string kv_type = copy("kv-type");
    ASSERT_TRUE(overwrite_after(kv_type, gguf_string("general.architecture"), 0, 99));

    const struct {
        const std::string &path;
        const char *reason;
    } refused[] = {{tensor_type, "'token_embd.weight' has invalid ggml type 1000"},
                   {kv_type, "'general.architecture' has invalid GGUF type 99"}};
    for (const auto &r : refused) {
        LogCapture log;
        jllama_model *model = load_tiny(r.path, {});
        log.stop();
        EXPECT_EQ(model, nullptr) << r.path;
        if (model != nullptr) {
            jllama_model_free(model);
        }
        // the refusal is the type, not some other damage the overwrite did
        EXPECT_NE(log.text().find(r.reason), std::string::npos) << log.text();
    }
    jllama_model *model = load_tiny(good, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model, 4).empty());
    jllama_model_free(model);
    std::error_code ec;
    for (const std::string &path : {good, tensor_type, kv_type}) {
        std::filesystem::remove(path, ec);
    }
}

TEST(CApiControlVector, AControlVectorChangesTheAnswerOnlyOnTheLayersOfItsRange) {
    // --control-vector and --control-vector-layer-range START END (ModelParameters.setControlVector,
    // setControlVectorLayerRange): llama.cpp adds the vector's direction to the output of every layer
    // of the range. No test loaded a control vector before; setControlVectorLayerRange wrote "START,END"
    // as one argument until 2026-11-19, which failed every load (upstream reads two).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-cvec-model");
    const std::string cvec = write_tiny_control_vector_gguf();
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(cvec.empty());
    const auto answer = [&](const std::vector<std::string> &extra) {
        jllama_model *model = load_tiny(path, extra);
        EXPECT_NE(model, nullptr) << jllama_last_error();
        if (model == nullptr) {
            return std::string("<no model>");
        }
        const std::string text = greedy_answer(model, 16);
        jllama_model_free(model);
        return text;
    };
    const std::string plain = answer({});
    const std::string steered = answer({"--control-vector", cvec});
    EXPECT_NE(steered, plain) << "the control vector changed nothing";
    // scaled by 0 it adds nothing
    EXPECT_EQ(answer({"--control-vector-scaled", cvec + ":0"}), plain);
    // its range covers every layer (1 and 2) or none of them
    EXPECT_EQ(answer({"--control-vector", cvec, "--control-vector-layer-range", "1", "2"}), steered);
    EXPECT_EQ(answer({"--control-vector", cvec, "--control-vector-layer-range", "5", "9"}), plain);
    // the range is two arguments: as one ("1,2") the next argument is read as END
    EXPECT_EQ(load_tiny(path, {"--control-vector-layer-range", "1,2", "--ctx-size", "64"}), nullptr);
    // a vector llama.cpp cannot apply is refused by the load: upstream logs it and runs without it
    // (common_init_from_params returns the context before the vector, the LoRA adapters and the warmup)
    const std::string wide = write_tiny_control_vector_gguf("jllama-control-vector-wide", N_LAYER, 2 * N_EMBD);
    EXPECT_EQ(load_tiny(path, {"--control-vector", wide}), nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("values per layer and the model 32"), std::string::npos)
        << jllama_last_error();
    const std::string deep = write_tiny_control_vector_gguf("jllama-control-vector-deep", N_LAYER + 1, N_EMBD);
    EXPECT_EQ(load_tiny(path, {"--control-vector", deep}), nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("cannot be applied to this model"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(load_tiny(path, {"--control-vector", cvec + ".missing"}), nullptr);
    EXPECT_EQ(answer({"--control-vector", cvec}), steered) << "the model loads with a vector that fits";
    std::error_code ec;
    for (const std::string &p : {path, cvec, wide, deep}) {
        std::filesystem::remove(p, ec);
    }
}

TEST(CApiShard, AModelSplitByGgufSplitLoadsFromItsFirstFile) {
    // A model in the files llama-gguf-split writes: <name>-00001-of-00002.gguf carries split.count, and
    // the loader builds the names of the others from the first one's (llama_split_prefix /
    // llama_split_path; since b12540, #32557, the buffers are sized from that path instead of
    // PATH_MAX). The generator writes the same weights whole and split; the split model must answer
    // what the whole one answers -- also from a non-ASCII directory, whose bytes the names are built
    // from --, refuse to load from its second file or without it, and take its second file from
    // another directory when that path is given (the loader matches it by file name).
    const std::string whole = write_tiny_llama_gguf("jllama-c-api-gguf-split");
    ASSERT_FALSE(whole.empty());
    jllama_model *reference = load_tiny(whole, {});
    ASSERT_NE(reference, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(reference);
    jllama_model_free(reference);
    ASSERT_FALSE(expected.empty());

    const std::filesystem::path base =
        std::filesystem::temp_directory_path() / ("jllama-c-api-gguf-split-" + std::to_string(std::random_device{}()));
    const auto utf8 = [](const std::filesystem::path &p) {
        const auto u8 = p.u8string();
        return std::string(u8.begin(), u8.end());
    };
    // Fewer than 260 characters, more than 260 bytes of UTF-8: until b12540 the loader built the split
    // names in a buffer of PATH_MAX bytes, which is MAX_PATH (260) on Windows, so this directory failed
    // there while the file system took it (MAX_PATH counts UTF-16 units). Linux's PATH_MAX is 4096.
    std::string cjk_40;
    for (int i = 0; i < 40; ++i) {
        cjk_40 += "\xE6\x97\xA5"; // 日, three bytes
    }
    const std::filesystem::path deep = base / std::filesystem::u8path("d-\xC3\xA4-\xE6\x97\xA5") /
                                       std::filesystem::u8path(cjk_40) / std::filesystem::u8path(cjk_40);
    ASSERT_GT(utf8(deep / "jllama-c-api-gguf-split-00001-of-00002.gguf").size(), 260u);
    for (const std::filesystem::path &dir : {base / "plain", deep}) {
        SCOPED_TRACE(utf8(dir));
        tiny_model_spec spec;
        spec.prefix = "jllama-c-api-gguf-split";
        spec.split_dir = utf8(dir);
        const std::string first = write_tiny_llama_gguf(spec);
        ASSERT_FALSE(first.empty());
        const std::filesystem::path second = dir / "jllama-c-api-gguf-split-00002-of-00002.gguf";
        ASSERT_TRUE(std::filesystem::exists(second));

        jllama_model *split = load_tiny(first, {});
        ASSERT_NE(split, nullptr) << jllama_last_error();
        EXPECT_EQ(greedy_answer(split), expected);
        jllama_model_free(split);

        // llama.cpp's reason ("model must be loaded with the first split") reaches the log only
        EXPECT_EQ(load_tiny(utf8(second), {}), nullptr) << "a model is loaded from its first split";

        // the second file elsewhere: missing where the first file's name says, found where --shard says
        const std::filesystem::path elsewhere = base / "elsewhere";
        std::filesystem::create_directories(elsewhere);
        const std::filesystem::path moved = elsewhere / second.filename();
        std::filesystem::rename(second, moved);
        EXPECT_EQ(load_tiny(first, {}), nullptr) << "the model without its second split must not load";
        jllama_model *relocated = load_tiny(first, {"--shard", utf8(moved)});
        ASSERT_NE(relocated, nullptr) << jllama_last_error();
        EXPECT_EQ(greedy_answer(relocated), expected);
        jllama_model_free(relocated);
        std::filesystem::rename(moved, second);
    }
    // the quantizer reads every split and writes one file
    const std::string first = utf8(base / "plain" / "jllama-c-api-gguf-split-00001-of-00002.gguf");
    const std::string quantized = utf8(base / "quantized-q8_0.gguf");
    ASSERT_EQ(jllama_quantize(first.c_str(), quantized.c_str(), 7, 2, false), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_model *q8 = load_tiny(quantized, {});
    ASSERT_NE(q8, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(q8).empty());
    jllama_model_free(q8);

    std::error_code ignored;
    std::filesystem::remove(whole, ignored);
    std::filesystem::remove_all(base, ignored);
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

// HRM-Text (llama.cpp's hrm_text) expands its low and high stacks into block_count cache slots,
// layers_per_stack * h_cycles * (l_cycles + 1) of them. Until b12556 (#32601) the loader asserted that
// product -- a GGUF whose count did not match ended the process -- and computed it in 32 bits, so a
// product of block_count + 2^32 passed and the tensor loader indexed past the layers. It throws now,
// in 64 bits. The GGUF carries the hparams and a byte vocabulary, no tensors: the check stands before
// the tensors, so a count that matches fails the load at the first missing tensor instead, which tells
// the refusal apart from any other failed load.
TEST(CApiHrm, ASlotCountThatDoesNotMatchTheBlockCountFailsTheLoadAndTheProcessGoesOn) {
    const auto write = [](const std::string &prefix, uint32_t per_stack, uint32_t h_cycles, uint32_t l_cycles) {
        const std::string path = (std::filesystem::temp_directory_path() / (prefix + ".gguf")).string();
        gguf_context *g = gguf_init_empty();
        const std::string a = "hrm_text";
        gguf_set_val_str(g, "general.architecture", a.c_str());
        gguf_set_val_u32(g, (a + ".context_length").c_str(), 64);
        gguf_set_val_u32(g, (a + ".embedding_length").c_str(), 32);
        gguf_set_val_u32(g, (a + ".block_count").c_str(), 4);
        gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), 64);
        gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), 2);
        gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), 2);
        gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
        gguf_set_val_u32(g, (a + ".hrm.layers_per_stack").c_str(), per_stack);
        gguf_set_val_u32(g, (a + ".hrm.h_cycles").c_str(), h_cycles);
        gguf_set_val_u32(g, (a + ".hrm.l_cycles").c_str(), l_cycles);
        const std::vector<std::string> tokens = jllama_test::byte_tokens();
        std::vector<const char *> token_ptrs;
        for (const std::string &t : tokens) {
            token_ptrs.push_back(t.c_str());
        }
        const std::vector<int32_t> token_types(tokens.size(), 1);
        const char *merges[] = {"Ġ Ġ"};
        gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
        gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
        gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
        const bool ok = gguf_write_to_file(g, path.c_str(), false);
        gguf_free(g);
        return ok ? path : std::string();
    };
    const auto load_log = [](const std::string &path) {
        LogCapture log;
        EXPECT_EQ(load_tiny(path, {}), nullptr);
        log.stop();
        return log.text();
    };

    // 1 * 1 * (1 + 1) = 2 slots against 4 blocks: an assertion that ended the process until b12556
    const std::string mismatch = write("jllama-c-api-hrm-mismatch", 1, 1, 1);
    ASSERT_FALSE(mismatch.empty());
    EXPECT_NE(load_log(mismatch).find("hrm: cache-slot count 2 (layers_per_stack 1, h_cycles 1, l_cycles 1) does not "
                                      "match block_count 4"),
              std::string::npos);

    // 1073741825 * 2 * (1 + 1) = 2^32 + 4: in 32 bits that was 4 and passed
    const std::string wrapping = write("jllama-c-api-hrm-wrapping", 1073741825u, 2, 1);
    ASSERT_FALSE(wrapping.empty());
    EXPECT_NE(load_log(wrapping).find("hrm: cache-slot count 4294967300"), std::string::npos);

    // 1 * 2 * (1 + 1) = 4 matches: the load passes the check and fails at the tensors the GGUF does not have
    const std::string matching = write("jllama-c-api-hrm-matching", 1, 2, 1);
    ASSERT_FALSE(matching.empty());
    const std::string text = load_log(matching);
    EXPECT_EQ(text.find("hrm: cache-slot count"), std::string::npos) << text;
    EXPECT_NE(text.find("tensor 'token_embd.weight' not found"), std::string::npos) << text;

    // and the process loads and answers the next model
    const std::string tiny = write_tiny_llama_gguf("jllama-c-api-hrm-after");
    jllama_model *model = load_tiny(tiny, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model).empty());
    jllama_model_free(model);

    for (const std::string &p : {mismatch, wrapping, matching, tiny}) {
        std::remove(p.c_str());
    }
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

TEST(CApiDraft, TheDraftIsSampledOnTheBackendOrOnTheCpuWithTheSameAnswer) {
    // --spec-draft-backend-sampling / --no-spec-draft-backend-sampling
    // (ModelParameters.setSpecDraftBackendSampling): where the draft is sampled. Greedily, both
    // places pick the argmax, so the model drafts either way and answers the same text.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-draft-backend-sampling");
    ASSERT_FALSE(path.empty());
    std::vector<std::string> answers;
    for (const char *flag : {"--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling"}) {
        SCOPED_TRACE(flag);
        jllama_model *model = load_tiny(path, {"--spec-draft-model", path, "--spec-draft-ngl", "0", flag});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        answers.push_back(greedy_answer(model, 48));
        EXPECT_GT(drafted_tokens(model), 0);
        jllama_model_free(model);
    }
    EXPECT_FALSE(answers[0].empty());
    EXPECT_EQ(answers[0], answers[1]);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiDraft, APlainDraftModelGivesTheGreedyAnswerOfTheModelAlone) {
    // Speculative decoding verifies the draft with the target, so greedily the answer is the
    // target's: the model with itself as a draft-simple draft proposes tokens and answers what the
    // model alone answers. Both are fresh models, so each evaluates its prompt whole.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-draft-same");
    ASSERT_FALSE(path.empty());
    jllama_model *alone = load_tiny(path, {});
    ASSERT_NE(alone, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(alone, 32);
    EXPECT_EQ(drafted_tokens(alone), 0);
    jllama_model_free(alone);
    jllama_model *drafted =
        load_tiny(path, {"--spec-draft-model", path, "--spec-draft-ngl", "0", "--spec-type", "draft-simple"});
    ASSERT_NE(drafted, nullptr) << jllama_last_error();
    EXPECT_FALSE(expected.empty());
    EXPECT_EQ(greedy_answer(drafted, 32), expected);
    EXPECT_GT(drafted_tokens(drafted), 0);
    jllama_model_free(drafted);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiQwen35, ARejectedDraftLeavesTheDeltaNetStateAsTheModelAloneHasIt) {
    // Qwen3.5 is a model llama.cpp rolls a recurrent state back for: its MTP block drafts, and the state
    // of the gated delta net must forget the draft tokens the model rejects. Since b12676 (#31669)
    // llama.cpp keeps one state and the inputs of the last ubatch and replays the kept tokens
    // (rs_replay, which the load logs) where it kept a snapshot per draft token before. The random MTP
    // block is rejected most of the time; greedily the answer must still be the model's alone, every
    // token of it. (A plain draft model, draft-simple, takes no rollback: the server keeps context
    // checkpoints for it instead.)
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::qwen35("jllama-qwen35-mtp", 1));
    ASSERT_FALSE(path.empty());
    jllama_model *alone = load_tiny(path, {});
    ASSERT_NE(alone, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(alone, 48);
    EXPECT_EQ(drafted_tokens(alone), 0);
    jllama_model_free(alone);

    struct Sink {
        std::mutex m;
        std::string text;
    } sink;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *ud) {
                      auto *s = static_cast<Sink *>(ud);
                      std::lock_guard<std::mutex> lk(s->m);
                      s->text += text;
                  },
                  &sink),
              JLLAMA_STATUS_OK);
    jllama_model *drafted = load_tiny(path, {"--spec-type", "draft-mtp", "-lv", "4"});
    ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK); // flushes
    ASSERT_NE(drafted, nullptr) << jllama_last_error();
    EXPECT_NE(sink.text.find("rs_replay             = true"), std::string::npos)
        << "the target context does not replay its delta net";
    EXPECT_FALSE(expected.empty());
    EXPECT_EQ(greedy_answer(drafted, 48), expected);
    Owned metrics;
    ASSERT_EQ(jllama_metrics(drafted, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json m = json::parse(metrics.str());
    const int n_drafted = m.at("n_draft_tokens_total").get<int>();
    const int n_accepted = m.at("n_draft_accepted_total").get<int>();
    EXPECT_GT(n_drafted, 0);
    EXPECT_LT(n_accepted, n_drafted) << "no draft token was rejected, so nothing was rolled back";
    jllama_model_free(drafted);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiAlias, TheAliasIsTheModelOfEveryOpenAiAnswer) {
    // --alias names the model to the API (model_name, the first alias): every OpenAI answer and chunk
    // carries it, blocking and streamed, for chat and completions, as upstream's server does.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-alias");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--alias", "my-model"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("name").get<std::string>(), "my-model") << meta.str();

    const char *chat = R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":2})";
    const char *completion = R"({"prompt":"Hi","max_tokens":2})";
    Owned blocking_chat;
    ASSERT_EQ(jllama_chat_completions(model, chat, &blocking_chat.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(blocking_chat.str()).at("model").get<std::string>(), "my-model");
    Owned blocking_completion;
    ASSERT_EQ(jllama_completions_oai(model, completion, &blocking_completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(blocking_completion.str()).at("model").get<std::string>(), "my-model");
    const auto streamed_models = [model](int task) {
        std::set<std::string> models;
        bool stop = false;
        for (int i = 0; i < 20 && !stop; ++i) {
            Owned envelope;
            EXPECT_EQ(jllama_chat_completions_stream_receive(model, task, &envelope.s, &stop), JLLAMA_STATUS_OK)
                << jllama_last_error();
            if (envelope.s == nullptr) {
                break;
            }
            const json data = json::parse(envelope.str()).at("data");
            for (const json &chunk : data.is_array() ? data : json::array({data})) {
                models.insert(chunk.at("model").get<std::string>());
            }
        }
        EXPECT_TRUE(stop);
        return models;
    };
    int task = -1;
    ASSERT_EQ(jllama_chat_completions_stream_request(model, chat, &task), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(streamed_models(task), std::set<std::string>{"my-model"});
    ASSERT_EQ(jllama_completions_oai_stream_request(model, completion, &task), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(streamed_models(task), std::set<std::string>{"my-model"});
    jllama_model_free(model);

    // and the OpenAI embeddings of an encoder, unless the request names a model of its own
    const std::string encoder = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-alias");
    ASSERT_FALSE(encoder.empty());
    jllama_model *embedder = load_tiny(encoder, {"--embedding", "--alias", "my-encoder"});
    ASSERT_NE(embedder, nullptr) << jllama_last_error();
    Owned embeddings;
    ASSERT_EQ(jllama_embeddings(embedder, R"({"input":"hello"})", true, &embeddings.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(embeddings.str()).at("model").get<std::string>(), "my-encoder") << embeddings.str();
    jllama_model_free(embedder);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(encoder, ignored);
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

TEST(CApiSlotFiles, ASlotIdOutOfRangeWrapsAroundAsUpstreamsServerDoesAndANegativeOneIsRefused) {
    // llama.cpp's server takes a slot id modulo the slot count ("allow id_slot to be out of bounds
    // (wrap around)", server_context::get_slot_by_id), deliberately, for the slots route and a
    // request's id_slot alike: slot 3 of two slots is slot 1, slot 5 of one slot is slot 0. A negative
    // id is refused by the engine: upstream takes the modulo in size_t, so -1 named some slot as well
    // (slot 1 of two) and its "Invalid slot ID" branch was unreachable.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-ids");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-slot-ids-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    const json request = {
        {"prompt", "Hello there, a prompt of some length"}, {"n_predict", 2}, {"temperature", 0}, {"id_slot", 1}};
    {
        jllama_model *model = load_tiny(path, {"-c", "256", "-np", "2", "--slot-save-path", dir.string()});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned answer;
        ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &answer.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        // slot 3 is slot 1, which holds the prompt; slot 0 holds nothing
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 3, "wrapped.bin", &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(json::parse(saved.str()).at("id_slot").get<int>(), 3) << saved.str();
        EXPECT_GT(json::parse(saved.str()).at("n_saved").get<int>(), 0) << "slot 3 was not slot 1: " << saved.str();
        Owned empty;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 2, "empty.bin", &empty.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(json::parse(empty.str()).at("n_saved").get<int>(), 0) << "slot 2 was not slot 0: " << empty.str();
        Owned erased;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 3, nullptr, &erased.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model_free(model);
    }
    {
        // one slot: every id is slot 0 -- slot 5 saves it, slot 7 erases it, and the file restores
        // into slot 0, from which the next request takes its prompt
        // no host-memory prompt cache, which could hand the erased state back
        jllama_model *model = load_tiny(path, {"-c", "256", "--cache-ram", "0", "--slot-save-path", dir.string()});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        json again = request;
        again.erase("id_slot");
        Owned answer;
        ASSERT_EQ(jllama_completions(model, again.dump().c_str(), &answer.s), JLLAMA_STATUS_OK) << jllama_last_error();
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 5, "one.bin", &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_GT(json::parse(saved.str()).at("n_saved").get<int>(), 0) << "slot 5 was not slot 0: " << saved.str();
        Owned erased;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 7, nullptr, &erased.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned emptied;
        ASSERT_EQ(jllama_completions(model, again.dump().c_str(), &emptied.s), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_EQ(json::parse(emptied.str()).at("timings").at("cache_n").get<int>(), 0)
            << "slot 7 did not erase slot 0: " << emptied.str();
        Owned restored;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "one.bin", &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned kept;
        ASSERT_EQ(jllama_completions(model, again.dump().c_str(), &kept.s), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_GT(json::parse(kept.str()).at("timings").at("cache_n").get<int>(), 0) << kept.str();
        // a negative id is no slot: refused by every action, before the file is looked at
        for (const jllama_slot_op op : {JLLAMA_SLOT_SAVE, JLLAMA_SLOT_RESTORE, JLLAMA_SLOT_ERASE}) {
            Owned refused;
            EXPECT_EQ(jllama_slot_action(model, op, -1, op == JLLAMA_SLOT_ERASE ? nullptr : "one.bin", &refused.s),
                      JLLAMA_STATUS_INVALID_REQUEST)
                << op;
            EXPECT_NE(std::string(jllama_last_error()).find("Invalid slot ID -1"), std::string::npos)
                << jllama_last_error();
        }
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(CApiSlotFiles, ASlotFileOfAnotherModelIsRefusedAndTheModelGoesOn) {
    // A slot file holds the KV rows of the model that wrote it and the prompt's token ids. One of a
    // model with other rows (wider), of another vocabulary (merge tokens this one does not have) or of
    // a longer context than this one's slot is refused as an invalid slot file, the slot is left
    // empty, and the model answers the next request.
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      ("jllama-c-api-slot-foreign-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    const std::string target_path = write_tiny_llama_gguf("jllama-c-api-slot-foreign-target");
    tiny_model_spec bpe;
    bpe.prefix = "jllama-c-api-slot-foreign-bpe";
    bpe.tokenizer = tiny_model_spec::tokenizer_kind::BPE_MERGES;
    const std::string bpe_path = write_tiny_llama_gguf(bpe);
    const std::string wide_path = write_tiny_llama_gguf(tiny_model_spec::wide(64, "jllama-c-api-slot-foreign-wide"));
    // the target's weights with a training context that does not cap the slot at 128 tokens
    tiny_model_spec long_spec;
    long_spec.prefix = "jllama-c-api-slot-foreign-long";
    long_spec.n_ctx_train = 1024;
    const std::string long_path = write_tiny_llama_gguf(long_spec);
    ASSERT_FALSE(target_path.empty());
    ASSERT_FALSE(long_path.empty());
    ASSERT_FALSE(bpe_path.empty());
    ASSERT_FALSE(wide_path.empty());
    const auto save = [&dir](const std::string &model_path, const std::vector<std::string> &extra,
                             const std::string &prompt, const char *name) {
        std::vector<std::string> args = {"--slot-save-path", dir.string()};
        args.insert(args.end(), extra.begin(), extra.end());
        jllama_model *model = load_tiny(model_path, args);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned answer;
        ASSERT_EQ(jllama_completions(model, json{{"prompt", prompt}, {"n_predict", 2}}.dump().c_str(), &answer.s),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name, &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model_free(model);
    };
    save(wide_path, {}, "hello world", "wide.bin");
    save(bpe_path, {}, "hello world hello world", "bpe.bin");
    save(long_path, {"-c", "512"}, std::string(300, 'a'), "long.bin");

    struct row {
        const char *name;
        std::string model;
    };
    // the long file is restored into the same model with a context of 256 tokens (llama.cpp pads to
    // 256), shorter than the state's 300
    for (const row &r : {row{"wide.bin", target_path}, row{"bpe.bin", target_path}, row{"long.bin", long_path}}) {
        SCOPED_TRACE(r.name);
        jllama_model *target = load_tiny(r.model, {"--slot-save-path", dir.string()});
        ASSERT_NE(target, nullptr) << jllama_last_error();
        const char *name = r.name;
        Owned refused;
        EXPECT_EQ(jllama_slot_action(target, JLLAMA_SLOT_RESTORE, 0, name, &refused.s), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("Unable to restore slot"), std::string::npos)
            << jllama_last_error();
        Owned next;
        ASSERT_EQ(jllama_completions(target, R"({"prompt":"Hi","n_predict":2,"temperature":0})", &next.s),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(json::parse(next.str()).at("timings").at("cache_n").get<int>(), 0)
            << "a refused state stayed: " << next.str();
        jllama_model_free(target);
    }
    std::error_code ignored;
    std::filesystem::remove(target_path, ignored);
    std::filesystem::remove(long_path, ignored);
    std::filesystem::remove(bpe_path, ignored);
    std::filesystem::remove(wide_path, ignored);
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

TEST(CApiQwen3Audio, AClipCostsTheEmbeddingsOfItsFramesWithoutThePaddingOfItsLastChunk) {
    // Upstream #32697 (b12643): the qwen3a encoder (Qwen3-Omni, Qwen3-ASR) counts 13 embeddings for
    // every 100-frame chunk and, for a shorter last chunk, its frames halved three times (rounded up),
    // instead of padding that chunk to 100 frames and counting 13; and a clip is n / 160 frames
    // (WhisperFeatureExtractor drops the last STFT frame), at least one, where it was one more.
    // Before, 0.25 s was 13 embeddings instead of 4, and 1 s 26 instead of 13.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-qwen3a";
    spec.n_ctx_train = 1024;
    const std::string path = write_tiny_llama_gguf(spec);
    const std::string mmproj = write_tiny_qwen3a_mmproj_gguf();
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(mmproj.empty());
    jllama_model *model = load_tiny(path, {"--mmproj", mmproj, "-c", "1024", "-b", "1024", "-ub", "1024"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json m = json::parse(meta.str());
    EXPECT_TRUE(m.at("modalities").value("audio", false)) << meta.str();
    const std::string marker = m.value("media_marker", std::string());
    ASSERT_FALSE(marker.empty());
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
            ADD_FAILURE() << samples << " samples: " << jllama_last_error();
            return -1;
        }
        return json::parse(completion.str()).at("tokens_evaluated").get<int>();
    };
    // the embeddings of n frames, as upstream counts them: 13 per full chunk, ceil^3 of the rest
    const auto tokens_of_frames = [](int frames) {
        const int tail = frames % 100;
        return frames / 100 * 13 + (tail == 0 ? 0 : (((tail + 1) / 2 + 1) / 2 + 1) / 2);
    };
    // 99 frames are 13 embeddings in both counts (the old one padded them to 100), which calibrates the
    // text tokens around the clip, the audio start and end markers included
    const int base = evaluated(15999) - 13;
    // 1, 25, 100, 101, 800 (one window) and 850 frames (a full window and a tail of 50)
    for (const size_t samples :
         {size_t{100}, size_t{4000}, size_t{16000}, size_t{16160}, size_t{128000}, size_t{136000}}) {
        const int frames = std::max(1, static_cast<int>(samples / 160));
        const int expected = frames <= 800 ? tokens_of_frames(frames) : 104 + tokens_of_frames(frames - 800);
        EXPECT_EQ(evaluated(samples), base + expected) << samples << " samples, " << frames << " frames";
    }
    EXPECT_EQ(tokens_of_frames(1), 1);
    EXPECT_EQ(tokens_of_frames(25), 4);
    EXPECT_EQ(tokens_of_frames(101), 14);
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

TEST(CApiRecurrent, AStateThatOverflowedDoesNotReachTheNextSequence) {
    // llama.cpp cleared the recurrent cell of a new sequence by scaling it by 0, and an inf or NaN
    // left in it stayed NaN (upstream #31913, b12529, fills it with zeros). A Mamba whose state grows
    // with every token overflows on a long prompt; after the slot is erased, a short prompt must
    // answer what the same model answers fresh.
    tiny_model_spec spec = tiny_model_spec::recurrent("jllama-c-api-mamba-growth");
    spec.ssm_a_growth = 2.0f;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    const auto ask = [](jllama_model *model, const std::string &prompt) {
        const json request = {{"prompt", prompt}, {"n_predict", 4},        {"temperature", 0},
                              {"n_probs", 3},     {"cache_prompt", false}, {"id_slot", 0}};
        Owned response;
        EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()) : json::object();
    };
    const std::string short_prompt = "Hi";
    jllama_model *fresh = load_tiny(path, {"--cache-ram", "0", "-c", "256"});
    ASSERT_NE(fresh, nullptr) << jllama_last_error();
    const json reference = ask(fresh, short_prompt);
    jllama_model_free(fresh);
    ASSERT_FALSE(reference.at("content").get<std::string>().empty());
    const json first = reference.at("completion_probabilities").at(0).at("top_logprobs").at(0);
    ASSERT_TRUE(first.at("logprob").is_number()) << "the short prompt overflows already: " << reference.dump();

    jllama_model *model = load_tiny(path, {"--cache-ram", "0", "-c", "256"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::string long_prompt;
    for (int i = 0; i < 6; ++i) {
        long_prompt += "a longer prompt ";
    }
    const json overflowed = ask(model, long_prompt);
    const json overflowed_first = overflowed.at("completion_probabilities").at(0).at("top_logprobs").at(0);
    ASSERT_FALSE(overflowed_first.at("logprob").is_number())
        << "the long prompt did not overflow the state: " << overflowed.dump();
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json after = ask(model, short_prompt);
    EXPECT_EQ(after.at("content"), reference.at("content")) << after.dump();
    EXPECT_EQ(after.at("completion_probabilities"), reference.at("completion_probabilities")) << after.dump();
    jllama_model_free(model);
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

    // The whole response, for the timings: an answer compared with another must be computed over the
    // same cached prefix (F272), so every comparison below asserts the reuse it relies on.
    const auto answer = [model]() {
        Owned completion;
        const std::string request = R"({"prompt":"Hello there","temperature":0,"n_predict":16})";
        EXPECT_EQ(jllama_completions(model, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return completion.s != nullptr ? json::parse(completion.str()) : json::object();
    };
    const auto timing = [](const json &response, const char *key) { return response.at("timings").at(key).get<int>(); };
    const json first = answer();
    EXPECT_FALSE(first.at("content").get<std::string>().empty());
    const json second = answer();
    const json third = answer();
    EXPECT_GT(timing(second, "cache_n"), 0) << second.dump();
    EXPECT_EQ(timing(third, "cache_n"), timing(second, "cache_n")) << third.dump();
    EXPECT_EQ(timing(third, "prompt_n"), timing(second, "prompt_n")) << third.dump();
    EXPECT_EQ(second.at("content"), third.at("content")); // both over the cached prompt

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
    const json after_restore = answer();
    // the restored state is reused as the slot's own was, not evaluated again
    EXPECT_EQ(timing(after_restore, "cache_n"), timing(third, "cache_n")) << after_restore.dump();
    EXPECT_EQ(timing(after_restore, "prompt_n"), timing(third, "prompt_n")) << after_restore.dump();
    EXPECT_EQ(after_restore.at("content"), third.at("content"));

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

// Inkling 2 (b12406, #32217) through the server, on the random model of write_tiny_inkling2_gguf(): a
// hybrid memory of KV rows with their relative positions (the attention has no RoPE; a learned bias
// over the distance carries the positions) and a recurrent state per sequence (the last three inputs
// of each of its four short convolutions). A prompt longer than the sliding window of eight positions
// reaches the window's edge, the relative bias of both layer types and the convolution state.
namespace {
// The log-probabilities of the top tokens of every step of a completion.
std::vector<double> top_logprobs_of(const json &response) {
    std::vector<double> out;
    const json &steps = response.at("completion_probabilities");
    for (size_t s = 0; s < steps.size(); ++s) {
        const json &top = steps[s].at("top_logprobs");
        for (size_t t = 0; t < top.size(); ++t) {
            out.push_back(top[t].at("logprob").get<double>());
        }
    }
    return out;
}

std::string inkling2_request(const std::string &prompt, int n_predict) {
    return json{{"prompt", prompt}, {"n_predict", n_predict}, {"temperature", 0},
                {"n_probs", 3},     {"ignore_eos", true},     {"cache_prompt", false}}
        .dump();
}
} // namespace

TEST(CApiInkling2, AnInkling2ModelAnswersAndRestoresItsHybridState) {
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-inkling2-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();

    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("architecture").get<std::string>(), "inkling2") << meta.str();

    // an answer compared with another is computed over the same cached prefix (F272)
    const auto answer = [model]() {
        Owned completion;
        const std::string request = R"({"prompt":"Hello there, how are you","temperature":0,"n_predict":16})";
        EXPECT_EQ(jllama_completions(model, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return completion.s != nullptr ? json::parse(completion.str()) : json::object();
    };
    const auto timing = [](const json &response, const char *key) { return response.at("timings").at(key).get<int>(); };
    const json first = answer();
    EXPECT_FALSE(first.at("content").get<std::string>().empty());
    const json second = answer();
    const json third = answer();
    EXPECT_GT(timing(second, "cache_n"), 0) << second.dump();
    EXPECT_EQ(timing(third, "cache_n"), timing(second, "cache_n")) << third.dump();
    EXPECT_EQ(timing(third, "prompt_n"), timing(second, "prompt_n")) << third.dump();
    EXPECT_EQ(second.at("content"), third.at("content"));

    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "inkling2.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json saved_json = json::parse(saved.str());
    EXPECT_GT(saved_json.at("n_written").get<size_t>(), 0u);
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "inkling2.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json restored_json = json::parse(restored.str());
    EXPECT_EQ(restored_json.at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>()) << restored.str();
    const json after_restore = answer();
    EXPECT_EQ(timing(after_restore, "cache_n"), timing(third, "cache_n")) << after_restore.dump();
    EXPECT_EQ(timing(after_restore, "prompt_n"), timing(third, "prompt_n")) << after_restore.dump();
    EXPECT_EQ(after_restore.at("content"), third.at("content"));

    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(CApiInkling2, ThePromptInOneBatchAndOneTokenAtATimeGiveTheSameAnswer) {
    // Evaluated whole, the convolutions read the previous inputs from the batch and every query sees
    // its keys in one attention; one token at a time (-ub 1) they come from the recurrent state and
    // the KV rows the steps before left, with their relative positions. The same computation, so the
    // probabilities agree up to rounding. The values are recorded (--gtest_output=xml) so that two
    // llama.cpp versions can be compared.
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2-ubatch");
    ASSERT_FALSE(path.empty());
    std::vector<std::vector<double>> answers;
    std::vector<std::string> texts;
    for (const std::vector<std::string> extra : {std::vector<std::string>{}, std::vector<std::string>{"-ub", "1"}}) {
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned response;
        ASSERT_EQ(
            jllama_completions(model, inkling2_request("Yellow paint, how are you today", 6).c_str(), &response.s),
            JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json parsed = json::parse(response.str());
        answers.push_back(top_logprobs_of(parsed));
        texts.push_back(parsed.at("content").get<std::string>());
        jllama_model_free(model);
    }
    EXPECT_EQ(texts[0], texts[1]);
    ASSERT_EQ(answers[0].size(), 18u); // 6 steps of 3
    ASSERT_EQ(answers[1].size(), answers[0].size());
    std::string recorded;
    for (size_t i = 0; i < answers[0].size(); ++i) {
        EXPECT_NEAR(answers[0][i], answers[1][i], 1e-3) << i;
        char value[32];
        std::snprintf(value, sizeof(value), "%s%.6f", i == 0 ? "" : " ", answers[0][i]);
        recorded += value;
    }
    RecordProperty("logprobs", recorded);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiInkling2, TwoSlotsAtOnceAnswerAsEachAlone) {
    // Two requests on two slots are decoded in one batch, each with its own convolution state, KV
    // rows and relative positions. A state or a row read from the other sequence changes the
    // probabilities of an answer.
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2-slots");
    ASSERT_FALSE(path.empty());
    // the same last tokens: what tells the answers apart has to come through the attention and the state
    const std::vector<std::string> prompts = {"Hello there, how are you", "Yellow paint, how are you"};
    std::vector<std::vector<double>> alone;
    {
        jllama_model *model = load_tiny(path, {"-c", "256"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        for (const std::string &prompt : prompts) {
            Owned response;
            ASSERT_EQ(jllama_completions(model, inkling2_request(prompt, 4).c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            alone.push_back(top_logprobs_of(json::parse(response.str())));
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

    jllama_model *model = load_tiny(path, {"-c", "256", "-np", "2"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::vector<Owned> together(prompts.size());
    std::vector<jllama_status> status(prompts.size(), JLLAMA_STATUS_ERROR);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < prompts.size(); ++i) {
        threads.emplace_back([&, i]() {
            status[i] = jllama_completions(model, inkling2_request(prompts[i], 4).c_str(), &together[i].s);
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (size_t p = 0; p < prompts.size(); ++p) {
        ASSERT_EQ(status[p], JLLAMA_STATUS_OK) << p;
        const std::vector<double> got = top_logprobs_of(json::parse(together[p].str()));
        ASSERT_EQ(got.size(), alone[p].size()) << p;
        for (size_t i = 0; i < got.size(); ++i) {
            EXPECT_NEAR(got[i], alone[p][i], 1e-3) << "prompt " << p << ", value " << i;
        }
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

// The Inkling 2 DSpark draft (b12720, #32218): write_tiny_inkling2_dspark_gguf() over a target of
// write_tiny_inkling2_gguf() 160 wide. The target hands the draft the input of its second layer and its
// final norm output of every token (t_layer_inp, llama_set_embeddings_layer_inp); the draft mixes them
// into each layer's context K and V and, for the Markov head, into the state f_rnn of the last committed
// token, then proposes a block of three tokens in one pass. The target checks every draft, so with
// greedy sampling the answer is the target's alone; what the test holds is that the path runs and drafts.
namespace {
struct DsparkLog {
    std::mutex m;
    std::string text;
};

jllama_model *load_with_log(const std::string &path, const std::vector<std::string> &extra, DsparkLog &log) {
    EXPECT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *ud) {
                      auto *s = static_cast<DsparkLog *>(ud);
                      std::lock_guard<std::mutex> lk(s->m);
                      s->text += text;
                  },
                  &log),
              JLLAMA_STATUS_OK);
    jllama_model *model = load_tiny(path, extra);
    EXPECT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK); // flushes
    return model;
}

// With the draft: no --spec-type (the models handler reads draft-dflash from the draft's architecture, and
// the DFlash driver runs the DSpark head of the loaded draft), draft-dspark, draft-dflash and none.
void expect_dspark_drafts_and_answers_the_same(bool markov_context, const std::string &prefix) {
    const std::string path = write_tiny_inkling2_gguf(prefix + "-target", 8, 160);
    const std::string draft = write_tiny_inkling2_dspark_gguf(prefix, 160, 8, markov_context);
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(draft.empty());
    std::map<std::string, std::string> answers;
    std::map<std::string, int> drafted;
    for (const char *type : {"", "auto", "draft-dspark", "draft-dflash", "none"}) {
        SCOPED_TRACE(*type != 0 ? std::string("--spec-type ") + type : std::string("without the draft"));
        std::vector<std::string> extra;
        if (*type != 0) {
            extra = {"--spec-draft-model", draft, "--spec-draft-ngl", "0"};
        }
        if (*type != 0 && std::string(type) != "auto") {
            extra.insert(extra.end(), {"--spec-type", type});
        }
        DsparkLog log;
        jllama_model *model = load_with_log(path, extra, log);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        if (std::string(type) == "auto") {
            EXPECT_NE(log.text.find("auto-detected speculative type 'draft-dflash'"), std::string::npos) << log.text;
        }
        if (*type != 0 && std::string(type) != "none") {
            const std::string head = std::string("has_markov_context=") + (markov_context ? "true" : "false");
            EXPECT_NE(log.text.find(head), std::string::npos) << "the DSpark driver did not run";
        }
        // a fresh model evaluates the whole prompt: nothing is compared across a prompt-cache hit
        answers[type] = greedy_answer(model, 48);
        drafted[type] = drafted_tokens(model);
        jllama_model_free(model);
    }
    EXPECT_FALSE(answers[""].empty());
    for (const char *type : {"auto", "draft-dspark", "draft-dflash", "none"}) {
        EXPECT_EQ(answers[type], answers[""]) << type;
    }
    for (const char *type : {"auto", "draft-dspark", "draft-dflash"}) {
        EXPECT_GT(drafted[type], 0) << type << ": the DSpark draft drafted nothing";
    }
    EXPECT_EQ(drafted["none"], 0);
    EXPECT_EQ(drafted[""], 0);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(draft, ec);
}
} // namespace

TEST(CApiInkling2, ADSparkDraftWithItsMarkovContextDraftsAndAnswersWhatTheTargetAnswersAlone) {
    // The Inkling 2 head: the rnn Markov head that reads the context state f_rnn, which comes back from
    // the injection of every prompt token as h_nextn; without it for the token before a block, no draft.
    expect_dspark_drafts_and_answers_the_same(true, "jllama-c-api-inkling2-dspark");
}

TEST(CApiInkling2, ADSparkDraftWithThePlainMarkovHeadDraftsAndAnswersWhatTheTargetAnswersAlone) {
    // The plain Markov head over the same draft layers: a chain of markov_w1 rows from the argmax of
    // every block position, no state and no context.
    expect_dspark_drafts_and_answers_the_same(false, "jllama-c-api-inkling2-dspark-plain");
}

TEST(CApiInkling2, ADSparkDraftSampledAtATemperatureAnswersTheSameRequestTheSame) {
    // --spec-draft-sampling probabilistic: the graph draws every block token at the request's
    // temperature, from uniforms the driver writes into the block's state rows, and hands the target the
    // distribution of each draw for its rejection sampling. The generator of those uniforms is seeded
    // from the request's seed at the start of every request (begin()): carried over from the request
    // before, the second of two identical requests drew other uniforms, proposed other tokens and,
    // through the rejection sampling, answered differently.
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2-dspark-sampled-target", 8, 160);
    const std::string draft = write_tiny_inkling2_dspark_gguf("jllama-c-api-inkling2-dspark-sampled", 160);
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(draft.empty());
    jllama_model *model = load_tiny(path, {"--spec-draft-model", draft, "--spec-draft-ngl", "0", "--spec-type",
                                           "draft-dspark", "--spec-draft-sampling", "probabilistic"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const auto answer = [&](int seed) {
        const json request = without_prompt_cache(
            json{{"prompt", "Hello there"}, {"temperature", 0.8}, {"seed", seed}, {"n_predict", 32}});
        Owned completion;
        EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        if (completion.s == nullptr) {
            return std::string();
        }
        const json result = json::parse(completion.str());
        expect_whole_prompt_evaluated(result, 11);
        return result.at("content").get<std::string>();
    };
    const std::string first = answer(7);
    const int drafted_first = drafted_tokens(model);
    const std::string second = answer(7);
    const std::string other = answer(8);
    EXPECT_FALSE(first.empty());
    EXPECT_GT(drafted_first, 0) << "the DSpark draft drafted nothing";
    EXPECT_EQ(second, first) << "the same request with the same seed answered differently";
    EXPECT_NE(other, first) << "the seed does not decide the answer: nothing here is sampled";
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(draft, ec);
}

TEST(CApiInkling2, TwoSlotsDraftingWithDSparkAtOnceAnswerAsEachAlone) {
    // Two drafting slots put their blocks into one draft batch: the Markov head reads every block's
    // anchor and state rows as strided views of the batch, and each block its own context state.
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2-dspark-slots-target", 8, 160);
    const std::string draft = write_tiny_inkling2_dspark_gguf("jllama-c-api-inkling2-dspark-slots", 160);
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(draft.empty());
    const std::vector<std::string> prompts = {"Hello there, how are you", "Yellow paint, how are you"};
    const auto request = [](const std::string &prompt) {
        return json{{"prompt", prompt}, {"n_predict", 24}, {"temperature", 0}, {"cache_prompt", false}}.dump();
    };
    std::vector<std::string> alone;
    {
        jllama_model *model = load_tiny(path, {"-c", "256"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        for (const std::string &prompt : prompts) {
            Owned response;
            ASSERT_EQ(jllama_completions(model, request(prompt).c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            alone.push_back(json::parse(response.str()).at("content").get<std::string>());
        }
        jllama_model_free(model);
    }
    EXPECT_NE(alone[0], alone[1]);

    jllama_model *model = load_tiny(path, {"-c", "256", "-np", "2", "--spec-draft-model", draft, "--spec-draft-ngl",
                                           "0", "--spec-type", "draft-dspark"});
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
        EXPECT_EQ(json::parse(together[p].str()).at("content").get<std::string>(), alone[p]) << "prompt " << p;
    }
    EXPECT_GT(drafted_tokens(model), 0) << "the DSpark draft drafted nothing";
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(draft, ec);
}

TEST(CApiInkling2, ADSparkDraftWithAContextNormButNoMarkovContextIsRefused) {
    // markov_ctx_norm without has_markov_context true: the loader refuses the draft by name, and the
    // process loads the next model.
    const std::string path = write_tiny_inkling2_gguf("jllama-c-api-inkling2-dspark-stale-target", 8, 160);
    const std::string draft =
        write_tiny_inkling2_dspark_gguf("jllama-c-api-inkling2-dspark-stale", 160, 8, false, true);
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(draft.empty());
    DsparkLog log;
    jllama_model *model =
        load_with_log(path, {"--spec-draft-model", draft, "--spec-draft-ngl", "0", "--spec-type", "draft-dspark"}, log);
    EXPECT_NE(log.text.find("DSpark model has 'markov_ctx_norm.weight' but 'has_markov_context' is not true"),
              std::string::npos)
        << log.text;
    if (model != nullptr) {
        jllama_model_free(model);
    }
    jllama_model *next = load_tiny(path, {});
    ASSERT_NE(next, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(next).empty());
    jllama_model_free(next);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(draft, ec);
}

// Gemma 4 through the server (b12588, #32342). The server decodes the slots that are busy at the same
// time in one batch; with a batch of 8 tokens a prompt of a slot is evaluated in pieces, the piece
// before its last with no output, while the other slots generate -- sequences with other numbers of
// outputs in one batch, which the graph's pruning before the KV-shared layers takes apart slice by
// slice of the attention mask (test_gemma4.cpp holds it on llama.h). Three requests at once on three
// slots answer what each answers alone, with a unified and with a split KV cache. Red-checked on the
// pruning: pruned rows given the mask rows of the first tokens instead of their own fail it with the
// split cache; an output taken back by its place in the batch instead of in its slice does not reach
// the server's batches (it reaches test_gemma4.cpp's, which put the outputs where they want).
TEST(CApiGemma4, ThreeSlotsAtOnceAnswerAsEachAlone) {
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::gemma4(8, "jllama-c-api-gemma4-slots"));
    ASSERT_FALSE(path.empty());
    // longer than the window of 8, and of other lengths, so the slots are in other phases at a time
    const std::vector<std::string> prompts = {"Hello there, how are you", "Yellow paint dries in the sun today",
                                              "The quick brown fox jumps over the lazy dog at noon"};
    const auto request = [](const std::string &prompt) {
        return json{{"prompt", prompt}, {"n_predict", 12},    {"temperature", 0},
                    {"n_probs", 3},     {"ignore_eos", true}, {"cache_prompt", false}}
            .dump();
    };
    // the log-probabilities of the top tokens of every step, and the number of prompt tokens evaluated
    const auto read = [](const Owned &response) {
        std::vector<double> out;
        const json j = json::parse(response.str());
        for (const json &step : j.at("completion_probabilities")) {
            for (const json &top : step.at("top_logprobs")) {
                out.push_back(top.at("logprob").get<double>());
            }
        }
        out.push_back(j.at("timings").at("prompt_n").get<double>());
        return out;
    };
    std::vector<std::vector<double>> alone;
    {
        jllama_model *model = load_tiny(path, {"-c", "128", "-b", "8", "-ub", "8"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned meta;
        ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_EQ(json::parse(meta.str()).at("architecture").get<std::string>(), "gemma4");
        for (const std::string &prompt : prompts) {
            Owned response;
            ASSERT_EQ(jllama_completions(model, request(prompt).c_str(), &response.s), JLLAMA_STATUS_OK)
                << jllama_last_error();
            alone.push_back(read(response));
        }
        jllama_model_free(model);
    }
    for (size_t p = 0; p < prompts.size(); ++p) {
        ASSERT_EQ(alone[p].size(), 12u * 3u + 1u) << p;
        EXPECT_EQ(alone[p].back(), static_cast<double>(prompts[p].size())) << "the whole prompt, no cache";
    }
    double apart = 0.0;
    for (size_t i = 0; i + 1 < alone[0].size(); ++i) {
        apart = std::max(apart, std::fabs(alone[0][i] - alone[1][i]));
    }
    EXPECT_GT(apart, 1e-3) << "the prompts are answered with other probabilities";

    for (const char *kv : {"--kv-unified", "--no-kv-unified"}) {
        SCOPED_TRACE(kv);
        jllama_model *model = load_tiny(path, {"-c", "384", "-np", "3", "-b", "8", "-ub", "8", kv});
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
            const std::vector<double> got = read(together[p]);
            ASSERT_EQ(got.size(), alone[p].size()) << p;
            for (size_t i = 0; i < got.size(); ++i) {
                EXPECT_NEAR(got[i], alone[p][i], 1e-3) << "prompt " << p << ", value " << i;
            }
        }
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

// Domyn Edge (b12568, #32559): sliding-window layers with RoPE and heads half as wide, full-attention
// layers without positions, sandwich norms (tiny_model_spec::domyn_edge).
namespace {

// The first token's top log-probabilities of a greedy answer to `prompt`, the whole prompt evaluated.
std::vector<double> domyn_first_logprobs(jllama_model *model, const std::string &prompt) {
    const json request = {{"prompt", prompt}, {"n_predict", 1},     {"temperature", 0},
                          {"n_probs", 3},     {"ignore_eos", true}, {"cache_prompt", false}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    std::vector<double> out;
    if (response.s == nullptr) {
        return out;
    }
    const json parsed = json::parse(response.str()); // a range-for over a member of a temporary dangles
    for (const json &top : parsed.at("completion_probabilities").at(0).at("top_logprobs")) {
        out.push_back(top.at("logprob").get<double>());
    }
    EXPECT_EQ(out.size(), 3u) << response.str();
    return out;
}

double max_difference(const std::vector<double> &a, const std::vector<double> &b) {
    double d = a.size() == b.size() ? 0.0 : 1e9;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

} // namespace

TEST(CApiDomynEdge, ADomynEdgeModelAnswersAndRestoresItsSlot) {
    // A model of three layers with two head widths: the KV cache keeps rows of 4 per head for the
    // sliding-window layers and of 8 for the full-attention one, and the server keeps context
    // checkpoints for it (a sliding window cannot be rolled back). It has to load, answer the same over
    // the cached prompt, and from a slot saved, erased and restored, whose sliding-window cache holds
    // only the window: the slot file carries the checkpoints, and the request after the restore
    // continues from one of them, evaluating at most the last two prompt tokens.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::domyn_edge(8, "jllama-c-api-domyn"));
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-slots-domyn-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *model = load_tiny(path, {"--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(meta.str()).at("architecture").get<std::string>(), "domynedge") << meta.str();

    const auto answer = [model]() {
        Owned completion;
        const std::string request =
            R"({"prompt":"Hello there, a prompt longer than the window","temperature":0,"n_predict":8})";
        EXPECT_EQ(jllama_completions(model, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return completion.s != nullptr ? json::parse(completion.str()) : json::object();
    };
    const auto timing = [](const json &response, const char *key) { return response.at("timings").at(key).get<int>(); };
    const json first = answer();
    EXPECT_FALSE(first.at("content").get<std::string>().empty());
    const json second = answer();
    const json third = answer();
    EXPECT_GT(timing(second, "cache_n"), 0) << second.dump();
    EXPECT_EQ(timing(third, "cache_n"), timing(second, "cache_n")) << third.dump();
    EXPECT_EQ(second.at("content"), third.at("content")); // both over the cached prompt (F272)

    LogCapture lines;
    Owned saved;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "domyn.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json saved_json = json::parse(saved.str());
    EXPECT_GT(saved_json.at("n_written").get<size_t>(), 0u);
    Owned erased;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "domyn.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(restored.str()).at("n_read").get<size_t>(), saved_json.at("n_written").get<size_t>())
        << restored.str();
    const json after_restore = answer();
    lines.stop();
    EXPECT_EQ(lines.count("context checkpoint(s) ("), 1) << "the save appended no checkpoints: " << lines.text();
    EXPECT_EQ(lines.count("context checkpoint(s) from '"), 1) << "the restore read none: " << lines.text();
    const int prompt_tokens = after_restore.at("tokens_evaluated").get<int>();
    EXPECT_GE(timing(after_restore, "prompt_n"), 1) << after_restore.dump();
    EXPECT_LE(timing(after_restore, "prompt_n"), 2) << after_restore.dump();
    EXPECT_EQ(timing(after_restore, "cache_n") + timing(after_restore, "prompt_n"), prompt_tokens)
        << after_restore.dump();
    EXPECT_EQ(after_restore.at("content"), third.at("content"));
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(CApiDomynEdge, ThePromptInOneBatchAndOneTokenAtATimeGiveTheSameAnswer) {
    // Evaluated whole, the sliding-window layers mask the batch's own positions; one token at a time
    // (-ub 1) they read the window from the cache, which by then has dropped what lies outside it. A
    // prompt longer than the window of 8 takes both, and the answer and its first token's
    // probabilities agree up to rounding.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::domyn_edge(8, "jllama-c-api-domyn-ub"));
    ASSERT_FALSE(path.empty());
    const json request = {{"prompt", "Yellow paint on a long wall, how are you"},
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

TEST(CApiDomynEdge, TheSlidingWindowLimitsWhatItsLayersSee) {
    // The same weights with a window of 4 and of 512 (only the metadata differs): a prompt shorter than
    // 4 tokens fits both windows and gets the same probabilities, a longer one does not -- the
    // sliding-window layers of the narrow model no longer see its beginning. Red with the window left
    // unbounded at load (the second half), and a window that reached into a prompt shorter than it
    // fails the first.
    const std::string narrow = write_tiny_llama_gguf(tiny_model_spec::domyn_edge(4, "jllama-c-api-domyn-w4"));
    const std::string wide = write_tiny_llama_gguf(tiny_model_spec::domyn_edge(512, "jllama-c-api-domyn-w512"));
    ASSERT_FALSE(narrow.empty());
    ASSERT_FALSE(wide.empty());
    const std::string short_prompt = "Hi";
    const std::string long_prompt = "Yellow paint on a long wall, how are you";
    std::vector<std::vector<double>> got[2];
    for (int w = 0; w < 2; ++w) {
        jllama_model *model = load_tiny(w == 0 ? narrow : wide, {});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        got[w].push_back(domyn_first_logprobs(model, short_prompt));
        got[w].push_back(domyn_first_logprobs(model, long_prompt));
        jllama_model_free(model);
    }
    ASSERT_EQ(got[0][0].size(), 3u);
    EXPECT_LT(max_difference(got[0][0], got[1][0]), 1e-5) << "a prompt inside both windows";
    EXPECT_GT(max_difference(got[0][1], got[1][1]), 1e-3) << "a prompt longer than the narrow window";
    std::error_code ignored;
    std::filesystem::remove(narrow, ignored);
    std::filesystem::remove(wide, ignored);
}

TEST(CApiDomynEdge, ARotationWiderThanTheSlidingWindowHeadIsRefusedAndTheProcessGoesOn) {
    // RoPE cannot rotate more dimensions than a sliding-window head has (the CPU would abort, a GPU read
    // past the row): llama.cpp refuses such a GGUF at load, naming both numbers, and the next model
    // loads in the same process.
    tiny_model_spec spec = tiny_model_spec::domyn_edge(8, "jllama-c-api-domyn-rot");
    spec.n_rot_swa = spec.head_dim_swa + 2;
    const std::string bad = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(bad.empty());
    std::string text;
    {
        LogCapture log; // llama.cpp names the reason in its load log, the C API the failed load
        EXPECT_EQ(load_tiny(bad, {}), nullptr);
        log.stop();
        text = log.text();
    }
    EXPECT_NE(text.find("invalid n_rot_swa: 6, max 4"), std::string::npos) << text;
    const std::string good = write_tiny_llama_gguf(tiny_model_spec::domyn_edge(8, "jllama-c-api-domyn-rot-ok"));
    jllama_model *model = load_tiny(good, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model, 4).empty());
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(bad, ignored);
    std::filesystem::remove(good, ignored);
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

namespace {

// What a model loaded with extra answers greedily to one prompt, evaluated whole, with the prompt
// tokens it evaluated -- or why it did not load or answer.
struct BatchAnswer {
    std::string content;
    int prompt_n = -1;
    int drafted = -1;
    std::string error;
};

BatchAnswer batch_answer(jllama_model *model) {
    BatchAnswer a;
    Owned completion;
    const char *request = R"({"prompt":"Hello there","temperature":0,"n_predict":16,"cache_prompt":false})";
    if (jllama_completions(model, request, &completion.s) == JLLAMA_STATUS_OK) {
        const json j = json::parse(completion.str());
        a.content = j.at("content").get<std::string>();
        a.prompt_n = j.at("timings").at("prompt_n").get<int>();
    } else {
        a.error = jllama_last_error();
    }
    return a;
}

BatchAnswer batch_answer(const std::string &path, const std::vector<std::string> &extra) {
    jllama_model *model = load_tiny(path, extra);
    if (model == nullptr) {
        BatchAnswer refused;
        refused.error = jllama_last_error();
        return refused;
    }
    BatchAnswer a = batch_answer(model);
    a.drafted = drafted_tokens(model);
    jllama_model_free(model);
    return a;
}

std::string joined(const std::vector<std::string> &args) {
    std::string text;
    for (const auto &a : args) {
        text += (text.empty() ? "" : " ") + a;
    }
    return text;
}

} // namespace

TEST(CApiBatchSize, ABatchOfOneTokenLoadsAndAnswersAsUbatchesOfOneTokenDo) {
    // The server's load evaluates two tokens in one batch to learn whether the memory can drop part
    // of a sequence (common_context_can_seq_rm), and the warmup up to --batch-size tokens. A context
    // whose batch holds fewer -- --batch-size 1, or -c 1, since a causal context's batch is at most
    // its size -- aborted the process in common_batch::get_sub_batch: a JVM, for setBatchSize(1).
    // Since patches/0032 both evaluate the context's batch at a time, and such a model answers what a
    // model with a larger batch answers in ubatches of one token: the same evaluation, token by token.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-batch-size");
    ASSERT_FALSE(path.empty());
    const BatchAnswer reference = batch_answer(path, {"-b", "64", "-ub", "1"});
    ASSERT_TRUE(reference.error.empty()) << reference.error;
    ASSERT_FALSE(reference.content.empty());
    ASSERT_GT(reference.prompt_n, 1);
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"-b", "1"}, std::vector<std::string>{"-b", "1", "--warmup"},
          std::vector<std::string>{"-c", "1"}, std::vector<std::string>{"-c", "1", "--warmup"},
          std::vector<std::string>{"-b", "2", "-ub", "1", "--warmup"}}) {
        SCOPED_TRACE(joined(extra));
        const BatchAnswer a = batch_answer(path, extra);
        EXPECT_TRUE(a.error.empty()) << a.error;
        EXPECT_EQ(a.content, reference.content);
        EXPECT_EQ(a.prompt_n, reference.prompt_n);
    }
    // A draft model's proposals go into the target's batch next to the sampled token, and a slot
    // drafts at most its share of that batch (patches/0033; before it, a batch of three next to the
    // default draft of three tokens ended the process). A batch of one leaves no room and the model
    // generates without drafting, a batch of three drafts up to two tokens; checked in ubatches of
    // one token, the answer stays the same.
    for (const char *batch : {"1", "3"}) {
        const std::vector<std::string> extra = {
            "-b", batch, "-ub", "1", "--warmup", "--spec-draft-model", path, "--spec-draft-ngl", "0"};
        SCOPED_TRACE(joined(extra));
        const BatchAnswer a = batch_answer(path, extra);
        EXPECT_TRUE(a.error.empty()) << a.error;
        EXPECT_EQ(a.content, reference.content);
        if (std::string(batch) == "1") {
            EXPECT_EQ(a.drafted, 0);
        } else {
            EXPECT_GT(a.drafted, 0);
        }
    }
    // a batch of no tokens cannot evaluate anything: the load is refused by name, the process goes on
    for (const char *flag : {"-b", "--batch_size"}) {
        EXPECT_EQ(load_tiny(path, {flag, "0"}), nullptr) << flag;
        EXPECT_NE(std::string(jllama_last_error()).find("--batch-size must be at least 1"), std::string::npos)
            << jllama_last_error();
    }
    // and a negative ubatch, which an embedding context would take for its batch
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"-ub", "-1"}, std::vector<std::string>{"--embedding", "-ub", "-1"}}) {
        SCOPED_TRACE(joined(extra));
        EXPECT_EQ(load_tiny(path, extra), nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find("--ubatch-size must not be negative"), std::string::npos)
            << jllama_last_error();
    }
    // a ubatch of 0 is upstream's "the batch size"
    const BatchAnswer whole = batch_answer(path, {"-b", "8", "-ub", "0"});
    EXPECT_TRUE(whole.error.empty()) << whole.error;
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiBatchSize, SlotsShareABatchSmallerThanTheirTokens) {
    // Requests at once put a token of every slot into each generation step, more tokens than a small
    // batch holds: the server decodes them a batch at a time, and each request answers what it
    // answers alone. A context of more sequences than batch tokens ended the process while it was
    // being created (patches/0034); two slots that draft get half of a batch of four each
    // (patches/0033; no guard of it: the abort needs both slots to draft long at the same step, which
    // the random model does not do on demand).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-batch-size-slots");
    ASSERT_FALSE(path.empty());
    struct row {
        int slots;
        std::vector<std::string> extra;
    };
    for (const row &r :
         {row{2, {"-b", "1", "--warmup"}}, row{4, {"-b", "2", "-ub", "1", "--warmup"}},
          row{2, {"-b", "4", "-ub", "1", "--warmup", "--spec-draft-model", path, "--spec-draft-ngl", "0"}}}) {
        const std::string slots = std::to_string(r.slots);
        std::vector<std::string> extra = r.extra;
        extra.insert(extra.end(), {"-np", slots});
        SCOPED_TRACE(joined(extra));
        const BatchAnswer reference = batch_answer(path, {"-b", "64", "-ub", "1", "-np", slots});
        ASSERT_TRUE(reference.error.empty()) << reference.error;
        jllama_model *model = load_tiny(path, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        const bool drafts = std::find(extra.begin(), extra.end(), "--spec-draft-model") != extra.end();
        for (int round = 0; round < 3; ++round) {
            std::vector<BatchAnswer> answers(static_cast<size_t>(r.slots));
            std::vector<std::thread> callers;
            for (size_t i = 0; i < answers.size(); ++i) {
                callers.emplace_back([&answers, i, model]() { answers[i] = batch_answer(model); });
            }
            for (auto &caller : callers) {
                caller.join();
            }
            for (const BatchAnswer &a : answers) {
                EXPECT_TRUE(a.error.empty()) << a.error;
                EXPECT_EQ(a.content, reference.content);
            }
        }
        if (drafts) {
            EXPECT_GT(drafted_tokens(model), 0) << "the slots did not draft at all";
        }
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiBatchSize, AnEmbeddingContextWithAUbatchOfOneTokenLoads) {
    // An embedding context's batch is its ubatch (common_init_result clamps n_batch to it), so
    // --embedding -ub 1 reached the same probe and aborted the process; a non-causal encoder's warmup
    // encoded two tokens in a ubatch of one. Both load now, and an input longer than the ubatch is
    // refused by the server, which an encoder reads whole.
    const std::string llama = write_tiny_llama_gguf("jllama-c-api-batch-size-embedding");
    ASSERT_FALSE(llama.empty());
    for (const std::vector<std::string> &extra : {std::vector<std::string>{"--embedding", "-ub", "1"},
                                                  std::vector<std::string>{"--embedding", "-ub", "1", "--warmup"}}) {
        SCOPED_TRACE(joined(extra));
        jllama_model *model = load_tiny(llama, extra);
        ASSERT_NE(model, nullptr) << jllama_last_error();
        jllama_model_free(model);
    }
    const std::string bert = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-batch-size");
    ASSERT_FALSE(bert.empty());
    jllama_model *encoder = load_tiny(bert, {"--embedding", "-b", "1", "-ub", "1", "--warmup"});
    ASSERT_NE(encoder, nullptr) << jllama_last_error();
    Owned refused;
    EXPECT_NE(jllama_embeddings(encoder, R"({"content":"hello there"})", false, &refused.s), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("too large to process"), std::string::npos) << jllama_last_error();
    jllama_model_free(encoder);
    std::error_code ignored;
    std::filesystem::remove(llama, ignored);
    std::filesystem::remove(bert, ignored);
}

TEST(CApiBatchSize, ADecoderWithoutCausalAttentionEvaluatesItsInputAUbatchAtATime) {
    // Without causal attention a call must fit one ubatch (decode() asserts it), while the context's
    // batch was --batch-size: the load's memory probe and warmup with -ub 1, and with -b 64 -ub 2 the
    // first prompt longer than two tokens, ended the process (patches/0036). The batch of such a
    // context is its ubatch now, and every caller that splits by llama_n_batch() takes it.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-non-causal";
    spec.causal = false;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    const BatchAnswer reference = batch_answer(path, {"-b", "2", "-ub", "2"});
    ASSERT_TRUE(reference.error.empty()) << reference.error;
    ASSERT_FALSE(reference.content.empty());
    ASSERT_GT(reference.prompt_n, 2) << "the prompt must not fit one ubatch";
    for (const std::vector<std::string> &extra : {std::vector<std::string>{"-b", "64", "-ub", "2"},
                                                  std::vector<std::string>{"-b", "64", "-ub", "2", "--warmup"}}) {
        SCOPED_TRACE(joined(extra));
        const BatchAnswer a = batch_answer(path, extra);
        EXPECT_TRUE(a.error.empty()) << a.error;
        EXPECT_EQ(a.content, reference.content);
        EXPECT_EQ(a.prompt_n, reference.prompt_n);
    }
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"-ub", "1"}, std::vector<std::string>{"-ub", "1", "--warmup"}}) {
        SCOPED_TRACE(joined(extra));
        const BatchAnswer a = batch_answer(path, extra);
        EXPECT_TRUE(a.error.empty()) << a.error;
        EXPECT_FALSE(a.content.empty());
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiBatchSize, AServerWithABatchOfOneTokenStartsAndAnswers) {
    // The embedded server loads its model inside llama_server(), past the engine's load: with -b 1
    // the same probe aborted the process there (patches/0032), and so did its four default slots on a
    // batch of one token (patches/0034). It starts and answers now, with the batch from the argv or
    // from the environment; a batch of no tokens is refused before the worker starts.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-server-batch-size");
    ASSERT_FALSE(path.empty());
    const auto start = [&](const std::vector<std::string> &batch, jllama_server **server, int *port) {
        httplib::Server probe;
        *port = probe.bind_to_any_port("127.0.0.1");
        std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
        probe.wait_until_ready();
        probe.stop();
        probe_listener.join();
        const std::string port_text = std::to_string(*port);
        std::vector<std::string> args = {"llama-server", "-m",        path,     "-c",     "64", "-ngl", "0",
                                         "--host",       "127.0.0.1", "--port", port_text};
        args.insert(args.end(), batch.begin(), batch.end());
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        return jllama_server_start(static_cast<int>(argv.size()), argv.data(), server);
    };
    const auto answers = [](int port) {
        httplib::Client client("127.0.0.1", port);
        httplib::Result health;
        for (int i = 0; i < 200 && !(health && health->status == 200); i++) {
            health = client.Get("/health");
            if (!(health && health->status == 200)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        EXPECT_TRUE(health && health->status == 200);
        const auto completion =
            client.Post("/completion", R"({"prompt":"Hello there","n_predict":4,"temperature":0})", "application/json");
        EXPECT_TRUE(completion) << httplib::to_string(completion.error());
        if (completion) {
            EXPECT_EQ(completion->status, 200) << completion->body;
            EXPECT_EQ(json::parse(completion->body).at("tokens_predicted").get<int>(), 4) << completion->body;
        }
    };
    // upstream reads the number with std::stoi, so "1x" is 1
    for (const std::vector<std::string> &batch :
         {std::vector<std::string>{"-b", "1"}, std::vector<std::string>{"-b", "1x"},
          std::vector<std::string>{"-b", "4", "--batch-size", "1"}}) {
        SCOPED_TRACE(joined(batch));
        jllama_server *server = nullptr;
        int port = 0;
        ASSERT_EQ(start(batch, &server, &port), JLLAMA_STATUS_OK) << jllama_last_error();
        answers(port);
        EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    }
    {
        // from the environment, which the parse applies before the argv
        SCOPED_TRACE("LLAMA_ARG_BATCH");
#ifdef _WIN32
        _putenv_s("LLAMA_ARG_BATCH", "1 ");
#else
        setenv("LLAMA_ARG_BATCH", "1 ", 1);
#endif
        jllama_server *server = nullptr;
        int port = 0;
        const jllama_status status = start({}, &server, &port);
        const std::string error = jllama_last_error();
#ifdef _WIN32
        _putenv_s("LLAMA_ARG_BATCH", "");
#else
        unsetenv("LLAMA_ARG_BATCH");
#endif
        ASSERT_EQ(status, JLLAMA_STATUS_OK) << error;
        answers(port);
        EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    }
    for (const std::vector<std::string> &batch :
         {std::vector<std::string>{"-b", "0"}, std::vector<std::string>{"--batch_size", "0"},
          std::vector<std::string>{"-b", "4", "--batch-size", "0"}}) {
        SCOPED_TRACE(joined(batch));
        jllama_server *server = nullptr;
        int port = 0;
        EXPECT_EQ(start(batch, &server, &port), JLLAMA_STATUS_ERROR);
        EXPECT_EQ(server, nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find("--batch-size must be at least 1"), std::string::npos)
            << jllama_last_error();
    }
    {
        // an argv the parse refuses fails the start itself, not the worker later
        SCOPED_TRACE("refused argv");
        const char *argv[] = {"llama-server", "--no-such-option", "--host", "127.0.0.1", "--port", "0"};
        jllama_server *server = nullptr;
        EXPECT_EQ(jllama_server_start(6, argv, &server), JLLAMA_STATUS_ERROR);
        EXPECT_EQ(server, nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find("Failed to parse server parameters"), std::string::npos)
            << jllama_last_error();
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
#endif

TEST(CApiInterrupt, InterruptEndsACallInFlightAndTheModelIsStillFreed) {
    // jllama_model_interrupt is jllama_model_free's first half on its own: a host that waits for the
    // calls in flight before it frees (llama-ffm's lifecycle lock) interrupts first, or the wait lasts
    // as long as the generation. With --context-shift and no token limit a completion never ends by
    // itself, so here only the interrupt can end it.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-interrupt");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--context-shift"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::atomic<int> status{-1};
    std::thread caller([model, &status]() {
        Owned completion;
        status.store(jllama_completions(
            model, R"({"prompt":"Hello there","n_predict":-1,"ignore_eos":true,"temperature":0})", &completion.s));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(status.load(), -1) << "the endless completion ended by itself";
    const auto interrupted = std::chrono::steady_clock::now();
    jllama_model_interrupt(model);
    caller.join();
    EXPECT_LT(std::chrono::steady_clock::now() - interrupted, std::chrono::seconds(10));
    EXPECT_EQ(status.load(), JLLAMA_STATUS_INTERRUPTED);
    // a call that does not wait for the inference loop still answers
    Owned meta;
    EXPECT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_model_interrupt(model); // idempotent
    jllama_model_free(model);
    jllama_model_interrupt(nullptr);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiInterrupt, AnInterruptEndsAStreamInFlightAndRefusesANewOne) {
    // The streamed half of the interrupt: a receive that waits for the next chunk of an endless stream
    // returns JLLAMA_STATUS_INTERRUPTED (it answered "Task not found" once the reader was cleared, a
    // status that reads as the caller's mistake), and a stream asked for afterwards is refused with
    // the same status instead of a task nobody will answer.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-interrupt-stream");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--context-shift"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const char *endless = R"({"prompt":"Hello there","n_predict":-1,"ignore_eos":true,"temperature":0,"stream":true})";
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, endless, &task), JLLAMA_STATUS_OK) << jllama_last_error();
    std::atomic<int> status{-1};
    std::atomic<int> chunks{0};
    std::thread receiver([model, task, &status, &chunks]() {
        for (;;) {
            Owned chunk;
            bool stop = false;
            const jllama_status s = jllama_completions_stream_receive(model, task, &chunk.s, &stop);
            if (s != JLLAMA_STATUS_OK || stop) {
                status.store(stop && s == JLLAMA_STATUS_OK ? -2 : s);
                return;
            }
            chunks.fetch_add(1);
        }
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (chunks.load() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_GE(chunks.load(), 3) << "the stream produced nothing";
    EXPECT_EQ(status.load(), -1) << "the endless stream ended by itself";
    jllama_model_interrupt(model);
    receiver.join();
    EXPECT_EQ(status.load(), JLLAMA_STATUS_INTERRUPTED) << jllama_last_error();

    int later = -1;
    EXPECT_EQ(jllama_completions_stream_request(model, endless, &later), JLLAMA_STATUS_INTERRUPTED);
    Owned blocking;
    EXPECT_EQ(jllama_completions(model, endless, &blocking.s), JLLAMA_STATUS_INTERRUPTED);
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

namespace {
// The process's LC_NUMERIC set to a locale with a decimal comma while it lives, as a JVM started in
// such an environment leaves it; ok() is false when the machine has none.
class DecimalCommaLocale {
  public:
    DecimalCommaLocale() {
        const char *current = std::setlocale(LC_NUMERIC, nullptr);
        previous_ = current != nullptr ? current : "C";
        for (const char *name : {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "de-DE", "German_Germany.1252"}) {
            if (std::setlocale(LC_NUMERIC, name) != nullptr) {
                if (std::string(std::localeconv()->decimal_point) == ",") {
                    ok_ = true;
                    return;
                }
            }
        }
        std::setlocale(LC_NUMERIC, previous_.c_str());
    }
    ~DecimalCommaLocale() { std::setlocale(LC_NUMERIC, previous_.c_str()); }
    DecimalCommaLocale(const DecimalCommaLocale &) = delete;
    DecimalCommaLocale &operator=(const DecimalCommaLocale &) = delete;
    bool ok() const { return ok_; }

  private:
    std::string previous_;
    bool ok_ = false;
};
} // namespace

TEST(CApiLocale, ANumberOfTheArgvIsReadWithADecimalPointUnderADecimalCommaLocale) {
    // llama.cpp's option handlers read numbers with std::stof, which follows LC_NUMERIC, and a JVM
    // leaves the locale of its environment in effect: under de_DE "--temp 0.25" was read as 0 and
    // "--repeat-penalty 0.9" refused. The engine parses under the "C" numeric locale of the calling
    // thread (and leaves the process's as it found it).
    const DecimalCommaLocale comma;
    if (!comma.ok()) {
        if (std::getenv("JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE") != nullptr) {
            FAIL() << "JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE is set, and no locale with a decimal comma is installed";
        }
        GTEST_SKIP() << "no locale with a decimal comma is installed (locale-gen de_DE.UTF-8)";
    }
    const std::string path = write_tiny_llama_gguf("jllama-c-api-locale");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--temp", "0.25", "--top-p", "0.5", "--repeat-penalty", "0.9"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_EQ(std::string(std::localeconv()->decimal_point), ",") << "the load changed the process's locale";
    Owned completion;
    ASSERT_EQ(jllama_completions(model, R"({"prompt":"Hello there","n_predict":2})", &completion.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json settings = json::parse(completion.str()).at("generation_settings");
    EXPECT_FLOAT_EQ(settings.at("temperature").get<float>(), 0.25f) << settings.dump();
    EXPECT_FLOAT_EQ(settings.at("top_p").get<float>(), 0.5f) << settings.dump();
    EXPECT_FLOAT_EQ(settings.at("repeat_penalty").get<float>(), 0.9f) << settings.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiLocale, AChatTemplateReadsAndWritesItsNumbersWithADecimalPointUnderADecimalCommaLocale) {
    // llama.cpp's Jinja reads a float literal and the float filter with std::stod and writes a float
    // through the C library, both of which follow LC_NUMERIC: under de_DE a template read 0.5 as 0. The
    // template is parsed at the load and rendered by every request, on the caller's thread and the
    // server's.
    const DecimalCommaLocale comma;
    if (!comma.ok()) {
        if (std::getenv("JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE") != nullptr) {
            FAIL() << "JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE is set, and no locale with a decimal comma is installed";
        }
        GTEST_SKIP() << "no locale with a decimal comma is installed (locale-gen de_DE.UTF-8)";
    }
    const std::string path = write_tiny_llama_gguf("jllama-c-api-locale-template");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--jinja", "--chat-template",
                                           "[{{ 0.5 }}|{{ '0.25' | float }}|{{ 1.5 * 2 }}]{{ messages[0].content }}"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned prompt;
    ASSERT_EQ(jllama_apply_template(model, R"({"messages":[{"role":"user","content":"Hi"}]})", &prompt.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(prompt.str().find("[0.5|0.25|3.0]Hi"), std::string::npos) << prompt.str();
    EXPECT_EQ(std::string(std::localeconv()->decimal_point), ",") << "a call changed the process's locale";
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

TEST(CApiSystemOne, SpokenInstructionsAreRefusedByEveryModelButLfm3D1OmniWithAudioInput) {
    // b12592 (#32578): a question whose instructions are an input_audio part is answered by an
    // lfm3-d1-omni model with audio input only. Every other decision model refuses it with that
    // reason, as the caller's fault; until then only lfm3-d1 did, and on the generator's openjev
    // model, which has no projector, the request failed as a server error ("Failed to load image or
    // audio file", JLLAMA_STATUS_ERROR -- red-checked against the old condition).
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::decision("jllama-c-api-decision-spoken"));
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"-c", "1024"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const char *const spoken = R"({"state": "a call",
      "questions": {"angry": {"type": "noul",
        "instructions": {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}}}})";
    Owned refused;
    EXPECT_EQ(jllama_systemone(model, spoken, &refused.s), JLLAMA_STATUS_INVALID_REQUEST);
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find("questions.angry: spoken instructions are only supported by lfm3-d1-omni with audio input"),
              std::string::npos)
        << error;
    // a written question still answers
    Owned answered;
    EXPECT_EQ(jllama_systemone(model, R"({"state": "a call",
      "questions": {"angry": {"type": "noul", "instructions": "Is the caller angry?"}}})",
                               &answered.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
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

#ifndef _WIN32
TEST(CApiServer, AnApiKeyFileThatCanBeReadOnlyOnceProtectsTheServer) {
    // llama.cpp reads --api-key-file while it parses the argv. The embedded server's argv was parsed
    // twice -- by the engine for its own checks, then again inside llama_server() -- so a file that
    // gives its contents once (a pipe: /dev/stdin, a process substitution) was read empty by the
    // second parse, and the server ran without a key, answering every request. The engine's parse is
    // the only one now; the server takes its parameters.
    int fds[2];
    ASSERT_EQ(pipe(fds), 0);
    const std::string key = "jllama-test-key";
    const std::string line = key + "\n";
    ASSERT_EQ(write(fds[1], line.data(), line.size()), static_cast<ssize_t>(line.size()));
    close(fds[1]);
    const std::string key_file = "/dev/fd/" + std::to_string(fds[0]);
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-api-key-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string models_dir = dir.string();
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    probe_listener.join();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "--host",           "127.0.0.1",      "--port",        port_text.c_str(),
                          "--models-dir", models_dir.c_str(), "--api-key-file", key_file.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    httplib::Client client("127.0.0.1", port);
    httplib::Result health;
    for (int i = 0; i < 100 && !(health && health->status == 200); i++) {
        health = client.Get("/health");
        if (!(health && health->status == 200)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    EXPECT_TRUE(health && health->status == 200);
    const auto without = client.Get("/models");
    ASSERT_TRUE(without) << httplib::to_string(without.error());
    EXPECT_EQ(without->status, 401) << "the server answered without its key: " << without->body;
    const auto with = client.Get("/models", httplib::Headers{{"Authorization", "Bearer " + key}});
    ASSERT_TRUE(with) << httplib::to_string(with.error());
    EXPECT_EQ(with->status, 200) << with->body;
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    close(fds[0]);
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}
#endif

TEST(CApiServer, AServerWhoseModelFailsToLoadEndsAndTheNextServerStarts) {
    // The model is loaded on the server's worker thread, after jllama_server_start returned: a file
    // that is no GGUF ends llama_server(), the server reports that it no longer runs, a stop returns
    // at once, and the process can start the next server (one server per process, so the first must
    // really be gone).
    const std::filesystem::path broken = std::filesystem::temp_directory_path() /
                                         ("jllama-c-api-broken-" + std::to_string(std::random_device{}()) + ".gguf");
    {
        std::ofstream out(broken, std::ios::binary);
        out << "this is no gguf file";
    }
    const std::string broken_path = broken.string();
    const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", "0", "-m", broken_path.c_str(), "-ngl", "0"};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(server, nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (jllama_server_is_running(server) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_FALSE(jllama_server_is_running(server)) << "a server whose model failed to load still runs";
    const auto before_stop = std::chrono::steady_clock::now();
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_LT(std::chrono::steady_clock::now() - before_stop, std::chrono::seconds(5));

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-models-next-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string models_dir = dir.string();
    const char *router[] = {"llama-server", "--host", "127.0.0.1", "--port", "0", "--models-dir", models_dir.c_str()};
    jllama_server *next = nullptr;
    ASSERT_EQ(jllama_server_start(7, router, &next), JLLAMA_STATUS_OK) << jllama_last_error();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(jllama_server_is_running(next)) << "the next server did not start";
    EXPECT_EQ(jllama_server_stop(next), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove(broken, ignored);
}

namespace {
// the attached server of a model on a free loopback port with the HTTP options of `extra`, up once
// /health (a public route, no key needed) answers
struct AttachedServer {
    jllama_server *server = nullptr;
    int port = 0;
    explicit AttachedServer(jllama_model *model, const std::vector<std::string> &extra = {}) {
        httplib::Server probe;
        port = probe.bind_to_any_port("127.0.0.1");
        std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
        probe.wait_until_ready();
        probe.stop();
        probe_listener.join();
        std::vector<std::string> args = {"llama-server", "--host", "127.0.0.1", "--port", std::to_string(port)};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        EXPECT_EQ(jllama_server_start_attached(model, static_cast<int>(argv.size()), argv.data(), &server),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
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
    // A client of the server; a slow model never makes a request of these tests fail by timeout.
    httplib::Client client() const {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(std::chrono::seconds(30));
        return c;
    }
};

int status_of(const httplib::Result &res) { return res ? res->status : -1; }

const httplib::Headers BEARER_SECRET = {{"Authorization", "Bearer jllama-attach-secret"}};
} // namespace

TEST_F(CApi, TheAttachedServerKeepsItsApiKeyAndItsCorsOriginsAndAnswersTheDisabledFeaturesWith403) {
    // The attach argv configures the HTTP side (patches/0007): --api-key and --cors-origins reach
    // the attached frontend as they reach a full llama-server, so a model served this way is not
    // left open by the attach mode. /cors-proxy and /tools are the features attach mode does not
    // wire; they answer 403 "disabled", like a stock server built without them, and not 404.
    AttachedServer attached(model_, {"--api-key", "jllama-attach-secret", "--cors-origins", "http://app.example"});
    ASSERT_NE(attached.server, nullptr);
    httplib::Client client = attached.client();

    // the health route is public; every other route needs the key
    EXPECT_EQ(status_of(client.Get("/health")), 200);
    EXPECT_EQ(status_of(client.Get("/v1/models")), 401);
    EXPECT_EQ(status_of(client.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer wrong"}})), 401);
    EXPECT_EQ(status_of(client.Get("/v1/models", BEARER_SECRET)), 200);
    const char *completion = R"({"prompt":"Hi","n_predict":2,"temperature":0})";
    EXPECT_EQ(status_of(client.Post("/completion", completion, "application/json")), 401);
    const auto answered = client.Post("/completion", BEARER_SECRET, completion, "application/json");
    ASSERT_TRUE(answered) << httplib::to_string(answered.error());
    EXPECT_EQ(answered->status, 200) << answered->body;

    // --cors-origins (upstream #32098): the allowed origin is served, a foreign one refused with 403
    httplib::Headers allowed = BEARER_SECRET;
    allowed.emplace("Origin", "http://app.example");
    EXPECT_EQ(status_of(client.Get("/v1/models", allowed)), 200);
    httplib::Headers foreign = BEARER_SECRET;
    foreign.emplace("Origin", "http://evil.example");
    const auto refused = client.Get("/v1/models", foreign);
    ASSERT_TRUE(refused) << httplib::to_string(refused.error());
    EXPECT_EQ(refused->status, 403) << refused->body;
    EXPECT_NE(refused->body.find("permission_error"), std::string::npos) << refused->body;

    // the two stubs, behind the key like every other route
    EXPECT_EQ(status_of(client.Get("/tools")), 401);
    for (const char *path : {"/cors-proxy", "/tools"}) {
        SCOPED_TRACE(path);
        const auto get = client.Get(path, BEARER_SECRET);
        ASSERT_TRUE(get) << httplib::to_string(get.error());
        EXPECT_EQ(get->status, 403) << get->body;
        EXPECT_NE(get->body.find("feature_disabled"), std::string::npos) << get->body;
        const auto post = client.Post(path, BEARER_SECRET, "{}", "application/json");
        ASSERT_TRUE(post) << httplib::to_string(post.error());
        EXPECT_EQ(post->status, 403) << post->body;
    }
}

TEST(CApiServer, TheEndpointTogglesOfTheAttachedServerAreTheModelsParameters) {
    // The attached frontend serves the model's own server_routes (patches/0007), so --metrics,
    // --props and --slots are what the model was loaded with; the same options in the attach argv
    // configure nothing. A model loaded with the opposite of the defaults shows both halves.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-attach-toggles");
    ASSERT_FALSE(path.empty());
    const auto statuses = [](jllama_model *model, const std::vector<std::string> &attach_args) {
        AttachedServer attached(model, attach_args);
        EXPECT_NE(attached.server, nullptr);
        httplib::Client client = attached.client();
        return std::vector<int>{status_of(client.Get("/metrics")), status_of(client.Get("/slots")),
                                status_of(client.Post("/props", "{}", "application/json"))};
    };
    const std::vector<int> defaults = {501, 200, 501};
    const std::vector<int> opposite = {200, 501, 200};

    jllama_model *plain = load_tiny(path, {});
    ASSERT_NE(plain, nullptr) << jllama_last_error();
    EXPECT_EQ(statuses(plain, {}), defaults);
    EXPECT_EQ(statuses(plain, {"--metrics", "--props", "--no-slots"}), defaults)
        << "the attach argv switched an endpoint of the model";
    jllama_model_free(plain);

    jllama_model *toggled = load_tiny(path, {"--metrics", "--props", "--no-slots"});
    ASSERT_NE(toggled, nullptr) << jllama_last_error();
    EXPECT_EQ(statuses(toggled, {}), opposite);
    EXPECT_EQ(statuses(toggled, {"--slots"}), opposite);
    jllama_model_free(toggled);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
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

#ifndef _WIN32
namespace {

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0 // macOS: SO_NOSIGPIPE on each socket instead
#endif

void no_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
    const int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#else
    (void)fd;
#endif
}

// A TCP proxy on loopback in front of `target_port` that waits `delay` after it accepts one of its
// first `delayed` connections before it forwards anything: an RPC server that is slow to answer, as
// one across a slow network is. The later connections (the load's own, after its parse) are not
// delayed, which keeps the test short.
class delaying_proxy {
  public:
    delaying_proxy(int target_port, std::chrono::milliseconds delay, int delayed)
        : target_port_(target_port), delay_(delay), delayed_(delayed) {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        socklen_t len = sizeof(addr);
        if (listen_fd_ < 0 || ::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
            ::listen(listen_fd_, 16) != 0 ||
            ::getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&addr), &len) != 0) {
            return;
        }
        port_ = ntohs(addr.sin_port);
        acceptor_ = std::thread([this]() { accept_loop(); });
    }
    delaying_proxy(const delaying_proxy &) = delete;
    delaying_proxy &operator=(const delaying_proxy &) = delete;
    ~delaying_proxy() {
        stopping_ = true;
        if (acceptor_.joinable()) {
            acceptor_.join(); // polls every 50 ms and sees stopping_
        }
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int fd : open_) {
                ::shutdown(fd, SHUT_RDWR);
            }
        }
        for (auto &pipe : pipes_) {
            pipe.join();
        }
    }
    int port() const { return port_; }
    std::string endpoint() const { return "127.0.0.1:" + std::to_string(port_); }
    int connections() const { return connections_.load(); }

  private:
    void accept_loop() {
        while (!stopping_) {
            pollfd p{listen_fd_, POLLIN, 0};
            if (::poll(&p, 1, 50) <= 0) {
                continue;
            }
            const int client = ::accept(listen_fd_, nullptr, nullptr);
            if (client < 0) {
                continue;
            }
            no_sigpipe(client);
            const bool delay = connections_++ < delayed_;
            std::lock_guard<std::mutex> lock(mutex_);
            open_.push_back(client);
            pipes_.emplace_back([this, client, delay]() { forward(client, delay); });
        }
    }
    void forward(int client, bool delay) {
        if (delay) {
            std::this_thread::sleep_for(delay_);
        }
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(target_port_));
        if (server >= 0 && ::connect(server, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
            no_sigpipe(server);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                open_.push_back(server);
            }
            char buffer[65536];
            pollfd fds[2] = {{client, POLLIN, 0}, {server, POLLIN, 0}};
            while (!stopping_) {
                if (::poll(fds, 2, 50) <= 0) {
                    continue;
                }
                bool closed = false;
                for (int k = 0; k < 2 && !closed; ++k) {
                    if ((fds[k].revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
                        continue;
                    }
                    const ssize_t n = ::recv(fds[k].fd, buffer, sizeof(buffer), 0);
                    if (n <= 0) {
                        closed = true;
                        break;
                    }
                    for (ssize_t sent = 0; sent < n && !closed;) {
                        const ssize_t m =
                            ::send(fds[1 - k].fd, buffer + sent, static_cast<size_t>(n - sent), MSG_NOSIGNAL);
                        closed = m <= 0;
                        sent += m > 0 ? m : 0;
                    }
                }
                if (closed) {
                    break;
                }
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (int fd : {client, server}) {
            if (fd >= 0) {
                open_.erase(std::remove(open_.begin(), open_.end(), fd), open_.end());
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
            }
        }
    }

    int target_port_;
    std::chrono::milliseconds delay_;
    int delayed_;
    int listen_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<int> connections_{0};
    std::mutex mutex_;
    std::vector<int> open_;
    std::vector<std::thread> pipes_;
    std::thread acceptor_; // last: it reads the members above
};

// The longest a swap of the log callback waits while `load` runs on another thread. Every parse holds
// the lock of the process-wide log, which the swap takes too.
std::chrono::milliseconds longest_log_swap_during(const std::function<void()> &load) {
    std::atomic<bool> loading{true};
    std::chrono::steady_clock::duration longest{0};
    std::thread swapper([&]() {
        while (loading.load()) {
            const auto started = std::chrono::steady_clock::now();
            jllama_set_log_callback([](int, const char *, void *) {}, nullptr);
            longest = std::max(longest, std::chrono::steady_clock::now() - started);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    load();
    loading = false;
    swapper.join();
    jllama_set_log_callback(nullptr, nullptr);
    return std::chrono::duration_cast<std::chrono::milliseconds>(longest);
}

} // namespace

TEST(CApiRpc, ALoadWithASlowRpcServerDoesNotHoldTheLogLockWhileItConnects) {
    // The engine parses a load's argv under the lock of the process-wide log, and upstream's --rpc
    // handler registers -- connects to -- every server the parse sees. Before the review of 2026-11-19
    // the registration of an endpoint the engine had just registered connected to it again inside the
    // parse (patches/0015 re-checked the cache hit there), and LLAMA_ARG_RPC's servers were registered
    // there for the first time: a slow server held every logger swap and every other load for as long
    // as its connect took. The servers are registered before the parse now, and the parse connects to
    // none.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-rpc-slow");
    ASSERT_FALSE(path.empty());
    int port = 0;
    std::unique_ptr<rpc_serve_run> server;
    for (int attempt = 0; attempt < 50 && port == 0; ++attempt) {
        const int candidate = rpc_test_port(attempt);
        server.reset(new rpc_serve_run(candidate));
        for (int i = 0;
             i < 500 && !jllama_rpc_server_listening() && !server->returned_within(std::chrono::milliseconds(0)); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (jllama_rpc_server_listening()) {
            port = candidate;
        } else {
            server.reset();
        }
    }
    ASSERT_NE(port, 0) << "no RPC server could bind";
    // The first two connections through a proxy wait this long: the registration, and the re-check of
    // a cached registration that the parse made before (or the parse's own registration of
    // LLAMA_ARG_RPC's server). A parse that connected under the log lock held a swap for about as
    // long; the bound leaves the rest for a slow machine (qemu, s390x) to parse in.
    const auto delay = std::chrono::milliseconds(1500);
    const int delayed = 2;
    struct env_reset {
        ~env_reset() { unsetenv("LLAMA_ARG_RPC"); }
    } reset;
    for (const bool from_env : {false, true}) {
        SCOPED_TRACE(from_env ? "LLAMA_ARG_RPC" : "--rpc");
        delaying_proxy proxy(port, delay, delayed); // a new endpoint each round: nothing of it is registered yet
        ASSERT_NE(proxy.port(), 0);
        std::vector<std::string> args = {"jllama", "-m", path, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2"};
        if (from_env) {
            setenv("LLAMA_ARG_RPC", proxy.endpoint().c_str(), 1);
        } else {
            args.insert(args.end(), {"--rpc", proxy.endpoint()});
        }
        jllama_model *model = nullptr;
        const auto longest = longest_log_swap_during([&]() {
            std::vector<const char *> argv;
            for (const auto &a : args) {
                argv.push_back(a.c_str());
            }
            EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
                      JLLAMA_STATUS_OK)
                << jllama_last_error();
        });
        unsetenv("LLAMA_ARG_RPC");
        EXPECT_GT(proxy.connections(), 0) << "the load never reached the server";
        EXPECT_LT(longest, delay / 2) << "a logger swap waited " << longest.count() << " ms for the parse";
        if (model != nullptr) {
            Owned completion;
            EXPECT_EQ(jllama_completions(model, R"({"prompt":"Hello","n_predict":2,"temperature":0})", &completion.s),
                      JLLAMA_STATUS_OK)
                << jllama_last_error();
            jllama_model_free(model);
        }
    }
#ifdef JLLAMA_HAS_NATIVE_SERVER
    {
        // the embedded server parses its argv once, in jllama_server_start, under the same lock
        SCOPED_TRACE("jllama_server_start");
        delaying_proxy proxy(port, delay, delayed);
        ASSERT_NE(proxy.port(), 0);
        const std::vector<std::string> args = {
            "llama-server", "-m",        path,     "-c", "64",    "-ngl",          "0",
            "--host",       "127.0.0.1", "--port", "0",  "--rpc", proxy.endpoint()};
        jllama_server *native = nullptr;
        const auto longest = longest_log_swap_during([&]() {
            std::vector<const char *> argv;
            for (const auto &a : args) {
                argv.push_back(a.c_str());
            }
            EXPECT_EQ(jllama_server_start(static_cast<int>(argv.size()), argv.data(), &native), JLLAMA_STATUS_OK)
                << jllama_last_error();
        });
        EXPECT_GT(proxy.connections(), 0) << "the start never reached the server";
        EXPECT_LT(longest, delay / 2) << "a logger swap waited " << longest.count() << " ms for the parse";
        if (native != nullptr) {
            EXPECT_EQ(jllama_server_stop(native), JLLAMA_STATUS_OK) << jllama_last_error();
        }
    }
    {
        // an attached server parses its argv the same way; its --rpc servers are registered (and an
        // unreachable one refused) before the parse, as for a start of its own
        SCOPED_TRACE("jllama_server_start_attached");
        jllama_model *model = load_tiny(path, {});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        delaying_proxy proxy(port, delay, delayed);
        ASSERT_NE(proxy.port(), 0);
        const std::vector<std::string> args = {"llama-server", "--host",        "127.0.0.1", "--port", "0",
                                               "--rpc",        proxy.endpoint()};
        jllama_server *attached = nullptr;
        const auto longest = longest_log_swap_during([&]() {
            std::vector<const char *> argv;
            for (const auto &a : args) {
                argv.push_back(a.c_str());
            }
            EXPECT_EQ(jllama_server_start_attached(model, static_cast<int>(argv.size()), argv.data(), &attached),
                      JLLAMA_STATUS_OK)
                << jllama_last_error();
        });
        EXPECT_GT(proxy.connections(), 0) << "the attach never registered the server";
        EXPECT_LT(longest, delay / 2) << "a logger swap waited " << longest.count() << " ms for the parse";
        if (attached != nullptr) {
            EXPECT_EQ(jllama_server_stop(attached), JLLAMA_STATUS_OK) << jllama_last_error();
        }
        const std::vector<std::string> unreachable = {"llama-server", "--host",     "127.0.0.1", "--port", "0",
                                                      "--rpc",        "127.0.0.1:1"};
        std::vector<const char *> argv;
        for (const auto &a : unreachable) {
            argv.push_back(a.c_str());
        }
        jllama_server *refused = nullptr;
        EXPECT_NE(jllama_server_start_attached(model, static_cast<int>(argv.size()), argv.data(), &refused),
                  JLLAMA_STATUS_OK);
        EXPECT_EQ(refused, nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find("127.0.0.1:1"), std::string::npos) << jllama_last_error();
        jllama_model_free(model);
    }
#endif
    server.reset();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
#endif

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

// jllama_tts_options_init is a static inline over jllama_tts_options_init_size with the header's
// sizeof, so a library that grew never writes past an older caller's struct: asked for the 64 bytes
// of 6.1.0's first header it writes those and struct_size 64; asked for more than it knows it writes
// its own sizeof; a size that does not hold struct_size writes nothing.
TEST(CApiTts, TheOptionsInitWritesNoMoreThanTheCallerDeclares) {
    const size_t first = offsetof(jllama_tts_options, speaker_reference_text);
    unsigned char bytes[256];
    std::memset(bytes, 0x5a, sizeof(bytes));
    auto *options = reinterpret_cast<jllama_tts_options *>(bytes);
    jllama_tts_options_init_size(options, first);
    EXPECT_EQ(options->struct_size, first);
    EXPECT_FLOAT_EQ(options->loudness, 0.0f);
    for (size_t i = first; i < sizeof(bytes); i++) {
        ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " is behind the caller's struct";
    }
    std::memset(bytes, 0x5a, sizeof(bytes));
    jllama_tts_options_init_size(options, sizeof(bytes));
    EXPECT_EQ(options->struct_size, sizeof(jllama_tts_options));
    for (size_t i = sizeof(jllama_tts_options); i < sizeof(bytes); i++) {
        ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " is behind the library's struct";
    }
    std::memset(bytes, 0x5a, sizeof(bytes));
    jllama_tts_options_init_size(options, sizeof(size_t) - 1);
    EXPECT_EQ(bytes[0], 0x5a);
    jllama_tts_options_init_size(nullptr, sizeof(jllama_tts_options)); // ignored
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
    // a size between the sizes of the header's versions cuts a field: half of the transcript's pointer
    // read as one, or cfg_scale without flow_steps
    for (const size_t cut : {offsetof(jllama_tts_options, speaker_reference_text) + 4,
                             offsetof(jllama_tts_options, flow_steps), sizeof(jllama_tts_options) - 4, size_t(40)}) {
        options.struct_size = cut;
        EXPECT_EQ(jllama_tts_synthesize_options(tts, "Hi", &options, &wav, &length), JLLAMA_STATUS_INVALID_ARGUMENT)
            << cut;
        EXPECT_EQ(wav, &sentinel);
    }

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
    // synthesis are the same samples as a synthesis that stops there. Over a backbone that cannot end
    // its speech, so every synthesis runs to its frame cap on every platform.
    const std::string endless = jllama_test::write_tiny_qwen3tts_gguf("jllama-qwen3tts-endless", false);
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(endless.c_str(), mmproj().c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    const auto synthesize = [&](int max_frames, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = max_frames;
        options.seed = 7;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        ASSERT_EQ(jllama_tts_synthesize_options(tts, "hello world", &options, &bytes, &length), JLLAMA_STATUS_OK)
            << jllama_last_error();
        wav->assign(bytes, bytes + length);
        jllama_bytes_free(bytes);
    };
    std::vector<unsigned char> one;
    std::vector<unsigned char> two;
    std::vector<unsigned char> three;
    synthesize(72, &one);
    synthesize(73, &two);
    synthesize(150, &three);
    EXPECT_EQ(frames_of(one), 72);
    EXPECT_EQ(frames_of(two), 73);
    EXPECT_EQ(frames_of(three), 150);
    ASSERT_GE(three.size(), one.size());
    EXPECT_TRUE(std::equal(one.begin() + 44, one.end(), three.begin() + 44))
        << "the window boundary must not change the samples before it";
    jllama_tts_free(tts);
    std::remove(endless.c_str());
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

TEST_F(CApiQwen3Tts, TopPAndTheRepetitionPenaltyReachTheSampler) {
    // /tts's top_p and repeat_penalty (b12431) configure the backbone's sampler: with a seed, each
    // changes the codes against the defaults, and the same options give the same audio twice. A
    // value that reached nothing would leave the audio of the defaults.
    const auto speak_with = [](float top_p, float repeat_penalty, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.max_frames = 16;
        options.seed = 7;
        options.top_p = top_p;
        options.repeat_penalty = repeat_penalty;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "hello world", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    jllama_tts_options defaults;
    jllama_tts_options_init(&defaults);
    std::vector<unsigned char> plain;
    ASSERT_EQ(speak_with(defaults.top_p, defaults.repeat_penalty, &plain), JLLAMA_STATUS_OK) << jllama_last_error();
    for (const auto &option : {std::make_pair(0.5f, defaults.repeat_penalty), std::make_pair(defaults.top_p, 1.3f)}) {
        SCOPED_TRACE("top_p " + std::to_string(option.first) + ", repeat_penalty " + std::to_string(option.second));
        std::vector<unsigned char> once;
        std::vector<unsigned char> twice;
        ASSERT_EQ(speak_with(option.first, option.second, &once), JLLAMA_STATUS_OK) << jllama_last_error();
        ASSERT_EQ(speak_with(option.first, option.second, &twice), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_EQ(once, twice) << "the same options and seed give the same audio";
        EXPECT_EQ(frames_of(once), 16);
        EXPECT_NE(once, plain) << "the option did not change the codes";
    }
}

TEST_F(CApiQwen3Tts, TwoSynthesesOnOneHandleAtOnceEqualTheSameOneAfterTheOther) {
    // One jllama_tts_model is shared by two threads: the engine serialises the syntheses
    // (synthesize_mutex), each of which clears the KV cache and resets its sampler, so neither sees
    // the other's state and each answers what it answers alone.
    std::vector<unsigned char> first;
    std::vector<unsigned char> second;
    ASSERT_EQ(speak("hello world", &first, 12, 7), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak("good morning", &second, 12, 8), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(first, second);
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        std::vector<unsigned char> a;
        std::vector<unsigned char> b;
        jllama_status status_a = JLLAMA_STATUS_ERROR;
        jllama_status status_b = JLLAMA_STATUS_ERROR;
        std::thread ta([&]() { status_a = speak("hello world", &a, 12, 7); });
        std::thread tb([&]() { status_b = speak("good morning", &b, 12, 8); });
        ta.join();
        tb.join();
        ASSERT_EQ(status_a, JLLAMA_STATUS_OK);
        ASSERT_EQ(status_b, JLLAMA_STATUS_OK);
        EXPECT_EQ(a, first);
        EXPECT_EQ(b, second);
    }
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
            // the server reports a token's probabilities only with its text, which an end of generation or
            // a byte of an unfinished UTF-8 character has none of (on Windows arm64 the greedy answer to
            // one clip was such a token); the grammar keeps the answer an ASCII letter, and the probabilities
            // stay the model's own softmax (post_sampling_probs is off)
            {"grammar", "root ::= [a-z]"},
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

TEST(CApiAudio, AStreamedTranscriptionCarriesTheTranscriptAndNotTheReasoning) {
    // b12597 (#32571): the streamed deltas of /v1/audio/transcriptions are the parsed content of the
    // answer, as the final event's text is; a reasoning part (Qwen3-ASR's "language ..." prefix) is not
    // transcript. Until then every delta was the raw generated text. The generator's audio projector
    // on the plain decoder with a template that opens a thinking block, and a reasoning budget of two
    // tokens for every request (form fields reach the route as strings), make each answer a reasoning
    // part, the forced end tag and the content. The model's end of generation is a byte of its own, so
    // the '>' of the end tag does not end the answer (the byte vocabulary's default EOS is '>').
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-audio-asr";
    spec.n_ctx_train = 512;
    spec.eos = 4;
    const std::string path = write_tiny_llama_gguf(spec);
    const std::string mmproj = write_tiny_audio_mmproj_gguf("jllama-c-api-audio-asr-mmproj");
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(mmproj.empty());
    jllama_model *model = load_tiny(
        path, {"--mmproj", mmproj, "-c", "512", "--chat-template", THINKING_TEMPLATE, "--reasoning-budget", "2"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::vector<float> tone(16000);
    for (size_t i = 0; i < tone.size(); ++i) {
        tone[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 16000.0f);
    }
    const std::vector<uint8_t> wav = jllama_tts::pcm_to_wav16_bytes(tone, 16000);
    const auto transcribe = [&](httplib::Client &client, bool stream) {
        const httplib::UploadFormDataItems items = {
            {"file", std::string(wav.begin(), wav.end()), "clip.wav", "audio/wav"},
            {"temperature", "0", "", ""},
            {"max_tokens", "20", "", ""},
            {"stream", stream ? "true" : "false", "", ""},
        };
        const auto response = client.Post("/v1/audio/transcriptions", items);
        EXPECT_TRUE(response) << httplib::to_string(response.error());
        EXPECT_EQ(response ? response->status : 0, 200) << (response ? response->body : std::string());
        return response ? response->body : std::string();
    };
    std::string text;
    std::string deltas;
    int delta_events = 0;
    json done;
    {
        AttachedServer attached(model);
        ASSERT_NE(attached.server, nullptr);
        httplib::Client client("127.0.0.1", attached.port);
        client.set_read_timeout(std::chrono::seconds(60));
        const json whole = json::parse(transcribe(client, false));
        text = whole.at("text").get<std::string>();
        std::istringstream sse(transcribe(client, true));
        for (std::string line; std::getline(sse, line);) {
            if (line.rfind("data: ", 0) != 0 || line == "data: [DONE]") {
                continue;
            }
            const json event = json::parse(line.substr(6));
            const std::string type = event.value("type", std::string());
            if (type == "transcript.text.delta") {
                deltas += event.at("delta").get<std::string>();
                ++delta_events;
            } else if (type == "transcript.text.done") {
                done = event;
            }
        }
    }
    ASSERT_TRUE(done.is_object()) << "no transcript.text.done event";
    EXPECT_FALSE(text.empty());
    EXPECT_EQ(done.at("text").get<std::string>(), text);
    EXPECT_EQ(deltas, text) << delta_events << " deltas";
    EXPECT_EQ(deltas.find("</think>"), std::string::npos) << deltas;
    // the answer had a reasoning part: more tokens were generated than the transcript holds
    int *tokens = nullptr;
    size_t count = 0;
    ASSERT_EQ(jllama_tokenize(model, text.c_str(), false, &tokens, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tokens_free(tokens);
    EXPECT_GT(done.at("usage").at("output_tokens").get<size_t>(), count + 8) << done.dump();
    jllama_model_free(model);
    std::remove(path.c_str());
    std::remove(mmproj.c_str());
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

TEST(CApiLfm3AudioSleep, ARealtimeSessionWakesASleepingModel) {
    // The /v1/realtime handler read the model's projector and vocabulary before it woke the server
    // (upstream #32567, b12528): on a model that had fallen asleep (--sleep-idle-seconds) those were
    // freed, and a session -- a conversation or a live transcription -- touched freed memory. It wakes
    // the model first now, holds it awake for the session, and the model sleeps again afterwards.
    const std::string backbone = jllama_test::write_tiny_lfm3_gguf("jllama-lfm3-sleep", true, true);
    const std::string mmproj = jllama_test::write_tiny_lfm3_mmproj_gguf("jllama-lfm3-sleep-mmproj", -10.0f, true, 0.0f);
    ASSERT_FALSE(backbone.empty());
    ASSERT_FALSE(mmproj.empty());
    struct Lines {
        std::mutex mutex;
        int sleeps = 0;
        int count() {
            std::lock_guard<std::mutex> lock(mutex);
            return sleeps;
        }
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
    jllama_model *model = load_tiny(backbone, {"--mmproj", mmproj, "-c", "1024", "-b", "1024", "-ub", "1024", "--seed",
                                               "5", "--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const auto asleep = [&lines](int n) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (lines.count() < n && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return lines.count() >= n;
    };
    {
        AttachedServer attached(model);
        ASSERT_NE(attached.server, nullptr);
        ASSERT_TRUE(asleep(1)) << "the model did not fall asleep";
        {
            SCOPED_TRACE("live transcription");
            RealtimeSession session(attached.port, "?intent=transcription");
            ASSERT_TRUE(session.open());
            const json created = session.next();
            ASSERT_EQ(created.value("type", std::string()), "session.created") << created.dump();
            EXPECT_EQ(created.at("session").at("audio").at("input").at("format").at("rate").get<int>(), 16000);
            std::vector<uint8_t> pcm;
            for (int i = 0; i < 16000; ++i) {
                const auto v = static_cast<int16_t>(16000.0f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f));
                pcm.push_back(static_cast<uint8_t>(v & 0xff));
                pcm.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
            }
            session.send({{"type", "input_audio_buffer.append"}, {"audio", base64_of(pcm)}});
            session.send({{"type", "input_audio_buffer.commit"}, {"final", true}});
            const json completed = session.await("conversation.item.input_audio_transcription.completed");
            EXPECT_EQ(completed.value("type", std::string()), "conversation.item.input_audio_transcription.completed")
                << completed.dump();
            session.ws.close();
        }
        ASSERT_TRUE(asleep(2)) << "the model did not fall asleep again after the session";
        {
            SCOPED_TRACE("conversation");
            RealtimeSession session(attached.port);
            ASSERT_TRUE(session.open());
            const json created = session.next();
            ASSERT_EQ(created.value("type", std::string()), "session.created") << created.dump();
            session.user_text("Hello there");
            session.response_create("text", 4);
            const auto answer = session.collect();
            EXPECT_EQ(answer.done.value("type", std::string()), "response.done") << answer.done.dump();
            session.ws.close();
        }
        ASSERT_TRUE(asleep(3)) << "the model did not fall asleep again after the conversation";
    }
    jllama_model_free(model);
    std::remove(backbone.c_str());
    std::remove(mmproj.c_str());
}

TEST_F(CApiLfm3Audio, AnIdleRealtimeSessionDoesNotHoldTheServerStop) {
    // A /v1/realtime client that connected and then sends nothing holds a WebSocket the server reads
    // from; the stop must close it rather than wait for the client. Run on a thread of its own, so a
    // stop that hangs fails the test instead of the whole run.
    auto attached = std::make_unique<AttachedServer>(model());
    ASSERT_NE(attached->server, nullptr);
    RealtimeSession session(attached->port);
    ASSERT_TRUE(session.open());
    const json created = session.await("session.created");
    ASSERT_EQ(created.value("type", std::string()), "session.created") << created.dump();
    jllama_server *server = attached->server;
    attached->server = nullptr; // stopped here, not by the destructor
    auto stopped = std::make_shared<std::atomic<int>>(-1);
    std::thread([server, stopped]() { stopped->store(jllama_server_stop(server)); }).detach();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (stopped->load() == -1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ASSERT_EQ(stopped->load(), JLLAMA_STATUS_OK) << "the stop waited for the idle realtime client";
    EXPECT_TRUE(session.next().is_null()) << "the session outlived the server";
    // the model is the model's, not the server's: it still answers
    EXPECT_GT(hear(wav_of(8000, 440.0f)).value("tokens_evaluated", -1), 0);
}

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

TEST(CApiTinyVoca, APipelineWithAModelOfItsOwnSpeaksAfterAnRpcServerWentAway) {
    // tiny-voca's two stacks and VoxCPM2's residual LM are llama models the projector builds itself
    // (mtmd_owned_llama_load). Kept on the CPU, they had no device list, so llama.cpp gave them every
    // registered device -- and an RPC server a load registered stays registered for the life of the
    // process: once it was gone, their context failed ("failed to initialize RPC0 backend") although no
    // layer was on it, and so did every synthesis. patches/0037 gives them the empty list.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-rpc-then-tts");
    ASSERT_FALSE(path.empty());
    int port = 0;
    {
        std::unique_ptr<rpc_serve_run> server;
        for (int attempt = 0; attempt < 50 && port == 0; ++attempt) {
            const int candidate = rpc_test_port(attempt);
            server.reset(new rpc_serve_run(candidate));
            for (int i = 0;
                 i < 500 && !jllama_rpc_server_listening() && !server->returned_within(std::chrono::milliseconds(0));
                 ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (jllama_rpc_server_listening()) {
                port = candidate;
            } else {
                server.reset();
            }
        }
        ASSERT_NE(port, 0) << "no RPC server could bind";
        jllama_model *model = load_tiny(path, {"--rpc", "127.0.0.1:" + std::to_string(port)});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        jllama_model_free(model);
    } // the server stops here; its device stays registered
    std::remove(path.c_str());

    TinyVocaPair voca("", 3, 1.0f, 0);
    ASSERT_NE(voca.tts, nullptr);
    std::vector<unsigned char> wav;
    EXPECT_EQ(voca.speak(TINY_VOCA_HELLO, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(wav_samples(wav), 0u);

    const std::string backbone = jllama_test::write_tiny_voxcpm2_gguf("jllama-voxcpm2-after-rpc");
    const std::string mmproj = jllama_test::write_tiny_voxcpm2_mmproj_gguf("jllama-voxcpm2-after-rpc-mmproj", 100.0f);
    jllama_tts_model *voxcpm2 = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &voxcpm2), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    unsigned char *bytes = nullptr;
    size_t length = 0;
    EXPECT_EQ(jllama_tts_synthesize_options(voxcpm2, "Hi", &options, &bytes, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_GT(length, 44u);
    jllama_bytes_free(bytes);
    jllama_tts_free(voxcpm2);
    std::remove(backbone.c_str());
    std::remove(mmproj.c_str());
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

TEST_F(CApiZonos3, ADeliveryNextToGuidanceIsRefusedNamingBothCauses) {
    // ZONOS3 speaks a speed and has no solver steps: with both asked for, llama.cpp's set_input refuses
    // the request (it checks the delivery first, the guidance after it) and says which in its log only.
    // The message names both, instead of blaming the guidance alone as it did until 2026-11-19.
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 40;
    options.seed = 7;
    options.speed = 1.2f;
    options.flow_steps = 8;
    unsigned char *bytes = nullptr;
    size_t length = 0;
    EXPECT_EQ(jllama_tts_synthesize_options(tts(), "Hello.", &options, &bytes, &length), JLLAMA_STATUS_INVALID_REQUEST);
    jllama_bytes_free(bytes);
    const std::string error = jllama_last_error();
    EXPECT_NE(error.find("speed or loudness"), std::string::npos) << error;
    EXPECT_NE(error.find("cfg_scale or flow_steps"), std::string::npos) << error;
    // and the speed alone is spoken
    options.flow_steps = 0;
    bytes = nullptr;
    EXPECT_EQ(jllama_tts_synthesize_options(tts(), "Hello.", &options, &bytes, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_bytes_free(bytes);
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

// A random YuE2 pair (tiny_llama_gguf.hpp, llama.cpp b12459, #32369): songs in stereo at 48 kHz. The
// backbone writes an ABC score, then semantic codes; the NAR of the mmproj -- a qwen3 layer that mtmd
// builds into a llama model of its own and runs over the backbone's keys and values -- turns the codes
// into latents by flow matching, and the VAE a latent frame into 1920 samples per channel. llama.cpp's
// driver (yue2_gen_audio_pipeline, mtmd-helper-gen.cpp) cuts the last 64 samples of a song (vae_trim),
// so n codes are 1920 * n - 64 samples, and a song cannot end by itself before 200 codes (the code
// minimum of protocol.py): the frame cap decides its length. An engine step is one backbone token -- a
// score token, the step that closes the score with </abc><|music_start|>, or a code -- and the
// generator's score is exactly 32 tokens long. On noise weights, so what is asserted is the contract.
class CApiYuE2 : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path() = jllama_test::write_tiny_yue2_gguf();
        mmproj_path() = jllama_test::write_tiny_yue2_mmproj_gguf();
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
    struct song {
        const char *voice = "off"; // the template's mode: off, melody or full (the default)
        int max_frames = 12;
        unsigned int seed = 11;
        float cfg_scale = 0.0f;
        int flow_steps = 2;
        int raw_prompt = 0;
        const char *speaker_reference = nullptr;
    };
    static jllama_status sing(const char *text, const song &s, std::vector<unsigned char> *wav) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.voice = s.voice;
        options.max_frames = s.max_frames;
        options.seed = s.seed;
        options.cfg_scale = s.cfg_scale;
        options.flow_steps = s.flow_steps;
        options.raw_prompt = s.raw_prompt;
        options.speaker_reference_path = s.speaker_reference;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK && wav != nullptr) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    }
    // 16-bit stereo behind the 44-byte header
    static size_t samples_per_channel(const std::vector<unsigned char> &wav) { return (wav.size() - 44) / 4; }
    static size_t song_of(size_t codes) { return 1920 * codes - 64; }
    static constexpr const char *LYRICS = "[Tags]\npop\n[Lyrics]\nla la la";
};

TEST_F(CApiYuE2, TheProjectorLoadsWithoutTheRerankingHead) {
    // The NAR layers are a qwen3 model that mtmd builds from the mmproj's metadata, and llama.cpp's
    // qwen3 asks for cls.output.weight, a reranking head no converted YuE2 carries. b12553 (#32606) lets
    // it be absent; before it every YuE2 pair failed to load ("tensor cls.output.weight is missing from
    // the mmproj").
    gguf_init_params params = {true, nullptr};
    gguf_context *g = gguf_init_from_file(mmproj_path().c_str(), params);
    ASSERT_NE(g, nullptr);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        EXPECT_EQ(std::string(gguf_get_tensor_name(g, i)).find("cls"), std::string::npos) << gguf_get_tensor_name(g, i);
    }
    gguf_free(g);
    jllama_tts_model *again = nullptr;
    ASSERT_EQ(jllama_tts_load(model_path().c_str(), mmproj_path().c_str(), 0, 2, &again), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_tts_free(again);
}

TEST_F(CApiYuE2, ASongIsAStereoWavAt48kHzOfWholeLatentFrames) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(sing(LYRICS, song{}, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(std::string(wav.begin(), wav.begin() + 4), "RIFF");
    EXPECT_EQ(std::string(wav.begin() + 8, wav.begin() + 12), "WAVE");
    const uint32_t rate =
        uint32_t(wav[24]) | uint32_t(wav[25]) << 8 | uint32_t(wav[26]) << 16 | uint32_t(wav[27]) << 24;
    EXPECT_EQ(rate, 48000u) << "the VAE's rate";
    EXPECT_EQ(wav[22] | wav[23] << 8, 2) << "YuE2 is stereo (mtmd_gen_audio_info::n_channels)";
    // the "off" mode renders <abc></abc><|music_start|>: every one of the 12 steps is a code
    EXPECT_EQ(samples_per_channel(wav), song_of(12));
    std::vector<unsigned char> again;
    ASSERT_EQ(sing(LYRICS, song{}, &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same song";
    song reseeded;
    reseeded.seed = 12;
    std::vector<unsigned char> other_seed;
    ASSERT_EQ(sing(LYRICS, reseeded, &other_seed), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(wav, other_seed) << "the seed draws the codes and the noise the flow starts from";
    EXPECT_EQ(other_seed.size(), wav.size());
    std::vector<unsigned char> other_text;
    ASSERT_EQ(sing("[Tags]\njazz\n[Lyrics]\nna na na", song{}, &other_text), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(wav, other_text) << "another text is another prompt";
    song one;
    one.max_frames = 1;
    std::vector<unsigned char> shortest;
    ASSERT_EQ(sing(LYRICS, one, &shortest), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(shortest), song_of(1));
}

TEST_F(CApiYuE2, WithoutTheOffModeTheBackboneWritesAScoreOf32TokensBeforeTheCodes) {
    // The default mode ends the prompt with <abc>: the backbone writes a score, at least 32 tokens
    // (protocol.py), and the generator's ends there; the 33rd step writes </abc><|music_start|>, and only
    // the steps after it are codes.
    song full;
    full.voice = nullptr;
    full.max_frames = 33;
    EXPECT_EQ(sing(LYRICS, full, nullptr), JLLAMA_STATUS_ERROR) << "a score without a code is no song";
    full.max_frames = 34;
    std::vector<unsigned char> one_code;
    ASSERT_EQ(sing(LYRICS, full, &one_code), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(one_code), song_of(1));
    full.max_frames = 33 + 12;
    std::vector<unsigned char> twelve_codes;
    ASSERT_EQ(sing(LYRICS, full, &twelve_codes), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(twelve_codes), song_of(12));
    std::vector<unsigned char> off;
    ASSERT_EQ(sing(LYRICS, song{}, &off), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(twelve_codes, off) << "the score and the instruction are part of what the codes follow";
    // "melody" asks for a score without chords: another instruction, the same rule
    song melody = full;
    melody.voice = "melody";
    std::vector<unsigned char> melody_codes;
    ASSERT_EQ(sing(LYRICS, melody, &melody_codes), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(melody_codes), song_of(12));
    EXPECT_NE(melody_codes, twelve_codes);
    // the error above left the model as it was
    std::vector<unsigned char> off_again;
    ASSERT_EQ(sing(LYRICS, song{}, &off_again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(off_again, off);
}

TEST_F(CApiYuE2, ACoverBringsItsScoreAndTheOffModeDropsIt) {
    // A text with "\n[ABC]\n" puts its score into the prompt (cover mode): the codes come at once.
    const std::string cover = std::string(LYRICS) + "\n[ABC]\nX:1\nK:C\nCDEF GABc";
    song mode;
    mode.voice = nullptr;
    std::vector<unsigned char> covered;
    ASSERT_EQ(sing(cover.c_str(), mode, &covered), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(covered), song_of(12));
    std::vector<unsigned char> off;
    ASSERT_EQ(sing(LYRICS, song{}, &off), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(covered, off);
    // the off mode keeps the text before the score and renders an empty one
    std::vector<unsigned char> cover_off;
    ASSERT_EQ(sing(cover.c_str(), song{}, &cover_off), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(cover_off, off);
}

TEST_F(CApiYuE2, GuidanceAndSolverStepsAreTakenAndChangeTheSong) {
    // b12459 passes cfg_scale and flow_steps through. The off mode guides its codes by 1.01 by itself
    // (protocol.py), with the bare instruction as the unconditioned prompt in a second sequence.
    std::vector<unsigned char> own;
    ASSERT_EQ(sing(LYRICS, song{}, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    song guided;
    guided.cfg_scale = 3.0f;
    std::vector<unsigned char> stronger;
    ASSERT_EQ(sing(LYRICS, guided, &stronger), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(own, stronger);
    // cfg_scale 1 runs no unconditioned sequence at all, so its negative prompt -- the instruction up to
    // the prompt's first line break -- is never built: a raw prompt without a line break sings with 1
    // and is refused with the model's own 1.01. Whether 1.01 and 1 give other audio is up to rounding
    // (the guidance moves the code logits by 1 % of their difference; the same WAV on aarch64 and macOS)
    const char *no_line_break = "<|endoftext|>Generate music la la la<abc></abc><|music_start|>";
    song unguided_raw;
    unguided_raw.raw_prompt = 1;
    unguided_raw.cfg_scale = 1.0f;
    std::vector<unsigned char> unguided;
    ASSERT_EQ(sing(no_line_break, unguided_raw, &unguided), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(unguided.size(), 44u);
    song own_raw;
    own_raw.raw_prompt = 1;
    EXPECT_EQ(sing(no_line_break, own_raw, nullptr), JLLAMA_STATUS_ERROR)
        << "the model's own 1.01 builds the negative prompt, which needs the line break";
    song steps;
    steps.flow_steps = 5;
    std::vector<unsigned char> more_steps;
    ASSERT_EQ(sing(LYRICS, steps, &more_steps), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_NE(own, more_steps) << "the ODE solved in 5 steps ends elsewhere than in 2";
    EXPECT_EQ(own.size(), more_steps.size()) << "the steps change the latents, not their number";
    // 0 is the model's own: the 32 midpoint steps of the reference
    steps.flow_steps = 0;
    std::vector<unsigned char> model_steps;
    ASSERT_EQ(sing(LYRICS, steps, &model_steps), JLLAMA_STATUS_OK) << jllama_last_error();
    steps.flow_steps = 32;
    std::vector<unsigned char> thirty_two;
    ASSERT_EQ(sing(LYRICS, steps, &thirty_two), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(model_steps, thirty_two);
    EXPECT_NE(model_steps, own);
}

TEST_F(CApiYuE2, ALongSongIsDecodedInVaeTilesOfWholeFrames) {
    // The VAE decodes tiles of 128 latent frames with 16 frames of context on each side and keeps only
    // their cores, as decode_tiled() of the reference: 128 frames are one tile, 140 two, and the song
    // keeps every frame of both.
    song tile;
    tile.flow_steps = 1;
    tile.max_frames = 128;
    std::vector<unsigned char> one_tile;
    ASSERT_EQ(sing(LYRICS, tile, &one_tile), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(one_tile), song_of(128));
    tile.max_frames = 140;
    std::vector<unsigned char> two_tiles;
    ASSERT_EQ(sing(LYRICS, tile, &two_tiles), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(samples_per_channel(two_tiles), song_of(140));
    std::vector<unsigned char> again;
    ASSERT_EQ(sing(LYRICS, tile, &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(two_tiles, again);
}

TEST_F(CApiYuE2, ItClonesNoVoiceAndHasNoPredefinedOnes) {
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported) << "YuE2 reads no reference clip";
    char **names = nullptr;
    size_t count = 7;
    ASSERT_EQ(jllama_tts_voices(tts(), &names, &count), JLLAMA_STATUS_OK);
    EXPECT_EQ(count, 0u) << "its voice is the template's mode";
    jllama_strings_free(names, count);
    song cloned;
    cloned.speaker_reference = "/nonexistent/yue2-reference.wav";
    EXPECT_EQ(sing(LYRICS, cloned, nullptr), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("no voice cloning"), std::string::npos) << jllama_last_error();
}

TEST_F(CApiYuE2, ARawPromptMustBeThePromptOfItsTemplate) {
    // b12494's raw_prompt hands the text over as the prompt: the rendered prompt sings as the template
    // does, and llama.cpp refuses one that does not start with <|endoftext|> or does not end with <abc>
    // or <|music_start|>
    const std::string rendered =
        std::string("<|endoftext|>Generate music with codec tokens from the given conditions.\n") + LYRICS +
        "\n<abc></abc><|music_start|>";
    std::vector<unsigned char> templated;
    ASSERT_EQ(sing(LYRICS, song{}, &templated), JLLAMA_STATUS_OK) << jllama_last_error();
    song raw;
    raw.raw_prompt = 1;
    std::vector<unsigned char> given;
    ASSERT_EQ(sing(rendered.c_str(), raw, &given), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(given, templated);
    EXPECT_EQ(sing(LYRICS, raw, nullptr), JLLAMA_STATUS_ERROR) << "no <|endoftext|> in front";
    EXPECT_EQ(sing("<|endoftext|>Generate music.\nla la la", raw, nullptr), JLLAMA_STATUS_ERROR)
        << "neither <abc> nor <|music_start|> at the end";
    std::vector<unsigned char> after;
    ASSERT_EQ(sing(LYRICS, song{}, &after), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(after, templated);
}

// A random VoxCPM2 pair (tiny_llama_gguf.hpp, llama.cpp b12531, #32443), the first pipeline in the
// suite whose backbone emits no audio token at all: the mmproj turns every hidden state into a patch of
// four latents -- FSQ, the residual LM (a libllama context of its own), the LocDiT flow matching with
// classifier-free guidance, LocEnc for the next backbone row -- and the causal AudioVAE turns a patch
// into 7680 samples at 48 kHz. The stop head of the mmproj carries a fixed score, which makes every
// length exact: llama.cpp's driver (voxcpm2_gen_audio_pipeline, mtmd-helper-gen.cpp) stops on a score
// > 0 once more than min_len + 1 = 3 frames are out, so +100 gives exactly 4 frames, and with -100 it
// runs to max_len = 6 * text tokens + 10 (the limit of the reference). The text goes to the tokenizer
// without BOS, a text of n characters without spaces is n + 1 tokens here (the space prefix), and the
// two mmprojs differ in the stop head only, so their latents are the same frame for frame. On noise
// weights, so what is asserted is the contract, not the voice.
class CApiVoxCpm2 : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        backbone() = jllama_test::write_tiny_voxcpm2_gguf();
        never() = jllama_test::write_tiny_voxcpm2_mmproj_gguf("jllama-voxcpm2-never", -100.0f);
        at_once() = jllama_test::write_tiny_voxcpm2_mmproj_gguf("jllama-voxcpm2-at-once", 100.0f);
        ASSERT_FALSE(backbone().empty());
        ASSERT_FALSE(never().empty());
        ASSERT_FALSE(at_once().empty());
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), never().c_str(), 0, 2, &budget_tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
        ASSERT_EQ(jllama_tts_load(backbone().c_str(), at_once().c_str(), 0, 2, &eos_tts()), JLLAMA_STATUS_OK)
            << jllama_last_error();
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        vocab_model() = llama_model_load_from_file(backbone().c_str(), params);
        ASSERT_NE(vocab_model(), nullptr);
    }
    static void TearDownTestSuite() {
        jllama_tts_free(budget_tts());
        jllama_tts_free(eos_tts());
        llama_model_free(vocab_model());
        budget_tts() = nullptr;
        eos_tts() = nullptr;
        vocab_model() = nullptr;
        for (const std::string &p : {backbone(), never(), at_once()}) {
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
    static jllama_tts_model *&budget_tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static jllama_tts_model *&eos_tts() {
        static jllama_tts_model *t = nullptr;
        return t;
    }
    static llama_model *&vocab_model() {
        static llama_model *m = nullptr;
        return m;
    }
    // the tokens of `text` the way the driver tokenizes it: no BOS, special tokens as plain text
    static int tokens_of(const std::string &text) {
        std::vector<llama_token> ids(text.size() + 16);
        return llama_tokenize(llama_model_get_vocab(vocab_model()), text.c_str(), static_cast<int32_t>(text.size()),
                              ids.data(), static_cast<int32_t>(ids.size()), false, false);
    }
    // A synthesis with seed 7 and the defaults, what `set` changes on top; the WAV in *wav.
    static jllama_status speak(jllama_tts_model *tts, const char *text, std::vector<unsigned char> *wav,
                               const std::function<void(jllama_tts_options &)> &set = nullptr) {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.seed = 7;
        if (set) {
            set(options);
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
    // the frames of a 16-bit mono WAV, -1 when its samples are no whole number of 7680-sample frames
    static long frames_of(const std::vector<unsigned char> &wav) {
        const size_t samples = (wav.size() - 44) / 2;
        return samples % 7680 == 0 ? static_cast<long>(samples / 7680) : -1;
    }
    // the samples of the first `frames` frames, as the bytes of the WAV
    static std::vector<unsigned char> head(const std::vector<unsigned char> &wav, size_t frames) {
        const size_t end = std::min(wav.size(), 44 + frames * 7680 * 2);
        return std::vector<unsigned char>(wav.begin() + 44, wav.begin() + static_cast<std::ptrdiff_t>(end));
    }
};

TEST_F(CApiVoxCpm2, ItHasNoPredefinedVoicesAndWithoutItsEncoderClonesNone) {
    char **names = nullptr;
    size_t count = 7;
    ASSERT_EQ(jllama_tts_voices(eos_tts(), &names, &count), JLLAMA_STATUS_OK);
    EXPECT_EQ(count, 0u) << "a voice is designed in the text or cloned, the mmproj ships none";
    jllama_strings_free(names, count);
    bool supported = true;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(eos_tts(), &supported), JLLAMA_STATUS_OK);
    EXPECT_FALSE(supported) << "the mmproj has no AudioVAE encoder";
    // the reference is refused by name before the model is touched, and the model answers the next request
    EXPECT_EQ(speak(eos_tts(), "Hi", nullptr,
                    [](jllama_tts_options &o) { o.speaker_reference_path = "/nonexistent/voxcpm2-reference.wav"; }),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("does not accept a speaker reference"), std::string::npos)
        << jllama_last_error();
    EXPECT_NE(std::string(jllama_last_error()).find("voxcpm2-reference.wav"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(eos_tts(), "Hi", nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiVoxCpm2, ASynthesisIsAMonoWavOfWholeFramesAt48kHz) {
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak(eos_tts(), "Hi", &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_GT(wav.size(), 44u);
    EXPECT_EQ(std::string(wav.begin(), wav.begin() + 4), "RIFF");
    EXPECT_EQ(std::string(wav.begin() + 8, wav.begin() + 12), "WAVE");
    EXPECT_EQ(le32(wav, 24), 48000u) << "the AudioVAE decoder's rate";
    EXPECT_EQ(wav[22] | wav[23] << 8, 1) << "mono";
    EXPECT_EQ(le32(wav, 40), wav.size() - 44) << "the data chunk holds every sample";
    // a stop at the first frame is held back to 4 frames (n_frames > min_len + 1), 0.64 s
    EXPECT_EQ(frames_of(wav), 4) << wav.size();
    // the generator keeps the decoder inside tanh's range (measured: RMS 5114, peak 30080 of 32767), so the
    // comparisons below are made on samples, not on silence or on the clamp of the 16-bit conversion. The
    // samples are little-endian whatever the host (read as host int16 on s390x they reached 32768).
    int peak = 0;
    for (size_t i = 44; i + 1 < wav.size(); i += 2) {
        const auto sample = static_cast<int16_t>(static_cast<uint16_t>(wav[i] | wav[i + 1] << 8));
        peak = std::max(peak, std::abs(static_cast<int>(sample)));
    }
    EXPECT_GT(peak, 1000);
    EXPECT_LT(peak, 32767);
    std::vector<unsigned char> again;
    ASSERT_EQ(speak(eos_tts(), "Hi", &again), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(wav, again) << "the same seed gives the same audio";
    std::vector<unsigned char> other;
    ASSERT_EQ(speak(eos_tts(), "Hi", &other, [](jllama_tts_options &o) { o.seed = 8; }), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_NE(wav, other) << "the flow matching starts from noise of the seed";
    EXPECT_EQ(other.size(), wav.size());
}

TEST_F(CApiVoxCpm2, WithoutTheStopTheFramesRunToSixPerTextTokenPlusTen) {
    std::vector<unsigned char> hi;
    std::vector<unsigned char> hi_dot;
    ASSERT_EQ(speak(budget_tts(), "Hi", &hi), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi.", &hi_dot), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(tokens_of("Hi"), 3) << "the space piece, H, i";
    ASSERT_EQ(tokens_of("Hi."), 4);
    EXPECT_EQ(frames_of(hi), 6 * 3 + 10);
    EXPECT_EQ(frames_of(hi_dot), 6 * 4 + 10) << "one token more, six frames more";
    // a cap of the caller wins on both sides: below the reference's limit, and below its minimum of 4
    std::vector<unsigned char> capped;
    ASSERT_EQ(speak(budget_tts(), "Hi", &capped, [](jllama_tts_options &o) { o.max_frames = 5; }), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(frames_of(capped), 5);
    ASSERT_EQ(speak(eos_tts(), "Hi", &capped, [](jllama_tts_options &o) { o.max_frames = 2; }), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(frames_of(capped), 2);
}

TEST_F(CApiVoxCpm2, TheStopHeadDecidesTheLengthAndNothingElse) {
    // the two mmprojs differ in the stop head only, and the decoder is causal: the 4 frames of the
    // stopping one are the first 4 frames of the one that runs to its limit, and a cap of 20 frames is
    // the first 20 of 28 -- although the decoder takes 16 frames per call, so its last call there is 4
    // frames and here 12
    std::vector<unsigned char> full;
    std::vector<unsigned char> stopped;
    std::vector<unsigned char> capped;
    ASSERT_EQ(speak(budget_tts(), "Hi", &full), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &stopped), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(budget_tts(), "Hi", &capped, [](jllama_tts_options &o) { o.max_frames = 20; }), JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(frames_of(full), 28);
    ASSERT_EQ(frames_of(stopped), 4);
    ASSERT_EQ(frames_of(capped), 20);
    EXPECT_EQ(head(stopped, 4), head(full, 4));
    EXPECT_EQ(head(capped, 20), head(full, 20));
}

TEST_F(CApiVoxCpm2, AVoiceDescriptionInParenthesesIsSpokenAsPartOfThePrompt) {
    // llama.cpp parses nothing out of "(...)": the description is text of the prompt the model was
    // trained to read as a voice design. So it is not refused, it is not stripped -- its 7 tokens raise
    // the frame limit by 42 -- and it changes the audio of a seed.
    std::vector<unsigned char> plain;
    std::vector<unsigned char> designed;
    ASSERT_EQ(speak(eos_tts(), "Hi", &plain), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "(warm)Hi", &designed), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(frames_of(designed), 4);
    EXPECT_NE(designed, plain);
    std::vector<unsigned char> budget;
    ASSERT_EQ(speak(budget_tts(), "(warm)Hi", &budget), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(tokens_of("(warm)Hi"), 9) << "the space piece, (, w, a, r, m, ), H, i";
    EXPECT_EQ(frames_of(budget), 6 * 9 + 10);
}

TEST_F(CApiVoxCpm2, AChineseWordOfTheVocabularyCountsItsCharacters) {
    // 你好 is one piece of the vocabulary, which the driver splits back into its two characters (VoxCPM's
    // mask_multichar_chinese_tokens): the tokenizer gives the space piece and 你好, the driver three tokens
    // and 28 frames, where the unsplit piece would give 22
    const std::string nihao = "\xE4\xBD\xA0\xE5\xA5\xBD";
    ASSERT_EQ(tokens_of(nihao), 2);
    std::vector<unsigned char> wav;
    ASSERT_EQ(speak(budget_tts(), nihao.c_str(), &wav), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(frames_of(wav), 6 * 3 + 10);
}

TEST_F(CApiVoxCpm2, TheWhitespaceOfTheTextIsCollapsed) {
    // re.sub(r"\s+", " ", text) as the reference: a run of spaces, tabs and line breaks is one space piece
    std::vector<unsigned char> one;
    std::vector<unsigned char> many;
    ASSERT_EQ(speak(eos_tts(), "Hi there", &one), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi \t\n  there", &many), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(many, one);
}

TEST_F(CApiVoxCpm2, TheTemperatureScalesTheFlowNoiseAndAtZeroTheSeedDoesNotMatter) {
    // the temperature is the standard deviation of the noise the flow matching starts from (mtmd-subcontext.cpp,
    // VOXCPM2_GEN), 1 by default; at 0 the noise is zero, and nothing else of this pipeline draws from the
    // seed (the backbone's sampled token is not read), so the seed no longer matters
    const auto at = [](float temperature, unsigned seed) {
        return [temperature, seed](jllama_tts_options &o) {
            o.temperature = temperature;
            o.seed = seed;
        };
    };
    std::vector<unsigned char> own;
    std::vector<unsigned char> one;
    std::vector<unsigned char> half;
    std::vector<unsigned char> cold;
    std::vector<unsigned char> cold_other_seed;
    ASSERT_EQ(speak(eos_tts(), "Hi", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &one, at(1.0f, 7)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &half, at(0.5f, 7)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &cold, at(0.0f, 7)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &cold_other_seed, at(0.0f, 8)), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(own, one) << "the model's default is 1";
    EXPECT_NE(half, own);
    EXPECT_NE(cold, own);
    EXPECT_NE(cold, half);
    EXPECT_EQ(cold, cold_other_seed);
}

TEST_F(CApiVoxCpm2, GuidanceAndSolverStepsAreTakenAndChangeTheAudio) {
    // VoxCPM2 is the third model that takes both, after YuE2 and MiniMax Music 3 (b12459's fields): 0 is the model's
    // own, 2.0 and 10 Euler steps like the reference, so naming those changes nothing; a scale of 1 is the conditioned
    // velocity alone, and 2 and 32 steps solve the same ODE more coarsely and more finely
    const auto with = [](float cfg_scale, int flow_steps) {
        return [cfg_scale, flow_steps](jllama_tts_options &o) {
            o.cfg_scale = cfg_scale;
            o.flow_steps = flow_steps;
        };
    };
    std::vector<unsigned char> own;
    std::vector<unsigned char> named;
    std::vector<unsigned char> unguided;
    std::vector<unsigned char> stronger;
    std::vector<unsigned char> coarse;
    std::vector<unsigned char> fine;
    ASSERT_EQ(speak(eos_tts(), "Hi", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &named, with(2.0f, 10)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &unguided, with(1.0f, 0)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &stronger, with(3.0f, 0)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &coarse, with(0.0f, 2)), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_EQ(speak(eos_tts(), "Hi", &fine, with(0.0f, 32)), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(named, own);
    EXPECT_NE(unguided, own);
    EXPECT_NE(stronger, own);
    EXPECT_NE(stronger, unguided);
    EXPECT_NE(coarse, own);
    EXPECT_NE(fine, own);
    EXPECT_NE(fine, coarse);
    for (const auto *wav : {&unguided, &stronger, &coarse, &fine}) {
        EXPECT_EQ(frames_of(*wav), 4) << "they change the latents, not their number";
    }
}

TEST_F(CApiVoxCpm2, SolverStepsAndAGuidanceOutsideTheirRangeAreRefused) {
    const auto status_of = [](float cfg_scale, int flow_steps) {
        return speak(eos_tts(), "Hi", nullptr, [cfg_scale, flow_steps](jllama_tts_options &o) {
            o.cfg_scale = cfg_scale;
            o.flow_steps = flow_steps;
        });
    };
    // VoxCPM2's own range, 2 to 32 (its flow graph unrolls the steps, the first is skipped by CFG-Zero*):
    // set_input answers 2, which the engine reports as the request's fault: the message names the model's
    // own range as the other cause ("or not these values"), llama.cpp's log line the range itself
    EXPECT_EQ(status_of(0.0f, 1), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("flow_steps"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(status_of(0.0f, 33), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("flow_steps"), std::string::npos) << jllama_last_error();
    // the engine's own range, before the model is touched
    EXPECT_EQ(status_of(0.0f, -1), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("flow_steps must be between"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(status_of(0.0f, 1025), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(status_of(-1.0f, 0), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("cfg_scale must be"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(status_of(std::nanf(""), 0), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(status_of(0.0f, 0), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiVoxCpm2, TheOptionsItDoesNotReadLeaveTheAudioAsIs) {
    // the language is the text's (the driver never reads it), the sampler settings reach only the
    // backbone token the driver ignores, the transcript of a reference is not needed (llama.cpp warns),
    // a voice name has neither a predefined voice nor a "tts" template to go to, and raw_prompt has no
    // template to skip; speed and loudness belong to ZONOS3 and are refused
    std::vector<unsigned char> own;
    ASSERT_EQ(speak(eos_tts(), "Hi", &own), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::vector<std::pair<const char *, std::function<void(jllama_tts_options &)>>> unread = {
        {"lang", [](jllama_tts_options &o) { o.lang = "de"; }},
        {"top_k", [](jllama_tts_options &o) { o.top_k = 1; }},
        {"top_p", [](jllama_tts_options &o) { o.top_p = 0.5f; }},
        {"repeat_penalty", [](jllama_tts_options &o) { o.repeat_penalty = 1.3f; }},
        {"speaker_reference_text", [](jllama_tts_options &o) { o.speaker_reference_text = "what a clip says"; }},
        {"voice", [](jllama_tts_options &o) { o.voice = "alba"; }},
        {"raw_prompt", [](jllama_tts_options &o) { o.raw_prompt = 1; }},
    };
    for (const auto &option : unread) {
        std::vector<unsigned char> wav;
        ASSERT_EQ(speak(eos_tts(), "Hi", &wav, option.second), JLLAMA_STATUS_OK)
            << option.first << ": " << jllama_last_error();
        EXPECT_EQ(wav, own) << option.first;
    }
    EXPECT_EQ(speak(eos_tts(), "Hi", nullptr, [](jllama_tts_options &o) { o.speed = 2.0f; }),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("speed or loudness"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(speak(eos_tts(), "Hi", nullptr, [](jllama_tts_options &o) { o.loudness = -20.0f; }),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(speak(eos_tts(), "Hi", nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
}

// The same backbone with the AudioVAE encoder in its mmproj (voxcpm2_spkenc), which turns a reference
// clip into one backbone row per 160 ms. llama.cpp's mtmd_helper_get_capabilities().audio_gen_spk_ref
// names Qwen3-TTS and Pocket TTS only (mtmd-helper.cpp, unchanged at b12542), so the engine's first
// rule refused the reference by name; the engine now decides by the pipeline and the projector
// (tts_params.hpp, pair_takes_speaker_reference), as upstream's /tts route asks the projector only.
// Red with the old rule. The pipeline itself is driven below the way tts_engine.cpp drives it, to
// show the clip's rows reach the prompt whatever the C API says.
TEST(CApiVoxCpm2Clone, TheEncoderClonesAVoiceThroughTheCApiAndThePipeline) {
    const std::string backbone = jllama_test::write_tiny_voxcpm2_gguf("jllama-voxcpm2-clone");
    const std::string mmproj = jllama_test::write_tiny_voxcpm2_mmproj_gguf("jllama-voxcpm2-clone-mmproj", 100.0f, true);
    ASSERT_FALSE(backbone.empty());
    ASSERT_FALSE(mmproj.empty());
    const auto clip = [](float hz) {
        std::vector<float> tone(16000); // one second at the encoder's 16 kHz: 7 rows of 2560 samples
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.4f * std::sin(2.0f * 3.14159265f * hz * static_cast<float>(i) / 16000.0f);
        }
        const std::string path = (std::filesystem::temp_directory_path() /
                                  ("jllama-voxcpm2-ref-" + std::to_string(static_cast<int>(hz)) + ".wav"))
                                     .string();
        const std::vector<uint8_t> wav = jllama_tts::pcm_to_wav16_bytes(tone, 16000);
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(wav.data()), wav.size());
        return path;
    };
    const std::string low = clip(220.0f);
    const std::string high = clip(880.0f);

    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    bool supported = false;
    ASSERT_EQ(jllama_tts_supports_speaker_reference(tts, &supported), JLLAMA_STATUS_OK);
    // llama.cpp's audio_gen_spk_ref does not name VoxCPM2 (unchanged at b12543); the engine's own rule does
    EXPECT_TRUE(supported);
    // the WAV of "Hi" with seed 7, cloned from the clip at `reference` (null: none); empty on a failure
    const auto synthesize = [&](const char *reference) -> std::vector<uint8_t> {
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        options.speaker_reference_path = reference;
        options.seed = 7;
        unsigned char *bytes = nullptr;
        size_t length = 0;
        if (jllama_tts_synthesize_options(tts, "Hi", &options, &bytes, &length) != JLLAMA_STATUS_OK) {
            ADD_FAILURE() << jllama_last_error();
            return {};
        }
        std::vector<uint8_t> wav(bytes, bytes + length);
        jllama_bytes_free(bytes);
        return wav;
    };
    const std::vector<uint8_t> api_own = synthesize(nullptr);
    const std::vector<uint8_t> api_low = synthesize(low.c_str());
    const std::vector<uint8_t> api_high = synthesize(high.c_str());
    ASSERT_FALSE(api_own.empty());
    EXPECT_EQ(api_low.size(), api_own.size());
    EXPECT_NE(api_low, api_own) << "the clip's rows are part of the prompt";
    EXPECT_NE(api_low, api_high) << "another clip, other rows";
    jllama_tts_free(tts);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    // the CPU alone: an RPC server an earlier test of the same process registered stays a device
    ggml_backend_dev_t no_devices[] = {nullptr};
    model_params.devices = no_devices;
    llama_model *model = llama_model_load_from_file(backbone.c_str(), model_params);
    ASSERT_NE(model, nullptr);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.embeddings = true;
    context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context *lctx = llama_init_from_model(model, context_params);
    ASSERT_NE(lctx, nullptr);
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.n_threads = 2;
    mtmd_params.print_timings = false;
    mtmd::context_ptr mctx(mtmd_init_from_file(mmproj.c_str(), model, mtmd_params));
    ASSERT_NE(mctx, nullptr);
    EXPECT_TRUE(mtmd_support_audio(mctx.get()));
    EXPECT_EQ(mtmd_get_audio_sample_rate(mctx.get()), 16000);
    // the PCM of "Hi" with seed 7, cloned from the clip at `reference` (null: none); empty on a failure
    const auto speak = [&](const std::string *reference) -> std::vector<float> {
        mtmd::bitmap_ptr ref;
        if (reference != nullptr) {
            ref.reset(
                mtmd_helper_bitmap_init_from_file(mctx.get(), reference->c_str(), false, mtmd_helper_init_opt_default())
                    .bitmap);
            if (!ref) {
                return {};
            }
        }
        mtmd_helper::gen_audio gen;
        gen.init(lctx, mctx.get());
        if (!gen.valid()) {
            return {};
        }
        mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
        inp.prompt = "Hi";
        inp.prompt_len = 2;
        inp.speaker_ref = ref.get();
        inp.seed = 7;
        inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        llama_memory_seq_rm(llama_get_memory(lctx), 0, -1, -1);
        if (gen.set_input(&inp) != 0) {
            return {};
        }
        int32_t remaining = 0;
        while ((remaining = gen.step_prompt(static_cast<int32_t>(llama_n_batch(lctx)))) > 0) {
        }
        if (remaining < 0) {
            return {};
        }
        const float *h = llama_get_embeddings_ith(lctx, -1);
        bool stop = false;
        while (!stop) {
            const float *next = nullptr;
            if (gen.step_gen(LLAMA_TOKEN_NULL, h, &next, &stop) != 0) {
                return {};
            }
            if (next == nullptr) {
                break;
            }
            h = next;
        }
        int32_t rate = 0;
        const char *data = nullptr;
        size_t n_bytes = 0;
        if (gen.get_output(&rate, &data, &n_bytes) != 0 || rate != 48000) {
            return {};
        }
        const auto *pcm = reinterpret_cast<const float *>(data);
        return std::vector<float>(pcm, pcm + n_bytes / sizeof(float));
    };
    const std::vector<float> own = speak(nullptr);
    const std::vector<float> cloned_low = speak(&low);
    const std::vector<float> cloned_low_again = speak(&low);
    const std::vector<float> cloned_high = speak(&high);
    // 4 frames each: the stop head of this mmproj fires at once, and the voice rows are prompt rows
    EXPECT_EQ(own.size(), 4u * 7680u);
    ASSERT_EQ(cloned_low.size(), own.size());
    ASSERT_EQ(cloned_high.size(), own.size());
    EXPECT_NE(cloned_low, own) << "the clip's rows are part of the prompt";
    EXPECT_EQ(cloned_low, cloned_low_again);
    EXPECT_NE(cloned_low, cloned_high) << "another clip, other rows";
    mctx.reset();
    llama_free(lctx);
    llama_model_free(model);
    for (const std::string &p : {backbone, mmproj, low, high}) {
        std::remove(p.c_str());
    }
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

namespace {

// The tensors of a GGUF file, name -> data.
std::map<std::string, std::vector<uint8_t>> gguf_tensor_bytes(const std::string &path) {
    std::map<std::string, std::vector<uint8_t>> tensors;
    ggml_context *data = nullptr;
    gguf_init_params params = {false, &data};
    gguf_context *g = gguf_init_from_file(path.c_str(), params);
    EXPECT_NE(g, nullptr) << path;
    if (g == nullptr) {
        return tensors;
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char *name = gguf_get_tensor_name(g, i);
        const ggml_tensor *t = ggml_get_tensor(data, name);
        const auto *bytes = static_cast<const uint8_t *>(t->data);
        tensors[name] = std::vector<uint8_t>(bytes, bytes + ggml_nbytes(t));
    }
    gguf_free(g);
    ggml_free(data);
    return tensors;
}

// A corpus of words over the printable ASCII the model knows.
std::string training_corpus(int sentences) {
    std::string corpus;
    for (int i = 0; i < sentences; ++i) {
        corpus += "the quick brown fox jumps over the lazy dog. ";
    }
    return corpus;
}

} // namespace

TEST_F(CApi, FinetuneChangesTheWeightsWithEitherOptimizerAndWithAValidationSplit) {
    // A run that writes a file is not yet a run that trained: the output's tensors are the input's,
    // and at least one of them moved -- with AdamW, with SGD (optimizer 1), and with a part of the
    // corpus held out for validation.
    const auto input = gguf_tensor_bytes(path_);
    ASSERT_FALSE(input.empty());
    struct run {
        const char *what;
        int optimizer;
        double learning_rate;
        double val_split;
    };
    for (const run &r :
         {run{"AdamW", 0, 1e-4, 0.0}, run{"SGD", 1, 1e-3, 0.0}, run{"AdamW, validation", 0, 1e-4, 0.25}}) {
        SCOPED_TRACE(r.what);
        const std::string out = path_ + ".trained.gguf";
        const json config = {{"model_path", path_},
                             {"training_text", training_corpus(24)},
                             {"output_path", out},
                             {"epochs", 1},
                             {"learning_rate", r.learning_rate},
                             {"optimizer", r.optimizer},
                             {"n_ctx", 64},
                             {"n_gpu_layers", 0},
                             {"val_split", r.val_split}};
        ASSERT_EQ(jllama_finetune(config.dump().c_str()), JLLAMA_STATUS_OK) << jllama_last_error();
        const auto output = gguf_tensor_bytes(out);
        ASSERT_EQ(output.size(), input.size());
        int changed = 0;
        for (const auto &tensor : input) {
            ASSERT_EQ(output.count(tensor.first), 1u) << tensor.first;
            ASSERT_EQ(output.at(tensor.first).size(), tensor.second.size()) << tensor.first;
            changed += output.at(tensor.first) != tensor.second ? 1 : 0;
        }
        EXPECT_GT(changed, 0) << "the trainer wrote the input's weights back";
        // and the trained model loads and answers
        jllama_model *trained = load_tiny(out, {});
        ASSERT_NE(trained, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(trained, 4).empty());
        jllama_model_free(trained);
        std::error_code ec;
        std::filesystem::remove(out, ec);
    }
}

TEST_F(CApi, FinetuneRefusesACorpusShorterThanOneTrainingWindowAndASplitWithoutTrainingData) {
    // llama.cpp's dataset cuts the corpus into windows of n_ctx + 1 tokens, one every n_ctx / 2
    // tokens (common_opt_dataset_init), and computes their count in size_t: a corpus shorter than a
    // window wrapped to a huge count, and one shorter than a window and a stride gave none, which
    // ggml_opt_dataset_init asserts on -- both ended the process. A validation split outside [0, 1), or
    // one that leaves no window to train on, ended it too (llama_opt_epoch asserts on the split) or
    // wrote the untouched model as a success. Each is refused by name, before anything is written.
    const std::string out = path_ + ".refused.gguf";
    std::error_code ec;
    std::filesystem::remove(out, ec);
    const auto finetune = [&out](const std::string &text, double val_split) {
        const json config = {{"model_path", path_}, {"training_text", text}, {"output_path", out},
                             {"epochs", 1},         {"learning_rate", 1e-4}, {"n_ctx", 64},
                             {"n_gpu_layers", 0},   {"val_split", val_split}};
        return jllama_finetune(config.dump().c_str());
    };
    for (const std::string &text : {std::string("abc"), std::string(300, 'a')}) {
        SCOPED_TRACE(text.size());
        EXPECT_EQ(finetune(text, 0.0), JLLAMA_STATUS_ERROR);
        EXPECT_NE(std::string(jllama_last_error()).find("too few tokens"), std::string::npos) << jllama_last_error();
        EXPECT_FALSE(std::filesystem::exists(out));
    }
    // the shortest corpus that trains is the one the refusal names: one window and one stride
    finetune("abc", 0.0);
    const std::string message = jllama_last_error();
    std::smatch need;
    ASSERT_TRUE(std::regex_search(message, need, std::regex("need at least ([0-9]+)"))) << message;
    const size_t minimum = std::stoul(need[1].str());
    // as many tokens as the trainer counts: the same vocabulary, with the special tokens
    const auto tokens_of = [](const std::string &text) {
        int *tokens = nullptr;
        size_t count = 0;
        EXPECT_EQ(jllama_tokenize(model_, text.c_str(), true, &tokens, &count), JLLAMA_STATUS_OK);
        jllama_tokens_free(tokens);
        return count;
    };
    const size_t specials = tokens_of("");
    const std::string exact(minimum - specials, 'a');
    ASSERT_EQ(tokens_of(exact), minimum);
    const std::string one_short(minimum - specials - 1, 'a');
    EXPECT_EQ(finetune(one_short, 0.0), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("too few tokens"), std::string::npos) << jllama_last_error();
    // one window: a split holds it out and leaves nothing to train on
    EXPECT_EQ(finetune(exact, 0.5), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("val_split"), std::string::npos) << jllama_last_error();
    EXPECT_FALSE(std::filesystem::exists(out));
    for (const double val_split : {-0.5, 1.0, 1.5}) {
        SCOPED_TRACE(val_split);
        EXPECT_EQ(finetune(training_corpus(24), val_split), JLLAMA_STATUS_ERROR);
        EXPECT_NE(std::string(jllama_last_error()).find("val_split"), std::string::npos) << jllama_last_error();
        EXPECT_FALSE(std::filesystem::exists(out));
    }
    // and the window itself trains
    ASSERT_EQ(finetune(exact, 0.0), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(std::filesystem::exists(out));
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

// The bits per weight of a GGUF file, as llama.cpp's target search counts them: the bytes of every
// tensor over its elements (metadata not counted), and the file type the file names.
struct gguf_bits {
    double bits_per_weight = 0.0;
    int file_type = -1;
    std::set<int> tensor_types;
};

gguf_bits read_gguf_bits(const std::string &file) {
    gguf_bits bits;
    ggml_context *meta = nullptr;
    gguf_init_params params = {true, &meta};
    gguf_context *g = gguf_init_from_file(file.c_str(), params);
    if (g == nullptr) {
        return bits;
    }
    double bytes = 0.0;
    double elements = 0.0;
    for (ggml_tensor *t = ggml_get_first_tensor(meta); t != nullptr; t = ggml_get_next_tensor(meta, t)) {
        bytes += static_cast<double>(ggml_nbytes(t));
        elements += static_cast<double>(ggml_nelements(t));
        bits.tensor_types.insert(static_cast<int>(t->type));
    }
    bits.bits_per_weight = elements > 0.0 ? 8.0 * bytes / elements : 0.0;
    const int64_t key = gguf_find_key(g, "general.file_type");
    if (key >= 0) {
        bits.file_type = static_cast<int>(gguf_get_val_u32(g, key));
    }
    gguf_free(g);
    ggml_free(meta);
    return bits;
}

TEST_F(CApiEnumSweep, ATargetOfBitsPerWeightIsReachedAndTheModelAnswers) {
    // llama.cpp b12669 (#15550): with a target the quantizer picks the type of every tensor itself.
    // The file names the type llama-quantize starts from (the first whose bits per weight are not
    // below the target), and the tensors land near the target -- 3, 4.5 and 6 bits per weight give three
    // different mixes, each just below its target.
    struct expected {
        double bpw;
        int file_type;
    };
    double previous = 0.0;
    for (const expected &target : {expected{3.0, 23}, expected{4.5, 15}, expected{6.0, 18}}) {
        SCOPED_TRACE("target " + std::to_string(target.bpw) + " bpw");
        const std::string out = path_ + ".target-bpw.gguf";
        ASSERT_EQ(jllama_quantize_target(path_.c_str(), out.c_str(), 1, false, imatrix_.c_str(), target.bpw, -1),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        const gguf_bits bits = read_gguf_bits(out);
        EXPECT_EQ(bits.file_type, target.file_type);
        EXPECT_GT(bits.bits_per_weight, previous) << "a larger target gives a larger file";
        // measured 2.98, 4.49 and 5.95: the search stays below the target, by less than a step
        EXPECT_LE(bits.bits_per_weight, target.bpw + 0.05);
        EXPECT_GE(bits.bits_per_weight, target.bpw - 0.5);
        previous = bits.bits_per_weight;
        jllama_model *model = load_tiny(out, {});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        Owned completion;
        EXPECT_EQ(jllama_completions(model, R"({"prompt":"Hello there","temperature":0,"n_predict":4})", &completion.s),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model_free(model);
        std::error_code ec;
        std::filesystem::remove(out, ec);
    }
}

TEST_F(CApiEnumSweep, ATargetSizeIsKeptAndTheFileNamesF16) {
    // A size target starts from F16, as llama-quantize does: without the target reaching the
    // quantizer the output would be F16 throughout, twice the size asked for.
    const std::string f16 = path_ + ".target-f16.gguf";
    ASSERT_EQ(jllama_quantize(path_.c_str(), f16.c_str(), 1, 1, false), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    const auto f16_size = static_cast<int64_t>(std::filesystem::file_size(f16, ec));
    ASSERT_GT(f16_size, 0);
    const int64_t target = f16_size / 3;
    const std::string out = path_ + ".target-size.gguf";
    ASSERT_EQ(jllama_quantize_target(path_.c_str(), out.c_str(), 1, false, imatrix_.c_str(), -1.0, target),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const auto size = static_cast<int64_t>(std::filesystem::file_size(out, ec));
    // measured 525408 bytes for a target of 526976 (F16 1580928)
    EXPECT_LE(size, target + target / 100) << "F16 was " << f16_size;
    EXPECT_GE(size, target - target / 5) << "F16 was " << f16_size;
    const gguf_bits bits = read_gguf_bits(out);
    EXPECT_EQ(bits.file_type, 1);
    EXPECT_GT(bits.tensor_types.size(), 1u) << "the search keeps F32 norms next to the quantized weights";
    jllama_model *model = load_tiny(out, {});
    EXPECT_NE(model, nullptr) << jllama_last_error();
    if (model != nullptr) {
        jllama_model_free(model);
    }
    std::filesystem::remove(out, ec);
    std::filesystem::remove(f16, ec);
}

TEST_F(CApiEnumSweep, ATargetThatCannotBeQuantizedToIsRefusedWithItsReason) {
    const std::string out = path_ + ".bad-target.gguf";
    struct refused {
        const char *imatrix;
        double bpw;
        int64_t size;
        const char *reason;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::string im = imatrix_;
    for (const refused &r :
         {refused{im.c_str(), 4.0, 1000, "exactly one target"}, refused{im.c_str(), -1.0, -1, "exactly one target"},
          refused{im.c_str(), 0.0, -1, "above 0 and at most 16"},
          refused{im.c_str(), 16.5, -1, "above 0 and at most 16"},
          refused{im.c_str(), nan, -1, "above 0 and at most 16"}, refused{im.c_str(), -1.0, 0, "above 0 bytes"},
          refused{nullptr, 4.0, -1, "needs an importance matrix"},
          refused{"", -1.0, 1000, "needs an importance matrix"}}) {
        SCOPED_TRACE(r.reason);
        EXPECT_EQ(jllama_quantize_target(path_.c_str(), out.c_str(), 1, false, r.imatrix, r.bpw, r.size),
                  JLLAMA_STATUS_INVALID_ARGUMENT);
        EXPECT_NE(std::string(jllama_last_error()).find(r.reason), std::string::npos) << jllama_last_error();
        std::error_code ec;
        EXPECT_FALSE(std::filesystem::exists(out, ec));
    }
    EXPECT_EQ(jllama_quantize_target(nullptr, out.c_str(), 1, false, imatrix_.c_str(), 4.0, -1),
              JLLAMA_STATUS_INVALID_ARGUMENT);
    // a matrix that cannot be read is the quantizer's error, as for a file type
    EXPECT_EQ(jllama_quantize_target(path_.c_str(), out.c_str(), 1, false, "/nonexistent/imatrix.gguf", 4.0, -1),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("importance matrix"), std::string::npos) << jllama_last_error();
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

TEST(CApiSleep, AWakeDoesNotCallTheLoadProgressCallbackAgain) {
    // The server keeps the caller's load-progress callback in its params and runs load_model() again
    // when it wakes from the idle sleep -- on the worker thread, long after jllama_model_load returned
    // and the caller's user data is gone (a stack object of LlamaModel's constructor, an FFM upcall
    // stub whose arena is closed); patches/0002 keeps a callback that is not the server's own. The
    // engine forwards to the host during the first load only.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sleep-progress");
    ASSERT_FALSE(path.empty());
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
    std::atomic<int> calls{0};
    const auto progress = [](float, void *user) -> bool {
        static_cast<std::atomic<int> *>(user)->fetch_add(1);
        return true;
    };
    const std::vector<std::string> args = {
        "jllama", "-m", path, "-c", "64", "-ngl", "0", "--no-warmup", "-t", "2", "--sleep-idle-seconds", "1"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), progress, &calls, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const int at_load = calls.load();
    EXPECT_GT(at_load, 0) << "the first load reports its progress";
    const auto slept = [&lines]() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(lines.mutex);
                if (lines.sleeps > 0) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    };
    ASSERT_TRUE(slept()) << "the model did not sleep";
    EXPECT_FALSE(greedy_answer(model, 4).empty()) << "the sleeping model did not wake";
    EXPECT_EQ(calls.load(), at_load) << "the wake reported its reload to the first load's callback";
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

namespace {

// Counts the server's "entering sleeping state" lines through the log callback, which it holds
// for its own lifetime.
class SleepWatch {
  public:
    SleepWatch() {
        EXPECT_EQ(jllama_set_log_callback(
                      [](int, const char *text, void *user) {
                          auto *self = static_cast<SleepWatch *>(user);
                          if (std::string(text).find("server is entering sleeping state") != std::string::npos) {
                              std::lock_guard<std::mutex> lock(self->mutex_);
                              ++self->sleeps_;
                          }
                      },
                      this),
                  JLLAMA_STATUS_OK);
    }
    ~SleepWatch() { jllama_set_log_callback(nullptr, nullptr); }
    SleepWatch(const SleepWatch &) = delete;
    SleepWatch &operator=(const SleepWatch &) = delete;

    // Waits for one more sleep than the last call saw; false after 20 s without one.
    bool next_sleep() {
        const int wanted = ++expected_;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (sleeps_ >= wanted) {
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    }

  private:
    std::mutex mutex_;
    int sleeps_ = 0;
    int expected_ = 0;
};

} // namespace

TEST(CApiSleep, EveryOperationAsTheFirstCallAfterASleepLetsTheModelSleepAgain) {
    // The other operations of the C API, each as the first call after a sleep: it wakes the model,
    // answers, and gives its wake back (jllama_hold_server_awake), so the idle timer puts the model
    // to sleep again. A call that kept its count kept the weights in memory for good, silently. The
    // model has FIM tokens, for the infill, and a slot directory, for the slot save and restore (each
    // after a sleep, so the slot saved and restored is the empty one a wake leaves).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sleep-every", true);
    ASSERT_FALSE(path.empty());
    const std::filesystem::path slots =
        std::filesystem::temp_directory_path() / ("jllama-c-api-sleep-every-" + std::to_string(std::random_device{}()));
    ASSERT_TRUE(std::filesystem::create_directories(slots));
    SleepWatch watch;
    jllama_model *model =
        load_tiny(path, {"-c", "128", "--sleep-idle-seconds", "1", "--slot-save-path", slots.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::vector<std::pair<const char *, std::function<void()>>> calls = {
        {"jllama_metrics",
         [model]() {
             Owned metrics;
             EXPECT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
         }},
        {"jllama_lora_adapters",
         [model]() {
             Owned adapters;
             EXPECT_EQ(jllama_lora_adapters(model, &adapters.s), JLLAMA_STATUS_OK) << jllama_last_error();
         }},
        {"slot erase",
         [model]() {
             Owned erased;
             EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
        {"a stream received to its end",
         [model]() {
             int task = -1;
             ASSERT_EQ(
                 jllama_completions_stream_request(model, R"({"prompt":"Hi","n_predict":3,"temperature":0})", &task),
                 JLLAMA_STATUS_OK)
                 << jllama_last_error();
             EXPECT_TRUE(receive_to_stop(model, task).at("stop").get<bool>());
         }},
        {"jllama_detokenize",
         [model]() {
             const int ids[] = {'H', 'i'};
             Owned text;
             size_t length = 0;
             EXPECT_EQ(jllama_detokenize(model, ids, 2, &text.s, &length), JLLAMA_STATUS_OK) << jllama_last_error();
         }},
        {"jllama_apply_template",
         [model]() {
             Owned prompt;
             EXPECT_EQ(jllama_apply_template(model, R"({"messages":[{"role":"user","content":"Hi"}]})", &prompt.s),
                       JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
        {"jllama_infill",
         [model]() {
             Owned infill;
             EXPECT_EQ(jllama_infill(model, INFILL_REQUEST, &infill.s), JLLAMA_STATUS_OK) << jllama_last_error();
         }},
        {"jllama_configure",
         [model]() {
             EXPECT_EQ(jllama_configure(model, R"({"n_threads":2,"slot_prompt_similarity":0.5})"), JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
        {"slot save",
         [model]() {
             Owned saved;
             EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "slot0.bin", &saved.s), JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
        {"slot restore",
         [model]() {
             Owned restored;
             EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "slot0.bin", &restored.s), JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
        {"a chat stream received to its end",
         [model]() {
             int task = -1;
             ASSERT_EQ(
                 jllama_chat_completions_stream_request(
                     model, R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0})", &task),
                 JLLAMA_STATUS_OK)
                 << jllama_last_error();
             bool stop = false;
             for (int i = 0; i < 50 && !stop; ++i) {
                 Owned envelope;
                 ASSERT_EQ(jllama_chat_completions_stream_receive(model, task, &envelope.s, &stop), JLLAMA_STATUS_OK)
                     << jllama_last_error();
             }
             EXPECT_TRUE(stop);
         }},
        {"jllama_completions_oai",
         [model]() {
             Owned completion;
             EXPECT_EQ(
                 jllama_completions_oai(model, R"({"prompt":"Hi","max_tokens":2,"temperature":0})", &completion.s),
                 JLLAMA_STATUS_OK)
                 << jllama_last_error();
         }},
    };
    // the load itself is followed by a sleep
    ASSERT_TRUE(watch.next_sleep()) << "the model did not sleep after its load";
    for (const auto &call : calls) {
        SCOPED_TRACE(call.first);
        call.second();
        EXPECT_TRUE(watch.next_sleep()) << "the model did not sleep after " << call.first;
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove_all(slots, ignored);
}

TEST(CApiSleep, AnEncoderSleepsAgainAfterItsEmbeddingsAndAfterARerank) {
    // The same for the operations of an encoder: embeddings (both entry points) and a rerank, each
    // the first call after a sleep.
    const std::string encoder = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-sleep");
    const std::string reranker = write_tiny_bert_gguf(true, 4 /* LLAMA_POOLING_TYPE_RANK */, "jllama-bert-sleep-rank");
    ASSERT_FALSE(encoder.empty());
    ASSERT_FALSE(reranker.empty());
    SleepWatch watch;
    {
        jllama_model *model = load_tiny(encoder, {"--embedding", "--sleep-idle-seconds", "1"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        ASSERT_TRUE(watch.next_sleep()) << "the encoder did not sleep after its load";
        Owned embeddings;
        EXPECT_EQ(jllama_embeddings(model, R"({"content":"hello there"})", false, &embeddings.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_TRUE(watch.next_sleep()) << "the encoder did not sleep after jllama_embeddings";
        float *values = nullptr;
        size_t count = 0;
        EXPECT_EQ(jllama_embed(model, "hello there", &values, &count), JLLAMA_STATUS_OK) << jllama_last_error();
        jllama_floats_free(values);
        EXPECT_TRUE(watch.next_sleep()) << "the encoder did not sleep after jllama_embed";
        jllama_model_free(model);
    }
    {
        jllama_model *model = load_tiny(reranker, {"--reranking", "--sleep-idle-seconds", "1"});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        ASSERT_TRUE(watch.next_sleep()) << "the reranker did not sleep after its load";
        const char *documents[] = {"berlin", "the cat sat on the mat"};
        Owned ranked;
        EXPECT_EQ(jllama_rerank(model, "the capital", 2, documents, &ranked.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_TRUE(watch.next_sleep()) << "the reranker did not sleep after jllama_rerank";
        jllama_model_free(model);
    }
    std::error_code ignored;
    std::filesystem::remove(encoder, ignored);
    std::filesystem::remove(reranker, ignored);
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
        // jllama_embed (LlamaModel.embed) answers one vector: the pooled one, or with --pooling none
        // the first token's row
        float *values = nullptr;
        size_t count = 0;
        ASSERT_EQ(jllama_embed(model, text.c_str(), &values, &count), JLLAMA_STATUS_OK) << jllama_last_error();
        ASSERT_FALSE(by_pooling[pooling].empty());
        const json first = json::parse(by_pooling[pooling].front());
        ASSERT_EQ(count, first.size());
        for (size_t j = 0; j < count; ++j) {
            EXPECT_FLOAT_EQ(values[j], first.at(j).get<float>()) << "column " << j;
        }
        jllama_floats_free(values);
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

TEST(CApiBert, AnInputLongerThanTheUbatchIsRefusedAndTheEncoderGoesOn) {
    // An encoder reads its whole input in one ubatch. The server refuses a longer one before it
    // decodes ("input ... is too large to process"); below it llama.cpp asserted until b12525
    // (#32553 made that an error too), so this refusal is what stands between a long text and the
    // end of the process. The model answers the next, shorter input.
    const std::string path = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-ubatch");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--embedding", "-b", "8", "-ub", "8"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned refused;
    const std::string long_text = json{{"content", "hello there world hello there world hello there world"}}.dump();
    EXPECT_NE(jllama_embeddings(model, long_text.c_str(), false, &refused.s), JLLAMA_STATUS_OK);
    EXPECT_NE(std::string(jllama_last_error()).find("too large to process"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(embedding_rows(model, "hello there world", 32).size(), 1u);
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiBert, TheOpenAiEmbeddingsAnswerKeepsTheOrderOfItsInputsAndDecodesFromBase64) {
    // jllama_embeddings(..., oai_compat = true) is /v1/embeddings: an OpenAI "list" of one
    // "embedding" object per input, in the order of the inputs -- with two slots the inputs are
    // computed side by side, and the answers are placed back by their index -- the model's name
    // unless the request names one, and with "encoding_format": "base64" each vector as the
    // little-endian float32 bytes of the same numbers.
    const std::string path = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-oai");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--embedding", "-np", "2"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const auto oai = [model](const json &request) {
        Owned response;
        EXPECT_EQ(jllama_embeddings(model, request.dump().c_str(), true, &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()) : json();
    };
    Owned meta;
    ASSERT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string name = json::parse(meta.str()).at("name").get<std::string>();

    const std::vector<std::string> inputs = {"hello there", "the cat sat on the mat", "world"};
    std::vector<std::vector<float>> alone;
    for (const std::string &input : inputs) {
        const json single = oai({{"input", input}});
        ASSERT_EQ(single.at("data").size(), 1u) << single.dump();
        alone.push_back(single.at("data").at(0).at("embedding").get<std::vector<float>>());
        ASSERT_EQ(alone.back().size(), 32u);
    }
    const json all = oai({{"input", inputs}});
    EXPECT_EQ(all.at("object").get<std::string>(), "list") << all.dump();
    EXPECT_EQ(all.at("model").get<std::string>(), name) << all.dump();
    EXPECT_GT(all.at("usage").at("prompt_tokens").get<int>(), 0) << all.dump();
    EXPECT_EQ(all.at("usage").at("total_tokens"), all.at("usage").at("prompt_tokens"));
    ASSERT_EQ(all.at("data").size(), inputs.size()) << all.dump();
    for (size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(inputs[i]);
        const json &item = all.at("data").at(i);
        EXPECT_EQ(item.at("object").get<std::string>(), "embedding");
        EXPECT_EQ(item.at("index").get<size_t>(), i);
        const std::vector<float> vector = item.at("embedding").get<std::vector<float>>();
        ASSERT_EQ(vector.size(), alone[i].size());
        // computed in another batch than alone: equal up to the rounding of the batch, and nearer
        // to its own input's vector than to any other
        for (size_t j = 0; j < vector.size(); ++j) {
            EXPECT_NEAR(vector[j], alone[i][j], 1e-4f) << "column " << j;
        }
    }
    EXPECT_NE(alone[0], alone[1]);
    EXPECT_NE(alone[1], alone[2]);
    EXPECT_EQ(oai({{"input", "world"}, {"model", "my-name"}}).at("model").get<std::string>(), "my-name");

    const json encoded = oai({{"input", inputs}, {"encoding_format", "base64"}});
    ASSERT_EQ(encoded.at("data").size(), inputs.size()) << encoded.dump();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const json &item = encoded.at("data").at(i);
        EXPECT_EQ(item.at("encoding_format").get<std::string>(), "base64");
        const std::vector<uint8_t> bytes = base64_bytes(item.at("embedding").get<std::string>());
        ASSERT_EQ(bytes.size(), 32u * sizeof(float));
        std::vector<float> decoded(32);
        std::memcpy(decoded.data(), bytes.data(), bytes.size());
        EXPECT_EQ(decoded, all.at("data").at(i).at("embedding").get<std::vector<float>>()) << inputs[i];
    }
    Owned refused;
    EXPECT_EQ(jllama_embeddings(model, R"({"input":"x","encoding_format":"hex"})", true, &refused.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    jllama_model_free(model);

    // one row per token has no OpenAI shape
    jllama_model *unpooled = load_tiny(path, {"--embedding", "--pooling", "none"});
    ASSERT_NE(unpooled, nullptr) << jllama_last_error();
    EXPECT_EQ(jllama_embeddings(unpooled, R"({"input":"hello there"})", true, &refused.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("Pooling type 'none' is not OAI compatible"), std::string::npos)
        << jllama_last_error();
    EXPECT_EQ(embedding_rows(unpooled, "hello there", 32).size(), 4u) << "the native shape still answers";
    jllama_model_free(unpooled);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(CApiBert, ARerankOfNoDocumentsAnswersAnEmptyList) {
    const std::string path = write_tiny_bert_gguf(true, 4 /* LLAMA_POOLING_TYPE_RANK */, "jllama-bert-rerank-none");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--reranking"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    Owned none;
    ASSERT_EQ(jllama_rerank(model, "the query", 0, nullptr, &none.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(none.str()), json::array()) << none.str();
    Owned refused;
    EXPECT_EQ(jllama_rerank(model, "the query", -1, nullptr, &refused.s), JLLAMA_STATUS_INVALID_ARGUMENT);
    // and the model ranks afterwards
    const char *documents[] = {"berlin"};
    Owned one;
    ASSERT_EQ(jllama_rerank(model, "the query", 1, documents, &one.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(json::parse(one.str()).size(), 1u) << one.str();
    jllama_model_free(model);
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

TEST(CApiRpc, TheImportanceMatrixAndTheTrainerRunAfterAnRpcServerWentAway) {
    // jllama_imatrix and jllama_finetune load a model of their own, without an argv for prepare_argv
    // to read, so they keep a stale RPC device out by exclude_stale_devices: an RPC server a load
    // registered stays a device for the life of the process, and llama.cpp's default device list
    // puts it first. Once the server is gone, a model placed on it fails or ends the process.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-rpc-then-imatrix");
    ASSERT_FALSE(path.empty());
    int port = 0;
    {
        std::unique_ptr<rpc_serve_run> server;
        for (int attempt = 0; attempt < 50 && port == 0; ++attempt) {
            const int candidate = rpc_test_port(attempt);
            server.reset(new rpc_serve_run(candidate));
            for (int i = 0;
                 i < 500 && !jllama_rpc_server_listening() && !server->returned_within(std::chrono::milliseconds(0));
                 ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (jllama_rpc_server_listening()) {
                port = candidate;
            } else {
                server.reset();
            }
        }
        ASSERT_NE(port, 0) << "no RPC server could bind";
        jllama_model *model = load_tiny(path, {"--rpc", "127.0.0.1:" + std::to_string(port)});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        jllama_model_free(model);
    } // the server stops here; its device stays registered

    const std::string text = write_calibration_text("jllama-rpc-then-imatrix", 100);
    const std::string imatrix = path + ".imatrix.gguf";
    int chunks = -1;
    EXPECT_EQ(jllama_imatrix(path.c_str(), text.c_str(), imatrix.c_str(), 32, 0, 2, 0, nullptr, nullptr, &chunks),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_GT(chunks, 0);

    std::string corpus;
    for (int i = 0; i < 16; ++i) {
        corpus += "the quick brown fox jumps over the lazy dog. ";
    }
    const std::string out = path + ".finetuned.gguf";
    const json config = {{"model_path", path}, {"training_text", corpus}, {"output_path", out},
                         {"epochs", 1},        {"learning_rate", 1e-4},   {"n_ctx", 64},
                         {"n_gpu_layers", 0},  {"val_split", 0.0}};
    EXPECT_EQ(jllama_finetune(config.dump().c_str()), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    EXPECT_GT(std::filesystem::file_size(out, ec), 0u);
    std::filesystem::remove(out, ec);
    std::filesystem::remove(imatrix, ec);
    std::filesystem::remove(text, ec);
    std::filesystem::remove(path, ec);
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

TEST(CApiMtp, ADraftWithAnMtpBlockIsReadAsDraftMtpWithoutASpecType) {
    // The load runs upstream's models handler (common_models_handler_apply), which reads a draft's type
    // from its file: a draft with an MTP block is draft-mtp. Until 2026-11-19 the load skipped it --
    // the server example's parse leaves it to llama_server() -- and the engine used every draft without
    // a type as draft-simple, which it logs. Without a type the draft must draft exactly as with
    // --spec-type draft-mtp (greedy, so the same tokens are drafted and accepted).
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::mtp("jllama-mtp-as-draft"));
    ASSERT_FALSE(path.empty());
    struct run {
        std::string answer;
        int drafted = -1;
        int accepted = -1;
        bool simple = false;
    };
    std::map<std::string, run> runs;
    for (const char *type : {"", "draft-mtp", "draft-simple"}) {
        SCOPED_TRACE(std::string("--spec-type ") + type);
        std::vector<std::string> extra = {"--spec-draft-model", path, "--spec-draft-ngl", "0"};
        if (*type != 0) {
            extra.insert(extra.end(), {"--spec-type", type});
        }
        struct Sink {
            std::mutex m;
            std::string text;
        } sink;
        ASSERT_EQ(jllama_set_log_callback(
                      [](int, const char *text, void *ud) {
                          auto *s = static_cast<Sink *>(ud);
                          std::lock_guard<std::mutex> lk(s->m);
                          s->text += text;
                      },
                      &sink),
                  JLLAMA_STATUS_OK);
        jllama_model *model = load_tiny(path, extra);
        ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK); // flushes
        ASSERT_NE(model, nullptr) << jllama_last_error();
        run &r = runs[type];
        r.simple = sink.text.find("using it as draft-simple") != std::string::npos;
        r.answer = greedy_answer(model, 48);
        Owned metrics;
        ASSERT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
        const json m = json::parse(metrics.str());
        r.drafted = m.at("n_draft_tokens_total").get<int>();
        r.accepted = m.at("n_draft_accepted_total").get<int>();
        jllama_model_free(model);
    }
    EXPECT_FALSE(runs[""].simple) << "the draft's MTP block was not read";
    EXPECT_GT(runs["draft-mtp"].drafted, 0);
    EXPECT_EQ(runs[""].drafted, runs["draft-mtp"].drafted);
    EXPECT_EQ(runs[""].accepted, runs["draft-mtp"].accepted);
    EXPECT_EQ(runs[""].answer, runs["draft-mtp"].answer);
    EXPECT_EQ(runs["draft-simple"].answer, runs["draft-mtp"].answer);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiDownload, AModelUrlIsDownloadedToItsPathAndLoaded) {
    // --model-url (ModelParameters.setModelUrl) with -m as the target: upstream's models handler
    // downloads it before the load. The load skipped the handler until 2026-11-19 and failed on a path
    // nobody had written. A loopback HTTP server serves the generator's model; a URL it answers with 404
    // fails the load by the download's own message, and nothing is left at the target.
    const std::string source = write_tiny_llama_gguf("jllama-c-api-download-source");
    ASSERT_FALSE(source.empty());
    std::string bytes;
    {
        std::ifstream in(source, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    ASSERT_GT(bytes.size(), 0u);
    jllama_model *reference = load_tiny(source, {});
    ASSERT_NE(reference, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(reference);
    jllama_model_free(reference);

    httplib::Server http;
    http.Get("/model.gguf", [&bytes](const httplib::Request &, httplib::Response &res) {
        res.set_content(bytes, "application/octet-stream");
    });
    const int port = http.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread serving([&http]() { http.listen_after_bind(); });
    struct stop_http {
        httplib::Server &http;
        std::thread &serving;
        ~stop_http() {
            http.stop();
            serving.join();
        }
    } stop{http, serving};
    http.wait_until_ready();

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-download-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string target = (dir / "downloaded.gguf").string();
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/model.gguf";
    jllama_model *downloaded = load_tiny(target, {"--model-url", url});
    ASSERT_NE(downloaded, nullptr) << jllama_last_error();
    EXPECT_EQ(std::filesystem::file_size(target), bytes.size());
    EXPECT_EQ(greedy_answer(downloaded), expected);
    jllama_model_free(downloaded);

    const std::string missing = (dir / "missing.gguf").string();
    EXPECT_EQ(load_tiny(missing, {"--model-url", "http://127.0.0.1:" + std::to_string(port) + "/no-such.gguf"}),
              nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("failed to fetch the model files"), std::string::npos)
        << jllama_last_error();
    EXPECT_FALSE(std::filesystem::exists(missing));

    std::error_code ec;
    std::filesystem::remove(source, ec);
    std::filesystem::remove_all(dir, ec);
}

namespace {
// a loopback HTTP server that serves `bytes` at /model.gguf and holds the body after its first half
// until release(): a download the test can watch while it runs
struct HeldModelServer {
    httplib::Server http;
    std::thread serving;
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    int port = 0;
    explicit HeldModelServer(const std::string &bytes) {
        http.Get("/model.gguf", [this, &bytes](const httplib::Request &, httplib::Response &res) {
            res.set_content_provider(bytes.size(), "application/octet-stream",
                                     [this, &bytes](size_t offset, size_t, httplib::DataSink &sink) {
                                         const size_t half = bytes.size() / 2;
                                         if (offset < half) {
                                             return sink.write(bytes.data() + offset, half - offset);
                                         }
                                         std::unique_lock<std::mutex> lock(mutex);
                                         cv.wait_for(lock, std::chrono::seconds(60), [this] { return released; });
                                         return sink.write(bytes.data() + offset, bytes.size() - offset);
                                     });
        });
        port = http.bind_to_any_port("127.0.0.1");
        serving = std::thread([this]() { http.listen_after_bind(); });
        http.wait_until_ready();
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
    ~HeldModelServer() {
        release();
        http.stop();
        serving.join();
    }
};

// the events of GET /models/sse, read on a thread of their own from the moment the server listens
// until it ends the stream
struct ModelsSseReader {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<json> events;
    bool ended = false;
    int status = 0;
    std::string content_type;
    std::thread reading;
    explicit ModelsSseReader(int port) {
        reading = std::thread([this, port]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            std::string pending;
            while (std::chrono::steady_clock::now() < deadline) {
                httplib::Client client("127.0.0.1", port);
                client.set_read_timeout(std::chrono::seconds(60));
                bool connected = false;
                const auto res = client.Get(
                    "/models/sse",
                    [&](const httplib::Response &r) {
                        connected = true;
                        std::lock_guard<std::mutex> lock(mutex);
                        status = r.status;
                        content_type = r.get_header_value("Content-Type");
                        cv.notify_all();
                        return true;
                    },
                    [&](const char *data, size_t length) {
                        pending.append(data, length);
                        size_t end;
                        while ((end = pending.find("\n\n")) != std::string::npos) {
                            const std::string line = pending.substr(0, end);
                            pending.erase(0, end + 2);
                            if (line.rfind("data: ", 0) == 0) {
                                std::lock_guard<std::mutex> lock(mutex);
                                events.push_back(json::parse(line.substr(6)));
                                cv.notify_all();
                            }
                        }
                        return true;
                    });
                if (connected) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::lock_guard<std::mutex> lock(mutex);
            ended = true;
            cv.notify_all();
        });
    }
    // whether the stream has answered, waiting for it up to `seconds`: the server registers a client
    // before it answers, so from then on every event reaches this reader
    bool await_connected(int seconds = 30) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::seconds(seconds), [&] { return status != 0 || ended; });
        return status != 0;
    }
    // the first event that matches, waiting for it up to `seconds`
    json await(const std::function<bool(const json &)> &match, int seconds = 30) {
        std::unique_lock<std::mutex> lock(mutex);
        json found;
        cv.wait_for(lock, std::chrono::seconds(seconds), [&] {
            for (const auto &e : events) {
                if (match(e)) {
                    found = e;
                    return true;
                }
            }
            return ended;
        });
        return found;
    }
    bool await_end(int seconds) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(seconds), [&] { return ended; });
    }
    std::vector<json> snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return events;
    }
    ~ModelsSseReader() {
        if (reading.joinable()) {
            reading.join();
        }
    }
};

int free_loopback_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    listener.join();
    return port;
}

bool is_status(const json &e, const char *status) {
    return e.value("event", std::string()) == "status_change" && e.contains("data") &&
           e.at("data").value("status", std::string()) == status;
}

// the position of the first event that matches, or the number of events
size_t index_of(const std::vector<json> &events, const std::function<bool(const json &)> &match) {
    for (size_t i = 0; i < events.size(); ++i) {
        if (match(events[i])) {
            return i;
        }
    }
    return events.size();
}
} // namespace

TEST(CApiServer, ASingleModelServerReportsItsDownloadAndLoadOverModelsSse) {
    // Since b12637 (#32398) a single-model server listens before it downloads and loads its model, and
    // serves GET /models/sse meanwhile: /health is 503 while the model is on its way, and the stream
    // carries the download's progress, its end, the load and the loaded model. The loopback server
    // holds the download at half the file, so the client sees it in progress whatever the timing.
    const std::string source = write_tiny_llama_gguf("jllama-c-api-sse-source");
    ASSERT_FALSE(source.empty());
    std::string bytes;
    {
        std::ifstream in(source, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    ASSERT_GT(bytes.size(), 2u);
    HeldModelServer model_server(bytes);
    ASSERT_GT(model_server.port, 0);
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-sse-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string target = (dir / "downloaded.gguf").string();
    const std::string url = "http://127.0.0.1:" + std::to_string(model_server.port) + "/model.gguf";
    const int port = free_loopback_port();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "-m",     target.c_str(),   "--model-url", url.c_str(),
                          "-c",           "64",     "-ngl",           "0",           "--host",
                          "127.0.0.1",    "--port", port_text.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(13, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    {
        ModelsSseReader sse(port);
        const json progress =
            sse.await([](const json &e) { return e.value("event", std::string()) == "download_progress"; });
        ASSERT_FALSE(progress.is_null()) << "no download progress before the model arrived";
        EXPECT_EQ(sse.status, 200);
        EXPECT_EQ(sse.content_type.rfind("text/event-stream", 0), 0u) << sse.content_type;
        EXPECT_EQ(progress.value("model", std::string()), target) << progress.dump();
        const json file = progress.at("data").at("progress").at(url);
        EXPECT_EQ(file.at("total").get<size_t>(), bytes.size()) << progress.dump();
        EXPECT_LE(file.at("done").get<size_t>(), bytes.size() / 2) << progress.dump();
        {
            httplib::Client client("127.0.0.1", port);
            const auto health = client.Get("/health");
            ASSERT_TRUE(health) << httplib::to_string(health.error());
            EXPECT_EQ(health->status, 503) << "the server is ready before its model arrived: " << health->body;
        }
        model_server.release();
        const json loaded = sse.await([](const json &e) { return is_status(e, "loaded"); });
        ASSERT_FALSE(loaded.is_null()) << "the model was not reported loaded";
        const auto events = sse.snapshot();
        const size_t finished =
            index_of(events, [](const json &e) { return e.value("event", std::string()) == "download_finished"; });
        const size_t loading = index_of(events, [](const json &e) { return is_status(e, "loading"); });
        const size_t done = index_of(events, [](const json &e) { return is_status(e, "loaded"); });
        EXPECT_LT(finished, loading) << json(events).dump();
        EXPECT_LT(loading, done) << json(events).dump();
        EXPECT_EQ(index_of(events, [](const json &e) { return e.value("event", std::string()) == "download_failed"; }),
                  events.size());
        EXPECT_TRUE(loaded.at("data").contains("info")) << loaded.dump();
        EXPECT_EQ(std::filesystem::file_size(target), bytes.size());
        httplib::Client client("127.0.0.1", port);
        const auto health = client.Get("/health");
        ASSERT_TRUE(health) << httplib::to_string(health.error());
        EXPECT_EQ(health->status, 200) << health->body;
        // a client that comes after the load gets the last event replayed: the loaded model, by the
        // id GET /models lists
        {
            ModelsSseReader late(port);
            const json replayed = late.await([](const json &) { return true; });
            ASSERT_FALSE(replayed.is_null());
            EXPECT_TRUE(is_status(replayed, "loaded")) << replayed.dump();
            const auto models = client.Get("/models");
            ASSERT_TRUE(models) << httplib::to_string(models.error());
            const std::string id = json::parse(models->body).at("data").at(0).at("id").get<std::string>();
            EXPECT_EQ(replayed.value("model", std::string()), id);
            EXPECT_EQ(replayed.at("data").at("info").value("id", std::string()), id) << replayed.dump();
            // the stop ends both streams: the SSE clients do not hold it (the shutdown handler of
            // patches/0006 stops them, as upstream's does)
            auto stopped = std::make_shared<std::atomic<int>>(-1);
            std::thread([server, stopped]() { stopped->store(jllama_server_stop(server)); }).detach();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (stopped->load() == -1 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            ASSERT_EQ(stopped->load(), JLLAMA_STATUS_OK) << "the stop waited for the SSE clients";
            EXPECT_TRUE(late.await_end(10)) << "the late SSE stream outlived the server";
        }
        EXPECT_TRUE(sse.await_end(10)) << "the SSE stream outlived the server";
    }
    std::error_code ec;
    std::filesystem::remove(source, ec);
    std::filesystem::remove_all(dir, ec);
}

TEST(CApiServer, ASingleModelServerWhoseModelFailsToLoadSaysSoOverModelsSseAndEnds) {
    // The same flow over bytes that are no GGUF: the download finishes, the load fails, and the
    // stream's last event is the model unloaded with the exit code before the server ends -- what
    // upstream's README promises for a failed download or load. The server no longer runs afterwards.
    // The load fails within a millisecond, and upstream stopped the HTTP server right after the
    // notification, before the stream had sent it: the event was lost whenever the stream had not
    // asked for its next chunk yet (macOS CI, run 45073054816; deterministic with a 30 ms pause after
    // every chunk). patches/0046 lets the SSE clients send it and end first.
    const std::string bytes(4096, 'x');
    HeldModelServer model_server(bytes);
    ASSERT_GT(model_server.port, 0);
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-sse-bad-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string target = (dir / "downloaded.gguf").string();
    const std::string url = "http://127.0.0.1:" + std::to_string(model_server.port) + "/model.gguf";
    const int port = free_loopback_port();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "-m",     target.c_str(),   "--model-url", url.c_str(),
                          "-c",           "64",     "-ngl",           "0",           "--host",
                          "127.0.0.1",    "--port", port_text.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(13, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    {
        ModelsSseReader sse(port);
        ASSERT_FALSE(
            sse.await([](const json &e) { return e.value("event", std::string()) == "download_progress"; }).is_null());
        model_server.release();
        const json unloaded = sse.await([](const json &e) { return is_status(e, "unloaded"); });
        ASSERT_FALSE(unloaded.is_null()) << json(sse.snapshot()).dump();
        EXPECT_EQ(unloaded.at("data").value("exit_code", 0), 1) << unloaded.dump();
        const auto events = sse.snapshot();
        EXPECT_EQ(index_of(events, [](const json &e) { return is_status(e, "loaded"); }), events.size());
        EXPECT_LT(index_of(events, [](const json &e) { return is_status(e, "loading"); }),
                  index_of(events, [](const json &e) { return is_status(e, "unloaded"); }));
        EXPECT_TRUE(sse.await_end(15)) << "the SSE stream outlived the failed server";
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (jllama_server_is_running(server) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_FALSE(jllama_server_is_running(server)) << "a server whose model failed to load still runs";
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(CApiServer, ASingleModelServerWhoseDownloadFailsSaysSoOverModelsSseAndEnds) {
    // The other failure path of a single-model server: the model URL answers 404, held until the SSE
    // client is connected. The stream reports the failed download and then the model unloaded with
    // the exit code, and ends; the server no longer runs. The download fails at once, so without
    // patches/0046 the two events raced the stop of the HTTP server as the failed load's do.
    struct HeldNotFound {
        httplib::Server http;
        std::thread serving;
        std::mutex mutex;
        std::condition_variable cv;
        bool released = false;
        int port = 0;
        HeldNotFound() {
            http.Get("/model.gguf", [this](const httplib::Request &, httplib::Response &res) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait_for(lock, std::chrono::seconds(60), [this] { return released; });
                res.status = 404;
            });
            port = http.bind_to_any_port("127.0.0.1");
            serving = std::thread([this]() { http.listen_after_bind(); });
            http.wait_until_ready();
        }
        void release() {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
            cv.notify_all();
        }
        ~HeldNotFound() {
            release();
            http.stop();
            serving.join();
        }
    } model_server;
    const int model_port = model_server.port;
    ASSERT_GT(model_port, 0);
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-sse-404-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string target = (dir / "downloaded.gguf").string();
    const std::string url = "http://127.0.0.1:" + std::to_string(model_port) + "/model.gguf";
    const int port = free_loopback_port();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "-m",     target.c_str(),   "--model-url", url.c_str(),
                          "-c",           "64",     "-ngl",           "0",           "--host",
                          "127.0.0.1",    "--port", port_text.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(13, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    {
        ModelsSseReader sse(port);
        ASSERT_TRUE(sse.await_connected()) << "the SSE client did not connect";
        EXPECT_EQ(sse.status, 200);
        model_server.release();
        const json unloaded = sse.await([](const json &e) { return is_status(e, "unloaded"); });
        ASSERT_FALSE(unloaded.is_null()) << json(sse.snapshot()).dump();
        EXPECT_EQ(unloaded.at("data").value("exit_code", 0), 1) << unloaded.dump();
        EXPECT_TRUE(sse.await_end(15)) << "the SSE stream outlived the failed server";
        const auto events = sse.snapshot();
        const size_t failed =
            index_of(events, [](const json &e) { return e.value("event", std::string()) == "download_failed"; });
        EXPECT_LT(failed, index_of(events, [](const json &e) { return is_status(e, "unloaded"); }))
            << json(events).dump();
        EXPECT_EQ(index_of(events, [](const json &e) { return is_status(e, "loading"); }), events.size())
            << json(events).dump();
        EXPECT_TRUE(is_status(events.back(), "unloaded")) << json(events).dump();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (jllama_server_is_running(server) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_FALSE(jllama_server_is_running(server)) << "a server whose download failed still runs";
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

#ifndef _WIN32
TEST(CApiServer, ARouterCancelsTheLoadOfAModelWhoseWorkerHasNotComeUpAndCanLoadItAgain) {
    // b12652 (#32225) gives the WebUI a button that cancels a model's load: it posts /models/unload
    // while the model is loading, and waits on /models/sse for the model to go down. This is the router's
    // side of it in the embedded server: the worker -- here a script that never comes up, set through
    // jllama_server_set_worker_command (patches/0008) -- is force-killed, the stream reports the model
    // unloaded, the worker process is gone, and a second load starts a new worker, which the server's
    // stop ends too. No model is loaded: the file in --models-dir only gives the router a model to list.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-cancel-" + std::to_string(std::random_device{}()));
    const std::filesystem::path models = dir / "models";
    std::filesystem::create_directories(models);
    const std::string generated = write_tiny_llama_gguf("jllama-c-api-cancel");
    std::filesystem::copy_file(generated, models / "tiny.gguf");
    std::filesystem::remove(generated);
    const std::filesystem::path pids = dir / "pids";
    const std::filesystem::path worker = dir / "worker.sh";
    {
        std::ofstream script(worker, std::ios::binary);
        script << "#!/bin/sh\necho $$ >> '" << pids.string() << "'\nexec sleep 60\n";
    }
    std::filesystem::permissions(worker, std::filesystem::perms::owner_all);
    ASSERT_EQ(jllama_server_set_worker_command(worker.string().c_str()), JLLAMA_STATUS_OK);
    // the worker command is the process's; a later test in this process must not spawn the script
    struct ResetWorkerCommand {
        const std::filesystem::path &dir;
        ~ResetWorkerCommand() {
            jllama_server_set_worker_command("");
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    } reset_worker_command{dir};

    // the pids the workers wrote, waiting until there are `count`
    const auto worker_pids = [&](size_t count) {
        std::vector<pid_t> found;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            found.clear();
            std::ifstream in(pids);
            for (long pid; in >> pid;) {
                found.push_back(static_cast<pid_t>(pid));
            }
            if (found.size() >= count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return found;
    };
    const auto gone = [](pid_t pid, int seconds) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (std::chrono::steady_clock::now() < deadline) {
            if (kill(pid, 0) != 0 && errno == ESRCH) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return false;
    };

    const int port = free_loopback_port();
    const std::string port_text = std::to_string(port);
    const std::string models_dir = models.string();
    const char *argv[] = {"llama-server",    "--host",       "127.0.0.1",       "--port",
                          port_text.c_str(), "--models-dir", models_dir.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(7, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    {
        ModelsSseReader sse(port);
        // on every way out of this block, a failed assertion included, the server stops before the
        // reader waits for the end of its stream (and a worker left behind ends with its sleep)
        struct StopOnExit {
            jllama_server *&server;
            ~StopOnExit() {
                if (server != nullptr) {
                    jllama_server_stop(server);
                }
            }
        } stop_on_exit{server};
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(std::chrono::seconds(30));
        // the model's entry of GET /models, once the router answers
        const auto entry = [&]() {
            const auto res = client.Get("/models");
            if (!res || res->status != 200) {
                return json();
            }
            const json listed = json::parse(res->body); // the loop must not run over a temporary's member
            for (const auto &m : listed.at("data")) {
                if (m.value("id", std::string()) == "tiny") {
                    return json(m);
                }
            }
            return json();
        };
        const auto listed_by = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (entry().is_null()) {
            ASSERT_LT(std::chrono::steady_clock::now(), listed_by) << "the router never listed the model";
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto post = [&](const char *path) {
            const auto res = client.Post(path, R"({"model":"tiny"})", "application/json");
            return res ? res->status : -1;
        };
        // the router reports a load it starts as model_status and a worker's exit as status_change; the
        // WebUI reads both
        const auto status_of = [](const json &e, const char *status) {
            const std::string event = e.value("event", std::string());
            return e.value("model", std::string()) == "tiny" && (event == "model_status" || event == "status_change") &&
                   e.contains("data") && e.at("data").value("status", std::string()) == status;
        };

        // the router replays no event, so the stream must be open before the load: a load posted earlier
        // was missed by a reader that connected a moment later (the macOS 15 job of run 45157679242)
        ASSERT_TRUE(sse.await_connected()) << "the SSE client never connected";
        ASSERT_EQ(post("/models/load"), 200);
        ASSERT_FALSE(sse.await([&](const json &e) { return status_of(e, "loading"); }).is_null())
            << json(sse.snapshot()).dump();
        const std::vector<pid_t> first = worker_pids(1);
        ASSERT_EQ(first.size(), 1u) << "the router started no worker";
        EXPECT_EQ(kill(first[0], 0), 0) << "the worker is not running";
        EXPECT_EQ(entry().at("status").value("value", std::string()), "loading") << entry().dump();

        // at once: a worker that is not loading is asked to exit and killed only after its stop timeout,
        // 10 s by default
        ASSERT_EQ(post("/models/unload"), 200);
        const json unloaded = sse.await([&](const json &e) { return status_of(e, "unloaded"); }, 5);
        ASSERT_FALSE(unloaded.is_null()) << json(sse.snapshot()).dump();
        EXPECT_TRUE(gone(first[0], 5)) << "the cancelled worker still runs";
        const json after = entry();
        EXPECT_EQ(after.at("status").value("value", std::string()), "unloaded") << after.dump();
        EXPECT_EQ(post("/models/unload"), 400) << "a model that is not running cannot be unloaded";

        // the cancel leaves the model loadable
        ASSERT_EQ(post("/models/load"), 200);
        const std::vector<pid_t> both = worker_pids(2);
        ASSERT_EQ(both.size(), 2u) << "the second load started no worker";
        const pid_t second = both[1];
        EXPECT_NE(second, first[0]);
        const auto loading_again_by = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (entry().at("status").value("value", std::string()) != "loading") {
            ASSERT_LT(std::chrono::steady_clock::now(), loading_again_by) << entry().dump();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        EXPECT_EQ(kill(second, 0), 0) << "the second worker is not running";

        EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
        server = nullptr;
        EXPECT_TRUE(gone(second, 30)) << "the server's stop left the loading worker running";
        EXPECT_TRUE(sse.await_end(15)) << "the SSE stream outlived the server";
    }
}
#endif // _WIN32
#endif // JLLAMA_HAS_NATIVE_SERVER

// ---------------------------------------------------------------------------------------------------
// EAGLE-3 (--spec-type draft-eagle3): a draft of one decoder layer that drafts from the inputs of three
// layers of the target, which the target's context extracts for it (llama_set_embeddings_layer_inp) --
// so the target's graph has to name those inputs (t_layer_inp). write_tiny_eagle3_gguf writes the draft.
// ---------------------------------------------------------------------------------------------------

namespace {

// The target written from `target` answers the same greedily alone, with the EAGLE-3 draft loaded and
// unused (--spec-type none) and drafting with it (--spec-type draft-eagle3, or no --spec-type, which
// the engine reads from the draft's architecture), and only those two draft.
// The draft reads the inputs of layers [2, 1, 0] of a target of three layers; with own_head it has a
// head over a smaller vocabulary mapped by d2t, without one it takes the target's.
void expect_eagle3_drafts_and_answers_the_same(tiny_model_spec target, bool own_head) {
    target.n_layer = 3;
    const std::string path = write_tiny_llama_gguf(target);
    const std::string draft = write_tiny_eagle3_gguf(target, own_head, target.prefix + "-eagle3");
    ASSERT_FALSE(path.empty());
    ASSERT_FALSE(draft.empty());
    std::map<std::string, std::string> answers;
    std::map<std::string, int> drafted;
    // "auto": the draft without --spec-type, which the engine reads as draft-eagle3 from its file
    for (const char *type : {"", "none", "draft-eagle3", "auto"}) {
        SCOPED_TRACE(*type != 0 ? std::string("--spec-type ") + type : std::string("without the draft"));
        std::vector<std::string> extra;
        if (*type != 0) {
            extra = {"--spec-draft-model", draft, "--spec-draft-ngl", "0"};
        }
        if (*type != 0 && std::string(type) != "auto") {
            extra.insert(extra.end(), {"--spec-type", type});
        }
        struct Sink {
            std::mutex m;
            std::string text;
        } sink;
        ASSERT_EQ(jllama_set_log_callback(
                      [](int, const char *text, void *ud) {
                          auto *s = static_cast<Sink *>(ud);
                          std::lock_guard<std::mutex> lk(s->m);
                          s->text += text;
                      },
                      &sink),
                  JLLAMA_STATUS_OK);
        jllama_model *model = load_tiny(path, extra);
        ASSERT_EQ(jllama_set_log_callback(nullptr, nullptr), JLLAMA_STATUS_OK); // flushes
        ASSERT_NE(model, nullptr) << jllama_last_error();
        if (std::string(type) == "auto") {
            EXPECT_NE(sink.text.find("without --spec-type: using it as draft-eagle3"), std::string::npos)
                << "an EAGLE-3 draft run as draft-simple reads layer inputs nobody wrote";
        }
        // a fresh model evaluates the whole prompt: nothing is compared across a prompt-cache hit
        answers[type] = greedy_answer(model, 48);
        drafted[type] = drafted_tokens(model);
        jllama_model_free(model);
    }
    EXPECT_FALSE(answers[""].empty());
    // the target checks every draft, so a draft changes the speed and never the answer
    EXPECT_EQ(answers["draft-eagle3"], answers[""]);
    EXPECT_EQ(answers["auto"], answers[""]);
    EXPECT_EQ(answers["none"], answers[""]);
    EXPECT_GT(drafted["draft-eagle3"], 0) << "the EAGLE-3 draft drafted nothing";
    EXPECT_GT(drafted["auto"], 0) << "the EAGLE-3 draft drafted nothing without --spec-type";
    EXPECT_EQ(drafted["none"], 0);
    EXPECT_EQ(drafted[""], 0);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(draft, ec);
}

} // namespace

TEST(CApiEagle3, ADraftOverTheLlamaDecoderDraftsAndAnswersTheSameWithout) {
    // The layout of upstream's Llama 3.1 EAGLE-3 conversions: the draft's own head over a draft
    // vocabulary smaller than the target's, which d2t maps to the target's ids, and the target's token
    // embedding.
    tiny_model_spec target;
    target.prefix = "jllama-eagle3-llama";
    expect_eagle3_drafts_and_answers_the_same(target, true);
}

TEST(CApiEagle3, ADraftOverAQwen3VlTargetGetsItsLayerInputs) {
    // llama.cpp #32589 (b12542): qwen3vl's graph did not name its layer inputs (res->t_layer_inp), so
    // the first target batch with the layers an EAGLE-3 draft reads turned on ended the process in
    // llm_graph_result::set_outputs ("layer input tensor is null"). The draft here borrows the
    // target's head and token embedding as well, the other path of the EAGLE-3 decoder.
    expect_eagle3_drafts_and_answers_the_same(tiny_model_spec::qwen3vl("jllama-eagle3-qwen3vl"), false);
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

TEST(CApiTokenizers, AnInkling2VocabularyTakesAWholeWordItHasAsOneTokenWithoutMerges) {
    // b12641 (#32799) gives the "inkling2" pre-tokenizer ignore_merges, as Inkling 2's tokenizer.json
    // says: a pre-tokenized word that is a token of the vocabulary is that token, whether or not a merge
    // builds it. Here no merge does -- the bytes, then "hello" and "Ġworld" -- so "gpt-4o", the same
    // regex without the flag, spells both words out byte by byte, as inkling2 did before.
    const auto load_with_pre = [](const char *pre, const char *prefix) {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.pre_tokenizer = pre;
        spec.unmerged_tokens = {"hello", "\xC4\xA0world"}; // Ġ: GPT-2's byte-level space
        return write_tiny_llama_gguf(spec);
    };
    const std::string inkling_path = load_with_pre("inkling2", "jllama-bpe-inkling2");
    const std::string gpt4o_path = load_with_pre("gpt-4o", "jllama-bpe-gpt4o");
    jllama_model *inkling = load_tiny(inkling_path, {});
    jllama_model *gpt4o = load_tiny(gpt4o_path, {});
    ASSERT_NE(inkling, nullptr) << jllama_last_error();
    ASSERT_NE(gpt4o, nullptr) << jllama_last_error();

    const int hello = 256;
    const int world = 257;
    const std::string text = "hello world";
    EXPECT_EQ(token_ids(inkling, text), (std::vector<int>{hello, world}));
    EXPECT_EQ(token_ids(gpt4o, text).size(), text.size()) << "without the flag a word no merge builds is bytes";
    // a word the vocabulary does not have stays bytes; " there" is one pre-tokenized word of six bytes
    EXPECT_EQ(token_ids(inkling, "hello there"), (std::vector<int>{hello, ' ', 't', 'h', 'e', 'r', 'e'}));
    // "helloworld" is one word, not "hello" + "world": the whole word is looked up, not its parts
    EXPECT_EQ(token_ids(inkling, "helloworld").size(), 10u);
    for (jllama_model *model : {inkling, gpt4o}) {
        for (const std::string &round_trip : {text, std::string("hello there"), std::string("Hello, World!")}) {
            EXPECT_EQ(detokenized(model, token_ids(model, round_trip)), round_trip);
        }
        EXPECT_FALSE(greedy_answer(model, 4).empty());
    }
    jllama_model_free(inkling);
    jllama_model_free(gpt4o);
    std::error_code ec;
    std::filesystem::remove(inkling_path, ec);
    std::filesystem::remove(gpt4o_path, ec);
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
    // the byte piece throws the map's "unordered_map::at", which the engine names). An error, not <unk>
    // and not a crash; a converted SentencePiece model always has the byte pieces.
    const std::string gap = "has no token for, and the vocabulary has no byte tokens";
    {
        int *tokens = nullptr;
        size_t count = 0;
        EXPECT_NE(jllama_tokenize(without, e_acute.c_str(), false, &tokens, &count), JLLAMA_STATUS_OK);
        EXPECT_EQ(tokens, nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find(gap), std::string::npos) << jllama_last_error();
        char *response = nullptr;
        EXPECT_NE(jllama_tokenize_json(without, e_acute.c_str(), false, true, &response), JLLAMA_STATUS_OK);
        EXPECT_EQ(response, nullptr);
        EXPECT_NE(std::string(jllama_last_error()).find(gap), std::string::npos) << jllama_last_error();
    }
    // a completion of such a text fails too, and the model answers the next request
    {
        const json request = {{"prompt", "caf" + e_acute}, {"n_predict", 2}, {"temperature", 0}};
        Owned response;
        EXPECT_NE(jllama_completions(without, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK);
        EXPECT_NE(std::string(jllama_last_error()).find(gap), std::string::npos) << jllama_last_error();
        static_cast<void>(greedy_answer(without, 2)); // asserts the status
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

// patches/0042 on the request paths: a response_format (chat) and a json_schema (raw completion) whose
// bounds no value meets are the caller's fault, named, and the model answers the next request. Without
// the patch the chat answers were "[99996]" and "\"\"" -- one item for minItems 2 and the empty string for
// minLength 3 --, two schemas failed as "failed to parse grammar", and the integer from 6 to 5 failed the
// generation as a server error ("Unexpected empty grammar stack").
TEST_F(CApiStructured, ARequestWithASchemaNoValueSatisfiesIsRefusedAndTheModelGoesOn) {
    const char *const schemas[] = {
        R"({"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":2})",
        R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":1})",
        R"({"type":"string","minLength":5,"maxLength":2})",
        R"({"type":"string","minLength":3,"maxLength":0})",
        R"({"type":"integer","minimum":6,"maximum":5})",
    };
    for (const char *schema : schemas) {
        Owned chat;
        const std::string chat_request =
            std::string(R"({"messages":[{"role":"user","content":"Answer."}],"temperature":0,"max_tokens":24,)"
                        R"("response_format":{"type":"json_schema","schema":)") +
            schema + "}}";
        EXPECT_EQ(jllama_chat_completions(model_, chat_request.c_str(), &chat.s), JLLAMA_STATUS_INVALID_REQUEST)
            << schema << ": " << (chat.s != nullptr ? chat.str() : std::string());
        EXPECT_NE(std::string(jllama_last_error()).find("JSON schema error"), std::string::npos) << jllama_last_error();

        Owned raw;
        const std::string raw_request =
            std::string(R"({"prompt":"Answer.","temperature":0,"n_predict":8,"json_schema":)") + schema + "}";
        EXPECT_EQ(jllama_completions(model_, raw_request.c_str(), &raw.s), JLLAMA_STATUS_INVALID_REQUEST)
            << schema << ": " << (raw.s != nullptr ? raw.str() : std::string());
        EXPECT_NE(std::string(jllama_last_error()).find("JSON schema error"), std::string::npos) << jllama_last_error();
    }
    // the bounds that meet still constrain the answer, and the model answers after the refusals
    const std::string two =
        chat_content(model_, R"(,"response_format":{"type":"json_schema","schema":)"
                             R"({"type":"array","items":{"enum":[1,2]},"minItems":2,"maxItems":2}})");
    EXPECT_TRUE(std::regex_match(two, std::regex(R"(\[\s*[12],\s*[12]\s*\])"))) << two;
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

TEST(CApiNonAsciiPaths, TheProcessReadsANarrowPathInACodePageOtherThanUtf8) {
    // The tests of this suite are the guards of patches/0024-0030 only where a narrow path is read in
    // the active code page and that is not UTF-8: a process whose code page is UTF-8 (an application
    // manifest, the "Beta: Use Unicode UTF-8" system setting) reads every narrow path as UTF-8, and the
    // tests pass without the patches -- as they do outside Windows, where a path is bytes. java.exe
    // carries no such manifest, so the JVM reads in the ANSI code page, and so must this binary.
#ifdef _WIN32
    EXPECT_NE(GetACP(), static_cast<UINT>(CP_UTF8))
        << "the active code page is UTF-8: CApiNonAsciiPaths passes here with or without patches 0024-0030";
#else
    GTEST_SKIP() << "a narrow path is bytes outside Windows; the code page matters on Windows only";
#endif
}

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

namespace {
// Whether a log line starts with common_log's timestamp ("0.01.234.567 ").
bool starts_with_timestamp(const std::string &line) {
    static const std::regex stamp(R"(^\d+\.\d{2}\.\d{3}\.\d{3} )");
    return std::regex_search(line, stamp);
}

// The lines of a drained log file, and how many of them start with a timestamp.
std::pair<size_t, size_t> lines_and_stamped(const std::filesystem::path &file) {
    drained_size(file);
    std::ifstream in(file);
    size_t lines = 0;
    size_t stamped = 0;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) {
            lines++;
            stamped += starts_with_timestamp(line) ? 1 : 0;
        }
    }
    return {lines, stamped};
}
} // namespace

TEST(CApiLogFile, TheLogPrefixAndTimestampOptionsOfTheArgvTakeEffect) {
    // common_init() turns common_log's prefix and timestamps on, and the engine called it after the
    // argv was parsed -- so --no-log-prefix / --no-log-timestamps (ModelParameters' log options) had no
    // effect on the console and the --log-file. llama_server() calls it before its parse; so do the load
    // and the embedded server now.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-format");
    ASSERT_FALSE(model.empty());
    const auto plain = log_file_path("jllama-c-api-log-format-plain");
    const auto stamped = log_file_path("jllama-c-api-log-format-stamped");
    std::error_code ec;
    std::filesystem::remove(plain, ec);
    std::filesystem::remove(stamped, ec);
    jllama_model *without = load_tiny(model, {"--log-file", plain.string(), "--no-log-timestamps", "--no-log-prefix"});
    ASSERT_NE(without, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(without).empty());
    jllama_model_free(without);
    const auto [plain_lines, plain_stamped] = lines_and_stamped(plain);
    EXPECT_GT(plain_lines, 0u);
    EXPECT_EQ(plain_stamped, 0u) << "--no-log-timestamps did not take effect";
    // without the options the load has upstream's default, timestamps on
    jllama_model *with = load_tiny(model, {"--log-file", stamped.string()});
    ASSERT_NE(with, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(with).empty());
    jllama_model_free(with);
    const auto [stamped_lines, stamped_count] = lines_and_stamped(stamped);
    EXPECT_GT(stamped_count, 0u) << stamped_lines << " lines, none with a timestamp";
#ifdef JLLAMA_HAS_NATIVE_SERVER
    // the embedded server: router mode over an empty directory needs no model
    const auto server_log = log_file_path("jllama-c-api-log-format-server");
    std::filesystem::remove(server_log, ec);
    const std::filesystem::path models_dir =
        std::filesystem::temp_directory_path() /
        ("jllama-c-api-log-format-models-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(models_dir);
    const std::vector<std::string> args = {
        "llama-server", "--models-dir",      models_dir.string(),  "--host", "127.0.0.1", "--port", "0",
        "--log-file",   server_log.string(), "--no-log-timestamps"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(static_cast<int>(argv.size()), argv.data(), &server), JLLAMA_STATUS_OK)
        << jllama_last_error();
    for (int i = 0; i < 100 && drained_size(server_log) == 0; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    const auto [server_lines, server_stamped] = lines_and_stamped(server_log);
    EXPECT_GT(server_lines, 0u);
    EXPECT_EQ(server_stamped, 0u) << "--no-log-timestamps did not take effect on the embedded server";
    std::filesystem::remove(server_log, ec);
    std::filesystem::remove_all(models_dir, ec);
#endif
    std::filesystem::remove(plain, ec);
    std::filesystem::remove(stamped, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiLogFile, AVocabOnlyModelOwnsItsLogFileAndAMissingFileIsRefused) {
    // A --vocab-only load takes its own path through the engine (no server context): it must own the
    // --log-file it opened like a full load and close it at its free, and a file that is not there
    // is refused like any load.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-vocab");
    ASSERT_FALSE(model.empty());
    jllama_model *missing = load_tiny(model + ".missing.gguf", {"--vocab-only"});
    EXPECT_EQ(missing, nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("could not load tokenizer"), std::string::npos)
        << jllama_last_error();
    jllama_model_free(missing);

    const auto log = log_file_path("jllama-c-api-log-file-vocab");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    jllama_model *vocab = load_tiny(model, {"--vocab-only", "--log-file", log.string()});
    ASSERT_NE(vocab, nullptr) << jllama_last_error();
    int *tokens = nullptr;
    size_t count = 0;
    EXPECT_EQ(jllama_tokenize(vocab, "Hello", false, &tokens, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    jllama_tokens_free(tokens);
    jllama_model_free(vocab);
    const auto written = drained_size(log);
    EXPECT_GT(written, 0u) << "the vocab-only load logged nothing into its file";

    jllama_model *later = load_tiny(model, {});
    ASSERT_NE(later, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(later).empty());
    jllama_model_free(later);
    EXPECT_EQ(drained_size(log), written) << "a later model wrote into the file of a freed vocab-only one";

    // a missing file with --log-file closes the file it opened at once
    const auto failed_log = log_file_path("jllama-c-api-log-file-vocab-failed");
    std::filesystem::remove(failed_log, ec);
    jllama_model *failed = load_tiny(model + ".missing.gguf", {"--vocab-only", "--log-file", failed_log.string()});
    EXPECT_EQ(failed, nullptr);
    const auto failed_written = drained_size(failed_log);
    jllama_model *after = load_tiny(model, {});
    ASSERT_NE(after, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(after).empty());
    jllama_model_free(after);
    EXPECT_EQ(drained_size(failed_log), failed_written) << "a failed vocab-only load left its log file open";
    EXPECT_TRUE(std::filesystem::remove(log, ec)) << ec.message();
    std::filesystem::remove(failed_log, ec);
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

TEST(CApiLogFile, ARefusedParseNeitherOpensNorClosesALogFile) {
    // The owner of the log file used to be decided from the argv's text: a load whose parse refused
    // an option before it reached --log-file still closed the process-wide file -- another model's.
    // The engine takes --log-file out of the argv and opens it only for an accepted parse. Upstream
    // has no "--log-file=<path>" spelling; the parse refuses it like any unknown option. A batch of no
    // tokens and rank pooling on a model without a classification head are refused after an accepted
    // parse, and the file was opened before those checks once.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-refused");
    ASSERT_FALSE(model.empty());
    const auto log_a = log_file_path("jllama-c-api-log-file-refused-a");
    const auto log_b = log_file_path("jllama-c-api-log-file-refused-b");
    std::error_code ec;
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(log_b, ec);
    jllama_model *a = load_tiny(model, {"--log-file", log_a.string()});
    ASSERT_NE(a, nullptr) << jllama_last_error();
    for (const std::vector<std::string> refused :
         {std::vector<std::string>{"--no-such-option", "--log-file", log_b.string()},
          std::vector<std::string>{"--log-file=" + log_b.string()},
          std::vector<std::string>{"-b", "0", "--log-file", log_b.string()},
          std::vector<std::string>{"--embedding", "--pooling", "rank", "--log-file", log_b.string()}}) {
        EXPECT_EQ(load_tiny(model, refused), nullptr) << refused.front();
        const auto before = drained_size(log_a);
        EXPECT_FALSE(greedy_answer(a).empty());
        EXPECT_GT(drained_size(log_a), before) << "a refused parse closed the log file of a live model";
        EXPECT_FALSE(std::filesystem::exists(log_b, ec)) << "a refused parse opened its log file";
    }
    jllama_model_free(a);
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(log_b, ec);
    std::filesystem::remove(model, ec);
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiLogFile, ARefusedServerArgvNeitherOpensNorClosesALogFile) {
    // The embedded server parses its argv inside llama_server(); its --log-file was opened before that
    // parse, so a refused argv closed a live model's file as the refused load once did. The start
    // parses the argv first now and opens nothing for a refused one.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-server-refused");
    ASSERT_FALSE(model.empty());
    const auto log_a = log_file_path("jllama-c-api-log-file-server-refused-a");
    const auto log_b = log_file_path("jllama-c-api-log-file-server-refused-b");
    std::error_code ec;
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(log_b, ec);
    jllama_model *a = load_tiny(model, {"--log-file", log_a.string()});
    ASSERT_NE(a, nullptr) << jllama_last_error();
    const std::string b_name = log_b.string();
    const char *argv[] = {"llama-server", "--no-such-option", "--log-file", b_name.c_str(), "--port", "0"};
    jllama_server *server = nullptr;
    EXPECT_EQ(jllama_server_start(6, argv, &server), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(server, nullptr);
    const auto before = drained_size(log_a);
    EXPECT_FALSE(greedy_answer(a).empty());
    EXPECT_GT(drained_size(log_a), before) << "a refused server argv closed the log file of a live model";
    EXPECT_FALSE(std::filesystem::exists(log_b, ec)) << "a refused server argv opened its log file";
    jllama_model_free(a);
    std::filesystem::remove(log_a, ec);
    std::filesystem::remove(model, ec);
}
#endif

TEST(CApiLogFile, LogDisableAfterTheLogFileStillDisablesTheLog) {
    // The engine opens --log-file after the parse, which applied --log-disable by pausing the log
    // worker; opening the file resumed it, so "--log-file f --log-disable" logged into f. It stays
    // paused, as upstream's order leaves it.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-disabled");
    ASSERT_FALSE(model.empty());
    const auto log = log_file_path("jllama-c-api-log-file-disabled");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    jllama_model *loaded = load_tiny(model, {"--log-file", log.string(), "--log-disable"});
    ASSERT_NE(loaded, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(loaded).empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // a running worker would have written by now
    const auto written = std::filesystem::file_size(log, ec);
    EXPECT_EQ(written, 0u) << "the disabled log was written into " << log.string();
    jllama_model_free(loaded);
    jllama_set_log_callback(nullptr, nullptr); // a swap resumes the worker for whatever runs next
    std::filesystem::remove(log, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiLogFile, TheFileOfLlamaArgLogFileIsOwnedLikeTheOption) {
    // LLAMA_ARG_LOG_FILE opens its file inside the parse, before the argv, so the engine cannot take it
    // out of the argv as it takes --log-file: the load whose parse opened it owns it -- its free closes
    // it, a load refused after its parse closes it at once, and so does a refused server argv.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-env");
    ASSERT_FALSE(model.empty());
    const auto log = log_file_path("jllama-c-api-log-file-env");
    const std::string log_name = log.string();
    std::error_code ec;
    std::filesystem::remove(log, ec);
    struct EnvReset {
        ~EnvReset() { set_env_utf8("LLAMA_ARG_LOG_FILE", nullptr); }
    } env_reset;
    const auto closed_after = [&](const char *what) {
        set_env_utf8("LLAMA_ARG_LOG_FILE", nullptr);
        const auto written = drained_size(log);
        jllama_model *later = load_tiny(model, {});
        ASSERT_NE(later, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(later).empty());
        jllama_model_free(later);
        EXPECT_EQ(drained_size(log), written) << what;
    };
    {
        SCOPED_TRACE("freed");
        set_env_utf8("LLAMA_ARG_LOG_FILE", log_name.c_str());
        jllama_model *owner = load_tiny(model, {});
        ASSERT_NE(owner, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(owner).empty());
        jllama_model_free(owner);
        EXPECT_GT(drained_size(log), 0u) << "the model logged nothing into the file of the variable";
        closed_after("the file of a freed model's variable stayed open");
    }
    {
        SCOPED_TRACE("refused after the parse");
        set_env_utf8("LLAMA_ARG_LOG_FILE", log_name.c_str());
        EXPECT_EQ(load_tiny(model, {"-b", "0"}), nullptr);
        closed_after("the file a refused load's parse opened stayed open");
    }
#ifdef JLLAMA_HAS_NATIVE_SERVER
    {
        SCOPED_TRACE("refused server argv");
        set_env_utf8("LLAMA_ARG_LOG_FILE", log_name.c_str());
        const char *argv[] = {"llama-server", "--no-such-option", "--host", "127.0.0.1", "--port", "0"};
        jllama_server *server = nullptr;
        EXPECT_EQ(jllama_server_start(6, argv, &server), JLLAMA_STATUS_ERROR);
        EXPECT_EQ(server, nullptr);
        closed_after("the file a refused server argv's parse opened stayed open");
    }
#endif
    std::filesystem::remove(log, ec);
    std::filesystem::remove(model, ec);
}

TEST(CApiLogFile, TheUnderscoreSpellingIsOwnedAndClosedLikeTheOther) {
    // The parse reads "--log_file" as "--log-file" (it maps '_' to '-' in a long option); a scan of
    // the argv that knew one spelling left the file open for the rest of the process.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-underscore");
    ASSERT_FALSE(model.empty());
    const auto log = log_file_path("jllama-c-api-log-file-underscore");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    jllama_model *first = load_tiny(model, {"--log_file", log.string()});
    ASSERT_NE(first, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(first).empty());
    jllama_model_free(first);
    const auto written = drained_size(log);
    EXPECT_GT(written, 0u) << "the model logged nothing into its file";
    jllama_model *second = load_tiny(model, {});
    ASSERT_NE(second, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(second).empty());
    jllama_model_free(second);
    EXPECT_EQ(drained_size(log), written) << "the file of a freed model stayed open";
    std::filesystem::remove(log, ec);
    std::filesystem::remove(model, ec);
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiLogFile, AStoppedServerClosesTheLogFileItsArgvOpened) {
    // An embedded server parses its argv inside llama_server() / llama_server_attach(), where a
    // --log-file opens the process-wide log file like a model load does -- and nothing closed it:
    // the file took every later line of the process until the JVM ended. The server owns it until it
    // returns, in both modes.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-log-file-server");
    ASSERT_FALSE(model.empty());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      ("jllama-c-api-log-file-models-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string models_dir = dir.string();
    std::error_code ec;
    {
        SCOPED_TRACE("router mode");
        const auto log = log_file_path("jllama-c-api-log-file-router");
        std::filesystem::remove(log, ec);
        const std::string log_name = log.string();
        const char *argv[] = {"llama-server", "--host",           "127.0.0.1",  "--port",        "0",
                              "--models-dir", models_dir.c_str(), "--log-file", log_name.c_str()};
        jllama_server *server = nullptr;
        ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        ASSERT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
        const auto written = drained_size(log);
        EXPECT_GT(written, 0u) << "the server logged nothing into its file";
        jllama_model *later = load_tiny(model, {});
        ASSERT_NE(later, nullptr) << jllama_last_error();
        EXPECT_FALSE(greedy_answer(later).empty());
        jllama_model_free(later);
        EXPECT_EQ(drained_size(log), written) << "a model loaded after the server wrote into the server's file";
        EXPECT_TRUE(std::filesystem::remove(log, ec)) << ec.message();
    }
    {
        SCOPED_TRACE("attached");
        const auto log = log_file_path("jllama-c-api-log-file-attached");
        std::filesystem::remove(log, ec);
        const std::string log_name = log.string();
        jllama_model *served = load_tiny(model, {});
        ASSERT_NE(served, nullptr) << jllama_last_error();
        const char *argv[] = {"llama-server", "--host", "127.0.0.1", "--port", "0", "--log-file", log_name.c_str()};
        jllama_server *server = nullptr;
        ASSERT_EQ(jllama_server_start_attached(served, 7, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        EXPECT_FALSE(greedy_answer(served).empty());
        ASSERT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
        const auto written = drained_size(log);
        EXPECT_GT(written, 0u) << "the attached server logged nothing into its file";
        EXPECT_FALSE(greedy_answer(served).empty());
        EXPECT_EQ(drained_size(log), written) << "the model kept writing into the stopped server's file";
        jllama_model_free(served);
        EXPECT_TRUE(std::filesystem::remove(log, ec)) << ec.message();
    }
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(model, ec);
}
#endif

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

namespace {

void write_text(const std::filesystem::path &path, const std::string &text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

// The content of a raw completion of "Hello", or "" when it failed.
std::string raw_content(jllama_model *model, int n_predict) {
    Owned completion;
    const std::string request =
        R"({"prompt":"Hello","temperature":0,"cache_prompt":false,"n_predict":)" + std::to_string(n_predict) + "}";
    EXPECT_EQ(jllama_completions(model, request.c_str(), &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
    return completion.s != nullptr ? json::parse(completion.str()).at("content").get<std::string>() : std::string();
}

} // namespace

TEST(CApiNonAsciiPaths, ArgumentFilesAreReadUnderNonAsciiNames) {
    // patches/0039: the files the server's options name were opened through narrow std::ifstreams --
    // read_file() behind --grammar-file and --chat-template-file, and the streams of
    // --json-schema-file and --api-key-file -- which MSVC reads in the active code page, so on code
    // page 1252 a name outside ASCII failed the load with "failed to open file". Each file is written
    // under both non-ASCII places and loaded through its option, and what it holds is shown to take
    // effect. Outside Windows a path is bytes and this passes with or without the patch; only a
    // Windows job can turn it red (see TheProcessReadsANarrowPathInACodePageOtherThanUtf8). The
    // model ends its answers with a control token of its own, so a grammar's answer is its text only.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-utf8-argfiles";
    spec.eos = 4;
    spec.eos_is_control = true;
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    // unconstrained, the random model writes neither the grammar's "a" nor the opening of an object
    std::string free_answer;
    {
        jllama_model *plain = load_tiny(model, {});
        ASSERT_NE(plain, nullptr) << jllama_last_error();
        free_answer = raw_content(plain, 4);
        jllama_model_free(plain);
    }
    ASSERT_FALSE(free_answer.empty());
    ASSERT_NE(free_answer, "a");
    ASSERT_NE(free_answer.front(), '{');
    for (const auto &place : non_ascii_places("jllama-c-api-utf8-argfiles")) {
        SCOPED_TRACE(place.what);
        const auto grammar = in_place(place, ".gbnf");
        const auto chat_template = in_place(place, ".jinja");
        const auto schema = in_place(place, ".schema.json");
        const auto keys = in_place(place, ".keys.txt");
        write_text(grammar, "root ::= \"a\"\n");
        write_text(chat_template, "[from the file]{{ messages[0].content }}");
        write_text(schema, R"({"type":"object"})");
        write_text(keys, "# one key\njllama-utf8-key\n");

        {
            SCOPED_TRACE("--grammar-file");
            jllama_model *loaded = load_tiny(model, {"--grammar-file", utf8(grammar)});
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
            EXPECT_EQ(raw_content(loaded, 4), "a") << "the grammar of the file is every request's";
            jllama_model_free(loaded);
        }
        {
            SCOPED_TRACE("--json-schema-file");
            jllama_model *loaded = load_tiny(model, {"--json-schema-file", utf8(schema)});
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
            const std::string answer = raw_content(loaded, 4);
            ASSERT_FALSE(answer.empty());
            EXPECT_EQ(answer.front(), '{') << answer;
            jllama_model_free(loaded);
        }
        {
            SCOPED_TRACE("--chat-template-file");
            jllama_model *loaded = load_tiny(model, {"--jinja", "--chat-template-file", utf8(chat_template)});
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
            Owned prompt;
            ASSERT_EQ(jllama_apply_template(loaded, R"({"messages":[{"role":"user","content":"Hi"}]})", &prompt.s),
                      JLLAMA_STATUS_OK)
                << jllama_last_error();
            EXPECT_NE(prompt.str().find("[from the file]Hi"), std::string::npos) << prompt.str();
            jllama_model_free(loaded);
        }
        {
            SCOPED_TRACE("--api-key-file");
            jllama_model *loaded = load_tiny(model, {"--api-key-file", utf8(keys)});
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
#ifdef JLLAMA_HAS_NATIVE_SERVER
            // the keys configure an HTTP frontend: the attach argv reads the same option, and the
            // server it starts asks for the key of the file
            {
                AttachedServer attached(loaded, {"--api-key-file", utf8(keys)});
                ASSERT_NE(attached.server, nullptr);
                httplib::Client client = attached.client();
                EXPECT_EQ(status_of(client.Get("/v1/models")), 401);
                EXPECT_EQ(
                    status_of(client.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer jllama-utf8-key"}})),
                    200);
            }
#endif
            jllama_model_free(loaded);
        }
        std::error_code ec;
        for (const auto &f : {grammar, chat_template, schema, keys}) {
            std::filesystem::remove(f, ec);
        }
        std::filesystem::remove_all(place.dir, ec);
    }
    std::error_code ignored;
    std::filesystem::remove(model, ignored);
}

TEST(CApiNonAsciiPaths, LookupCachesAreReadAndWrittenUnderNonAsciiNames) {
    // patches/0039: the n-gram lookup caches of --spec-type ngram-cache are read
    // (common_ngram_cache_load, behind -lcs and -lcd) and written (common_ngram_cache_save, the dynamic
    // one when the model is freed: beside the file and renamed over it) by their names, and the
    // dynamic one is looked for first (std::filesystem::exists) -- all narrow on Windows before, so a
    // cache outside ASCII was neither found nor written there. The dynamic cache is written under
    // both non-ASCII places and read back as a static one. Outside Windows a path is bytes and this
    // passes with or without the patch; only a Windows job can turn it red.
    const std::string model = write_tiny_llama_gguf("jllama-c-api-utf8-lookup");
    ASSERT_FALSE(model.empty());
    for (const auto &place : non_ascii_places("jllama-c-api-utf8-lookup")) {
        SCOPED_TRACE(place.what);
        const auto cache = in_place(place, ".lookup.bin");
        const auto cache_tmp = in_place(place, ".lookup.bin.tmp");
        {
            jllama_model *loaded = load_tiny(model, {"--spec-type", "ngram-cache", "-lcd", utf8(cache)});
            ASSERT_NE(loaded, nullptr) << jllama_last_error();
            EXPECT_FALSE(greedy_answer(loaded, 16).empty());
            jllama_model_free(loaded); // writes the dynamic cache
        }
        std::error_code ec;
        EXPECT_TRUE(std::filesystem::exists(cache, ec)) << "the dynamic cache is not under its name";
        EXPECT_GT(std::filesystem::file_size(cache, ec), 0u) << "the cache holds no n-gram";
        EXPECT_FALSE(std::filesystem::exists(cache_tmp, ec)) << "the save left its temporary file";

        // read back as the static cache: the speculative type starts with it
        const auto load_static = [&model](const std::filesystem::path &file) {
            LogCapture log;
            jllama_model *loaded = load_tiny(model, {"--spec-type", "ngram-cache", "-lcs", utf8(file)});
            EXPECT_NE(loaded, nullptr) << jllama_last_error();
            if (loaded != nullptr) {
                EXPECT_FALSE(greedy_answer(loaded, 8).empty());
                jllama_model_free(loaded);
            }
            log.stop();
            return log.text();
        };
        const std::string read = load_static(cache);
        EXPECT_EQ(read.find("failed to open static lookup cache"), std::string::npos) << read;
        EXPECT_EQ(read.find("failed to initialize speculative decoding"), std::string::npos) << read;
        // the capture sees the refusal it looks for: a static cache that is not there is reported
        const std::string missing = load_static(in_place(place, ".missing.bin"));
        EXPECT_NE(missing.find("failed to open static lookup cache"), std::string::npos) << missing;

        std::filesystem::remove(cache, ec);
        std::filesystem::remove_all(place.dir, ec);
    }
    std::error_code ignored;
    std::filesystem::remove(model, ignored);
}

// ---------------------------------------------------------------------------------------------------
// The fourth review round of 2026-11-19: what the reviewers found, each case red without its fix.

namespace {
// Sets (or, with nullptr, removes) an environment variable for the life of the object.
struct scoped_env {
    std::string name;
    scoped_env(std::string n, const char *value) : name(std::move(n)) { set(value); }
    ~scoped_env() { set(nullptr); }
    void set(const char *value) const {
#ifdef _WIN32
        _putenv_s(name.c_str(), value != nullptr ? value : "");
#else
        if (value != nullptr) {
            setenv(name.c_str(), value, 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }
};

std::filesystem::path fresh_dir(const std::string &prefix) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / (prefix + "-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    return dir;
}

void write_bytes(const std::filesystem::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
} // namespace

TEST(CApiDraft, ADraftThatIsNoModelFailsTheLoadAndTheProcessGoesOn) {
    // patches/0038: the models handler reads the speculative type of a draft given without --spec-type
    // from its GGUF, and llama.cpp read <arch>.block_count there without looking whether the key is
    // there -- a projector or a control vector named as the draft aborted the process (the JVM). The
    // load refuses such a draft now; a --vocab-only load, which loads no draft, loads.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-draft-no-model");
    const std::string mmproj = write_tiny_mmproj_gguf("jllama-c-api-draft-no-model-mmproj");
    const std::string cvec = write_tiny_control_vector_gguf("jllama-c-api-draft-no-model-cvec");
    ASSERT_FALSE(path.empty());
    for (const std::string &draft : {mmproj, cvec}) {
        SCOPED_TRACE(draft);
        EXPECT_EQ(load_tiny(path, {"-md", draft}), nullptr);
        EXPECT_NE(std::strlen(jllama_last_error()), 0u);
        jllama_model *vocab = load_tiny(path, {"-md", draft, "--vocab-only"});
        EXPECT_NE(vocab, nullptr) << jllama_last_error();
        jllama_model_free(vocab);
    }
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_FALSE(greedy_answer(model).empty());
    jllama_model_free(model);
    for (const std::string &p : {path, mmproj, cvec}) {
        std::remove(p.c_str());
    }
}

TEST(CApiNgramCache, ALookupCacheThatCannotBeReadDoesNotEndTheProcess) {
    // patches/0038: --spec-type ngram-cache with a static cache that cannot be read (-lcs), or a
    // dynamic one that exists and cannot be read (-lcd), was GGML_ABORT, and so was a cache cut short:
    // common_ngram_cache_load() asserted on every short read. The server now runs without the type and
    // answers. A cache cut inside an n-gram is refused too; an empty file is an empty cache.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-ngram-cache");
    ASSERT_FALSE(path.empty());
    jllama_model *reference = load_tiny(path, {});
    ASSERT_NE(reference, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(reference);
    jllama_model_free(reference);
    const std::filesystem::path dir = fresh_dir("jllama-c-api-ngram-cache");
    const auto missing = dir / "missing.bin";
    const auto cut = dir / "cut.bin";         // an n-gram of 16 bytes and two bytes of its count
    const auto partial = dir / "partial.bin"; // five bytes: cut inside the first n-gram
    const auto empty = dir / "empty.bin";
    write_bytes(cut, std::string(18, '\x01'));
    write_bytes(partial, std::string(5, '\x01'));
    write_bytes(empty, std::string());
    for (const char *option : {"-lcs", "-lcd"}) {
        for (const auto &file : {missing, cut, partial, empty}) {
            SCOPED_TRACE(std::string(option) + " " + file.filename().string());
            jllama_model *model = load_tiny(path, {"--spec-type", "ngram-cache", option, file.string()});
            ASSERT_NE(model, nullptr) << jllama_last_error();
            EXPECT_EQ(greedy_answer(model), expected);
            jllama_model_free(model);
        }
    }
    // the dynamic cache is written when the model is freed: beside the file and renamed over it, so a
    // cut save cannot leave a cut cache; the cache written is read by the next load
    const auto dynamic = dir / "dynamic.bin";
    for (int run = 0; run < 2; ++run) {
        jllama_model *model = load_tiny(path, {"--spec-type", "ngram-cache", "-lcd", dynamic.string()});
        ASSERT_NE(model, nullptr) << jllama_last_error();
        EXPECT_EQ(greedy_answer(model, 16).substr(0, expected.size()), expected);
        jllama_model_free(model);
        EXPECT_TRUE(std::filesystem::exists(dynamic)) << run;
        EXPECT_FALSE(std::filesystem::exists(dynamic.string() + ".tmp")) << run;
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(path, ec);
}

TEST(CApiControlVector, ADirectionOnTheMtpBlockIsRefused) {
    // llama.cpp loads a control vector for llama_model_n_layer() layers, the blocks without the
    // multi-token-prediction ones after them; the engine's refusal counted <arch>.block_count, so a
    // vector with a direction on the MTP block passed it, and upstream then dropped the vector (and
    // the LoRA adapters and the warmup after it) with a log line.
    const tiny_model_spec spec = tiny_model_spec::mtp("jllama-c-api-cvec-mtp");
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    const std::string fits = write_tiny_control_vector_gguf("jllama-c-api-cvec-mtp-fits", spec.n_layer, spec.n_embd);
    const std::string on_mtp =
        write_tiny_control_vector_gguf("jllama-c-api-cvec-mtp-deep", spec.n_layer + spec.n_layer_nextn, spec.n_embd);
    jllama_model *model = load_tiny(path, {"--control-vector", fits});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    jllama_model_free(model);
    EXPECT_EQ(load_tiny(path, {"--control-vector", on_mtp}), nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("cannot be applied to this model"), std::string::npos)
        << jllama_last_error();
    for (const std::string &p : {path, fits, on_mtp}) {
        std::remove(p.c_str());
    }
}

TEST(CApiTokenizers, AVocabOnlyModelRendersThePiecesAsTheFullModelDoes) {
    // jllama_tokenize_json with pieces on a --vocab-only model rendered a special token as "" (it asked
    // for the pieces without the special tokens) and cut a piece at 256 bytes; the full model renders
    // "<s>". Both answer the same JSON now.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-vocab-pieces";
    spec.tokenizer = tiny_model_spec::tokenizer_kind::SENTENCEPIECE;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    const auto pieces = [&](const std::vector<std::string> &extra) {
        jllama_model *model = load_tiny(path, extra);
        EXPECT_NE(model, nullptr) << jllama_last_error();
        Owned out;
        EXPECT_EQ(jllama_tokenize_json(model, "hello world", true, true, &out.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model_free(model);
        return out.str();
    };
    const std::string full = pieces({});
    EXPECT_NE(full.find("\"<s>\""), std::string::npos) << full;
    EXPECT_EQ(pieces({"--vocab-only"}), full);
    std::remove(path.c_str());
}

TEST(CApiEmbed, AnEmptyEmbeddingInputIsTheCallersMistake) {
    // jllama_embed("") on a vocabulary that adds no BOS posted an empty task, which the server answers as
    // a completion, and the call reported "embedding result is empty" -- a server error, where
    // jllama_embeddings and upstream's handler answer "Input content cannot be empty" (400).
    const std::string path = write_tiny_llama_gguf("jllama-c-api-embed-empty");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--embedding", "--pooling", "mean"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    float *values = nullptr;
    size_t count = 0;
    EXPECT_EQ(jllama_embed(model, "", &values, &count), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("cannot be empty"), std::string::npos) << jllama_last_error();
    Owned oai;
    EXPECT_EQ(jllama_embeddings(model, R"({"input":""})", false, &oai.s), JLLAMA_STATUS_INVALID_REQUEST);
    // and the model answers the next one
    ASSERT_EQ(jllama_embed(model, "hello", &values, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(count, 0u);
    jllama_floats_free(values);
    jllama_model_free(model);
    std::remove(path.c_str());
}

TEST_F(CApi, ALoraListThatIsNoArrayIsTheCallersMistake) {
    // upstream's POST /lora-adapters answers a body that is no array with 400; the engine threw a
    // server error for it
    Owned response;
    EXPECT_EQ(jllama_set_lora_adapters(model_, "{}", &response.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("JSON array"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(jllama_set_lora_adapters(model_, "[]", &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST_F(CApiFim, MalformedInfillFieldsAreRefusedAsUpstreamRefusesThem) {
    // upstream's /infill validates its input; the engine read it with json_value(), which turned a value
    // of another type into its default: a prompt that was no string was dropped, a chunk without a text
    // made the prompt builder throw a type error, and a filename of another type became "tmp"
    const std::pair<const char *, const char *> cases[] = {
        {R"({"input_prefix":"ab","input_suffix":"cd","prompt":5})", "\"prompt\" must be a string"},
        {R"({"input_prefix":"ab","input_suffix":"cd","input_extra":{}})", "\"input_extra\" must be an array"},
        {R"({"input_prefix":"ab","input_suffix":"cd","input_extra":[{"filename":"a"}]})", "\"text\" field"},
        {R"({"input_prefix":"ab","input_suffix":"cd","input_extra":["text"]})", "\"text\" field"},
        {R"({"input_prefix":"ab","input_suffix":"cd","input_extra":[{"text":"x","filename":3}]})",
         "\"filename\" field must be a string"},
    };
    for (const auto &[body, message] : cases) {
        SCOPED_TRACE(body);
        Owned infill;
        EXPECT_EQ(jllama_infill(model_, body, &infill.s), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find(message), std::string::npos) << jllama_last_error();
        Owned raw;
        EXPECT_EQ(jllama_completions(model_, body, &raw.s), JLLAMA_STATUS_INVALID_REQUEST);
    }
    Owned ok;
    EXPECT_EQ(jllama_infill(model_,
                            R"({"input_prefix":"ab","input_suffix":"cd","n_predict":1,)"
                            R"("input_extra":[{"text":"x","filename":"f.py"}]})",
                            &ok.s),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
}

TEST(CApiTrainer, AnOutputThatCannotBeWrittenAndAConfigItWouldChangeAreRefused) {
    // llama_model_save_to_file() returns nothing and ignores a failed write, so a run into a directory
    // that does not exist reported success and left no file; and the C API trained what
    // TrainingParameters refuses: 0 epochs as one, an unknown optimizer as AdamW.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-trainer-refusals");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir = fresh_dir("jllama-c-api-trainer-refusals");
    const auto config = [&](const std::string &out) {
        return json{{"model_path", path},    {"training_text", training_corpus(24)},
                    {"output_path", out},    {"epochs", 1},
                    {"learning_rate", 1e-4}, {"n_ctx", 64},
                    {"n_gpu_layers", 0},     {"val_split", 0.0}};
    };
    const auto refused = [](const json &cfg, const char *message) {
        EXPECT_EQ(jllama_finetune(cfg.dump().c_str()), JLLAMA_STATUS_ERROR) << cfg.dump();
        EXPECT_NE(std::string(jllama_last_error()).find(message), std::string::npos) << jllama_last_error();
    };
    const auto nowhere = dir / "no-such-dir" / "out.gguf";
    refused(config(nowhere.string()), "cannot write the output file");
    EXPECT_FALSE(std::filesystem::exists(nowhere));
    refused(config(""), "output_path is required");
    refused(config(dir.string()), "cannot write the output file"); // a directory
    json zero = config((dir / "zero.gguf").string());
    zero["epochs"] = 0;
    refused(zero, "epochs must be at least 1");
    json unknown = config((dir / "unknown.gguf").string());
    unknown["optimizer"] = 7;
    refused(unknown, "optimizer must be 0 (AdamW) or 1 (SGD)");
    json missing = config((dir / "missing.gguf").string());
    missing.erase("training_text");
    missing["training_file"] = (dir / "no-corpus.txt").string();
    refused(missing, "cannot open training file");
    for (const char *name : {"zero.gguf", "unknown.gguf", "missing.gguf"}) {
        EXPECT_FALSE(std::filesystem::exists(dir / name)) << name;
    }
    // a run that can write still writes, over a file that is there
    const auto out = dir / "trained.gguf";
    write_bytes(out, "old");
    ASSERT_EQ(jllama_finetune(config(out.string()).dump().c_str()), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(std::filesystem::file_size(out), 3u);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(path, ec);
}

namespace {
// a fine-tuning run of one epoch over the generator's model spec, into out
jllama_status finetune_spec(const tiny_model_spec &spec, const std::filesystem::path &out) {
    const std::string path = write_tiny_llama_gguf(spec);
    if (path.empty()) {
        return JLLAMA_STATUS_ERROR;
    }
    const json cfg = {{"model_path", path},          {"training_text", training_corpus(24)},
                      {"output_path", out.string()}, {"epochs", 1},
                      {"learning_rate", 1e-4},       {"n_ctx", 64},
                      {"n_gpu_layers", 0},           {"val_split", 0.0}};
    const jllama_status status = jllama_finetune(cfg.dump().c_str());
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return status;
}
} // namespace

TEST(CApiTrainer, AModelWhoseGraphHasAnOpWithoutABackwardPassIsRefusedByTheOpAndTheProcessGoesOn) {
    // ggml_build_backward_expand ended the process (GGML_ABORT) on an op ggml_compute_backward has no case
    // for, so fine-tuning a mixture of experts, a recurrent model, Gemma 4 or Cohere2 ended the JVM;
    // patches/0044 refuses the model before the backward graph is built and names the op
    tiny_model_spec moe;
    moe.prefix = "jllama-c-api-train-moe";
    moe.n_expert = 4;
    moe.n_expert_used = 2;
    const struct {
        const char *name;
        tiny_model_spec spec;
        const char *op;
    } rows[] = {
        {"moe", moe, "MUL_MAT_ID"},
        {"mamba", tiny_model_spec::recurrent("jllama-c-api-train-mamba"), "SSM_SCAN"},
        {"gemma4", tiny_model_spec::gemma4(8, "jllama-c-api-train-gemma4"), "TANH"},
        {"qwen3next", tiny_model_spec::qwen3next("jllama-c-api-train-qwen3next"), "MUL_MAT_ID"},
        {"zaya1", tiny_model_spec::zaya1("jllama-c-api-train-zaya1"), "MUL_MAT_ID"},
        {"cohere2", tiny_model_spec::swa(16, "jllama-c-api-train-cohere2"), "NORM"},
    };
    const std::filesystem::path dir = fresh_dir("jllama-c-api-train-refused");
    for (const auto &row : rows) {
        const auto out = dir / (std::string(row.name) + ".gguf");
        EXPECT_EQ(finetune_spec(row.spec, out), JLLAMA_STATUS_ERROR) << row.name;
        const std::string error = jllama_last_error();
        EXPECT_NE(error.find(std::string("cannot train this model: ggml has no backward pass for ") + row.op + " "),
                  std::string::npos)
            << row.name << ": " << error;
        EXPECT_FALSE(std::filesystem::exists(out)) << row.name;
    }
    // and the process trains the next model
    const auto out = dir / "llama.gguf";
    ASSERT_EQ(finetune_spec(tiny_model_spec(), out), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_GT(std::filesystem::file_size(out), 0u);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(CApiTrainer, TheGeneratorsArchitecturesGgmlCanDifferentiateStillTrain) {
    // the check of patches/0044 refuses no graph the backward pass can be built for: sliding-window and
    // full attention with norms on the heads (Domyn Edge), an MTP block (HY-V3) and Qwen3-VL's text model
    const struct {
        const char *name;
        tiny_model_spec spec;
    } rows[] = {
        {"domyn", tiny_model_spec::domyn_edge(8, "jllama-c-api-train-domyn")},
        {"hy_v3", tiny_model_spec::mtp("jllama-c-api-train-hy")},
        {"qwen3vl", tiny_model_spec::qwen3vl("jllama-c-api-train-qwen3vl")},
    };
    const std::filesystem::path dir = fresh_dir("jllama-c-api-train-trains");
    for (const auto &row : rows) {
        const auto out = dir / (std::string(row.name) + ".gguf");
        EXPECT_EQ(finetune_spec(row.spec, out), JLLAMA_STATUS_OK) << row.name << ": " << jllama_last_error();
        EXPECT_TRUE(std::filesystem::exists(out)) << row.name;
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_F(CApiEnumSweep, ALongImatrixNameIsRecordedAsValidUtf8) {
    // quantize.imatrix.file is a 127-byte field of llama_model_kv_override; a longer name was cut at a
    // byte, and inside a character it wrote invalid UTF-8 into the GGUF (60 CJK characters are 180 bytes)
    std::string name;
    for (int i = 0; i < 60; ++i) {
        name += "\xE6\x97\xA5"; // 日
    }
    const std::filesystem::path dir = fresh_dir("jllama-c-api-imatrix-name");
    const std::filesystem::path long_name = dir / std::filesystem::u8path(name + ".gguf");
    std::filesystem::copy_file(imatrix_, long_name);
    const std::string out = (dir / "q4.gguf").string();
    ASSERT_EQ(jllama_quantize_imatrix(path_.c_str(), out.c_str(), 15, 1, false, long_name.u8string().c_str()),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    gguf_init_params no_data = {true, nullptr};
    gguf_context *g = gguf_init_from_file(out.c_str(), no_data);
    ASSERT_NE(g, nullptr);
    const int64_t key = gguf_find_key(g, "quantize.imatrix.file");
    ASSERT_GE(key, 0);
    const std::string recorded = gguf_get_val_str(g, key);
    gguf_free(g);
    EXPECT_LE(recorded.size(), 127u);
    EXPECT_EQ(recorded.size() % 3, 0u) << "cut inside a character";
    EXPECT_EQ(name.compare(0, recorded.size(), recorded), 0) << "not the start of the name";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(CApiSystemOne, AnInterruptedDecisionIsReportedAsInterrupted) {
    // upstream's handler returns its response untouched when the results are terminated -- status 200
    // and no body -- and the engine handed that out as an empty answer: jllama_model_interrupt (and
    // LlamaModel.close()) during jllama_systemone gave JLLAMA_STATUS_OK with "". A decision over a long
    // state, one token per batch on one thread, takes long enough to be interrupted: upstream's wait
    // looks at the stop flag once a second, so the decision has to outlast that first look. 40
    // sentences (2837 tokens) took 2.3 s here and 0.68 s on a GitHub runner, which then answered before
    // the look (main, run 45052971364); 160 take several seconds on either, and the test about one.
    tiny_model_spec spec = tiny_model_spec::wide(256, "jllama-c-api-decision-interrupt");
    spec.decision_type = "openjev";
    spec.n_ctx_train = 16384;
    spec.n_layer = 2;
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"-c", "16384", "-t", "1", "-tb", "1", "-b", "1", "-ub", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    std::string state;
    for (int i = 0; i < 160; ++i) {
        state += "I was charged twice for my order last week and nobody has replied. ";
    }
    const json request = {
        {"state", state},
        {"questions", {{"angry", {{"type", "noul"}, {"instructions", "Is the customer angry about the charge?"}}}}}};
    std::atomic<int> status{-1};
    std::string response;
    std::string error;
    std::thread caller([&]() {
        char *out = nullptr;
        status = jllama_systemone(model, request.dump().c_str(), &out);
        response = out != nullptr ? out : "";
        error = jllama_last_error();
        jllama_string_free(out);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    jllama_model_interrupt(model);
    caller.join();
    EXPECT_EQ(status.load(), JLLAMA_STATUS_INTERRUPTED) << "answer: '" << response << "' " << error;
    jllama_model_free(model);
    std::remove(path.c_str());
}

TEST(CApiSleep, ASleepingModelWhoseFileIsGoneIsNotWokenAndAnswersWhenItIsBack) {
    // A wake from the idle sleep loads the model's files again, and upstream ends the process when that
    // fails (GGML_ABORT "failed to reload model after sleeping"): a model whose file was moved while it
    // slept took its host with it on the next request. The engine refuses the wake now, the model stays
    // asleep, and it answers once the file is back.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sleep-moved");
    ASSERT_FALSE(path.empty());
    struct Sleeps {
        std::atomic<int> n{0};
    } sleeps;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *user) {
                      if (std::string(text).find("server is entering sleeping state") != std::string::npos) {
                          static_cast<Sleeps *>(user)->n++;
                      }
                  },
                  &sleeps),
              JLLAMA_STATUS_OK);
    struct LogCallbackReset {
        ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); }
    } log_callback_reset;
    jllama_model *model = load_tiny(path, {"--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::string expected = greedy_answer(model);
    const auto asleep = [&sleeps](int count) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (sleeps.n.load() < count && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return sleeps.n.load() >= count;
    };
    ASSERT_TRUE(asleep(1));
    const std::string moved = path + ".moved";
    std::filesystem::rename(path, moved);
    Owned refused;
    EXPECT_EQ(jllama_completions(model, R"({"prompt":"Hello there","temperature":0,"n_predict":8})", &refused.s),
              JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("cannot be woken"), std::string::npos) << jllama_last_error();
    EXPECT_NE(std::string(jllama_last_error()).find(path), std::string::npos) << jllama_last_error();
    Owned meta;
    EXPECT_EQ(jllama_model_meta(model, &meta.s), JLLAMA_STATUS_ERROR);
    std::filesystem::rename(moved, path);
    EXPECT_EQ(greedy_answer(model), expected);
    jllama_model_free(model);
    std::remove(path.c_str());
}

TEST_F(CApiZonos3, AnEmptyTextIsTheCallersMistake) {
    // upstream's /tts refuses an empty input (400); the pipelines answered one, and a text of spaces,
    // each its own way -- audio of nothing, "failed to set TTS input", "no audio was generated" --, none
    // of them the caller's mistake
    for (const char *text : {"", "   ", " \n\t "}) {
        SCOPED_TRACE(std::string("'") + text + "'");
        jllama_tts_options options;
        jllama_tts_options_init(&options);
        unsigned char *bytes = nullptr;
        size_t length = 0;
        EXPECT_EQ(jllama_tts_synthesize_options(tts(), text, &options, &bytes, &length), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("nothing to speak"), std::string::npos) << jllama_last_error();
        EXPECT_EQ(bytes, nullptr);
    }
    std::vector<unsigned char> wav;
    EXPECT_EQ(speak(1.0f, 0.0f, &wav), JLLAMA_STATUS_OK) << jllama_last_error();
}

TEST(CApiQwen3TtsClone, AnImageAsTheSpeakerReferenceIsTheCallersMistake) {
    // mtmd's helper reads images too: a picture named as the reference reached set_input, which refused it
    // as "failed to set TTS input (prompt/speaker-reference/lang rejected)", a server error
    const std::string backbone = jllama_test::write_tiny_qwen3tts_gguf("jllama-qwen3tts-clone-image");
    const std::string mmproj = jllama_test::write_tiny_qwen3tts_mmproj_gguf("jllama-qwen3tts-clone-image-mmproj", true);
    jllama_tts_model *tts = nullptr;
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), mmproj.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    const std::string png = (std::filesystem::temp_directory_path() / "jllama-qwen3tts-reference.png").string();
    const std::vector<uint8_t> bytes = base64_bytes(TINY_PNG_BASE64);
    write_bytes(png, std::string(bytes.begin(), bytes.end()));
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 10;
    options.speaker_reference_path = png.c_str();
    unsigned char *wav = nullptr;
    size_t length = 0;
    EXPECT_EQ(jllama_tts_synthesize_options(tts, "hello", &options, &wav, &length), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("is no audio"), std::string::npos) << jllama_last_error();
    options.speaker_reference_path = nullptr;
    ASSERT_EQ(jllama_tts_synthesize_options(tts, "hello", &options, &wav, &length), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_bytes_free(wav);
    jllama_tts_free(tts);
    for (const std::string &p : {backbone, mmproj, png}) {
        std::remove(p.c_str());
    }
}

TEST(CApiLocale, TheNumericLocaleGuardLeaksNothing) {
    // Every C API call runs under jllama::c_numeric_locale. Apple's newlocale() copies its base and
    // leaves it to the caller (POSIX lets it consume the base), so the duplocale() the guard gave it
    // leaked one locale object per call -- a stream, one per token; bionic ignores the base altogether.
    // A hundred thousand guards must leave the heap as it was.
#if defined(__APPLE__) || (defined(__GLIBC__) && (__GLIBC__ > 2 || __GLIBC_MINOR__ >= 33))
    const auto in_use = []() -> size_t {
#if defined(__APPLE__)
        malloc_statistics_t stats;
        malloc_zone_statistics(nullptr, &stats);
        return stats.size_in_use;
#else
        return mallinfo2().uordblks;
#endif
    };
    {
        const jllama::c_numeric_locale warm;
    }
    const size_t before = in_use();
    for (int i = 0; i < 100000; ++i) {
        const jllama::c_numeric_locale numbers;
    }
    const size_t after = in_use();
    EXPECT_LT(after, before + 1024 * 1024) << "the guard leaked " << (after - before) << " bytes";
#else
    GTEST_SKIP() << "no heap statistics on this platform";
#endif
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiLogFile, AnAttachedServerKeepsTheModelsLogFormat) {
    // An attached server parsed its argv after common_init(), which turns common_log's prefix and
    // timestamps on process-wide: it undid the --no-log-prefix / --no-log-timestamps of the model it
    // serves. The model's load calls common_init(); the attach does not any more, as llama_server_attach()
    // never did.
    const std::string model_path = write_tiny_llama_gguf("jllama-c-api-attach-log-format");
    ASSERT_FALSE(model_path.empty());
    const auto log = log_file_path("jllama-c-api-attach-log-format");
    std::error_code ec;
    std::filesystem::remove(log, ec);
    jllama_model *model = load_tiny(model_path, {"--log-file", log.string(), "--no-log-timestamps", "--no-log-prefix"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    {
        AttachedServer attached(model);
        ASSERT_NE(attached.server, nullptr);
        auto client = attached.client();
        const auto res = client.Post("/completion", R"({"prompt":"Hello","n_predict":4})", "application/json");
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 200);
    }
    EXPECT_FALSE(greedy_answer(model).empty());
    jllama_model_free(model);
    const auto [lines, stamped] = lines_and_stamped(log);
    EXPECT_GT(lines, 0u);
    EXPECT_EQ(stamped, 0u) << "the attach turned the timestamps on again";
    std::ifstream in(log);
    static const std::regex prefixed(R"(^[DIWE] )");
    size_t with_prefix = 0;
    for (std::string line; std::getline(in, line);) {
        with_prefix += std::regex_search(line, prefixed) ? 1 : 0;
    }
    EXPECT_EQ(with_prefix, 0u) << "the attach turned the prefix on again";
    std::filesystem::remove(log, ec);
    std::filesystem::remove(model_path, ec);
}

TEST(CApiServer, ASecondEmbeddedServerIsRefusedWhileOneRuns) {
    // llama.cpp keeps an embedded server's shutdown state in globals (patches/0006, 0007): a second
    // server replaced the first one's shutdown handler, so the first one's stop() stopped the second and
    // then waited for itself for good. A start while a server runs is refused now; once it returned, the
    // next start works.
    const std::filesystem::path models_dir = fresh_dir("jllama-c-api-two-servers");
    const std::vector<std::string> args = {
        "llama-server", "--models-dir", models_dir.string(), "--host", "127.0.0.1", "--port", "0"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_server *first = nullptr;
    ASSERT_EQ(jllama_server_start(static_cast<int>(argv.size()), argv.data(), &first), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_server *second = nullptr;
    EXPECT_EQ(jllama_server_start(static_cast<int>(argv.size()), argv.data(), &second), JLLAMA_STATUS_ERROR);
    EXPECT_NE(std::string(jllama_last_error()).find("already running"), std::string::npos) << jllama_last_error();
    EXPECT_EQ(second, nullptr);
    const std::string model_path = write_tiny_llama_gguf("jllama-c-api-two-servers-model");
    jllama_model *model = load_tiny(model_path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const std::vector<const char *> attach_argv = {"llama-server", "--host", "127.0.0.1", "--port", "0"};
    jllama_server *attached = nullptr;
    EXPECT_EQ(jllama_server_start_attached(model, static_cast<int>(attach_argv.size()), attach_argv.data(), &attached),
              JLLAMA_STATUS_ERROR);
    EXPECT_EQ(attached, nullptr);
    // the first stop returns (on a thread of its own, so a stop that hangs fails the test rather than
    // the run -- the thread is left behind then, as is the server)
    auto status = std::make_shared<std::promise<jllama_status>>();
    std::future<jllama_status> stopped = status->get_future();
    std::thread([first, status]() { status->set_value(jllama_server_stop(first)); }).detach();
    ASSERT_EQ(stopped.wait_for(std::chrono::seconds(60)), std::future_status::ready) << "the stop hangs";
    EXPECT_EQ(stopped.get(), JLLAMA_STATUS_OK);
    ASSERT_EQ(jllama_server_start_attached(model, static_cast<int>(attach_argv.size()), attach_argv.data(), &attached),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(jllama_server_stop(attached), JLLAMA_STATUS_OK);
    jllama_model_free(model);
    std::error_code ec;
    std::filesystem::remove_all(models_dir, ec);
    std::filesystem::remove(model_path, ec);
}

TEST(CApiDownload, ARerankerWithoutAHeadIsRefusedAfterItsDownload) {
    // The refusal of rank pooling without a classification head read the model file at the parse, before
    // the models handler downloaded it: --model-url (or -hf) of an encoder without a head was never
    // refused, and it reranked with the first value of the pooled embedding. It is read after the download.
    const std::string encoder = write_tiny_bert_gguf(false, 2 /* LLAMA_POOLING_TYPE_CLS */, "jllama-bert-download");
    ASSERT_FALSE(encoder.empty());
    std::string bytes;
    {
        std::ifstream in(encoder, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    httplib::Server http;
    http.Get("/encoder.gguf", [&bytes](const httplib::Request &, httplib::Response &res) {
        res.set_content(bytes, "application/octet-stream");
    });
    const int port = http.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread serving([&http]() { http.listen_after_bind(); });
    struct stop_http {
        httplib::Server &http;
        std::thread &serving;
        ~stop_http() {
            http.stop();
            serving.join();
        }
    } stop{http, serving};
    http.wait_until_ready();
    const std::filesystem::path dir = fresh_dir("jllama-c-api-download-reranker");
    const std::string target = (dir / "encoder.gguf").string();
    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/encoder.gguf";
    EXPECT_EQ(load_tiny(target, {"--model-url", url, "--reranking"}), nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("has no classification head"), std::string::npos)
        << jllama_last_error();
    // the file is there now, and as an embedding model it loads
    jllama_model *embedding = load_tiny(target, {"--embedding"});
    EXPECT_NE(embedding, nullptr) << jllama_last_error();
    jllama_model_free(embedding);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove(encoder, ec);
}

TEST(CApiDownload, ARepositoryOfPresetsIsRefusedByItsCause) {
    // -hf of a repository with a preset.ini at its root is a set of models: the models handler clears the
    // model path so that llama_server() starts in router mode, and the load failed with "could not load
    // model from given file path". A loopback server answers the Hugging Face API for such a repository.
    const std::string commit = "0123456789abcdef0123456789abcdef01234567";
    const std::string preset = "[*]\nctx-size = 64\n";
    httplib::Server hub;
    hub.Get("/api/models/jllama/presets/refs", [&commit](const httplib::Request &, httplib::Response &res) {
        res.set_content(json{{"branches", json::array({{{"name", "main"}, {"targetCommit", commit}}})}}.dump(),
                        "application/json");
    });
    hub.Get("/api/models/jllama/presets/tree/" + commit, [&preset](const httplib::Request &, httplib::Response &res) {
        res.set_content(json::array({{{"type", "file"}, {"path", "preset.ini"}, {"size", preset.size()}}}).dump(),
                        "application/json");
    });
    hub.Get("/jllama/presets/resolve/" + commit + "/preset.ini",
            [&preset](const httplib::Request &, httplib::Response &res) { res.set_content(preset, "text/plain"); });
    const int port = hub.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread serving([&hub]() { hub.listen_after_bind(); });
    struct stop_http {
        httplib::Server &http;
        std::thread &serving;
        ~stop_http() {
            http.stop();
            serving.join();
        }
    } stop{hub, serving};
    hub.wait_until_ready();
    const std::filesystem::path cache = fresh_dir("jllama-c-api-hf-cache");
    const std::string endpoint = "http://127.0.0.1:" + std::to_string(port) + "/";
    const scoped_env model_endpoint("MODEL_ENDPOINT", endpoint.c_str());
    const scoped_env hf_home("HF_HOME", cache.string().c_str());
    const scoped_env llama_cache("LLAMA_CACHE", cache.string().c_str());
    const std::vector<const char *> argv = {"jllama", "-hf", "jllama/presets", "-c", "64", "-ngl", "0", "--no-warmup"};
    jllama_model *model = nullptr;
    EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model),
              JLLAMA_STATUS_ERROR);
    EXPECT_EQ(model, nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("repository of model presets"), std::string::npos)
        << jllama_last_error();
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);
}
#endif // JLLAMA_HAS_NATIVE_SERVER

#ifndef _WIN32
TEST(CApiLogFile, ClosingALogSinkLeavesADisabledLogDisabled) {
    // patches/0040: --log-disable pauses common_log, and every setter resumes it -- also the ones that
    // take a sink away: freeing the model closed its --log-file (common_log_set_file(log, nullptr)) and
    // clearing the logger (what LlamaModel.setLogger(format, null) does to flush) turned the log of the
    // process back on, onto the console. A later load prints nothing to stdout or stderr now.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-log-disabled");
    ASSERT_FALSE(path.empty());
    const auto log = log_file_path("jllama-c-api-log-disabled");
    jllama_model *quiet = load_tiny(path, {"--log-disable", "--log-file", log.string()});
    ASSERT_NE(quiet, nullptr) << jllama_last_error();
    jllama_model_free(quiet);                  // closes its file
    jllama_set_log_callback(nullptr, nullptr); // clears the logger
    const std::filesystem::path captured = std::filesystem::temp_directory_path() / "jllama-c-api-log-captured.txt";
    std::fflush(stdout);
    std::fflush(stderr);
    const int fd = open(captured.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    ASSERT_GE(fd, 0);
    const int saved_out = dup(1);
    const int saved_err = dup(2);
    dup2(fd, 1);
    dup2(fd, 2);
    jllama_model *loud = load_tiny(path, {});
    const bool loaded = loud != nullptr;
    const std::string answer = loaded ? greedy_answer(loud) : std::string();
    jllama_model_free(loud);
    common_log_pause(common_log_main()); // whatever a running worker queued is printed first
    std::fflush(stdout);
    std::fflush(stderr);
    dup2(saved_out, 1);
    dup2(saved_err, 2);
    close(saved_out);
    close(saved_err);
    close(fd);
    common_log_resume(common_log_main()); // the rest of this process logs again
    ASSERT_TRUE(loaded) << jllama_last_error();
    EXPECT_FALSE(answer.empty());
    std::ifstream in(captured);
    const std::string printed((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(printed.find("loading model"), std::string::npos) << printed.substr(0, 400);
    std::error_code ec;
    std::filesystem::remove(captured, ec);
    std::filesystem::remove(log, ec);
    std::filesystem::remove(path, ec);
}
#endif

// ---------------------------------------------------------------------------------------------------
// Test gaps of 2026-11-19: paths the suite did not reach yet.

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiServer, FreeingAModelWaitsForTheServerAttachedToIt) {
    // An attached server holds a user reference on the model for the life of its worker
    // (native_server_engine.cpp), and jllama_model_free waits for users == 0 before it stops the
    // inference loop and deletes the context. A free issued while the server runs must therefore
    // block until the server is stopped, not tear the model down under the HTTP routes. The free runs
    // on a second thread; the server is stopped on this one after a pause in which the free must not
    // have returned and the server must still answer.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-free-attached");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    AttachedServer attached(model);
    ASSERT_NE(attached.server, nullptr);
    httplib::Client client = attached.client();
    ASSERT_EQ(status_of(client.Get("/health")), 200);

    std::atomic<bool> free_called{false};
    std::atomic<bool> freed{false};
    std::chrono::steady_clock::time_point freed_at;
    std::thread freer([&]() {
        free_called = true;
        jllama_model_free(model);
        freed_at = std::chrono::steady_clock::now();
        freed = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    EXPECT_TRUE(free_called.load());
    EXPECT_FALSE(freed.load()) << "jllama_model_free returned while a server was attached to the model";
    EXPECT_TRUE(jllama_server_is_running(attached.server));
    EXPECT_EQ(status_of(client.Get("/health")), 200) << "the server stopped answering while the free waited";

    const auto stop_called_at = std::chrono::steady_clock::now();
    EXPECT_EQ(jllama_server_stop(attached.server), JLLAMA_STATUS_OK) << jllama_last_error();
    attached.server = nullptr; // stopped here, not by the destructor
    freer.join();
    EXPECT_TRUE(freed.load());
    EXPECT_GE(freed_at, stop_called_at) << "the free returned before the server was stopped";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiLocale, TheAttachedServerRendersATemplateNumberWithADecimalPointUnderADecimalCommaLocale) {
    // The attached server's HTTP threads render the chat template (POST /apply-template, every chat
    // route), and they are not threads of the C API: no jllama::c_numeric_locale guards them. What
    // keeps their numbers right is the server setting the "C" numeric locale for the process when it
    // starts (llama_server_attach_parsed, patches/0007; llama_server in 0006 for the classic server),
    // as jllama_api.h documents. Measured with the process put back to de_DE while the server ran, the
    // same request rendered "[0,5|0,|3,]Hi": 0.5 written with a comma, '0.25' read as 0.
    const DecimalCommaLocale comma;
    if (!comma.ok()) {
        if (std::getenv("JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE") != nullptr) {
            FAIL() << "JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE is set, and no locale with a decimal comma is installed";
        }
        GTEST_SKIP() << "no locale with a decimal comma is installed (locale-gen de_DE.UTF-8)";
    }
    const std::string path = write_tiny_llama_gguf("jllama-c-api-locale-attached");
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {"--jinja", "--chat-template",
                                           "[{{ 0.5 }}|{{ '0.25' | float }}|{{ 1.5 * 2 }}]{{ messages[0].content }}"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    ASSERT_EQ(std::string(std::localeconv()->decimal_point), ",") << "the load changed the process's locale";
    {
        AttachedServer attached(model);
        ASSERT_NE(attached.server, nullptr);
        EXPECT_EQ(std::string(std::localeconv()->decimal_point), ".")
            << "the server did not set the \"C\" numeric locale for the process";
        httplib::Client client = attached.client();
        const auto rendered =
            client.Post("/apply-template", R"({"messages":[{"role":"user","content":"Hi"}]})", "application/json");
        ASSERT_TRUE(rendered) << httplib::to_string(rendered.error());
        ASSERT_EQ(rendered->status, 200) << rendered->body;
        EXPECT_NE(json::parse(rendered->body).at("prompt").get<std::string>().find("[0.5|0.25|3.0]Hi"),
                  std::string::npos)
            << rendered->body;
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
#endif // JLLAMA_HAS_NATIVE_SERVER

TEST(CApiSleep, AStreamDrainedAfterTheModelFellAsleepDeliversEveryChunk) {
    // A stream's results wait in the server's result queue, which the idle sleep does not touch: the
    // sleep frees the model and the context (server_context_impl::destroy) but not the results of a
    // task that ended, and receiving them does not wake the model (stream_next waits on the reader
    // alone). A stream received to its first chunk is left until the model sleeps -- it does: a
    // reader that is not drained keeps no wake count, and the task itself ends in milliseconds -- and
    // the rest is received from the sleeping model, every chunk. The next request wakes the model and
    // answers what the stream answered.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-sleep-stream");
    ASSERT_FALSE(path.empty());
    LogCapture log;
    jllama_model *model = load_tiny(path, {"-c", "128", "--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const char *SLEEPING = "server is entering sleeping state";
    const char *WAKING = "server is exiting sleeping state";
    ASSERT_TRUE(log.wait_for(SLEEPING, 1, std::chrono::seconds(20))) << "the model did not sleep after its load";
    const json request = {
        {"prompt", "Hello there"}, {"n_predict", 8}, {"temperature", 0}, {"ignore_eos", true}, {"cache_prompt", false}};
    json streaming = request;
    streaming["stream"] = true;
    int task = -1;
    ASSERT_EQ(jllama_completions_stream_request(model, streaming.dump().c_str(), &task), JLLAMA_STATUS_OK)
        << jllama_last_error();
    // a log line reaches the capture on common_log's worker thread, so a line just logged is waited for
    EXPECT_TRUE(log.wait_for(WAKING, 1, std::chrono::seconds(5))) << "the stream request did not wake the model";
    std::string streamed;
    int partial = 0;
    bool stop = false;
    {
        Owned first;
        ASSERT_EQ(jllama_completions_stream_receive(model, task, &first.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        ASSERT_FALSE(stop) << first.str();
        streamed += json::parse(first.str()).at("content").get<std::string>();
        ++partial;
    }
    // 10 s rather than 3: the sleep comes within about two seconds of the task's end (an idle second
    // and the loop's one-second poll), and a slow machine must not turn that into a failure
    ASSERT_TRUE(log.wait_for(SLEEPING, 2, std::chrono::seconds(10)))
        << "the model did not sleep while the stream was left half received";
    json last;
    while (!stop) {
        Owned chunk;
        ASSERT_EQ(jllama_completions_stream_receive(model, task, &chunk.s, &stop), JLLAMA_STATUS_OK)
            << jllama_last_error();
        last = json::parse(chunk.str());
        if (!stop) {
            streamed += last.at("content").get<std::string>();
            ++partial;
        }
    }
    EXPECT_EQ(log.count(WAKING), 1) << "receiving the stream woke the model";
    EXPECT_EQ(partial, 8) << "a chunk was lost in the sleep";
    EXPECT_EQ(last.at("tokens_predicted").get<int>(), 8) << last.dump();
    EXPECT_EQ(last.at("timings").at("cache_n").get<int>(), 0) << last.dump();

    Owned again;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &again.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_TRUE(log.wait_for(WAKING, 2, std::chrono::seconds(5)))
        << "the request after the stream did not wake the model";
    const json answer = json::parse(again.str());
    EXPECT_EQ(answer.at("timings").at("cache_n").get<int>(), 0) << answer.dump();
    EXPECT_EQ(answer.at("content").get<std::string>(), streamed);
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiSleep, ADecisionAsTheFirstCallAfterASleepLetsTheModelSleepAgain) {
    // /v1/systemone goes to upstream's route handler, which wakes the server itself
    // (server_res_generator: wait_until_no_sleep) and gives the wake back when its response is
    // destroyed -- the engine does not wake it (jllama_engine.cpp, systemone). A decision as the first
    // call after a sleep must answer and let the model sleep again, twice in a row.
    const std::string path = write_tiny_llama_gguf(tiny_model_spec::decision("jllama-c-api-sleep-decision"));
    ASSERT_FALSE(path.empty());
    SleepWatch watch;
    jllama_model *model = load_tiny(path, {"-c", "1024", "--sleep-idle-seconds", "1"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    ASSERT_TRUE(watch.next_sleep()) << "the decision model did not sleep after its load";
    for (int round = 0; round < 2; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        Owned response;
        ASSERT_EQ(jllama_systemone(model, SYSTEMONE_REQUEST, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
        EXPECT_EQ(json::parse(response.str()).at("answers").size(), 3u) << response.str();
        EXPECT_TRUE(watch.next_sleep()) << "the model did not sleep after jllama_systemone";
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiSlotFiles, RestoringAFileThatWasNeverSavedEmptiesTheSlotAndTheModelGoesOn) {
    // A restore of a name with no file behind it: llama_state_seq_load_file cannot open it and reads
    // nothing, which the server answers as an invalid slot file (ERROR_TYPE_INVALID_REQUEST) and,
    // as for every failed restore, with the slot cleared (prompt_clear) -- whatever it held before.
    // Nothing is created under the name, and the model answers the next request.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-slot-never-saved");
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir = fresh_dir("jllama-c-api-slot-never-saved");
    jllama_model *model = load_tiny(path, {"-c", "128", "--slot-save-path", dir.string()});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const char *request = R"({"prompt":"Hello there, a prompt of some tokens","n_predict":2,"temperature":0})";
    const auto complete = [&]() {
        Owned completion;
        EXPECT_EQ(jllama_completions(model, request, &completion.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return completion.s != nullptr ? json::parse(completion.str()) : json::object();
    };
    complete();
    // the slot holds the prompt: the same request takes it from the cache
    ASSERT_GT(complete().at("timings").at("cache_n").get<int>(), 0);

    Owned refused;
    EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "never-saved.bin", &refused.s),
              JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_EQ(refused.s, nullptr);
    EXPECT_NE(std::string(jllama_last_error()).find("invalid slot save file"), std::string::npos)
        << jllama_last_error();
    EXPECT_FALSE(std::filesystem::exists(dir / "never-saved.bin"));
    const json after = complete();
    EXPECT_EQ(after.at("timings").at("cache_n").get<int>(), 0) << "the slot kept its prompt: " << after.dump();
    EXPECT_FALSE(after.at("content").get<std::string>().empty());
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove(path, ignored);
}

TEST(CApiSlotFiles, ASlotSavedWithALargerContextIsRefusedByASmallerOne) {
    // A slot file is the model's, not the context's: the state of a prompt of 300 tokens that a
    // -c 512 load saved cannot go into the same model with a smaller context. llama.cpp pads a context
    // to 256 cells, so -c 64 holds 256 tokens: the restore finds no room for the state ("No available
    // space in KV cache"). A context of 512 whose slot --kv-unified-per-slot limits to 128 has the room
    // but not the slot: the restored prompt is longer than the slot ("does not fit in the slot
    // context"). Both are invalid requests that leave the slot empty, and the same file restores into
    // a -c 512 load again, which shows the refusals are the size's.
    tiny_model_spec spec;
    spec.prefix = "jllama-c-api-slot-larger";
    spec.n_ctx_train = 1024; // or every slot is capped at the 128 of the default model
    const std::string path = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(path.empty());
    const std::filesystem::path dir = fresh_dir("jllama-c-api-slot-larger");
    const auto load = [&](const std::vector<std::string> &context) {
        std::vector<std::string> args = context;
        args.insert(args.end(), {"--slot-save-path", dir.string()});
        jllama_model *model = load_tiny(path, args);
        EXPECT_NE(model, nullptr) << jllama_last_error();
        return model;
    };
    const auto slot_context = [](jllama_model *model) {
        Owned metrics;
        EXPECT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
        return metrics.s != nullptr ? json::parse(metrics.str()).at("slots").at(0).at("n_ctx").get<int>() : -1;
    };
    int saved_tokens = 0;
    {
        jllama_model *large = load({"-c", "512"});
        ASSERT_NE(large, nullptr);
        const json request = {{"prompt", std::string(300, 'a')}, {"n_predict", 2}, {"temperature", 0}};
        Owned completion;
        ASSERT_EQ(jllama_completions(large, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        Owned saved;
        ASSERT_EQ(jllama_slot_action(large, JLLAMA_SLOT_SAVE, 0, "large.bin", &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        saved_tokens = json::parse(saved.str()).at("n_saved").get<int>();
        jllama_model_free(large);
    }
    ASSERT_GT(saved_tokens, 256);
    const auto refused = [&](const std::vector<std::string> &context, const char *reason) {
        SCOPED_TRACE(reason);
        jllama_model *model = load(context);
        ASSERT_NE(model, nullptr);
        ASSERT_LT(slot_context(model), saved_tokens) << "the slot holds the saved prompt";
        Owned response;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "large.bin", &response.s),
                  JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find(reason), std::string::npos) << jllama_last_error();
        Owned completion;
        ASSERT_EQ(jllama_completions(model, R"({"prompt":"Hello","n_predict":2,"temperature":0})", &completion.s),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(json::parse(completion.str()).at("timings").at("cache_n").get<int>(), 0) << completion.str();
        jllama_model_free(model);
    };
    refused({"-c", "64"}, "No available space in KV cache");
    refused({"-c", "512", "--kv-unified-per-slot", "128"}, "does not fit in the slot context");
    {
        jllama_model *again = load({"-c", "512"});
        ASSERT_NE(again, nullptr);
        Owned restored;
        ASSERT_EQ(jllama_slot_action(again, JLLAMA_SLOT_RESTORE, 0, "large.bin", &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        EXPECT_EQ(json::parse(restored.str()).at("n_restored").get<int>(), saved_tokens) << restored.str();
        jllama_model_free(again);
    }
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::remove(path, ignored);
}

namespace {

// The mean per column of every entry of an importance matrix: sum / count, with the count of the
// matrix (expert) the column belongs to -- one count for all when the file has one.
std::map<std::string, std::vector<double>> imatrix_means(const imatrix_file &file) {
    std::map<std::string, std::vector<double>> means;
    const std::string suffix = ".in_sum2";
    for (const auto &[name, sums] : file.tensors) {
        if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string weight = name.substr(0, name.size() - suffix.size());
        const std::vector<float> &counts = file.tensors.at(weight + ".counts");
        const size_t per_count = sums.size() / counts.size();
        std::vector<double> &m = means[weight];
        for (size_t i = 0; i < sums.size(); ++i) {
            m.push_back(static_cast<double>(sums[i]) / static_cast<double>(counts[i / per_count]));
        }
    }
    return means;
}

} // namespace

TEST(CApiImatrix, AMixtureOfExpertsKeepsItsPerExpertMeansThroughTheLegacyFormat) {
    // The legacy format has one call count per entry, while an expert tensor of a GGUF matrix has a
    // count per expert: save_legacy writes each expert's columns as its mean times the one count
    // (imatrix_engine.cpp), and the conversion back reads them over that count. An expert's mean must
    // survive both ways -- which only says something if the experts were routed to unequally, so
    // that one count for all would give other means; the test asserts that they were.
    const tiny_model_spec spec = moe_spec();
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text = write_calibration_text("jllama-imatrix-moe-legacy", 200);
    const std::string gguf = model + ".imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), gguf.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const imatrix_file original = read_imatrix(gguf);
    const std::string up = "blk.0.ffn_up_exps.weight";
    ASSERT_EQ(original.tensors.count(up + ".counts"), 1u);
    const std::vector<float> &counts = original.tensors.at(up + ".counts");
    ASSERT_EQ(counts.size(), static_cast<size_t>(spec.n_expert));
    for (const float c : counts) {
        ASSERT_GT(c, 0.0f) << "an expert no token was routed to";
    }
    ASSERT_NE(*std::min_element(counts.begin(), counts.end()), *std::max_element(counts.begin(), counts.end()))
        << "every expert got as many tokens as the others, so one count would do";

    const std::string legacy = model + ".imatrix.dat";
    const std::string back = model + ".back.imatrix.gguf";
    const char *only_gguf[] = {gguf.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_gguf, 1, legacy.c_str(), true, nullptr), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const char *only_legacy[] = {legacy.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_legacy, 1, back.c_str(), false, nullptr), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const auto before = imatrix_means(original);
    const auto after = imatrix_means(read_imatrix(back));
    ASSERT_EQ(after.size(), before.size());
    int expert_entries = 0;
    for (const auto &[weight, means] : before) {
        SCOPED_TRACE(weight);
        ASSERT_EQ(after.count(weight), 1u);
        const std::vector<double> &round_trip = after.at(weight);
        ASSERT_EQ(round_trip.size(), means.size());
        for (size_t i = 0; i < means.size(); ++i) {
            ASSERT_NEAR(round_trip[i], means[i], 1e-4 * std::abs(means[i]) + 1e-9) << "column " << i;
        }
        expert_entries += weight.find("_exps.") != std::string::npos ? 1 : 0;
    }
    EXPECT_EQ(expert_entries, 3 * spec.n_layer) << "ffn_gate_exps, ffn_up_exps and ffn_down_exps of every layer";
    std::error_code ec;
    for (const std::string &file : {model, text, gguf, legacy, back}) {
        std::filesystem::remove(file, ec);
    }
}

TEST(CApiImatrix, AMixtureOfExpertsMatrixMergesWithALegacyOneInEitherOrder) {
    // merge() reads a legacy entry with one count and a GGUF expert entry with one per expert: a
    // legacy file first has its count spread over the experts when the GGUF one follows, and a GGUF
    // file first has the legacy count added to every expert's. Both orders are the same sum -- the
    // GGUF sums plus the legacy file's, the GGUF counts plus the legacy count -- exactly, since a
    // float sum of two values does not depend on their order. The legacy format records no chunk
    // size: next to the GGUF file it is read at that one's (32), alone at the default of 512, so its
    // part of the sum is the legacy file converted alone, scaled by 32 / 512 (a power of two, exact).
    const tiny_model_spec spec = moe_spec();
    const std::string model = write_tiny_llama_gguf(spec);
    ASSERT_FALSE(model.empty());
    const std::string text_a = write_calibration_text("jllama-imatrix-moe-mix-a", 200);
    const std::string text_b = write_calibration_text("jllama-imatrix-moe-mix-b", 400);
    const std::string a = model + ".a.imatrix.gguf";
    const std::string b = model + ".b.imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text_a.c_str(), a.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(jllama_imatrix(model.c_str(), text_b.c_str(), b.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const std::string legacy = model + ".b.imatrix.dat";
    const char *only_b[] = {b.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_b, 1, legacy.c_str(), true, nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
    // the legacy file converted alone, at the chunk size of 512 (see above)
    const std::string legacy_back = model + ".b.back.imatrix.gguf";
    const char *only_legacy[] = {legacy.c_str()};
    ASSERT_EQ(jllama_imatrix_merge(only_legacy, 1, legacy_back.c_str(), false, nullptr), JLLAMA_STATUS_OK)
        << jllama_last_error();

    const std::string gguf_first = model + ".gguf-first.imatrix.gguf";
    const std::string legacy_first = model + ".legacy-first.imatrix.gguf";
    const char *ab[] = {a.c_str(), legacy.c_str()};
    const char *ba[] = {legacy.c_str(), a.c_str()};
    int chunks_ab = 0;
    int chunks_ba = 0;
    ASSERT_EQ(jllama_imatrix_merge(ab, 2, gguf_first.c_str(), false, &chunks_ab), JLLAMA_STATUS_OK)
        << jllama_last_error();
    ASSERT_EQ(jllama_imatrix_merge(ba, 2, legacy_first.c_str(), false, &chunks_ba), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(chunks_ab, 18);
    EXPECT_EQ(chunks_ba, 18);

    const imatrix_file fa = read_imatrix(a);
    const imatrix_file fl = read_imatrix(legacy_back);
    const imatrix_file f1 = read_imatrix(gguf_first);
    const imatrix_file f2 = read_imatrix(legacy_first);
    ASSERT_FALSE(f1.tensors.empty());
    ASSERT_EQ(f1.tensors.size(), fa.tensors.size());
    ASSERT_EQ(f2.tensors.size(), fa.tensors.size());
    for (const auto &[name, values] : f1.tensors) {
        SCOPED_TRACE(name);
        ASSERT_EQ(f2.tensors.count(name), 1u);
        EXPECT_EQ(values, f2.tensors.at(name)) << "the two orders differ";
        const std::vector<float> &va = fa.tensors.at(name);
        const std::vector<float> &vl = fl.tensors.at(name);
        ASSERT_EQ(values.size(), va.size()) << "the merged entry lost the shape of the GGUF one";
        for (size_t i = 0; i < values.size(); ++i) {
            // a count of the legacy file is one for all experts
            const float legacy_part = vl[vl.size() == 1 ? 0 : i] * (32.0f / 512.0f);
            ASSERT_FLOAT_EQ(values[i], va[i] + legacy_part) << "[" << i << "]";
        }
    }
    EXPECT_EQ(f1.tensors.at("blk.0.ffn_up_exps.weight.counts").size(), static_cast<size_t>(spec.n_expert));
    std::error_code ec;
    for (const std::string &file : {model, text_a, text_b, a, b, legacy, legacy_back, gguf_first, legacy_first}) {
        std::filesystem::remove(file, ec);
    }
}

TEST(CApiImatrix, AnOutputInADirectoryThatDoesNotExistIsRefusedByItsPath) {
    // jllama_imatrix writes its result as GGUF and jllama_imatrix_merge as GGUF or in the legacy
    // format: an output whose directory does not exist is JLLAMA_STATUS_ERROR naming the path, in each
    // of the three, and no directory is created on the way. The same calls into a directory that
    // exists succeed (the matrix computed first is the merge's input).
    const std::string model = write_tiny_llama_gguf("jllama-imatrix-no-dir");
    ASSERT_FALSE(model.empty());
    const std::string text = write_calibration_text("jllama-imatrix-no-dir", 200);
    const std::string written = model + ".imatrix.gguf";
    ASSERT_EQ(jllama_imatrix(model.c_str(), text.c_str(), written.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const std::filesystem::path missing_dir = std::filesystem::temp_directory_path() /
                                              ("jllama-imatrix-no-such-dir-" + std::to_string(std::random_device{}()));
    ASSERT_FALSE(std::filesystem::exists(missing_dir));
    const auto refused = [](jllama_status status, const std::string &output, const char *what) {
        EXPECT_EQ(status, JLLAMA_STATUS_ERROR) << what;
        EXPECT_NE(std::string(jllama_last_error()).find(output), std::string::npos)
            << what << ": " << jllama_last_error();
    };
    const std::string computed = (missing_dir / "computed.imatrix.gguf").string();
    refused(jllama_imatrix(model.c_str(), text.c_str(), computed.c_str(), 32, 0, 2, 0, nullptr, nullptr, nullptr),
            computed, "jllama_imatrix");
    const char *input[] = {written.c_str()};
    const std::string merged = (missing_dir / "merged.imatrix.gguf").string();
    refused(jllama_imatrix_merge(input, 1, merged.c_str(), false, nullptr), merged, "merge as GGUF");
    const std::string legacy = (missing_dir / "merged.imatrix.dat").string();
    refused(jllama_imatrix_merge(input, 1, legacy.c_str(), true, nullptr), legacy, "merge in the legacy format");
    EXPECT_FALSE(std::filesystem::exists(missing_dir));
    const std::string beside = model + ".legacy.imatrix.dat";
    EXPECT_EQ(jllama_imatrix_merge(input, 1, beside.c_str(), true, nullptr), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ec;
    for (const std::string &file : {model, text, written, beside}) {
        std::filesystem::remove(file, ec);
    }
}

TEST(CApiTts, ABackboneThatLoadsIsRefusedByAProjectorThatIsMissingOrDoesNotSpeak) {
    // jllama_tts_load loads the backbone first and the projector after it (tts_engine.cpp,
    // engine_init): a projector that is not there, and one that is no speech projector at all (a
    // vision projector of the same width, which mtmd loads), fail the load after the backbone came
    // up, each with its own message naming the projector -- not the backbone's. The backbone and its
    // own projector load, so the backbone was never the cause.
    const std::string backbone = jllama_test::write_tiny_zonos3_gguf("jllama-c-api-tts-refused");
    const std::string vision = write_tiny_mmproj_gguf("jllama-c-api-tts-refused-vision");
    const std::string speech = jllama_test::write_tiny_zonos3_mmproj_gguf("jllama-c-api-tts-refused-mmproj");
    ASSERT_FALSE(backbone.empty());
    ASSERT_FALSE(vision.empty());
    ASSERT_FALSE(speech.empty());
    const std::string missing = backbone + ".missing-mmproj.gguf";
    jllama_tts_model *tts = nullptr;
    EXPECT_EQ(jllama_tts_load(backbone.c_str(), missing.c_str(), 0, 2, &tts), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(tts, nullptr);
    EXPECT_EQ(std::string(jllama_last_error()), "failed to load TTS mmproj: " + missing);
    EXPECT_EQ(jllama_tts_load(backbone.c_str(), vision.c_str(), 0, 2, &tts), JLLAMA_STATUS_ERROR);
    EXPECT_EQ(tts, nullptr);
    EXPECT_EQ(std::string(jllama_last_error()), "mmproj does not support audio generation: " + vision);
    ASSERT_EQ(jllama_tts_load(backbone.c_str(), speech.c_str(), 0, 2, &tts), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(tts, nullptr);
    jllama_tts_free(tts);
    for (const std::string &file : {backbone, vision, speech}) {
        std::remove(file.c_str());
    }
}

TEST(CApiConcurrency, TwoModelsAnswerAtOnceOnTwoThreadsAsEachAlone) {
    // Two models in one process, each with its own server context and inference loop, asked at the
    // same time on two threads: each must answer what it answers alone -- two different models, so an
    // answer that came from the other one would show. Every answer is the first request of a fresh
    // model, so none of them takes anything from a prompt cache. The two are loaded one after the
    // other here; loads at the same time are the next test.
    const std::string path_a = write_tiny_llama_gguf("jllama-c-api-two-models-a");
    const std::string path_b = write_tiny_llama_gguf(tiny_model_spec::wide(64, "jllama-c-api-two-models-b"));
    ASSERT_FALSE(path_a.empty());
    ASSERT_FALSE(path_b.empty());
    const auto alone = [](const std::string &path) {
        jllama_model *model = load_tiny(path, {});
        EXPECT_NE(model, nullptr) << jllama_last_error();
        const std::string answer = model != nullptr ? greedy_answer(model, 16) : std::string();
        jllama_model_free(model);
        return answer;
    };
    const std::string expected_a = alone(path_a);
    const std::string expected_b = alone(path_b);
    ASSERT_FALSE(expected_a.empty());
    ASSERT_NE(expected_a, expected_b) << "the two models must answer differently";

    jllama_model *model_a = load_tiny(path_a, {});
    ASSERT_NE(model_a, nullptr) << jllama_last_error();
    jllama_model *model_b = load_tiny(path_b, {});
    ASSERT_NE(model_b, nullptr) << jllama_last_error();
    std::atomic<int> ready{0};
    const auto answer_when_both_are_ready = [&ready](jllama_model *model, std::string &answer) {
        ready.fetch_add(1);
        while (ready.load() < 2) {
            std::this_thread::yield();
        }
        answer = greedy_answer(model, 16);
    };
    std::string answer_a;
    std::string answer_b;
    std::thread thread_a([&]() { answer_when_both_are_ready(model_a, answer_a); });
    std::thread thread_b([&]() { answer_when_both_are_ready(model_b, answer_b); });
    thread_a.join();
    thread_b.join();
    EXPECT_EQ(answer_a, expected_a);
    EXPECT_EQ(answer_b, expected_b);
    jllama_model_free(model_a);
    jllama_model_free(model_b);
    std::error_code ignored;
    std::filesystem::remove(path_a, ignored);
    std::filesystem::remove(path_b, ignored);
}

TEST(CApiConcurrency, TwoModelsLoadedAtOnceOnTwoThreadsAnswerAsEachAlone) {
    // Two models in one process, loaded at the same time on two threads and asked at the same time:
    // the loads share common_log, the backend registry and the log lock of the engine's parse, and
    // each model has its own server context. patches/0041: common_fit_params swapped llama.cpp's
    // process-wide logger for the length of a load (common_fit_log_scope, its state on the loading
    // thread's stack), and two loads at once restored the logger of a scope that had ended -- a
    // SIGSEGV in llama_log_internal_v in 4 of 5 runs. Each must answer what it answers loaded alone -- two
    // different models, so an answer that came from the other one would show. Every answer is the
    // first request of a fresh model, so none of them takes anything from a prompt cache.
    const std::string path_a = write_tiny_llama_gguf("jllama-c-api-two-models-a");
    const std::string path_b = write_tiny_llama_gguf(tiny_model_spec::wide(64, "jllama-c-api-two-models-b"));
    ASSERT_FALSE(path_a.empty());
    ASSERT_FALSE(path_b.empty());
    const auto alone = [](const std::string &path) {
        jllama_model *model = load_tiny(path, {});
        EXPECT_NE(model, nullptr) << jllama_last_error();
        const std::string answer = model != nullptr ? greedy_answer(model, 16) : std::string();
        jllama_model_free(model);
        return answer;
    };
    const std::string expected_a = alone(path_a);
    const std::string expected_b = alone(path_b);
    ASSERT_FALSE(expected_a.empty());
    ASSERT_NE(expected_a, expected_b) << "the two models must answer differently";

    std::atomic<int> at_load{0};
    std::atomic<int> at_answer{0};
    const auto meet = [](std::atomic<int> &barrier) {
        barrier.fetch_add(1);
        while (barrier.load() < 2) {
            std::this_thread::yield();
        }
    };
    const auto both = [&](const std::string &path, std::string &answer) {
        meet(at_load);
        jllama_model *model = load_tiny(path, {});
        EXPECT_NE(model, nullptr) << jllama_last_error();
        meet(at_answer); // reached also by a failed load, so the other thread never waits for good
        if (model != nullptr) {
            answer = greedy_answer(model, 16);
            jllama_model_free(model);
        }
    };
    std::string answer_a;
    std::string answer_b;
    std::thread thread_a([&]() { both(path_a, answer_a); });
    std::thread thread_b([&]() { both(path_b, answer_b); });
    thread_a.join();
    thread_b.join();
    EXPECT_EQ(answer_a, expected_a);
    EXPECT_EQ(answer_b, expected_b);
    std::error_code ignored;
    std::filesystem::remove(path_a, ignored);
    std::filesystem::remove(path_b, ignored);
}

TEST(CApiKvUnified, APerSlotLimitIsTheContextOfEverySlotOfAUnifiedCache) {
    // --kv-unified-per-slot (llama.cpp's server option, LLAMA_EXAMPLE_SERVER only): with a unified KV
    // cache every slot may use the whole cache, and the option caps each slot's context
    // (server_context_impl::n_ctx_slot). Two slots over a unified cache of 256 cells get 128 each
    // without it -- the cache, capped at the model's training context of 128 -- and 64 with it, which
    // a prompt of 100 tokens then exceeds.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-kv-per-slot");
    ASSERT_FALSE(path.empty());
    const auto slot_contexts = [](jllama_model *model) {
        std::vector<int> contexts;
        Owned metrics;
        EXPECT_EQ(jllama_metrics(model, &metrics.s), JLLAMA_STATUS_OK) << jllama_last_error();
        if (metrics.s != nullptr) {
            const json parsed = json::parse(metrics.str());
            for (const json &slot : parsed.at("slots")) {
                contexts.push_back(slot.at("n_ctx").get<int>());
            }
        }
        return contexts;
    };
    const json long_prompt = {{"prompt", std::string(100, 'a')}, {"n_predict", 2}, {"temperature", 0}};

    jllama_model *capped = load_tiny(path, {"-np", "2", "--kv-unified", "-c", "256", "--kv-unified-per-slot", "64"});
    ASSERT_NE(capped, nullptr) << jllama_last_error();
    EXPECT_EQ(slot_contexts(capped), (std::vector<int>{64, 64}));
    Owned refused;
    EXPECT_EQ(jllama_completions(capped, long_prompt.dump().c_str(), &refused.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("exceeds the available context size"), std::string::npos)
        << jllama_last_error();
    jllama_model_free(capped);

    jllama_model *uncapped = load_tiny(path, {"-np", "2", "--kv-unified", "-c", "256"});
    ASSERT_NE(uncapped, nullptr) << jllama_last_error();
    EXPECT_EQ(slot_contexts(uncapped), (std::vector<int>{128, 128}));
    Owned answered;
    EXPECT_EQ(jllama_completions(uncapped, long_prompt.dump().c_str(), &answered.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    jllama_model_free(uncapped);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST_F(CApiMusic3, AnOptionsStructOfTheHeaderBeforeGuidanceLeavesGuidanceAndStepsToTheModel) {
    // A caller built against the header of b12453 declares a struct that ends with
    // speaker_reference_text (struct_size == offsetof(jllama_tts_options, cfg_scale)): the library
    // reads only those bytes, so whatever lies behind them in the caller's memory is not taken for
    // guidance or solver steps (b12459), and the model's own -- the defaults, 0 -- apply. Set to values
    // that change the song when read, they must leave it as the defaults sing it.
    const auto sing_with = [](const jllama_tts_options &options, std::vector<unsigned char> *wav) {
        unsigned char *bytes = nullptr;
        size_t length = 0;
        const jllama_status status = jllama_tts_synthesize_options(tts(), "la la la", &options, &bytes, &length);
        if (status == JLLAMA_STATUS_OK) {
            wav->assign(bytes, bytes + length);
        }
        jllama_bytes_free(bytes);
        return status;
    };
    jllama_tts_options options;
    jllama_tts_options_init(&options);
    options.max_frames = 12;
    options.seed = 11;
    ASSERT_EQ(options.cfg_scale, 0.0f);
    ASSERT_EQ(options.flow_steps, 0);
    std::vector<unsigned char> own;
    ASSERT_EQ(sing_with(options, &own), JLLAMA_STATUS_OK) << jllama_last_error();
    options.cfg_scale = 3.0f;
    options.flow_steps = 5;
    std::vector<unsigned char> read;
    ASSERT_EQ(sing_with(options, &read), JLLAMA_STATUS_OK) << jllama_last_error();
    ASSERT_NE(read, own) << "guidance 3 and five steps are not the model's own";
    options.struct_size = offsetof(jllama_tts_options, cfg_scale);
    std::vector<unsigned char> older;
    ASSERT_EQ(sing_with(options, &older), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(older, own) << "the fields behind the declared size were read";
}

TEST_F(CApiVideoDecoder, AVocabOnlyModelReleasesItsDecoderOnceAtItsFree) {
    // A --vocab-only load has no server to hand the decoder to, but jllama_model_load_video takes the
    // decoder over all the same: the context keeps it (jllama_context::video_decoder) and its release
    // runs exactly once, when the model is freed -- not at the load, which succeeded, and not twice.
    FakeVideoDecoder decoder;
    const jllama_video_decoder c_decoder = decoder.c_decoder();
    jllama_model *model = load_tiny_vision(path_, mmproj_, &c_decoder, {"--vocab-only"});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    EXPECT_EQ(decoder.released.load(), 0) << "the load released the decoder of a model it returned";
    int *tokens = nullptr;
    size_t n_tokens = 0;
    EXPECT_EQ(jllama_tokenize(model, "hello", false, &tokens, &n_tokens), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_EQ(n_tokens, 5u);
    jllama_tokens_free(tokens);
    jllama_model_free(model);
    EXPECT_EQ(decoder.released.load(), 1);
    EXPECT_EQ(decoder.opened.load(), 0);
}

TEST(CApiGraphReuse, AHybridSlidingWindowModelReusesItsGraphWithFlashAttention) {
    // llama.cpp b12549 (#32429): the memory input of a model that mixes a recurrent state with a
    // sliding-window and a full attention cache (llm_graph_input_mem_hybrid_iswa: LFM2, LFM3,
    // Inkling2) built its masks dense, while its reuse check expected the kv_rows masks it gets with
    // flash attention over a unified cache -- so the check failed on every token and the graph was
    // built anew each time: "graphs reused" was 0 before b12549 (measured with the two mask lines of
    // build_inp_mem_hybrid_iswa put back: 0). The generator's LFM3 backbone (a short convolution, a
    // sliding-window and a full attention layer) with -fa on and --kv-unified -- without it llama.cpp
    // uses no kv_rows, and the graph was reused before the fix too -- reuses it now, and answers
    // exactly as with reuse turned off (LLAMA_GRAPH_REUSE_DISABLE, read when the context is created),
    // whose count is 0, which shows the count is the reuse's.
    const std::string path = write_tiny_lfm3_gguf("jllama-c-api-graph-reuse");
    ASSERT_FALSE(path.empty());
    const auto generate = [&path](int *reused) {
        *reused = -1;
        LogCapture log;
        jllama_model *model = load_tiny(path, {"-c", "256", "-fa", "on", "--kv-unified"});
        EXPECT_NE(model, nullptr) << jllama_last_error();
        json result;
        if (model != nullptr) {
            Owned completion;
            EXPECT_EQ(jllama_completions(model,
                                         R"({"prompt":"Hello there","n_predict":16,"temperature":0,"ignore_eos":true,)"
                                         R"("return_tokens":true,"n_probs":2})",
                                         &completion.s),
                      JLLAMA_STATUS_OK)
                << jllama_last_error();
            result = completion.s != nullptr ? json::parse(completion.str()) : json();
            jllama_model_free(model);
        }
        log.stop();
        const std::string text = log.text();
        std::smatch match;
        if (std::regex_search(text, match, std::regex("graphs reused =\\s*(\\d+)"))) {
            *reused = std::stoi(match[1].str());
        }
        return result;
    };
    int reused = -1;
    const json with_reuse = generate(&reused);
    ASSERT_TRUE(with_reuse.is_object());
    ASSERT_EQ(with_reuse.at("tokens").size(), 16u) << with_reuse.dump();
    EXPECT_GT(reused, 0) << "the hybrid sliding-window model built its graph anew for every token";
    int reused_off = -1;
    json without_reuse;
    {
        const scoped_env off("LLAMA_GRAPH_REUSE_DISABLE", "1");
        without_reuse = generate(&reused_off);
    }
    ASSERT_TRUE(without_reuse.is_object());
    EXPECT_EQ(reused_off, 0) << "LLAMA_GRAPH_REUSE_DISABLE did not turn the reuse off";
    EXPECT_EQ(with_reuse.at("tokens"), without_reuse.at("tokens"));
    EXPECT_EQ(with_reuse.at("completion_probabilities"), without_reuse.at("completion_probabilities"));
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

namespace {
// the text without the end-of-generation byte 4 at its end: write_tiny_llama_gguf(prefix, false, 4)
// makes that byte the model's EOS, an ordinary byte token whose piece the answer keeps; a grammar
// that is done leaves the model nothing else to write
std::string without_eos_byte(std::string text) {
    if (!text.empty() && text.back() == '\x04') {
        text.pop_back();
    }
    return text;
}

// the single token of one character of the generator's byte vocabulary
int single_token(jllama_model *model, const char *text) {
    int *tokens = nullptr;
    size_t count = 0;
    EXPECT_EQ(jllama_tokenize(model, text, false, &tokens, &count), JLLAMA_STATUS_OK) << jllama_last_error();
    const int token = count == 1 ? tokens[0] : -1;
    jllama_tokens_free(tokens);
    EXPECT_GE(token, 0) << "'" << text << "' is not one token";
    return token;
}

std::string raw_under_grammar(jllama_model *model, const std::string &grammar, const json &logit_bias) {
    json request = {{"prompt", "Hello there"}, {"temperature", 0}, {"n_predict", 8}, {"grammar", grammar}};
    if (!logit_bias.is_null()) {
        request["logit_bias"] = logit_bias;
    }
    Owned completion;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &completion.s), JLLAMA_STATUS_OK)
        << grammar << ": " << jllama_last_error();
    return completion.s != nullptr ? without_eos_byte(json::parse(completion.str()).at("content").get<std::string>())
                                   : std::string();
}
} // namespace

TEST(CApiGrammar, ATokenRuleTakesSetsAndRangesOfIdsAndItsTokensLeaveTheCharacterRules) {
    // b12638 (#29307): a token rule takes several ids and inclusive ranges (<[1,5-9]>, negated
    // !<[...]>), and a token a token rule names is no longer matched by a character rule -- [a-c] does
    // not take the token "b" once the grammar names it. A logit bias makes the biased token the greedy
    // choice wherever the grammar allows it, so each answer below is exact. At b12637 the sets did not
    // parse (the request was refused) and [a-c] took "b": the last answer was "bbb".
    const std::string path = write_tiny_llama_gguf("jllama-c-api-grammar-token-sets", false, 4);
    ASSERT_FALSE(path.empty());
    jllama_model *model = load_tiny(path, {});
    ASSERT_NE(model, nullptr) << jllama_last_error();
    const int a = single_token(model, "a");
    const int b = single_token(model, "b");
    const int c = single_token(model, "c");
    const int x = single_token(model, "x");
    const int z = single_token(model, "z");
    ASSERT_EQ(b, a + 1);
    ASSERT_EQ(c, a + 2);
    const std::string range = std::to_string(a) + "-" + std::to_string(c);
    const std::string set = std::to_string(x) + "," + std::to_string(z);

    // a range and a set, the biased member of each taken
    const std::string sets = "root ::= <[" + range + "]> <[" + set + "]>";
    EXPECT_EQ(raw_under_grammar(model, sets, json::array({json::array({c, 100}), json::array({x, 100})})), "cx");
    EXPECT_EQ(raw_under_grammar(model, sets, json::array({json::array({b, 100}), json::array({z, 100})})), "bz");
    // without a bias the answer is still a member of each
    const std::string free_answer = raw_under_grammar(model, sets, nullptr);
    ASSERT_EQ(free_answer.size(), 2u) << free_answer;
    EXPECT_NE(std::string("abc").find(free_answer[0]), std::string::npos) << free_answer;
    EXPECT_NE(std::string("xz").find(free_answer[1]), std::string::npos) << free_answer;
    // a negated set refuses the biased token it names
    const std::string negated =
        raw_under_grammar(model, "root ::= !<[" + range + "]>", json::array({json::array({b, 100})}));
    ASSERT_FALSE(negated.empty());
    EXPECT_EQ(std::string("abc").find(negated[0]), std::string::npos) << negated;
    // the token a token rule names leaves the character rules: [a-c] takes a and c only
    const std::string opaque = raw_under_grammar(model, "root ::= [a-c] [a-c] <[" + std::to_string(b) + "]>",
                                                 json::array({json::array({b, 100})}));
    ASSERT_EQ(opaque.size(), 3u) << opaque;
    EXPECT_NE(opaque[0], 'b') << opaque;
    EXPECT_NE(opaque[1], 'b') << opaque;
    EXPECT_EQ(opaque[2], 'b') << opaque;
    // and a grammar that does not name it keeps the token in the character rule
    EXPECT_EQ(raw_under_grammar(model, "root ::= [a-c] [a-c] [a-c]", json::array({json::array({b, 100})})), "bbb");
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiChatTokens, AThinkingTagEndsTheReasoningOnlyAsItsTokenNotSpelledOut) {
    // b12638 (#29307) parses the Qwen3-Coder format (also Qwen3.5, Nemotron Nano 3) by token: with
    // the thinking tags control tokens of the vocabulary, as in those models, the reasoning ends at the
    // token </think>, and the same characters written out are text. A grammar fixes what the model
    // writes: the tags as tokens give reasoning "plan" and content "answer"; the end tag spelled out
    // (the token </think> is named in an unused rule, so the character rule cannot take it) ends
    // nothing, and the text keeps it. Parsed by text, at b12637, the second answer split like the first.
    const std::string path = write_tiny_llama_gguf("jllama-c-api-chat-tokens", false, 4,
                                                   {"<think>", "</think>", "<tool_call>", "</tool_call>"});
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{#- a Qwen3-Coder template: <tool_call> <function= <parameter= <think> -#}"
                             "{%- for message in messages -%}"
                             "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
                             "{%- endfor -%}"
                             "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";
    jllama_model *model = load_with_chat_template(path, tmpl);
    ASSERT_NE(model, nullptr);
    const int think = N_VOCAB;
    const int end_think = N_VOCAB + 1;
    const auto chat = [&](const std::string &grammar) {
        const json request = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                              {"temperature", 0},
                              {"n_predict", 30},
                              {"grammar", grammar}};
        Owned response;
        EXPECT_EQ(jllama_chat_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()).at("choices").at(0).at("message") : json();
    };
    const json tokens =
        chat("root ::= <[" + std::to_string(think) + "]> \"plan\" <[" + std::to_string(end_think) + "]> \"answer\"");
    EXPECT_EQ(tokens.value("reasoning_content", std::string()), "plan") << tokens.dump();
    EXPECT_EQ(without_eos_byte(tokens.value("content", std::string())), "answer") << tokens.dump();
    const json spelled = chat("root ::= <[" + std::to_string(think) + "]> \"plan</think>answer\"\nunused ::= <[" +
                              std::to_string(end_think) + "]>");
    const std::string text =
        spelled.value("reasoning_content", std::string()) + spelled.value("content", std::string());
    EXPECT_NE(text.find("plan</think>answer"), std::string::npos) << spelled.dump();
    EXPECT_NE(without_eos_byte(spelled.value("content", std::string())), "answer") << spelled.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(CApiChatTokens, AQwen3CoderToolCallStartsAtItsTokenAndIsParsedIntoTheCall) {
    // The Qwen3-Coder format's tool calls since b12638 (#29307): the grammar opens a call at the token
    // <tool_call> and closes it at </tool_call> (token rules over the vocabulary's control tokens), the
    // argument between them by the parameter's schema. With tool_choice "required" and a logit bias on
    // <tool_call>, the greedy model calls the one tool; a boolean parameter leaves it two answers, and
    // a bias on the end-of-generation byte ends the answer after the call. The parser turns it into
    // the OpenAI tool call with the boolean as its argument. (This holds the format the bump moved
    // onto tokens; parsed by text it gave the same call.)
    const std::string path = write_tiny_llama_gguf("jllama-c-api-qwen3-coder-tools", false, 4,
                                                   {"<think>", "</think>", "<tool_call>", "</tool_call>"});
    ASSERT_FALSE(path.empty());
    const std::string tmpl = "{#- a Qwen3-Coder template without thinking: <tool_call> <function= <parameter= -#}"
                             "{%- for message in messages -%}"
                             "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
                             "{%- endfor -%}"
                             "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}";
    jllama_model *model = load_with_chat_template(path, tmpl);
    ASSERT_NE(model, nullptr);
    const int tool_call = N_VOCAB + 2;
    const int eos = single_token(model, "\x04");
    const json tool = {{"type", "function"},
                       {"function",
                        {{"name", "set"},
                         {"description", "sets the flag"},
                         {"parameters",
                          {{"type", "object"},
                           {"properties", {{"flag", {{"type", "boolean"}}}}},
                           {"required", json::array({"flag"})}}}}}};
    const json request = {{"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
                          {"tools", json::array({tool})},
                          {"tool_choice", "required"},
                          {"temperature", 0},
                          {"n_predict", 200}, // every byte is a token of its own: the call alone is about 60
                          {"logit_bias", json::array({json::array({tool_call, 100}), json::array({eos, 100})})}};
    Owned response;
    ASSERT_EQ(jllama_chat_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json choice = json::parse(response.str()).at("choices").at(0);
    EXPECT_EQ(choice.value("finish_reason", std::string()), "tool_calls") << choice.dump();
    const json calls = choice.at("message").value("tool_calls", json::array());
    ASSERT_EQ(calls.size(), 1u) << choice.dump();
    EXPECT_EQ(calls.at(0).at("function").value("name", std::string()), "set") << choice.dump();
    const json arguments = json::parse(calls.at(0).at("function").value("arguments", std::string("{}")));
    ASSERT_TRUE(arguments.contains("flag")) << choice.dump();
    EXPECT_TRUE(arguments.at("flag").is_boolean()) << choice.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

#ifdef JLLAMA_HAS_NATIVE_SERVER
TEST(CApiServerTools, TheShellToolAsksForADescriptionFirstAndRunsWithoutOne) {
    // b12667 (#32894): exec_shell_command's schema asks for a short `description` of the command, as its
    // first and a required property, so that a client can show it while the call streams; the tool
    // ignores it, and a call without it still runs. At b12666 the schema had no such property.
    const std::filesystem::path models =
        std::filesystem::temp_directory_path() / ("jllama-c-api-shell-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(models);
    const std::string models_dir = models.string();
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    probe_listener.join();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server",     "--host",          "127.0.0.1",
                          "--port",           port_text.c_str(), "--models-dir",
                          models_dir.c_str(), "--tools",         "exec_shell_command"};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    httplib::Client client("127.0.0.1", port);
    httplib::Result health;
    for (int i = 0; i < 100 && !(health && health->status == 200); i++) {
        health = client.Get("/health");
        if (!(health && health->status == 200)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ASSERT_TRUE(health && health->status == 200);

    const auto listed = client.Get("/tools");
    ASSERT_TRUE(listed && listed->status == 200) << (listed ? listed->body : httplib::to_string(listed.error()));
    // ordered: the order of the properties is the point
    const auto tools = nlohmann::ordered_json::parse(listed->body);
    nlohmann::ordered_json parameters;
    for (const auto &tool : tools) {
        if (tool.value("tool", std::string()) == "exec_shell_command") {
            parameters = tool.at("definition").at("function").at("parameters");
        }
    }
    ASSERT_TRUE(parameters.is_object()) << listed->body;
    ASSERT_FALSE(parameters.at("properties").empty()) << parameters.dump();
    EXPECT_EQ(parameters.at("properties").begin().key(), "description") << parameters.dump();
    const auto required = parameters.at("required");
    EXPECT_NE(std::find(required.begin(), required.end(), "description"), required.end()) << parameters.dump();
    EXPECT_NE(std::find(required.begin(), required.end(), "command"), required.end()) << parameters.dump();

    // the tool still runs a call without the description
    const json body = {{"tool", "exec_shell_command"}, {"params", {{"command", "echo jllama-shell-tool"}}}};
    const auto ran = client.Post("/tools", body.dump(), "application/json");
    ASSERT_TRUE(ran) << httplib::to_string(ran.error());
    EXPECT_EQ(ran->status, 200) << ran->body;
    EXPECT_NE(ran->body.find("jllama-shell-tool"), std::string::npos) << ran->body;

    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ignored;
    std::filesystem::remove_all(models, ignored);
}

#ifndef _WIN32
TEST(CApiServerTools, AnMcpToolResultCarriesItsImageAndAudioAsDataUris) {
    // A tool of an MCP server (--mcp-servers-json, a subprocess speaking JSON-RPC over stdio) is called
    // through POST /tools like a built-in one; its result's content parts are joined into
    // plain_text_response, one line each. Since b12708 (#32064) an image or audio part is a line of its
    // own as a data URI, its base64 joined when the server wrapped it; a part of another type is still
    // left out. At b12707 only the text part was kept. The stand-in MCP server is a POSIX shell script
    // that answers initialize, tools/list (one tool, "picture") and tools/call by the request's id.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-mcp-" + std::to_string(std::random_device{}()));
    const std::filesystem::path models = dir / "models";
    std::filesystem::create_directories(models);
    const std::filesystem::path script = dir / "mcp-stand-in.sh";
    {
        std::ofstream s(script, std::ios::binary);
        s << "while IFS= read -r line; do\n"
             "  id=$(printf '%s' \"$line\" | sed -n 's/.*\"id\":\\([0-9][0-9]*\\).*/\\1/p')\n"
             "  case \"$line\" in\n"
             "    *'\"method\":\"initialize\"'*)\n"
             "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2024-11-05\","
             "\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"stand-in\",\"version\":\"1\"}}}\\n' "
             "\"$id\";;\n"
             "    *'\"method\":\"tools/list\"'*)\n"
             "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"tools\":[{\"name\":\"picture\","
             "\"description\":\"draws a picture\",\"inputSchema\":{\"type\":\"object\",\"properties\":{}}}]}}\\n' "
             "\"$id\";;\n"
             "    *'\"method\":\"tools/call\"'*)\n"
             "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":["
             "{\"type\":\"text\",\"text\":\"here it is\"},"
             "{\"type\":\"image\",\"mimeType\":\"image/png\",\"data\":\"iVBORw0K\\\\nGgo=\"},"
             "{\"type\":\"audio\",\"mimeType\":\"audio/wav\",\"data\":\"UklGRg==\"},"
             "{\"type\":\"resource\",\"resource\":{\"uri\":\"file:///x\"}}]}}\\n' \"$id\";;\n"
             "  esac\n"
             "done\n";
    }
    const json mcp = {
        {"mcpServers", {{"standin", {{"command", "/bin/sh"}, {"args", json::array({script.string()})}}}}}};
    const std::string mcp_text = mcp.dump();
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    probe_listener.join();
    const std::string port_text = std::to_string(port);
    const std::string models_dir = models.string();
    const char *argv[] = {"llama-server", "--host",           "127.0.0.1",          "--port",        port_text.c_str(),
                          "--models-dir", models_dir.c_str(), "--mcp-servers-json", mcp_text.c_str()};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    httplib::Client client("127.0.0.1", port);
    httplib::Result health;
    for (int i = 0; i < 100 && !(health && health->status == 200); i++) {
        health = client.Get("/health");
        if (!(health && health->status == 200)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ASSERT_TRUE(health && health->status == 200);

    const auto listed = client.Get("/tools");
    ASSERT_TRUE(listed && listed->status == 200) << (listed ? listed->body : httplib::to_string(listed.error()));
    bool found = false;
    for (const auto &tool : json::parse(listed->body)) {
        if (tool.value("tool", std::string()) == "standin_picture") {
            found = true;
            EXPECT_EQ(tool.value("type", std::string()), "mcp") << listed->body;
        }
    }
    ASSERT_TRUE(found) << listed->body;

    const json body = {{"tool", "standin_picture"}, {"params", json::object()}};
    const auto called = client.Post("/tools", body.dump(), "application/json");
    ASSERT_TRUE(called) << httplib::to_string(called.error());
    EXPECT_EQ(called->status, 200) << called->body;
    const json result = json::parse(called->body);
    EXPECT_EQ(result.value("plain_text_response", std::string()),
              "here it is\ndata:image/png;base64,iVBORw0KGgo=\ndata:audio/wav;base64,UklGRg==")
        << called->body;

    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}
#endif

TEST(CApiServerTools, GrepSearchPagesItsMatchesByOffsetInTheOrderOfTheFiles) {
    // The embedded server's built-in tools (--tools, POST /tools): since b12640 (#32792) grep_search
    // takes an offset, the number of matching lines to skip, searches the files of a directory in the
    // order of their paths, and ends a full page of 100 matches with the offset to continue at. Two
    // files give 150 matches: a.txt 90 (every second of its lines), b.txt 60. The first page is a.txt's
    // 90 and b.txt's first 10, the second the other 50 of b.txt and no continuation. At b12639 the
    // offset was ignored and the second call answered the first page again.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("jllama-c-api-grep-" + std::to_string(std::random_device{}()));
    const std::filesystem::path data = dir / "data";
    std::filesystem::create_directories(data);
    {
        // binary: the tool reads a file as bytes and keeps a '\r' in its line, so a text-mode
        // stream on Windows would end every answered line with "\r\n"
        std::ofstream b(data / "b.txt", std::ios::binary);
        for (int i = 0; i < 60; ++i) {
            b << "match b " << i << "\n";
        }
        std::ofstream a(data / "a.txt", std::ios::binary);
        for (int i = 0; i < 90; ++i) {
            a << "match a " << i << "\nother line\n";
        }
    }
    const std::filesystem::path models = dir / "models";
    std::filesystem::create_directories(models);
    const std::string models_dir = models.string();
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread probe_listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    probe_listener.join();
    const std::string port_text = std::to_string(port);
    const char *argv[] = {"llama-server", "--host",           "127.0.0.1", "--port",     port_text.c_str(),
                          "--models-dir", models_dir.c_str(), "--tools",   "grep_search"};
    jllama_server *server = nullptr;
    ASSERT_EQ(jllama_server_start(9, argv, &server), JLLAMA_STATUS_OK) << jllama_last_error();
    httplib::Client client("127.0.0.1", port);
    httplib::Result health;
    for (int i = 0; i < 100 && !(health && health->status == 200); i++) {
        health = client.Get("/health");
        if (!(health && health->status == 200)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ASSERT_TRUE(health && health->status == 200);
    const auto search = [&](const json &offset) {
        json params = {{"path", data.string()}, {"pattern", "^match"}};
        if (!offset.is_null()) {
            params["offset"] = offset;
        }
        const json body = {{"tool", "grep_search"}, {"params", params}};
        const auto res = client.Post("/tools", body.dump(), "application/json");
        EXPECT_TRUE(res) << httplib::to_string(res.error());
        if (!res) {
            return std::string();
        }
        EXPECT_EQ(res->status, 200) << res->body;
        const json answer = json::parse(res->body, nullptr, false);
        return answer.is_object() ? answer.value("plain_text_response", std::string()) : res->body;
    };
    const auto count = [](const std::string &text, const std::string &what) {
        size_t n = 0;
        for (size_t pos = text.find(what); pos != std::string::npos; pos = text.find(what, pos + 1)) {
            ++n;
        }
        return n;
    };
    const std::string first = search(nullptr);
    EXPECT_NE(first.find("Total matches: 100"), std::string::npos) << first;
    EXPECT_EQ(count(first, "match a "), 90u) << first;
    EXPECT_EQ(count(first, "match b "), 10u) << first;
    EXPECT_NE(first.find("match b 9\n"), std::string::npos) << first;
    EXPECT_NE(first.find("Continue with offset=100"), std::string::npos) << first;
    EXPECT_LT(first.find("match a 89"), first.find("match b 0")) << "the files are not in the order of their paths";

    const std::string second = search(100);
    EXPECT_NE(second.find("Total matches: 50"), std::string::npos) << second;
    EXPECT_EQ(count(second, "match a "), 0u) << second;
    EXPECT_EQ(count(second, "match b "), 50u) << second;
    EXPECT_NE(second.find("match b 10\n"), std::string::npos) << second;
    EXPECT_EQ(second.find("Continue with offset="), std::string::npos) << second;

    // an offset past the last match finds none, and a negative one counts as 0
    EXPECT_NE(search(150).find("Total matches: 0"), std::string::npos);
    EXPECT_NE(search(-5).find("Total matches: 100"), std::string::npos);
    EXPECT_EQ(jllama_server_stop(server), JLLAMA_STATUS_OK) << jllama_last_error();
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}
#endif // JLLAMA_HAS_NATIVE_SERVER
