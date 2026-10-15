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
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace {

using nlohmann::json;

constexpr int N_VOCAB = 256; // one GPT-2 byte token per byte
constexpr int N_EMBD = 32;
constexpr int N_HEAD = 4;
constexpr int N_HEAD_KV = 4;
constexpr int HEAD_DIM = 8;
constexpr int N_FF = 64;
constexpr int N_LAYER = 2;
constexpr int N_CTX_TRAIN = 128;

// The 256 printable code points GPT-2's byte-level BPE maps the bytes to, as UTF-8.
std::vector<std::string> byte_tokens() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b)
        bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b)
        bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b)
        bs.push_back(b);
    std::vector<int> cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + extra++);
        }
    }
    std::vector<std::string> tokens(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        const int cp = cs[i];
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        }
        tokens[bs[i]] = s;
    }
    return tokens;
}

ggml_tensor *add_random_tensor(gguf_context *g, ggml_context *ctx, std::mt19937 &rng, const std::string &name,
                               std::initializer_list<int64_t> ne, float scale) {
    const std::vector<int64_t> d(ne);
    ggml_tensor *t = d.size() == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d[0])
                                   : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d[0], d[1]);
    std::normal_distribution<float> dist(0.0f, scale);
    auto *data = (float *)t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        data[i] = d.size() == 1 ? 1.0f : dist(rng); // norms are ones, weights small noise
    }
    ggml_set_name(t, name.c_str());
    gguf_add_tensor(g, t);
    return t;
}

// Keep the generated tokens printable ASCII (never BOS/EOS): the output rows of every other
// byte are scaled to almost nothing, so greedy sampling over random logits lands on an ASCII
// byte with overwhelming probability. That matters for the streaming test -- a byte >= 0x80 is
// an incomplete UTF-8 sequence, which the server holds back instead of sending as a partial
// result, and a random model that happens to emit only those streams no partial chunk at all.
void confine_output_to_ascii(ggml_tensor *output) {
    auto *data = (float *)output->data;
    for (int64_t tok = 0; tok < output->ne[1]; ++tok) {
        const bool printable = tok >= ' ' && tok < 0x7F && tok != '<' && tok != '>';
        if (!printable) {
            for (int64_t i = 0; i < output->ne[0]; ++i) {
                data[tok * output->ne[0] + i] *= 1e-3f;
            }
        }
    }
}

// A two-layer llama with random weights and the GPT-2 byte tokenizer: enough for the server to
// load it, apply its fallback chat template and generate tokens.
std::string write_tiny_llama_gguf() {
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("jllama-c-api-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(42);
    gguf_context *g = gguf_init_empty();
    const std::string a = "llama";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".context_length").c_str(), N_CTX_TRAIN);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), N_EMBD);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), N_LAYER);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), N_FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD_KV);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(g, (a + ".rope.dimension_count").c_str(), HEAD_DIM);

    const std::vector<std::string> tokens = byte_tokens();
    std::vector<const char *> token_ptrs;
    for (const std::string &s : tokens) {
        token_ptrs.push_back(s.c_str());
    }
    std::vector<int32_t> token_types(N_VOCAB, 1); // LLAMA_TOKEN_TYPE_NORMAL
    const char *no_merges[] = {"Ġ Ġ"};
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", no_merges, 1);
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", '<');
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", '>');

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {N_EMBD, N_VOCAB}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {N_EMBD}, 1.0f);
    confine_output_to_ascii(add_random_tensor(g, ctx, rng, "output.weight", {N_EMBD, N_VOCAB}, 0.1f));
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add_random_tensor(g, ctx, rng, p + "attn_norm.weight", {N_EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "attn_q.weight", {N_EMBD, N_HEAD * HEAD_DIM}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_k.weight", {N_EMBD, N_HEAD_KV * HEAD_DIM}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_v.weight", {N_EMBD, N_HEAD_KV * HEAD_DIM}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_output.weight", {N_HEAD * HEAD_DIM, N_EMBD}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_norm.weight", {N_EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_gate.weight", {N_EMBD, N_FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {N_EMBD, N_FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {N_FF, N_EMBD}, 0.1f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

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
