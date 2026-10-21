// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The runnable guard for patches/0021: a slot's prompt cache is computed under the LoRA adapters
// the slot ran with, so it must not survive a change of those adapters. Upstream clears it when a
// request names its own adapters (lora_should_clear_cache) but not when the adapters come from
// params_base -- changed by SET_LORA (LlamaModel.setLoraAdapter, POST /lora-adapters) or restored
// after a request with its own adapters. The slot then went on with KV entries computed under the
// old scales and answered like neither model: measured at b11639 with stories15M_MOE and its
// Shakespeare adapter (J9 of the bug-hunt measurement brief, F67).
//
// Hermetic: a tiny random llama model and a random LoRA adapter written here with the public gguf
// API. The adapter touches K, V and the FFN output of every layer, so the KV cache depends on it.
// Greedy decoding makes every answer deterministic, and the reference for "the model with these
// adapters" is the same request on an emptied slot.

#include "jllama_api.h"
#include "tiny_llama_gguf.hpp"

#include "ggml.h"
#include "gguf.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

using nlohmann::json;
using namespace jllama_test;

constexpr int LORA_RANK = 4;

// A LoRA adapter for write_tiny_llama_gguf()'s model. lora_a is {in, rank}, lora_b {rank, out} in
// ggml order (llama-adapter.cpp validates exactly that); alpha = rank keeps the scale at 1.
std::string write_tiny_lora_gguf() {
    static const unsigned run_id = std::random_device{}();
    const std::string path =
        (std::filesystem::temp_directory_path() / ("jllama-lora-cache-adapter-" + std::to_string(run_id) + ".gguf"))
            .string();
    std::mt19937 rng(7);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", static_cast<float>(LORA_RANK));

    ggml_init_params ip = {4u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char *name : {"attn_k", "attn_v"}) {
            add_random_tensor(g, ctx, rng, p + name + ".weight.lora_a", {N_EMBD, LORA_RANK}, 1.0f);
            add_random_tensor(g, ctx, rng, p + name + ".weight.lora_b", {LORA_RANK, N_HEAD_KV * HEAD_DIM}, 1.0f);
        }
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight.lora_a", {N_FF, LORA_RANK}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight.lora_b", {LORA_RANK, N_EMBD}, 1.0f);
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

// One slot, so every request lands on the same cache; long enough a prompt that most of it is
// reused from the cache when it is sent again.
const char *PROMPT = "The quick brown fox jumps over the lazy dog while the cache keeps every token";

class LoraCache : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path_ = write_tiny_llama_gguf("jllama-lora-cache-model");
        adapter_path_ = write_tiny_lora_gguf();
        ASSERT_FALSE(model_path_.empty());
        ASSERT_FALSE(adapter_path_.empty());
        // --cache-ram 0: no host-memory prompt cache, so the slot's own KV cache is the only one.
        const std::vector<std::string> args = {"jllama", "-m",          model_path_, "--lora",     adapter_path_, "-c",
                                               "128",    "-np",         "1",         "-ngl",       "0",           "-t",
                                               "2",      "--cache-ram", "0",         "--no-warmup"};
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
        std::filesystem::remove(model_path_, ignored);
        std::filesystem::remove(adapter_path_, ignored);
    }
    void SetUp() override {
        ASSERT_NE(model_, nullptr);
        set_scale(1.0f);
        erase_slot();
    }

    static json complete(const json &extra = json::object()) {
        json request = {{"prompt", PROMPT}, {"n_predict", 8},       {"temperature", 0},
                        {"top_k", 1},       {"cache_prompt", true}, {"id_slot", 0}};
        request.update(extra);
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return json::parse(response.str());
    }
    static void set_scale(float scale) {
        Owned response;
        const json adapters = json::array({{{"id", 0}, {"scale", scale}}});
        ASSERT_EQ(jllama_set_lora_adapters(model_, adapters.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void erase_slot() {
        Owned response;
        ASSERT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_ERASE, 0, nullptr, &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    // How many prompt tokens the request took from the slot's cache (upstream's n_past). Not
    // tokens_cached: that is every token the slot holds afterwards, positive after any request.
    static int cache_n(const json &response) { return response.at("timings").at("cache_n").get<int>(); }
    // The answer of the model with the current adapters, computed on an empty slot.
    static std::string fresh_answer() {
        erase_slot();
        return complete().at("content").get<std::string>();
    }

    static std::string model_path_;
    static std::string adapter_path_;
    static jllama_model *model_;
};
std::string LoraCache::model_path_;
std::string LoraCache::adapter_path_;
jllama_model *LoraCache::model_ = nullptr;

} // namespace

// Without these two, the tests below could pass for the wrong reason: an adapter that changes
// nothing, or a slot that never reuses its cache.
TEST_F(LoraCache, TheAdapterChangesTheAnswerAndTheSlotReusesItsCache) {
    const std::string with_adapter = fresh_answer();
    const json again = complete();
    EXPECT_GT(cache_n(again), 0) << again.dump();
    EXPECT_LT(again.at("timings").at("prompt_n").get<int>(), again.at("tokens_evaluated").get<int>()) << again.dump();
    EXPECT_EQ(again.at("content").get<std::string>(), with_adapter);

    set_scale(0.0f);
    EXPECT_NE(fresh_answer(), with_adapter);
}

TEST_F(LoraCache, ASetLoraChangeIsNotAnsweredFromTheOldCache) {
    complete(); // the slot's cache is now computed with the adapter at scale 1
    set_scale(0.0f);
    const json after_change = complete();

    EXPECT_EQ(cache_n(after_change), 0) << after_change.dump();
    EXPECT_EQ(after_change.at("content").get<std::string>(), fresh_answer());
}

TEST_F(LoraCache, ASetLoraChangeBackIsNotAnsweredFromTheOldCache) {
    set_scale(0.0f);
    complete(); // cache computed without the adapter
    set_scale(1.0f);
    const json after_change = complete();

    EXPECT_EQ(cache_n(after_change), 0) << after_change.dump();
    EXPECT_EQ(after_change.at("content").get<std::string>(), fresh_answer());
}

TEST_F(LoraCache, TheServerAdaptersAfterARequestWithItsOwnAreNotAnsweredFromItsCache) {
    // A request with its own adapters (scale 0) fills the cache; the next request names none, so it
    // runs with the server's adapters (scale 1) and must not reuse what the first one computed.
    complete({{"lora", json::array({{{"id", 0}, {"scale", 0.0}}})}});
    const json after_change = complete();

    EXPECT_EQ(cache_n(after_change), 0) << after_change.dump();
    EXPECT_EQ(after_change.at("content").get<std::string>(), fresh_answer());
}

namespace {

// Two slots and a slot-save directory, for the one case the shared fixture above cannot reach: a
// slot that has never run a task. Its adapters must already be the server's, so a state restored
// into it is kept like on any other slot (patches/0021, init() hunk).
class LoraCacheFreshSlot : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path_ = write_tiny_llama_gguf("jllama-lora-cache-fresh-model");
        adapter_path_ = write_tiny_lora_gguf();
        save_dir_ = (std::filesystem::temp_directory_path() /
                     ("jllama-lora-cache-slots-" + std::to_string(std::random_device{}())))
                        .string();
        ASSERT_FALSE(model_path_.empty());
        ASSERT_FALSE(adapter_path_.empty());
        ASSERT_TRUE(std::filesystem::create_directories(save_dir_));
        const std::vector<std::string> args = {
            "jllama",  "-m",         model_path_, "--lora", adapter_path_, "-c",          "256", "-np",
            "2",       "-ngl",       "0",         "-t",     "2",           "--cache-ram", "0",   "--slot-save-path",
            save_dir_, "--no-warmup"};
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
        std::filesystem::remove(model_path_, ignored);
        std::filesystem::remove(adapter_path_, ignored);
        std::filesystem::remove_all(save_dir_, ignored);
    }

    static json complete_on(int slot) {
        const json request = {{"prompt", PROMPT}, {"n_predict", 8},       {"temperature", 0},
                              {"top_k", 1},       {"cache_prompt", true}, {"id_slot", slot}};
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return json::parse(response.str());
    }
    static void slot_action(jllama_slot_op op, int slot, const char *filename) {
        Owned response;
        ASSERT_EQ(jllama_slot_action(model_, op, slot, filename, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    }

    static std::string model_path_;
    static std::string adapter_path_;
    static std::string save_dir_;
    static jllama_model *model_;
};
std::string LoraCacheFreshSlot::model_path_;
std::string LoraCacheFreshSlot::adapter_path_;
std::string LoraCacheFreshSlot::save_dir_;
jllama_model *LoraCacheFreshSlot::model_ = nullptr;

} // namespace

TEST_F(LoraCacheFreshSlot, AStateRestoredIntoANeverUsedSlotIsKept) {
    ASSERT_NE(model_, nullptr);
    const json first = complete_on(0);
    // The C API resolves the file name under --slot-save-path, as upstream's slots route does;
    // TearDownTestSuite's remove_all(save_dir_) takes the file away again.
    slot_action(JLLAMA_SLOT_SAVE, 0, "slot0.bin");
    ASSERT_TRUE(std::filesystem::exists(std::filesystem::path(save_dir_) / "slot0.bin"))
        << "the saved state is not under " << save_dir_;
    slot_action(JLLAMA_SLOT_RESTORE, 1, "slot0.bin"); // slot 1 has never run a task

    const json restored = complete_on(1);

    // Computed under the same adapters as the restored state, so all of the prompt but the token
    // the slot always re-evaluates comes from it; a slot that took this for an adapter change
    // cleared it and processed the whole prompt again.
    const int prompt_n = restored.at("timings").at("prompt_n").get<int>();
    const int evaluated = restored.at("tokens_evaluated").get<int>();
    EXPECT_LE(prompt_n, 2) << "of " << evaluated << " prompt tokens: " << restored.dump();
    EXPECT_EQ(restored.at("content").get<std::string>(), first.at("content").get<std::string>());
}
