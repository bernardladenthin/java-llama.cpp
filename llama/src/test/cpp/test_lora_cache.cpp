// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The runnable guard for patches/0021, and for upstream's cache checks behind it: a slot's prompt
// cache is computed under the LoRA adapters the slot ran with, so it must not survive a change of
// those adapters. Until b12416 upstream cleared it only when a request named its own adapters, not
// when they came from params_base -- changed by SET_LORA (LlamaModel.setLoraAdapter,
// POST /lora-adapters) or restored after a request with its own adapters -- and the slot went on with
// KV entries computed under the old scales (measured at b11639 with stories15M_MOE and its
// Shakespeare adapter, F67); upstream #30367 and #32434 (b12498) compare them since. What upstream
// still lacks is the config of a state restored from a slot file, which patches/0021 writes into
// the file as a section of its own.
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

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;
using namespace jllama_test;

constexpr int LORA_RANK = 4;

// A LoRA adapter for write_tiny_llama_gguf()'s model. lora_a is {in, rank}, lora_b {rank, out} in
// ggml order (llama-adapter.cpp validates exactly that); alpha = rank keeps the scale at 1. Another
// name and seed give a second adapter of other weights; the defaults write the one every test used
// before there were two.
std::string write_tiny_lora_gguf(const std::string &name = "adapter", unsigned seed = 7) {
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("jllama-lora-cache-" + name + "-" + std::to_string(run_id) + ".gguf"))
                                 .string();
    std::mt19937 rng(seed);
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
    // How many prompt tokens the request evaluated.
    static int prompt_n(const json &response) { return response.at("timings").at("prompt_n").get<int>(); }
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
// nothing, or a slot that never reuses its cache. The reuse is compared between two requests over the
// cache, not with the fresh answer: a request over the cache re-evaluates only the last prompt token,
// in a batch of its own (upstream [TAG_PROMPT_LOGITS]), which does not reproduce the logits of the
// fresh answer's one batch bit for bit, so a greedy near tie could flip between the two (F272). The
// two requests over the cache take the same tokens from it and compute the same thing.
TEST_F(LoraCache, TheAdapterChangesTheAnswerAndTheSlotReusesItsCache) {
    const std::string with_adapter = fresh_answer();
    const json again = complete();
    EXPECT_GT(cache_n(again), 0) << again.dump();
    EXPECT_LT(again.at("timings").at("prompt_n").get<int>(), again.at("tokens_evaluated").get<int>()) << again.dump();
    const json once_more = complete();
    EXPECT_GT(cache_n(once_more), 0) << once_more.dump();
    EXPECT_LT(prompt_n(once_more), once_more.at("tokens_evaluated").get<int>()) << once_more.dump();
    EXPECT_EQ(cache_n(once_more), cache_n(again)) << once_more.dump() << "\n" << again.dump();
    EXPECT_EQ(prompt_n(once_more), prompt_n(again)) << once_more.dump() << "\n" << again.dump();
    EXPECT_EQ(once_more.at("content").get<std::string>(), again.at("content").get<std::string>());

    set_scale(0.0f);
    EXPECT_NE(fresh_answer(), with_adapter);
}

// jllama_set_lora_adapters answers upstream's {"success": true}, not the list (the header said
// otherwise once), and an adapter the array does not name gets scale 0, as POST /lora-adapters does.
TEST_F(LoraCache, SettingTheAdaptersAnswersSuccessAndAnAdapterNotNamedGetsScaleZero) {
    const auto scale_of_adapter_0 = []() {
        Owned list;
        EXPECT_EQ(jllama_lora_adapters(model_, &list.s), JLLAMA_STATUS_OK) << jllama_last_error();
        const json adapters = json::parse(list.str());
        EXPECT_EQ(adapters.size(), 1u) << adapters.dump();
        return adapters.at(0).at("scale").get<float>();
    };
    Owned response;
    ASSERT_EQ(jllama_set_lora_adapters(model_, R"([{"id":0,"scale":0.5}])", &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(response.str()), json({{"success", true}})) << response.str();
    EXPECT_FLOAT_EQ(scale_of_adapter_0(), 0.5f);
    Owned none;
    ASSERT_EQ(jllama_set_lora_adapters(model_, "[]", &none.s), JLLAMA_STATUS_OK) << jllama_last_error();
    EXPECT_FLOAT_EQ(scale_of_adapter_0(), 0.0f) << "an adapter the array does not name";
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

// Two slots and a slot-save directory: a state restored from a slot file is labelled with the LoRA
// config it was saved with (patches/0021), into a slot that has never run a task as into any other.
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
    // Another prompt on `slot`, so that the slot holds other tokens and a label of the current config
    // before a restore: a restore that left either alone would show.
    static void complete_other_on(int slot) {
        const json request = {{"prompt", "A different prompt, so the slot holds other tokens"},
                              {"n_predict", 4},
                              {"temperature", 0},
                              {"cache_prompt", true},
                              {"id_slot", slot}};
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    static void slot_action(jllama_slot_op op, int slot, const char *filename) {
        Owned response;
        ASSERT_EQ(jllama_slot_action(model_, op, slot, filename, &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    }
    static void set_scale(float scale) {
        Owned response;
        const json adapters = json::array({{{"id", 0}, {"scale", scale}}});
        ASSERT_EQ(jllama_set_lora_adapters(model_, adapters.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
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

TEST_F(LoraCacheFreshSlot, ARequestWithoutASlotGoesToTheSlotOfItsLoraConfig) {
    // b12498 (#32434): the slot scheduler skips a slot whose state another LoRA config computed, which
    // the task would clear. Two slots hold the same prompt, one computed at scale 1, one at scale 0; a
    // request that names no slot takes the slot of its own config and reuses its cache. Without the
    // check the first slot of the best similarity won whatever its config, and the request cleared it.
    set_scale(1.0f);
    slot_action(JLLAMA_SLOT_ERASE, 0, nullptr);
    slot_action(JLLAMA_SLOT_ERASE, 1, nullptr);
    const auto complete = [](int slot, const json &lora) {
        json request = {{"prompt", PROMPT}, {"n_predict", 8}, {"temperature", 0}, {"top_k", 1}, {"cache_prompt", true}};
        if (slot >= 0) {
            request["id_slot"] = slot;
        }
        if (!lora.is_null()) {
            request["lora"] = lora;
        }
        Owned response;
        EXPECT_EQ(jllama_completions(model_, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return json::parse(response.str());
    };
    const json scale_zero = json::array({{{"id", 0}, {"scale", 0.0f}}});
    complete(0, json());     // slot 0: the server's scale 1
    complete(1, scale_zero); // slot 1: the request's scale 0
    const json at_zero = complete(-1, scale_zero);
    EXPECT_EQ(at_zero.at("id_slot").get<int>(), 1) << "a request at scale 0 went to the slot of scale 1";
    EXPECT_GT(at_zero.at("timings").at("cache_n").get<int>(), 0) << at_zero.dump();
    const json at_one = complete(-1, json());
    EXPECT_EQ(at_one.at("id_slot").get<int>(), 0) << "a request at scale 1 went to the slot of scale 0";
    EXPECT_GT(at_one.at("timings").at("cache_n").get<int>(), 0) << at_one.dump();
}

TEST_F(LoraCacheFreshSlot, AStateRestoredIntoANeverUsedSlotIsKept) {
    ASSERT_NE(model_, nullptr);
    complete_on(0); // the state to save: the prompt, evaluated in one batch, and an answer
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

    // The kept state answers like the slot it was saved from, asked again over its own cache. Not
    // like slot 0's first answer: that one evaluated the prompt in one batch, while both requests here
    // re-evaluate only the last prompt token in a batch of its own (upstream [TAG_PROMPT_LOGITS]),
    // which does not reproduce the batched logits bit for bit, so a greedy near tie could flip (F272).
    // Slot 0 still holds the saved state, and each slot has a KV stream of its own (-np 2 without a
    // unified cache), so the two compute the same thing.
    const json saved_slot = complete_on(0);
    EXPECT_LE(saved_slot.at("timings").at("prompt_n").get<int>(), 2)
        << "of " << saved_slot.at("tokens_evaluated").get<int>() << " prompt tokens: " << saved_slot.dump();
    EXPECT_EQ(saved_slot.at("timings").at("cache_n").get<int>(), restored.at("timings").at("cache_n").get<int>())
        << saved_slot.dump() << "\n"
        << restored.dump();
    EXPECT_EQ(saved_slot.at("timings").at("prompt_n").get<int>(), prompt_n) << saved_slot.dump() << "\n"
                                                                            << restored.dump();
    EXPECT_EQ(restored.at("content").get<std::string>(), saved_slot.at("content").get<std::string>());
}

TEST_F(LoraCacheFreshSlot, AStateIsKeptOnlyUnderTheScalesItWasSavedWith) {
    // The label of a restored state is the config it was saved with, which the file carries, not the
    // server's at the time of the restore: a state saved with the adapter at scale 1 and restored
    // after setLoraAdapters() set scale 0 was reused, and the answer at scale 0 came from KV computed
    // at scale 1. A copy of the file under another name carries the same config and is kept too.
    ASSERT_NE(model_, nullptr);
    complete_on(0); // computed with the adapter at scale 1, the --lora default
    Owned saved;
    ASSERT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_SAVE, 0, "scaled.bin", &saved.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const auto scaled = std::filesystem::path(save_dir_) / "scaled.bin";
    // the bytes the save reports are the whole file, the section included
    EXPECT_EQ(json::parse(saved.str()).at("n_written").get<uintmax_t>(), std::filesystem::file_size(scaled))
        << saved.str();

    // Before each restore slot 1 runs another prompt at the current scale, so its label is that config
    // and its tokens are others: a restore that left the label alone would keep it (server_prompt::clear
    // does not reset it) and reuse a state computed under another config.
    set_scale(0.0f);
    complete_other_on(1);
    slot_action(JLLAMA_SLOT_RESTORE, 1, "scaled.bin");
    const json other_scale = complete_on(1);
    EXPECT_EQ(other_scale.at("timings").at("cache_n").get<int>(), 0) << other_scale.dump();
    set_scale(1.0f);

    std::error_code ec;
    std::filesystem::copy_file(scaled, std::filesystem::path(save_dir_) / "copied.bin",
                               std::filesystem::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
    complete_other_on(1);
    Owned restored;
    ASSERT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_RESTORE, 1, "copied.bin", &restored.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    EXPECT_EQ(json::parse(restored.str()).at("n_read").get<uintmax_t>(), std::filesystem::file_size(scaled))
        << restored.str();
    const json copied = complete_on(1);
    EXPECT_LE(copied.at("timings").at("prompt_n").get<int>(), 2) << copied.dump();

    // and the file it saved under these adapters is kept: slot 1 holds another prompt, so only the
    // restored state can supply this one
    complete_other_on(1);
    slot_action(JLLAMA_SLOT_RESTORE, 1, "scaled.bin");
    const json same_scale = complete_on(1);
    EXPECT_LE(same_scale.at("timings").at("prompt_n").get<int>(), 2) << same_scale.dump();
}

TEST_F(LoraCacheFreshSlot, ASectionThatCannotBeReadRefusesTheRestore) {
    // A config read half is no config: the restore is refused and the slot left empty, rather than
    // the state kept under a label nobody can vouch for.
    ASSERT_NE(model_, nullptr);
    complete_on(0);
    slot_action(JLLAMA_SLOT_SAVE, 0, "whole.bin");
    const auto whole = std::filesystem::path(save_dir_) / "whole.bin";
    const auto cut = std::filesystem::path(save_dir_) / "cut.bin";
    std::error_code ec;
    std::filesystem::copy_file(whole, cut, std::filesystem::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
    std::filesystem::resize_file(cut, std::filesystem::file_size(whole) - 2, ec); // into the last scale
    ASSERT_FALSE(ec) << ec.message();
    Owned refused;
    EXPECT_EQ(jllama_slot_action(model_, JLLAMA_SLOT_RESTORE, 1, "cut.bin", &refused.s), JLLAMA_STATUS_INVALID_REQUEST);
    EXPECT_NE(std::string(jllama_last_error()).find("truncated LoRA config"), std::string::npos) << jllama_last_error();
    const json after = complete_on(1);
    EXPECT_EQ(after.at("timings").at("cache_n").get<int>(), 0) << "the refused state stayed: " << after.dump();
}

namespace {

// A model of write_tiny_llama_gguf() with the slot directory and, when adapter is not empty, the adapter.
jllama_model *load_with_slots(const std::string &model, const std::string &adapter, const std::string &dir) {
    std::vector<std::string> args = {"jllama", "-m",          model,  "-c",          "256",
                                     "-np",    "1",           "-ngl", "0",           "-t",
                                     "2",      "--cache-ram", "0",    "--no-warmup", "--slot-save-path",
                                     dir};
    if (!adapter.empty()) {
        args.insert(args.end(), {"--lora", adapter});
    }
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *loaded = nullptr;
    EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &loaded),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    return loaded;
}

json complete_slot_zero(jllama_model *model) {
    const json request = {{"prompt", PROMPT}, {"n_predict", 8},       {"temperature", 0},
                          {"top_k", 1},       {"cache_prompt", true}, {"id_slot", 0}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    return response.s != nullptr ? json::parse(response.str()) : json::object();
}

jllama_status slot_file(jllama_model *model, jllama_slot_op op, const char *name) {
    Owned response;
    return jllama_slot_action(model, op, 0, name, &response.s);
}

// The section's magic is a uint32_t the server writes in the host's byte order ("SLRA" on a
// little-endian host, "ARLS" on s390x).
bool holds_section(const std::filesystem::path &file) {
    std::ifstream in(file, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const uint32_t magic = 0x41524c53;
    char native[sizeof(magic)];
    std::memcpy(native, &magic, sizeof(magic));
    return bytes.find(std::string(native, sizeof(native))) != std::string::npos;
}

} // namespace

TEST(LoraCacheFiles, AFileCarriesItsConfigIntoTheNextProcess) {
    // The config travels with the file, so a model loaded again -- the next process, as far as the
    // server knows -- keeps a state saved under its adapters. Until the section existed the server
    // remembered the config of the files it had saved itself, by path, and a model loaded again
    // counted every file as computed under none: the first task cleared it.
    const std::string model_path = write_tiny_llama_gguf("jllama-lora-cache-files-model");
    const std::string adapter_path = write_tiny_lora_gguf();
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("jllama-lora-cache-files-" + std::to_string(std::random_device{}())))
            .string();
    ASSERT_FALSE(model_path.empty());
    ASSERT_FALSE(adapter_path.empty());
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    {
        jllama_model *first = load_with_slots(model_path, adapter_path, dir);
        ASSERT_NE(first, nullptr);
        complete_slot_zero(first); // at scale 1
        EXPECT_EQ(slot_file(first, JLLAMA_SLOT_SAVE, "scaled.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        Owned response;
        EXPECT_EQ(jllama_set_lora_adapters(first, R"([{"id":0,"scale":0}])", &response.s), JLLAMA_STATUS_OK);
        complete_slot_zero(first); // at scale 0: the adapter is in the config, and computes nothing
        EXPECT_EQ(slot_file(first, JLLAMA_SLOT_SAVE, "zero.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        jllama_model_free(first);
    }
    EXPECT_TRUE(holds_section(std::filesystem::path(dir) / "scaled.bin"));
    {
        // Without the adapter no task computes under scale 1: the restore is refused, by the adapter's
        // name, and the slot left empty. A state at scale 0 is a state without it, and is kept. A state
        // this model saves has no config to write, and the file has no section -- upstream's format.
        jllama_model *plain = load_with_slots(model_path, "", dir);
        ASSERT_NE(plain, nullptr);
        EXPECT_EQ(slot_file(plain, JLLAMA_SLOT_RESTORE, "scaled.bin"), JLLAMA_STATUS_INVALID_REQUEST);
        const std::string error = jllama_last_error();
        EXPECT_NE(error.find(adapter_path), std::string::npos) << error;
        EXPECT_NE(error.find("which this server has not loaded"), std::string::npos) << error;
        const json refused = complete_slot_zero(plain);
        EXPECT_EQ(refused.at("timings").at("cache_n").get<int>(), 0) << refused.dump();
        EXPECT_EQ(slot_file(plain, JLLAMA_SLOT_RESTORE, "zero.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        const json zero = complete_slot_zero(plain);
        EXPECT_LE(zero.at("timings").at("prompt_n").get<int>(), 2) << zero.dump();
        EXPECT_EQ(slot_file(plain, JLLAMA_SLOT_SAVE, "plain.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        jllama_model_free(plain);
    }
    EXPECT_FALSE(holds_section(std::filesystem::path(dir) / "plain.bin"));
    {
        jllama_model *again = load_with_slots(model_path, adapter_path, dir);
        ASSERT_NE(again, nullptr);
        EXPECT_EQ(slot_file(again, JLLAMA_SLOT_RESTORE, "scaled.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        const json kept = complete_slot_zero(again);
        EXPECT_LE(kept.at("timings").at("prompt_n").get<int>(), 2) << kept.dump();
        // a file without a section counts as computed under none, which the task at scale 1 clears
        EXPECT_EQ(slot_file(again, JLLAMA_SLOT_RESTORE, "plain.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
        const json none = complete_slot_zero(again);
        EXPECT_EQ(none.at("timings").at("cache_n").get<int>(), 0) << none.dump();
        jllama_model_free(again);
    }
    std::error_code ignored;
    std::filesystem::remove(model_path, ignored);
    std::filesystem::remove(adapter_path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(LoraCacheFiles, ASectionLostBehindADamagedDraftStateRefusesTheRestore) {
    // The LoRA section follows the checkpoints and the draft state. When one of those cannot be read
    // -- here the draft state of a model with a draft model, cut -- the restore cannot find the
    // section behind it, and counting the state as computed under no adapters would reuse it on a
    // server without any. The restore is refused, on a server with the adapter and on one without,
    // and the uncut file restores and is kept.
    const std::string model_path = write_tiny_llama_gguf("jllama-lora-cache-draft-model");
    const std::string adapter_path = write_tiny_lora_gguf();
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("jllama-lora-cache-draft-" + std::to_string(std::random_device{}())))
            .string();
    ASSERT_FALSE(model_path.empty());
    ASSERT_FALSE(adapter_path.empty());
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    const auto load = [&](bool with_adapter) {
        std::vector<std::string> args = {"jllama",
                                         "-m",
                                         model_path,
                                         "-c",
                                         "256",
                                         "-np",
                                         "1",
                                         "-ngl",
                                         "0",
                                         "-t",
                                         "2",
                                         "--cache-ram",
                                         "0",
                                         "--no-warmup",
                                         "--slot-save-path",
                                         dir,
                                         "--spec-draft-model",
                                         model_path,
                                         "--spec-draft-ngl",
                                         "0"};
        if (with_adapter) {
            args.insert(args.end(), {"--lora", adapter_path});
        }
        std::vector<const char *> argv;
        for (const auto &a : args) {
            argv.push_back(a.c_str());
        }
        jllama_model *loaded = nullptr;
        EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &loaded),
                  JLLAMA_STATUS_OK)
            << jllama_last_error();
        return loaded;
    };
    jllama_model *model = load(true);
    ASSERT_NE(model, nullptr);
    complete_slot_zero(model);
    ASSERT_EQ(slot_file(model, JLLAMA_SLOT_SAVE, "whole.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
    const auto whole = std::filesystem::path(dir) / "whole.bin";
    ASSERT_TRUE(holds_section(whole));
    // the LoRA section: magic, version, count, then the adapter's path (with its length) and scale
    const uintmax_t section = 3 * sizeof(uint32_t) + sizeof(uint64_t) + adapter_path.size() + sizeof(float);
    std::error_code ec;
    std::filesystem::copy_file(whole, std::filesystem::path(dir) / "cut.bin",
                               std::filesystem::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << ec.message();
    std::filesystem::resize_file(std::filesystem::path(dir) / "cut.bin",
                                 std::filesystem::file_size(whole) - section - 16, ec); // into the draft state
    ASSERT_FALSE(ec) << ec.message();
    for (const bool with_adapter : {true, false}) {
        SCOPED_TRACE(with_adapter ? "with the adapter" : "without adapters");
        if (!with_adapter) {
            jllama_model_free(model);
            model = load(false);
            ASSERT_NE(model, nullptr);
        }
        EXPECT_EQ(slot_file(model, JLLAMA_SLOT_RESTORE, "cut.bin"), JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("LoRA config is unknown"), std::string::npos)
            << jllama_last_error();
        const json after = complete_slot_zero(model);
        EXPECT_EQ(after.at("timings").at("cache_n").get<int>(), 0) << "the refused state stayed: " << after.dump();
    }
    jllama_model_free(model);
    model = load(true);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(slot_file(model, JLLAMA_SLOT_RESTORE, "whole.bin"), JLLAMA_STATUS_OK) << jllama_last_error();
    const json kept = complete_slot_zero(model);
    EXPECT_LE(kept.at("timings").at("prompt_n").get<int>(), 2) << kept.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(model_path, ignored);
    std::filesystem::remove(adapter_path, ignored);
    std::filesystem::remove_all(dir, ignored);
}

TEST(LoraCacheSleep, AStateRestoredAfterTheModelSleptIsKept) {
    // A wake from the idle sleep loads the adapters again, under new pointers, and llama.cpp compares
    // adapter configs by pointer and scale. The first form of patches/0021 stored the adapter infos of
    // a saved file, so after a sleep the label of a restored state pointed into freed memory -- a
    // mismatch at best (the state cleared), a read of the freed adapter at worst. The patch stores
    // paths and scales in the file and rebuilds the label from the adapters loaded now: the state is
    // kept. Not red
    // against that first form in every run: the allocator may give the reloaded adapter its old
    // address, and then the stale pointer compares equal; what this pins is the behaviour.
    const std::string model_path = write_tiny_llama_gguf("jllama-lora-cache-sleep-model");
    const std::string adapter_path = write_tiny_lora_gguf();
    const std::string save_dir = (std::filesystem::temp_directory_path() /
                                  ("jllama-lora-cache-sleep-slots-" + std::to_string(std::random_device{}())))
                                     .string();
    ASSERT_FALSE(model_path.empty());
    ASSERT_FALSE(adapter_path.empty());
    ASSERT_TRUE(std::filesystem::create_directories(save_dir));
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
    const std::vector<std::string> args = {"jllama",     "-m",
                                           model_path,   "--lora",
                                           adapter_path, "-c",
                                           "256",        "-np",
                                           "1",          "-ngl",
                                           "0",          "-t",
                                           "2",          "--cache-ram",
                                           "0",          "--slot-save-path",
                                           save_dir,     "--sleep-idle-seconds",
                                           "1",          "--no-warmup"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json request = {
        {"prompt", PROMPT}, {"n_predict", 8}, {"temperature", 0}, {"top_k", 1}, {"cache_prompt", true}};
    {
        Owned response;
        ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    {
        Owned saved;
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, "slept.bin", &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    const auto slept = [&lines]() {
        std::lock_guard<std::mutex> lock(lines.mutex);
        return lines.sleeps > 0;
    };
    while (!slept() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(slept()) << "the model did not sleep";
    {
        Owned restored; // wakes the model: its adapters are loaded again
        ASSERT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "slept.bin", &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
    }
    Owned response;
    ASSERT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    const json after = json::parse(response.str());
    EXPECT_LE(after.at("timings").at("prompt_n").get<int>(), 2) << after.dump();
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(model_path, ignored);
    std::filesystem::remove(adapter_path, ignored);
    std::filesystem::remove_all(save_dir, ignored);
}

namespace {

// The host-memory prompt cache (--cache-ram, on by default) keeps the LoRA config of every state
// since upstream #32434 (b12498): a state computed under other adapters is not loaded into a slot,
// and one computed under the task's adapters is, after a change and back. Before it the cache
// stored states without their adapters (and the change back threw the matching state away).
class LoraCacheHostMemory : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        model_path_ = write_tiny_llama_gguf("jllama-lora-cache-ram-model");
        adapter_path_ = write_tiny_lora_gguf();
        ASSERT_FALSE(model_path_.empty());
        ASSERT_FALSE(adapter_path_.empty());
        const std::vector<std::string> args = {"jllama", "-m",          model_path_, "--lora",     adapter_path_, "-c",
                                               "128",    "-np",         "1",         "-ngl",       "0",           "-t",
                                               "2",      "--cache-ram", "64",        "--no-warmup"};
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
    static json complete() {
        const json request = {{"prompt", PROMPT}, {"n_predict", 8},       {"temperature", 0},
                              {"top_k", 1},       {"cache_prompt", true}, {"id_slot", 0}};
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

    static std::string model_path_;
    static std::string adapter_path_;
    static jllama_model *model_;
};
std::string LoraCacheHostMemory::model_path_;
std::string LoraCacheHostMemory::adapter_path_;
jllama_model *LoraCacheHostMemory::model_ = nullptr;

int timing(const json &response, const char *key) { return response.at("timings").at(key).get<int>(); }

} // namespace

TEST_F(LoraCacheHostMemory, AStateIsLoadedOnlyUnderTheAdaptersItWasComputedWith) {
    ASSERT_NE(model_, nullptr);
    complete(); // the slot holds the prompt computed with the adapter at scale 1
    set_scale(0.0f);
    // the state goes to the host-memory cache when the next task takes the slot; it is not taken back
    const json without = complete();
    EXPECT_EQ(timing(without, "cache_n"), 0) << without.dump();

    set_scale(1.0f);
    // the slot's state (scale 0) goes to the cache, the one of scale 1 comes back from it
    const json back = complete();
    EXPECT_GT(timing(back, "cache_n"), 0) << back.dump();
    EXPECT_LE(timing(back, "prompt_n"), 2) << back.dump();
    // and it answers like a second request over the same cache (two requests over a cached prompt
    // compute the same thing, F272)
    const json again = complete();
    EXPECT_EQ(timing(again, "cache_n"), timing(back, "cache_n")) << again.dump() << "\n" << back.dump();
    EXPECT_EQ(again.at("content").get<std::string>(), back.at("content").get<std::string>());
}

namespace {

// A model of write_tiny_llama_gguf() with one slot, the slot directory and the options of `extra`.
jllama_model *load_with(const std::string &model, const std::string &dir, const std::vector<std::string> &extra) {
    std::vector<std::string> args = {"jllama", "-m",          model,  "-c",          "256",
                                     "-np",    "1",           "-ngl", "0",           "-t",
                                     "2",      "--cache-ram", "0",    "--no-warmup", "--slot-save-path",
                                     dir};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *loaded = nullptr;
    EXPECT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &loaded),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    return loaded;
}

// The adapters the server lists, path -> scale.
std::map<std::string, float> listed_adapters(jllama_model *model) {
    Owned list;
    EXPECT_EQ(jllama_lora_adapters(model, &list.s), JLLAMA_STATUS_OK) << jllama_last_error();
    std::map<std::string, float> scales;
    if (list.s != nullptr) {
        for (const auto &adapter : json::parse(list.str())) {
            scales[adapter.at("path").get<std::string>()] = adapter.at("scale").get<float>();
        }
    }
    return scales;
}

// The answer to PROMPT on an emptied slot, its whole prompt evaluated in one batch.
std::string fresh_answer_of(jllama_model *model) {
    Owned erased;
    EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_ERASE, 0, nullptr, &erased.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const json request = {{"prompt", PROMPT}, {"n_predict", 8},        {"temperature", 0},
                          {"top_k", 1},       {"cache_prompt", false}, {"id_slot", 0}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
    if (response.s == nullptr) {
        return std::string();
    }
    const json answer = json::parse(response.str());
    EXPECT_EQ(answer.at("timings").at("cache_n").get<int>(), 0) << answer.dump();
    return answer.at("content").get<std::string>();
}

void set_scales(jllama_model *model, const json &adapters) {
    Owned response;
    EXPECT_EQ(jllama_set_lora_adapters(model, adapters.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
        << jllama_last_error();
}

// Another prompt on slot 0, so a later restore shows whether it put its own state there.
void complete_other(jllama_model *model) {
    const json request = {{"prompt", "A different prompt, so the slot holds other tokens"},
                          {"n_predict", 4},
                          {"temperature", 0},
                          {"cache_prompt", true},
                          {"id_slot", 0}};
    Owned response;
    EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK) << jllama_last_error();
}

} // namespace

TEST(LoraCacheTwoAdapters, BothAdaptersApplyAndASlotFileMatchesThemByPathInEitherOrder) {
    // --lora a,b loads two adapters, both applied at scale 1; each changes the answer. A slot file
    // saved under the pair at scales 1 and 0.5 carries both in its section (patches/0021); a model that
    // loads them in the other order, or through --lora-scaled, matches them by path, not by place: at
    // the same scales the state is kept, at the scales swapped it is cleared.
    const std::string model_path = write_tiny_llama_gguf("jllama-lora-two-model");
    const std::string a = write_tiny_lora_gguf("adapter-a", 7);
    const std::string b = write_tiny_lora_gguf("adapter-b", 11);
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("jllama-lora-two-" + std::to_string(std::random_device{}())))
            .string();
    ASSERT_FALSE(model_path.empty());
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    std::string scaled_answer; // a at 1, b at 0.5
    {
        jllama_model *pair = load_with(model_path, dir, {"--lora", a + "," + b});
        ASSERT_NE(pair, nullptr);
        const std::map<std::string, float> listed = listed_adapters(pair);
        EXPECT_EQ(listed, (std::map<std::string, float>{{a, 1.0f}, {b, 1.0f}}));
        const std::string both = fresh_answer_of(pair);
        set_scales(pair, json::array({{{"id", 0}, {"scale", 1.0f}}, {{"id", 1}, {"scale", 0.0f}}}));
        const std::string only_a = fresh_answer_of(pair);
        set_scales(pair, json::array({{{"id", 0}, {"scale", 0.0f}}, {{"id", 1}, {"scale", 1.0f}}}));
        const std::string only_b = fresh_answer_of(pair);
        EXPECT_NE(both, only_a) << "the second adapter changed nothing";
        EXPECT_NE(both, only_b) << "the first adapter changed nothing";

        set_scales(pair, json::array({{{"id", 0}, {"scale", 1.0f}}, {{"id", 1}, {"scale", 0.5f}}}));
        scaled_answer = fresh_answer_of(pair);
        EXPECT_NE(scaled_answer, both) << "the scale 0.5 changed nothing";
        Owned saved;
        ASSERT_EQ(jllama_slot_action(pair, JLLAMA_SLOT_SAVE, 0, "pair.bin", &saved.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        jllama_model_free(pair);
    }
    const auto restore_and_complete = [](jllama_model *model) {
        complete_other(model);
        Owned restored;
        EXPECT_EQ(jllama_slot_action(model, JLLAMA_SLOT_RESTORE, 0, "pair.bin", &restored.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        const json request = {{"prompt", PROMPT}, {"n_predict", 8},       {"temperature", 0},
                              {"top_k", 1},       {"cache_prompt", true}, {"id_slot", 0}};
        Owned response;
        EXPECT_EQ(jllama_completions(model, request.dump().c_str(), &response.s), JLLAMA_STATUS_OK)
            << jllama_last_error();
        return response.s != nullptr ? json::parse(response.str()) : json::object();
    };
    {
        // the other order: id 0 is b now
        jllama_model *reversed = load_with(model_path, dir, {"--lora", b + "," + a});
        ASSERT_NE(reversed, nullptr);
        set_scales(reversed, json::array({{{"id", 0}, {"scale", 0.5f}}, {{"id", 1}, {"scale", 1.0f}}}));
        const json kept = restore_and_complete(reversed);
        EXPECT_LE(kept.at("timings").at("prompt_n").get<int>(), 2) << kept.dump();
        // the same places at the file's scales are the scales swapped: b at 1, a at 0.5
        set_scales(reversed, json::array({{{"id", 0}, {"scale", 1.0f}}, {{"id", 1}, {"scale", 0.5f}}}));
        const json swapped = restore_and_complete(reversed);
        EXPECT_EQ(swapped.at("timings").at("cache_n").get<int>(), 0) << swapped.dump();
        jllama_model_free(reversed);
    }
    // --lora-scaled reads FNAME:SCALE and splits on every colon, so an absolute path with a drive
    // letter cannot be expressed (ModelParameters.addLoraScaledAdapter says so): relative paths.
    std::error_code ec;
    const std::string rel_a = std::filesystem::relative(a, std::filesystem::current_path(), ec).string();
    const std::string rel_b = std::filesystem::relative(b, std::filesystem::current_path(), ec).string();
    if (!rel_a.empty() && !rel_b.empty() && rel_a.find(':') == std::string::npos &&
        rel_b.find(':') == std::string::npos) {
        jllama_model *scaled = load_with(model_path, dir, {"--lora-scaled", rel_b + ":0.5," + rel_a + ":1"});
        ASSERT_NE(scaled, nullptr);
        const std::map<std::string, float> listed = listed_adapters(scaled);
        EXPECT_EQ(listed, (std::map<std::string, float>{{rel_a, 1.0f}, {rel_b, 0.5f}}));
        // both applied at their own scales from the load on: the answer of the pair set to them
        EXPECT_EQ(fresh_answer_of(scaled), scaled_answer);
        // The section names the paths the first model was given, and the match is by path: these are
        // other spellings of the same files, so the file names adapters this server has not loaded,
        // and the restore is refused -- the conservative answer, never a state reused under adapters
        // it may not have been computed with.
        Owned restored;
        EXPECT_EQ(jllama_slot_action(scaled, JLLAMA_SLOT_RESTORE, 0, "pair.bin", &restored.s),
                  JLLAMA_STATUS_INVALID_REQUEST);
        EXPECT_NE(std::string(jllama_last_error()).find("which this server has not loaded"), std::string::npos)
            << jllama_last_error();
        jllama_model_free(scaled);
    } else {
        ADD_FAILURE() << "no relative path to the adapters from " << std::filesystem::current_path();
    }
    std::filesystem::remove(model_path, ec);
    std::filesystem::remove(a, ec);
    std::filesystem::remove(b, ec);
    std::filesystem::remove_all(dir, ec);
}

TEST(LoraCacheTwoAdapters, AnAdapterLoadedWithoutApplyAnswersAsNoAdapterUntilItIsSet) {
    // --lora-init-without-apply loads an adapter without applying it ("apply later via POST
    // /lora-adapters"): the model answers as without the adapter until jllama_set_lora_adapters
    // gives it a scale, and then as the model loaded with the adapter applied.
    const std::string model_path = write_tiny_llama_gguf("jllama-lora-init-model");
    const std::string adapter = write_tiny_lora_gguf("adapter-init", 7);
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("jllama-lora-init-" + std::to_string(std::random_device{}())))
            .string();
    ASSERT_FALSE(model_path.empty());
    ASSERT_FALSE(adapter.empty());
    ASSERT_TRUE(std::filesystem::create_directories(dir));
    jllama_model *plain = load_with(model_path, dir, {});
    ASSERT_NE(plain, nullptr);
    const std::string without = fresh_answer_of(plain);
    jllama_model_free(plain);
    jllama_model *applied = load_with(model_path, dir, {"--lora", adapter});
    ASSERT_NE(applied, nullptr);
    const std::string with = fresh_answer_of(applied);
    jllama_model_free(applied);
    ASSERT_NE(without, with) << "the adapter changes nothing";

    jllama_model *deferred = load_with(model_path, dir, {"--lora", adapter, "--lora-init-without-apply"});
    ASSERT_NE(deferred, nullptr);
    // Upstream's server ignores the option (b12541): common_init_from_params leaves the adapters
    // unapplied, but the server lists them at the scale --lora gave and applies the slot's adapters to
    // every batch (common_set_adapter_lora in update_slots), so the model answered as with the adapter
    // from the first request on. The engine starts such adapters at scale 0 (red without it).
    EXPECT_EQ(listed_adapters(deferred), (std::map<std::string, float>{{adapter, 0.0f}}));
    EXPECT_EQ(fresh_answer_of(deferred), without);
    set_scales(deferred, json::array({{{"id", 0}, {"scale", 1.0f}}}));
    EXPECT_EQ(listed_adapters(deferred), (std::map<std::string, float>{{adapter, 1.0f}}));
    EXPECT_EQ(fresh_answer_of(deferred), with);
    jllama_model_free(deferred);
    std::error_code ec;
    std::filesystem::remove(model_path, ec);
    std::filesystem::remove(adapter, ec);
    std::filesystem::remove_all(dir, ec);
}
