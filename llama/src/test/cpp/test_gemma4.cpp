// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Guard for llama.cpp's Gemma 4 graph with several sequences in one batch (b12593, #32342).
//
// The last layers of Gemma 4 keep no KV of their own (they read the KV of an earlier layer), so the
// graph drops every row that asks for no output before the first of them. Until b12593 it did so
// only in a context of one sequence; since #32342 it also prunes a batch of several, slice by slice
// of the attention mask: the rows of each sequence (a stream of a split KV cache) are padded to the
// largest number of outputs of a slice, and the outputs are taken back by their place in their
// slice. A wrong place hands one sequence the logits of another -- which no answer of a single slot
// would show.
//
// The model is tiny_model_spec::gemma4() (random, two layers with KV of their own and two that
// share it, sliding window 8). Three prompts longer than the window are decoded as three sequences
// of one batch, each with outputs at other places and a different number of them, then a next token
// per sequence in one batch, in a context with a split and with a unified KV cache; every output
// must be what the same prompt gives decoded alone (measured: 2.3e-5 apart at most, the tolerance is
// 1e-3). Red-checked against the patched tree: an output taken back by its place in the batch
// instead of in its slice is 2.45 apart in the split cache, and pruned rows given the mask rows of
// the first tokens instead of their own 1.15 (split) and 1.72 (unified). A unified cache is one
// slice, so only the second reaches it.

#include "llama.h"

#include "tiny_llama_gguf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void quiet_log(ggml_log_level, const char *, void *) {}

// one sequence of a batch: its tokens from position first on, and which of them ask for logits
struct part {
    llama_seq_id seq;
    std::vector<llama_token> tokens;
    std::vector<bool> output;
    llama_pos first = 0;
};

struct gemma4_model {
    std::string path;
    llama_model *model = nullptr;

    gemma4_model() {
        path = jllama_test::write_tiny_llama_gguf(jllama_test::tiny_model_spec::gemma4());
        llama_backend_init();
        llama_log_set(quiet_log, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        // the CPU alone: an RPC server another test of the same process registered stays a device
        static ggml_backend_dev_t no_devices[] = {nullptr};
        mp.devices = no_devices;
        if (!path.empty()) {
            model = llama_model_load_from_file(path.c_str(), mp);
        }
        llama_log_set(nullptr, nullptr);
    }
    ~gemma4_model() {
        if (model != nullptr) {
            llama_model_free(model);
        }
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
    gemma4_model(const gemma4_model &) = delete;
    gemma4_model &operator=(const gemma4_model &) = delete;

    // a context of n_seq sequences; F32 cache and no flash attention, so the comparison is about the
    // graph and not about the cache's rounding
    llama_context *context(uint32_t n_seq, bool unified) const {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 128 * n_seq;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_seq_max = n_seq;
        cp.kv_unified = unified;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.op_offload = false;
        llama_log_set(quiet_log, nullptr);
        llama_context *ctx = llama_init_from_model(model, cp);
        llama_log_set(nullptr, nullptr);
        return ctx;
    }

    int n_vocab() const { return llama_vocab_n_tokens(llama_model_get_vocab(model)); }
};

// decodes the parts as one batch and returns the logits of every output, part by part in order
std::vector<std::vector<float>> decode(llama_context *ctx, const std::vector<part> &parts, int n_vocab) {
    std::vector<std::vector<float>> res;
    llama_batch_ext *batch = llama_batch_ext_init(ctx);
    std::vector<int32_t> outputs;
    bool ok = batch != nullptr;
    for (const part &p : parts) {
        for (size_t i = 0; ok && i < p.tokens.size(); ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, p.seq, p.tokens[i]);
            const llama_pos pos = p.first + static_cast<llama_pos>(i);
            ok = idx >= 0 && llama_batch_ext_set_pos(batch, idx, &pos) &&
                 (!p.output[i] || llama_batch_ext_set_output_logits(batch, idx, true));
            if (ok && p.output[i]) {
                outputs.push_back(idx);
            }
        }
    }
    if (ok && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0) {
        for (const int32_t idx : outputs) {
            const float *l = llama_get_logits_ith(ctx, idx);
            res.emplace_back(l, l + n_vocab);
        }
    }
    llama_batch_ext_free(batch);
    return res;
}

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return d;
}

std::vector<llama_token> prompt(int seed, int n) {
    std::vector<llama_token> tokens;
    for (int i = 0; i < n; ++i) {
        tokens.push_back(static_cast<llama_token>(33 + (seed * 37 + i * 11) % 90));
    }
    return tokens;
}

std::vector<bool> outputs_at(int n, std::initializer_list<int> at) {
    std::vector<bool> out(static_cast<size_t>(n), false);
    for (const int i : at) {
        out.at(static_cast<size_t>(i)) = true;
    }
    return out;
}

// Three prompts of 12 tokens (longer than the window of 8) with 1, 3 and 2 outputs at other
// places, then one next token each: every output, decoded together, against the same prompt alone.
void expect_each_sequence_as_alone(const gemma4_model &m, bool unified) {
    const int n_vocab = m.n_vocab();
    constexpr int N = 12;
    const std::vector<part> prompts = {
        {0, prompt(1, N), outputs_at(N, {N - 1})},
        {1, prompt(2, N), outputs_at(N, {3, N - 2, N - 1})},
        {2, prompt(3, N), outputs_at(N, {0, N - 1})},
    };
    const std::vector<llama_token> next = {65, 66, 67};

    llama_context *together = m.context(3, unified);
    ASSERT_NE(together, nullptr);
    const std::vector<std::vector<float>> got = decode(together, prompts, n_vocab);
    std::vector<part> step;
    for (size_t s = 0; s < prompts.size(); ++s) {
        step.push_back({prompts[s].seq, {next[s]}, {true}, N});
    }
    const std::vector<std::vector<float>> got_next = decode(together, step, n_vocab);
    llama_free(together);
    ASSERT_EQ(got.size(), 6u);
    ASSERT_EQ(got_next.size(), 3u);

    llama_context *alone = m.context(1, true);
    ASSERT_NE(alone, nullptr);
    size_t at = 0;
    double worst = 0.0;
    for (size_t s = 0; s < prompts.size(); ++s) {
        llama_memory_clear(llama_get_memory(alone), true);
        part p = prompts[s];
        p.seq = 0;
        const std::vector<std::vector<float>> want = decode(alone, {p}, n_vocab);
        const std::vector<std::vector<float>> want_next = decode(alone, {{0, {next[s]}, {true}, N}}, n_vocab);
        ASSERT_EQ(want_next.size(), 1u);
        for (const std::vector<float> &w : want) {
            ASSERT_LT(at, got.size());
            const double d = max_abs_diff(got[at], w);
            worst = std::max(worst, d);
            EXPECT_LT(d, 1e-3) << "sequence " << s << ", output " << at;
            ++at;
        }
        const double d = max_abs_diff(got_next[s], want_next[0]);
        worst = std::max(worst, d);
        EXPECT_LT(d, 1e-3) << "next token of sequence " << s;
    }
    llama_free(alone);
    EXPECT_EQ(at, got.size());
    std::printf("[gemma4] %s KV cache: largest difference %.3g\n", unified ? "unified" : "split", worst);
}

} // namespace

TEST(Gemma4, ThreeSequencesInOneBatchOfASplitCacheGetTheLogitsEachGetsAlone) {
    gemma4_model m;
    ASSERT_NE(m.model, nullptr);
    expect_each_sequence_as_alone(m, false);
}

TEST(Gemma4, ThreeSequencesInOneBatchOfAUnifiedCacheGetTheLogitsEachGetsAlone) {
    gemma4_model m;
    ASSERT_NE(m.model, nullptr);
    expect_each_sequence_as_alone(m, true);
}

// The comparison would pass vacuously if the three prompts gave the same logits.
TEST(Gemma4, TheThreePromptsGiveDifferentLogits) {
    gemma4_model m;
    ASSERT_NE(m.model, nullptr);
    llama_context *ctx = m.context(1, true);
    ASSERT_NE(ctx, nullptr);
    std::vector<std::vector<float>> last;
    for (int s = 1; s <= 3; ++s) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const std::vector<std::vector<float>> l = decode(ctx, {{0, prompt(s, 12), outputs_at(12, {11})}}, m.n_vocab());
        ASSERT_EQ(l.size(), 1u);
        last.push_back(l[0]);
    }
    llama_free(ctx);
    EXPECT_GT(max_abs_diff(last[0], last[1]), 1e-2);
    EXPECT_GT(max_abs_diff(last[1], last[2]), 1e-2);
}
