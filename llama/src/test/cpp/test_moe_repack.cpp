// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Runnable guard for patches/0022 (ggml-cpu: batch the tokens of an expert in the repack MUL_MAT_ID,
// and hand its work out dynamically).
//
// MUL_MAT_ID is the expert matmul of a mixture-of-experts model. With the expert weights in the
// CPU_REPACK buffer it runs through forward_mul_mat_id() of ggml-cpu's repack code, which the patch
// splits into two paths behind a process-wide switch: upstream's (one gemv per token, a static row
// split per thread) and the batched one (every group of four or more tokens of an expert through one
// gemm, the work handed out as (expert, column chunk) units). Both must compute what ggml's generic
// MUL_MAT_ID computes over the same weights in a plain buffer -- that path is independent code
// (ggml-cpu.c), quantizes the activations the same way, and is the reference here.
//
// The routings give an expert 0, 1, 3, 4 and 17 tokens (the gemv tail alone, one gemm group, a full
// 16-row gemm group plus a gemv tail), one slot of -1 (no expert: the output row stays zero), and a
// single token (token generation). Each runs with one input row per token (the gate/up projections)
// and one per slot (the down projection), on 1, 3 and 4 threads -- 3 does not divide the column blocks
// evenly, and 1 thread leaves units past the last column. Four weight types reach the repack path on
// AVX2 (MXFP4 of gpt-oss, Q4_0, IQ4_NL, Q4_K with its Q8_K activations); a type the running CPU does
// not repack is skipped, not passed.
//
// The switch is resolved through the CPU backend's get_proc_address, the way libjllama reaches it in a
// GGML_BACKEND_DL build, so a llama.cpp bump that drops the patch reds this file at once.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t K = 256;      // row length: one Q4_K/Q8_K super-block, eight 32-blocks
constexpr int64_t N = 72;       // rows per expert: 9 column blocks of 8, not a multiple of 3 or 4
constexpr int64_t N_EXPERT = 6; // experts

using switch_set_fn = void (*)(bool);
using switch_get_fn = bool (*)();

template <typename Fn> Fn cpu_proc(const char *name) {
    return reinterpret_cast<Fn>(ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), name));
}

void set_batched(bool batched) {
    auto set = cpu_proc<switch_set_fn>("ggml_backend_cpu_set_repack_mmid_batched");
    ASSERT_NE(set, nullptr);
    set(batched);
}

bool get_batched() {
    auto get = cpu_proc<switch_get_fn>("ggml_backend_cpu_get_repack_mmid_batched");
    return get != nullptr && get();
}

ggml_backend_buffer_type_t repack_buffer_type() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_dev_get_extra_bufts"));
    if (dev == nullptr || get_extra == nullptr) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t *buft = get_extra(dev); buft != nullptr && *buft != nullptr; ++buft) {
        if (std::string(ggml_backend_buft_name(*buft)) == "CPU_REPACK") {
            return *buft;
        }
    }
    return nullptr;
}

// ids[token][slot] -> expert, -1 for none
struct Routing {
    const char *name;
    int n_used;
    std::vector<std::vector<int32_t>> ids;
};

// 20 tokens, 2 experts each: e3 gets 17 tokens, e5 15, e2 4, e1 3, e0 1, e4 none
Routing spread(bool with_empty_slot) {
    Routing r{with_empty_slot ? "spread_with_an_empty_slot" : "spread", 2, {}};
    for (int t = 0; t < 20; ++t) {
        const int32_t first = t <= 16 ? 3 : 5;
        const int32_t second = t <= 11 ? 5 : t <= 15 ? 2 : t == 16 ? 0 : 1;
        r.ids.push_back({first, second});
    }
    if (with_empty_slot) {
        r.ids[19][1] = -1;
    }
    return r;
}

Routing single_token() { return Routing{"single_token", 4, {{5, 0, 3, 2}}}; }

struct Case {
    ggml_type type;
    Routing routing;
    bool row_per_slot; // src1 has one row per slot (down projection) instead of one per token
};

std::string case_name(const testing::TestParamInfo<Case> &info) {
    return std::string(ggml_type_name(info.param.type)) + "_" + info.param.routing.name +
           (info.param.row_per_slot ? "_row_per_slot" : "_row_per_token");
}

class MoeRepack : public testing::TestWithParam<Case> {
  protected:
    void SetUp() override { initial_ = get_batched(); }
    void TearDown() override { set_batched(initial_); }

  private:
    bool initial_ = true;
};

// out = mul_mat_id(w, x, ids) on n_threads; w lives in the given buffer, or in the context when null
std::vector<float> run(ggml_tensor *w, const std::vector<float> &x_data, const Routing &routing, int64_t ne11,
                       int n_threads) {
    const int64_t n_tokens = (int64_t)routing.ids.size();
    ggml_init_params ip = {64u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, n_tokens);
    std::memcpy(x->data, x_data.data(), ggml_nbytes(x));
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, routing.n_used, n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < routing.n_used; ++s) {
            ((int32_t *)ids->data)[t * routing.n_used + s] = routing.ids[t][s];
        }
    }
    ggml_tensor *out = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx, gf, n_threads), GGML_STATUS_SUCCESS);
    std::vector<float> result((const float *)out->data, (const float *)out->data + ggml_nelements(out));
    ggml_free(ctx);
    return result;
}

TEST_P(MoeRepack, BothPathsComputeWhatTheGenericPathComputes) {
    const Case &c = GetParam();
    ggml_backend_buffer_type_t repack = repack_buffer_type();
    if (repack == nullptr) {
        GTEST_SKIP() << "this build has no CPU_REPACK buffer";
    }

    // random weights, quantized to the type, the same bytes in both buffers
    std::mt19937 rng(22);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> w_f32(K * N * N_EXPERT);
    for (float &v : w_f32) {
        v = 0.1f * dist(rng);
    }
    std::vector<uint8_t> w_q(ggml_row_size(c.type, K) * N * N_EXPERT);
    ggml_quantize_chunk(c.type, w_f32.data(), w_q.data(), 0, N * N_EXPERT, K, nullptr);

    // the reference: the weights in plain memory, ggml's generic MUL_MAT_ID
    ggml_init_params plain_ip = {w_q.size() + 4096, nullptr, false};
    ggml_context *plain_ctx = ggml_init(plain_ip);
    ggml_tensor *w_plain = ggml_new_tensor_3d(plain_ctx, c.type, K, N, N_EXPERT);
    std::memcpy(w_plain->data, w_q.data(), w_q.size());

    // the subject: the same weights repacked
    ggml_init_params repack_ip = {ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context *repack_ctx = ggml_init(repack_ip);
    ggml_tensor *w_repack = ggml_new_tensor_3d(repack_ctx, c.type, K, N, N_EXPERT);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(repack_ctx, repack);
    ASSERT_NE(buffer, nullptr);
    if (w_repack->extra == nullptr) {
        ggml_backend_buffer_free(buffer);
        ggml_free(repack_ctx);
        ggml_free(plain_ctx);
        GTEST_SKIP() << ggml_type_name(c.type) << " is not repacked on this CPU";
    }
    ggml_backend_tensor_set(w_repack, w_q.data(), 0, w_q.size());

    const int64_t ne11 = c.row_per_slot ? c.routing.n_used : 1;
    std::vector<float> x(K * ne11 * c.routing.ids.size());
    for (float &v : x) {
        v = dist(rng);
    }

    const std::vector<float> expected = run(w_plain, x, c.routing, ne11, 1);
    float scale = 0.0f;
    for (float v : expected) {
        scale = std::max(scale, std::fabs(v));
    }
    ASSERT_GT(scale, 0.0f);

    for (bool batched : {true, false}) {
        set_batched(batched);
        for (int n_threads : {1, 3, 4}) {
            const std::vector<float> actual = run(w_repack, x, c.routing, ne11, n_threads);
            ASSERT_EQ(actual.size(), expected.size());
            size_t worst = 0;
            float worst_diff = 0.0f;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) {
                    worst_diff = INFINITY;
                    worst = i;
                    break;
                }
                const float diff = std::fabs(actual[i] - expected[i]);
                if (diff > worst_diff) {
                    worst_diff = diff;
                    worst = i;
                }
            }
            EXPECT_LE(worst_diff, 1e-3f * scale)
                << (batched ? "batched" : "upstream's") << " path, " << n_threads << " thread(s): element " << worst
                << " is " << actual[worst] << ", the generic path gives " << expected[worst];
        }
    }
    // a slot routed to no expert stays zero on every path (the generic path's reference included)
    for (size_t t = 0; t < c.routing.ids.size(); ++t) {
        for (int s = 0; s < c.routing.n_used; ++s) {
            if (c.routing.ids[t][s] == -1) {
                for (int64_t n = 0; n < N; ++n) {
                    EXPECT_EQ(expected[(t * c.routing.n_used + s) * N + n], 0.0f);
                }
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(repack_ctx);
    ggml_free(plain_ctx);
}

std::vector<Case> cases() {
    std::vector<Case> out;
    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K}) {
        for (const Routing &routing : {spread(false), spread(true), single_token()}) {
            for (bool row_per_slot : {false, true}) {
                out.push_back({type, routing, row_per_slot});
            }
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Types, MoeRepack, testing::ValuesIn(cases()), case_name);

TEST(MoeRepackSwitch, IsReachableThroughTheCpuBackendAndRoundTrips) {
    auto set = cpu_proc<switch_set_fn>("ggml_backend_cpu_set_repack_mmid_batched");
    auto get = cpu_proc<switch_get_fn>("ggml_backend_cpu_get_repack_mmid_batched");
    ASSERT_NE(set, nullptr) << "the CPU backend has no switch: llama/patches/0022 is not applied";
    ASSERT_NE(get, nullptr) << "the CPU backend has no switch: llama/patches/0022 is not applied";
    const bool initial = get();
    const char *env = std::getenv("GGML_CPU_REPACK_MMID_BATCHED");
    if (env == nullptr) {
        // nothing in this process has flipped it yet: the test cases restore what they found
        EXPECT_TRUE(initial) << "the batched path is the default";
    }
    set(false);
    EXPECT_FALSE(get());
    set(true);
    EXPECT_TRUE(get());
    set(initial);
}

} // namespace
