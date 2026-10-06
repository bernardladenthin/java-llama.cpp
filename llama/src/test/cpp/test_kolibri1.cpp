// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Runnable guard for patches/0016 (Aleph Alpha Kolibri-1, architecture "kolibri1").
//
// Writes tiny random Kolibri-1 GGUFs, runs them through the real library on the CPU and compares
// every logit with an independent reference written from Aleph Alpha's own vLLM implementation
// (aleph_alpha_inference/kolibri1.py at 049a6a7) -- not from the patch. The model is too small to
// mean anything, but it exercises each architectural decision the patch had to get right:
//
//  - RoPE on sliding-window layers only (the layer pattern puts a full-attention layer in the
//    middle, so a "rope every layer" or "no rope at all" graph is off immediately);
//  - the sliding-window mask, with sequences longer than the window, both as one batch and token
//    by token through the iSWA KV cache;
//  - the router: top-k on logits + expert_bias, weights = unbiased sigmoid(logits). The biases are
//    large enough that DeepSeek-V3's router (top-k on sigmoid(logits) + bias) picks other experts;
//    one test asserts exactly that, so the comparison cannot pass vacuously;
//  - sandwich norms, the ungated shared expert, optional renormalization (norm_topk_prob);
//  - both published GGUF dialects (gating_func 2 + pre "qwen2" + output.weight, and gating_func 5
//    + pre "kolibri1" + tied output), and the rejection of a gating function that is neither.
//
// A llama.cpp bump that drops the patch fails these at load time ("unknown model architecture").
// When upstream adds kolibri1 itself and 0016 is dropped, these tests are what must keep passing.

#include "llama.h"
#include "gguf.h"
#include "ggml.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N_VOCAB = 256; // one GPT-2 byte token per byte
constexpr int N_EMBD = 64;
constexpr int N_HEAD = 4;
constexpr int N_HEAD_KV = 2;
constexpr int HEAD_DIM = 16;
constexpr int N_EXPERT = 8;
constexpr int N_EXPERT_USED = 2;
constexpr int N_FF_EXP = 32;
constexpr int N_FF_SHEXP = 48;
constexpr int N_SWA = 4;
constexpr int N_CTX_TRAIN = 128;
constexpr float RMS_EPS = 1e-6f;
constexpr float ROPE_BASE = 10000.0f;
// sliding, sliding, FULL, sliding, FULL -- a full layer in the middle, unlike the default 4:1 pattern
const std::vector<bool> IS_SWA = {true, true, false, true, false};
const int N_LAYER = (int)IS_SWA.size();

using Mat = std::vector<float>; // row-major [rows][cols], i.e. the GGML layout of ne = {cols, rows}

struct Layer {
    Mat attn_norm, attn_post_norm, wq, wk, wv, wo, q_norm, k_norm;
    Mat ffn_norm, ffn_post_norm, gate_inp, exp_bias;
    std::vector<Mat> exp_gate, exp_up, exp_down; // per expert: [n_ff][n_embd], [n_ff][n_embd], [n_embd][n_ff]
    Mat sh_gate, sh_up, sh_down;
};

struct Weights {
    Mat tok_embd, output_norm, output; // output empty = tied to tok_embd
    std::vector<Layer> layers;
};

Mat random(std::mt19937 &rng, size_t n, float scale, float offset = 0.0f) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Mat m(n);
    for (float &v : m) {
        v = offset + scale * dist(rng);
    }
    return m;
}

Weights make_weights(uint32_t seed, bool tied_output) {
    std::mt19937 rng(seed);
    Weights w;
    w.tok_embd = random(rng, (size_t)N_VOCAB * N_EMBD, 1.0f);
    w.output_norm = random(rng, N_EMBD, 0.1f, 1.0f);
    if (!tied_output) {
        w.output = random(rng, (size_t)N_VOCAB * N_EMBD, 0.2f);
    }
    for (int il = 0; il < N_LAYER; ++il) {
        Layer l;
        l.attn_norm = random(rng, N_EMBD, 0.1f, 1.0f);
        l.attn_post_norm = random(rng, N_EMBD, 0.1f, 1.0f);
        l.wq = random(rng, (size_t)N_HEAD * HEAD_DIM * N_EMBD, 0.2f);
        l.wk = random(rng, (size_t)N_HEAD_KV * HEAD_DIM * N_EMBD, 0.2f);
        l.wv = random(rng, (size_t)N_HEAD_KV * HEAD_DIM * N_EMBD, 0.2f);
        l.wo = random(rng, (size_t)N_EMBD * N_HEAD * HEAD_DIM, 0.2f);
        l.q_norm = random(rng, HEAD_DIM, 0.1f, 1.0f);
        l.k_norm = random(rng, HEAD_DIM, 0.1f, 1.0f);
        l.ffn_norm = random(rng, N_EMBD, 0.1f, 1.0f);
        l.ffn_post_norm = random(rng, N_EMBD, 0.1f, 1.0f);
        l.gate_inp = random(rng, (size_t)N_EXPERT * N_EMBD, 0.5f);
        // trained Kolibri biases reach ~20; large biases are what separate the two routers
        l.exp_bias = random(rng, N_EXPERT, 5.0f);
        for (int e = 0; e < N_EXPERT; ++e) {
            l.exp_gate.push_back(random(rng, (size_t)N_FF_EXP * N_EMBD, 0.2f));
            l.exp_up.push_back(random(rng, (size_t)N_FF_EXP * N_EMBD, 0.2f));
            l.exp_down.push_back(random(rng, (size_t)N_EMBD * N_FF_EXP, 0.2f));
        }
        l.sh_gate = random(rng, (size_t)N_FF_SHEXP * N_EMBD, 0.2f);
        l.sh_up = random(rng, (size_t)N_FF_SHEXP * N_EMBD, 0.2f);
        l.sh_down = random(rng, (size_t)N_EMBD * N_FF_SHEXP, 0.2f);
        w.layers.push_back(std::move(l));
    }
    return w;
}

// ---------------------------------------------------------------------------------------------
// The reference: Kolibri1DecoderLayer / Kolibri1Attention / sigmoid_logit_add_routing, in double.
// ---------------------------------------------------------------------------------------------

enum class Router { KOLIBRI, DEEPSEEK };

using Vec = std::vector<double>;

Vec matvec(const Mat &m, const Vec &x, int rows, int cols) {
    Vec y(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            y[r] += (double)m[(size_t)r * cols + c] * x[c];
        }
    }
    return y;
}

Vec rms_norm(const Vec &x, const Mat &w, size_t off = 0, size_t n = 0) {
    if (n == 0) {
        n = x.size();
    }
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ss += x[off + i] * x[off + i];
    }
    const double scale = 1.0 / std::sqrt(ss / (double)n + (double)RMS_EPS);
    Vec y(n);
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[off + i] * scale * (double)w[i];
    }
    return y;
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }
double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// NEOX rotary embedding over one head: dimension i pairs with i + HEAD_DIM/2
void rope_neox(Vec &h, int pos) {
    const int half = HEAD_DIM / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = (double)pos * std::pow((double)ROPE_BASE, -2.0 * i / HEAD_DIM);
        const double c = std::cos(theta), s = std::sin(theta);
        const double x0 = h[i], x1 = h[i + half];
        h[i] = x0 * c - x1 * s;
        h[i + half] = x0 * s + x1 * c;
    }
}

Vec swiglu(const Mat &gate, const Mat &up, const Mat &down, const Vec &x, int n_ff) {
    const Vec g = matvec(gate, x, n_ff, N_EMBD);
    const Vec u = matvec(up, x, n_ff, N_EMBD);
    Vec a(n_ff);
    for (int i = 0; i < n_ff; ++i) {
        a[i] = silu(g[i]) * u[i];
    }
    return matvec(down, a, N_EMBD, n_ff);
}

std::vector<int> route(const Layer &l, const Vec &logits, Router router) {
    std::vector<double> score(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) {
        score[e] = router == Router::KOLIBRI ? logits[e] + l.exp_bias[e] : sigmoid(logits[e]) + l.exp_bias[e];
    }
    std::vector<int> ids(N_EXPERT);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + N_EXPERT_USED, ids.end(),
                      [&](int a, int b) { return score[a] > score[b]; });
    ids.resize(N_EXPERT_USED);
    return ids;
}

// Logits of every position of `tokens` (one causal pass), [n_tokens][N_VOCAB].
std::vector<Vec> reference(const Weights &w, const std::vector<llama_token> &tokens, bool renorm,
                           Router router = Router::KOLIBRI) {
    const int n = (int)tokens.size();
    std::vector<Vec> x(n, Vec(N_EMBD));
    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < N_EMBD; ++i) {
            x[t][i] = w.tok_embd[(size_t)tokens[t] * N_EMBD + i];
        }
    }
    const int group = N_HEAD / N_HEAD_KV;
    for (int il = 0; il < N_LAYER; ++il) {
        const Layer &l = w.layers[il];
        // q/k/v of every position, q/k normed per head, RoPE on sliding-window layers only
        std::vector<Vec> q(n), k(n), v(n);
        for (int t = 0; t < n; ++t) {
            const Vec h = rms_norm(x[t], l.attn_norm);
            const Vec qr = matvec(l.wq, h, N_HEAD * HEAD_DIM, N_EMBD);
            const Vec kr = matvec(l.wk, h, N_HEAD_KV * HEAD_DIM, N_EMBD);
            v[t] = matvec(l.wv, h, N_HEAD_KV * HEAD_DIM, N_EMBD);
            q[t].resize(qr.size());
            k[t].resize(kr.size());
            for (int hd = 0; hd < N_HEAD; ++hd) {
                Vec qh = rms_norm(qr, l.q_norm, (size_t)hd * HEAD_DIM, HEAD_DIM);
                if (IS_SWA[il]) {
                    rope_neox(qh, t);
                }
                std::copy(qh.begin(), qh.end(), q[t].begin() + (size_t)hd * HEAD_DIM);
            }
            for (int hd = 0; hd < N_HEAD_KV; ++hd) {
                Vec kh = rms_norm(kr, l.k_norm, (size_t)hd * HEAD_DIM, HEAD_DIM);
                if (IS_SWA[il]) {
                    rope_neox(kh, t);
                }
                std::copy(kh.begin(), kh.end(), k[t].begin() + (size_t)hd * HEAD_DIM);
            }
        }
        std::vector<Vec> next(n);
        for (int t = 0; t < n; ++t) {
            // causal attention; a sliding-window layer sees the last N_SWA positions (itself included)
            Vec attn(N_HEAD * HEAD_DIM, 0.0);
            for (int hd = 0; hd < N_HEAD; ++hd) {
                const int kvh = hd / group;
                const int first = IS_SWA[il] ? std::max(0, t - N_SWA + 1) : 0;
                std::vector<double> s;
                double mx = -1e300;
                for (int p = first; p <= t; ++p) {
                    double d = 0.0;
                    for (int i = 0; i < HEAD_DIM; ++i) {
                        d += q[t][(size_t)hd * HEAD_DIM + i] * k[p][(size_t)kvh * HEAD_DIM + i];
                    }
                    s.push_back(d / std::sqrt((double)HEAD_DIM));
                    mx = std::max(mx, s.back());
                }
                double sum = 0.0;
                for (double &e : s) {
                    e = std::exp(e - mx);
                    sum += e;
                }
                for (int p = first; p <= t; ++p) {
                    const double pr = s[p - first] / sum;
                    for (int i = 0; i < HEAD_DIM; ++i) {
                        attn[(size_t)hd * HEAD_DIM + i] += pr * v[p][(size_t)kvh * HEAD_DIM + i];
                    }
                }
            }
            const Vec o = rms_norm(matvec(l.wo, attn, N_EMBD, N_HEAD * HEAD_DIM), l.attn_post_norm);
            Vec x1(N_EMBD);
            for (int i = 0; i < N_EMBD; ++i) {
                x1[i] = x[t][i] + o[i];
            }
            // MoE: route, routed experts weighted by the unbiased sigmoid, plus the shared expert
            const Vec h2 = rms_norm(x1, l.ffn_norm);
            const Vec logits = matvec(l.gate_inp, h2, N_EXPERT, N_EMBD);
            const std::vector<int> sel = route(l, logits, router);
            std::vector<double> wt;
            double wsum = 0.0;
            for (int e : sel) {
                wt.push_back(sigmoid(logits[e]));
                wsum += wt.back();
            }
            Vec ffn = swiglu(l.sh_gate, l.sh_up, l.sh_down, h2, N_FF_SHEXP);
            for (size_t j = 0; j < sel.size(); ++j) {
                const double weight = renorm ? wt[j] / wsum : wt[j];
                const Vec y = swiglu(l.exp_gate[sel[j]], l.exp_up[sel[j]], l.exp_down[sel[j]], h2, N_FF_EXP);
                for (int i = 0; i < N_EMBD; ++i) {
                    ffn[i] += weight * y[i];
                }
            }
            const Vec f = rms_norm(ffn, l.ffn_post_norm);
            next[t].resize(N_EMBD);
            for (int i = 0; i < N_EMBD; ++i) {
                next[t][i] = x1[i] + f[i];
            }
        }
        x = std::move(next);
    }
    const Mat &out = w.output.empty() ? w.tok_embd : w.output;
    std::vector<Vec> res(n);
    for (int t = 0; t < n; ++t) {
        res[t] = matvec(out, rms_norm(x[t], w.output_norm), N_VOCAB, N_EMBD);
    }
    return res;
}

// ---------------------------------------------------------------------------------------------
// GGUF writer (public gguf API; F32 tensors, GGML ne = {cols, rows[, n_expert]})
// ---------------------------------------------------------------------------------------------

struct Dialect {
    uint32_t gating_func; // 0 = key absent
    const char *pre;
    bool write_weights_norm;
    bool renorm;
    bool tied_output;
};

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

void add_tensor(gguf_context *g, ggml_context *ctx, const std::string &name, const Mat &data,
                std::initializer_list<int64_t> ne) {
    ggml_tensor *t = nullptr;
    const std::vector<int64_t> d(ne);
    if (d.size() == 1) {
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d[0]);
    } else if (d.size() == 2) {
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d[0], d[1]);
    } else {
        t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d[0], d[1], d[2]);
    }
    ASSERT_EQ((size_t)ggml_nelements(t), data.size()) << name;
    std::copy(data.begin(), data.end(), (float *)t->data);
    ggml_set_name(t, name.c_str());
    gguf_add_tensor(g, t);
}

Mat concat(const std::vector<Mat> &parts) {
    Mat m;
    for (const Mat &p : parts) {
        m.insert(m.end(), p.begin(), p.end());
    }
    return m;
}

std::string write_gguf(const Weights &w, const Dialect &d) {
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("jllama-kolibri1-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();

    gguf_context *g = gguf_init_empty();
    const std::string a = "kolibri1";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".context_length").c_str(), N_CTX_TRAIN);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), N_EMBD);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), N_LAYER);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), N_FF_EXP);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD_KV);
    gguf_set_val_u32(g, (a + ".attention.key_length").c_str(), HEAD_DIM);
    gguf_set_val_u32(g, (a + ".attention.value_length").c_str(), HEAD_DIM);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), RMS_EPS);
    gguf_set_val_u32(g, (a + ".attention.sliding_window").c_str(), N_SWA);
    std::vector<int8_t> pattern(IS_SWA.begin(), IS_SWA.end()); // gguf bool is one byte
    gguf_set_arr_data(g, (a + ".attention.sliding_window_pattern").c_str(), GGUF_TYPE_BOOL, pattern.data(),
                      pattern.size());
    gguf_set_val_u32(g, (a + ".rope.dimension_count").c_str(), HEAD_DIM);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), ROPE_BASE);
    gguf_set_val_u32(g, (a + ".expert_count").c_str(), N_EXPERT);
    gguf_set_val_u32(g, (a + ".expert_used_count").c_str(), N_EXPERT_USED);
    gguf_set_val_u32(g, (a + ".expert_feed_forward_length").c_str(), N_FF_EXP);
    gguf_set_val_u32(g, (a + ".expert_shared_feed_forward_length").c_str(), N_FF_SHEXP);
    gguf_set_val_u32(g, (a + ".expert_shared_count").c_str(), 1);
    if (d.write_weights_norm) {
        gguf_set_val_bool(g, (a + ".expert_weights_norm").c_str(), d.renorm);
    }
    if (d.gating_func != 0) {
        gguf_set_val_u32(g, (a + ".expert_gating_func").c_str(), d.gating_func);
    }

    const std::vector<std::string> tokens = byte_tokens();
    std::vector<const char *> token_ptrs;
    for (const std::string &s : tokens) {
        token_ptrs.push_back(s.c_str());
    }
    std::vector<int32_t> token_types(N_VOCAB, 1); // LLAMA_TOKEN_TYPE_NORMAL
    const char *no_merges[] = {"Ġ Ġ"};
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", d.pre);
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", no_merges, 1);

    ggml_init_params ip = {64u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_tensor(g, ctx, "token_embd.weight", w.tok_embd, {N_EMBD, N_VOCAB});
    add_tensor(g, ctx, "output_norm.weight", w.output_norm, {N_EMBD});
    if (!w.output.empty()) {
        add_tensor(g, ctx, "output.weight", w.output, {N_EMBD, N_VOCAB});
    }
    for (int il = 0; il < N_LAYER; ++il) {
        const Layer &l = w.layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        add_tensor(g, ctx, p + "attn_norm.weight", l.attn_norm, {N_EMBD});
        add_tensor(g, ctx, p + "post_attention_norm.weight", l.attn_post_norm, {N_EMBD});
        add_tensor(g, ctx, p + "attn_q.weight", l.wq, {N_EMBD, N_HEAD * HEAD_DIM});
        add_tensor(g, ctx, p + "attn_k.weight", l.wk, {N_EMBD, N_HEAD_KV * HEAD_DIM});
        add_tensor(g, ctx, p + "attn_v.weight", l.wv, {N_EMBD, N_HEAD_KV * HEAD_DIM});
        add_tensor(g, ctx, p + "attn_output.weight", l.wo, {N_HEAD * HEAD_DIM, N_EMBD});
        add_tensor(g, ctx, p + "attn_q_norm.weight", l.q_norm, {HEAD_DIM});
        add_tensor(g, ctx, p + "attn_k_norm.weight", l.k_norm, {HEAD_DIM});
        add_tensor(g, ctx, p + "ffn_norm.weight", l.ffn_norm, {N_EMBD});
        add_tensor(g, ctx, p + "post_ffw_norm.weight", l.ffn_post_norm, {N_EMBD});
        add_tensor(g, ctx, p + "ffn_gate_inp.weight", l.gate_inp, {N_EMBD, N_EXPERT});
        add_tensor(g, ctx, p + "exp_probs_b.bias", l.exp_bias, {N_EXPERT});
        add_tensor(g, ctx, p + "ffn_gate_exps.weight", concat(l.exp_gate), {N_EMBD, N_FF_EXP, N_EXPERT});
        add_tensor(g, ctx, p + "ffn_up_exps.weight", concat(l.exp_up), {N_EMBD, N_FF_EXP, N_EXPERT});
        add_tensor(g, ctx, p + "ffn_down_exps.weight", concat(l.exp_down), {N_FF_EXP, N_EMBD, N_EXPERT});
        add_tensor(g, ctx, p + "ffn_gate_shexp.weight", l.sh_gate, {N_EMBD, N_FF_SHEXP});
        add_tensor(g, ctx, p + "ffn_up_shexp.weight", l.sh_up, {N_EMBD, N_FF_SHEXP});
        add_tensor(g, ctx, p + "ffn_down_shexp.weight", l.sh_down, {N_FF_SHEXP, N_EMBD});
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// ---------------------------------------------------------------------------------------------
// Running the library
// ---------------------------------------------------------------------------------------------

struct Loaded {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    ~Loaded() {
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
    }
};

void quiet_log(ggml_log_level, const char *, void *) {}

bool load(const std::string &path, Loaded &out) {
    llama_backend_init();
    llama_log_set(quiet_log, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    out.model = llama_model_load_from_file(path.c_str(), mp);
    llama_log_set(nullptr, nullptr);
    if (out.model == nullptr) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    out.ctx = llama_init_from_model(out.model, cp);
    return out.ctx != nullptr;
}

// logits of every position, decoding the whole sequence in one batch
std::vector<std::vector<float>> run_batch(Loaded &m, const std::vector<llama_token> &tokens) {
    llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos)i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = (int32_t)tokens.size();
    std::vector<std::vector<float>> res;
    if (llama_decode(m.ctx, batch) == 0) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            const float *l = llama_get_logits_ith(m.ctx, (int32_t)i);
            res.emplace_back(l, l + N_VOCAB);
        }
    }
    llama_batch_free(batch);
    return res;
}

// logits of every position, decoding token by token through the KV cache
std::vector<std::vector<float>> run_incremental(Loaded &m, const std::vector<llama_token> &tokens) {
    llama_memory_clear(llama_get_memory(m.ctx), true);
    std::vector<std::vector<float>> res;
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[0] = tokens[i];
        batch.pos[0] = (llama_pos)i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        if (llama_decode(m.ctx, batch) != 0) {
            res.clear();
            break;
        }
        const float *l = llama_get_logits_ith(m.ctx, 0);
        res.emplace_back(l, l + N_VOCAB);
    }
    llama_batch_free(batch);
    return res;
}

double max_abs_diff(const std::vector<std::vector<float>> &got, const std::vector<Vec> &want) {
    double d = 0.0;
    for (size_t t = 0; t < want.size(); ++t) {
        for (int i = 0; i < N_VOCAB; ++i) {
            d = std::max(d, std::fabs((double)got[t][i] - want[t][i]));
        }
    }
    return d;
}

double max_abs(const std::vector<Vec> &v) {
    double m = 0.0;
    for (const Vec &r : v) {
        for (double x : r) {
            m = std::max(m, std::fabs(x));
        }
    }
    return m;
}

const std::vector<llama_token> TOKENS = {3, 141, 59, 26, 53, 58, 97, 93, 238, 46, 26, 43}; // 12 > N_SWA

void expect_matches_reference(const Dialect &d, uint32_t seed) {
    const Weights w = make_weights(seed, d.tied_output);
    const std::string path = write_gguf(w, d);
    ASSERT_FALSE(path.empty());
    {
        Loaded m;
        ASSERT_TRUE(load(path, m)) << "the library did not load the kolibri1 GGUF (pre=" << d.pre
                                   << ", gating_func=" << d.gating_func << ")";

        const std::vector<Vec> want = reference(w, TOKENS, d.renorm);
        const double tol = 1e-3 * std::max(1.0, max_abs(want));

        const auto batch = run_batch(m, TOKENS);
        ASSERT_EQ(batch.size(), TOKENS.size());
        EXPECT_LT(max_abs_diff(batch, want), tol) << "batch decode";

        const auto inc = run_incremental(m, TOKENS);
        ASSERT_EQ(inc.size(), TOKENS.size());
        EXPECT_LT(max_abs_diff(inc, want), tol) << "token-by-token decode through the iSWA KV cache";
    }
    std::remove(path.c_str());
}

} // namespace

// The AFMoE-based converter's dialect: gating_func 2, pre "qwen2", an explicit output.weight.
TEST(Kolibri1, AfmoeDialectMatchesTheReference) { expect_matches_reference({2, "qwen2", true, false, false}, 1); }

// The Qwen3-MoE-based converter's dialect: gating_func 5, pre "kolibri1", tied output, no norm key.
TEST(Kolibri1, Qwen3MoeDialectWithTiedOutputMatchesTheReference) {
    expect_matches_reference({5, "kolibri1", false, false, true}, 2);
}

// norm_topk_prob = true renormalizes the selected sigmoid weights (Kolibri-1 itself ships it off).
TEST(Kolibri1, RenormalizedRoutingMatchesTheReference) { expect_matches_reference({0, "qwen2", true, true, false}, 3); }

// Keeps the comparisons above honest: with these biases DeepSeek-V3's router (top-k on
// sigmoid(logits) + bias) gives a different model, so a graph that used it would fail them.
TEST(Kolibri1, TheDeepSeekRouterWouldGiveDifferentLogits) {
    const Weights w = make_weights(1, false);
    const std::vector<Vec> kolibri = reference(w, TOKENS, false, Router::KOLIBRI);
    const std::vector<Vec> deepseek = reference(w, TOKENS, false, Router::DEEPSEEK);
    double d = 0.0;
    for (size_t t = 0; t < kolibri.size(); ++t) {
        for (int i = 0; i < N_VOCAB; ++i) {
            d = std::max(d, std::fabs(kolibri[t][i] - deepseek[t][i]));
        }
    }
    EXPECT_GT(d, 1e-1 * std::max(1.0, max_abs(kolibri)));
}

// Only the router Kolibri-1 has is accepted; a GGUF claiming another gating function is refused.
TEST(Kolibri1, AnotherGatingFunctionIsRejected) {
    const Weights w = make_weights(4, false);
    const std::string path = write_gguf(w, {1, "qwen2", true, false, false}); // 1 = softmax
    ASSERT_FALSE(path.empty());
    {
        Loaded m;
        EXPECT_FALSE(load(path, m));
    }
    std::remove(path.c_str());
}
