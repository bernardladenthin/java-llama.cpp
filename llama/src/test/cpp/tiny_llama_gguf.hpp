// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// A tiny random llama-architecture GGUF written with the public gguf API, shared by the tests that
// drive a real model without downloading one (test_c_api.cpp, test_lora_cache.cpp). The output of
// such a model is noise; it is deterministic for a given seed, which is all a contract test needs.

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

namespace jllama_test {

constexpr int N_VOCAB = 256; // one GPT-2 byte token per byte
constexpr int N_EMBD = 32;
constexpr int N_HEAD = 4;
constexpr int N_HEAD_KV = 4;
constexpr int HEAD_DIM = 8;
constexpr int N_FF = 64;
constexpr int N_LAYER = 2;
constexpr int N_CTX_TRAIN = 128;

// The 256 printable code points GPT-2's byte-level BPE maps the bytes to, as UTF-8.
inline std::vector<std::string> byte_tokens() {
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

inline ggml_tensor *add_random_tensor(gguf_context *g, ggml_context *ctx, std::mt19937 &rng, const std::string &name,
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
inline void confine_output_to_ascii(ggml_tensor *output) {
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
// load it, apply its fallback chat template and generate tokens. With fim, the byte tokens 1, 2 and
// 3 are declared its fill-in-the-middle prefix, suffix and middle tokens, which is all llama.cpp needs
// to format an infill prompt (the bytes never occur in the tests' text). The end-of-generation token
// is the byte eos, '>' unless a test needs to generate that character as text. Each string of special
// is appended to the vocabulary as one control token (ids 256, 257, ...), the way a model's vocabulary
// holds its chat-template tags; their output rows are scaled down like every non-printable byte's, so
// they are generated only when a sampler forces them.
inline std::string write_tiny_llama_gguf(const std::string &prefix = "jllama-c-api", bool fim = false,
                                         uint32_t eos = '>', const std::vector<std::string> &special = {}) {
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
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

    std::vector<std::string> tokens = byte_tokens();
    tokens.insert(tokens.end(), special.begin(), special.end());
    const int n_vocab = static_cast<int>(tokens.size());
    std::vector<const char *> token_ptrs;
    for (const std::string &s : tokens) {
        token_ptrs.push_back(s.c_str());
    }
    std::vector<int32_t> token_types(N_VOCAB, 1); // LLAMA_TOKEN_TYPE_NORMAL
    token_types.resize(n_vocab, 3);               // LLAMA_TOKEN_TYPE_CONTROL
    const char *no_merges[] = {"Ġ Ġ"};
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", no_merges, 1);
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", '<');
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", eos);
    if (fim) {
        gguf_set_val_u32(g, "tokenizer.ggml.fim_pre_token_id", 1);
        gguf_set_val_u32(g, "tokenizer.ggml.fim_suf_token_id", 2);
        gguf_set_val_u32(g, "tokenizer.ggml.fim_mid_token_id", 3);
    }

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {N_EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {N_EMBD}, 1.0f);
    confine_output_to_ascii(add_random_tensor(g, ctx, rng, "output.weight", {N_EMBD, n_vocab}, 0.1f));
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

// A random llava-style (MLP) vision projector for the model above: 32x32 images in 16x16 patches,
// a two-layer encoder of width 16 and a projection to the model's N_EMBD. Loaded with --mmproj it
// makes the server multimodal, so the image paths run end to end without a download; like the
// model, its answers are noise. llava's preprocessing resizes every image to 32x32, so any image
// stb_image decodes will do.
inline std::string write_tiny_mmproj_gguf(const std::string &prefix = "jllama-mmproj") {
    constexpr int IMG = 32, PATCH = 16, V_EMBD = 16, V_HEAD = 2, V_FF = 32, V_LAYER = 2, PROJ = 32;
    constexpr int N_PATCHES = (IMG / PATCH) * (IMG / PATCH);
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(7);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_str(g, "clip.projector_type", "mlp");
    gguf_set_val_bool(g, "clip.has_vision_encoder", true);
    gguf_set_val_bool(g, "clip.use_gelu", true);
    gguf_set_val_u32(g, "clip.vision.image_size", IMG);
    gguf_set_val_u32(g, "clip.vision.patch_size", PATCH);
    gguf_set_val_u32(g, "clip.vision.embedding_length", V_EMBD);
    gguf_set_val_u32(g, "clip.vision.feed_forward_length", V_FF);
    gguf_set_val_u32(g, "clip.vision.block_count", V_LAYER);
    gguf_set_val_u32(g, "clip.vision.projection_dim", PROJ);
    gguf_set_val_u32(g, "clip.vision.attention.head_count", V_HEAD);
    gguf_set_val_f32(g, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
    const float mean_std[] = {0.5f, 0.5f, 0.5f};
    gguf_set_arr_data(g, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean_std, 3);
    gguf_set_arr_data(g, "clip.vision.image_std", GGUF_TYPE_FLOAT32, mean_std, 3);

    ggml_init_params ip = {4u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: small noise (weights, biases); otherwise every element is value (norms).
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        auto *data = (float *)t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            data[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    };
    add("v.patch_embd.weight", {PATCH, PATCH, 3, V_EMBD}, -1);
    add("v.patch_embd.bias", {V_EMBD}, -1);
    add("v.position_embd.weight", {V_EMBD, N_PATCHES}, -1);
    for (int il = 0; il < V_LAYER; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        for (const char *n : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            add(p + n + ".weight", {V_EMBD, V_EMBD}, -1);
            add(p + n + ".bias", {V_EMBD}, -1);
        }
        for (const char *n : {"ln1", "ln2"}) {
            add(p + n + ".weight", {V_EMBD}, 1.0f);
            add(p + n + ".bias", {V_EMBD}, 0.0f);
        }
        add(p + "ffn_up.weight", {V_EMBD, V_FF}, -1);
        add(p + "ffn_up.bias", {V_FF}, -1);
        add(p + "ffn_down.weight", {V_FF, V_EMBD}, -1);
        add(p + "ffn_down.bias", {V_EMBD}, -1);
    }
    add("mm.0.weight", {V_EMBD, PROJ}, -1);
    add("mm.0.bias", {PROJ}, -1);
    add("mm.2.weight", {PROJ, N_EMBD}, -1);
    add("mm.2.bias", {N_EMBD}, -1);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// A 2x2 red PNG, base64: an image stb_image decodes, small enough to inline, for the projector above.
constexpr const char *TINY_PNG_BASE64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

} // namespace jllama_test
