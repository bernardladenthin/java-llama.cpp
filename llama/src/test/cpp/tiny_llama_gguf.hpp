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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <random>
#include <cstdio>
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
    ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(d.size()), d.data());
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
// they are generated only when a sampler forces them. With shard_path, the tensors of the last layer
// are written to a second GGUF in a directory of its own instead -- an extra shard as --shard takes
// it (llama.cpp b12233): tensors only, no metadata, a name the split numbering does not know. The
// weights are the ones the unsplit model gets (same seed, same order), so both answer alike.
// The shape of one random model. The defaults are the model every test used before the struct
// existed, and a default spec writes exactly that file (same seed, same tensor order), so a test
// that compares answers across models keeps its meaning. A test that needs another shape names only
// what differs: wider rows for a quantization type with a block size (Q1_0 wants row lengths
// divisible by 128), a pooling type declared by the model itself, more layers.
struct tiny_model_spec {
    std::string prefix = "jllama-c-api";
    bool fim = false;
    uint32_t eos = '>';
    // Declares the end token a control token, as a real model's is: its piece is then not part of
    // the text (a normal token's is, so an answer of the default fixture ends with the byte).
    bool eos_is_control = false;
    std::vector<std::string> special = {};
    int n_embd = N_EMBD;
    int n_head = N_HEAD;
    int n_head_kv = N_HEAD_KV;
    int head_dim = HEAD_DIM;
    int n_ff = N_FF;
    int n_layer = N_LAYER;
    int n_ctx_train = N_CTX_TRAIN;
    int pooling_type = -1; // llama_pooling_type written as <arch>.pooling_type; -1 writes none
    // A mixture of experts when n_expert > 0: every layer's feed-forward is a router
    // (ffn_gate_inp) over n_expert experts, of which n_expert_used answer each token.
    int n_expert = 0;
    int n_expert_used = 0;
    // The architecture. "llama" is the plain decoder; "hy_v3" (dense, no experts) is the smallest
    // one llama.cpp has a multi-token-prediction graph for: the same decoder with a norm on the
    // query and key heads, and after the n_layer trunk layers n_layer_nextn MTP blocks -- a decoder
    // block each, plus the projection that joins a token embedding with the trunk's hidden state.
    std::string arch = "llama";
    int n_layer_nextn = 0;
    // A sliding attention window of n_swa positions, with arch "cohere2": the llama decoder without
    // the norm before the feed-forward and with the output tied to the token embedding, every layer
    // attending to the last n_swa positions only. Such a model cannot roll its memory back to an
    // earlier position, so the server keeps context checkpoints for it and appends them to a slot
    // file -- the one path the plain decoder never takes.
    int n_swa = 0;
    // arch "mamba" is a recurrent model: no attention and no KV cache, a state per layer that is
    // overwritten with every token (a convolution window and the state-space model's hidden state).
    // The server can only checkpoint such a model, and a slot file holds its state instead of rows.
    // A decision model, as /v1/systemone asks for: <arch>.decision.type names how the answers are
    // read, and the model carries a "systemone" chat template that renders one question. "openjev"
    // is the type a plain decoder can be: the answer is the distribution over the label tokens A, B,
    // ... at the end of the prompt, one single-byte token each in this vocabulary.
    std::string decision_type = {};
    // KV cache scales: one F32 scalar per layer and side (blk.N.attn_k.k_scale, attn_v.v_scale), which
    // an 8-bit float cache (--cache-type-k/-v f8_e4m3) divides K and V by before it stores them and
    // multiplies back in the attention, so the stored values fit the format's range. No other cache
    // type reads them. A scale that is zero or not finite fails the context with such a cache.
    bool cache_scales = false;
    float k_cache_scale = 1.0f;
    float v_cache_scale = 1.0f;
    // Weight scales (NVFP4's second scale, an FP8 checkpoint's per-tensor scale): an F32 tensor
    // <matrix>.scale next to a weight matrix, one value ({1}) or one per expert ({n_expert}), by which
    // llama.cpp multiplies the matrix's product (build_lora_mm). Written next to every matrix that has
    // one in llama.cpp's generic loader -- attention, feed-forward, experts, shared experts, the
    // recurrent in/out projections, the output -- when weight_scale is not 0. With
    // weight_scale_folded the matrices hold their weights times the scale instead and no scale tensor
    // is written: the same random draws, so the two files are one model written two ways.
    float weight_scale = 0.0f;
    bool weight_scale_folded = false;
    // The vocabulary. BYTES is one GPT-2 byte token per byte and no merge that ever applies, so a
    // text is as many tokens as bytes. BPE_MERGES adds the merges that make "hello" and " world" one
    // token each (ids 256 + 3 and 256 + 8), the way a real BPE vocabulary works. SENTENCEPIECE is the
    // other tokenizer family ("llama"): scored pieces with U+2581 for a space, <unk>/<s>/</s> as
    // ids 0/1/2, every printable ASCII character as a piece and the pieces that build "hello" and
    // "world" as words.
    // SENTENCEPIECE_BYTES adds the 256 byte pieces (<0x00> .. <0xFF>) a real SentencePiece vocabulary
    // falls back to for a character it has no piece for. UNIGRAM is T5's tokenizer ("t5": the best
    // scoring segmentation, <pad> </s> <unk> first), RWKV the greedy longest match over byte strings
    // ("rwkv": escaped tokens, no special ones of its own).
    enum class tokenizer_kind { BYTES, BPE_MERGES, SENTENCEPIECE, SENTENCEPIECE_BYTES, UNIGRAM, RWKV };
    tokenizer_kind tokenizer = tokenizer_kind::BYTES;

    // A decoder with one MTP block, which --spec-type draft-mtp drafts with.
    static tiny_model_spec mtp(const std::string &prefix = "jllama-mtp") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.arch = "hy_v3";
        spec.n_layer_nextn = 1;
        return spec;
    }

    // A sliding-window decoder, for which the server writes context checkpoints.
    static tiny_model_spec swa(int window = 16, const std::string &prefix = "jllama-swa") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.arch = "cohere2";
        spec.n_swa = window;
        return spec;
    }

    // A recurrent model (Mamba).
    static tiny_model_spec recurrent(const std::string &prefix = "jllama-mamba") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.arch = "mamba";
        return spec;
    }

    // Qwen3-Next: a gated delta net (layer 0) and gated attention (layer 1), both with a mixture of
    // experts and a shared expert behind them; the delta net reads its projections from attn_qkv and
    // attn_gate (the converter's layout, not the legacy ssm_in).
    static tiny_model_spec qwen3next(const std::string &prefix = "jllama-qwen3next") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.arch = "qwen3next";
        spec.n_head_kv = 2;
        spec.n_expert = 2;
        spec.n_expert_used = 1;
        return spec;
    }

    // A decision model of type "openjev" (see decision_type).
    static tiny_model_spec decision(const std::string &prefix = "jllama-decision") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.decision_type = "openjev";
        spec.n_ctx_train = 1024; // a question, its options and the state are one prompt
        return spec;
    }

    // A model whose every weight row is `width` long (width must be a multiple of the head count).
    static tiny_model_spec wide(int width, const std::string &prefix = "jllama-wide") {
        tiny_model_spec spec;
        spec.prefix = prefix;
        spec.n_embd = width;
        spec.head_dim = width / spec.n_head;
        spec.n_ff = 2 * width;
        return spec;
    }
};

inline std::string write_tiny_llama_gguf(const tiny_model_spec &spec, std::string *shard_path = nullptr) {
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (spec.prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(42);
    gguf_context *g = gguf_init_empty();
    const std::string a = spec.arch;
    const bool hy_v3 = spec.arch == "hy_v3";
    const bool cohere2 = spec.arch == "cohere2";
    const bool mamba = spec.arch == "mamba";
    const bool qwen3next = spec.arch == "qwen3next";
    constexpr int SSM_CONV = 4, SSM_STATE = 8, SSM_DT_RANK = 4;
    constexpr int GDN_HEADS = 2, GDN_HEAD_DIM = 8, GDN_INNER = GDN_HEADS * GDN_HEAD_DIM;
    const int ssm_inner = 2 * spec.n_embd; // the only expansion factor llama.cpp loads
    const int n_layer_all = spec.n_layer + spec.n_layer_nextn;
    gguf_set_val_str(g, "general.architecture", a.c_str());
    if (hy_v3) {
        gguf_set_val_u32(g, (a + ".expert_feed_forward_length").c_str(), static_cast<uint32_t>(spec.n_ff));
    }
    if (spec.n_layer_nextn > 0) {
        gguf_set_val_u32(g, (a + ".nextn_predict_layers").c_str(), static_cast<uint32_t>(spec.n_layer_nextn));
    }
    gguf_set_val_u32(g, (a + ".context_length").c_str(), spec.n_ctx_train);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), spec.n_embd);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), n_layer_all);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), spec.n_ff);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), spec.n_head);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), spec.n_head_kv);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(g, (a + ".rope.dimension_count").c_str(), spec.head_dim);
    if (mamba) {
        gguf_set_val_u32(g, (a + ".ssm.conv_kernel").c_str(), SSM_CONV);
        gguf_set_val_u32(g, (a + ".ssm.inner_size").c_str(), static_cast<uint32_t>(ssm_inner));
        gguf_set_val_u32(g, (a + ".ssm.state_size").c_str(), SSM_STATE);
        gguf_set_val_u32(g, (a + ".ssm.time_step_rank").c_str(), SSM_DT_RANK);
    }
    if (qwen3next) {
        gguf_set_val_u32(g, (a + ".expert_feed_forward_length").c_str(), static_cast<uint32_t>(spec.n_ff));
        gguf_set_val_u32(g, (a + ".expert_shared_feed_forward_length").c_str(), static_cast<uint32_t>(spec.n_ff));
        gguf_set_val_u32(g, (a + ".ssm.conv_kernel").c_str(), SSM_CONV);
        gguf_set_val_u32(g, (a + ".ssm.inner_size").c_str(), GDN_INNER);
        gguf_set_val_u32(g, (a + ".ssm.state_size").c_str(), GDN_HEAD_DIM);
        gguf_set_val_u32(g, (a + ".ssm.time_step_rank").c_str(), GDN_HEADS);
        gguf_set_val_u32(g, (a + ".ssm.group_count").c_str(), GDN_HEADS);
        gguf_set_val_u32(g, (a + ".full_attention_interval").c_str(), 2);
    }
    if (cohere2) {
        gguf_set_val_u32(g, (a + ".attention.sliding_window").c_str(), static_cast<uint32_t>(spec.n_swa));
        gguf_set_val_f32(g, (a + ".attention.layer_norm_epsilon").c_str(), 1e-5f);
        gguf_set_val_f32(g, (a + ".logit_scale").c_str(), 1.0f);
    }
    if (spec.n_expert > 0) {
        gguf_set_val_u32(g, (a + ".expert_count").c_str(), static_cast<uint32_t>(spec.n_expert));
        gguf_set_val_u32(g, (a + ".expert_used_count").c_str(), static_cast<uint32_t>(spec.n_expert_used));
    }
    if (!spec.decision_type.empty()) {
        gguf_set_val_str(g, (a + ".decision.type").c_str(), spec.decision_type.c_str());
        // the template gets the raw values of the request; what is no text is written as JSON
        gguf_set_val_str(g, "tokenizer.chat_template.systemone",
                         "Question: {{ instructions | tojson }}\nState: {{ state | tojson }}\n"
                         "Options: {{ options | tojson }}\nAnswer: ");
    }
    if (spec.pooling_type >= 0) {
        gguf_set_val_u32(g, (a + ".pooling_type").c_str(), static_cast<uint32_t>(spec.pooling_type));
    }

    using kind = tiny_model_spec::tokenizer_kind;
    const bool unigram = spec.tokenizer == kind::UNIGRAM;
    const bool rwkv = spec.tokenizer == kind::RWKV;
    // the two SentencePiece kinds and the Unigram one are scored pieces over U+2581
    const bool sentencepiece =
        spec.tokenizer == kind::SENTENCEPIECE || spec.tokenizer == kind::SENTENCEPIECE_BYTES || unigram;
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types; // 1 normal, 2 unknown, 3 control (llama_token_type)
    std::vector<float> scores;
    std::vector<std::string> merges = {"Ġ Ġ"}; // never applies: two spaces are two tokens
    if (sentencepiece) {
        const std::string sp = "\xE2\x96\x81"; // U+2581, SentencePiece's space
        const auto piece = [&](const std::string &text, float score, int32_t type = 1) {
            tokens.push_back(text);
            scores.push_back(score);
            token_types.push_back(type);
        };
        if (unigram) {
            piece("<pad>", 0.0f, 3);
            piece("</s>", 0.0f, 3);
            piece("<unk>", 0.0f, 2);
        } else {
            piece("<unk>", 0.0f, 2);
            piece("<s>", 0.0f, 3);
            piece("</s>", 0.0f, 3);
        }
        piece(sp, -2.0f);
        for (char c = '!'; c <= '~'; ++c) {
            piece(std::string(1, c), -10.0f);
        }
        // a word is reached pair by pair, the longer piece always scoring higher
        for (const std::string word : {"hello", "world"}) {
            for (size_t n = 1; n <= word.size(); ++n) {
                piece(sp + word.substr(0, n), -1.0f + 0.1f * static_cast<float>(n));
            }
        }
        if (spec.tokenizer == kind::SENTENCEPIECE_BYTES) {
            for (int byte = 0; byte < 256; ++byte) {
                char name[8];
                std::snprintf(name, sizeof(name), "<0x%02X>", byte);
                piece(name, 0.0f, 6); // LLAMA_TOKEN_TYPE_BYTE
            }
        }
    } else if (rwkv) {
        // one token per byte, written the way the RWKV vocabulary escapes them, then two words
        for (int byte = 0; byte < 256; ++byte) {
            char name[8];
            if (byte == '\\') {
                std::snprintf(name, sizeof(name), "\\\\");
            } else if (byte >= 0x20 && byte < 0x7F) {
                std::snprintf(name, sizeof(name), "%c", byte);
            } else {
                std::snprintf(name, sizeof(name), "\\x%02x", byte);
            }
            tokens.push_back(name);
        }
        tokens.push_back("hello");
        tokens.push_back(" world");
        token_types.assign(tokens.size(), 1);
    } else {
        tokens = byte_tokens();
        token_types.assign(tokens.size(), 1);
        if (spec.tokenizer == tiny_model_spec::tokenizer_kind::BPE_MERGES) {
            merges = {"h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld"};
            for (const char *merged : {"he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "ld", "Ġworld"}) {
                tokens.push_back(merged);
                token_types.push_back(1);
            }
        }
    }
    tokens.insert(tokens.end(), spec.special.begin(), spec.special.end());
    const int n_vocab = static_cast<int>(tokens.size());
    token_types.resize(n_vocab, 3);
    if (spec.eos_is_control && !sentencepiece) {
        token_types[spec.eos] = 3;
    }
    scores.resize(sentencepiece ? n_vocab : 0, 0.0f);
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    std::vector<const char *> merge_ptrs;
    for (const std::string &m : merges) {
        merge_ptrs.push_back(m.c_str());
    }
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    if (unigram) {
        gguf_set_val_str(g, "tokenizer.ggml.model", "t5");
        gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        // what the converter writes for a SentencePiece model with add_dummy_prefix, as T5's has;
        // llama.cpp's own default for this tokenizer is no space in front of the text
        gguf_set_val_bool(g, "tokenizer.ggml.add_space_prefix", true);
        gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", 0);
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 1);
        gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 2);
    } else if (sentencepiece) {
        gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
        gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);
    } else if (rwkv) {
        gguf_set_val_str(g, "tokenizer.ggml.model", "rwkv");
        gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", '<');
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", spec.eos);
    } else {
        gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
        gguf_set_arr_str(g, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());
        gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", '<');
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", spec.eos);
    }
    if (spec.fim) {
        gguf_set_val_u32(g, "tokenizer.ggml.fim_pre_token_id", 1);
        gguf_set_val_u32(g, "tokenizer.ggml.fim_suf_token_id", 2);
        gguf_set_val_u32(g, "tokenizer.ggml.fim_mid_token_id", 3);
    }

    ggml_init_params ip = {64u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // a weight matrix with the scale llama.cpp's generic loader reads next to it (see weight_scale)
    const auto matrix = [&](gguf_context *to, const std::string &name, std::initializer_list<int64_t> ne) {
        ggml_tensor *t = add_random_tensor(to, ctx, rng, name, ne, 0.1f);
        if (spec.weight_scale == 0.0f) {
            return t;
        }
        const bool per_expert = name.find("_exps.") != std::string::npos;
        if (spec.weight_scale_folded) {
            auto *data = static_cast<float *>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                data[i] *= spec.weight_scale;
            }
            return t;
        }
        ggml_tensor *s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, per_expert ? spec.n_expert : 1);
        for (int64_t i = 0; i < ggml_nelements(s); ++i) {
            static_cast<float *>(s->data)[i] = spec.weight_scale;
        }
        ggml_set_name(s, (name.substr(0, name.size() - std::string("weight").size()) + "scale").c_str());
        gguf_add_tensor(to, s);
        return t;
    };
    ggml_tensor *token_embd = add_random_tensor(g, ctx, rng, "token_embd.weight", {spec.n_embd, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {spec.n_embd}, 1.0f);
    // cohere2 has no output layer of its own: the token embedding is the output
    ggml_tensor *output =
        cohere2 ? token_embd : add_random_tensor(g, ctx, rng, "output.weight", {spec.n_embd, n_vocab}, 0.1f);
    if (!sentencepiece) {
        confine_output_to_ascii(output); // token ids are bytes there; every SentencePiece piece is printable
    }
    if (!cohere2 && spec.weight_scale != 0.0f) {
        if (spec.weight_scale_folded) {
            for (int64_t i = 0; i < ggml_nelements(output); ++i) {
                static_cast<float *>(output->data)[i] *= spec.weight_scale;
            }
        } else {
            ggml_tensor *s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            *static_cast<float *>(s->data) = spec.weight_scale;
            ggml_set_name(s, "output.scale");
            gguf_add_tensor(g, s);
        }
    }
    gguf_context *shard = shard_path != nullptr ? gguf_init_empty() : nullptr;
    for (int il = 0; il < n_layer_all; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        gguf_context *to = shard != nullptr && il == spec.n_layer - 1 ? shard : g;
        if (mamba) {
            add_random_tensor(to, ctx, rng, p + "attn_norm.weight", {spec.n_embd}, 1.0f);
            matrix(to, p + "ssm_in.weight", {spec.n_embd, 2 * ssm_inner});
            add_random_tensor(to, ctx, rng, p + "ssm_conv1d.weight", {SSM_CONV, ssm_inner}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "ssm_conv1d.bias", {ssm_inner}, 1.0f);
            add_random_tensor(to, ctx, rng, p + "ssm_x.weight", {ssm_inner, SSM_DT_RANK + 2 * SSM_STATE}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "ssm_dt.weight", {SSM_DT_RANK, ssm_inner}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "ssm_dt.bias", {ssm_inner}, 1.0f);
            // the state decays with exp(dt * A): a positive entry would let it grow with every token
            ggml_tensor *ssm_a = add_random_tensor(to, ctx, rng, p + "ssm_a", {SSM_STATE, ssm_inner}, 0.1f);
            auto *decay = (float *)ssm_a->data;
            for (int64_t i = 0; i < ggml_nelements(ssm_a); ++i) {
                decay[i] = -std::fabs(decay[i]) - 0.1f;
            }
            add_random_tensor(to, ctx, rng, p + "ssm_d", {ssm_inner}, 1.0f);
            matrix(to, p + "ssm_out.weight", {ssm_inner, spec.n_embd});
            continue;
        }
        if (qwen3next) {
            add_random_tensor(to, ctx, rng, p + "attn_norm.weight", {spec.n_embd}, 1.0f);
            add_random_tensor(to, ctx, rng, p + "post_attention_norm.weight", {spec.n_embd}, 1.0f);
            if (il % 2 == 0) { // the gated delta net
                matrix(to, p + "attn_qkv.weight", {spec.n_embd, 3 * GDN_INNER});
                matrix(to, p + "attn_gate.weight", {spec.n_embd, GDN_INNER});
                add_random_tensor(to, ctx, rng, p + "ssm_conv1d.weight", {SSM_CONV, 3 * GDN_INNER}, 0.1f);
                add_random_tensor(to, ctx, rng, p + "ssm_dt.bias", {GDN_HEADS}, 1.0f);
                ggml_tensor *ssm_a = add_random_tensor(to, ctx, rng, p + "ssm_a", {GDN_HEADS}, 1.0f);
                for (int64_t i = 0; i < ggml_nelements(ssm_a); ++i) {
                    static_cast<float *>(ssm_a->data)[i] = -0.5f; // -A_log.exp(): the state decays
                }
                add_random_tensor(to, ctx, rng, p + "ssm_ba.weight", {spec.n_embd, 2 * GDN_HEADS}, 0.1f);
                add_random_tensor(to, ctx, rng, p + "ssm_norm.weight", {GDN_HEAD_DIM}, 1.0f);
                matrix(to, p + "ssm_out.weight", {GDN_INNER, spec.n_embd});
            } else { // attention, the query carrying its output gate
                matrix(to, p + "attn_q.weight", {spec.n_embd, 2 * spec.n_head * spec.head_dim});
                matrix(to, p + "attn_k.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim});
                matrix(to, p + "attn_v.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim});
                matrix(to, p + "attn_output.weight", {spec.n_head * spec.head_dim, spec.n_embd});
                add_random_tensor(to, ctx, rng, p + "attn_q_norm.weight", {spec.head_dim}, 1.0f);
                add_random_tensor(to, ctx, rng, p + "attn_k_norm.weight", {spec.head_dim}, 1.0f);
            }
            add_random_tensor(to, ctx, rng, p + "ffn_gate_inp.weight", {spec.n_embd, spec.n_expert}, 1.0f);
            matrix(to, p + "ffn_gate_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert});
            matrix(to, p + "ffn_down_exps.weight", {spec.n_ff, spec.n_embd, spec.n_expert});
            matrix(to, p + "ffn_up_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert});
            add_random_tensor(to, ctx, rng, p + "ffn_gate_inp_shexp.weight", {spec.n_embd}, 1.0f);
            matrix(to, p + "ffn_gate_shexp.weight", {spec.n_embd, spec.n_ff});
            matrix(to, p + "ffn_up_shexp.weight", {spec.n_embd, spec.n_ff});
            matrix(to, p + "ffn_down_shexp.weight", {spec.n_ff, spec.n_embd});
            continue;
        }
        if (hy_v3) {
            add_random_tensor(to, ctx, rng, p + "attn_q_norm.weight", {spec.head_dim}, 1.0f);
            add_random_tensor(to, ctx, rng, p + "attn_k_norm.weight", {spec.head_dim}, 1.0f);
        }
        if (il >= spec.n_layer) {
            add_random_tensor(to, ctx, rng, p + "nextn.eh_proj.weight", {2 * spec.n_embd, spec.n_embd}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "nextn.enorm.weight", {spec.n_embd}, 1.0f);
            add_random_tensor(to, ctx, rng, p + "nextn.hnorm.weight", {spec.n_embd}, 1.0f);
        }
        add_random_tensor(to, ctx, rng, p + "attn_norm.weight", {spec.n_embd}, 1.0f);
        matrix(to, p + "attn_q.weight", {spec.n_embd, spec.n_head * spec.head_dim});
        matrix(to, p + "attn_k.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim});
        matrix(to, p + "attn_v.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim});
        matrix(to, p + "attn_output.weight", {spec.n_head * spec.head_dim, spec.n_embd});
        if (spec.cache_scales) {
            // constants, so the random weights are those of the same model without scales
            for (const auto &scale : {std::pair<const char *, float>{"attn_k.k_scale", spec.k_cache_scale},
                                      std::pair<const char *, float>{"attn_v.v_scale", spec.v_cache_scale}}) {
                ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                *static_cast<float *>(t->data) = scale.second;
                ggml_set_name(t, (p + scale.first).c_str());
                gguf_add_tensor(to, t);
            }
        }
        if (!cohere2) {
            add_random_tensor(to, ctx, rng, p + "ffn_norm.weight", {spec.n_embd}, 1.0f);
        }
        if (spec.n_expert > 0) {
            // the router needs spread-out logits, or every token takes the same experts
            add_random_tensor(to, ctx, rng, p + "ffn_gate_inp.weight", {spec.n_embd, spec.n_expert}, 1.0f);
            matrix(to, p + "ffn_gate_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert});
            matrix(to, p + "ffn_down_exps.weight", {spec.n_ff, spec.n_embd, spec.n_expert});
            matrix(to, p + "ffn_up_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert});
            continue;
        }
        matrix(to, p + "ffn_gate.weight", {spec.n_embd, spec.n_ff});
        matrix(to, p + "ffn_up.weight", {spec.n_embd, spec.n_ff});
        matrix(to, p + "ffn_down.weight", {spec.n_ff, spec.n_embd});
    }
    bool ok = gguf_write_to_file(g, path.c_str(), false);
    if (shard != nullptr) {
        // another directory than the model's, which is what --shard is for
        const std::filesystem::path dir = std::filesystem::path(path).replace_extension(".shards");
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        *shard_path = (dir / "last-layer.gguf").string();
        ok = ok && !ec && gguf_write_to_file(shard, shard_path->c_str(), false);
        gguf_free(shard);
    }
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

inline std::string write_tiny_llama_gguf(const std::string &prefix = "jllama-c-api", bool fim = false,
                                         uint32_t eos = '>', const std::vector<std::string> &special = {},
                                         std::string *shard_path = nullptr) {
    tiny_model_spec spec;
    spec.prefix = prefix;
    spec.fim = fim;
    spec.eos = eos;
    spec.special = special;
    return write_tiny_llama_gguf(spec, shard_path);
}

// A random BERT: an encoder with learned positions, a WordPiece vocabulary of a few words and,
// with classifier, the pooler and a one-label classification head, which is what makes a model a
// reranker (--reranking, pooling type rank). The second architecture next to the llama decoder: it
// has no causal attention, a [CLS] token for cls pooling and no output layer to generate with. Its
// answers are noise like the decoder's; what a test asserts is that the path runs and the shapes fit.
// pooling_type is written as bert.pooling_type (llama_pooling_type; -1 writes none).
// With shard_path, the tensors of the last layer -- and the classification head, when there is one --
// are written to a second GGUF in a directory of its own, as --shard takes it. The weights are the
// same either way, so the split model answers what the whole one answers.
inline std::string write_tiny_bert_gguf(bool classifier = false, int pooling_type = 2,
                                        const std::string &prefix = "jllama-bert", std::string *shard_path = nullptr) {
    constexpr int B_EMBD = 32, B_HEAD = 4, B_FF = 64, B_LAYER = 2, B_CTX = 64;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(1234);
    gguf_context *g = gguf_init_empty();
    const std::string a = "bert";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".context_length").c_str(), B_CTX);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), B_EMBD);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), B_LAYER);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), B_FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), B_HEAD);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_epsilon").c_str(), 1e-12f);
    gguf_set_val_bool(g, (a + ".attention.causal").c_str(), false);
    if (pooling_type >= 0) {
        gguf_set_val_u32(g, (a + ".pooling_type").c_str(), static_cast<uint32_t>(pooling_type));
    }

    // WordPiece as the converter stores it: a word start carries U+2581, a continuation nothing
    const std::string w = "\xE2\x96\x81";
    std::vector<std::string> tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    for (const char *word : {"hello", "there", "world",   "the", "cat",    "dog",    "sat",     "on", "mat", "is",
                             "what",  "paris", "capital", "of",  "france", "berlin", "germany", ".",  "?",   ","}) {
        tokens.push_back(w + word);
    }
    for (char c = 'a'; c <= 'z'; ++c) {
        tokens.push_back(w + std::string(1, c));
        tokens.push_back(std::string(1, c));
    }
    const int n_vocab = static_cast<int>(tokens.size());
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    std::vector<int32_t> token_types(n_vocab, 1); // LLAMA_TOKEN_TYPE_NORMAL
    for (int special = 0; special < 5; ++special) {
        token_types[special] = 3; // LLAMA_TOKEN_TYPE_CONTROL
    }
    gguf_set_val_str(g, "tokenizer.ggml.model", "bert");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_val_u32(g, "tokenizer.ggml.token_type_count", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.cls_token_id", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 3);
    gguf_set_val_u32(g, "tokenizer.ggml.seperator_token_id", 3);
    gguf_set_val_u32(g, "tokenizer.ggml.mask_token_id", 4);

    ggml_init_params ip = {8u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    gguf_context *shard = shard_path != nullptr ? gguf_init_empty() : nullptr;
    gguf_context *to = g; // where the next tensors go
    const auto bias = [&](const std::string &name, int n) {
        ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (int i = 0; i < n; ++i) {
            ((float *)t->data)[i] = dist(rng);
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(to, t);
    };
    add_random_tensor(g, ctx, rng, "token_embd.weight", {B_EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "token_types.weight", {B_EMBD, 2}, 0.1f);
    add_random_tensor(g, ctx, rng, "position_embd.weight", {B_EMBD, B_CTX}, 0.1f);
    add_random_tensor(g, ctx, rng, "token_embd_norm.weight", {B_EMBD}, 1.0f);
    bias("token_embd_norm.bias", B_EMBD);
    for (int il = 0; il < B_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        if (shard != nullptr && il == B_LAYER - 1) {
            to = shard; // the last layer and everything after it
        }
        for (const char *t : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            add_random_tensor(to, ctx, rng, p + t + ".weight", {B_EMBD, B_EMBD}, 0.1f);
            bias(p + t + ".bias", B_EMBD);
        }
        add_random_tensor(to, ctx, rng, p + "attn_output_norm.weight", {B_EMBD}, 1.0f);
        bias(p + "attn_output_norm.bias", B_EMBD);
        add_random_tensor(to, ctx, rng, p + "ffn_up.weight", {B_EMBD, B_FF}, 0.1f);
        bias(p + "ffn_up.bias", B_FF);
        add_random_tensor(to, ctx, rng, p + "ffn_down.weight", {B_FF, B_EMBD}, 0.1f);
        bias(p + "ffn_down.bias", B_EMBD);
        add_random_tensor(to, ctx, rng, p + "layer_output_norm.weight", {B_EMBD}, 1.0f);
        bias(p + "layer_output_norm.bias", B_EMBD);
    }
    if (classifier) {
        add_random_tensor(to, ctx, rng, "cls.weight", {B_EMBD, B_EMBD}, 0.3f);
        bias("cls.bias", B_EMBD);
        add_random_tensor(to, ctx, rng, "cls.output.weight", {B_EMBD, 1}, 0.5f);
        bias("cls.output.bias", 1);
    }
    bool ok = gguf_write_to_file(g, path.c_str(), false);
    if (shard != nullptr) {
        const std::filesystem::path dir = std::filesystem::path(path).replace_extension(".shards");
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        *shard_path = (dir / "last-layer.gguf").string();
        ok = ok && !ec && gguf_write_to_file(shard, shard_path->c_str(), false);
        gguf_free(shard);
    }
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// An importance matrix for the model of spec, in the GGUF format llama-imatrix writes: for every
// weight matrix a tensor <name>.in_sum2 with one value per column and <name>.counts with the number
// of tokens they were summed over, plus the imatrix.* metadata. The values are random and positive;
// what matters to a test is that the quantizer finds an entry of the right size for every tensor.
inline std::string write_tiny_imatrix_gguf(const tiny_model_spec &spec = tiny_model_spec(),
                                           const std::string &prefix = "jllama-imatrix") {
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.5f, 2.0f);
    gguf_context *g = gguf_init_empty();
    const char *datasets[] = {"random.txt"};
    gguf_set_val_str(g, "general.type", "imatrix");
    gguf_set_arr_str(g, "imatrix.datasets", datasets, 1);
    gguf_set_val_u32(g, "imatrix.chunk_count", 3);
    gguf_set_val_u32(g, "imatrix.chunk_size", 16);
    ggml_init_params ip = {8u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    const auto add = [&](const std::string &name, int n_columns) {
        ggml_tensor *sums = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_columns);
        for (int i = 0; i < n_columns; ++i) {
            ((float *)sums->data)[i] = 48.0f * dist(rng); // the sum over 48 tokens
        }
        ggml_set_name(sums, (name + ".in_sum2").c_str());
        gguf_add_tensor(g, sums);
        ggml_tensor *counts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ((float *)counts->data)[0] = 48.0f;
        ggml_set_name(counts, (name + ".counts").c_str());
        gguf_add_tensor(g, counts);
    };
    for (int il = 0; il < spec.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char *t : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate.weight", "ffn_up.weight"}) {
            add(p + t, spec.n_embd);
        }
        add(p + "attn_output.weight", spec.n_head * spec.head_dim);
        add(p + "ffn_down.weight", spec.n_ff);
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

// A random ultravox-style audio projector for the model above: a Whisper encoder (two
// convolutions over a 16-bin mel spectrogram, one transformer layer of width 16 over the 1500
// positions of a 30 s chunk), four frames stacked into one, and the projection to the model's
// N_EMBD. Loaded with --mmproj it lets the server hear: a clip of up to 30 s is one chunk and 375
// embeddings, so the model needs a context that holds them. llama.cpp computes the mel filter
// bank itself; like the model, what the encoder hears is noise.
//
// a_embd and stack are the two shapes that decide how llama.cpp stacks the 1500 encoder frames of a
// chunk (measured at b12493, see CApiAudio): with a stack factor that divides 1500 every shape
// agrees. With one that does not, llama.cpp counts align(3000, stack) / stack / 2 embeddings before
// it builds the graph, and the graph pads with GGML_PAD, which rounds by bit mask and is right only
// for a power of two: a stride a_embd * stack that is none (Ultravox v0.5: 1280 * 8) pads less than
// a row and keeps the count, one that is (16 * 8) pads a whole row more and aborts the process.
inline std::string write_tiny_audio_mmproj_gguf(const std::string &prefix = "jllama-mmproj-audio", int a_embd = 16,
                                                int stack = 4) {
    const int A_EMBD = a_embd, STACK = stack;
    constexpr int A_HEAD = 2, A_FF = 32, A_LAYER = 1, A_MEL = 16, A_POS = 1500, HIDDEN = 32;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(11);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_str(g, "clip.projector_type", "ultravox");
    gguf_set_val_bool(g, "clip.has_audio_encoder", true);
    gguf_set_val_u32(g, "clip.audio.num_mel_bins", A_MEL);
    gguf_set_val_u32(g, "clip.audio.embedding_length", A_EMBD);
    gguf_set_val_u32(g, "clip.audio.feed_forward_length", A_FF);
    gguf_set_val_u32(g, "clip.audio.block_count", A_LAYER);
    gguf_set_val_u32(g, "clip.audio.projection_dim", N_EMBD);
    gguf_set_val_u32(g, "clip.audio.attention.head_count", A_HEAD);
    gguf_set_val_f32(g, "clip.audio.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_u32(g, "clip.audio.projector.stack_factor", STACK);

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
    add("a.conv1d.1.weight", {3, A_MEL, A_EMBD}, -1);
    add("a.conv1d.1.bias", {1, A_EMBD}, -1);
    add("a.conv1d.2.weight", {3, A_EMBD, A_EMBD}, -1);
    add("a.conv1d.2.bias", {1, A_EMBD}, -1);
    add("a.position_embd.weight", {A_EMBD, A_POS}, -1);
    for (int il = 0; il < A_LAYER; ++il) {
        const std::string p = "a.blk." + std::to_string(il) + ".";
        for (const char *n : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            add(p + n + ".weight", {A_EMBD, A_EMBD}, -1);
            if (std::string(n) != "attn_k") { // Whisper's key projection has no bias
                add(p + n + ".bias", {A_EMBD}, -1);
            }
        }
        for (const char *n : {"ln1", "ln2"}) {
            add(p + n + ".weight", {A_EMBD}, 1.0f);
            add(p + n + ".bias", {A_EMBD}, 0.0f);
        }
        add(p + "ffn_up.weight", {A_EMBD, A_FF}, -1);
        add(p + "ffn_up.bias", {A_FF}, -1);
        add(p + "ffn_down.weight", {A_FF, A_EMBD}, -1);
        add(p + "ffn_down.bias", {A_EMBD}, -1);
    }
    add("a.post_ln.weight", {A_EMBD}, 1.0f);
    add("a.post_ln.bias", {A_EMBD}, 0.0f);
    add("mm.a.norm_pre.weight", {A_EMBD * STACK}, 1.0f);
    add("mm.a.mlp.1.weight", {A_EMBD * STACK, HIDDEN}, -1);
    add("mm.a.norm_mid.weight", {HIDDEN / 2}, 1.0f); // the gated activation halves the width
    add("mm.a.mlp.2.weight", {HIDDEN / 2, N_EMBD}, -1);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The pair of a random ZONOS3 (llama.cpp b12431, #32020): the one speech pipeline with delivery
// control, whose prompt carries a loudness and a speaking-rate condition token picked by the
// request's loudness and speed. Small enough to synthesize in a unit test, written in the layout
// upstream's converter writes (conversion/zonos3.py).
//
// The backbone: two layers, a gated delta net (GDN) and an attention layer (full_attention_interval
// 2), both dense (leading_dense_block_count = block_count, so no expert tensors), with the routed
// residuals of the architecture (attn_res_score, ffn_res_score, output_res_score). The vocabulary is
// the GPT-2 byte tokens behind <|unused_0|> <|unused_1|> <|bos|> <|eos|> (ids 0-3, BOS 2, EOS 3, both
// added to every tokenization), and the condition tokens the pipeline looks up by name.
// A dictionary of a speech model for the tts template's tts_dict(name, key) (llama.cpp b12480, #32397):
// written as tokenizer.tts_dict.<name>.keys / .values, both string arrays, the keys sorted by their
// bytes and each value a JSON document. More values than keys or fewer make the dictionary malformed,
// which llama.cpp ignores (the template then names an unknown dictionary).
struct tiny_tts_dict {
    std::string name;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

// tts_template: a "tts" chat template (tokenizer.chat_template.tts), empty for none; dicts: the
// dictionaries it may look up, listed by name in tokenizer.tts_dicts.
inline std::string write_tiny_zonos3_gguf(const std::string &prefix = "jllama-zonos3",
                                          const std::string &tts_template = "",
                                          const std::vector<tiny_tts_dict> &dicts = {}) {
    constexpr int EMBD = 32, N_HEAD_A = 2, HEAD = 16, FF = 64, GDN_HEADS = 2, GDN_HEAD_DIM = 8, GDN_KERNEL = 4;
    constexpr int D_INNER = GDN_HEADS * GDN_HEAD_DIM;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(23);
    gguf_context *g = gguf_init_empty();
    const std::string a = "zonos3";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 8192);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), 2);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD_A);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD_A);
    gguf_set_val_u32(g, (a + ".attention.key_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".attention.value_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".rope.dimension_count").c_str(), HEAD);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(g, (a + ".full_attention_interval").c_str(), 2);
    gguf_set_val_u32(g, (a + ".ssm.conv_kernel").c_str(), GDN_KERNEL);
    gguf_set_val_u32(g, (a + ".ssm.inner_size").c_str(), D_INNER);
    gguf_set_val_u32(g, (a + ".ssm.state_size").c_str(), GDN_HEAD_DIM);
    gguf_set_val_u32(g, (a + ".ssm.time_step_rank").c_str(), GDN_HEADS);
    gguf_set_val_u32(g, (a + ".ssm.group_count").c_str(), GDN_HEADS);
    gguf_set_val_u32(g, (a + ".leading_dense_block_count").c_str(), 2);
    gguf_set_val_u32(g, (a + ".expert_feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".moe_latent_size").c_str(), 16);

    std::vector<std::string> tokens = {"<|unused_0|>", "<|unused_1|>", "<|bos|>", "<|eos|>"};
    std::vector<int32_t> token_types(tokens.size(), 3);
    for (const std::string &t : byte_tokens()) {
        tokens.push_back(t);
        token_types.push_back(1);
    }
    std::vector<std::string> conditions = {"<|text_pad|>"};
    for (int i = 0; i < 10; ++i) { // nine loudness edges, ten buckets
        conditions.push_back("<|loudness_" + std::to_string(i) + "|>");
    }
    conditions.push_back("<|loudness_unk|>");
    conditions.push_back("<|fmax_unk|>");
    for (int i = 0; i < 30; ++i) { // 29 speaking-rate edges
        conditions.push_back("<|speaking_rate_" + std::to_string(i) + "|>");
    }
    conditions.push_back("<|speaking_rate_unk|>");
    tokens.insert(tokens.end(), conditions.begin(), conditions.end());
    token_types.resize(tokens.size(), 3);
    const int n_vocab = static_cast<int>(tokens.size());
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    const char *merges[] = {"Ġ Ġ"}; // never applies
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "zonos3");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 3);
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", true);
    if (!tts_template.empty()) {
        gguf_set_val_str(g, "tokenizer.chat_template.tts", tts_template.c_str());
    }
    if (!dicts.empty()) {
        std::vector<const char *> names;
        for (const tiny_tts_dict &d : dicts) {
            names.push_back(d.name.c_str());
            std::vector<const char *> keys;
            std::vector<const char *> values;
            for (const std::string &k : d.keys) {
                keys.push_back(k.c_str());
            }
            for (const std::string &v : d.values) {
                values.push_back(v.c_str());
            }
            gguf_set_arr_str(g, ("tokenizer.tts_dict." + d.name + ".keys").c_str(), keys.data(), keys.size());
            gguf_set_arr_str(g, ("tokenizer.tts_dict." + d.name + ".values").c_str(), values.data(), values.size());
        }
        gguf_set_arr_str(g, "tokenizer.tts_dicts", names.data(), names.size());
    }

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    auto ones = [&](const std::string &name, std::initializer_list<int64_t> ne) {
        const std::vector<int64_t> d(ne);
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(d.size()), d.data());
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = 1.0f;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
        return t;
    };
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {EMBD}, 1.0f);
    add_random_tensor(g, ctx, rng, "output_res_score.weight", {EMBD}, 1.0f);
    for (int il = 0; il < 2; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add_random_tensor(g, ctx, rng, p + "attn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "attn_res_score.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_res_score.weight", {EMBD}, 1.0f);
        if (il == 0) { // the gated delta net
            add_random_tensor(g, ctx, rng, p + "attn_qkv.weight", {EMBD, 3 * D_INNER}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_gate.weight", {EMBD, D_INNER}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "ssm_conv1d.weight", {GDN_KERNEL, 3 * D_INNER}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "ssm_dt.bias", {GDN_HEADS}, 1.0f);
            // -A_log.exp(): negative, so the state decays
            ggml_tensor *ssm_a = add_random_tensor(g, ctx, rng, p + "ssm_a", {GDN_HEADS}, 1.0f);
            for (int64_t i = 0; i < ggml_nelements(ssm_a); ++i) {
                ((float *)ssm_a->data)[i] = -0.5f;
            }
            add_random_tensor(g, ctx, rng, p + "ssm_alpha.weight", {EMBD, GDN_HEADS}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "ssm_beta.weight", {EMBD, GDN_HEADS}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "ssm_norm.weight", {GDN_HEAD_DIM}, 1.0f);
            add_random_tensor(g, ctx, rng, p + "ssm_out.weight", {D_INNER, EMBD}, 0.1f);
        } else { // attention with a per-head output gate and a weighted QK norm on Q
            add_random_tensor(g, ctx, rng, p + "attn_q.weight", {EMBD, N_HEAD_A * HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_k.weight", {EMBD, N_HEAD_A * HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_v.weight", {EMBD, N_HEAD_A * HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_output.weight", {N_HEAD_A * HEAD, EMBD}, 0.1f);
            ones(p + "attn_q_norm.weight", {HEAD, N_HEAD_A});
            add_random_tensor(g, ctx, rng, p + "attn_gate.weight", {EMBD, N_HEAD_A}, 0.1f);
        }
        add_random_tensor(g, ctx, rng, p + "ffn_gate.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {FF, EMBD}, 0.1f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above: the ten codebook heads and the summed code embedding
// (a.gen.code.*, 4098 rows per codebook: the codes, the end of audio and the pad), one predefined
// voice ("test", three rows) and the ZAC codec decoder (a.gen.wav.*) at the shapes llama.cpp fixes --
// five stages, the patch-up between them, the 2948 rows of the two iSTFT branches, the 736 x 735
// resampler -- with every free width made small and no transformer layer in a stage (the loader
// takes a stage without one). No speaker encoder, so the pair clones no voice. The weights are
// small, so the log magnitudes the iSTFT exponentiates stay near zero.
inline std::string write_tiny_zonos3_mmproj_gguf(const std::string &prefix = "jllama-zonos3-mmproj") {
    constexpr int EMBD = 32, N_CB = 10, N_CB_ROWS = 4098, N_CODES = 4096, SEM = 8, LATENT = 8, STAGE = 8, K = 3,
                  PW = 16;
    constexpr int SPEC_ROWS = 736 + 2 + 2208 + 2;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(29);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "zonos3_gen");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", 0);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 0);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", 0);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);

    ggml_init_params ip = {64u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude (weights); otherwise every element is value (norms, biases)
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    };
    add("a.gen.code.head.weight", {EMBD, N_CB * N_CB_ROWS}, -0.1f);
    add("a.gen.code.out_embd.weight", {EMBD, N_CB * N_CB_ROWS}, -0.05f);
    add("voice.test", {EMBD, 3}, -0.1f);

    const std::string w = "a.gen.wav";
    auto convnext = [&](const std::string &q, int c) {
        add(q + ".dw.weight", {K, c}, -0.1f);
        add(q + ".dw.bias", {c}, 0.0f);
        add(q + ".norm.weight", {c}, 1.0f);
        add(q + ".norm.bias", {c}, 0.0f);
        add(q + ".pw1.weight", {c, PW}, -0.1f);
        add(q + ".pw1.bias", {PW}, 0.0f);
        add(q + ".pw2.weight", {PW, c}, -0.1f);
        add(q + ".pw2.bias", {c}, 0.0f);
    };
    add(w + ".sem.table", {SEM, N_CODES}, -0.1f);
    add(w + ".sem.bias", {SEM}, 0.0f);
    convnext(w + ".sem.0", SEM);
    add(w + ".sem.out.weight", {SEM, LATENT}, -0.1f);
    add(w + ".sem.out.bias", {LATENT}, 0.0f);
    add(w + ".res.table", {LATENT, (N_CB - 1) * N_CODES}, -0.1f);
    add(w + ".res.bias", {LATENT, N_CB - 1}, 0.0f);
    // each stage projects to STAGE channels; between two stages the patch-up halves them
    int channels = LATENT;
    for (int s = 0; s < 5; ++s) {
        add(w + ".stage." + std::to_string(s) + ".in.weight", {channels, STAGE}, -0.1f);
        channels = s < 4 ? STAGE / 2 : STAGE;
    }
    convnext(w + ".head.0", STAGE);
    add(w + ".head.norm.weight", {STAGE}, 1.0f);
    add(w + ".head.norm.bias", {STAGE}, 0.0f);
    add(w + ".head.out.weight", {STAGE, SPEC_ROWS}, -0.01f);
    add(w + ".head.out.bias", {SPEC_ROWS}, 0.0f);
    add(w + ".mixer.weight", {3, 2}, -0.1f);
    add(w + ".resample.weight", {736, 735}, -0.01f);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The pair of a random MiniMax Music 3 (llama.cpp b12480, #32435): songs from a caption and lyrics,
// stereo at 44.1 kHz. In the layout upstream's converter writes (conversion/minimax_music3.py), with
// every free width made small; what llama.cpp fixes stays (the token ids, 128 latent channels in two
// halves of 64, a vocoder stride of 512, DiT heads of 64).
//
// The backbone is a qwen3 with classifier-free guidance (the uncond prompt runs as a second sequence,
// so common_init doubles the sequences), tied output, and the checkpoint's vocabulary length: the
// pipeline reads the logits of <|audio_end|> (151670) and of the 16384 semantic codes from 151675 on,
// and looks for <|im_start|> (151644), <|audio_cfg|> (151654) and <|audio_start|> (151669) by id. The
// GPT-2 byte tokens are ids 0-255, every other id a placeholder. Its "tts" template renders
// <|im_start|> text <|audio_start|>, the frame of upstream's template without the caption and lyrics
// cleanup, which needs a "[Tags]\n...\n[Lyrics]\n..." text.
inline std::string write_tiny_music3_gguf(const std::string &prefix = "jllama-music3") {
    constexpr int EMBD = 32, N_HEAD = 2, HEAD = 16, FF = 64, N_VOCAB = 151675 + 16384;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(31);
    gguf_context *g = gguf_init_empty();
    const std::string a = "qwen3";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 8192);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".block_count").c_str(), 1);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.key_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".attention.value_length").c_str(), HEAD);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    gguf_set_val_bool(g, (a + ".classifier_free_guidance").c_str(), true);

    std::vector<std::string> tokens = byte_tokens();
    std::vector<int32_t> token_types(tokens.size(), 1);
    tokens.reserve(N_VOCAB);
    for (int id = static_cast<int>(tokens.size()); id < N_VOCAB; ++id) {
        std::string name;
        int32_t type = 3;
        if (id == 151644) {
            name = "<|im_start|>";
        } else if (id == 151654) {
            name = "<|audio_cfg|>";
        } else if (id == 151669) {
            name = "<|audio_start|>";
        } else if (id == 151670) {
            name = "<|audio_end|>";
        } else if (id >= 151675) {
            name = "<|audio_code_" + std::to_string(id - 151675) + "|>";
        } else {
            name = "<|unused_" + std::to_string(id) + "|>";
            type = 1;
        }
        tokens.push_back(name);
        token_types.push_back(type);
    }
    std::vector<const char *> token_ptrs;
    token_ptrs.reserve(tokens.size());
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    const char *merges[] = {"Ġ Ġ"}; // never applies
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen2");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 151670);
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_str(g, "tokenizer.chat_template.tts",
                     "{{- '<|im_start|>' ~ messages[0]['content'] ~ '<|audio_start|>' -}}");

    ggml_init_params ip = {32u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, N_VOCAB}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {EMBD}, 1.0f);
    const std::string p = "blk.0.";
    add_random_tensor(g, ctx, rng, p + "attn_norm.weight", {EMBD}, 1.0f);
    add_random_tensor(g, ctx, rng, p + "attn_q.weight", {EMBD, N_HEAD * HEAD}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "attn_k.weight", {EMBD, N_HEAD * HEAD}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "attn_v.weight", {EMBD, N_HEAD * HEAD}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "attn_output.weight", {N_HEAD * HEAD, EMBD}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "attn_q_norm.weight", {HEAD}, 1.0f);
    add_random_tensor(g, ctx, rng, p + "attn_k_norm.weight", {HEAD}, 1.0f);
    add_random_tensor(g, ctx, rng, p + "ffn_norm.weight", {EMBD}, 1.0f);
    add_random_tensor(g, ctx, rng, p + "ffn_gate.weight", {EMBD, FF}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {EMBD, FF}, 0.1f);
    add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {FF, EMBD}, 0.1f);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above: the RVQ depth decoder (a.gen.code.*: one layer over the backbone
// hidden and a row per residual code, two residual codebooks of 16 codes, the embeddings of the 16384
// semantic codes), the condition encoder (a.gen.cond.*: the mix of the depth decoder's three hidden
// states and a k = 3 conv), the flow-matching DiT (a.gen.flow.*: one layer of one 64-wide head) and the
// DAC vocoder (a.gen.vae.*: a k = 1 projection from the 64 latent channels of one stereo side, three
// blocks of stride 8, four channels). The vocoder ends in tanh, so the samples stay in [-1, 1].
inline std::string write_tiny_music3_mmproj_gguf(const std::string &prefix = "jllama-music3-mmproj") {
    constexpr int EMBD = 32, N_HEAD = 2, FF = 64, N_CB = 2, N_CODES = 16, N_SEMANTIC = 16384;
    constexpr int N_COND = 8, N_LATENT = 128, N_IN = 2 * N_LATENT + N_COND, DIT = 64, DIT_FF = 64, FREQ = 8, TIME = 16,
                  VAE = 4;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(37);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "minimax_music3_gen");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", FF);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 1);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", N_HEAD);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-6f);

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude (weights); otherwise every element is value (norms, biases)
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    };
    const std::string c = "a.gen.code";
    add(c + ".blk.0.ln1.weight", {EMBD}, 1.0f);
    add(c + ".blk.0.attn_q.weight", {EMBD, EMBD}, -0.1f);
    add(c + ".blk.0.attn_k.weight", {EMBD, EMBD}, -0.1f);
    add(c + ".blk.0.attn_v.weight", {EMBD, EMBD}, -0.1f);
    add(c + ".blk.0.attn_out.weight", {EMBD, EMBD}, -0.1f);
    add(c + ".blk.0.ln2.weight", {EMBD}, 1.0f);
    add(c + ".blk.0.ffn_gate.weight", {EMBD, FF}, -0.1f);
    add(c + ".blk.0.ffn_up.weight", {EMBD, FF}, -0.1f);
    add(c + ".blk.0.ffn_down.weight", {FF, EMBD}, -0.1f);
    add(c + ".position_embd.weight", {EMBD, N_CB + 1}, -0.1f);
    add(c + ".post_ln.weight", {EMBD}, 1.0f);
    add(c + ".proj_in.weight", {EMBD, EMBD}, -0.1f);
    add(c + ".embd.weight", {EMBD, N_CODES, N_CB}, -0.1f);
    add(c + ".head.weight", {EMBD, N_CODES, N_CB}, -0.1f);
    add(c + ".out_embd.weight", {EMBD, N_SEMANTIC}, -0.1f);

    add("a.gen.cond.layer_w", {N_CB + 1}, 1.0f / (N_CB + 1));
    add("a.gen.cond.proj.weight", {3, EMBD, N_COND}, -0.1f);
    add("a.gen.cond.proj.bias", {N_COND}, 0.0f);

    const std::string f = "a.gen.flow";
    add(f + ".time_freq", {FREQ}, -1.0f);
    add(f + ".time_in.weight", {2 * FREQ, TIME}, -0.1f);
    add(f + ".time_in.bias", {TIME}, 0.0f);
    add(f + ".time_out.weight", {TIME, DIT}, -0.1f);
    add(f + ".time_out.bias", {DIT}, 0.0f);
    add(f + ".pre.weight", {N_IN, N_IN}, -0.01f);
    add(f + ".proj_in.weight", {N_IN, DIT}, -0.05f);
    add(f + ".proj_out.weight", {DIT, N_LATENT}, -0.05f);
    add(f + ".post.weight", {N_LATENT, N_LATENT}, -0.01f);
    add(f + ".blk.0.ln1.weight", {DIT}, 1.0f);
    add(f + ".blk.0.ln1.bias", {DIT}, 0.0f);
    add(f + ".blk.0.attn_q.weight", {DIT, DIT}, -0.1f);
    add(f + ".blk.0.attn_k.weight", {DIT, DIT}, -0.1f);
    add(f + ".blk.0.attn_v.weight", {DIT, DIT}, -0.1f);
    add(f + ".blk.0.attn_out.weight", {DIT, DIT}, -0.1f);
    add(f + ".blk.0.ln2.weight", {DIT}, 1.0f);
    add(f + ".blk.0.ln2.bias", {DIT}, 0.0f);
    add(f + ".blk.0.ffn_up.weight", {DIT, DIT_FF}, -0.1f);
    add(f + ".blk.0.ffn_up.bias", {DIT_FF}, 0.0f);
    add(f + ".blk.0.ffn_gate.weight", {DIT, DIT_FF}, -0.1f);
    add(f + ".blk.0.ffn_gate.bias", {DIT_FF}, 0.0f);
    add(f + ".blk.0.ffn_down.weight", {DIT_FF, DIT}, -0.1f);
    add(f + ".blk.0.ffn_down.bias", {DIT}, 0.0f);

    const std::string v = "a.gen.vae";
    const auto snake = [&](const std::string &q) {
        add(q + ".alpha", {VAE}, 1.0f);
        add(q + ".beta", {VAE}, 0.5f);
    };
    add(v + ".in_proj.weight", {1, N_LATENT / 2, VAE}, -0.1f);
    add(v + ".in_proj.bias", {VAE}, 0.0f);
    add(v + ".conv_in.weight", {3, VAE, VAE}, -0.3f);
    add(v + ".conv_in.bias", {VAE}, 0.0f);
    for (int b = 0; b < 3; ++b) { // strides 8 * 8 * 8 = 512 samples per latent frame
        const std::string q = v + ".blk." + std::to_string(b);
        snake(q + ".snake");
        add(q + ".up.weight", {16, VAE, VAE}, -0.3f);
        add(q + ".up.bias", {VAE}, 0.0f);
        for (int r = 0; r < 3; ++r) {
            const std::string rq = q + ".res." + std::to_string(r);
            snake(rq + ".snake1");
            add(rq + ".conv1.weight", {7, VAE, VAE}, -0.1f);
            add(rq + ".conv1.bias", {VAE}, 0.0f);
            snake(rq + ".snake2");
            add(rq + ".conv2.weight", {1, VAE, VAE}, -0.1f);
            add(rq + ".conv2.bias", {VAE}, 0.0f);
        }
    }
    snake(v + ".post_snake");
    add(v + ".conv_out.weight", {7, VAE, 1}, -0.3f);
    add(v + ".conv_out.bias", {1}, 0.0f);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The pair of a random Pocket TTS (kyutai, llama.cpp's pockettts / pockettts_gen), in the layout of
// CI's real english_2026-04 pair, read from its GGUFs: the backbone is a pre-norm transformer with
// LayerNorm and a GELU feed-forward, no output head (llama.cpp reuses the token embedding), and the
// T5 Unigram vocabulary of the checkpoint's shape -- <unk>, <s>, </s>, <pad>, 256 byte pieces, the
// pieces, and the three control tokens the pipeline looks up by text (<|pad|>, <|bos_before_voice|>,
// <|audio_bos|>), no BOS or EOS added. The pipeline feeds no token back: the flow net in the mmproj
// turns each hidden state into a latent, the Mimi decoder the latents into 24 kHz audio.
inline std::string write_tiny_pockettts_gguf(const std::string &prefix = "jllama-pockettts") {
    constexpr int EMBD = 32, N_HEAD = 2, FF = 64, N_LAYER = 2;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(41);
    gguf_context *g = gguf_init_empty();
    const std::string a = "pockettts";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".block_count").c_str(), N_LAYER);
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 4096);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_epsilon").c_str(), 1e-5f);

    std::vector<std::string> tokens;
    std::vector<int32_t> types; // 1 normal, 2 unknown, 3 control, 6 byte
    std::vector<float> scores;
    const auto piece = [&](const std::string &text, float score, int32_t type) {
        tokens.push_back(text);
        scores.push_back(score);
        types.push_back(type);
    };
    piece("<unk>", 0.0f, 2);
    piece("<s>", 0.0f, 3);
    piece("</s>", 0.0f, 3);
    piece("<pad>", 0.0f, 3);
    for (int byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", byte);
        piece(name, 0.0f, 6);
    }
    const std::string sp = "\xE2\x96\x81"; // U+2581
    piece(sp, -2.0f, 1);
    for (char c = '!'; c <= '~'; ++c) {
        piece(std::string(1, c), -10.0f, 1);
    }
    for (const std::string word : {"hello", "world"}) {
        for (size_t n = 1; n <= word.size(); ++n) {
            piece(sp + word.substr(0, n), -1.0f + 0.1f * static_cast<float>(n), 1);
        }
    }
    piece("<|pad|>", 0.0f, 3);
    piece("<|bos_before_voice|>", 0.0f, 3);
    piece("<|audio_bos|>", 0.0f, 3);
    const int n_vocab = static_cast<int>(tokens.size());
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    gguf_set_val_str(g, "tokenizer.ggml.model", "t5");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_bool(g, "tokenizer.ggml.add_space_prefix", true);
    gguf_set_val_bool(g, "tokenizer.ggml.remove_extra_whitespaces", false);
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", false);

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {EMBD}, 1.0f);
    add_random_tensor(g, ctx, rng, "output_norm.bias", {EMBD}, 0.0f);
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add_random_tensor(g, ctx, rng, p + "attn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "attn_norm.bias", {EMBD}, 0.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_norm.bias", {EMBD}, 0.0f);
        for (const char *m : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            add_random_tensor(g, ctx, rng, p + m + ".weight", {EMBD, EMBD}, 0.1f);
        }
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {FF, EMBD}, 0.1f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above (pockettts_gen), every free width made small, what llama.cpp
// fixes kept: the Mimi decoder's three SEANet stages of strides 6, 5 and 4 (kernels 12, 10, 8), the
// depthwise upsampling by 16 (kernel 32), 24 kHz. The flow net has two AdaLN blocks and the two time
// embeddings, the end-of-speech head a bias the caller picks: llama.cpp ends a chunk once its score
// passes -4 (and a few frames after), else at the budget of the chunk's text, so a large bias ends
// every chunk at once and a very negative one never. voices: predefined voices, one row of backbone
// width per position (the real pack has 25, 73 to 154 rows each); none for a model without them.
inline std::string write_tiny_pockettts_mmproj_gguf(const std::string &prefix = "jllama-pockettts-mmproj",
                                                    float eos_bias = -100.0f,
                                                    const std::vector<std::string> &voices = {"alba", "marius"}) {
    constexpr int EMBD = 32, LAT = 8, FC = 16, TF = 8, WE = 16, WFF = 32, C0 = 16;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(43);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "pockettts_gen");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", WE);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", WFF);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 1);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", 2);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_str(g, "clip.gen.audio.model_variant", "english_2026-04");

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude (weights); otherwise every element is value (norms, biases)
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    };
    int rows = 3;
    for (const std::string &v : voices) {
        add("voice." + v, {EMBD, rows++}, -0.1f);
    }
    const std::string f = "a.gen.flow";
    add("a.gen.emb_mean", {LAT}, 0.0f);
    add("a.gen.emb_std", {LAT}, 1.0f);
    add(f + ".cond_embd.weight", {EMBD, FC}, -0.1f);
    add(f + ".cond_embd.bias", {FC}, 0.0f);
    add(f + ".final.ada.weight", {FC, 2 * FC}, -0.1f);
    add(f + ".final.ada.bias", {2 * FC}, 0.0f);
    add(f + ".final.proj.weight", {FC, LAT}, -0.1f);
    add(f + ".final.proj.bias", {LAT}, 0.0f);
    add(f + ".input_proj.weight", {LAT, FC}, -0.1f);
    add(f + ".input_proj.bias", {FC}, 0.0f);
    for (int b = 0; b < 2; ++b) {
        const std::string q = f + ".blk." + std::to_string(b);
        add(q + ".ada.weight", {FC, 3 * FC}, -0.1f);
        add(q + ".ada.bias", {3 * FC}, 0.0f);
        add(q + ".norm.weight", {FC}, 1.0f);
        add(q + ".norm.bias", {FC}, 0.0f);
        add(q + ".up.weight", {FC, FC}, -0.1f);
        add(q + ".up.bias", {FC}, 0.0f);
        add(q + ".down.weight", {FC, FC}, -0.1f);
        add(q + ".down.bias", {FC}, 0.0f);
    }
    for (int i = 0; i < 2; ++i) {
        const std::string q = f + ".time." + std::to_string(i);
        add(q + ".freqs", {TF}, 0.5f);
        add(q + ".up.weight", {2 * TF, FC}, -0.1f);
        add(q + ".up.bias", {FC}, 0.0f);
        add(q + ".down.weight", {FC, FC}, -0.1f);
        add(q + ".down.bias", {FC}, 0.0f);
        add(q + ".norm", {FC}, 1.0f);
    }
    add("a.gen.input_linear.weight", {LAT, EMBD}, -0.1f);
    add("a.gen.out_eos.weight", {EMBD, 1}, -0.01f);
    add("a.gen.out_eos.bias", {1}, eos_bias);

    const std::string w = "a.gen.wav";
    add(w + ".quant_out.weight", {LAT, WE}, -0.1f);
    add(w + ".upsample.weight", {32, 1, WE}, -0.1f);
    add(w + ".seanet.conv_in.weight", {7, WE, C0}, -0.1f);
    add(w + ".seanet.conv_in.bias", {C0}, 0.0f);
    const int ratios[] = {6, 5, 4};
    int c = C0;
    for (int i = 0; i < 3; ++i) {
        const std::string q = w + ".seanet.blk." + std::to_string(i);
        add(q + ".scale_conv.weight", {2 * ratios[i], c / 2, c}, -0.1f);
        add(q + ".scale_conv.bias", {c / 2}, 0.0f);
        c /= 2;
        add(q + ".res_conv1.weight", {3, c, c / 2}, -0.1f);
        add(q + ".res_conv1.bias", {c / 2}, 0.0f);
        add(q + ".res_conv2.weight", {1, c / 2, c}, -0.1f);
        add(q + ".res_conv2.bias", {c}, 0.0f);
    }
    add(w + ".seanet.conv_out.weight", {3, c, 1}, -0.1f);
    add(w + ".seanet.conv_out.bias", {1}, 0.0f);
    const std::string t = w + ".tfm.blk.0";
    add(t + ".ls1.weight", {WE}, 0.1f);
    add(t + ".ls2.weight", {WE}, 0.1f);
    add(t + ".ln1.weight", {WE}, 1.0f);
    add(t + ".ln1.bias", {WE}, 0.0f);
    add(t + ".ln2.weight", {WE}, 1.0f);
    add(t + ".ln2.bias", {WE}, 0.0f);
    add(t + ".ffn_up.weight", {WE, WFF}, -0.1f);
    add(t + ".ffn_down.weight", {WFF, WE}, -0.1f);
    for (const char *m : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
        add(t + "." + m + ".weight", {WE, WE}, -0.1f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The pair of a random Qwen3-TTS (llama.cpp's qwen3tts / qwen3tts_gen), in the layout of CI's real
// 12Hz-1.7B-Base pair, read from its GGUFs. The backbone is a qwen3vl (q/k norm, interleaved M-RoPE)
// whose output head covers only the codec range at the end of the vocabulary: llama.cpp pads the
// logits of every id before it with -inf. The vocabulary keeps the checkpoint's order -- the GPT-2
// bytes with a merge chain for "assistant" (the pipeline slices "<|im_start|>assistant\n" as three
// tokens), the chat and <tts_*> control tokens, then CODEC codec tokens whose first CODEBOOK are the
// codes of the first codebook and whose rest holds what the pipeline looks up by text (languages,
// pad, bos, eos, think). suppress_tokens names the codec ids above the codebook but the end of
// speech, as the real file does, so the backbone samples a code or ends.
inline constexpr int TINY_QWEN3TTS_CODEC = 32;
inline constexpr int TINY_QWEN3TTS_CODEBOOK = 16;
inline std::string write_tiny_qwen3tts_gguf(const std::string &prefix = "jllama-qwen3tts") {
    constexpr int EMBD = 32, N_HEAD = 2, N_HEAD_KV = 1, HEAD = 16, FF = 64, N_LAYER = 2;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(47);
    gguf_context *g = gguf_init_empty();
    const std::string a = "qwen3tts";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".block_count").c_str(), N_LAYER);
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 4096);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD_KV);
    gguf_set_val_u32(g, (a + ".attention.key_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".attention.value_length").c_str(), HEAD);
    const int32_t sections[4] = {4, 2, 2, 0}; // half the head, as the real 24/20/20 of 128
    gguf_set_arr_data(g, (a + ".rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections, 4);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 1000000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

    std::vector<std::string> tokens = byte_tokens();
    std::vector<int32_t> types(tokens.size(), 1);
    std::vector<std::string> merges;
    const std::string word = "assistant";
    for (size_t n = 2; n <= word.size(); ++n) {
        merges.push_back(word.substr(0, n - 1) + " " + word.substr(n - 1, 1));
        tokens.push_back(word.substr(0, n));
        types.push_back(1);
    }
    for (const char *t : {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tts_pad>", "<tts_text_bos>",
                          "<tts_text_eod>", "<tts_text_bos_single>"}) {
        tokens.push_back(t);
        types.push_back(3);
    }
    const int codec_0 = static_cast<int>(tokens.size());
    const std::vector<std::string> codec_specials = {
        "<|codec_language_english|>", "<|codec_language_german|>", "<|codec_pad|>",     "<|codec_bos|>",
        "<|codec_eos_token|>",        "<|codec_think|>",           "<|codec_nothink|>", "<|codec_think_bos|>",
        "<|codec_think_eos|>"};
    for (int i = 0; i < TINY_QWEN3TTS_CODEC; ++i) {
        const int special = i - TINY_QWEN3TTS_CODEBOOK - 2;
        tokens.push_back(special >= 0 && special < static_cast<int>(codec_specials.size())
                             ? codec_specials[static_cast<size_t>(special)]
                             : "<|codec_" + std::to_string(i) + "|>");
        types.push_back(3);
    }
    const int n_vocab = static_cast<int>(tokens.size());
    const int codec_eos = codec_0 + TINY_QWEN3TTS_CODEBOOK + 2 + 4;
    std::vector<int32_t> suppress;
    for (int id = codec_0 + TINY_QWEN3TTS_CODEBOOK; id < n_vocab; ++id) {
        if (id != codec_eos) {
            suppress.push_back(id);
        }
    }
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    std::vector<const char *> merge_ptrs;
    for (const std::string &m : merges) {
        merge_ptrs.push_back(m.c_str());
    }
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen2");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.suppress_tokens", GGUF_TYPE_INT32, suppress.data(), suppress.size());
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", static_cast<uint32_t>(codec_eos));
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", false);

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {EMBD}, 1.0f);
    // the codec head; the end of speech scores low, so a synthesis speaks until its frame cap
    ggml_tensor *head = add_random_tensor(g, ctx, rng, "output.weight", {EMBD, TINY_QWEN3TTS_CODEC}, 0.1f);
    for (int i = 0; i < EMBD; ++i) {
        ((float *)head->data)[(codec_eos - codec_0) * EMBD + i] = 0.0f;
    }
    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add_random_tensor(g, ctx, rng, p + "attn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_norm.weight", {EMBD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "attn_q.weight", {EMBD, N_HEAD * HEAD}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_k.weight", {EMBD, N_HEAD_KV * HEAD}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_v.weight", {EMBD, N_HEAD_KV * HEAD}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_output.weight", {N_HEAD * HEAD, EMBD}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "attn_q_norm.weight", {HEAD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "attn_k_norm.weight", {HEAD}, 1.0f);
        add_random_tensor(g, ctx, rng, p + "ffn_gate.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {FF, EMBD}, 0.1f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above (qwen3tts_gen, no speaker encoder): the code predictor (one
// layer, three acoustic codebooks after the first, so four codes a frame) and code2wav at what
// llama.cpp fixes -- 8 transformer layers of 16 heads, a sliding window of 72 frames, two
// upsampling blocks by 2 and the DAC decoder of four blocks -- with every free width small. The DAC
// keeps the real strides 8, 5, 4, 3 (kernels 16, 10, 8, 6), so a frame stays 1920 samples at 24 kHz.
// speaker_encoder: also the ECAPA-TDNN speaker encoder of the real pair (qwen3tts_spkenc: a stem TDNN,
// three SE-Res2Net blocks at llama.cpp's fixed scale 8 and dilations 2, 3, 4, attentive statistics
// pooling, the projection to one backbone row), which makes the pair clone a voice from a clip.
inline std::string write_tiny_qwen3tts_mmproj_gguf(const std::string &prefix = "jllama-qwen3tts-mmproj",
                                                   bool speaker_encoder = false) {
    constexpr int EMBD = 32, CP = 16, CP_HEAD = 8, CP_FF = 32, N_AC = 3, QD = 8, H0 = 16, C1 = 16, W2 = 16, W_HEAD = 2,
                  W_FF = 32, D0 = 32;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(53);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "qwen3tts_gen");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", CP);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", CP_FF);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 1);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", 2);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count_kv", 1);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-6f);

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude (weights); otherwise every element is value (norms, biases)
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
    };
    const int K = TINY_QWEN3TTS_CODEBOOK;
    // the code predictor
    add("a.gen.code.proj_in.weight", {EMBD, CP}, -0.1f);
    add("a.gen.code.proj_in.bias", {CP}, 0.0f);
    add("a.gen.code.head.weight", {CP, K, N_AC}, -0.1f);
    add("a.gen.code.embd.weight", {EMBD, K, N_AC}, -0.1f);
    add("a.gen.code.out_embd.weight", {EMBD, TINY_QWEN3TTS_CODEC}, -0.1f);
    add("a.gen.code.output_norm.weight", {CP}, 1.0f);
    const std::string c = "a.gen.code.blk.0";
    add(c + ".ln1.weight", {CP}, 1.0f);
    add(c + ".ln2.weight", {CP}, 1.0f);
    add(c + ".attn_q.weight", {CP, 2 * CP_HEAD}, -0.1f);
    add(c + ".attn_k.weight", {CP, CP_HEAD}, -0.1f);
    add(c + ".attn_v.weight", {CP, CP_HEAD}, -0.1f);
    add(c + ".attn_out.weight", {2 * CP_HEAD, CP}, -0.1f);
    add(c + ".attn_q_norm.weight", {CP_HEAD}, 1.0f);
    add(c + ".attn_k_norm.weight", {CP_HEAD}, 1.0f);
    add(c + ".ffn_gate.weight", {CP, CP_FF}, -0.1f);
    add(c + ".ffn_up.weight", {CP, CP_FF}, -0.1f);
    add(c + ".ffn_down.weight", {CP_FF, CP}, -0.1f);

    // code2wav
    const std::string w = "a.gen.wav";
    add(w + ".quant.first.in_proj.weight", {H0, QD}, -0.1f);
    add(w + ".quant.first.out_proj.weight", {QD, H0}, -0.1f);
    add(w + ".quant.first.codebook.weight", {QD, K}, -0.5f);
    add(w + ".quant.rest.in_proj.weight", {H0, QD}, -0.1f);
    add(w + ".quant.rest.out_proj.weight", {QD, H0}, -0.1f);
    add(w + ".quant.rest.codebook.weight", {QD, K, N_AC}, -0.5f);
    add(w + ".pre_conv.weight", {3, H0, C1}, -0.1f);
    add(w + ".pre_conv.bias", {C1}, 0.0f);
    add(w + ".tfm.in_proj.weight", {C1, W2}, -0.1f);
    add(w + ".tfm.in_proj.bias", {W2}, 0.0f);
    add(w + ".tfm.out_proj.weight", {W2, C1}, -0.1f);
    add(w + ".tfm.out_proj.bias", {C1}, 0.0f);
    add(w + ".tfm.output_norm.weight", {W2}, 1.0f);
    for (int il = 0; il < 8; ++il) {
        const std::string t = w + ".tfm.blk." + std::to_string(il);
        add(t + ".ln1.weight", {W2}, 1.0f);
        add(t + ".ln2.weight", {W2}, 1.0f);
        add(t + ".ls1.weight", {W2}, 0.1f);
        add(t + ".ls2.weight", {W2}, 0.1f);
        add(t + ".attn_q.weight", {W2, 16 * W_HEAD}, -0.1f);
        add(t + ".attn_k.weight", {W2, 16 * W_HEAD}, -0.1f);
        add(t + ".attn_v.weight", {W2, 16 * W_HEAD}, -0.1f);
        add(t + ".attn_out.weight", {16 * W_HEAD, W2}, -0.1f);
        add(t + ".ffn_gate.weight", {W2, W_FF}, -0.1f);
        add(t + ".ffn_up.weight", {W2, W_FF}, -0.1f);
        add(t + ".ffn_down.weight", {W_FF, W2}, -0.1f);
    }
    for (int il = 0; il < 2; ++il) {
        const std::string u = w + ".up.blk." + std::to_string(il);
        add(u + ".conv.weight", {2, C1, C1}, -0.1f);
        add(u + ".conv.bias", {C1}, 0.0f);
        add(u + ".dwconv.weight", {7, 1, C1}, -0.1f);
        add(u + ".dwconv.bias", {C1}, 0.0f);
        add(u + ".norm.weight", {C1}, 1.0f);
        add(u + ".norm.bias", {C1}, 0.0f);
        add(u + ".pw1.weight", {C1, 4 * C1}, -0.1f);
        add(u + ".pw1.bias", {4 * C1}, 0.0f);
        add(u + ".pw2.weight", {4 * C1, C1}, -0.1f);
        add(u + ".pw2.bias", {C1}, 0.0f);
        add(u + ".gamma", {C1}, 0.1f);
    }
    add(w + ".dac.entry.weight", {7, C1, D0}, -0.1f);
    add(w + ".dac.entry.bias", {D0}, 0.0f);
    const int strides[] = {8, 5, 4, 3};
    int d = D0;
    for (int il = 0; il < 4; ++il) {
        const std::string b = w + ".dac.blk." + std::to_string(il);
        add(b + ".snake.alpha", {d}, 1.0f);
        add(b + ".snake.beta", {d}, 1.0f);
        add(b + ".conv.weight", {2 * strides[il], d / 2, d}, -0.1f);
        add(b + ".conv.bias", {d / 2}, 0.0f);
        d /= 2;
        for (int ir = 0; ir < 3; ++ir) {
            const std::string r = b + ".res." + std::to_string(ir);
            add(r + ".act1.alpha", {d}, 1.0f);
            add(r + ".act1.beta", {d}, 1.0f);
            add(r + ".conv1.weight", {7, d, d}, -0.1f);
            add(r + ".conv1.bias", {d}, 0.0f);
            add(r + ".act2.alpha", {d}, 1.0f);
            add(r + ".act2.beta", {d}, 1.0f);
            add(r + ".conv2.weight", {1, d, d}, -0.1f);
            add(r + ".conv2.bias", {d}, 0.0f);
        }
    }
    add(w + ".dac.post_snake.alpha", {d}, 1.0f);
    add(w + ".dac.post_snake.beta", {d}, 1.0f);
    add(w + ".dac.post_conv.weight", {7, d, 1}, -0.1f);
    add(w + ".dac.post_conv.bias", {1}, 0.0f);
    if (speaker_encoder) {
        constexpr int MEL = 16, SC = 16, ATT = 8;
        gguf_set_val_bool(g, "clip.has_audio_encoder", true);
        gguf_set_val_str(g, "clip.audio.projector_type", "qwen3tts_spkenc");
        gguf_set_val_u32(g, "clip.audio.projection_dim", EMBD);
        gguf_set_val_u32(g, "clip.audio.num_mel_bins", MEL);
        gguf_set_val_u32(g, "clip.audio.block_count", 3);
        gguf_set_val_u32(g, "clip.audio.embedding_length", 3 * SC);
        gguf_set_val_u32(g, "clip.audio.attention.head_count", 1);
        gguf_set_val_u32(g, "clip.audio.feed_forward_length", 3 * SC);
        gguf_set_val_f32(g, "clip.audio.attention.layer_norm_epsilon", 1e-5f);
        add("a.conv1d.0.weight", {5, MEL, SC}, -0.1f);
        add("a.conv1d.0.bias", {SC}, 0.0f);
        for (int b = 1; b <= 3; ++b) {
            const std::string q = "a.blk." + std::to_string(b);
            for (int x = 0; x < 7; ++x) {
                add(q + ".res2." + std::to_string(x) + ".weight", {3, SC / 8, SC / 8}, -0.1f);
                add(q + ".res2." + std::to_string(x) + ".bias", {SC / 8}, 0.0f);
            }
            add(q + ".se_conv1.weight", {1, SC, SC / 2}, -0.1f);
            add(q + ".se_conv1.bias", {SC / 2}, 0.0f);
            add(q + ".se_conv2.weight", {1, SC / 2, SC}, -0.1f);
            add(q + ".se_conv2.bias", {SC}, 0.0f);
            add(q + ".conv_pw1.weight", {1, SC, SC}, -0.1f);
            add(q + ".conv_pw1.bias", {SC}, 0.0f);
            add(q + ".conv_pw2.weight", {1, SC, SC}, -0.1f);
            add(q + ".conv_pw2.bias", {SC}, 0.0f);
        }
        add("a.conv_out.weight", {1, 3 * SC, 3 * SC}, -0.1f);
        add("a.conv_out.bias", {3 * SC}, 0.0f);
        add("a.asp_tdnn.weight", {1, 9 * SC, ATT}, -0.1f);
        add("a.asp_tdnn.bias", {ATT}, 0.0f);
        add("a.asp_attn.weight", {1, ATT, 3 * SC}, -0.1f);
        add("a.asp_attn.bias", {3 * SC}, 0.0f);
        add("mm.a.fc.weight", {1, 6 * SC, EMBD}, -0.1f);
        add("mm.a.fc.bias", {EMBD}, 0.0f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The pair of a random LFM3-Audio (llama.cpp's lfm3 / lfm3a_gen), in the layout of CI's real 2B pair,
// read from its GGUFs. The backbone mixes LFM3's three layer kinds -- a short convolution, a sliding
// window attention and a full attention layer, each attention with a per-head sigmoid gate, q/k norm
// and sandwich norms -- over the GPT-2 bytes and the control tokens the pipeline looks up by text. Its
// "tts" template is this project's own, in the shape of the real one: a system turn naming the
// language and the voice (female or male, the only two llama.cpp knows for this model), the text as
// the user turn, and the assistant turn opened with <|audio_start|>, which puts the pipeline in audio.
// chat_templates: the default chat template and the "asr" one as well, as the real file has them. The
// default is a plain ChatML one marked with <additional_tools> in a comment, the text llama.cpp
// recognises an LFM3 template by (is_lfm3_template: it then builds the transcription prompt as
// "Perform ASR[ in <language>]." in the system turn); the "asr" one is the real one's, the prompt of
// live transcription.
inline std::string write_tiny_lfm3_gguf(const std::string &prefix = "jllama-lfm3", bool tts_template = true,
                                        bool chat_templates = false) {
    constexpr int EMBD = 32, N_HEAD = 2, HEAD = 16, FF = 64, L_CACHE = 3;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(59);
    gguf_context *g = gguf_init_empty();
    const std::string a = "lfm3";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".block_count").c_str(), 3);
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 4096);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    const int32_t kv[3] = {0, 1, 1}; // 0: the short convolution
    gguf_set_arr_data(g, (a + ".attention.head_count_kv").c_str(), GGUF_TYPE_INT32, kv, 3);
    const bool swa[3] = {false, true, false};
    gguf_set_arr_data(g, (a + ".attention.sliding_window_pattern").c_str(), GGUF_TYPE_BOOL, swa, 3);
    gguf_set_val_u32(g, (a + ".attention.sliding_window").c_str(), 64);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 1000000.0f);
    gguf_set_val_f32(g, (a + ".rope.freq_base_swa").c_str(), 10000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(g, (a + ".attention.key_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".attention.value_length").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".rope.dimension_count").c_str(), HEAD);
    gguf_set_val_u32(g, (a + ".shortconv.l_cache").c_str(), L_CACHE);

    std::vector<std::string> tokens = byte_tokens();
    std::vector<int32_t> types(tokens.size(), 1);
    const std::vector<std::string> control = {"<|pad|>",         "<|startoftext|>", "<|endoftext|>",
                                              "<|im_start|>",    "<|im_end|>",      "<|text_end|>",
                                              "<|audio_start|>", "<|mixed_start|>", "<|mixed_end|>"};
    const int first_control = static_cast<int>(tokens.size());
    for (const std::string &t : control) {
        tokens.push_back(t);
        types.push_back(3);
    }
    const int n_vocab = static_cast<int>(tokens.size());
    std::vector<const char *> token_ptrs;
    for (const std::string &t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    const char *merges[] = {"Ġ Ġ"};
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "lfm2");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", static_cast<uint32_t>(first_control));
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", static_cast<uint32_t>(first_control + 1));
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", static_cast<uint32_t>(first_control + 4));
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", false);
    if (tts_template) {
        gguf_set_val_str(
            g, "tokenizer.chat_template.tts",
            "{%- set names = {'en': 'English', 'de': 'German'} -%}"
            "{%- set code = (language | lower) if language else 'en' -%}"
            "{%- set lang = names[code] if code in names else language -%}"
            "{%- set v = (voice | lower) if voice and (voice | lower) in ['female', 'male'] else 'female' -%}"
            "{{- bos_token -}}"
            "{{- '<|im_start|>system\\nSpeak ' + lang + ' with a ' + v + ' voice.<|im_end|>\\n' -}}"
            "{{- '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
            "{{- '<|im_start|>assistant\\n<|audio_start|>' -}}");
        if (!chat_templates) {
            const char *template_names[] = {"tts"};
            gguf_set_arr_str(g, "tokenizer.chat_templates", template_names, 1);
        }
    }
    if (chat_templates) {
        gguf_set_val_str(g, "tokenizer.chat_template",
                         "{#- an LFM3 template: <additional_tools> -#}{{- bos_token -}}"
                         "{%- for m in messages -%}"
                         "{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' -}}"
                         "{%- endfor -%}"
                         "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}");
        gguf_set_val_str(g, "tokenizer.chat_template.asr",
                         "{{- bos_token -}}{{- '<|im_start|>system\\nPerform live ASR.<|im_end|>\\n' -}}"
                         "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}");
        if (tts_template) {
            const char *template_names[] = {"tts", "asr"};
            gguf_set_arr_str(g, "tokenizer.chat_templates", template_names, 2);
        } else {
            const char *template_names[] = {"asr"};
            gguf_set_arr_str(g, "tokenizer.chat_templates", template_names, 1);
        }
    }

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {EMBD}, 1.0f);
    add_random_tensor(g, ctx, rng, "rope_freqs.weight", {HEAD / 2}, 1.0f);
    for (int il = 0; il < 3; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        for (const char *n : {"attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm"}) {
            add_random_tensor(g, ctx, rng, p + n + ".weight", {EMBD}, 1.0f);
        }
        add_random_tensor(g, ctx, rng, p + "ffn_gate.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {EMBD, FF}, 0.1f);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {FF, EMBD}, 0.1f);
        if (il == 0) {
            add_random_tensor(g, ctx, rng, p + "shortconv.conv.weight", {L_CACHE, EMBD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "shortconv.in_proj.weight", {EMBD, 3 * EMBD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "shortconv.out_proj.weight", {EMBD, EMBD}, 0.1f);
        } else {
            add_random_tensor(g, ctx, rng, p + "attn_q.weight", {EMBD, N_HEAD * HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_k.weight", {EMBD, HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_v.weight", {EMBD, HEAD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_output.weight", {N_HEAD * HEAD, EMBD}, 0.1f);
            add_random_tensor(g, ctx, rng, p + "attn_q_norm.weight", {HEAD}, 1.0f);
            add_random_tensor(g, ctx, rng, p + "attn_k_norm.weight", {HEAD}, 1.0f);
            add_random_tensor(g, ctx, rng, p + "attn_gate.weight", {EMBD, N_HEAD}, 0.1f);
        }
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above (lfm3a_gen): the depthformer (one gated attention layer over the 12
// codebook positions llama.cpp fixes, a code vocabulary of 16 codes plus the end of audio) and the
// codec decoder at what llama.cpp fixes -- 8 layers, the 30-frame window, six STFT frames a codec frame
// and 1282 polar rows for its 1280-point STFT, so 1920 samples a frame at 24 kHz. The decoder is 512
// wide: llama.cpp sizes the state of its convolution layers from that constant, not from the tensors.
// Its layers alternate the short convolution and the gated attention, as both kinds occur in the
// real one. eos_scale: the end-of-audio row of the tied code head (see below).
//
// audio_encoder: the audio input half as well (lfm3a, the real file holds both): a downsampler of
// three stride-2 convolutions over time and mel bins (8x each way) with LFM3A_CH channels, a stack of
// two layers -- a short convolution and an attention layer with q/k norms and a sliding window of
// LFM3A_WINDOW positions, the two kinds the real 17 layers mix -- and the LayerNorm-GELU projector
// to the backbone's width. 16 mel bins (the real 128), from 16 kHz audio with a 160-sample hop.
constexpr int LFM3A_MEL = 16, LFM3A_CH = 4, LFM3A_EMBD = 16, LFM3A_WINDOW = 4;
// audio_temp: the projector's default temperature of the audio codes (clip.gen.audio.sampling.temp, the
// real one's 1.0), which a caller that names none -- the realtime conversation -- samples with.
inline std::string write_tiny_lfm3_mmproj_gguf(const std::string &prefix = "jllama-lfm3-mmproj", float eos_scale = 0.0f,
                                               bool audio_encoder = false, float audio_temp = 1.0f) {
    constexpr int EMBD = 32, CP = 16, CP_HEAD = 8, N_CB = 12, V = 17, QO = 16, DEC = 512, D_HEAD = 8, DEC_FF = 32;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(61);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "lfm3a_gen");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", CP);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", 32);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 1);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", 2);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count_kv", 1);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_f32(g, "clip.gen.audio.sampling.temp", audio_temp);
    gguf_set_val_i32(g, "clip.gen.audio.sampling.top_k", 128);
    gguf_set_val_f32(g, "clip.gen.audio.sampling.penalty_repeat", 1.1f);
    gguf_set_val_i32(g, "clip.gen.audio.sampling.penalty_last_n", 4096);
    if (audio_encoder) {
        gguf_set_val_bool(g, "clip.has_audio_encoder", true);
        gguf_set_val_str(g, "clip.audio.projector_type", "lfm3a");
        gguf_set_val_u32(g, "clip.audio.projection_dim", EMBD);
        gguf_set_val_u32(g, "clip.audio.embedding_length", LFM3A_EMBD);
        gguf_set_val_u32(g, "clip.audio.feed_forward_length", 32);
        gguf_set_val_u32(g, "clip.audio.block_count", 2);
        gguf_set_val_u32(g, "clip.audio.attention.head_count", 2);
        gguf_set_val_u32(g, "clip.audio.attention.head_count_kv", 1);
        gguf_set_val_f32(g, "clip.audio.attention.layer_norm_epsilon", 1e-5f);
        gguf_set_val_u32(g, "clip.audio.num_mel_bins", LFM3A_MEL);
        gguf_set_val_u32(g, "clip.audio.window_size", LFM3A_WINDOW);
    }

    ggml_init_params ip = {64u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude (weights); otherwise every element is value (norms, biases)
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
        return t;
    };
    // the depthformer; its head is tied to the code embedding
    ggml_tensor *embd = add("a.gen.code.embd.weight", {CP, V, N_CB}, -0.1f);
    // the end-of-audio row of every codebook is eos_scale in every element: 0 gives it the logit 0,
    // which greedy decoding never picks over 16 random codes; a large value wins in every codebook
    // whose normed hidden state sums positive, which with 12 codebooks is all but certain
    for (int cb = 0; cb < N_CB; ++cb) {
        for (int i = 0; i < CP; ++i) {
            // eos_scale < 0: the row is eos_scale in the first element only, which the input bias below
            // keeps positive after every norm, so the end of audio never wins (a sign the random weights
            // cannot flip, unlike a row of one value over a normed random vector)
            ((float *)embd->data)[(static_cast<int64_t>(cb) * V + (V - 1)) * CP + i] =
                eos_scale < 0 && i > 0 ? 0.0f : eos_scale;
        }
    }
    add("a.gen.code.head_norm.weight", {CP, N_CB}, 1.0f);
    add("a.gen.code.mask_embd", {CP}, 0.0f);
    add("a.gen.code.out_embd.weight", {EMBD, N_CB * V}, -0.1f);
    add("a.gen.code.output_norm.weight", {CP}, 1.0f);
    add("a.gen.code.proj_in.weight", {EMBD, N_CB * CP}, -0.1f);
    ggml_tensor *proj_in_bias = add("a.gen.code.proj_in.bias", {N_CB * CP}, 0.0f);
    if (eos_scale < 0) {
        for (int cb = 0; cb < N_CB; ++cb) {
            ((float *)proj_in_bias->data)[cb * CP] = 50.0f; // dominates the first element of every codebook input
        }
    }
    const std::string c = "a.gen.code.blk.0";
    add(c + ".ln1.weight", {CP}, 1.0f);
    add(c + ".ln2.weight", {CP}, 1.0f);
    add(c + ".attn_q.weight", {CP, 2 * CP_HEAD}, -0.1f);
    add(c + ".attn_k.weight", {CP, CP_HEAD}, -0.1f);
    add(c + ".attn_v.weight", {CP, CP_HEAD}, -0.1f);
    add(c + ".attn_out.weight", {2 * CP_HEAD, CP}, -0.1f);
    add(c + ".attn_q_norm.weight", {CP_HEAD}, 1.0f);
    add(c + ".attn_k_norm.weight", {CP_HEAD}, 1.0f);
    add(c + ".attn_gate.weight", {CP, 2}, -0.1f);
    add(c + ".ffn_gate.weight", {CP, 32}, -0.1f);
    add(c + ".ffn_up.weight", {CP, 32}, -0.1f);
    add(c + ".ffn_down.weight", {32, CP}, -0.1f);

    // the codec decoder: FSQ levels (4 per code), one projection over all codebooks, the LFM layers
    const std::string w = "a.gen.wav";
    add(w + ".quant.first.codebook.weight", {4, V - 1}, -0.5f);
    add(w + ".quant.first.out_proj.weight", {4 * N_CB, QO}, -0.1f);
    add(w + ".quant.first.out_proj.bias", {QO}, 0.0f);
    add(w + ".tfm.in_proj.weight", {QO, DEC}, -0.1f);
    add(w + ".tfm.in_proj.bias", {DEC}, 0.0f);
    add(w + ".tfm.out_proj.weight", {DEC, 1282}, -0.01f);
    add(w + ".tfm.out_proj.bias", {1282}, 0.0f);
    add(w + ".tfm.output_norm.weight", {DEC}, 1.0f);
    for (int il = 0; il < 8; ++il) {
        const std::string t = w + ".tfm.blk." + std::to_string(il);
        for (const char *n : {"ln1", "attn_post_norm", "ln2", "ffn_post_norm"}) {
            add(t + "." + n + ".weight", {DEC}, 1.0f);
        }
        add(t + ".ffn_gate.weight", {DEC, DEC_FF}, -0.05f);
        add(t + ".ffn_up.weight", {DEC, DEC_FF}, -0.05f);
        add(t + ".ffn_down.weight", {DEC_FF, DEC}, -0.05f);
        if (il % 2 == 0) {
            add(t + ".conv_pw1.weight", {DEC, 3 * DEC}, -0.05f);
            add(t + ".conv_dw.weight", {7, DEC}, -0.1f);
            add(t + ".conv_pw2.weight", {DEC, DEC}, -0.05f);
        } else {
            add(t + ".attn_q.weight", {DEC, 2 * D_HEAD}, -0.05f);
            add(t + ".attn_k.weight", {DEC, D_HEAD}, -0.05f);
            add(t + ".attn_v.weight", {DEC, D_HEAD}, -0.05f);
            add(t + ".attn_out.weight", {2 * D_HEAD, DEC}, -0.05f);
            add(t + ".attn_q_norm.weight", {D_HEAD}, 1.0f);
            add(t + ".attn_k_norm.weight", {D_HEAD}, 1.0f);
            add(t + ".attn_gate.weight", {DEC, 2}, -0.05f);
        }
    }
    if (audio_encoder) {
        constexpr int E = LFM3A_EMBD, D_A = 8, FF_A = 32;
        // the downsampler: a full 3x3 convolution from the one input channel, then twice a depthwise
        // 3x3 and a pointwise 1x1 (tensor indices 0, 2+3, 5+6 as the converter writes them)
        add("a.conv1d.0.weight", {3, 3, 1, LFM3A_CH}, -0.3f);
        add("a.conv1d.0.bias", {1, 1, LFM3A_CH}, 0.0f);
        for (int i : {2, 5}) {
            add("a.conv1d." + std::to_string(i) + ".weight", {3, 3, 1, LFM3A_CH}, -0.3f);
            add("a.conv1d." + std::to_string(i) + ".bias", {1, 1, LFM3A_CH}, 0.0f);
            add("a.conv1d." + std::to_string(i + 1) + ".weight", {1, 1, LFM3A_CH, LFM3A_CH}, -0.3f);
            add("a.conv1d." + std::to_string(i + 1) + ".bias", {1, 1, LFM3A_CH}, 0.0f);
        }
        add("a.pre_encode.out.weight", {LFM3A_CH * LFM3A_MEL / 8, E}, -0.3f);
        add("a.pre_encode.out.bias", {E}, 0.0f);
        for (int il = 0; il < 2; ++il) {
            const std::string b = "a.blk." + std::to_string(il);
            add(b + ".ln1.weight", {E}, 1.0f);
            add(b + ".ln2.weight", {E}, 1.0f);
            add(b + ".ffn_gate.weight", {E, FF_A}, -0.2f);
            add(b + ".ffn_up.weight", {E, FF_A}, -0.2f);
            add(b + ".ffn_down.weight", {FF_A, E}, -0.2f);
            if (il == 0) {
                add(b + ".conv_pw1.weight", {E, 3 * E}, -0.2f);
                add(b + ".conv_dw.weight", {3, E}, -0.3f);
                add(b + ".conv_pw2.weight", {E, E}, -0.2f);
            } else {
                add(b + ".attn_q.weight", {E, E}, -0.2f);
                add(b + ".attn_k.weight", {E, D_A}, -0.2f);
                add(b + ".attn_v.weight", {E, D_A}, -0.2f);
                add(b + ".attn_out.weight", {E, E}, -0.2f);
                add(b + ".attn_q_norm.weight", {D_A}, 1.0f);
                add(b + ".attn_k_norm.weight", {D_A}, 1.0f);
            }
        }
        add("a.post_ln.weight", {E}, 1.0f);
        add("mm.a.mlp.0.weight", {E}, 1.0f);
        add("mm.a.mlp.0.bias", {E}, 0.0f);
        add("mm.a.mlp.1.weight", {E, EMBD}, -0.3f);
        add("mm.a.mlp.1.bias", {EMBD}, 0.0f);
        add("mm.a.mlp.3.weight", {EMBD, EMBD}, -0.3f);
        add("mm.a.mlp.3.bias", {EMBD}, 0.0f);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// tiny-voca's 76 symbols, in the order of the real vocabulary: four specials, <unk>, punctuation, the IPA
// consonants and vowels of the model.
inline std::vector<std::string> tiny_voca_tokens() {
    return {"<pad>",
            "<bos>",
            "<eos>",
            "<sp>",
            "<unk>",
            ",",
            ".",
            "?",
            "!",
            ";",
            ":",
            "\"",
            "'",
            "(",
            ")",
            "\xE2\x80\x94",
            "\xE2\x80\xA6",
            "p",
            "b",
            "t",
            "d",
            "k",
            "\xC9\xA1",
            "f",
            "v",
            "\xCE\xB8",
            "\xC3\xB0",
            "s",
            "z",
            "\xCA\x83",
            "\xCA\x92",
            "h",
            "m",
            "n",
            "\xC5\x8B",
            "l",
            "\xC9\xB9",
            "j",
            "w",
            "t\xCA\x83",
            "d\xCA\x92",
            "\xC9\xBE",
            "\xCA\x94",
            "\xC9\x99",
            "\xC9\x90",
            "\xC9\x9A",
            "\xE1\xB5\xBB",
            "\xC9\xAA",
            "i",
            "i\xCB\x90",
            "\xC9\x9B",
            "\xC3\xA6",
            "\xCA\x8C",
            "\xC9\x91\xCB\x90",
            "\xC9\x94",
            "\xC9\x94\xCB\x90",
            "\xCA\x8A",
            "u\xCB\x90",
            "\xC9\x9C\xCB\x90",
            "e\xC9\xAA",
            "a\xC9\xAA",
            "\xC9\x94\xC9\xAA",
            "o\xCA\x8A",
            "a\xCA\x8A",
            "i\xC9\x99",
            "o\xCB\x90",
            "\xC9\x91\xCB\x90\xC9\xB9",
            "\xC9\x94\xCB\x90\xC9\xB9",
            "o\xCB\x90\xC9\xB9",
            "\xC9\xAA\xC9\xB9",
            "\xC9\x9B\xC9\xB9",
            "\xCA\x8A\xC9\xB9",
            "a\xC9\xAA\xC9\x9A",
            "a\xC9\xAA\xC9\x99",
            "\xC9\x99l",
            "n\xCC\xA9"};
}

// The class of each symbol, as the real a.gen.token_class: 0 for the specials and punctuation (a pause,
// whose presence the duration head decides), 1 for <unk> and the consonants, 2 for the vowels (always
// voiced).
inline float tiny_voca_token_class(int id) { return id == 4 ? 1.0f : id < 17 ? 0.0f : id < 43 ? 1.0f : 2.0f; }

// One stack of tiny-voca's layer kind under a prefix ("" is the backbone), plus its output norm.
inline void tiny_voca_stack(gguf_context *g, ggml_context *ctx, std::mt19937 &rng, const std::string &prefix,
                            int n_layer, int64_t embd, int64_t ff) {
    auto norm = [&](const std::string &name) {
        add_random_tensor(g, ctx, rng, name + ".weight", {embd}, 1.0f); // ones
        ggml_tensor *b = add_random_tensor(g, ctx, rng, name + ".bias", {embd}, 1.0f);
        std::fill((float *)b->data, (float *)b->data + embd, 0.0f);
    };
    auto bias = [&](const std::string &name, int64_t n) {
        ggml_tensor *b = add_random_tensor(g, ctx, rng, name, {n}, 1.0f);
        std::fill((float *)b->data, (float *)b->data + n, 0.0f);
    };
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = prefix + "blk." + std::to_string(il) + ".";
        norm(p + "attn_norm");
        for (const char *n : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            add_random_tensor(g, ctx, rng, p + n + ".weight", {embd, embd}, 0.2f);
            bias(p + n + ".bias", embd);
        }
        norm(p + "ffn_norm");
        add_random_tensor(g, ctx, rng, p + "ssm_conv1d.weight", {7, embd}, 0.2f);
        bias(p + "ssm_conv1d.bias", embd);
        add_random_tensor(g, ctx, rng, p + "ffn_up.weight", {embd, ff}, 0.2f);
        bias(p + "ffn_up.bias", ff);
        add_random_tensor(g, ctx, rng, p + "ffn_down.weight", {ff, embd}, 0.2f);
        bias(p + "ffn_down.bias", embd);
    }
    norm(prefix + "output_norm");
}

// The pair of a random tiny-voca (llama.cpp b12497, #32427), in the layout of ggml-org's real pair, read
// from its GGUFs: the backbone is the phoneme encoder (arch tiny-voca: a non-causal stack whose layers
// carry attention, a depthwise convolution of 7 and a feed-forward, each with biases and LayerNorms),
// over the real 76-symbol vocabulary -- IPA phonemes and punctuation, which the pipeline looks up by text
// (<pad>, <bos>, <eos>, <sp>, <unk> and "."). No tts template unless one is given: the prompt is then
// the phonemes, separated by spaces.
inline std::string write_tiny_voca_gguf(const std::string &prefix = "jllama-tiny-voca",
                                        const std::string &tts_template = "") {
    constexpr int EMBD = 16, N_HEAD = 2, FF = 32, N_LAYER = 2;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(71);
    gguf_context *g = gguf_init_empty();
    const std::string a = "tiny-voca";
    gguf_set_val_str(g, "general.architecture", a.c_str());
    gguf_set_val_u32(g, (a + ".block_count").c_str(), N_LAYER);
    gguf_set_val_u32(g, (a + ".context_length").c_str(), 512);
    gguf_set_val_u32(g, (a + ".embedding_length").c_str(), EMBD);
    gguf_set_val_u32(g, (a + ".feed_forward_length").c_str(), FF);
    gguf_set_val_u32(g, (a + ".attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(g, (a + ".attention.head_count_kv").c_str(), N_HEAD);
    gguf_set_val_f32(g, (a + ".rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_f32(g, (a + ".attention.layer_norm_epsilon").c_str(), 1e-5f);
    gguf_set_val_bool(g, (a + ".attention.causal").c_str(), false);
    gguf_set_val_u32(g, (a + ".pooling_type").c_str(), 0);

    const std::vector<std::string> tokens = tiny_voca_tokens();
    std::vector<const char *> token_ptrs;
    std::vector<int32_t> types;
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_ptrs.push_back(tokens[i].c_str());
        types.push_back(i < 4 ? 3 : 1); // the four specials are control tokens, <unk> a normal one, as in the real file
    }
    const std::vector<float> scores(tokens.size(), 0.0f);
    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "default");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 4);
    gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", false);
    if (!tts_template.empty()) {
        gguf_set_val_str(g, "tokenizer.chat_template.tts", tts_template.c_str());
        const char *template_names[] = {"tts"};
        gguf_set_arr_str(g, "tokenizer.chat_templates", template_names, 1);
    }

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    add_random_tensor(g, ctx, rng, "token_embd.weight", {EMBD, static_cast<int64_t>(tokens.size())}, 0.1f);
    tiny_voca_stack(g, ctx, rng, "", N_LAYER, EMBD, FF);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// The mmproj of the backbone above (tiny_voca): the duration predictor (two convolutions of 3 with
// LayerNorms, then a head of two outputs per token), the prosody (2 layers) and acoustic (5 layers)
// stacks llama.cpp fixes, run as models of the backbone's architecture, the pitch / voicing / energy
// head and its projection, and the source-filter heads at the sizes llama.cpp fixes (80 + 16 causal and
// anti-causal cepstra + 2 harmonic steps, 32 noise cepstra + 4 noise steps, so a frame is 256 samples at
// 24 kHz). Two voices, F2 and M2 (the default, as the real pack). Every width is small.
//
// The duration head's weight is zero, so every token gets the same outputs, its bias: frames is the
// number of frames of a phone (log1p(frames), which the pipeline rounds back), and presence the logit
// that decides whether a token of class 0 (punctuation, <bos>, <eos>) lasts as long (> 0) or no frame at
// all (< 0). g2p_symbol (1 to 69): a letter-to-sound model whose output bias makes every slot of every
// word that symbol of CMUdict (llama.cpp's list, "AA0" is 1), so tts_g2p() answers it for any word;
// 0 writes none.
inline std::string write_tiny_voca_mmproj_gguf(const std::string &prefix = "jllama-tiny-voca-mmproj", int frames = 3,
                                               float presence = 1.0f, int g2p_symbol = 0) {
    constexpr int EMBD = 16, N_HEAD = 2, FF = 32, DUR = 8, G2P_W = 4, G2P_H = 4;
    static std::atomic<int> counter{0};
    static const unsigned run_id = std::random_device{}();
    const std::string path = (std::filesystem::temp_directory_path() /
                              (prefix + "-" + std::to_string(run_id) + "-" + std::to_string(counter++) + ".gguf"))
                                 .string();
    std::mt19937 rng(73);
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_gen_audio_encoder", true);
    gguf_set_val_str(g, "clip.gen.audio.projector_type", "tiny_voca");
    gguf_set_val_u32(g, "clip.gen.audio.projection_dim", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.embedding_length", EMBD);
    gguf_set_val_u32(g, "clip.gen.audio.feed_forward_length", FF);
    gguf_set_val_u32(g, "clip.gen.audio.block_count", 5);
    gguf_set_val_u32(g, "clip.gen.audio.attention.head_count", N_HEAD);
    gguf_set_val_f32(g, "clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_str(g, "clip.gen.audio.default_voice", "M2");

    ggml_init_params ip = {16u * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    // value < 0: noise of that magnitude; otherwise every element is value
    auto add = [&](const std::string &name, std::vector<int64_t> ne, float value) {
        ggml_tensor *t = ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
        std::normal_distribution<float> dist(0.0f, value < 0 ? -value : 0.0f);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            ((float *)t->data)[i] = value < 0 ? dist(rng) : value;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g, t);
        return t;
    };
    const std::vector<std::string> tokens = tiny_voca_tokens();
    const int64_t n_vocab = static_cast<int64_t>(tokens.size());
    if (g2p_symbol > 0) {
        add("a.gen.g2p.slot", {G2P_W, 2}, -0.3f);
        add("a.gen.g2p.embed.weight", {G2P_W, 28}, -0.3f);
        for (const char *dir : {"", "_reverse"}) {
            const std::string sfx = std::string("_l0") + dir;
            add("a.gen.g2p.gru.weight_ih" + sfx, {G2P_W, 3 * G2P_H}, -0.3f);
            add("a.gen.g2p.gru.weight_hh" + sfx, {G2P_H, 3 * G2P_H}, -0.3f);
            add("a.gen.g2p.gru.bias_ih" + sfx, {3 * G2P_H}, 0.0f);
            add("a.gen.g2p.gru.bias_hh" + sfx, {3 * G2P_H}, 0.0f);
        }
        add("a.gen.g2p.out.weight", {2 * G2P_H, 70}, 0.0f);
        ggml_tensor *bias = add("a.gen.g2p.out.bias", {70}, 0.0f);
        ((float *)bias->data)[g2p_symbol] = 10.0f; // the blank (0) and every other symbol stay at 0
    }
    add("voice.F2", {2 * EMBD}, -0.5f);
    add("voice.M2", {2 * EMBD}, -0.5f);
    ggml_tensor *token_class = add("a.gen.token_class", {n_vocab}, 0.0f);
    for (int64_t i = 0; i < n_vocab; ++i) {
        ((float *)token_class->data)[i] = tiny_voca_token_class(static_cast<int>(i));
    }
    add("a.gen.tok_embd.weight", {EMBD, n_vocab}, -0.3f);
    add("a.gen.stress_embd.weight", {EMBD, 3}, -0.3f);
    add("a.gen.dur.conv1.weight", {3, EMBD, DUR}, -0.3f);
    add("a.gen.dur.conv1.bias", {DUR}, 0.0f);
    add("a.gen.dur.conv2.weight", {3, DUR, DUR}, -0.3f);
    add("a.gen.dur.conv2.bias", {DUR}, 0.0f);
    add("a.gen.dur.norm1.weight", {DUR}, 1.0f);
    add("a.gen.dur.norm1.bias", {DUR}, 0.0f);
    add("a.gen.dur.norm2.weight", {DUR}, 1.0f);
    add("a.gen.dur.norm2.bias", {DUR}, 0.0f);
    add("a.gen.dur.out.weight", {DUR, 2}, 0.0f);
    ggml_tensor *dur_bias = add("a.gen.dur.out.bias", {2}, 0.0f);
    ((float *)dur_bias->data)[0] = std::log1p(static_cast<float>(frames));
    ((float *)dur_bias->data)[1] = presence;
    add("a.gen.frame_pos.weight", {3, EMBD}, -0.3f);
    add("a.gen.frame_pos.bias", {EMBD}, 0.0f);
    tiny_voca_stack(g, ctx, rng, "a.gen.prosody.", 2, EMBD, FF);
    add("a.gen.prosody_head.weight", {EMBD, 3}, -0.3f);
    add("a.gen.prosody_head.bias", {3}, 0.0f);
    add("a.gen.prosody_proj.weight", {3, EMBD}, -0.3f);
    add("a.gen.prosody_proj.bias", {EMBD}, 0.0f);
    tiny_voca_stack(g, ctx, rng, "a.gen.acoustic.", 5, EMBD, FF);
    add("a.gen.voc.harmonic.weight", {EMBD, 98}, -0.1f);
    add("a.gen.voc.harmonic.bias", {98}, 0.0f);
    add("a.gen.voc.noise.weight", {EMBD, 36}, -0.1f);
    add("a.gen.voc.noise.bias", {36}, 0.0f);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? path : std::string();
}

// A 2x2 red PNG, base64: an image stb_image decodes, small enough to inline, for the projector above.
constexpr const char *TINY_PNG_BASE64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

} // namespace jllama_test
