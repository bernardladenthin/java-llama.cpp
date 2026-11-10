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
    if (spec.n_expert > 0) {
        gguf_set_val_u32(g, (a + ".expert_count").c_str(), static_cast<uint32_t>(spec.n_expert));
        gguf_set_val_u32(g, (a + ".expert_used_count").c_str(), static_cast<uint32_t>(spec.n_expert_used));
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
    add_random_tensor(g, ctx, rng, "token_embd.weight", {spec.n_embd, n_vocab}, 0.1f);
    add_random_tensor(g, ctx, rng, "output_norm.weight", {spec.n_embd}, 1.0f);
    ggml_tensor *output = add_random_tensor(g, ctx, rng, "output.weight", {spec.n_embd, n_vocab}, 0.1f);
    if (!sentencepiece) {
        confine_output_to_ascii(output); // token ids are bytes there; every SentencePiece piece is printable
    }
    gguf_context *shard = shard_path != nullptr ? gguf_init_empty() : nullptr;
    for (int il = 0; il < n_layer_all; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        gguf_context *to = shard != nullptr && il == spec.n_layer - 1 ? shard : g;
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
        add_random_tensor(to, ctx, rng, p + "attn_q.weight", {spec.n_embd, spec.n_head * spec.head_dim}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "attn_k.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "attn_v.weight", {spec.n_embd, spec.n_head_kv * spec.head_dim}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "attn_output.weight", {spec.n_head * spec.head_dim, spec.n_embd}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "ffn_norm.weight", {spec.n_embd}, 1.0f);
        if (spec.n_expert > 0) {
            // the router needs spread-out logits, or every token takes the same experts
            add_random_tensor(to, ctx, rng, p + "ffn_gate_inp.weight", {spec.n_embd, spec.n_expert}, 1.0f);
            add_random_tensor(to, ctx, rng, p + "ffn_gate_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "ffn_down_exps.weight", {spec.n_ff, spec.n_embd, spec.n_expert}, 0.1f);
            add_random_tensor(to, ctx, rng, p + "ffn_up_exps.weight", {spec.n_embd, spec.n_ff, spec.n_expert}, 0.1f);
            continue;
        }
        add_random_tensor(to, ctx, rng, p + "ffn_gate.weight", {spec.n_embd, spec.n_ff}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "ffn_up.weight", {spec.n_embd, spec.n_ff}, 0.1f);
        add_random_tensor(to, ctx, rng, p + "ffn_down.weight", {spec.n_ff, spec.n_embd}, 0.1f);
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

// A 2x2 red PNG, base64: an image stb_image decodes, small enough to inline, for the projector above.
constexpr const char *TINY_PNG_BASE64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

} // namespace jllama_test
