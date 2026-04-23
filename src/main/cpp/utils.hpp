#pragma once

// server-common.h provides: JSON_ASSERT, json, raw_buffer, json_value<T>,
// server_grammar_trigger, server_tokens, error_type, SRV_*/SLT_* macros,
// and many utility function declarations (implemented in server-common.cpp).
#include "server-common.h"

#include "download.h" // common_remote_get_content, common_remote_params
#include "base64.hpp"
#include "build-info.h"
#include "mtmd-helper.h"

#include <cinttypes>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"

// server-common.h uses slot.task->id; redefine with our simpler slot.id_task
#undef SLT_INF
#undef SLT_CNT
#undef SLT_WRN
#undef SLT_ERR
#undef SLT_DBG
#define SLT_INF(slot, fmt, ...)                                                                                        \
    LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...)                                                                                        \
    LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...)                                                                                        \
    LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...)                                                                                        \
    LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, (slot).id_task, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Token-piece JSON serialisation helpers
//
// There are two distinct wire formats for representing a token piece that may
// not be valid UTF-8, used in different parts of the API.  The helpers below
// implement each format exactly once and are documented so the two are never
// accidentally conflated.
//
// 1. token_piece_value()   — llama.cpp /tokenize endpoint (native format)
//    Schema  : a single JSON value that is EITHER a string OR a byte array.
//    Use for : handleTokenize, and any endpoint that follows the llama.cpp
//              /tokenize wire format.
//    Example : {"id": 123, "piece": "hello"}
//              {"id": 456, "piece": [195, 169]}
//
// 2. token_piece_oai_fields() — OpenAI completion probabilities format
//    Schema  : a partial JSON object with BOTH "token" (truncated UTF-8
//              string) AND "bytes" (full raw-byte array) always present.
//    Use for : completion_token_output::to_json / probs_vector_to_json, and
//              any endpoint that follows the OpenAI logprobs wire format.
//    Example : {"token": "hell", "bytes": [104,101,108,108,111], ...}
//
// Shared building block used by both:
//
// 3. str_to_bytes() — converts every byte of a string to an int in a JSON
//    array.  Used directly by token_piece_value (invalid-UTF-8 branch) and
//    token_piece_oai_fields ("bytes" field).
// ---------------------------------------------------------------------------

// Converts every byte of `str` to its integer value and returns them as a
// JSON array.  The raw bytes are preserved exactly — no UTF-8 truncation.
static json str_to_bytes(const std::string &str) {
    json bytes = json::array();
    bytes.get_ref<json::array_t &>().reserve(str.size());
    for (unsigned char c : str) {
        bytes.push_back(static_cast<int>(c));
    }
    return bytes;
}

// Returns the JSON value for the "piece" key in a llama.cpp /tokenize
// response.  Valid UTF-8 pieces become a JSON string; invalid ones become a
// JSON array of byte values (via str_to_bytes).
//
// NEVER use this for completion probability responses — use
// token_piece_oai_fields() instead, which always emits both "token" and
// "bytes" per the OpenAI spec.
static json token_piece_value(const std::string &piece) {
    if (is_valid_utf8(piece)) {
        return piece;
    }
    return str_to_bytes(piece);
}

// Returns a partial JSON object {"token": <truncated-utf8>, "bytes": <raw>}
// for use in OpenAI-compatible completion probability responses.
// "token" is always a string (piece truncated at the last valid UTF-8
// boundary).  "bytes" is always the full raw-byte array via str_to_bytes.
//
// NEVER use this for /tokenize responses — use token_piece_value() instead,
// which follows the llama.cpp native "piece" field schema.
static json token_piece_oai_fields(const std::string &piece) {
    std::string txt = piece;
    txt.resize(validate_utf8(txt));
    return json{{"token", txt}, {"bytes", str_to_bytes(piece)}};
}

//
// template utils
//

// format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static llama_tokens format_rerank(const struct llama_vocab *vocab, const llama_tokens &query, const llama_tokens &doc) {
    llama_tokens result;

    // Get EOS token - use SEP token as fallback if EOS is not available
    llama_token eos_token = llama_vocab_eos(vocab);
    if (eos_token == LLAMA_TOKEN_NULL) {
        eos_token = llama_vocab_sep(vocab);
    }

    result.reserve(doc.size() + query.size() + 4);
    result.push_back(llama_vocab_bos(vocab));
    result.insert(result.end(), query.begin(), query.end());
    result.push_back(eos_token);
    result.push_back(llama_vocab_sep(vocab));
    result.insert(result.end(), doc.begin(), doc.end());
    result.push_back(eos_token);

    return result;
}

// format infill task
static llama_tokens format_infill(const llama_vocab *vocab, const json &input_prefix, const json &input_suffix,
                                  const json &input_extra, const int n_batch, const int n_predict, const int n_ctx,
                                  const bool spm_infill, const llama_tokens &tokens_prompt) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto &chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text = json_value(chunk, "text", std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                                      0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(), 3 * (n_batch / 4));
    const int n_suffix_take =
        std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch / 4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take,
            (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch)-2 * n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(), tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int)extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

// clang-format off
// ---- BEGIN COPY FROM llama.cpp tools/server/server-common.cpp ---------------
// base64_chars / is_base64 / base64_decode are declared `static` in
// server-common.cpp (internal linkage). Even though server-common.cpp is
// compiled into the same shared library, C++ static linkage makes the symbols
// invisible to every other translation unit — there is no declaration in
// server-common.h to call through. These copies are therefore unavoidable and
// must be kept in sync manually whenever llama.cpp upgrades server-common.cpp.
// Removing them is only possible if upstream moves them to a header as `inline`.
static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static inline bool is_base64(uint8_t c) { return (isalnum(c) || (c == '+') || (c == '/')); }

static inline raw_buffer base64_decode(const std::string &encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;
    int in_len = encoded_string.size();
    uint8_t char_array_4[4];
    uint8_t char_array_3[3];
    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_++];
        if (i == 4) {
            for (i = 0; i < 4; i++) char_array_4[i] = base64_chars.find(char_array_4[i]);
            char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; i < 3; i++) ret.push_back(char_array_3[i]);
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 4; j++) char_array_4[j] = 0;
        for (j = 0; j < 4; j++) char_array_4[j] = base64_chars.find(char_array_4[j]);
        char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        for (j = 0; j < i - 1; j++) ret.push_back(char_array_3[j]);
    }
    return ret;
}
// ---- END COPY FROM llama.cpp tools/server/server-common.cpp -----------------
// clang-format on

// Strip an exact-match flag (no value) from an argv array.
// Returns a new vector of pointers (non-owning) with every occurrence removed.
// Sets *found = true if the flag was present at least once.
static std::vector<char *> strip_flag_from_argv(char **argv, int argc, const char *flag, bool *found) {
    *found = false;
    std::vector<char *> out;
    out.reserve(argc);
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], flag) == 0) {
            *found = true;
        } else {
            out.push_back(argv[i]);
        }
    }
    return out;
}

static json format_tokenizer_response(const json &tokens) { return json{{"tokens", tokens}}; }

static json format_detokenized_response(const std::string &content) { return json{{"content", content}}; }

static json format_logit_bias(const std::vector<llama_logit_bias> &logit_bias) {
    json data = json::array();
    for (const auto &lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

// parse lora config from JSON request, returned a copy of lora_base with updated scale
static std::vector<common_adapter_lora_info> parse_lora_request(const std::vector<common_adapter_lora_info> &lora_base,
                                                                const json &data) {
    std::vector<common_adapter_lora_info> lora(lora_base);
    int max_idx = lora.size();

    // clear existing value
    for (auto &entry : lora) {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto &entry : data) {
        int id = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx) {
            lora[id].scale = scale;
        } else {
            throw std::runtime_error("invalid adapter id");
        }
    }

    return lora;
}
