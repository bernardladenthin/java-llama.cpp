#pragma once

// server-common.h provides: JSON_ASSERT, json, raw_buffer, json_value<T>,
// server_grammar_trigger, server_tokens, error_type, SRV_* macros,
// and many utility function declarations (implemented in server-common.cpp).
#include "server-common.h"

#include <string>
#include <vector>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"

// ---------------------------------------------------------------------------
// Token-piece JSON serialisation helpers
//
// There are two distinct wire formats for representing a token piece that may
// not be valid UTF-8, used in different parts of the API.  The helpers below
// implement each format exactly once and are documented so the two are never
// accidentally conflated.
//
// 1. token_piece_value()  — llama.cpp /tokenize endpoint (native format)
//    Schema: a single JSON value that is EITHER a string (valid UTF-8) OR a
//    byte-integer array (invalid UTF-8).
//    Used by: handleTokenize at jllama.cpp:1165.
//
// 2. str_to_bytes() — converts every byte of a string to an int in a JSON
//    array; used by token_piece_value for the invalid-UTF-8 branch.
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
static json token_piece_value(const std::string &piece) {
    if (is_valid_utf8(piece)) {
        return piece;
    }
    return str_to_bytes(piece);
}

//
// template utils
//

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
