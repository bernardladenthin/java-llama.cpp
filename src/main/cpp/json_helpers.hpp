#pragma once

// json_helpers.hpp — Pure JSON transformation helpers.
//
// Every function in this file is pure data transformation:
//   - input:  nlohmann::json values, server_task_result_ptr, or plain C++ types
//   - output: nlohmann::json, std::vector, std::optional, or plain C++ types
//   - zero JNI calls (no JNIEnv*, jclass, jstring, …)
//   - zero llama state (no llama_context*, llama_vocab*, server_context*)
//
// All functions are unit-testable with JSON literals and fake result objects;
// no JVM and no loaded model are required.
//
// IMPORTANT — include order:
//   Upstream server headers (server-context.h, server-queue.h, server-task.h,
//   server-common.h, server-chat.h) and utils.hpp must be included by the
//   including translation unit BEFORE this header.  Those headers define:
//     server_task_result_ptr, task_response_type, TASK_RESPONSE_TYPE_OAI_EMBD,
//     format_embeddings_response_oaicompat, and the `json` type alias.
//
// Declaration order:
//   1.  get_result_error_message        — used by nothing above it
//   2.  results_to_json                 — used by nothing above it
//   3.  rerank_results_to_json          — used by nothing above it
//   4.  build_embeddings_response_json  — used by nothing above it
//   5.  parse_encoding_format           — used by nothing above it
//   6.  extract_embedding_prompt        — used by nothing above it
//   7.  is_infill_request               — used by nothing above it
//   8.  parse_slot_prompt_similarity    — used by nothing above it
//   9.  parse_positive_int_config       — used by nothing above it

#include "nlohmann/json.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// get_result_error_message
//
// Extracts the human-readable error string from a failed task result.
// Equivalent to result->to_json()["message"].get<std::string>().
//
// Used by recv_slot_task_result_impl and collect_task_results_impl in
// jni_helpers.hpp, and directly in receiveCompletionJson, embed, and
// handleRerank in jllama.cpp.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string get_result_error_message(
        const server_task_result_ptr &result) {
    return result->to_json()["message"].get<std::string>();
}

// ---------------------------------------------------------------------------
// results_to_json
//
// Converts a vector of task results to a single json value.
//
// One result  → the result's JSON object directly (no wrapping array).
// Many results → a JSON array of each result's JSON object.
// Empty vector → empty JSON array.
//
// This mirrors the OpenAI API convention used by handleCompletions,
// handleCompletionsOai, handleChatCompletions, and handleInfill.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json results_to_json(
        const std::vector<server_task_result_ptr> &results) {
    if (results.size() == 1) {
        return results[0]->to_json();
    }
    json arr = json::array();
    for (const auto &res : results) {
        arr.push_back(res->to_json());
    }
    return arr;
}

// ---------------------------------------------------------------------------
// rerank_results_to_json
//
// Converts a collected vector of rerank task results to a JSON array.
// Each element contains the original document text (looked up via the
// result's "index" field), the index, and the relevance score.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json rerank_results_to_json(
        const std::vector<server_task_result_ptr> &results,
        const std::vector<std::string>            &documents) {
    json arr = json::array();
    for (const auto &result : results) {
        const auto out = result->to_json();
        int   index = out["index"].get<int>();
        float score = out["score"].get<float>();
        arr.push_back({
            {"document", documents[index]},
            {"index",    index},
            {"score",    score}
        });
    }
    return arr;
}

// ---------------------------------------------------------------------------
// build_embeddings_response_json
//
// Collects task results into a JSON array, then formats the final response:
//   - TASK_RESPONSE_TYPE_OAI_EMBD → wraps via format_embeddings_response_oaicompat
//     (adds "object":"list", "usage", and per-embedding "object":"embedding")
//   - any other res_type          → returns the bare JSON array
//
// Symmetric counterpart to rerank_results_to_json.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json build_embeddings_response_json(
        const std::vector<server_task_result_ptr> &results,
        const json                                &body,
        task_response_type                         res_type,
        bool                                       use_base64) {
    json responses = json::array();
    for (const auto &result : results) {
        responses.push_back(result->to_json());
    }
    if (res_type == TASK_RESPONSE_TYPE_OAI_EMBD) {
        return format_embeddings_response_oaicompat(body, json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL)), responses, use_base64);
    }
    return responses;
}

// ---------------------------------------------------------------------------
// parse_encoding_format
//
// Reads the optional "encoding_format" field from `body`.
//
// Returns false  — field absent, or value is "float"  → use float encoding.
// Returns true   — value is "base64"                  → use base64 encoding.
// Throws std::invalid_argument — value is present but neither "float" nor
//   "base64", with a message suitable for forwarding to JNI ThrowNew.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool parse_encoding_format(const json &body) {
    if (!body.contains("encoding_format")) {
        return false;
    }
    const std::string format = body.at("encoding_format").get<std::string>();
    if (format == "base64") { return true; }
    if (format == "float")  { return false; }
    throw std::invalid_argument("encoding_format must be \"float\" or \"base64\"");
}

// ---------------------------------------------------------------------------
// extract_embedding_prompt
//
// Selects the prompt value from an embedding request body using OAI-style
// key precedence: "input" is preferred (OAI path); "content" is the fallback
// (legacy non-OAI path).
//
// On success: returns the prompt JSON value.  Sets force_no_oaicompat=true
//   when "content" was used — the caller must downgrade oaicompat to NONE.
// Throws std::invalid_argument if neither "input" nor "content" is present.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json extract_embedding_prompt(const json &body,
                                                    bool       &force_no_oaicompat) {
    force_no_oaicompat = false;
    if (body.count("input") != 0) {
        return body.at("input");
    }
    if (body.contains("content")) {
        force_no_oaicompat = true;
        return body.at("content");
    }
    throw std::invalid_argument("\"input\" or \"content\" must be provided");
}

// ---------------------------------------------------------------------------
// is_infill_request
//
// Returns true if the request data contains "input_prefix" or "input_suffix",
// indicating that the caller wants fill-in-the-middle (infill) inference
// rather than plain completion.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool is_infill_request(const json &data) {
    return data.contains("input_prefix") || data.contains("input_suffix");
}

// ---------------------------------------------------------------------------
// parse_slot_prompt_similarity
//
// Reads the optional "slot_prompt_similarity" field from `config`.
//
// Returns empty optional — field absent, no change needed.
// Returns float          — validated value in [0.0, 1.0].
// Throws std::invalid_argument — present but outside [0.0, 1.0].
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::optional<float>
parse_slot_prompt_similarity(const json &config) {
    if (!config.contains("slot_prompt_similarity")) {
        return std::nullopt;
    }
    const float v = config["slot_prompt_similarity"].get<float>();
    if (v < 0.0f || v > 1.0f) {
        throw std::invalid_argument("slot_prompt_similarity must be between 0.0 and 1.0");
    }
    return v;
}

// ---------------------------------------------------------------------------
// parse_positive_int_config
//
// Reads an optional integer field `key` from `config` and validates it is > 0.
//
// Returns empty optional — field absent, no change needed.
// Returns int            — validated value > 0.
// Throws std::invalid_argument("<key> must be greater than 0") — present but ≤ 0.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::optional<int>
parse_positive_int_config(const json &config, const char *key) {
    if (!config.contains(key)) {
        return std::nullopt;
    }
    const int v = config[key].get<int>();
    if (v <= 0) {
        throw std::invalid_argument(std::string(key) + " must be greater than 0");
    }
    return v;
}
