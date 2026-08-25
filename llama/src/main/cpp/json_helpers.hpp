// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

#pragma once

// json_helpers.hpp — Pure JSON transformation helpers.
//
// Every function in this file is pure data transformation:
//   - input:  `json` values (the upstream alias — `common_json` since llama.cpp b10585,
//             `nlohmann::ordered_json` before it), server_task_result_ptr, or plain C++ types
//   - output: `json`, std::vector, std::optional, or plain C++ types
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
//   4.  parse_encoding_format           — used by nothing above it
//   5.  extract_embedding_prompt        — used by nothing above it
//   6.  is_infill_request               — used by nothing above it
//   7.  parse_slot_prompt_similarity    — used by nothing above it
//   8.  parse_positive_int_config       — used by nothing above it
//   9.  wrap_stream_chunk               — used by nothing above it
//  10.  server_metrics_to_json          — used by nothing above it

#include <cmath>
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
[[nodiscard]] inline std::string get_result_error_message(const server_task_result_ptr &result) {
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
[[nodiscard]] inline json results_to_json(const std::vector<server_task_result_ptr> &results) {
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
[[nodiscard]] inline json rerank_results_to_json(const std::vector<server_task_result_ptr> &results,
                                                 const std::vector<std::string> &documents) {
    json arr = json::array();
    for (const auto &result : results) {
        const auto out = result->to_json();
        // Defensive: a malformed/absent "index" or an out-of-range value would otherwise
        // throw json::type_error or index documents[] out of bounds.
        if (!out.contains("index")) {
            throw std::invalid_argument("rerank result is missing the 'index' field");
        }
        const int index = out["index"].get<int>();
        if (index < 0 || static_cast<size_t>(index) >= documents.size()) {
            throw std::invalid_argument("rerank result index " + std::to_string(index) + " out of range");
        }
        float score = out["score"].get<float>();
        arr.push_back({{"document", documents[index]}, {"index", index}, {"score", score}});
    }
    return arr;
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
    if (format == "base64") {
        return true;
    }
    if (format == "float") {
        return false;
    }
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
[[nodiscard]] inline json extract_embedding_prompt(const json &body, bool &force_no_oaicompat) {
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
[[nodiscard]] inline std::optional<float> parse_slot_prompt_similarity(const json &config) {
    if (!config.contains("slot_prompt_similarity")) {
        return std::nullopt;
    }
    // Coerce via double so an integer-valued config (e.g. 0 or 1) is accepted rather than
    // throwing json::type_error; the range check below preserves the [0.0, 1.0] contract.
    const double v = config["slot_prompt_similarity"].get<double>();
    if (v < 0.0 || v > 1.0) {
        throw std::invalid_argument("slot_prompt_similarity must be between 0.0 and 1.0");
    }
    return static_cast<float>(v);
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
[[nodiscard]] inline std::optional<int> parse_positive_int_config(const json &config, const char *key) {
    if (!config.contains(key)) {
        return std::nullopt;
    }
    // Coerce via double so an integer-valued config is accepted rather than throwing
    // json::type_error; require a positive integer per the documented contract.
    const double raw = config[key].get<double>();
    // Reject non-positive, non-integer, or > INT_MAX values: a whole-number JSON value above
    // INT_MAX would overflow static_cast<int> (UB).
    if (raw <= 0.0 || raw != std::floor(raw) || raw > 2147483647.0) {
        throw std::invalid_argument(std::string(key) + " must be a positive integer");
    }
    return static_cast<int>(raw);
}

// ---------------------------------------------------------------------------
// wrap_stream_chunk
//
// Wraps one streaming chat result payload together with its stop flag into a
// single transport object so the Java side has a uniform shape to parse:
//
//   {"data": <payload>, "stop": <bool>}
//
// `payload` is whatever a streaming OAI chat result's to_json() produced — a
// single chat.completion.chunk object for a partial token, or a JSON array of
// chunk objects for the final result (final delta + optional usage chunk).
// The Java consumer reads "stop" and emits each element of "data" as its own
// SSE `data:` event.  Used by receiveChatCompletionChunk in jllama.cpp.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json wrap_stream_chunk(json payload, bool stop) {
    json out;
    out["data"] = std::move(payload);
    out["stop"] = stop;
    return out;
}

// ---------------------------------------------------------------------------
// server_metrics_to_json
//
// Merges the two halves of the server-introspection payload back into the one
// object LlamaModel.getMetrics() has always returned.
//
// Until llama.cpp b10408 (upstream #26920) a single SERVER_TASK_TYPE_METRICS
// task produced that object.  b10408 reduced its to_json() to the slot array,
// and b10519 (#27376) split the task in two: METRICS now carries only the
// counters (rendered as Prometheus text by to_metrics(); its to_json() is
// unused and returns JSON null) and SERVER_TASK_TYPE_SLOT_GET carries the slot
// array plus the idle-slot count.  Rather than let the Java contract follow
// upstream's transport split, the JNI layer posts both tasks and rebuilds the
// object here.
//
// Key names are the pre-b10408 ones — `idle`, `processing`, `deferred`,
// `t_start`, the `n_*`/`t_*` counter pairs, the speculative-decoding tallies and
// `slots` — so every existing consumer keeps working.  The speculative counters
// deliberately keep upstream's own historical spelling (`n_draft_verif_steps_total`,
// `n_accepted_per_pos_total`), not a tidied-up one, so the payload stays a faithful
// reproduction.  Only `n_prompt_tokens_cached_total` is genuinely new: `n_prompt_cached`
// did not exist in the struct before b10408 and has no Prometheus counterpart either.
//
// Durations are microseconds upstream and milliseconds here, matching what the
// pre-b10408 payload used.
// ---------------------------------------------------------------------------
[[nodiscard]] inline json server_metrics_to_json(const server_task_result_metrics &metrics_result,
                                                 const server_task_result_slots &slots_result) {
    const server_metrics &m = metrics_result.metrics;

    // Microseconds -> milliseconds, as a double so sub-millisecond timings survive.
    const auto to_ms = [](uint64_t time_us) { return static_cast<double>(time_us) / 1000.0; };

    json out;
    out["idle"] = slots_result.n_idle_slots;
    out["processing"] = metrics_result.n_processing_slots;
    out["deferred"] = metrics_result.n_tasks_deferred;
    out["t_start"] = m.t_start;

    // Cumulative since server start.
    out["n_prompt_tokens_processed_total"] = m.prompt.count;
    out["t_prompt_processing_total"] = to_ms(m.prompt.time);
    out["n_tokens_predicted_total"] = m.predict.count;
    out["t_tokens_generation_total"] = to_ms(m.predict.time);
    out["n_decode_total"] = m.n_decode;
    out["n_busy_slots_total"] = m.n_busy_slots;
    out["n_tokens_max"] = m.n_tokens_max;

    // Current measurement window: reset by an HTTP /metrics scrape, never by this call
    // (the JNI task leaves server_task::metrics_reset_bucket at its default false).
    out["n_prompt_tokens_processed"] = m.prompt_bucket.count;
    out["t_prompt_processing"] = to_ms(m.prompt_bucket.time);
    out["n_tokens_predicted"] = m.predict_bucket.count;
    out["t_tokens_generation"] = to_ms(m.predict_bucket.time);

    // Cache counter (new since b10408) plus the speculative-decoding tallies, which the
    // pre-b10408 payload already carried under exactly these names.
    out["n_prompt_tokens_cached_total"] = m.n_prompt_cached;
    out["n_draft_tokens_total"] = m.n_draft_tokens;
    out["n_draft_accepted_total"] = m.n_draft_accepted;
    out["n_draft_verif_steps_total"] = m.n_draft_verif_steps;
    out["n_accepted_per_pos_total"] = m.n_accepted_per_pos;

    out["slots"] = slots_result.slots_data;
    return out;
}
