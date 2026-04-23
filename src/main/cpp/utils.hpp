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

//
// tokenizer and input processing utils
//

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56]]
 *
 * Overload without mtmd_context: returns plain llama_tokens instead of server_tokens.
 * Used by jni_helpers.hpp for non-multimodal inference paths.
 */
static std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab *vocab, const json &json_prompt,
                                                        bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_mixed(vocab, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto &p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else {
                throw std::runtime_error(
                    "element of \"prompt\" must be a string, an list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error(
            "\"prompt\" must be a string, an list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

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

//
// other common utils
//

// Iterator-range overloads of tokens_to_str — upstream server-common.cpp provides
// const-ref (llama_tokens) versions; these template variants take Iter begin/end
// and are used by server.hpp completion-output paths.
template <class Iter> static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }
    return ret;
}

template <class Iter> static std::string tokens_to_str(const llama_vocab *vocab, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(vocab, *begin);
    }
    return ret;
}

//
// OAI utils
//

struct oaicompat_parser_options {
    bool use_jinja = false;
    bool prefill_assistant = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    common_chat_templates *tmpls = nullptr;
    bool allow_image = false;
    bool allow_audio = false;
    bool enable_thinking = false;
};

// used by /chat/completions endpoint
static json oaicompat_chat_params_parse(json &body, /* openai api json semantics */
                                        const oaicompat_parser_options &opt, std::vector<raw_buffer> &out_files) {
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " +
                                     response_type);
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    json &messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("Expected 'messages' to be an array");
    }
    for (auto &msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::runtime_error("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::runtime_error("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json &content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::runtime_error("Expected 'content' to be a string or an array");
        }

        for (auto &p : content) {
            std::string type = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need "
                                             "to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                std::string url = json_value(image_url, "url", std::string());
                if (string_starts_with(url, "http")) {
                    // download remote image
                    // TODO @ngxson : maybe make these params configurable
                    common_remote_params params;
                    params.headers.push_back({"User-Agent", "llama.cpp/" + std::string(llama_build_info())});
                    params.max_size = 1024 * 1024 * 10; // 10MB
                    params.timeout = 10;                // seconds
                    SRV_INF("downloading image from '%s'\n", url.c_str());
                    auto res = common_remote_get_content(url, params);
                    if (200 <= res.first && res.first < 300) {
                        SRV_INF("downloaded %ld bytes\n", res.second.size());
                        raw_buffer data;
                        data.insert(data.end(), res.second.begin(), res.second.end());
                        out_files.push_back(data);
                    } else {
                        throw std::runtime_error("Failed to download image");
                    }

                } else {
                    // try to decode base64 image
                    std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
                    if (parts.size() != 2) {
                        throw std::runtime_error("Invalid image_url.url value");
                    } else if (!string_starts_with(parts[0], "data:image/")) {
                        throw std::runtime_error("Invalid image_url.url format: " + parts[0]);
                    } else if (!string_ends_with(parts[0], "base64")) {
                        throw std::runtime_error("image_url.url must be base64 encoded");
                    } else {
                        auto base64_data = parts[1];
                        auto decoded_data = base64_decode(base64_data);
                        out_files.push_back(decoded_data);
                    }
                }

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("image_url");

            } else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need "
                                             "to provide the mmproj");
                }

                json input_audio = json_value(p, "input_audio", json::object());
                std::string data = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                // while we also support flac, we don't allow it here so we matches the OAI spec
                if (format != "wav" && format != "mp3") {
                    throw std::runtime_error("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data); // expected to be base64 encoded
                out_files.push_back(decoded_data);

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("input_audio");

            } else if (type != "text") {
                throw std::runtime_error("unsupported content[].type");
            }
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar = grammar;
    inputs.use_jinja = opt.use_jinja;
    inputs.parallel_tool_calls = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format = opt.reasoning_format;
    inputs.enable_thinking = opt.enable_thinking;
    // Extract custom template kwargs from request body (JSON object with string values).
    // Values are stored as JSON-serialized strings because upstream does json::parse(value).
    if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
        for (auto &el : body.at("chat_template_kwargs").items()) {
            inputs.chat_template_kwargs[el.key()] = el.value().dump();
        }
    }
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::runtime_error("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // if the assistant message appears at the end of list, we do not add end-of-turn token
    // for ex. this can be useful to modify the reasoning process in reasoning models
    bool prefill_assistant_message =
        !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        /* sanity check, max one assistant message at the end of the list */
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            throw std::runtime_error("Cannot have 2 or more assistant messages at the end of the list.");
        }

        /* TODO: test this properly */
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        inputs.add_generation_prompt = true;
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        chat_params.prompt += last_message.content;
    }

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"] = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto &trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    llama_params["generation_prompt"] = chat_params.generation_prompt;
    for (const auto &stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may
    // need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::runtime_error("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto &item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json format_embeddings_response_oaicompat(const json &request, const json &embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto &elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto &vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char *data_ptr = reinterpret_cast<const char *>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {{"embedding", base64::encode(data_ptr, data_size)},
                             {"index", i++},
                             {"object", "embedding"},
                             {"encoding_format", "base64"}};
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())}, {"index", i++}, {"object", "embedding"}};
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json{{"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
                    {"object", "list"},
                    {"usage", json{{"prompt_tokens", n_tokens}, {"total_tokens", n_tokens}}},
                    {"data", data}};

    return res;
}

static json format_response_rerank(const json &request, const json &ranks, bool is_tei_format,
                                   std::vector<std::string> &texts, int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements;
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto &rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json &a, const json &b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    elements.resize(std::min(top_n, (int) elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{{"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
                   {"object", "list"},
                   {"usage", json{{"prompt_tokens", n_tokens}, {"total_tokens", n_tokens}}},
                   {"results", results}};

    return res;
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
