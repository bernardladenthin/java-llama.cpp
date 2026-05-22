// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

#pragma once

// log_helpers.hpp — Pure log-formatting helpers.
//
// No JNI, no llama state. Depends only on the ggml_log_level enum (declared
// in ggml.h) and nlohmann/json. Unit-testable without a JVM or model.

#include "ggml.h"
#include "nlohmann/json.hpp"

#include <ctime>
#include <string>

// Returns the canonical short name for a ggml log level. INFO is the default
// fall-through to mirror llama.cpp's own log routing.
[[nodiscard]] inline const char *log_level_name(ggml_log_level level) {
    switch (level) {
    case GGML_LOG_LEVEL_ERROR: return "ERROR";
    case GGML_LOG_LEVEL_WARN:  return "WARN";
    case GGML_LOG_LEVEL_DEBUG: return "DEBUG";
    case GGML_LOG_LEVEL_INFO:
    default:                   return "INFO";
    }
}

// Pure variant taking an explicit timestamp so tests are deterministic.
[[nodiscard]] inline std::string format_log_as_json(
        ggml_log_level level, const char *text, std::time_t timestamp) {
    nlohmann::json log_obj = {
        {"timestamp", timestamp},
        {"level",     log_level_name(level)},
        {"message",   text ? text : ""},
    };
    return log_obj.dump();
}
