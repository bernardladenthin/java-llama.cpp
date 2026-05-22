// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

// Tests for log_helpers.hpp — pure log-formatting helpers.

#include <gtest/gtest.h>

#include "log_helpers.hpp"
#include "nlohmann/json.hpp"

#include <ctime>
#include <string>

using json = nlohmann::json;

// ============================================================
// log_level_name
// ============================================================

TEST(LogLevelName, Error) {
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_ERROR), "ERROR");
}

TEST(LogLevelName, Warn) {
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_WARN), "WARN");
}

TEST(LogLevelName, Info) {
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_INFO), "INFO");
}

TEST(LogLevelName, Debug) {
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_DEBUG), "DEBUG");
}

TEST(LogLevelName, NoneFallsBackToInfo) {
    // GGML_LOG_LEVEL_NONE is not explicitly mapped; the default arm returns INFO
    // so callers always get a usable label.
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_NONE), "INFO");
}

TEST(LogLevelName, ContFallsBackToInfo) {
    EXPECT_STREQ(log_level_name(GGML_LOG_LEVEL_CONT), "INFO");
}

// ============================================================
// format_log_as_json
// ============================================================

TEST(FormatLogAsJson, BasicShape) {
    const std::string out = format_log_as_json(GGML_LOG_LEVEL_INFO, "hello", 1700000000);
    const json j = json::parse(out);
    EXPECT_EQ(j.at("level").get<std::string>(),   "INFO");
    EXPECT_EQ(j.at("message").get<std::string>(), "hello");
    EXPECT_EQ(j.at("timestamp").get<std::int64_t>(), 1700000000);
}

TEST(FormatLogAsJson, ErrorLevel) {
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_ERROR, "boom", 42));
    EXPECT_EQ(j.at("level").get<std::string>(), "ERROR");
    EXPECT_EQ(j.at("message").get<std::string>(), "boom");
}

TEST(FormatLogAsJson, WarnLevel) {
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_WARN, "careful", 0));
    EXPECT_EQ(j.at("level").get<std::string>(), "WARN");
}

TEST(FormatLogAsJson, DebugLevel) {
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_DEBUG, "trace", 0));
    EXPECT_EQ(j.at("level").get<std::string>(), "DEBUG");
}

TEST(FormatLogAsJson, NullTextBecomesEmptyString) {
    // The original implementation passed text directly to nlohmann::json,
    // which crashes on nullptr. The new helper normalises null to "".
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_INFO, nullptr, 7));
    EXPECT_EQ(j.at("message").get<std::string>(), "");
}

TEST(FormatLogAsJson, EmbedsExplicitTimestamp) {
    const std::time_t t = 1234567890;
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_INFO, "x", t));
    EXPECT_EQ(j.at("timestamp").get<std::int64_t>(), t);
}

TEST(FormatLogAsJson, KeysExactlyThree) {
    const json j = json::parse(format_log_as_json(GGML_LOG_LEVEL_INFO, "x", 0));
    EXPECT_EQ(j.size(), 3u);
    EXPECT_TRUE(j.contains("timestamp"));
    EXPECT_TRUE(j.contains("level"));
    EXPECT_TRUE(j.contains("message"));
}
