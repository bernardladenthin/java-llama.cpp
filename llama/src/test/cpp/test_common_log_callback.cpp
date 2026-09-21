// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Runnable guard for patches/0014-common-log-callback-sink.patch: the common_log_set_callback()
// sink that lets LlamaModel.setLogger receive the server's SRV_*/SLT_* lines (which never pass
// through llama_log_set). Drives a private common_log instance, never common_log_main(), so the
// process-wide logger the other tests print through is untouched. A llama.cpp bump that drops the
// patch fails this file at compile time on every platform instead of silently muting Java logging.

#include <gtest/gtest.h>

#include "log.h"

#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using entries = std::vector<std::pair<ggml_log_level, std::string>>;

// The sink is a plain function pointer, so the capture goes through user_data. The worker thread
// writes and the test thread reads only after common_log_flush()/common_log_free() joined it, but
// the mutex keeps the recorder honest if a future test reads while the worker is live.
struct recorder {
    std::mutex mtx;
    entries got;
};

void record(ggml_log_level level, const char *text, void *user_data) {
    auto *rec = static_cast<recorder *>(user_data);
    std::lock_guard<std::mutex> lk(rec->mtx);
    rec->got.emplace_back(level, text);
}

entries snapshot(recorder &rec) {
    std::lock_guard<std::mutex> lk(rec.mtx);
    return rec.got;
}

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string temp_log_path(const char *tag) {
    return (std::string(testing::TempDir()) + "jllama-common-log-" + tag + ".log");
}

} // namespace

TEST(CommonLogCallback, CallbackReceivesFormattedMessageAndLevel) {
    recorder rec;
    common_log *log = common_log_init();
    common_log_set_callback(log, record, &rec);

    common_log_add(log, GGML_LOG_LEVEL_INFO, "hello %d\n", 42);
    common_log_add(log, GGML_LOG_LEVEL_ERROR, "boom\n");
    common_log_flush(log);

    const entries got = snapshot(rec);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0].first, GGML_LOG_LEVEL_INFO);
    EXPECT_EQ(got[0].second, "hello 42\n");
    EXPECT_EQ(got[1].first, GGML_LOG_LEVEL_ERROR);
    EXPECT_EQ(got[1].second, "boom\n");

    common_log_free(log);
}

TEST(CommonLogCallback, TextCarriesNoPrefixOrTimestampEvenWhenEnabled) {
    // common_init() turns both on for every model load; the sink must still get the bare message,
    // because the Java side formats (or JSON-wraps) it itself.
    recorder rec;
    common_log *log = common_log_init();
    common_log_set_prefix(log, true);
    common_log_set_timestamps(log, true);
    common_log_set_callback(log, record, &rec);

    common_log_add(log, GGML_LOG_LEVEL_WARN, "plain\n");
    common_log_flush(log);

    const entries got = snapshot(rec);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].second, "plain\n");

    common_log_free(log);
}

TEST(CommonLogCallback, ClearingTheCallbackStopsDelivery) {
    recorder rec;
    common_log *log = common_log_init();
    common_log_set_callback(log, record, &rec);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "seen\n");

    common_log_set_callback(log, nullptr, nullptr);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "unseen\n");
    common_log_flush(log);

    const entries got = snapshot(rec);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].second, "seen\n");

    common_log_free(log);
}

TEST(CommonLogCallback, EntriesQueuedBeforeASwapReachThePreviousSink) {
    // This is what makes LlamaModel.setLogger(format, null) a synchronous drain: pause() flushes
    // through the old sink before the new one is installed.
    recorder first;
    recorder second;
    common_log *log = common_log_init();
    common_log_set_callback(log, record, &first);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "before\n");

    common_log_set_callback(log, record, &second);
    common_log_add(log, GGML_LOG_LEVEL_INFO, "after\n");
    common_log_flush(log);

    const entries got_first = snapshot(first);
    const entries got_second = snapshot(second);
    ASSERT_EQ(got_first.size(), 1u);
    EXPECT_EQ(got_first[0].second, "before\n");
    ASSERT_EQ(got_second.size(), 1u);
    EXPECT_EQ(got_second[0].second, "after\n");

    common_log_free(log);
}

TEST(CommonLogCallback, FileOutputIsKeptWhileTheCallbackIsSet) {
    // The sink replaces the console only; --log-file keeps working alongside a Java logger.
    const std::string path = temp_log_path("file-kept");
    recorder rec;
    common_log *log = common_log_init();
    common_log_set_file(log, path.c_str());
    common_log_set_callback(log, record, &rec);

    common_log_add(log, GGML_LOG_LEVEL_INFO, "to both\n");
    common_log_free(log); // joins the worker and closes the file

    const entries got = snapshot(rec);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].second, "to both\n");
    EXPECT_NE(read_file(path).find("to both"), std::string::npos);
    std::remove(path.c_str());
}

TEST(CommonLogCallback, LevelsPassThroughUnchanged) {
    recorder rec;
    common_log *log = common_log_init();
    common_log_set_callback(log, record, &rec);

    common_log_add(log, GGML_LOG_LEVEL_DEBUG, "d\n");
    common_log_add(log, GGML_LOG_LEVEL_NONE, "o\n");
    common_log_add(log, GGML_LOG_LEVEL_CONT, "c\n");
    common_log_flush(log);

    const entries got = snapshot(rec);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0].first, GGML_LOG_LEVEL_DEBUG);
    EXPECT_EQ(got[1].first, GGML_LOG_LEVEL_NONE);
    EXPECT_EQ(got[2].first, GGML_LOG_LEVEL_CONT);

    common_log_free(log);
}
