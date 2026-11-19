// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// jllama_configure() across the idle sleep. A model loaded with --sleep-idle-seconds frees its
// context when it falls asleep, and the wake builds a new one from the load-time parameters -- the
// threads of llama_set_n_threads() and the slot scheduler's similarity (load_model() copies it from
// params_base) were the load's again, silently. The engine applies what configure() set after every
// wake (wake_server). Neither value is part of an answer, so this file looks at the context and the
// server the handle holds, which needs the engine's own headers (and therefore a file apart from
// test_c_api.cpp, whose json is nlohmann's, not the server's).

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"

#include "jllama_api.h"
#include "jllama_context.hpp"
#include "jllama_engine.hpp"
#include "tiny_llama_gguf.hpp"

#include "llama.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct SleepLines {
    std::mutex mutex;
    int sleeps = 0;
    int count() {
        std::lock_guard<std::mutex> lock(mutex);
        return sleeps;
    }
};

struct Tuning {
    int threads = 0;
    int threads_batch = 0;
    float similarity = -1.0f;
};

// The threads of the context and the scheduler's similarity, read while the call that woke the
// server still holds it awake -- as every C API call does -- so the context cannot sleep under it.
Tuning tuning_after_a_wake(jllama_model *model) {
    auto *jctx = reinterpret_cast<jllama_context *>(model);
    jllama_context_guard guard{acquire_jllama_context(jctx)};
    jllama::engine::wake_server(jctx);
    Tuning t;
    llama_context *lctx = jctx->server.get_llama_context();
    if (lctx != nullptr) {
        t.threads = llama_n_threads(lctx);
        t.threads_batch = llama_n_threads_batch(lctx);
    }
    t.similarity = jctx->server.get_slot_prompt_similarity();
    return t;
}

} // namespace

TEST(EngineTuning, WhatConfigureSetHoldsAfterTheIdleSleep) {
    const std::string path = jllama_test::write_tiny_llama_gguf("jllama-engine-tuning");
    ASSERT_FALSE(path.empty());
    SleepLines lines;
    ASSERT_EQ(jllama_set_log_callback(
                  [](int, const char *text, void *user) {
                      auto *sink = static_cast<SleepLines *>(user);
                      std::lock_guard<std::mutex> lock(sink->mutex);
                      sink->sleeps += std::string(text).find("server is entering sleeping state") != std::string::npos;
                  },
                  &lines),
              JLLAMA_STATUS_OK);
    struct LogCallbackReset {
        ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); }
    } log_callback_reset;
    const std::vector<std::string> args = {"jllama",
                                           "-m",
                                           path,
                                           "-c",
                                           "64",
                                           "-ngl",
                                           "0",
                                           "--no-warmup",
                                           "-t",
                                           "2",
                                           "-tb",
                                           "2",
                                           "--slot-prompt-similarity",
                                           "0.5",
                                           "--sleep-idle-seconds",
                                           "1"};
    std::vector<const char *> argv;
    for (const auto &a : args) {
        argv.push_back(a.c_str());
    }
    jllama_model *model = nullptr;
    ASSERT_EQ(jllama_model_load(static_cast<int>(argv.size()), argv.data(), nullptr, nullptr, &model), JLLAMA_STATUS_OK)
        << jllama_last_error();
    const Tuning loaded = tuning_after_a_wake(model);
    EXPECT_EQ(loaded.threads, 2);
    EXPECT_FLOAT_EQ(loaded.similarity, 0.5f);

    ASSERT_EQ(jllama_configure(model, R"({"n_threads":1,"n_threads_batch":1,"slot_prompt_similarity":0.25})"),
              JLLAMA_STATUS_OK)
        << jllama_last_error();
    const Tuning configured = tuning_after_a_wake(model);
    EXPECT_EQ(configured.threads, 1);
    EXPECT_EQ(configured.threads_batch, 1);
    EXPECT_FLOAT_EQ(configured.similarity, 0.25f);

    for (int sleep = 1; sleep <= 2; ++sleep) {
        SCOPED_TRACE("after sleep " + std::to_string(sleep));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (lines.count() < sleep && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT_GE(lines.count(), sleep) << "the model did not sleep";
        const Tuning woken = tuning_after_a_wake(model);
        EXPECT_EQ(woken.threads, 1) << "the wake built the context with the load's threads";
        EXPECT_EQ(woken.threads_batch, 1);
        EXPECT_FLOAT_EQ(woken.similarity, 0.25f) << "the wake took the load's similarity";
    }
    jllama_model_free(model);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
