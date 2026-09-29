// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The shutdown path of the embedded llama-server (patches/0006 + 0007, native_server.cpp).
//
// upstream's llama_server() installs a process-wide `shutdown_handler` lambda that captures its own
// locals (ctx_http, models_routes, mcp_mgr) by reference, and never clears it. NativeServer stops
// the server by calling llama_server_request_shutdown() -- repeatedly, until the worker reports it
// finished, because a stop issued before the handler is installed would otherwise be lost. A call
// that lands after llama_server() returned therefore ran the lambda over destroyed locals: the JVM
// died with SIGSEGV in server_http_context::stop() (CI, RouterModeIntegrationTest.tearDown, run
// 36485159455). Router mode made the window wide: its clean-up stops worker processes first.
//
// These tests drive the real llama_server() in router mode over an empty models directory -- no
// model, no network beyond an ephemeral loopback port -- so they run in C++ Tests on every desktop
// platform and fail deterministically instead of once in a while in a Java job.

#include "native_server_bridge.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

// An empty, unique models directory: router mode with nothing to route to starts and stops fast.
std::filesystem::path empty_models_dir() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("jllama-router-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    return dir;
}

// Runs llama_server() on a thread of its own, the way native_server.cpp does.
class embedded_router {
  public:
    embedded_router() : dir_(empty_models_dir()) {
        args_ = {"llama-server", "--host", "127.0.0.1", "--port", "0", "--models-dir", dir_.string()};
        for (auto &arg : args_) {
            argv_.push_back(arg.data());
        }
        argv_.push_back(nullptr);
        llama_server_set_embedded(true);
        thread_ = std::thread([this]() {
            exit_code_ = llama_server(static_cast<int>(args_.size()), argv_.data());
            finished_.store(true);
        });
    }

    ~embedded_router() {
        if (thread_.joinable()) {
            stop();
        }
        std::error_code ignored;
        std::filesystem::remove_all(dir_, ignored);
    }

    // What NativeServer.stopNativeServer does: signal until the server has actually returned.
    void stop() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!finished_.load() && std::chrono::steady_clock::now() < deadline) {
            llama_server_request_shutdown();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ASSERT_TRUE(finished_.load()) << "llama_server() did not return after shutdown requests";
        thread_.join();
    }

    int exit_code() const { return exit_code_; }

  private:
    std::filesystem::path dir_;
    std::vector<std::string> args_;
    std::vector<char *> argv_;
    std::thread thread_;
    std::atomic<bool> finished_{false};
    int exit_code_ = -1;
};

} // namespace

TEST(NativeServerShutdown, RouterStopsOnRequest) {
    embedded_router router;
    router.stop();
    EXPECT_EQ(router.exit_code(), 0);
}

// The crash itself, made deterministic: a request after llama_server() returned must be a no-op.
// Before the fix it invoked the lambda over the destroyed ctx_http and segfaulted.
TEST(NativeServerShutdown, ARequestAfterTheServerReturnedIsANoOp) {
    {
        embedded_router router;
        router.stop();
    }
    llama_server_request_shutdown();
    llama_server_request_shutdown();
    SUCCEED();
}

// The race the CI run hit, from the other side: requests hammering the handler while the server is
// cleaning up and tearing down its locals, with no pause between them.
TEST(NativeServerShutdown, RequestsDuringTeardownAreSafe) {
    for (int round = 0; round < 5; ++round) {
        std::atomic<bool> done{false};
        std::thread hammer([&done]() {
            while (!done.load()) {
                llama_server_request_shutdown();
            }
        });
        {
            embedded_router router;
            router.stop();
        }
        done.store(true);
        hammer.join();
    }
    SUCCEED();
}
