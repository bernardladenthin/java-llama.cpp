// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// What `--cors-origins` does to a request of the embedded llama-server (upstream #32098, b12309).
//
// Until b12309 the option only decided the Access-Control-Allow-Origin header: a browser hid the
// response of a foreign page, and the request itself ran. Since then a value other than `*` refuses
// the request with 403 -- a foreign Origin (CSRF), and on a loopback connection a Host that names
// neither loopback nor an allowed origin (DNS rebinding: a page whose own name resolves to
// 127.0.0.1). NativeServer forwards the option verbatim, so this is the behaviour of a
// `NativeServer` started with it; the default `*` leaves every request alone, which the last test
// holds, because a changed default would lock out every client that names the machine by its host
// name.
//
// The real llama_server() in router mode over an empty models directory: no model, a loopback port.

#include "native_server_bridge.h"

#include <cpp-httplib/httplib.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

// A port nothing listens on: bound by a throw-away server, which is stopped again.
int free_loopback_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    std::thread listener([&probe]() { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    listener.join();
    return port;
}

class cors_router {
  public:
    explicit cors_router(const std::vector<std::string> &extra)
        : dir_(std::filesystem::temp_directory_path() /
               ("jllama-cors-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))),
          port_(free_loopback_port()) {
        std::filesystem::create_directories(dir_);
        args_ = {"llama-server", "--host", "127.0.0.1", "--port", std::to_string(port_), "--models-dir", dir_.string()};
        args_.insert(args_.end(), extra.begin(), extra.end());
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

    ~cors_router() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!finished_.load() && std::chrono::steady_clock::now() < deadline) {
            llama_server_request_shutdown();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (finished_.load()) {
            thread_.join();
        } else {
            thread_.detach(); // the test has failed already; do not hang the binary on top of it
        }
        std::error_code ignored;
        std::filesystem::remove_all(dir_, ignored);
    }

    int port() const { return port_; }

    // GET /health with the given headers; a Host among them replaces the client's own.
    httplib::Result get(const httplib::Headers &headers) const {
        httplib::Client client("127.0.0.1", port_);
        client.set_connection_timeout(5);
        client.set_read_timeout(10);
        return client.Get("/health", headers);
    }

    // The server answers once it listens; until then the connection is refused.
    bool wait_until_listening() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline && !finished_.load()) {
            const httplib::Result res = get({});
            if (res && res->status == 200) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return false;
    }

  private:
    std::filesystem::path dir_;
    int port_;
    std::vector<std::string> args_;
    std::vector<char *> argv_;
    std::thread thread_;
    std::atomic<bool> finished_{false};
    int exit_code_ = -1;
};

int status_of(const httplib::Result &res) { return res ? res->status : -1; }

} // namespace

TEST(NativeServerCors, AnAllowedOriginListRefusesAForeignHostAndOrigin) {
    cors_router router({"--cors-origins", "http://app.example"});
    ASSERT_TRUE(router.wait_until_listening());
    const std::string port = std::to_string(router.port());

    // the client's own Host (127.0.0.1:<port>) and no Origin: a local tool, served
    EXPECT_EQ(status_of(router.get({})), 200);
    EXPECT_EQ(status_of(router.get({{"Host", "localhost:" + port}})), 200);

    // DNS rebinding: the connection is loopback, the name in Host is somebody else's
    const httplib::Result rebound = router.get({{"Host", "evil.example:" + port}});
    ASSERT_TRUE(rebound);
    EXPECT_EQ(rebound->status, 403);
    EXPECT_NE(rebound->body.find("permission_error"), std::string::npos) << rebound->body;
    EXPECT_NE(rebound->body.find("--cors-origins"), std::string::npos) << rebound->body;

    // the host of an allowed origin may be the name the server is reached under
    EXPECT_EQ(status_of(router.get({{"Host", "app.example"}})), 200);

    // CSRF: the allowed origin is served and named, another one is refused
    const httplib::Result allowed = router.get({{"Origin", "http://app.example"}});
    ASSERT_TRUE(allowed);
    EXPECT_EQ(allowed->status, 200);
    EXPECT_EQ(allowed->get_header_value("Access-Control-Allow-Origin"), "http://app.example");
    EXPECT_EQ(status_of(router.get({{"Origin", "http://evil.example"}})), 403);
    // the scheme is part of an origin
    EXPECT_EQ(status_of(router.get({{"Origin", "https://app.example"}})), 403);

    // a browser's no-cors request carries no Origin, but says where it comes from
    EXPECT_EQ(status_of(router.get({{"Sec-Fetch-Site", "cross-site"}})), 403);
    EXPECT_EQ(status_of(router.get({{"Sec-Fetch-Site", "same-origin"}})), 200);

    // a page served by the server itself: its Origin is the Host it was reached under
    EXPECT_EQ(status_of(router.get({{"Origin", "http://127.0.0.1:" + port}})), 200);
}

TEST(NativeServerCors, TheValueLocalhostServesLocalOriginsOnly) {
    cors_router router({"--cors-origins", "localhost"});
    ASSERT_TRUE(router.wait_until_listening());
    const std::string port = std::to_string(router.port());

    const httplib::Result local = router.get({{"Origin", "http://localhost:5173"}});
    ASSERT_TRUE(local);
    EXPECT_EQ(local->status, 200);
    EXPECT_EQ(local->get_header_value("Access-Control-Allow-Origin"), "http://localhost:5173");

    const httplib::Result foreign = router.get({{"Origin", "http://evil.example"}});
    ASSERT_TRUE(foreign);
    EXPECT_EQ(foreign->status, 403);
    EXPECT_EQ(foreign->get_header_value("Access-Control-Allow-Origin"), "");

    EXPECT_EQ(status_of(router.get({{"Host", "evil.example:" + port}})), 403);
    EXPECT_EQ(status_of(router.get({{"Host", "127.0.0.2:" + port}})), 200);
}

TEST(NativeServerCors, WithoutTheOptionEveryHostAndOriginIsServed) {
    cors_router router({});
    ASSERT_TRUE(router.wait_until_listening());
    const std::string port = std::to_string(router.port());

    EXPECT_EQ(status_of(router.get({{"Host", "evil.example:" + port}})), 200);
    const httplib::Result foreign = router.get({{"Origin", "http://evil.example"}});
    ASSERT_TRUE(foreign);
    EXPECT_EQ(foreign->status, 200);
    // upstream's default pair (`*` with credentials) names the origin that asked: a browser refuses `*` there
    EXPECT_EQ(foreign->get_header_value("Access-Control-Allow-Origin"), "http://evil.example");
    EXPECT_EQ(status_of(router.get({{"Sec-Fetch-Site", "cross-site"}})), 200);
}
