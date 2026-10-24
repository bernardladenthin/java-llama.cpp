// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Real TCP protocol-v8 clients of the C API used by JNI/FFM. No models, GPU or external network.
// Each scenario runs in a fresh subprocess: a regression in stop must produce a bounded failure,
// rather than hang the test runner or leave a server thread affecting the following cases.

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "jllama_api.h"
#include "ggml.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;
#ifdef _WIN32
using fd_t = SOCKET;
constexpr fd_t invalid_fd = INVALID_SOCKET;
void close_fd(fd_t fd) { closesocket(fd); }
#else
using fd_t = int;
constexpr fd_t invalid_fd = -1;
void close_fd(fd_t fd) { close(fd); }
#endif

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "RPC comm regression: %s\n", message);
        std::fflush(stderr);
        std::_Exit(1);
    }
}

void init_sockets() {
#ifdef _WIN32
    static const bool initialized = []() {
        WSADATA data;
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }();
    require(initialized, "WSAStartup failed");
#endif
}

struct socket_owner {
    fd_t fd = invalid_fd;
    explicit socket_owner(fd_t value = invalid_fd) : fd(value) {}
    ~socket_owner() {
        if (fd != invalid_fd) {
            close_fd(fd);
        }
    }
    socket_owner(const socket_owner &) = delete;
    socket_owner &operator=(const socket_owner &) = delete;
};

sockaddr_in address(const char *host, int port) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(host);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    return addr;
}

fd_t new_socket() {
    fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    require(fd != invalid_fd, "socket creation failed");
#ifdef _WIN32
    const DWORD timeout = 5000;
#else
    const timeval timeout{5, 0};
#endif
    require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&timeout), sizeof(timeout)) == 0,
            "receive timeout setup failed");
    require(setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&timeout), sizeof(timeout)) == 0,
            "send timeout setup failed");
#ifdef SO_NOSIGPIPE
    const int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    return fd;
}

bool can_bind(const char *host, int port) {
    socket_owner sock(new_socket());
    auto addr = address(host, port);
    return bind(sock.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
}

int free_port() {
    socket_owner sock(new_socket());
    auto addr = address("127.0.0.1", 0);
    require(bind(sock.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "ephemeral bind failed");
#ifdef _WIN32
    int len = sizeof(addr);
#else
    socklen_t len = sizeof(addr);
#endif
    require(getsockname(sock.fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0, "getsockname failed");
    return ntohs(addr.sin_port);
}

fd_t try_connect(const char *host, int port) {
    socket_owner sock(new_socket());
    auto addr = address(host, port);
    if (connect(sock.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        return invalid_fd;
    }
    fd_t result = sock.fd;
    sock.fd = invalid_fd;
    return result;
}

template <class Predicate> bool eventually(Predicate predicate, std::chrono::milliseconds budget = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    do {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

void send_bytes(fd_t fd, const void *data, size_t size) {
    auto bytes = static_cast<const char *>(data);
    while (size != 0) {
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;
#endif
        const auto n = send(fd, bytes, static_cast<int>(size), flags);
        require(n > 0, "wire send failed");
        bytes += n;
        size -= static_cast<size_t>(n);
    }
}

void recv_bytes(fd_t fd, void *data, size_t size) {
    auto bytes = static_cast<char *>(data);
    while (size != 0) {
        const auto n = recv(fd, bytes, static_cast<int>(size), 0);
        require(n > 0, "wire receive failed or timed out");
        bytes += n;
        size -= static_cast<size_t>(n);
    }
}

void command(fd_t fd, uint8_t cmd, const void *payload, uint64_t size) {
    // Like upstream, the protocol sends fixed-width fields in native byte order.
    send_bytes(fd, &cmd, sizeof(cmd));
    send_bytes(fd, &size, sizeof(size));
    send_bytes(fd, payload, static_cast<size_t>(size));
}

void hello(fd_t fd) {
    std::array<uint8_t, 24> caps{};
    command(fd, 14, caps.data(), caps.size());
    uint64_t size = 0;
    recv_bytes(fd, &size, sizeof(size));
    require(size == 28, "HELLO response changed: update the protocol-v8 fixture");
    std::array<uint8_t, 28> response{};
    recv_bytes(fd, response.data(), response.size());
    require(response[0] == 8, "expected RPC protocol v8");
}

void comm_init(fd_t fd, int port, bool rank_one) {
    struct request {
        uint32_t device, rank, world, port;
        char host[64];
    } req{0, rank_one ? 1u : 0u, 2, static_cast<uint32_t>(port), "127.0.0.1"};
    static_assert(sizeof(req) == 80, "COMM_INIT wire layout changed");
    command(fd, 20, &req, sizeof(req));
}

// Observe the real listener without occupying its port (a bind probe can race create_server),
// connecting a recovery peer, or adding a production test hook. Debug logs are enabled before
// the backend is initialized in this fresh subprocess. A changed log/protocol fails explicitly.
struct trace_state {
    std::string peer_endpoint;
    std::atomic<bool> listening{false};
    std::atomic<bool> accepted{false};
};

void trace_rpc(ggml_log_level, const char *text, void *context) {
    auto &state = *static_cast<trace_state *>(context);
    if (std::strstr(text, "[create_server] listening on ") && std::strstr(text, state.peer_endpoint.c_str())) {
        state.listening = true;
    }
    if (state.listening && std::strstr(text, "[accept] accepted client connection")) {
        state.accepted = true;
    }
}

class watchdog {
  public:
    watchdog()
        : thread_([this]() {
              std::unique_lock<std::mutex> lock(mutex_);
              if (!changed_.wait_for(lock, 25s, [this]() { return done_; })) {
                  require(false, "subprocess exceeded the 25-second safety deadline");
              }
          }) {}
    ~watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        changed_.notify_all();
        thread_.join();
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false;
    std::thread thread_;
};

class running_server {
  public:
    explicit running_server(int port, const char *host = "127.0.0.1")
        : worker_([this, port, host]() {
              status_ = jllama_rpc_server_serve(host, port, 1, nullptr, "CPU");
              done_ = true;
          }) {
        require(eventually([this]() { return jllama_rpc_server_listening() || done_.load(); }),
                "server startup timed out");
        require(jllama_rpc_server_listening(), "server failed to bind");
    }
    ~running_server() { stop(); }
    void stop(std::chrono::milliseconds budget = 5s) {
        if (!worker_.joinable()) {
            return;
        }
        std::atomic<bool> stopped{false};
        jllama_status stop_status = JLLAMA_STATUS_ERROR;
        std::thread stopper([&]() {
            stop_status = jllama_rpc_server_stop();
            stopped = true;
        });
        require(eventually([&]() { return stopped.load() && done_.load(); }, budget),
                "stop/serve exceeded its test deadline without help from a peer");
        stopper.join();
        worker_.join();
        require(stop_status == JLLAMA_STATUS_OK && status_ == JLLAMA_STATUS_OK, "stop/serve returned an error");
        require(!jllama_rpc_server_listening(), "listener flag survived stop");
    }

  private:
    std::atomic<bool> done_{false};
    jllama_status status_ = JLLAMA_STATUS_ERROR;
    std::thread worker_;
};

enum class scenario { bind_local, bind_network, no_peer, rank_zero_handshake, rank_one_handshake, established };

void run_scenario(scenario mode) {
    init_sockets();
#ifdef _WIN32
    _putenv_s("GGML_RPC_DEBUG", "2");
#else
    setenv("GGML_RPC_DEBUG", "2", 1);
#endif
    watchdog deadline;
    const int main_port = free_port();
    running_server server(main_port, mode == scenario::bind_network ? "0.0.0.0" : "127.0.0.1");
    socket_owner client(try_connect("127.0.0.1", main_port));
    require(client.fd != invalid_fd, "main connection failed");
    hello(client.fd);
    const int peer_port = free_port();
    trace_state trace;
    trace.peer_endpoint = ":" + std::to_string(peer_port) + "\n";
    ggml_log_set(trace_rpc, &trace);
    socket_owner listener;
    if (mode == scenario::rank_one_handshake) {
        listener.fd = new_socket();
        auto addr = address("127.0.0.1", peer_port);
        require(bind(listener.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake peer bind failed");
        require(listen(listener.fd, 1) == 0, "fake peer listen failed");
    }
    comm_init(client.fd, peer_port, mode == scenario::rank_one_handshake);
    socket_owner peer;
    if (mode == scenario::rank_one_handshake) {
        // select bounds accept even with the old implementation. No model is needed to reach it.
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(listener.fd, &read_set);
        timeval timeout{5, 0};
        require(select(static_cast<int>(listener.fd + 1), &read_set, nullptr, nullptr, &timeout) > 0,
                "rank one did not connect to the fake peer");
        peer.fd = accept(listener.fd, nullptr, nullptr);
        require(peer.fd != invalid_fd, "fake peer accept failed");
        std::array<uint8_t, 24> caps{};
        recv_bytes(peer.fd, caps.data(), caps.size()); // server is now waiting for our caps
    } else {
        require(eventually([&]() { return trace.listening.load(); }), "COMM_INIT did not open its listener");
        if (mode == scenario::bind_local || mode == scenario::bind_network) {
            socket_owner other(try_connect("127.0.0.2", peer_port));
            require((other.fd != invalid_fd) == (mode == scenario::bind_network),
                    "COMM_INIT did not preserve the configured bind address");
        }
        if (mode != scenario::no_peer && mode != scenario::bind_network) {
            peer.fd = try_connect("127.0.0.1", peer_port);
            require(peer.fd != invalid_fd, "peer connection failed");
            require(eventually([&]() { return trace.accepted.load(); }), "server did not accept the peer");
        }
        if (mode == scenario::established || mode == scenario::bind_local) {
            std::array<uint8_t, 24> caps{};
            send_bytes(peer.fd, caps.data(), caps.size());
            recv_bytes(peer.fd, caps.data(), caps.size());
            uint64_t size = 0;
            uint8_t ok = 0;
            recv_bytes(client.fd, &size, sizeof(size));
            require(size == 1, "COMM_INIT response size changed");
            recv_bytes(client.fd, &ok, sizeof(ok));
            require(ok == 1, "COMM_INIT rejected a valid pair");
        }
    }
    server.stop();
    // No recovery peer is supplied. Stop must have closed both ports by itself.
    socket_owner old_main(try_connect("127.0.0.1", main_port));
    socket_owner old_peer(try_connect("127.0.0.1", peer_port));
    require(old_main.fd == invalid_fd, "main listener survived shutdown");
    if (mode != scenario::rank_one_handshake) {
        require(old_peer.fd == invalid_fd, "peer listener survived shutdown");
    }
    for (int port : {main_port, peer_port}) {
        if (mode == scenario::rank_one_handshake && port == peer_port) {
            continue; // this listener belongs to the test, not the RPC server
        }
        running_server restarted(port);
        socket_owner next(try_connect("127.0.0.1", port));
        require(next.fd != invalid_fd, "restart could not reuse the old port");
        hello(next.fd);
        restarted.stop();
    }
    ggml_log_set(nullptr, nullptr);
}

void check_scenario(scenario mode) {
    // Re-exec instead of forking a process that may have ggml worker threads from earlier tests.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        {
            run_scenario(mode);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), "");
}

} // namespace

TEST(RpcComm, LocalServerDoesNotOpenAWildcardPeerListener) {
    init_sockets();
    if (!can_bind("127.0.0.2", 0)) {
        GTEST_SKIP() << "this host has no second loopback address; shutdown scenarios still run";
    }
    check_scenario(scenario::bind_local);
}

TEST(RpcComm, NetworkBindRemainsAnExplicitOptIn) {
    init_sockets();
    if (!can_bind("127.0.0.2", 0)) {
        GTEST_SKIP() << "this host has no second loopback address";
    }
    check_scenario(scenario::bind_network);
}

TEST(RpcComm, StopWithoutPeerClosesBothListenersAndAllowsRestart) { check_scenario(scenario::no_peer); }
TEST(RpcComm, StopInterruptsRankZeroCapsHandshake) { check_scenario(scenario::rank_zero_handshake); }
TEST(RpcComm, StopInterruptsRankOneCapsHandshake) { check_scenario(scenario::rank_one_handshake); }
TEST(RpcComm, SuccessfulPeerSetupStillWorksAndStops) { check_scenario(scenario::established); }
