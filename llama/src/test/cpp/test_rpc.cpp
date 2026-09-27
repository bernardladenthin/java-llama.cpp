// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// RPC support: the device-selection rules of rpc_support.hpp (pure, literal inputs), and the real
// ggml-rpc client and server end to end over loopback -- no model, no second machine, no GPU.
//
// The loopback half is the runnable guard for patches/0015: it links ggml_backend_rpc_stop_server()
// and ggml_backend_rpc_server_listening(), so a bump that drops the patch fails this binary at link
// time on every platform, and it pins the two behaviours the patch exists for -- an unreachable
// server is an ordinary failure (nullptr / std::invalid_argument) rather than an abort of the
// process, and a running server can be stopped from another thread.

#include "rpc_support.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"
#include "ggml.h"
#include "llama.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using jllama::rpc::device_info;
using jllama::rpc::device_override;
using jllama::rpc::has_mmproj_device_option;
using jllama::rpc::mmproj_device_override;

namespace {

device_info gpu(const std::string &name, const std::string &id = "") {
    device_info d;
    d.name = name;
    d.is_gpu = true;
    d.device_id = id;
    return d;
}

device_info igpu(const std::string &name) {
    device_info d;
    d.name = name;
    d.is_igpu = true;
    return d;
}

device_info rpc_dev(const std::string &name, const std::string &endpoint) {
    device_info d;
    d.name = name;
    d.is_gpu = true;
    d.endpoint = endpoint;
    return d;
}

device_info cpu() {
    device_info d;
    d.name = "CPU";
    return d;
}

} // namespace

// ---- pure rules ------------------------------------------------------------------------------

TEST(RpcSupport, SplitEndpointsDropsEmptyEntries) {
    EXPECT_EQ(jllama::rpc::split_endpoints("a:1,,b:2,"), (std::vector<std::string>{"a:1", "b:2"}));
    EXPECT_TRUE(jllama::rpc::split_endpoints("").empty());
}

TEST(RpcSupport, RequestedEndpointsAccumulateAcrossRepeatedOptions) {
    const std::vector<std::string> argv = {"llama", "--rpc", "a:1,b:2", "-m", "x.gguf", "--rpc", "c:3"};
    EXPECT_EQ(jllama::rpc::requested_endpoints(argv), (std::vector<std::string>{"a:1", "b:2", "c:3"}));
}

TEST(RpcSupport, RpcAsLastArgumentWithoutValueRequestsNothing) {
    EXPECT_TRUE(jllama::rpc::requested_endpoints({"llama", "--rpc"}).empty());
}

TEST(RpcSupport, DeviceOptionIsRecognisedInBothSpellings) {
    EXPECT_TRUE(jllama::rpc::has_device_option({"llama", "--device", "CUDA0"}));
    EXPECT_TRUE(jllama::rpc::has_device_option({"llama", "-dev", "none"}));
    EXPECT_FALSE(jllama::rpc::has_device_option({"llama", "--mmproj-device", "CUDA0"}));
}

TEST(RpcSupport, NoRpcDeviceLeavesTheArgvAlone) {
    EXPECT_FALSE(device_override({cpu(), gpu("CUDA0")}, {}).has_value());
}

TEST(RpcSupport, OnlyRequestedRpcDevicesLeaveTheArgvAlone) {
    EXPECT_FALSE(device_override({cpu(), rpc_dev("RPC0", "a:1"), gpu("CUDA0")}, {"a:1"}).has_value());
}

TEST(RpcSupport, StaleRpcDeviceOnACpuOnlyHostMeansNone) {
    EXPECT_EQ(device_override({cpu(), rpc_dev("RPC0", "a:1")}, {}).value(), "none");
}

TEST(RpcSupport, StaleRpcDeviceIsReplacedByTheLocalGpus) {
    EXPECT_EQ(device_override({cpu(), rpc_dev("RPC0", "a:1"), gpu("CUDA0"), gpu("CUDA1")}, {}).value(), "CUDA0,CUDA1");
}

TEST(RpcSupport, RequestedRpcDevicesComeFirstAndStaleOnesAreDropped) {
    const std::vector<device_info> devices = {cpu(), gpu("CUDA0"), rpc_dev("RPC0", "old:1"), rpc_dev("RPC1", "new:2"),
                                              rpc_dev("RPC2", "new:2")};
    EXPECT_EQ(device_override(devices, {"new:2"}).value(), "RPC1,RPC2,CUDA0");
}

TEST(RpcSupport, TheSameGpuSeenByTwoBackendsIsKeptOnce) {
    const std::vector<device_info> devices = {rpc_dev("RPC0", "a:1"), gpu("CUDA0", "0000:01:00.0"),
                                              gpu("Vulkan0", "0000:01:00.0"), gpu("Vulkan1", "0000:02:00.0")};
    EXPECT_EQ(device_override(devices, {}).value(), "CUDA0,Vulkan1");
}

TEST(RpcSupport, GpusWithoutADeviceIdAreNeverTreatedAsDuplicates) {
    EXPECT_EQ(device_override({rpc_dev("RPC0", "a:1"), gpu("A"), gpu("B")}, {}).value(), "A,B");
}

TEST(RpcSupport, IntegratedGpuOnlyWhenThereIsNoDiscreteOne) {
    EXPECT_EQ(device_override({rpc_dev("RPC0", "a:1"), igpu("iGPU0")}, {}).value(), "iGPU0");
    EXPECT_EQ(device_override({rpc_dev("RPC0", "a:1"), igpu("iGPU0"), gpu("CUDA0")}, {}).value(), "CUDA0");
}

TEST(RpcSupport, MmprojDeviceIsLeftAloneWithoutAStaleRpcDevice) {
    EXPECT_FALSE(mmproj_device_override({cpu(), gpu("CUDA0")}, {}).has_value());
    EXPECT_FALSE(mmproj_device_override({cpu(), rpc_dev("RPC0", "a:1")}, {"a:1"}).has_value());
}

TEST(RpcSupport, MmprojDeviceSkipsAStaleRpcDeviceTheWayClipWouldChoose) {
    // clip takes the first GPU in registry order, else the first iGPU; RPC devices are type GPU
    EXPECT_EQ(mmproj_device_override({cpu(), rpc_dev("RPC0", "a:1")}, {}).value(), "") << "CPU-only host";
    EXPECT_EQ(mmproj_device_override({gpu("CUDA0"), cpu(), rpc_dev("RPC0", "a:1")}, {}).value(), "CUDA0");
    EXPECT_EQ(mmproj_device_override({igpu("iGPU0"), cpu(), rpc_dev("RPC0", "a:1")}, {}).value(), "iGPU0");
    // a requested RPC device is as eligible as it is upstream
    EXPECT_EQ(mmproj_device_override({cpu(), rpc_dev("RPC0", "a:1"), rpc_dev("RPC1", "b:2")}, {"b:2"}).value(), "RPC1");
}

TEST(RpcSupport, MmprojDeviceOptionIsRecognisedInEverySpelling) {
    EXPECT_TRUE(has_mmproj_device_option({"x", "--mmproj-device", "CUDA0"}));
    EXPECT_TRUE(has_mmproj_device_option({"x", "-mmdev", "CUDA0"}));
    EXPECT_TRUE(has_mmproj_device_option({"x", "--no-mmproj-offload"}));
    EXPECT_FALSE(has_mmproj_device_option({"x", "--mmproj-offload", "--mmproj", "m.gguf"}));
}

// ---- the real client and server over loopback --------------------------------------------------

TEST(RpcBuild, RpcBackendIsCompiledIn) { EXPECT_TRUE(llama_supports_rpc()); }

namespace {

// Runs ggml's blocking RPC server on a thread of its own, on the first free port of a small range
// (a fixed port would collide with parallel jobs on one runner). stop() + the destructor end it.
class loopback_server {
  public:
    // Serves the CPU: ggml-rpc's client reports every operation as supported, so a served GPU that
    // cannot run one aborts the server -- the paravirtual Metal device of the macOS CI runners has
    // no MUL_MAT. The transport is what these tests are about, not the device behind it.
    loopback_server() {
        devices_ = jllama::rpc::server_devices({"CPU"});
        for (int attempt = 0; attempt < 50 && !listening_; ++attempt) {
            port_ = 42000 +
                    static_cast<int>(
                        (std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + attempt * 97) % 20000);
            const std::string endpoint = "127.0.0.1:" + std::to_string(port_);
            auto done = std::make_shared<std::atomic<bool>>(false);
            thread_ = std::thread([this, endpoint, done]() {
                ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 2, devices_.size(), devices_.data());
                done->store(true);
            });
            // Listening, or returned because the port could not be bound.
            for (int i = 0; i < 500 && !ggml_backend_rpc_server_listening() && !done->load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (ggml_backend_rpc_server_listening()) {
                listening_ = true;
            } else {
                thread_.join();
            }
        }
    }

    ~loopback_server() { stop(); }

    void stop() {
        if (thread_.joinable()) {
            ggml_backend_rpc_stop_server();
            thread_.join();
        }
    }

    bool listening() const { return listening_; }
    std::string endpoint() const { return "127.0.0.1:" + std::to_string(port_); }

  private:
    std::vector<ggml_backend_dev_t> devices_;
    std::thread thread_;
    bool listening_ = false;
    int port_ = 0;
};

// A port the loop above never picks, with nothing listening on it.
std::string closed_endpoint() { return "127.0.0.1:1"; }

// y = W x with small integers, computed on the given backend.
std::vector<float> mul_mat_on(ggml_backend_t backend) {
    const int64_t k = 4;
    const int64_t n = 3;
    ggml_init_params params = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> w_data(static_cast<size_t>(k * n));
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = static_cast<float>(i) - 5.0f;
    }
    const std::vector<float> x_data = {1.0f, -2.0f, 3.0f, 0.5f};
    ggml_backend_tensor_set(w, w_data.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));

    EXPECT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);
    std::vector<float> out(static_cast<size_t>(n));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return out;
}

} // namespace

TEST(RpcServerDevices, NamedDevicesAreFoundWithoutRegardToCaseAndDeduplicated) {
    const auto devices = jllama::rpc::server_devices({"cpu", "CPU"});
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(ggml_backend_dev_type(devices[0]), GGML_BACKEND_DEVICE_TYPE_CPU);
}

TEST(RpcServerDevices, NoNamesMeansTheDefaultChoice) {
    const auto named = jllama::rpc::server_devices(std::vector<std::string>{});
    const auto fallback = jllama::rpc::server_devices();
    EXPECT_EQ(named, fallback);
    EXPECT_FALSE(named.empty());
}

TEST(RpcServerDevices, AnUnknownNameListsTheAvailableDevices) {
    try {
        (void)jllama::rpc::server_devices({"CPU", "NO-SUCH-DEVICE"});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("unknown device 'NO-SUCH-DEVICE'"), std::string::npos) << what;
        EXPECT_NE(what.find("available: "), std::string::npos) << what;
        EXPECT_NE(what.find("CPU"), std::string::npos) << what;
    }
}

TEST(RpcLoopback, ServerComputesTheSameResultAsTheLocalCpu) {
    loopback_server server;
    ASSERT_TRUE(server.listening()) << "no free port found for the loopback RPC server";

    std::vector<float> remote;
    {
        ggml_backend_t rpc = ggml_backend_rpc_init(server.endpoint().c_str(), 0);
        ASSERT_NE(rpc, nullptr);
        remote = mul_mat_on(rpc);
        ggml_backend_free(rpc);
    }
    ggml_backend_t local = ggml_backend_cpu_init();
    const auto expected = mul_mat_on(local);
    ggml_backend_free(local);

    ASSERT_EQ(remote.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(remote[i], expected[i]) << "row " << i;
    }
}

TEST(RpcLoopback, StopEndsTheServerAndFreesThePort) {
    loopback_server server;
    ASSERT_TRUE(server.listening());
    server.stop();
    EXPECT_FALSE(ggml_backend_rpc_server_listening());
    // the port is free again: a new server can bind the very same endpoint
    const std::string endpoint = server.endpoint();
    auto devices = jllama::rpc::server_devices({"CPU"});
    std::thread again(
        [&]() { ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, devices.size(), devices.data()); });
    for (int i = 0; i < 500 && !ggml_backend_rpc_server_listening(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(ggml_backend_rpc_server_listening());
    ggml_backend_rpc_stop_server();
    again.join();
}

TEST(RpcLoopback, StopDisconnectsAClientThatIsStillConnected) {
    loopback_server server;
    ASSERT_TRUE(server.listening());
    // an open backend keeps its connection -- the server is inside rpc_serve_client when stopped
    ggml_backend_t rpc = ggml_backend_rpc_init(server.endpoint().c_str(), 0);
    ASSERT_NE(rpc, nullptr);
    server.stop(); // must return rather than wait for the client to hang up
    EXPECT_FALSE(ggml_backend_rpc_server_listening());
    ggml_backend_free(rpc);
}

TEST(RpcClient, UnreachableServerIsAFailureNotAnAbort) {
    EXPECT_EQ(ggml_backend_rpc_add_server(closed_endpoint().c_str()), nullptr);
}

TEST(RpcClient, MalformedEndpointIsAFailureNotAnAbort) {
    EXPECT_EQ(ggml_backend_rpc_add_server("no-port-here"), nullptr);
}

TEST(RpcClient, PrepareArgvRejectsAnUnreachableServerWithItsName) {
    try {
        (void)jllama::rpc::prepare_argv({"llama", "--rpc", closed_endpoint()});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find(closed_endpoint()), std::string::npos) << e.what();
    }
}

TEST(RpcClient, AStaleServerIsKeptOutOfALaterLoadThatDidNotAskForIt) {
    loopback_server server;
    ASSERT_TRUE(server.listening());
    const std::string endpoint = server.endpoint();

    // the load that asks for it registers it and is left alone
    const auto with_rpc = jllama::rpc::prepare_argv({"llama", "--rpc", endpoint});
    EXPECT_FALSE(jllama::rpc::has_device_option(with_rpc));

    // a later load without --rpc gets an explicit device list that leaves the RPC device out
    const auto without = jllama::rpc::prepare_argv({"llama", "-m", "x.gguf"});
    ASSERT_TRUE(jllama::rpc::has_device_option(without));
    const std::string &devices = without.back();
    for (const auto &info : jllama::rpc::registered_devices()) {
        if (!info.endpoint.empty()) {
            EXPECT_EQ(devices.find(info.name), std::string::npos) << devices;
        }
    }

    // ...and the multimodal projector is pinned too: clip would otherwise take the stale RPC device
    // on a host without a local GPU
    ASSERT_TRUE(jllama::rpc::has_mmproj_device_option(without));
    for (const auto &info : jllama::rpc::registered_devices()) {
        if (!info.endpoint.empty()) {
            EXPECT_EQ(std::find(without.begin(), without.end(), info.name), without.end()) << info.name;
        }
    }

    // an explicit --device from the caller is never overridden (the mmproj device still is pinned)
    const auto explicit_devices = jllama::rpc::prepare_argv({"llama", "-dev", "none", "--no-mmproj-offload"});
    EXPECT_EQ(explicit_devices, (std::vector<std::string>{"llama", "-dev", "none", "--no-mmproj-offload"}));

    // the params-level guard for TextToSpeech / LlamaTrainer, which never parse an argv
    std::vector<ggml_backend_dev_t> params_devices;
    const auto mmproj = jllama::rpc::exclude_stale_devices(params_devices);
    ASSERT_FALSE(params_devices.empty());
    EXPECT_EQ(params_devices.back(), nullptr) << "null-terminated like parse_device_list";
    for (size_t i = 0; i + 1 < params_devices.size(); ++i) {
        ASSERT_NE(params_devices[i], nullptr);
        EXPECT_FALSE(jllama::rpc::is_rpc_device(params_devices[i])) << ggml_backend_dev_name(params_devices[i]);
    }
    ASSERT_TRUE(mmproj.has_value());
    if (!mmproj->empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(mmproj->c_str());
        ASSERT_NE(dev, nullptr);
        EXPECT_FALSE(jllama::rpc::is_rpc_device(dev));
    }
    // an explicit device list is kept as it is
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    std::vector<ggml_backend_dev_t> chosen = {cpu_dev, nullptr};
    (void)jllama::rpc::exclude_stale_devices(chosen);
    EXPECT_EQ(chosen, (std::vector<ggml_backend_dev_t>{cpu_dev, nullptr}));

    // and a registered RPC device can never be served back out by name (it would forward to itself)
    size_t checked = 0;
    for (const auto &info : jllama::rpc::registered_devices()) {
        if (!info.endpoint.empty()) {
            try {
                (void)jllama::rpc::server_devices({info.name});
                ADD_FAILURE() << "served the RPC device " << info.name;
            } catch (const std::invalid_argument &e) {
                EXPECT_NE(std::string(e.what()).find("itself a remote RPC device"), std::string::npos) << e.what();
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

TEST(RpcClient, ARegisteredServerThatWentAwayIsReportedOnTheNextRegistration) {
    std::string endpoint;
    {
        loopback_server server;
        ASSERT_TRUE(server.listening());
        endpoint = server.endpoint();
        ASSERT_NE(ggml_backend_rpc_add_server(endpoint.c_str()), nullptr);
    }
    // the endpoint is still cached in ggml-rpc's registry map; it must be re-checked, not trusted
    EXPECT_EQ(ggml_backend_rpc_add_server(endpoint.c_str()), nullptr);
}

TEST(RpcClient, AGoneServersDeviceReportsNoMemoryInsteadOfAborting) {
    // common_init() lists the memory of EVERY registered device at -lv 4, and --list-devices does
    // too; a server that stopped after registering must not take the process down there.
    std::string endpoint;
    {
        loopback_server server;
        ASSERT_TRUE(server.listening());
        endpoint = server.endpoint();
        jllama::rpc::register_server(endpoint);
    }
    bool found = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (jllama::rpc::is_rpc_device(dev) && endpoint == ggml_backend_dev_description(dev)) {
            found = true;
            size_t free = 1;
            size_t total = 1;
            ggml_backend_dev_memory(dev, &free, &total);
            EXPECT_EQ(free, 0u);
            EXPECT_EQ(total, 0u);
        }
    }
    EXPECT_TRUE(found) << "the stopped server's device should still be in the registry";
}
