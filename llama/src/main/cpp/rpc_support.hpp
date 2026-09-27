// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// RPC support shared by every JNI entry point that parses a llama.cpp argv (LlamaModel,
// NativeServer) and by the RpcServer bridge.
//
// Why this exists. ggml keeps ONE process-wide backend registry, and an RPC server registered by
// `--rpc host:port` stays in it for the life of the process -- there is no unregister. llama.cpp's
// default device selection then puts every registered RPC device in front of the local GPUs. In a
// standalone llama-server that is harmless (one model, one argv, one process). In a JVM it is not:
// a second model loaded WITHOUT `--rpc` would silently offload its layers to a server the first
// model asked for, and if that server is gone by then, the first tensor upload aborts the whole
// JVM (ggml-rpc treats a vanished server as fatal). So before an argv is parsed:
//
//   1. every endpoint it names in `--rpc` is registered here, up front, so an unreachable server
//      fails the load with a clear message instead of reaching llama.cpp at all (patches/0015 makes
//      ggml_backend_rpc_add_server return nullptr for it instead of aborting);
//   2. if the registry holds RPC devices this argv did NOT ask for, and the caller did not choose
//      devices itself (`--device`/`-dev`), an explicit `--device` list is appended: the requested
//      RPC devices plus the local GPUs, chosen the way llama.cpp's default would choose them.
//
// When no stale RPC device exists the argv is returned untouched, so a process that never uses RPC
// sees exactly upstream's behaviour.
//
// The first half of this header is pure (strings and plain structs, no ggml calls) so the selection
// rules are unit-tested with literal inputs; the second half is the thin glue over the registry.

#pragma once

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace jllama::rpc {

// ---- pure part ------------------------------------------------------------------------------

// One device as the selection rules need to see it.
struct device_info {
    std::string name;      // ggml_backend_dev_name(), e.g. "CUDA0", "RPC0"
    std::string endpoint;  // for an RPC device: the host:port it lives on; empty otherwise
    bool is_gpu = false;   // GGML_BACKEND_DEVICE_TYPE_GPU (RPC devices report GPU too)
    bool is_igpu = false;  // GGML_BACKEND_DEVICE_TYPE_IGPU
    std::string device_id; // ggml_backend_dev_props::device_id, used to drop duplicates; may be empty
};

// Splits a comma-separated `--rpc` value the way upstream's add_rpc_devices does (empty entries
// are kept out, so "a:1,,b:2" is two endpoints). RpcServer's device-name list uses it too.
[[nodiscard]] inline std::vector<std::string> split_endpoints(const std::string &value) {
    std::vector<std::string> out;
    std::string current;
    for (char c : value) {
        if (c == ',') {
            if (!current.empty()) {
                out.push_back(current);
            }
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

// Every endpoint the argv asks for, in order, across repeated `--rpc` options (each one registers
// its servers, so they accumulate in upstream too). argv[0] is treated like any other element.
[[nodiscard]] inline std::vector<std::string> requested_endpoints(const std::vector<std::string> &argv) {
    std::vector<std::string> out;
    for (size_t i = 0; i + 1 < argv.size(); ++i) {
        if (argv[i] == "--rpc") {
            for (auto &ep : split_endpoints(argv[i + 1])) {
                out.push_back(ep);
            }
        }
    }
    return out;
}

// true when the caller chose the devices itself; its choice is never overridden.
[[nodiscard]] inline bool has_device_option(const std::vector<std::string> &argv) {
    for (const auto &arg : argv) {
        if (arg == "--device" || arg == "-dev") {
            return true;
        }
    }
    return false;
}

// The `--device` value to append, or nullopt when the argv must be left as it is (no RPC device
// this argv did not ask for). Mirrors llama.cpp's default selection (src/llama.cpp,
// llama_model_load): requested RPC devices first, then the GPUs with duplicates by device id
// dropped, and the integrated GPUs only when there is no discrete one. "none" = CPU only.
[[nodiscard]] inline std::optional<std::string> device_override(const std::vector<device_info> &devices,
                                                                const std::vector<std::string> &requested) {
    auto is_requested = [&requested](const std::string &endpoint) {
        for (const auto &r : requested) {
            if (r == endpoint) {
                return true;
            }
        }
        return false;
    };

    bool stale = false;
    for (const auto &d : devices) {
        if (!d.endpoint.empty() && !is_requested(d.endpoint)) {
            stale = true;
        }
    }
    if (!stale) {
        return std::nullopt;
    }

    std::vector<std::string> rpc;
    std::vector<std::string> gpus;
    std::vector<std::string> gpu_ids;
    std::vector<std::string> igpus;
    for (const auto &d : devices) {
        if (!d.endpoint.empty()) {
            if (is_requested(d.endpoint)) {
                rpc.push_back(d.name);
            }
        } else if (d.is_gpu) {
            bool duplicate = false;
            if (!d.device_id.empty()) {
                for (const auto &id : gpu_ids) {
                    if (id == d.device_id) {
                        duplicate = true;
                    }
                }
            }
            if (!duplicate) {
                gpus.push_back(d.name);
                gpu_ids.push_back(d.device_id);
            }
        } else if (d.is_igpu) {
            igpus.push_back(d.name);
        }
    }

    std::vector<std::string> chosen = rpc;
    chosen.insert(chosen.end(), gpus.begin(), gpus.end());
    if (gpus.empty()) {
        chosen.insert(chosen.end(), igpus.begin(), igpus.end());
    }
    if (chosen.empty()) {
        return std::string("none");
    }
    std::string joined;
    for (size_t i = 0; i < chosen.size(); ++i) {
        if (i > 0) {
            joined.push_back(',');
        }
        joined += chosen[i];
    }
    return joined;
}

// ---- glue over the ggml registry ------------------------------------------------------------

// ggml's RPC devices all belong to the one backend reg named "RPC"; their description is the
// endpoint they were registered for.
[[nodiscard]] inline bool is_rpc_device(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && std::string(ggml_backend_reg_name(reg)) == "RPC";
}

[[nodiscard]] inline std::vector<device_info> registered_devices() {
    std::vector<device_info> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        device_info info;
        info.name = ggml_backend_dev_name(dev);
        const auto type = ggml_backend_dev_type(dev);
        info.is_gpu = type == GGML_BACKEND_DEVICE_TYPE_GPU;
        info.is_igpu = type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (is_rpc_device(dev)) {
            info.endpoint = ggml_backend_dev_description(dev);
        } else {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            if (props.device_id != nullptr) {
                info.device_id = props.device_id;
            }
        }
        out.push_back(info);
    }
    return out;
}

// Registers one RPC server; throws std::invalid_argument when it cannot be reached (patches/0015
// turns that case into a nullptr instead of an abort of the process).
inline void register_server(const std::string &endpoint) {
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
    if (reg == nullptr) {
        throw std::invalid_argument("cannot reach RPC server " + endpoint +
                                    " (is an rpc-server / RpcServer listening there?)");
    }
    ggml_backend_register(reg);
}

// See the header comment. Returns the argv to parse: the input, possibly with `--device <list>`
// appended. Throws std::invalid_argument for an unreachable `--rpc` endpoint.
[[nodiscard]] inline std::vector<std::string> prepare_argv(std::vector<std::string> argv) {
    const auto requested = requested_endpoints(argv);
    for (const auto &endpoint : requested) {
        register_server(endpoint);
    }
    if (has_device_option(argv)) {
        return argv;
    }
    auto value = device_override(registered_devices(), requested);
    if (value) {
        argv.emplace_back("--device");
        argv.push_back(*value);
    }
    return argv;
}

// Devices an in-process RPC server offers: every accelerator, else the CPU (upstream rpc-server's
// default), but never an RPC device -- serving a remote device back out would forward to itself.
[[nodiscard]] inline std::vector<ggml_backend_dev_t> server_devices() {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_ACCEL && !is_rpc_device(dev)) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            devices.push_back(cpu);
        }
    }
    return devices;
}

// Devices an in-process RPC server offers when the caller names them (upstream rpc-server's
// `--device`); an empty list means the default above. Naming them matters because ggml-rpc's client
// reports every operation as supported (upstream TODO in ggml_backend_rpc_device_supports_op), so a
// served device that cannot run an operation aborts the server process on the first graph that
// uses it -- e.g. the MUL_MAT-less paravirtual Metal GPU of a macOS VM. Serving the CPU instead is
// then the only way to use such a machine.
[[nodiscard]] inline std::vector<ggml_backend_dev_t> server_devices(const std::vector<std::string> &names) {
    if (names.empty()) {
        return server_devices();
    }
    std::vector<ggml_backend_dev_t> devices;
    for (const auto &name : names) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) {
            std::string available;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
                if (!is_rpc_device(candidate)) {
                    available += (available.empty() ? "" : ", ") + std::string(ggml_backend_dev_name(candidate));
                }
            }
            throw std::invalid_argument("unknown device '" + name + "' to serve over RPC; available: " + available);
        }
        if (is_rpc_device(dev)) {
            throw std::invalid_argument("device '" + name +
                                        "' is itself a remote RPC device and cannot be served over RPC");
        }
        bool duplicate = false;
        for (auto *seen : devices) {
            duplicate = duplicate || seen == dev;
        }
        if (!duplicate) {
            devices.push_back(dev);
        }
    }
    return devices;
}

} // namespace jllama::rpc
