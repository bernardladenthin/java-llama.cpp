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
//   1. every endpoint it names in `--rpc` or LLAMA_ARG_RPC is registered here, up front and before
//      the caller takes any lock, so an unreachable server fails the load with a clear message
//      instead of reaching llama.cpp at all, and the parse after it -- which the engine runs under
//      the lock of the process-wide log -- finds every endpoint registered and connects to none
//      (the argv it parses has no `--rpc` left, see without_rpc_options; LLAMA_ARG_RPC's endpoints
//      are answered from ggml-rpc's cache). An endpoint registered before is asked again, since
//      ggml-rpc answers it from its cache even when its server has gone away;
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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <mutex>
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

// Splits a comma-separated list and keeps empty entries out, so "a:1,,b:2" is two entries
// (RpcServer's device-name list, the `--device` value built below).
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

// Splits a `--rpc` value the way upstream's add_rpc_devices does (string_split): an empty entry is
// kept, and registering it fails, so "a:1," is refused here as the parse refused it.
[[nodiscard]] inline std::vector<std::string> split_rpc_value(const std::string &value) {
    std::vector<std::string> out(1);
    for (char c : value) {
        if (c == ',') {
            out.emplace_back();
        } else {
            out.back().push_back(c);
        }
    }
    return out;
}

// Every endpoint the argv asks for, in order, across repeated `--rpc` options (each one registers
// its servers, so they accumulate in upstream too). argv[0] is treated like any other element. A
// parse applies the environment before the argv, so LLAMA_ARG_RPC's endpoints (`env`, may be null
// or empty: not set) come first. The argv is read by name, not by position: a value of another
// option that is literally "--rpc" is taken for the option (as take_log_file_option does for
// --log-file), which a parse would not do.
[[nodiscard]] inline std::vector<std::string> requested_endpoints(const std::vector<std::string> &argv,
                                                                  const char *env = nullptr) {
    std::vector<std::string> out;
    if (env != nullptr && *env != '\0') {
        out = split_rpc_value(env);
    }
    for (size_t i = 0; i + 1 < argv.size(); ++i) {
        if (argv[i] == "--rpc") {
            for (auto &ep : split_rpc_value(argv[i + 1])) {
                out.push_back(ep);
            }
        }
    }
    return out;
}

// The argv without its `--rpc <servers>` pairs: the argv a parse sees once prepare_argv() has
// registered those servers, so that the parse -- whose `--rpc` handler only registers devices --
// does no network I/O. A `--rpc` without a value is left for the parse to refuse. Read by name, like
// requested_endpoints().
[[nodiscard]] inline std::vector<std::string> without_rpc_options(const std::vector<std::string> &argv) {
    std::vector<std::string> out;
    out.reserve(argv.size());
    for (size_t i = 0; i < argv.size(); ++i) {
        if (argv[i] == "--rpc" && i + 1 < argv.size()) {
            ++i;
            continue;
        }
        out.push_back(argv[i]);
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

// true when the caller chose the multimodal projector's device (or kept it off the GPU) itself.
[[nodiscard]] inline bool has_mmproj_device_option(const std::vector<std::string> &argv) {
    for (const auto &arg : argv) {
        if (arg == "--mmproj-device" || arg == "-mmdev" || arg == "--no-mmproj-offload") {
            return true;
        }
    }
    return false;
}

// The multimodal projector does not go through the device list: when no device is named, clip
// takes the FIRST registered device of type GPU, else IGPU (tools/mtmd/clip.cpp). RPC devices are
// type GPU and are registered after every local backend, so on a host without a GPU a stale RPC
// device is exactly what it would take. Returns the device name to pin, "" for "keep it on the
// CPU", or nullopt when nothing is stale and the argv must be left alone.
[[nodiscard]] inline std::optional<std::string> mmproj_device_override(const std::vector<device_info> &devices,
                                                                       const std::vector<std::string> &requested) {
    auto is_stale = [&requested](const device_info &d) {
        if (d.endpoint.empty()) {
            return false;
        }
        for (const auto &r : requested) {
            if (r == d.endpoint) {
                return false;
            }
        }
        return true;
    };
    bool stale = false;
    for (const auto &d : devices) {
        stale = stale || is_stale(d);
    }
    if (!stale) {
        return std::nullopt;
    }
    for (const auto &d : devices) {
        if (d.is_gpu && !is_stale(d)) {
            return d.name;
        }
    }
    for (const auto &d : devices) {
        if (d.is_igpu && !is_stale(d)) {
            return d.name;
        }
    }
    return std::string();
}

// ---- glue over the ggml registry ------------------------------------------------------------

// ggml's backend registry has no lock, and an RPC registration appends to it. Every registration
// and every read of the registry in this file holds this mutex, so two loads that register servers
// at once do not race on its vectors. It is never held while the engine holds the log lock: the
// engine calls prepare_argv() before it takes that lock. Not covered: llama.cpp's own reads of the
// registry in a load that runs at the same time on another thread (TODO.md).
inline std::recursive_mutex &registry_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

// Resolves an RPC entry point through the RPC backend's proc-address table instead of linking it,
// as upstream's own common/arg.cpp does: with GGML_BACKEND_DL the RPC backend is a module loaded at
// run time, and a direct call would not even link. Every name used here is exported there, the
// two of patches/0015 included. Usage: rpc_proc<decltype(&ggml_backend_rpc_stop_server)>("...").
template <typename Fn> [[nodiscard]] Fn rpc_proc(const char *name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    void *fn = reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
    if (fn == nullptr) {
        throw std::runtime_error(std::string("the RPC backend does not provide ") + name);
    }
    return reinterpret_cast<Fn>(fn);
}

// ggml's RPC devices all belong to the one backend reg named "RPC"; their description is the
// endpoint they were registered for.
[[nodiscard]] inline bool is_rpc_device(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && std::string(ggml_backend_reg_name(reg)) == "RPC";
}

[[nodiscard]] inline std::vector<device_info> registered_devices() {
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
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

// true when ggml-rpc registered the endpoint before in this process (its devices carry it as their
// description); a registration is never undone.
[[nodiscard]] inline bool is_registered(const std::string &endpoint) {
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (is_rpc_device(dev) && endpoint == ggml_backend_dev_description(dev)) {
            return true;
        }
    }
    return false;
}

// Registers one RPC server; throws std::invalid_argument when it cannot be reached (upstream answers
// that with a nullptr since b12283). ggml-rpc answers an endpoint it registered before from its
// cache, without contacting the server, so such an endpoint is asked for its device count
// (patches/0015 puts that function into the proc-address table): a server that has gone away fails
// the load now, instead of the first tensor upload aborting the process.
inline void register_server(const std::string &endpoint) {
    if (endpoint.empty()) {
        throw std::invalid_argument("an empty RPC server endpoint (a ',' too many in --rpc or LLAMA_ARG_RPC?)");
    }
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
    const bool known = is_registered(endpoint);
    ggml_backend_reg_t reg =
        rpc_proc<decltype(&ggml_backend_rpc_add_server)>("ggml_backend_rpc_add_server")(endpoint.c_str());
    if (reg == nullptr ||
        (known && rpc_proc<uint32_t (*)(const char *)>("ggml_backend_rpc_get_device_count")(endpoint.c_str()) == 0)) {
        throw std::invalid_argument("cannot reach RPC server " + endpoint +
                                    " (is an rpc-server / RpcServer listening there?)");
    }
    ggml_backend_register(reg);
}

// See the header comment. Returns the argv to parse: the input, possibly with `--device <list>`
// appended (the caller parses it without its `--rpc` pairs, without_rpc_options). Throws
// std::invalid_argument for an unreachable endpoint of `--rpc` or LLAMA_ARG_RPC. Takes no lock: the
// connects happen here, before the caller's parse.
[[nodiscard]] inline std::vector<std::string> prepare_argv(std::vector<std::string> argv) {
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
    const auto requested = requested_endpoints(argv, std::getenv("LLAMA_ARG_RPC"));
    std::vector<std::string> registered;
    for (const auto &endpoint : requested) {
        bool done = false;
        for (const auto &r : registered) {
            done = done || r == endpoint;
        }
        if (!done) {
            register_server(endpoint);
            registered.push_back(endpoint);
        }
    }
    const auto devices = registered_devices();
    // The parse applies the environment before the argv, so what is appended here overrides a choice
    // the caller made through it as well: LLAMA_ARG_DEVICE is a device list like --device, and
    // LLAMA_ARG_MMPROJ_OFFLOAD off keeps the projector on the CPU like --no-mmproj-offload (on, it
    // names no device, as --mmproj-offload does not, and the projector is pinned as without it).
    const auto env_value = [](const char *name) {
        const char *value = std::getenv(name);
        return value != nullptr ? std::string(value) : std::string();
    };
    std::string mmproj_offload = env_value("LLAMA_ARG_MMPROJ_OFFLOAD");
    std::transform(mmproj_offload.begin(), mmproj_offload.end(), mmproj_offload.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool mmproj_kept_off =
        mmproj_offload == "0" || mmproj_offload == "false" || mmproj_offload == "off" || mmproj_offload == "disabled";
    if (!has_mmproj_device_option(argv) && !mmproj_kept_off) {
        auto mmproj = mmproj_device_override(devices, requested);
        if (mmproj && mmproj->empty()) {
            argv.emplace_back("--no-mmproj-offload");
        } else if (mmproj) {
            argv.emplace_back("--mmproj-device");
            argv.push_back(*mmproj);
        }
    }
    if (has_device_option(argv) || !env_value("LLAMA_ARG_DEVICE").empty()) {
        return argv;
    }
    auto value = device_override(devices, requested);
    if (value) {
        argv.emplace_back("--device");
        argv.push_back(*value);
    }
    return argv;
}

// The same guard for the entry points that build common_params themselves instead of parsing an
// argv (TextToSpeech, LlamaTrainer): they never ask for an RPC server, so every registered one is
// stale. Without it a TTS model loaded after an RPC load in the same JVM offloads to that server --
// and aborts the process once it is gone (seen in CI: RpcIntegrationTest, then TtsIntegrationTest).
// Leaves an explicit params.devices alone. Returns what the multimodal projector should use, in
// mmproj_device_override's terms; the caller applies it to its own mtmd/clip parameters.
inline std::optional<std::string> exclude_stale_devices(std::vector<ggml_backend_dev_t> &params_devices) {
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
    const auto devices = registered_devices();
    if (params_devices.empty()) {
        auto value = device_override(devices, {});
        if (value) {
            if (*value != "none") {
                for (const auto &name : split_endpoints(*value)) {
                    params_devices.push_back(ggml_backend_dev_by_name(name.c_str()));
                }
            }
            params_devices.push_back(nullptr); // the list is null-terminated, as parse_device_list builds it
        }
    }
    return mmproj_device_override(devices, {});
}

// Devices an in-process RPC server offers: every accelerator, else the CPU (upstream rpc-server's
// default), but never an RPC device -- serving a remote device back out would forward to itself.
[[nodiscard]] inline std::vector<ggml_backend_dev_t> server_devices() {
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
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
    std::lock_guard<std::recursive_mutex> lock(registry_mutex());
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

// ---- test seam (defined in jllama_api.cpp) ---------------------------------------------------

// Called by jllama_rpc_server_serve() on the serving thread after the devices are selected and
// before ggml's server starts; nullptr (the default) for none. Only test_c_api.cpp sets it: it is
// the one way to issue a stop deterministically before the socket is bound. Not exported.
void set_serve_hook_for_testing(void (*hook)());

// Test seam of jllama_rpc_server_stop(): called on the stopping thread once the stop is recorded
// for the serves in progress and before it waits for them, so a test can start the next serve
// right there. nullptr clears it. Not exported; the C++ tests reach it, nothing else does.
void set_stop_hook_for_testing(void (*hook)());

} // namespace jllama::rpc
