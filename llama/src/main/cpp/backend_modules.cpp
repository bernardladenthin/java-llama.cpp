// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "backend_modules.hpp"

#include "ggml-backend.h"
#include "log.h"

#ifdef GGML_BACKEND_DL
// load_backends_next_to_this_library() needs this library's own path: dladdr on POSIX,
// GetModuleHandleExW + GetModuleFileNameW on Windows. On Android the modules are loaded by soname
// instead (load_backends_by_soname, JLLAMA_MODULES_BY_SONAME), through dlopen/dlsym.
#ifdef JLLAMA_MODULES_BY_SONAME
#include "jllama_modules.h"
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX // or windows.h's min/max macros break std::min / std::max below
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include <mutex>
#include <string>
#include <vector>

namespace {

#ifdef GGML_BACKEND_DL
// What the process ended up with after the modules were loaded: a module whose runtime is missing
// fails silently (ggml logs it at GGML_LOG_DEBUG only, and on Windows SEM_FAILCRITICALERRORS
// suppresses the dialog), so say which backends ggml registered and which devices they report.
// Reaches LlamaModel.setLogger.
static void log_backend_registry(const char *where, const std::string &from) {
    std::string devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        devices += (i == 0 ? "" : ", ");
        devices += ggml_backend_dev_name(dev);
        devices += " (";
        devices += ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        devices += ")";
    }
    std::string backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        backends += (i == 0 ? "" : ", ");
        backends += ggml_backend_reg_name(ggml_backend_reg_get(i));
    }
    LOG_INF("%s: %zu backend(s) registered from %s [%s], %zu device(s): %s\n", where, ggml_backend_reg_count(),
            from.c_str(), backends.c_str(), ggml_backend_dev_count(), devices.c_str());
}

#ifdef JLLAMA_MODULES_BY_SONAME
// Android: an app's native libraries stay inside the APK (extractNativeLibs=false is the default from
// API 23 on), so there is no directory to scan -- dladdr() reports a path into the zip, and ggml's
// directory iteration finds nothing. bionic resolves a bare soname against the app's own libraries
// (the same lookup System.loadLibrary uses), so the modules are opened by name instead: the CPU
// variants of this build (generated into jllama_modules.h at configure time) are scored the way
// ggml_backend_load_best() scores them and the best one is registered, every other module of the
// build (ggml-rpc) and every GPU module another AAR may add next to this one (the fixed list below)
// is registered when it is there. A missing one is silent, as a missing module is on the desktop.
static int score_module_by_soname(const char *soname) {
    void *handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return -1;
    }
    int score = 1; // ggml treats a module without a score function as always usable
    using score_fn_t = int (*)();
    if (auto score_fn = reinterpret_cast<score_fn_t>(dlsym(handle, "ggml_backend_score"))) {
        score = score_fn();
    }
    dlclose(handle);
    return score;
}

static void load_backends_by_soname() {
    const std::vector<const char *> cpu_modules = JLLAMA_CPU_MODULE_SONAMES;
    const std::vector<const char *> other_modules = JLLAMA_OTHER_MODULE_SONAMES;
    static const char *const gpu_modules[] = {"libggml-opencl.so", "libggml-vulkan.so"};
    const char *best = nullptr;
    int best_score = 0;
    for (const char *soname : cpu_modules) {
        const int score = score_module_by_soname(soname);
        if (score > best_score) {
            best = soname;
            best_score = score;
        }
    }
    if (best != nullptr) {
        ggml_backend_load(best);
    } else {
        LOG_ERR("%s: none of the %zu CPU backend module(s) of this build is usable on this device\n", __func__,
                cpu_modules.size());
    }
    for (const char *soname : other_modules) {
        if (score_module_by_soname(soname) > 0) {
            ggml_backend_load(soname);
        }
    }
    for (const char *soname : gpu_modules) {
        if (score_module_by_soname(soname) > 0) {
            ggml_backend_load(soname);
        }
    }
    log_backend_registry(__func__, "the APK (by soname)");
}
#else
// With GGML_BACKEND_DL (JLLAMA_BACKEND_DL) the CPU and RPC backends -- and every GPU backend whose
// natives jar is on the classpath -- are modules ggml loads at run time, and ggml_backend_load_all()
// looks next to the executable -- `java`, here. LlamaLoader extracts the modules next to this
// library, so they are loaded from this library's own directory before anything touches the backend
// registry. ggml_backend_load_best() picks the CPU module whose instruction set the running CPU
// supports best, and loads the GPU modules in its fixed order (cuda, hip, sycl, vulkan, opencl,
// openvino): one whose vendor runtime is missing fails to open and registers nothing.
static void load_backends_next_to_this_library() {
#ifdef _WIN32
    // GetModuleHandleExW on an address inside this DLL -- GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT
    // so the handle needs no release. The path must reach ggml as UTF-8: it opens the modules with
    // the narrow CRT, and a directory holding a non-ASCII character (an umlaut in a Windows user
    // name reaches the temp directory LlamaLoader extracts into) loaded nothing at all when the
    // path was handed over in the active ANSI code page -- without any error being logged.
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&load_backends_next_to_this_library), &self)) {
        return;
    }
    std::wstring wpath(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(self, wpath.data(), static_cast<DWORD>(wpath.size()));
        if (n == 0) {
            return;
        }
        if (n < wpath.size()) {
            wpath.resize(n);
            break;
        }
        if (wpath.size() > 32768) { // the documented ceiling for an extended-length path
            return;
        }
        wpath.resize(wpath.size() * 2);
    }
    const size_t sep = wpath.find_last_of(L"\\/");
    const std::wstring wdir = sep == std::wstring::npos ? std::wstring(L".") : wpath.substr(0, sep);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) {
        return;
    }
    std::string dir(static_cast<size_t>(bytes - 1), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), -1, dir.data(), bytes, nullptr, nullptr) <= 0) {
        return;
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&load_backends_next_to_this_library), &info) == 0 ||
        info.dli_fname == nullptr) {
        return;
    }
    std::string dir(info.dli_fname);
    const size_t slash = dir.find_last_of('/');
    dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
#endif
    ggml_backend_load_all_from_path(dir.c_str());
    log_backend_registry(__func__, dir);
}
#endif // JLLAMA_MODULES_BY_SONAME
#endif // GGML_BACKEND_DL

} // namespace

namespace jllama {

void load_backend_modules() {
    static std::once_flag once;
    std::call_once(once, []() {
#ifdef JLLAMA_MODULES_BY_SONAME
        load_backends_by_soname();
#elif defined(GGML_BACKEND_DL)
        load_backends_next_to_this_library();
#endif
    });
}

} // namespace jllama
