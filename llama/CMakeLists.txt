cmake_minimum_required(VERSION 3.22)

# The oldest macOS the dylib loads on. Without it the linker takes the build host's own version, so
# moving a macOS job to a newer runner image would silently raise the floor for every user. Must be
# set before the first project(); a -DCMAKE_OSX_DEPLOYMENT_TARGET=... or MACOSX_DEPLOYMENT_TARGET
# from outside still wins. Ignored on every other platform. Checked by .github/smoke-native-macos.sh.
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version of libjllama.dylib")
endif()

project(jllama CXX)

# HYBRID CRT on Windows, for every toolchain but the two named below: the STL and vcruntime are
# linked statically (/MT),
# the UCRT stays dynamic. So msvcp140.dll / vcruntime140.dll / vcruntime140_1.dll are not required
# on the end-user's machine, while ucrtbase.dll comes from the OS.
# Must be set before any FetchContent_MakeAvailable() so that llama.cpp and all
# other subprojects inherit the same CRT choice (mixing /MT and /MD in a single
# link is a linker error).
#
# WHY NOT A PLAIN /MT, which would need no UCRT at all: /MT links all THREE layers statically,
# the UCRT included, which freezes a copy of it inside jllama.dll -- so Microsoft's UCRT security
# updates, which Windows Update delivers through ucrtbase.dll, never reach a shipped artifact
# until we rebuild and release. For a library that parses network data (cpp-httplib, JSON) and
# untrusted files (GGUF) that is the deciding argument, not the 0.3 MB the hybrid also saves
# (measured: 10.1 MB with /MT, 9.8 MB hybrid). It is Microsoft's own recommendation for
# self-contained binaries, and it makes both toolchains use ONE CRT strategy instead of two.
# The cost is one requirement, which README.md states: **Windows 10 or newer**, where the UCRT is
# an OS component. `nativedeps.py` guards this in BOTH directions for every Windows CPU directory:
# msvcp140/vcruntime140 appearing fails the build (the static half went dynamic), and the
# api-ms-win-crt-* forwarders disappearing fails it too (the dynamic half went static) -- the second
# matters because a static CRT only ever REMOVES imports, so losing one of the linker flags below
# would otherwise pass unnoticed.
#
# EXCEPTION: the Intel oneAPI SYCL and OpenVINO backends must use the DYNAMIC /MD
# runtime — `icx -fsycl` rejects /MT outright ("invalid argument 'MT' not allowed
# with '-fsycl'") and the OpenVINO import libraries are built /MD (mixing would be a
# link error). Those classifiers already require the vendor runtime on the host, so
# the self-contained-DLL rationale does not apply to them; the CPU + CUDA/Vulkan/OpenCL
# classifiers keep /MT.
#
# PLAIN CLANG on Windows needs the HYBRID CRT instead, and gets it from the second branch.
# CMake's `MSVC` is TRUE for cl.exe and for clang-cl, but FALSE for clang with the GNU driver
# (upstream's cmake/x64-windows-llvm.cmake, the only toolchain that yields all 14 CPU variants --
# see CLAUDE.md "CPU variants"), so without that branch such a build silently links the DLL
# runtime and needs msvcp140.dll / vcruntime140.dll after all. Three facts, all measured on
# 2026-10-09 (clang 23.1.3, MSVC 14.51.36231 headers):
#   * The GNU driver does NOT ignore CMAKE_MSVC_RUNTIME_LIBRARY -- it emits -D_DLL -D_MT from it,
#     which is why passing it alone fails to link (undefined __declspec(dllimport) expf). The flag
#     that actually selects the libraries is -fms-runtime-lib.
#   * A FULLY static CRT (static UCRT too) combined with shared libraries -- which
#     JLLAMA_BACKEND_DL requires, since GGML_BACKEND_DL needs BUILD_SHARED_LIBS -- gives every
#     DLL its own CRT heap; such a build does not even start (and crashed with 0xC0000409 in an
#     earlier attempt).
#   * Hybrid CRT (static STL + vcruntime, dynamic UCRT -- Microsoft's own recommendation for
#     self-contained binaries) leaves the complete jllama.dll with NO non-OS dependency at all:
#     KERNEL32, ADVAPI32, SHELL32, WS2_32 and the api-ms-win-crt-* forwarders, nothing else.
#     ucrtbase.dll is an OS component from Windows 10 on and is not the DLL behind the
#     msvcp140 incident in README.md.
if(MSVC AND NOT GGML_SYCL AND NOT GGML_OPENVINO)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    # Keep the UCRT dynamic: drop the static import /MT pulls in, add the DLL one.
    add_link_options(
        $<IF:$<CONFIG:Debug>,/NODEFAULTLIB:libucrtd.lib,/NODEFAULTLIB:libucrt.lib>
        $<IF:$<CONFIG:Debug>,/DEFAULTLIB:ucrtd.lib,/DEFAULTLIB:ucrt.lib>)
elseif(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT GGML_SYCL AND NOT GGML_OPENVINO)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    add_compile_options($<IF:$<CONFIG:Debug>,-fms-runtime-lib=static_dbg,-fms-runtime-lib=static>)
    # Same hybrid, through the GNU driver: these have to reach the linker, not clang.
    add_link_options(
        "SHELL:-Xlinker $<IF:$<CONFIG:Debug>,/nodefaultlib:libucrtd.lib,/nodefaultlib:libucrt.lib>"
        "SHELL:-Xlinker $<IF:$<CONFIG:Debug>,ucrtd.lib,ucrt.lib>")
endif()

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
# Android NDK only declares posix_spawn_file_actions_t as a type alias but
# does not implement the posix_spawn_* functions; subprocess.h (pulled in by
# server-tools.cpp) uses them and fails to compile.  The server tools are not
# needed by the jllama JNI library, so skip them on Android.
# Must use CACHE BOOL FORCE to override llama.cpp's own option() defaults.
if(ANDROID_ABI)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
endif()

option(LLAMA_VERBOSE	"llama: verbose output"		OFF)

# Declared here, before any FetchContent_MakeAvailable(), and not next to the tests below: a
# subproject that calls include(CTest) creates BUILD_TESTING as a cache variable defaulting to ON,
# after which this option() would be a no-op. CCCL does exactly that, unconditionally, and ggml
# fetches it whenever GGML_CUDA_CCCL_VERSION is set (our CUDA builds pin v3.4.3) -- the CUDA job
# then built jllama_test, whose gtest discovery needs libcuda.so.1 and failed the build on the
# GPU-less runner. -DBUILD_TESTING=ON still enables the tests.
option(BUILD_TESTING "Build C++ unit tests for jni_helpers / json_helpers / utils" OFF)

# Modular natives (CLAUDE.md "Modular natives"): ggml is built the way upstream builds its own
# release binaries -- GGML_BACKEND_DL, with GGML_CPU_ALL_VARIANTS where upstream has a variant list
# -- so the CPU natives directory holds libjllama plus libggml/libggml-base and the backend modules
# ggml loads at run time: one CPU module per instruction-set level (x64, sse42, sandybridge ...
# zen4, sapphirerapids on x86-64; armv8.0_1 ... armv9.2_2 on aarch64), of which
# ggml_backend_load_best picks the best for the running CPU, and ggml-rpc. A single static build can
# not do that: it targets the Haswell baseline below, so it crashes with SIGILL on an older CPU and
# leaves AVX-512/VNNI/AMX unused on a newer one. Only ggml becomes shared; llama, mtmd and
# llama-common stay static inside libjllama.
#
# JLLAMA_MODULE_ONLY is the other half: a GPU natives jar holds ONLY its backend module
# (libggml-cuda.so, ggml-vulkan.dll, ...), built against the same ggml with GGML_CPU=OFF, and
# LlamaLoader puts it next to the CPU jar's libraries, where ggml finds it. Exactly upstream's
# Windows release layout (win-cpu + win-cuda holding one ggml-cuda.dll). Both OFF by default: a plain
# `cmake -B build` is the static single-library build (macOS/Metal and s390x ship that way).
option(JLLAMA_BACKEND_DL "Build ggml as shared libraries and the backends as modules ggml loads at run time (GGML_BACKEND_DL)" OFF)
option(JLLAMA_MODULE_ONLY "Build only the GPU backend module of the one GGML_<backend> flag set, for a GPU natives jar (implies JLLAMA_BACKEND_DL)" OFF)
if(JLLAMA_MODULE_ONLY)
    set(JLLAMA_BACKEND_DL ON)
endif()

#################### json ####################

FetchContent_Declare(
	json
	GIT_REPOSITORY https://github.com/nlohmann/json
	GIT_TAG        v3.12.0
)
FetchContent_MakeAvailable(json)

#################### llama.cpp ####################

# Android bionic gates several libc declarations behind a minimum API level:
#   >= 24: getifaddrs/freeifaddrs (<ifaddrs.h>), needed by cpp-httplib.
#   >= 28: posix_spawn / posix_spawnp / posix_spawn_file_actions_* (<spawn.h>),
#          needed by mtmd-helper.cpp (vendor/sheredom/subprocess.h).
# The symbols exist in libc.so at all API levels; bionic only hides the
# *declarations* below the introducing API via:
#     #if __BIONIC_AVAILABILITY_GUARD(api)        // <android/versioning.h>
# which expands to (__ANDROID_MIN_SDK_VERSION__ >= api) UNLESS
# __ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__ is defined, in which case it is
# always 1 (all declarations visible, newer-API symbols become weak references
# resolved at load time — present on every API-28+ device we target).
#
# We define __ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__ rather than bumping
# __ANDROID_API__ because the dockcross-android-arm64 image used by CI does NOT
# use the Google NDK CMake toolchain: it is a Debian-style cross-clang
# (/usr/aarch64-linux-android/bin/clang) that (a) never sets the ANDROID /
# ANDROID_ABI CMake variables and (b) ignores -DANDROID_PLATFORM (CMake reports
# it as an unused variable). On that toolchain clang predefines __ANDROID_API__
# from the baked-in target triple, so -D__ANDROID_API__=28 would only clash with
# the builtin (a -Wmacro-redefined) without moving __ANDROID_MIN_SDK_VERSION__.
# The weak-symbols macro is never compiler-predefined, so defining it is clean
# and deterministically exposes the guarded declarations. It also covers the
# getifaddrs (API 24) case, replacing the old __ANDROID_API__ bump entirely.
#
# Detection must not rely on ANDROID_ABI (unset by this toolchain). OS_NAME is
# passed as -DOS_NAME=Linux-Android by the CI cmake invocation and is already
# used the same way at the server-models.cpp exclusion below; we also accept the
# NDK-style ANDROID/ANDROID_ABI signals so a future switch back to the NDK
# toolchain keeps working.
if(ANDROID OR ANDROID_ABI OR OS_NAME MATCHES "Android" OR CMAKE_CXX_COMPILER MATCHES "android")
    add_compile_definitions(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)
    # 16 KB page-size compliance (Google Play requirement for apps targeting
    # Android 15+): every LOAD segment of the shipped .so must be aligned to a
    # multiple of 16384. The dockcross cross-linker currently emits 0x4000
    # alignment by default, but that is a toolchain default, not a guarantee —
    # pin it explicitly so a toolchain image bump cannot silently regress Play
    # compatibility. CI asserts the resulting alignment with readelf.
    add_link_options(-Wl,-z,max-page-size=16384)
    # The shipped .so must dlopen with ONLY bionic system libraries present —
    # an app that consumes the AAR bundles no other native libs. The dockcross
    # cross-clang, left alone, emits two DT_NEEDED entries that do not exist on
    # any Android device and made System.loadLibrary fail with
    # UnsatisfiedLinkError (caught on the API-30 emulator CI job; readelf on the
    # released 5.0.5 arm64 lib shows the same latent defect):
    #   - libomp.so: ggml's OpenMP backend links LLVM's libomp, which Android
    #     does not ship. Turn OpenMP off; ggml falls back to its own std::thread
    #     pool (same trade the Windows-arm64 clang-cl job makes with
    #     -DGGML_OPENMP=OFF, and what the Google NDK toolchain does by default).
    #   - libc++_shared.so: the NDK's shared C++ runtime, only present when an
    #     app packages it itself. Link libc++ statically instead (upstream
    #     llama.cpp's Android builds do the same); bionic system libs are the
    #     only remaining DT_NEEDED. CI enforces this with a whitelist check on
    #     the AAR's .so files.
    set(GGML_OPENMP OFF CACHE BOOL "Android: no libomp.so on devices; use ggml's std::thread pool" FORCE)
    add_link_options(-static-libstdc++)
endif()

set(LLAMA_BUILD_COMMON ON)

# HTTPS -- model downloads from https:// URLs (common/download.cpp: --model-url, --hf-repo) and the
# embedded server's --ssl-key-file / --ssl-cert-file -- comes from BoringSSL, which upstream's
# vendor/cpp-httplib/CMakeLists.txt fetches at a pinned tag and links STATICALLY, on every platform
# that ships a desktop jar. A shipped library must not depend on an OpenSSL the target machine may
# not have: the macOS build used to pick up the runner's Homebrew openssl@3 and failed to load on a
# Mac without it, and the manylinux images have no OpenSSL that upstream's version check accepts, so
# the Linux jars had no HTTPS at all. Upstream's own release builds pass the same flag (macos-cpu,
# windows-*, android-*). jllama links upstream's cpp-httplib target (further down) instead of
# compiling httplib.cpp a second time, so the download code and the embedded server see ONE copy of
# cpp-httplib built with ONE setting. Off where BoringSSL is not built here: Android (the dockcross
# cross-clang, and the AAR has no download use case), s390x (big-endian is outside BoringSSL's
# supported set) and the Linux SYCL builds (icx/icpx, compiler id IntelLLVM, which BoringSSL's CMake
# treats as neither GCC nor Clang; upstream's ubuntu-24-sycl release job passes LLAMA_OPENSSL=OFF too,
# while its windows-sycl job builds BoringSSL -- as this file did before). LLAMA_OPENSSL=OFF there
# keeps find_package from linking a system OpenSSL by accident, which verify-native-deps.py would
# reject.
if(ANDROID_ABI OR OS_NAME MATCHES "Android" OR OS_ARCH STREQUAL "s390x" OR CMAKE_SYSTEM_PROCESSOR MATCHES "s390x"
   OR (GGML_SYCL AND NOT WIN32))
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_BORINGSSL OFF CACHE BOOL "" FORCE)
else()
    set(LLAMA_BUILD_BORINGSSL ON CACHE BOOL "" FORCE)
endif()

# Instruction-set policy: target the "Haswell" baseline (x86-64-v3).
#
# This set of flags matches GGML's own "haswell" named variant in
# GGML_CPU_ALL_VARIANTS and covers every x86-64 CPU since:
#   - Intel Haswell   (2013)
#   - AMD Ryzen / EPYC (2017)
#
# GGML_NATIVE is OFF so the build never probes the build machine's CPU.
# Without this, MSVC runs FindSIMD.cmake which shadow-sets GGML_AVX512=ON
# via a local variable that bypasses our CACHE FORCE, and GCC/Clang uses
# -march=native which embeds whatever the build machine supports.
#
# The individual flags are set explicitly because with GGML_NATIVE=OFF
# they all default to OFF.  On MSVC the elseif chain in ggml-cpu cmake
# picks the highest level (/arch:AVX2) and bundles FMA + F16C defines
# automatically; SSE42, AVX, FMA, F16C have no additional effect there
# but are needed for GCC/Clang where each flag independently adds its
# -m flag and GGML_* preprocessor define.
#
# BMI2 is enabled on 64-bit targets only.  MSVC's 32-bit (x86) compiler
# does not expose __pdep_u64 / __pext_u64; ggml's quants.c uses them in
# _ggml_vec_dot_iq1_m_q8_K, causing an unresolved-external link error.
# GCC/Clang on x86 can lower the 64-bit intrinsics to two 32-bit ops,
# but disabling BMI2 entirely is safer and consistent across compilers.
#
# AVX-512 stays OFF:
#   - Many CPUs lack it (AMD EPYC 7763, all Intel desktop since Alder Lake).
#   - MSVC's /arch:AVX512 applies to the entire TU — no per-function gating.
#   - Frequency throttling and power draw make it a net loss for bursty work.
set(GGML_NATIVE  OFF CACHE BOOL "" FORCE)
if(NOT JLLAMA_BACKEND_DL)
set(GGML_SSE42   ON  CACHE BOOL "" FORCE)
set(GGML_AVX     ON  CACHE BOOL "" FORCE)
set(GGML_AVX2    ON  CACHE BOOL "" FORCE)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(GGML_BMI2 ON  CACHE BOOL "" FORCE)
else()
    set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
endif()
set(GGML_FMA     ON  CACHE BOOL "" FORCE)
set(GGML_F16C    ON  CACHE BOOL "" FORCE)
set(GGML_AVX512  OFF CACHE BOOL "" FORCE)
endif()
# RPC backend (ggml-rpc): lets a model offload layers to rpc-servers on other machines
# (--rpc host:port) and lets this library serve its own devices (RpcServer). It is a static
# library linked into jllama like every other ggml backend, so the shipped artifact stays ONE
# jllama library. TCP only: plain BSD sockets / Winsock, which jllama already links (cpp-httplib
# pulls in ws2_32), so no new runtime dependency on any platform.
#
# GGML_RPC_RDMA must be forced OFF, not left to upstream's default: ggml-rpc's CMakeLists
# switches it ON whenever the BUILD host has libibverbs (Linux) or librdma (Apple), which would
# make libjllama.so NEED libibverbs.so.1 at load time -- it would then fail to load on every
# machine without rdma-core. libibverbs cannot be linked statically in any useful way either
# (it dlopens its hardware providers), and upstream notes the Apple weak link does not survive a
# static ggml-rpc. CI asserts the resulting DT_NEEDED set, so a toolchain image that grows
# rdma-core cannot turn this back on unnoticed.
set(GGML_RPC      ON  CACHE BOOL "" FORCE)
set(GGML_RPC_RDMA OFF CACHE BOOL "" FORCE)
# b9305 removed the top-level LLAMA_BUILD_WEBUI -> LLAMA_BUILD_UI shim; set the
# new name directly. (The old name no longer forwards at top level; the shim
# survives in tools/ui/CMakeLists.txt but that subdir is not configured in
# FetchContent mode, so the old setting would be inert anyway.)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
# b9284 flipped LLAMA_BUILD_APP default to ON; we don't build the unified binary
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
# Local source patches for the fetched llama.cpp tree. Every patches/*.patch|*.diff is applied
# (sorted, idempotently, fail-loud) by cmake/apply-llama-patches.cmake — see that file's header.
# This runs for every C++ build (all CI jobs + local) from one place. <SOURCE_DIR> is substituted
# by FetchContent/ExternalProject to the fetched llama.cpp source root.
# With JLLAMA_BACKEND_DL, SOURCE_SUBDIR names a directory that does not exist, so
# FetchContent_MakeAvailable only fetches and patches the source; ggml and llama.cpp are then added
# below in two steps (llama.cpp skips its own add_subdirectory(ggml) when a ggml target exists), and a
# JLLAMA_MODULE_ONLY build adds ggml alone.
if(JLLAMA_BACKEND_DL)
    set(JLLAMA_LLAMA_SOURCE_SUBDIR SOURCE_SUBDIR jllama-added-below)
endif()
FetchContent_Declare(
	llama.cpp
	GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
	GIT_TAG        b11538
	PATCH_COMMAND  ${CMAKE_COMMAND}
		-DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
		-DLLAMA_SRC=<SOURCE_DIR>
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-llama-patches.cmake
	${JLLAMA_LLAMA_SOURCE_SUBDIR}
)
FetchContent_MakeAvailable(llama.cpp)

# Every ggml source allocating with ggml_aligned_malloc must also release with ggml_aligned_free.
# The AMX buffer type did not, which corrupted the Windows heap on AMX-capable CPUs and killed the
# process at cleanup; patches/0018 fixes it. Checked here on every platform in every build,
# because no machine without AMX can notice the defect at run time.
include(cmake/check-aligned-free-pairing.cmake)
jllama_check_aligned_free_pairing(${llama.cpp_SOURCE_DIR})
if(JLLAMA_BACKEND_DL)
    # Where ggml can be loaded as modules here (CLAUDE.md "Modular natives"): Linux x86-64 / aarch64
    # (the dockcross Android toolchain reports Linux too, see the Android block above) and Windows
    # x86-64 / arm64. CPU variants wherever ggml has a list for the OS/arch: x86-64 everywhere, ARM on
    # Linux/Android (ggml-cpu/CMakeLists.txt has no ARM list for Windows, so Windows arm64 gets the
    # one ggml-cpu module upstream ships there too). macOS stays static (Metal is compiled in, one
    # CPU generation per Metal build), s390x has no variants upstream and no GPU.
    set(_jllama_dl_ok OFF)
    set(_jllama_all_variants OFF)
    if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android)$" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64)$")
        set(_jllama_dl_ok ON)
        set(_jllama_all_variants ON)
    elseif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_jllama_dl_ok ON)
        set(_jllama_all_variants ON)
    elseif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|aarch64|arm64)$")
        set(_jllama_dl_ok ON)
    endif()
    if(NOT _jllama_dl_ok)
        message(FATAL_ERROR "JLLAMA_BACKEND_DL is supported on Linux/Android x86-64 and aarch64 and on Windows x86-64 and arm64 only")
    endif()
    if(BUILD_TESTING)
        # jllama_test calls the CPU and RPC backends directly (ggml_backend_cpu_init, ...), which are
        # loadable modules here, not link-time libraries.
        message(FATAL_ERROR "JLLAMA_BACKEND_DL cannot be combined with BUILD_TESTING")
    endif()
    if(JLLAMA_MODULE_ONLY)
        # One GPU backend, nothing else: no CPU backend (GGML_CPU=OFF, as upstream's windows-vulkan
        # release job builds its module) and no ggml-rpc -- both come with the CPU jar. The module
        # is the only shipped file; ggml and ggml-base are built because the module links them, and
        # are discarded (the CPU jar's copies are the ones loaded next to the module).
        set(_jllama_gpu_flags)
        foreach(flag GGML_CUDA GGML_HIP GGML_SYCL GGML_VULKAN GGML_OPENCL GGML_OPENVINO)
            if(${flag})
                list(APPEND _jllama_gpu_flags ${flag})
            endif()
        endforeach()
        list(LENGTH _jllama_gpu_flags _jllama_gpu_count)
        if(NOT _jllama_gpu_count EQUAL 1)
            message(FATAL_ERROR "JLLAMA_MODULE_ONLY needs exactly one of GGML_CUDA, GGML_HIP, GGML_SYCL, GGML_VULKAN, GGML_OPENCL, GGML_OPENVINO (got: '${_jllama_gpu_flags}')")
        endif()
        set(GGML_CPU OFF CACHE BOOL "" FORCE)
        set(GGML_RPC OFF CACHE BOOL "" FORCE)
        set(_jllama_all_variants OFF)
    elseif(_jllama_all_variants AND WIN32 AND MSVC)
        # Only PLAIN clang produces the full x86 set here: CMake sets MSVC for clang-cl as for
        # cl.exe, and ggml's `if (NOT MSVC)` then drops ivybridge, piledriver, cooperlake, zen4 and
        # sapphirerapids -- a Zen 4/5 machine would fall back to icelake and lose the BF16 kernels.
        # clang-cl additionally fails to build alderlake (/arch:AVX2 + __AVXVNNI__ without
        # -mavxvnni, an upstream gap). Fail loud rather than silently shipping 9 variants. A
        # module-only build is exempt: it builds no CPU variant (upstream builds its CUDA module
        # with cl.exe), and so is Windows arm64 (one module, clang-cl as the arm64 job uses).
        message(FATAL_ERROR
            "The CPU variants on Windows x86-64 need plain clang (the GNU driver), not cl.exe or "
            "clang-cl: ggml's `if (NOT MSVC)` block drops 5 of the 14 x86 variants, and "
            "clang-cl cannot build alderlake. Use upstream's cmake/x64-windows-llvm.cmake.")
    endif()
    set(BUILD_SHARED_LIBS ON)
    set(GGML_BACKEND_DL       ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ${_jllama_all_variants} CACHE BOOL "" FORCE)
    # The ggml defaults llama.cpp's own CMakeLists changes before its add_subdirectory(ggml): every
    # `set(GGML_*_DEFAULT ...)` there ("change the default for these ggml options") must be
    # repeated here, because adding ggml first skips that block entirely. Re-check on every bump
    # with `grep -n _DEFAULT CMakeLists.txt` at the new tag. Missing GGML_CUDA_GRAPHS_DEFAULT cost a
    # CUDA module built this way two thirds of its token generation (measured, TODO.md "GPU
    # backends as modules"): ggml's own default is OFF, llama.cpp turns it ON for itself.
    set(GGML_LLAMAFILE_DEFAULT   ON)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
    add_subdirectory(${llama.cpp_SOURCE_DIR}/ggml ${llama.cpp_BINARY_DIR}/ggml)
    set(BUILD_SHARED_LIBS OFF)
    # Unversioned: LlamaLoader extracts plain files, not the libggml.so -> .so.0 -> .so.0.x.y
    # symlink chain, so the file name, the SONAME and every DT_NEEDED entry naming it -- libjllama's
    # and each module's -- must be one name. Set in both build kinds for that reason: a module built
    # here records `libggml-base.so`, which is what the CPU jar's library is called.
    set_property(TARGET ggml ggml-base PROPERTY VERSION)
    set_property(TARGET ggml ggml-base PROPERTY SOVERSION)
    if(NOT JLLAMA_MODULE_ONLY)
        add_subdirectory(${llama.cpp_SOURCE_DIR} ${llama.cpp_BINARY_DIR})
    endif()
endif()

# ROCm/HIP: compress the embedded GPU code. Without it every HIP translation unit carries one
# UNcompressed code object per GPU target, so the shipped library holds the complete kernel set
# (flash attention, mmq per quant type, ...) once per architecture: ~1 GB for the 23 Windows
# targets, which LlamaLoader then extracts to the temp dir on every start. --offload-compress
# makes clang store each bundle zstd-compressed (a "CCOB" bundle); the HIP runtime inflates it
# when the module is loaded. Scoped to the ggml-hip target (the only one with device code):
# its sources are LANGUAGE HIP on Linux and CXX on Windows (upstream compiles HIP as C++ there).
# CI (.github/verify-hip-offload-compressed.py) fails the ROCm jobs if an uncompressed bundle
# ever reappears.
if(TARGET ggml-hip)
    target_compile_options(ggml-hip PRIVATE $<$<COMPILE_LANGUAGE:HIP,CXX>:--offload-compress>)
endif()

# mtmd lives in tools/mtmd, which is only built when LLAMA_BUILD_TOOLS=ON.
# LLAMA_BUILD_TOOLS defaults to LLAMA_STANDALONE, which is OFF when llama.cpp
# is consumed via FetchContent.  Build mtmd explicitly so the target exists.
if(NOT JLLAMA_MODULE_ONLY AND NOT TARGET mtmd)
    # LLAMA_VERSION_BASE / LLAMA_VERSION_MAJOR are set inside llama.cpp's directory
    # scope and are not visible here.  tools/mtmd/CMakeLists.txt uses them in
    # set_target_properties() as VERSION/SOVERSION values; if a variable is empty the
    # token list after PROPERTIES becomes odd-length and CMake aborts with "incorrect
    # number of arguments".  Both are cosmetic for this build (BUILD_SHARED_LIBS is
    # OFF, so mtmd is a static library and VERSION/SOVERSION are no-ops), hence the
    # neutral placeholder rather than a mirror of upstream's numbers.
    # (b10398 replaced the older LLAMA_INSTALL_VERSION with these two — upstream
    # #26839, "cmake: introduce semantic versioning".)
    if(NOT DEFINED LLAMA_VERSION_BASE)
        set(LLAMA_VERSION_BASE "0")
    endif()
    if(NOT DEFINED LLAMA_VERSION_MAJOR)
        set(LLAMA_VERSION_MAJOR "0")
    endif()
    add_subdirectory(${llama.cpp_SOURCE_DIR}/tools/mtmd ${llama.cpp_BINARY_DIR}/tools/mtmd)
endif()

# Workaround: server-common.h (included transitively by llama-cli) includes
# mtmd.h, but the mtmd include path is not propagated to llama-cli consumers.
if(TARGET llama-cli)
    target_include_directories(llama-cli PRIVATE ${llama.cpp_SOURCE_DIR}/tools/mtmd)
endif()
if(TARGET server-context)
    target_include_directories(server-context PUBLIC ${llama.cpp_SOURCE_DIR}/tools/mtmd)
endif()

#################### jllama ####################

# find which OS we build for if not set (make sure to run mvn compile first)
if(NOT DEFINED OS_NAME)
    if(ANDROID_ABI)
        set(OS_NAME "Android")
    else()
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
	execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes net.ladenthin.llama.loader.OSInfo --os
      OUTPUT_VARIABLE OS_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
endif()
if(NOT OS_NAME)
    message(FATAL_ERROR "Could not determine OS name")
endif()

# find which architecture we build for if not set  (make sure to run mvn compile first)
if(NOT DEFINED OS_ARCH)
    if(ANDROID_ABI)
        set(OS_ARCH ${ANDROID_ABI})
    else()
    find_package(Java REQUIRED)
    find_program(JAVA_EXECUTABLE NAMES java)
    execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes net.ladenthin.llama.loader.OSInfo --arch
      OUTPUT_VARIABLE OS_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
endif()
if(NOT OS_ARCH)
    message(FATAL_ERROR "Could not determine CPU architecture")
endif()

# Native-library output tree: src/main/natives/net/ladenthin/llama/<OS>/<ARCH>/<backend>/.
# Every build writes into its own backend directory, so all builds of one pipeline share one
# tree without overwriting each other, and each directory becomes exactly one natives jar
# (classifier <backend>-<os>-<arch>, see llama/pom.xml and .github/natives.csv). `cpu` and `metal`
# hold the library (LlamaLoader.LIBRARY_BACKENDS), every other name a GPU backend module
# (LlamaLoader.MODULE_BACKENDS); both lists must name every backend set here. `cuda13` names the
# CUDA major version the classifier promises; change it together with the classifiers when CUDA's
# major version changes. `sycl` is the fp16 build on every platform (GGML_SYCL_F16, upstream's
# recommended setting): two SYCL builds would both be called libggml-sycl.so and could not share
# the CPU jar's directory.
if(NOT DEFINED JLLAMA_BACKEND)
    if(GGML_CUDA)
        set(JLLAMA_BACKEND cuda13)
    elseif(GGML_VULKAN)
        set(JLLAMA_BACKEND vulkan)
    elseif(GGML_OPENCL)
        set(JLLAMA_BACKEND opencl)
    elseif(GGML_HIP)
        set(JLLAMA_BACKEND rocm)
    elseif(GGML_SYCL)
        set(JLLAMA_BACKEND sycl)
    elseif(GGML_OPENVINO)
        set(JLLAMA_BACKEND openvino)
    elseif(APPLE AND GGML_METAL)
        set(JLLAMA_BACKEND metal)
    else()
        set(JLLAMA_BACKEND cpu)
    endif()
endif()
set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/natives/net/ladenthin/llama/${OS_NAME}/${OS_ARCH}/${JLLAMA_BACKEND})
message(STATUS "Backend '${JLLAMA_BACKEND}' - installing files to ${JLLAMA_DIR}")

# jllama-build.txt: the build every natives directory came from, read by LlamaLoader before it puts
# a GPU module next to a CPU jar's library -- the two must be the same llama.cpp tag (the ggml ABI
# between libggml-base and a module is that commit's), and a mismatch is a LlamaException naming
# both, never a silent attempt. The tag is read back from this file's own GIT_TAG line so the
# version bump keeps touching that one line (build-webui and llama-next-version.sh grep it too).
file(READ ${CMAKE_CURRENT_LIST_FILE} _jllama_self)
string(REGEX MATCH "GIT_TAG[ \t]+(b[0-9]+)" _jllama_tag_match "${_jllama_self}")
set(JLLAMA_LLAMA_CPP_TAG "${CMAKE_MATCH_1}")
unset(_jllama_self)
if(NOT JLLAMA_LLAMA_CPP_TAG)
    message(FATAL_ERROR "could not read the llama.cpp GIT_TAG (b<nnnn>) from ${CMAKE_CURRENT_LIST_FILE}")
endif()
if(JLLAMA_MODULE_ONLY)
    set(_jllama_kind module)
else()
    set(_jllama_kind library)
endif()
file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-build.txt CONTENT
     "# Written by llama/CMakeLists.txt: the build this directory came from (read by LlamaLoader)\nllama.cpp=${JLLAMA_LLAMA_CPP_TAG}\nbackend=${JLLAMA_BACKEND}\nkind=${_jllama_kind}\n"
     NEWLINE_STYLE LF)

if(JLLAMA_MODULE_ONLY)
    # The GPU natives jar: exactly one file, the backend module, next to the two list files. Built
    # with the plain `cmake --build` of the CI scripts -- everything below this block (llama, mtmd,
    # jllama, the tests) is skipped, so `all` is ggml, ggml-base and the module.
    set(_jllama_module_targets cuda13 ggml-cuda rocm ggml-hip sycl ggml-sycl vulkan ggml-vulkan opencl ggml-opencl openvino ggml-openvino)
    list(FIND _jllama_module_targets ${JLLAMA_BACKEND} _jllama_module_index)
    if(_jllama_module_index EQUAL -1)
        message(FATAL_ERROR "JLLAMA_MODULE_ONLY: '${JLLAMA_BACKEND}' is not a GPU backend module")
    endif()
    math(EXPR _jllama_module_index "${_jllama_module_index} + 1")
    list(GET _jllama_module_targets ${_jllama_module_index} JLLAMA_MODULE_TARGET)
    if(NOT TARGET ${JLLAMA_MODULE_TARGET})
        message(FATAL_ERROR "JLLAMA_MODULE_ONLY: ggml did not create the target ${JLLAMA_MODULE_TARGET} for backend '${JLLAMA_BACKEND}'")
    endif()
    # A MODULE library is a LIBRARY artifact on every platform, Windows included, and a multi-config
    # generator appends the configuration name unless the per-config form is set (measured on the
    # CPU variants, see the output-directory block further down).
    set_target_properties(${JLLAMA_MODULE_TARGET} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${JLLAMA_DIR}
      LIBRARY_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
      LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
      RUNTIME_OUTPUT_DIRECTORY ${JLLAMA_DIR}
      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
      # $ORIGIN only: libggml-base.so is the sibling LlamaLoader extracts next to the module.
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN")
    # The counterpart of the CPU jar's list: LlamaLoader extracts what is named here into the CPU
    # library's directory and never loads it from Java -- ggml loads the module from there.
    file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-files.txt
         CONTENT "# Written by llama/CMakeLists.txt (JLLAMA_MODULE_ONLY): extracted next to the CPU library, loaded by ggml\n$<TARGET_FILE_NAME:${JLLAMA_MODULE_TARGET}>\n"
         NEWLINE_STYLE LF)
    message(STATUS "Module-only build: ${JLLAMA_MODULE_TARGET} -> ${JLLAMA_DIR}")
    return()
endif()

# include jni.h and jni_md.h
if(NOT DEFINED JNI_INCLUDE_DIRS)
    # The bundled JNI headers live in .github/include/ at the REPO ROOT, one level above
    # this module (the core moved into llama/). Anchor to the parent of the CMake source
    # dir so the path is correct regardless of the build CWD.
    if(OS_NAME MATCHES "^Linux" OR OS_NAME STREQUAL "Mac" OR OS_NAME STREQUAL "Darwin")
        set(JNI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../.github/include/unix)
    elseif(OS_NAME STREQUAL "Windows")
        set(JNI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../.github/include/windows)
    # if we don't have provided headers, try to find them via Java
    else()
        find_package(Java REQUIRED)
        find_program(JAVA_EXECUTABLE NAMES java)

        find_path(JNI_INCLUDE_DIRS NAMES jni.h HINTS ENV JAVA_HOME PATH_SUFFIXES include)

        # find "jni_md.h" include directory if not set
        file(GLOB_RECURSE JNI_MD_PATHS RELATIVE "${JNI_INCLUDE_DIRS}" "${JNI_INCLUDE_DIRS}/**/jni_md.h")
        foreach(PATH IN LISTS JNI_MD_PATHS)
            get_filename_component(DIR ${PATH} DIRECTORY)
            list(APPEND JNI_INCLUDE_DIRS "${JNI_INCLUDE_DIRS}/${DIR}")
        endforeach()
    endif()
endif()
if(NOT JNI_INCLUDE_DIRS)
    if(ANDROID_ABI)
        find_package(JNI REQUIRED)
        set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIRS})
    else()
    message(FATAL_ERROR "Could not determine JNI include directories")
endif()
endif()

add_library(jllama SHARED
    src/main/cpp/jllama.cpp
    src/main/cpp/tts_engine.cpp
    src/main/cpp/train_engine.cpp
    src/main/cpp/rpc_bridge.cpp
    src/main/cpp/utils.hpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-common.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-chat.cpp)

# Phase 1 refactoring: compile upstream server library units directly into jllama
# server.hpp has been replaced by direct upstream includes in jllama.cpp.
# server-context.cpp, server-queue.cpp, server-task.cpp compile on all platforms
# including Android.  server-models.cpp is excluded on Android because it pulls
# in subprocess.h which calls posix_spawn_*, declared but not implemented by the
# Android NDK.  Guard with both ANDROID_ABI (NDK toolchain convention) and
# OS_NAME (always set to "Linux-Android" by the CI cmake invocation).
#
# server-stream.cpp (added upstream in b9829) owns the resumable-streaming SSE
# replay buffer (g_stream_sessions, stream_session_attach_pipe,
# stream_aware_should_stop, stream_conv_id_from_headers, stream_pipe_*).
# server-context.cpp / server-http.cpp / server-models.cpp now #include
# "server-stream.h" and reference those symbols, so it MUST be compiled in or the
# link fails with undefined references.  It is platform-neutral (threads + the
# std mutex/condvar primitives, no subprocess.h / posix_spawn_*), so it builds on
# Android too and stays outside the server-models Android guard below.  jllama
# wires its own routes and never calls g_stream_sessions.start_gc() (only the
# standalone server.cpp main does), so the GC thread stays dormant here.
#
# server-decision.cpp (added upstream in b11361, #29818) implements the typed
# decision models behind POST /v1/systemone (server_decision_context, the
# SERVER_TASK_TYPE_DECISION prompts). server-context.cpp holds one as a member
# and calls into it on every load, so it is mandatory for the same reason as
# server-stream.cpp -- a missing unit is an undefined reference, latent on Linux
# but a hard link error on macOS/ld64 and Windows/MSVC. Platform-neutral, so it
# stays outside the Android guard too.
target_sources(jllama PRIVATE
    ${llama.cpp_SOURCE_DIR}/tools/server/server-context.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-queue.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-task.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-schema.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-stream.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-decision.cpp
)
if(NOT ANDROID_ABI AND NOT OS_NAME MATCHES "Android")
    target_sources(jllama PRIVATE
        ${llama.cpp_SOURCE_DIR}/tools/server/server-models.cpp
    )
endif()

# Native-server mode (net.ladenthin.llama.server.NativeServer): compile the standalone server
# entry point (server.cpp's `llama_server`, made embeddable by patches/0006) and its tools helper
# (server-tools.cpp); jllama's JNI bridge (native_server.cpp) then calls llama_server on a worker
# thread. This runs the *full* upstream HTTP server — WebUI included, every llama-server flag
# forwarded — inside libjllama, with no separate llama-server executable. server.cpp,
# server-tools.cpp and server-mcp.cpp all pull in vendor/sheredom/subprocess.h (posix_spawn_*), so
# they share the non-Android guard used for server-models.cpp above; native_server.cpp links against
# llama_server and is guarded too. On Android the NativeServer native methods are simply absent (its
# JNI calls throw UnsatisfiedLinkError) — use OpenAiCompatServer there.
#
# server-mcp.cpp is new in b10154 (upstream MCP-server support): both server.cpp (llama_server's
# mcp_mgr lifecycle) and server-tools.cpp (tools.setup(..., mcp_mgr) / server_mcp::call_tool) now
# reference server_mcp, so it MUST be compiled in or the link fails with undefined server_mcp
# symbols. This is latent on Linux (a shared lib tolerates undefined symbols) but a hard link error
# on macOS/ld64 and Windows. It is not compiled into jllama_test (that target links neither
# server.cpp nor server-tools.cpp, so it never references server_mcp).
if(NOT ANDROID_ABI AND NOT OS_NAME MATCHES "Android")
    target_sources(jllama PRIVATE
        ${llama.cpp_SOURCE_DIR}/tools/server/server-mcp.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-tools.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server.cpp
        ${CMAKE_SOURCE_DIR}/src/main/cpp/native_server.cpp
    )
endif()

# b10154's server-tools.cpp embeds a U+2192 (→) in a narrow string literal. MSVC (and clang-cl)
# decode narrow literals in the system code page (1252 on the CI runners) and emit warning C4566
# ("character ... cannot be represented in the current code page"), substituting the arrow at
# runtime. /utf-8 makes the compiler read sources as UTF-8 and use a UTF-8 execution charset so the
# arrow survives; it is the fix Microsoft documents for C4566. Scoped to jllama's own TUs and gated
# on MSVC (true for both cl.exe and clang-cl; /utf-8 is invalid on gcc/clang), so non-Windows builds
# are unaffected.
if(MSVC)
    target_compile_options(jllama PRIVATE /utf-8)
endif()

# Phase 2: also compile the upstream HTTP transport (server-http.cpp) and its
# cpp-httplib backend directly into jllama, so the OpenAI-compatible server can be
# driven natively from JNI — shipped inside libjllama, with no separate
# llama-server executable (a JNI .so/.dll/.dylib loads everywhere a JVM runs,
# unlike a standalone binary).  Only server.cpp (the standalone main() + route
# wiring) stays excluded for now; this first step just makes the HTTP layer build
# and link.
#
# server-http.cpp does `#include "ui.h"` — the WebUI asset table that tools/ui
# normally GENERATES.  The WebUI is built once in CI (the build-webui job) and
# shared to every native build as a generated, platform-independent ui.cpp/ui.h;
# the "WebUI assets" block below compiles it in when present and otherwise falls
# back to the empty-asset stub (src/main/cpp/webui_stub/ui.h).
# <cpp-httplib/httplib.h> already resolves through llama-common: upstream #27304
# (b10488) replaced its `PUBLIC . ../vendor` include dir with PUBLIC links to the
# vendor::nlohmann / vendor::sheredom INTERFACE targets, and each of those exports
# the vendor/ root (`target_include_directories(<t> INTERFACE ..)`), so the same
# directory still reaches jllama transitively.  Its bundled nlohmann/json is the
# same 3.12.0 as our FetchContent copy, so nothing added here shadows it.
target_sources(jllama PRIVATE
    ${llama.cpp_SOURCE_DIR}/tools/server/server-http.cpp
)

# cpp-httplib itself is upstream's `cpp-httplib` static target (vendor/cpp-httplib/CMakeLists.txt,
# built whenever LLAMA_BUILD_COMMON is on), linked below -- NOT compiled into jllama a second time.
# That target carries CPPHTTPLIB_OPENSSL_SUPPORT as a PUBLIC definition whenever an SSL library is
# configured (BoringSSL here, see the HTTPS block above), so server-http.cpp, jllama's own TUs and
# common/download.cpp all agree on it. Until 5.2.0 this file compiled httplib.cpp itself, without
# SSL, which put two incompatible copies of the same classes into one link; which copy the linker
# chose decided per platform whether download.cpp got an SSL-capable client (macOS: yes, via the
# Homebrew libssl it then depended on) or not (Linux: an https:// download threw).

# MinGW needs ws2_32 explicitly; MSVC auto-links it via a #pragma in httplib.h.
if(WIN32 AND NOT MSVC)
    target_link_libraries(jllama PRIVATE ws2_32)
endif()

# WebUI assets.  The llama.cpp WebUI is built once in CI by the build-webui job
# (upstream Svelte build + llama-ui-embed, run against the pinned llama.cpp tag)
# and shared to every native build as the generated, platform-independent
# ui.cpp / ui.h under webui-generated/ (git-ignored; downloaded as the
# "webui-generated" artifact — this repo commits no build outputs).  When present,
# compile it in so libjllama serves the real WebUI: the generated ui.h #defines
# LLAMA_UI_HAS_ASSETS, which activates server-http.cpp's static-asset routes.  When
# absent (local builds, or any job without the artifact) fall back to the empty-
# asset stub so the server still builds and runs, just without an embedded WebUI.
set(JLLAMA_WEBUI_GENERATED_DIR ${CMAKE_SOURCE_DIR}/webui-generated)
if(EXISTS ${JLLAMA_WEBUI_GENERATED_DIR}/ui.cpp AND EXISTS ${JLLAMA_WEBUI_GENERATED_DIR}/ui.h)
    message(STATUS "WebUI: embedding generated assets from ${JLLAMA_WEBUI_GENERATED_DIR}")
    target_sources(jllama PRIVATE ${JLLAMA_WEBUI_GENERATED_DIR}/ui.cpp)
    target_include_directories(jllama PRIVATE ${JLLAMA_WEBUI_GENERATED_DIR})
else()
    message(STATUS "WebUI: no generated assets found; using empty-asset stub (no embedded WebUI)")
    target_include_directories(jllama PRIVATE ${CMAKE_SOURCE_DIR}/src/main/cpp/webui_stub)
endif()

set_target_properties(jllama PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(jllama PRIVATE
    src/main/cpp
    ${JNI_INCLUDE_DIRS}
    ${llama.cpp_SOURCE_DIR}/tools/mtmd
    ${llama.cpp_SOURCE_DIR}/tools/server)
# Note: the WebUI ui.h include dir (generated webui-generated/ or the empty stub
# src/main/cpp/webui_stub) is added by the "WebUI assets" conditional above.
target_link_libraries(jllama PRIVATE llama-common mtmd llama nlohmann_json cpp-httplib)
target_compile_features(jllama PRIVATE cxx_std_11)

target_compile_definitions(jllama PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
    # cpp-httplib tuning — mirror the defines upstream's cpp-httplib target sets PRIVATELY for
    # httplib.cpp, so every TU here that includes httplib.h (server-http.cpp) agrees with it on
    # the inline behaviour these macros control.
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    CPPHTTPLIB_LISTEN_BACKLOG=512
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    CPPHTTPLIB_TCP_NODELAY=1
)

if(OS_NAME STREQUAL "Windows")
    set_target_properties(jllama llama ggml PROPERTIES
	  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
	  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
	  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
	)
else()
	set_target_properties(jllama llama ggml PROPERTIES
	  LIBRARY_OUTPUT_DIRECTORY ${JLLAMA_DIR}
	)
endif()

if(JLLAMA_BACKEND_DL)
    # Every ggml library and backend module goes next to libjllama, which finds the two it links
    # (libggml, libggml-base) through $ORIGIN and loads the backend modules from its own directory
    # at JNI_OnLoad. jllama-files.txt lists them for LlamaLoader, which extracts them without
    # loading them: choosing the CPU module is ggml's job.
    set(JLLAMA_GGML_TARGETS)
    foreach(dir ${llama.cpp_SOURCE_DIR}/ggml/src ${llama.cpp_SOURCE_DIR}/ggml/src/ggml-rpc)
        get_property(dir_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
        foreach(t ${dir_targets})
            get_target_property(t_type ${t} TYPE)
            if(t_type STREQUAL "SHARED_LIBRARY" OR t_type STREQUAL "MODULE_LIBRARY")
                list(APPEND JLLAMA_GGML_TARGETS ${t})
            endif()
        endforeach()
    endforeach()
    if(OS_NAME STREQUAL "Windows")
        # Windows needs BOTH properties, per configuration, and that is not a belt-and-braces
        # guess -- CMake routes the two artifact kinds in this list differently:
        #   * ggml / ggml-base are SHARED libraries, whose .dll is a RUNTIME artifact.
        #   * the CPU backend variants and ggml-rpc are MODULE libraries, whose output CMake treats
        #     as a LIBRARY artifact on EVERY platform, Windows included.
        # And a multi-config generator appends the configuration name unless the per-config form is
        # set. Measured both ways: with only LIBRARY_OUTPUT_DIRECTORY the 14 modules landed in
        # <dir>/Release/, with only the RUNTIME ones they stayed in the build tree entirely -- while
        # jllama-files.txt names them next to jllama.dll either way, so the loader would extract
        # nothing and ggml would find no backend at all.
        set_target_properties(${JLLAMA_GGML_TARGETS} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
          RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
          LIBRARY_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
          LIBRARY_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
          LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
        )
    else()
        set_target_properties(${JLLAMA_GGML_TARGETS} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${JLLAMA_DIR})
    endif()
    # $ORIGIN only: the default build run path would also carry this machine's absolute build path.
    # (VERSION/SOVERSION of ggml and ggml-base are cleared where ggml is added, see there.)
    set_target_properties(jllama ${JLLAMA_GGML_TARGETS} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
    set(JLLAMA_FILES_CONTENT "# Written by llama/CMakeLists.txt (JLLAMA_BACKEND_DL): extracted, not loaded\n")
    foreach(t ${JLLAMA_GGML_TARGETS})
        # On WINDOWS the two libraries jllama LINKS (ggml, ggml-base) must be left out of this list,
        # which puts them into jllama-extras.txt instead (merge-native-artifacts.sh derives that from
        # everything not named here) so LlamaLoader loads them by full path BEFORE jllama.dll.
        # There is no $ORIGIN on Windows: System.load with a full path does not add the DLL's own
        # directory to the dependency search, so jllama.dll's import of ggml.dll would be resolved
        # from the *executable's* directory (java.exe) or PATH -- where a foreign llama.cpp install
        # (winget ggml.llamacpp, Ollama, LM Studio) can satisfy it with another build's binary.
        # Pre-loading by full path puts ours in the loaded-module list first, which the Windows
        # loader checks before any directory search. The alphabetical order the merge script writes
        # is also the dependency order (ggml-base.dll < ggml.dll), and it has to be: loading
        # ggml.dll first would have Windows resolve ggml-base.dll by directory search again.
        # The backend MODULES stay in this list -- extracted, never loaded from Java, because
        # choosing one is ggml_backend_load_best()'s job.
        if(OS_NAME STREQUAL "Windows" AND (t STREQUAL "ggml" OR t STREQUAL "ggml-base"))
            continue()
        endif()
        string(APPEND JLLAMA_FILES_CONTENT "$<TARGET_FILE_NAME:${t}>\n")
    endforeach()
    # NEWLINE_STYLE LF on both lists: on Windows file(GENERATE) otherwise writes CRLF (measured), and
    # merge-native-artifacts.sh reads them on Linux -- it strips a CR itself now, but the extras
    # file below is meant to be byte-identical to what that script writes, which is LF.
    file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-files.txt CONTENT "${JLLAMA_FILES_CONTENT}" NEWLINE_STYLE LF)
    if(OS_NAME STREQUAL "Windows")
        # The counterpart of the two names left out above: in CI merge-native-artifacts.sh derives
        # jllama-extras.txt as "every file next to the library that jllama-files.txt does not name,
        # sorted", which here is exactly ggml-base.dll then ggml.dll. Writing the same two lines from
        # the build (same order, no comment line, so the merge step rewrites an identical file) makes
        # a local variants build loadable without the merge step: without the file LlamaLoader
        # extracts jllama.dll alone, System.load resolves its ggml.dll import from java.exe's
        # directory or PATH, and every model-backed test fails on the load.
        file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-extras.txt
             CONTENT "$<TARGET_FILE_NAME:ggml-base>\n$<TARGET_FILE_NAME:ggml>\n" NEWLINE_STYLE LF)
    endif()
    message(STATUS "ggml modules: ${JLLAMA_GGML_TARGETS}")
    # Android: the modules are not next to the library as files -- an app's native libraries stay
    # inside the APK unless it opts into legacy extraction -- so JNI_OnLoad loads them by soname
    # through the app's linker namespace instead of scanning a directory (jllama.cpp,
    # load_backends_by_soname). The names it needs are this build's module targets, generated into a
    # header: the CPU modules it scores against each other, and the rest (ggml-rpc) it loads as is.
    # The GPU modules another AAR may add (libggml-opencl.so) are a fixed list in jllama.cpp.
    if(OS_NAME MATCHES "Android" OR ANDROID OR ANDROID_ABI)
        set(_jllama_cpu_sonames "")
        set(_jllama_other_sonames "")
        foreach(t ${JLLAMA_GGML_TARGETS})
            if(t STREQUAL "ggml" OR t STREQUAL "ggml-base")
                continue() # DT_NEEDED of libjllama, resolved by the linker
            elseif(t MATCHES "^ggml-cpu")
                string(APPEND _jllama_cpu_sonames "    \"$<TARGET_FILE_NAME:${t}>\", \\\n")
            else()
                string(APPEND _jllama_other_sonames "    \"$<TARGET_FILE_NAME:${t}>\", \\\n")
            endif()
        endforeach()
        file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/jllama_modules.h CONTENT
             "// Generated by llama/CMakeLists.txt (JLLAMA_BACKEND_DL on Android): the module sonames of this build.\n#pragma once\n#define JLLAMA_CPU_MODULE_SONAMES \\\n{ \\\n${_jllama_cpu_sonames}}\n#define JLLAMA_OTHER_MODULE_SONAMES \\\n{ \\\n${_jllama_other_sonames}}\n"
             NEWLINE_STYLE LF)
        target_include_directories(jllama PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
        target_compile_definitions(jllama PRIVATE JLLAMA_MODULES_BY_SONAME)
    endif()
endif()

if (LLAMA_METAL AND NOT LLAMA_METAL_EMBED_LIBRARY)
    # copy ggml-common.h and ggml-metal.metal to bin directory
    configure_file(${llama.cpp_SOURCE_DIR}/ggml-metal.metal ${JLLAMA_DIR}/ggml-metal.metal COPYONLY)
endif()

#################### C++ unit tests ####################

# BUILD_TESTING is declared at the top of this file, before any subproject (see there).
if(BUILD_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        # No constraint behind this exact tag — GoogleTest is only used by this repo's own
        # C++ unit tests (jllama_test), not by the shipped library and not tied to llama.cpp.
        # It is just "latest stable at the time"; bump it from time to time (see CLAUDE.md).
        GIT_TAG        v1.18.0
    )
    # Keep GTest on the same CRT as the rest of the project.
    # OFF means GTest respects CMAKE_MSVC_RUNTIME_LIBRARY (static /MT here).
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    # Make the three wire-name registries the Java layer declares available to the C++ contract
    # tests. The Java enums are the single source of truth; see cmake/extract-java-wire-names.cmake
    # for why a textual sweep of the receiver's source is not a substitute (a CLI option can be
    # present in common/arg.cpp yet scoped away from the server example, and an unknown request key
    # is discarded without a word).
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extract-java-wire-names.cmake)
    file(GLOB JLLAMA_REQUEST_READER_SOURCES
        ${llama.cpp_SOURCE_DIR}/tools/server/*.cpp
        ${llama.cpp_SOURCE_DIR}/common/*.cpp)
    jllama_extract_java_wire_names(
        JAVA_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/net/ladenthin/llama/args/ModelFlag.java
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/net/ladenthin/llama/args/ModelOption.java
        ARRAY_PREFIX     JLLAMA_JAVA_CLI
        DEFAULT_CONTRACT SERVER_PARSER
        MIN_COUNT        50
        OUTPUT_HEADER    ${CMAKE_CURRENT_BINARY_DIR}/generated/jllama_java_cli_flags.h
    )
    jllama_extract_java_wire_names(
        JAVA_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/net/ladenthin/llama/parameters/RequestField.java
        ARRAY_PREFIX     JLLAMA_JAVA_REQUEST
        DEFAULT_CONTRACT SCHEMA
        MIN_COUNT        30
        OUTPUT_HEADER    ${CMAKE_CURRENT_BINARY_DIR}/generated/jllama_java_request_fields.h
        # An OAI_LAYER key never reaches make_llama_cmpl_schema, so the schema cannot vouch for
        # it. Sweep the layer that does consume it instead -- globbed, not listed, so an upstream
        # file split does not quietly narrow the corpus.
        READER_CONTRACT  OAI_LAYER
        READER_SOURCES   ${JLLAMA_REQUEST_READER_SOURCES}
    )
    jllama_extract_java_wire_names(
        JAVA_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/net/ladenthin/llama/parameters/TrainingField.java
        ARRAY_PREFIX     JLLAMA_JAVA_TRAINING
        DEFAULT_CONTRACT ENGINE
        MIN_COUNT        10
        OUTPUT_HEADER    ${CMAKE_CURRENT_BINARY_DIR}/generated/jllama_java_training_fields.h
    )

    add_executable(jllama_test
        src/test/cpp/test_utils.cpp
        src/test/cpp/test_server.cpp
        src/test/cpp/test_jni_helpers.cpp
        src/test/cpp/test_json_helpers.cpp
        src/test/cpp/test_log_helpers.cpp
        src/test/cpp/test_common_log_callback.cpp
        src/test/cpp/test_tts_wav.cpp
        src/test/cpp/test_tts_params.cpp
        src/test/cpp/test_model_split.cpp
        src/test/cpp/test_prefetch.cpp
        src/test/cpp/test_kolibri1.cpp
        src/test/cpp/test_model_flags.cpp
        src/test/cpp/test_wire_contracts.cpp
        src/test/cpp/test_rpc.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-common.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-chat.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-context.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-queue.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-task.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-schema.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-models.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-stream.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-decision.cpp
    )

    target_include_directories(jllama_test PRIVATE
        src/main/cpp
        # mtmd.h is not always propagated transitively — add it explicitly
        ${llama.cpp_SOURCE_DIR}/tools/mtmd
        # jni.h / jni_md.h needed by jni_helpers.hpp (mock JNI tests, no JVM required)
        ${JNI_INCLUDE_DIRS}
        ${llama.cpp_SOURCE_DIR}/tools/server
        # llama.cpp's internal src/ -- test_model_split.cpp includes llama-model.h to drive the
        # two load_tensors() split helpers that patches/0012 extracts. Upstream's own tests do the
        # same (tests/test-batch-alloc.cpp, tests/test-quantize-stats.cpp include "../src/...").
        # No header name in that directory collides with tools/server, tools/mtmd or src/main/cpp.
        ${llama.cpp_SOURCE_DIR}/src
        # ggml's internal src/ and the x86 CPU-backend directory -- test_prefetch.cpp includes
        # ggml-common.h (for block_q4_0 / block_q8_0, whose sizes the prefetch offsets depend on)
        # and prefetch.h, the helper patches/0017 adds. Same reasoning as the line above: the
        # guard has to drive the patched code, and that code is not on any public header path.
        ${llama.cpp_SOURCE_DIR}/ggml/src
        ${llama.cpp_SOURCE_DIR}/ggml/src/ggml-cpu/arch/x86
        # The generated Java-CLI-flag list consumed by test_model_flags.cpp.
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )
    target_link_libraries(jllama_test PRIVATE llama-common mtmd llama nlohmann_json cpp-httplib GTest::gtest_main)
    target_compile_features(jllama_test PRIVATE cxx_std_17)

    # The embedded llama-server's shutdown path (test_native_server_shutdown.cpp) drives the real
    # llama_server() in router mode, so the test binary needs server.cpp and what it pulls in: the
    # HTTP transport (cpp-httplib itself is the linked upstream target) and the tools/MCP units
    # server.cpp references. Same
    # non-Android guard as jllama (subprocess.h / posix_spawn_*). The empty-asset WebUI stub is
    # always enough here: no test serves the UI.
    if(NOT ANDROID_ABI AND NOT OS_NAME MATCHES "Android")
        target_sources(jllama_test PRIVATE
            src/test/cpp/test_native_server_shutdown.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-tools.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-mcp.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-http.cpp
        )
        target_include_directories(jllama_test PRIVATE ${CMAKE_SOURCE_DIR}/src/main/cpp/webui_stub)
        if(MSVC)
            target_compile_options(jllama_test PRIVATE /utf-8)
        endif()
        if(WIN32 AND NOT MSVC)
            target_link_libraries(jllama_test PRIVATE ws2_32)
        endif()
    endif()

    target_compile_definitions(jllama_test PRIVATE
        SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
    )

    # gtest_discover_tests runs the freshly built jllama_test executable at build
    # time (POST_BUILD) to enumerate test cases. The default discovery timeout is
    # 5s. The 32-bit Windows (Win32) build links the entire llama/ggml/server tree
    # statically into one large binary whose startup + test enumeration sits right
    # at that 5s boundary on shared CI runners: the same b9682 binary discovered
    # within 5s in one run but was killed at the 5s timeout in another (empty
    # output, process still alive — a timeout, not a crash). x64/Linux/macOS finish
    # well under the default. Raise the budget so 32-bit discovery is not flaky;
    # this is a maximum, so fast platforms still return immediately.
    gtest_discover_tests(jllama_test DISCOVERY_TIMEOUT 120)
endif()
