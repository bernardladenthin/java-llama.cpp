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

#################### json: upstream's vendored copy, and only that one ####################
#
# There is no FetchContent for nlohmann/json any more, because there never was a compile that used
# it. llama.cpp carries its own copy at vendor/nlohmann/ and exports that directory through the
# vendor::nlohmann INTERFACE target which llama-common links PUBLIC, and that directory comes BEFORE
# a FetchContent include dir on every jllama compile line -- measured at the b11556 bump in
# CMakeFiles/jllama.dir/flags.make: `vendor/nlohmann/..` was -I number 7 and `json-src/include`
# number 12. So the two project headers that include nlohmann/json.hpp (log_helpers.hpp and
# train_engine.cpp) always resolved to upstream's copy, exactly as every upstream TU does.
#
# That is the good case and the reason to keep it: ONE copy in every translation unit of the library.
# Two copies of the same classes with different layouts in one link is what the httplib incident was
# (see "HTTPS: BoringSSL, statically" in CLAUDE.md) -- the linker's choice per platform decided the
# behaviour there, and it differed per platform.
#
# What made the duplicate indefensible rather than merely idle: since b11556 upstream applies
# vendor/nlohmann/json-deep-nesting.patch to its copy (nlohmann/json#5387 -- deep values crashed
# copy, dump() and operator==), so the fetched v3.12.0 was not even the same code. It cost a clone
# per fresh build directory, two link entries, and a pinned tag that looked like something to keep
# in sync with a library this project does not choose the version of.
#
# The guard that replaces it is below, next to the llama.cpp fetch: upstream's copy must be there.

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
# Linux: no libgomp dependency, because the jar has to load in the images Java is actually deployed
# in. `libggml-base.so` -- the first file LlamaLoader loads -- declares `libgomp.so.1` whenever ggml
# builds with OpenMP, and GCC's -fopenmp links it DYNAMICALLY, so it is a runtime requirement of the
# shipped jar. A plain Ubuntu 24.04 server image does not have it, nor does a distroless Java image:
# measured on EC2, the whole jar then fails with "The native library of backend 'cpu' could not be
# loaded", which reads like a packaging defect and is a missing distro package. It never surfaced
# before because every probe ran on Amazon Linux, which ships libgomp with its gcc runtime.
#
# What it costs: nothing measurable. Upstream b11538 built twice on one 16-vCPU machine with
# identical flags except this one and benchmarked with llama-bench (3 repetitions) gives token
# generation slightly FASTER without OpenMP at 4, 8 and 16 threads (+0.9%, +0.8%, +0.7%) and prompt
# processing lower by 0.4-2.1%, every difference inside the error bars -- see
# docs/benchmarks/cpu-backend-modules.md. The ~6% this project previously recorded was a single
# sample already flagged as inside its own spread and does not reproduce.
#
# Scope, deliberately: ONLY OpenMP. libstdc++.so.6 and libgcc_s.so.1 stay dynamic, because every
# image that can run this library has them -- HotSpot's own libjvm.so links both -- so
# -static-libstdc++ would buy nothing here while giving each of our shared objects its own unwinder,
# which is a real hazard for an exception crossing from ggml into libjllama. Android is the different
# case and keeps its own block above: there libc++_shared genuinely is absent.
#
# Upstream is not one answer but three, and this is the same kind of per-deployment decision: their
# Android build passes GGML_OPENMP=OFF, their Windows build passes GGML_OPENMP=ON with
# GGML_OPENMP_FETCH=ON (bundling LLVM's libomp), and their Linux build leaves it ON against the
# distro's libgomp -- right for a zip onto a developer's distro, wrong for a Maven dependency landing
# in someone else's container. s390x already passed -DGGML_OPENMP=OFF from its job for a different
# reason (cross-libgomp), and the Windows jobs pass it because it measured 1.9x/2.7x FASTER there.
if(OS_NAME MATCHES "^Linux" AND NOT OS_NAME MATCHES "Android" AND NOT ANDROID_ABI)
    set(GGML_OPENMP OFF CACHE BOOL
        "Linux: no libgomp.so.1 dependency -- slim and distroless images do not have it, and it costs nothing measurable"
        FORCE)
endif()

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
# ---- The llama.cpp pin ------------------------------------------------------------------------
# JLLAMA_GIT_TAG is THE pin, a b<nnnn> release number, and everything that names the version reads
# it: jllama-build.txt (the loader's same-build check), the build-webui job, llama-next-version.sh,
# and by hand README.md, CLAUDE.md and LlamaCppVersion.java. Upstream numbers its tags by
# `git rev-list --count` of the master commit, so a number names exactly one commit whether or not
# upstream has pushed the tag yet. That is what JLLAMA_GIT_COMMIT is for: while the tag is not
# published, set it to that commit's full hash (a "virtual" b-number) -- the fetch below takes the
# hash, every other consumer keeps the number, and the pin check after FetchContent_MakeAvailable
# asserts that the fetched commit's count IS the number. Clear it once the tag exists;
# `llama-next-version.sh` says when, and `llama-next-version.sh --resolve b<nnnn>` prints the hash.
set(JLLAMA_GIT_TAG    b11639)
set(JLLAMA_GIT_COMMIT "a0e6dc7384edb8f5d927155e1d26c10b4a310e6d")   # the commit b11639 names; kept after the tag exists
if(NOT JLLAMA_GIT_TAG MATCHES "^b[0-9]+$")
    message(FATAL_ERROR "JLLAMA_GIT_TAG must be a llama.cpp release number b<nnnn>, got '${JLLAMA_GIT_TAG}'")
endif()
if(JLLAMA_GIT_COMMIT AND NOT JLLAMA_GIT_COMMIT MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "JLLAMA_GIT_COMMIT must be empty or a full 40-hex commit hash, got '${JLLAMA_GIT_COMMIT}'")
endif()
# A tag name is a moving reference: upstream can re-point it, and `git rev-list --count` only proves
# the number, not the content. So the commit stays the pin for good -- JLLAMA_GIT_COMMIT is kept
# after upstream publishes the tag, not cleared -- and CI refuses to fetch by name
# (JLLAMA_REQUIRE_COMMIT, set by build.sh/build.bat under GitHub Actions; TODO.md, delivery
# roadmap, item 3). A local build with an empty commit still works, by the tag, for convenience.
option(JLLAMA_REQUIRE_COMMIT "Refuse to fetch llama.cpp by tag name: JLLAMA_GIT_COMMIT must name the commit" OFF)
if(JLLAMA_GIT_COMMIT)
    string(LENGTH "${JLLAMA_GIT_COMMIT}" _jllama_commit_len)
    if(NOT _jllama_commit_len EQUAL 40)
        message(FATAL_ERROR "JLLAMA_GIT_COMMIT must be the full 40-hex commit hash, got '${JLLAMA_GIT_COMMIT}' (${_jllama_commit_len} chars)")
    endif()
    set(_jllama_fetch_ref "${JLLAMA_GIT_COMMIT}")
    message(STATUS "llama.cpp ${JLLAMA_GIT_TAG} is fetched by commit ${JLLAMA_GIT_COMMIT}")
elseif(JLLAMA_REQUIRE_COMMIT)
    message(FATAL_ERROR "JLLAMA_REQUIRE_COMMIT is ON and JLLAMA_GIT_COMMIT is empty: a release build never fetches llama.cpp by the moving tag name. Run `.github/scripts/llama-next-version.sh --resolve ${JLLAMA_GIT_TAG}` and set the commit.")
else()
    set(_jllama_fetch_ref "${JLLAMA_GIT_TAG}")
    message(STATUS "llama.cpp ${JLLAMA_GIT_TAG} is fetched by tag name (JLLAMA_GIT_COMMIT is empty; CI would refuse this)")
endif()

# Where the pinned commit is fetched from. The default is upstream; the mirror
# bernardladenthin/llama.cpp (every pinned commit is pushed there as the tag jllama/b<nnnn> by the
# mirror-llama-source job of publish.yml) serves the same commit when upstream's repository or tag
# is gone -- the commit hash is identical in both, so nothing else changes. The archived patched
# source of every release (llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz on the GitHub release) is
# the third way: -DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<unpacked archive>. TODO.md, delivery roadmap,
# item 6.
set(JLLAMA_GIT_REPOSITORY "https://github.com/ggerganov/llama.cpp.git" CACHE STRING
    "The git repository llama.cpp is fetched from (upstream, or the mirror https://github.com/bernardladenthin/llama.cpp.git)")
FetchContent_Declare(
	llama.cpp
	GIT_REPOSITORY ${JLLAMA_GIT_REPOSITORY}
	GIT_TAG        ${_jllama_fetch_ref}
	PATCH_COMMAND  ${CMAKE_COMMAND}
		-DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
		-DLLAMA_SRC=<SOURCE_DIR>
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-llama-patches.cmake
	${JLLAMA_LLAMA_SOURCE_SUBDIR}
)
# The commit of an ARCHIVED llama.cpp source tree: the release asset
# llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz (publish.yml, archive-llama-source) is the patched
# checkout with its .git removed, so git cannot answer for it -- but the applier's stamp it carries
# names the commit it was patched at (`head <hash>`). Empty for a git work tree or a tree without
# the stamp; the two build-info writers fall back to it. Such a tree only ever arrives through
# FETCHCONTENT_SOURCE_DIR_LLAMA.CPP, which is why this runs BEFORE FetchContent_MakeAvailable: the
# static build adds llama.cpp's root inside that call, and the pair below must be set by then.
function(jllama_archived_llama_commit out_var dir)
    set(${out_var} "" PARENT_SCOPE)
    set(_stamp "${dir}/.jllama-patches-applied")
    if(EXISTS "${_stamp}" AND NOT EXISTS "${dir}/.git")
        file(STRINGS "${_stamp}" _head_lines REGEX "^head [0-9a-f]+$")
        list(LENGTH _head_lines _n)
        if(_n EQUAL 1)
            string(REGEX REPLACE "^head " "" _head "${_head_lines}")
            set(${out_var} "${_head}" PARENT_SCOPE)
        endif()
    endif()
endfunction()
# For such a tree upstream's own build-info (common/CMakeLists.txt via cmake/build-info.cmake:
# `git rev-list --count` / `rev-parse --short` in the source tree) would report `b0-unknown`; the
# root CMakeLists honours a predefined LLAMA_BUILD_NUMBER / LLAMA_BUILD_COMMIT, and these normal
# variables reach every subdirectory, so llama_build_info() of an archive build names
# `b<nnnn>-<commit>` like a fetched one (NativeLibraryLoadSmokeTest holds it to LlamaCppVersion).
set(_jllama_archived_commit "")
if(FETCHCONTENT_SOURCE_DIR_LLAMA.CPP)
    jllama_archived_llama_commit(_jllama_archived_commit "${FETCHCONTENT_SOURCE_DIR_LLAMA.CPP}")
endif()
if(_jllama_archived_commit)
    string(REGEX REPLACE "^b" "" LLAMA_BUILD_NUMBER "${JLLAMA_GIT_TAG}")
    string(SUBSTRING "${_jllama_archived_commit}" 0 9 LLAMA_BUILD_COMMIT)
    message(STATUS "llama.cpp ${JLLAMA_GIT_TAG} from an archived source tree (no .git); build info b${LLAMA_BUILD_NUMBER}-${LLAMA_BUILD_COMMIT} from its patch stamp")
endif()

FetchContent_MakeAvailable(llama.cpp)

# llama.cpp pin check: the fetched commit's count must be the number JLLAMA_GIT_TAG names. Fetched by
# tag, that is upstream's own invariant (release.yml numbers the tag by that count); fetched by
# JLLAMA_GIT_COMMIT, it is the proof that the hash resolves the number -- a wrong hash would otherwise
# ship as b<nnnn> with another commit's code, and the Java pin check (NativeLibraryLoadSmokeTest
# against build-info, which derives its number the same way) would see it only after the natives
# were built. Skipped, with a message, where it cannot be answered: no git, a source tree that is not
# a git checkout (-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP), or a shallow clone (count is meaningless).
find_package(Git QUIET)
if(GIT_EXECUTABLE AND EXISTS "${llama.cpp_SOURCE_DIR}/.git" AND NOT EXISTS "${llama.cpp_SOURCE_DIR}/.git/shallow")
    string(REGEX REPLACE "^b" "" _jllama_tag_number "${JLLAMA_GIT_TAG}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
        WORKING_DIRECTORY "${llama.cpp_SOURCE_DIR}"
        OUTPUT_VARIABLE _jllama_head_count
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _jllama_count_res)
    if(NOT _jllama_count_res EQUAL 0)
        message(WARNING "llama.cpp pin check skipped: `git rev-list --count HEAD` failed in ${llama.cpp_SOURCE_DIR}")
    elseif(NOT _jllama_head_count STREQUAL _jllama_tag_number)
        message(FATAL_ERROR "llama.cpp pin check: JLLAMA_GIT_TAG is ${JLLAMA_GIT_TAG}, but the fetched commit (${_jllama_fetch_ref}) has commit count ${_jllama_head_count}, i.e. it is b${_jllama_head_count}. With JLLAMA_GIT_COMMIT set, that hash does not resolve the number (run `.github/scripts/llama-next-version.sh --resolve ${JLLAMA_GIT_TAG}`); without it, upstream's tag and its count disagree, which has never happened.")
    else()
        message(STATUS "llama.cpp pin check: fetched commit count ${_jllama_head_count} = ${JLLAMA_GIT_TAG}")
    endif()
else()
    message(STATUS "llama.cpp pin check skipped: no git, or ${llama.cpp_SOURCE_DIR} is not a full git checkout")
endif()

# Every ggml source allocating with ggml_aligned_malloc must also release with ggml_aligned_free.
# The AMX buffer type did not, which corrupted the Windows heap on AMX-capable CPUs and killed the
# process at cleanup; patches/0018 fixes it. Checked here on every platform in every build,
# because no machine without AMX can notice the defect at run time.
include(cmake/check-aligned-free-pairing.cmake)
jllama_check_aligned_free_pairing(${llama.cpp_SOURCE_DIR})

# nlohmann/json comes from upstream's vendored copy and from nowhere else (see the json block above).
# Asserted here rather than left to the compiler: a missing header fails every TU that includes it
# with a message about the header, while this one says where the file is expected and what to do.
# A future upstream move of vendor/nlohmann/ is the case it catches.
if(NOT EXISTS "${llama.cpp_SOURCE_DIR}/vendor/nlohmann/json.hpp")
    message(FATAL_ERROR
            "nlohmann/json.hpp is not at ${llama.cpp_SOURCE_DIR}/vendor/nlohmann/json.hpp.\n"
            "This project takes that header from llama.cpp's vendored copy, which reaches jllama "
            "through the vendor::nlohmann interface target of llama-common, and keeps no copy of its "
            "own on purpose: two copies of the same classes in one link is the httplib failure mode. "
            "If upstream moved the directory at this tag, point the include at the new location; do "
            "not re-add a FetchContent for it, which would put a second, differently-patched copy "
            "back on the include path.")
endif()
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
        # Layer A of the delivery roadmap (TODO.md, item 5): llama and mtmd are SHARED libraries next
        # to libjllama, exactly as upstream's release zips ship them (libllama.so / llama.dll,
        # libmtmd.so / mtmd.dll) -- so a consumer gets upstream's own C API (llama.h, mtmd.h, ggml.h)
        # from this project's patched source, and libjllama imports them instead of carrying a
        # copy. llama-common stays STATIC inside jllama: it is a C++ library (no stable ABI, and
        # WINDOWS_EXPORT_ALL_SYMBOLS when shared), and the server sources compiled into jllama are
        # its only consumer here. Upstream's common/CMakeLists.txt follows BUILD_SHARED_LIBS like
        # src/ does, so the root is added with LLAMA_BUILD_COMMON OFF (src/ -> llama SHARED), and
        # vendor/cpp-httplib (STATIC by declaration, BoringSSL with BUILD_SHARED_LIBS forced OFF by
        # upstream) plus common/ are added afterwards with BUILD_SHARED_LIBS OFF. mtmd is added
        # the same way further down (its own block, as in every build), with BUILD_SHARED_LIBS ON
        # there. Unversioned like ggml (the loader extracts plain files).
        set(LLAMA_BUILD_COMMON OFF)
        set(BUILD_SHARED_LIBS ON)
        add_subdirectory(${llama.cpp_SOURCE_DIR} ${llama.cpp_BINARY_DIR})
        set(BUILD_SHARED_LIBS OFF)
        set(LLAMA_BUILD_COMMON ON)
        get_target_property(_jllama_t_type llama TYPE)
        if(NOT _jllama_t_type STREQUAL "SHARED_LIBRARY")
            message(FATAL_ERROR "JLLAMA_BACKEND_DL: upstream built 'llama' as ${_jllama_t_type}, not SHARED -- re-check src/CMakeLists.txt at the new tag")
        endif()
        set_property(TARGET llama PROPERTY VERSION)
        set_property(TARGET llama PROPERTY SOVERSION)
        # common/ reads four build-info values and the version pair from the root's directory
        # scope, which is not visible from here: the number and the short hash of the fetched commit
        # (what llama_build_info() reports as `b<number>-<hash>`, held to LlamaCppVersion by
        # NativeLibraryLoadSmokeTest), the compiler and the target, each derived the way
        # cmake/build-info.cmake derives them, only in the fetched tree instead of in ours.
        if(GIT_EXECUTABLE AND EXISTS "${llama.cpp_SOURCE_DIR}/.git")
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
                WORKING_DIRECTORY "${llama.cpp_SOURCE_DIR}" OUTPUT_VARIABLE _jllama_common_commit
                OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _jllama_common_commit_res)
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
                WORKING_DIRECTORY "${llama.cpp_SOURCE_DIR}" OUTPUT_VARIABLE _jllama_common_number
                OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _jllama_common_number_res)
        endif()
        if(DEFINED LLAMA_BUILD_NUMBER AND DEFINED LLAMA_BUILD_COMMIT)
            # An archived tree: the pair was set from its stamp above.
        elseif(_jllama_common_commit_res EQUAL 0 AND _jllama_common_number_res EQUAL 0)
            set(LLAMA_BUILD_NUMBER ${_jllama_common_number})
            set(LLAMA_BUILD_COMMIT ${_jllama_common_commit})
        else()
            string(REGEX REPLACE "^b" "" LLAMA_BUILD_NUMBER "${JLLAMA_GIT_TAG}")
            set(LLAMA_BUILD_COMMIT "unknown")
            message(WARNING "llama-common build info: git could not answer in ${llama.cpp_SOURCE_DIR}; using ${JLLAMA_GIT_TAG}-unknown")
        endif()
        set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
        if(CMAKE_VS_PLATFORM_NAME)
            set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
        else()
            set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        if(NOT DEFINED LLAMA_VERSION_BASE)
            set(LLAMA_VERSION_BASE "0")
        endif()
        if(NOT DEFINED LLAMA_VERSION_MAJOR)
            set(LLAMA_VERSION_MAJOR "0")
        endif()
        # vendor/ adds cpp-httplib (and with it BoringSSL) only under LLAMA_BUILD_COMMON, so it is
        # added here, before common/, which links it.
        add_subdirectory(${llama.cpp_SOURCE_DIR}/vendor/cpp-httplib ${llama.cpp_BINARY_DIR}/vendor/cpp-httplib)
        add_subdirectory(${llama.cpp_SOURCE_DIR}/common ${llama.cpp_BINARY_DIR}/common)
        foreach(t llama-common cpp-httplib)
            get_target_property(_jllama_t_type ${t} TYPE)
            if(NOT _jllama_t_type STREQUAL "STATIC_LIBRARY")
                message(FATAL_ERROR "JLLAMA_BACKEND_DL: upstream built '${t}' as ${_jllama_t_type}, not STATIC -- it would ship as one more file; re-check upstream's CMake at the new tag")
            endif()
        endforeach()
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
    # And silence one warning class on this target only, because on Windows it is not a warning but a
    # flood: **at least 1,374,762** warning lines in one ROCm job, against 0 in the CUDA job and 8 in
    # the SYCL one (run 38149667195). The figure is a floor, not a total -- both counts were taken
    # while that job was still running, and the earlier one of 571,559 was simply read sooner; a
    # broader pattern returning *fewer* matches than a narrower one later is how the growth showed
    # itself. `ggml.h` defines GGML_API as `__declspec(dllimport) extern` for a consumer of
    # a shared ggml on _WIN32, and HIP compiles each translation unit once per GPU architecture for
    # the **device** target amdgcn, which has no DLL storage class -- so clang reports
    # "__declspec attribute 'dllimport' is not supported" for every GGML_API declaration, times 23
    # Windows GPU targets. Nothing is wrong: the attribute applies on the host pass and is
    # meaningless on the device one.
    #
    # Why suppress rather than live with it: this is the pipeline's critical path and it builds
    # uncached, the log becomes unreadable for a real error, and a log that large is its own risk
    # (GitHub truncates). Scoped to `ggml-hip` because that is the only target with device code, and
    # it holds upstream's kernels rather than any code of ours, so hiding an ignored attribute there
    # costs nothing we would otherwise learn. Guarded by WIN32 because the macro expands to
    # `visibility("default")` elsewhere and there is nothing to ignore.
    #
    # Upstream has the same flood in its own windows-rocm job -- worth an issue with these counts
    # (TODO.md, "Upstream submissions"), since the real fix belongs in ggml.h: a device pass should
    # not see the dllimport form at all.
    #
    # Measured again in run 38172107721, and the comparison is what decided the form of this: the
    # Windows ROCm job's log was **714,345 lines with 83,362 warning messages**, while the Linux ROCm
    # job -- the same upstream sources, the same TheRock clang, the same 23 GPU targets -- had **9
    # warnings in 2,455 lines**. The difference is not the arch count but the language: upstream
    # compiles the HIP sources as plain C++ on Windows (`CXX_IS_HIPCC` on `WIN32`) and as the HIP
    # language on Linux, and clang's C++ frontend reports what the HIP one does not --
    # `-Wsign-compare` 29,673 times, `-Wnested-anon-types` 17,964, `-Wmissing-noreturn` 16,378, then
    # unused variables, missing braces, unused parameters, cast-qual, unreachable breaks. **Every one
    # of them is in `ggml/src/ggml-cuda/*.cu`**, upstream's own code: not a single line of ours.
    #
    # So `-w` on this target on Windows, which subsumes the dllimport suppression above. Suppressing a
    # whole warning set is normally the wrong instrument, and the reason it is right here is the Linux
    # job: it compiles the same sources and keeps every warning, in a log short enough to read. The
    # information is preserved where it is legible and dropped where it is multiplied by 23 -- rather
    # than chasing one class per toolchain bump, which the eight above would otherwise require.
    # Errors are untouched: a real compile failure still fails the job.
    if(WIN32)
        target_compile_options(ggml-hip PRIVATE $<$<COMPILE_LANGUAGE:HIP,CXX>:-w>)
    endif()
endif()

# mtmd lives in tools/mtmd, which is only built when LLAMA_BUILD_TOOLS=ON.
# LLAMA_BUILD_TOOLS defaults to LLAMA_STANDALONE, which is OFF when llama.cpp
# is consumed via FetchContent.  Build mtmd explicitly so the target exists.
if(NOT JLLAMA_MODULE_ONLY AND NOT TARGET mtmd)
    # LLAMA_VERSION_BASE / LLAMA_VERSION_MAJOR are set inside llama.cpp's directory
    # scope and are not visible here.  tools/mtmd/CMakeLists.txt uses them in
    # set_target_properties() as VERSION/SOVERSION values; if a variable is empty the
    # token list after PROPERTIES becomes odd-length and CMake aborts with "incorrect
    # number of arguments".  Both are cosmetic for this build (static mtmd ignores
    # VERSION/SOVERSION, and the shared mtmd of a JLLAMA_BACKEND_DL build clears them
    # below), hence the neutral placeholder rather than a mirror of upstream's numbers.
    # (b10398 replaced the older LLAMA_INSTALL_VERSION with these two — upstream
    # #26839, "cmake: introduce semantic versioning".)
    if(NOT DEFINED LLAMA_VERSION_BASE)
        set(LLAMA_VERSION_BASE "0")
    endif()
    if(NOT DEFINED LLAMA_VERSION_MAJOR)
        set(LLAMA_VERSION_MAJOR "0")
    endif()
    if(JLLAMA_BACKEND_DL)
        # Layer A (see the JLLAMA_BACKEND_DL block): mtmd is a SHARED library next to libjllama,
        # unversioned like llama and ggml.
        set(BUILD_SHARED_LIBS ON)
    endif()
    add_subdirectory(${llama.cpp_SOURCE_DIR}/tools/mtmd ${llama.cpp_BINARY_DIR}/tools/mtmd)
    set(BUILD_SHARED_LIBS OFF)
    if(JLLAMA_BACKEND_DL)
        get_target_property(_jllama_t_type mtmd TYPE)
        if(NOT _jllama_t_type STREQUAL "SHARED_LIBRARY")
            message(FATAL_ERROR "JLLAMA_BACKEND_DL: upstream built 'mtmd' as ${_jllama_t_type}, not SHARED -- re-check tools/mtmd/CMakeLists.txt at the new tag")
        endif()
        set_property(TARGET mtmd PROPERTY VERSION)
        set_property(TARGET mtmd PROPERTY SOVERSION)
    endif()
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
# both, never a silent attempt. The tag is JLLAMA_GIT_TAG, the pin at the top of the FetchContent
# block -- the b<nnnn> number also when JLLAMA_GIT_COMMIT fetched the commit for it (a module jar and
# a library jar of the same number are the same commit either way, the pin check saw to that).
# The rest of the file -- the fetched upstream commit, this repository's version and commit, the
# patch set and its hash, compiler and toolkit -- plus the CycloneDX BOM next to it come from
# cmake/jllama-build-info.cmake; merge-native-artifacts.sh requires both files of every artifact.
include(cmake/jllama-build-info.cmake)
jllama_write_build_info()

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
    src/main/cpp/jllama_engine.cpp
    src/main/cpp/jllama_api.cpp
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
        ${CMAKE_SOURCE_DIR}/src/main/cpp/native_server_engine.cpp
        ${CMAKE_SOURCE_DIR}/src/main/cpp/native_server.cpp
    )
    # The C API's jllama_server_* functions (jllama_api.cpp) exist where the engine does.
    target_compile_definitions(jllama PRIVATE JLLAMA_HAS_NATIVE_SERVER)
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
# directory still reaches jllama transitively.  The same link is what supplies
# nlohmann/json.hpp to log_helpers.hpp and train_engine.cpp -- see the json block at
# the top of this file for why there is no second copy of it.
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
# Export hygiene (TODO.md, delivery roadmap, item 4): libjllama exports the JNI entry points and the
# jllama_* C API and nothing else. Hidden visibility for its own translation units (JNIEXPORT and
# JLLAMA_API mark the exported ones), and -- because the libraries it links statically were
# compiled with default visibility -- a linker export list that keeps every llama/ggml/mtmd/
# BoringSSL symbol local: a version script on ELF (Linux, Android), an exported-symbols list on
# macOS. Windows exports nothing without __declspec(dllexport), so it needs no list. The shipped
# export set is held to the same list by verify-native-deps.py.
set_target_properties(jllama PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(APPLE)
    target_link_options(jllama PRIVATE "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/jllama-exports-macos.txt")
    set_property(TARGET jllama APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jllama-exports-macos.txt)
elseif(NOT WIN32)
    target_link_options(jllama PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/jllama-exports.map")
    set_property(TARGET jllama APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jllama-exports.map)
endif()
target_include_directories(jllama PRIVATE
    src/main/cpp
    ${JNI_INCLUDE_DIRS}
    ${llama.cpp_SOURCE_DIR}/tools/mtmd
    ${llama.cpp_SOURCE_DIR}/tools/server)
# Note: the WebUI ui.h include dir (generated webui-generated/ or the empty stub
# src/main/cpp/webui_stub) is added by the "WebUI assets" conditional above.
target_link_libraries(jllama PRIVATE llama-common mtmd llama cpp-httplib)

# The C API (src/main/cpp/jllama_api.h, layer B of the delivery roadmap, TODO.md item 7): JLLAMA_BUILD
# selects dllexport on Windows (the ELF/Mach-O export lists above already let jllama_* through);
# JLLAMA_VERSION_STRING is what jllama_api_version() returns -- the net.ladenthin:llama version this
# build belongs to, resolved by cmake/jllama-build-info.cmake from the pom (or -DJLLAMA_VERSION=).
# The header ships in every library directory, next to upstream's llama.h/mtmd.h/ggml.h of layer A:
# named in jllama-files.txt so the loader extracts it next to the library and never loads it (the
# static platforms get a jllama-files.txt for that one entry; without the list the merge step would
# derive the header into jllama-extras.txt and LlamaLoader would System.load a header).
target_compile_definitions(jllama PRIVATE JLLAMA_BUILD JLLAMA_VERSION_STRING="${JLLAMA_VERSION_RESOLVED}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/jllama_api.h DESTINATION ${JLLAMA_DIR})
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
    set_target_properties(jllama llama mtmd ggml PROPERTIES
	  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${JLLAMA_DIR}
	  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
	  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${JLLAMA_DIR}
	)
else()
	set_target_properties(jllama llama mtmd ggml PROPERTIES
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
    # The two shared upstream libraries of layer A (llama, mtmd; absent in a module-only build) ride
    # with the ggml set: same directory, same $ORIGIN, same list files. On Windows they are imports
    # of jllama.dll like ggml.dll, so they go into jllama-extras.txt (preloaded by full path, in
    # dependency order) rather than jllama-files.txt.
    set(JLLAMA_UPSTREAM_SHARED)
    if(NOT JLLAMA_MODULE_ONLY)
        set(JLLAMA_UPSTREAM_SHARED llama mtmd)
    endif()
    set_target_properties(jllama ${JLLAMA_UPSTREAM_SHARED} ${JLLAMA_GGML_TARGETS} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
    set(JLLAMA_FILES_CONTENT "# Written by llama/CMakeLists.txt (JLLAMA_BACKEND_DL): extracted, not loaded\njllama_api.h\n")
    foreach(t ${JLLAMA_UPSTREAM_SHARED} ${JLLAMA_GGML_TARGETS})
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
        if(OS_NAME STREQUAL "Windows" AND (t STREQUAL "ggml" OR t STREQUAL "ggml-base" OR t STREQUAL "llama" OR t STREQUAL "mtmd"))
            continue()
        endif()
        string(APPEND JLLAMA_FILES_CONTENT "$<TARGET_FILE_NAME:${t}>\n")
    endforeach()
    # NEWLINE_STYLE LF on both lists: on Windows file(GENERATE) otherwise writes CRLF (measured), and
    # merge-native-artifacts.sh reads them on Linux -- it strips a CR itself now, but the extras
    # file below is meant to be byte-identical to what that script writes, which is LF.
    file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-files.txt CONTENT "${JLLAMA_FILES_CONTENT}" NEWLINE_STYLE LF)
else()
    # The static build (macOS, s390x, a plain local build) ships one library and the C API header;
    # the list exists so the header is extracted and not loaded (see the C API block above).
    file(WRITE ${JLLAMA_DIR}/jllama-files.txt "# Written by llama/CMakeLists.txt (static): extracted, not loaded\njllama_api.h\n")
endif()
if(JLLAMA_BACKEND_DL)
    if(OS_NAME STREQUAL "Windows")
        # The counterpart of the two names left out above: in CI merge-native-artifacts.sh derives
        # jllama-extras.txt as "every file next to the library that jllama-files.txt does not name,
        # sorted", which here is exactly ggml-base.dll then ggml.dll. Writing the same two lines from
        # the build (same order, no comment line, so the merge step rewrites an identical file) makes
        # a local variants build loadable without the merge step: without the file LlamaLoader
        # extracts jllama.dll alone, System.load resolves its ggml.dll import from java.exe's
        # directory or PATH, and every model-backed test fails on the load.
        # Dependency order, which is also the alphabetical order the merge script writes:
        # ggml-base.dll < ggml.dll < llama.dll (imports both) < mtmd.dll (imports llama and ggml).
        if(JLLAMA_MODULE_ONLY)
            file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-extras.txt
                 CONTENT "$<TARGET_FILE_NAME:ggml-base>\n$<TARGET_FILE_NAME:ggml>\n" NEWLINE_STYLE LF)
        else()
            file(GENERATE OUTPUT ${JLLAMA_DIR}/jllama-extras.txt
                 CONTENT "$<TARGET_FILE_NAME:ggml-base>\n$<TARGET_FILE_NAME:ggml>\n$<TARGET_FILE_NAME:llama>\n$<TARGET_FILE_NAME:mtmd>\n" NEWLINE_STYLE LF)
        endif()
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

# The plain-C consumer of both layers: a modular build (JLLAMA_BACKEND_DL, not module-only) builds
# src/test/c/test_c_abi.c against the shared llama and ggml-base libraries it just put next to
# libjllama (layer A) and against libjllama itself (layer B: jllama_api.h must compile as C and the
# jllama_* symbols must be exported) and registers it as the ctest `c_abi` -- the first and only test of the modular build
# jobs, which refuse BUILD_TESTING (jllama_test needs the CPU backend statically). build.sh and
# build.bat run `ctest -R c_abi` right after the build. Off when cross-compiling (the Android jobs
# cannot run the executable). Also useful locally: `ctest --test-dir build -R c_abi -V`.
option(JLLAMA_C_ABI_TEST "Build and run the plain-C test of the shared upstream libraries of a modular build" ON)
if(JLLAMA_BACKEND_DL AND NOT JLLAMA_MODULE_ONLY AND JLLAMA_C_ABI_TEST AND NOT CMAKE_CROSSCOMPILING)
    enable_language(C)
    add_executable(jllama_c_abi src/test/c/test_c_abi.c)
    target_compile_features(jllama_c_abi PRIVATE c_std_11)
    target_link_libraries(jllama_c_abi PRIVATE jllama llama ggml-base)
    target_include_directories(jllama_c_abi PRIVATE src/main/cpp)
    # The executable lives in the build tree and must resolve libllama/libggml* from the natives
    # directory: a run path pointing there on ELF, PATH for the test on Windows (where the
    # libraries are found through the loaded-module list and PATH, never the executable's rpath).
    set_target_properties(jllama_c_abi PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c-abi
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/c-abi
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/c-abi
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/c-abi
        BUILD_RPATH "${JLLAMA_DIR}"
        BUILD_WITH_INSTALL_RPATH FALSE
        SKIP_BUILD_RPATH FALSE)
    enable_testing()
    add_test(NAME c_abi COMMAND jllama_c_abi "${JLLAMA_DIR}")
    if(WIN32)
        set_tests_properties(c_abi PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${JLLAMA_DIR}")
    endif()
endif()

# BUILD_TESTING is declared at the top of this file, before any subproject (see there).
if(BUILD_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        # No constraint behind this exact version — GoogleTest is only used by this repo's own
        # C++ unit tests (jllama_test), not by the shipped library and not tied to llama.cpp.
        # It is just "latest stable at the time"; bump it from time to time (see CLAUDE.md).
        # Pinned by commit, not by the tag name it stands for (v1.18.0): tags move, commits do not.
        GIT_TAG        063de7e9578f82b369302001269680b4b1553359 # v1.18.0
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
        src/test/cpp/test_c_api.cpp
        src/main/cpp/jllama_engine.cpp
        src/main/cpp/jllama_api.cpp
        # The engines behind the C API's TTS and trainer functions (test_c_api.cpp drives them).
        src/main/cpp/tts_engine.cpp
        src/main/cpp/train_engine.cpp
        src/test/cpp/test_utils.cpp
        src/test/cpp/test_server.cpp
        src/test/cpp/test_jni_helpers.cpp
        src/test/cpp/test_json_helpers.cpp
        src/test/cpp/test_log_helpers.cpp
        src/test/cpp/test_common_log_callback.cpp
        src/test/cpp/test_tts_wav.cpp
        src/test/cpp/test_tts_params.cpp
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
        # ggml's internal src/ and the x86 CPU-backend directory -- test_prefetch.cpp includes
        # ggml-common.h (for block_q4_0 / block_q8_0, whose sizes the prefetch offsets depend on)
        # and prefetch.h, the helper patches/0017 adds. Same reasoning as the line above: the
        # guard has to drive the patched code, and that code is not on any public header path.
        ${llama.cpp_SOURCE_DIR}/ggml/src
        ${llama.cpp_SOURCE_DIR}/ggml/src/ggml-cpu/arch/x86
        # The generated Java-CLI-flag list consumed by test_model_flags.cpp.
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )
    target_link_libraries(jllama_test PRIVATE llama-common mtmd llama cpp-httplib GTest::gtest_main)
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
            src/main/cpp/native_server_engine.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-tools.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-mcp.cpp
            ${llama.cpp_SOURCE_DIR}/tools/server/server-http.cpp
        )
        target_include_directories(jllama_test PRIVATE ${CMAKE_SOURCE_DIR}/src/main/cpp/webui_stub)
        target_compile_definitions(jllama_test PRIVATE JLLAMA_HAS_NATIVE_SERVER)
        if(MSVC)
            target_compile_options(jllama_test PRIVATE /utf-8)
        endif()
        if(WIN32 AND NOT MSVC)
            target_link_libraries(jllama_test PRIVATE ws2_32)
        endif()
    endif()

    target_compile_definitions(jllama_test PRIVATE
        SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
        # test_c_api.cpp drives the C API compiled into this binary (jllama_api.cpp above), so the
        # same two definitions as the library: JLLAMA_BUILD (dllexport is harmless in an executable)
        # and the version jllama_api_version() reports.
        JLLAMA_BUILD
        JLLAMA_VERSION_STRING="${JLLAMA_VERSION_RESOLVED}"
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
