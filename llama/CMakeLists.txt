cmake_minimum_required(VERSION 3.22)

project(jllama CXX)

# Use static MSVC runtime (/MT) instead of the default DLL runtime (/MD).
# This embeds the C++ runtime into jllama.dll so msvcp140.dll / vcruntime140.dll
# are not required on the end-user's machine.
# Must be set before any FetchContent_MakeAvailable() so that llama.cpp and all
# other subprojects inherit the same CRT choice (mixing /MT and /MD in a single
# link is a linker error).
#
# EXCEPTION: the Intel oneAPI SYCL and OpenVINO backends must use the DYNAMIC /MD
# runtime — `icx -fsycl` rejects /MT outright ("invalid argument 'MT' not allowed
# with '-fsycl'") and the OpenVINO import libraries are built /MD (mixing would be a
# link error). Those classifiers already require the vendor runtime on the host, so
# the self-contained-DLL rationale does not apply to them; the CPU + CUDA/Vulkan/OpenCL
# classifiers keep /MT.
if(MSVC AND NOT GGML_SYCL AND NOT GGML_OPENVINO)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
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
set(LLAMA_CURL OFF)

option(LLAMA_VERBOSE	"llama: verbose output"		OFF)

#################### json ####################

FetchContent_Declare(
	json
	GIT_REPOSITORY https://github.com/nlohmann/json
	GIT_TAG        v3.12.0
)
FetchContent_MakeAvailable(json)

#################### llama.cpp ####################

# GCC < 9 requires explicit linking of stdc++fs for std::filesystem (C++17).
# This affects cross-compilation toolchains such as dockcross/linux-arm64-lts.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    link_libraries(stdc++fs)
endif()

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
# Enable HTTPS model downloads via curl
set(LLAMA_CURL ON)
# Build BoringSSL to include OpenSSL DLLs in Windows packages for HTTPS support
if(WIN32)
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
FetchContent_Declare(
	llama.cpp
	GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
	GIT_TAG        b10092
	PATCH_COMMAND  ${CMAKE_COMMAND}
		-DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
		-DLLAMA_SRC=<SOURCE_DIR>
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-llama-patches.cmake
)
FetchContent_MakeAvailable(llama.cpp)

# OuteTTS native pipeline: DERIVE the upstream tts.cpp helpers (DSP + prompt + text + the default
# speaker profile) into a compilable translation unit at configure time, rather than hand-copying
# them — a hand copy is a DRY/maintenance hazard that silently diverges on every llama.cpp upgrade.
# tts.cpp cannot simply be added to target_sources because it defines its own main(); the generator
# drops main() and gives the helpers external linkage. See cmake/generate-tts-upstream.cmake. The
# generated file is never committed; it is regenerated from whatever tts.cpp the pinned GIT_TAG
# resolves to, so a version bump is picked up automatically. The tag below is cosmetic provenance in
# the generated banner — keep it in sync with the llama.cpp GIT_TAG above.
set(JLLAMA_TTS_GEN_DIR ${CMAKE_BINARY_DIR}/tts_generated)
set(JLLAMA_TTS_GEN_CPP ${JLLAMA_TTS_GEN_DIR}/tts_upstream_gen.cpp)
file(MAKE_DIRECTORY ${JLLAMA_TTS_GEN_DIR})
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DTTS_SRC=${llama.cpp_SOURCE_DIR}/tools/tts/tts.cpp
        -DOUT_CPP=${JLLAMA_TTS_GEN_CPP}
        -DLLAMA_TAG=b10092
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate-tts-upstream.cmake
    RESULT_VARIABLE JLLAMA_TTS_GEN_RESULT
)
if(NOT JLLAMA_TTS_GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "OuteTTS extraction failed; see cmake/generate-tts-upstream.cmake")
endif()

# b8831 added ggml_graph_next_uid() which calls _InterlockedIncrement64 via
# <intrin.h> on x86. The intrinsic only exists on x64; provide the
# implementation in a compat TU so the linker resolves __InterlockedIncrement64.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_sources(ggml-base PRIVATE
        ${CMAKE_SOURCE_DIR}/src/main/cpp/compat/ggml_x86_compat.c)
endif()

# mtmd lives in tools/mtmd, which is only built when LLAMA_BUILD_TOOLS=ON.
# LLAMA_BUILD_TOOLS defaults to LLAMA_STANDALONE, which is OFF when llama.cpp
# is consumed via FetchContent.  Build mtmd explicitly so the target exists.
if(NOT TARGET mtmd)
    # LLAMA_INSTALL_VERSION is set inside llama.cpp's directory scope and is not
    # visible here.  tools/mtmd/CMakeLists.txt uses it in set_target_properties()
    # as a VERSION value; if the variable is empty the token list after PROPERTIES
    # becomes odd-length and CMake aborts with "incorrect number of arguments".
    if(NOT DEFINED LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION "0")
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

# Backend + OS routing for the native-library output tree. Each GPU backend ships
# under its own Maven classifier, so it must land in a backend-specific resource
# root (the default CPU tree stays src/main/resources/). The GPU branches are
# OS-aware because the same GGML flag is used on more than one platform:
#   - GGML_CUDA     -> Linux (resources_linux_cuda)   AND Windows (resources_windows_cuda)
#   - GGML_OPENCL   -> Android (resources_android_opencl) AND Windows (resources_windows_opencl)
#   - GGML_VULKAN   -> Windows (resources_windows_vulkan) AND Linux (resources_linux_vulkan)
#   - GGML_HIP      -> Linux (resources_linux_rocm)    AND Windows (resources_windows_rocm)   [AMD ROCm/HIP]
#   - GGML_SYCL     -> Windows (resources_windows_sycl) AND Linux (fp16/fp32 split, see below) [Intel oneAPI]
#   - GGML_OPENVINO -> Linux (resources_linux_openvino) AND Windows (resources_windows_openvino) [Intel OpenVINO]
# The classifier->tree mapping is mirrored by the matching Maven profile in pom.xml. The Linux
# Vulkan tree holds both x86_64 and aarch64 under Linux/${OS_ARCH}; two Maven profiles
# (vulkan-linux / vulkan-linux-aarch64) split it into one single-arch classifier JAR each. The
# Windows OpenCL tree likewise holds both x86_64 (desktop ICD) and aarch64 (Snapdragon/Adreno),
# split by the opencl-windows / opencl-windows-aarch64 profiles. Linux SYCL ships two precision
# variants at the SAME arch, so it is routed to two distinct trees by GGML_SYCL_F16 (fp16 vs fp32).
if(GGML_CUDA)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_cuda/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (CUDA Windows) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_cuda/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (CUDA Linux) build - Installing files to ${JLLAMA_DIR}")
    endif()
elseif(GGML_VULKAN)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_vulkan/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (Vulkan Windows) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_vulkan/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (Vulkan Linux) build - Installing files to ${JLLAMA_DIR}")
    endif()
elseif(GGML_OPENCL)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_opencl/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (OpenCL Windows) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_android_opencl/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (OpenCL Android) build - Installing files to ${JLLAMA_DIR}")
    endif()
elseif(GGML_HIP)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_rocm/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (ROCm/HIP Windows) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_rocm/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (ROCm/HIP Linux) build - Installing files to ${JLLAMA_DIR}")
    endif()
elseif(GGML_SYCL)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_sycl/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (SYCL Windows) build - Installing files to ${JLLAMA_DIR}")
    elseif(GGML_SYCL_F16)
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_sycl_fp16/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (SYCL Linux fp16) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_sycl_fp32/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (SYCL Linux fp32) build - Installing files to ${JLLAMA_DIR}")
    endif()
elseif(GGML_OPENVINO)
    if(OS_NAME STREQUAL "Windows")
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_windows_openvino/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (OpenVINO Windows) build - Installing files to ${JLLAMA_DIR}")
    else()
        set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_openvino/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
        message(STATUS "GPU (OpenVINO Linux) build - Installing files to ${JLLAMA_DIR}")
    endif()
else()
    set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources/net/ladenthin/llama/${OS_NAME}/${OS_ARCH})
    message(STATUS "CPU build - Installing files to ${JLLAMA_DIR}")
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
    ${JLLAMA_TTS_GEN_CPP}
    src/main/cpp/utils.hpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-common.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-chat.cpp)

# The generated TU keeps the whole pre-main() span of tts.cpp, so a few upstream CLI-only
# helpers (print_usage, save_wav16, xterm colour) come along unused. Silence the resulting
# unused-function warning on that one file (non-MSVC; MSVC's C4505 is off by default).
if(NOT MSVC)
    set_source_files_properties(${JLLAMA_TTS_GEN_CPP} PROPERTIES COMPILE_FLAGS "-Wno-unused-function")
endif()

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
target_sources(jllama PRIVATE
    ${llama.cpp_SOURCE_DIR}/tools/server/server-context.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-queue.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-task.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-schema.cpp
    ${llama.cpp_SOURCE_DIR}/tools/server/server-stream.cpp
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
# forwarded — inside libjllama, with no separate llama-server executable. server.cpp and
# server-tools.cpp both pull in vendor/sheredom/subprocess.h (posix_spawn_*), so they share the
# non-Android guard used for server-models.cpp above; native_server.cpp links against llama_server
# and is guarded too. On Android the NativeServer native methods are simply absent (its JNI calls
# throw UnsatisfiedLinkError) — use OpenAiCompatServer there.
if(NOT ANDROID_ABI AND NOT OS_NAME MATCHES "Android")
    target_sources(jllama PRIVATE
        ${llama.cpp_SOURCE_DIR}/tools/server/server-tools.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server.cpp
        ${CMAKE_SOURCE_DIR}/src/main/cpp/native_server.cpp
    )
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
# <cpp-httplib/httplib.h> already resolves via llama-common's vendor/ include dir,
# whose bundled nlohmann/json is the same 3.12.0 as our FetchContent copy, so
# adding nothing there shadows it.
target_sources(jllama PRIVATE
    ${llama.cpp_SOURCE_DIR}/tools/server/server-http.cpp
    ${llama.cpp_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp
)

# cpp-httplib is third-party: silence its warnings (matching upstream's own
# cpp-httplib target, which compiles it with -w / /w).  No SSL is enabled —
# CPPHTTPLIB_OPENSSL_SUPPORT is left undefined — so the embedded server is
# plain-HTTP for now (bind to localhost or front it with a TLS proxy).
if(MSVC)
    set_source_files_properties(
        ${llama.cpp_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp
        PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(
        ${llama.cpp_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp
        PROPERTIES COMPILE_FLAGS "-w")
endif()

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
target_link_libraries(jllama PRIVATE llama-common mtmd llama nlohmann_json)
target_compile_features(jllama PRIVATE cxx_std_11)

target_compile_definitions(jllama PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
    # cpp-httplib tuning — mirror the defines upstream's cpp-httplib target sets so
    # httplib.cpp and every TU that includes httplib.h (server-http.cpp) agree on
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

if (LLAMA_METAL AND NOT LLAMA_METAL_EMBED_LIBRARY)
    # copy ggml-common.h and ggml-metal.metal to bin directory
    configure_file(${llama.cpp_SOURCE_DIR}/ggml-metal.metal ${JLLAMA_DIR}/ggml-metal.metal COPYONLY)
endif()

#################### C++ unit tests ####################

option(BUILD_TESTING "Build C++ unit tests for jni_helpers / json_helpers / utils" OFF)

if(BUILD_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        # No constraint behind this exact tag — GoogleTest is only used by this repo's own
        # C++ unit tests (jllama_test), not by the shipped library and not tied to llama.cpp.
        # It is just "latest stable at the time"; bump it from time to time (see CLAUDE.md).
        GIT_TAG        v1.17.0
    )
    # Keep GTest on the same CRT as the rest of the project.
    # OFF means GTest respects CMAKE_MSVC_RUNTIME_LIBRARY (static /MT here).
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(jllama_test
        src/test/cpp/test_utils.cpp
        src/test/cpp/test_server.cpp
        src/test/cpp/test_jni_helpers.cpp
        src/test/cpp/test_json_helpers.cpp
        src/test/cpp/test_log_helpers.cpp
        src/test/cpp/test_tts_wav.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-common.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-chat.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-context.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-queue.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-task.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-schema.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-models.cpp
        ${llama.cpp_SOURCE_DIR}/tools/server/server-stream.cpp
    )

    target_include_directories(jllama_test PRIVATE
        src/main/cpp
        # mtmd.h is not always propagated transitively — add it explicitly
        ${llama.cpp_SOURCE_DIR}/tools/mtmd
        # jni.h / jni_md.h needed by jni_helpers.hpp (mock JNI tests, no JVM required)
        ${JNI_INCLUDE_DIRS}
        ${llama.cpp_SOURCE_DIR}/tools/server
    )
    target_link_libraries(jllama_test PRIVATE llama-common mtmd llama nlohmann_json GTest::gtest_main)
    target_compile_features(jllama_test PRIVATE cxx_std_17)

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
