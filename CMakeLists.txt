cmake_minimum_required(VERSION 3.14)

project(jllama CXX)

include(FetchContent)

set(BUILD_SHARED_LIBS ON)
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

# Android standalone toolchains default to API level 21, but cpp-httplib uses
# getifaddrs/freeifaddrs which are only declared in <ifaddrs.h> when
# __ANDROID_API__ >= 24. NDK unified sysroots (r14b+) include the symbol in
# libc regardless of level; this exposes the declaration for all targets.
if(ANDROID_ABI)
    add_compile_definitions(__ANDROID_API__=24)
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
FetchContent_Declare(
	llama.cpp
	GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
	GIT_TAG        b8887
)
FetchContent_MakeAvailable(llama.cpp)

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
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --os
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
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --arch
      OUTPUT_VARIABLE OS_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
endif()
if(NOT OS_ARCH)
    message(FATAL_ERROR "Could not determine CPU architecture")
endif()

if(GGML_CUDA)
    set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources_linux_cuda/de/kherud/llama/${OS_NAME}/${OS_ARCH})
    message(STATUS "GPU (CUDA Linux) build - Installing files to ${JLLAMA_DIR}")
else()
    set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources/de/kherud/llama/${OS_NAME}/${OS_ARCH})
    message(STATUS "CPU build - Installing files to ${JLLAMA_DIR}")
endif()

# include jni.h and jni_md.h
if(NOT DEFINED JNI_INCLUDE_DIRS)
    if(OS_NAME MATCHES "^Linux" OR OS_NAME STREQUAL "Mac" OR OS_NAME STREQUAL "Darwin")
        set(JNI_INCLUDE_DIRS .github/include/unix)
    elseif(OS_NAME STREQUAL "Windows")
        set(JNI_INCLUDE_DIRS .github/include/windows)
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

add_library(jllama SHARED src/main/cpp/jllama.cpp src/main/cpp/server.hpp src/main/cpp/utils.hpp)

set_target_properties(jllama PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(jllama PRIVATE src/main/cpp ${JNI_INCLUDE_DIRS} ${llama.cpp_SOURCE_DIR}/tools/mtmd)
target_link_libraries(jllama PRIVATE llama-common mtmd llama nlohmann_json)
target_compile_features(jllama PRIVATE cxx_std_11)

target_compile_definitions(jllama PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
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

option(BUILD_TESTING "Build C++ unit tests for server.hpp / utils.hpp" OFF)

if(BUILD_TESTING)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Prevent GTest from overriding our compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(jllama_test
        src/test/cpp/test_utils.cpp
        src/test/cpp/test_server.cpp
        src/test/cpp/test_jni_helpers.cpp
        src/test/cpp/test_json_helpers.cpp
    )

    target_include_directories(jllama_test PRIVATE
        src/main/cpp
        # mtmd.h is not always propagated transitively — add it explicitly
        ${llama.cpp_SOURCE_DIR}/tools/mtmd
        # jni.h / jni_md.h needed by jni_helpers.hpp (mock JNI tests, no JVM required)
        ${JNI_INCLUDE_DIRS}
    )
    target_link_libraries(jllama_test PRIVATE llama-common mtmd llama nlohmann_json GTest::gtest_main)
    target_compile_features(jllama_test PRIVATE cxx_std_17)

    target_compile_definitions(jllama_test PRIVATE
        SERVER_VERBOSE=$<BOOL:${LLAMA_VERBOSE}>
    )

    gtest_discover_tests(jllama_test)
endif()
