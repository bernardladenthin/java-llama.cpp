# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# msvc-runtime-gnu-clang.cmake -- the MSVC runtime of plain clang on Windows, selected by the driver.
#
# Loaded through CMAKE_USER_MAKE_RULES_OVERRIDE (set before project() in llama/CMakeLists.txt), i.e.
# once per enabled language, right after CMake's own Platform/Windows-Clang.cmake. For clang with the
# GNU front end, that file implements CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded as
#
#     -Xclang -flto-visibility-public-std -D_MT -Xclang --dependent-lib=libcmt
#
# and sccache (0.18.0, and main as of 2026-10-22) refuses every TU that carries an `-Xclang` argument it
# does not know -- `-flto-visibility-public-std` is one ("Can't handle UnknownFlag arguments with
# -Xclang"). So the Windows ROCm job (TheRock's clang compiles the ~150 HIP TUs as C++ for 23 GPU
# targets) and the plain-clang CPU job cached nothing at all.
#
# The driver flag -fms-runtime-lib=static hands cc1 the same runtime selection (-D_MT,
# -flto-visibility-public-std, --dependent-lib=libcmt, plus --dependent-lib=oldnames as cl.exe /MT
# does), and llama/CMakeLists.txt already adds it for this toolchain. Measured with TheRock's clang
# 23 and MSVC 14.44 (the windows-2022 runner's): every section of every object identical -- .text and
# the HIP device code included -- except .drectve, where CMake's sequence repeated
# /DEFAULTLIB:libcmt.lib. CLAUDE.md ("sccache on every Windows Ninja job") has the measurement.
#
# Why the option variables and not an empty CMAKE_MSVC_RUNTIME_LIBRARY: the variable is handed to
# try_compile, and an empty one leaves the compiler check linking -nostdlib without any CRT
# (`undefined symbol: mainCRTStartup`, configure fails). With this override the variable keeps its
# value and try_compile keeps working; only the flags it maps to change.
#
# clang-cl, cl.exe and icx (IntelLLVM) are not Clang with the GNU front end, so they are untouched.
if(WIN32)
    foreach(_jllama_lang C CXX)
        if(CMAKE_${_jllama_lang}_COMPILER_ID STREQUAL "Clang"
           AND CMAKE_${_jllama_lang}_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
            set(CMAKE_${_jllama_lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded -fms-runtime-lib=static)
            set(CMAKE_${_jllama_lang}_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug -fms-runtime-lib=static_dbg)
        endif()
    endforeach()
    unset(_jllama_lang)
endif()
