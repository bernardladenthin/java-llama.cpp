REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM
REM SPDX-License-Identifier: MIT OR Apache-2.0
REM
REM Windows x86_64 build with the OpenCL backend enabled, shipped as the
REM `opencl-windows-x86-64` classifier. The windows-2025 runner image ships
REM neither OpenCL headers nor an OpenCL import library, so this script first
REM stages Khronos OpenCL-Headers and builds OpenCL-ICD-Loader (producing
REM OpenCL.lib) before delegating the jllama configure+build to build.bat with
REM the OpenCL paths. Mirrors build_opencl_android.sh.
REM
REM At runtime the GPU vendor's ICD (System32\OpenCL.dll, installed by the
REM NVIDIA/AMD/Intel driver) provides the actual OpenCL symbols; we link only
REM against the loader's import library, so no OpenCL.dll is shipped.

@echo off
setlocal enabledelayedexpansion

set "OPENCL_STAGE=%RUNNER_TEMP%\opencl-stage"
if "%RUNNER_TEMP%"=="" set "OPENCL_STAGE=%TEMP%\opencl-stage"
set "HEADERS_DIR=%OPENCL_STAGE%\OpenCL-Headers"
set "LOADER_DIR=%OPENCL_STAGE%\OpenCL-ICD-Loader"
set "LOADER_BUILD=%LOADER_DIR%\build"

REM Pinned tags for reproducibility (OpenCL 3.1.1; match build_opencl_android.sh).
set "HEADERS_TAG=v2026.05.29"
set "LOADER_TAG=v2026.05.29"

if not exist "%HEADERS_DIR%" (
    git clone --depth 1 --branch %HEADERS_TAG% https://github.com/KhronosGroup/OpenCL-Headers.git "%HEADERS_DIR%"
    if errorlevel 1 exit /b 1
)

if not exist "%LOADER_BUILD%\Release\OpenCL.lib" if not exist "%LOADER_BUILD%\OpenCL.lib" (
    if not exist "%LOADER_DIR%" (
        git clone --depth 1 --branch %LOADER_TAG% https://github.com/KhronosGroup/OpenCL-ICD-Loader.git "%LOADER_DIR%"
        if errorlevel 1 exit /b 1
    )
    cmake -B "%LOADER_BUILD%" -S "%LOADER_DIR%" -DOPENCL_ICD_LOADER_HEADERS_DIR="%HEADERS_DIR%" -DBUILD_TESTING=OFF
    if errorlevel 1 exit /b 1
    cmake --build "%LOADER_BUILD%" --config Release
    if errorlevel 1 exit /b 1
)

REM Resolve the import library: multi-config generators emit build\Release\OpenCL.lib,
REM single-config ones emit build\OpenCL.lib.
set "OPENCL_LIB=%LOADER_BUILD%\Release\OpenCL.lib"
if not exist "%OPENCL_LIB%" set "OPENCL_LIB=%LOADER_BUILD%\OpenCL.lib"

REM Delegate to build.bat so the jllama build inherits the sccache probe + Depot
REM cache launcher and --show-stats output. The OpenCL paths satisfy ggml's
REM find_package(OpenCL); the caller appends -G/-DGGML_OPENCL/-DOS_* via %*.
call .github\build.bat -DOpenCL_INCLUDE_DIR="%HEADERS_DIR%" -DOpenCL_LIBRARY="%OPENCL_LIB%" %*
exit /b %ERRORLEVEL%
