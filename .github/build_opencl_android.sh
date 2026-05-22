#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Android arm64 build with the OpenCL backend enabled and Adreno-tuned
# kernels embedded. Runs inside the dockcross/android-arm64 container.
#
# The dockcross image does not ship OpenCL headers or a libOpenCL.so stub,
# so this script first stages Khronos OpenCL-Headers and cross-builds
# OpenCL-ICD-Loader to satisfy `find_package(OpenCL REQUIRED)` at link
# time. At runtime the device's vendor ICD (e.g. Qualcomm Adreno driver)
# provides the actual OpenCL symbols.

set -eu

# Forward args to the inner cmake invocation.
EXTRA_ARGS=("$@")

OPENCL_STAGE=/tmp/opencl-stage
HEADERS_DIR="$OPENCL_STAGE/OpenCL-Headers"
LOADER_DIR="$OPENCL_STAGE/OpenCL-ICD-Loader"
LOADER_BUILD="$LOADER_DIR/build"

# Pinned tags for reproducibility.
HEADERS_TAG=v2025.07.22
LOADER_TAG=v2025.07.22

if [ ! -d "$HEADERS_DIR" ]; then
    mkdir -p "$OPENCL_STAGE"
    git clone --depth 1 --branch "$HEADERS_TAG" \
        https://github.com/KhronosGroup/OpenCL-Headers.git "$HEADERS_DIR"
fi

if [ ! -f "$LOADER_BUILD/libOpenCL.so" ]; then
    if [ ! -d "$LOADER_DIR" ]; then
        git clone --depth 1 --branch "$LOADER_TAG" \
            https://github.com/KhronosGroup/OpenCL-ICD-Loader.git "$LOADER_DIR"
    fi
    cmake -B "$LOADER_BUILD" -S "$LOADER_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENCL_ICD_LOADER_HEADERS_DIR="$HEADERS_DIR" \
        -DBUILD_TESTING=OFF
    cmake --build "$LOADER_BUILD" --config Release -j"$(nproc)"
fi

mkdir -p build
cmake -Bbuild \
    -DOpenCL_INCLUDE_DIR="$HEADERS_DIR" \
    -DOpenCL_LIBRARY="$LOADER_BUILD/libOpenCL.so" \
    "${EXTRA_ARGS[@]}"
cmake --build build --config Release -j"$(nproc)"
