#!/bin/sh

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

# A Cuda 13.3 install script for RHEL8/Rocky8/Manylinux_2.28
# Available versions can be found at:
# https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/

sudo dnf install -y kernel-devel kernel-headers
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

sudo dnf install -y cuda-toolkit-13-3

# CUDA target architectures — LOCAL-dev build-speed knob.
#
# Default (CUDA_FAST_BUILD unset): we do NOT pass CMAKE_CUDA_ARCHITECTURES, so ggml/llama.cpp
# compiles its full default arch set — exactly what release artifacts must ship (every supported
# GPU generation). On a cold cache this is the slow part (nvcc recompiles each .cu kernel once per
# architecture), but build.sh now wraps nvcc with sccache (-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache),
# so on a warm Depot cache the per-arch .cu device passes are served from cache (verified: 100%
# CUDA hit rate, ~51 min cold -> ~15 min warm). CI therefore always builds the full set and relies
# on the warm cache for speed — it no longer sets CUDA_FAST_BUILD.
#
# Dev fast build (CUDA_FAST_BUILD=1): compile for a SINGLE architecture instead of the full set,
# removing most of the nvcc time on a COLD cache. Defaults to `native` (the build machine's own
# GPU — needs a GPU present at configure time); override with CUDA_ARCH, e.g. CUDA_ARCH=90. This is
# a MANUAL local-dev knob only: CI and release never set it, because an artifact built this
# way runs on a single GPU generation. (Direct-cmake equivalent: -DCMAKE_CUDA_ARCHITECTURES=native.)
CUDA_ARCH_ARGS=""
case "${CUDA_FAST_BUILD:-}" in
  1 | true | TRUE | yes | on)
    CUDA_ARCH_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native}"
    echo "build_cuda_linux.sh: CUDA_FAST_BUILD set -> ${CUDA_ARCH_ARGS} (DEV ONLY — not release-distributable)"
    ;;
esac

exec .github/build.sh $@ -DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc $CUDA_ARCH_ARGS
