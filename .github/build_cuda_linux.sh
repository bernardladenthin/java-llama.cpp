#!/bin/sh

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# A Cuda 13.4 install script for RHEL8/Rocky8/Manylinux_2.28
# Available versions can be found at:
# https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/

sudo dnf install -y kernel-devel kernel-headers
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

sudo dnf install -y cuda-toolkit-13-4

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
# Release default: an EXPLICIT arch list, because ggml's own default leaves the datacenter GPUs on
# PTX and this project's consumers are server applications.
#
# ggml states its policy in ggml/src/ggml-cuda/CMakeLists.txt: build *virtual* architectures "to
# cover all features needed for best performance" and *real* ones "for the most commonly used GPUs"
# -- and for CUDA >= 13 the real ones are 86 (RTX 3000), 89 (RTX 4000) and consumer Blackwell, while
# 75 (T4, RTX 20xx), 80 (A100) and 90 (H100) are `-virtual`, i.e. PTX that the driver must JIT on
# first run. Upstream passes no CMAKE_CUDA_ARCHITECTURES at all and says so in a comment, so their
# binaries carry the same gap -- which is right for their users, who download a zip for an RTX card.
#
# It is wrong for ours. A Maven dependency lands in server applications, and those run on T4, A100,
# L4 and H100: for this project the "most commonly used GPUs" are exactly the ones upstream leaves
# as PTX. And the failure is not graceful. Measured on a Tesla T4 with driver 595.91.07 (AWS Deep
# Learning AMI): the module opened, ggml enumerated the GPU, offloaded 29/29 layers and then
# ggml_abort killed the JVM with "the provided PTX was compiled with an unsupported toolchain" --
# the driver's JIT refusing PTX from CUDA 13.4. `-ngl 0` aborts too, so having the jar on the
# classpath is enough, and abort() cannot be caught from Java. See
# docs/benchmarks/gpu-module-jars.md.
#
# So 75, 80 and 90 become `-real` as well. Everything ggml lists stays, including the virtual tail
# that lets a future architecture JIT. The price is paid in the build, not by the consumer: nvcc
# compiles every .cu once per real architecture, so this roughly doubles the device-compile half of
# a job that already takes 47-80 minutes uncached, and the module grows from 154 MB by about the
# same proportion. That trade is the point -- a jar that is larger and slower to build, against one
# that kills the JVM on the most common cloud GPU.
#
# On a bump, re-read ggml's list at the new tag (`grep -A 30 CUDA_ARCHITECTURES
# ggml/src/ggml-cuda/CMakeLists.txt`) and extend this one rather than dropping it: new hardware
# arrives as a new entry there, and silently inheriting their default would reintroduce the gap.
CUDA_ARCH_RELEASE="75-real;80-real;86-real;89-real;90-real;100-real;120-real;121-real;121-virtual"

CUDA_ARCH_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_RELEASE}"
case "${CUDA_FAST_BUILD:-}" in
  1 | true | TRUE | yes | on)
    CUDA_ARCH_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native}"
    echo "build_cuda_linux.sh: CUDA_FAST_BUILD set -> ${CUDA_ARCH_ARGS} (DEV ONLY — not release-distributable)"
    ;;
esac

# CCCL v3.4.3 is fetched instead of the one CUDA 13.4 bundles, as upstream's own CUDA release builds do
# (llama.cpp #29792): ggml's CUB DeviceTopK path needs >= 3.4.3 (an earlier race, NVIDIA/cccl#10627) and
# falls back to a sort below it. Drop the flag once the toolkit is 13.5+, which bundles CCCL 3.5.
exec .github/build.sh $@ -DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc -DGGML_CUDA_CCCL_VERSION=v3.4.3 $CUDA_ARCH_ARGS
