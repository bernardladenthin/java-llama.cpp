#!/bin/sh

# A Cuda 13.2 install script for RHEL8/Rocky8/Manylinux_2.28
# Available versions can be found at:
# https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/

sudo dnf install -y kernel-devel kernel-headers
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

sudo dnf install -y cuda-toolkit-13-2

exec .github/build.sh $@ -DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
