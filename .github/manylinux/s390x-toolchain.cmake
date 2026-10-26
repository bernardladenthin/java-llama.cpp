# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# CMake toolchain for the cpu-linux-s390x natives jar (publish.yml build-linux-s390x): clang on the
# x86-64 runner cross-compiles for IBM Z against the manylinux_2_28 s390x image unpacked as a sysroot
# by manylinux_2_28-s390x-sysroot, and GNU ld of binutils-s390x-linux-gnu links. The result has the
# floor of the other Linux CPU jars: glibc 2.28, GLIBCXX_3.4.25, CXXABI_1.3.11 (verify-native-deps.py
# holds GLIBC; the README states all three).
#
#   JLLAMA_S390X_SYSROOT   the unpacked sysroot (cache variable or environment variable, required)
#   CMAKE_C_COMPILER /
#   CMAKE_CXX_COMPILER     clang-18 / clang++-18 by default (Ubuntu 24.04's clang; the runner is
#                          pinned to ubuntu-24.04 for it)
#
# Why clang and not a GCC: no packaged x86-64 GCC targets s390x with glibc 2.28, and the image's own
# gcc-toolset-14 is an s390x binary, which would have to run under qemu for the whole build (hours,
# and every tag build is uncached). clang is a cross compiler by construction: --gcc-install-dir
# points it at gcc-toolset-14's directory inside the sysroot, so it takes the same crt files, libgcc
# and libstdc++ (GCC 14 headers, the system libstdc++.so.6 plus libstdc++_nonshared.a through the
# toolset's linker script) that a native gcc-toolset build would. The linker scripts name absolute
# paths (/usr/lib64/libstdc++.so.6, /lib64/libgcc_s.so.1), which ld resolves inside --sysroot.
#
# -march=z13 is the arch11 baseline the previous GCC cross build defaulted to (and RHEL 8's): z13 is
# the oldest machine RHEL 8, SLES 15 and Ubuntu support. It does not switch on ggml's vector code:
# that needs __VEC__ (-mzvector), which nothing passes -- the job builds the scalar s390x path, see
# the GGML_NATIVE note in publish.yml.
#
# CMAKE_CROSSCOMPILING_EMULATOR runs the test binaries under qemu-user with the sysroot as the
# loader prefix, so ctest exercises the jar's libraries against glibc 2.28 and the GCC 8
# libstdc++.so.6 -- the oldest runtime the jar promises, not the build host's. The prefix goes in
# as QEMU_LD_PREFIX, not as qemu's -L: googletest's death tests re-execute the test binary, and that
# child reaches qemu through binfmt_misc without any argument, so only the environment carries the
# prefix to it (with -L the six RpcComm death tests failed on "Could not open '/lib/ld64.so.1'").

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR s390x)

# Cached, so that the reconfigure a build triggers on its own (a changed CMakeLists.txt) still finds
# the sysroot when the environment variable is gone.
if(NOT JLLAMA_S390X_SYSROOT AND DEFINED ENV{JLLAMA_S390X_SYSROOT})
    set(JLLAMA_S390X_SYSROOT "$ENV{JLLAMA_S390X_SYSROOT}" CACHE PATH
        "The manylinux_2_28 s390x sysroot (.github/manylinux/manylinux_2_28-s390x-sysroot)")
endif()
if(NOT JLLAMA_S390X_SYSROOT OR NOT EXISTS "${JLLAMA_S390X_SYSROOT}/usr/lib64/libc.so.6")
    message(FATAL_ERROR "s390x-toolchain.cmake: set JLLAMA_S390X_SYSROOT to a sysroot unpacked by "
                        ".github/manylinux/manylinux_2_28-s390x-sysroot (got '${JLLAMA_S390X_SYSROOT}')")
endif()
# try_compile projects re-read this file; hand them the sysroot too.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES JLLAMA_S390X_SYSROOT)

set(CMAKE_SYSROOT "${JLLAMA_S390X_SYSROOT}")
if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang-18)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++-18)
endif()
set(CMAKE_C_COMPILER_TARGET s390x-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET s390x-linux-gnu)

set(_jllama_s390x_gcc "${JLLAMA_S390X_SYSROOT}/opt/rh/gcc-toolset-14/root/usr/lib/gcc/s390x-redhat-linux/14")
set(CMAKE_C_FLAGS_INIT "--gcc-install-dir=${_jllama_s390x_gcc} -march=z13")
set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_jllama_s390x_gcc} -march=z13")
# s390x-typeinfo-names.ld: clang aligns typeinfo name strings like char, the GCC code of
# libstdc++_nonshared.a loads them with LARL, which needs an even address (see the script).
foreach(_kind EXE SHARED MODULE)
    set(CMAKE_${_kind}_LINKER_FLAGS_INIT
        "--ld-path=/usr/bin/s390x-linux-gnu-ld -Wl,-T,${CMAKE_CURRENT_LIST_DIR}/s390x-typeinfo-names.ld")
endforeach()

set(CMAKE_CROSSCOMPILING_EMULATOR /usr/bin/env "QEMU_LD_PREFIX=${JLLAMA_S390X_SYSROOT}" /usr/bin/qemu-s390x-static)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
