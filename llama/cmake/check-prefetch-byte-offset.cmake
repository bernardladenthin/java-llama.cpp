# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Every _mm_prefetch call of ggml's x86 quants.c takes its address from ggml_prefetch_at() -- checked
# at configure time, on every platform, in every build.
#
# Why a check of its own and not only src/test/cpp/test_prefetch.cpp: that test includes the header
# patches/0017 adds (prefetch.h) and holds its arithmetic, but it cannot see the call sites. Since
# b12525 upstream's own form of the four calls, `_mm_prefetch((const char *) (&x[ib] + sizeof(...)))`,
# compiles everywhere -- the cast around the whole expression silences clang 22 and keeps an address
# 18x/34x too far, because sizeof() is added in elements. A refresh of 0017 that kept the header and
# lost the quants.c hunks would therefore pass the test, the build and check-patches.py, and ship the
# wrong address again. This reads the patched file instead.

function(jllama_check_prefetch_byte_offset root)
    set(source "${root}/ggml/src/ggml-cpu/arch/x86/quants.c")
    if(NOT EXISTS "${source}")
        message(STATUS "prefetch byte offset: ${source} not found, nothing to check")
        return()
    endif()
    file(READ "${source}" text)
    string(REGEX MATCHALL "_mm_prefetch\\([^;]*" calls "${text}")
    set(offenders "")
    foreach(call IN LISTS calls)
        if(NOT call MATCHES "^_mm_prefetch\\(ggml_prefetch_at\\(")
            list(APPEND offenders "${call}")
        endif()
    endforeach()
    list(LENGTH calls n_calls)
    if(offenders)
        string(REPLACE ";" "\n  - " shown "${offenders}")
        message(FATAL_ERROR
            "_mm_prefetch in ggml/src/ggml-cpu/arch/x86/quants.c without ggml_prefetch_at():\n  - ${shown}\n"
            "&x[ib] + sizeof(block) adds sizeof() elements, not bytes; ggml_prefetch_at() (patches/0017) "
            "casts to a byte pointer first. Check whether 0017's quants.c hunks still apply -- the drop-check "
            "in its header says when upstream's own form would make the patch unnecessary. "
            "See llama/cmake/check-prefetch-byte-offset.cmake.")
    endif()
    message(STATUS "prefetch byte offset: ${n_calls} _mm_prefetch call(s) in quants.c, all through ggml_prefetch_at()")
endfunction()
