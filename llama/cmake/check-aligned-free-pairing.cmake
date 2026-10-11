# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Every ggml source that allocates with ggml_aligned_malloc must also release with
# ggml_aligned_free -- checked at configure time, on every platform, in every build.
#
# Why this is worth a check of its own. On Windows ggml_aligned_malloc is `_aligned_malloc` and
# ggml_aligned_free is `_aligned_free` (ggml.c), and passing an `_aligned_malloc` pointer to plain
# `free()` is undefined behaviour that corrupts the process heap. The AMX buffer type did exactly
# that -- allocating with ggml_aligned_malloc at amx.cpp and releasing with `free(buffer->context)`
# -- so every model load that used an AMX-capable CPU killed the process at cleanup with
# 0xC0000374 STATUS_HEAP_CORRUPTION, after the inference had already succeeded. Measured on a rented
# Granite Rapids machine: six runs with the `sapphirerapids` module crashed (including at one
# thread, which rules out a data race), while `cooperlake`, `icelake` and `haswell` on the same
# machine and model were clean. In CI it showed up only as `Tests run: 0` on whichever runs happened
# to land on AMX hardware.
#
# The pairing is what makes it checkable without that hardware: five callers in the tree get it
# right and one got it wrong, and the wrong one is recognisable from the source alone -- a file that
# calls ggml_aligned_malloc and never calls ggml_aligned_free. That is a coarse rule (it cannot see
# *which* pointer goes where), but it is exact for the shape that occurred, costs milliseconds, and
# fails the configure on any platform rather than only on the machines where the bug is fatal.
#
# A failure here means one of three things: the local patch that fixes it was dropped, upstream
# reintroduced the shape somewhere else, or upstream fixed it in a way that no longer calls
# ggml_aligned_free in that file. Read the file before changing this check.

function(jllama_check_aligned_free_pairing root)
    file(GLOB_RECURSE sources
         "${root}/ggml/src/*.c"
         "${root}/ggml/src/*.cpp"
         "${root}/ggml/src/*.cu"
         "${root}/ggml/src/*.m"
         "${root}/ggml/src/*.mm")

    set(offenders "")
    set(checked 0)
    foreach(source IN LISTS sources)
        file(READ "${source}" text)
        # ggml.c defines both, so it names them for reasons other than calling them.
        string(FIND "${text}" "ggml_aligned_malloc" has_malloc)
        if(has_malloc EQUAL -1)
            continue()
        endif()
        math(EXPR checked "${checked} + 1")
        string(FIND "${text}" "ggml_aligned_free" has_free)
        if(has_free EQUAL -1)
            file(RELATIVE_PATH shown "${root}" "${source}")
            list(APPEND offenders "${shown}")
        endif()
    endforeach()

    if(offenders)
        string(REPLACE ";" "\n  - " shown "${offenders}")
        message(FATAL_ERROR
            "ggml_aligned_malloc without ggml_aligned_free in:\n  - ${shown}\n"
            "On Windows ggml_aligned_malloc is _aligned_malloc, whose memory MUST be released with "
            "_aligned_free (ggml_aligned_free); plain free() corrupts the process heap and kills it "
            "at cleanup with STATUS_HEAP_CORRUPTION, after the work already succeeded. This is the "
            "defect llama/patches/0018 fixes -- check whether that patch still applies, or whether "
            "upstream reintroduced the shape elsewhere. See llama/cmake/check-aligned-free-pairing.cmake.")
    endif()

    message(STATUS "aligned-free pairing: ${checked} ggml source(s) use ggml_aligned_malloc, all paired")
endfunction()
