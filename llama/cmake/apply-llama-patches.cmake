# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# apply-llama-patches.cmake — applies every patch in the repo-root `patches/` directory to the
# llama.cpp source tree fetched by FetchContent. Wired as the llama.cpp `PATCH_COMMAND` in the
# top-level CMakeLists.txt, so it runs for EVERY C++ build (all CI jobs + local) from one place,
# rather than per-build-step.
#
# Design:
#   * Cross-platform: invoked via `cmake -P`, so it behaves identically on Linux, macOS and
#     Windows (the dockcross/native/MSVC jobs all call the same code path).
#   * Every `patches/*.patch` and `patches/*.diff` is applied, sorted by filename (so a numeric
#     prefix like 0001-, 0002- defines a deterministic order).
#   * Idempotent ONLY while no two patches touch the same file: `git apply --reverse --check`
#     detects an already-applied patch and skips it, so a CMake reconfigure over an already-patched
#     source tree normally does not fail. It DOES fail when a later patch rewrites a region an
#     earlier one also patched (today: 0006/0007 vs 0001 in tools/server/server.cpp) — the earlier
#     patch's reverse-check then no longer matches and the forward apply aborts the configure with
#     a misleading "does not apply cleanly". Configure into a fresh build dir; see TODO.md.
#   * Fail-loud: a patch that no longer applies (e.g. after a llama.cpp version bump shifts the
#     context) aborts the configure with a clear message, so a stale patch can never be silently
#     dropped from a release build.
#
# Invoked as:
#   cmake -DPATCH_DIR=<repo>/patches -DLLAMA_SRC=<fetched-src> -P cmake/apply-llama-patches.cmake

if(NOT DEFINED PATCH_DIR OR NOT DEFINED LLAMA_SRC)
    message(FATAL_ERROR "apply-llama-patches: both PATCH_DIR and LLAMA_SRC must be defined")
endif()

find_program(GIT_EXECUTABLE NAMES git)
if(NOT GIT_EXECUTABLE)
    message(FATAL_ERROR "apply-llama-patches: 'git' not found on PATH (required to apply patches)")
endif()

file(GLOB patch_files "${PATCH_DIR}/*.patch" "${PATCH_DIR}/*.diff")
list(SORT patch_files)

if(NOT patch_files)
    message(STATUS "apply-llama-patches: no patches in ${PATCH_DIR} (nothing to apply)")
    return()
endif()

foreach(patch IN LISTS patch_files)
    get_filename_component(patch_name "${patch}" NAME)

    # Already applied? A successful reverse-apply check means the change is present already.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --reverse --check "${patch}"
        RESULT_VARIABLE reverse_rc
        OUTPUT_QUIET ERROR_QUIET)
    if(reverse_rc EQUAL 0)
        message(STATUS "apply-llama-patches: ${patch_name} already applied — skipping")
        continue()
    endif()

    # Not applied yet — confirm it applies cleanly before touching the tree.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --check "${patch}"
        RESULT_VARIABLE check_rc
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT check_rc EQUAL 0)
        message(FATAL_ERROR
            "apply-llama-patches: ${patch_name} does not apply cleanly to ${LLAMA_SRC}.\n"
            "  A llama.cpp version bump probably shifted the patched code — refresh the patch "
            "against the new source and recommit it.")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply "${patch}"
        RESULT_VARIABLE apply_rc)
    if(NOT apply_rc EQUAL 0)
        message(FATAL_ERROR "apply-llama-patches: failed to apply ${patch_name}")
    endif()
    message(STATUS "apply-llama-patches: applied ${patch_name}")
endforeach()
