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
#   * Idempotent, via a stamp file rather than per-patch probing. The stamp
#     (`${LLAMA_SRC}/.jllama-patches-applied`) records the checked-out llama.cpp commit plus the
#     SHA-256 of every patch, and the decision is driven by whether the source tree is pristine:
#       - clean tree  -> nothing is applied yet (a fresh fetch, or a re-checkout after a version
#                        bump), so apply all patches forward and write the stamp;
#       - dirty tree  -> already patched; skip when the stamp matches this exact commit + patch
#                        set, and fail loudly when it does not.
#     A per-patch `git apply --reverse --check` cannot do this: `--check` never mutates the tree,
#     so an earlier patch whose region a later patch rewrote (today 0001 vs 0006/0007 in
#     tools/server/server.cpp) always reverse-checks as "not applied" and the forward re-apply
#     then aborts an otherwise harmless reconfigure. The stamp is state, but it is state derived
#     from — and invalidated by — both inputs that matter.
#   * Fail-loud: a patch that no longer applies (e.g. after a llama.cpp version bump shifts the
#     context) aborts the configure with a clear message, so a stale patch can never be silently
#     dropped from a release build.
#   * A source tree that is not a git work tree (e.g. supplied via
#     `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<path>`) has no clean/dirty oracle and no HEAD, so it
#     falls back to the legacy per-patch reverse-check path, with the caveat described above.
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

set(STAMP_NAME ".jllama-patches-applied")
set(stamp_file "${LLAMA_SRC}/${STAMP_NAME}")

# Applies one patch, aborting the configure when it no longer fits the source tree.
function(apply_one_patch patch)
    get_filename_component(patch_name "${patch}" NAME)
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
endfunction()

# ---------------------------------------------------------------------------
# Is the source tree a git work tree? Without one there is no HEAD to pin the
# stamp to and no clean/dirty oracle, so fall back to the legacy behaviour.
# ---------------------------------------------------------------------------
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" rev-parse HEAD
    RESULT_VARIABLE head_rc
    OUTPUT_VARIABLE llama_head
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

if(NOT head_rc EQUAL 0)
    message(STATUS "apply-llama-patches: ${LLAMA_SRC} is not a git work tree — "
                   "using per-patch detection (a reconfigure over a patched tree may fail)")
    foreach(patch IN LISTS patch_files)
        get_filename_component(patch_name "${patch}" NAME)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" apply --reverse --check "${patch}"
            RESULT_VARIABLE reverse_rc
            OUTPUT_QUIET ERROR_QUIET)
        if(reverse_rc EQUAL 0)
            message(STATUS "apply-llama-patches: ${patch_name} already applied — skipping")
            continue()
        endif()
        apply_one_patch("${patch}")
    endforeach()
    return()
endif()

# ---------------------------------------------------------------------------
# Build the manifest: the checked-out commit plus every patch's content hash.
# Any llama.cpp version bump changes HEAD; any patch edit changes a hash.
# ---------------------------------------------------------------------------
set(manifest "head ${llama_head}\n")
foreach(patch IN LISTS patch_files)
    get_filename_component(patch_name "${patch}" NAME)
    file(SHA256 "${patch}" patch_hash)
    string(APPEND manifest "${patch_name} ${patch_hash}\n")
endforeach()

# ---------------------------------------------------------------------------
# Clean tree => nothing applied yet. Untracked files count as dirty (a future
# patch may add a file), except the stamp itself, which we write ourselves.
# ---------------------------------------------------------------------------
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" status --porcelain
    RESULT_VARIABLE status_rc
    OUTPUT_VARIABLE status_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT status_rc EQUAL 0)
    message(FATAL_ERROR "apply-llama-patches: 'git status' failed in ${LLAMA_SRC}")
endif()

set(tree_is_dirty FALSE)
if(NOT status_out STREQUAL "")
    string(REPLACE "\n" ";" status_lines "${status_out}")
    foreach(line IN LISTS status_lines)
        string(STRIP "${line}" line)
        if(line STREQUAL "" OR line MATCHES "${STAMP_NAME}$")
            continue()
        endif()
        set(tree_is_dirty TRUE)
        break()
    endforeach()
endif()

if(NOT tree_is_dirty)
    foreach(patch IN LISTS patch_files)
        apply_one_patch("${patch}")
    endforeach()
    file(WRITE "${stamp_file}" "${manifest}")
    return()
endif()

# ---------------------------------------------------------------------------
# Dirty tree: already patched. Only a stamp matching this exact commit + patch
# set proves the modifications are ours and complete.
# ---------------------------------------------------------------------------
set(stamp_matches FALSE)
if(EXISTS "${stamp_file}")
    file(READ "${stamp_file}" stamp_content)
    if(stamp_content STREQUAL manifest)
        set(stamp_matches TRUE)
    endif()
endif()

if(stamp_matches)
    list(LENGTH patch_files patch_count)
    message(STATUS "apply-llama-patches: ${patch_count} patch(es) already applied — skipping")
    return()
endif()

message(FATAL_ERROR
    "apply-llama-patches: ${LLAMA_SRC} has local modifications that do not match the current "
    "patch set.\n"
    "  Patches cannot be applied on top of an already-patched tree, and the previous state is "
    "unknown (the tree was patched with a different patch set or llama.cpp commit, or edited by "
    "hand).\n"
    "  Configure into a FRESH build directory so FetchContent re-checks-out a pristine "
    "llama.cpp, then build again.")
