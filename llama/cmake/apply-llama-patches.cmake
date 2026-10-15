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
#     `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<path>`) has no clean/dirty oracle and no HEAD. When it
#     carries the stamp -- the archived source tree of a release
#     (llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz, written by publish.yml's archive-llama-source
#     job after patching, .git removed) -- the stamp is the oracle: it must name exactly the patches
#     on disk, with their hashes, and then nothing is applied; another patch set fails loudly.
#     Without a stamp it falls back to the legacy per-patch reverse-check path, with the caveat
#     described above.
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
    # An archived source tree -- the release asset llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz,
    # which publish.yml's archive-llama-source job writes from a patched checkout with its .git
    # removed -- carries this stamp, and the stamp is its only oracle: no HEAD, no clean/dirty
    # state, and the per-patch reverse check below cannot answer for it either (0001's region is
    # rewritten by 0006/0007, so 0001 always reverse-checks as "not applied" and the forward apply
    # then aborts). So: every patch line of the stamp names a patch on disk with the same hash and
    # the counts agree => the tree is exactly the patched tree the stamp describes, nothing to do.
    # A stamp from another patch set (a patch edited or added since the archive was written) is a
    # mismatch, reported as such rather than silently papered over with the legacy path.
    if(EXISTS "${stamp_file}")
        file(STRINGS "${stamp_file}" stamp_patch_lines REGEX "^[^ ]+\\.(patch|diff) [0-9a-f]+$")
        set(stamp_matches TRUE)
        list(LENGTH stamp_patch_lines stamp_n)
        list(LENGTH patch_files disk_n)
        if(NOT stamp_n EQUAL disk_n)
            set(stamp_matches FALSE)
        endif()
        foreach(patch IN LISTS patch_files)
            get_filename_component(patch_name "${patch}" NAME)
            file(SHA256 "${patch}" patch_hash)
            list(FIND stamp_patch_lines "${patch_name} ${patch_hash}" idx)
            if(idx EQUAL -1)
                set(stamp_matches FALSE)
            endif()
        endforeach()
        if(stamp_matches)
            message(STATUS "apply-llama-patches: ${LLAMA_SRC} is an archived (non-git) tree whose stamp "
                           "names exactly the ${disk_n} patch(es) on disk — already applied, nothing to do")
            return()
        endif()
        message(FATAL_ERROR
            "apply-llama-patches: ${LLAMA_SRC} is not a git work tree, and its stamp ${stamp_file} "
            "describes another patch set than ${PATCH_DIR} (${stamp_n} patch line(s) in the stamp, "
            "${disk_n} on disk, or a hash differs). An archived source tree is patched once, when "
            "the archive is written; build this patch set from a fresh fetch instead.")
    endif()
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
# Content oracle for the WORKING TREE.
#
# The commit + patch hashes below describe what SHOULD be applied; they say nothing about what
# the files actually contain. Reverting one patched file after a successful apply therefore left
# the stamp valid and the tree still dirty (the other patched files are still modified), so the
# dirty-tree branch reported "already applied — skipping", exited 0, and the build compiled
# unpatched code. This fingerprint closes that: it hashes the tree's actual modifications, so a
# reverted or hand-edited file changes it and the stamp stops matching.
#
# Two sources, because neither alone is enough: `git diff` carries the CONTENT of modifications
# to tracked files, and `git status --porcelain` carries the PRESENCE of untracked files (patch
# 0012 adds tests/test-model-split.cpp, which `git diff` never sees). The stamp itself is
# untracked and is filtered out, or writing it would change the fingerprint that describes it.
#
# Residual hole, stated rather than papered over: an edit to the *body* of an untracked file a
# patch added is not detected. Closing that would mean hashing every untracked file's contents;
# the CI-side .github/verify-patches-applied.sh covers the case that actually matters.
# ---------------------------------------------------------------------------
function(compute_tree_fingerprint out_var)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" status --porcelain
        OUTPUT_VARIABLE fp_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    set(fp_status_filtered "")
    if(NOT fp_status STREQUAL "")
        string(REPLACE "\n" ";" fp_lines "${fp_status}")
        foreach(fp_line IN LISTS fp_lines)
            string(STRIP "${fp_line}" fp_line)
            if(fp_line STREQUAL "" OR fp_line MATCHES "${STAMP_NAME}$")
                continue()
            endif()
            string(APPEND fp_status_filtered "${fp_line}\n")
        endforeach()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_SRC}" diff --no-color
        OUTPUT_VARIABLE fp_diff
        ERROR_QUIET)
    string(SHA256 fp_hash "${fp_status_filtered}${fp_diff}")
    set(${out_var} "${fp_hash}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Build the manifest: the checked-out commit plus every patch's content hash.
# Any llama.cpp version bump changes HEAD; any patch edit changes a hash.
# The tree fingerprint is appended after applying (it cannot be known before).
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
    # Fingerprint the result, not the intent: this is what a later reconfigure compares against.
    compute_tree_fingerprint(applied_fingerprint)
    file(WRITE "${stamp_file}" "${manifest}tree ${applied_fingerprint}\n")
    return()
endif()

# ---------------------------------------------------------------------------
# Dirty tree: already patched. Only a stamp matching this exact commit + patch
# set proves the modifications are ours and complete.
# ---------------------------------------------------------------------------
compute_tree_fingerprint(current_fingerprint)
set(expected_stamp "${manifest}tree ${current_fingerprint}\n")

set(stamp_matches FALSE)
if(EXISTS "${stamp_file}")
    file(READ "${stamp_file}" stamp_content)
    if(stamp_content STREQUAL expected_stamp)
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
    "unknown: the tree was patched with a different patch set or llama.cpp commit, or one of the "
    "patched files was reverted or edited by hand (the stamp records a fingerprint of the tree's "
    "modifications, so a single reverted file lands here rather than silently building unpatched "
    "code).\n"
    "  Configure into a FRESH build directory so FetchContent re-checks-out a pristine "
    "llama.cpp, then build again.")
