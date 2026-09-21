#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Asserts that every llama/patches/*.patch really reached the fetched llama.cpp tree.
#
# WHY THIS EXISTS. The patch applier (llama/cmake/apply-llama-patches.cmake) is fail-loud on
# "does not apply", so a *stale* patch cannot ship silently. What it cannot detect is the
# applier never having run at all, or a patched tree being reverted after the fact — the stamp
# bookkeeping and the tree's dirty state are the only evidence of that, and this script asserts
# both. It runs in the always-on `C++ Tests` job, needs no model, and costs milliseconds.
#
# Every patch in the set does also have a runnable guard that reds CI on every platform if it
# goes missing, so these checks are a second line rather than the only one:
#
#   0003, 0006, 0007, 0008  -> jllama.cpp / native_server.cpp call the symbols they add,
#                              so dropping one is a compile or link error.
#   0012                    -> src/test/cpp/test_model_split.cpp.
#   0014                    -> src/test/cpp/test_common_log_callback.cpp (link error).
#   0001, 0002              -> model-gated Java jobs (Windows argv, LoadProgressCallbackTest).
#
# It used to carry a third, patch-specific check for `0010`, the one patch with no runnable
# guard (it cast an enum inside upstream's `static get_res_model_info()`, unreachable from
# jllama_test). That patch was DROPPED at the b11080 bump — upstream #28518 gave
# `common_json_value` an enum constructor, fixing the defect at its root — so the check retired
# with it. If a future patch is ever added that likewise cannot be reached from `ctest`, add a
# check for it here rather than relying on a model-gated Java test.
#
# Usage: .github/verify-patches-applied.sh [<llama.cpp-src-dir>]
# Exit codes: 0 all good, 1 a check failed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/llama/build/_deps/llama.cpp-src}"
PATCH_DIR="$ROOT/llama/patches"
STAMP="$SRC/.jllama-patches-applied"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[ -d "$SRC" ] || fail "llama.cpp source dir not found: $SRC (configure the build first)"
[ -f "$STAMP" ] || fail "patch stamp not found: $STAMP — the applier never ran, so the tree is unpatched"

# --- 1. every patch on disk is named in the stamp -----------------------------------------------
# Self-maintaining on purpose: adding a patch file needs no edit here. The stamp carries metadata
# lines ("head <commit>", "tree <fingerprint>") plus one "<patch filename> <sha256>" line per patch.
# Count the patch lines by their own shape rather than by subtracting a fixed number of metadata
# lines: that subtraction was "- 1" and silently went stale the moment the applier gained its
# "tree" line, failing every correct build with "the build dir is stale".
on_disk=0
for p in "$PATCH_DIR"/*.patch; do
    [ -e "$p" ] || fail "no *.patch files in $PATCH_DIR"
    on_disk=$((on_disk + 1))
    name="$(basename "$p")"
    grep -qF "$name" "$STAMP" || fail "patch '$name' is on disk but absent from the stamp $STAMP"
done

in_stamp="$(grep -cE '^[^[:space:]]+\.(patch|diff)[[:space:]]' "$STAMP" || true)"
[ "$in_stamp" -eq "$on_disk" ] \
    || fail "stamp lists $in_stamp patch(es) but $on_disk are on disk — the build dir is stale; configure into a fresh one"

# --- 2. the tree is actually modified ------------------------------------------------------------
# A valid stamp over a clean tree means the patches were reverted after the fact.
if git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
    if git -C "$SRC" diff --quiet; then
        fail "stamp says $on_disk patch(es) applied but '$SRC' is clean — the patched files were reverted"
    fi
fi

echo "patches verified: $on_disk applied, tree dirty"
