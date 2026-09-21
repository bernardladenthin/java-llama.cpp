#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Asserts that every llama/patches/*.patch really reached the fetched llama.cpp tree.
#
# WHY THIS EXISTS. The patch applier (llama/cmake/apply-llama-patches.cmake) is fail-loud on
# "does not apply", so a *stale* patch cannot ship silently. What it cannot detect is a patch
# that stops having an effect while still applying, and most patches do not need this check
# because they have a runnable guard that reds CI on every platform if they go missing:
#
#   0003, 0006, 0007, 0008  -> jllama.cpp / native_server.cpp call the symbols they add,
#                              so dropping one is a compile or link error.
#   0012                    -> src/test/cpp/test_model_split.cpp.
#   0001, 0002              -> model-gated Java jobs (Windows argv, LoadProgressCallbackTest).
#
# `0010` is the exception and the reason for this script. It casts one enum to int inside
# upstream's `get_res_model_info()`, which is `static` in server-context.cpp and therefore
# unreachable from jllama_test; reverting it leaves `ctest` completely green. Its only guard is
# NativeServerAttachIntegrationTest.models_reportNumericVocabType, which is model-gated — so the
# day a platform stops downloading models, the regression ships. This check runs in the
# always-on `C++ Tests` job, needs no model, and costs milliseconds.
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

# --- 3. the one patch with no runnable guard ------------------------------------------------------
VOCAB_CAST='(int) meta.model_vocab_type'
SERVER_CONTEXT="$SRC/tools/server/server-context.cpp"
[ -f "$SERVER_CONTEXT" ] || fail "not found: $SERVER_CONTEXT"
grep -qF "$VOCAB_CAST" "$SERVER_CONTEXT" \
    || fail "patches/0010 is not present in $SERVER_CONTEXT: expected '$VOCAB_CAST'.
         Without the cast, common_json binds the unscoped enum to its bool constructor and
         GET /models + GET /v1/models report vocab_type as true/false instead of a number.
         If upstream added the cast themselves, DROP patch 0010 and update this check."

echo "patches verified: $on_disk applied, tree dirty, patches/0010 cast present"
