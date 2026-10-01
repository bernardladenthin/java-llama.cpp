#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Pick the NEXT llama.cpp tag to bump the pin to, one reviewable chunk at a time.
#
# The runbook this supports is docs/upgrade/llama-cpp-version-bump.md. Strategy:
#   * TARGET   = the topmost RELEASE on the GitHub releases page (read from the release atom feed),
#                or an explicit "b<nnnn>" passed as $1.
#   * CURRENT  = the pinned tag in llama/CMakeLists.txt (GIT_TAG b<nnnn>).
#   * If `git diff CURRENT..TARGET` is smaller than the threshold (default 100 KiB), bump straight
#     to TARGET. Otherwise CHUNK: pick the largest intermediate b<nnnn> tag whose diff from CURRENT
#     is still under the threshold, so each bump stays a small, reviewable patch. Re-run after each
#     bump to walk the remaining chunks up to TARGET.
#
# This tool only READS (a cached mirror clone + the pin file); it never edits the repo. Apply the
# bump by hand per the runbook. It prints the compare/.patch URLs for the chosen step.
#
# Env:
#   LLAMA_BUMP_MAX_DIFF_KB   per-step diff-size threshold in KiB (default 100)
#   LLAMA_BUMP_EXCLUDE_WEBUI if "1", size the diff EXCLUDING tools/ui (the auto-followed WebUI, which
#                            does not need per-bump review); default 0 = the full diff you paste/review
#   LLAMA_BUMP_CACHE         mirror-clone location (default ~/.cache/jllama-llamacpp-mirror)
#
# Network: needs read access to github.com (git clone/fetch + the release atom feed). No token.

set -euo pipefail

THRESHOLD_KB="${LLAMA_BUMP_MAX_DIFF_KB:-100}"
THRESHOLD=$((THRESHOLD_KB * 1024))
EXCLUDE_WEBUI="${LLAMA_BUMP_EXCLUDE_WEBUI:-0}"
REPO="ggml-org/llama.cpp"
GIT_URL="https://github.com/${REPO}.git"
CACHE="${LLAMA_BUMP_CACHE:-$HOME/.cache/jllama-llamacpp-mirror}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMAKELISTS="$ROOT/llama/CMakeLists.txt"

# --- current pinned tag number, e.g. "GIT_TAG b9866" -> 9866 -----------------------------------
cur="$(grep -oE 'GIT_TAG[[:space:]]+b[0-9]+' "$CMAKELISTS" | grep -oE '[0-9]+' | head -1 || true)"
[ -n "$cur" ] || { echo "ERROR: could not read 'GIT_TAG b<nnnn>' from $CMAKELISTS" >&2; exit 1; }

# --- cached blobless mirror of llama.cpp (clone once, then fetch tags) --------------------------
if [ -d "$CACHE/.git" ]; then
    git -C "$CACHE" fetch --quiet --tags --prune origin || true
else
    echo "cloning ${REPO} (blobless) into $CACHE (one-time) ..." >&2
    git clone --filter=blob:none --no-checkout --quiet "$GIT_URL" "$CACHE"
fi

# --- target: explicit "$1" (b<nnnn>) or the latest RELEASE from the atom feed -------------------
if [ "${1:-}" != "" ]; then
    target="$(printf '%s' "$1" | grep -oE '[0-9]+' | head -1)"
    [ -n "$target" ] || { echo "ERROR: '$1' is not a b<nnnn> tag" >&2; exit 1; }
else
    feed="$(curl -sSL --fail --retry 4 --retry-delay 2 "https://github.com/${REPO}/releases.atom" 2>/dev/null || true)"
    [ -n "$feed" ] || { echo "ERROR: cannot fetch the releases feed (network/rate limit). Read the topmost release at https://github.com/${REPO}/releases and pass it: $0 b<nnnn>" >&2; exit 2; }
    target="$(printf '%s' "$feed" | grep -oE 'releases/tag/b[0-9]+' | grep -oE '[0-9]+' | sort -un | tail -1)"
    [ -n "$target" ] || { echo "ERROR: parsed no release tags from the feed." >&2; exit 3; }
fi

git -C "$CACHE" rev-parse -q --verify "b${cur}^{commit}"    >/dev/null 2>&1 || { echo "ERROR: b$cur is not a tag in the mirror" >&2; exit 3; }
git -C "$CACHE" rev-parse -q --verify "b${target}^{commit}" >/dev/null 2>&1 || { echo "ERROR: b$target is not a tag in the mirror" >&2; exit 3; }

# diff byte size between two tag numbers, honoring the WebUI-exclusion toggle
diffsize() {
    if [ "$EXCLUDE_WEBUI" = "1" ]; then
        git -C "$CACHE" diff "b$1" "b$2" -- . ':(exclude)tools/ui' 2>/dev/null | wc -c
    else
        git -C "$CACHE" diff "b$1" "b$2" 2>/dev/null | wc -c
    fi
}

scope="full diff"
[ "$EXCLUDE_WEBUI" = "1" ] && scope="diff excluding tools/ui"
echo "current pin    : b$cur"
echo "latest release : b$target"
echo "threshold      : ${THRESHOLD_KB} KiB per step (${scope})"

if [ "$cur" -ge "$target" ]; then
    echo "=> up to date — no bump needed."
    exit 0
fi

# --- choose next step: TARGET if it fits, else the largest intermediate tag under the threshold -
if [ "$(diffsize "$cur" "$target")" -lt "$THRESHOLD" ]; then
    next="$target"
else
    # existing b-tags strictly after cur, up to and including target, ascending
    # shellcheck disable=SC2207
    cands=($(git -C "$CACHE" tag -l 'b*' | grep -oE 'b[0-9]+' | grep -oE '[0-9]+' | sort -un \
             | awk -v c="$cur" -v t="$target" '$1 > c && $1 <= t'))
    # binary search for the largest candidate whose diff from cur is under the threshold
    # (diff size grows monotonically enough with the tag number for this to be a safe heuristic)
    lo=0; hi=$(( ${#cands[@]} - 1 )); best=""
    while [ "$lo" -le "$hi" ]; do
        mid=$(( (lo + hi) / 2 )); T="${cands[$mid]}"
        if [ "$(diffsize "$cur" "$T")" -lt "$THRESHOLD" ]; then best="$T"; lo=$(( mid + 1 )); else hi=$(( mid - 1 )); fi
    done
    if [ -n "$best" ]; then
        next="$best"
    else
        next="${cands[0]}"
        echo "NOTE: even b$cur..b$next exceeds ${THRESHOLD_KB} KiB — a single-commit step this large is unavoidable." >&2
    fi
fi

full=$(git -C "$CACHE" diff "b$cur" "b$next" | wc -c)
noui=$(git -C "$CACHE" diff "b$cur" "b$next" -- . ':(exclude)tools/ui' | wc -c)
commits=$(git -C "$CACHE" rev-list --count "b$cur".."b$next")
echo
echo "next step      : b$cur -> b$next"
echo "  diff size    : $((full / 1024)) KiB full  /  $((noui / 1024)) KiB excluding tools/ui (auto-followed WebUI)"
echo "  commits      : $commits"
if [ "$next" -eq "$target" ]; then
    echo "  progress     : reaches the latest release — final chunk"
else
    echo "  progress     : intermediate chunk — re-run this script after the bump for the next one"
fi
echo "  review diff  : https://github.com/${REPO}/compare/b$cur...b$next"
echo "  raw .patch   : https://github.com/${REPO}/compare/b$cur...b$next.patch"
echo
echo "Apply this bump per docs/upgrade/llama-cpp-version-bump.md (b$cur -> b$next)."
