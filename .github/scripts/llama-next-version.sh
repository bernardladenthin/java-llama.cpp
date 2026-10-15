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
#   * CURRENT  = the pinned number in llama/CMakeLists.txt (set(JLLAMA_GIT_TAG b<nnnn>)). It may be a
#                "virtual" number -- one upstream has not tagged yet, pinned through JLLAMA_GIT_COMMIT --
#                in which case it is resolved to the master commit whose `git rev-list --count` is the
#                number, the way upstream's release.yml numbers its tags.
#   * `--resolve b<nnnn>` prints that commit for any number (the tag's commit when the tag exists) and
#                exits: the hash to put into JLLAMA_GIT_COMMIT.
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

# --- current pinned number, e.g. "set(JLLAMA_GIT_TAG b9866)" -> 9866, and the optional commit ------
cur="$(grep -oE '^set\(JLLAMA_GIT_TAG[[:space:]]+b[0-9]+\)' "$CMAKELISTS" | grep -oE '[0-9]+' | head -1 || true)"
[ -n "$cur" ] || { echo "ERROR: could not read 'set(JLLAMA_GIT_TAG b<nnnn>)' from $CMAKELISTS" >&2; exit 1; }
pinned_commit="$(grep -oE '^set\(JLLAMA_GIT_COMMIT[[:space:]]+"[0-9a-f]{40}"\)' "$CMAKELISTS" | grep -oE '[0-9a-f]{40}' | head -1 || true)"

# --- cached blobless mirror of llama.cpp (clone once, then fetch tags) --------------------------
if [ -d "$CACHE/.git" ]; then
    git -C "$CACHE" fetch --quiet --tags --prune origin || true
else
    echo "cloning ${REPO} (blobless) into $CACHE (one-time) ..." >&2
    git clone --filter=blob:none --no-checkout --quiet "$GIT_URL" "$CACHE"
fi

# --- a number -> the ref that carries it: the tag when upstream published it, else the master -----
# commit whose `git rev-list --count` is the number (upstream's own numbering rule). A virtual number
# is recent by nature, so the walk down first-parent master is short; it stops at the first commit
# whose count is <= the number and errors when that count is not the number (a number no master
# commit carries, or one further back than the walk looks).
ref_of_number() {
    local n="$1" c count i=0
    if git -C "$CACHE" rev-parse -q --verify "b${n}^{commit}" >/dev/null 2>&1; then
        echo "b$n"; return 0
    fi
    git -C "$CACHE" fetch --quiet origin master 2>/dev/null || true
    while read -r c; do
        count="$(git -C "$CACHE" rev-list --count "$c")"
        if [ "$count" -eq "$n" ]; then echo "$c"; return 0; fi
        if [ "$count" -lt "$n" ]; then break; fi
        i=$((i + 1)); [ "$i" -lt 2000 ] || break
    done < <(git -C "$CACHE" rev-list --first-parent origin/master)
    echo "ERROR: b$n is not a tag in the mirror and no recent master commit has commit count $n" >&2
    return 1
}

if [ "${1:-}" = "--resolve" ]; then
    n="$(printf '%s' "${2:-}" | grep -oE '[0-9]+' | head -1)"
    [ -n "$n" ] || { echo "usage: $0 --resolve b<nnnn>" >&2; exit 1; }
    ref="$(ref_of_number "$n")" || exit 3
    sha="$(git -C "$CACHE" rev-parse "${ref}^{commit}")"
    if [ "$ref" = "b$n" ]; then
        echo "b$n is published: $sha -- set(JLLAMA_GIT_COMMIT \"$sha\") (the commit stays the pin; CI refuses a fetch by tag name)"
    else
        echo "b$n is not published yet; it will be $sha -- set(JLLAMA_GIT_COMMIT \"$sha\")"
    fi
    exit 0
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

curref="$(ref_of_number "$cur")" || exit 3
git -C "$CACHE" rev-parse -q --verify "b${target}^{commit}" >/dev/null 2>&1 || { echo "ERROR: b$target is not a tag in the mirror" >&2; exit 3; }
if [ -n "$pinned_commit" ]; then
    # JLLAMA_GIT_COMMIT is set: it must be the commit the number names (and it stays set once the tag exists)
    pinned_count="$(git -C "$CACHE" rev-list --count "$pinned_commit" 2>/dev/null || true)"
    if [ "$pinned_count" != "$cur" ]; then
        echo "ERROR: JLLAMA_GIT_COMMIT $pinned_commit has commit count ${pinned_count:-?}, not $cur -- it does not resolve b$cur (run: $0 --resolve b$cur)" >&2; exit 3
    fi
elif [ "$curref" != "b$cur" ]; then
    echo "NOTE: b$cur is not a published tag and JLLAMA_GIT_COMMIT is empty -- the native fetch will fail; set(JLLAMA_GIT_COMMIT \"$curref\")" >&2
else
    echo "NOTE: JLLAMA_GIT_COMMIT is empty -- CI refuses a fetch by tag name (JLLAMA_REQUIRE_COMMIT); set(JLLAMA_GIT_COMMIT \"$(git -C "$CACHE" rev-parse "b$cur^{commit}")\")" >&2
fi

# diff byte size between the current ref and a tag number, honoring the WebUI-exclusion toggle
diffsize() {
    if [ "$EXCLUDE_WEBUI" = "1" ]; then
        git -C "$CACHE" diff "$1" "b$2" -- . ':(exclude)tools/ui' 2>/dev/null | wc -c
    else
        git -C "$CACHE" diff "$1" "b$2" 2>/dev/null | wc -c
    fi
}

scope="full diff"
[ "$EXCLUDE_WEBUI" = "1" ] && scope="diff excluding tools/ui"
echo "current pin    : b$cur$( [ "$curref" = "b$cur" ] || echo " (virtual: not tagged upstream yet, resolved to $curref)")"
echo "latest release : b$target"
echo "threshold      : ${THRESHOLD_KB} KiB per step (${scope})"

if [ "$cur" -ge "$target" ]; then
    echo "=> up to date — no bump needed."
    exit 0
fi

# --- choose next step: TARGET if it fits, else the largest intermediate tag under the threshold -
if [ "$(diffsize "$curref" "$target")" -lt "$THRESHOLD" ]; then
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
        if [ "$(diffsize "$curref" "$T")" -lt "$THRESHOLD" ]; then best="$T"; lo=$(( mid + 1 )); else hi=$(( mid - 1 )); fi
    done
    if [ -n "$best" ]; then
        next="$best"
    else
        next="${cands[0]}"
        echo "NOTE: even b$cur..b$next exceeds ${THRESHOLD_KB} KiB — a single-commit step this large is unavoidable." >&2
    fi
fi

full=$(git -C "$CACHE" diff "$curref" "b$next" | wc -c)
noui=$(git -C "$CACHE" diff "$curref" "b$next" -- . ':(exclude)tools/ui' | wc -c)
commits=$(git -C "$CACHE" rev-list --count "$curref".."b$next")
echo
echo "next step      : b$cur -> b$next"
echo "  diff size    : $((full / 1024)) KiB full  /  $((noui / 1024)) KiB excluding tools/ui (auto-followed WebUI)"
echo "  commits      : $commits"
if [ "$next" -eq "$target" ]; then
    echo "  progress     : reaches the latest release — final chunk"
else
    echo "  progress     : intermediate chunk — re-run this script after the bump for the next one"
fi
echo "  review diff  : https://github.com/${REPO}/compare/$curref...b$next"
echo "  raw .patch   : https://github.com/${REPO}/compare/$curref...b$next.patch"
echo
echo "Apply this bump per docs/upgrade/llama-cpp-version-bump.md (b$cur -> b$next)."
