#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Pushes the pinned llama.cpp commit to the project's mirror (bernardladenthin/llama.cpp, a fork
# of ggml-org/llama.cpp) as the tag `jllama/b<nnnn>`, so the commit a release was built from stays
# fetchable when upstream's tag or repository goes away: `-DJLLAMA_GIT_REPOSITORY=<mirror>` in
# llama/CMakeLists.txt then serves the same commit (TODO.md, delivery roadmap, item 6). The patches
# stay files in this repository; the mirror holds pristine upstream commits only, so the review
# surface remains the patch file.
#
# Idempotent: a tag that already names this commit is left alone; a tag that names ANOTHER commit
# fails loudly -- a number maps to exactly one commit (upstream's own rule), so that is a defect,
# never something to overwrite.
#
# Usage: LLAMA_MIRROR_TOKEN=<token> .github/mirror-llama-source.sh <llama.cpp checkout> [<mirror repo>]
#   <llama.cpp checkout>  a full git clone at the pinned commit (fetch-depth 0: the push needs the
#                         commit's history, which the mirror shares with upstream anyway)
#   <mirror repo>         owner/name, default bernardladenthin/llama.cpp
#   LLAMA_MIRROR_TOKEN    a token with contents:write on the mirror (an environment secret of the
#                         release job); the script fails without it rather than skipping
# Exit codes: 0 mirrored (or already there), 1 a check failed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: mirror-llama-source.sh <llama.cpp checkout> [<mirror repo>]}"
MIRROR="${2:-bernardladenthin/llama.cpp}"
CMAKELISTS="$ROOT/llama/CMakeLists.txt"

fail() {
    echo "::error::mirror-llama-source: $*" >&2
    exit 1
}

[ -n "${LLAMA_MIRROR_TOKEN:-}" ] || fail "LLAMA_MIRROR_TOKEN is empty -- the release job needs that secret to push to $MIRROR"
[[ "$MIRROR" =~ ^[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$ ]] \
    && [ "${MIRROR##*/}" != "." ] && [ "${MIRROR##*/}" != ".." ] \
    || fail "mirror must be a GitHub owner/repository name"
[ -d "$SRC/.git" ] || fail "$SRC is not a git checkout"

TAG=$(grep -oE '^set\(JLLAMA_GIT_TAG[[:space:]]+b[0-9]+\)' "$CMAKELISTS" | grep -oE 'b[0-9]+' | head -1)
[ -n "$TAG" ] || fail "could not resolve JLLAMA_GIT_TAG (b<nnnn>) from $CMAKELISTS"
COMMIT=$(grep -oE '^set\(JLLAMA_GIT_COMMIT[[:space:]]+"[0-9a-f]{40}"\)' "$CMAKELISTS" | grep -oE '[0-9a-f]{40}' | head -1 || true)
HEAD_COMMIT=$(git -C "$SRC" rev-parse HEAD)
if [ -n "$COMMIT" ] && [ "$HEAD_COMMIT" != "$COMMIT" ]; then
    fail "$SRC is at $HEAD_COMMIT, but JLLAMA_GIT_COMMIT pins $COMMIT"
fi
[ -f "$SRC/.git/shallow" ] && fail "$SRC is a shallow clone; the push needs the full history (fetch-depth 0)"
COUNT=$(git -C "$SRC" rev-list --count HEAD)
[ "$COUNT" = "${TAG#b}" ] || fail "the checked-out commit $HEAD_COMMIT is b$COUNT, not $TAG"

MIRROR_URL="https://github.com/$MIRROR.git"
REF="refs/tags/jllama/$TAG"
# Match only this mirror URL, replacing any inherited checkout header for it. Pass the header
# through the environment, not argv or a persistent Git config. Requires Git >= 2.31.
JLLAMA_MIRROR_AUTH_HEADER="AUTHORIZATION: basic $(printf '%s' "x-access-token:$LLAMA_MIRROR_TOKEN" | base64 -w0)"
export JLLAMA_MIRROR_AUTH_HEADER
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "::add-mask::${JLLAMA_MIRROR_AUTH_HEADER#AUTHORIZATION: basic }"
fi
mirror_git() {
    GIT_TERMINAL_PROMPT=0 git -C "$SRC" \
        -c "credential.$MIRROR_URL.helper=" \
        -c "http.$MIRROR_URL.extraheader=" \
        --config-env="http.$MIRROR_URL.extraheader=JLLAMA_MIRROR_AUTH_HEADER" "$@"
}

EXISTING=$(mirror_git ls-remote --tags "$MIRROR_URL" "$REF" | cut -f1)
if [ -n "$EXISTING" ]; then
    if [ "$EXISTING" = "$HEAD_COMMIT" ]; then
        echo "mirror-llama-source: $MIRROR already has $REF = $HEAD_COMMIT; nothing to push"
        exit 0
    fi
    fail "$MIRROR has $REF = $EXISTING, but $TAG is $HEAD_COMMIT here -- one number, one commit; resolve by hand"
fi
mirror_git push "$MIRROR_URL" "$HEAD_COMMIT:$REF"
echo "mirror-llama-source: pushed $HEAD_COMMIT to $MIRROR as $REF"
