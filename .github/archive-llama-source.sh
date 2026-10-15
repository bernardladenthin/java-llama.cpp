#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Writes the release asset `llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz`: the pinned llama.cpp
# commit with every llama/patches/*.patch applied, as a plain source tree (no .git), reproducibly
# packed. It is the exact source every natives jar of the release was compiled from, kept with the
# release so a binary stays rebuildable when upstream's tag or repository goes away (TODO.md,
# delivery roadmap, item 6): `cmake -B build -DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<unpacked dir>`
# builds from it, and the applier's stamp inside the tree (`.jllama-patches-applied`) tells the
# configure that the patches are already in (llama/cmake/apply-llama-patches.cmake).
#
# Usage: .github/archive-llama-source.sh <llama.cpp checkout> <output directory>
#   <llama.cpp checkout>  a full git clone of ggml-org/llama.cpp at the pinned commit (the archive
#                         job checks it out with fetch-depth 0; the commit count is checked here)
#   <output directory>    receives <name>.tar.gz, <name>.tar.gz.sha256 and a one-line
#                         `archive-name.txt` naming the file
# Exit codes: 0 archive written, 1 a check failed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: archive-llama-source.sh <llama.cpp checkout> <output directory>}"
OUT="${2:?usage: archive-llama-source.sh <llama.cpp checkout> <output directory>}"
CMAKELISTS="$ROOT/llama/CMakeLists.txt"

fail() {
    echo "::error::archive-llama-source: $*" >&2
    exit 1
}

[ -d "$SRC/.git" ] || fail "$SRC is not a git checkout"

# --- the pin: the number, and the commit it names ------------------------------------------------
TAG=$(grep -oE '^set\(JLLAMA_GIT_TAG[[:space:]]+b[0-9]+\)' "$CMAKELISTS" | grep -oE 'b[0-9]+' | head -1)
[ -n "$TAG" ] || fail "could not resolve JLLAMA_GIT_TAG (b<nnnn>) from $CMAKELISTS"
COMMIT=$(grep -oE '^set\(JLLAMA_GIT_COMMIT[[:space:]]+"[0-9a-f]{40}"\)' "$CMAKELISTS" | grep -oE '[0-9a-f]{40}' | head -1 || true)
HEAD_COMMIT=$(git -C "$SRC" rev-parse HEAD)
if [ -n "$COMMIT" ] && [ "$HEAD_COMMIT" != "$COMMIT" ]; then
    fail "$SRC is at $HEAD_COMMIT, but JLLAMA_GIT_COMMIT pins $COMMIT"
fi
if [ -f "$SRC/.git/shallow" ]; then
    fail "$SRC is a shallow clone; the commit count cannot be checked (clone with fetch-depth 0)"
fi
COUNT=$(git -C "$SRC" rev-list --count HEAD)
[ "$COUNT" = "${TAG#b}" ] || fail "the checked-out commit $HEAD_COMMIT is b$COUNT, not $TAG"
git -C "$SRC" diff --quiet && git -C "$SRC" diff --cached --quiet \
    || fail "$SRC is not a clean checkout; the archive is written from a pristine tree plus the patches"

# --- this project's version, from the reactor pom (the first <version> of a parent-less pom) ------
VERSION=$(grep -oE '<version>[^<]+</version>' "$ROOT/pom.xml" | head -1 | sed -E 's|</?version>||g')
[ -n "$VERSION" ] || fail "could not read the reactor version from $ROOT/pom.xml"

# --- apply the patches exactly as every build does -----------------------------------------------
cmake -DPATCH_DIR="$ROOT/llama/patches" -DLLAMA_SRC="$SRC" -P "$ROOT/llama/cmake/apply-llama-patches.cmake"
bash "$ROOT/.github/verify-patches-applied.sh" "$SRC"
[ -f "$SRC/.jllama-patches-applied" ] || fail "the applier wrote no stamp into $SRC"
grep -q "^head $HEAD_COMMIT\$" "$SRC/.jllama-patches-applied" \
    || fail "the stamp does not name $HEAD_COMMIT (the archived tree's build info reads the commit from it)"

# --- pack: the tree without .git, under one directory, reproducibly ---------------------------------
NAME="llama.cpp-$TAG-jllama-$VERSION-src"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME" "$OUT"
# `git ls-files` would drop the files the patches add; copy the tree and drop the repository instead.
tar -C "$SRC" --exclude=.git -cf - . | tar -C "$STAGE/$NAME" -xf -
[ -f "$STAGE/$NAME/.jllama-patches-applied" ] || fail "the stamp did not make it into the staged tree"
[ ! -e "$STAGE/$NAME/.git" ] || fail ".git made it into the staged tree"
# Reproducible: a fixed order, the commit's own time as every file's mtime, no owner, and gzip
# without a name or timestamp in its header -- the same inputs give the same bytes.
COMMIT_TIME=$(git -C "$SRC" log -1 --format=%ct HEAD)
tar --sort=name --mtime="@$COMMIT_TIME" --owner=0 --group=0 --numeric-owner \
    -C "$STAGE" -cf - "$NAME" | gzip -n -9 > "$OUT/$NAME.tar.gz"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
echo "$NAME.tar.gz" > "$OUT/archive-name.txt"
echo "archive-llama-source: $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1)) from $TAG = $HEAD_COMMIT, patches $(grep -cE '\.(patch|diff) ' "$SRC/.jllama-patches-applied")"
cat "$OUT/$NAME.tar.gz.sha256"
