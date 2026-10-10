#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Assembles one smoke set per smoke target of .github/natives.csv (check-natives.py smoke-targets):
# <out-dir>/<target>/ holds the classes jar, its runtime dependencies and every natives jar of the
# target -- the library jar and each GPU module jar -- so a smoke job on a runner of that OS/arch
# starts the server with `java -cp '<set>/*'`, which is exactly a consumer's classpath (llama-platform
# plus the GPU classifiers), with no fat jar in between. The sets are the only thing the smoke jobs
# download: the natives jars of the other platforms (ROCm alone is hundreds of MB) stay behind.
#
# Fail-loud: a natives jar the list names that is not in <llama/target> aborts (the package job's
# natives profile builds one per row; a missing one means a row without a build), and so does a
# target without any jar.
#
# Usage: package-smoke-sets.sh <llama/target> <deps-dir> <out-dir>
#   <llama/target>  directory holding llama-<v>.jar and the llama-<v>-<classifier>.jar natives jars
#   <deps-dir>      the classes jar's runtime dependencies as jars (mvn dependency:copy-dependencies)
#   <out-dir>       output directory; one subdirectory per smoke target
set -euo pipefail

TARGET_DIR="${1:?usage: package-smoke-sets.sh <llama/target> <deps-dir> <out-dir>}"
DEPS_DIR="${2:?usage: package-smoke-sets.sh <llama/target> <deps-dir> <out-dir>}"
OUT_DIR="${3:?usage: package-smoke-sets.sh <llama/target> <deps-dir> <out-dir>}"
CHECK="$(dirname "$0")/check-natives.py"

fail() {
    echo "::error::$*" >&2
    exit 1
}

classes="$(find "$TARGET_DIR" -maxdepth 1 -name 'llama-*.jar' | grep -E '/llama-[0-9][0-9.]*(-SNAPSHOT)?\.jar$' || true)"
[ -n "$classes" ] && [ "$(wc -l <<< "$classes")" -eq 1 ] || fail "expected exactly one classes jar in $TARGET_DIR, got: ${classes:-none}"
version="$(basename "$classes" .jar)"
version="${version#llama-}"
deps_count="$(find "$DEPS_DIR" -maxdepth 1 -name '*.jar' | wc -l)"
[ "$deps_count" -gt 0 ] || fail "no dependency jars in $DEPS_DIR"
mkdir -p "$OUT_DIR"

targets="$(python3 "$CHECK" smoke-targets)"
[ -n "$targets" ] || fail "check-natives.py smoke-targets printed no target"
for target in $targets; do
    set_dir="$OUT_DIR/$target"
    mkdir -p "$set_dir"
    cp "$classes" "$set_dir/"
    cp "$DEPS_DIR"/*.jar "$set_dir/"
    for classifier in $(python3 "$CHECK" smoke-set "$target"); do
        jar="$TARGET_DIR/llama-$version-$classifier.jar"
        [ -f "$jar" ] || fail "natives jar missing for smoke target $target: $jar (a row of natives.csv without a build?)"
        cp "$jar" "$set_dir/"
    done
    echo "smoke set $target ($(du -sh "$set_dir" | cut -f1)):"
    find "$set_dir" -maxdepth 1 -name 'llama-*.jar' -exec basename {} \; | sort | sed 's/^/  /'
done
echo "$(wc -w <<< "$targets") smoke set(s) in $OUT_DIR"
