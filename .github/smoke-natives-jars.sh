#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Loads the library from the published jars exactly as a consumer does: the classes jar, its
# runtime dependencies and EVERY natives jar at once -- first on the classpath, then on the module
# path. Each natives jar holds its own <OS>/<ARCH>/<backend>/ directory, so all of them fit on one
# classpath; on the module path each is an automatic module and needs a unique
# Automatic-Module-Name, or the JVM refuses to start ("Two versions of module ... found").
#
# On a GPU-less runner the GPU backends normally fail their load (no vendor runtime) and the loader
# falls through to the CPU library. A GPU library whose runtime happens to be installed may load and
# find no device, which is benign, so any selected backend passes; the log names the one chosen.
#
# Usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>
#   <llama/target>       directory holding llama-<v>.jar and the llama-<v>-<classifier>.jar natives jars
#   <runtime-classpath>  the classes jar's runtime dependencies (mvn dependency:build-classpath)
set -euo pipefail

TARGET="${1:?usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>}"
DEPS="${2:?usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>}"
SMOKE="$(dirname "$0")/smoke/NativeLoadSmoke.java"

# The classes jar is the one without a classifier; every other jar but sources/javadoc/the fat jar
# is a natives jar.
classes="$(find "$TARGET" -maxdepth 1 -name 'llama-*.jar' | grep -E '/llama-[0-9][0-9.]*(-SNAPSHOT)?\.jar$' || true)"
[ "$(wc -l <<< "$classes")" -eq 1 ] && [ -n "$classes" ] || { echo "::error::expected one classes jar in $TARGET: $classes" >&2; exit 2; }
mapfile -t natives < <(find "$TARGET" -maxdepth 1 -name 'llama-*.jar' ! -path "$classes" ! -name '*-sources.jar' \
    ! -name '*-javadoc.jar' ! -name '*-jar-with-dependencies.jar' | sort)
[ "${#natives[@]}" -gt 0 ] || { echo "::error::no natives jars in $TARGET" >&2; exit 2; }
path="$classes:$(IFS=:; echo "${natives[*]}"):$DEPS"
echo "classes jar: $classes; ${#natives[@]} natives jars"

run() {
    local label="$1"
    shift
    local out
    out="$(java "$@" "$SMOKE" 2>&1)" || { echo "$out" >&2; echo "::error::$label: the library did not load" >&2; exit 1; }
    backend="$(grep -o "\[jllama\] using native backend '[^']*'" <<< "$out" || true)"
    [ -n "$backend" ] || { echo "$out" >&2; echo "::error::$label: the loader reported no backend" >&2; exit 1; }
    echo "OK ($label): $backend; $(grep 'native load smoke OK' <<< "$out")"
}

run "classpath" -cp "$path"
run "module path" --module-path "$path" --add-modules ALL-MODULE-PATH
