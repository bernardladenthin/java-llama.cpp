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
# The library jar of this runner's platform is loaded, and every GPU module jar of the platform is
# put next to it (the loader line names them all -- checked against .github/natives.csv, so a module
# jar the loader skipped would fail here). On a GPU-less runner ggml then fails to open each module
# (no vendor runtime) and runs on the CPU, which is benign; a module whose runtime happens to be
# installed registers no device, which is benign too.
#
# Usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>
#   <llama/target>       directory holding llama-<v>.jar and the llama-<v>-<classifier>.jar natives jars
#   <runtime-classpath>  the classes jar's runtime dependencies (mvn dependency:build-classpath)
set -euo pipefail

TARGET="${1:?usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>}"
DEPS="${2:?usage: smoke-natives-jars.sh <llama/target> <runtime-classpath>}"
SMOKE="$(dirname "$0")/smoke/NativeLoadSmoke.java"
LIST="$(dirname "$0")/natives.csv"

# The module backends of this runner's platform, from the list: what the loader line must name.
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) tree="Linux/x86_64" ;;
    Linux-aarch64) tree="Linux/aarch64" ;;
    Darwin-arm64) tree="Mac/aarch64" ;;
    *) echo "::error::no smoke expectation for $(uname -s)-$(uname -m)" >&2; exit 2 ;;
esac
expected_modules="$(grep -v -e '^#' -e '^classifier,' -e '^$' "$LIST" | tr -d '\r' \
    | awk -F, -v t="$tree/" 'index($2, t) == 1 && $5 == "module" { sub(t, "", $2); print $2 }')"

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
    backend="$(grep -o "\[jllama\] native backend '[^']*' loaded .*" <<< "$out" || true)"
    [ -n "$backend" ] || { echo "$out" >&2; echo "::error::$label: the loader reported no backend" >&2; exit 1; }
    if [ -n "$expected_modules" ]; then
        while IFS= read -r module; do
            grep -qE "GPU module\(s\) ([a-z0-9-]+, )*$module(,|;|$)" <<< "$backend" \
                || { echo "$out" >&2; echo "::error::$label: the loader did not put the $module module jar of $tree in place" >&2; exit 1; }
        done <<< "$expected_modules"
    else
        grep -q "with no GPU module" <<< "$backend" || { echo "$out" >&2; echo "::error::$label: unexpected modules" >&2; exit 1; }
    fi
    echo "OK ($label): $backend; $(grep 'native load smoke OK' <<< "$out")"
}

run "classpath" -cp "$path"
run "module path" --module-path "$path" --add-modules ALL-MODULE-PATH
