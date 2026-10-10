#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Shared by the bash smoke scripts (sourced, not run): what a smoke set is and what the loader
# must have said about it. A smoke set (.github/package-smoke-sets.sh) is a directory of jars: the
# classes jar, its runtime dependencies, the library jar of the platform and every GPU module jar
# of it; `java -cp '<set>/*'` is the consumer's classpath.

SMOKE_MAIN="net.ladenthin.llama.server.ServerLauncher"
SMOKE_LIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/natives.csv"

# The classpath of a smoke set: every jar in it.
smoke_classpath() {
    local set_dir="$1"
    [ -d "$set_dir" ] || { echo "::error::smoke set directory '$set_dir' does not exist" >&2; return 1; }
    find "$set_dir" -maxdepth 1 -name '*.jar' | grep -q . || { echo "::error::no jars in the smoke set $set_dir" >&2; return 1; }
    echo "$set_dir/*"
}

# The GPU module backends whose jars are in the smoke set (from .github/natives.csv), one per line.
smoke_set_modules() {
    local set_dir="$1"
    grep -v -e '^#' -e '^classifier,' -e '^$' "$SMOKE_LIST" | tr -d '\r' | awk -F, '$5 == "module" { print $1 "," $2 }' \
        | while IFS=, read -r classifier dir; do
            if find "$set_dir" -maxdepth 1 -name "llama-*-$classifier.jar" | grep -q .; then
                basename "$dir"
            fi
        done
}

# Asserts the loader line in the given logs: the library loaded, and every GPU module jar of the set
# named in it -- a module jar the loader skipped would otherwise pass as "runs on the CPU anyway".
smoke_assert_loader_line() {
    local set_dir="$1"
    shift
    local line
    line="$(grep -h "\[jllama\] native backend '[^']*' loaded " "$@" | head -1 || true)"
    [ -n "$line" ] || { echo "::error::no loader line ([jllama] native backend ... loaded) in $*" >&2; return 1; }
    local module
    while IFS= read -r module; do
        [ -z "$module" ] && continue
        grep -qE "GPU module\(s\) ([a-z0-9-]+, )*$module(,|;|$)" <<< "$line" \
            || { echo "::error::the loader did not put the $module module in place: $line" >&2; return 1; }
    done <<< "$(smoke_set_modules "$set_dir")"
    echo "loader: $line"
}
