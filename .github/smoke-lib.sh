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
    # $6 == "yes" is an opt-in module (natives.csv): its jar on the classpath does nothing until
    # net.ladenthin.llama.backend names it, so the loader deliberately leaves it out and the line
    # below must not expect it. smoke-natives-jars.sh names them instead, which is where the opt-in
    # path itself is exercised.
    grep -v -e '^#' -e '^classifier,' -e '^$' "$SMOKE_LIST" | tr -d '\r' \
        | awk -F, '$5 == "module" && $6 != "yes" { print $1 "," $2 }' \
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

# The classpath of a smoke set WITHOUT its natives jars: the classes jar and its runtime
# dependencies only -- what a consumer has who ships the natives as a directory next to the
# application and names it with net.ladenthin.llama.lib.path (CLAUDE.md, "Natives jars": the
# natives directory contract). Every jar of the set whose name ends in a natives.csv classifier is
# left out; the result is a colon-separated list, never a wildcard.
smoke_classpath_without_natives() {
    local set_dir="$1"
    local sep=":"
    case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) sep=";" ;; esac
    local classifiers
    classifiers="$(grep -v -e '^#' -e '^classifier,' -e '^$' "$SMOKE_LIST" | tr -d '\r' | cut -d, -f1)"
    local cp="" jar base keep c
    while IFS= read -r jar; do
        base="$(basename "$jar" .jar)"
        keep=1
        while IFS= read -r c; do
            case "$base" in *"-$c") keep=0 ;; esac
        done <<< "$classifiers"
        [ "$keep" -eq 1 ] && cp="${cp:+$cp$sep}$jar"
    done < <(find "$set_dir" -maxdepth 1 -name '*.jar' | sort)
    [ -n "$cp" ] || { echo "::error::no non-natives jar in the smoke set $set_dir" >&2; return 1; }
    echo "$cp"
}

# Unpacks the natives jars of a smoke set into ONE natives directory, the way a consumer does by
# hand (README, "Natives outside Maven"): the library jar's directory
# net/ladenthin/llama/<OS>/<ARCH>/<backend>/ is the directory, and every GPU module jar's files are
# copied flat into it -- the module's library next to the CPU set, its jllama-extras.txt lines
# appended to the library's (a vendor loader a module ships must still be preloaded), its
# jllama-files.txt lines appended likewise, its own jllama-build.txt and jllama-sbom.json dropped
# (same tag by contract, which the loader checked when the jars were built). Prints the directory.
smoke_natives_directory() {
    # Absolute, because the jars are unpacked after a cd into the output directory and the
    # workflows pass the set relative to the job's directory (`smoke`).
    local set_dir out="$2"
    set_dir="$(cd "$1" && pwd)" || { echo "::error::smoke set directory '$1' does not exist" >&2; return 1; }
    rm -rf "$out" && mkdir -p "$out"
    local rows
    rows="$(grep -v -e '^#' -e '^classifier,' -e '^$' "$SMOKE_LIST" | tr -d '\r')"
    local lib_dir="" classifier dir lib platform kind optin jar
    while IFS=, read -r classifier dir lib platform kind optin; do
        [ "$kind" = "library" ] || continue
        jar="$(find "$set_dir" -maxdepth 1 -name "llama-*-$classifier.jar" | head -1)"
        [ -n "$jar" ] || continue
        [ -z "$lib_dir" ] || { echo "::error::two library jars in the smoke set: $lib_dir and $classifier" >&2; return 1; }
        mkdir -p "$out/library"
        (cd "$out/library" && unzip -q "$jar") || { echo "::error::cannot unzip $jar" >&2; return 1; }
        lib_dir="$out/library/net/ladenthin/llama/$dir"
        [ -f "$lib_dir/$lib" ] || { echo "::error::$jar holds no net/ladenthin/llama/$dir/$lib" >&2; return 1; }
    done <<< "$rows"
    [ -n "$lib_dir" ] || { echo "::error::no library jar in the smoke set $set_dir" >&2; return 1; }
    while IFS=, read -r classifier dir lib platform kind optin; do
        [ "$kind" = "module" ] || continue
        jar="$(find "$set_dir" -maxdepth 1 -name "llama-*-$classifier.jar" | head -1)"
        [ -n "$jar" ] || continue
        mkdir -p "$out/$classifier"
        (cd "$out/$classifier" && unzip -q "$jar") || { echo "::error::cannot unzip $jar" >&2; return 1; }
        local mod_dir="$out/$classifier/net/ladenthin/llama/$dir" f
        [ -f "$mod_dir/$lib" ] || { echo "::error::$jar holds no net/ladenthin/llama/$dir/$lib" >&2; return 1; }
        for f in "$mod_dir"/*; do
            case "$(basename "$f")" in
                jllama-build.txt|jllama-sbom.json) ;;
                jllama-extras.txt|jllama-files.txt)
                    tr -d '\r' < "$f" | grep -v -e '^#' -e '^$' >> "$lib_dir/$(basename "$f")" || true ;;
                *) [ ! -e "$lib_dir/$(basename "$f")" ] \
                       || { echo "::error::$classifier ships $(basename "$f"), which the library directory already holds" >&2; return 1; }
                   cp "$f" "$lib_dir/" ;;
            esac
        done
    done <<< "$rows"
    echo "$lib_dir"
}
