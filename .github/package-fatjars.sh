#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Checks every natives jar and assembles the per-OS "all backends" server fat jars distributed
# as GitHub Release assets (never deployed to Maven Central).
#
# A natives jar holds exactly one directory, net/ladenthin/llama/<OS>/<ARCH>/<backend>/, named
# by its classifier <backend>-<os>-<arch>. Because the directories never overlap, an all-backends
# jar is a plain merge: the default fat jar (classes + Java deps + the CPU natives of every
# platform), minus every native tree of another OS/arch, plus every natives jar of its own OS/arch.
# LlamaLoader tries the backend directories in its fixed priority order (BACKEND_PRIORITY) and
# falls back to the CPU one.
#
# Only the exact OS+arch of the jar name is kept, so all-windows-x86-64 also loses Windows/x86
# and Windows/aarch64: a 32-bit JVM finds no natives there. The default fat jar (step 5) still
# carries every platform.
#
# The natives jars are the rows of .github/natives.csv (classifier, directory, library); that
# the pom, llama-platform, the workflow and LlamaLoader agree with it is checked by
# check-natives.py. This script checks the built jars.
#
# Fail-loud invariants (a broken invariant must red the pipeline, never skip):
#   * the listed natives jars and the natives jars on disk match exactly,
#   * every natives jar holds its own directory, with its library, and nothing else, and names
#     the Automatic-Module-Name the module path needs (without it the JVM silently drops the jar),
#   * a combined jar keeps its own CPU library (byte-identical), carries no other OS/arch,
#     had at least one tree removed, keeps every .class of the default fat jar, holds every
#     added backend byte-identical to its natives jar, and keeps its Main-Class.
#
# Usage: package-fatjars.sh <jars-dir> <out-dir>
#   jars-dir  directory holding the `llama-jars` artifact (llama/target/*.jar)
#   out-dir   output directory for the combined fat jars + sha256 files
set -euo pipefail

JARS_DIR="${1:?usage: package-fatjars.sh <jars-dir> <out-dir>}"
OUT_DIR="${2:?usage: package-fatjars.sh <jars-dir> <out-dir>}"
LIST="$(dirname "$0")/natives.csv"

fail() {
    echo "::error::$*" >&2
    exit 1
}

[ -d "$JARS_DIR" ] || fail "jars dir not found: $JARS_DIR"
JARS_DIR="$(cd "$JARS_DIR" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Backends that get no place in an all-backends jar: msvc is the same CPU build from another
# generator, so it would only be a second CPU library the loader tries before the first.
EXCLUDED_BACKENDS=" msvc "
# Platforms that get no all-backends jar: `java -jar` does not apply on Android (the AAR does).
EXCLUDED_OSES=" Linux-Android "

MAIN_CLASS="net.ladenthin.llama.server.ServerLauncher"

# Native trees are the upper-case DIRECTORIES below net/ladenthin/llama/ (OSInfo folder names:
# Linux, Linux-Android, Mac, Windows); Java packages there are lower-case, and top-level class
# files such as LlamaModel.class are upper-case too, so a match must require the directory's
# trailing slash. Prints <OS>/<ARCH>, sorted.
native_trees() {
    unzip -Z1 "$1" | sed -n -E 's#^net/ladenthin/llama/([A-Z][^/]*/[^/]+)/.*#\1#p' | sort -u
}

class_count() {
    unzip -Z1 "$1" | grep -c '\.class$'
}

# --- 1. The listed natives jars, the default fat jar, and the natives jars on disk ---------
declare -A DIR LIB
LISTED=()
while IFS=, read -r classifier dir lib _; do
    LISTED+=("$classifier") DIR[$classifier]="$dir" LIB[$classifier]="$lib"
done < <(grep -v -e '^#' -e '^classifier,' -e '^$' "$LIST")
[ "${#LISTED[@]}" -gt 0 ] || fail "no rows in $LIST"
mapfile -t LISTED < <(printf '%s\n' "${LISTED[@]}" | sort)

mapfile -t BASE_FAT_JARS < <(find "$JARS_DIR" -maxdepth 1 -name 'llama-*-jar-with-dependencies.jar' | sort)
[ "${#BASE_FAT_JARS[@]}" -eq 1 ] \
    || fail "expected exactly 1 default jar-with-dependencies in $JARS_DIR, got ${#BASE_FAT_JARS[@]}: ${BASE_FAT_JARS[*]:-none}"
BASE_FAT_JAR="${BASE_FAT_JARS[0]}"
VERSION="$(basename "$BASE_FAT_JAR")"
VERSION="${VERSION#llama-}"
VERSION="${VERSION%-jar-with-dependencies.jar}"
echo "base fat jar: $BASE_FAT_JAR (version $VERSION)"

DISK_CLASSIFIERS=()
for jar in "$JARS_DIR"/llama-"$VERSION"-*.jar; do
    [ -e "$jar" ] || continue
    classifier="$(basename "$jar")"
    classifier="${classifier#llama-"$VERSION"-}"
    classifier="${classifier%.jar}"
    case "$classifier" in
        sources | javadoc | jar-with-dependencies) continue ;;
    esac
    DISK_CLASSIFIERS+=("$classifier")
done
[ "${#DISK_CLASSIFIERS[@]}" -gt 0 ] || fail "no natives jars found in $JARS_DIR for version $VERSION"
mapfile -t DISK_CLASSIFIERS < <(printf '%s\n' "${DISK_CLASSIFIERS[@]}" | sort -u)
if ! diff <(printf '%s\n' "${LISTED[@]}") <(printf '%s\n' "${DISK_CLASSIFIERS[@]}"); then
    fail "natives jars in $JARS_DIR differ from $LIST (see diff above)"
fi
echo "natives jars match $LIST (${#LISTED[@]})"

# --- 2. Every natives jar holds its own directory and nothing else -------------------------
declare -A TARGET_CLASSIFIERS # "<os>-<arch>" -> space-separated classifiers to merge
declare -A TARGET_FOLDER      # "<os>-<arch>" -> "<OS>/<ARCH> <library>"
for classifier in "${LISTED[@]}"; do
    dir="net/ladenthin/llama/${DIR[$classifier]}/" lib="${LIB[$classifier]}"
    jar="$JARS_DIR/llama-$VERSION-$classifier.jar"
    entries="$(unzip -Z1 "$jar" | grep -v -e '^META-INF/' -e '/$' || true)"
    grep -qxF "$dir$lib" <<< "$entries" || fail "$(basename "$jar") lacks $dir$lib"
    stray="$(grep -v "^$dir" <<< "$entries" || true)"
    [ -z "$stray" ] || fail "$(basename "$jar") holds files outside $dir: $(echo "$stray" | tr '\n' ' ')"
    # Manifest lines are folded at 72 bytes; a continuation line starts with one space.
    module="$(unzip -p "$jar" META-INF/MANIFEST.MF | tr -d '\r' | sed ':a;N;$!ba;s/\n //g' \
        | sed -n 's/^Automatic-Module-Name: //p')"
    [ "$module" = "net.ladenthin.llama.natives.${classifier//-/_}" ] \
        || fail "$(basename "$jar") declares Automatic-Module-Name '$module', expected net.ladenthin.llama.natives.${classifier//-/_}"
    echo "natives jar OK: $classifier -> $dir ($module)"
    backend="${DIR[$classifier]##*/}" tree="${DIR[$classifier]%/*}"
    case "$EXCLUDED_BACKENDS" in *" $backend "*) continue ;; esac
    case "$EXCLUDED_OSES" in *" ${tree%%/*} "*) continue ;; esac
    target="${classifier#"$backend"-}"
    TARGET_CLASSIFIERS[$target]="${TARGET_CLASSIFIERS[$target]:-} $classifier"
    TARGET_FOLDER[$target]="$tree $lib"
done

# --- 3. One combined jar per OS/arch that has a backend besides the CPU one -----------------
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

for target in $(printf '%s\n' "${!TARGET_CLASSIFIERS[@]}" | sort); do
    classifiers="${TARGET_CLASSIFIERS[$target]}"
    [ "$(wc -w <<< "$classifiers")" -gt 1 ] || continue
    case "$classifiers " in *" cpu-$target "*) ;; *) fail "$target has GPU backends but no cpu-$target natives jar" ;; esac
    read -r tree lib <<< "${TARGET_FOLDER[$target]}"
    own="net/ladenthin/llama/$tree/"

    staging="$WORK_DIR/$target"
    mkdir -p "$staging"
    for classifier in $classifiers; do
        unzip -q -o "$JARS_DIR/llama-$VERSION-$classifier.jar" 'net/*' -d "$staging"
    done

    out_jar="$OUT_DIR/llama-$VERSION-all-$target-jar-with-dependencies.jar"
    cp "$BASE_FAT_JAR" "$out_jar"
    # Drop every native tree but this target's. Matched by exact path components, never by
    # prefix: a Linux* glob would also hit Linux-Android.
    foreign_entries="$WORK_DIR/foreign-$target.txt"
    unzip -Z1 "$out_jar" \
        | awk -v keep="$own" -v own_os="net/ladenthin/llama/${tree%%/*}/" \
            '/^net\/ladenthin\/llama\/[A-Z][^\/]*\// && index($0, keep) != 1 && $0 != own_os' \
            > "$foreign_entries"
    [ -s "$foreign_entries" ] \
        || fail "$out_jar: no foreign native tree to remove — did the OSInfo folder names change?"
    zip -q -d -nw "$out_jar" -@ < "$foreign_entries" || fail "$out_jar: removing foreign native trees failed"
    (cd "$staging" && zip -q -ur "$out_jar" net)

    # --- Verify the combined jar --------------------------------------------------------
    for classifier in $classifiers; do
        backend="${classifier%-"$target"}"
        unzip -p "$out_jar" "$own$backend/$lib" | cmp -s - "$staging/$own$backend/$lib" \
            || fail "$out_jar: $own$backend/$lib is missing or differs from its natives jar"
    done
    unzip -p "$out_jar" "${own}cpu/$lib" | cmp -s - <(unzip -p "$BASE_FAT_JAR" "${own}cpu/$lib") \
        || fail "$out_jar: CPU fallback ${own}cpu/$lib is missing or differs from the default fat jar's"
    out_trees="$(native_trees "$out_jar")"
    [ "$out_trees" = "$tree" ] \
        || fail "$out_jar: expected only the native tree $tree, found: $(echo "$out_trees" | tr '\n' ' ')"
    [ "$(class_count "$out_jar")" -eq "$(class_count "$BASE_FAT_JAR")" ] \
        || fail "$out_jar: .class count differs from the default fat jar — the tree filter removed classes"
    unzip -p "$out_jar" META-INF/MANIFEST.MF | grep -qF "Main-Class: $MAIN_CLASS" \
        || fail "$out_jar: Main-Class $MAIN_CLASS did not survive the zip update"

    (cd "$OUT_DIR" && sha256sum "$(basename "$out_jar")" > "$(basename "$out_jar").sha256")
    echo "OK: $(basename "$out_jar") ($(du -h "$out_jar" | cut -f1);$classifiers)"
done

# --- 4. Exactly the targets the list derives were produced ---------------------------------
# check-natives.py derives them from natives.csv as well and checks every consumer against the
# same derivation (smoke jobs, the agent jar's Class-Path, README), so a target this loop drops
# or invents cannot reach a release unlaunched.
produced="$(cd "$OUT_DIR" && ls llama-"$VERSION"-all-*-jar-with-dependencies.jar \
    | sed -e "s/^llama-$VERSION-all-//" -e 's/-jar-with-dependencies\.jar$//' | sort)"
expected="$(python3 "$(dirname "$0")/check-natives.py" fatjar-targets | sort)"
[ -n "$expected" ] || fail "check-natives.py derived no all-backends fat jar from $LIST"
[ "$produced" = "$expected" ] \
    || fail "all-backends fat jars produced: $(echo $produced) -- natives.csv derives: $(echo $expected)"

# --- 5. The default (all-platform CPU) fat jar is a release asset too -------------------------
cp "$BASE_FAT_JAR" "$OUT_DIR/"
(cd "$OUT_DIR" && sha256sum "$(basename "$BASE_FAT_JAR")" > "$(basename "$BASE_FAT_JAR").sha256")
echo "OK: $(basename "$BASE_FAT_JAR") (default CPU fat jar, copied as-is)"

ls -lh "$OUT_DIR"
