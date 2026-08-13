#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Merges the per-artifact native-library trees downloaded by the `*-libraries` glob into the
# single default-JAR resource tree — and FAILS LOUD if two artifacts claim the same file path.
#
# Why this exists: the `package` / `publish-snapshot` / `publish-release` jobs pull every build
# job whose artifact name ends in `-libraries` with one globbed `actions/download-artifact`.
# That is convenient (a new CPU platform ships in the default JAR by naming its artifact
# `<Something>-libraries`, no packaging change) but it is silently unsafe: an artifact name says
# nothing about which `{OS}/{ARCH}` subdirectory the job's CMake run actually wrote. When two
# artifacts carry the same relative path, `merge-multiple: true` extracts both onto that one
# path and the survivor can be a byte-level hybrid of the two, not either input.
#
# That is exactly what happened to macOS arm64: all three macOS build jobs write
# `Mac/aarch64/libjllama.dylib` (none of them passes -DOS_NAME/-DOS_ARCH, so CMakeLists
# auto-detects the same subdir) and all three used to upload under a `*-libraries` name. The
# published dylib became a hybrid whose ad-hoc linker signature no longer matched its own
# __TEXT pages, so macOS SIGKILLed every process that loaded it — shipped broken in 5.0.6 and
# several 5.0.7 snapshots. The immediate fix renamed those artifacts out of the glob; this
# script is the backstop that stops the same hole from being reopened by a future job.
#
# NOTE ON WHAT *CANNOT* WORK AS A GUARD: asserting "exactly one library per {OS}/{ARCH}" on the
# merged tree does not detect this. The collision overwrites one path, so the merged tree still
# holds exactly one file there — a corrupt one. The collision is only observable BEFORE the
# merge, which is why this script does the merge itself instead of checking afterwards.
#
# Usage: merge-native-artifacts.sh <staging-dir> <dest-dir>
#   <staging-dir>  output of `actions/download-artifact` with `pattern: "*-libraries"` and
#                  `merge-multiple: false`, i.e. one subdirectory per artifact name.
#   <dest-dir>     the tree the artifacts are merged into, e.g.
#                  llama/src/main/resources/net/ladenthin/llama/
#
# Fail-loud: aborts when the staging directory holds no artifacts (a silently empty default JAR
# is worse than a red job) and when any relative path is claimed by more than one artifact.

set -euo pipefail

STAGING="${1:?usage: merge-native-artifacts.sh <staging-dir> <dest-dir>}"
DEST="${2:?usage: merge-native-artifacts.sh <staging-dir> <dest-dir>}"

if [ ! -d "$STAGING" ]; then
  echo "::error::staging directory '$STAGING' does not exist — the globbed download did not run." >&2
  exit 1
fi

# One subdirectory per downloaded artifact. Depth 1 only: everything below is artifact content.
artifacts=()
while IFS= read -r d; do artifacts+=("$(basename "$d")"); done < <(find "$STAGING" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "${#artifacts[@]}" -eq 0 ]; then
  echo "::error::no '*-libraries' artifacts found in '$STAGING' — the default JAR would ship without native libraries." >&2
  exit 1
fi

echo "Merging ${#artifacts[@]} native-library artifact(s) into $DEST"
for a in "${artifacts[@]}"; do echo "  - $a"; done

# relpath -> space-separated list of artifacts that carry it. Bash 3.2 (macOS) has no
# associative arrays, so this stays a sorted "<relpath>\t<artifact>" stream processed by awk.
collisions="$(
  for a in "${artifacts[@]}"; do
    (cd "$STAGING/$a" && find . -type f | sed 's|^\./||' | while IFS= read -r f; do printf '%s\t%s\n' "$f" "$a"; done)
  done | sort | awk -F'\t' '
    { if ($1 == prev) { owners = owners " " $2; n++ } else { if (n > 1) print prev "\t" owners; prev = $1; owners = $2; n = 1 } }
    END { if (n > 1) print prev "\t" owners }
  '
)"

if [ -n "$collisions" ]; then
  echo "::error::two or more '*-libraries' artifacts write the same path — merging them would produce a hybrid, corrupt native library." >&2
  while IFS=$'\t' read -r path owners; do
    echo "::error::  $path  <- claimed by:$owners" >&2
  done <<< "$collisions"
  cat >&2 <<'EOF'
::error::Fix: only ONE artifact per {OS}/{ARCH} may be named `*-libraries`. Rename the extra
::error::build jobs' artifacts outside the glob (as the macOS jobs do: macos-15-metal /
::error::macos-14-metal / macos-15-no-metal) and download the variant that ships explicitly by
::error::name. See CLAUDE.md, "macOS arm64: three build jobs, one shipped dylib".
EOF
  exit 1
fi

mkdir -p "$DEST"
for a in "${artifacts[@]}"; do
  cp -R "$STAGING/$a/." "$DEST/"
done

echo "Merged native tree:"
find "$DEST" -type f | sort | sed 's|^|  |'
