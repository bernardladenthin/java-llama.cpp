#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Merges the per-build native-library artifacts downloaded by the `natives-*` glob into the one
# natives tree (llama/src/main/natives/net/ladenthin/llama/) that the `natives` Maven profile
# packages — and FAILS LOUD when an artifact does not hold exactly what its name promises.
#
# Every shipped build uploads its tree as `natives-<classifier>`, one per row of
# .github/natives.csv, which names the one directory it may write. Checked before anything is
# merged:
#   * every listed artifact arrived, with its library, and no unlisted one did;
#   * every file lies below the artifact's own directory. A build whose CMake picked another
#     backend name or platform (the routing lives in llama/CMakeLists.txt) would otherwise land in
#     some other natives jar, or in none;
#   * no relative path is claimed by two artifacts. This follows from the second check, and is
#     kept anyway because it is the failure that actually shipped: all three macOS build jobs
#     wrote Mac/aarch64 and used to share one glob, the merge produced a byte-level hybrid of two
#     dylibs, and macOS SIGKILLed every process that loaded it (5.0.6 and several 5.0.7
#     snapshots). A check on the merged tree cannot see it — the collision leaves exactly one
#     file on the path, a corrupt one — so it has to run before the merge.
#
# After the merge, a backend directory holding files beside its library gets a
# jllama-extras.txt listing them; LlamaLoader loads those first (e.g. a vendor loader a build ships
# next to itself). Not listed there: the files a build's own jllama-files.txt names -- ggml's shared
# libraries and the CPU backend modules of a JLLAMA_BACKEND_DL build, and the one GPU module of a
# JLLAMA_MODULE_ONLY build (CLAUDE.md, "Modular natives"), which the loader only extracts next to the
# library; loading them from Java would bypass ggml's choice of the module. Each file that list names
# must exist, or the loader would fail the backend on every machine. Every directory must carry its
# jllama-build.txt (the llama.cpp tag, CMake writes it): LlamaLoader refuses a GPU module without one.
#
# Usage: merge-native-artifacts.sh <staging-dir> <dest-dir>
#   <staging-dir>  output of `actions/download-artifact` with `pattern: "natives-*"` and
#                  `merge-multiple: false`, i.e. one subdirectory per artifact name.
#   <dest-dir>     the tree the artifacts are merged into,
#                  llama/src/main/natives/net/ladenthin/llama/
#
# Fail-loud: also aborts when the staging directory holds no artifacts.

set -euo pipefail

STAGING="${1:?usage: merge-native-artifacts.sh <staging-dir> <dest-dir>}"
DEST="${2:?usage: merge-native-artifacts.sh <staging-dir> <dest-dir>}"
LIST="$(dirname "$0")/natives.csv"

if [ ! -d "$STAGING" ]; then
  echo "::error::staging directory '$STAGING' does not exist — the globbed download did not run." >&2
  exit 1
fi

# One subdirectory per downloaded artifact. Depth 1 only: everything below is artifact content.
artifacts=()
while IFS= read -r d; do artifacts+=("$(basename "$d")"); done < <(find "$STAGING" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "${#artifacts[@]}" -eq 0 ]; then
  echo "::error::no 'natives-*' artifacts found in '$STAGING' — there would be no natives jars." >&2
  exit 1
fi

echo "Merging ${#artifacts[@]} native-library artifact(s) into $DEST"
listed=0
while IFS=, read -r classifier dir lib _platform kind; do
  listed=$((listed + 1))
  a="natives-$classifier"
  if [ ! -d "$STAGING/$a" ]; then
    echo "::error::no artifact '$a' -- the build job for this row of $LIST did not upload it" >&2
    exit 1
  fi
  stray="$(cd "$STAGING/$a" && find . -type f | sed 's|^\./||' | grep -v "^$dir/" || true)"
  if [ -n "$stray" ] || [ ! -f "$STAGING/$a/$dir/$lib" ]; then
    echo "::error::artifact '$a' must hold $dir/$lib and nothing outside $dir/; outside it:" >&2
    printf '%s\n' "$stray" | sed 's|^|::error::  |' >&2
    exit 1
  fi
  if [ ! -f "$STAGING/$a/$dir/jllama-build.txt" ] || ! grep -q '^llama.cpp=b[0-9]' "$STAGING/$a/$dir/jllama-build.txt"; then
    echo "::error::artifact '$a' has no $dir/jllama-build.txt naming its llama.cpp tag -- llama/CMakeLists.txt writes it for every build" >&2
    exit 1
  fi
  if [ "$kind" = "module" ]; then
    # a GPU jar is its module and the two list files, nothing else: no second library, no CPU set
    others="$(cd "$STAGING/$a/$dir" && find . -maxdepth 1 -type f ! -name "$lib" ! -name 'jllama-*.txt' | sed 's|^\./||' || true)"
    if [ -n "$others" ] || ! grep -qx "$lib" "$STAGING/$a/$dir/jllama-files.txt" 2>/dev/null; then
      echo "::error::artifact '$a' is a module jar: $dir/ must hold $lib (named in jllama-files.txt) and the jllama-*.txt lists only; found: $others" >&2
      exit 1
    fi
  fi
  echo "  - $a -> $dir/ ($kind)"
done < <(grep -v -e '^#' -e '^classifier,' -e '^$' "$LIST" | tr -d '\r')
if [ "$listed" -ne "${#artifacts[@]}" ]; then
  echo "::error::$STAGING holds ${#artifacts[@]} natives-* artifacts but $LIST lists $listed: $(printf '%s ' "${artifacts[@]}")" >&2
  exit 1
fi

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
  echo "::error::two or more 'natives-*' artifacts write the same path — merging them would produce a hybrid, corrupt native library." >&2
  while IFS=$'\t' read -r path owners; do
    echo "::error::  $path  <- claimed by:$owners" >&2
  done <<< "$collisions"
  echo "::error::Fix: only one shipped build per <backend>-<os>-<arch>; name a test-only variant outside the glob (see CLAUDE.md, \"macOS arm64\")." >&2
  exit 1
fi

mkdir -p "$DEST"
for a in "${artifacts[@]}"; do
  cp -R "$STAGING/$a/." "$DEST/"
done

# Sibling files of a backend's library are loaded before it, in name order -- except the ones its
# jllama-files.txt names (extracted, never loaded; see the header) and that list itself.
find "$DEST" -mindepth 3 -maxdepth 3 -type d | sort | while IFS= read -r dir; do
  plain=""
  if [ -f "$dir/jllama-files.txt" ]; then
    # tr -d '\r': a Windows build may write the list with CRLF; a name carrying its CR would fail
    # the existence check below and, worse, miss the awk skip-list, so the CPU modules would land in
    # jllama-extras.txt and LlamaLoader would load every one of them.
    plain="$(tr -d '\r' < "$dir/jllama-files.txt" | grep -v -e '^#' -e '^$' || true)"
    while IFS= read -r f; do
      [ -z "$f" ] || [ -f "$dir/$f" ] \
        || { echo "::error::${dir#"$DEST"}/jllama-files.txt names '$f', which the build did not produce" >&2; exit 1; }
    done <<< "$plain"
    echo "files extracted, not loaded, for ${dir#"$DEST"}: $(echo "$plain" | tr '\n' ' ')"
  fi
  extras="$(cd "$dir" && find . -maxdepth 1 -type f ! -name 'libjllama.*' ! -name 'jllama.dll' ! -name '*.metal' \
    ! -name jllama-extras.txt ! -name jllama-files.txt ! -name jllama-build.txt | sed 's|^\./||' | sort \
    | awk -v plain="$plain" 'BEGIN { n = split(plain, a, "\n"); for (i = 1; i <= n; i++) skip[a[i]] = 1 } !($0 in skip)')"
  if [ -n "$extras" ]; then
    printf '%s\n' "$extras" > "$dir/jllama-extras.txt"
    echo "extras for ${dir#"$DEST"}: $(echo "$extras" | tr '\n' ' ')"
  fi
done

echo "Merged native tree:"
find "$DEST" -type f | sort | sed 's|^|  |'
