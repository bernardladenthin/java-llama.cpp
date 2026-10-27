#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run llama-bench with exactly ONE CPU backend module, so a module the build machine would never pick
# (alderlake on an AVX-512 host, say) can be measured there. docs/benchmarks/moe-mxfp4-cpu-alderlake.md
# is the measurement this was written for, and has the build recipe.
#
#   BUILD=<llama.cpp build dir> bench.sh <module> <out.jsonl> <llama-bench args...>
#       appends one JSON line per test to <out.jsonl>, tagged with "variant":"<module>"
#   BUILD=<dir> PROFILE=<out-prefix> [PERF_DELAY=<ms>] bench.sh <module> - <llama-bench args...>
#       one run under `perf record -e cpu-clock` instead; writes <out-prefix>.data and a flat
#       per-symbol report <out-prefix>.sym.txt. PERF_DELAY skips the model load (repacking a 12 GB
#       model alone is seconds of samples).
#
# The build must be a GGML_BACKEND_DL one with its libggml-cpu-*.so moved from <BUILD>/bin to
# <BUILD>/cpu-modules: ggml searches the executable's directory for CPU modules and would load the
# best one from there next to the one named here. GGML_BACKEND_PATH then loads the single module.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '6,19p' "$0"
    exit 2
fi
: "${BUILD:?set BUILD to the llama.cpp build directory}"
module=$1
out=$2
shift 2

lib="$BUILD/cpu-modules/libggml-cpu-$module.so"
if [ ! -f "$lib" ]; then
    echo "bench.sh: no module $lib" >&2
    exit 1
fi
if ls "$BUILD"/bin/libggml-cpu-*.so > /dev/null 2>&1; then
    echo "bench.sh: $BUILD/bin still holds CPU modules; move them to $BUILD/cpu-modules first" >&2
    exit 1
fi
export GGML_BACKEND_PATH="$lib"

if [ -n "${PROFILE:-}" ]; then
    # Ubuntu's /usr/bin/perf is a wrapper that refuses a kernel it has no package for (any VM or
    # container kernel); the versioned binary behind it works for user-space sampling regardless
    perf=
    for p in "$(command -v perf || true)" /usr/lib/linux-tools/*/perf; do
        if [ -x "$p" ] && "$p" --version > /dev/null 2>&1; then
            perf=$p
            break
        fi
    done
    if [ -z "$perf" ]; then
        echo "bench.sh: no working perf (Ubuntu: apt-get install linux-tools-generic)" >&2
        exit 1
    fi
    "$perf" record -q -D "${PERF_DELAY:-0}" -e cpu-clock -F 1999 -o "$PROFILE.data" -- \
        "$BUILD/bin/llama-bench" -r 1 --no-warmup "$@"
    "$perf" report -q -i "$PROFILE.data" --no-children --sort sym --percent-limit 0.3 2> /dev/null \
        | sed 's/ \+$//' > "$PROFILE.sym.txt"
    head -25 "$PROFILE.sym.txt"
    exit 0
fi

"$BUILD/bin/llama-bench" -o jsonl "$@" | sed "s/^{/{\"variant\":\"$module\",/" >> "$out"
