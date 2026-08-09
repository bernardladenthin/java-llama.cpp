#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

# The core project (CMakeLists.txt + src/) lives in the `llama/` module of the Maven
# reactor. Re-root here once — every native build delegates to this script (incl. the
# build_cuda_linux.sh / build_opencl_*.sh wrappers that `exec .github/build.sh`) — so
# cmake configures the module regardless of the caller's CWD. Anchored to this script's
# own location (via BASH_SOURCE), so it works from the repo root and inside the dockcross
# container whose workdir is the mounted repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/../llama" || exit 1

mkdir -p build

# Build parallelism. Defaults to all cores; RAM-limited CI runners (notably GitHub's
# ~7 GB macOS arm64) export BUILD_JOBS lower (e.g. 2) so the large httplib.cpp + the 134
# llama.cpp model TUs do not exhaust memory and get the runner OOM-killed mid-compile
# (which surfaces as a SIGTERM / "runner received a shutdown signal", not a clean timeout).
JOBS="${BUILD_JOBS:-}"
if [ -z "$JOBS" ]; then
  JOBS="$( { command -v nproc >/dev/null 2>&1 && nproc; } || sysctl -n hw.ncpu 2>/dev/null || echo 4 )"
fi

# Fetch sccache when caching is requested but the runner/container doesn't ship it — the
# dockcross cross-compile containers (manylinux/Android) and Linux hosts have no sccache,
# while macOS installs it via brew in the workflow. Best-effort and inert-safe: any failure
# leaves sccache absent, so the build just proceeds uncached. A static musl release exists for
# both Linux arches we build on: x86_64 (the dockcross cross-compile hosts) and aarch64 (the
# native ubuntu-24.04-arm runner); any other arch leaves SCCACHE_DL_ARCH empty and is skipped.
#
# SCCACHE_DL_VERSION is overridable per-job, so a container that crashes one sccache build can
# try another without editing this script (the in-container panic that stalled phase 2 was on
# v0.8.2; v0.16.0 is the latest release and the default). A wrong/unavailable version just fails
# the `curl -f` and falls back to an uncached build, so bumping it can never red a build.
SCCACHE_DL_VERSION="${SCCACHE_DL_VERSION:-0.16.0}"
case "$(uname -m)" in
  x86_64)        SCCACHE_DL_ARCH="x86_64" ;;
  aarch64|arm64) SCCACHE_DL_ARCH="aarch64" ;;
  *)             SCCACHE_DL_ARCH="" ;;
esac
if [ "${USE_CACHE:-true}" = "true" ] && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ] \
   && ! command -v sccache >/dev/null 2>&1 \
   && [ "$(uname -s)" = "Linux" ] && [ -n "$SCCACHE_DL_ARCH" ]; then
  SCCACHE_REL="sccache-v${SCCACHE_DL_VERSION}-${SCCACHE_DL_ARCH}-unknown-linux-musl"
  echo "build.sh: fetching ${SCCACHE_REL} (no sccache on PATH)..."
  if curl -fsSL --proto =https --proto-redir =https \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_DL_VERSION}/${SCCACHE_REL}.tar.gz" \
        -o /tmp/sccache.tgz && tar -xzf /tmp/sccache.tgz -C /tmp; then
    export PATH="/tmp/${SCCACHE_REL}:$PATH"
    echo "build.sh: sccache -> $(command -v sccache || echo 'still missing')"
  else
    echo "build.sh: sccache fetch failed; continuing without cache"
  fi
fi

# Health-check before trusting sccache as the compiler launcher. Because sccache *is* the
# launcher (cmake runs `sccache <compiler> ...` for every TU), a present-but-crashing sccache
# fails every compile and reds the whole build — exactly the in-container panic that stalled
# phase 2 (the static-musl binary panicked while wrapping the cross-compiler, failing ggml.c.o).
# The probe runs the real compiler through sccache on a trivial TU; only if that succeeds is the
# launcher enabled. On any failure it logs the captured output (the Rust panic backtrace, plus
# the detached server's SCCACHE_ERROR_LOG when a job sets one) and the build runs WITHOUT the
# cache — a clean, uncached -O3 build that still goes green. This closes the gap the old
# absent-only guard left: it handled sccache *missing*, not sccache *crashing*.
sccache_can_wrap_compiler() {
  probe_cc="${CC:-}"
  if [ -z "$probe_cc" ]; then
    for c in cc gcc clang; do
      if command -v "$c" >/dev/null 2>&1; then probe_cc="$c"; break; fi
    done
  fi
  if [ -z "$probe_cc" ]; then
    echo "build.sh: sccache probe: no C compiler on PATH to probe; building uncached"
    return 1
  fi
  probe_dir="$(mktemp -d 2>/dev/null || echo "/tmp/sccache-probe.$$")"
  mkdir -p "$probe_dir" || return 1
  printf 'int main(void){return 0;}\n' > "$probe_dir/probe.c"
  probe_out="$(sccache "$probe_cc" -c "$probe_dir/probe.c" -o "$probe_dir/probe.o" 2>&1)"
  probe_rc=$?
  rm -rf "$probe_dir"
  if [ "$probe_rc" -ne 0 ]; then
    echo "build.sh: sccache probe FAILED (rc=${probe_rc}) wrapping '${probe_cc}' — building WITHOUT cache."
    [ -n "$probe_out" ] && printf '%s\n' "$probe_out" | sed 's/^/build.sh:   sccache-probe| /'
    if [ -n "${SCCACHE_ERROR_LOG:-}" ] && [ -f "${SCCACHE_ERROR_LOG}" ]; then
      echo "build.sh:   --- detached server log (${SCCACHE_ERROR_LOG}) ---"
      sed 's/^/build.sh:   sccache-srv| /' "${SCCACHE_ERROR_LOG}" 2>/dev/null || true
    fi
    return 1
  fi
  echo "build.sh: sccache probe OK (wrapped '${probe_cc}')"
  return 0
}

# Optional shared compiler cache: sccache fronting Depot Cache (WebDAV). Enabled only when
# USE_CACHE is true AND sccache + a cache token are present AND the probe confirms sccache can
# wrap the compiler — so it stays inert before the DEPOT_TOKEN secret is configured, on fork PRs
# (secrets hidden), and when sccache would crash; all of those just compile normally. sccache is
# content-addressed, so a cache hit is bit-identical to a fresh -O3 compile (release-safe), and
# it degrades to direct compilation if the cache is unreachable.
LAUNCH=""
if [ "${USE_CACHE:-true}" = "true" ] && command -v sccache >/dev/null 2>&1 \
   && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ] \
   && sccache_can_wrap_compiler; then
  LAUNCH="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
  # CUDA builds: also wrap nvcc so the per-arch .cu device passes are cached too — not just
  # the gcc host TUs. Those per-architecture device-pass objects are the dominant cost of the
  # full-arch CUDA job, and sccache does support nvcc as a compiler. Scoped to CUDA builds
  # (GGML_CUDA in the cmake args): CMAKE_CUDA_COMPILER_LAUNCHER is inert when CUDA is not an
  # enabled language, but keeping it scoped leaves the CPU/Android jobs' configure output clean.
  # If sccache cannot wrap nvcc it runs it directly (uncached); and the mid-build retry below
  # also catches an sccache "Compiler not supported" failure and rebuilds without the launcher,
  # so an nvcc-hostile sccache can never red the build.
  case " $* " in
    *" -DGGML_CUDA=1 "* | *" -DGGML_CUDA=ON "* | *" -DGGML_CUDA=on "*)
      LAUNCH="$LAUNCH -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"
      echo "build.sh: sccache will also wrap nvcc (CUDA build detected)" ;;
  esac
  echo "build.sh: sccache ON (endpoint=${SCCACHE_WEBDAV_ENDPOINT:-default}), building with -j${JOBS}"
else
  echo "build.sh: sccache OFF, building with -j${JOBS}"
fi

cmake -Bbuild $LAUNCH $@ || exit 1

# Build. The pre-build probe only proves the cache was reachable at one instant; it cannot
# foresee a cache outage that strikes *during* the build. When sccache is the launcher and its
# backend fails mid-build — e.g. an intermittent Depot 403 on the server-startup .sccache_check,
# or the tokenless 403 a fork PR hits because secrets are withheld — sccache makes every TU fatal
# and reds the whole build. sccache exposes no "ignore backend errors" switch for that startup
# check, so recover by retrying the build once WITHOUT the launcher: a from-scratch uncached -O3
# build is content-identical and release-safe, so the cache can never red the build. The retry is
# gated on the failure output actually showing an sccache cache error, so a genuine compile error
# still fails fast (and is reported) instead of triggering a wasteful uncached rebuild. The
# "Compiler not supported" signature additionally covers the CUDA case: if wrapping nvcc breaks
# (sccache declining/erroring on the nvcc driver), the retry rebuilds the full-arch CUDA job
# without any launcher rather than redding it. "Compiler killed by signal" covers a second CUDA
# failure mode seen with CUDA 13.3's nvcc: sccache's nvcc wrapping loses an intermediate .ptx file
# for a "-virtual" architecture target (e.g. 75-virtual), so fatbinary aborts with "Could not open
# input file '*.ptx'" and sccache reports the underlying nvcc invocation as killed — an sccache/nvcc
# incompatibility, not a real compile error, so it gets the same uncached-retry treatment.
build_log="$(mktemp 2>/dev/null || echo "/tmp/jllama-build.$$.log")"
cmake --build build --config Release -j"${JOBS}" 2>&1 | tee "$build_log"
build_rc=${PIPESTATUS[0]}
if [ "$build_rc" -ne 0 ]; then
  if [ -n "$LAUNCH" ] && grep -qiE 'sccache: error|Server startup failed|cache storage failed|Compiler not supported|Compiler killed by signal' "$build_log"; then
    echo "build.sh: build failed via an sccache cache error — retrying WITHOUT cache (clean reconfigure)."
    rm -f "$build_log"
    rm -rf build && mkdir -p build
    cmake -Bbuild $@ || exit 1
    cmake --build build --config Release -j"${JOBS}" || exit 1
    LAUNCH=""  # cache disabled for this run; skip the stats query below
  else
    rm -f "$build_log"
    exit 1
  fi
fi
rm -f "$build_log"

# Only query stats when sccache was actually used as the launcher; if the probe rejected a
# crashing sccache (or the mid-build retry disabled it), re-invoking it here would just repeat
# the crash output (harmless but noisy).
if [ -n "$LAUNCH" ] && command -v sccache >/dev/null 2>&1; then
  sccache_stats="$(sccache --show-stats 2>/dev/null || true)"
  printf '%s\n' "$sccache_stats"
  # KISS per-job cache summary in the GitHub Actions job summary (like upstream llama.cpp's
  # ccache-action table). Parse the text stats: the top-level "Compile requests" line is the
  # total and the top-level "Cache hits" line is the hits (the per-language "Cache hits (C/C++)"
  # line has "(" after the label, so the digit-anchored regex skips it). Only runs in CI
  # (GITHUB_STEP_SUMMARY set); local runs are untouched. Best-effort — skips silently if the two
  # numbers can't be parsed or there were no requests.
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ] && [ -n "$sccache_stats" ]; then
    sccache_req="$(printf '%s\n' "$sccache_stats" | awk '/^Compile requests[[:space:]]+[0-9]/{print $NF; exit}')"
    sccache_hits="$(printf '%s\n' "$sccache_stats" | awk '/^Cache hits[[:space:]]+[0-9]/{print $NF; exit}')"
    if [ -n "$sccache_req" ] && [ -n "$sccache_hits" ] && [ "$sccache_req" -gt 0 ] 2>/dev/null; then
      sccache_rate="$(awk "BEGIN{printf \"%.1f\", ($sccache_hits/$sccache_req)*100}")"
      {
        echo "### sccache statistics"
        echo ""
        echo "| Cache hits | Requests | Hit rate |"
        echo "|------------|----------|----------|"
        echo "| ${sccache_hits} | ${sccache_req} | ${sccache_rate}% |"
      } >> "$GITHUB_STEP_SUMMARY"
    fi
  fi
fi
