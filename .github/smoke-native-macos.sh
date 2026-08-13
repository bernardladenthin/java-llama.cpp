#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# macOS post-`package` smoke: verifies the libjllama.dylib that is actually INSIDE the packaged jar
# — its code signature, and that a JVM can load it and cross the JNI boundary.
#
# Why this exists (the gap it closes): the three macOS Java test jobs each run against the dylib
# THEIR OWN build job produced. Nothing in the pipeline ever loaded the one that goes into the
# published jar. So when the `*-libraries` artifact glob merged three different macOS dylibs onto
# one path and produced a byte-level hybrid, the result — a library whose ad-hoc linker signature no
# longer matched its own __TEXT pages, which macOS SIGKILLs on load — shipped in 5.0.6 and several
# 5.0.7 snapshots with an all-green pipeline. Linux and Windows already had the equivalent gate
# (`smoke-fatjar-linux` / `smoke-fatjar-windows`, downstream of `package`); macOS had none.
#
# This is the macOS member of the cross-repo "no release asset is attached that CI has not run"
# convention (workspace/policies/fat-jar-release-assets.md). It is NOT the shared
# smoke-fatjar-cli.sh that BitcoinAddressFinder and srcmorph run: this jar's Main-Class is a server
# that never exits, and the assertion that matters here is native-library loadability, not a CLI
# exit code. Same job shape, repo-specific assertions.
#
# Deliberately model-free: no GGUF, no cache restore, no network — it runs in ~1 min. A full
# model-backed macOS server smoke would be strictly more, but the failure class that actually
# shipped is caught here, so this is the version that is cheap enough to always run.
#
# Usage: smoke-native-macos.sh <jar-dir> <jar-glob>
#   <jar-dir>   directory to search for the jar (recursively)
#   <jar-glob>  filename glob; must match EXACTLY ONE jar

set -euo pipefail

JAR_DIR="${1:?usage: smoke-native-macos.sh <jar-dir> <jar-glob>}"
JAR_GLOB="${2:?usage: smoke-native-macos.sh <jar-dir> <jar-glob>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() {
    echo "::error::$*" >&2
    exit 1
}

[ -d "$JAR_DIR" ] || fail "jar directory '$JAR_DIR' does not exist"

jars=()
while IFS= read -r j; do jars+=("$j"); done < <(find "$JAR_DIR" -type f -name "$JAR_GLOB" | sort)
[ "${#jars[@]}" -eq 1 ] \
    || fail "expected exactly 1 jar matching '$JAR_GLOB' under '$JAR_DIR', got ${#jars[@]}: ${jars[*]:-none}"
JAR="$(cd "$(dirname "${jars[0]}")" && pwd)/$(basename "${jars[0]}")"
echo "smoke jar: $JAR"

DYLIB_ENTRY="net/ladenthin/llama/Mac/aarch64/libjllama.dylib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

unzip -o -q "$JAR" "$DYLIB_ENTRY" -d "$WORK" \
    || fail "the jar does not contain $DYLIB_ENTRY — the macOS natives never reached the package job"
DYLIB="$WORK/$DYLIB_ENTRY"
echo "extracted: $(cd "$(dirname "$DYLIB")" && pwd)/$(basename "$DYLIB") ($(wc -c < "$DYLIB") bytes)"

# 1) Signature vs. content. `--strict` re-hashes the code pages and compares them against the
#    signature's stored hashes, so a dylib assembled from two different builds fails here with the
#    exact page mismatch — the direct check for the shipped corruption. An ad-hoc signature (what
#    the linker emits on arm64) is expected and fine; only a MISMATCH is a failure.
echo "== codesign --verify --strict =="
codesign --verify --strict --verbose=2 "$DYLIB" \
    || fail "code signature does not match the dylib's own pages — the packaged library is corrupt (macOS would SIGKILL any process that loads it)"

# 2) The JVM must actually be able to map it and call through JNI. This is what a consumer does,
#    and it is the only check that covers load-time failures the signature check cannot see
#    (missing dependent library, wrong architecture, unresolved JNI_OnLoad class lookup).
echo "== JVM load + JNI round-trip =="
java -cp "$JAR" "$SCRIPT_DIR/smoke/NativeLoadSmoke.java" \
    || fail "the packaged native library did not load in a JVM"

echo "smoke test PASSED"
