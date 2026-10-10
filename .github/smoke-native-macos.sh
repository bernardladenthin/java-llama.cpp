#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# macOS post-`package` smoke: verifies the libjllama.dylib that is actually INSIDE the published
# natives jar (metal-macos-aarch64, taken from the macOS smoke set) -- its code signature, and that
# a JVM can load it from the jars and cross the JNI boundary.
#
# Why this exists (the gap it closes): the three macOS Java test jobs each run against the dylib
# THEIR OWN build job produced. Nothing in the pipeline ever loaded the one that goes into the
# published jar. So when an artifact glob merged three different macOS dylibs onto
# one path and produced a byte-level hybrid, the result — a library whose ad-hoc linker signature no
# longer matched its own __TEXT pages, which macOS SIGKILLs on load — shipped in 5.0.6 and several
# 5.0.7 snapshots with an all-green pipeline. Linux and Windows already had the equivalent gate
# (the `smoke-natives` matrix, downstream of `package`); macOS had none.
#
# This is the macOS member of the cross-repo "no artifact ships that CI has not run" convention
# (workspace/policies/fat-jar-release-assets.md; this repository ships no fat jar, its artifacts are
# the natives jars, and every one of a platform is launched from a smoke set). It is NOT the shared
# smoke-fatjar-cli.sh that BitcoinAddressFinder and srcmorph run: the assertion that matters here
# is native-library loadability, not a CLI exit code.
#
# No cache restore and no model of the CI set: it runs in ~1 min. A full model-backed macOS server
# smoke would be strictly more, but the failure class that actually shipped is caught here, so this
# is the version that is cheap enough to always run. The one model it touches is the 1 MB
# stories260K.gguf of .github/models.csv, which step 3 downloads itself over HTTPS -- that download
# IS the check: the library links BoringSSL statically (llama/CMakeLists.txt, "HTTPS") and this
# dylib used to depend on the runner's Homebrew OpenSSL, which no check here could see because the
# runner has it. Step 3 proves the HTTPS client against the macOS certificate store and the
# embedded server behind its own TLS certificate, the same round the Linux and Windows rows of
# smoke-natives run.
#
# Usage: smoke-native-macos.sh <set-dir> [port]
# Server output of step 3 is written to server-tls-out.log / server-tls-err.log in the working dir.
#   <set-dir>   the macOS smoke set (.github/package-smoke-sets.sh): the classes jar, its
#               dependencies and the metal-macos-aarch64 natives jar

set -euo pipefail

SET_DIR="${1:?usage: smoke-native-macos.sh <set-dir> [port]}"
TLS_PORT="${2:-18081}"
HTTPS_MODEL_NAME="stories260K.gguf"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=smoke-lib.sh
. "$SCRIPT_DIR/smoke-lib.sh"

fail() {
    echo "::error::$*" >&2
    exit 1
}

CP="$(smoke_classpath "$SET_DIR")" || exit 1
jars=()
while IFS= read -r j; do jars+=("$j"); done < <(find "$SET_DIR" -maxdepth 1 -type f -name 'llama-*-metal-macos-aarch64.jar' | sort)
[ "${#jars[@]}" -eq 1 ] \
    || fail "expected exactly 1 metal-macos-aarch64 natives jar in '$SET_DIR', got ${#jars[@]}: ${jars[*]:-none}"
JAR="$(cd "$(dirname "${jars[0]}")" && pwd)/$(basename "${jars[0]}")"
echo "natives jar: $JAR"

DYLIB_ENTRY="net/ladenthin/llama/Mac/aarch64/metal/libjllama.dylib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

unzip -o -q "$JAR" "$DYLIB_ENTRY" -d "$WORK" \
    || fail "the natives jar does not contain $DYLIB_ENTRY — the macOS natives never reached the package job"
DYLIB="$WORK/$DYLIB_ENTRY"
echo "extracted: $(cd "$(dirname "$DYLIB")" && pwd)/$(basename "$DYLIB") ($(wc -c < "$DYLIB") bytes)"

# 0) The oldest macOS it loads on. CMAKE_OSX_DEPLOYMENT_TARGET (llama/CMakeLists.txt) pins it;
#    without that pin the linker takes the build host's own version, so a build job moved to a
#    newer runner image would ship a library newer macOS releases alone can load -- and nothing
#    else in the pipeline would notice, since every test runs on that same newer image.
MAX_MINOS="${JLLAMA_MAX_MINOS:-15.0}"
echo "== LC_BUILD_VERSION (minos must be <= $MAX_MINOS) =="
otool -l "$DYLIB" | grep -A4 LC_BUILD_VERSION \
    || fail "no LC_BUILD_VERSION load command in the dylib -- cannot tell which macOS it needs"
MINOS="$(otool -l "$DYLIB" | awk '/cmd LC_BUILD_VERSION/{f=1} f && $1=="minos"{print $2; exit}')"
[ -n "$MINOS" ] || fail "could not read minos from the dylib's LC_BUILD_VERSION"
awk -v have="$MINOS" -v max="$MAX_MINOS" 'BEGIN {
    split(have, h, "."); split(max, m, ".")
    for (i = 1; i <= 3; i++) { if (h[i] + 0 != m[i] + 0) exit (h[i] + 0 > m[i] + 0) }
    exit 0
}' || fail "the dylib needs macOS $MINOS, but macOS $MAX_MINOS is the supported floor (CMAKE_OSX_DEPLOYMENT_TARGET in llama/CMakeLists.txt)"
echo "minos $MINOS <= $MAX_MINOS: ok"

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
java -cp "$CP" "$SCRIPT_DIR/smoke/NativeLoadSmoke.java" \
    || fail "the packaged native library did not load in a JVM"

# 3) HTTPS in both directions out of the packaged jar: the server behind a self-signed certificate
#    (--ssl-key-file / --ssl-cert-file) serving a model it downloaded from an https:// URL, verified
#    against the macOS certificate store (Security.framework). -ngl 0: this is about TLS, not Metal.
echo "== HTTPS server + https:// model download =="
HTTPS_MODEL_URL="$(grep -E "^${HTTPS_MODEL_NAME//./\\.}," "$SCRIPT_DIR/models.csv" | cut -d, -f2- || true)"
[ -n "$HTTPS_MODEL_URL" ] || fail "$HTTPS_MODEL_NAME has no row in $SCRIPT_DIR/models.csv"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$WORK/tls-key.pem" -out "$WORK/tls-cert.pem" -days 2 \
    -subj "/CN=127.0.0.1" > /dev/null 2>&1 || fail "could not create the self-signed TLS certificate"
export LLAMA_CACHE="$WORK/llama-cache"
mkdir -p "$LLAMA_CACHE"
# -m names where the download lands; with --model-url alone the server starts in ROUTER mode.
java -cp "$CP" "$SMOKE_MAIN" -m "$LLAMA_CACHE/$HTTPS_MODEL_NAME" --model-url "$HTTPS_MODEL_URL" \
    --host 127.0.0.1 --port "$TLS_PORT" --chat-template chatml \
    -ngl 0 --ssl-key-file "$WORK/tls-key.pem" --ssl-cert-file "$WORK/tls-cert.pem" \
    > server-tls-out.log 2> server-tls-err.log &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2> /dev/null || true; rm -rf "$WORK"' EXIT
CODE=""
for _ in $(seq 1 100); do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
        echo "--- server-tls-out.log ---" && cat server-tls-out.log
        echo "--- server-tls-err.log ---" && cat server-tls-err.log
        fail "the TLS server exited before becoming healthy (download or load failed)"
    fi
    CODE="$(curl -sk -o /dev/null -w '%{http_code}' "https://127.0.0.1:$TLS_PORT/health" || true)"
    [ "$CODE" = "200" ] && break
    sleep 3
done
if [ "$CODE" != "200" ]; then
    echo "--- server-tls-err.log (tail) ---" && tail -50 server-tls-err.log
    fail "https://127.0.0.1:$TLS_PORT/health never returned 200 (last code: ${CODE:-none})"
fi
echo "HTTPS health OK"
if curl -s -o /dev/null --max-time 10 "http://127.0.0.1:$TLS_PORT/health"; then
    fail "the TLS port answered a plain-HTTP request — the server did not use the certificate"
fi
echo "plain HTTP on the TLS port refused: OK"
find "$LLAMA_CACHE" -type f -name '*.gguf' | grep -q . \
    || fail "no .gguf under $LLAMA_CACHE — the https:// download did not happen"
echo "https:// model download OK: $(find "$LLAMA_CACHE" -type f -name '*.gguf' -exec basename {} \; | head -1)"

echo "smoke test PASSED"
