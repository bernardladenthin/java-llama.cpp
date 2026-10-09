#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Smoke test for an all-backends server fat jar on a GPU-less runner, in two rounds.
#
# Round 1 (plain HTTP): `java -jar` must start the embedded server — with every GPU backend
# failing its load cleanly and the loader falling back to the CPU backend — then answer GET
# /health with 200 and a POST /v1/chat/completions with a valid choice. This exercises backend
# probing, per-backend extraction, and the fallback chain end-to-end through a real fat-jar launch.
#
# Round 2 (HTTPS, both directions): the library links BoringSSL statically (llama/CMakeLists.txt,
# "HTTPS"), so out of the release asset two things must work on every platform -- the embedded
# server behind its own TLS key and certificate (--ssl-key-file / --ssl-cert-file), and a model
# download from an https:// URL verified against the OS certificate store (crypt32 on Windows,
# Security on macOS, /etc/ssl on Linux). Neither held before 5.2.0: the Linux jars had no SSL at
# all (an https:// download threw "HTTPS is not supported"), and macOS depended on the runner's
# Homebrew OpenSSL. The download is the 1 MB stories260K.gguf of .github/models.csv, into a cache
# directory of this run (LLAMA_CACHE), so round 2 also proves the HTTPS client path with a real
# certificate chain, which no local check can.
#
# Usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]
# Server output is written to server-out.log / server-err.log (round 1) and server-tls-out.log /
# server-tls-err.log (round 2) in the working dir (uploaded by the CI job on failure).
set -euo pipefail

JAR_DIR="${1:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
JAR_GLOB="${2:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
MODEL="${3:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
PORT="${4:-18080}"
TLS_PORT=$((PORT + 1))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTTPS_MODEL_NAME="stories260K.gguf"

fail() {
    echo "::error::$*" >&2
    exit 1
}

mapfile -t JARS < <(find "$JAR_DIR" -maxdepth 1 -name "$JAR_GLOB" | sort)
[ "${#JARS[@]}" -eq 1 ] || fail "expected exactly 1 jar matching $JAR_GLOB in $JAR_DIR, got ${#JARS[@]}: ${JARS[*]:-none}"
JAR="${JARS[0]}"
[ -f "$MODEL" ] || fail "model file missing: $MODEL"
echo "smoke jar: $JAR"

HTTPS_MODEL_URL="$(grep -E "^${HTTPS_MODEL_NAME//./\\.}," "$SCRIPT_DIR/models.csv" | cut -d, -f2- || true)"
[ -n "$HTTPS_MODEL_URL" ] || fail "$HTTPS_MODEL_NAME has no row in $SCRIPT_DIR/models.csv (round 2 downloads it over HTTPS)"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2> /dev/null || true; }
trap cleanup EXIT

# Poll /health until 200 (model loaded); 100 x 3 s = 5 min budget. An early server exit (e.g. an
# UnsatisfiedLinkError the fallback chain failed to absorb) fails fast. $1 is the base URL, $2/$3
# the server's stdout/stderr logs.
wait_healthy() {
    local base="$1" out="$2" err="$3" code=""
    for _ in $(seq 1 100); do
        if ! kill -0 "$SERVER_PID" 2> /dev/null; then
            echo "--- $out ---" && cat "$out"
            echo "--- $err ---" && cat "$err"
            fail "server process exited before becoming healthy ($base)"
        fi
        code="$(curl -sk -o /dev/null -w '%{http_code}' "$base/health" || true)"
        [ "$code" = "200" ] && return 0
        sleep 3
    done
    echo "--- $out (tail) ---" && tail -50 "$out"
    echo "--- $err (tail) ---" && tail -50 "$err"
    fail "$base/health never returned 200 (last code: ${code:-none})"
}

# A chat completion with one valid choice, over $1 (the base URL). `-k` accepts the self-signed
# certificate of round 2 and is a no-op over plain HTTP.
chat_completion() {
    local base="$1" response
    response="$(curl -sSk --fail -X POST "$base/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}')" \
        || fail "chat completion request failed ($base)"
    echo "$response" | python3 -c '
import json, sys
response = json.load(sys.stdin)
message = response["choices"][0]["message"]
assert message is not None, "choices[0].message missing"
print("chat completion OK:", json.dumps(message)[:200])
' || fail "malformed chat completion response: $response"
}

# ---- Round 1: plain HTTP, the cached model -------------------------------------------------------
java -jar "$JAR" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --chat-template chatml \
    > server-out.log 2> server-err.log &
SERVER_PID=$!
wait_healthy "http://127.0.0.1:$PORT" server-out.log server-err.log
echo "health OK"
chat_completion "http://127.0.0.1:$PORT"

# The loader must have reported its backend decision (normally the CPU fallback on a
# GPU-less runner; a GPU backend whose runtime happens to be installed may load and
# find no device, which is benign) — this pins that the smoke ran the backend probing.
grep -hE '\[jllama\] using native backend' server-out.log server-err.log \
    || fail "no backend-selection log line found — the loader did not report a backend"
kill "$SERVER_PID" 2> /dev/null || true
wait "$SERVER_PID" 2> /dev/null || true
SERVER_PID=""

# ---- Round 2: HTTPS server + https:// model download ---------------------------------------------
command -v openssl > /dev/null || fail "the openssl CLI is needed to mint the TLS test certificate"
openssl req -x509 -newkey rsa:2048 -nodes -keyout tls-key.pem -out tls-cert.pem -days 2 \
    -subj "/CN=127.0.0.1" > /dev/null 2>&1 || fail "could not create the self-signed TLS certificate"
# -m names where the download lands; with --model-url alone the server starts in ROUTER mode
# (upstream decides on model.path / hf_repo / docker_repo before the URL is resolved).
export LLAMA_CACHE="$PWD/llama-cache"
rm -rf "$LLAMA_CACHE" && mkdir -p "$LLAMA_CACHE"
java -jar "$JAR" -m "$LLAMA_CACHE/$HTTPS_MODEL_NAME" --model-url "$HTTPS_MODEL_URL" \
    --host 127.0.0.1 --port "$TLS_PORT" --chat-template chatml \
    --ssl-key-file tls-key.pem --ssl-cert-file tls-cert.pem \
    > server-tls-out.log 2> server-tls-err.log &
SERVER_PID=$!
wait_healthy "https://127.0.0.1:$TLS_PORT" server-tls-out.log server-tls-err.log
echo "HTTPS health OK"
chat_completion "https://127.0.0.1:$TLS_PORT"
# The port must really speak TLS: a plain-HTTP request to it is refused (curl exits non-zero).
if curl -s -o /dev/null --max-time 10 "http://127.0.0.1:$TLS_PORT/health"; then
    fail "the TLS port answered a plain-HTTP request — the server did not use the certificate"
fi
echo "plain HTTP on the TLS port refused: OK"
# The model came over HTTPS into this run's cache, not from the runner's model directory.
find "$LLAMA_CACHE" -type f -name '*.gguf' | grep -q . \
    || fail "no .gguf under $LLAMA_CACHE — the https:// download did not happen"
echo "https:// model download OK: $(find "$LLAMA_CACHE" -type f -name '*.gguf' -exec basename {} \; | head -1)"

echo "smoke test PASSED"
