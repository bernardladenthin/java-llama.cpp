#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# Smoke test for an all-backends server fat jar on a GPU-less runner:
# `java -jar` must start the embedded server — with every GPU backend failing its
# load cleanly and the loader falling back to the CPU backend — then answer GET
# /health with 200 and a POST /v1/chat/completions with a valid choice. This
# exercises backend probing, per-backend extraction, and the fallback chain
# end-to-end through a real fat-jar launch.
#
# Usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]
# Server output is written to server-out.log / server-err.log in the working dir
# (uploaded by the CI job on failure).
set -euo pipefail

JAR_DIR="${1:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
JAR_GLOB="${2:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
MODEL="${3:?usage: smoke-test-fatjar.sh <jar-dir> <jar-glob> <model-path> [port]}"
PORT="${4:-18080}"

fail() {
    echo "::error::$*" >&2
    exit 1
}

mapfile -t JARS < <(find "$JAR_DIR" -maxdepth 1 -name "$JAR_GLOB" | sort)
[ "${#JARS[@]}" -eq 1 ] || fail "expected exactly 1 jar matching $JAR_GLOB in $JAR_DIR, got ${#JARS[@]}: ${JARS[*]:-none}"
JAR="${JARS[0]}"
[ -f "$MODEL" ] || fail "model file missing: $MODEL"
echo "smoke jar: $JAR"

java -jar "$JAR" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --chat-template chatml \
    > server-out.log 2> server-err.log &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2> /dev/null || true; }
trap cleanup EXIT

# Poll /health until 200 (model loaded); 100 x 3 s = 5 min budget. An early server
# exit (e.g. an UnsatisfiedLinkError the fallback chain failed to absorb) fails fast.
CODE=""
for _ in $(seq 1 100); do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
        echo "--- server-out.log ---" && cat server-out.log
        echo "--- server-err.log ---" && cat server-err.log
        fail "server process exited before becoming healthy"
    fi
    CODE="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)"
    [ "$CODE" = "200" ] && break
    sleep 3
done
if [ "$CODE" != "200" ]; then
    echo "--- server-out.log (tail) ---" && tail -50 server-out.log
    echo "--- server-err.log (tail) ---" && tail -50 server-err.log
    fail "/health never returned 200 (last code: ${CODE:-none})"
fi
echo "health OK"

RESPONSE="$(curl -sS --fail -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}')" \
    || fail "chat completion request failed"
echo "$RESPONSE" | python3 -c '
import json, sys
response = json.load(sys.stdin)
message = response["choices"][0]["message"]
assert message is not None, "choices[0].message missing"
print("chat completion OK:", json.dumps(message)[:200])
' || fail "malformed chat completion response: $RESPONSE"

# The loader must have reported its backend decision (normally the CPU fallback on a
# GPU-less runner; a GPU backend whose runtime happens to be installed may load and
# find no device, which is benign) — this pins that the smoke ran the backend probing.
grep -hE '\[jllama\] using native backend' server-out.log server-err.log \
    || fail "no backend-selection log line found — the loader did not report a backend"

echo "smoke test PASSED"
