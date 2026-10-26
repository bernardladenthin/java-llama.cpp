#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Post-release check 15 (docs/post-release-check.md): the embedded WebUI of the RELEASED library. A
# local cmake build compiles the empty-asset stub, so "the WebUI is served" is a property of the
# published jar and of no build a developer makes: the stub answers GET / with 404, the release
# with the gzip-compressed app shell. Also /props, and a chat completion whose system_fingerprint
# names the build the natives jars declare (jllama-build.txt `llama.cpp`).
#
# Usage: post-release-webui.sh <set-dir> <model-path> [port]   (a smoke set, e.g. from post-release-consumer.sh)
set -euo pipefail

SET_DIR="${1:?usage: post-release-webui.sh <set-dir> <model-path> [port]}"
MODEL="${2:?usage: post-release-webui.sh <set-dir> <model-path> [port]}"
PORT="${3:-18090}"
BASE="http://127.0.0.1:$PORT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=smoke-lib.sh
. "$SCRIPT_DIR/smoke-lib.sh"

fail() {
    echo "::error::$*" >&2
    exit 1
}

CP="$(smoke_classpath "$SET_DIR")" || exit 1
java -cp "$CP" "$SMOKE_MAIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --chat-template chatml \
    > webui-out.log 2> webui-err.log &
PID=$!
trap 'kill "$PID" 2> /dev/null || true' EXIT
for _ in $(seq 1 100); do
    kill -0 "$PID" 2> /dev/null || { cat webui-out.log webui-err.log; fail "the server exited before /health answered"; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" || true)" = "200" ] && break
    sleep 3
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = "200" ] || fail "/health never answered 200"

curl -s -D webui-headers.txt -o webui-index.gz -H 'Accept-Encoding: gzip' "$BASE/" || fail "GET / failed"
status="$(head -1 webui-headers.txt | tr -d '\r')"
grep -q ' 200' <<< "$status" || fail "GET / answered '$status' -- the empty-asset stub answers 404, a release must serve the WebUI"
grep -qi '^content-type: text/html' webui-headers.txt || fail "GET / is not text/html: $(grep -i '^content-type' webui-headers.txt)"
grep -qi '^content-encoding: gzip' webui-headers.txt || fail "GET / is not gzip-encoded"
python3 - webui-index.gz << 'EOF' || fail "the WebUI body is not the app shell"
import gzip, sys
html = gzip.decompress(open(sys.argv[1], "rb").read()).decode("utf-8", "replace").lower()
assert "<!doctype html" in html and "<script" in html, html[:300]
print(f"GET / OK: {len(html)} bytes of HTML (gzip)")
EOF
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/props")" = "200" ] || fail "/props did not answer 200"
echo "/props OK"

build="$(unzip -p "$(find "$SET_DIR" -maxdepth 1 -name 'llama-*-cpu-*.jar' -o -maxdepth 1 -name 'llama-*-metal-*.jar' | head -1)" \
    '*/jllama-build.txt' | tr -d '\r' | sed -n 's/^llama\.cpp=//p' | head -1)"
response="$(curl -s --fail -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0}')" || fail "chat completion failed"
BUILD="$build" python3 -c '
import json, os, sys
r = json.loads(sys.stdin.read())
assert r["choices"][0]["message"] is not None, r
fp = r.get("system_fingerprint", "")
build = os.environ["BUILD"]
assert build and build in fp, f"system_fingerprint {fp!r} does not name {build}"
print("chat completion OK, system_fingerprint", fp)
' <<< "$response" || fail "unexpected chat completion: $response"
echo "WebUI check PASSED"
