#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# RPC smoke test over two JVMs, on the real release asset:
#
#   JVM A  java -cp <fatjar> net.ladenthin.llama.RpcServer   serves this runner's devices
#   JVM B  java -jar <fatjar> -m <model> --rpc 127.0.0.1:<A>  the default NativeServer, which
#                                                              offloads its layers to A
#
# and checks that B answers a chat completion, that its load log shows a model buffer on A's
# endpoint (the layers really went over RPC, not silently to the CPU), and that A accepted a
# client. A third launch names a server nobody runs and must fail with a message naming it and a
# normal exit -- not a SIGABRT, which is what ggml-rpc did before patches/0015.
#
# Usage: smoke-rpc-fatjar.sh <jar-dir> <jar-glob> <model-path>
# Output lands in rpc-server.log, rpc-client-out.log, rpc-client-err.log, rpc-unreachable.log
# (uploaded by the CI job on failure).
set -euo pipefail

JAR_DIR="${1:?usage: smoke-rpc-fatjar.sh <jar-dir> <jar-glob> <model-path>}"
JAR_GLOB="${2:?usage: smoke-rpc-fatjar.sh <jar-dir> <jar-glob> <model-path>}"
MODEL="${3:?usage: smoke-rpc-fatjar.sh <jar-dir> <jar-glob> <model-path>}"
RPC_PORT="${RPC_PORT:-50152}"
HTTP_PORT="${HTTP_PORT:-18181}"
UNUSED_PORT="${UNUSED_PORT:-50153}"

fail() {
    echo "::error::$*" >&2
    for f in rpc-server.log rpc-client-out.log rpc-client-err.log rpc-unreachable.log; do
        [ -f "$f" ] && { echo "--- $f (tail) ---"; tail -60 "$f"; }
    done
    exit 1
}

mapfile -t JARS < <(find "$JAR_DIR" -maxdepth 1 -name "$JAR_GLOB" | sort)
[ "${#JARS[@]}" -eq 1 ] || fail "expected exactly 1 jar matching $JAR_GLOB in $JAR_DIR, got ${#JARS[@]}: ${JARS[*]:-none}"
JAR="${JARS[0]}"
[ -f "$MODEL" ] || fail "model file missing: $MODEL"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2> /dev/null || true
    done
}
trap cleanup EXIT

# --- JVM A: the RPC server ----------------------------------------------------------------------
java -cp "$JAR" net.ladenthin.llama.RpcServer --port "$RPC_PORT" --threads 2 > rpc-server.log 2>&1 &
PIDS+=($!)
SERVER_PID=$!
for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2> /dev/null || fail "RpcServer exited before listening"
    grep -q "RpcServer listening on 127.0.0.1:$RPC_PORT" rpc-server.log && break
    sleep 1
done
grep -q "RpcServer listening on 127.0.0.1:$RPC_PORT" rpc-server.log || fail "RpcServer never reported listening"
echo "RPC server up: $(grep 'RpcServer listening' rpc-server.log)"

# --- JVM B: the model, offloaded over RPC --------------------------------------------------------
java -jar "$JAR" -m "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" --chat-template chatml \
    --rpc "127.0.0.1:$RPC_PORT" -ngl 99 -lv 4 > rpc-client-out.log 2> rpc-client-err.log &
PIDS+=($!)
CLIENT_PID=$!
CODE=""
for _ in $(seq 1 100); do
    kill -0 "$CLIENT_PID" 2> /dev/null || fail "the RPC client server exited before becoming healthy"
    CODE="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$HTTP_PORT/health" || true)"
    [ "$CODE" = "200" ] && break
    sleep 3
done
[ "$CODE" = "200" ] || fail "/health never returned 200 (last code: ${CODE:-none})"

RESPONSE="$(curl -sS --fail -X POST "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0}')" \
    || fail "chat completion over RPC failed"
echo "$RESPONSE" | python3 -c '
import json, sys
message = json.load(sys.stdin)["choices"][0]["message"]
assert message is not None, "choices[0].message missing"
print("chat completion over RPC OK:", json.dumps(message)[:200])
' || fail "malformed chat completion response: $RESPONSE"

grep -h "model buffer size" rpc-client-out.log rpc-client-err.log | grep -q "127.0.0.1:$RPC_PORT" \
    || fail "no model buffer on the RPC server in the load log -- the layers did not go over RPC"
grep -q "Accepted client connection" rpc-server.log || fail "the RPC server never accepted a client"
echo "layers offloaded over RPC: $(grep -h 'model buffer size' rpc-client-out.log rpc-client-err.log | grep "127.0.0.1:$RPC_PORT" | head -1)"

kill "$CLIENT_PID" 2> /dev/null || true
wait "$CLIENT_PID" 2> /dev/null || true

# --- an unreachable server fails the start cleanly -----------------------------------------------
set +e
timeout 120 java -jar "$JAR" -m "$MODEL" --host 127.0.0.1 --port "$((HTTP_PORT + 1))" \
    --rpc "127.0.0.1:$UNUSED_PORT" > rpc-unreachable.log 2>&1
status=$?
set -e
[ "$status" -ne 0 ] || fail "a server naming an unreachable RPC endpoint started anyway"
[ "$status" -ne 124 ] || fail "a server naming an unreachable RPC endpoint hung instead of failing"
# 134 = SIGABRT: the GGML_ABORT patches/0015 removed from the registration path
[ "$status" -ne 134 ] || fail "an unreachable RPC endpoint aborted the JVM (exit 134)"
grep -q "127.0.0.1:$UNUSED_PORT" rpc-unreachable.log || fail "the failure does not name the unreachable endpoint"
echo "unreachable endpoint rejected (exit $status): $(grep -m1 "127.0.0.1:$UNUSED_PORT" rpc-unreachable.log)"

echo "RPC smoke test PASSED"
