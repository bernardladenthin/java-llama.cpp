#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Smoke test for the published llama-atmosphere-agent on the published core jars: the agent's thin
# jar with its own dependencies (what Maven Central resolves for `jbang net.ladenthin:llama-atmosphere-agent`)
# on one classpath with a smoke set of the core (.github/package-smoke-sets.sh: the classes jar, its
# dependencies, the CPU natives jar and every GPU module jar of this platform). That is the check
# that the two publications fit together -- the agent built against this very core, the natives
# reached through the loader -- which no test run from the source tree can see.
#
#   1. the agent carries no core: started without the core set, loading a model fails with
#      NoClassDefFoundError for net.ladenthin.llama.LlamaModel;
#   2. --help exits 0 and prints the usage;
#   3. a one-shot prompt with the model loaded in-process answers "2+2" with a 4;
#   4. a one-shot prompt that needs a reading tool (read_file) reads a marker file from
#      --workspace, so the whole tool loop runs through the shipped jars;
#   5. --web starts the browser front end (embedded Jetty + the Atmosphere console pages,
#      which the assembly unpacks from a jar it otherwise leaves out): without the token
#      the console is 401, the token link sets the session cookie, and the console page
#      is served with it;
#   6. --acp speaks the Agent Client Protocol on stdin/stdout the way an editor drives it
#      (smoke/agent_acp_smoke.py): handshake, a streamed answer, a read_file round, and a
#      clean exit when the editor hangs up.
#
# Usage: smoke-agent.sh <agent-dir> <set-dir> <model-path>
# <agent-dir> holds the agent's thin jar (llama-atmosphere-agent-<v>.jar) and its dependency jars
# without the core (mvn dependency:copy-dependencies -DexcludeGroupIds=net.ladenthin); <set-dir> is
# the core's smoke set. Output of each run is kept in agent-*.log in the working directory
# (uploaded by the CI job on failure).
set -euo pipefail

AGENT_DIR="${1:?usage: smoke-agent.sh <agent-dir> <set-dir> <model-path>}"
SET_DIR="${2:?usage: smoke-agent.sh <agent-dir> <set-dir> <model-path>}"
MODEL="${3:?usage: smoke-agent.sh <agent-dir> <set-dir> <model-path>}"
AGENT_MAIN="net.ladenthin.llama.atmosphere.LocalAgent"
# shellcheck source=smoke-lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke-lib.sh"
TIMEOUT="${AGENT_SMOKE_TIMEOUT:-600}"
# The checks grep plain text; never let a CI runner that forces colour put escapes in it.
export NO_COLOR=1
unset CLICOLOR_FORCE

fail() {
    echo "::error::$*" >&2
    exit 1
}

[ -f "$MODEL" ] || fail "model not found: $MODEL"
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"
SET_DIR="$(cd "$SET_DIR" && pwd)"

mapfile -t AGENTS < <(find "$AGENT_DIR" -maxdepth 1 -name 'llama-atmosphere-agent-*.jar' ! -name '*-sources.jar' ! -name '*-javadoc.jar' | sort)
[ "${#AGENTS[@]}" -eq 1 ] || fail "expected exactly 1 agent jar in $AGENT_DIR, got ${#AGENTS[@]}: ${AGENTS[*]:-none}"
AGENT="${AGENTS[0]}"
echo "Agent jar: $(basename "$AGENT") ($(du -h "$AGENT" | cut -f1)), $(find "$AGENT_DIR" -maxdepth 1 -name '*.jar' | wc -l) jar(s) in $AGENT_DIR"
if find "$AGENT_DIR" -maxdepth 1 -name 'llama-[0-9]*.jar' | grep -q .; then
    fail "the agent directory carries a core jar: $(find "$AGENT_DIR" -maxdepth 1 -name 'llama-[0-9]*.jar')"
fi
CORE_CP="$(smoke_classpath "$SET_DIR")" || exit 1
CP="$AGENT_DIR/*:$CORE_CP"
echo "Core smoke set: $SET_DIR (GPU module jars: $(smoke_set_modules "$SET_DIR" | tr '\n' ' '))"

# 1. Without the core on the classpath the agent must not work: it brings none of its own.
set +e
timeout "$TIMEOUT" java -cp "$AGENT_DIR/*" "$AGENT_MAIN" --model "$MODEL" --plain --prompt hi \
    > agent-alone.log 2>&1 < /dev/null
rc=$?
set -e
[ "$rc" -ne 0 ] || fail "the agent ran without the core on the classpath - does it bundle the core?"
grep -q 'NoClassDefFoundError: net/ladenthin/llama/LlamaModel' agent-alone.log \
    || { cat agent-alone.log; fail "agent without core failed, but not for the missing core (see above)"; }
echo "OK: the agent carries no core"

# 2. --help
java -cp "$CP" "$AGENT_MAIN" --help > agent-help.log 2>&1 < /dev/null || { cat agent-help.log; fail "--help exited non-zero"; }
grep -q 'Usage: LocalAgent' agent-help.log || { cat agent-help.log; fail "--help printed no usage"; }
echo "OK: --help"

run_agent() {
    local log="$1"
    shift
    set +e
    timeout "$TIMEOUT" java -cp "$CP" "$AGENT_MAIN" --model "$MODEL" --ngl 0 --plain --temperature 0 "$@" \
        > "$log" 2>"${log%.log}.err.log" < /dev/null
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "===== $log =====" && cat "$log"
        echo "===== ${log%.log}.err.log (last 80 lines) =====" && tail -n 80 "${log%.log}.err.log"
        fail "agent exited with $status ($log)"
    fi
}

# 3. A plain answer through the in-process server.
run_agent agent-answer.log --prompt 'What is 2 + 2? Answer with one short sentence.'
grep -q '4' agent-answer.log || { cat agent-answer.log; fail "the answer does not contain 4"; }
echo "OK: plain answer"

# 4. A tool round: the marker exists only in the file, so it reaches the output only
# through read_file (whose result the console prints) or an answer built from it.
WORKSPACE="$(mktemp -d)"
MARKER="AGENT_SMOKE_$(date +%s)_$RANDOM"
printf '%s\n' "$MARKER" > "$WORKSPACE/marker.txt"
run_agent agent-tool.log --workspace "$WORKSPACE" \
    --prompt 'Read the file marker.txt with the read_file tool and tell me its exact content.'
rm -rf "$WORKSPACE"
grep -q 'read_file' agent-tool.log || { cat agent-tool.log; fail "the model did not call read_file"; }
grep -q "$MARKER" agent-tool.log || { cat agent-tool.log; fail "the marker never reached the output"; }
echo "OK: tool round (read_file)"

# 5. The browser front end. Port 0 lets the OS pick; the banner says which one it got.
WEB_LOG=agent-web.log
java -cp "$CP" "$AGENT_MAIN" --model "$MODEL" --ngl 0 --web --web-port 0 > agent-web.out.log 2> "$WEB_LOG" < /dev/null &
WEB_PID=$!
trap 'kill "$WEB_PID" 2>/dev/null || true' EXIT
URL=""
for _ in $(seq 1 "$TIMEOUT"); do
    URL="$(sed -n 's/^Open in a browser: //p' "$WEB_LOG" | head -n 1)"
    [ -n "$URL" ] && break
    kill -0 "$WEB_PID" 2>/dev/null || { tail -n 80 "$WEB_LOG"; fail "--web exited before it listened"; }
    sleep 1
done
[ -n "$URL" ] || { tail -n 80 "$WEB_LOG"; fail "--web printed no address within ${TIMEOUT}s"; }
BASE="${URL%%/?token=*}"
code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/atmosphere/console/")"
[ "$code" = "401" ] || fail "the console without a token answered $code, not 401"
COOKIES="$(mktemp)"
code="$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIES" "$URL")"
[ "$code" = "302" ] || fail "the token link answered $code, not a redirect"
grep -q jllama_agent "$COOKIES" || fail "the token link set no session cookie"
curl -s -f -b "$COOKIES" -o agent-web-console.log "$BASE/atmosphere/console/" \
    || fail "the console page is not served with the session cookie"
grep -qi '<script' agent-web-console.log || { head -c 2000 agent-web-console.log; fail "the console page has no script"; }
# The page ships with a placeholder the server replaces per response; left in, the browser's CSP
# blocks the console's own script and the page stays blank.
if grep -q '__ATMO_CSP_NONCE__' agent-web-console.log; then fail "the console page still carries the CSP nonce placeholder"; fi
rm -f "$COOKIES"
kill "$WEB_PID" 2>/dev/null || true
wait "$WEB_PID" 2>/dev/null || true
trap - EXIT
echo "OK: --web (token, cookie, console page)"

# 6. The editor front end.
WORKSPACE="$(mktemp -d)"
python3 "$(dirname "$0")/smoke/agent_acp_smoke.py" "$WORKSPACE" agent-acp.log -- \
    java -cp "$CP" "$AGENT_MAIN" --model "$MODEL" --ngl 0 --temperature 0 --acp \
    || { tail -n 80 agent-acp.err.log 2>/dev/null || true; fail "--acp smoke failed"; }
rm -rf "$WORKSPACE"
echo "OK: --acp"

# The core was reached through the loader: the line names every GPU module jar of the set.
smoke_assert_loader_line "$SET_DIR" agent-answer.err.log || fail "loader line check failed"
echo "Agent smoke test passed."
