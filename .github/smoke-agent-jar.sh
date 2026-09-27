#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# Smoke test for the llama-atmosphere-agent release asset, run exactly the way the README
# tells a user to run it: the agent jar lies next to a core fat jar and is started with
# `java -jar`, so the core is found only through the agent manifest's Class-Path. That
# makes this the check that the two release assets actually fit together (same version in
# the file names, nothing missing on either side), which no test run from the source tree
# can see.
#
#   1. the agent jar carries no core: started alone, loading a model fails with
#      NoClassDefFoundError for net.ladenthin.llama.LlamaModel;
#   2. --help exits 0 and prints the usage;
#   3. a one-shot prompt with the model loaded in-process answers "2+2" with a 4;
#   4. a one-shot prompt that needs a reading tool (read_file) reads a marker file from
#      --workspace, so the whole tool loop runs through the shipped jars.
#
# Usage: smoke-agent-jar.sh <jar-dir> <model-path>
# <jar-dir> must hold exactly one llama-atmosphere-agent-*-jar-with-dependencies.jar and at
# least one core fat jar its Class-Path names. Output of each run is kept in agent-*.log in
# the working directory (uploaded by the CI job on failure).
set -euo pipefail

JAR_DIR="${1:?usage: smoke-agent-jar.sh <jar-dir> <model-path>}"
MODEL="${2:?usage: smoke-agent-jar.sh <jar-dir> <model-path>}"
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
JAR_DIR="$(cd "$JAR_DIR" && pwd)"

mapfile -t AGENTS < <(find "$JAR_DIR" -maxdepth 1 -name 'llama-atmosphere-agent-*-jar-with-dependencies.jar' | sort)
[ "${#AGENTS[@]}" -eq 1 ] || fail "expected exactly 1 agent jar in $JAR_DIR, got ${#AGENTS[@]}: ${AGENTS[*]:-none}"
AGENT="${AGENTS[0]}"
echo "Agent jar: $(basename "$AGENT") ($(du -h "$AGENT" | cut -f1))"

# The manifest names the core jars by file name; at least one of them must be here, or
# `java -jar` would start without a core. unzip wraps manifest lines at 72 bytes with a
# leading space, so the continuation lines are joined first.
CLASS_PATH="$(unzip -p "$AGENT" META-INF/MANIFEST.MF | tr -d '\r' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n //g' \
    | sed -n 's/^Class-Path: //p')"
[ -n "$CLASS_PATH" ] || fail "agent manifest has no Class-Path"
found=""
for entry in $CLASS_PATH; do
    if [ -f "$JAR_DIR/$entry" ]; then
        found="$entry"
        break
    fi
done
[ -n "$found" ] || fail "none of the core jars the agent manifest names is in $JAR_DIR: $CLASS_PATH (present: $(ls "$JAR_DIR"))"
echo "Core jar picked up via Class-Path: $found"

# 1. Without a core next to it the agent must not work: that is what makes it small.
ALONE="$(mktemp -d)"
cp "$AGENT" "$ALONE/"
set +e
timeout "$TIMEOUT" java -jar "$ALONE/$(basename "$AGENT")" --model "$MODEL" --plain --prompt hi \
    > agent-alone.log 2>&1 < /dev/null
rc=$?
set -e
rm -rf "$ALONE"
[ "$rc" -ne 0 ] || fail "the agent jar ran without a core jar next to it - does it bundle the core?"
grep -q 'NoClassDefFoundError: net/ladenthin/llama/LlamaModel' agent-alone.log \
    || { cat agent-alone.log; fail "agent without core failed, but not for the missing core (see above)"; }
echo "OK: the agent jar carries no core"

# 2. --help
java -jar "$AGENT" --help > agent-help.log 2>&1 < /dev/null || { cat agent-help.log; fail "--help exited non-zero"; }
grep -q 'Usage: LocalAgent' agent-help.log || { cat agent-help.log; fail "--help printed no usage"; }
echo "OK: --help"

run_agent() {
    local log="$1"
    shift
    set +e
    timeout "$TIMEOUT" java -jar "$AGENT" --model "$MODEL" --ngl 0 --plain --temperature 0 "$@" \
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

echo "Agent release asset smoke test passed."
