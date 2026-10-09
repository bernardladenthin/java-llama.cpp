#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Fails a Java test job when the suite silently stopped running tests.
#
# WHY THIS EXISTS. Surefire's working directory is the module basedir while the shared GGUF cache
# restores to the reactor root, so every model path resolved one directory too deep and EVERY
# model-gated class aborted in its @BeforeAll assumption — on every test-java-* job, for months,
# while the pipeline stayed green. Several stale assertions rode along unnoticed.
#
# The shape is what defeats the obvious guard: a CLASS-LEVEL assumption failure makes Surefire
# record tests="0" errors="0" failures="0" skipped="0" for that class. It contributes NO test
# entries at all, so "did this run skip anything?" is structurally blind to it — a skipped test is
# still a reported test. Three checks catch it:
#
#   1. Any testsuite reporting tests="0". This is the exact signature above, it is precise, and it
#      is platform-independent: a class that runs nowhere is a bug on every OS.
#   2. A floor on the number of tests EXECUTED (run minus skipped). This is the other shape of the
#      same failure: a METHOD-LEVEL assumption (Assume.assumeTrue inside a test) does report the
#      test, as skipped. Measured on a green run at b11529, every test-java-* job runs 1869 tests
#      and skips 4 (Linux), 11 (macOS, all three jobs) or 13 (Windows, Ninja and MSVC), i.e. it
#      executes 1856 to 1865; the same tree in a checkout WITHOUT the models runs 1869 and skips
#      280, i.e. executes 1589. The jobs pass --min-executed 1800: that catches the model set going
#      missing (or a resolver regression) with a margin of ~200 either way, and tolerates ordinary
#      test removals of up to ~56 tests before someone has to lower it, deliberately.
#   3. A floor on the total (--min-total), the backstop for a whole class file going missing from
#      the run rather than reporting zero. Implied by check 2 when both are given.
#
# Usage: .github/verify-test-counts.sh <surefire-reports-dir> [--min-total N] [--min-executed N]
# Exit codes: 0 all good, 1 a check failed, 2 nothing to scan.

set -euo pipefail

REPORT_DIR="${1:-}"
MIN_TOTAL=0
MIN_EXECUTED=0
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --min-total) MIN_TOTAL="$2"; shift 2 ;;
        --min-executed) MIN_EXECUTED="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -n "$REPORT_DIR" ] || { echo "usage: $0 <surefire-reports-dir> [--min-total N] [--min-executed N]" >&2; exit 1; }
[ -d "$REPORT_DIR" ] || { echo "ERROR: no such directory: $REPORT_DIR" >&2; exit 2; }

shopt -s nullglob
reports=("$REPORT_DIR"/TEST-*.xml)
shopt -u nullglob

# An empty input is a failure, never a pass — the same rule verify-bytecode-version.sh follows,
# for the same reason: a job that produced no reports at all has not proved anything.
if [ "${#reports[@]}" -eq 0 ]; then
    echo "ERROR: no TEST-*.xml under $REPORT_DIR — the suite did not run" >&2
    exit 2
fi

# An attribute of the <testsuite> element; the first match, so a nested element carrying the same
# attribute name cannot shift the number (<testcase> has neither, a skipped test is a child element).
suite_attr() {
    local n
    n="$(grep -o "$2=\"[0-9]*\"" "$1" | head -1 | grep -o '[0-9]*' || true)"
    echo "${n:-0}"
}

total=0
skipped=0
empty_suites=()
for f in "${reports[@]}"; do
    n="$(suite_attr "$f" tests)"
    total=$((total + n))
    skipped=$((skipped + $(suite_attr "$f" skipped)))
    if [ "$n" -eq 0 ]; then
        empty_suites+=("$(basename "$f")")
    fi
done
executed=$((total - skipped))

status=0

if [ "${#empty_suites[@]}" -gt 0 ]; then
    echo "ERROR: ${#empty_suites[@]} test class(es) contributed ZERO test entries:" >&2
    printf '  %s\n' "${empty_suites[@]}" >&2
    echo "  A class-level @BeforeAll assumption that fails looks exactly like this. It is NOT a" >&2
    echo "  skip — the class reports no tests at all, so it is invisible to a skip check. If a" >&2
    echo "  model is missing, validate-models should have failed the job before this point." >&2
    status=1
fi

if [ "$total" -lt "$MIN_TOTAL" ]; then
    echo "ERROR: only $total test(s) run, below the floor of $MIN_TOTAL" >&2
    status=1
fi

if [ "$executed" -lt "$MIN_EXECUTED" ]; then
    echo "ERROR: only $executed test(s) executed ($total run, $skipped skipped), below the floor of $MIN_EXECUTED" >&2
    echo "  Method-level assumptions skipping in bulk look like this: a model file the tests expect is" >&2
    echo "  missing or resolves to the wrong directory (TestConstants.resolveModelPath), or a native" >&2
    echo "  library failed to load. If the suite legitimately shrank, lower --min-executed in the" >&2
    echo "  workflow deliberately." >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "test counts verified: $total test(s) run, $skipped skipped, $executed executed across ${#reports[@]} class(es), none empty (floors: total $MIN_TOTAL, executed $MIN_EXECUTED)"
fi
exit "$status"
