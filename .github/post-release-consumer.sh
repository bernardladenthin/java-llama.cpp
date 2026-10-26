#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Post-release check 11 (docs/post-release-check.md): a consumer resolves the published release from
# Maven Central. Nothing in the publish workflow ever resolves the release from Central -- the
# smoke sets are assembled from the build tree -- so the parent pom's property substitutions, the
# scopes a consumer actually gets and the presence of every platform jar are first tried here.
#
# A throwaway consumer pom depends on net.ladenthin:llama-platform (type pom) plus the given extra
# classifiers of net.ladenthin:llama, and is resolved against an EMPTY local repository into
# <set-dir>: the classes jar, its runtime dependencies and the natives jars, flat -- a smoke set, so
# .github/smoke-test-natives.sh and friends run on it unchanged. Asserted on the resolved graph:
#   - the classes jar and the library jar of every `platform=yes` row of natives.csv (at the tag);
#   - jackson-databind, slf4j-api and jspecify at compile scope, slf4j-simple at runtime;
#   - neither logback nor checker-qual anywhere: both break a Java 8 consumer, and only the pom
#     scopes keep them out, which no compile-time check sees (6.0.0's AAR poms got this wrong).
# Afterwards the natives jars of other platforms are removed from <set-dir>, so it is the smoke set
# of <target> (<os>-<arch>, e.g. linux-x86-64) that the smoke scripts expect: one library jar.
#
# Usage: post-release-consumer.sh <version> <natives.csv at the tag> <set-dir> <target> [extra classifier...]
set -euo pipefail

USAGE="usage: post-release-consumer.sh <version> <natives.csv> <set-dir> <target> [classifier...]"
VERSION="${1:?$USAGE}"
NATIVES_CSV="${2:?$USAGE}"
SET_DIR="${3:?$USAGE}"
TARGET="${4:?$USAGE}"
shift 4

fail() {
    echo "::error::$*" >&2
    exit 1
}

[ -f "$NATIVES_CSV" ] || fail "no natives list at $NATIVES_CSV"
[ ! -e "$SET_DIR" ] || [ -z "$(ls -A "$SET_DIR")" ] || fail "$SET_DIR is not empty"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

extra=""
for classifier in "$@"; do
    extra="$extra
    <dependency><groupId>net.ladenthin</groupId><artifactId>llama</artifactId><version>$VERSION</version><classifier>$classifier</classifier></dependency>"
done
cat > "$WORK/pom.xml" << EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>local.post-release</groupId>
  <artifactId>consumer</artifactId>
  <version>1</version>
  <dependencies>
    <dependency><groupId>net.ladenthin</groupId><artifactId>llama-platform</artifactId><version>$VERSION</version><type>pom</type></dependency>$extra
  </dependencies>
</project>
EOF

mvn_consumer() {
    mvn -B -ntp -f "$WORK/pom.xml" "-Dmaven.repo.local=$WORK/m2" "$@"
}
mvn_consumer -q dependency:list -DincludeScope=runtime -DoutputFile="$WORK/deps.txt" -DexcludeTransitive=false \
    || fail "net.ladenthin:llama-platform:$VERSION does not resolve from Maven Central"
mvn_consumer -q dependency:copy-dependencies -DincludeScope=runtime "-DoutputDirectory=$(cd "$(dirname "$SET_DIR")" && pwd)/$(basename "$SET_DIR")"
# dependency:list writes "   group:artifact:type[:classifier]:version:scope" (plus a module suffix on newer plugins)
deps="$(sed -n 's/^ *\([^ ]*:[^ ]*\).*/\1/p' "$WORK/deps.txt" | grep ':' || true)"
echo "resolved from Central (runtime scope):"
sed 's/^/  /' <<< "$deps"

has() { grep -qE "^$1:" <<< "$deps"; }
scope_of() { grep -E "^$1:" <<< "$deps" | head -1 | awk -F: '{print $NF}'; }

for dep in com.fasterxml.jackson.core:jackson-databind org.slf4j:slf4j-api org.jspecify:jspecify; do
    has "$dep" || fail "$dep is not in the consumer's graph"
    [ "$(scope_of "$dep")" = "compile" ] || fail "$dep has scope $(scope_of "$dep"), expected compile"
done
has org.slf4j:slf4j-simple || fail "org.slf4j:slf4j-simple (the shipped binding) is not in the consumer's graph"
[ "$(scope_of org.slf4j:slf4j-simple)" = "runtime" ] || fail "slf4j-simple has scope $(scope_of org.slf4j:slf4j-simple), expected runtime"
if grep -qE '^(ch\.qos\.logback|org\.checkerframework):' <<< "$deps"; then
    fail "logback or checker-qual reached the consumer: $(grep -E '^(ch\.qos\.logback|org\.checkerframework):' <<< "$deps" | tr '\n' ' ')"
fi
[ -f "$SET_DIR/llama-$VERSION.jar" ] || fail "the classes jar llama-$VERSION.jar was not resolved"
missing=""
while IFS=, read -r classifier _ _ platform _; do
    [ "$platform" = "yes" ] || continue
    [ -f "$SET_DIR/llama-$VERSION-$classifier.jar" ] || missing="$missing $classifier"
done < <(grep -v -e '^#' -e '^classifier,' -e '^$' "$NATIVES_CSV" | tr -d '\r')
[ -z "$missing" ] || fail "llama-platform did not bring the library jar of:$missing"
for classifier in "$@"; do
    [ -f "$SET_DIR/llama-$VERSION-$classifier.jar" ] || fail "the classifier $classifier was not resolved"
done
echo "consumer resolution OK: $(find "$SET_DIR" -maxdepth 1 -name '*.jar' | wc -l | tr -d ' ') jars, no logback, no checker-qual"
while IFS=, read -r classifier _; do
    case "$classifier" in *-"$TARGET") ;; *) rm -f "$SET_DIR/llama-$VERSION-$classifier.jar" ;; esac
done < <(grep -v -e '^#' -e '^classifier,' -e '^$' "$NATIVES_CSV" | tr -d '\r')
find "$SET_DIR" -maxdepth 1 -name "llama-$VERSION-*-$TARGET.jar" | grep -q . || fail "no natives jar of $TARGET in the set"
echo "smoke set of $TARGET: $(cd "$SET_DIR" && ls llama-*.jar | tr '\n' ' ')"
