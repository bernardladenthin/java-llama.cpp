#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Authenticode-signs every DLL of the Windows natives directories with SSL.com eSigner (the
# project's IV code-signing certificate, held in SSL.com's cloud HSM) and flips `signed=no` to
# `signed=yes` in each directory's jllama-build.txt (TODO.md, delivery roadmap, item 11). Runs in
# the `sign-windows-natives` job on a Linux runner over the downloaded `natives-*-windows-*`
# artifacts, which the job re-uploads signed before `package` merges them, so what gets packaged is
# already signed and the Windows smoke rows run the signed sets. The signer is jsign (Java, pinned
# by version and SHA-256), which talks to eSigner over HTTPS; no Windows and no signtool needed.
#
# Credentials: the four values SSL.com's own documentation names, as secrets of the `maven-central`
# environment (the GPG key's environment). The TOTP secret lets whoever holds it sign in the
# owner's name, which is why the job runs only on an explicit `sign_windows` dispatch and only
# with that environment's rules. The passwords reach jsign through its `env:` prefix, never as
# process arguments (readable in /proc/<pid>/cmdline by every process of the runner user);
# nothing here prints an argument, `set -x` stays off.
#
# Usage: sign-windows-natives.sh <root>   (a tree holding */Windows/<arch>/<backend>/*.dll)
# Environment: ES_USERNAME, ES_PASSWORD, CREDENTIAL_ID, ES_TOTP_SECRET; optional ES_ENDPOINT
#   (default https://cs.ssl.com; https://cs-try.ssl.com is SSL.com's sandbox, whose test
#   certificate Windows does not trust) and JSIGN_JAR (a local jsign jar instead of the Maven fetch).
set -euo pipefail

ROOT="${1:?natives root}"
JSIGN_VERSION="7.2"
JSIGN_SHA256="9a99673bb011cc1d7faf00bc840a6b333fc8a9b596098da2f92946b68297f067"
TSA_URL="http://ts.ssl.com"
ES_ENDPOINT="${ES_ENDPOINT:-https://cs.ssl.com}"

for v in ES_USERNAME ES_PASSWORD CREDENTIAL_ID ES_TOTP_SECRET; do
  if [ -z "${!v:-}" ]; then
    echo "::error::sign-windows-natives.sh: $v is not set (a secret of the maven-central environment)" >&2
    exit 1
  fi
done

mapfile -t dlls < <(find "$ROOT" -path '*/Windows/*' -name '*.dll' | sort)
if [ "${#dlls[@]}" -eq 0 ]; then
  echo "::error::no Windows DLL under $ROOT -- nothing to sign is a failure, not a pass" >&2
  exit 1
fi

# --- jsign, pinned (`-N`: the goal needs no reactor, and the checkout root is one)
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
if [ -n "${JSIGN_JAR:-}" ]; then
  jar="$JSIGN_JAR"
else
  mvn -q -B -N --no-transfer-progress dependency:copy -Dartifact="net.jsign:jsign:${JSIGN_VERSION}" -DoutputDirectory="$work"
  jar="$work/jsign-${JSIGN_VERSION}.jar"
fi
echo "${JSIGN_SHA256}  ${jar}" | sha256sum -c - >/dev/null

# --- sign: eSigner authenticates with "<username>|<password>" as the store password, the
# credential id as the alias and the TOTP secret as the key password (jsign generates the one-time
# code per signature itself); both passwords are read from the environment (`env:<name>`), so the
# command line carries no secret; an RFC 3161 timestamp from SSL.com's TSA.
export JSIGN_STOREPASS="${ES_USERNAME}|${ES_PASSWORD}"
echo "signing ${#dlls[@]} DLL(s) with SSL.com eSigner at ${ES_ENDPOINT}"
java -jar "$jar" sign \
  --storetype ESIGNER \
  --keystore "${ES_ENDPOINT}" \
  --storepass env:JSIGN_STOREPASS \
  --alias "${CREDENTIAL_ID}" \
  --keypass env:ES_TOTP_SECRET \
  --alg SHA-256 --tsaurl "${TSA_URL}" --tsmode RFC3161 --tsretries 3 --tsretrywait 10 \
  --quiet \
  "${dlls[@]}"

# --- record it: every directory a DLL was signed in says signed=yes
mapfile -t dirs < <(printf '%s\n' "${dlls[@]}" | xargs -n1 dirname | sort -u)
for dir in "${dirs[@]}"; do
  build="$dir/jllama-build.txt"
  if [ ! -f "$build" ] || ! grep -q '^signed=no$' "$build"; then
    echo "::error::$build does not say signed=no (jllama-build-info.cmake writes it; the merge step requires it)" >&2
    exit 1
  fi
  sed -i 's/^signed=no$/signed=yes/' "$build"
  echo "signed $(find "$dir" -maxdepth 1 -name '*.dll' | wc -l) DLL(s) in $(realpath --relative-to="$ROOT" "$dir")"
done
