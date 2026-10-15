#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Authenticode-signs every DLL of the Windows natives directories with Azure Trusted Signing and
# flips `signed=no` to `signed=yes` in each directory's jllama-build.txt (TODO.md, delivery roadmap,
# item 11). Runs in the `package` job on a Linux runner, after merge-native-artifacts.sh and before
# the jars are built, so what gets packaged is already signed and the Windows smoke rows run the
# signed sets. The signer is jsign (Java, pinned by version and SHA-256), which talks to Trusted
# Signing over HTTPS; no Windows and no signtool needed.
#
# Credentials: none stored. The job's GitHub OIDC token is exchanged for an Azure access token
# through the federated credential of an app registration (client-credentials grant with a
# client_assertion), which the owner binds to this repository's `main` ref and `v*` tags. The step
# runs only where the workflow's `if:` says so (the signing variables set, main or a tag).
#
# Usage: sign-windows-natives.sh <natives-root>   (the net/ladenthin/llama/ tree)
# Environment: AZURE_TENANT_ID, AZURE_CLIENT_ID, TRUSTED_SIGNING_ENDPOINT (e.g.
#   https://weu.codesigning.azure.net), TRUSTED_SIGNING_ACCOUNT, TRUSTED_SIGNING_PROFILE, and the
#   runner's ACTIONS_ID_TOKEN_REQUEST_URL / ACTIONS_ID_TOKEN_REQUEST_TOKEN (permissions: id-token: write).
set -euo pipefail

ROOT="${1:?natives root}"
JSIGN_VERSION="7.2"
JSIGN_SHA256="9a99673bb011cc1d7faf00bc840a6b333fc8a9b596098da2f92946b68297f067"
TSA_URL="http://timestamp.acs.microsoft.com"

for v in AZURE_TENANT_ID AZURE_CLIENT_ID TRUSTED_SIGNING_ENDPOINT TRUSTED_SIGNING_ACCOUNT TRUSTED_SIGNING_PROFILE \
         ACTIONS_ID_TOKEN_REQUEST_URL ACTIONS_ID_TOKEN_REQUEST_TOKEN; do
  if [ -z "${!v:-}" ]; then
    echo "::error::sign-windows-natives.sh: $v is not set" >&2
    exit 1
  fi
done

mapfile -t dlls < <(find "$ROOT" -path '*/Windows/*' -name '*.dll' | sort)
if [ "${#dlls[@]}" -eq 0 ]; then
  echo "::error::no Windows DLL under $ROOT -- nothing to sign is a failure, not a pass" >&2
  exit 1
fi

# --- jsign, pinned
work="$(mktemp -d)"
mvn -q -B --no-transfer-progress dependency:copy -Dartifact="net.jsign:jsign:${JSIGN_VERSION}" -DoutputDirectory="$work"
jar="$work/jsign-${JSIGN_VERSION}.jar"
echo "${JSIGN_SHA256}  ${jar}" | sha256sum -c - >/dev/null

# --- the access token: GitHub OIDC -> Azure (federated credential, no stored secret)
github_jwt="$(curl -sSf -H "Authorization: bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN}" \
  "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=api://AzureADTokenExchange" | jq -r .value || true)"
if [ -z "$github_jwt" ] || [ "$github_jwt" = "null" ]; then
  echo "::error::no OIDC token from the runner (permissions: id-token: write?)" >&2
  exit 1
fi
token="$(curl -sSf -X POST "https://login.microsoftonline.com/${AZURE_TENANT_ID}/oauth2/v2.0/token" \
  --data-urlencode "client_id=${AZURE_CLIENT_ID}" \
  --data-urlencode "grant_type=client_credentials" \
  --data-urlencode "scope=https://codesigning.azure.net/.default" \
  --data-urlencode "client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer" \
  --data-urlencode "client_assertion=${github_jwt}" | jq -r .access_token || true)"
if [ -z "$token" ] || [ "$token" = "null" ]; then
  echo "::error::Azure refused the federated credential exchange (is the app registration's federated credential bound to this repository and ref?)" >&2
  exit 1
fi
echo "::add-mask::$token"

# --- sign, with an RFC 3161 timestamp from Microsoft's TSA
java -jar "$jar" sign \
  --storetype TRUSTEDSIGNING \
  --keystore "${TRUSTED_SIGNING_ENDPOINT}" \
  --storepass "${token}" \
  --alias "${TRUSTED_SIGNING_ACCOUNT}/${TRUSTED_SIGNING_PROFILE}" \
  --alg SHA-256 --tsaurl "${TSA_URL}" --tsmode RFC3161 --tsretries 3 --tsretrywait 10 \
  --replace --quiet \
  "${dlls[@]}"

# --- record it: every directory a DLL was signed in says signed=yes
for dir in $(printf '%s\n' "${dlls[@]}" | xargs -n1 dirname | sort -u); do
  build="$dir/jllama-build.txt"
  if [ ! -f "$build" ] || ! grep -q '^signed=no$' "$build"; then
    echo "::error::$build does not say signed=no (jllama-build-info.cmake writes it; the merge step requires it)" >&2
    exit 1
  fi
  sed -i 's/^signed=no$/signed=yes/' "$build"
  echo "signed $(find "$dir" -maxdepth 1 -name '*.dll' | wc -l) DLL(s) in $(realpath --relative-to="$ROOT" "$dir")"
done
rm -rf "$work"
