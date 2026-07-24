#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# GPG-signs the all-backends server fat jars (jar-with-dependencies) with a detached,
# armored .asc signature — the authenticity counterpart to the .sha256 integrity files
# that package-fatjars.sh already emits. Signature parity with the thin jars (which
# maven-gpg signs at deploy time) and with the BAF / srcmorph sibling fat jars.
#
# Signed in the GitHub-Release attach jobs (github-snapshot / github-release-signed),
# not in package-fatjars, because only that dispatch-gated path receives the signing
# key: the GPG_PRIVATE_KEY / GPG_PASSPHRASE secrets are scoped to the `maven-central`
# environment. The cross-repo mechanism is documented in
# workspace/policies/fat-jar-release-assets.md.
#
# Usage: sign-fatjars.sh <asset-dir>
#   <asset-dir>  directory holding the downloaded fat jars (and the thin jars); every
#                *-jar-with-dependencies*.jar in it is signed in place (-> <jar>.asc).
#
# Env: GPG_PRIVATE_KEY   armored secret key (required)
#      GPG_PASSPHRASE    passphrase for the key (may be empty)
#
# Fail-loud: aborts if the key is absent, if no fat jar is found, or if any signature
# fails to verify.

set -euo pipefail

DIR="${1:?usage: sign-fatjars.sh <asset-dir>}"

if [ -z "${GPG_PRIVATE_KEY:-}" ]; then
  echo "::error::GPG_PRIVATE_KEY is empty — cannot sign the fat jars. The maven-central environment did not deliver the secret to this ref." >&2
  exit 1
fi
if [ -n "${GPG_PASSPHRASE:-}" ]; then echo "::add-mask::${GPG_PASSPHRASE}"; fi

# Ephemeral, private keyring; removed on exit (do NOT touch the runner's default one).
GNUPGHOME="$(mktemp -d)"
export GNUPGHOME
chmod 700 "$GNUPGHOME"
cleanup() { gpgconf --kill gpg-agent >/dev/null 2>&1 || true; rm -rf "$GNUPGHOME"; }
trap cleanup EXIT

printf '%s\n' "$GPG_PRIVATE_KEY" | gpg --batch --import
KEYID="$(gpg --list-secret-keys --with-colons --fixed-list-mode | awk -F: '$1=="sec"{print $5; exit}')"
if [ -z "$KEYID" ]; then
  echo "::error::No secret key imported from GPG_PRIVATE_KEY." >&2
  exit 1
fi

shopt -s nullglob
jars=("$DIR"/*-jar-with-dependencies*.jar)
shopt -u nullglob
if [ "${#jars[@]}" -eq 0 ]; then
  echo "::error::No *-jar-with-dependencies*.jar found in '$DIR' to sign." >&2
  exit 1
fi

for f in "${jars[@]}"; do
  # Skip a signature file itself if the glob ever catches one.
  case "$f" in *.asc) continue ;; esac
  printf '%s' "${GPG_PASSPHRASE:-}" | gpg --batch --yes --pinentry-mode loopback \
    --passphrase-fd 0 --local-user "$KEYID" --detach-sign --armor "$f"
  gpg --batch --verify "$f.asc" "$f"
  echo "signed + verified: $(basename "$f") -> $(basename "$f").asc"
done

echo "Signed ${#jars[@]} fat jar(s) in '$DIR'."
