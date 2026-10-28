#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Downloads models of the CI set (.github/models.csv) into models/ and checks each against the
# SHA-256 of its row -- the one place CI gets a model, and the same command works locally.
#
# Usage: fetch-models.sh [feature ...]
#   no argument   every model of the list
#   feature ...   only the rows whose feature column names one of them (draft, tts, vision, ...);
#                 a feature no row has is an error, so a typo cannot fetch nothing and pass
#
# Environment:
#   HF_TOKEN     optional Hugging Face token (CI: the HF_TOKEN secret, absent on fork pull
#                requests). With it the downloads count against that account's rate limit instead
#                of the shared one of the runner's IP address, which answered 429 before.
#   MODELS_DIR   target directory (default: models, relative to the working directory)
#
# A file that is already there with the right SHA-256 is kept, so a local models/ directory is
# reused; one with another hash is downloaded again. Downloads land in <file>.part and are moved
# into place only after the hash matched. Runs under Linux, macOS (shasum) and Git Bash on Windows.

set -euo pipefail

MODELS_CSV="$(dirname "$0")/models.csv"
DEST="${MODELS_DIR:-models}"
[ -f "$MODELS_CSV" ] || { echo "ERROR: model manifest not found: $MODELS_CSV" >&2; exit 1; }

sha256_of() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Read the whole manifest first: curl must not inherit a loop's stdin.
NAMES=() FEATURES=() HASHES=() URLS=()
while IFS=, read -r name feature hash url rest; do
    name="${name%$'\r'}"
    case "$name" in '' | \#*) continue ;; esac
    url="${url%$'\r'}"
    if [ -z "$url" ] || [ -n "${rest:-}" ]; then
        echo "ERROR: $MODELS_CSV: row '$name' is not filename,feature,sha256,url" >&2
        exit 1
    fi
    NAMES+=("$name") FEATURES+=("$feature") HASHES+=("$hash") URLS+=("$url")
done < "$MODELS_CSV"
[ "${#NAMES[@]}" -gt 0 ] || { echo "ERROR: no models parsed from $MODELS_CSV" >&2; exit 1; }

for wanted in "$@"; do
    found=no
    for feature in "${FEATURES[@]}"; do [ "$feature" = "$wanted" ] && found=yes; done
    [ "$found" = yes ] || { echo "ERROR: no model of $MODELS_CSV has the feature '$wanted'" >&2; exit 1; }
done

CURL_AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
    # The header goes through a file, so the token is not on curl's command line. curl sends a
    # custom Authorization header only to the host it was given, not to the CDN it is redirected to.
    HEADER_FILE="$(mktemp)"
    trap 'rm -f "$HEADER_FILE"' EXIT
    printf 'Authorization: Bearer %s\n' "$HF_TOKEN" > "$HEADER_FILE"
    CURL_AUTH=(-H "@$HEADER_FILE")
    echo "Hugging Face: with a token (HF_TOKEN)"
else
    echo "Hugging Face: anonymous (no HF_TOKEN; the runner's shared per-IP rate limit applies)"
fi

mkdir -p "$DEST"
fetched=0 kept=0
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}" feature="${FEATURES[$i]}" hash="${HASHES[$i]}" url="${URLS[$i]}"
    if [ "$#" -gt 0 ]; then
        selected=no
        for wanted in "$@"; do [ "$feature" = "$wanted" ] && selected=yes; done
        [ "$selected" = yes ] || continue
    fi
    file="$DEST/$name"
    if [ -f "$file" ]; then
        if [ "$(sha256_of "$file")" = "$hash" ]; then
            echo "- $name ($feature): present, SHA-256 matches"
            kept=$((kept + 1))
            continue
        fi
        echo "- $name ($feature): present with another SHA-256, downloading again"
    fi
    start=$SECONDS
    curl -L --proto =https --proto-redir =https --fail -sS --retry 5 --retry-all-errors \
        --connect-timeout 30 ${CURL_AUTH[@]+"${CURL_AUTH[@]}"} -o "$file.part" "$url"
    actual="$(sha256_of "$file.part")"
    if [ "$actual" != "$hash" ]; then
        rm -f "$file.part"
        echo "ERROR: $name: SHA-256 $actual, the manifest says $hash ($url)" >&2
        exit 1
    fi
    mv -f "$file.part" "$file"
    echo "- $name ($feature): downloaded, $(wc -c < "$file" | tr -d ' ') bytes in $((SECONDS - start)) s, SHA-256 matches"
    fetched=$((fetched + 1))
done
echo "models: $fetched downloaded, $kept already present (${*:-all features})"
