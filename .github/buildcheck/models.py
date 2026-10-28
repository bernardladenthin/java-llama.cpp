# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The CI model set, .github/models.csv: its rows (filename,feature,sha256,url, the format
fetch-models.sh reads), and the names publish.yml uses from it -- the GGUF file names of its `env:`
(for the smoke scripts, the Android emulator jobs and the integration jobs outside the llama module)
and the features its jobs ask fetch-models for. A name or feature the list lacks is a model no job
downloads: its consumer would fail or self-skip. (The llama module's tests default to the list
itself; TestConstantsTest checks that side.)
"""

import re

ENV_MODEL = re.compile(r'^  ([A-Z0-9_]+):\s*"?([^"\s]+\.gguf)"?\s*$', re.M)
FEATURES_INPUT = re.compile(r"^ +features: *(.*?) *$", re.M)
FILENAME = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")
FEATURE = re.compile(r"^[a-z]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# A file of a Hugging Face model repository at a commit: resolve/main would let the file change
# under the test set, and the SHA-256 would then fail every job at once.
PINNED_URL = re.compile(r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/(?:.+/)?([^/]+)$")


def rows(models_csv_text):
    return [[field.strip() for field in line.split(",")] for line in models_csv_text.splitlines()
            if line.strip() and not line.startswith("#")]


def filenames(models_csv_text):
    return {fields[0] for fields in rows(models_csv_text)}


def features(models_csv_text):
    return {fields[1] for fields in rows(models_csv_text) if len(fields) > 1}


def check_rows(models_csv_text):
    failures, seen = [], set()
    for fields in rows(models_csv_text):
        if len(fields) != 4:
            failures.append(f"models.csv row {','.join(fields)!r} is not filename,feature,sha256,url")
            continue
        name, feature, sha256, url = fields
        if not FILENAME.match(name):
            failures.append(f"models.csv: {name!r} is no plain .gguf file name")
        if name in seen:
            failures.append(f"models.csv: {name} is listed twice")
        seen.add(name)
        if not FEATURE.match(feature):
            failures.append(f"models.csv: {name}: feature {feature!r} is not one lower-case word")
        if not SHA256.match(sha256):
            failures.append(f"models.csv: {name}: {sha256!r} is no lower-case SHA-256")
        pinned = PINNED_URL.match(url)
        if not pinned:
            failures.append(f"models.csv: {name}: {url} is no Hugging Face URL pinned to a commit "
                            "(https://huggingface.co/<owner>/<repo>/resolve/<40 hex>/<file>)")
        elif pinned.group(1) != name:
            failures.append(f"models.csv: {name}: the URL names another file, {pinned.group(1)}")
    return failures


def check(models_csv_text, workflow_text):
    listed, known = filenames(models_csv_text), features(models_csv_text)
    failures = check_rows(models_csv_text)
    failures += [f"publish.yml env {name}={value} is not a filename of .github/models.csv -- no job downloads it"
                 for name, value in ENV_MODEL.findall(workflow_text) if value not in listed]
    failures += [f"publish.yml asks fetch-models for the feature {feature!r}, which no row of "
                 ".github/models.csv has"
                 for line in FEATURES_INPUT.findall(workflow_text) for feature in line.split()
                 if feature not in known]
    return failures
