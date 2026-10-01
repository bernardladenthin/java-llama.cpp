# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The GGUF file names publish.yml's `env:` names (for the smoke scripts, the Android emulator jobs
and the integration jobs outside the llama module), checked against .github/models.csv -- the list
the download-models job fetches and validate-models.sh requires. A name the list lacks is a model
no job downloads: its consumer would fail or self-skip. (The llama module's tests default to the
list itself; TestConstantsTest checks that side.)
"""

import re

ENV_MODEL = re.compile(r'^  ([A-Z0-9_]+):\s*"?([^"\s]+\.gguf)"?\s*$', re.M)


def filenames(models_csv_text):
    return {line.split(",", 1)[0].strip() for line in models_csv_text.splitlines()
            if line.strip() and not line.startswith("#")}


def check(models_csv_text, workflow_text):
    listed = filenames(models_csv_text)
    return [f"publish.yml env {name}={value} is not a filename of .github/models.csv -- no job downloads it"
            for name, value in ENV_MODEL.findall(workflow_text) if value not in listed]
