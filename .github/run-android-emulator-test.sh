#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Executed INSIDE reactivecircus/android-emulator-runner's `script:` (emulator booted,
# adb connected). Lives in a file because the runner executes the `script:` input LINE BY
# LINE through sh — multi-line if-blocks are a syntax error there (exit code 2).
#
# Pushes the cached tiny draft model to the emulator (world-readable so the app-uid
# instrumentation can open it) and runs the consumer fixture's instrumented tests.
# A missing model degrades to a warning — the on-device tests then self-skip
# (Assume) instead of failing the fixture.
set -euo pipefail

if [ -f "models/${DRAFT_MODEL_NAME}" ]; then
  adb push "models/${DRAFT_MODEL_NAME}" /data/local/tmp/jllama-test-model.gguf
  adb shell chmod 644 /data/local/tmp/jllama-test-model.gguf
else
  echo "::warning::draft model missing from the GGUF cache — on-device tests will self-skip"
fi

VERSION=$(mvn -q -DforceStdout help:evaluate -Dexpression=project.version | tail -n1)
echo "Running connectedDebugAndroidTest against llama-android ${VERSION}"
gradle -p .github/android-consumer-test connectedDebugAndroidTest "-PjllamaVersion=${VERSION}"
