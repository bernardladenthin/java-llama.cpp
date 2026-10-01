<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->
# llama-android consumer-test fixture (CI only)

Minimal AGP app that consumes the `net.ladenthin:llama-android` AAR from `mavenLocal`.
Not a shipped project — it exists so CI proves Android Studio consumption end to end.

Two CI jobs drive it (`.github/workflows/publish.yml`):

1. **`package-android-aar`** — `assembleRelease` with R8: validates AAR format, manifest
   minSdk merge, `jni/{arm64-v8a,x86_64}` packaging, and the shipped consumer ProGuard
   rules (asserts the APK still carries the binding and both `.so` files).
2. **`test-android-emulator`** — boots a KVM-accelerated **x86_64** emulator, adb-pushes a
   small GGUF (the already-cached draft model) to
   `/data/local/tmp/jllama-test-model.gguf`, and runs `connectedDebugAndroidTest`:
   `OnDeviceInferenceTest` loads the binding via `System.loadLibrary` from the AAR's
   `jni/x86_64/libjllama.so`, reads the GGUF with the pure-Java `GgufInspector`, and runs
   real native inference on the emulator. Tests self-skip when the model was not pushed,
   so a local `connectedAndroidTest` against a bare emulator stays green.

What the emulator job deliberately does NOT cover: arm64 kernels (real-device ABI — the
example app on hardware covers that) and the Adreno/OpenCL flavor (no OpenCL ICD in the
emulator).
