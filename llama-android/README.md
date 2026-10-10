<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# llama-android

Android AAR packaging of [java-llama.cpp](https://github.com/bernardladenthin/java-llama.cpp):
the `net.ladenthin:llama` Java API plus the CI-built native library and ggml backend modules,
consumable from any Android project as a normal Maven dependency — no git submodule, no NDK
build, no manual ProGuard rules.

```kotlin
// build.gradle.kts of your app — that's all.
dependencies {
    implementation("net.ladenthin:llama-android:5.2.0")
    // additionally, for Qualcomm Adreno GPUs (device must provide an OpenCL ICD) -- it depends
    // on llama-android, so the line above may also be left out:
    // implementation("net.ladenthin:llama-android-opencl:5.2.0")
}
```

- **minSdk 28** (Android 9.0 Pie) — enforced at build time via the AAR manifest.
- **Multi-ABI**: `arm64-v8a` (devices) + `x86_64` (Android Studio emulator, Chromebooks,
  x86-64 Android hardware). App bundles split per ABI, so phones download only arm64.
- **R8/ProGuard safe** — consumer rules ship inside the AAR (`proguard.txt`) and apply
  automatically.
- **16 KB page-size compliant** native libraries (Google Play requirement for Android 15+ targets).
- **Modular natives, as on the desktop**: `libjllama.so` plus ggml's shared libraries and one
  `libggml-cpu-<variant>.so` module per ARM instruction-set level (ggml's seven Android levels,
  `android_armv8.0_1` … `android_armv9.2_2`: dotprod, fp16, i8mm, SVE/SVE2, SME; the x86-64 ABI
  the 14 desktop levels); at start the library scores every module against the
  device's CPU and loads the best one. The OpenCL AAR adds `libggml-opencl.so`, which the same
  start loads when it is in the APK.
- The Kotlin coroutines/Flow façade lives in the separate, optional
  [`llama-kotlin`](../llama-kotlin) artifact.

Use `LlamaModel` exactly as on the JVM (see the core README). On Android the loader resolves the
native library via `System.loadLibrary("jllama")` from the APK's native-lib directory — where the
AAR's `jni/arm64-v8a/*.so` land; `JNI_OnLoad` then loads the backend modules from the same
directory by name.

> Do **not** combine this artifact with a `net.ladenthin:llama` JAR dependency in the same app:
> the AAR already contains those classes (and only the Android native library, whereas the JAR
> would drag ~70 MB of desktop natives into your APK as Java resources).

Models are ordinary GGUF files on device storage; download them at runtime (or bundle small ones
as assets and copy them to files dir) and pass the absolute path to `ModelParameters.setModel`.

## Two AAR flavors

| Artifact | Content | Requirement |
|---|---|---|
| `llama-android` | the library + the CPU backend modules | any arm64-v8a or x86_64 Android environment, API 28+ |
| `llama-android-opencl` | the OpenCL backend module only (Adreno-tuned kernels, arm64-v8a); depends on `llama-android` | device OpenCL ICD (`libOpenCL.so`) — Qualcomm Adreno drivers ship one. Without an ICD the module fails to load and the model runs on the CPU modules, with a log line; the artifact is additive, never a replacement |

## How this build works

This directory is a **standalone plain-Gradle build** (no Android Gradle Plugin, no Android SDK
required to build): an AAR is a documented zip, and Gradle's built-in `maven-publish` can publish
it with `<packaging>aar</packaging>` — which plain Maven cannot (`android-maven-plugin` is
unmaintained). It is intentionally *not* a Maven reactor module, but it stays version-locked to
the reactor: `build.gradle.kts` parses the version (and the mirrored dependency versions) out of
the Maven poms at configure time, so `mvn versions:set` remains the single bump point.

The AAR's `classes.jar` repackages the **byte-identical Maven-built core classes** (no
recompilation) minus `module-info.class`; the Android `.so` files ship under `jni/<abi>/`
instead of as Java resources (the desktop natives jars carry `jllama-files.txt` /
`jllama-build.txt` lists for the JVM loader, which an AAR does not need and does not contain).

### Building locally

```bash
# 1. Build the core jar (from the repo root)
mvn -pl llama -am -DskipTests package

# 2. Stage the Android native libraries (CI artifacts, or a dockcross build) -- every .so of
#    the natives-* artifact's directory:
#    natives/cpu/arm64-v8a/*.so       (libjllama, libggml, libggml-base, libggml-rpc, libggml-cpu-*)
#    natives/cpu/x86_64/*.so
#    natives/opencl/arm64-v8a/libggml-opencl.so

# 3. Assemble + publish to the local staging repo / mavenLocal
gradle -p llama-android aarCpu aarOpencl
gradle -p llama-android publishToMavenLocal
```

CI (`.github/workflows/publish.yml`) assembles both AARs from the freshly built native
artifacts, asserts the AAR structure and the 16 KB LOAD-segment alignment, and compiles a
minimal AGP consumer app against the published AAR as a smoke test.
