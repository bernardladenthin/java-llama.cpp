<p align="center">
  <img src="docs/branding/java-llama-cpp.svg" alt="java-llama.cpp" width="720">
</p>

> [!NOTE]
> **No IDE or local setup required.** This repository is optimized for fully AI-assisted development using [Claude Code](https://claude.ai/code). No local toolchain, no IDE, nothing to install — everything works completely through Claude.

**AI:**  
[![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff)](#)  

**Build:**  
![Java 8+](https://img.shields.io/badge/Java-8%2B-informational)  
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android-lightgrey)  
[![llama.cpp b10092](https://img.shields.io/badge/llama.cpp-%23b10092-informational)](https://github.com/ggml-org/llama.cpp/releases/tag/b10092)  
[![JPMS](https://img.shields.io/badge/JPMS-modular%20JAR-25A162)](https://openjdk.org/projects/jigsaw/)  
![JUnit](https://img.shields.io/badge/tested%20with-JUnit6-25A162)  
[![JSpecify](https://img.shields.io/badge/JSpecify-1.0.0%20%40NullMarked-25A162)](https://jspecify.dev)  
[![NullAway](https://img.shields.io/badge/NullAway-strict%20JSpecify-25A162)](https://github.com/uber/NullAway)  
[![Checker Framework](https://img.shields.io/badge/Checker%20Framework-Nullness-25A162)](https://checkerframework.org)  
[![Error Prone](https://img.shields.io/badge/Error%20Prone-12%20patterns%20at%20ERROR-25A162)](https://errorprone.info)  
[![Maven Enforcer](https://img.shields.io/badge/Maven%20Enforcer-strict-25A162)](https://maven.apache.org/enforcer/)  
[![Lombok](https://img.shields.io/badge/Lombok-1.18.46-bc3f3c)](https://projectlombok.org/)  
[![jqwik](https://img.shields.io/badge/tested%20with-jqwik-1f6feb)](https://jqwik.net)  
[![ArchUnit](https://img.shields.io/badge/tested%20with-ArchUnit-c71a36)](https://www.archunit.org)  
[![SpotBugs](https://img.shields.io/badge/analyzed%20with-SpotBugs-3b5998)](https://spotbugs.github.io)  
[![jcstress](https://img.shields.io/badge/tested%20with-jcstress-007396)](https://openjdk.org/projects/code-tools/jcstress/)  
[![Lincheck](https://img.shields.io/badge/tested%20with-Lincheck-7F52FF)](https://github.com/JetBrains/lincheck)  
[![vmlens](https://img.shields.io/badge/tested%20with-vmlens-ff6f00)](https://vmlens.com)  
[![JMH](https://img.shields.io/badge/benchmarked%20with-JMH-25A162)](https://openjdk.org/projects/code-tools/jmh/)  
[![Publish](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/publish.yml/badge.svg)](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/publish.yml)  
[![CodeQL](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/codeql.yml/badge.svg)](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/codeql.yml)  

**Build cache:**  
[![Build cache by Depot](https://img.shields.io/badge/build%20cache-Depot-FF5C35)](https://depot.dev)  

**Coverage:**  
[![Coverage Status](https://coveralls.io/repos/github/bernardladenthin/java-llama.cpp/badge.svg?branch=main)](https://coveralls.io/github/bernardladenthin/java-llama.cpp?branch=main)  
[![codecov](https://codecov.io/gh/bernardladenthin/java-llama.cpp/graph/badge.svg)](https://codecov.io/gh/bernardladenthin/java-llama.cpp)  
[![JaCoCo](https://img.shields.io/codecov/c/github/bernardladenthin/java-llama.cpp?label=JaCoCo&logo=java)](https://codecov.io/gh/bernardladenthin/java-llama.cpp)  
[![PIT Mutation](https://img.shields.io/badge/PIT%20mutation-100%25%20(1%20class)-brightgreen)](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/publish.yml)  

**Quality:**  
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=bernardladenthin_java-llama.cpp&metric=alert_status)](https://sonarcloud.io/dashboard?id=bernardladenthin_java-llama.cpp)  
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=bernardladenthin_java-llama.cpp&metric=code_smells)](https://sonarcloud.io/dashboard?id=bernardladenthin_java-llama.cpp)  
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=bernardladenthin_java-llama.cpp&metric=security_rating)](https://sonarcloud.io/dashboard?id=bernardladenthin_java-llama.cpp)  

**Security:**  
<!--
Coverity Scan is not configured for this repository.
To enable, register the project at https://scan.coverity.com/ and add a badge
using the assigned ID:
[![Coverity Scan Build Status](https://scan.coverity.com/projects/<ID>/badge.svg)](https://scan.coverity.com/projects/<ID>)
-->
[![Known Vulnerabilities](https://snyk.io/test/github/bernardladenthin/java-llama.cpp/badge.svg?targetFile=pom.xml)](https://snyk.io/test/github/bernardladenthin/java-llama.cpp?targetFile=pom.xml)  
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fbernardladenthin%2Fjava-llama.cpp.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fbernardladenthin%2Fjava-llama.cpp?ref=badge_shield)  
[![Dependencies](https://img.shields.io/librariesio/github/bernardladenthin/java-llama.cpp)](https://libraries.io/github/bernardladenthin/java-llama.cpp)  
[![OSV-Scanner](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/osv-scanner.yml/badge.svg)](https://github.com/bernardladenthin/java-llama.cpp/actions/workflows/osv-scanner.yml)  

**Package:**  
[![Maven Central](https://img.shields.io/maven-central/v/net.ladenthin/llama)](https://central.sonatype.com/artifact/net.ladenthin/llama)  
[![Snapshot](https://img.shields.io/badge/snapshot-latest-informational)](https://central.sonatype.com/repository/maven-snapshots/net/ladenthin/llama/)  
![Release Date](https://img.shields.io/github/release-date/bernardladenthin/java-llama.cpp)  
![Last Commit](https://img.shields.io/github/last-commit/bernardladenthin/java-llama.cpp)  

**License:**  
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)  

**Community:**  
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12862/badge)](https://www.bestpractices.dev/projects/12862)  
[![Contribute with Gitpod](https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/bernardladenthin/java-llama.cpp)  
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bernardladenthin/java-llama.cpp/badge)](https://scorecard.dev/viewer/?uri=github.com/bernardladenthin/java-llama.cpp)  
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-success?logo=dependabot)](./.github/dependabot.yml)  
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org/en/v1.0.0/)  
[![Keep a Changelog](https://img.shields.io/badge/changelog-Keep%20a%20Changelog-blue)](https://keepachangelog.com/en/1.1.0/)  
[![SemVer](https://img.shields.io/badge/SemVer-2.0.0-blue)](https://semver.org/spec/v2.0.0.html)  
[![REUSE](https://api.reuse.software/badge/github.com/bernardladenthin/java-llama.cpp)](https://api.reuse.software/info/github.com/bernardladenthin/java-llama.cpp)  
[![Maintained?](https://isitmaintained.com/badge/resolution/bernardladenthin/java-llama.cpp.svg)](https://isitmaintained.com/project/bernardladenthin/java-llama.cpp)  
[![Issues](https://img.shields.io/github/issues/bernardladenthin/java-llama.cpp)](https://github.com/bernardladenthin/java-llama.cpp/issues)  
[![Pull Requests](https://img.shields.io/github/issues-pr/bernardladenthin/java-llama.cpp)](https://github.com/bernardladenthin/java-llama.cpp/pulls)  
[![GitHub Stars](https://img.shields.io/github/stars/bernardladenthin/java-llama.cpp?style=social)](https://github.com/bernardladenthin/java-llama.cpp/stargazers)  
[![Treeware](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=Treeware&query=%24.total&url=https%3A%2F%2Fpublic.offset.earth%2Fusers%2Ftreeware%2Ftrees)](https://treeware.earth)  
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/StandWithUkraine.svg)](https://stand-with-ukraine.pp.ua)

# Java Bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)
> **Forked from** [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp): many thanks to [@kherud](https://github.com/kherud) for the great work!

Inference of Meta's LLaMA model (and others) in pure C/C++.

**You are welcome to contribute**

1. [Features](#features)
2. [Quick Start](#quick-start)  
    2.1 [No Setup required](#no-setup-required)   
    2.2 [Setup required](#setup-required)
3. [Documentation](#documentation)  
    3.1 [Example](#example)  
    3.2 [Inference](#inference)  
    3.3 [Chat Completion](#chat-completion)  
    3.4 [Infilling](#infilling)  
    3.5 [Embeddings & Reranking](#embeddings--reranking)  
    3.6 [Raw JSON Endpoints](#raw-json-endpoints)
4. [Android](#importing-in-android)
5. [Feature Ideas](#feature-ideas)

## Features

- Text completion (blocking and streaming) with full control over sampling parameters.
- OpenAI-compatible **chat completion** with automatic chat-template application, including streaming and tool/function calling support via the upstream server.
- **Embeddings** (single and native-batched via `embed(Collection<String>)`) and **reranking** for retrieval pipelines.
- **Runtime LoRA adapter control** — list the loaded adapters and change their scales at runtime without reloading the model (`getLoraAdapters()` / `setLoraAdapters(Map)`), the typed counterpart of the upstream `GET`/`POST /lora-adapters` endpoints.
- **Text-to-speech** (`TextToSpeech`) over the two-model OuteTTS + WavTokenizer pipeline, returning WAV audio.
- **In-JVM GGUF quantization** (`LlamaQuantizer`) over llama.cpp's `llama_model_quantize` — convert a GGUF to another quantization scheme without shelling out to `llama-quantize`.
- **Infilling** (fill-in-the-middle) for code models.
- **Tokenize / detokenize** and **JSON-schema → grammar** conversion.
- **Raw JSON endpoint handlers** mirroring the upstream llama.cpp HTTP server (`/completions`, `/v1/completions`, `/embeddings`, `/infill`, `/tokenize`, `/detokenize`).
- **Two runnable HTTP server modes, one fat-jar entry.** The fat jar's `Main-Class` is `ServerLauncher`, which dispatches on the `--jllama-openai-compat` flag. Without it, `java -jar …-jar-with-dependencies.jar -m model.gguf --port 8080` runs the full upstream llama.cpp server (embedded **WebUI**, every llama-server flag forwarded) hosted inside `libjllama` over JNI — no separate `llama-server.exe`. With it, `java -jar … --jllama-openai-compat --model model.gguf --port 8080` runs the Java-transport, zero-extra-dependency **OpenAI-compatible** server (`OpenAiCompatServer`, streaming SSE) instead. Both are also runnable directly by class name via `java -cp … net.ladenthin.llama.server.{NativeServer,OpenAiCompatServer}`.
- **Model metadata** access (`getModelMeta()`) and **server management** (metrics, slot save/restore, runtime thread reconfiguration).
- **Conversation checkpoints** — `Session.checkpoint(...)` / `rewind(...)` / `fork(...)` branch and roll back a chat (KV-cache slot save/restore + transcript snapshot) without re-prefilling.
- **GGUF metadata inspection** without loading the model (`GgufInspector` — pure Java, reads header + key/value table only, big-endian aware).
- **Multi-model router mode** (`--models-dir` + per-request model selection, managed via the typed `RouterClient`) and **attach mode** (`NativeServer(LlamaModel, ...)` serves an already-loaded model over the full upstream HTTP frontend — one copy of the weights).
- Pre-built native binaries in the default JAR for Linux (x86-64, aarch64, s390x), macOS (x86-64, arm64 — Metal included), Windows (x86-64, x86, arm64) and Android (arm64, x86-64); GPU backends (CUDA, Vulkan, OpenCL, ROCm/HIP, SYCL, OpenVINO) ship as Maven classifiers — see [Choosing the right classifier](#choosing-the-right-classifier). Android additionally ships as the [`llama-android` AAR](#importing-in-android) with the optional `llama-kotlin` coroutines façade.

## Quick Start

Access this library via Maven (released versions on Maven Central):

```xml
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama</artifactId>
    <version>5.0.6</version>
</dependency>
```

There are multiple [examples](llama/src/test/java/examples).

### Snapshot builds

Every push to `main` publishes a snapshot to the [Sonatype Central snapshot repository](https://central.sonatype.com/repository/maven-snapshots/net/ladenthin/llama/).

To use the latest snapshot, add the repository and dependency to your `pom.xml`:

```xml
<repositories>
  <repository>
    <id>sonatype-snapshots</id>
    <url>https://central.sonatype.com/repository/maven-snapshots/</url>
    <snapshots><enabled>true</enabled></snapshots>
    <releases><enabled>false</enabled></releases>
  </repository>
</repositories>

<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama</artifactId>
    <version>5.0.7-SNAPSHOT</version>
</dependency>
```

No credentials are required — the repository is publicly readable.

### No Setup required

We support CPU inference for the following platforms out of the box:

- Linux x86-64, aarch64
- MacOS x86-64, aarch64 (M-series)
- Windows x86-64, x64

If any of these match your platform, you can include the Maven dependency and get started.

### Choosing the right classifier

The Maven coordinate `net.ladenthin:llama` publishes one default JAR (CPU-only;
its Windows natives are built with the Ninja Multi-Config + MSVC toolchain) plus
optional JARs selected via a Maven `<classifier>`: NVIDIA CUDA (Linux / Windows),
Vulkan (Linux x86-64 / aarch64, Windows), AMD ROCm/HIP (Linux / Windows), Intel
SYCL (Linux fp16 / fp32, Windows) and OpenVINO (Linux / Windows) GPU builds, OpenCL
(Android Adreno, Windows x86-64 / Snapdragon-arm64), and an alternate-toolchain MSVC
Windows CPU build. Pick at most one GPU/accelerator classifier — those are mutually
exclusive — and optionally a CPU Windows build.

| Classifier | Backend | Target platform | Runtime requirement |
|---|---|---|---|
| _(none)_ | CPU | Linux x86-64 / aarch64 / s390x, macOS x86-64 / aarch64, Windows x86-64 / x86 / aarch64 (Ninja Multi-Config + MSVC), Android aarch64 + x86-64 (CPU) | A JDK 8+ JVM. **Linux `aarch64` additionally requires glibc ≥ 2.39** (e.g. Ubuntu 24.04+, Debian 13+) — it is built natively on `ubuntu-24.04-arm`, matching upstream llama.cpp's own ARM binaries; older-glibc ARM hosts (Ubuntu 22.04, Debian 12, RHEL 8/9, Amazon Linux 2023) are not supported. Linux x86-64 keeps a glibc 2.17 floor (manylinux2014). **Windows `aarch64`** (Windows on ARM — Snapdragon X / Surface) is built natively on `windows-11-arm` and ships in the default JAR alongside the x86-64 / x86 natives. |
| `msvc-windows` | CPU (MSVC / Visual Studio generator) | Windows x86-64 and x86 | None beyond a JDK 8+ JVM. Same CPU backend as the default JAR's Windows natives, but compiled with the Visual Studio generator instead of `Ninja Multi-Config`. Both use the same MSVC toolchain (static `/MT` CRT), so they are functionally equivalent — provided as an alternate-toolchain option. |
| `cuda13-windows-x86-64` | CUDA 13 | Windows x86-64 with NVIDIA GPU | NVIDIA driver + CUDA 13 Toolkit installed on the host (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` resolvable on `PATH`). The runtime libraries are **not bundled** in the JAR; native-library load fails with `UnsatisfiedLinkError` if they are absent. No CPU fallback. |
| `vulkan-windows-x86-64` | Vulkan | Windows x86-64 with a Vulkan 1.2+ GPU (NVIDIA / AMD / Intel) | A Vulkan runtime (`vulkan-1.dll`), which current GPU drivers install. No Vulkan SDK is needed at runtime. The most portable Windows GPU option (vendor-independent). |
| `opencl-windows-x86-64` | OpenCL | Windows x86-64 with an OpenCL 2.0+ GPU | A vendor OpenCL ICD (`OpenCL.dll`, installed by the GPU driver). **Note:** the GGML OpenCL backend is Adreno-tuned; on desktop GPUs CUDA or Vulkan are better supported. |
| `cuda13-linux-x86-64` | CUDA 13 | Linux x86-64 with NVIDIA GPU | NVIDIA driver + CUDA 13 runtime libraries (`libcudart.so.13`, `libcublas.so.13`) installed on the host. The shared library is dynamically linked against them and will fail to `dlopen` if they are absent — there is no automatic fallback to CPU. |
| `vulkan-linux-x86-64` | Vulkan | Linux x86-64 with a Vulkan 1.2+ GPU (NVIDIA / AMD / Intel) | A Vulkan runtime (`libvulkan.so.1`), which current GPU drivers install. No Vulkan SDK is needed at runtime. The most portable Linux GPU option (vendor-independent, no CUDA toolkit). Built natively on `ubuntu-latest`, so it shares the aarch64 build's higher glibc floor (≈ 2.39). |
| `vulkan-linux-aarch64` | Vulkan | Linux aarch64 with a Vulkan 1.2+ GPU | A Vulkan runtime (`libvulkan.so.1`) from the device/driver. glibc ≥ 2.39 (built on `ubuntu-24.04-arm`). |
| `opencl-android-aarch64` | OpenCL (Adreno) | Android aarch64 with Qualcomm Adreno GPU | A device-supplied OpenCL ICD (`libOpenCL.so`). Devices without an ICD (e.g. most non-Snapdragon Android hardware) must use the default CPU JAR. |
| `rocm-linux-x86-64` | ROCm / HIP | Linux x86-64 with AMD GPU | An installed AMD ROCm runtime (`libamdhip64.so`, `librocblas.so`, `libhipblas.so`) on the host. Not bundled; native load fails without it. No CPU fallback. |
| `rocm-windows-x86-64` | ROCm / HIP | Windows x86-64 with AMD GPU | The AMD HIP SDK runtime DLLs (`amdhip64.dll`, `rocblas.dll`, `hipblas.dll`) on `PATH`. Not bundled. No CPU fallback. |
| `sycl-fp16-linux-x86-64` | SYCL (Intel oneAPI, fp16) | Linux x86-64 with Intel GPU (Arc / iGPU) | An installed Intel oneAPI / Level-Zero runtime. fp16 accumulation (faster, slightly lower precision). Not bundled. |
| `sycl-fp32-linux-x86-64` | SYCL (Intel oneAPI, fp32) | Linux x86-64 with Intel GPU (Arc / iGPU) | An installed Intel oneAPI / Level-Zero runtime. fp32 accumulation (higher precision). Not bundled. |
| `sycl-windows-x86-64` | SYCL (Intel oneAPI) | Windows x86-64 with Intel GPU (Arc / iGPU) | The Intel oneAPI / Level-Zero runtime DLLs on `PATH`. Not bundled. |
| `opencl-windows-aarch64` | OpenCL (Adreno) | Windows-on-ARM aarch64 (Snapdragon X) with Adreno GPU | A device-supplied OpenCL ICD (`OpenCL.dll`, from the Adreno driver). Not bundled. |
| `openvino-linux-x86-64` | OpenVINO | Linux x86-64 (Intel GPU / NPU / CPU) | An installed Intel OpenVINO runtime. Not bundled. |
| `openvino-windows-x86-64` | OpenVINO | Windows x86-64 (Intel GPU / NPU / CPU) | The Intel OpenVINO runtime DLLs on `PATH`. Not bundled. |

> [!NOTE]
> The AMD (`rocm-*`), Intel SYCL (`sycl-*`), Windows-on-ARM OpenCL
> (`opencl-windows-aarch64`) and Intel OpenVINO (`openvino-*`) classifiers are
> newly added GPU backends. Like the other GPU classifiers they are validated
> **build-only** in CI (GitHub runners have no matching GPU), so end-to-end
> inference is verified locally / on self-hosted hardware. As with every GPU JAR,
> the vendor runtime is supplied by the consumer's driver/toolkit and is not bundled.

For the default CPU JAR, omit the `<classifier>`. For a GPU/accelerator or
alternate-CPU build, add the `<classifier>` for your platform from the table
above — the backend, target platform and runtime requirement are all listed
there. Pick **at most one** classifier (they are mutually exclusive):

```xml
<!-- Default (CPU) — no classifier -->
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama</artifactId>
    <version>5.0.6</version>
</dependency>

<!-- GPU / accelerator or alternate-CPU build: add the <classifier> from the
     table above. Example shown — CUDA 13 on Linux x86-64. -->
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama</artifactId>
    <version>5.0.6</version>
    <classifier>cuda13-linux-x86-64</classifier>
</dependency>
```

> [!IMPORTANT]
> The GPU JARs are **GPU-only at runtime**. On a host without the matching
> GPU driver/runtime the JVM fails at native-library load time with
> `UnsatisfiedLinkError`: the CUDA JARs are dynamically linked against the
> CUDA runtime (`libcudart.so.13` on Linux, `cudart64_13.dll` /
> `cublas64_13.dll` / `cublasLt64_13.dll` on Windows — the Windows CUDA
> runtime is **not bundled**, install the CUDA 13 Toolkit), the Vulkan JAR
> needs a Vulkan runtime (`vulkan-1.dll`, shipped with current GPU drivers),
> and the OpenCL JARs need a vendor OpenCL ICD. There is no automatic
> fallback to CPU. If you want a single artifact that works on both CPU and
> GPU hosts, depend on the default (CPU) JAR; users who want GPU acceleration
> on an unlisted platform must compile locally with the matching `-DGGML_*=ON`
> flag (see [Setup required](#setup-required)).

> [!NOTE]
> Android `armeabi-v7a` (32-bit ARM) is **not** published. Only 64-bit
> Android binaries are shipped: `aarch64` (devices) and `x86_64`
> (emulators, Chromebooks, x86-64 Android hardware) in the CPU-only default
> JAR and the `llama-android` AAR, plus `aarch64` as
> `opencl-android-aarch64`. 32-bit Android devices are unsupported
> by the released artifacts; building from source via the
> `.github/dockcross/dockcross-android-arm` toolchain is possible but not
> wired into CI.
>
> The minimum required Android version is **API 28 (Android 9.0 Pie)**.
> Devices running Android 8.1 (API 27) or earlier are not supported.

### Standalone server fat jars (GitHub Releases)

For running the [embedded server](#native-server-with-the-built-in-webui-nativeserver)
without any Maven setup, every tagged [GitHub Release](https://github.com/bernardladenthin/java-llama.cpp/releases)
(and the rolling `snapshot` pre-release from `main`) attaches self-contained
**all-backends fat jars** — one download per OS/arch, runnable directly:

```bash
java -jar llama-<version>-all-linux-x86-64-jar-with-dependencies.jar -m model.gguf --port 8080
```

| Release asset | Bundled GPU backends | CPU fallback |
|---|---|---|
| `llama-<version>-all-linux-x86-64-jar-with-dependencies.jar` | CUDA 13, ROCm, SYCL fp16/fp32, Vulkan, OpenVINO | yes |
| `llama-<version>-all-linux-aarch64-jar-with-dependencies.jar` | Vulkan | yes |
| `llama-<version>-all-windows-x86-64-jar-with-dependencies.jar` | CUDA 13, ROCm, SYCL, Vulkan, OpenCL, OpenVINO | yes |
| `llama-<version>-all-windows-aarch64-jar-with-dependencies.jar` | OpenCL (Adreno / Snapdragon X) | yes |
| `llama-<version>-jar-with-dependencies.jar` | none (CPU only, incl. macOS Metal) | — |

Each all-backends jar contains the library classes, all Java runtime
dependencies, the default CPU natives for **every** platform, and every GPU
backend for its named OS/arch. At startup the loader tries the bundled backends
in priority order (CUDA → ROCm → SYCL → Vulkan → OpenCL → OpenVINO) and uses
the **first one whose native library loads**; if none loads — e.g. no GPU
driver/toolkit installed — it falls back to the CPU natives, so the jar starts
everywhere. The usual GPU policy applies: vendor runtimes are **not** bundled
(see the classifier table above for what each backend needs on the host).
Force a specific backend with `-Dnet.ladenthin.llama.backend=<name>`
(e.g. `vulkan`; fails loud instead of falling back) or force CPU with
`-Dnet.ladenthin.llama.backend=default`. A `.sha256` checksum file accompanies
every jar. These fat jars are GitHub download assets only — they are **not**
published to Maven Central (Maven users combine the thin classifier jars
instead, see above).

### Setup required

If none of the above listed platforms matches yours, currently you have to compile the library yourself (also if you 
want GPU acceleration).

This consists of two steps: 1) Compiling the libraries and 2) putting them in the right location.

##### Library Compilation

First, have a look at [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) to know which build arguments to use (e.g. for CUDA support).
Any build option of llama.cpp works equivalently for this project.
You then have to run the following commands in the `llama/` module directory (the native core lives
there; the repository root is just the Maven reactor aggregator):

```shell
cd llama       # the native core module
mvn compile    # don't forget this line
cmake -B build # add any other arguments for your backend, e.g. -DGGML_CUDA=ON
cmake --build build --config Release
```

> [!TIP]
> Use `-DLLAMA_CURL=ON` to download models via Java code using `ModelParameters#setModelUrl(String)`.

All compiled libraries will be put in a resources directory matching your platform, which will appear in the cmake output. For example something like:

```shell
--  Installing files to /java-llama.cpp/llama/src/main/resources/net/ladenthin/llama/Linux/x86_64
```

#### Library Location

This project has to load a single shared library `jllama`.

Note, that the file name varies between operating systems, e.g., `jllama.dll` on Windows, `jllama.so` on Linux, and `jllama.dylib` on macOS.

The application will search in the following order in the following locations:

- In **net.ladenthin.llama.lib.path**: Use this option if you want a custom location for your shared libraries, i.e., set VM option `-Dnet.ladenthin.llama.lib.path=/path/to/directory`.
- In **java.library.path**: These are predefined locations for each OS, e.g., `/usr/java/packages/lib:/usr/lib64:/lib64:/lib:/usr/lib` on Linux.
  You can find out the locations using `System.out.println(System.getProperty("java.library.path"))`.
  Use this option if you want to install the shared libraries as system libraries.
- From the **JAR**: If any of the libraries weren't found yet, the application will try to use a prebuilt shared library.
  This of course only works for the [supported platforms](#no-setup-required) .

#### System Properties Reference

Every `net.ladenthin.llama.*` system property recognised by the library, deep-scanned from the source. Runtime properties are resolved through `LlamaSystemProperties`; test-only properties are declared in the test sources (`TestConstants`) and consumed by individual test classes.

| Property | Default | Scope | Consumer | Description |
|---|---|---|---|---|
| `net.ladenthin.llama.lib.path` | unset (falls back to `java.library.path`) | runtime | `LlamaLoader` | Directory containing the native `jllama` shared library. Checked first, before `java.library.path`. Set with `-Dnet.ladenthin.llama.lib.path=/path/to/dir`. |
| `net.ladenthin.llama.tmpdir` | unset (falls back to `java.io.tmpdir`) | runtime | `LlamaLoader` | Custom temporary directory used when extracting the native library from the JAR. |
| `net.ladenthin.llama.osinfo.architecture` | unset (uses `os.arch`) | runtime | `OSInfo` | Override for the architecture string used to locate the bundled library inside the JAR. Useful when `os.arch` reports an unexpected value (e.g. inside dockcross / chrooted environments). |
| `net.ladenthin.llama.backend` | unset (auto: first loadable backend, then CPU) | runtime | `LlamaLoader` | Backend override for the [all-backends fat jars](#standalone-server-fat-jars-github-releases) (jars carrying a `jllama-backends.txt` manifest). Names one bundled backend (e.g. `cuda13`, `vulkan`) to load exclusively — failure is then fatal instead of falling back — or `default`/`cpu` to skip all GPU backends. Ignored by jars without a backend manifest. |
| `net.ladenthin.llama.test.ngl` | `43` for the general suite; `0` for `ToolCallingIntegrationTest` | test | Model-backed integration tests | Number of GPU layers used during testing. Pin to `0` on CPU-only hosts: `mvn test -Dnet.ladenthin.llama.test.ngl=0`. The tool test also selects device `none` at zero layers so Metal/CUDA is not initialized. |
| `net.ladenthin.llama.tool.model` | `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (test self-skips if missing) | test | `ToolCallingIntegrationTest` | Path to a tool-capable GGUF used to verify required blocking and streaming tool calls. The default matches the Qwen2.5 model in upstream llama.cpp's tool-call test matrix. |
| `net.ladenthin.llama.nomic.path` | unset (test self-skips) | test | `LlamaEmbeddingsTest#testNomicEmbedLoads` | Path to a Nomic embedding model (`nomic-embed-text-v1.5.f16.gguf` or a compatible BERT-family encoder). Regression test for upstream issue #98 (BERT-encoder `result_output` assertion). |
| `net.ladenthin.llama.vision.model` | unset (test self-skips) | test | `MultimodalIntegrationTest` | Path to a vision-capable model GGUF. Any vision-capable GGUF works; CI default is `SmolVLM-500M-Instruct-Q8_0.gguf`. |
| `net.ladenthin.llama.vision.mmproj` | unset (test self-skips) | test | `MultimodalIntegrationTest` | Matching mmproj GGUF for the vision model. |
| `net.ladenthin.llama.vision.image` | `llama/src/test/resources/images/test-image.jpg` (a CC-BY-4.0 / MIT-granted photo committed to the repo) | test | `MultimodalIntegrationTest` | Visual prompt image. Any png/jpeg/webp/gif works; the extension drives MIME detection. |
| `net.ladenthin.llama.audio.model` | unset (test self-skips) | test | `AudioInputIntegrationTest` (llama.cpp discussion #13759) | Path to an audio-input model GGUF (e.g. Ultravox, Qwen2.5-Omni). |
| `net.ladenthin.llama.audio.mmproj` | unset (test self-skips) | test | `AudioInputIntegrationTest` | Matching audio mmproj (encoder) GGUF. |
| `net.ladenthin.llama.audio.input` | `src/test/resources/audios/sample.wav` (committed) | test | `AudioInputIntegrationTest` | `.wav`/`.mp3` audio prompt clip; the extension drives format detection. |
| `net.ladenthin.llama.tts.ttc.model` | unset (test self-skips) | test | `TtsIntegrationTest` | Path to the OuteTTS text-to-codes GGUF. CI default is `OuteTTS-0.2-500M-Q4_K_M.gguf`. |
| `net.ladenthin.llama.tts.vocoder.model` | unset (test self-skips) | test | `TtsIntegrationTest` | Path to the matching codes-to-speech vocoder GGUF. CI default is `WavTokenizer-Large-75-F16.gguf`. |

`MultimodalIntegrationTest` self-skips when any of the three `vision.*` properties points at a missing path, so a partial setup (just the vision model + the committed image, no mmproj) lets the test class load without erroring. `AudioInputIntegrationTest` self-skips the same way over the three `audio.*` properties. `TtsIntegrationTest` likewise self-skips unless both `tts.ttc.model` and `tts.vocoder.model` point at existing files.

## Documentation

### Example

This is a short example on how to use this library:

```java
public class Example {

    public static void main(String... args) throws IOException {
        ModelParameters modelParams = new ModelParameters()
                .setModel("models/mistral-7b-instruct-v0.2.Q2_K.gguf")
                .setGpuLayers(43);

        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
                "requests immediately and with precision.\n";
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        try (LlamaModel model = new LlamaModel(modelParams)) {
            System.out.print(system);
            String prompt = system;
            while (true) {
                prompt += "\nUser: ";
                System.out.print("\nUser: ");
                String input = reader.readLine();
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
                InferenceParameters inferParams = new InferenceParameters(prompt)
                        .setTemperature(0.7f)
                        .setPenalizeNl(true)
                        .setMiroStat(MiroStat.V2)
                        .setStopStrings("User:");
                for (LlamaOutput output : model.generate(inferParams)) {
                    System.out.print(output);
                    prompt += output;
                }
            }
        }
    }
}
```

Also have a look at the other [examples](llama/src/test/java/examples).

### Inference

There are multiple inference tasks. In general, `LlamaModel` is stateless, i.e., you have to append the output of the 
model to your prompt in order to extend the context. If there is repeated content, however, the library will internally
cache this, to improve performance.

```java
ModelParameters modelParams = new ModelParameters().setModel("/path/to/model.gguf");
InferenceParameters inferParams = new InferenceParameters("Tell me a joke.");
try (LlamaModel model = new LlamaModel(modelParams)) {
    // Stream a response and access more information about each output.
    for (LlamaOutput output : model.generate(inferParams)) {
        System.out.print(output);
    }
    // Calculate a whole response before returning it.
    String response = model.complete(inferParams);
    // Returns the hidden representation of the context + prompt.
    float[] embedding = model.embed("Embed this");
}
```

> [!NOTE]
> Since llama.cpp allocates memory that can't be garbage collected by the JVM, `LlamaModel` is implemented as an
> AutoClosable. If you use the objects with `try-with` blocks like the examples, the memory will be automatically
> freed when the model is no longer needed. This isn't strictly required, but avoids memory leaks if you use different
> models throughout the lifecycle of your application.

### Chat Completion

For chat models, build a list of role/content pairs and let the library apply the model's chat template.
`chatComplete()` returns the full response, `generateChat()` streams tokens, and `chatCompleteText()` returns
just the text content of the assistant message.

```java
List<Pair<String, String>> messages = new ArrayList<>();
messages.add(new Pair<>("user", "Write a haiku about Java."));

InferenceParameters inferParams = new InferenceParameters("")
        .setMessages("You are a helpful assistant.", messages)
        .setUseChatTemplate(true);

try (LlamaModel model = new LlamaModel(modelParams)) {
    // Streaming
    for (LlamaOutput output : model.generateChat(inferParams)) {
        System.out.print(output);
    }
    // Or blocking, returns the OpenAI-compatible JSON envelope
    String json = model.chatComplete(inferParams);
    // Or just the assistant text
    String text = model.chatCompleteText(inferParams);
}
```

Reasoning/thinking models can receive custom Jinja template variables via
`ModelParameters#setChatTemplateKwargs(Map)`.

### Vision / Multimodal Chat

Load a vision-capable GGUF with its matching projector, then place text and image parts in the
same user message. Images may come from a file, raw bytes, a data URI, or an HTTP(S) URL:

```java
ModelParameters modelParams = new ModelParameters()
        .setModel("models/SmolVLM-500M-Instruct-Q8_0.gguf")
        .setMmproj("models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf");

ChatMessage message = ChatMessage.userMultimodal(
        ContentPart.text("Describe this image in one short sentence."),
        ContentPart.imageFile(Paths.get("photo.jpg")));

try (LlamaModel model = new LlamaModel(modelParams)) {
    String answer = model.chatCompleteText(InferenceParameters.empty()
            .withMessages(Collections.singletonList(message))
            .withNPredict(64));
    System.out.println(answer);
}
```

The same multipart `messages[].content` shape works through `ChatRequest` and the embedded
OpenAI-compatible `/v1/chat/completions` server. For a strictly CPU-only run, use
`setDevices("none").setMmprojOffload(false)` in addition to `setGpuLayers(0)`; projector offload
has its own upstream default.

**Audio input** works identically — load an audio-capable model (Ultravox, Qwen2.5-Omni, …) with its
audio `--mmproj` and add a `ContentPart.audioFile(...)` (or `inputAudio(bytes, "wav"|"mp3")`) part. It
serializes to the OpenAI `input_audio` content part and routes through the same `mtmd` pipeline:

```java
ModelParameters modelParams = new ModelParameters()
        .setModel("models/ultravox-v0_5-llama-3_2-1b.gguf")
        .setMmproj("models/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf");

ChatMessage message = ChatMessage.userMultimodal(
        ContentPart.text("Transcribe the audio."),
        ContentPart.audioFile(Paths.get("speech.wav")));

try (LlamaModel model = new LlamaModel(modelParams)) {
    System.out.println(model.supportsAudio()); // true
    String answer = model.chatCompleteText(InferenceParameters.empty()
            .withMessages(Collections.singletonList(message))
            .withNPredict(64));
    System.out.println(answer);
}
```

`LlamaModel.supportsVision()` / `supportsAudio()` report which modalities the loaded projector enables.

### Tool Calling

Use a tool-aware instruct model and enable Jinja when loading it. A typed request can either return
the model's tool calls through `chat`, or execute registered handlers until the model produces a
normal assistant response through `chatWithTools`:

```java
ToolDefinition weather = new ToolDefinition(
        "get_weather",
        "Get the current weather for a city",
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
                + "\"required\":[\"city\"]}");

ChatRequest request = ChatRequest.empty()
        .appendMessage("user", "What is the weather in Paris?")
        .appendTool(weather)
        .withToolChoice("auto")
        .withParallelToolCalls(Boolean.FALSE);

Map<String, ToolHandler> handlers = Collections.singletonMap(
        "get_weather", argumentsJson -> "{\"temperature_c\":21,\"condition\":\"sunny\"}");

try (LlamaModel model = new LlamaModel(new ModelParameters()
        .setModel("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
        .enableJinja())) {
    ChatResponse response = model.chatWithTools(request, handlers);
    System.out.println(response.getFirstContent());
}
```

`tool_choice` is the OpenAI-compatible string form (`auto`, `none`, or `required`). Set
`parallel_tool_calls` to `false` when handlers should be issued one at a time. Handler failures and
unknown tool names are returned to the model as valid `{"error":"..."}` tool-result JSON.

### Infilling

You can simply set `InferenceParameters#setInputPrefix(String)` and `InferenceParameters#setInputSuffix(String)`.

### Embeddings & Reranking

Load the model with `enableEmbedding()` (or `enableReranking()`) and call `embed(String)` to get a sentence
embedding, or `rerank(query, documents...)` to get relevance scores.

```java
ModelParameters modelParams = new ModelParameters()
        .setModel("/path/to/embedding-model.gguf")
        .enableEmbedding();
try (LlamaModel model = new LlamaModel(modelParams)) {
    float[] embedding = model.embed("Embed this sentence");
    // Batch form: one native dispatch for many inputs, results in request order.
    List<float[]> embeddings = model.embed(Arrays.asList("First sentence", "Second sentence"));
}
```

### Runtime LoRA adapter control

Adapters loaded at model-load time (`addLoraAdapter(...)` / `addLoraScaledAdapter(...)`, optionally
`setLoraInitWithoutApply()` to start disabled) can be listed and re-scaled at runtime without
reloading the model — the typed counterpart of the upstream `GET`/`POST /lora-adapters` endpoints:

```java
ModelParameters modelParams = new ModelParameters()
        .setModel("models/base.gguf")
        .addLoraScaledAdapter("models/adapter.gguf", 1.0f);
try (LlamaModel model = new LlamaModel(modelParams)) {
    List<LoraAdapter> adapters = model.getLoraAdapters();      // [{id=0, path=..., scale=1.0}]
    model.setLoraAdapter(0, 0.5f);                             // re-scale at runtime
    model.setLoraAdapters(Collections.emptyMap());             // disable all adapters
}
```

Per the upstream contract, a scale update lists the adapters to keep active — any adapter missing
from the map is set to scale `0` (disabled). The native side clears affected KV caches when the
effective adapter set changes.

### Text-to-Speech

`TextToSpeech` synthesizes audio from text over llama.cpp's OuteTTS pipeline. It is a separate
`AutoCloseable` native type (not a `LlamaModel`) because TTS is a **two-model** pipeline: a
text-to-codes model (OuteTTS) and a codes-to-speech vocoder (WavTokenizer). `synthesize(String)`
returns a 24&nbsp;kHz mono 16-bit WAV byte stream.

```java
try (TextToSpeech tts = new TextToSpeech(
        "models/OuteTTS-0.2-500M-Q4_K_M.gguf",
        "models/WavTokenizer-Large-75-F16.gguf")) {
    byte[] wav = tts.synthesize("Hello from llama dot c p p.");
    Files.write(Paths.get("out.wav"), wav);
}
```

Add `(ttcPath, vocoderPath, gpuLayers, threads)` to offload to the GPU, or
`synthesize(text, maxCodeTokens, topK, seed)` for explicit sampling. As with `LlamaModel`, native
memory is not GC-managed — use try-with-resources or call `close()`. Synthesis uses the built-in
default speaker profile; English number words are expanded for speech (`3` → "three"), and
non-English text is not romanized.

Compatible GGUFs (the CI test defaults): OuteTTS
[`OuteTTS-0.2-500M-GGUF`](https://huggingface.co/second-state/OuteTTS-0.2-500M-GGUF) +
[`WavTokenizer`](https://huggingface.co/ggml-org/WavTokenizer).

### GGUF Quantization

`LlamaQuantizer` converts a GGUF to another quantization scheme in-process (llama.cpp's
`llama_model_quantize` — the `llama-quantize` tool without the separate binary):

```java
LlamaQuantizer.quantize("model-f16.gguf", "model-q4_k_m.gguf", QuantizationType.Q4_K_M);
// Re-quantizing an already-quantized GGUF degrades quality and must be opted into:
LlamaQuantizer.quantize("model-q8_0.gguf", "model-q4_0.gguf", QuantizationType.Q4_0,
        /* threads */ 0, /* allowRequantize */ true);
```

### Raw JSON Endpoints

For direct access to the upstream llama.cpp server API, the following methods take a JSON request and return
a JSON response, matching the HTTP server's contract:

`handleCompletions`, `handleCompletionsOai`, `handleChatCompletions`, `handleInfill`,
`handleEmbeddings`, `handleTokenize`, `handleDetokenize`.

Server state is exposed via `getMetrics()`, `eraseSlot(int)`, `saveSlot(int, String)`,
`restoreSlot(int, String)`, and `getModelMeta()`.

### Conversation checkpoints: rewind + fork (`Session`)

A `Session` can be snapshotted and branched — the KV-cache slot state and the transcript move
together, so native state and history can never drift apart:

```java
try (Session session = new Session(model, 0, "You are terse.")) {
    session.send("My name is Alice.");
    SessionCheckpoint cp = session.checkpoint("checkpoints/turn1.bin");

    session.send("Tell me a joke.");
    session.rewind(cp);                     // undo everything after the checkpoint
    session.send("Tell me a story instead."); // retry from the branch point

    // Branch into a second slot (model loaded with setParallel(2)+):
    try (Session forked = session.fork(1, "checkpoints/branch.bin")) {
        forked.send("Answer as a pirate.");   // both sessions continue independently
    }
}
```

Checkpoint files are caller-managed (KV dumps grow with context usage) and both operations are
rejected while a stream is in progress. For plain transformer models a rewind is also achievable
cheaply by resending a truncated history with `cache_prompt` (prefix reuse); checkpoints make the
branch point exact and are the only reliable rollback for recurrent/hybrid models (e.g.
Granite-4), whose state cannot be recomputed from a prefix.

### GGUF metadata inspection (no model load)

`GgufInspector` reads a GGUF's header and key/value table **without loading the model** — pure
Java, no native library, cost independent of file size (parsing stops before the tensor data).
Useful for model pickers and download validators:

```java
GgufMetadata meta = GgufInspector.read(Paths.get("models/Qwen3-0.6B-Q4_K_M.gguf"));
meta.getArchitecture();   // Optional[qwen3]
meta.getModelName();      // Optional[Qwen3 0.6B]
meta.getParameterCount(); // OptionalLong[751632384]
meta.getContextLength();  // OptionalLong[40960]  (<arch>.context_length)
meta.getFileType();       // OptionalLong[15]     (llama_ftype, cf. QuantizationType)
meta.getChatTemplate();   // Optional[{{- ... }}]
meta.getEntries();        // full decoded key/value table
```

Supports GGUF v2/v3, little- and big-endian (auto-detected), and fails loud on v1/corrupt files.
For metadata of an already-loaded model use `getModelMeta()` instead.

### Prompt and KV Cache Reuse

Prompt-prefix reuse is enabled by default in llama.cpp and can be controlled per request with
`InferenceParameters.withCachePrompt(boolean)`. `withCacheReuse(int)` enables non-prefix chunk reuse,
while `withSlotId(int)` pins a request to a specific server slot. `Session` applies its slot id to every
request, so generation and `save`/`restore` operate on the same KV state.

Typed results expose logical prompt, generated, cached prompt, and evaluated prompt counts through
`Usage`. Per-request timing also remains available through `Timings.getCacheN()`.
`LlamaModel.getMetricsTyped().getSlotMetrics()` reports each slot's logical, processed, cached,
decoded, and remaining token counts.

The embedded HTTP server exposes the same native JSON at authenticated `GET /metrics`, with the slot
array alone at `GET /slots`. OpenAI responses preserve
`usage.prompt_tokens_details.cached_tokens`; Responses API output uses
`usage.input_tokens_details.cached_tokens`; Anthropic output uses `cache_read_input_tokens`.

### OpenAI-compatible HTTP server

`net.ladenthin.llama.server.OpenAiCompatServer` turns a loaded model into a local
OpenAI-compatible HTTP endpoint using only the JDK's built-in `com.sun.net.httpserver` — no extra
dependency and no separate server process. It is embeddable, and runnable via
`java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …` (the fat jar's default
`Main-Class` is instead `NativeServer` — see "Native server with the built-in WebUI" below). It
serves:

| Method &amp; path | Backed by |
|---|---|
| `POST /v1/chat/completions` | `LlamaModel.streamChatCompletion` (streaming SSE) / `chatComplete` (blocking) |
| `POST /v1/completions` | `LlamaModel.handleCompletionsOai` |
| `POST /v1/embeddings` (requires `--embedding`) | `LlamaModel.handleEmbeddings` |
| `POST /v1/rerank` (requires `--reranking`) | `LlamaModel.handleRerank` (reshaped to `results`/`data`) |
| `POST /infill` | `LlamaModel.handleInfill` (fill-in-the-middle autocomplete) |
| `GET /v1/models` | the configured model id |
| `GET /metrics` | native server and per-slot token/cache counters (JSON) |
| `GET /slots` | native per-slot token/cache counters (JSON array) |
| `GET /health` | static `{"status":"ok"}` (unauthenticated) |

Chat completions support **streaming via Server-Sent Events** and non-streaming, forwarding
`messages`/`tools` verbatim. The streaming path carries `delta.tool_calls` and (with
`stream_options.include_usage`) a trailing `usage` chunk, so **agent/tool-calling clients work** —
this is the recommended surface for VS Code Copilot agent mode, Cline, Roo Code and Continue.
`response_format` (`json_object` / `json_schema`) is forwarded for structured outputs. Completions,
embeddings, rerank and infill are non-streaming.

Every route is also reachable **without the `/v1` prefix**, the server answers **CORS preflight**
(`OPTIONS`) and stamps `Access-Control-Allow-Origin` (so browser/webview clients work), and
`POST /infill` is the llama.cpp-native FIM endpoint for local ghost-text autocomplete plugins
(llama.vscode, Twinny, Tabby, Continue's `llama.cpp` provider). Note: GitHub Copilot's **inline**
completions cannot be served by any local endpoint — only its chat/agent surfaces — so use one of
those autocomplete plugins for ghost text.

**Alternative protocol surfaces.** For clients that don't speak OpenAI Chat Completions, the same model
is exposed through additional protocols (pure translation over the OpenAI core — no extra inference
path), all supporting tools and streaming:

| Surface | Routes | For |
|---|---|---|
| **Ollama-native** | `GET /api/version`, `GET /api/tags`, `POST /api/show`, `POST /api/chat` (NDJSON streaming), `POST /api/generate` (prompt completion / FIM) | Copilot's built-in **Ollama** provider; Ollama-hardcoded tools |
| **Anthropic Messages** | `POST /v1/messages` (SSE event stream) | Claude-shaped clients (Claude Code); Copilot `messages` apiType |
| **OpenAI Responses** | `POST /v1/responses` (SSE event stream) | Copilot `responses` apiType; Responses-API clients |

`/api/show` advertises the model's capabilities (`tools`, `insert`, and `vision` when `--mmproj` is set)
and context length, which Copilot's Ollama provider reads to enable agent mode. The llama.cpp-native
`GET /props` reports `default_generation_settings.n_ctx` and a `modalities` block, which autocomplete
clients such as llama.vscode read to size their context window.

Embed it in your app:

```java
ModelParameters modelParams = new ModelParameters().setModel("models/model.gguf").setParallel(2);
OpenAiServerConfig config = OpenAiServerConfig.builder().port(8080).modelId("local-model").build();
try (LlamaModel model = new LlamaModel(modelParams);
     OpenAiCompatServer server = new OpenAiCompatServer(model, config).start()) {
    Thread.currentThread().join(); // serve until interrupted
}
```

…or run it standalone. The fat jar's `Main-Class` is the `ServerLauncher` dispatcher, so add
`--jllama-openai-compat` to select this Java server (the launcher strips that flag and forwards the rest);
or name the class explicitly via `-cp`:

```bash
# fat jar (bundles the native lib + Java deps) — select the Java server with --jllama-openai-compat
java -jar target/llama-<version>-jar-with-dependencies.jar --jllama-openai-compat \
    --model models/Qwen3-0.6B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers 99

# or name the class explicitly (fat jar or plain library jar)
java -cp target/llama-<version>.jar net.ladenthin.llama.server.OpenAiCompatServer \
  --model models/model.gguf --port 8080 --model-id local-model
```

Run with `--help` for the full option list (`-m/--model`, `--host`, `-p/--port`, `-c/--ctx-size`,
`-b/--batch-size`, `-ub/--ubatch-size`, `-ngl/--n-gpu-layers`, `-t/--threads`, `-tb/--threads-batch`,
`-ctk/--cache-type-k`, `-ctv/--cache-type-v`, `--jinja`, `--chat-template-kwargs`, `--parallel`,
`--model-id`, `--api-key`, `--mmproj`, `--embedding`, `--reranking`). The tuning flags mirror
llama.cpp's server, so an invocation like
`--jinja --chat-template-kwargs '{"reasoning_effort":"low"}' -ctk q8_0 -ctv q8_0 -b 4096 -ub 2048`
works directly.

Verify with curl (streaming chat):

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

**VS Code Copilot setup:** Command Palette → **Chat: Manage Language Models** → **Add Models** →
**Custom Endpoint**; enter a group name, a display name and any non-empty API key, and pick API type
**Chat Completions**. VS Code then opens `chatLanguageModels.json` — set the model `url` to your
endpoint (the host/port go here, not in the form):

```json
[
  {
    "name": "Local llama.cpp",
    "vendor": "customendpoint",
    "apiKey": "local-dummy-key",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "local-model",
        "name": "Local model",
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 6144,
        "maxOutputTokens": 2048
      }
    ]
  }
]
```

Notes: BYOK powers the chat/agent experience only (inline completions and embeddings still require a
GitHub account). On CPU, prefer a smaller model and a modest context window — the server emits SSE
heartbeats so a long prompt prefill does not trip the client's stream-inactivity timeout. Agent-mode
tool calling depends on the model's own tool-calling quality. Pass `--api-key` (or
`OpenAiServerConfig.apiKey(...)`) to require an `Authorization: Bearer` token; the server binds to
`127.0.0.1` by default.

### Native server with the built-in WebUI (`NativeServer`)

`OpenAiCompatServer` above is a JSON **API** server (its `/` is a 404 — no web page). If you want
the **full upstream llama.cpp server, including its bundled Svelte WebUI**, use
`net.ladenthin.llama.server.NativeServer`. It runs the real `llama_server` inside `libjllama` over
JNI — no separate `llama-server.exe` — and **forwards the raw llama-server arguments verbatim**, so
every flag works exactly as it does for the standalone binary. The fat jar runs it **by default**
(when `--jllama-openai-compat` is absent), forwarding its args to the native server (pass `--help` for the
full llama-server option list):

```bash
java -jar target/llama-<version>-jar-with-dependencies.jar \
    -m models/model.gguf --host 127.0.0.1 --port 8080 -c 65536 --jinja
# then open http://127.0.0.1:8080/ for the WebUI
```

Or embed it:

```java
try (NativeServer server = new NativeServer(
        "-m", "gpt-oss-20b-UD-Q4_K_XL.gguf",
        "--host", "127.0.0.1", "--port", "8080",
        "-c", "65536", "-b", "4096", "-ub", "2048",
        "--jinja", "-ngl", "0", "-t", "8", "-tb", "16",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--chat-template-kwargs", "{\"reasoning_effort\":\"low\"}",
        "--parallel", "1").start()) {
    // Open http://127.0.0.1:8080/ in a browser for the WebUI; the OpenAI API is at /v1/... too.
    Thread.currentThread().join();
}
```

Differences from `OpenAiCompatServer`: with the classic constructor it **loads its own model** from
the arguments (an independent lifecycle, like `llama-server.exe`), it is **single-instance per
process**, it serves the **WebUI** (in released jars — local `cmake` builds ship the empty-asset
stub, so no UI there), and it is **not available on Android** (the upstream server needs
`posix_spawn`). Readiness: poll `GET /health`. No SSL (plain HTTP — bind localhost or front with a
TLS proxy).

#### Attach mode — serve an already-loaded `LlamaModel`

`NativeServer` can also **attach** the full upstream HTTP frontend (routes, WebUI, resumable
streaming) to a `LlamaModel` you already loaded — one copy of the weights, shared between direct
JNI calls and HTTP:

```java
try (LlamaModel model = new LlamaModel(new ModelParameters().setModel("models/model.gguf"));
     NativeServer server = new NativeServer(model, "--host", "127.0.0.1", "--port", "8080").start()) {
    // HTTP (incl. WebUI in released jars) and direct Java calls share the same loaded model.
    String direct = model.complete(new InferenceParameters("2+2=").withNPredict(4));
    Thread.currentThread().join();
}
```

In attach mode the arguments carry only the HTTP-side flags (`--host`, `--port`, `--api-key`, …;
no `-m`), the server reports healthy immediately (the model is already loaded), and the **caller
keeps ownership of the model** — close the server before the model, never the other way around.

#### Router mode — multi-model management

Started **without** a model argument, the upstream server runs in **router mode**: it lists models
from `--models-dir`, loads/unloads them on demand (`GET /models`, `POST /models/load`,
`POST /models/unload`, per-request `"model"` selection) and serves each model from a **worker
subprocess**. Upstream spawns workers by re-executing its own binary — inside a JVM that binary is
`java`, so before starting an embedded router you must point the worker spawn at this library's
bootstrap:

```java
String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
NativeServer.setWorkerCommand(javaBin, "-cp", System.getProperty("java.class.path"),
        "net.ladenthin.llama.server.NativeServer");
try (NativeServer router = new NativeServer(
        "--host", "127.0.0.1", "--port", "8080", "--models-dir", "models").start()) {
    Thread.currentThread().join(); // each loaded model runs as a fresh worker JVM
}
```

Worker-command tokens may not contain whitespace (the value is whitespace-split natively).

**Typed model management (`RouterClient`).** Instead of hand-rolling HTTP+JSON against the
management endpoints, use `server.RouterClient` — a plain-HTTP typed client (works against the
embedded router above or any external `llama-server` router):

```java
RouterClient client = new RouterClient(8080);
List<RouterModel> models = client.listModels();          // GET /models, typed status per entry
client.loadModel("Qwen3-0.6B-Q4_K_M");                   // POST /models/load (non-blocking)
client.awaitModelLoaded("Qwen3-0.6B-Q4_K_M", 240_000L);  // poll until LOADED; fails fast if the
                                                         // worker died (exit code in the message)
client.unloadModel("Qwen3-0.6B-Q4_K_M");                 // POST /models/unload
```

`RouterModel` carries the identifier, the lifecycle status
(`UNLOADED`/`LOADING`/`LOADED`/`SLEEPING`/`DOWNLOADING`/`DOWNLOADED`), and the router's
failed-worker marker. Chat requests then select a model per request via the standard
`"model"` field on `POST /v1/chat/completions`.

### LangChain4j integration

A separate artifact, **`net.ladenthin:llama-langchain4j`**, adapts a `LlamaModel` to
[LangChain4j](https://github.com/langchain4j/langchain4j)'s `ChatModel`, `StreamingChatModel`,
`EmbeddingModel` and `ScoringModel` interfaces **in-process over JNI** — no HTTP hop, no separate
server. It is a separate `artifactId` (not a classifier of the core) because LangChain4j 1.x
requires **Java 17** while the core `net.ladenthin:llama` stays Java 8; keeping it separate avoids
forcing that floor on every core consumer. It ships and versions in lockstep with the core.

```xml
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama-langchain4j</artifactId>
    <version>5.0.6</version>
</dependency>
```

Each adapter **borrows** a `LlamaModel` you already loaded — it never loads or closes the native
model, so you manage its lifecycle (try-with-resources), and one `LlamaModel` can back several
adapters at once:

```java
try (LlamaModel llama = new LlamaModel(new ModelParameters().setModel("models/qwen3-0.6b.gguf"))) {
    ChatModel chat = new JllamaChatModel(llama);
    String reply = chat.chat("Write a haiku about lazy senior devs.");
    System.out.println(reply);
}
```

| Adapter | LangChain4j interface | java-llama.cpp call |
|---------|-----------------------|---------------------|
| `JllamaChatModel` | `ChatModel` | `LlamaModel.chat(...)` |
| `JllamaStreamingChatModel` | `StreamingChatModel` | `LlamaModel.generateChat(...)` (token streaming) |
| `JllamaEmbeddingModel` | `EmbeddingModel` | `LlamaModel.embed(...)` (model loaded with `enableEmbedding()`) |
| `JllamaScoringModel` | `ScoringModel` (re-ranking) | `LlamaModel.handleRerank(...)` (model loaded with `enableReranking()`) |

See [`llama-langchain4j/README.md`](llama-langchain4j/) for streaming/embedding/re-ranking
examples and the current mapping limitations (tool calling, JSON mode, and multimodal input are
not yet forwarded).

### Model/Inference Configuration

There are two sets of parameters you can configure, `ModelParameters` and `InferenceParameters`. Both provide builder 
classes to ease configuration. `ModelParameters` are once needed for loading a model, `InferenceParameters` are needed
for every inference task. All non-specified options have sensible defaults.

```java
ModelParameters modelParams = new ModelParameters()
        .setModel("/path/to/model.gguf")
        .addLoraAdapter("/path/to/lora/adapter");
String grammar = """
		root  ::= (expr "=" term "\\n")+
		expr  ::= term ([-+*/] term)*
		term  ::= [0-9]""";
InferenceParameters inferParams = new InferenceParameters("")
        .setGrammar(grammar)
        .setTemperature(0.8);
try (LlamaModel model = new LlamaModel(modelParams)) {
    model.generate(inferParams);
}
```

### Reactive integration (Reactor, RxJava, Kotlin Flow, Akka)

`LlamaIterable` (returned by `model.generate(...)` and `model.generateChat(...)`)
implements `Iterable<LlamaOutput> & AutoCloseable`, so every mainstream reactive
library wraps it in a few lines without `java-llama.cpp` pulling in a runtime
reactive dependency.

**Always wrap with the library's resource-management primitive** — `Flux.using`,
`Flowable.using`, Kotlin `use {}`, etc. — so that subscription cancellation
flows into `LlamaIterable.close()` and from there into llama.cpp's native
`cancelCompletion`. A plain `Flux.fromIterable(iterable)` or `for (x in iter)`
loop will NOT close the iterable on cancel; the native task slot stays
occupied until the model is closed.

#### Project Reactor (Spring WebFlux)
```java
Flux<LlamaOutput> tokens = Flux.using(
        () -> model.generate(params),
        Flux::fromIterable,
        LlamaIterable::close)
    .subscribeOn(Schedulers.boundedElastic());
```

#### RxJava 3 (also for RxAndroid)
```java
Flowable<LlamaOutput> tokens = Flowable.using(
        () -> model.generate(params),
        Flowable::fromIterable,
        LlamaIterable::close)
    .subscribeOn(Schedulers.io());
```

#### Kotlin Flow (Android / coroutines)

Ready-made: the optional [`net.ladenthin:llama-kotlin`](llama-kotlin/README.md) artifact ships
`generateFlow`/`generateChatFlow` extensions (close-on-cancellation included) plus `suspend`
wrappers whose coroutine cancellation is wired to the binding's cooperative `CancellationToken`:

```kotlin
model.generateChatFlow(params).flowOn(Dispatchers.IO).collect { print(it.text) }
```

Hand-rolled equivalent (no extra dependency):
```kotlin
fun llama(model: LlamaModel, params: InferenceParameters) = flow {
    model.generate(params).use { iterable ->
        for (output in iterable) emit(output)
    }
}.flowOn(Dispatchers.IO)
```
The companion Android sample [LLaMAndroid](https://github.com/Rattlyy/LLaMAndroid)
demonstrates the `flow { for (output in model.generate(params)) emit(output) }`
shape against the upstream binding. Wrap the `for` loop in
`.use { }` if your collector may cancel mid-stream — otherwise the native task
slot will not be released until the model is closed.

#### Akka Streams
```scala
val tokens: Source[LlamaOutput, NotUsed] = Source
    .fromIterator(() => model.generate(params).iterator())
    .async("blocking-io-dispatcher")
```

**Why no built-in `Publisher`?** Earlier snapshots of this fork shipped a
hand-rolled `LlamaModel.streamPublisher(...)` returning a Reactive Streams
`Publisher<LlamaOutput>`. Since every reactive library bridges blocking
iterables in a few lines via its own resource-management primitive, the binding
now stays free of any reactive runtime dependency — pick whichever library your
app already uses. The pattern is verified end-to-end by
`ReactorIntegrationTest` in the test sources.

### Logging

Per default, logs are written to stdout.
This can be intercepted via the static method `LlamaModel.setLogger(LogFormat, BiConsumer<LogLevel, String>)`. 
There is text- and JSON-based logging. The default is JSON.
Note, that text-based logging will include additional output of the GGML backend, while JSON-based logging
only provides request logs (while still writing GGML messages to stdout).
To only change the log format while still writing to stdout, `null` can be passed for the callback. 
Logging can be disabled by passing an empty callback.

```java
// Re-direct log messages however you like (e.g. to a logging library)
LlamaModel.setLogger(LogFormat.TEXT, (level, message) -> System.out.println(level.name() + ": " + message));
// Log to stdout, but change the format
LlamaModel.setLogger(LogFormat.TEXT, null);
// Disable logging by passing a no-op
LlamaModel.setLogger(null, (level, message) -> {});
```

The `LogLevel` enum values passed to the callback correspond to the native llama.cpp log levels:

| Value | Meaning |
|---|---|
| `DEBUG` | Verbose diagnostic output |
| `INFO` | Informational messages about model loading and inference |
| `WARN` | Non-fatal warnings |
| `ERROR` | Errors that may affect inference results |

## Importing in Android

> [!IMPORTANT]
> **Minimum Android version: API 28 (Android 9.0 Pie).** Devices running
> Android 8.1 (API 27) or earlier are not supported.

### Option 1 (recommended): the `llama-android` AAR from Maven Central

One dependency line in Android Studio — no submodule, no NDK build, no manual ProGuard rules:

```kotlin
dependencies {
    implementation("net.ladenthin:llama-android:5.0.6")
    // or, for Qualcomm Adreno GPUs (device must provide an OpenCL ICD):
    // implementation("net.ladenthin:llama-android-opencl:5.0.6")

    // optional Kotlin coroutines facade (Flow streaming + suspend wrappers):
    implementation("net.ladenthin:llama-kotlin:5.0.6")
}
```

The AAR carries the full `net.ladenthin:llama` Java API, the CI-built native libraries for
`arm64-v8a` (devices) **and** `x86_64` (Android Studio emulator, Chromebooks — app bundles
split per ABI so phones download only arm64), both 16 KB page-size compliant, consumer
R8/ProGuard rules (applied automatically), and a manifest `minSdkVersion 28` that AGP
enforces against your app. CI boots an x86_64 emulator and runs real on-device inference
against every AAR build.
Do **not** also depend on the desktop `net.ladenthin:llama` JAR in the same app — the AAR
already contains those classes, and the JAR would drag ~70 MB of desktop natives into your
APK. See [`llama-android/README.md`](llama-android/README.md) and
[`llama-kotlin/README.md`](llama-kotlin/README.md) for details.

**Runnable example app — "LLM Service".** A minimal, KISS, fully-offline on-device chat app
— pick a GGUF from the file system, then chat with it, tokens streaming into a Jetpack
Compose UI, with a 13-language flag picker and private local save/load — lives in
[`android-llmservice/`](android-llmservice/README.md) (`net.ladenthin.android.llmservice`).
It builds with plain Gradle/AGP (no Android Studio required), produces a Play-shaped signed
`.aab`, and is validated in CI by a real on-device emulator UI test. See its README for the
build, signing/Play, and testing walkthrough.

### Option 2 (advanced): build from source inside your app

Use this only if you need to patch the native layer or build for an ABI this project does
not ship.
1. Add java-llama.cpp as a submodule in your an droid `app` project directory
```shell
git submodule add https://github.com/bernardladenthin/java-llama.cpp 
```
2. Declare the library as a source in your build.gradle
```gradle
android {
    val jllamaLib = file("java-llama.cpp")

    // Execute "mvn compile" in the llama/ core module if its target/ doesn't exist
    // (the repository root is the Maven reactor aggregator; the native core lives in llama/).
    if (!file("$jllamaLib/llama/target").exists()) {
        exec {
            commandLine = listOf("mvn", "compile")
            workingDir = file("java-llama.cpp/llama/")
        }
    }

    ...
    defaultConfig {
	...
        externalNativeBuild {
            cmake {
		// Add an flags if needed
                cppFlags += ""
                arguments += ""
            }
        }
    }

    // Declare c++ sources
    externalNativeBuild {
        cmake {
            path = file("$jllamaLib/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // Declare java sources
    sourceSets {
        named("main") {
            // Add source directory for java-llama.cpp
            java.srcDir("$jllamaLib/src/main/java")
        }
    }
}
```
3. Exclude `net.ladenthin.llama` in proguard-rules.pro
```proguard
keep class net.ladenthin.llama.** { *; }
```

## TODO

Open work items live in [`TODO.md`](TODO.md).

- **Expand PIT mutation-testing scope.** PIT is wired in `pom.xml` and runs on every CI build (in the `test-java-linux-x86_64` job) with `<mutationThreshold>100</mutationThreshold>`. `<targetClasses>` currently covers `net.ladenthin.llama.value.*`, `exception.*`, `args.*` and four `json` parsers (295 mutations, 100% killed, hermetic — no model or fixture needed); widen it incrementally as additional classes reach mutation-test parity. Final target: `<param>net.ladenthin.llama.*</param>` matching the streambuffer pattern.

## Feature Ideas

Forward-looking ideas being tracked for this fork:

- **Adopt feature ideas from the Kotlin Llama Stack client.** Candidates (multimodal image input, typed chat messages, async API, batch inference, typed usage/timings) are inventoried with effort estimates in [`docs/feature-investigation-llama-stack-client-kotlin.md`](docs/feature-investigation-llama-stack-client-kotlin.md), derived from [`ogx-ai/llama-stack-client-kotlin`](https://github.com/ogx-ai/llama-stack-client-kotlin).
- **Ship a directly Android-capable artifact — DONE.** `net.ladenthin:llama-android` / `llama-android-opencl` (AAR, arm64-v8a, minSdk 28, consumer ProGuard rules, 16 KB page-size compliant) plus the optional `net.ladenthin:llama-kotlin` coroutines façade ship from this repo — see [Importing in Android](#importing-in-android). Typed image input for VLMs is covered by `ContentPart.imageBytes(...)` / `imageFile(...)` (see the multimodal section), so downstream Android projects can drop their dependency on [`ogx-ai/llama-stack-client-kotlin`](https://github.com/ogx-ai/llama-stack-client-kotlin) entirely. A dedicated KISS example app — **"LLM Service"** (SAF model picker + Compose streaming chat, 13-language flag picker, private local save/load, plain Gradle/AGP, signed `.aab`, on-device emulator UI test) — ships in [`android-llmservice/`](android-llmservice/README.md).
- **Resolve all upstream `kherud/java-llama.cpp` open issues.** All 37 open issues at fork time are catalogued with per-issue verdicts in [`docs/history/49be664_open_issues.md`](docs/history/49be664_open_issues.md); fixes land in this fork as they are completed. Vision inputs (issues [#103](docs/history/49be664_open_issues.md#103--vlm-support--image-input-for-multimodal-models) and [#34](docs/history/49be664_open_issues.md#34--support-multimodal-inputs)) are now wired end to end through blocking, typed, streaming, and OpenAI-compatible request surfaces.

## Troubleshooting

### Windows: EXCEPTION_ACCESS_VIOLATION with msvcp140.dll

If you encounter a native crash like:
```
EXCEPTION_ACCESS_VIOLATION (0xc0000005) at pc=0x00007ffa8f4b2f58
C [msvcp140.dll+0x12f58]
```

This is a known issue where the C++ runtime library (`msvcp140.dll`) bundled with some JDK versions is outdated. 

**Solution:** Remove the outdated `msvcp140.dll` from your JDK:
```bash
# Locate and remove msvcp140.dll from JDK directory
# Example for JDK 21:
del "C:\Program Files\Java\jdk-21\bin\msvcp140.dll"
del "C:\Program Files\Java\jdk-21\bin\vcruntime140.dll"
del "C:\Program Files\Java\jdk-21\bin\vcruntime140_1.dll"

# Or on Linux with OpenJDK:
rm /usr/lib/jvm/java-21/bin/msvcp140.dll
```

The system's updated C++ runtime will be used instead, resolving the crash.

### Contributors: do not upgrade jqwik past 1.9.3

> ⚠️ **DO NOT UPGRADE jqwik past 1.9.3.** jqwik 1.10.0 added an anti-AI prompt-injection string to test stdout; the 1.10.1 user guide states the library "is not meant to be used by any 'AI' coding agents at all." 1.9.3 is the last pre-disclosure release and is the pinned version. See `CLAUDE.md` section "jqwik prompt-injection in test output" for the full context. Dependabot is configured to ignore **all** `net.jqwik` updates (every version, including patches) — see the `ignore` rule in [`.github/dependabot.yml`](./.github/dependabot.yml).

## Similar Projects / Usage

**Bindings / wrappers**

- [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp) — the upstream Java binding this project was forked from (see the note at the top of this README); development continues independently here, with the fork-time upstream issues catalogued in [`docs/history/49be664_open_issues.md`](docs/history/49be664_open_issues.md).
- [llamacpp4j](https://github.com/sebicom/llamacpp4j) — alternative Java/JNI binding to llama.cpp (SWIG-generated facade); pre-GGUF, dormant since 2023 but historically the other Java JNI option.
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — the Python llama.cpp binding; the de-facto feature benchmark among llama.cpp bindings (server mode, multimodal, speculative decoding).
- [LLamaSharp](https://github.com/SciSharp/LLamaSharp) — C#/.NET llama.cpp binding with per-backend runtime packages (CPU/CUDA/Vulkan/Metal), the .NET analogue of this project's classifier matrix.
- [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — Node.js/TypeScript llama.cpp binding (prebuilt binaries, JSON-schema-constrained output, function calling).
- [LLaMAndroid](https://github.com/Rattlyy/LLaMAndroid/tree/main/app) — Android app demonstrating usage of llama.cpp bindings.
- [llama-stack-client-kotlin](https://github.com/ogx-ai/llama-stack-client-kotlin) — Kotlin client for the Llama Stack API with an ExecuTorch-backed local-inference path (the [`llama-android`](llama-android/) AAR + [`llama-kotlin`](llama-kotlin/) façade cover the same on-device ground natively).
- [llama.cpp-android-tutorial](https://github.com/JackZeng0208/llama.cpp-android-tutorial) — Step-by-step tutorial for running llama.cpp on Android.

**Other local inference stacks (no llama.cpp JVM binding)**

- [Ollama](https://github.com/ollama/ollama) — llama.cpp-based local model runner with its own HTTP API and model registry. This project's OpenAI-compatible server implements the Ollama-native API surface (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`), so Ollama-speaking clients (e.g. VS Code Copilot's Ollama provider) work against an in-process jllama model.
- [ExecuTorch](https://github.com/pytorch/executorch) — PyTorch's on-device inference runtime (`.pte` models, XNNPACK/NPU delegates); the engine behind `llama-stack-client-kotlin`'s local mode and the main non-llama.cpp alternative for Android on-device inference (GGUF is not supported there — different model format ecosystem).

**Pure-Java single-model inference (no JNI / no llama.cpp)** — Alfonso² Peterssen's `*.java` family of standalone, dependency-free Java inference runtimes, one per model architecture. Useful when JNI is unavailable (e.g. some sandboxes / GraalVM native-image scenarios) or when you want a single jar with no native side at all. Different design point from this project, which prioritises GGUF compatibility and llama.cpp performance via JNI.

- [llama3.java](https://github.com/mukel/llama3.java) — Llama 3 / 3.1 / 3.2 inference.
- [gemma4.java](https://github.com/mukel/gemma4.java) — Gemma 4 (and earlier Gemma 2/3) inference.
- [gptoss.java](https://github.com/mukel/gptoss.java) — GPT-OSS architecture inference.
- [qwen35.java](https://github.com/mukel/qwen35.java) — Qwen 3.5 inference.
- [nemotron3.java](https://github.com/mukel/nemotron3.java) — NVIDIA Nemotron-3 inference.

**Pure-Java inference engines (no JNI / no llama.cpp)**

- [Jlama](https://github.com/tjake/Jlama) — a full pure-Java LLM inference engine for the JVM (multiple model architectures, quantization, and distributed inference) built on the Java Vector API. A no-native alternative to the JNI approach here; different design point (pure JVM portability vs. GGUF compatibility and llama.cpp performance via JNI).

**Frameworks / orchestration**

- [LangChain4j](https://github.com/langchain4j/langchain4j) — LLM-application framework for Java (chat, embeddings, RAG, tool calling, agents) over a unified provider API. This project ships a first-class **in-process** integration — see the [`llama-langchain4j`](llama-langchain4j/) module — so a llama.cpp model plugs straight into LangChain4j's `ChatModel` / `StreamingChatModel` / `EmbeddingModel` / `ScoringModel` without an HTTP hop.
