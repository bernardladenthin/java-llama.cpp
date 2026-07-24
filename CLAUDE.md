# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b10092**

## Upgrading CUDA Version

Current CUDA version: **13.2**

To change the CUDA version, update the following **three** places:

1. **`.github/build_cuda_linux.sh`** — Line 10: `sudo dnf install -y cuda-toolkit-13-2`
2. **`.github/build_cuda_linux.sh`** — Line 12: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`
3. **`llama/pom.xml`** — The `<classifier>` tag in the `cuda` jar execution: `cuda13-linux-x86-64`

Also update the header comment in `build_cuda_linux.sh` and the job name in `.github/workflows/release.yaml` for clarity.

Available CUDA versions for RHEL8/Manylinux_2_28 can be browsed at:
```
https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/
```

**Note:** Each CUDA version supports only certain GCC versions. If the dockcross container uses a newer GCC than CUDA supports, the build will fail with `unsupported GNU version`. Check NVIDIA's compatibility table before downgrading CUDA.

Example: To upgrade from 13.2 to a hypothetical 13.3:
```bash
# Edit .github/build_cuda_linux.sh:
#   line 10: cuda-toolkit-13-2 -> cuda-toolkit-13-3
#   line 12: /usr/local/cuda-13.2/bin/nvcc -> /usr/local/cuda-13.3/bin/nvcc
# Edit llama/pom.xml classifier: cuda13-linux-x86-64 (major version only, no need to change for minor bumps)
# Edit CLAUDE.md line: Current CUDA version: **13.2** -> **13.3**
git add .github/build_cuda_linux.sh llama/pom.xml CLAUDE.md
git commit -m "Upgrade CUDA from 13.2 to 13.3"
```

### Fast local CUDA builds (`CUDA_FAST_BUILD`) — single-arch speed knob

The CUDA artifact must ship kernels for **every supported GPU generation**, so the default
build — and every CI build — compiles the **full `CMAKE_CUDA_ARCHITECTURES` set** that
ggml/llama.cpp selects. nvcc recompiles each `.cu` kernel once per architecture, which is the
dominant cost of the ~70 min CUDA job. **`sccache` now wraps nvcc too:** `build.sh` adds
`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache` for CUDA builds (it detects `GGML_CUDA` in the cmake
args), so the per-arch `.cu` device passes are cached over Depot alongside the gcc C/C++ TUs.
Because the kernels are content-addressed and llama.cpp is pinned, a **warm** cache recompiles
only what changed — so CI keeps the **full arch set on every run** (release-safe everywhere)
and relies on the cache, not a reduced arch set, for speed. The first (cold-cache) run still
pays the full nvcc cost; the win shows on subsequent warm runs.

`CUDA_FAST_BUILD` remains as a **local-dev** single-arch knob (CI no longer sets it).
`build_cuda_linux.sh` honors it — default **off** (full arch set, release-safe):

```bash
# Full release build (default): all archs — slow, runs on every GPU generation.
.github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"

# Fast local dev build: one arch only. Defaults to `native` (the build machine's own GPU;
# needs a GPU present at configure time). Override with CUDA_ARCH=<cc>, e.g. CUDA_ARCH=90.
CUDA_FAST_BUILD=1 .github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"
CUDA_FAST_BUILD=1 CUDA_ARCH=90 .github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"
# Direct-cmake equivalent: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

**Default + CI policy (release-safety is the invariant).** An artifact built with `CUDA_FAST_BUILD`
runs on only the single GPU generation it was compiled for, so the **distributed jar must always be
the full arch set**. The script default is **off** (full) so any *local/manual* build is
release-safe, and **CI no longer sets `CUDA_FAST_BUILD` at all** — the `crosscompile-linux-x86_64-cuda`
job always builds the full set on PR / push / dispatch / publish, so every artifact (not just the ones
that reach Central) runs on every GPU generation. The full-arch CI cost is absorbed by the
sccache-over-Depot cache, which now wraps nvcc (`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache`, added by
`build.sh` for CUDA builds, gated behind the same probe). The launcher is safe to enable
unconditionally: if sccache cannot wrap nvcc it runs it directly (uncached), and `build.sh`'s
mid-build retry treats an sccache `Compiler not supported` failure like any other cache error and
rebuilds the job without the launcher rather than redding it. **Verified:** a warm run in the
manylinux_2_28 container hit **100%** on CUDA / CUBIN / device-code (139 CUDA hits, 99.86% overall,
3 misses) and cut the job from **~51 min cold to ~15 min warm** — nvcc caching works here. `build.sh`
prints `sccache --show-stats` at the end of every run so the hit table stays visible.

## Android minimum API level

Current Android minimum API level: **28** (Android 9.0 Pie)

This is enforced through bionic's **weak-symbol** mechanism, *not* by bumping
`__ANDROID_API__` or passing `-DANDROID_PLATFORM`. See "How the API gate is
satisfied" below for why. To change anything here, update:

1. **`llama/CMakeLists.txt`** — the `add_compile_definitions(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)`
   block and its Android-detection guard (`OS_NAME MATCHES "Android"` etc.).
2. **`CLAUDE.md`** (this file) — the "Current Android minimum API level" line above.
3. **`README.md`** — the minimum-API note (the `[!NOTE]` block near the Android
   classifier entries and the "Importing in Android" section).

**Why API 28?** `mtmd-helper.cpp` (part of the upstream llama.cpp `mtmd`
multimodal library) includes `vendor/sheredom/subprocess.h`, which calls
`posix_spawn`, `posix_spawnp`, and `posix_spawn_file_actions_*`. Bionic only
exposes those `<spawn.h>` declarations once the minimum SDK is ≥ 28 (and
`getifaddrs`/`freeifaddrs` in `<ifaddrs.h>`, used by cpp-httplib, at ≥ 24). The
symbols exist in `libc.so` at all API levels; bionic only hides the
*declarations* below the introducing API.

**How the API gate is satisfied (important — the obvious fixes do not work).**
The CI cross-compiler is the `dockcross-android-arm64` image, which is **not**
the Google NDK CMake toolchain — it is a Debian-style cross-clang at
`/usr/aarch64-linux-android/bin/clang`. Consequently:

- It never sets the `ANDROID` / `ANDROID_ABI` CMake variables, so any
  `if(ANDROID_ABI)`-guarded logic silently does nothing.
- It **ignores** `-DANDROID_PLATFORM=android-28` (CMake prints it as a
  "Manually-specified variables were not used by the project" warning).
- `clang` predefines `__ANDROID_API__` from its baked-in target triple, so
  `-D__ANDROID_API__=28` would only clash with the builtin (`-Wmacro-redefined`)
  and would *not* move `__ANDROID_MIN_SDK_VERSION__`, which is what bionic's
  `__BIONIC_AVAILABILITY_GUARD(api)` actually tests.

The working fix is `add_compile_definitions(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)`
for the Android build. That macro forces `__BIONIC_AVAILABILITY_GUARD(api)` to
`1` for every API level (declarations always visible) and makes any symbol newer
than the toolchain's baked-in min-SDK a **weak** reference resolved by the
dynamic linker at load time — present on every API-28+ device the artifact
targets. It is never compiler-predefined, so defining it is clean. The guard
detects Android via `OS_NAME MATCHES "Android"` (CI passes
`-DOS_NAME=Linux-Android`) and the compiler path, not `ANDROID_ABI`.

## OpenCL / Adreno backend on Android

A second Android arm64 artifact is built with the OpenCL backend enabled and
Adreno-tuned kernels embedded. It ships under the Maven classifier
`opencl-android-aarch64` and is consumed only when callers explicitly request it.
The default Android arm64 JAR remains CPU-only.

Three places wire it together (mirrors the CUDA classifier pattern):

1. **`llama/CMakeLists.txt`** — `elseif(GGML_OPENCL)` branch routes artifacts to
   `src/main/resources_android_opencl/net/ladenthin/llama/${OS_NAME}/${OS_ARCH}/`.
2. **`.github/workflows/publish.yml`** — `crosscompile-android-aarch64-opencl`
   job runs the dockcross-android-arm64 build with
   `-DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON`
   and uploads as artifact `android-libraries-opencl`. The `package`,
   `publish-snapshot`, and `publish-release` jobs download it into
   `resources_android_opencl/` and activate the `opencl-android` Maven profile.
3. **`llama/pom.xml`** — the `opencl-android` profile produces a second JAR with
   `<classifier>opencl-android-aarch64</classifier>` from the
   `${project.build.outputDirectory}_opencl_android` tree.

Local sanity build:
```bash
.github/dockcross/dockcross-android-arm64 .github/build_opencl_android.sh \
  "-DOS_NAME=Linux-Android -DOS_ARCH=aarch64 \
   -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
   -DGGML_OPENCL_USE_ADRENO_KERNELS=ON"
```
Artifacts land in `src/main/resources_android_opencl/net/ladenthin/llama/Linux-Android/aarch64/`.

The dockcross image does not ship OpenCL headers or a stub `libOpenCL.so`, so
`build_opencl_android.sh` first stages Khronos `OpenCL-Headers` and
cross-builds `OpenCL-ICD-Loader` into `/tmp/opencl-stage/` before invoking the
main project cmake with `-DOpenCL_INCLUDE_DIR=...` and `-DOpenCL_LIBRARY=...`.
At runtime the device must provide its own OpenCL ICD (`libOpenCL.so`);
Qualcomm Adreno drivers do. Devices without an ICD should use the default
CPU-only Android JAR.

## Windows native classifiers (default Ninja CPU + MSVC classifier + CUDA/Vulkan/OpenCL GPU)

The Windows native libraries ship in **five** forms. The **default JAR's** Windows natives are now
built with the **`Ninja Multi-Config`** generator (the *default flip*); the Visual Studio / MSVC
build is shipped as the **`msvc-windows`** classifier; and three GPU backends ship as
**`cuda13-windows-x86-64`**, **`vulkan-windows-x86-64`**, and **`opencl-windows-x86-64`** (all
**x86_64 only**, all Ninja).

**Why Ninja is the default (the flip).** The Visual Studio generator ignores
`CMAKE_{C,CXX}_COMPILER_LAUNCHER`, so only Ninja Multi-Config can front `cl.exe` with sccache over
Depot WebDAV. **Both generators use the same MSVC toolchain** (`cl.exe`, static `/MT` CRT via
`CMAKE_MSVC_RUNTIME_LIBRARY`, same Release flags, same runner), so the produced
`jllama.dll` binaries are **functionally equivalent with identical runtime
dependencies** — the only difference is build-system plumbing + caching. Making Ninja the default
gives the most-pulled JAR the sccache cache; MSVC stays available as a classifier for anyone who
wants the Visual-Studio-generator build. (Upstream llama.cpp also builds its Windows artifacts with
Ninja Multi-Config + MSVC.) Both Windows CPU builds are validated end-to-end with the full
model-backed Java suite (`test-java-windows-x86_64` = default/Ninja, `test-java-windows-x86_64-msvc`
= MSVC classifier).

**GPU runtime libraries are NOT bundled.** The GPU JARs ship only the single monolithic
`jllama.dll` (llama.cpp + ggml + the backend are statically linked in — `BUILD_SHARED_LIBS OFF`). The consumer's driver/toolkit must supply the runtime: CUDA needs the
installed CUDA 13 Toolkit (`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll` on `PATH`); Vulkan
needs `vulkan-1.dll` (ships with current GPU drivers); OpenCL needs the vendor ICD
(`System32\OpenCL.dll`). Not bundling = no NVIDIA-EULA redistribution obligation. **GitHub-hosted
Windows runners have NO GPU**, so the GPU jobs **build the artifact only** (no `-DBUILD_TESTING`/`ctest`)
— a GPU-linked `jllama_test.exe` can't even be enumerated on a GPU-less runner (it errors probing for a
device, so `gtest_discover_tests` registers a failing `*_NOT_BUILT` sentinel). The CPU-only C++ unit
suite is fully covered by the `C++ Tests` job + the CPU Windows jobs; model-backed GPU inference is
local / self-hosted.

Wiring (mirrors the CUDA-Linux / OpenCL-Android classifier pattern):

1. **`llama/CMakeLists.txt`** — the `if(GGML_CUDA) … elseif(GGML_VULKAN) … elseif(GGML_OPENCL) … else()`
   chain is **OS-aware**: CUDA → `resources_windows_cuda` on Windows (else `resources_linux_cuda`),
   Vulkan → `resources_windows_vulkan` on Windows (else `resources_linux_vulkan` — see "Linux Vulkan
   classifiers" above), OpenCL → `resources_windows_opencl` on Windows (else
   `resources_android_opencl`). The default CPU build (both generators) still emits to the canonical
   `src/main/resources/.../Windows/{x86_64,x86}/`, so the Ninja-vs-MSVC split is purely a
   CI-artifact-name + pom-profile concern (no CMake change for it).
2. **`.github/build.bat`** — the sccache probe guard (mirrors `build.sh`) wraps the **cl.exe** C/C++ TUs
   only. Unlike `build.sh` (Linux), it does **not** wrap `nvcc`: sccache on Windows can't parse the nvcc
   command line (`sccache: error: Could not parse shell line`) and fails every `.cu` compile, so CUDA
   device code builds with nvcc directly (uncached). `build.bat` also propagates a `cmake --build`
   failure as a non-zero exit (a prior bug let a failed CUDA build exit 0 → empty artifact → late
   `package` failure); the GPU upload steps additionally use `if-no-files-found: error` as a backstop.
3. **`.github/build_opencl_windows.bat`** — stages Khronos OpenCL-Headers + builds OpenCL-ICD-Loader
   (`OpenCL.lib`), then delegates to `build.bat` with `-DOpenCL_INCLUDE_DIR`/`-DOpenCL_LIBRARY`
   (the Windows analogue of `build_opencl_android.sh`).
4. **`.github/workflows/publish.yml`** — build jobs (all `windows-2025-vs2026`, `ilammy/msvc-dev-cmd@v1`,
   sccache v0.16.0 zip + Depot WebDAV):
   - `build-windows-x86_64` / `build-windows-x86` — **Ninja CPU**, artifacts `Windows-{arch}-libraries`
     → picked up by the `package` job's `pattern: "*-libraries"` into the **default** tree.
   - `build-windows-x86_64-msvc` / `build-windows-x86-msvc` — **MSVC CPU**, artifacts `Windows-{arch}-msvc`.
   - `build-windows-x86_64-cuda` — `Jimver/cuda-toolkit@v0.2.35` (CUDA `13.2.0`) + `-DGGML_CUDA=ON`,
     artifact `Windows-x86_64-cuda`.
   - `build-windows-x86_64-vulkan` — `jakoch/install-vulkan-sdk-action` + `-DGGML_VULKAN=ON`, artifact
     `Windows-x86_64-vulkan`.
   - `build-windows-x86_64-opencl` — `build_opencl_windows.bat -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON`,
     artifact `Windows-x86_64-opencl`.
   The `package`, `publish-snapshot`, and `publish-release` jobs download each non-default artifact into
   its `src/main/resources_windows_{msvc,cuda,vulkan,opencl}/` tree and activate the
   `windows-msvc,cuda-windows,vulkan-windows,opencl-windows` Maven profiles.
5. **`llama/pom.xml`** — profiles `windows-msvc` / `cuda-windows` / `vulkan-windows` / `opencl-windows`,
   each a separate compile pass + resource copy + classified jar (classifiers `msvc-windows` /
   `cuda13-windows-x86-64` / `vulkan-windows-x86-64` / `opencl-windows-x86-64`). Activated only in CI.
6. **`README.md`** — the classifier table + dependency snippets in "Choosing the right classifier".

`src/main/resources_windows_{msvc,cuda,vulkan,opencl}/` are git-ignored (staged by CI, never committed).

**First CI run (PR #276, run 28327740376):** the default Ninja CPU flip, the MSVC classifier, and the
**OpenCL** job were green on the first try. Two GPU jobs needed a toolchain fix: **CUDA** failed with
`Version not available: 13.0.0` because the pinned `Jimver/cuda-toolkit@v0.2.24` predated CUDA 13.x →
bumped to `@v0.2.35` + `13.2.0` (matches the Linux pin, classifier stays `cuda13-…`); **Vulkan** failed
`find_package(Vulkan)` because `humbletim/install-vulkan-sdk` set `VULKAN_SDK` but laid the SDK out in a
way CMake's `FindVulkan` couldn't read → switched to `jakoch/install-vulkan-sdk-action` (purpose-built,
FindVulkan-compatible). Because all five Windows build jobs are in the `package`/publish `needs:` graph, a
GPU-toolchain failure blocks packaging — the same release-gating policy the Linux-CUDA / Android-OpenCL
jobs already follow.

**Local sanity builds** (need MSVC + Ninja on PATH; sccache optional; GPU builds also need the matching SDK):
```bat
mvn -q compile
.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DBUILD_TESTING=ON
ctest --test-dir build --output-on-failure
:: GPU (needs the matching SDK installed + on PATH):
.github\build.bat -G "Ninja Multi-Config" -DGGML_CUDA=ON   -DOS_NAME=Windows -DOS_ARCH=x86_64
.github\build.bat -G "Ninja Multi-Config" -DGGML_VULKAN=ON -DOS_NAME=Windows -DOS_ARCH=x86_64
.github\build_opencl_windows.bat -G "Ninja Multi-Config" -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DOS_NAME=Windows -DOS_ARCH=x86_64
```

## Linux Vulkan classifiers + Windows arm64 CPU

Three additional artifacts extend the matrix toward upstream llama.cpp's release set. They follow
the same classifier/resource-tree pattern as CUDA-Linux and Vulkan-Windows.

**Linux Vulkan (`vulkan-linux-x86-64` + `vulkan-linux-aarch64`).** A vendor-neutral GPU jar for
Linux (NVIDIA / AMD / Intel) with no CUDA toolkit — the intersection of the existing Vulkan-Windows
and CUDA-Linux wiring. Four places:

1. **`llama/CMakeLists.txt`** — the `elseif(GGML_VULKAN)` branch is now **OS-aware** (mirrors
   `GGML_CUDA`): Windows → `resources_windows_vulkan`, else → `resources_linux_vulkan`
   (`.../Linux/${OS_ARCH}/`). One tree holds both arches under `Linux/{x86_64,aarch64}`.
2. **`.github/workflows/publish.yml`** — `build-linux-x86_64-vulkan` (native `ubuntu-latest`, **not**
   dockcross — the Vulkan SDK is a trivial apt install and upstream builds ubuntu-vulkan the same way)
   and `build-linux-aarch64-vulkan` (`ubuntu-24.04-arm` + GCC 14). Both `apt-get install libvulkan-dev
   glslc glslang-tools`, build `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF`, and are **build-only** (no
   `ctest`: a Vulkan-linked `jllama_test` errors enumerating devices on a GPU-less runner — same as the
   Windows GPU jobs). Artifacts `Linux-{x86_64,aarch64}-vulkan` → both downloaded into the **one**
   `resources_linux_vulkan/` tree by `package`/`publish-*`. Glibc floor rises to the ubuntu baseline
   (like the aarch64 CPU jar); acceptable for a GPU artifact.
3. **`llama/pom.xml`** — profiles `vulkan-linux` (classifier `vulkan-linux-x86-64`) and
   `vulkan-linux-aarch64` (classifier `vulkan-linux-aarch64`). Both read the shared
   `resources_linux_vulkan` tree but the resource-copy `<includes>` is **arch-scoped**
   (`net/ladenthin/llama/Linux/{x86_64,aarch64}/**`), so each classifier JAR carries only its own
   arch (verified: each jar contains exactly one `libjllama.so`). Separate output dirs
   `_linux_vulkan` / `_linux_vulkan_aarch64` avoid collision. Activated in CI via
   `-P …,vulkan-linux,vulkan-linux-aarch64,…`.
4. **`README.md`** — classifier table + dependency snippets.

`src/main/resources_linux_vulkan/` is git-ignored (staged by CI, never committed). GPU runtime
`libvulkan.so.1` is supplied by the consumer's driver — nothing is bundled (same policy as every GPU
classifier).

**Windows arm64 CPU (default JAR, no classifier).** `build-windows-arm64` runs natively on GitHub's
free `windows-11-arm` runner (`ilammy/msvc-dev-cmd` `arch: arm64`, Ninja Multi-Config, `-DOS_ARCH=aarch64`,
build + `ctest`). It emits to the **canonical** `resources/.../Windows/aarch64/` and uploads
`Windows-aarch64-libraries`, which the `package`/`publish-*` `*-libraries` glob merges into the default
tree — so it ships in the **default** JAR alongside Windows x86-64 / x86 (like those, it is not a
classifier). No Java change was needed: `OSInfo` already maps a Windows-on-ARM JVM (`os.arch=aarch64`)
to `Windows/aarch64` (it isn't in `archMapping`, so it falls through `translateArchNameToFolderName`).
sccache is intentionally omitted (the shared install step pulls the x86_64 sccache zip; not worth an
arm64 path for one CPU job — `build.bat` just builds uncached). **Compiler: `clang-cl`, not MSVC
`cl.exe`.** ggml's `ggml-cpu/CMakeLists.txt` aborts with *"MSVC is not supported for ARM, use clang"*
via `if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")`; `clang-cl` (LLVM's MSVC-compatible driver)
satisfies that guard (compiler id `"Clang"`) while keeping CMake's `MSVC=TRUE`, so the static `/MT` CRT
block still applies and the generator stays Ninja Multi-Config. The job passes
`-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl`; `msvc-dev-cmd` supplies the MSVC
headers/libs/linker **and the bundled clang-cl/lld-link** (`VC\Tools\Llvm\ARM64`), so no separate
LLVM install is needed. It also passes **`-DGGML_OPENMP=OFF`**: with clang-cl, ggml links LLVM's
OpenMP (`libomp.lib` → `libomp140.aarch64.dll` at runtime), which — unlike MSVC's ambient
`vcomp140.dll` on x64 — is not on `PATH`, so the test exe (and any consumer) failed to launch with
`0xc0000135` (`STATUS_DLL_NOT_FOUND`). Disabling OpenMP makes ggml use its own `std::thread`
threadpool, leaving the arm64 `jllama.dll` self-contained (the x86_64/x86 jobs keep OpenMP via MSVC
`vcomp`). (Upstream llama.cpp instead cross-compiles arm64 from an
x64 runner with `vcvarsall amd64_arm64` + a `clang`/`clang++` toolchain file and no arm64 tests; the
native-runner + `clang-cl` route here keeps the `/MT` CRT and lets `ctest` run on real ARM hardware.)

## Additional GPU-backend classifiers (ROCm/HIP, SYCL, Win-arm64 OpenCL, OpenVINO)

Eight further GPU classifiers extend the matrix toward upstream llama.cpp's full release set. They
follow the **exact same 5-place wiring** as the CUDA/Vulkan classifiers (no special cases — KISS): a
`CMakeLists.txt` backend branch, a `publish.yml` build job (in `package.needs`, **fail-loud** — a
broken build reds the pipeline, same policy as every GPU job), a `pom.xml` classifier profile, a
`README.md` row, and a git-ignored `resources_*` tree. All are **build-only** (GitHub runners have no
matching GPU) and bundle **no** vendor runtime.

| Classifier | GGML flag(s) | Job runner / toolchain | Tree |
|---|---|---|---|
| `rocm-linux-x86-64` | `GGML_HIP=ON -DAMDGPU_TARGETS=…` | `ubuntu-latest` + ROCm apt repo (`/opt/rocm/llvm/bin/clang`) | `resources_linux_rocm` |
| `rocm-windows-x86-64` | `GGML_HIP=ON` | `windows-2025-vs2026` + AMD HIP SDK | `resources_windows_rocm` |
| `sycl-fp16-linux-x86-64` | `GGML_SYCL=ON -DGGML_SYCL_F16=ON` (`icx`/`icpx`) | `ubuntu-latest` + Intel oneAPI apt | `resources_linux_sycl_fp16` |
| `sycl-fp32-linux-x86-64` | `GGML_SYCL=ON` (`icx`/`icpx`) | `ubuntu-latest` + Intel oneAPI apt | `resources_linux_sycl_fp32` |
| `sycl-windows-x86-64` | `GGML_SYCL=ON` (`icx`) | `windows-2025-vs2026` + oneAPI installer | `resources_windows_sycl` |
| `opencl-windows-aarch64` | `GGML_OPENCL=ON …ADRENO_KERNELS=ON` (clang-cl, `GGML_OPENMP=OFF`) | `windows-11-arm` (arm64 CPU job's toolchain) | `resources_windows_opencl` (arch subdir `aarch64`) |
| `openvino-linux-x86-64` | `GGML_OPENVINO=ON` | `ubuntu-latest` + OpenVINO apt | `resources_linux_openvino` |
| `openvino-windows-x86-64` | `GGML_OPENVINO=ON` | `windows-2025-vs2026` + OpenVINO archive | `resources_windows_openvino` |

Two routing notes mirror existing precedent: **Linux SYCL** ships two precision variants at the *same*
arch, so `CMakeLists.txt` routes them to two *distinct* trees by `GGML_SYCL_F16` (fp16 vs fp32).
**Windows OpenCL** now holds both `x86_64` (desktop ICD) and `aarch64` (Snapdragon/Adreno) in the one
`resources_windows_opencl` tree, split by the `opencl-windows` / `opencl-windows-aarch64` profiles'
arch-scoped `<includes>` — exactly like the `vulkan-linux` / `vulkan-linux-aarch64` split.

The vendor toolchain install steps in `publish.yml` are **first-pass** (apt repos / vendor installers
pinned to a specific version): if a URL/version 404s in CI, the job fails loud and the step is adjusted
— the failure is intentional signal, not a regression to hide behind `continue-on-error`.
`src/main/resources_{linux_rocm,windows_rocm,linux_sycl_fp16,linux_sycl_fp32,windows_sycl,linux_openvino,windows_openvino}/`
are all git-ignored (staged by CI, never committed).

## All-backends server fat jars (GitHub Release assets, never Maven Central)

Every pipeline run assembles **per-OS multi-backend server fat jars** and, on the release
paths, attaches them to GitHub: `llama-<version>-all-<os>-<arch>-jar-with-dependencies.jar`
for `linux-x86-64`, `linux-aarch64`, `windows-x86-64`, `windows-aarch64`, plus the default
CPU fat jar — each with a `.sha256` **and** a detached GPG `.asc` signature. They are
**download assets only**: the Central deploy invocations run without the `assembly` profile
and are untouched. (The cross-repo "fat jar → GitHub Release, never Central, signed `.asc`"
convention shared with BAF and srcmorph is documented in
[`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md).)

Mechanism (three pieces):

1. **`.github/package-fatjars.sh`** — run by the `package-fatjars` job (`needs: [package]`,
   downloads `llama-jars`). Enumerates the `<classifier>` set from `llama/pom.xml` (source of
   truth) and cross-checks it in **both** directions against the built classifier jars; parses
   each classifier as `<backend>-<os>-<arch>`; **fails loud** on unparseable classifiers,
   backends missing from its priority table, missing native trees, or zip-update corruption
   (entry list, `Main-Class`, sample byte-compare). Excluded by design: `msvc-windows`
   (redundant CPU variant) and `opencl-android-aarch64` (no `java -jar` on Android). For each
   OS/arch it copies the default fat jar and adds every backend's native tree under
   `net/ladenthin/llama/<OS>/<ARCH>/<backend>/` plus a **`jllama-backends.txt`** manifest
   (backends in priority order `cuda13 rocm sycl-fp16 sycl-fp32 sycl vulkan opencl openvino`;
   extra tokens per line list sibling files such as openvino-windows' bundled `OpenCL.dll`).
   **A new classifier fails this script until it is consciously ranked/excluded** — that is the
   no-silent-gaps guarantee.
2. **`LlamaLoader` backend selection** — when (and only when) the manifest resource exists,
   the loader tries each backend subdirectory in order: extract into a per-backend temp subdir
   (`jllama-backend-<name>/`; backends share file names), load manifest extras first, then the
   backend's `jllama` library. A load failure (missing vendor runtime → `UnsatisfiedLinkError`)
   moves to the next backend; after the list it falls back to the default CPU natives. System
   property `net.ladenthin.llama.backend` forces one backend (fail-loud) or `default`/`cpu`.
   Jars without a manifest take the unchanged legacy path. A backend whose extra module is
   already resident from a previously failed attempt is skipped (by-name import cross-wiring).
3. **`publish.yml` wiring** — `smoke-fatjar-linux` / `smoke-fatjar-windows` run the
   `all-<os>-x86-64` jar via real `java -jar` on GPU-less runners (cached draft model,
   `--chat-template chatml`): poll `/health` to 200, assert a `/v1/chat/completions` choice,
   and require the loader's backend-selection log line. `publish-snapshot`/`publish-release`
   `need` `package-fatjars` + both smokes (fail-loud gating); `github-release-signed` and
   `github-snapshot` additionally download `llama-fatjars` into their asset directory, then
   **GPG-sign each fat jar** via `.github/sign-fatjars.sh` (a detached `.asc` alongside the
   `.sha256`), so the fat jars land signed on the tag release and the rolling `snapshot`
   pre-release. Both jobs declare `environment: maven-central` (where the signing key secret is
   scoped; it has no approval gate) and `checkout` the repo so the script is present. Signing
   happens in these attach jobs — not in `package-fatjars` — because only this dispatch-gated
   release path receives the key.

A backend loading successfully but finding **zero usable devices** (e.g. CUDA toolkit
installed, no NVIDIA GPU) is benign: ggml's backend registry contributes no devices and
inference runs on CPU inside that library. The known trade-off is that such a host never
reaches a *different* GPU backend later in the list — the `net.ladenthin.llama.backend`
override is the escape hatch (documented in the README table).

## WebUI (llama.cpp Svelte UI) embedding

The llama.cpp WebUI is **built once in CI and shared to every native build**, then
compiled into `libjllama` so the embedded server (`server-http.cpp`) can serve it.
This repo commits no build outputs, so the assets are produced per-pipeline, never
checked in (same policy as the native libs).

Pipeline (`.github/workflows/publish.yml`):

1. **`build-webui` job** (ubuntu — the *only* job that runs `npm`): resolves the
   pinned `b<nnnn>` tag from `llama/CMakeLists.txt`'s `GIT_TAG`, sparse-checks-out
   `ggml-org/llama.cpp@<tag>` `tools/ui`, runs the upstream Svelte build
   (`npm ci && npm run build`), gzips `dist/` into `dist/_gzip/` (LLAMA_UI_GZIP
   parity), builds the self-contained `llama-ui-embed` host tool (plain C++17, **no
   npm**) and runs it to produce the platform-independent **`webui-generated/ui.cpp`
   + `ui.h`**, uploaded as the `webui-generated` artifact.
2. **Every native build job** (`needs: [startgate, build-webui]`) downloads that
   artifact into `webui-generated/` before building. npm never runs in the dockcross
   cross-compilers (which have no node) or per-platform.
3. **CMake** (the "WebUI assets" block in `CMakeLists.txt`): if
   `webui-generated/ui.cpp` + `ui.h` exist, compiles `ui.cpp` in and adds its dir to
   the include path — the generated `ui.h` `#define`s `LLAMA_UI_HAS_ASSETS`, which
   activates `server-http.cpp`'s static-asset routes. If absent, it falls back to the
   empty-asset stub `src/main/cpp/webui_stub/ui.h` (no embedded UI) so local builds —
   and any job without the artifact — still build and run.

The WebUI version **auto-follows** the pinned `GIT_TAG`: a llama.cpp version bump
needs no extra step here, `build-webui` re-reads the tag and rebuilds the matching UI.

**Building the WebUI locally** (optional — a plain `cmake` build uses the stub and
ships no UI):
```bash
# needs node/npm + network; embed.cpp is plain C++17 (no npm)
git clone --depth 1 --branch b10092 https://github.com/ggml-org/llama.cpp /tmp/lc
( cd /tmp/lc/tools/ui && npm ci && npm run build \
  && ( cd dist && find . -type f -not -path './_gzip/*' \
       | while read -r f; do mkdir -p "_gzip/$(dirname "$f")"; gzip -9 -c "$f" > "_gzip/$f"; done ) \
  && g++ -O2 -std=c++17 -o /tmp/llama-ui-embed embed.cpp )
mkdir -p webui-generated
/tmp/llama-ui-embed webui-generated/ui.cpp webui-generated/ui.h /tmp/lc/tools/ui/dist
cmake -B build && cmake --build build --target jllama   # now embeds the real UI
```
`webui-generated/` is git-ignored.

## CI build cache & parallelism (sccache + Depot)

The native build dominates CI time (134 llama.cpp model TUs + ggml + the 16.6k-line
`httplib.cpp`, all at `-O3`). Two knobs in **`.github/build.sh`**, both behind the
`use_cache` `workflow_dispatch` input (default **true**), keep it fast and stop the macOS
runners OOM-ing.

**`BUILD_JOBS` — compile parallelism.** `build.sh` builds with `cmake --build -j${BUILD_JOBS}`
(default: all cores, via portable `nproc` → `sysctl -n hw.ncpu` → `4` detection). GitHub's
~7 GB **macOS arm64** runners OOM under full `-j` when `httplib.cpp` co-schedules with the
model TUs; the runner is then killed as **SIGTERM / exit 143** ("received a shutdown
signal"), which *looks* like a timeout but is an out-of-memory kill. The three macOS build
jobs therefore set `BUILD_JOBS: 2` to bound peak memory.

**`sccache` → Depot Cache — shared compiler cache.** When `USE_CACHE=true` **and** `sccache`
plus a cache token are present, `build.sh` adds
`-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache` and prints
`sccache --show-stats`. **Per-job cache summary:** when running in CI (`GITHUB_STEP_SUMMARY` set),
`build.sh`/`build.bat` also parse those stats and append a small `### sccache statistics` table
(`Cache hits | Requests | Hit rate`) to the job summary — the sccache/Depot analogue of upstream
llama.cpp's `ccache-action` "CCache Statistics" table, per-job (GitHub does not merge job
summaries). It is best-effort (skipped silently if the numbers can't be parsed) and only emitted
when sccache was actually the launcher; local runs (no `GITHUB_STEP_SUMMARY`) are untouched. The
cache lives in **Depot Cache** over sccache's **WebDAV** backend:

- `SCCACHE_WEBDAV_ENDPOINT: https://cache.depot.dev`
- `SCCACHE_WEBDAV_TOKEN: ${{ secrets.DEPOT_TOKEN }}` — a Depot **organization** token, stored
  as the repo secret **`DEPOT_TOKEN`**.

Because `sccache` is **content-addressed** and llama.cpp is pinned (`GIT_TAG b10092`), the
~280 upstream object files are byte-identical every run, so a warm cache recompiles only the
*changed* files. Depot's cache is **shared across all branches** (unlike GitHub's
per-branch `actions/cache`), so every branch builds incrementally; a `b<nnnn>` version bump
naturally invalidates the upstream entries (their content changed) with no manual step. It
stays `-O3` and is **bit-identical** to a clean build (release-safe).

**Safety / transparency.** It is **inert** until `DEPOT_TOKEN` is configured and on **fork
PRs** (secrets are hidden there) — those simply compile normally; the `Install sccache` step
is `continue-on-error`; and `use_cache=false` forces a pristine, from-scratch build. Crucially,
`build.sh` runs a **probe-compile health-check** (`sccache_can_wrap_compiler`) before trusting
sccache as the launcher: it compiles a trivial TU *through* sccache, and only sets
`-DCMAKE_{C,CXX}_COMPILER_LAUNCHER=sccache` if that succeeds. So a sccache that is present but
**crashes** (the in-container panic that stalled phase 2) also falls back to an uncached, green
`-O3` build — it logs the Rust panic backtrace (and the detached server's `SCCACHE_ERROR_LOG`,
when a job sets one) for diagnosis but never reds the build. This closes the gap the original
absent-only guard left.

**The fork-PR `.sccache_check` 403 (mac-only symptom) and its two guards.** A fork PR (e.g.
`vaiju1981/java-llama.cpp` → upstream) runs with secrets withheld, so `SCCACHE_WEBDAV_TOKEN`
(`= secrets.DEPOT_TOKEN`) is **empty**. Depot rejects the unauthenticated server-startup
`.sccache_check` with **403 Forbidden** (`PermissionDenied (temporary) … Forbidden`), and
because sccache treats a failed startup check as fatal, *every* TU dies. The symptom looked
**mac-only** purely because of an asymmetry in how sccache reaches `PATH`: the macOS jobs ran
`brew install sccache` **unconditionally** (`if: USE_CACHE == 'true'`), whereas the
Linux/dockcross/aarch64 jobs only **fetch** sccache when a token is present (the `[ -n
"$SCCACHE_WEBDAV_TOKEN…" ]` guard in `build.sh`'s fetch block) — so on a tokenless fork PR
mac was the only platform with sccache on `PATH` to misfire. Two independent guards now prevent
it: **(1)** every `Install sccache` step is gated `if: env.USE_CACHE == 'true' && env.SCCACHE_WEBDAV_TOKEN
!= ''`, so a tokenless fork PR never even installs sccache (mac now matches Linux); and **(2)**
`build.sh`'s build step **retries once without the launcher** when the build fails *and* the
output shows an sccache cache error (`sccache: error` / `Server startup failed` / `cache storage
failed`) — a clean uncached `-O3` rebuild that is content-identical and release-safe. The retry
is gated on that error signature so a genuine compile error still fails fast and is reported
(no wasteful uncached rebuild). Guard (2) also covers an *intermittent* 403 that strikes a
valid-token job mid-build, which the one-shot probe cannot foresee.

**Rollout.** **Phase 1 — DONE & proven: the 3 macOS build jobs** (slowest + OOM-prone) —
`brew install sccache` + the env above + `BUILD_JOBS: 2`. macOS build dropped **~40 min → ~6 min**
with a warm cache. **Phase 2 — DONE: all 5 dockcross cross-compile jobs** now have the same
steady-state env (`USE_CACHE` + `SCCACHE_WEBDAV_*` + `DOCKCROSS_ARGS`). The probe makes it safe
to enable them all at once — any container where sccache crashes falls back to an uncached green
build automatically. (The first attempt enabled all four at once without the probe and was
reverted: the static-musl sccache v0.8.2 panicked in-container and redded the build. With
v0.16.0 + the probe this is no longer a risk.) Job-by-job status:
1. `crosscompile-linux-x86_64` (manylinux2014) — ✅ **verified green** in PR #245: sccache
   **v0.16.0** probe passed in-container (devtoolset-10 gcc), `sccache ON` over Depot WebDAV,
   warm cache 277/278 hits (99.64%), 1m46s build time.
2. `crosscompile-linux-x86_64-cuda` (via `build_cuda_linux.sh`, which execs `build.sh`) —
   ✅ **verified green with nvcc caching, full-arch always.** `build.sh` also wraps nvcc
   (`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache`, scoped to CUDA builds), so both the gcc C/C++ TUs
   (134 model files + ggml + httplib) **and** the per-arch `.cu` device passes cache over Depot.
   CI dropped the single-arch validation shortcut (`CUDA_FAST_BUILD`/`CUDA_ARCH` removed from the
   job) — every run builds the full arch set and leans on the warm cache for speed. A warm run hit
   **100%** on CUDA / CUBIN / device-code (139 CUDA hits, 99.86% overall, 3 misses), cutting the job
   from **~51 min cold to ~15 min warm**. The first-run debug diagnostics (`SCCACHE_LOG` /
   `SCCACHE_ERROR_LOG` / `RUST_BACKTRACE`) were dropped once confirmed; `sccache --show-stats` still
   prints the hit table every run.
3. `crosscompile-linux-aarch64` — ✅ **enabled**, now a **native `ubuntu-24.04-arm` build** (not
   dockcross): `build.sh` self-fetches the aarch64 static-musl sccache (the fetch block in
   `build.sh` maps `uname -m` → `x86_64`/`aarch64`) and the probe guards it. See "Linux aarch64:
   native ARM build" below for why it moved off the cross-compiler.
4. `crosscompile-android-aarch64` — ✅ **enabled** (same steady-state env; probe guards it).
5. `crosscompile-android-aarch64-opencl` — ✅ **enabled**. `build_opencl_android.sh` stages the
   OpenCL headers/loader, then delegates the jllama cmake build to `build.sh` via `exec`
   (same pattern as `build_cuda_linux.sh`), so it inherits the probe and launcher automatically.

Per-job recipe: add `env:` { `USE_CACHE`, `SCCACHE_WEBDAV_ENDPOINT`, `SCCACHE_WEBDAV_TOKEN` } and
`DOCKCROSS_ARGS: "-e SCCACHE_WEBDAV_ENDPOINT -e SCCACHE_WEBDAV_TOKEN -e USE_CACHE"` — the
dockcross wrapper only forwards host env it is explicitly told to via `-e`. The fetched sccache
version is the `SCCACHE_DL_VERSION` knob in `build.sh` (default **0.16.0**; overridable per-job
to try a different build against a container that crashed another). **Windows** is handled
separately (the Visual Studio generator ignores `CMAKE_*_COMPILER_LAUNCHER`): see
"Windows native classifiers" below — the **default** Windows CPU JAR now uses the **Ninja
Multi-Config** generator (so it caches) with a `build.bat` sccache probe and a direct sccache zip
download (not `mozilla-actions/sccache-action`); the uncached MSVC build ships as the `msvc-windows`
classifier, and the three Windows GPU classifiers (CUDA/Vulkan/OpenCL) use the same Ninja path.

**Cross-repo scope.** This Depot/sccache compiler cache makes sense only for java-llama.cpp —
it is the only sibling repo with a native (C++/JNI) build. It does not apply to the pure-Maven
siblings; why (and why the `DEPOT_TOKEN` org secret and the README "Build cache by Depot" badge
are kept jllama-only) is explained in the cross-repo status under "Deliberate non-parity":
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md).

## Local llama.cpp source patches (`patches/`)

The fetched llama.cpp source is patched before it compiles, via a generic mechanism:

- **`patches/`** (repo root) — drop any number of `*.patch` / `*.diff` files here. They are applied
  in **filename order** (use a numeric prefix, e.g. `0001-`, `0002-`), so keep them independent or
  ordered. Each must be a `git apply`-compatible unified diff with paths relative to the llama.cpp
  source root (`a/common/arg.cpp` / `b/common/arg.cpp`, i.e. `-p1`).
- **`cmake/apply-llama-patches.cmake`** — the applier. Cross-platform (`cmake -P`, so identical on
  Linux/macOS/Windows), **idempotent** (`git apply --reverse --check` skips already-applied patches
  so a reconfigure never double-applies) and **fail-loud** (a patch that no longer applies aborts
  the configure — a stale patch can't be silently dropped from a release build).
- **`CMakeLists.txt`** — wired as the llama.cpp `FetchContent_Declare(... PATCH_COMMAND ...)`, so it
  runs for **every** C++ build (all CI jobs *and* local `cmake -B build`) from one place — no
  per-build-step plumbing.

**On a llama.cpp version bump, every patch must still apply** — if a bump shifts the patched code,
the configure fails with an "does not apply cleanly" error; refresh the diff against the new source
and recommit. Treat `patches/` as part of the upgrade checklist below.

Current patches:

| Patch | Fixes |
|-------|-------|
| `0001-win32-arg-parse-embed-guard.patch` | Windows JNI regression from llama.cpp **#24779** (introduced b9739): on Windows `common_params_parse` re-derived argv from the **process** command line (`GetCommandLineW`) and adopted it, so an embedded/JNI caller (`java.exe`) lost its `--model …` args → "Failed to parse model parameters". b9789 narrowed the unconditional override to a **count-guard** (`if (static_cast<int>(utf8.buf.size()) == argc) { argv = utf8.ptrs.data(); }`), but that is exactly the variant the project already found breaks its Windows server-integration tests (when the embedded argv length coincides with `java.exe`'s). The patch carries the **complete upstream change** (so it can be submitted to llama.cpp verbatim and then dropped here): **(1)** `common_params_parse` parses **exactly the argv it is given** (no `GetCommandLineW` magic) and a new `common_params_parse_main()` wrapper holds the UTF-8 recovery for the standalone tools' `main()` (`common/arg.{cpp,h}`); **(2)** the **~34 standalone `main()` call sites** (every `common_params_parse(argc, argv, …)` across `tools/*`, `examples/*` and the `tests/*` programs) flip to `common_params_parse_main()`; **(3)** a `tests/test-arg-parser.cpp` regression case pins that `common_params_parse` honors a caller-supplied argv. The embedded caller (`jllama.cpp`) keeps calling `common_params_parse` and is never overridden. **Our subproject build compiles only the `arg.{cpp,h}` core** — `LLAMA_BUILD_TOOLS`/`LLAMA_BUILD_TESTS` are OFF for a FetchContent subproject — so the flips + test are applied-but-not-compiled here; they were validated via a one-off `-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=ON` build (the new test compiles and its asserts pass; `test-arg-parser`'s only red there is the live `ggml.ai` download check, which is sandbox-network, not the patch). Because it spans **37 files** it must be refreshed on every llama.cpp bump (the applier fails loud). |
| `0002-server-preserve-caller-load-progress-callback.patch` | Load-progress-callback regression introduced in llama.cpp **b9789**: `server_context::load_model` (`tools/server/server-context.cpp`) now **unconditionally** installs the server's own load-progress reporter on `params_base.load_progress_callback` immediately before `common_init_from_params`, clobbering any callback the embedding caller already set. libjllama's `LoadProgressCallback` feature wires `common_params.load_progress_callback` to a JNI trampoline *before* calling `load_model`, so the bump silently killed it — `LoadProgressCallbackTest` saw zero progress updates and the abort-on-`false` path never threw. The patch guards the assignment with `if (params_base.load_progress_callback == nullptr)`, so the server installs its own reporter **only when the caller hasn't** — a caller-supplied callback survives and fires during load. Standalone `llama-server` (no caller callback, so the field is null) is unaffected. Same JNI-vs-standalone divergence class as `0001`. |
| `0003-pr22393-server-add-slot-prompt-similarity-getter-setter.patch` | **Upstream-PR carry** of [ggml-org/llama.cpp#22393](https://github.com/ggml-org/llama.cpp/pull/22393) ("server : add slot_prompt_similarity getter/setter") while it is still open upstream. Purely additive: adds `server_context::get_slot_prompt_similarity()` / `set_slot_prompt_similarity(float)` (`tools/server/server-context.{cpp,h}`) so an embedding/JNI caller can query and tune the slot-selection threshold at runtime without reloading the model. Verbatim copy of the PR — drop it once a pinned `b<nnnn>` includes the change. |
| `0007-server-attach-http-frontend.patch` | **Adds `llama_server_attach(argc, argv, server_context&)`** so the `NativeServer` *attach mode* can serve an **already-loaded `LlamaModel`** over the full upstream HTTP frontend — no second model load, no `start_loop()`; the LlamaModel's worker keeps driving the shared `server_context` and the HTTP routes post tasks to its queue (the queue is the synchronization point). Mechanically: (1) extracts the common route table + the CORS-proxy/tools **route bindings** out of `llama_server()` into `llama_server_register_common_routes(...)` (shared, so the entry points cannot drift; returns `false` on tools-setup failure) — the per-caller **experimental-feature warnings** (upstream's b10015/#25655 `cors_origins` security warning + the router/MCP/tools `warn_names` block) stay in `llama_server` since they depend on router state the shared helper does not carry; (2) adds `llama_server_attach`, which parses only the HTTP-side argv via `common_params_parse`, starts `g_stream_sessions` GC + `server_http_context`, registers the common routes plus the non-router resumable-streaming handlers, marks ready immediately (model already loaded), and blocks on the HTTP thread until `llama_server_request_shutdown()` — never calling `common_init()`, backend init, `ctx_server.terminate()` or `llama_backend_free()` (the embedding caller owns those). Applies after `0001`+`0006` (same file); closes the "NativeServer — reuse an already-loaded LlamaModel" TODO. Upstream-submittable ("server: let embedding callers attach the HTTP frontend to an existing server_context"). |
| `0008-server-models-worker-cmd-override.patch` | **Makes router mode usable in-JVM.** The router (`server-models.cpp`) spawns each model worker by re-executing its own binary (`get_server_exec_path()` = `/proc/self/exe` & friends) — inside a JVM that binary is `java`, not a llama-server, so embedded router workers could never start. The patch adds env `LLAMA_SERVER_WORKER_CMD` (whitespace-split; read in `server_model_meta::update_args`) which replaces only the leading binary-path token of the rendered worker args, letting an embedding host relaunch workers through its own bootstrap — e.g. `java -cp app.jar net.ladenthin.llama.server.NativeServer` (each worker is then a fresh JVM running the classic single-model `NativeServer`). Exposed in Java as `NativeServer.setWorkerCommand(String...)` (JNI `setenv`); exercised by `RouterModeIntegrationTest` (Linux CI). Upstream-submittable (also useful for containerized/wrapped deployments). |
| `0006-server-embed-native-server-jni.patch` | **Makes `server.cpp`'s `llama_server` embeddable in the JVM** so the `NativeServer` JNI bridge can run the full upstream HTTP server (WebUI included) inside `libjllama` — see "Two server modes" below. b9870 already exposes `int llama_server(int, char**)` (non-static; no `main` in the file), so the patch only adds embedded-mode support: (1) a `g_llama_server_embedded` flag + `llama_server_set_embedded()` / `llama_server_request_shutdown()` (declared in the committed `src/main/cpp/native_server_bridge.h`); (2) skips installing the process-wide SIGINT/SIGTERM handlers when embedded (they would hijack the JVM's); (3) in embedded mode parses the **forwarded** argv via `common_params_parse` instead of `common_params_parse_main` (whose `GetCommandLineW` recovery would pick up `java.exe`'s command line — the same Windows class of bug `0001` fixes). `llama_server_request_shutdown()` mirrors the SIGTERM path (invokes the installed `shutdown_handler` → `ctx_server.terminate()` unblocks `start_loop()`), giving JNI an out-of-band stop since `ctx_server` is loop-local. Applies **after `0001`** (which flips this call site to `common_params_parse_main`), so its context is the post-`0001` tree; regenerate against `0001`+source on a bump. Only touches `tools/server/server.cpp`. |

**`0005` was dropped at the b9981 bump.** Upstream's own `server-context.cpp` picked up an
equivalent — and broader — fix for the same checkpoint-starvation problem: `create_checkpoint`
now tracks `id_task` and evicts a stale checkpoint that sits within `checkpoint_min_step` of a
newer one (instead of our pre-creation duplicate-position check), and the `do_checkpoint` gate
now exempts **every** near-prompt-end checkpoint from the min-step spacing (`near_prompt_end`,
unconditionally) rather than only recurrent/hybrid (`ctx_tgt_seq_rm_type` `FULL`/`RS`) models as
our patch scoped it. The upstream version is a strict superset of what `0005` did, so it applies
cleanly with the patch removed and needs no replacement. If a regression in agentic multi-turn
prefill behavior shows up, re-check `tools/server/server-context.cpp`'s `create_checkpoint` /
`do_checkpoint` logic against this description before reintroducing a local patch.

**`0004` was dropped at the b9982 bump.** Upstream merged
[ggml-org/llama.cpp#23116](https://github.com/ggml-org/llama.cpp/pull/23116) verbatim: the
`b9981...b9982` diff carries the exact same `oaicompat_chat_params_parse` precedence change
(`reasoning_budget_tokens` > `thinking_budget_tokens` alias > server default, plus the per-request
`reasoning_budget_message` override) and the same two `tests/test-chat.cpp` regression cases
(`test_reasoning_budget_tokens_per_request` / `test_reasoning_budget_message_per_request`,
byte-identical body). No local patch needed — the tree already matches what `0004` used to add.

## OuteTTS build-time extraction (`cmake/generate-tts-upstream.cmake`)

The `TextToSpeech` native pipeline reuses llama.cpp's OuteTTS helpers (`tools/tts/tts.cpp`)
**without hand-copying them**. A verbatim copy would be a DRY/maintenance hazard that silently
diverges on every upgrade, and `tts.cpp` cannot simply be added to `target_sources` — it defines its
own `main()`, which would clash at link time (the same reason `tools/server/server.cpp` is excluded
while `server-*.cpp` are compiled in), and all its helpers are `static` (internal linkage), so they
are unreachable from another TU even if it were linked.

Instead the helpers are **DERIVED mechanically at configure time** from the pinned upstream source:

- **`cmake/generate-tts-upstream.cmake`** — reads `${llama.cpp_SOURCE_DIR}/tools/tts/tts.cpp`, keeps
  the pre-`main()` span (the DSP `fill_hann_window`/`irfft`/`fold`/`embd_to_audio`, the prompt/text
  helpers incl. `process_text`'s number-to-words, the `outetts_version` enum), strips `static` from
  the handful the JNI engine calls (giving them external linkage), and extracts the two hard-coded
  default-speaker literals out of `main()` into `extern const` strings. Writes
  `build/tts_generated/tts_upstream_gen.cpp`.
- **`CMakeLists.txt`** — runs the generator via `execute_process` right after
  `FetchContent_MakeAvailable(llama.cpp)`, then compiles the generated TU into `jllama`. The file is
  **never committed** (build artifact, like the native libs / WebUI assets); it is regenerated from
  whatever `tts.cpp` the pinned `GIT_TAG` resolves to, so a version bump is picked up automatically.
- **`src/main/cpp/tts_upstream.h`** — committed, hand-written declarations of the extracted symbols
  (interface facts, not the implementation). `tts_engine.cpp` includes it and links against the
  generated definitions. The in-memory WAV writer (`tts_wav.hpp`) is ours, not extracted.

**Fail-loud on drift (same contract as `patches/`):** the generator asserts every anchor — the
`int main(` split point, each `static <signature>` it de-statics, the `outetts_version` enum
(enumerators + order, kept ODR-identical to the hand-written copy in `tts_upstream.h`), both
`prompt_add` overloads the header declares (the bare `void prompt_add(` prefix de-statics all three
upstream overloads, so the two the header relies on are pinned individually), and both speaker
literals. If an upgrade renames a helper, reorders the enum, or moves a literal, the **configure step
aborts** with a pointer to the generator; if upstream changes a *type*, `tts_upstream.h` stops
matching and the **link fails**. Either way a silent divergence is impossible. On a llama.cpp bump,
re-verify the generator the same way you re-verify `patches/`.

## Upgrading/Downgrading llama.cpp Version

**Runbook (documentation root):** [`docs/upgrade/llama-cpp-version-bump.md`](docs/upgrade/llama-cpp-version-bump.md)
covers the full bump process end-to-end — picking the target (topmost GitHub release, via the atom
feed), **chunking by `git diff` byte-size** (bump straight to the target when the diff is < 100 KiB,
else step through the largest intermediate tag still under the threshold), the
`.github/scripts/llama-next-version.sh` helper that computes the next reviewable step, and the
edit/verify/commit loop below. Use it for any non-trivial bump; the steps here are the mechanical core.

To change the llama.cpp version, update the following **four** files (and re-verify `patches/`):

1. **llama/CMakeLists.txt** — the `GIT_TAG` line for llama.cpp: `GIT_TAG        b8831` (and the
   cosmetic `-DLLAMA_TAG=b8831` a few lines below, passed to the TTS generator — keep them equal)
2. **README.md** — the badge and link line with the version number
3. **CLAUDE.md** — the "Current llama.cpp pinned version" line
4. **llama/src/main/java/net/ladenthin/llama/value/LlamaCppVersion.java** — the
   `LLAMA_CPP_VERSION` constant (the compile-time pin exposed to Java/Kotlin consumers, e.g. the
   Android version badge). It is the *pure-Java mirror* of `GIT_TAG` and must stay equal to it —
   the native `LlamaModel.getLlamaCppBuildInfo()` getter reports the actually-linked build
   (`b<number>-<commit>`), and `NativeLibraryLoadSmokeTest.nativeBuildInfoMatchesPinnedVersionConstant`
   **fails the build** if this constant and the linked binary drift apart.

Example: To upgrade from b8808 to b8831:
```bash
# Edit llama/CMakeLists.txt: change GIT_TAG b8808 to b8831 (and the -DLLAMA_TAG line)
# Edit README.md: change b8808 to b8831 (in both badge and link)
# Edit CLAUDE.md: change b8808 to b8831
# Edit LlamaCppVersion.java: change LLAMA_CPP_VERSION "b8808" to "b8831"
git add llama/CMakeLists.txt README.md CLAUDE.md \
        llama/src/main/java/net/ladenthin/llama/value/LlamaCppVersion.java
git commit -m "Upgrade llama.cpp from b8808 to b8831"
git push -u origin <your-branch>
```

**Note:** Always test the build with `cmake -B build && cmake --build build --config Release` after version changes to catch compatibility issues early.

### Inspecting API changes between versions

Use the GitHub compare URL to diff any two llama.cpp builds:

```
https://github.com/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

Example — what changed between b6721 and b6732:
```
https://github.com/ggml-org/llama.cpp/compare/b6721...b6732
```

The GitHub HTML page may time out for large ranges; fall back to the API:
```
https://api.github.com/repos/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

For individual file content at a specific build:
```
https://raw.githubusercontent.com/ggerganov/llama.cpp/b<VERSION>/common/chat.h
```

### Files to check for API compatibility

The three project C++ files (`jllama.cpp`, `server.hpp`, `utils.hpp`) pull in the following
llama.cpp headers. Any of these can introduce breaking changes on upgrade.

**Include dependency graph:**
```
jllama.cpp / server.hpp / utils.hpp
│
├── arg.h ──────────────────────────► common.h ─┐
├── common.h ──────────────────────────────────►├── ggml-opt.h ──► ggml.h
├── chat.h ─────────────► common.h, peg-parser.h └── ggml-backend.h ──► ggml-alloc.h
├── speculative.h ──────► llama.h, common.h
├── sampling.h ─────────► llama.h, common.h
├── download.h ─────────► (stdlib only, no deps)
├── log.h ──────────────► ggml.h
├── llama.h ────────────────────────────────────► ggml.h, ggml-cpu.h, ggml-backend.h, ggml-opt.h
│                                                  └── llama-cpp.h ──► llama.h
├── json-schema-to-grammar.h
├── base64.hpp
├── mtmd.h
└── mtmd-helper.h
```

**Priority-ordered review list for upgrade diffs** (highest break risk first)

The top 8 rows cover all known API-level breaking changes from b5022 → b8831.
For future upgrades, provide diffs for at least these 8 files rather than the full patch.
Also review the project `CMakeLists.txt` for build-system-level breaks (e.g. renamed link targets, new required headers) — those are not visible in header file diffs alone.

| File | What to watch for |
|------|-------------------|
| `common/common.h` | `common_params`/`common_params_speculative` struct fields, `model_alias` container type, `common_init_result` shape, `build_info` symbol (removed in b8831 — now `llama_build_info()` from `build-info.h`) |
| `common/chat.h` | `common_chat_parser_params` (was `common_chat_syntax`), `to_json_oaicompat`, `common_chat_msg_diff_to_json_oaicompat`, `set_tool_call_ids` |
| `common/speculative.h` | `common_speculative_init`, `common_speculative_draft`, `common_speculative_accept` signatures, struct names |
| `tools/mtmd/mtmd.h` | `mtmd_context_params` fields, `image_marker`/`media_marker` API, deprecated symbols (was `common/mtmd.h` before ~b8190) |
| `include/llama-cpp.h` | `common_init_result_ptr` type, access pattern changes (`.get()` vs `->method()`) |
| `common/arg.h` | `n_parallel` sentinel value, what moved to `download.h` across versions |
| `include/llama.h` | Core llama_ function signatures, token types, `llama_model_ptr`, renamed structs |
| `common/download.h` | `common_remote_params` struct, `headers` field format (string vs key-value pair) |
| `common/common.cpp` | Implementation of any inline API used directly |
| `common/speculative.cpp` | Speculative decoding implementation details |
| `common/chat.cpp` | Chat parsing implementation |
| `common/sampling.h` | Sampler API, `common_sampler_*` functions |
| `common/log.h` | Log macro signatures |
| `tools/mtmd/mtmd-helper.h` | Multimodal helper functions |
| `common/json-schema-to-grammar.h` | Grammar API |
| `ggml/include/ggml.h` | `ggml_type` enum values (e.g. `GGML_TYPE_F16`), tensor primitives |
| `ggml/include/ggml-backend.h` | Backend/device abstraction types |
| `ggml/include/ggml-opt.h` | Optimizer params pulled in via `common.h` |

**Safe to skip** (have never caused a break; not used directly by project code):
`common/sampling.h`, `common/log.h`, `tools/mtmd/mtmd-helper.h`, `common/json-schema-to-grammar.h`,
`ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `ggml/include/ggml-opt.h`,
`ggml-alloc.h`, `ggml-cpu.h`, `peg-parser.h`, `base64.hpp`

For the full record of upstream API breaks across version ranges (b5022 &#x2192; current), including which rows required project source changes vs. which stayed inside upstream-compiled translation units, see [`docs/history/llama-cpp-breaking-changes.md`](docs/history/llama-cpp-breaking-changes.md). When bumping the `llama.cpp` version, append a new row to that file covering the upgrade range.

## Build Commands

### Java (Maven)
```bash
mvn compile          # Compiles Java and generates JNI headers
mvn test             # Run all tests (requires native library and model files)
mvn package          # Build JAR
mvn -P assembly package  # Also build the fat jar-with-dependencies uber JAR (library + Java deps + native libs); CI builds it and uploads it in the `llama-jars` artifact
mvn test -Dtest=LlamaModelTest#testGenerate  # Run a single test method
```

### Native Library (CMake)
Must run `mvn compile` first to generate JNI headers, then:
```bash
# CPU only
cmake -B build
cmake --build build --config Release

# CUDA (Linux)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (macOS)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

# Optional: enable model downloading via URL
cmake -B build -DLLAMA_CURL=ON
```

Built libraries are placed in `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.

### Building the native library for local Java tests

`mvn test` does **not** build the native library — Maven only compiles Java
and runs surefire. The shared library must already exist on disk under the
platform-specific resource path that `LlamaLoader` resolves at runtime.
Without it the JVM throws `UnsatisfiedLinkError` and every Java test fails
immediately (it does not auto-skip).

The output path is derived by `CMakeLists.txt` from `OS_NAME` and `OS_ARCH`
detected by the helper script `.github/dockcross/dockcross-resolve-host`
(falls back to `uname` on hosts where the script is absent). The mapping
mirrors `OSInfo.translateOSNameToFolderName` on the Java side, so the same
folder name is produced on both ends.

| Host | Library file | Resource path produced by `cmake --build` |
|------|--------------|-------------------------------------------|
| Linux x86_64 | `libjllama.so` | `src/main/resources/net/ladenthin/llama/Linux/x86_64/` |
| Linux aarch64 | `libjllama.so` | `src/main/resources/net/ladenthin/llama/Linux/aarch64/` |
| macOS Apple Silicon | `libjllama.dylib` | `src/main/resources/net/ladenthin/llama/Mac/aarch64/` |
| macOS Intel | `libjllama.dylib` | `src/main/resources/net/ladenthin/llama/Mac/x86_64/` |
| Windows x86_64 | `jllama.dll` | `src/main/resources/net/ladenthin/llama/Windows/x86_64/` |

On every platform exactly **one** `jllama` library is produced: `CMakeLists.txt` forces
`BUILD_SHARED_LIBS OFF`, so upstream `llama` and `ggml` are static libraries linked into
`jllama` (the `RUNTIME_OUTPUT_DIRECTORY_*` block that also names the `llama`/`ggml` targets
is a no-op for them — verified against the published 5.0.5 jars, which contain only
`jllama.dll` per Windows arch). `LlamaLoader` accordingly extracts and loads a single file
from the jar (plus `ggml-metal.metal` on macOS). Historical note: upstream kherud once
shipped split `ggml` + `jllama` libraries, which is where stale "three co-located DLLs"
claims came from.

End-to-end local workflow for running Java tests:

```bash
# 1. Generate JNI headers (one-time per Java API change)
mvn -q compile

# 2. Configure + build the native library for the current host
cmake -B build
cmake --build build --config Release -j$(nproc)
# The shared lib lands directly in src/main/resources/.../{OS}/{ARCH}/ —
# no separate install step is needed.

# 3. Ensure model files referenced by tests are present under models/.
#    The default test models (downloaded by CI in publish.yml) are:
curl -L --fail "$MODEL_URL"          --create-dirs -o models/codellama-7b.Q2_K.gguf
curl -L --fail "$RERANKING_MODEL_URL" --create-dirs -o models/jina-reranker-v1-tiny-en-Q4_0.gguf
curl -L --fail "$DRAFT_MODEL_URL"     --create-dirs -o models/AMD-Llama-135m-code.Q2_K.gguf
curl -L --fail "$REASONING_MODEL_URL" --create-dirs -o models/Qwen3-0.6B-Q4_K_M.gguf

# 4. Run tests. Tests that need a model file self-skip via Assume.assumeTrue()
#    when their GGUF is absent, so partial model availability is OK.
mvn test
# CPU-only host (no GPU): pin GPU layers to 0
mvn test -Dnet.ladenthin.llama.test.ngl=0
# Run a single test class or method
mvn test -Dtest=MemoryManagementTest
mvn test -Dtest=LlamaModelTest#testGenerateAnswer
```

**Optional models** referenced by individual tests are gated on a system
property so CI can skip them cleanly when the GGUF is not downloaded.
The full property → consumer → default table for every `net.ladenthin.llama.*`
property the library understands (runtime + test) is the user-facing
**[System Properties Reference](README.md#system-properties-reference)** in
the README. The summary below covers only the optional-model bindings:

| Property | Default test that uses it | Model |
|----------|---------------------------|-------|
| `net.ladenthin.llama.nomic.path` | `LlamaEmbeddingsTest#testNomicEmbedLoads` | `nomic-embed-text-v1.5.f16.gguf` (issue #98 regression) |
| `net.ladenthin.llama.vision.model` | `MultimodalIntegrationTest` | `SmolVLM-500M-Instruct-Q8_0.gguf` (any vision-capable GGUF works) |
| `net.ladenthin.llama.vision.mmproj` | `MultimodalIntegrationTest` | matching mmproj for the vision model, e.g. `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` |
| `net.ladenthin.llama.vision.image` | `MultimodalIntegrationTest` | committed default `src/test/resources/images/test-image.jpg`; override to any png/jpeg/webp/gif on disk |
| `net.ladenthin.llama.audio.model` | `AudioInputIntegrationTest` (llama.cpp discussion #13759) | audio-input model GGUF, e.g. `ultravox-v0_5-llama-3_2-1b.gguf` |
| `net.ladenthin.llama.audio.mmproj` | `AudioInputIntegrationTest` | matching audio mmproj/encoder, e.g. `mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf` |
| `net.ladenthin.llama.audio.input` | `AudioInputIntegrationTest` | committed default `src/test/resources/audios/sample.wav`; override to any `.wav`/`.mp3` on disk |
| `net.ladenthin.llama.tts.ttc.model` | `TtsIntegrationTest` | OuteTTS text-to-codes model, e.g. `OuteTTS-0.2-500M-Q4_K_M.gguf` |
| `net.ladenthin.llama.tts.vocoder.model` | `TtsIntegrationTest` | matching codes-to-speech vocoder, e.g. `WavTokenizer-Large-75-F16.gguf` |

Run those tests by setting the property:
```bash
mvn test -Dtest=LlamaEmbeddingsTest#testNomicEmbedLoads \
         -Dnet.ladenthin.llama.nomic.path=models/nomic-embed-text-v1.5.f16.gguf
mvn test -Dtest=MultimodalIntegrationTest \
         -Dnet.ladenthin.llama.vision.model=models/SmolVLM-500M-Instruct-Q8_0.gguf \
         -Dnet.ladenthin.llama.vision.mmproj=models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
# The vision.image property defaults to src/test/resources/images/test-image.jpg
# (a CC-BY-4.0 / MIT-granted photo of flowers and bees by the project author);
# override only if you want to test a different image.

# Audio input (Ultravox / Qwen2.5-Omni; the audio clip has no committed default):
mvn test -Dtest=AudioInputIntegrationTest \
         -Dnet.ladenthin.llama.audio.model=models/ultravox-v0_5-llama-3_2-1b.gguf \
         -Dnet.ladenthin.llama.audio.mmproj=models/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf \
         -Dnet.ladenthin.llama.audio.input=/path/to/speech.wav   # optional: defaults to the committed src/test/resources/audios/sample.wav
mvn test -Dtest=TtsIntegrationTest \
         -Dnet.ladenthin.llama.tts.ttc.model=models/OuteTTS-0.2-500M-Q4_K_M.gguf \
         -Dnet.ladenthin.llama.tts.vocoder.model=models/WavTokenizer-Large-75-F16.gguf
```

`MultimodalIntegrationTest` self-skips when any of the three vision properties
points at a missing path, so a partial setup (just the vision model + the
committed image, no mmproj) lets the test class load without erroring.

**Restricted-network environments.** Some hosts (e.g. ephemeral remote
execution sandboxes) block outbound traffic to `huggingface.co`. In that
case downloading models for the Java tests is not possible from the host
itself; the native library can still be built and the C++ test suite
(`ctest --test-dir build`) still runs because it depends only on the
upstream sources fetched at CMake configure time. Java tests should then
be exercised either in CI (via `.github/workflows/publish.yml`) or on a
developer machine with HF access; pre-staged models can also be uploaded
into `models/` out-of-band.

**Verifying the native library *loads* without models (model-free smoke).**
Even with HuggingFace blocked you can still do the one piece of *real native*
verification that does not need a GGUF: confirm the library loads and its
`JNI_OnLoad` resolves every Java class it looks up by name. The model-gated
tests cannot do this in a restricted sandbox — they self-skip via
`Assume.assumeTrue(model present)` **before** the lib is ever loaded, so a plain
`mvn test` is silent on load-time breakage. The full local recipe:

```bash
# 1. Build the native lib locally (FetchContent pulls llama.cpp from GitHub,
#    which is reachable even when huggingface.co is not):
mvn -q compile
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j$(nproc)   # -> src/main/resources/.../<os>/<arch>/libjllama.so
# 2. Force LlamaModel.<clinit> (System.load -> JNI_OnLoad) with no model:
mvn test -Dtest=NativeLibraryLoadSmokeTest
```

`NativeLibraryLoadSmokeTest` (in the `loader` package) calls
`Class.forName("net.ladenthin.llama.LlamaModel")`, which runs
`LlamaLoader.initialize() -> System.load() -> JNI_OnLoad`, which in turn calls
`FindClass(...)` for every JNI-referenced Java class. It **passes** when the lib
loads cleanly, **fails** if the native-resource path in `LlamaLoader` is wrong
(lib not found) or a `FindClass`/field-signature FQN in
`src/main/cpp/jllama.cpp` is stale after a Java package move (lib loads but
`JNI_OnLoad` throws `NoClassDefFoundError: net/ladenthin/llama/...`), and
**self-skips** when `libjllama` is not on the classpath (pure-Java checkout, no
CMake build) so it never breaks a build-less `mvn test`.

Both of those failure modes shipped on a branch once — the layered-package
restructure left (a) `LlamaLoader.getNativeResourcePath()` deriving the resource
root from the loader's own package (which moved to `…loader`) and (b)
`jllama.cpp` still `FindClass`-ing the old flat paths — and neither was visible
to a local `mvn test` (model tests skipped) or to the pure-Java unit tests.
**When you move a Java class the JNI layer references by name** (`LlamaModel`
[root], `exception.LlamaException`, `value.LogLevel`, `args.LogFormat`,
`callback.LoadProgressCallback`), update the matching `FindClass` / `"L…;"`
signature string in `src/main/cpp/jllama.cpp` and keep the native-resource root
anchored at `net/ladenthin/llama/` in `LlamaLoader.NATIVE_RESOURCE_BASE` (it must
not track the loader's own Java package). This is the same
"FQN/path not updated after a package move" class as the stale
`spotbugs-exclude.xml`, PIT `targetClasses`, and `CMakeLists.txt` OSInfo repairs.

### Code Formatting

C++ formatting is **enforced in CI** (`.github/workflows/clang-format.yml`) with a **pinned**
clang-format — currently **22.1.5**, installed via `pip install clang-format==22.1.5`. Format with
that exact version before committing; a different clang-format version reflows code differently and
will fail the check.

```bash
pip install "clang-format==22.1.5"
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp src/test/cpp/*.cpp   # Format C++ code
```

The generated JNI header `src/main/cpp/jllama.h` (produced by `javac -h`) is intentionally excluded.
To bump the enforced version, update the pin in **both** the workflow (`CLANG_FORMAT_VERSION`) and
this line, then reformat the whole tree with the new version in the same commit.

**`.clang-format` sets `SortIncludes: Never` — do not re-enable include sorting.** The project has
order-sensitive includes (see the "Include order rule" above): the upstream `server-*.h` headers and
`utils.hpp` must precede `json_helpers.hpp` / `jni_helpers.hpp`, which use the `json` alias those
headers define. Alphabetical sorting moves the helper headers first and breaks the build with
`'json' does not name a type` (it slips past a local build whose toolchain resolves `json` anyway,
but fails the manylinux/aarch64/Android CI compilers). Keep include order manual.

### Javadoc — must build cleanly before `mvn package`

The release packaging job runs `mvn package` with the `release` profile, which attaches
a javadoc jar via `maven-javadoc-plugin`. The plugin treats Javadoc tool **errors** as
build failures (warnings are tolerated). After changing any public/protected Java API,
verify the javadoc build succeeds locally:

```bash
mvn clean javadoc:jar -DskipTests=true -Dgpg.skip=true
# expected: BUILD SUCCESS
```

Common Javadoc errors that fail the build (not warnings):

- **Unbalanced HTML**: `</p>` without a matching `<p>`, mismatched `<ul>`/`<li>`, stray
  closing tags. Symptom: `error: unexpected end tag: </p>`.
- **Invalid `{@link …}` targets**: typo'd class, method, or parameter name.
- **Self-closing void HTML elements written as `<br>` inside `<pre>` blocks** in HTML5
  mode (rare but seen).

Common Javadoc *warnings* (do not fail the build, but should be cleaned up on new code):

- `no main description` — a doc comment containing only `@param`/`@return`/`@throws`
  tags with no leading prose. Fix: add a one-line description before the tags.
- `no @return` / `no @param` — public method missing the tag. Fix: add it.
- `no comment` — public method/field/enum constant has no doc comment at all.
- `use of default constructor, which does not provide a comment` — public class with
  no explicit constructor (the synthetic default has no Javadoc). Fix: add an explicit
  no-arg constructor with a Javadoc comment.

Preferred doc-comment shapes for getters and small value types:

```java
/**
 * Brief one-line description of the value.
 *
 * @return the value
 */
public T getThing() { ... }
```

A bare `/** @return … */` triggers `no main description`; add a leading sentence.

If the local check passes (`BUILD SUCCESS`), the `mvn package` job in
`.github/workflows/publish.yml` will pass the `attach-javadocs` step.

## Architecture

### Two-Layer Design

**Java layer** (`src/main/java/net/ladenthin/llama/`):
- `LlamaModel` — Main API class (AutoCloseable). Wraps native context for inference, embeddings, re-ranking, and tokenization.
- `TextToSpeech` — Separate AutoCloseable native type for speech synthesis over the two-model OuteTTS (text-to-codes) + WavTokenizer (codes-to-speech vocoder) pipeline; `synthesize(text)` returns a 24 kHz mono 16-bit WAV byte stream. Native orchestration in `tts_engine.{h,cpp}`; the OuteTTS DSP / prompt / text helpers + default speaker are **derived at build time from upstream `tts.cpp`** (see "OuteTTS build-time extraction" below), not hand-copied; the in-memory WAV writer is `tts_wav.hpp`.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes that serialize to JSON (extend `JsonParameters`) for passing to native code.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.
- **`server` package — OpenAI-compatible HTTP endpoint (a single implementation).**
  - `server.OpenAiCompatServer` — built only on the JDK's `com.sun.net.httpserver` (no new dependency), embeddable and runnable via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …` (the fat-jar default `Main-Class` is now `NativeServer` — see "Two server modes"). Serves `POST /v1/chat/completions` (streaming via SSE + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST /infill`, `GET /v1/models` and `GET /health` (every route is also reachable without the `/v1` prefix), so editors that speak the OpenAI protocol (e.g. VS Code Copilot "Custom Endpoint", Cline, Roo Code, Continue) can drive a local model. Streaming chat uses the native OAI chunk path (`LlamaModel.streamChatCompletion` → `requestChatCompletionStream` / `receiveChatCompletionChunk` + the C++ `wrap_stream_chunk` helper), preserving `delta.tool_calls`; completions/embeddings/infill forward verbatim to the matching `LlamaModel.handle*`; rerank reshapes `handleRerank` into the OAI `results`/`data` shape. The chat mapper forwards `stream_options` and `response_format` and defaults `cache_prompt=true`; a CORS `Filter` answers `OPTIONS` preflights; `OpenAiSseFormatter.ensureUsageCachedTokens` guarantees `usage.prompt_tokens_details.cached_tokens` on the streamed usage chunk (Copilot crash fix, microsoft/vscode #273482). **Agentic tool-calling is the primary target**; a C++ guard (`test_server.cpp`) pins `tool_calls.function.arguments` as a JSON string (llama.cpp #20198).
  - **Alternative protocol surfaces** (pure translation over the OpenAI chat core — no second inference path; each reconstructs streamed tool calls via `ToolCallDeltaAccumulator`): **Ollama-native** (`GET /api/version`, `/api/tags`, `POST /api/show`, `/api/chat` with NDJSON streaming, `/api/generate` prompt-completion/FIM — `OllamaApiSupport`; `/api/show` advertises tools/insert/vision capabilities + context length for Copilot's Ollama provider), **Anthropic Messages** (`POST /v1/messages`, SSE event stream — `AnthropicApiSupport` + `AnthropicStreamTranslator`), and **OpenAI Responses** (`POST /v1/responses`, SSE event stream — `ResponsesApiSupport` + `ResponsesStreamTranslator`). The llama.cpp-native `GET /props` (context length + `modalities`) is served via `OpenAiSseFormatter.propsJson` for autocomplete clients that size their context from it.
  - Supporting classes: `OpenAiServerConfig` (builder; optional bearer auth; binds `127.0.0.1`; `corsAllowOrigin`; `supportsVision`), `OpenAiServerCli` (testable CLI arg parser → `ModelParameters` + `OpenAiServerConfig`; flags incl. `--mmproj`/`--embedding`/`--reranking`), `OpenAiRequestMapper` (OAI chat request → `InferenceParameters`), `OpenAiSseFormatter` (SSE/models/error JSON + usage normalization), `OaiRerankSupport` (pure rerank request/response shaping), and the model-free test seam `OpenAiBackend`/`ChunkSink` + `LlamaModelBackend`. The streaming envelope is parsed by `json.ChatStreamChunkParser`.
  - The `server` package is a dedicated top layer in the ArchUnit `layeredArchitecture` rule (the only layer allowed to access the root `Api`); `noInternalJdkImports` carries an explicit exception for the supported `com.sun.net.httpserver` (the exported `jdk.httpserver` module, which `module-info.java` `requires`). See README "OpenAI-compatible HTTP server".

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp. ~1,650 lines; 33 native methods (29 `LlamaModel` + 3 `TextToSpeech` + 1 `LlamaQuantizer`).
- `utils.hpp` — Helper utilities (format helpers, argv stripping, token-piece serialisation).
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- Uses `nlohmann/json` for JSON deserialization of parameters.
- The upstream server library (`server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-schema.cpp`, `server-models.cpp`, and — since b9829 — `server-stream.cpp`) is compiled directly into `jllama` via CMake — there is no hand-ported `server.hpp` fork. **`server-stream.cpp` is mandatory, not optional:** it defines the resumable-streaming SSE replay buffer (`g_stream_sessions`, `stream_session_attach_pipe`, `stream_aware_should_stop`, `stream_conv_id_from_headers`, the `stream_pipe_*` types) that `server-context.cpp` / `server-http.cpp` / `server-models.cpp` now `#include "server-stream.h"` and call, so omitting it fails the link with undefined references. It is platform-neutral (threads + std mutex/condvar, no `subprocess.h`/`posix_spawn_*`), so it builds on Android too and sits outside the `server-models.cpp` Android guard. `jllama` wires its own JNI routes and never calls `g_stream_sessions.start_gc()` (only the excluded standalone `server.cpp` `main()` does), so its GC thread stays dormant. **Phase 2:** the upstream HTTP transport (`tools/server/server-http.cpp`) and its `cpp-httplib` backend (`vendor/cpp-httplib/httplib.cpp`) are now compiled into `jllama` too, so the OpenAI-compatible server can be driven natively from JNI *inside* `libjllama` — no separate `llama-server` executable (a JNI shared library loads anywhere a JVM runs, which a standalone binary does not). `server-http.cpp` does `#include "ui.h"` (the WebUI asset table that `tools/ui`/`llama-ui` normally generates); since the Svelte WebUI is not shipped, `src/main/cpp/webui_stub/ui.h` supplies the upstream **empty-asset** interface and leaves `LLAMA_UI_HAS_ASSETS` undefined (all static-asset-serving blocks compile out). `<cpp-httplib/httplib.h>` already resolves via `llama-common`'s `vendor/` include dir (same nlohmann/json 3.12.0 as the FetchContent copy). No SSL: `CPPHTTPLIB_OPENSSL_SUPPORT` is left undefined (plain-HTTP; bind localhost / front with a TLS proxy). **`server.cpp` is now compiled in too** (on non-Android — it and `server-tools.cpp` pull in `subprocess.h`/`posix_spawn_*`, so they share `server-models.cpp`'s Android guard): b9870 exposes its entry as `int llama_server(int, char**)` (no `main` in the file), and `patches/0006` makes it embeddable (no process signal handlers, forwarded-argv parse, out-of-band shutdown). The `NativeServer` JNI bridge (`src/main/cpp/native_server.cpp`) calls `llama_server` on a worker thread, so the **full** upstream server — WebUI and all — runs inside `libjllama`. See "Two server modes" below.

### Two server modes (`OpenAiCompatServer` vs `NativeServer`)

The library exposes **two** ways to serve a model over HTTP, on two different transports. The fat jar's `Main-Class` is `server.ServerLauncher`, a tiny dispatcher: it runs `OpenAiCompatServer` when `--jllama-openai-compat` is present (that marker is stripped, the rest forwarded) and the default `NativeServer` otherwise. Both mains are also runnable directly by class name via `java -cp`. The two modes:

1. **`server.OpenAiCompatServer` (Java transport).** OpenAI/Ollama/Anthropic-compatible JSON API on the JDK's `com.sun.net.httpserver`, driving the compiled server *core* over JNI. Embeddable, no extra dependency, and it can share/reuse a `LlamaModel`. It serves **no** static assets — its `/` route is a 404, so **no WebUI**. It has its own `main` (run via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …`); its CLI (`OpenAiServerCli`) maps a curated flag subset (`-m/-c/-b/-ub/-ngl/-t/-tb/-ctk/-ctv/--jinja/--chat-template-kwargs/--host/--port/--parallel/--mmproj/--api-key/--embedding/--reranking`).
2. **`server.NativeServer` (native transport) — the default fat-jar server (when `--jllama-openai-compat` is absent).** Runs the **full upstream `llama_server`** (via `patches/0006` + `native_server.cpp`) inside `libjllama`, forwarding the raw llama-server argv verbatim — so **every** llama-server flag works and the **embedded WebUI is served** (when the assets are compiled in; CI's released jars have them, local `cmake` builds use the empty-asset stub). With the classic constructor it is an **independent lifecycle** (loads its own model from the argv, like `llama-server.exe`; owns the process's llama backend + stderr logging while running); the **attach constructor** (`NativeServer(LlamaModel, String...)`, via `patches/0007`'s `llama_server_attach`) instead serves an **already-loaded `LlamaModel`** — one copy of the weights, the model's worker keeps driving inference, the HTTP routes post to its queue; caller closes the server before the model. **Router mode** (start without a model argument: `--models-dir`, `GET/POST /models`, per-request model selection) works in-JVM after `NativeServer.setWorkerCommand(...)` redirects the worker spawn to a fresh JVM (`patches/0008` — upstream re-execs its own binary, which in a JVM is `java`); the typed `server.RouterClient` (+ `value.RouterModel`, `json.RouterModelsResponseParser`) wraps the model-management endpoints (list/load/unload/await-loaded with fail-fast on failed workers) so callers don't hand-roll HTTP+JSON. Either way it is **single-instance per process** (upstream keeps shutdown state in file-scope globals) and **not available on Android** (the `subprocess.h` guard). `libjllama` loading anywhere a JVM runs is what makes this "no separate `llama-server.exe`" possible.

### Native Helper Architecture

The project C++ helpers follow a strict semantic split:

**`json_helpers.hpp`** — Pure data transforms.
- Input: `nlohmann::json`, `server_task_result_ptr`, plain C++ types.
- Output: `json`, `std::vector`, `std::optional`, plain C++ types.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama state (`llama_context*`, `llama_vocab*`, `server_context*` never appear).
- Functions are named without `_impl` suffix — they are the canonical implementation.
- Testable with JSON literals and fake result objects; no JVM and no loaded model required.
- Upstream server headers must be included by the translation unit first (they define `server_task_result_ptr`, `json`, etc.).

Functions: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`,
`parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`,
`parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk`.

**`log_helpers.hpp`** — Pure log-formatting transforms.
- Input: `ggml_log_level`, message text (`const char*`), an explicit `std::time_t` timestamp.
- Output: `const char*` level label / `std::string` JSON.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama/server state — depends only on the `ggml_log_level` enum (from `ggml.h`) and
  nlohmann/json; no upstream server headers required (more standalone than `json_helpers.hpp`).
- Functions are `[[nodiscard]] inline`, named without an `_impl` suffix — the canonical implementation.
- Testable with literal levels/strings and a fixed timestamp; no JVM and no loaded model required.

Functions: `log_level_name`, `format_log_as_json`.

**`jni_helpers.hpp`** — JNI bridge helpers, split into two layers:

*Layer A* (no server headers required): handle management.
- `jllama_context` struct — owns `server_context` (value member, pimpl inside), background
  worker thread, cached `vocab`, saved `params`, and a `readers` map for streaming tasks.
- `get_jllama_context_impl` — reads Java `ctx` handle, returns the `jllama_context*` wrapper.
  Does NOT throw on zero handle (valid no-op for destructor-style calls).
- `require_json_field_impl` — throws `"<field> is required"` if key is absent.
- `jint_array_to_tokens_impl` — reads a Java `int[]` into `std::vector<int32_t>`.

*Layer B* (requires upstream server headers in the TU before `jni_helpers.hpp`): orchestration.
Includes `json_helpers.hpp` so all bridge helpers can call transforms directly.
- `utf8_to_jstring_impl` — builds a `java.lang.String` from raw standard-UTF-8 bytes via the cached
  `String(byte[], "UTF-8")` constructor. **Payload text must never go through `NewStringUTF`**: JNI
  specifies *Modified* UTF-8 input there, so standard UTF-8 containing supplementary-plane
  characters (every 4-byte emoji) is spec-invalid — Android CheckJNI aborts on it. The mirror of
  `parse_jstring`'s `String.getBytes("UTF-8")` input path.
- `json_to_jstring_impl` — serialises any `json` value to a JNI string via upstream
  `safe_json_to_str` (dump with `error_handler_t::replace`, so content ending in an incomplete
  UTF-8 sequence yields U+FFFD instead of throwing `json::type_error 316`) + `utf8_to_jstring_impl`.
- `results_to_jstring_impl` — delegates to `results_to_json` then `json_to_jstring_impl`.
- `vec_to_jarray_impl<JArray,JElem,CppElem>` — generic C++ vector → JNI primitive array.
- `embedding_to_jfloat_array_impl` — converts `std::vector<float>` to `jfloatArray`.
- `tokens_to_jint_array_impl` — converts `std::vector<int32_t>` to `jintArray`.

Functions with `_impl` suffix are called directly from `jllama.cpp`.

**Include order rule:**
```
// In jllama.cpp and any TU that uses Layer B helpers:
#include "server-context.h"   // upstream server headers must come first
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "jni_helpers.hpp"    // includes json_helpers.hpp internally
```

**Adding a new pure transform** (e.g. a new JSON field parser):
- Add it to `json_helpers.hpp`. No JNI, no llama types.
- Add tests to `src/test/cpp/test_json_helpers.cpp`.

**Adding a new JNI bridge helper:**
- Add it to `jni_helpers.hpp` in the appropriate layer.
- If it needs upstream server types, put it in Layer B (after the `json_helpers.hpp` include).
- Add tests to `src/test/cpp/test_jni_helpers.cpp`.

### Parameter Flow
Java parameters are serialized to JSON strings and passed to native code, which deserializes them using nlohmann/json. This avoids complex JNI field mapping for the many llama.cpp parameters.

### Native Library Resolution
`LlamaLoader` tries in order:
1. System property `net.ladenthin.llama.lib.path`
2. `java.library.path`
3. Extracts from JAR resources at `net/ladenthin/llama/{os}/{arch}/`

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for **Android** targets (and the
x86_64 manylinux jobs). **Linux `aarch64` is no longer cross-compiled** — it builds natively on a
GitHub `ubuntu-24.04-arm` runner (see "Linux aarch64: native ARM build" below). The
`.github/dockcross/dockcross-linux-arm64-lts` wrapper is now unused by CI (left in place; harmless).

### Linux aarch64: native ARM build

The `crosscompile-linux-aarch64` job (id kept for its downstream `needs:` reference; display name is
now **"Build and Test Linux aarch64"**) builds **natively on `ubuntu-24.04-arm`**, mirroring upstream
llama.cpp's own `ubuntu-cpu` aarch64 release job (`ubuntu-24.04-arm` + **GCC 14**).

**Why it moved off dockcross.** The old `dockcross/linux-arm64-lts` image ships **GCC 8.5 / glibc
2.17**; llama.cpp **b9789** uses C++17 CTAD-in-`new`, which needs **GCC ≥ 12**, so the cross build
stopped compiling. Upstream solved the same problem by building natively on `ubuntu-24.04-arm` with
GCC 14 and ships a **glibc ≈ 2.39** ARM binary with no old-glibc compatibility layer. This repo now
does the same: the aarch64 artifact's **glibc floor rises 2.17 → ~2.39** — the same envelope
upstream's own ARM binaries require (the x86_64 artifact stays at manylinux2014 / glibc 2.17).

Wiring (mirrors the macOS native jobs, not the dockcross jobs):
- `runs-on: ubuntu-24.04-arm`; `setup-java` → `mvn compile` (generates the JNI header) → `build.sh`.
- Installs `gcc-14`/`g++-14` and exports `CC`/`CXX` (upstream parity).
- `build.sh` flags: `-DGGML_NATIVE=OFF` (portable across ARMv8 CPU generations — no build-host
  `-march` baked in) `-DBUILD_TESTING=ON`, then **`ctest` runs the C++ unit suite on real ARM
  hardware** (the cross build ran no tests at all).
- sccache: `build.sh`'s Linux auto-fetch now covers `aarch64` as well as `x86_64` (it maps
  `uname -m` to the matching static-musl release); the probe still gates it, so a miss just builds
  uncached.
- Branch protection: if a required check pinned the old name "Cross-Compile Linux aarch64 (LTS)",
  repoint it to "Build and Test Linux aarch64".

### Linux s390x: big-endian cross-build + qemu test gate

`build-linux-s390x` extends the default JAR to **IBM Z (s390x, big-endian)** — the one target whose
byte order differs from every other platform. It **cross-compiles** with the GCC s390x toolchain
(`g++-s390x-linux-gnu`, native x86 speed — no emulated build) and then runs the **full C++ unit suite
under `qemu-user`** (`CMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-s390x-static`, `QEMU_LD_PREFIX=/usr/s390x-linux-gnu`).
That `ctest` run is a **real big-endian correctness gate** for the byte-order-sensitive surface — the
little-endian WAV writer (`tts_wav.hpp`), the JSON/token/embedding transforms, and the JNI helpers —
which is where an endian bug in *our* code could hide. Model-backed **Java** tests are deliberately
**not** run under emulation (a JVM + GGUF inference under `qemu-user` is slow and flaky); the Java↔JNI
boundary uses host-native array copies (endian-transparent), so the C++ gate covers the actual risk.
`-DGGML_OPENMP=OFF` sidesteps cross-libgomp issues (ggml uses its own `std::thread` pool). s390x is a
CPU platform like aarch64, so it ships in the **default** JAR (`Linux-s390x-libraries` merges via the
`*-libraries` glob; `OSInfo` maps `os.arch=s390x` → `Linux/s390x`) — no classifier, no pom profile.
**Fail-loud** and in `package.needs` like every other build. (Upstream llama.cpp already supports s390x
— it ships `ubuntu-s390x` with GGUF big-endian handling — so the native inference path is upstream's
concern; this job validates only *our* layer's endian-safety.)

## Testing

### Java tests
Require a model file. The CI downloads models from HuggingFace:
- **LlamaModel tests**: CodeLlama-7B-GGUF (`codellama-7b.Q2_K.gguf`)
- **RerankingModel tests**: Jina-Reranker model

**CI model policy (publish.yml): the full model set is downloaded and exercised on EVERY
Java test job** — Linux x86_64, all three macOS arm64 jobs (Metal / no-Metal / Metal-15), and
both Windows jobs (MSVC + Ninja). That includes the nomic embedding model, the SmolVLM vision
model + mmproj, and the OuteTTS + WavTokenizer TTS pair, with their `-Dnet.ladenthin.llama.*`
properties set, so `LlamaEmbeddingsTest`, `MultimodalIntegrationTest`, and `TtsIntegrationTest`
**run on every platform** rather than self-skipping. `validate-models.{sh,bat}` treats all of
these as **required** (a missing model hard-fails the job before tests run, so a download
regression can never silently downgrade to a skip). The only model still self-skipping is the
audio-input model (`AudioInputIntegrationTest`) — the prompt clip is committed
(`src/test/resources/audios/sample.wav`) but the audio model + mmproj have no CI download.
The model set has a **single source of truth: `.github/models.csv`** (one `filename,url` row per
model; `#` comments). Everything derives from it: the **`download-models`** job (ubuntu,
`needs: startgate`) is the only place models are fetched from HuggingFace (one manifest-driven
`curl` loop; files already restored from cache are skipped) and the **only writer** of the shared
GGUF cache (path `models/`, key **`gguf-models-<hash of models.csv>`** — so *editing the manifest
automatically creates a fresh complete cache entry*; no manual cache deletion on a model-set
change). The writer sets **`enableCrossOsArchive: true`**, making the one ubuntu-built entry the
same entry macOS and Windows restore. A **`verify-model-cache` matrix job** (ubuntu / macOS /
Windows, `needs: download-models`) then proves the entry is restorable
(`actions/cache/restore` + `fail-on-cache-miss: true`) **and complete**
(`validate-models.{sh,bat}`, which read their required list from the same manifest) on every OS
**before any model-consuming job starts** — all `test-java-*` jobs, the langchain4j integration
job, the Android emulator job and the fat-jar smoke jobs `need: verify-model-cache` and use the
**restore-only** action themselves (no per-job download, no save — a consumer can never re-save
an empty/partial entry), keeping validate as a per-job integrity guard. (This design hardened
after run 28805360584: without the cross-OS flag, cache entries are versioned per-OS and the
unreachable Windows-side entry had been re-saved **empty** (343 B) after an eviction; and
`validate-models.bat`'s quoted `MODELS` list broke cmd's for-tokenization so its `exit /b 1`
never fired — the empty cache sailed through the "gate" and the Windows jobs silently
self-skipped every model-backed test until the fat-jar smoke's hard check caught it.) The
`*_MODEL_NAME` workflow env vars remain consumer-side wiring for the `-Dnet.ladenthin.llama.*`
test properties and must match the manifest's filename column — locally the model tests still
self-skip when a GGUF is absent (`Assume.assumeTrue`), so a partial local checkout is fine.

Set the model path via system property or environment variable (see test files for exact property names).

Test files are in `src/test/java/net/ladenthin/llama/` and `src/test/java/examples/`.

### C++ unit tests

**No JVM and no model file required.** All tests run on pure data structures using mock
objects. The binary is named `jllama_test` and is built by CMake when `BUILD_TESTING=ON`.

#### Commands

```bash
# 1. Configure (once per fresh clone or after CMakeLists.txt changes)
cmake -B build -DBUILD_TESTING=ON

# 2. Build (incremental; -j$(nproc) uses all CPU cores)
cmake --build build --config Release -j$(nproc)

# 3. Run all tests
ctest --test-dir build --output-on-failure

# Count tests across all files
grep -rn "^TEST\b\|^TEST_F\b\|^TEST_P\b" src/test/cpp/ | wc -l

# Run a single named test (GoogleTest filter syntax)
ctest --test-dir build --output-on-failure -R "ResultsToJson"
```

#### Test files

| File | Tests | Scope |
|------|-------|-------|
| `src/test/cpp/test_utils.cpp` | 162 | Upstream helpers: `server_tokens`, `server_grammar_trigger`, `gen_tool_call_id`, `json_value`, `json_get_nested_values`, UTF-8 helpers, `format_response_rerank`, `format_embeddings_response_oaicompat`, `oaicompat_completion_params_parse`, `oaicompat_chat_params_parse`, `are_lora_equal`, `strip_flag_from_argv`, `token_piece_value`, `json_is_array_and_contains_numbers`, `format_oai_sse`, `format_oai_resp_sse`, `format_anthropic_sse`, `parse_lora_request` |
| `src/test/cpp/test_server.cpp` | 201 | Upstream result types: `result_timings`, `task_params::to_json()` (incl. `dry_sequence_breakers`, `preserved_tokens`, `timings_per_token`), `completion_token_output`, `server_task_result_cmpl_partial` (non-oaicompat + `to_json_oaicompat` + logprobs + `to_json_oaicompat_chat` + `to_json_anthropic` + dispatcher), `server_task_result_cmpl_final` (non-oaicompat + `to_json_oaicompat` + `to_json_oaicompat_chat` + `to_json_oaicompat_chat_stream` + `to_json_anthropic` + `to_json_anthropic_stream` + tool_calls + dispatcher), `server_task_result_embd`, `server_task_result_rerank`, `server_task_result_metrics`, `server_task_result_slot_save_load`, `server_task_result_slot_erase`, `server_task_result_apply_lora`, `server_task_result_get_lora`, `server_task_result_error`, `format_error_response`, `server_task::need_sampling()`, `server_task::n_tokens()`, `server_schema::eval_llama_cmpl_schema()` (parsing pipeline + grammar routing + error paths + per-request `dry_*` and `sse_ping_interval` field round-trips incl. hard-limit + server-default inheritance), `response_fields` projection |
| `src/test/cpp/test_json_helpers.cpp` | 50 | All functions in `json_helpers.hpp`: `get_result_error_message`, `results_to_json`, `rerank_results_to_json` (incl. missing/out-of-range `index` rejection), `parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk` |
| `src/test/cpp/test_log_helpers.cpp` | 13 | All functions in `log_helpers.hpp`: `log_level_name`, `format_log_as_json` |
| `src/test/cpp/test_jni_helpers.cpp` | 54 | All functions in `jni_helpers.hpp` using a zero-filled `JNINativeInterface_` mock (incl. the `utf8_to_jstring_impl` byte-array string path: emoji byte-preservation, truncated-UTF-8 replace-not-throw) |
| `src/test/cpp/test_tts_wav.cpp` | 5 | The in-memory WAV writer `pcm_to_wav16_bytes` in `tts_wav.hpp` (WAV header/payload + little-endian clamping) plus the pure OuteTTS codec-window filter `filter_outetts_codec_tokens` (`tts_engine.h`: boundary keep/drop + rebase). The OuteTTS DSP it pairs with is derived from upstream `tts.cpp` and covered end-to-end by the Java `TtsIntegrationTest`, not unit-tested here. |

**Current total: 485 tests (all passing).**

#### Upstream source location (in CMake build tree)

llama.cpp is fetched via CMake FetchContent, pinned to `GIT_TAG b10092`.

**GoogleTest** is a separate `BUILD_TESTING`-only FetchContent (`GIT_TAG v1.17.0`), used solely
by the `jllama_test` C++ unit-test binary — not by the shipped library, and not coupled to the
llama.cpp pin or the bundled nlohmann/json. There is **no constraint behind the exact tag**; it
is just the latest stable at the time it was last touched. Bump it from time to time (nothing
auto-tracks it), pairing the bump with a green `C++ Tests` CI run.

```
build/_deps/llama.cpp-src/tools/server/   ← server-task.h, server-common.h, etc.
build/_deps/llama.cpp-src/include/        ← llama.h, llama-cpp.h
build/_deps/llama.cpp-src/common/         ← common.h, chat.h, arg.h, etc.
```

When reading a `to_json()` implementation to write tests against it, read from:
`build/_deps/llama.cpp-src/tools/server/server-task.cpp`

#### Mock JNI pattern used in test_jni_helpers.cpp

```cpp
// Zero-fill the interface so all unpatched fn pointers are nullptr
JNINativeInterface_ iface = {};
// Patch only the stubs this test needs, e.g.:
iface.GetLongField  = [](JNIEnv*, jobject, jfieldID) -> jlong { return some_handle; };
iface.ThrowNew      = [](JNIEnv*, jclass, const char*) -> jint { return 0; };
// Wire up the env
JNIEnv_ fake_env = {};
fake_env.functions = &iface;
JNIEnv *env = &fake_env;
```

Any stub that is called but not patched will crash (null function pointer) — deliberately,
so missing stubs are caught immediately rather than silently.

#### How to add a new C++ test

1. Open the appropriate `src/test/cpp/test_*.cpp`:
   - Pure JSON transform → `test_json_helpers.cpp`
   - JNI helper → `test_jni_helpers.cpp`
   - Upstream result type `to_json()` → `test_server.cpp`
   - `utils.hpp` function or upstream utility → `test_utils.cpp`
2. Add a `TEST(SuiteName, TestName) { ... }` block using GoogleTest macros.
3. Rebuild: `cmake --build build --config Release -j$(nproc)`
4. Run: `ctest --test-dir build --output-on-failure`
5. Commit with message summarising coverage added and new test total.

#### Finding untested code paths

```bash
# List all functions defined in a header
grep -n "^inline\|^static\|^\[\[nodiscard\]\]" src/main/cpp/utils.hpp

# Check which functions already have tests
grep -n "function_name" src/test/cpp/*.cpp

# Find all fields in an upstream to_json() method
grep -n "\"field_name\"" build/_deps/llama.cpp-src/tools/server/server-task.cpp

# Check which JSON fields Java actually reads (important: must test these)
grep -rn "field_name" src/main/java/net/ladenthin/llama/
```

#### Testing complex scenarios — methodology

Simple tests verify individual field values on a default-constructed struct.
Complex tests verify **control flow**: switch dispatchers, cross-cutting flags, and
multi-step parameter pipelines.  The same build/run/commit loop applies.

**1. Dispatcher (switch) coverage**

Every `to_json()` that is a switch on `res_type` has one test per arm:

```cpp
// Pattern: set is_updated=true, set res_type, call to_json(), check the
// distinguishing field that differs between arms.
server_task_result_cmpl_final f;
f.is_updated = true;
f.stream     = false;
f.res_type   = TASK_RESPONSE_TYPE_OAI_CMPL;
// ... set required fields ...
const json j = f.to_json();
EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
```

The same pattern handles the `stream` flag fork inside `OAI_CHAT`:
`stream=false` → single object with `"object":"chat.completion"`;
`stream=true`  → JSON array of chunks with `"object":"chat.completion.chunk"`.

**2. Cross-cutting flag interaction**

Some flags (verbose, include_usage, timings.prompt_n) cut across multiple formatters.
Test each flag in one formatter only — they share the same code path:

```cpp
// verbose=true must add __verbose to the first chunk/top-level object
f.verbose = true;
EXPECT_TRUE(j.contains("__verbose"));

// timings absent when prompt_n < 0 (default), present when >= 0
f.timings.prompt_n = 5;
EXPECT_TRUE(j.contains("timings"));
```

**3. Parameter parsing (`eval_llama_cmpl_schema`) without a model**

`server_schema::eval_llama_cmpl_schema(vocab, params_base, n_ctx_slot, logit_bias_eog, data)`
can be called with `nullptr` vocab **if the JSON does not trigger grammar/preserved_tokens
tokenisation** (those are the only vocab-dependent paths).  This lets us test the full
parsing pipeline including error throws:

```cpp
common_params          params_base;
std::vector<llama_logit_bias> no_bias;
const int n_ctx = 512;

// test: repeat_last_n=-1 is expanded to n_ctx_slot
json data = {{"repeat_last_n", -1}};
auto p = server_schema::eval_llama_cmpl_schema(nullptr, params_base, n_ctx, no_bias, data);
EXPECT_EQ(p.sampling.penalty_last_n, n_ctx);

// test: invalid value throws std::runtime_error
json bad = {{"dry_sequence_breakers", json::array()}};  // empty → error
EXPECT_THROW(server_schema::eval_llama_cmpl_schema(nullptr, params_base, n_ctx, no_bias, bad),
             std::runtime_error);
```

**4. Array-returning formatters**

Some methods (e.g. `to_json_oaicompat_chat_stream()`) return a JSON array of event objects,
not a single object.  Check with `is_array()` first, then iterate or index:

```cpp
const json j = f.to_json_oaicompat_chat_stream();
ASSERT_TRUE(j.is_array());
ASSERT_GE(j.size(), 1u);
// Last chunk always has a non-null finish_reason
EXPECT_FALSE(j.back().at("choices")[0].at("finish_reason").is_null());
```

**5. `response_fields` projection**

`to_json_non_oaicompat()` supports a projection list via `response_fields`.
When non-empty, only those dot-separated paths survive:

```cpp
f.response_fields = {"content", "tokens_predicted"};
const json j = f.to_json_non_oaicompat();
EXPECT_TRUE(j.contains("content"));
EXPECT_FALSE(j.contains("stop_type"));  // filtered out
```

## Key Constraints

- **Java 8+** runtime required. Built with JDK 21 targeting bytecode 1.8 for broad compatibility.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.

## Javadoc Conventions

See [`../workspace/policies/javadoc-conventions.md`](../workspace/policies/javadoc-conventions.md).

## SpotBugs Suppressions

See [`../workspace/policies/spotbugs-suppressions.md`](../workspace/policies/spotbugs-suppressions.md).

## Spotless Formatting

See [`../workspace/policies/spotless-formatting.md`](../workspace/policies/spotless-formatting.md).
Run `mvn spotless:apply` before every commit that touches `.java` files.

## jqwik Policy

See [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md).

## Lombok Config

See [`../workspace/policies/lombok-config.md`](../workspace/policies/lombok-config.md).

## CI Test Diagnostics

See [`../workspace/policies/ci-test-diagnostics.md`](../workspace/policies/ci-test-diagnostics.md).

## PIT Mutation Testing

See [`../workspace/policies/pit-mutation-testing.md`](../workspace/policies/pit-mutation-testing.md).
Run PIT with the lifecycle prefix — `mvn test-compile org.pitest:pitest-maven:mutationCoverage`
(from the repo root add `-f llama/pom.xml`). The gate is **hermetic** — no model or audio fixture
needed: `ContentPartTest`'s `@TempDir` tests cover `value.ContentPart.audioFile(Path)` (verified
295/295, 0 NO_COVERAGE in a fixture-less sandbox; the former audio-fixture gotcha is resolved,
see `TODO.md`).

## JPMS Module Descriptor

This repo ships a `module-info.java` compiled in a separate `release 9` execution. Javadoc
currently runs in **classpath mode** (javadoc `<source>` is `1.8`), which is the *only* thing
keeping it clear of the JPMS module-mode javadoc trap that bit BAF. **Before raising the Java /
javadoc source level to ≥ 9, read**
[`../workspace/policies/jpms-module-descriptor.md`](../workspace/policies/jpms-module-descriptor.md).

## Repository layout — Maven reactor (`llama/` + `llama-langchain4j/` + `llama-kotlin/`) + the `llama-android/` Gradle build

The repo root is a thin **aggregator/parent POM** (`net.ladenthin:llama-parent`,
`packaging=pom`) with three modules:

- **`llama/`** — the native JNI core (`net.ladenthin:llama`). *All the core sources and build
  files live here now:* `llama/src/`, `llama/CMakeLists.txt`, `llama/cmake/`, `llama/patches/`,
  `llama/pom.xml`, `llama/spotbugs-exclude.xml`, `llama/lombok.config`, `llama/.clang-format`.
  Its published coordinates are unchanged (`net.ladenthin:llama`), so consumers are unaffected.
- **`llama-langchain4j/`** — the LangChain4j adapters (see below).
- **`llama-kotlin/`** — the Kotlin coroutines façade (see "Android AAR + Kotlin façade" below).

All modules inherit the single `<version>` from the parent, so they **ship in lockstep by
construction** (no CI guard needed). The parent also holds the shared `release` profile (GPG +
Central Publishing), so one reactor `mvn -P release deploy` signs and publishes all four
Maven artifacts (`llama-parent` pom, `llama`, `llama-langchain4j`, `llama-kotlin`) at the same
version.

**`llama-android/` is deliberately NOT a reactor module** but a standalone plain-Gradle build
(no AGP, no Android SDK needed to build): Maven cannot produce or deploy an artifact with
`<packaging>aar</packaging>` (the only android-maven-plugin is dead), while Gradle's built-in
`maven-publish` can. It stays version-locked anyway — `llama-android/build.gradle.kts` parses
the version and the mirrored dependency versions out of the Maven poms at configure time, so
`mvn versions:set` remains the single bump point (no Gradle-side edit on a bump). See
"Android AAR + Kotlin façade" below.

**Consequences for build commands:** the core's cmake/native build runs *in `llama/`*.
`.github/build.sh` / `build.bat` `cd` into `llama/` themselves (relative to the script), so CI
and the dockcross containers (whose workdir stays the repo root) are unaffected. Locally, run
core cmake builds from `llama/` (e.g. `cd llama && cmake -B build && cmake --build build`), and
target the core with Maven via `-f llama/pom.xml` (or `-pl llama -am` from the root). A plain
`mvn` at the root builds the whole reactor. **When a build-command example elsewhere in this
file shows `cmake -B build` / `src/main/...` / `mvn compile` at the root, read it as running in
`llama/`** (the paths moved; the recipes are otherwise unchanged).

**Version bump:** the child modules declare **no `<version>` of their own** — their *project*
version is inherited from the parent. But each child still hardcodes the parent version inside its
`<parent><version>` pointer (Maven requires a literal there — there is **no `${revision}`/CI-friendly
versioning** here), so a version change must be applied to **all four poms in lockstep**:

- `pom.xml` (root) — `<version>`
- `llama/pom.xml` — `<parent><version>`
- `llama-langchain4j/pom.xml` — `<parent><version>`
- `llama-kotlin/pom.xml` — `<parent><version>`

(`llama-android/` needs **no** edit — its Gradle build reads the root pom's version at
configure time.)

The safe way is `mvn -q versions:set -DnewVersion=X.Y.Z -DgenerateBackupPoms=false` from the repo
root (it updates the parent and every child `<parent>` reference at once). Changing only the root
`<version>` leaves the children pointing at a non-existent parent and **fails the reactor build**
(`Could not find artifact net.ladenthin:llama-parent:pom:X.Y.Z`).

`versions:set` only rewrites the **poms**. The **two README files** that carry hardcoded
release-version dependency snippets must be bumped **manually and in the same commit** — miss either
and the published docs point consumers at the previous release. (The `llama-langchain4j/README.md`
snippet was exactly the one forgotten on the `5.0.4 → 5.0.5` bump; it is listed here so it is not
missed again.)

- **`README.md`** (root) — the install snippet, the two classifier-example snippets (default + the
  `<classifier>` template), and the `llama-langchain4j` snippet. The Maven Central **badge**
  auto-pulls the latest released version, so leave it. The **`-SNAPSHOT` line** in the "Snapshot
  builds" section documents the snapshot channel — set it to the *next* dev version, not the release.
  (The per-classifier snippets were **deduplicated** to a single canonical + template pair, so the
  release version now appears in only ~4 spots here, not ~20 — the runtime details live once in the
  classifier table.)
- **`llama-langchain4j/README.md`** — its own `<dependency>` snippet.
- **`llama-android/README.md`** and **`llama-kotlin/README.md`** — their Gradle dependency
  snippets, plus the `llama-android`/`llama-kotlin` snippets in the root README's
  "Importing in Android" section.

(If single-source ergonomics are wanted, the Maven
CI-friendly `${revision}` property + `flatten-maven-plugin` would let a bump touch only the root —
that plugin is not configured today, so do not rely on "root only".)

## LangChain4j integration (`llama-langchain4j` reactor module)

`llama-langchain4j/` adapts a `LlamaModel` to LangChain4j's `ChatModel`,
`StreamingChatModel`, `EmbeddingModel` and `ScoringModel` interfaces **in-process over
JNI** (no HTTP hop). It is a **reactor module** alongside the core `llama` module (see
"Repository layout" above), so it is built, versioned and released together with the core.

Why it is a **separate artifact** and not a classifier of the core: langchain4j 1.x
requires **Java 17** (the core stays Java 8), and classifiers share the core's single POM —
adding `langchain4j-core` there would force it (and the Java 17 floor) on every plain
`net.ladenthin:llama` consumer. A separate `artifactId` (its own module POM) is the only way to
keep that dependency (and Java floor) off the core. It is pure Java with **no per-classifier
matrix**: it compiles against the core's Java API, which is identical across every native
classifier; the backend (CPU/CUDA/OpenCL/Vulkan) is a runtime classpath choice for the
consumer.

Wiring:

1. **`llama-langchain4j/pom.xml`** — `net.ladenthin:llama-langchain4j`, `release 17`, a child of
   `net.ladenthin:llama-parent` (so it **inherits `${project.version}`** — no hardcoded *dependency*
   version, no lockstep guard; the `<parent><version>` literal itself is still bumped in lockstep,
   see "Version bump" above). Depends on `net.ladenthin:llama:${project.version}` and
   `dev.langchain4j:langchain4j-core`. Builds its own sources/javadoc jars; the `release`
   profile (GPG + Central Publishing) is **inherited from the parent**, not duplicated here.
   Java package stays `net.ladenthin.llama.langchain4j` (package name need not track the artifactId).
2. **`.github/workflows/publish.yml`** — the `test-java-llama-langchain4j` job installs
   parent + core into the local repo (`mvn -pl llama -am -DskipTests install`), then
   `mvn -f llama-langchain4j/pom.xml verify` (7 model-free mapping unit tests run; the 4
   model-backed integration tests self-skip without a GGUF; `verify` also builds the javadoc
   jar so a release-time javadoc break is caught in PR CI). The `publish-snapshot`/
   `publish-release` jobs `needs:` this job; deployment is a **single reactor**
   `mvn -P release deploy` (no separate module deploy step — the parent's inherited `release`
   profile signs and publishes parent + llama + llama-langchain4j together at the same version).
   A separate **`test-java-llama-langchain4j-integration`** job runs the model-backed tests
   (chat/streaming/embedding/scoring adapters) by **reusing** the shared GGUF cache
   (`gguf-models-v1`, restore-only — no extra download) and the `Linux-x86_64-libraries` native
   artifact: it `needs: [crosscompile-linux-x86_64, download-models]` (so the cache is already
   populated and it runs in parallel), installs parent+core with the downloaded native lib
   bundled, and passes the already-cached chat (`REASONING_MODEL_NAME`), nomic-embedding and
   jina-reranker model paths via the module's
   `-Dnet.ladenthin.llama.langchain4j.{embedding,rerank}.model` / `net.ladenthin.llama.model.path`
   properties. It is validation-only (not a release gate); a cold cache degrades to a self-skip.

**Mapped** (since 5.0.6): blocking tool calling (`ToolSpecification` ↔ jllama `ToolDefinition`
via the module's own `JsonSchemaElementSerializer` — langchain4j's serializer lives in its
`internal` package, so the module carries a public-API-only recursive walk emitting the same
`$defs`/`#/$defs/…` conventions; tool-call turns round-trip in both directions),
`response_format`/JSON mode (`json_object` + `json_schema` structured output), and multimodal
user input (`ImageContent`/`AudioContent` → `ContentPart` array-form content; needs `--mmproj`).
Streaming (since 5.0.6, second pass): `JllamaStreamingChatModel` now streams over the native
OAI chunk path via the module's `StreamingChunkAssembler` — streamed tool calls
(`onPartialToolCall`/`onCompleteToolCall` + `toolExecutionRequests()` on the final response),
per-token thinking events (`onPartialThinking` + `AiMessage.thinking()`), real finish reason and
token usage. **Open follow-up** (documented in `llama-langchain4j/README.md`): `modelName()` is
ignored (one model per adapter).

## Android AAR + Kotlin façade (`llama-android/` + `llama-kotlin/`)

Two consumable Android-facing artifacts, replacing the submodule/NDK source-integration flow as
the recommended path (README "Importing in Android", Option 1):

- **`net.ladenthin:llama-android`** / **`llama-android-opencl`** — AARs (`<packaging>aar</packaging>`)
  carrying the core classes + the CI-built `libjllama.so` natives under `jni/` — the CPU AAR is
  **multi-ABI** (`arm64-v8a` devices + `x86_64` emulators/Chromebooks, built by the
  `crosscompile-android-x86_64` dockcross job whose artifact also merges into the default JAR's
  `Linux-Android/x86_64` tree via the `*-libraries` glob; the OpenCL flavor stays arm64-only —
  Adreno is Qualcomm ARM hardware), a
  `minSdkVersion 28` manifest (AGP enforces the floor on consumers), and consumer R8/ProGuard
  rules (`consumer-proguard.txt` → `proguard.txt` in the AAR; keeps `net.ladenthin.llama.**` for
  the JNI `FindClass`/Jackson reflection surface). The AAR's `classes.jar` is the
  **byte-identical Maven-built core jar** minus the desktop/Android native resource trees
  (~70 MB APK bloat otherwise) and `module-info.class` (D8 rejects it); on Android `LlamaLoader`
  resolves via `System.loadLibrary("jllama")`, which finds the AAR-installed `.so` — no loader
  change was needed. Built by the **standalone plain-Gradle build** in `llama-android/`
  (see "Repository layout" for why it is not a Maven module); the POM mirrors the core's
  compile-scope deps (jackson/slf4j-api/jspecify/checker-qual, versions parsed from
  `llama/pom.xml` — deliberately NOT logback, which is the JVM-only runtime binding).
- **`net.ladenthin:llama-kotlin`** — Maven reactor module; pure-Kotlin (2.4, jvmTarget 1.8)
  coroutines façade: `generateFlow`/`generateChatFlow` (cold `Flow`, source closed on
  completion/error/cancellation) and `completeSuspend`/`chatSuspend`/`chatCompleteTextSuspend`/
  `embedSuspend` (`completeSuspend` wires coroutine cancellation into the cooperative
  `CancellationToken`). The core dep is **provided-scope** so Android consumers pair it with the
  AAR instead of transitively pulling the fat desktop JAR. 6 model-free unit tests fake the
  `Iterable & AutoCloseable` seam (`closeableIterableFlow`/`withCancellationToken` internals).

**16 KB page-size invariant (Google Play, Android 15+ targets):** `llama/CMakeLists.txt` pins
`-Wl,-z,max-page-size=16384` in the Android guard block, and the `package-android-aar` CI job
asserts every LOAD segment of the shipped `.so` is 16384-aligned via `readelf` — a dockcross
toolchain bump cannot silently regress Play compatibility.

**dlopen-ability invariant (bionic-only DT_NEEDED):** the same Android guard block sets
`GGML_OPENMP OFF` (ggml uses its std::thread pool — Android ships no `libomp.so`; same trade
as the Windows-arm64 clang-cl job) and links `-static-libstdc++` (no `libc++_shared.so`
dependency — that runtime only exists when an app packages it itself). Without both, the
dockcross cross-clang emitted `DT_NEEDED` on `libomp.so` + `libc++_shared.so`, which made
`System.loadLibrary("jllama")` fail with `UnsatisfiedLinkError` on every device (caught by the
`test-android-emulator` job; the released 5.0.5 arm64 lib had the same latent defect). The
`package-android-aar` job enforces a per-`.so` `DT_NEEDED` whitelist (`libc.so libm.so libdl.so
liblog.so libandroid.so`, plus `libOpenCL.so` for the OpenCL flavor) via `readelf -dW`, and
`LlamaLoader` now includes the swallowed `System.loadLibrary` message in its
"Directly from .apk/lib (…)" tried-path entry so a future dlopen reason is never invisible.

**CI (`publish.yml`):** `test-java-llama-kotlin` (model-free unit tests);
`package-android-aar` (needs both Android native jobs) builds the core jar, stages the natives,
assembles both AARs, validates structure (entries, minSdk, classes.jar content, 16 KB alignment),
publishes to mavenLocal, and runs the **AGP consumer smoke test** — the minimal app fixture in
`.github/android-consumer-test/` resolves the AAR from mavenLocal and runs a full R8
`assembleRelease` on the runner's preinstalled Android SDK (this is what actually validates
AGP/Android Studio consumption). **On-device runtime IS now CI-covered** via
`test-android-emulator`: a KVM-accelerated x86_64 emulator (API 30) runs the fixture's
`connectedDebugAndroidTest` — `System.loadLibrary` from the AAR's `jni/x86_64`, on-device
`GgufInspector`, and real native inference against the adb-pushed cached draft model
(AMD-Llama-135m). The job is a **release gate** (in both
publish `needs:` graphs) since PR #298, after running flake-free through the PR's validation
cycle. arm64 kernels + the Adreno/OpenCL flavor remain out of emulator scope —
the planned example app covers those on hardware.
Both publish jobs `need` these jobs (fail-loud release gating) and publish the AARs via Gradle:
snapshots to the Central snapshots repo (`publishAllPublicationsToCentralSnapshotsRepository`),
releases as a signed Central Portal bundle upload (staging repo → zip → Publisher API).
`llama-kotlin` rides the normal reactor `mvn -P release deploy`.

## Android app "LLM Service" (`android-llmservice/`)

A shippable, **KISS fully-offline on-device chat app** consuming the `llama-android` AAR +
`llama-kotlin` façade. App label **"LLM Service"**, applicationId/namespace/package
**`net.ladenthin.android.llmservice`**. One Compose screen: pick a GGUF from the file system
(Storage Access Framework), stream a chat reply fully on-device, **tune sampling** via a Settings
sheet (⚙️ — temperature/Top-K/Top-P/Min-P/**repeat penalty**/repeat range/max tokens; the
repeat-penalty default is what stops the small-model repetition loop), **Stop/Copy/Regenerate/Clear**
the chat, watch an **in-app log** (bottom strip + full viewer with copy-all / save-as-`.txt` / clear),
switch UI language via a **flag picker (13 languages)**, and **Save/Load** the conversation to
**private local storage**. No `INTERNET`/storage permission — nothing leaves the device. Like `llama-android/` and
`.github/android-consumer-test/`, it is a **standalone plain-Gradle/AGP build, NOT a Maven
reactor module** (and NOT published to Maven Central — it is an app, not a library). It is the
"real app" counterpart to the headless `.github/android-consumer-test/` fixture (which only
compiles/loads the API); this one has a UI and is driven end-to-end. The name is intentionally
**generic/descriptive** (sidesteps trademark conflicts); the applicationId under the owner's
domain is the only string that must be globally unique.

Structure (mirrors the consumer-test's plumbing):
- **`settings.gradle.kts`** — `rootProject.name = "android-llmservice"`; pins AGP `9.2.1` + the
  Compose compiler plugin (`2.4.0`); `mavenLocal()` first so the freshly-built AAR + façade
  resolve there in CI (Maven Central for real users). `9.2.1` (not `9.2.0`) fixes a real R8
  regression (`ClassNotFoundException` on `com.android.tools.r8.RecordTag` after upgrading
  Gradle to 9.x with AGP 9.2.0) that hits this project directly since `buildTypes.release`
  sets `isMinifyEnabled = true`; the `.github/android-consumer-test` fixture is pinned the
  same way for the same reason. AGP 9.x requires Gradle >= 9.4.1 and
  JDK 17+; CI already runs JDK 21 everywhere (`env.JAVA_VERSION`), so only the `gradle-version`
  pin on the jobs that build this project (and the `.github/android-consumer-test` fixture)
  needed bumping. AGP 9.0+ has **built-in Kotlin support** (a runtime dependency on Kotlin
  Gradle plugin 2.2.10+), so the standalone `org.jetbrains.kotlin.android` plugin is no longer
  applied — applying it now fails the build with "no longer required for Kotlin support since
  AGP 9.0" (`app/build.gradle.kts` line 7). The Compose compiler plugin still applies
  separately and its 2.4.0 pin exceeds AGP's 2.2.10 floor, so no other version changed.
- **`app/build.gradle.kts`** — `namespace`/`applicationId` `net.ladenthin.android.llmservice`,
  `minSdk 28` (AAR floor), `compileSdk 37` (raised from 35 — Compose/lifecycle/activity AAR
  metadata now requires it), `targetSdk 35`, Jetpack Compose, `androidx.appcompat` (only for
  the per-app language API), ABIs `arm64-v8a` + `x86_64`. Depends on `net.ladenthin:llama-android` +
  `net.ladenthin:llama-kotlin` at `-PjllamaVersion=<reactor version>` (defaults to the last release
  when built by hand). The release `signingConfig` reads an **upload keystore** (env vars
  `JLLAMA_UPLOAD_STORE_FILE` / `_STORE_PASSWORD` / `_KEY_ALIAS` / `_KEY_PASSWORD`, or the matching `-P`
  props) and **falls back to debug signing** when none is set, so forks/PRs/local builds stay green.
- **i18n:** every UI string is a resource; translations ship for **13 languages**
  (`values-{de,es,fr,it,pt,ru,tr,ar,hi,zh-rCN,ja,ko}/strings.xml` + default English), listed in
  `res/xml/locales_config.xml` and `Languages.kt`. The flag dropdown switches language in-app via
  `AppCompatDelegate.setApplicationLocales` (AppCompat `autoStoreLocales` service persists it;
  `MainActivity` extends `AppCompatActivity`, theme `Theme.LlmService` = AppCompat DayNight). The
  ViewModel survives the locale-change recreation, so the chat isn't lost. `app_name` ("LLM Service")
  is the one string deliberately NOT translated.
- **`ChatViewModel.kt`** — the logic: loads a `LlamaModel` (SAF `content://` copied into `filesDir`
  because llama.cpp mmaps a real path), streams via `generateChatFlow` and forwards the `GenerationSettings`
  sampling knobs to `InferenceParameters`; supports **stop** (`generation?.cancel()` — cancellation keeps
  the partial reply, not an error) and **regenerate** (drop the trailing reply, re-run the last user turn,
  shared `startGeneration` path); keeps a capped rolling in-app **log**. **`SessionStore.kt`** — private
  local save/load: the conversation + model path as JSON in `filesDir/session.json` (`org.json`, no dep
  added), readable only by the app. **`LlmServiceApp.kt`** — `Application` that **wipes the transient
  working data** (the copied model `current-model.gguf` + cache) on every **cold start**
  (`onCreate`) — a privacy guarantee independent of the OS calling `onDestroy`; `MainActivity` also
  wipes best-effort on finish. Only the **opt-in saved session** persists. **`MainActivity.kt`** —
  Compose UI with a **two-row app bar** (row 1 = horizontally-scrollable model name + a ❌ unload
  button when a model is loaded, row 2 = all action icons) + SAF picker + flag menu + Save/Load + the ⚙️ Settings sheet
  (sampling knobs + a **Model** section: CPU threads / context length applied via **Reload model**) +
  the 🧾 log strip/viewer + Stop/Copy/Regenerate/Clear chat actions (+ **long-press any bubble to
  copy**) + **prompt shortcut chips** + a **device-readiness card** (free RAM / storage / battery via
  `DeviceInfo`) on the picker + the offline badge; reads optional `MODEL_PATH` / `CHAT_TEMPLATE`
  intent extras as a **test hook** (the shipping UI never sets them).
- **`app/src/androidTest/kotlin/.../ChatFlowInstrumentedTest.kt`** — the real end-to-end test:
  launches the activity with a preloaded model (bypassing the system SAF dialog, which only
  UiAutomator could drive), types a prompt, taps Send, asserts a non-empty streamed reply.
  Self-skips (JUnit `Assume`) when the adb-pushed model is absent.

**Requirements / roadmap docs (keep in sync with the code).** The app has **no unit tests** for its
logic (only the one on-device `ChatFlowInstrumentedTest` above), so
[`android-llmservice/requirements.md`](android-llmservice/requirements.md) is the **spec of record** —
it enumerates, with stable `Rn.m` IDs, every feature the app implements today. Any change to app
behavior must update `requirements.md` in the same commit (add/amend/retire a row); the roadmap of
not-yet-built features lives in [`android-llmservice/TODO.md`](android-llmservice/TODO.md), and
`README.md`'s feature claims must match `requirements.md`.

**Recommended real model: Gemma 3 4B Instruct** (a `*-it` GGUF; carries its own chat template, so
no override needed). Any instruct GGUF works. The CI on-device test uses the tiny cached draft model
with a forced `chatml` template just to prove tokens generate.

**Key signing fact:** the Maven Central **GPG key cannot sign an APK/AAB** (different
cryptosystem). Android needs a Java keystore (PKCS12/JKS + RSA) **upload key**; Play App Signing
manages the final app-signing key. CI wires it from the optional `ANDROID_UPLOAD_KEYSTORE_BASE64`
+ password/alias secrets.

**CI wiring** — **two split jobs** in `.github/workflows/publish.yml` (both `needs:` the two Android
native jobs; the test job additionally `needs: verify-model-cache`), so the installable artifacts are
**always** produced even when the on-device UI test is flaky/slow:
- **`build-android-llmservice`** — installs the core + `llama-kotlin` to mavenLocal (`llama-kotlin`'s
  core dep is provided-scope, so the desktop JAR is never pulled into the APK), stages the CI-built
  Android natives, publishes the CPU AAR to mavenLocal, builds the **release `.aab`** (signed with the
  upload key when the `ANDROID_UPLOAD_KEYSTORE_BASE64` secret is present, else debug-signed) + the
  installable **`.apk`**, and uploads them (`android-llmservice-aab` / `android-llmservice-apk`).
  **No emulator** — a flaky emulator can never block getting the APK.
- **`test-android-llmservice`** — a **separate, non-gating** check that repeats the setup, then boots
  the KVM x86_64 emulator and runs the Compose UI test via **`.github/run-android-llmservice-test.sh`**
  (adb-pushes the cached `AMD-Llama-135m` draft model, `connectedDebugAndroidTest`). It can go **red on
  its own** without stopping the build/artifacts; keep it **non-required** in branch protection to keep
  it optional (visible-but-non-blocking).

Both emulator jobs (`test-android-llmservice` + `test-android-emulator`) prepend a **free-disk step**
before the emulator (delete every restored model except `DRAFT_MODEL_NAME` + large unused preinstalled
toolchains) because the AVD userdata partition needs ~7.4 GB and the full ~10 GB GGUF cache restore
otherwise FATALs the emulator ("Not enough space to create userdata partition"). Neither
llmservice job is **yet a publish gate** (not in the `publish-snapshot`/`publish-release` `needs:`
graphs) so a Compose/AGP version-pin hiccup can't block a library release.
`android-llmservice/**/build/` is git-ignored.

## Open TODOs

Open TODOs for this repo live in [`TODO.md`](TODO.md). Cross-repo status
tracking lives in [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md).
