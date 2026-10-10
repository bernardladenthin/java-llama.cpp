<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Local agent report -- Windows workstation at llama.cpp b11538

Answers the three deliverables of the handover prompt, on `main` at `ed9b4f12` (llama.cpp
**b11538**). Every number below was measured on this machine on 2026-10-10; nothing is carried over
from the earlier session's notes except where a row says it is being *verified*. Where a question
could not be answered, it says so instead of narrowing the question.

**Machine.** Ryzen 7 5800H (Zen 3, 8 cores / 16 threads), 63 GB RAM, RTX 3070 + AMD iGPU,
Windows 11 Pro 10.0.26200.

## Summary

| Deliverable | Result |
|---|---|
| 1 -- the `mvn compile` blocker | **Solved, and it is not a project defect**: Oracle JDK 21 does not support the flag NullAway requires. Temurin 21.0.12.1 compiles the same tree cleanly |
| 2 -- the Java suite with models at b11538 | **Green**: ctest 607/607, `mvn test` 1885 run / 0 failures / 0 errors / 13 skipped, the test-count floor passed. The three new 400-case classes all pass |
| 3 -- Windows CPU variants (4 open items) | **All 4 answered** (sccache in Addendum 1). The build works and the suite passes -- **but only after a file CMake does not write**, which is the finding worth acting on. The module indirection costs nothing |
| Addendum -- sccache, GPU inference, GPU-as-module | sccache wraps clang but **caches nothing** (719/720); CUDA 28x and Vulkan 20x the CPU on pp512; one directory can carry the CPU set **and** a GPU module -- free for Vulkan, **-66.6 % token generation for CUDA**, caused by a missing `GGML_CUDA_GRAPHS_DEFAULT` line in this repo |

## Deliverable 1 -- the `mvn compile` blocker

### Environment

| | |
|---|---|
| `java -version` | `java version "21.0.9" 2025-10-21 LTS`, `Java HotSpot(TM) 64-Bit Server VM (build 21.0.9+7-LTS-338)` |
| `mvn -version` | `Apache Maven 3.9.11`, Maven home `C:\apache-maven-3.9.11` |
| Maven's JDK | `Java version: 21.0.9, vendor: Oracle Corporation, runtime: C:\Program Files\Java\jdk-21` |
| JDKs installed | **only that one.** No Temurin, no Microsoft build, no `~/.jdks` |
| `.java-version` / CI | `21`, with `distribution: temurin` in `publish.yml` |

### The stack trace

```
mvn -e -f llama/pom.xml compile
```

```
error-prone version: 2.50.0
BugPattern: NullAway
java.lang.IllegalStateException: Running NullAway in JSpecify mode requires either JDK 22+ or
passing the flag -XDaddTypeAnnotationsToSymbol=true to an older JDK that supports it; see
https://github.com/uber/NullAway/wiki/JSpecify-Support#supported-jdk-versions for details.
The flag -XDaddTypeAnnotationsToSymbol=true was passed, but it is not supported by the running
JDK (version 21.0.9+7-LTS-338). Typically, JDK 17.0.19+ or 21.0.8+ is required for flag support
(may vary by distribution), and Oracle JDK 17/21 may not support the flag.
    at com.uber.nullaway.NullAway.matchCompilationUnit(NullAway.java:1927)
    at com.google.errorprone.scanner.ErrorProneScanner.processMatchers(ErrorProneScanner.java:541)
```

on `llama/src/main/java/net/ladenthin/llama/args/package-info.java:[11,1]`, failing
`maven-compiler-plugin:3.16.0:compile (default-compile)`.

### The cause, and why it is not the project's

NullAway names it itself in the last sentence: the flag **was passed** and the running JDK does not
support it. Two things verified rather than assumed:

* the flag is configured in the failing execution -- `llama/pom.xml:499`, inside `default-compile`;
* it **reaches javac** -- `mvn -X` shows it exactly once on the compiler command line, alongside
  `-Xplugin:ErrorProne`.

So nothing in the POM needs changing, and that is also why the remote sandbox compiles this tree:
it runs a different JDK build.

### The counter-test the prompt asked for

| JDK | `mvn -q -f llama/pom.xml compile` |
|---|---|
| Oracle 21.0.9+7-LTS-338 | **exit 1**, the trace above |
| **Temurin 21.0.12.1+1** (portable zip, extracted to a temp directory) | **exit 0**, zero NullAway errors, `target/classes` populated |

```
JAVA_HOME=<temurin-21.0.12.1+1> mvn -q -f llama/pom.xml compile
```

**Everything in deliverables 2 and 3 therefore ran under Temurin 21.0.12.1+1**, not under the JDK
installed on this machine. For "does the Windows suite pass at b11538" that is equivalent to CI,
which uses Temurin; it is not this workstation's everyday environment.

## Deliverable 2 -- the Java suite with models, the way CI does it

### Commands

```
mvn -q -f llama/pom.xml compile
.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DGGML_OPENMP=OFF -DBUILD_TESTING=ON
ctest --test-dir llama\build -C Release
mvn -f llama/pom.xml test -Dnet.ladenthin.llama.test.ngl=0
.github/verify-test-counts.sh llama/target/surefire-reports --min-executed 1800
```

The build ran with MSVC 14.51.36231 from `Visual Studio\18\BuildTools` after `vcvarsall.bat x64`;
`build.bat` expects a prepared MSVC environment (it never calls `vcvars` itself), which is what
`ilammy/msvc-dev-cmd` provides in CI. `-ngl 0` because the Windows CI runners have no GPU, and this
machine does -- without it the suite would measure a configuration the report does not describe.

### Results

| Step | Result |
|---|---|
| `build.bat` | **`BUILD_EXIT=0`**, `jllama.dll` **11.0 MB** |
| `ctest -C Release` | **100% tests passed, 0 tests failed out of 607**, 145.85 s |
| `mvn test` | **`Tests run: 1885, Failures: 0, Errors: 0, Skipped: 13`** -- `BUILD SUCCESS` |
| `verify-test-counts.sh` | **exit 0** -- `1885 test(s) run, 13 skipped, 1872 executed across 130 class(es), none empty (floors: total 0, executed 1800)` |

The 607 also confirm independently what `CLAUDE.md` records as the C++ total, including the patch
guards that would fail at **link** time if a patch were lost (`test_prefetch.cpp` for `0017`,
`test_kolibri1.cpp` for `0016`, `test_rpc.cpp` for `0015`, `test_model_split.cpp` for `0012`).

### The three classes the prompt named -- all pass

Never run on Windows before this report.

| Class | tests | failures | errors | skipped |
|---|---:|---:|---:|---:|
| `ErrorHandlingTest` | 16 | 0 | 0 | 0 |
| `OpenAiServerEmbeddingsIntegrationTest` | 4 | 0 | 0 | 0 |
| `OpenAiServerCompletionIntegrationTest` | 5 | 0 | 0 | 0 |

So the `InvalidRequestException` work of b11538 -- the typed 400 answers -- behaves on Windows as it
does on Linux.

### The 13 skipped tests, by name

```
grep -l '<skipped' llama/target/surefire-reports/TEST-*.xml
```

| Class | Skipped tests | Why |
|---|---|---|
| `LlamaModelTest` | `testLogStdout` | `@Disabled` |
| `AudioInputIntegrationTest` | `audioInputProducesNonEmptyReply` | audio model + mmproj are not in the CI model set |
| `SystemOneIntegrationTest` | `aDecisionModelAnswersEveryQuestionType`, `aDecisionModelRejectsAnUnknownQuestionType` | decision model is not in the CI model set |
| `RouterModeIntegrationTest` | `chatCompletion_isProxiedToWorker`, `models_listContainsLoadedModel` | Linux-only |
| `BackendLoadTest` | 5 tests | needs several natives jars on the classpath |
| `ProcessRunnerTest` | `returnsStandardOutput`, `commandThatDoesNotEndInTimeIsKilledAndReported` | |

**The count matching CI's 13 is a coincidence, not agreement.** On Linux `RouterModeIntegrationTest`
runs and these two do not appear; the composition differs. Likewise the **1872 executed** is above
the 1856-1865 band `CLAUDE.md` records for CI, which fits a branch that added tests.

### Models

All 11 rows of `.github/models.csv` downloaded into `Q:\Modelle` and copied to `models/`, 6.2 GB,
no failures. `validate-models.sh` passed before the suite ran.

## Deliverable 3 -- Windows CPU variants (the four open items)

### The `build.bat` command line that works -- answered

```
.github\build.bat -G "Ninja Multi-Config" -DJLLAMA_CPU_VARIANTS=ON ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
  -DGGML_OPENMP=OFF -DOS_NAME=Windows -DOS_ARCH=x86_64
```

`BUILD_EXIT=0`, with `C:\Program Files\LLVM\bin` ahead of everything on `PATH` and
`vcvarsall.bat x64` for the MSVC headers and import libraries. CMake reports
`The CXX compiler identification is Clang 23.1.3 with GNU-like command-line` and
`Check for working CXX compiler: C:/Program Files/LLVM/bin/clang++.exe`. `build.bat` needed no
change, and `patches/0017` is what lets `quants.c` compile at this clang version.

### sccache -- answered in Addendum 1

No sccache was installed when this pass ran. It was installed afterwards, and **Addendum 1** has the
answer: the build is green and the cache is inert.

### The directory and the dependency check -- verified, both as previously measured

18 DLLs in `llama/src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu/`: `jllama.dll`,
`ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll` and all **14** `ggml-cpu-<level>.dll` (`x64`, `sse42`,
`sandybridge`, `ivybridge`, `piledriver`, `haswell`, `skylakex`, `cannonlake`, `cascadelake`,
`icelake`, `cooperlake`, `zen4`, `alderlake`, `sapphirerapids`). No `Release/` subdirectory.
`jllama-files.txt` names **15** files -- the 14 modules plus `ggml-rpc.dll` -- in a 16-line file
whose first line is a comment (a bare `wc -l` reads 16; the earlier measurement of 15 is correct).

```
python .github/verify-native-deps.py llama/src/main/natives
```

> `18 native libraries checked, 0 violations`

The printed dependency lists also show the hybrid CRT doing its job: only `api-ms-win-crt-*`
forwarders, `KERNEL32`, `ADVAPI32`, `WS2_32`, `SHELL32`, `CRYPT32` and the siblings -- no
`msvcp140.dll`, no `vcruntime140.dll`, no `vcomp140.dll`.

### The Java suite against the variant build -- the item that mattered, and it found something

**As the build leaves CMake, the suite fails outright:**

| | Result |
|---|---|
| `mvn test` (first attempt) | **`Tests run: 1651, Failures: 7, Errors: 57, Skipped: 12`** -- `BUILD FAILURE` |
| `verify-test-counts.sh` | **fails**: `only 1639 test(s) executed (1651 run, 12 skipped), below the floor of 1800` |

The library does not load at all:

```
java.lang.UnsatisfiedLinkError: No native library found for os.name=Windows, os.arch=x86_64, paths=[...]
    at net.ladenthin.llama.loader.LlamaLoader.loadNativeLibrary(LlamaLoader.java:298)
    at net.ladenthin.llama.loader.LlamaLoader.load(LlamaLoader.java:181)
    at net.ladenthin.llama.LlamaModel.<clinit>(LlamaModel.java:79)
```

53 further errors are that same failure carried as `ExceptionInInitializerError`, plus
`NoClassDefFoundError` for `LlamaModel`, `RpcServerNative` and `LlamaQuantizer`.

**The cause is the one `CLAUDE.md` already describes as "correct, not a defect": `jllama-extras.txt`
is absent, because CMake never writes it** -- in CI `merge-native-artifacts.sh` derives it. With no
`$ORIGIN` on Windows, `jllama.dll`'s import of `ggml.dll` cannot resolve and the load fails.

Reproducing that derivation exactly (the script's own find/sort/minus-`jllama-files.txt` logic)
yields two lines:

```
ggml-base.dll
ggml.dll
```

**With that file in place the variant build passes identically to the static one:**

| | Static (step 2) | Variants, no `jllama-extras.txt` | Variants, with it |
|---|---|---|---|
| `Tests run` | 1885 | 1651 | **1885** |
| Failures / Errors | 0 / 0 | 7 / 57 | **0 / 0** |
| Skipped | 13 | 12 | **13** |
| executed | 1872 | 1639 (**below floor**) | **1872** |
| verdict | `BUILD SUCCESS` | `BUILD FAILURE` | **`BUILD SUCCESS`** |

So the variant path works end to end on Windows -- including `RpcIntegrationTest`, i.e. a CPU
backend and an RPC backend that are *loadable modules* rather than linked code.

### Which variant ggml picks on the 5800H -- `haswell`, read from the process

`-lv 4` does not name it (ggml logs module decisions at `GGML_LOG_DEBUG`), so this was read off the
running JVM's mapped modules rather than inferred from the CPU:

```powershell
Get-Process java | ForEach-Object { $_.Modules } | Where-Object ModuleName -like 'ggml*'
```

| Mapped | |
|---|---|
| `ggml-cpu-haswell.dll` | **the chosen module** |
| `ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll`, `jllama.dll` | |

Exactly one CPU module stays mapped: ggml scored the other 13 and unloaded them. `haswell` is what
Zen 3 should take (AVX2, no AVX-512; `alderlake` needs AVX-VNNI it lacks), now measured rather than
expected.

### pp512 / tg128 -- and why a two-way comparison would have been wrong

`llama-bench` cannot be built from this tree (`llama/CMakeLists.txt:85` sets
`LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE`, and the prompt forbids source edits), so throughput was
measured **through the shipped library**: a 512-token prompt and 128 predicted tokens via
`completeWithStats`, reading upstream's own `prompt_per_second` / `predicted_per_second` out of the
result's `timings`. Qwen3-0.6B Q4_K_M, 8 threads, `-ngl 0`, `cache_prompt=false`, 8 rounds each,
medians:

| Configuration | pp512 median | pp range | tg128 median | tg range |
|---|---:|---|---:|---|
| MSVC `cl.exe`, static (what step 2 built) | 478.3 | 417-485 | 58.5 | 58.3-59.9 |
| clang 23.1.3, static | 497.2 | 449-506 | 56.9 | 55.5-59.3 |
| **clang 23.1.3, variants** | **508.8** | 478-521 | **62.0** | 59.5-62.8 |

A straight "variants versus the static build" would have compared **two** changes at once, because
the static build of step 2 is MSVC and the variant build must be clang. A third build -- static,
plain clang -- isolates them:

| Isolated change | pp512 | tg128 |
|---|---:|---:|
| compiler only (static: `cl.exe` -> clang) | +4.0 % | -2.7 % |
| **module indirection only (clang: static -> variants)** | **+2.3 %** | **+8.8 %** |

**The module indirection costs nothing measurable** -- the answer the item was asking for. It
measured *faster*, and on token generation the two sample ranges do not even overlap (55.5-59.3
against 59.5-62.8), so it is not noise in this measurement. **I cannot explain why a `haswell`
module would beat a host-native static build of the same source**, and I am not going to guess:
it deserves a second look, ideally with `llama-bench` against both, and on other hardware.

**These numbers are not comparable to the `llama-bench` figures in the handover (pp512 391.6 /
tg128 82.3).** Different harness (the server path, sampling included), a different quantisation,
and a different definition of the two metrics. They are internally comparable, which is what the
static-versus-variants question needs.

## Findings and proposals (nothing was changed)

1. **`jllama-extras.txt` makes a local Windows variant build untestable.** A developer following
   "CPU variants" gets 57 errors and a `BUILD FAILURE` that looks like a defect of the variant path,
   when the build is fine and only the *testability* is missing. **Proposal:** have
   `llama/CMakeLists.txt` write `jllama-extras.txt` for a `JLLAMA_CPU_VARIANTS` build with the same
   derivation `merge-native-artifacts.sh` uses (everything in the directory that `jllama-files.txt`
   does not name, sorted). The merge step can keep overwriting it in CI -- deriving it twice is
   harmless, and the check in `merge-native-artifacts.sh` stays the authority. **Consequence for the
   planned CI job either way:** it must run the merge step *before* the Java suite, not after.
2. **`CLAUDE.md` names no JDK requirement beyond the version.** One sentence under "Building the
   native library for local Java tests" -- that the build needs a JDK supporting
   `-XDaddTypeAnnotationsToSymbol`, and that Oracle 21 does not -- would have saved the whole of
   deliverable 1.
3. **`MODEL_PATH` is the only heavily used test model with no `-D` override.** Every
   capability-specific model has one (`tool.model`, `nomic.path`, `vision.model`, `tts.model`,
   `train.model`, ...), but `MODEL_PATH`, `DRAFT_MODEL_PATH`, `REASONING_MODEL_PATH` and
   `RERANKING_MODEL_PATH` are fixed filenames -- and `MODEL_PATH` is what all three classes this
   prompt named depend on. A `resolveModelProperty` wrapper on those four would let a developer run
   the suite against models they already have.
4. **Five of the eleven CI models cannot be substituted at all**, checked rather than assumed: of 49
   GGUFs on this machine, none is a reranker, an embedding model, a vision pair, a TTS pair or an F32
   `stories260K`. Those tests need a *capability*, so "use a model that is already there" silently
   turns them into skips.
5. **`jllama-files.txt` is written with CRLF line endings** on Windows. The suite passing proves the
   loader handles it; noted because a future reader of that file should not assume LF.

## What this report does not cover

* **Kolibri-1 (`patches/0016`) against the real model.** Both shards are on this machine (41.6 GB +
  18.2 GB = 59.8 GB) but 63 GB of RAM means mmap would page from disk for the whole run, so any
  number would describe the disk. Deferred to a machine that can hold the weights. The GGUF header
  *was* read, and it confirms the architecture the patch describes: `general.architecture`
  `kolibri1`, `expert_gating_func` **5** with `tokenizer.ggml.pre` `kolibri1` (so the **second** of
  the two published dialects), `expert_weights_norm` 0, 384 experts with 6 used and **1** shared,
  `attention.sliding_window` 513 with a `0` at every fifth of 50 layers (the full-attention NoPE
  layer), GQA 48/4, context 262144.
* **GPU inference**, the interactive agent, and the per-CPU variant question on other hardware --
  these are the parts CI structurally cannot cover either, and they are now written down as
  [`docs/local-test-plan.md`](../local-test-plan.md) rather than left in a handover prompt.

## Working-tree state

`llama/src/main/natives/` (git-ignored) now holds three backend directories from the addendum runs
-- `cpu/` (CPU variants), `cuda13/` and `vulkan/`, the latter two with the GPU module copied in by
hand -- and the 11 GGUFs are in `models/` (also git-ignored). Newly installed on the machine:
sccache 0.18.0 (in a temp directory, not on the permanent `PATH`) and the LunarG Vulkan SDK
1.4.363.0 under `C:\VulkanSDK` (`VULKAN_SDK` was set per build invocation, not persisted).
Nothing in the repository was touched -- no source file was edited, per the prompt.

## Addendum (same day) -- the three follow-up items

Three things CI structurally cannot show, measured on the same machine after the first pass. Two
needed installs that were not there before: **sccache 0.18.0** (the `SCCACHE_DL_VERSION` of
`build.sh`) and the **LunarG Vulkan SDK 1.4.363.0**. The SDK needed an elevated run -- two silent
attempts failed with `Cannot elevate access rights while running from command line`, with a
user-writable `--root` and a reduced component set as well -- so it was installed through an
explicit UAC prompt. No reboot was required; the Vulkan *runtime* (`C:\Windows\System32\vulkan-1.dll`)
already comes with the NVIDIA driver.

**CUDA here is toolkit 13.3**, while the project pins **13.4**. Both GPU builds are
**single-architecture** (`-DCMAKE_CUDA_ARCHITECTURES=86`, the RTX 3070's `sm_86`): right for a
measurement, wrong for a release, and `compute capability 8.6` in the load log confirms the match.

### Addendum 1 -- sccache 0.18.0 with plain clang: the build is green and the cache is inert

```
set USE_CACHE=true
.github\build.bat -G "Ninja Multi-Config" -DJLLAMA_CPU_VARIANTS=ON ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
  -DGGML_OPENMP=OFF -DOS_NAME=Windows -DOS_ARCH=x86_64
```

| | |
|---|---|
| `build.bat` probe | **`sccache probe OK (wrapped cl.exe)`** -- the launcher *is* set |
| Build | **`BUILD_EXIT=0`**, 745 compile steps, 19 files produced |
| `sccache clang` in isolation (a trivial TU) | **exit 0**, cacheable, 0 failed |
| Compile requests | **720** |
| Compile requests executed | **1** |
| **Non-cacheable calls** | **719** |
| Cache hit rate | **0.00 %** |

The reason is printed by sccache itself:

```
Non-cacheable reasons:
Can't handle UnknownFlag arguments with -Xclang     719
```

and the flag comes from **upstream llama.cpp**, not from this project -- `CMakeLists.txt`:

```cmake
add_compile_options(
    "$<$<COMPILE_LANG_AND_ID:C,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>")
```

set **unconditionally**, with no option in front of it, and only for Clang -- which is why the
existing Ninja jobs that compile with `cl.exe` are unaffected and their measured hit rates stand.

**So the probe's verdict is right in substance and wrong in scope:** sccache *can* wrap plain clang
(proved in isolation), but it cannot cache this project's clang command lines. **Consequence for the
planned Windows CPU-variants CI job: it would build cold every time**, because that job must use
plain clang -- `clang-cl` drops 5 of the 14 variants. That belongs in the job's cost estimate.
Two ways out, both outside this repository: upstream gating the flag behind
`CMAKE_DISABLE_PRECOMPILE_HEADERS`, or sccache passing unknown `-Xclang` arguments through instead
of discarding the translation unit.

### Addendum 2 -- Part B of the local test plan: GPU inference at b11538

Throughput as in deliverable 3 (through the shipped library, `completeWithStats`, Qwen3-0.6B
Q4_K_M, 8 threads, `-ngl 99`), 4 rounds per backend on an idle machine, round 1 excluded as warm-up:

| Backend | pp512 | tg128 | vs CPU |
|---|---:|---:|---|
| `cuda13` (static) | **14 124** | **265.6** | pp 28x, tg 4.3x |
| `vulkan` (static) | **10 127** | **243.7** | pp 20x, tg 4.0x |
| `cpu` (variants) | 503 | 61.3 | -- |

Both offload every layer. The loader's own line is what names the backend
(`net.ladenthin.llama.backend` forces one, since `cuda13` has priority over `vulkan`):

```
[jllama] using native backend 'cuda13'
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8191 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Laptop GPU, compute capability 8.6, VMM: yes
load_tensors: offloading 27 repeating layers to GPU
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =   372.65 MiB
load_tensors:   CPU_Mapped model buffer size =   121.71 MiB
```

**Two things only real hardware shows.**

**(a) Vulkan sees two devices and picks the right one.**

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64
ggml_vulkan: 1 = NVIDIA GeForce RTX 3070 Laptop GPU (NVIDIA)     | uma: 0 | fp16: 1 | bf16: 1 | warp size: 32
load_tensors:      Vulkan1 model buffer size =   372.65 MiB
```

It takes `Vulkan1`, the discrete NVIDIA, not the integrated AMD -- which is the documented default
(integrated only when there is no discrete GPU). Note that here **`Vulkan0` is the iGPU and
`Vulkan1` the NVIDIA**, the opposite assignment to the one `CLAUDE.md` records from a run with CUDA
also loaded. That is the same point its RPC section makes: **name the backend, never the index.**

**(b) Vulkan's first call after a cold pipeline cache costs three orders of magnitude.** In the very
first run on this machine, with no shader cache, round 1 measured **pp 25.9** against ~10 100 in the
rounds after it. With the cache warm, round 1 was 6745. A consumer sees this once per machine, not
once per process, but it is large enough to be mistaken for a hang.

### Addendum 3 -- E1: the GPU backend as a module beside the CPU set

The question: can **one** directory carry the 14-module CPU set *and* a GPU backend as a loadable
module, i.e. one jar with GPU plus a full CPU fallback? Both combinations build
(`-DJLLAMA_CPU_VARIANTS=ON` with `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`, `BUILD_EXIT=0`) -- and
**nvcc accepts the plain-clang host compiler**, which the variants path forces and which nvcc on
Windows officially does not support.

**It works at runtime.** With the module placed in the directory, the process maps the CPU module
*and* the GPU backend at the same time:

```
ggml-cpu-haswell.dll    ggml-cuda.dll    cublas64_13.dll    cublasLt64_13.dll
ggml.dll    ggml-base.dll    ggml-rpc.dll    jllama.dll
```

`offloaded 29/29 layers to GPU`, and `verify-native-deps.py` accepts the directory
(`37 native libraries checked, 0 violations`; 56 with the Vulkan directory beside it).

**Finding A -- the GPU module is built but never shipped.** `ggml-cuda.dll` (50.3 MB) and
`ggml-vulkan.dll` (43.9 MB) are produced in `llama/build/bin/Release/` and **not** copied into
`src/main/natives/.../<backend>/`. The copy list in `llama/CMakeLists.txt` -- the same block that
writes `jllama-files.txt` -- knows only the CPU modules plus `ggml`, `ggml-base` and `ggml-rpc`. For
these measurements the module was copied in by hand and both file lists were extended. Any future
"one jar for everything" work has to teach that block about GPU backend modules.

**Finding B -- CUDA loses two thirds of its token generation, and the cause is a missing line in
this repository.** Measured with 8 rounds (round 1 excluded), medians:

| | pp512 | tg128 |
|---|---:|---:|
| CUDA static | 14 124 | 265.6 |
| CUDA as a module | 11 328 (**-19.8 %**) | **88.8 (-66.6 %)** |
| Vulkan static | 10 127 | 243.7 |
| Vulkan as a module | 10 115 (-0.1 %) | **256.4 (+5.2 %)** |

The tg sample range for CUDA-as-a-module is 81.9-94.5, nowhere near 265.6, so it is not noise. And
**Vulkan loses nothing**, which rules out the module boundary and the shared `ggml` as the cause and
pointed at something CUDA-specific. It is:

| `CMakeCache.txt` of a configure | `GGML_CUDA_GRAPHS` |
|---|---|
| CUDA static | **ON** |
| CPU variants + CUDA | **OFF** |

CUDA Graphs replace many per-token kernel launches with one graph replay, so they act on *generation*
and barely on prompt processing -- exactly the shape of the measurement -- and Vulkan has no
equivalent feature to lose.

Why it is off: ggml defaults it to `GGML_CUDA_GRAPHS_DEFAULT`, which is `OFF`
(`ggml/CMakeLists.txt:117`), and **llama.cpp sets two ggml defaults before adding ggml**:

```cmake
# llama.cpp/CMakeLists.txt
166:    set(GGML_LLAMAFILE_DEFAULT ON)
170:    set(GGML_CUDA_GRAPHS_DEFAULT ON)
```

The variants path in `llama/CMakeLists.txt` adds ggml itself, ahead of llama.cpp, and repeats only
the **first** of the two:

```cmake
318:    set(BUILD_SHARED_LIBS ON)
322:    set(GGML_LLAMAFILE_DEFAULT ON)
323:    add_subdirectory(${llama.cpp_SOURCE_DIR}/ggml ${llama.cpp_BINARY_DIR}/ggml)
```

**Proposal:** add `set(GGML_CUDA_GRAPHS_DEFAULT ON)` beside line 322. It is latent today -- no CI job
builds the variants path together with CUDA -- and it would cost two thirds of token-generation
throughput the moment one does, which is precisely what E1 was evaluating. `CLAUDE.md` says "**The
one** ggml default llama.cpp changes itself (`GGML_LLAMAFILE`) is repeated"; that sentence is now
wrong, and the general rule behind it is worth writing instead: **every `*_DEFAULT` llama.cpp sets
before its own `add_subdirectory(ggml)` must be repeated in the variants path**, because that path
configures ggml first.

### Verdict on E1

One directory carrying the CPU variant set plus a GPU backend module **is feasible and, for Vulkan,
free**. Before it could ship, two things must happen: the copy list has to place the GPU module
(Finding A), and the CUDA-graphs default has to be repeated (Finding B). A third is worth a thought:
such a jar is `jllama.dll` + `ggml`/`ggml-base`/`ggml-rpc` + 14 CPU modules + a 44-50 MB GPU module,
and `LlamaLoader` extracts every file of the chosen backend on first use.
