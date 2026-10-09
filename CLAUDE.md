# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b11512**

## Natives jars: one directory per backend (`.github/natives.csv`)

`net.ladenthin:llama` is the **Java classes only**. Every native build ships as its own jar of the
same artifact, classifier `<backend>-<os>-<arch>`, holding exactly one directory
`net/ladenthin/llama/<OS>/<ARCH>/<backend>/` (27 today: `cpu-*` for 8 platforms, `metal-macos-aarch64`,
`msvc-windows-*`, and the GPU backends). Because the directories never overlap, **any combination of
natives jars can share one classpath**, and `LlamaLoader` tries the backends it finds in a fixed
order (`BACKEND_PRIORITY`: cuda13, rocm, sycl-fp16, sycl-fp32, sycl, vulkan, opencl, openvino, metal,
msvc, cpu), taking the first whose library loads. Every build carries the CPU backend, so a GPU
library that loads can still run a model on the CPU (`-ngl 0`); one that fails to load (no vendor
runtime) falls through to the next directory, ending at `cpu`. `net.ladenthin.llama.backend=<dir>`
forces one and fails loud. `net.ladenthin:llama-platform` is a **pom-packaging** module (a
dependency list, no jar, nothing to upload) naming the classes jar plus the CPU/Metal jars of every
desktop platform; consumers use it with `<type>pom</type>`.

This replaced (in 5.2.0) the old layout, where the default jar carried the CPU natives of every
platform at `<OS>/<ARCH>/` and each GPU classifier was a complete replacement jar with its own
compile pass: two jars could not share a classpath, a GPU jar had no CPU fallback, and the
all-backends fat jars needed a `jllama-backends.txt` manifest to tell backends apart.

**The list is `.github/natives.csv`** (classifier, directory, library, platform yes/no) — the one
place a natives jar is declared, in the same spirit as `models.csv`. Everything else either reads
it or is checked against it:

| Place | Relation to the list |
|---|---|
| `llama/CMakeLists.txt` | names the backend directory (`JLLAMA_BACKEND`: GGML flag → name; `msvc` = Visual Studio generator; `metal` = Apple + `GGML_METAL`) and writes to `src/main/natives/net/ladenthin/llama/<OS>/<ARCH>/<backend>/` |
| `llama/pom.xml` `natives` profile | one `maven-jar-plugin` execution per row, **generated**: `python3 .github/check-natives.py pom` prints them |
| `llama-platform/pom.xml` | depends on the rows with `platform=yes` |
| `publish.yml` build jobs | each uploads its tree as artifact `natives-<classifier>`; `package` waits for every one of them (transitively) |
| `LlamaLoader.BACKEND_PRIORITY` | must contain every backend directory |
| the all-backends fat jars | **derived**: one `all-<os>-<arch>` per OS/arch with more than one natives jar (no `msvc`, no Android) — `check-natives.py fatjar-targets` prints them; each must be uploaded for and launched by a smoke job, named in the agent jar's `Class-Path` and in the README |
| `.github/check-natives.py` | runs in `code-style` (first minutes of every run); fails when any row of this table disagrees with the list — including CMake's backend names, the dependency allowlists and the README rows |
| `.github/merge-native-artifacts.sh` | reads the list: every listed `natives-*` artifact present, no other, each holding its library and nothing outside its directory, no path claimed twice; writes `jllama-extras.txt` (sibling files loaded before the library, e.g. OpenVINO's `OpenCL.dll` on Windows -- never the files a build's own `jllama-files.txt` names, which are only extracted; each of those must exist) |
| `.github/package-fatjars.sh` | reads the list: the built natives jars match it, each holds only its directory and the right `Automatic-Module-Name`; merges the all-backends fat jars (every file of every backend directory byte-identical to its natives jar, the CPU directory to the default fat jar's) and fails unless it produced exactly the targets `check-natives.py fatjar-targets` derives |
| `.github/verify-native-deps.py` | every library of a directory (18 in a CPU-variants directory): exact dependency allowlist per CPU directory (`cpu`/`metal`/`msvc`) plus the files next to it, and for the Android OpenCL build (bionic + `libOpenCL.so`), denylist for the other GPU ones; a sibling dependency only through run path `$ORIGIN`; the manylinux_2_28 builds (both Linux CPU jars, CUDA) held to `GLIBC_2.28`; every Android library 16 KB page-aligned (Google Play). Runs in `package` and on the staged AAR libraries |
| `.github/smoke-natives-jars.sh` (`package` job) | loads the real jars: classes + all 27 natives jars at once, on the classpath **and** the module path (on the GPU-less runner normally ending at `cpu`) |

**Adding a natives jar:** a row in `natives.csv`, the execution `check-natives.py pom` prints, a build
job uploading `natives-<classifier>`, the backend name in CMake and `BACKEND_PRIORITY` if new, a
README row. `check-natives.py` fails until they agree; nothing else needs touching — the fat-jar
targets follow by themselves (a new OS/arch with a GPU backend also needs its smoke job, which
`check-natives.py` then demands). The checks are Python in `.github/buildcheck/` with unit tests
(see "Build checks, shared files and the release gate").

**Why the build jobs are not spawned from the list as one matrix** (considered and rejected): the 27
builds use genuinely different toolchains — dockcross images, the CUDA redist archives, ROCm pip
wheels, oneAPI, OpenVINO, `clang-cl` on arm64, qemu for s390x, three macOS variants — so a single
matrix job would be a web of `if:` conditions; and `needs:` on a matrix waits for every entry, so
each test job would wait for the slowest build (CUDA). The list removes the scattering that matters
(the packaging facts); a build job's only contract is its artifact name, and the merge check
enforces it.

**Module path.** Resources are looked up through the `ClassLoader`, never `Class.getResource`
(which only sees the own module's resources on the module path). Each natives jar declares
`Automatic-Module-Name: net.ladenthin.llama.natives.<classifier with _ for ->`. **Without it every
natives jar derives the module name `llama` from its file name and the JVM does not refuse to start —
it silently keeps the first one and drops the others' natives** (measured; `package-fatjars.sh`
checks the manifest of every built jar for that reason). Nothing `requires` a natives module, so a
module-path launch needs `--add-modules` (or the natives jars on the classpath). `module-info.java`
`requires` Jackson and SLF4J — it did not before, so the classes jar never worked on the module path.

**Locally**, CMake writes into `llama/src/main/natives/` (git-ignored); surefire adds that directory
to the test classpath (`additionalClasspathElements`), so `mvn test` finds the library. `-P natives`
is CI-only (a local tree holds one directory); a missing directory there is caught by the merge
step, not by Maven.

## CPU variants (Linux x86-64 and aarch64): `JLLAMA_CPU_VARIANTS`

The two Linux CPU jars, `cpu-linux-x86-64` and `cpu-linux-aarch64`, are built since 5.2.0 the way
upstream builds its own release binaries: `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`, one small
`ggml-cpu-<level>` module per instruction-set level, of which `ggml_backend_load_best` loads the best
at run time (it scores each module's own feature check against the running CPU). Before, each jar
was **one** library for one level -- x86-64 the Haswell baseline (x86-64-v3, the block in
`llama/CMakeLists.txt` that forces `GGML_AVX2` & co.: a CPU without AVX2 died with SIGILL, a JVM
crash and not an exception, at the first matrix multiplication, and AVX-512/VNNI/AMX stayed unused
on CPUs that have them), aarch64 plain ARMv8. `-DJLLAMA_CPU_VARIANTS=ON` builds it; the two Linux
build jobs pass it, every other platform still ships the single static library, and the option
refuses them (the Windows plan is in `TODO.md`):

- **Only ggml becomes shared.** `FetchContent_MakeAvailable` is pointed at a non-existent
  `SOURCE_SUBDIR`, so it only fetches and patches; ggml is then added first with
  `BUILD_SHARED_LIBS ON` and llama.cpp afterwards with it `OFF` (llama.cpp skips its own
  `add_subdirectory(ggml)` when a `ggml` target exists). llama, mtmd and llama-common stay static in
  `libjllama.so`. The one ggml default llama.cpp changes itself (`GGML_LLAMAFILE`) is repeated.
- **The directory holds 18 files on x86-64**: `libjllama.so`, `libggml.so`, `libggml-base.so`
  (unversioned: the loader extracts plain files, so file name, SONAME and `DT_NEEDED` must agree), 14
  `libggml-cpu-<level>.so` (`x64`, `sse42`, `sandybridge`, `ivybridge`, `piledriver`, `haswell`,
  `skylakex`, `cannonlake`, `cascadelake`, `icelake`, `cooperlake`, `zen4`, `alderlake`,
  `sapphirerapids`) and `libggml-rpc.so`; **12 on aarch64** (8 modules, `armv8.0_1` to `armv9.2_2`:
  dotprod, fp16, SVE, i8mm, SVE2, SME) -- all with run path `$ORIGIN` only. `jllama-files.txt` lists
  everything besides `libjllama`; `LlamaLoader` extracts them next to it **without loading them**
  (`BACKEND_FILES_FILE`, the counterpart of `jllama-extras.txt`, which loads). `JNI_OnLoad` then calls
  `ggml_backend_load_all_from_path(<its own directory>)` (found via `dladdr`): ggml's default
  searches next to the executable, which here is `java`.
- **The extraction directory is keyed by the build**
  (`LlamaLoader.extractionDirectoryName`: `jllama-backend-<backend>-<CRC and size of the library's
  jar entry>`), because it is shared between JVMs and now holds many files. Keyed by the backend
  alone, a JVM running another jllama build wrote into the directory of a running one; on Windows
  that replaces exactly the modules the first JVM scored and unloaded again (which ones depends on
  the CPU) and leaves a mixture of two builds for the next start -- measured with upstream's DLLs.
- **RPC is a module too**, so `rpc_bridge.cpp`/`rpc_support.hpp` resolve its four entry points through
  `ggml_backend_reg_get_proc_address` (`rpc_proc<>()`, as upstream's `common/arg.cpp` does) in every
  build.
- **Built in manylinux_2_28** (AlmaLinux 8, gcc-toolset-14, glibc 2.28): x86-64 in the dockcross
  image the CUDA job already used, aarch64 natively on `ubuntu-24.04-arm` inside the pypa image
  (`.github/manylinux/manylinux_2_28-aarch64`, see "Linux aarch64"). manylinux2014's GCC 10 lacked
  `-mavxvnni` and `-mamx-tile` (`alderlake`, `sapphirerapids`; ggml does not check). The glibc floors
  meet at **2.28**: x86-64 rises from 2.17, aarch64 falls from ~2.39. `verify-native-deps.py` reads
  every library's version-needs table and holds both directories (and CUDA) to `GLIBC_2.28`; the
  modules themselves reference `GLIBC_2.27` at most (measured in the image).
- **No `ctest` in the two build jobs**: `jllama_test` calls the CPU/RPC backends directly, so the
  option refuses `BUILD_TESTING`; `test-cpp-linux-x86_64` and `test-cpp-linux-aarch64` build the
  default static library with the tests. The Java suite (`test-java-linux-x86_64`, the langchain4j
  integration job through `-Dnet.ladenthin.llama.lib.path`, the fat-jar smokes) runs real inference
  through the variant build -- a load alone proves little: in the Windows measurement every crash
  came after a successful load with a correct device count.
- **The natives-jar checks know directories with many files**: `merge-native-artifacts.sh` requires
  every file `jllama-files.txt` names and keeps them out of `jllama-extras.txt`; `package-fatjars.sh`
  compares every file of a backend directory; `verify-native-deps.py` checks every library and allows
  a sibling as a dependency only through `$ORIGIN`.
- **Measured.** Locally (GCC 13, a Xeon with AVX-512 + VNNI, no VBMI/BF16): the x86-64 directory is
  36 MB (13.4 MB zipped), `libjllama.so` 15 MB of it; ggml picked `cascadelake`, and with the AVX-512
  modules removed it fell back to `haswell` (skipping `alderlake`, whose AVX-VNNI the CPU lacks).
  On a Ryzen 7 5800H (Zen 3; upstream's b11476 Windows binaries, Qwen3-0.6B Q4_K_M, 8 threads):
  `haswell` 313 t/s pp512 and 14.3 t/s tg128, the plain `x64` module 30.0 and 8.8 -- a baseline-only
  build would be unusable for prompt processing, and the 14 x86 modules together are ~20 MB. The gain
  on AVX-512/AMX hardware is unmeasured (`TODO.md`).

## Build checks, shared files and the release gate (`.github/buildcheck/`)

The pipeline's static checks are a small **stdlib-only Python package**, `.github/buildcheck/`, with
unit tests in `.github/buildcheck/tests/` (`python3 -m unittest discover -s .github/buildcheck/tests
-t .github`, from the repo root). The `.github/*.py` scripts are thin command-line wrappers around it,
so every check can be driven with literal inputs in a test — including the ELF/PE/Mach-O parsers
(synthesized binaries) and the workflow parser. It deliberately does not use PyYAML: the runners do
not guarantee it, and the checks only need the job graph, which `workflow.py` reads from the two
shapes this repo writes and rejects anything else loudly.

| Module | CLI | What it checks |
|---|---|---|
| `natives.py` | `check-natives.py` (`code-style`) | everything that names a natives jar agrees with `natives.csv` (see "Natives jars") |
| `models.py` | `check-natives.py` | every `*_MODEL_NAME` of publish.yml's `env:` is a filename of `models.csv` |
| `nativedeps.py` | `verify-native-deps.py` (`package`, `package-android-aar`) | dependency allowlists, 16 KB Android alignment |
| `hipoffload.py` | `verify-hip-offload-compressed.py` (the two ROCm jobs) | no uncompressed GPU code bundle |
| `workflow.py` | — | the job graph of a workflow (`needs`, uploads, transitive closure) |
| `releasegate.py` | `check-release-gate.py` (`shared-files`) | every job gates **both** publish jobs, unless `.github/release-gate-exemptions.txt` names it with a reason |
| `sharedfiles.py` | `check-shared-files.py` (`shared-files`) | the files kept byte-identical across the four sibling repos |
| `versions.py` | `check-versions.py` (`shared-files`) | **warns** where a Maven dependency or plugin is used in another version than in a sibling repo |
| `runscripts.py` | `check-run-scripts.py` (`shared-files`) | every `run:` script of the workflows and composite actions that runs in bash parses (`bash -n`) |

**The release gate.** A job nothing waits for can go red while a release still ships — the natives
builds `package` once forgot to wait for, and the aarch64 fat jars that were signed and attached
without any job launching them, were both that shape. So "not gating" is a written decision: a line
`<job>: <reason>` in `.github/release-gate-exemptions.txt`. The check fails both ways — a job neither
gating nor listed, and a listed job that gates after all (a stale exemption hides the next job of that
name). Introducing it found `vmlens` gating nothing in all four repos; it now gates both publish jobs.

**Shared files (all four repos).** Files kept byte-identical in java-llama.cpp, BitcoinAddressFinder,
srcmorph and streambuffer are listed with their SHA-256 in each repo's **`.github/shared-files.sha256`**
(`sha256sum` format). An entry can also name **one job of a workflow**,
`.github/workflows/publish.yml#startgate`: `startgate`, `shared-files`, `verify-signing-key`,
`check-snapshot` and `check-tag` are identical in all four `publish.yml` files (plus
`verify-signing-key-gradle`, `github-snapshot` and `github-release` in the three Maven-only ones), and
such an entry hashes just that job -- so a job kept identical by convention is now checked like a
file, without moving it into a reusable workflow (no check name or `needs:` changes). An entry ending
in `?repo` covers a file or job identical up to the repository's name (hashed with the name replaced
by `{repo}`), e.g. `SUPPORT.md?repo`. The
`shared-files` job, gating both publish jobs, **fails** when a listed file changed here
alone, and **warns** when another repo's default branch lists the same file with a different hash (a
sync is one change per repo and lands in four steps, so a warning, not a failure). **The manifest is
the reference for what must stay equal**: changing a shared file means changing every copy, then
`python3 .github/check-shared-files.py --write` in each repo. Forgetting one is not a disaster — the
job says so, and the history shows it — but nobody can edit a shared script without learning that it
is shared. Duplicating the library across repos is therefore deliberate. A separate actions/library
repo was considered and rejected: every consumer would need a pinned SHA per use (Scorecard's
pinned-dependencies rule), a release process of its own, and a cross-repo checkout; copies with a
checksum keep each repo self-contained.

**Maven versions across the four repos (`check-versions.py`).** They pin the same build tooling and
Dependabot bumps each repo on its own, so the versions drift one pull request at a time; the
`shared-files` job therefore compares every `groupId:artifactId` the POMs use (plugins, dependencies,
annotation-processor paths, and the formatter version Spotless is configured with -- `${...}`
resolved) with the siblings' default branches and **warns** per difference. Warnings only, because a
bump lands in four pull requests and the first one merged must not turn the other three red; the
repos' own artifacts (`net.ladenthin:*`) are left out. It compares what is used, not property names:
those differ between repos and include values that must differ (`maven.compiler.release`). The POM
list per repo is `versions.POMS` -- a new module's POM goes there, in all four copies.

**Run scripts parse (`check-run-scripts.py`).** A cleanup once lost the trailing ` \` of six continued
lines in `publish.yml` -- each left a line starting with `||`/`&&`, a syntax error in a step only a
native build, the AAR job or the snapshot release reaches. actionlint did not see it (it parses
scripts only with shellcheck installed), `bash -n` sees every one. So the `shared-files` job runs
`bash -n` over every `run:` script of `.github/workflows/` and `.github/actions/*/action.yml` that
runs in bash, deciding the shell as the runner does (step `shell:`, job then workflow
`defaults.run.shell`, else PowerShell on a `windows` runner and bash elsewhere; a matrix/input runner
is Windows only where the step's `if:` says `runner.os == 'Windows'`). pwsh/powershell/cmd/python are
skipped, an unknown shell is an error. Literal blocks are checked line for line (errors point at the
file line), folded and plain values as YAML folds them, `${{ ... }}` becomes a word. Like
`workflow.py` it reads the two-space layout these workflows use, not general YAML.

Shared today: the `buildcheck` modules above marked shared in
`__init__.py` (+ their tests and CLIs), `print-crash-logs.sh`, `verify-signing-key.sh`,
`verify-bytecode-version.sh`, `sign-fatjars.sh` (jllama + srcmorph), `smoke-fatjar-cli.sh` (BAF +
srcmorph), the signing self-test, `lombok.config`, the issue/PR templates, `CODE_OF_CONDUCT.md`,
`claude.yml`, `claude-code-review.yml`, `codeql.yml`, `scorecard.yml`, `reuse.yml`, `osv-scanner.yml`,
`dependabot.yml`, `.java-version`, `.editorconfig`, `.gitattributes`, `FUNDING.yml`, `CODEOWNERS`,
`LICENSES/{MIT,Apache-2.0}.txt` and `.mvn/jvm.config`; the job `verify-signing-key-gradle` (its Gradle
version is a literal for that reason -- bump it with `GRADLE_VERSION`); and, identical up to the
repository name (`?repo` entries, see `sharedfiles.py`), `SUPPORT.md` and `ISSUE_TEMPLATE/config.yml`
(in the siblings also `CITATION.cff`, `sonarqube.yml` for srcmorph + streambuffer, and the
`code-style` job for BAF + streambuffer). **License of `.github`:** every file there is `MIT OR Apache-2.0` in all four repos
-- one header, so a file can be shared byte for byte -- except the third-party ones annotated in
`REUSE.toml` (the generated dockcross wrappers, the JNI headers).

**Composite actions (`.github/actions/`, this repo only).** `restore-models` (restore the shared GGUF
cache + `validate-models.sh`; the one place a job gets its models — 16 call sites) and
`install-sccache-windows` (10 call sites; the caller keeps `if:` and `continue-on-error`),
`build-core` (parent + `llama`, optionally more modules, built without tests/checks/javadoc/signing,
`install` or `package` -- the one copy of that skip-flag list, 9 call sites) and
`publish-cpu-aar-local` (download + stage the Android CPU natives, publish the CPU AAR to mavenLocal;
the emulator test and the two llmservice jobs). Composite actions for steps; for whole jobs, the two
shapes below.

**Reusable workflow for the macOS/Windows Java tests (`.github/workflows/java-tests.yml`).** The
three macOS arm64 test jobs and the two Windows x86-64 ones were five copies of one job differing in
the runner, the natives artifact, one Maven flag and the failure-artifact name. They are now five
**callers** (`uses: ./.github/workflows/java-tests.yml` + four `with:` inputs) that keep their ids,
names and `needs`, so `package`, the release gate and `check-natives.py` see the same graph; the
OS-specific steps (memory report, core dumps vs. WER minidumps) are chosen inside by `runner.os`.
Two constraints shaped it: a called workflow sees **none** of the caller's `env:`, and `with:` cannot
read `env` — which is why the JDK version moved to **`.java-version`** (`setup-java`'s
`java-version-file`, used by every job and by `codeql.yml`/`sonarqube.yml`, so one file names it)
and why the callee needs no model names (the tests default to the CI model set). The Linux x86-64
test job stays inline: it differs in far more than inputs.

**Matrix for the fat-jar smokes (`smoke-fatjar`).** See "All-backends server fat jars" below. A
matrix and not a reusable workflow, because the rows differ only in data (target, runner).

**Check names changed with both** — a called job shows as `<caller name> / Java tests`, a matrix row
as `Smoke test all-backends fat jar (<target>)`. That was measured, not assumed, before doing it:
none of the four repositories has a required status check (each has only a ruleset — no deletion,
no force push, PR required — and classic branch protection is off), so no setting pins a check
name. Should required checks ever be added, name these new forms.

## Upgrading CUDA Version

Current CUDA version: **13.4** (Linux `cuda-toolkit-13-4` from NVIDIA's rhel8 repo; Windows 13.4 redist archives)

To change the CUDA version, update the following places:

1. **`.github/build_cuda_linux.sh`** — the `sudo dnf install -y cuda-toolkit-13-4` line and the
   `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc` line (plus the header comment).
2. **`.github/workflows/publish.yml`** — the `build-windows-x86_64-cuda` job's
   "Install CUDA Toolkit … (NVIDIA redist archives)" step: the `v13.x` directory, `CUDA_PATH_V13_x`,
   and the per-component archive list. **Copy that list from upstream's
   `.github/actions/windows-setup-cuda/action.yml` at the pinned `GIT_TAG`** — the component versions
   differ per component (cuBLAS and CCCL have their own numbering) and cannot be derived from the CUDA
   version. (This replaced `Jimver/cuda-toolkit`, which never shipped 13.4.)
3. **Only for a major bump:** the backend name `cuda13` — in `llama/CMakeLists.txt`
   (`JLLAMA_BACKEND`), `.github/natives.csv` (the two `cuda13-*` rows), the generated pom executions
   (`check-natives.py pom`), the build jobs' artifact names, and `LlamaLoader.BACKEND_PRIORITY`.
   `check-natives.py` fails until they agree. No change for a minor bump.
4. **CCCL pin** — both CUDA builds pass `-DGGML_CUDA_CCCL_VERSION=v3.4.3` (`build_cuda_linux.sh` and the
   Windows CUDA job), as upstream's own CUDA release jobs do since llama.cpp #29792: ggml's CUB
   `DeviceTopK` path needs CCCL >= 3.4.3 and falls back to a sort below it, and CUDA 13.4 bundles an
   older 3.4. **Drop both flags once the toolkit is 13.5 or newer** (it bundles CCCL 3.5); follow
   upstream's `release.yml` matrix comment, which says the same.
   **The pin makes ggml fetch CCCL, and CCCL calls `include(CTest)` unconditionally**, which creates
   `BUILD_TESTING` as a cache variable defaulting to ON. That is why `llama/CMakeLists.txt` declares
   `option(BUILD_TESTING ... OFF)` before its first `FetchContent_MakeAvailable()`: declared after,
   the option was a no-op, both CUDA jobs built `jllama_test`, and its gtest discovery failed on the
   GPU-less runners (no `libcuda.so.1` / `nvcuda.dll`) -- the first Publish run after the bump
   (37550451678). Keep the option there.
5. **`CLAUDE.md`** — the "Current CUDA version" line above.

Available CUDA versions for RHEL8/Manylinux_2_28 can be browsed at:
```
https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/
```
and the Windows redist components at `https://developer.download.nvidia.com/compute/cuda/redist/`.

**Note:** Each CUDA version supports only certain GCC versions. If the dockcross container uses a newer GCC than CUDA supports, the build will fail with `unsupported GNU version`. Check NVIDIA's compatibility table before downgrading CUDA.

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

A second Android arm64 build has the OpenCL backend enabled and Adreno-tuned kernels embedded.
It ships as the natives jar `opencl-android-aarch64` (directory `Linux-Android/aarch64/opencl/`)
and as the `llama-android-opencl` AAR; the `cpu-android-*` jars and the `llama-android` AAR stay
CPU-only. The `crosscompile-android-aarch64-opencl` job runs the dockcross-android-arm64 build with
`-DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON` and uploads
`natives-opencl-android-aarch64`; the rest is the natives-jar wiring (see "Natives jars" above).

Local sanity build:
```bash
.github/dockcross/dockcross-android-arm64 .github/build_opencl_android.sh \
  "-DOS_NAME=Linux-Android -DOS_ARCH=aarch64 \
   -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
   -DGGML_OPENCL_USE_ADRENO_KERNELS=ON"
```
Artifacts land in `src/main/natives/net/ladenthin/llama/Linux-Android/aarch64/opencl/`.

The dockcross image does not ship OpenCL headers or a stub `libOpenCL.so`, so
`build_opencl_android.sh` first stages Khronos `OpenCL-Headers` and
cross-builds `OpenCL-ICD-Loader` into `/tmp/opencl-stage/` before invoking the
main project cmake with `-DOpenCL_INCLUDE_DIR=...` and `-DOpenCL_LIBRARY=...`.
At runtime the device must provide its own OpenCL ICD (`libOpenCL.so`);
Qualcomm Adreno drivers do. Devices without an ICD should use the CPU-only
`llama-android` AAR.

## Windows natives (Ninja CPU + MSVC + CUDA/Vulkan/OpenCL GPU)

The Windows x86-64 natives ship in **five** forms (natives jars): the CPU build with the
**`Ninja Multi-Config`** generator (`cpu-windows-x86-64` / `cpu-windows-x86`, the ones in
`llama-platform`), the Visual Studio / MSVC build (`msvc-windows-x86-64` / `msvc-windows-x86`), and
three GPU backends, **`cuda13-windows-x86-64`**, **`vulkan-windows-x86-64`** and
**`opencl-windows-x86-64`** (all Ninja).

**Why Ninja is the default (the flip).** The Visual Studio generator ignores
`CMAKE_{C,CXX}_COMPILER_LAUNCHER`, so only Ninja Multi-Config can front `cl.exe` with sccache over
Depot WebDAV. **Both generators use the same MSVC toolchain** (`cl.exe`, static `/MT` CRT via
`CMAKE_MSVC_RUNTIME_LIBRARY`, same Release flags, same runner), so the produced
`jllama.dll` binaries are **functionally equivalent with identical runtime
dependencies** — the only difference is build-system plumbing + caching. Making Ninja the `cpu`
build gives the most-pulled natives the sccache cache; MSVC stays available (directory `msvc`,
which the loader tries before `cpu`) for anyone who wants the Visual-Studio-generator build. (Upstream llama.cpp also builds its Windows artifacts with
Ninja Multi-Config + MSVC.) Both Windows CPU builds are validated end-to-end with the full
model-backed Java suite (`test-java-windows-x86_64` = Ninja, `test-java-windows-x86_64-msvc` = MSVC).

**`GGML_OPENMP=OFF` on every Windows CPU job -- it is faster, not a concession.** All four x86-64/x86
CPU jobs (both generators) pass it, as the arm64 jobs already did for a dependency reason. Measured
on 2026-10-09 (Ryzen 7 5800H, Qwen3-0.6B Q4_0, 8 threads, `GGML_NATIVE=ON`, static builds, runs
interleaved, t/s):

| | pp512 | tg128 |
|---|---:|---:|
| MSVC, OpenMP on | 359.8 | 41.7 |
| MSVC, OpenMP off | 371.0 | **79.9** |
| clang, OpenMP on | 390.6 | 30.2 |
| clang, OpenMP off | 391.6 | **82.3** |

Token generation is **1.9x (MSVC) / 2.7x (clang)** faster without OpenMP and prompt processing is
unchanged within error, so there is nothing to trade away. The cause is visible in the thread
scaling (clang, tg64): OpenMP peaks at 4 threads and *degrades* above it -- 36.0 (t=2), 37.6 (t=4),
29.1 (t=8), 21.5 (t=16) -- while ggml's own `std::thread` pool gives 54.5 / 74.2 / 71.4. Generation
is synchronisation-bound (little work per barrier), so the OpenMP runtime's barrier cost swamps it;
prompt processing has enough work per barrier to hide it. It drops `vcomp140.dll` from the
dependency list as a side effect (MSVC's OpenMP, a third redistributable DLL after `msvcp140` and
`vcruntime140`). **Do not remove the flag as redundant**: without it a *clang* Windows build also
imports `libomp140.x86_64.dll`, which no redistributable carries -- the same `0xc0000135` class the
arm64 job hit. The 32-bit job follows by analogy, not by measurement.

**Linux is the open counterpart, deliberately unchanged.** The Linux CPU jobs pass no `GGML_OPENMP`
either, so they ship with it ON (ggml's default) through **libgomp**, a third implementation that
was not measured -- two of two measured implementations (LLVM `libomp`, MSVC `vcomp`) show the
penalty, which makes it a strong suspicion and not a result. There is no dependency argument on
Linux (libgomp is everywhere), only the throughput one, so it needs a measurement on a Linux host
before the same one-line change is made (`TODO.md`).

**GPU runtime libraries are NOT bundled.** The GPU natives jars ship only the single monolithic
`jllama.dll` (llama.cpp + ggml + the backend are statically linked in — `BUILD_SHARED_LIBS OFF`). The consumer's driver/toolkit must supply the runtime: CUDA needs the
installed CUDA 13 Toolkit (`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll` on `PATH`); Vulkan
needs `vulkan-1.dll` (ships with current GPU drivers); OpenCL needs the vendor ICD
(`System32\OpenCL.dll`). Not bundling = no NVIDIA-EULA redistribution obligation. **GitHub-hosted
Windows runners have NO GPU**, so the GPU jobs **build the artifact only** (no `-DBUILD_TESTING`/`ctest`)
— a GPU-linked `jllama_test.exe` can't even be enumerated on a GPU-less runner (it errors probing for a
device, so `gtest_discover_tests` registers a failing `*_NOT_BUILT` sentinel). The CPU-only C++ unit
suite is fully covered by the `C++ Tests` job + the CPU Windows jobs; model-backed GPU inference is
local / self-hosted.

Windows-specific wiring (the rest is the natives-jar wiring, see "Natives jars" above; CMake names
the Visual Studio generator build `msvc`, the Ninja one `cpu`):

1. **`.github/build.bat`** — the sccache probe guard (mirrors `build.sh`) wraps the **cl.exe** C/C++ TUs
   only. Unlike `build.sh` (Linux), it does **not** wrap `nvcc`: sccache on Windows can't parse the nvcc
   command line (`sccache: error: Could not parse shell line`) and fails every `.cu` compile, so CUDA
   device code builds with nvcc directly (uncached). `build.bat` also propagates a `cmake --build`
   failure as a non-zero exit (a prior bug let a failed CUDA build exit 0 → empty artifact → late
   `package` failure); the GPU upload steps additionally use `if-no-files-found: error` as a backstop.
2. **`.github/build_opencl_windows.bat`** — stages Khronos OpenCL-Headers + builds OpenCL-ICD-Loader
   (`OpenCL.lib`), then delegates to `build.bat` with `-DOpenCL_INCLUDE_DIR`/`-DOpenCL_LIBRARY`
   (the Windows analogue of `build_opencl_android.sh`).
3. **`.github/workflows/publish.yml`** — build jobs (all `windows-2025-vs2026`, `ilammy/msvc-dev-cmd@v1`,
   sccache v0.18.0 zip + Depot WebDAV), each uploading `natives-<classifier>`:
   - `build-windows-x86_64` / `build-windows-x86` — **Ninja CPU**.
   - `build-windows-x86_64-msvc` / `build-windows-x86-msvc` — **MSVC CPU**.
   - `build-windows-x86_64-cuda` — CUDA `13.4` assembled from NVIDIA's redist archives (upstream's
     `windows-setup-cuda` component list; `Jimver/cuda-toolkit` stops at 13.3.1) + `-DGGML_CUDA=ON`.
   - `build-windows-x86_64-vulkan` — `jakoch/install-vulkan-sdk-action` + `-DGGML_VULKAN=ON`.
   - `build-windows-x86_64-opencl` — `build_opencl_windows.bat -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON`.

**First CI run (PR #276, run 28327740376):** the default Ninja CPU flip, the MSVC classifier, and the
**OpenCL** job were green on the first try. Two GPU jobs needed a toolchain fix: **CUDA** failed with
`Version not available: 13.0.0` because the pinned `Jimver/cuda-toolkit@v0.2.24` predated CUDA 13.x →
bumped to `@v0.2.35` + `13.2.0` (matches the Linux pin, classifier stays `cuda13-…`); **Vulkan** failed
`find_package(Vulkan)` because `humbletim/install-vulkan-sdk` set `VULKAN_SDK` but laid the SDK out in a
way CMake's `FindVulkan` couldn't read → switched to `jakoch/install-vulkan-sdk-action` (purpose-built,
FindVulkan-compatible). Because all five Windows build jobs are in the `package`/publish `needs:` graph, a
GPU-toolchain failure blocks packaging — the same release-gating policy every build job follows.

**sccache on every Windows Ninja job.** All ten Ninja build jobs install sccache (x86_64 or the
native `aarch64` release) through the composite action `.github/actions/install-sccache-windows`
(arch from `RUNNER_ARCH`; its version must equal `SCCACHE_DL_VERSION` in `build.sh`) with the same
`USE_CACHE` / `SCCACHE_WEBDAV_*` env; only the two MSVC-classifier
jobs cannot, because the Visual Studio generator ignores compiler launchers. It cannot red a build, by
three guards in `build.bat`: the install step is `continue-on-error`; a probe compiles through sccache
before it is trusted; and — because that probe only proves `cl.exe`, while arm64 builds with `clang-cl`,
ROCm with its own `clang` and SYCL with `icx` — **a configure or build that fails with sccache as the
launcher is retried once from a clean build dir without it**. The retry is unconditional (cmd cannot
tee the output to match an error signature the way `build.sh` does), so a genuine compile error costs
one extra uncached attempt before it fails. Every configure also passes `-DGGML_CCACHE=OFF`: without
it ggml self-enables any sccache it finds on `PATH` whenever no launcher is set — exactly the
probe-failed and retry cases — and the "uncached" build silently goes through sccache after all (the
same trap `build.sh`'s retry hit with nvcc). `build.bat` uses no `goto`/labels on purpose: it is
checked out with LF line endings, where cmd's label search is unreliable.

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

## Linux Vulkan natives + Windows arm64 CPU and Vulkan

Four natives jars that extend the matrix toward upstream llama.cpp's release set.

**Linux Vulkan (`vulkan-linux-x86-64` + `vulkan-linux-aarch64`).** A vendor-neutral GPU jar for
Linux (NVIDIA / AMD / Intel) with no CUDA toolkit. The build jobs are `build-linux-x86_64-vulkan`
(native `ubuntu-latest`, **not** dockcross — the Vulkan SDK is a trivial apt install and upstream
builds ubuntu-vulkan the same way) and `build-linux-aarch64-vulkan` (`ubuntu-24.04-arm` + GCC 14).
Both `apt-get install libvulkan-dev glslc glslang-tools`, build `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF`,
and are **build-only** (no `ctest`: a Vulkan-linked `jllama_test` errors enumerating devices on a
GPU-less runner — same as the Windows GPU jobs). Glibc floor rises to the ubuntu baseline (like the
aarch64 CPU jar); acceptable for a GPU artifact. GPU runtime `libvulkan.so.1` is supplied by the
consumer's driver — nothing is bundled (same policy as every GPU backend).

**Windows arm64 Vulkan (`vulkan-windows-aarch64`).** Added at the llama.cpp b11395 bump, when upstream
put a Windows arm64 Vulkan build into its release set (#29954). `build-windows-arm64-vulkan` uses the
arm64 CPU job's toolchain below (`windows-11-arm`, `clang-cl`, `GGML_OPENMP=OFF`) and installs the SDK
exactly as upstream's release job does: LunarG's x64 installer with the `com.lunarg.vulkan.arm64`
component (the arm64 import library), whose x64 `glslc` runs under the runner's x64 emulation; the SDK
version is upstream's `VULKAN_VERSION`, not the one the x86-64 Vulkan job pins through
`jakoch/install-vulkan-sdk-action`. Build-only like every GPU job, and part of the `all-windows-aarch64`
fat jar (tried before `opencl`, then `cpu`), so the `windows-aarch64` row of `smoke-fatjar` launches it.

**Windows arm64 CPU (`cpu-windows-aarch64`, in `llama-platform`).** `build-windows-arm64` runs
natively on GitHub's free `windows-11-arm` runner (`ilammy/msvc-dev-cmd` `arch: arm64`, Ninja
Multi-Config, `-DOS_ARCH=aarch64`, build + `ctest`) and writes `Windows/aarch64/cpu/`. No Java change
was needed: `OSInfo` already maps a Windows-on-ARM JVM (`os.arch=aarch64`)
to `Windows/aarch64` (it isn't in `archMapping`, so it falls through `translateArchNameToFolderName`).
sccache runs here too, from its native `aarch64-pc-windows-msvc` release (it wraps `clang-cl`; see
"sccache on every Windows Ninja job" above). **Compiler: `clang-cl`, not MSVC
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

## Additional GPU-backend natives (ROCm/HIP, SYCL, Win-arm64 OpenCL, OpenVINO)

Eight further GPU natives jars extend the matrix toward upstream llama.cpp's full release set, with
the same natives-jar wiring as every other one (see "Natives jars"; each build job is in
`package.needs`, **fail-loud**). All are **build-only** (GitHub runners have no matching GPU) and
bundle **no** vendor runtime.

| Classifier | GGML flag(s) | Job runner / toolchain | Directory |
|---|---|---|---|
| `rocm-linux-x86-64` | `GGML_HIP=ON -DCMAKE_HIP_COMPILER=… -DGPU_TARGETS=…` | `ubuntu-latest` + ROCm 10 TheRock wheels (pip, `rocm-sdk path`) | `Linux/x86_64/rocm` |
| `rocm-windows-x86-64` | `GGML_HIP=ON` | `windows-2022` + ROCm 10 TheRock wheels (pip) | `Windows/x86_64/rocm` |
| `sycl-fp16-linux-x86-64` | `GGML_SYCL=ON -DGGML_SYCL_F16=ON` (`icx`/`icpx`) | `ubuntu-latest` + Intel oneAPI apt | `Linux/x86_64/sycl-fp16` |
| `sycl-fp32-linux-x86-64` | `GGML_SYCL=ON` (`icx`/`icpx`) | `ubuntu-latest` + Intel oneAPI apt | `Linux/x86_64/sycl-fp32` |
| `sycl-windows-x86-64` | `GGML_SYCL=ON` (`icx`) | `windows-2025-vs2026` + oneAPI installer | `Windows/x86_64/sycl` |
| `opencl-windows-aarch64` | `GGML_OPENCL=ON …ADRENO_KERNELS=ON` (clang-cl, `GGML_OPENMP=OFF`) | `windows-11-arm` (arm64 CPU job's toolchain) | `Windows/aarch64/opencl` |
| `openvino-linux-x86-64` | `GGML_OPENVINO=ON` | `ubuntu-latest` + OpenVINO apt | `Linux/x86_64/openvino` |
| `openvino-windows-x86-64` | `GGML_OPENVINO=ON` | `windows-2025-vs2026` + OpenVINO archive | `Windows/x86_64/openvino` |

**ROCm comes from TheRock, and the version and GPU targets follow upstream.** Since ROCm 7.14 AMD
builds and releases ROCm through [TheRock](https://github.com/ROCm/TheRock); both ROCm jobs install
its Python wheels (`rocm[libraries,devel]` from `stable.repo.amd.com/rocm/whl-next/`) exactly as
llama.cpp's own `ubuntu-rocm` / `windows-rocm` release jobs do, and read the paths back with
`rocm-sdk path`. The ROCm version follows upstream's `release.yml` at the pinned `GIT_TAG` —
**re-check it on every llama.cpp bump**. The `GPU_TARGETS` lists deliberately go **further than
upstream's**: they are every target TheRock builds for that OS (its `SUPPORTED_GPUS.md`). On Linux
that adds gfx900/gfx906/gfx90c/gfx1153 to upstream's list, on Windows only gfx900/gfx906/gfx90c
(upstream's `windows-rocm` list already carries gfx1153, its `ubuntu-rocm` list does not — checked at
b11256). All four are "build passing" in TheRock, not release-ready.
Supporting more rather than fewer is the policy, with one limit: an extra stays only while it builds
without problems and without local patches; the moment one needs a patch or holds back a newer
ROCm/llama.cpp, drop it. The two lists differ **only** by the Instinct parts
(gfx908/gfx90a/gfx942/gfx950), which ROCm supports on Linux alone.

**The ROCm GPU code is compressed (`--offload-compress`), and CI enforces it.** Each HIP
translation unit embeds one code object per GPU target, i.e. the whole kernel set (flash attention,
mmq per quant type, …) once per architecture — stored **uncompressed** by default, which made the
Windows `jllama.dll` ~1 GB for its 23 targets (234 MB zipped, so the jar never showed it). That size
also lands on disk: `LlamaLoader` extracts the library to the temp dir on every start, and in the
all-backends fat jar ROCm is tried right after CUDA, i.e. on nearly every machine without an NVIDIA
card. `llama/CMakeLists.txt` therefore adds `--offload-compress` to the `ggml-hip` target only
(`$<COMPILE_LANGUAGE:HIP,CXX>`: its sources are HIP on Linux and CXX on Windows, where upstream
compiles HIP as C++), so clang stores every bundle zstd-compressed (a `CCOB` bundle) and the HIP
runtime inflates it when the module loads. Upstream llama.cpp does **not** do this; its
`ggml-hip.dll` carries the same uncompressed code (for 20 targets). `.github/verify-hip-offload-compressed.py` runs after the build in
both ROCm jobs, prints the library size and bundle counts (also into the job summary), and fails on
any uncompressed bundle (`__CLANG_OFFLOAD_BUNDLE__`) or on none compressed — a toolchain or upstream
change that drops the flag reds the job instead of quietly shipping the 1 GB library again. The
jar barely shrinks (zip already compressed the code); what shrinks is the extracted library.

One routing note: **Linux SYCL** ships two precision variants at the *same* arch, so
`CMakeLists.txt` names them two *distinct* backends by `GGML_SYCL_F16` (`sycl-fp16` vs `sycl-fp32`);
the loader tries fp16 first.

The Linux jobs that install a multi-GB vendor toolchain (CUDA, ROCm, both SYCL) start with
`ggml-org/free-disk-space` — the same guard upstream llama.cpp's CUDA/ROCm jobs use (ROCm also clears the
tool cache, as upstream does, which is safe only because the step runs before `setup-java`).

The vendor toolchain install steps in `publish.yml` are **first-pass** (apt repos / vendor installers
pinned to a specific version): if a URL/version 404s in CI, the job fails loud and the step is adjusted
— the failure is intentional signal, not a regression to hide behind `continue-on-error`.

## macOS arm64: three build jobs, one shipped dylib

macOS arm64 is the **only** platform built by more than one job; one of them ships, as the natives
jar `metal-macos-aarch64` (directory `Mac/aarch64/metal/`, in `llama-platform`):

| Job | Build flags | Artifact | Role |
|---|---|---|---|
| `build-macos-arm64-metal-15` (macos-15) | `-DLLAMA_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF` | `natives-metal-macos-aarch64` | **shipped** |
| `build-macos-arm64-metal` (macos-26) | `-DLLAMA_METAL_EMBED_LIBRARY=ON` (host-native) | `macos-26-metal` | test-only |
| `build-macos-arm64-no-metal` (macos-15) | `-DLLAMA_METAL=OFF -DGGML_NATIVE=OFF` | `macos-15-no-metal` | test-only (writes `Mac/aarch64/cpu/`) |

The shipped variant is the metal-15 build because it is the only one with **both** Metal **and**
`GGML_NATIVE=OFF` (portable across Apple-silicon generations); the other two exist to prove the
no-Metal path and the newest macOS SDK still build and pass the Java suite. The two Metal builds write
the **same** directory, which is why the test-only artifacts are named outside the `natives-*` glob.
The host-native Metal build ran on `macos-14`, which GitHub retires by 2026-11-02 (with brownouts in
October 2026 that fail every `macos-14` job); it moved to a **pinned**
`macos-26`, not `macos-latest`, so the image cannot change under it unnoticed, and not to `macos-15`,
which the other two jobs already cover. Its Java test job runs with `-ngl 0`, so it exercises the
Metal-enabled build on the CPU; only `Java Tests macOS 15 arm64 (Metal)` offloads to the GPU.

**Why that naming matters (the incident).** Two artifacts sharing a relative path, downloaded with
one glob and merged, get extracted onto one file — and the survivor can be a **byte-level hybrid** of
both, not either input. All three macOS jobs used to upload under one glob: the published dylib's
ad-hoc linker signature then no longer matched its own `__TEXT` pages and macOS **SIGKILLed every
process that loaded it** (shipped broken in 5.0.6 and several 5.0.7 snapshots; 66/4078 and 1141/4097
code pages failed their stored hashes).

**The guard: `.github/merge-native-artifacts.sh`.** The three consumer jobs download `natives-*`
**unmerged** (`merge-multiple` off → one subdirectory per artifact) and let that script do the merge.
Each artifact must hold exactly the directory its row in `natives.csv` names, and no relative path may
be claimed twice. Note **why the check runs before the merge**: a collision still leaves exactly one
file on the path, so a post-merge assertion cannot see it — it is only observable while the artifacts
are still separate. A future job that reopens the hole reds the pipeline instead of shipping a
corrupt binary.

**The end-to-end gate: `smoke-fatjar-macos`.** The three macOS Java test jobs each test the dylib
*their own job* built, so until now nothing exercised the **packaged** artifact on macOS — Linux and
Windows had fat-jar smoke jobs downstream of `package`, macOS had none, which is why this bug reached
three releases with a fully green pipeline. The job (`needs: [package]`, `macos-15`, gates both
publish jobs) downloads `llama-jars` and runs `.github/smoke-native-macos.sh`, which extracts
`Mac/aarch64/metal/libjllama.dylib` from the **default fat jar** and asserts two things: `codesign --verify
--strict` (re-hashes the code pages against the signature's stored hashes — the direct check for a
dylib assembled from two builds) and a real JVM load via `.github/smoke/NativeLoadSmoke.java`
(`java -cp <fatjar> …`, the JDK single-file source launcher), which forces
`LlamaLoader.initialize() → System.load() → JNI_OnLoad` and then crosses JNI for
`getLlamaCppBuildInfo()`, checked against the `LlamaCppVersion` pin. A third check comes first:
the dylib's `minos` (`otool -l`, `LC_BUILD_VERSION`) must not exceed **15.0**, the supported floor.

**Minimum macOS: 15.0, pinned, not inherited.** `llama/CMakeLists.txt` sets
`CMAKE_OSX_DEPLOYMENT_TARGET` to `15.0` before `project()` (an explicit `-D` or
`MACOSX_DEPLOYMENT_TARGET` still wins). Without it the linker takes the build host's own macOS
version, so moving the shipped build to a newer runner image would silently drop every user on the
older release, and no test would see it -- they all run on that same newer image. `build.sh` prints
`minos`/`sdk` of every dylib it builds, and the smoke above fails a shipped one above 15.0. Raise
the floor in both places together, deliberately: `JLLAMA_MAX_MINOS` in the smoke and the CMake
default.

Two macOS specifics: it targets the **default** fat jar because there is no `all-macos-*` fat jar to
target (macOS has no GPU backend besides Metal, which the default fat jar carries, so
`package-fatjars` builds no macOS variant), and it asserts native loadability plus the code
signature rather than a server round trip, so it does not reuse `smoke-test-fatjar.sh`.
Deliberately model-free (~1 min, no GGUF/cache restore/network): a full model-backed macOS server
smoke would be strictly more, but this catches the failure class that actually shipped and is cheap
enough to always run. It is the macOS member of the cross-repo convention in
[`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md)
("No release asset is attached that CI has not run"), which BAF and srcmorph implement with a shared
`smoke-fatjar-cli.sh`.

## All-backends server fat jars (GitHub Release assets, never Maven Central)

Every pipeline run assembles **per-OS multi-backend server fat jars** and, on the release
paths, attaches them to GitHub: `llama-<version>-all-<os>-<arch>-jar-with-dependencies.jar`
for `linux-x86-64`, `linux-aarch64`, `windows-x86-64`, `windows-aarch64`, plus the default
CPU fat jar — each with a `.sha256` **and** a detached GPG `.asc` signature. They are
**download assets only**: the Central deploy invocations run without the `assembly` profile
and are untouched. (The cross-repo "fat jar → GitHub Release, never Central, signed `.asc`"
convention shared with BAF and srcmorph is documented in
[`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md).)

Mechanism:

1. **`.github/package-fatjars.sh`** — run by the `package-fatjars` job (`needs: [package]`,
   downloads `llama-jars`). Takes the natives jars from `.github/natives.csv` and checks them in
   **both** directions against the built jars; every natives jar must hold exactly its own
   directory, its library, and its `Automatic-Module-Name`. Because the directories never overlap,
   an all-backends jar is a **plain merge**: for each OS/arch that has a GPU backend (Android and the
   `msvc` backend excluded — no `java -jar` on Android, and `msvc` would only be a second CPU library),
   copy the default fat jar, **remove the native trees of every other platform**, and unzip every
   natives jar of that OS/arch into it. No manifest: the loader's fixed priority order does the rest.

   **Only the jar's own OS+arch tree is kept.** Originally each combined jar was the default fat jar
   plus backends, i.e. it carried the CPU natives of all nine platforms — ~71.5 MB of dead weight in
   `all-windows-x86-64` (341.5 MB → ~270 MB, computed from the per-tree sizes of run 36606543343).
   That was how the jar was built, not a requirement: the introducing commit (`aff68f5e`) gives no
   reason. The rule is deliberately the strict one, so `all-windows-x86-64` also loses `Windows/x86`
   and `Windows/aarch64` — a 32-bit JVM on 64-bit Windows needs the default fat jar, which still runs
   everywhere. The "natives missing from a jar" incidents in the history are a different layer — the
   default jar's collection (`package` once lacked `needs:` on three build jobs, `7ad8066a`), which the
   merge step's completeness check against `natives.csv` now covers.
   Trees are found as the **upper-case directories** below `net/ladenthin/llama/` (OSInfo folder
   names; Java packages are lower-case) and matched by exact path component — a `Linux*` prefix
   would also hit `Linux-Android`, and a pattern without the trailing `/` of a directory matches
   upper-case class files such as `LlamaModel.class`. Fail-loud checks after the zip update: every
   added backend byte-identical to its natives jar, the own CPU library byte-identical to the default
   jar's, **no other `<OS>/<ARCH>` tree left**, the `.class` count equal to the default jar's (the
   check that caught the missing-slash variant during development), `Main-Class` intact, and all
   four release targets produced.
2. **`LlamaLoader` backend selection** — the same code path as for any classpath: every backend
   directory present is tried in `BACKEND_PRIORITY` order, each extracted into its own temp subdir
   (`jllama-backend-<name>/`; backends share file names), its `jllama-extras.txt` files loaded first.
   A load failure (missing vendor runtime → `UnsatisfiedLinkError`) moves to the next backend, ending
   at `cpu`. `net.ladenthin.llama.backend` forces one backend (fail-loud). A backend whose extra module
   is already resident from a previously failed attempt is skipped (by-name import cross-wiring).
3. **`publish.yml` wiring** — **every** OS/arch fat jar is launched by one row of the
   **`smoke-fatjar` matrix** (`fail-fast: false`; rows `linux-x86-64` on `ubuntu-latest`,
   `linux-aarch64` on `ubuntu-24.04-arm`, `windows-x86-64` on `windows-2025-vs2026`,
   `windows-aarch64` on `windows-11-arm`). Each row runs its jar via real `java -jar` on a
   GPU-less runner (cached draft model, `--chat-template chatml`): poll `/health` to 200, assert a
   `/v1/chat/completions` choice, and require the loader's backend-selection log line — so every
   GPU backend failing its load and falling back to the CPU backend is exercised on the actual
   release asset. Each row downloads one small single-jar artifact, `llama-fatjar-smoke-<target>`,
   rather than the multi-GB `llama-fatjars` set; the Linux rows also re-run the bytecode gate and
   the `linux-x86-64` row the RPC smoke. `check-natives.py` holds the uploads (name **and** the jar
   inside) and the matrix rows to the targets `natives.csv` derives, so a new fat jar cannot ship
   unlaunched — **the aarch64 rows close a real gap**: those jars were built, GPG-signed and
   attached to every release while `publish.yml` referenced them zero times, which is exactly what
   the cross-repo rule forbids — and that rule exists because a corrupt macOS dylib shipped in
   three releases under a fully green pipeline. `publish-snapshot`/`publish-release` `need`
   `package-fatjars` + `smoke-fatjar` (a matrix job is done when every row is; fail-loud gating);
   `github-release-signed` and `github-snapshot` additionally download `llama-fatjars` into their
   asset directory, then **GPG-sign each fat jar** via `.github/sign-fatjars.sh` (a detached `.asc` alongside the
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
   parity), then runs upstream's own **`scripts/ui-assets.cmake`** (a plain `cmake -P`
   script, **no npm and no host executable**) to produce the platform-independent
   **`webui-generated/ui.cpp` + `ui.h`**, uploaded as the `webui-generated` artifact.
   Upstream **#28445** deleted the `tools/ui/embed.cpp` host tool this job used to
   compile and replaced it with that script plus `ui.cpp.in`/`ui.h.in` templates; the
   job passes `BUILD_UI=OFF HF_ENABLED=OFF` so the script takes its priority-1 path
   ("pre-built assets in `<UI_SOURCE_DIR>/dist`") over the tree npm just built — no
   second npm run, no Hugging Face download. `LLAMA_UI_GZIP` is upstream's own knob
   and replaced the job's hand-rolled gzip loop. The sparse checkout therefore needs
   **`scripts` as well as `tools/ui`**. **The completeness guard cannot be a bare
   `grep LLAMA_UI_HAS_ASSETS`**: `ui.h.in` emits `/* #undef LLAMA_UI_HAS_ASSETS */`
   for an empty table, so that token is present either way and the check passes the
   failure case — the old `embed.cpp` emitted no such line, which is why the naive
   grep used to work. The job asserts the **active** `#define` plus a non-zero count
   parsed out of `std::array<llama_ui_asset, N>`.
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
# needs node/npm + network for the asset build; the embed step is plain cmake -P
git clone --depth 1 --branch b11512 https://github.com/ggml-org/llama.cpp /tmp/lc
( cd /tmp/lc/tools/ui && npm ci && npm run build )
mkdir -p webui-generated /tmp/ui-gen
cmake -DUI_SOURCE_DIR=/tmp/lc/tools/ui -DUI_BINARY_DIR=/tmp/ui-gen \
      -DLLAMA_SOURCE_DIR=/tmp/lc -DBUILD_UI=OFF -DHF_ENABLED=OFF -DLLAMA_UI_GZIP=ON \
      -P /tmp/lc/scripts/ui-assets.cmake
cp /tmp/ui-gen/ui.cpp /tmp/ui-gen/ui.h webui-generated/
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

Because `sccache` is **content-addressed** and llama.cpp is pinned (`GIT_TAG b11512`), the
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
1. `crosscompile-linux-x86_64` — ✅ **verified green** in PR #245, then on manylinux2014: sccache
   **v0.16.0** probe passed in-container (devtoolset-10 gcc), `sccache ON` over Depot WebDAV,
   warm cache 277/278 hits (99.64%), 1m46s build time. Since the CPU variants it runs in the
   manylinux_2_28 image the CUDA job already used (gcc-toolset-14), with the same env.
2. `crosscompile-linux-x86_64-cuda` (via `build_cuda_linux.sh`, which execs `build.sh`) —
   ✅ **verified green with nvcc caching, full-arch always.** `build.sh` also wraps nvcc
   (`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache`, scoped to CUDA builds), so both the gcc C/C++ TUs
   (134 model files + ggml + httplib) **and** the per-arch `.cu` device passes cache over Depot.
   CI dropped the single-arch validation shortcut (`CUDA_FAST_BUILD`/`CUDA_ARCH` removed from the
   job) — every run builds the full arch set and leans on the warm cache for speed. A warm run hit
   **100%** on CUDA / CUBIN / device-code (139 CUDA hits, 99.86% overall, 3 misses), cutting the job
   from **~51 min cold to ~15 min warm**. The first-run debug diagnostics (`SCCACHE_LOG` /
   `SCCACHE_ERROR_LOG` / `RUST_BACKTRACE`) were dropped once confirmed; `sccache --show-stats` still
   prints the hit table every run. **CUDA 13.3 regression (caught on the b10333/CUDA-13.3 publish
   dispatch, run 31339594933):** nvcc 13.3's device-compile pipeline broke sccache's nvcc wrapping —
   `fatbinary fatal: Could not open input file 'acc.compute_75.ptx'` immediately followed by
   `sccache: Compiler killed by signal 1`, on the very first `.cu` TU (a cold-cache miss, not a hit
   issue) — an sccache/nvcc incompatibility for `-virtual` architecture targets (e.g. `75-virtual`),
   not a real compile error. **Two fixes were needed, not one.** (1) The existing mid-build
   retry-without-cache mechanism (see below) didn't catch it because its trigger regex didn't include
   this failure's wording; `build.sh`'s regex now also matches `Compiler killed by signal`. (2) That
   alone still wasn't enough — verified live on the very next dispatch (run 31340386884): the retry
   fired but failed **identically**, because ggml's own `ggml/src/CMakeLists.txt` self-enables
   ccache/sccache (`GGML_CCACHE`, default `ON`) whenever it finds one on `PATH` and
   `CMAKE_C_COMPILER_LAUNCHER`/`CMAKE_CXX_COMPILER_LAUNCHER` are unset — exactly the retry's state,
   since sccache is still on `PATH` from the failed attempt (build.sh only clears its own `$LAUNCH`
   flags, not ggml's independent detection). ggml wires itself in via the **global**
   `RULE_LAUNCH_COMPILE` CMake property, which wraps nvcc too, not just C/C++ — so the "uncached"
   retry silently re-enabled the very launcher it was trying to avoid. Fix: the retry's `cmake
   -Bbuild` now also passes `-DGGML_CCACHE=OFF`, so a genuinely uncached build is guaranteed
   regardless of what's left on `PATH`. This fallback now falls back to a real, green `-O3` build
   like every other sccache/nvcc incompatibility instead of redding the job.
3. `crosscompile-linux-aarch64` — ✅ **enabled**, a **native `ubuntu-24.04-arm` build** inside the
   pypa manylinux_2_28 aarch64 image (`MANYLINUX_ARGS` forwards the env like `DOCKCROSS_ARGS`):
   `build.sh` self-fetches the aarch64 static-musl sccache in the container (the fetch block in
   `build.sh` maps `uname -m` → `x86_64`/`aarch64`) and the probe guards it. See "Linux aarch64:
   native build in the manylinux_2_28 image" below.
4. `crosscompile-android-aarch64` — ✅ **enabled** (same steady-state env; probe guards it).
5. `crosscompile-android-aarch64-opencl` — ✅ **enabled**. `build_opencl_android.sh` stages the
   OpenCL headers/loader, then delegates the jllama cmake build to `build.sh` via `exec`
   (same pattern as `build_cuda_linux.sh`), so it inherits the probe and launcher automatically.

Per-job recipe: `USE_CACHE` and `SCCACHE_WEBDAV_ENDPOINT` are workflow-level `env:` (every job has
them); a cached job adds only `SCCACHE_WEBDAV_TOKEN: ${{ secrets.DEPOT_TOKEN }}` to its own `env:` —
kept per job on purpose, since at workflow level the token would reach every job, including those
running third-party actions — and, for dockcross,
`DOCKCROSS_ARGS: "-e SCCACHE_WEBDAV_ENDPOINT -e SCCACHE_WEBDAV_TOKEN -e USE_CACHE"` — the
dockcross wrapper only forwards host env it is explicitly told to via `-e`. The fetched sccache
version is the `SCCACHE_DL_VERSION` knob in `build.sh` (default **0.18.0**; overridable per-job
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

## Wire-name registries (CLI options, request keys, trainer keys)

Three surfaces leave this library as names on a wire, and each is checked against the code that reads
them. The names are **enum constants**, not string literals, and each declares the contract it must
satisfy:

| Registry | Receiver it is checked against | Contract kinds | Guard |
|---|---|---|---|
| `args.ModelFlag` + `args.ModelOption` | `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options` | `SERVER_PARSER`, `PROJECT_PSEUDO` | `src/test/cpp/test_model_flags.cpp` |
| `parameters.RequestField` | `server_schema::make_llama_cmpl_schema(...)` | `SCHEMA`, `OAI_LAYER` | `src/test/cpp/test_wire_contracts.cpp` |
| `parameters.TrainingField` | `jllama_train::config_keys()` (`train_engine.h`) | — | `src/test/cpp/test_wire_contracts.cpp` |

`cmake/extract-java-wire-names.cmake` reads the registry `.java` files at configure time and emits
`{name, contract}` pairs into generated headers; the C++ tests feed them to the receivers. It matches
**enum constant declarations only**, so prose and javadoc cannot contribute a name, and it fails the
configure when a registry declares a name twice or extracts implausibly few. For `OAI_LAYER` keys —
which by definition never reach the schema — it additionally sweeps upstream's own sources
(`tools/server/*.cpp` + `common/*.cpp`, globbed) for a *reader shape* (`json_value(x, "k", …)`,
`.contains("k")`, `.at("k")`) and emits the hit count, because the schema cannot vouch for them.

**Why the failure modes differ, and why all three need a guard.** An unregistered CLI option is a hard
parse error — `loadModel()` throws `"Failed to parse model parameters"`, so the model does not load.
The other two are worse: llama.cpp's request schema discards an unknown key without a word, and
`train_engine.cpp` reads with `j.value(key, default)`, so a dead field simply stops having an effect.
Either way a Java test asserting the string mapping (`hasKey("--mlock")`) passes forever.

**Rules when touching a registry:**

1. **Adding a name** means adding a constant with its contract. There is no `put(String, ...)` to
   bypass — that is the point.
2. **A name with no counterpart is deleted, never deprecated.** A method that writes a key the
   receiver discards reads as configuration and behaves as a no-op.
3. **The exemption checks are inverted on purpose.** A `PROJECT_PSEUDO` / `OAI_LAYER` name cannot go
   stale by outliving its constant. It *can* go stale the other way — upstream may later register a
   name we exempted, hiding a real check — so the tests assert such a name is still unknown to the
   receiver, and that the exempt set is non-empty (a generator that lost the contract column would
   otherwise exempt everything).
4. **An `OAI_LAYER` name must additionally be read by *something* upstream.** Absence from the schema
   is satisfied just as well by a key nothing reads at all, so the inverted check alone left a hole
   the exact size of the problem — `chat_template` sat in it, written by a public builder method and
   read by nobody (upstream's only occurrence of that name is the `/props` payload it *emits*). The
   reader sweep above closes it. It is a source pattern, not the parser: it proves a key is read from
   some body, not that this endpoint reads it. That is enough for the failure that occurred, a count
   of zero.
5. **`WireNameRegistryTest` checks the other direction**: every declared constant must be reachable
   from some public builder method (driven reflectively), names are unique across both CLI
   registries, and every `OCP_OVERLY_CONCRETE_PARAMETER` suppression still names a real enum-valued
   setter.

`JsonParameters` additionally enforces that **every stored value is exactly one well-formed JSON
value**, checked on write. That is what stops a caller-supplied fragment
(`withJsonSchema`/`withResponseFormat`/`withStreamOptions`/`withMessagesJson`/`withToolsJson`) from
injecting sibling fields into a request body — a demonstrated defect, with duplicate keys resolving
last-wins in the native parser. Note `FAIL_ON_TRAILING_TOKENS` is load-bearing: plain `readTree`
parses the first value and ignores the rest, which would silently truncate such a fragment instead of
rejecting it.

The full record — which names were dead when, the fork-point archaeology, and the injection
reproducer — is in
[`docs/history/parameter-wire-surface.md`](docs/history/parameter-wire-surface.md).

## Local llama.cpp source patches (`patches/`)

The fetched llama.cpp source is patched before it compiles, via a generic mechanism:

- **`llama/patches/`** — drop any number of `*.patch` / `*.diff` files here. They are applied
  in **filename order** (use a numeric prefix, e.g. `0001-`, `0002-`), so keep them independent or
  ordered. Each must be a `git apply`-compatible unified diff with paths relative to the llama.cpp
  source root (`a/common/arg.cpp` / `b/common/arg.cpp`, i.e. `-p1`).
- **`llama/cmake/apply-llama-patches.cmake`** — the applier. Cross-platform (`cmake -P`, so identical on
  Linux/macOS/Windows), **idempotent** and **fail-loud** (a patch that no longer applies aborts the
  configure — a stale patch can't be silently dropped from a release build). Idempotency comes from
  a **stamp file** (`<llama.cpp-src>/.jllama-patches-applied`, recording the checked-out llama.cpp
  commit plus each patch's SHA-256) combined with git's clean/dirty state, not from per-patch
  probing: a **clean** source tree means nothing is applied yet (fresh fetch, or a re-checkout after
  a version bump) so everything is applied forward and the stamp written; a **dirty** tree is
  already patched, and the reconfigure is a no-op when the stamp matches this exact commit + patch
  set, or aborts with a "configure into a fresh build directory" message when it does not.
  A per-patch `git apply --reverse --check` cannot do this — `--check` never mutates the tree, so an
  earlier patch whose region a later one rewrote (`0001` vs `0006`/`0007` in
  `tools/server/server.cpp`) always reverse-checks as "not applied", and the forward re-apply then
  aborted every reconfigure of an existing build dir with a misleading "does not apply cleanly".
  A source tree supplied via `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<path>` that is not a git work
  tree has neither oracle and falls back to the old per-patch path (same caveat as before).
- **`llama/CMakeLists.txt`** — wired as the llama.cpp `FetchContent_Declare(... PATCH_COMMAND ...)`, so it
  runs for **every** C++ build (all CI jobs *and* local `cmake -B build`) from one place — no
  per-build-step plumbing.

**On a llama.cpp version bump, every patch must still apply** — if a bump shifts the patched code,
the configure fails with an "does not apply cleanly" error; refresh the diff against the new source
and recommit. Treat `patches/` as part of the upgrade checklist below.

Current patches:

| Patch | Fixes |
|-------|-------|
| `0001-win32-arg-parse-embed-guard.patch` | Windows JNI regression from llama.cpp **#24779** (introduced b9739): on Windows `common_params_parse` re-derived argv from the **process** command line (`GetCommandLineW`) and adopted it, so an embedded/JNI caller (`java.exe`) lost its `--model …` args → "Failed to parse model parameters". b9789 narrowed the unconditional override to a **count-guard** (`if (static_cast<int>(utf8.buf.size()) == argc) { argv = utf8.ptrs.data(); }`), but that is exactly the variant the project already found breaks its Windows server-integration tests (when the embedded argv length coincides with `java.exe`'s). The patch carries the **complete upstream change** (so it can be submitted to llama.cpp verbatim and then dropped here): **(1)** `common_params_parse` parses **exactly the argv it is given** (no `GetCommandLineW` magic) and a new `common_params_parse_main()` wrapper holds the UTF-8 recovery for the standalone tools' `main()` (`common/arg.{cpp,h}`); **(2)** the **~34 standalone `main()` call sites** (every `common_params_parse(argc, argv, …)` across `tools/*`, `examples/*` and the `tests/*` programs) flip to `common_params_parse_main()`; **(3)** a `tests/test-arg-parser.cpp` regression case pins that `common_params_parse` honors a caller-supplied argv. The embedded caller (`jllama.cpp`) keeps calling `common_params_parse` and is never overridden. **Our subproject build compiles only the `arg.{cpp,h}` core** — `LLAMA_BUILD_TOOLS`/`LLAMA_BUILD_TESTS` are OFF for a FetchContent subproject — so the flips + test are applied-but-not-compiled here; they were validated via a one-off `-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=ON` build (the new test compiles and its asserts pass; `test-arg-parser`'s only red there is the live `ggml.ai` download check, which is sandbox-network, not the patch). Because it spans **35 files** it must be refreshed on every llama.cpp bump (the applier fails loud). **Refreshed at the b10679 bump:** upstream rewrote `tests/test-save-load-state.cpp`'s `main()` to take a `--models DIR` option, which it strips itself into a `filtered_argv` before calling `common_params_parse(fargc, filtered_argv.data(), …)`. That call site therefore stopped qualifying for the `_main()` flip — by this patch's own rule a caller that builds its own argv must use `common_params_parse` directly, so its argv is kept — and the hunk was **dropped** rather than refreshed (37 → 36 files). Caveat for whoever submits this upstream: that `main()` now filters a possibly-mojibake Windows argv *before* any UTF-8 recovery, so the fully correct upstream form there is recover-then-filter, not a one-line flip. It is out of scope for the downstream carry because `LLAMA_BUILD_TESTS` is OFF here, so the file is never compiled. **Refreshed the same way at the b11236 bump:** upstream #29426 rewrote `tests/test-recurrent-state-rollback.cpp`'s `main()` into the identical `--models DIR` / `filtered_argv` shape, so that hunk was dropped too (36 → 35 files); the same recover-then-filter caveat applies to it. **Still required at b10679, verified rather than assumed:** `common_params_parse` in pristine `b10679:common/arg.cpp` still carries the `#ifdef _WIN32` count-guarded `argv = utf8.ptrs.data()` override, and `common_params_parse_main` appears nowhere in `b10679:common/arg.h` — upstream has not adopted the fix. The upstream-facing write-up, including a standalone reproducer that makes llama.cpp's own `test-arg-parser` fail on unmodified `master`, lives in [docs/upstream-investigation-win32-argv-substitution.md](docs/upstream-investigation-win32-argv-substitution.md). **Reported upstream as [ggml-org/llama.cpp#26416](https://github.com/ggml-org/llama.cpp/issues/26416)** (2026-08-01, label `bug-unconfirmed`, first bad commit `508a475`); the issue asks which of the two directions the maintainers prefer before a PR is opened, so this patch stays downstream until they answer. |
| `0002-server-preserve-caller-load-progress-callback.patch` | Load-progress-callback regression introduced in llama.cpp **b9789**: `server_context::load_model` (`tools/server/server-context.cpp`) now **unconditionally** installs the server's own load-progress reporter on `params_base.load_progress_callback` immediately before `common_init_from_params`, clobbering any callback the embedding caller already set. libjllama's `LoadProgressCallback` feature wires `common_params.load_progress_callback` to a JNI trampoline *before* calling `load_model`, so the bump silently killed it — `LoadProgressCallbackTest` saw zero progress updates and the abort-on-`false` path never threw. The patch guards the assignment with `if (params_base.load_progress_callback == nullptr)`, so the server installs its own reporter **only when the caller hasn't** — a caller-supplied callback survives and fires during load. Standalone `llama-server` (no caller callback, so the field is null) is unaffected. Same JNI-vs-standalone divergence class as `0001`. **The guard is `== nullptr || == load_progress_callback`, and the second disjunct must never be dropped:** `load_progress_text` is a **local** of `load_model()`, and upstream re-assigns both fields on every call so the `user_data` always points at the current frame. `load_model()` runs a **second** time when resuming from the sleeping state (`--sleep-idle-seconds`), and by then `params_base` holds *our own* callback from the first load — a bare nullptr check skips the re-assignment and leaves `user_data` pointing into a **dead stack frame**, which segfaults inside `load_progress_callback()` on the first request after an idle window. That was a latent defect in this patch from the day it was written; only a second `load_model()` can reach it, and nothing exercised sleep until `IdleSleepWakeIntegrationTest` was added. |
| `0003-pr22393-server-add-slot-prompt-similarity-getter-setter.patch` | **Upstream-PR carry** of [ggml-org/llama.cpp#22393](https://github.com/ggml-org/llama.cpp/pull/22393) ("server : add slot_prompt_similarity getter/setter"). Purely additive: adds `server_context::get_slot_prompt_similarity()` / `set_slot_prompt_similarity(float)` (`tools/server/server-context.{cpp,h}`) so an embedding/JNI caller can query and tune the slot-selection threshold at runtime without reloading the model. Verbatim copy of the PR, which **upstream closed without merging** (rejected as exposing unsafe internal state — see the patch header). Carried permanently; it will not be droppable via a version bump. |
| `0007-server-attach-http-frontend.patch` | **Adds `llama_server_attach(argc, argv, server_context&)`** so the `NativeServer` *attach mode* can serve an **already-loaded `LlamaModel`** over the upstream HTTP frontend — no second model load, no `start_loop()`; the LlamaModel's worker keeps driving the shared `server_context` and the HTTP routes post tasks to its queue (the queue is the synchronization point). Mechanically: (1) extracts the **pure core route table** (`health` … `slots`) out of `llama_server()` into `static void llama_server_register_common_routes(ctx_http, routes)` (shared, so the two entry points cannot drift on the core endpoint set). **Scope note (narrowed at the b10154 bump):** the helper deliberately carries **only** the stable, state-independent route table — **not** the resumable-streaming routes (their handlers differ between router / non-router), the GCP-compat shim, or the experimental **CORS-proxy / MCP-server / built-in-tools** wiring. b10154 (upstream MCP-server support) moved the streaming routes into the middle of that block and coupled tools/CORS to a per-call `server_mcp mcp_mgr` lifecycle, so the earlier contiguous "route-table + CORS-proxy + tools" extraction is no longer possible; `llama_server()` keeps all of that inline, **byte-identical to upstream b10154** (only the route-table block is factored out). (2) adds `llama_server_attach`, which parses only the HTTP-side argv via `common_params_parse`, starts the stream-session GC + `server_http_context`, registers the common route table, the **non-router** resumable-streaming handlers (upstream b10154 paths `/v1/stream` GET/DEL + `/v1/streams/lookup` POST), the GCP-compat shim, and **403 "disabled" stubs for `/cors-proxy` + `/tools`** (attach mode does not wire the experimental CORS-proxy / MCP / built-in-tools host — those belong to a full `llama-server`, not an embedded model), marks ready immediately (model already loaded), and blocks on the HTTP thread until `llama_server_request_shutdown()` — never calling `common_init()`, backend init, `ctx_server.terminate()` or `llama_backend_free()` (the embedding caller owns those). Applies after `0001`+`0006` (same file); closes the "NativeServer — reuse an already-loaded LlamaModel" TODO. Upstream-submittable ("server: let embedding callers attach the HTTP frontend to an existing server_context"). **Refreshed at the b10519 bump:** upstream #26347 dropped the API key from the `/models` + `/v1/models` public-endpoint set and deleted the two trailing `// public endpoint (no API key check)` comments on those route registrations. Those two lines sit inside this patch's route-table removal block, so `git apply` failed ("patch does not apply", `server.cpp:258`) at **every** tag from b10519 on; the fix was to drop the now-wrong comment from all four affected lines (2 on the `-` side, 2 in the extracted helper on the `+` side), keeping the helper byte-identical to the block it replaces. **This is the invariant to re-check on every bump:** the `+` side of `llama_server_register_common_routes()` must stay a verbatim copy of the route table it factors out of `llama_server()`. **Refreshed at the b11104 bump** (upstream #28690, multi-address `--host`): `server_http_context` lost its single `thread` and `listening_address` members in favour of `join()` and a `listening_addresses` vector, one listener thread per bound address. The patch still *applied* cleanly there — only its own `+` lines named the removed members — so the applier could not see it; `llama_server_attach` now logs every address and blocks in `ctx_http.join()`, exactly as upstream's `llama_server()` does. **Refreshed at the b11361 bump** (#29818 adds `POST /v1/systemone` to the route table: one line, added on both sides so the helper stays verbatim) **and at b11401** (#29895 adds a `server_child &` overload of `llama_server()` beside the declarations and a `server_child child;` at the top of the argv entry point, the context of `0007`'s first two hunks; only context moved, the `+`/`-` lines are unchanged). Attach mode builds its own `server_routes`, whose sleep callback outlives it -- on file in `TODO.md`. |
| `0008-server-models-worker-cmd-override.patch` | **Makes router mode usable in-JVM.** The router (`server-models.cpp`) spawns each model worker by re-executing its own binary (`get_server_exec_path()` = `/proc/self/exe` & friends) — inside a JVM that binary is `java`, not a llama-server, so embedded router workers could never start. The patch adds env `LLAMA_SERVER_WORKER_CMD` (whitespace-split; read in `server_model_meta::update_args`) which replaces only the leading binary-path token of the rendered worker args, letting an embedding host relaunch workers through its own bootstrap — e.g. `java -cp app.jar net.ladenthin.llama.server.NativeServer` (each worker is then a fresh JVM running the classic single-model `NativeServer`). Exposed in Java as `NativeServer.setWorkerCommand(String...)` (JNI `setenv`); exercised by `RouterModeIntegrationTest` (Linux CI). Upstream-submittable (also useful for containerized/wrapped deployments). **Refreshed at the b11429 bump:** #29987 (untagged `4d60b4d08`, the commit before b11429) moved `server_model_meta::update_args` and changed the function after it (`update_caps(const common_params &)`), the hunk's trailing context; the `+` lines are unchanged. Since b11401 (#29895) a worker is a router *child* whose stdout carries the state commands, which a JVM worker's early `System.out` line now trips (cosmetic, `TODO.md`). |
| `0006-server-embed-native-server-jni.patch` | **Makes `server.cpp`'s `llama_server` embeddable in the JVM** so the `NativeServer` JNI bridge can run the full upstream HTTP server (WebUI included) inside `libjllama` — see "Two server modes" below. b9870 already exposes `int llama_server(int, char**)` (non-static; no `main` in the file), so the patch only adds embedded-mode support: (1) a `g_llama_server_embedded` flag + `llama_server_set_embedded()` / `llama_server_request_shutdown()` (declared in the committed `src/main/cpp/native_server_bridge.h`); (2) skips installing the process-wide SIGINT/SIGTERM handlers when embedded (they would hijack the JVM's); (3) in embedded mode parses the **forwarded** argv via `common_params_parse` instead of `common_params_parse_main` (whose `GetCommandLineW` recovery would pick up `java.exe`'s command line — the same Windows class of bug `0001` fixes). `llama_server_request_shutdown()` mirrors the SIGTERM path (invokes the installed `shutdown_handler` → `ctx_server.terminate()` unblocks `start_loop()`), giving JNI an out-of-band stop since `ctx_server` is loop-local. **The handler is guarded, and both guards are load-bearing:** upstream's `shutdown_handler` lambdas capture `llama_server()`'s locals (`ctx_http`, `models_routes`, `mcp_mgr`, `ctx_server`) by reference and are never cleared, while `native_server.cpp` signals *repeatedly* until the worker reports it finished (a stop issued before the handler is installed, or before `start_loop()` begins, would otherwise be lost). A request landing after `llama_server()` returned therefore ran the lambda over destroyed objects — a `SIGSEGV` in `server_http_context::stop()` in `RouterModeIntegrationTest.tearDown` (run 36485159455), with router mode's `clean_up()` (unloading workers) making the window wide. The patch (1) takes `g_shutdown_handler_mutex` for every read, write and invocation of the handler (the caller is on another thread) and (2) declares a `llama_server_shutdown_handler_reset` **after** the captured locals, so the handler is cleared before any of them is destroyed; because invocation holds the same lock, that reset waits for an invocation in progress. `0007`'s `llama_server_attach` carries the same guard. Runnable guard: `src/test/cpp/test_native_server_shutdown.cpp`. Applies **after `0001`** (which flips this call site to `common_params_parse_main`), so its context is the post-`0001` tree; regenerate against `0001`+source on a bump. Only touches `tools/server/server.cpp`. |
| `0012-model-guard-zero-split-sum-and-name-the-device-index.patch` | **A GPU that reports zero free memory makes every model load fail with the unactionable `error loading model: vector`.** `llama_model_base::load_tensors` (`src/llama-model.cpp`) weights the per-device layer split by `ggml_backend_dev_memory()`'s `free`, then normalises: `splits[i] /= split_sum`. With a single device reporting `free == 0` that is `0/0` → **NaN** in every split point; NaN compares false against everything, so the `std::upper_bound` below returns the end iterator, `layer_gpu == n_devices()`, and `devices.at(layer_gpu)` throws `std::out_of_range` — whose libc++ `what()` is the bare string `"vector"`, which `llama.cpp`'s `catch (const std::exception &)` prints verbatim. Upstream's `free == 0 && total == 0` host-memory fallback does **not** fire, because `total` is `recommendedMaxWorkingSetSize` and is non-zero. **Reachable since b10618..b10797**: upstream `8c0b9cd04` ("metal : fix memory query under low-memory conditions", [#27701](https://github.com/ggml-org/llama.cpp/pull/27701)) changed `ggml-metal-device.m` to `*free = *total > cur ? *total - cur : 0`; before that clamp an over-committed device (`currentAllocatedSize > recommendedMaxWorkingSetSize`) *underflowed* to a huge `size_t`, which normalised fine, so the same precondition was harmless. That is why the `Java Tests macOS …` jobs went red at the b10792→b10797 step while every Linux/Windows job stayed green — **and why only a GPU build can fail this way at all**: `act_gpu_layers` is `devices.empty() ? 0 : …`, so with no GPU backend `devices` is empty, every layer returns early on `cpu_dev`, and the `.at()` line is unreachable. **Shape:** the two blocks are lifted out of `load_tensors` into free functions declared in `src/llama-model.h`, purely so they can be driven by a test — the failing state needs a real over-committed GPU and cannot be arranged through any public API. `llama_model_splits_normalize()` carries **the fix**: on `split_sum == 0` it `LLAMA_LOG_WARN`s and falls back to an even split (`splits[i] = float(i+1)/splits.size()`), the only neutral choice when no device can be preferred and exactly right for a single device. `llama_model_splits_select_device()` carries **the diagnostic**: it bounds-checks the index and throws a `std::runtime_error` naming the function, the offloaded layer, the device index, the split-point count **and the split points themselves** — with NaN splits that message prints `nan` and names the cause outright, which is precisely what was missing when this had to be diagnosed by reading source. **A second, backend-independent trigger reaches the same line**, found while writing this up and verified against the unfixed library: `--tensor-split` values are parsed with `std::stof` and never range-checked (`common/arg.cpp`), so `-ts 1,-1` cancels out, `split_sum` is 0 again, the split points become `[inf, -nan]`, and every layer maps one past the last device — on CUDA, Vulkan or ROCm just as much as on Metal, with no memory pressure involved. That is what makes this an ordinary upstream defect rather than a Metal edge case, and the warning names both causes rather than only the memory one. Also adds upstream `tests/test-model-split.cpp` (5 cases in upstream's `testing.h` style) + its `llama_build_and_test` registration. Touches `src/llama-model.{cpp,h}`, `tests/test-model-split.cpp` and `tests/CMakeLists.txt` — **none** of which any other patch touches, so it is independent of all of them. Upstream-submittable ("model: fall back to an even split when no device reports free memory"); **not yet filed upstream**. **Runnable guard: `src/test/cpp/test_model_split.cpp`** — a FetchContent subproject builds with `LLAMA_BUILD_TESTS=OFF`, so the upstream test above is applied-but-never-compiled here (same as `0001`'s test). That file drives the same two functions from `jllama_test`, which runs on **every** platform in `C++ Tests`, so a bump that drops this patch fails the build at link time everywhere instead of surfacing as one red macOS Java job. **Verification limit — read before assuming this can be dropped:** the *failing path* still cannot be reached without a GPU backend, so the guard pins the arithmetic (what actually broke), not the end-to-end load; the end-to-end proof is the macOS CI job. On a bump, re-check whether upstream added its own `split_sum == 0` guard (grep `split_sum` in `src/llama-model.cpp`) and **drop this patch rather than refreshing it** if they did — the fail-loud applier detects "does not apply", never "upstream already fixed this". |
| `0014-common-log-callback-sink.patch` | **Gives `common_log` a callback sink: `common_log_set_callback(log, cb, user_data)` (`common/log.{h,cpp}`).** This is what `LlamaModel.setLogger` hooks. Before it, the Java logger was a `llama_log_set()` callback, which has two holes, both found while chasing `slot print_timing` lines interleaving with the Atmosphere agent's streamed answer: **(a)** every model load runs `common_init()`, which re-points `llama_log_set()` at `common_log_default_callback` (`common.cpp:394`), so `setLogger(…)` *before* `new LlamaModel(…)` silently lost the callback; **(b)** the server's own `SRV_*`/`SLT_*` macros are `LOG_INF` and write straight into `common_log`, which `llama_log_set()` never carried, so the per-request `slot …`/`srv …` lines could not be routed to Java at all (the reason `LlamaModelTest#testLogText/JSON` sat `@Disabled` for years). `common_log` upstream offers file, colors, prefix, timestamps, verbosity and JSONL but no hook. The patch adds one: while a callback is set, the worker thread hands every entry to it **instead of** printing to stdout/stderr (a `--log-file` still receives them); the callback gets the bare formatted message (no prefix/timestamp/colors) with the `ggml_log_callback` signature; swapping the callback pauses the worker first, so queued entries reach the *previous* sink (which is what makes `setLogger(format, null)` a synchronous drain). With the sink, `common_init()`'s `llama_log_set()` reset is harmless — it points at the default callback that feeds `common_log`, i.e. exactly the path into the sink — so the ordering problem (a) disappears without any re-install logic, and the `srv`/`slot` lines (b) arrive because they are `common_log` entries. `jllama.cpp`'s `setLogger` therefore sets `common_log_set_callback(common_log_main(), trampoline)` **plus** `llama_log_set(common_log_default_callback)` (so llama/ggml lines feed `common_log` even before the first load). Two consequences to know: the callback runs on `common_log`'s **worker thread**, a plain `std::thread` llama.cpp re-creates on every pause/resume and never attaches to the JVM — the trampoline attaches per call and detaches again (`get_jni_env_attaching`; a thread that exits while attached leaks a `JavaThread`, and this thread is not ours — the leak-free choice, not the cheapest: each attach creates a `java.lang.Thread` object, so a `thread_local` guard that detaches once at thread exit is the optimisation on file in `TODO.md`), `setLogger` must call `common_log_set_callback` **outside** `g_log_mutex`, because the pause joins the worker, which needs that mutex to read the callback, and `setLogger` callers are serialized by a **separate** `g_set_logger_mutex`: two unserialized swaps race on the worker's `std::thread` (one joins it while the other assigns a fresh thread over the still-joinable object = `std::terminate`, the whole JVM), which `LlamaLoggerTest#concurrentSetLoggerCallsDoNotRaceOnTheLogWorker` reproduced before the mutex existed. Two caveats the Javadoc carries: a caller must not hold a lock the *previous* callback needs (the drain runs it on the worker while the caller waits), and the verbosity threshold is process-wide and reset by **every** load (`common_params_parse` ends with `common_log_set_verbosity_thold(params.verbosity)`, default 3), so a load without `-lv` puts it back to 3 — a review assumed the opposite, and `LlamaLoggerTest#verbosityThresholdIsProcessWideAndEveryLoadSetsIt` now pins the measured behaviour. And the verbosity threshold applies *before* the sink: at the default (`3`) the Java logger sees errors, warnings and the server's INFO lines, while llama/ggml INFO lines (`common_log_get_verbosity` maps them to TRACE = 4) arrive only from `setLogVerbosity(4)` on — the same filtering the console gets, and a behaviour change for consumers who captured the unfiltered `llama_log_set()` stream before. **Runnable guards:** `src/test/cpp/test_common_log_callback.cpp` (6 tests over a private `common_log_init()` instance: delivery, bare text under prefix+timestamps, clear, swap-drains-to-old-sink, file kept, levels pass through) links the function on every platform, so a bump that drops the patch reds `C++ Tests` at link time; `LlamaLoggerTest` (model-free, needs only `libjllama`: a logger set before a deliberately failing load on a non-GGUF file sees the `srv … loading model` INFO line and llama's ERROR line, in TEXT and JSON) and the re-enabled `LlamaModelTest#testLogText/testLogJSON` plus `#testLoggerSetBeforeLoadSurvivesTheLoad` (vocab-only load) cover the Java side. Upstream-submittable ("common : add a callback sink to common_log for embedding hosts"); **not yet filed upstream**. Touches only `common/log.{h,cpp}`, which no other patch touches. **Refreshed at the b11401 bump** (#29895 adds a `colors` member next to the sink's members and a `get_colors()` next to `set_callback()`; only context moved, the `+`/`-` lines are unchanged, and upstream still has no hook -- it added `common_log_get_colors` only). **On a bump, check whether upstream added a hook of its own (grep `callback` in `common/log.h`) and, if so, DROP this patch and port `setLogger` to theirs rather than refreshing it.** |
| `0015-rpc-embeddable-client-and-stoppable-server.patch` | **Makes ggml-rpc usable inside a JVM** (see "RPC backend" below). Upstream treats every client-side RPC problem as `GGML_ABORT` — a malformed endpoint, a server that is not running, a failed handshake — which in a JVM kills the application over a typo in `--rpc`. **(1)** The *registration* path reports failure instead: `rpc_dispatcher::start()` returns `false`, `try_get_dispatcher()` `nullptr`, `ggml_backend_rpc_get_device_count()` `0`, `ggml_backend_rpc_add_server()` `nullptr`, and `add_rpc_devices()` (`common/arg.cpp`) throws `std::invalid_argument` naming the server where it used to register `nullptr` and silently run without it. Every path *after* registration keeps the original contract through `get_dispatcher()`, which still aborts — a server that vanishes mid-inference stays fatal (TODO). **(2)** `add_server()`'s cache hit is re-checked (a stopped server used to pass registration and abort on the first tensor upload), and `ggml_backend_rpc_get_device_memory()` of a gone server returns 0/0 instead of aborting — RPC devices stay in ggml's process-wide registry forever, and `common_init()` queries the memory of **every** registered device at `-lv 4`, so a stopped server would otherwise take down an unrelated later load. **(3)** The server becomes stoppable: `ggml_backend_rpc_stop_server()` (disconnects the client being served, wakes `accept()` with a loopback connection — the one portable way) and `ggml_backend_rpc_server_listening()`; the loop now reaches its cleanup and frees its backends on every early return, but deliberately **not** `rpc_transport_shutdown()`, whose `WSACleanup()` would invalidate every other RPC socket in the process. **(4)** transport: `MSG_NOSIGNAL`/`SO_NOSIGPIPE`, `socket_t::shutdown()`, and the fds `connect()`/`create_server()`/`accept()` leaked on their error paths are closed. Touches `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/{ggml-rpc.cpp,transport.cpp,transport.h}` and one function in `common/arg.cpp` (`add_rpc_devices`, away from `0001`'s `common_params_parse` hunks). **Runnable guard: `src/test/cpp/test_rpc.cpp`** — links both new functions (a bump that drops the patch fails `C++ Tests` at link time on every platform) and drives registration, stop and the memory query over loopback. Upstream-submittable; **not yet filed upstream**. On a bump, check whether upstream grew its own stop function or non-aborting registration (grep `stop_server` / `GGML_ABORT("Failed to connect` in `ggml-rpc.cpp`). **Refreshed at the b11450 bump** (#26610, `-sm tensor` over RPC, ~800 lines in `ggml-rpc.cpp`, protocol 7 → 8): three hunks lost their context -- new dispatcher members beside `start()`, and new `get_proc_address` names before `return NULL` (the two new names now sit after `ggml_backend_comm_allreduce_tensor`); the `+`/`-` lines are unchanged. The server-to-server comm #26610 adds (a listener on `0.0.0.0`, an `accept()` a stop does not wake) is on file in `TODO.md`. |
| `0017-ggml-cpu-x86-prefetch-byte-offset-and-msvc-type.patch` | **Makes `GGML_CPU_ALL_VARIANTS=ON` buildable with plain clang on Windows, and fixes a prefetch address that is wrong everywhere.** The four `_mm_prefetch` calls in `ggml/src/ggml-cpu/arch/x86/quants.c` (the SSSE3-without-AVX branch of `ggml_vec_dot_q4_0_q8_0`, and the only `_mm_prefetch` calls in the whole ggml tree) carry two independent defects; both are fixed by routing the address through one new `static inline` helper in the new header `ggml/src/ggml-cpu/arch/x86/prefetch.h`, which is included relatively, so **no CMake change is needed**. **(1) Type -- fatal only on a current clang:** MSVC declares the intrinsic as `void _mm_prefetch(char const *, int)`; clang's own `xmmintrin.h` defines a casting macro but only `#ifndef _MSC_VER`, so on a `*-windows-msvc` target clang leaves the declaration to Microsoft and a typed pointer is `-Wincompatible-pointer-types`. **clang 22 promoted that to an error by default** ([llvm-project #157364](https://github.com/llvm/llvm-project/pull/157364); the [22.1.0 release notes](https://releases.llvm.org/22.1.0/tools/clang/docs/ReleaseNotes.html) add that `-Wno-error=incompatible-pointer-types` downgrades it) -- *not* clang 16, which promoted only `-Wincompatible-FUNCTION-pointer-types`; an earlier version of this row said otherwise. Measured on the real file at `-msse4.2`, unpatched → patched: clang **20.1.8** 4 warnings → 0/0, clang **22.1.3** 4 errors → 0/0, clang **23.1.3** 4 errors → 0/0. Plain clang is exactly what `cmake/x64-windows-llvm.cmake` selects -- and plain clang is the only way to get all **14** CPU variants, because `clang-cl` makes CMake's `MSVC` true and the `if (NOT MSVC)` block in `ggml/src/CMakeLists.txt` then drops `zen4`, `cooperlake`, `sapphirerapids`, `ivybridge` and `piledriver`. **(2) Arithmetic:** `&x[ib] + sizeof(block_q4_0)` adds `sizeof()` *elements*, not bytes -- +324 and +1156 instead of +18 and +34 (the sizes `ggml-common.h` pins by `static_assert`), and +648 instead of +36 for the `2 *` pair. That half is wrong on every platform, including where the call compiles. **Measured:** no throughput difference (see the test-table row), so this is a correctness and portability fix, not a performance one. **Upstream-submittable and meant to be submitted.** **Why upstream's own build is green is measured, not an open question:** their `windows-cpu` release job builds exactly this with plain clang, `GGML_CPU_ALL_VARIANTS=ON` and no suppressing flag (`CMAKE_ARGS` carries only build-target toggles), and ships `ggml-cpu-sse42.dll` -- because the `windows-2025-vs2026` image has **LLVM 20.1.8** on `PATH`, where these are warnings. That is narrower than it looks: the same image also ships **clang 22.1.3** inside Visual Studio 2026 (`VC/Tools/Llvm/x64`), which *fails* on the unpatched file -- upstream reaches the working compiler only because `C:\Program Files/LLVM/bin` comes first on `PATH`. It is **not** a header difference: all three MSVC toolsets on that image (14.29.30133, 14.44.35207, 14.51.36231) declare it identically as `extern void _mm_prefetch(char const*_A, int _Sel);` -- that was the first hypothesis and `.github/workflows/probe-msvc-prefetch.yml` disproved it. So this patch is **ahead of** upstream rather than diverging: their build breaks the moment the image ships a clang that makes the diagnostic fatal, and until then they compile four warnings and a prefetch address 18x/34x too far. **Runnable guard: `src/test/cpp/test_prefetch.cpp`.** On a bump, run `git grep -n _mm_prefetch ggml/src/ggml-cpu/arch/x86/quants.c` at the new tag: if the call sites already cast **before** the addition, **drop this patch rather than refreshing it**. |
| `0016-model-kolibri1.patch` | **Adds Aleph Alpha's Kolibri-1 (architecture `kolibri1`) ahead of upstream** -- a 78B German/English reasoning MoE (3.46B active) that upstream does not support yet ([ggml-org/llama.cpp#29922](https://github.com/ggml-org/llama.cpp/issues/29922), no PR at the time). A **temporary carry, not a fix**: drop it the moment upstream registers the architecture (`git grep -n kolibri src/llama-arch.cpp` at the new tag), and keep `test_kolibri1.cpp` -- its numerical comparisons must then pass against upstream's implementation, while its GGUF-format rows follow whatever format upstream's converter fixes (see the test-table row). **Architecture** (Aleph Alpha's vLLM reference, `aleph_alpha_inference/kolibri1.py` at `049a6a7`): GQA with per-head q/k RMSNorm; `layer_types` interleaves sliding-window layers (NEOX RoPE) with full-attention layers that have **no** positional encoding; sandwich norms around attention and MoE; every layer routed MoE + one ungated shared expert; the router selects top-k on **`logits + expert_bias`** and weights by the **unbiased `sigmoid(logits)`** -- not DeepSeek-V3's sigmoid router (selection on `sigmoid(logits) + bias`), which picks other experts as soon as the bias is non-zero. The graph therefore builds the selection itself and passes it to `build_moe_ffn` (`selected_experts_in`), leaving upstream's shared MoE code untouched. **Derived from two community ports, which are incompatible with each other's GGUFs** -- Eliasfpv28's AFMoE-based one (base b11378, `expert_gating_func` 2, pre-tokenizer `qwen2`) and the Qwen3-MoE-based one behind Hob-forge's GGUFs (base b11381, as carried in [mjwsolo/localcode#101](https://github.com/mjwsolo/localcode/pull/101): gating 5, pre-tokenizer `kolibri1`, rejects gating 2). This patch **loads both dialects**: gating 2, 5 or absent (all mean the one router), `kolibri1` mapped to the Qwen2 pre-tokenizer in `llama-vocab.cpp`, `expert_weights_norm` / `expert_shared_count` / `output.weight` optional. Every source, with license, is listed in the patch header; `REUSE.toml` annotates the file `MIT AND Apache-2.0` (Eliasfpv28's additions are Apache-2.0). Touches `src/llama-arch.{h,cpp}`, `src/llama-model.cpp` (mapping + NEOX rope list), `src/llama-vocab.cpp` (one name), `src/models/models.h` and the new `src/models/kolibri1.cpp`; no converter. **Runnable guard: `src/test/cpp/test_kolibri1.cpp`** (writes tiny random GGUFs in both dialects and compares every logit with an independent double-precision reference). **Not verified here:** the real 78B model (HuggingFace is unreachable from the sessions this was written in, and the smallest GGUF is 28.6 GB) and GPU backends -- the community ports report real-model checks (top-1 67/67 against a float64 reference for Q8_0, tool calls, German text). |

**Dropped patches** (`0004`, `0005`, `0009`, `0010`, `0011`, `0013` -- upstream fixed each defect
itself) are recorded in [`docs/history/dropped-llama-patches.md`](docs/history/dropped-llama-patches.md):
why each went, and the runnable guards that stayed behind re-pointed at upstream's contract. Read the
matching note before reintroducing a patch for the same defect. **The standing drop-check on every
bump:** the applier detects "does not apply", never "upstream already fixed this" -- for each
remaining patch ask whether the *defect* is still there, not only whether the patch still applies.

## RPC backend: `--rpc` client and the in-JVM `RpcServer`

llama.cpp's RPC backend (`ggml-rpc`) is compiled into **every** natives jar — CPU and GPU — with `GGML_RPC=ON` forced in `llama/CMakeLists.txt`. It stays **one** `jllama`
library: `ggml_add_backend_library` makes `ggml-rpc` a static library linked into `ggml`, and its
registration is compiled in (`GGML_USE_RPC`) -- except in the two Linux CPU-variant builds
(`GGML_BACKEND_DL`, see "CPU variants"), where `libggml-rpc.so` is a module next to the library that
`JNI_OnLoad` loads with the CPU modules; the entry points are resolved through
`ggml_backend_reg_get_proc_address` in every build for that reason. Client and server live in the same file, so the flag brings both.

**No new runtime dependency — and this is enforced, not assumed.** The transport is plain TCP:
BSD sockets from libc/libSystem/bionic, Winsock on Windows (`jllama.dll` already imported
`WS2_32.dll` through cpp-httplib). `GGML_RPC_RDMA` is forced **OFF**: upstream turns it on whenever
the *build host* has `libibverbs` (Linux) or `librdma` (Apple), which would make `libjllama.so` need
`libibverbs.so.1` at load time and fail on every machine without rdma-core; it cannot be linked
statically in a useful way either (it `dlopen`s its hardware providers), and upstream's own CMake
comment says the Apple weak link does not survive a static `ggml-rpc`. **`.github/verify-native-deps.py`**
(the `package` job; the code and its tests are `.github/buildcheck/nativedeps.py`) reads each shipped library's dependency list straight from the file — ELF
`DT_NEEDED`, PE import table including Windows arm64 (which binutils cannot read), Mach-O load
commands — and holds every CPU directory (`cpu`, `metal`, `msvc`) and the Android OpenCL build to an
exact per-directory allowlist (the other GPU directories only to a denylist, since they need their
vendor runtime). A new CPU
directory without an allowlist fails too. (That every build arrived is the merge step's check against
`natives.csv`.) **It found a pre-existing defect on its first run**: the macOS dylib links Homebrew's
`openssl@3` (`/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib` + `libcrypto`), so it does not load on a
Mac without that formula. Those two paths are in the allowlist marked as a known defect so the
check reports only *new* dependencies; the fix is on file in `TODO.md`.

**Three places every argv goes through (`src/main/cpp/rpc_support.hpp`).** ggml's backend registry
is process-wide and has no unregister, so an RPC server registered by one `--rpc` load stays a
device for the life of the JVM — and llama.cpp's default device selection puts RPC devices *first*.
`jllama::rpc::prepare_argv()` runs before `common_params_parse` in `LlamaModel`'s load and in
`NativeServer`'s start (attach mode parses only HTTP args and is exempt):
1. every `--rpc` endpoint is registered up front, and an unreachable one throws
   `std::invalid_argument` naming it → `LlamaException` (instead of upstream's generic parse failure);
2. if the registry holds an RPC device the argv did **not** ask for and the caller gave no
   `--device`/`-dev`, it appends `--device <requested RPC devices + local GPUs>` — the local part
   chosen exactly as llama.cpp's default (`src/llama.cpp`, `llama_model_load`) chooses: duplicates by
   `device_id` dropped, integrated GPUs only when there is no discrete one, `none` on a CPU-only host;
3. in the same situation the **multimodal projector's** device is pinned too (`--mmproj-device <local
   GPU>`, or `--no-mmproj-offload` without one), unless the caller named it: clip does not use the
   device list but takes the first registered device of type GPU, and RPC devices are type GPU and
   registered after every local backend — so on a GPU-less host a stale RPC device is exactly what
   it would pick;
4. otherwise the argv is untouched, so a process that never uses RPC sees upstream's behaviour.

**Entry points that build `common_params` themselves need the same guard, and it is easy to
forget.** `TextToSpeech` (`tts_engine.cpp`) and `LlamaTrainer` (`train_engine.cpp`) never parse an
argv, so `prepare_argv` never sees them; they call `jllama::rpc::exclude_stale_devices(params.devices)`
(same selection, applied to the null-terminated device vector; TTS also applies the mmproj answer
to its `mtmd_context_params`). Found in CI, not by reading: `RpcIntegrationTest` followed by
`TtsIntegrationTest` in one fork aborted the JVM in `get_dispatcher()` while `common_fit_params`
built a context over the gone server — `RpcIntegrationTest` now ends with a `TextToSpeech` load
that pins it (verified red with the call removed). **A new native entry point that calls
`common_init_from_params` must call it too.**
The selection rules are pure functions over a descriptor list (unit-tested with literals in
`test_rpc.cpp`); only `registered_devices()` / `register_server()` touch ggml. `RpcIntegrationTest`
pins the end-to-end reason: a model over RPC, the server stopped, then a load **without** `--rpc` in
the same JVM — which would offload to the stopped server and abort without step 2.

**`RpcServer` (root package) + `RpcServerNative` (`rpc_bridge.cpp`).** The native calls live in the
package-private `RpcServerNative`, behind the `RpcServer.Backend` seam, so `RpcServer` itself loads
without `libjllama` and its whole Java-side lifecycle is tested with a fake backend
(`RpcServerLifecycleTest`) — the analysis build (SonarCloud, no native library) sees it covered, and a
new lifecycle branch cannot hide behind "needs natives". The in-JVM `rpc-server`: ggml's blocking
`ggml_backend_rpc_start_server()` runs on a daemon Java thread, `close()` calls
`ggml_backend_rpc_stop_server()` (patch `0015`), and `start` waits for
`ggml_backend_rpc_server_listening()` so a bind failure is a `LlamaException` rather than a silently
dead thread. It serves every accelerator, else the CPU — **never an RPC device**
(`jllama::rpc::server_devices()`), which in a JVM that is also a client would forward to itself.
The overloads taking a device list (`--device` on the command line, upstream `rpc-server -d`) serve
named devices instead, resolved **before** the server thread starts so an unknown name is a
`LlamaException` listing the available ones. That is not a convenience: ggml-rpc's client answers
every `supports_op` with `true` (upstream TODO), so a served device that cannot run an operation
**aborts the process** — the macOS CI runners' paravirtual Metal GPU has no `MUL_MAT` and took the
first CI run down that way. **Every test that computes over RPC serves `CPU`** (C++ `loopback_server`,
`RpcIntegrationTest`); do not switch one back to the default choice.
**`RpcServer` is the first entry point that loads the library before `LlamaModel`, which exposed a
loader defect** (fixed in `LlamaLoader.runOnceOnThisThread`): `JNI_OnLoad`'s `GetFieldID` on
`LlamaModel` initializes that class, whose static block re-entered `LlamaLoader.initialize()` on the
loading thread — the lock is reentrant, so a second complete load ran while the first was inside
`System.load`, clearing the temp files and, with an all-backends jar, probing and extracting every
backend again over the library being loaded. The fat-jar RPC smoke timed out on it (the doubled
extraction of the CUDA/ROCm/SYCL libraries); it now waits 300 s like the other smoke and fails if the
backend is selected more than once. Only the *nested* call is skipped — later calls still run the body,
which `BackendLoadTest` relies on.
**Single instance per process** (ggml keeps the server state in globals). `startLocal` binds
loopback only; `startOnNetwork` is the explicit, warned opt-in, and binding needs an IPv4 literal
(the server uses `inet_addr`). Endpoints on the client side are `value.RpcEndpoint`
(`host:port`, IPv4 or host name — the transport has no IPv6, and upstream's parser would cut an IPv6
literal at its first colon). `main()` is the command-line form
(`java -cp <jar> net.ladenthin.llama.RpcServer --port 50052`).

**Tests, by layer.** C++ `test_rpc.cpp` (28, every platform in `C++ Tests`): selection rules plus the
real client/server over loopback. Java: `RpcEndpointTest`, `RpcServerOptionsTest` and `RpcServerLifecycleTest` (pure; the last one
drives start/stop, bind failure, start timeout, interrupt and the single-instance slot over a fake backend),
`RpcServerTest` (native, model-free: lifecycle, restart on the same port, single instance, bind
failure, unreachable `--rpc` load), `RpcIntegrationTest` (draft model: layers on the RPC server
proven from the load log's `model buffer size` line naming the endpoint, then the stale-server
load). CI: `.github/smoke-rpc-fatjar.sh` in the `linux-x86-64` row of `smoke-fatjar` runs **two JVMs from the release
asset** — `RpcServer` in one, the default `NativeServer` with `--rpc` in the other — and requires a
chat completion, the RPC model buffer in the log, an accepted client on the server, and a clean
non-SIGABRT failure naming the endpoint for a server nobody runs.

**One visible side effect, deliberate:** `llama_supports_gpu_offload()` is `true` on a CPU-only
build now (upstream ORs in `llama_supports_rpc()`), so `-ngl` no longer prints "no usable GPU found"
there and the load log shows "offloading N layers to GPU" lines even when every layer stays on the
CPU. Nothing else reads it (checked at b11222: `common/arg.cpp` warnings + one `llama-model.cpp` log
block), and upstream's own release binaries, which all build with `GGML_RPC=ON`, behave the same.

**Known limits (in `TODO.md`):** a server lost *mid-inference* still aborts the process (upstream has
no error path for it); Android needs the app's `INTERNET` permission even on loopback, which the AAR
deliberately does not request; the server serves one client at a time; no authentication or TLS.

## Qwen3-TTS via `mtmd_helper::gen_audio` (was: OuteTTS build-time extraction)

The `TextToSpeech` native pipeline (`tts_engine.{h,cpp}`) drives llama.cpp's upstream Qwen3-TTS
audio-generation pipeline directly through its public C++ API — `mtmd_helper::gen_audio`
(`tools/mtmd/mtmd-helper.h`) — rather than deriving/extracting anything from upstream source. `mtmd`
is already a `target_link_libraries(jllama ...)` dependency (vision/audio-input support), so this
needed no new CMake wiring at all: no generator, no build-time extraction, no hand-written interface
header to keep in sync. Loads a backbone text GGUF (a normal `llama_model`) plus an mmproj GGUF
(speaker encoder + code predictor + code2wav decoder, all bundled in one file by upstream's
`conversion/qwen3tts.py`), and drives the streaming API: `mtmd_helper_gen_audio_set_input()` (prompt
+ optional speaker-reference audio + language) → a `step_prompt()` loop → a `step_gen()` loop (the
engine owns semantic-token sampling via a `common_sampler`, feeding each sampled token + the
backbone's hidden state into `step_gen()` and receiving the next hidden state back — the same pattern
upstream's own `tools/tts/tts.cpp` `main()` uses) → `get_output()` for raw PCM, which the engine
encodes to WAV itself via `tts_wav.hpp` (not upstream's own WAV writer) so that already-tested code
stays in the loop.

**Why this replaced OuteTTS, not extended it.** Upstream #26254 ("mtmd: support Qwen3-TTS") deleted
the entire OuteTTS implementation from `tools/tts/tts.cpp` (it shrank from ~1450 to 205 lines) and
replaced the two-model OuteTTS-(text-to-codes)-+-WavTokenizer-(vocoder) design with the single
backbone+mmproj design above — there is no upstream code path for OuteTTS left at all past b10269
(`enum mtmd_gen_audio_type` has exactly `MTMD_GEN_AUDIO_TYPE_NONE` and `MTMD_GEN_AUDIO_TYPE_QWEN3TTS`).
See `docs/history/llama-cpp-breaking-changes.md`'s `b10269–b10270` row for the full investigation;
this was a breaking **public API** change (`TextToSpeech`'s constructor and `synthesize()` overloads
all changed shape) done deliberately — the project does not carry OuteTTS-compatibility shims.

**Nothing to re-verify on a llama.cpp bump.** Because there is no generator or extracted header
anymore, a version bump cannot silently break the TTS surface the way `patches/` or the old
extraction could — `mtmd_helper::gen_audio`'s API surface is upstream's own committed public header,
covered by the normal priority-8 API-compat review (`tools/mtmd/mtmd-helper.h` is on that list).

## Upgrading/Downgrading llama.cpp Version

**Runbook (documentation root):** [`docs/upgrade/llama-cpp-version-bump.md`](docs/upgrade/llama-cpp-version-bump.md)
covers the full bump process end-to-end — picking the target (topmost GitHub release, via the atom
feed), **chunking by `git diff` byte-size** (bump straight to the target when the diff is < 100 KiB,
else step through the largest intermediate tag still under the threshold), the
`.github/scripts/llama-next-version.sh` helper that computes the next reviewable step, and the
edit/verify/commit loop below. Use it for any non-trivial bump; the steps here are the mechanical core.

To change the llama.cpp version, update the following **four** files (and re-verify `patches/`):

1. **llama/CMakeLists.txt** — the `GIT_TAG` line for llama.cpp: `GIT_TAG        b8831`. (There is
   no second tag to keep in sync any more: the cosmetic `-DLLAMA_TAG=` that fed the old build-time
   TTS extraction went away with the Qwen3-TTS rework — see "Qwen3-TTS via `mtmd_helper::gen_audio`".)
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
# Edit llama/CMakeLists.txt: change GIT_TAG b8808 to b8831
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

The rows below cover the known **compile/link-level** breaks from b5022 to the current pin; start any
upgrade review with them rather than the full patch. Also review the project `CMakeLists.txt` for
build-system-level breaks (e.g. renamed link targets, new required headers) — those are not visible in
header file diffs alone.

**Two failure classes this list does NOT catch, both of which have bitten the project:**

1. **A same-repo header the project includes directly but that is not reachable through the
   dependency graph above.** `tools/server/server-schema.h` broke a full build at b10273 while sitting
   outside this table; it is in it now, and the `tools/server/*.h` rule in its row generalises that.
2. **A silent *contract* change behind an unchanged signature.** b10408 reduced
   `server_task_result_metrics::to_json()` to a bare slot array and b10519 split the task; no
   signature moved, every chunk compiled and linked clean, and `LlamaModel.getMetrics()` quietly
   returned the wrong shape for hundreds of builds. The same class hit `repeat_last_n` /
   `dry_penalty_last_n` at b10273, where only a *value range* moved. Two cheap mechanical checks catch
   these where a header diff cannot — run them on any bump that touches `tools/server/`:

```bash
# request-field set + their bounds, old tag vs new
git show <old>:tools/server/server-schema.cpp | grep -oE 'field_[a-z_]+\("[a-z_0-9]+"' | sort -u
git show <old>:tools/server/server-schema.cpp | tr '\n' ' ' \
  | grep -oE 'field_[a-z]+[^(]*\("[a-z_0-9]+"[^;]*?set_(hard_)?limits\([^)]*\)'
# response keys emitted by the result types
# response keys -- both emit forms; a single-form grep misses res["k"] = ... entirely
git show <old>:tools/server/server-task.cpp | { grep -oE '\{ *"[A-Za-z_0-9.]+" *,'; \
  git show <old>:tools/server/server-task.cpp | grep -oE '\[ *"[A-Za-z_0-9.]+" *\] *='; } | sort -u
```

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
| `tools/mtmd/mtmd-helper.h` | `mtmd_helper::gen_audio` (used directly by `tts_engine.cpp` since the Qwen3-TTS rework — no longer safe to skip), `mtmd_helper_bitmap_init_from_file` |
| `tools/server/server-schema.h` | `eval_llama_cmpl_schema` signature (called directly by `jllama.cpp`'s `populate_completion_task`) — **b10275 dropped its `n_ctx_slot` parameter** and broke a full build without any diff review catching it, because this file is a same-repo header `jllama.cpp` includes directly rather than one pulled in transitively through `common.h`/`llama.h`; it was outside this table until that incident (see `docs/history/llama-cpp-breaking-changes.md`'s b10270–b10275 row). Also watch `server-common.h`/`server-chat.h`/`server-task.h` the same way — anything under `tools/server/*.h` that `jllama.cpp`/`jni_helpers.hpp`/`json_helpers.hpp` `#include`s directly is in scope here, not just headers reachable from the dependency graph above. |
| `common/json-schema-to-grammar.h` | Grammar API |
| `ggml/include/ggml.h` | `ggml_type` enum values (e.g. `GGML_TYPE_F16`), tensor primitives |
| `ggml/include/ggml-backend.h` | Backend/device abstraction types |
| `ggml/include/ggml-opt.h` | Optimizer params pulled in via `common.h` |

**Safe to skip** (have never caused a break; not used directly by project code):
`common/sampling.h`, `common/log.h`, `common/json-schema-to-grammar.h`,
`ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `ggml/include/ggml-opt.h`,
`ggml-alloc.h`, `ggml-cpu.h`, `peg-parser.h`, `base64.hpp`

For the full record of upstream API breaks across version ranges (b5022 &#x2192; current), including which rows required project source changes vs. which stayed inside upstream-compiled translation units, see [`docs/history/llama-cpp-breaking-changes.md`](docs/history/llama-cpp-breaking-changes.md). When bumping the `llama.cpp` version, append a new row to that file covering the upgrade range.

## Build Commands

### Java (Maven)
```bash
mvn compile          # Compiles Java (jllama.h is maintained by hand, see "The JNI exception boundary")
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

Built libraries are placed in `src/main/natives/net/ladenthin/llama/{OS}/{ARCH}/{backend}/`
(`cpu` for a plain build; see "Natives jars").

### Building the native library for local Java tests

`mvn test` does **not** build the native library — Maven only compiles Java
and runs surefire. The shared library must already exist on disk under the
platform-specific path that `LlamaLoader` resolves at runtime; surefire puts
`src/main/natives/` on the test classpath (`additionalClasspathElements`).
Without it the native-backed tests skip themselves or fail with
`UnsatisfiedLinkError`.

The output path is derived by `CMakeLists.txt` from `OS_NAME` and `OS_ARCH`
detected by the helper script `.github/dockcross/dockcross-resolve-host`
(falls back to `uname` on hosts where the script is absent). The mapping
mirrors `OSInfo.translateOSNameToFolderName` on the Java side, so the same
folder name is produced on both ends.

| Host | Library file | Path produced by `cmake --build` |
|------|--------------|-------------------------------------------|
| Linux x86_64 | `libjllama.so` | `src/main/natives/net/ladenthin/llama/Linux/x86_64/cpu/` |
| Linux aarch64 | `libjllama.so` | `src/main/natives/net/ladenthin/llama/Linux/aarch64/cpu/` |
| macOS Apple Silicon | `libjllama.dylib` | `src/main/natives/net/ladenthin/llama/Mac/aarch64/metal/` |
| Windows x86_64 | `jllama.dll` | `src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu/` (Ninja) or `…/msvc/` (Visual Studio generator) |

On every platform but Linux x86-64/aarch64 exactly **one** `jllama` library is produced: `CMakeLists.txt` forces
`BUILD_SHARED_LIBS OFF`, so upstream `llama` and `ggml` are static libraries linked into
`jllama` (the `RUNTIME_OUTPUT_DIRECTORY_*` block that also names the `llama`/`ggml` targets
is a no-op for them — verified against the published 5.0.5 jars, which contain only
`jllama.dll` per Windows arch). `LlamaLoader` accordingly extracts and loads a single file
per backend (plus the files in its `jllama-extras.txt`, and `ggml-metal.metal` from a Metal build
that does not embed it). The two Linux CPU jars are the exception (`JLLAMA_CPU_VARIANTS`, see "CPU
variants"): there ggml is shared, and the directory holds its libraries and CPU modules, which the
loader extracts without loading them (`jllama-files.txt`). Historical note: upstream kherud once
shipped split `ggml` + `jllama` libraries, which is where stale "three co-located DLLs"
claims came from.

End-to-end local workflow for running Java tests:

```bash
# 1. Compile the Java classes (a new native method also needs its line in jllama.h, by hand)
mvn -q compile

# 2. Configure + build the native library for the current host
cmake -B build
cmake --build build --config Release -j$(nproc)
# The shared lib lands directly in src/main/natives/.../{OS}/{ARCH}/{backend}/ —
# no separate install step is needed; surefire puts that tree on the test classpath.

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

**Optional models.** Every model a test uses defaults to `models/<file>` from `.github/models.csv`
(`TestConstants.DEFAULT_*`), so downloading a model from that list into `models/` is all it takes —
the tests find it (module-relative or at the reactor root) and self-skip while it is missing. The
properties below only **override** a default. The full property → consumer → default table for every
`net.ladenthin.llama.*` property the library understands (runtime + test) is the user-facing
**[System Properties Reference](README.md#system-properties-reference)** in the README. The summary
below covers the model bindings:

| Property | Default test that uses it | Default (`models/…` unless noted) |
|----------|---------------------------|-------|
| `net.ladenthin.llama.tool.model` | `ToolCallingIntegrationTest`, `OpenAiServerToolCallingIntegrationTest` | `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` |
| `net.ladenthin.llama.nomic.path` | `LlamaEmbeddingsTest#testNomicEmbedLoads` | `nomic-embed-text-v1.5.f16.gguf` (issue #98 regression) |
| `net.ladenthin.llama.vision.model` | `MultimodalIntegrationTest` | `SmolVLM-500M-Instruct-Q8_0.gguf` (any vision-capable GGUF works) |
| `net.ladenthin.llama.vision.mmproj` | `MultimodalIntegrationTest` | `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` |
| `net.ladenthin.llama.vision.image` | `MultimodalIntegrationTest` | committed `src/test/resources/images/test-image.jpg`; any png/jpeg/webp/gif |
| `net.ladenthin.llama.tts.model` | `TtsIntegrationTest` | `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` (any Qwen3-TTS-family model works) |
| `net.ladenthin.llama.tts.mmproj` | `TtsIntegrationTest` | `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` |
| `net.ladenthin.llama.train.model` | `LlamaTrainerIntegrationTest` | `stories260K.gguf` (must be **F32**) |
| `net.ladenthin.llama.decision.model` | `SystemOneIntegrationTest` (decision-model half) | none (not in CI's set) — e.g. upstream's `ggml-org/tinylaya-for-testing-gguf` |
| `net.ladenthin.llama.audio.model` | `AudioInputIntegrationTest` (llama.cpp discussion #13759) | none (not in CI's set) — e.g. `ultravox-v0_5-llama-3_2-1b.gguf` |
| `net.ladenthin.llama.audio.mmproj` | `AudioInputIntegrationTest` | none — e.g. `mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf` |
| `net.ladenthin.llama.audio.input` | `AudioInputIntegrationTest` | committed `src/test/resources/audios/sample.wav`; any `.wav`/`.mp3` |

```bash
# CI's models, e.g. the vision pair (URLs in .github/models.csv) -- no property needed afterwards:
mvn test -Dtest=MultimodalIntegrationTest
# Another model than the default:
mvn test -Dtest=TtsIntegrationTest \
         -Dnet.ladenthin.llama.tts.model=/path/to/qwen3-tts-backbone.gguf \
         -Dnet.ladenthin.llama.tts.mmproj=/path/to/qwen3-tts-mmproj.gguf
# Audio input has no CI model, so it always needs the properties:
mvn test -Dtest=AudioInputIntegrationTest \
         -Dnet.ladenthin.llama.audio.model=models/ultravox-v0_5-llama-3_2-1b.gguf \
         -Dnet.ladenthin.llama.audio.mmproj=models/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf
```

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
cmake --build build --config Release -j$(nproc)   # -> src/main/natives/.../<os>/<arch>/cpu/libjllama.so
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
anchored at `net/ladenthin/llama` in `LlamaLoader.NATIVE_RESOURCE_BASE` (it must
not track the loader's own Java package; no leading slash, it is resolved through the ClassLoader). This is the same
"FQN/path not updated after a package move" class as the stale
`spotbugs-exclude.xml`, PIT `targetClasses`, and `CMakeLists.txt` OSInfo repairs.

### Code Formatting

C++ formatting is **enforced in CI** (`.github/workflows/clang-format.yml`) with a **pinned**
clang-format — currently **23.1.3**, installed via `pip install clang-format==23.1.3`. Format with
that exact version before committing; a different clang-format version reflows code differently and
will fail the check.

```bash
pip install "clang-format==23.1.3"
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp src/test/cpp/*.cpp   # Format C++ code
```

The JNI header `src/main/cpp/jllama.h` (originally `javac -h` output, now maintained by hand) is intentionally excluded.
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
- `TextToSpeech` — Separate AutoCloseable native type for speech synthesis over llama.cpp's upstream Qwen3-TTS pipeline (a backbone text GGUF + an mmproj GGUF bundling the speaker encoder, code predictor, and code2wav decoder); `synthesize(text)` returns a 24 kHz mono 16-bit WAV byte stream, with overloads for a cloned-voice speaker-reference clip and language. Native orchestration in `tts_engine.{h,cpp}` drives upstream's `mtmd_helper::gen_audio` streaming API directly (see "Qwen3-TTS via `mtmd_helper::gen_audio`" below) — there is nothing extracted or hand-copied from llama.cpp source; the in-memory WAV writer is `tts_wav.hpp`.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes. Every wire name they can
  emit is an enum constant carrying the contract it must satisfy (`args.ModelOption` + `args.ModelFlag` for
  argv, `parameters.RequestField` for the request body), and the base classes accept nothing else — see
  "Wire-name registries" below. `InferenceParameters.toJson()` renders the request body; its `toString()`
  is a redacted debug view and deliberately not valid JSON.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.
- **`server` package — OpenAI-compatible HTTP endpoint (a single implementation).**
  - `server.OpenAiCompatServer` — built only on the JDK's `com.sun.net.httpserver` (no new dependency), embeddable and runnable via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …` (the fat-jar default `Main-Class` is now `NativeServer` — see "Two server modes"). Serves `POST /v1/chat/completions` (streaming via SSE + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST /infill`, `GET /v1/models` and `GET /health` (every route is also reachable without the `/v1` prefix), so editors that speak the OpenAI protocol (e.g. VS Code Copilot "Custom Endpoint", Cline, Roo Code, Continue) can drive a local model. Streaming chat uses the native OAI chunk path (`LlamaModel.streamChatCompletion` → `requestChatCompletionStream` / `receiveChatCompletionChunk` + the C++ `wrap_stream_chunk` helper), preserving `delta.tool_calls`; completions/embeddings/infill forward verbatim to the matching `LlamaModel.handle*`; rerank reshapes `handleRerank` into the OAI `results`/`data` shape. The chat mapper forwards `stream_options` and `response_format` and defaults `cache_prompt=true`; a CORS `Filter` answers `OPTIONS` preflights; `OpenAiSseFormatter.ensureUsageCachedTokens` guarantees `usage.prompt_tokens_details.cached_tokens` on the streamed usage chunk (Copilot crash fix, microsoft/vscode #273482). **Agentic tool-calling is the primary target**; a C++ guard (`test_server.cpp`) pins `tool_calls.function.arguments` as a JSON string (llama.cpp #20198).
  - **Alternative protocol surfaces** (pure translation over the OpenAI chat core — no second inference path; each reconstructs streamed tool calls via `ToolCallDeltaAccumulator`): **Ollama-native** (`GET /api/version`, `/api/tags`, `POST /api/show`, `/api/chat` with NDJSON streaming, `/api/generate` prompt-completion/FIM — `OllamaApiSupport`; `/api/show` advertises tools/insert/vision capabilities + context length for Copilot's Ollama provider), **Anthropic Messages** (`POST /v1/messages`, SSE event stream — `AnthropicApiSupport` + `AnthropicStreamTranslator`), and **OpenAI Responses** (`POST /v1/responses`, SSE event stream — `ResponsesApiSupport` + `ResponsesStreamTranslator`). The llama.cpp-native `GET /props` (context length + `modalities`) is served via `OpenAiSseFormatter.propsJson` for autocomplete clients that size their context from it.
  - Supporting classes: `OpenAiServerConfig` (builder; optional bearer auth; binds `127.0.0.1`; `corsAllowOrigin`; `supportsVision`), `OpenAiServerCli` (testable CLI arg parser → `ModelParameters` + `OpenAiServerConfig`; flags incl. `--mmproj`/`-mmdev,--mmproj-device`/`--embedding`/`--reranking`), `OpenAiRequestMapper` (OAI chat request → `InferenceParameters`), `OpenAiSseFormatter` (SSE/models/error JSON + usage normalization), `OaiRerankSupport` (pure rerank request/response shaping), and the model-free test seam `OpenAiBackend`/`ChunkSink` + `LlamaModelBackend`. The streaming envelope is parsed by `json.ChatStreamChunkParser`.
  - The `server` package is a dedicated top layer in the ArchUnit `layeredArchitecture` rule (the only layer allowed to access the root `Api`); `noInternalJdkImports` carries an explicit exception for the supported `com.sun.net.httpserver` (the exported `jdk.httpserver` module, which `module-info.java` `requires`). See README "OpenAI-compatible HTTP server".

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp. ~1,900 lines; 35 native methods (31 `LlamaModel` + 3 `TextToSpeech` + 1 `LlamaQuantizer`) plus `JNI_OnLoad`/`JNI_OnUnload`.
- `utils.hpp` — Helper utilities (format helpers, argv stripping, token-piece serialisation).
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- **The `json` alias is upstream's `common_json`, not `nlohmann::ordered_json` (since llama.cpp b10585, upstream #27511).** `tools/server/server-common.h` now says `using json = common_json;` — a deliberately small pimpl wrapper (`common/json.{h,cpp}`, compiled into `llama-common`) around the vendored nlohmann copy. Two traps this cost the project once, both of which **compile silently**:
  1. **An unscoped enum became a JSON boolean — fixed upstream at b11080, and still worth knowing.** From b10585 to b11069, `common_json_value`'s integral constructor template was `std::is_integral`-gated, which excludes enums, so an enum bound to `common_json_value(bool)` and serialised as `true`/`false`. Upstream [#28518](https://github.com/ggml-org/llama.cpp/pull/28518) added an `std::is_enum`-gated constructor delegating to the underlying type, which retired `patches/0010` (see the drop note under the patches table). **Keep casting anyway** — `jllama.cpp`'s two `"vocab_type"` sites still `static_cast<int>(...)`, which is equivalent under the fix and immune to a revert of it. Guards, both in CI: `test_json_helpers.cpp`'s `CommonJsonEnum` trio pins the mechanism (uncast enum is numeric, explicit cast equivalent, `bool` still boolean); `LlamaModelTest`'s `isIntegralNumber()` assertion pins the real wire value (the model-gated suite no longer self-skips — see "CI model policy" below).
  2. **`common_json` converts to `std::string` implicitly**, so it binds happily to a `const nlohmann::json &` parameter (via nlohmann's string-constructible converting constructor) and then throws `json::type_error 302` at runtime. Never declare a project helper as taking `nlohmann::json` when callers pass the `json` alias — `require_json_field_impl` is a template for exactly this reason.
  Other differences to know: no `get_ref`/`array_t`/`type_name()`; a braced list in *value* position does not build an array (write `json::array({...})`); `at(key)` needs an explicit `.get<T>()`; errors are `common_json_error`; and `get<T>()` is limited to the types explicitly specialised in `common/json.cpp`. `log_helpers.hpp` and `train_engine.cpp` keep their own `nlohmann::json` alias — they never touch the server's `json`.
- Uses `nlohmann/json` for JSON deserialization of parameters in the two files named above; everything on the server path uses `common_json`.
- The upstream server library (`server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-schema.cpp`, `server-models.cpp`, and — since b9829 — `server-stream.cpp`) is compiled directly into `jllama` via CMake — there is no hand-ported `server.hpp` fork. **`server-stream.cpp` is mandatory, not optional:** it defines the resumable-streaming SSE replay buffer (`g_stream_sessions`, `stream_session_attach_pipe`, `stream_aware_should_stop`, `stream_conv_id_from_headers`, the `stream_pipe_*` types) that `server-context.cpp` / `server-http.cpp` / `server-models.cpp` now `#include "server-stream.h"` and call, so omitting it fails the link with undefined references. It is platform-neutral (threads + std mutex/condvar, no `subprocess.h`/`posix_spawn_*`), so it builds on Android too and sits outside the `server-models.cpp` Android guard. `jllama` wires its own JNI routes and never calls `g_stream_sessions.start_gc()` (only the excluded standalone `server.cpp` `main()` does), so its GC thread stays dormant. **Phase 2:** the upstream HTTP transport (`tools/server/server-http.cpp`) and its `cpp-httplib` backend (`vendor/cpp-httplib/httplib.cpp`) are now compiled into `jllama` too, so the OpenAI-compatible server can be driven natively from JNI *inside* `libjllama` — no separate `llama-server` executable (a JNI shared library loads anywhere a JVM runs, which a standalone binary does not). `server-http.cpp` does `#include "ui.h"` (the WebUI asset table that `tools/ui`/`llama-ui` normally generates); since the Svelte WebUI is not shipped, `src/main/cpp/webui_stub/ui.h` supplies the upstream **empty-asset** interface and leaves `LLAMA_UI_HAS_ASSETS` undefined (all static-asset-serving blocks compile out). `<cpp-httplib/httplib.h>` already resolves through `llama-common` — since upstream #27304 (b10488) not from a `PUBLIC ../vendor` include dir of its own but transitively, via the `vendor::nlohmann` / `vendor::sheredom` INTERFACE targets it links PUBLIC, each of which exports the `vendor/` root (same nlohmann/json 3.12.0 as the FetchContent copy). No SSL: `CPPHTTPLIB_OPENSSL_SUPPORT` is left undefined (plain-HTTP; bind localhost / front with a TLS proxy). **`server.cpp`, `server-tools.cpp` and `server-mcp.cpp` are now compiled in too** (on non-Android — they pull in `subprocess.h`/`posix_spawn_*`, so they share `server-models.cpp`'s Android guard): b9870 exposes `server.cpp`'s entry as `int llama_server(int, char**)` (no `main` in the file), and `patches/0006` makes it embeddable (no process signal handlers, forwarded-argv parse, out-of-band shutdown). **`server-mcp.cpp` is new in b10154** (upstream MCP-server support): both `server.cpp` (`llama_server`'s `mcp_mgr` lifecycle) and `server-tools.cpp` (`tools.setup(..., mcp_mgr)` / `server_mcp::call_tool`) reference `server_mcp`, so it **must** be in the `target_sources` list or the link fails with undefined `server_mcp::{start,shutdown,call_tool,list_tools,~server_mcp}` — **latent on Linux** (a shared object tolerates undefined symbols) but a **hard link error on macOS/ld64 and Windows/MSVC**. It is compiled into `jllama` and — on non-Android — into `jllama_test` too, together with `server.cpp`, `server-tools.cpp`, `server-http.cpp` and `httplib.cpp`, so `test_native_server_shutdown.cpp` can drive the real `llama_server()`. The `NativeServer` JNI bridge (`src/main/cpp/native_server.cpp`) calls `llama_server` on a worker thread, so the **full** upstream server — WebUI and all — runs inside `libjllama`. See "Two server modes" below.

### Two server modes (`OpenAiCompatServer` vs `NativeServer`)

The library exposes **two** ways to serve a model over HTTP, on two different transports. The fat jar's `Main-Class` is `server.ServerLauncher`, a tiny dispatcher: it runs `OpenAiCompatServer` when `--jllama-openai-compat` is present (that marker is stripped, the rest forwarded) and the default `NativeServer` otherwise. Both mains are also runnable directly by class name via `java -cp`. The two modes:

1. **`server.OpenAiCompatServer` (Java transport).** OpenAI/Ollama/Anthropic-compatible JSON API on the JDK's `com.sun.net.httpserver`, driving the compiled server *core* over JNI. Embeddable, no extra dependency, and it can share/reuse a `LlamaModel`. It serves **no** static assets — its `/` route is a 404, so **no WebUI**. It has its own `main` (run via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …`); its CLI (`OpenAiServerCli`) maps a curated flag subset (`-m/-c/-b/-ub/-ngl/-t/-tb/-ctk/-ctv/--jinja/--chat-template-kwargs/--host/--port/--parallel/--mmproj/--api-key/--embedding/--reranking/--rpc`).
2. **`server.NativeServer` (native transport) — the default fat-jar server (when `--jllama-openai-compat` is absent).** Runs the **full upstream `llama_server`** (via `patches/0006` + `native_server.cpp`) inside `libjllama`, forwarding the raw llama-server argv verbatim — so **every** llama-server flag works and the **embedded WebUI is served** (when the assets are compiled in; CI's released jars have them, local `cmake` builds use the empty-asset stub). With the classic constructor it is an **independent lifecycle** (loads its own model from the argv, like `llama-server.exe`; owns the process's llama backend + stderr logging while running); the **attach constructor** (`NativeServer(LlamaModel, String...)`, via `patches/0007`'s `llama_server_attach`) instead serves an **already-loaded `LlamaModel`** — one copy of the weights, the model's worker keeps driving inference, the HTTP routes post to its queue; caller closes the server before the model. **Router mode** (start without a model argument: `--models-dir`, `GET/POST /models`, per-request model selection) works in-JVM after `NativeServer.setWorkerCommand(...)` redirects the worker spawn to a fresh JVM (`patches/0008` — upstream re-execs its own binary, which in a JVM is `java`); the typed `server.RouterClient` (+ `value.RouterModel`, `json.RouterModelsResponseParser`) wraps the model-management endpoints (list/load/unload/await-loaded with fail-fast on failed workers) so callers don't hand-roll HTTP+JSON, and its `apiKey` constructors send `Authorization: Bearer <key>` — required for **every** one of those calls against a router started with `--api-key` since b10519 (#26347 dropped `/models` + `/v1/models` from the public-endpoint set; `/models/load` and `/models/unload` were always gated). `awaitModelLoaded` cannot observe a model hidden by a preset with `dedup-cache-models` (b10505/#27346 omits it from `GET /models` although it still loads and serves by name), so its "not listed" message names that cause explicitly; such a model is reached by issuing the request directly instead. Either way it is **single-instance per process** (upstream keeps shutdown state in file-scope globals) and **not available on Android** (the `subprocess.h` guard). `libjllama` loading anywhere a JVM runs is what makes this "no separate `llama-server.exe`" possible.

### `getMetrics()` — one object rebuilt from two upstream tasks

`LlamaModel.getMetrics()` / `getMetricsTyped()` return the single server-introspection object the
Java side has always documented: `idle` / `processing` / `deferred` / `t_start`, the cumulative and
current-window `n_*` / `t_*` counter pairs, and a `slots` array. Upstream stopped emitting that in
one piece — **b10408** (#26920) reduced `server_task_result_metrics::to_json()` to the slot array,
and **b10519** (#27376) split the task in two: `SERVER_TASK_TYPE_METRICS` keeps only the counters
(its `to_json()` is unused and returns JSON null; `to_metrics()` renders them as Prometheus text)
while `SERVER_TASK_TYPE_SLOT_GET` carries the slot array plus the idle-slot count.

`handleSlotAction(0, …)` therefore posts **both** tasks and merges the results through the pure
helper `server_metrics_to_json` (`json_helpers.hpp`, unit-tested in `test_json_helpers.cpp`), rather
than letting the Java contract follow upstream's transport split. Durations are converted from
upstream microseconds to the milliseconds the payload has always used. The merge also surfaces the
counters upstream added since — `n_prompt_tokens_cached_total` and the speculative-decoding tallies
(`n_draft_tokens_total`, `n_draft_accepted_total`, `n_draft_verif_steps_total`,
`n_accepted_per_pos_total` — upstream's own spellings, kept verbatim) — which upstream emits only
as Prometheus counters from `to_metrics()`, with no JSON representation at all; `value.ServerMetrics` exposes them with typed getters (plus a derived
`getDraftAcceptanceRate()`). No second JNI entry point and no Prometheus-text parser were needed.

The metrics task is posted with `server_task::metrics_reset_bucket` left at its default `false`, so
`getMetrics()` never resets the current-measurement window; only an HTTP `/metrics` scrape does.

### Native Helper Architecture

The project C++ helpers follow a strict semantic split:

**`json_helpers.hpp`** — Pure data transforms.
- Input: the `json` alias (upstream `common_json` since b10585), `server_task_result_ptr`, plain C++ types.
- Output: `json`, `std::vector`, `std::optional`, plain C++ types.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama state (`llama_context*`, `llama_vocab*`, `server_context*` never appear).
- Functions are named without `_impl` suffix — they are the canonical implementation.
- Testable with JSON literals and fake result objects; no JVM and no loaded model required.
- Upstream server headers must be included by the translation unit first (they define `server_task_result_ptr`, `json`, etc.).

Functions: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`,
`parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`,
`parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk`,
`server_metrics_to_json`, `route_error_message`.

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
- `require_json_field_impl` — throws `"<field> is required"` if key is absent. **Templated on the JSON type on purpose**: a plain `const nlohmann::json &` parameter still accepts a `common_json` (through its `operator std::string()`) and turns the presence check into a runtime `type_error 302`.
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

### The JNI exception boundary — `jni_guard_impl`

An exception that escapes a native method and unwinds across the JNI boundary is **undefined
behaviour and aborts the JVM** on most implementations. **Every `Java_*` entry point must therefore
convert anything that escapes into a Java exception**, and there are 45 of them across four TUs —
`jllama.cpp` (35), `native_server.cpp` (5), `rpc_bridge.cpp` (4), `train_engine.cpp` (1).

The mechanism is `jni_guard_impl(env, exception_class, [&]() -> Ret { … })` (`jni_helpers.hpp`,
Layer A). It is **additive**: an entry point that already converts `std::exception` itself keeps
doing so and never reaches the guard's handlers. What the guard adds everywhere is the
**`catch (...)` arm** — the case for an exception type not derived from `std::exception`, which
otherwise has no backstop at all. On a catch it returns the zero/`nullptr` sentinel for the entry
point's return type.

Two rules the handler keeps, both pinned by tests in `test_jni_helpers.cpp`:

- **Never `ThrowNew` over an already-pending Java exception.** The JNI spec forbids most calls in
  that state, and the pending exception is the more precise error — so it is left in place.
- **Never `ThrowNew` with a null class.**

**Three entry points are deliberately NOT routed through it, and each uses a function-try-block
instead** (which also avoids reindenting a `goto`-carrying body): `JNI_OnLoad` runs before
`c_llama_error` is cached and `JNI_OnUnload` after it is released, so neither has a class to throw
with — `JNI_OnLoad` returns `JNI_ERR` (the JVM surfaces that as `UnsatisfiedLinkError`) and
`JNI_OnUnload` swallows. `LlamaTrainer_finetuneNative` reports failure as its **return string**
rather than a Java exception, and `train_engine.cpp` deliberately keeps its own `nlohmann` alias
and never includes `jni_helpers.hpp`, so its backstop returns an error string to preserve that
contract.

**When you add a native method, wrap it.** The guard is not enforced by a test — a new unguarded
entry point is invisible until something throws through it in production.

**And declare it in `src/main/cpp/jllama.h` by hand.** That header is committed and maintained
manually -- nothing regenerates it, `mvn compile` included, whatever older notes here say -- and it
is what gives the `LlamaModel` entry points in `jllama.cpp` their C linkage (they are not inside an
`extern "C"` block there). A method missing from it compiles and links, but under its C++-mangled
name, so the JVM cannot find it and the first call throws `UnsatisfiedLinkError`. Caught this way
for `handleSystemOne` at the b11361 bump, before the first build.

### Parameter Flow
Java parameters are serialized to JSON strings and passed to native code, which deserializes them using nlohmann/json. This avoids complex JNI field mapping for the many llama.cpp parameters.

### Native Library Resolution
`LlamaLoader` tries in order:
1. System property `net.ladenthin.llama.lib.path`
2. `System.loadLibrary` on Android (the AAR's `jni/`), then `java.library.path`
3. The natives jars on the classpath: every `net/ladenthin/llama/{os}/{arch}/{backend}/` present, in
   `BACKEND_PRIORITY` order, extracted and loaded until one loads (see "Natives jars")

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for **Android** targets and the
two x86_64 manylinux_2_28 jobs (CPU variants, CUDA). **Linux `aarch64` is built natively** on a
GitHub `ubuntu-24.04-arm` runner inside the pypa manylinux_2_28 image, through
`.github/manylinux/manylinux_2_28-aarch64` (see "Linux aarch64" below). Deleted wrappers: the
`dockcross-linux-arm64-lts` cross-compiler (GCC 8.5), the never-wired 32-bit `dockcross-android-arm`,
and `dockcross-manylinux2014-x64` (GCC 10 cannot build the AVX-VNNI/AMX CPU variants).

### Linux aarch64: native build in the manylinux_2_28 image

The `crosscompile-linux-aarch64` job (id kept for its downstream `needs:` reference; display name
**"Build manylinux_2_28 aarch64 (CPU variants)"**) builds **natively on `ubuntu-24.04-arm`, inside
`quay.io/pypa/manylinux_2_28_aarch64`** (AlmaLinux 8, gcc-toolset-14 = GCC 14.2.1, glibc 2.28),
through the wrapper `.github/manylinux/manylinux_2_28-aarch64`, with the CPU variants
(`-DJLLAMA_CPU_VARIANTS=ON`, see "CPU variants").

**History, because each step had a reason.** The original `dockcross/linux-arm64-lts` cross image
(GCC 8.5 / glibc 2.17) could not compile llama.cpp b9789 (its C++17 CTAD-in-`new` needs GCC >= 12),
so the job moved to a native build on the arm64 runner with Ubuntu's `gcc-14`, as upstream's own
`ubuntu-cpu` aarch64 release job does -- at the price of a **glibc ~2.39** floor. The CPU variants
then asked for the same image generation as x86-64, and the pypa manylinux_2_28 aarch64 image gives
GCC 14 **and** glibc 2.28: the floor **falls** to 2.28 (RHEL 8, Ubuntu 20.04, Debian 10, Amazon Linux
2023 and later). Not dockcross's `manylinux_2_28-aarch64`: that is an **amd64 image with a
crosstool-NG cross-compiler** (measured: platform `linux/amd64`, 1.9 GB), i.e. a cross build on an
x86 runner; the pypa image is native arm64 (0.5 GB) and runs where the artifact runs.

Wiring:
- `runs-on: ubuntu-24.04-arm`; no `setup-java`/`mvn compile` (the JNI header is committed, and the
  dockcross x86-64 jobs never ran Maven either). The wrapper runs `build.sh` in the container as the
  runner's uid, with the repository mounted at `/work` and `HOME=/tmp`.
- `MANYLINUX_ARGS: "-e SCCACHE_WEBDAV_ENDPOINT -e SCCACHE_WEBDAV_TOKEN -e USE_CACHE"` forwards the
  cache env like `DOCKCROSS_ARGS`; `build.sh` fetches the aarch64 static-musl sccache inside the
  container (curl is in the image) and the probe guards it.
- The image is pinned by **digest** in the wrapper (tag `2026.10.03-1`). Bump it deliberately, with a
  Publish run: it decides the glibc floor `verify-native-deps.py` enforces (`GLIBC_2.28`) and the
  compiler.
- No `ctest` (the variant build has no static CPU backend for `jllama_test`): **`test-cpp-linux-aarch64`**
  builds the default static library with `gcc-14` on the same runner and runs the C++ suite on real
  ARM hardware, as this job did before.
- Branch protection: a required check pinned to "Build and Test Linux aarch64" must be repointed.

### Linux s390x: big-endian cross-build + qemu test gate

`build-linux-s390x` adds **IBM Z (s390x, big-endian)** — the one target whose
byte order differs from every other platform. It **cross-compiles** with the GCC s390x toolchain
(`g++-s390x-linux-gnu`, native x86 speed — no emulated build) and then runs the **full C++ unit suite
under `qemu-user`** (`CMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-s390x-static`, `QEMU_LD_PREFIX=/usr/s390x-linux-gnu`).
That `ctest` run is a **real big-endian correctness gate** for the byte-order-sensitive surface — the
little-endian WAV writer (`tts_wav.hpp`), the JSON/token/embedding transforms, and the JNI helpers —
which is where an endian bug in *our* code could hide. Model-backed **Java** tests are deliberately
**not** run under emulation (a JVM + GGUF inference under `qemu-user` is slow and flaky); the Java↔JNI
boundary uses host-native array copies (endian-transparent), so the C++ gate covers the actual risk.
`-DGGML_OPENMP=OFF` sidesteps cross-libgomp issues (ggml uses its own `std::thread` pool).
`-DGGML_NATIVE=OFF` also switches VXE off, so the binary is **scalar** on purpose; **do not "fix" a
VXE-related compile error with `-DGGML_VXE=ON`** (it needs `-march=z15` too and raises the hardware
floor) -- measured in the `0013` note of `docs/history/dropped-llama-patches.md`. s390x is a
CPU platform like aarch64, so it ships as the natives jar `cpu-linux-s390x`, part of `llama-platform`
(`OSInfo` maps `os.arch=s390x` → `Linux/s390x`).
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
model + mmproj, and the Qwen3-TTS backbone + mmproj (`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`,
smallest available quants: `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` backbone +
`mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` mmproj — no smaller mmproj quant is published), with
their `-Dnet.ladenthin.llama.*` properties set, so `LlamaEmbeddingsTest`, `MultimodalIntegrationTest`,
and `TtsIntegrationTest` are **intended** to run on every platform rather than self-skipping.

**How the paths resolve (this was silently broken until it was fixed after the b10618 bump).**
Surefire's working directory defaults to the **module** basedir (`<workspace>/llama`), while the
shared GGUF cache is restored to `<workspace>/models/` and every model path — the `TestConstants`
constants and the `-Dnet.ladenthin.llama.*` properties alike — is stated relative as `models/…`.
Those therefore resolved to `<workspace>/llama/models/…`, which does not exist: **every**
model-gated class aborted in its `@BeforeAll` `Assumptions.assumeTrue(file.exists())` and reported
as *nothing at all* while the job still went green, on every `test-java-*` job. Note the precise
shape, because it defeats the obvious guard: a class-level `@BeforeAll` assumption makes Surefire
record `tests="0" errors="0" skipped="0"` — the class contributes **no** test entries, so a check of
the form "did this run skip anything?" is blind to it. The only thing that catches it directly is a
floor on the number of tests actually executed (see `TODO.md`). It is why several stale
assertions (e.g. `LlamaModelTest#testGetMetrics` against a payload shape upstream had dropped at
b10408) never failed in CI. The fix is **`TestConstants.resolveModelPath` /
`resolveModelProperty`**, which accept either layout — module-relative first, then the reactor root
— so a developer with models under `llama/models/` and CI with them at the workspace root both
work, with no workflow change. Every `TestConstants` path constant is routed through it, as is
every `-Dnet.ladenthin.llama.*` fixture property; `TestConstantsTest` pins both the resolver and
the wiring (a future edit that drops the wrapper from a constant fails that test rather than
silently re-muting the suite). `llama-langchain4j` had the identical defect and carries the same
resolver as `TestModelPaths` (test classes are not shared between modules).

`validate-models.sh` treats all of these as **required** (a missing model hard-fails the job before
tests run, so a download regression can never silently downgrade to a skip). **The tests default to
exactly the CI model set**: every model-gated test reads its path through a `TestConstants`
`DEFAULT_*` constant (`models/<file>`), a `-Dnet.ladenthin.llama.*` property only overrides it, and
`TestConstantsTest.theModelDefaultsAreExactlyTheCiModelSet` asserts that the `.gguf` constants and the
rows of `models.csv` are the same set in both directions (a row no constant names is downloaded for
nothing; a constant naming a model the list lacks self-skips in CI). So the test jobs pass **no** model
properties. That also closed a gap this file used to describe: `LlamaTrainerIntegrationTest` defaults
to `stories260K.gguf` and runs on every Java test job, so the Java → JNI → native trainer round trip
has a runnable guard. **One** class still self-skips everywhere: `AudioInputIntegrationTest` — its
prompt clip is committed (`src/test/resources/audios/sample.wav`), but the audio model + mmproj have no
CI download. Half of `SystemOneIntegrationTest` does too (the `/v1/systemone` answers need a decision model,
which is not in the CI set either); its rejection case runs everywhere with the draft model.
The model set has a **single source of truth: `.github/models.csv`** (one `filename,url` row per
model; `#` comments). Everything derives from it: the **`download-models`** job (ubuntu,
`needs: startgate`) is the only place models are fetched from HuggingFace (one manifest-driven
`curl` loop; files already restored from cache are skipped) and the **only writer** of the shared
GGUF cache (path `models/`, key **`gguf-models-<hash of models.csv>`** — so *editing the manifest
automatically creates a fresh complete cache entry*; no manual cache deletion on a model-set
change). The writer sets **`enableCrossOsArchive: true`**, making the one ubuntu-built entry the
same entry macOS and Windows restore. Every consumer uses the composite action
**`.github/actions/restore-models`** — restore-only (`actions/cache/restore`; a consumer can never
re-save an empty/partial entry) plus `validate-models.sh`, which reads its required list from the
same manifest. A **`verify-model-cache` matrix job** (ubuntu / macOS / Windows, `needs:
download-models`) runs the same action with `fail-on-cache-miss: true` and so proves the entry is
restorable **and complete** on every OS **before any model-consuming job starts** — all `test-java-*`
jobs, the langchain4j integration job, the Android emulator job and the fat-jar smoke jobs `need:
verify-model-cache`. (This design hardened after run 28805360584: without the cross-OS flag, cache
entries are versioned per-OS and the unreachable Windows-side entry had been re-saved **empty** (343 B)
after an eviction; and the then-separate `validate-models.bat`'s quoted `MODELS` list broke cmd's
for-tokenization so its `exit /b 1` never fired — the empty cache sailed through the "gate" and the
Windows jobs silently self-skipped every model-backed test until the fat-jar smoke's hard check caught
it. The `.bat` is gone: Windows runs the one bash script under Git Bash, which is why it uses
`head -c`/`wc -c` rather than `xxd`/`stat`.) The remaining `*_MODEL_NAME` env vars of publish.yml
serve only the consumers outside the llama module's tests (smoke scripts, Android emulator jobs,
langchain4j/agent integration); `check-natives.py` fails when one is not a filename of `models.csv`.
Locally the model tests self-skip when a GGUF is absent (`Assume.assumeTrue`), so a partial local
checkout is fine.


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
| `src/test/cpp/test_utils.cpp` | 168 | Upstream helpers: `server_tokens`, `server_grammar_trigger`, `gen_tool_call_id`, `json_value`, `json_get_nested_values`, UTF-8 helpers, `format_response_rerank`, `format_embeddings_response_oaicompat`, `oaicompat_completion_params_parse`, `oaicompat_chat_params_parse`, `are_lora_equal`, `strip_flag_from_argv`, `token_piece_value`, `json_is_array_and_contains_numbers`, `format_oai_sse`, `format_oai_resp_sse`, `format_anthropic_sse`, `parse_lora_request`, `common_chat_parse` over malformed UTF-8 (the `ContentOnlyParseUtf8` guard, which pins upstream #29161's one-U+FFFD-per-invalid-run contract — formerly the guard for the dropped `patches/0011`) |
| `src/test/cpp/test_server.cpp` | 206 | Upstream result types: `server_slot_stats` (the `timings` JSON payload; replaced `result_timings` in b10408), `task_params::to_json()` (incl. `dry_sequence_breakers`, `preserved_tokens`, `timings_per_token`), `completion_token_output`, `server_task_result_cmpl_partial` (non-oaicompat + `to_json_oaicompat` + logprobs + `to_json_oaicompat_chat` + `to_json_anthropic` + dispatcher), `server_task_result_cmpl_final` (non-oaicompat + `to_json_oaicompat` + `to_json_oaicompat_chat` + `to_json_oaicompat_chat_stream` + `to_json_anthropic` + `to_json_anthropic_stream` + tool_calls + dispatcher), `server_task_result_embd`, `server_task_result_rerank`, `server_task_result_metrics` (`to_metrics()` = the `/metrics` Prometheus exposition text; its `to_json()` has been unused since b10519 and returns `json{}` = JSON null), `server_task_result_slots` (`to_json()` = the `/slots` array, fed by the b10519 `SERVER_TASK_TYPE_SLOT_GET` task), `server_task_result_slot_save_load`, `server_task_result_slot_erase`, `server_task_result_apply_lora`, `server_task_result_get_lora`, `server_task_result_error`, `format_error_response`, `server_task::need_sampling()`, `server_task::n_tokens()`, `server_schema::eval_llama_cmpl_schema()` (parsing pipeline + grammar routing + error paths + per-request `dry_*` and `sse_ping_interval` field round-trips incl. hard-limit + server-default inheritance), `response_fields` projection |
| `src/test/cpp/test_json_helpers.cpp` | 67 | All functions in `json_helpers.hpp`: `get_result_error_message`, `results_to_json`, `rerank_results_to_json` (incl. missing/out-of-range `index` rejection), `parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk`, `server_metrics_to_json`, `route_error_message` |
| `src/test/cpp/test_log_helpers.cpp` | 13 | All functions in `log_helpers.hpp`: `log_level_name`, `format_log_as_json` |
| `src/test/cpp/test_common_log_callback.cpp` | 6 | **The runnable guard for `patches/0014`**: `common_log_set_callback()` on a private `common_log_init()` instance (never `common_log_main()`, so the process-wide logger the other tests print through is untouched) — delivery of level + bare text, no prefix/timestamp even when both are on (what `common_init()` does), clearing stops delivery, a swap drains queued entries to the *previous* sink (the property behind `LlamaModel.setLogger(format, null)` being a synchronous flush), a `--log-file` keeps being written alongside the sink, and every `ggml_log_level` passes through unchanged. The Java half (`LlamaLoggerTest`, model-free) proves the JNI trampoline on top of it. |
| `src/test/cpp/test_jni_helpers.cpp` | 70 | All functions in `jni_helpers.hpp` using a zero-filled `JNINativeInterface_` mock (incl. the `utf8_to_jstring_impl` byte-array string path: emoji byte-preservation, truncated-UTF-8 replace-not-throw). Seven of them pin `jni_guard_impl` — the JNI exception boundary every `Java_*` entry point runs inside — including the `catch (...)` arm that is the only backstop for a non-`std::exception` type, and its two refusals (never `ThrowNew` over a pending Java exception, never with a null class). |
| `src/test/cpp/test_tts_wav.cpp` | 2 | The in-memory WAV writer `pcm_to_wav16_bytes` in `tts_wav.hpp` (WAV header/payload + little-endian clamping) — our own code, not upstream. The Qwen3-TTS pipeline it pairs with (`mtmd_helper::gen_audio`) is entirely upstream-owned (no project-side DSP to unit-test here). The load path is additionally covered by `test_tts_params.cpp` (3 tests over `tts_params.hpp`'s `build_tts_params`, plus 2 pinning the upstream `-1` default it depends on), which pins the CPU-thread resolution whose absence used to crash the JVM on every platform — see the `TODO.md` entry for the mechanism. End-to-end coverage is `TtsIntegrationTest`, which is model-gated. |
| `src/test/cpp/test_tts_params.cpp` | 13 | The **three** builders every hand-assembled `common_params` goes through: `build_tts_params` (`tts_params.hpp`), `build_train_params` (`train_params.hpp`) and the shared `jllama::resolve_cpu_params` (`cpu_params.hpp`). Each builder is guarded separately on purpose — testing the resolver alone does **not** cover its call sites, because `train_engine.cpp` is compiled into `jllama` only, never into `jllama_test`, and `LlamaTrainerIntegrationTest` is gated on `net.ladenthin.llama.train.model`, which no CI job sets. Without these the JVM-abort bug could regress in the trainer on every platform, unseen. |
| `src/test/cpp/test_model_split.cpp` | 7 | The two `load_tensors()` split helpers that `patches/0012` extracts out of llama.cpp's `src/llama-model.cpp` — `llama_model_splits_normalize` (proportional split, single device, and the zero-sum case that used to produce NaN, **and the cancelling `--tensor-split` case** — `-ts 1,-1` reaches the identical line on any backend with no GPU memory pressure at all) and `llama_model_splits_select_device` (every layer maps to a real device index; malformed split points throw a message that names the function, the layer, the index and the split values instead of libc++'s bare `"vector"`). **This is the runnable guard for `0012`**: the patch also ships an upstream `tests/test-model-split.cpp`, but a FetchContent subproject builds with `LLAMA_BUILD_TESTS=OFF`, so that one is applied-but-never-compiled here. This file is the only place the two functions are linked in CI, on every platform — so a bump that drops the patch fails the `C++ Tests` build outright rather than resurfacing as one red macOS Java job. It is the one test file that includes an **internal** upstream header (`llama-model.h`, via the `${llama.cpp_SOURCE_DIR}/src` include dir added for it), which is deliberate: a signature drift should fail loudly at compile time. |
| `src/test/cpp/test_prefetch.cpp` | 5 | **The runnable guard for `patches/0017`** (the x86 prefetch helper). The patch fixes two defects in the four `_mm_prefetch` calls of `ggml_vec_dot_q4_0_q8_0`'s SSSE3-without-AVX branch, and only one of them is a build error. The **type** half is caught by the compiler, but only by a recent one: plain clang on a `*-windows-msvc` target gets MSVC's `char const *` declaration (its own casting macro sits behind `#ifndef _MSC_VER`), and the resulting `-Wincompatible-pointer-types` is an **error by default from clang 22 on** -- measured on the real file at `-msse4.2`: 4 warnings on 20.1.8, 4 errors on 22.1.3 and 23.1.3, and 0/0 on all three once patched (`-msse4.2 -mavx` compiles either way, taking the AVX branch). So `GGML_CPU_ALL_VARIANTS=ON` stops building with plain clang on Windows once the compiler is new enough -- via the **`sse42`** variant, the only one that reaches `#elif defined(__SSSE3__)` (the `x64` baseline has no SSSE3 and takes the scalar branch; everything richer has AVX). Nothing in a test can cover that half. The **arithmetic** half is what this file guards, and it is silent: `&x[ib] + sizeof(block_q4_0)` adds `sizeof()` *elements*, so with the sizes `ggml-common.h` pins by `static_assert` (18 and 34) the prefetch targets +324 and +1156 bytes instead of +18 and +34, and the `+ 2 *` pair +648 instead of +36 -- wrong on **every** platform, including the ones where the call compiles. `ggml_prefetch_at()` casts to `const char *` first, which is what makes the offset byte-wise. **The case that makes the file worth having:** writing the cast around the whole expression, `(const char *)(&x[ib] + sizeof(block_q4_0))`, silences the compiler and keeps the wrong address -- that form builds and fails here. Verified both ways: 5/5 green against the patched header, 5/5 red against a deliberately element-wise one. Includes two internal headers (`ggml-common.h` for the block types, `prefetch.h` for the helper) via the `${llama.cpp_SOURCE_DIR}/ggml/src` and `.../ggml/src/ggml-cpu/arch/x86` include dirs added for it. **Measured, so the upstream report does not overclaim:** the wrong prefetch distance costs no measurable throughput (Ryzen 7 5800H forced onto the `sse42` kernel, 0.6B Q4_0: pp512 101.3 vs 100.9, tg128 37.3 vs 37.0, inside each other's error bars) -- it is a correctness and portability fix, not a performance one. |
| `src/test/cpp/test_kolibri1.cpp` | 5 | **The runnable guard for `patches/0016`** (Kolibri-1). Writes tiny random `kolibri1` GGUFs with the public `gguf` API, loads them through the real library on the CPU and compares every logit -- one batch, and token by token through the iSWA KV cache -- with an independent double-precision reference written from Aleph Alpha's vLLM implementation, not from the patch. The layer pattern puts a full-attention (NoPE) layer between sliding ones, the sequence is longer than the window, and the expert biases are large enough that DeepSeek-V3's router would pick other experts (one test asserts that, so the comparison cannot pass vacuously). Covers both published GGUF dialects (gating 2 + `qwen2` + `output.weight`; gating 5 + `kolibri1` + tied output), renormalized routing, and the rejection of another gating function. **What happens when upstream supports Kolibri-1 and `0016` is dropped** -- the file compiles unchanged (public `llama.h`/`gguf.h`/`ggml.h` only, nothing from the patch), but its tests split in two. The **numerical comparison** against the reference must stay green, and `TheDeepSeekRouterWouldGiveDifferentLogits` never touches the library at all; a red comparison means upstream computes something else than Aleph Alpha's reference (router, window boundary, positional encoding) and is a finding, not a test to adjust. The **GGUF format** each test writes -- architecture name, key and tensor names, `expert_gating_func`, pre-tokenizer, optional `output.weight` -- is upstream's converter's decision, not ours: `AfmoeDialectMatchesTheReference` (gating 2) is the likeliest to go red if upstream follows the gating-5 port, `RenormalizedRoutingMatchesTheReference` writes no gating key at all, and `AnotherGatingFunctionIsRejected` assumes a rejection upstream need not make. A red format row is a real signal as well -- the published GGUFs of that dialect stop loading without the patch -- so decide it deliberately (a small compatibility patch, or document that those files must be reconverted) and only then move the row to upstream's format: one `{gating, pre, ...}` entry per test, with the reference untouched. |
| `src/test/cpp/test_model_flags.cpp` | 4 | **The contract between the Java CLI-flag registries and llama.cpp's server argument parser.** CMake reads `ModelFlag.java` + `ModelOption.java` (`cmake/extract-java-wire-names.cmake` → a generated header of `{name, contract}` pairs), and this file asserts every `SERVER_PARSER` name is in `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options`. It exists because **no Java test can catch this class**: `ModelFlagTest`/`ModelParametersExtendedTest` pin the *string mapping* (`hasKey("--mlock")`), never that llama.cpp still accepts the string, so they stay green forever while the flag is dead — and `common_params_parse` treats an unregistered option as a hard error, so the affected builder method makes the model **unloadable**, not merely ineffective. **A grep over `arg.cpp` is not a substitute**: `--grp-attn-n`/`-w` are present there at every pinned tag but `set_examples()`-scoped to `LLAMA_EXAMPLE_COMPLETION`/`PASSKEY`, so the server parser rejects them exactly like a deleted flag — only the real option table sees that. `--vocab-only` is the one exemption, and it declares itself `CliContract.PROJECT_PSEUDO` on its own constant rather than appearing in a list inside this file; the test asserts such a name is **still unknown** to the parser (an exemption upstream later registers would be hiding a real check) and that the exempt set is non-empty. |
| `src/test/cpp/test_wire_contracts.cpp` | 6 | **The same contract for the two quieter surfaces.** `RequestField` against `server_schema::make_llama_cmpl_schema(...)` (5 tests) and `TrainingField` against `jllama_train::config_keys()` (1 test). Both receivers *silently ignore* an unknown key — the schema skips it, `train_engine.cpp` reads with `j.value(key, default)` and falls back — so a dead field produces no error anywhere and every string-mapping test keeps passing. `OAI_LAYER`-declared keys (consumed by `oaicompat_*_params_parse` before the schema) are exempt from the schema check, and are checked **both** ways: still unknown to the schema (the inverted check), and read by at least one upstream reader-shaped site (the configure-time sweep — this is what caught `chat_template`, a key a public builder wrote and nothing read). See [`docs/history/parameter-wire-surface.md`](docs/history/parameter-wire-surface.md). |
| `src/test/cpp/test_rpc.cpp` | 28 | **The runnable guard for `patches/0015` and for `rpc_support.hpp`.** The device-selection rules (pure, literal inputs: `--rpc` endpoints accumulate across repeated options, an explicit `--device`/`-dev` is never overridden, stale RPC devices are replaced by the local GPUs chosen the way llama.cpp's default does — duplicates by device id dropped, iGPUs only without a discrete GPU, `none` on a CPU-only host). Then the **real** ggml-rpc client and server over loopback, no model: a `mul_mat` graph computed on the RPC backend equals the local CPU result; `ggml_backend_rpc_stop_server()` ends the server, frees the port and disconnects a client that is still connected; an unreachable or malformed endpoint is `nullptr`/`std::invalid_argument` instead of an abort; a registered server that went away is re-checked on the next registration; its device reports 0/0 memory instead of aborting; and a stale server is kept out of a later load that did not ask for it. The loopback server picks a free port from a range so parallel jobs do not collide, and serves the **CPU** by name — a served GPU that lacks an op aborts the server (see the RPC section). Also: server devices chosen by name (case-insensitive, deduplicated), an unknown name listing the available devices, and a registered RPC device refused; the mmproj device pinned the way clip would choose it minus the stale server; and `exclude_stale_devices()`, the params-level guard for TTS and the trainer. |

| `src/test/cpp/test_native_server_shutdown.cpp` | 3 | **The runnable guard for the shutdown-handler guard in `patches/0006`/`0007`.** Runs the real `llama_server()` in router mode over an empty `--models-dir` on an ephemeral loopback port (no model, no worker) and stops it the way `native_server.cpp` does. Pins a clean stop (exit code 0), that `llama_server_request_shutdown()` **after** the server returned is a no-op (the deterministic form of the CI `SIGSEGV` — it crashed every run before the fix), and that requests hammered from another thread while the server tears down are safe (5 rounds). Compiled on non-Android only, like `server.cpp` itself. |

**Current total: 603 tests (all passing).**

#### Upstream source location (in CMake build tree)

llama.cpp is fetched via CMake FetchContent, pinned to `GIT_TAG b11512`.

**GoogleTest** is a separate `BUILD_TESTING`-only FetchContent (`GIT_TAG v1.18.0`), used solely
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
   - A function one of the local `patches/` adds to llama.cpp → its own file, e.g.
     `test_model_split.cpp` for `0012`. Give every patch that introduces a callable a
     guard here: upstream tests carried by a patch are **not** compiled (a FetchContent
     subproject sets `LLAMA_BUILD_TESTS=OFF`), so this is the only place a dropped patch
     reds CI on every platform instead of on whichever job happens to hit it.
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

`server_schema::eval_llama_cmpl_schema(vocab, params_base, logit_bias_eog, data)`
can be called with `nullptr` vocab **if the JSON does not trigger grammar/preserved_tokens
tokenisation** (those are the only vocab-dependent paths).  This lets us test the full
parsing pipeline including error throws.  **It takes four arguments** — the `n_ctx_slot`
parameter was dropped at b10275; a five-argument call has not compiled since.  `test_server.cpp`
wraps it in a `parse_params` helper, which is the form to copy:

```cpp
namespace {
task_params parse_params(const json &data) {
    common_params params_base;
    std::vector<llama_logit_bias> no_bias;
    return server_schema::eval_llama_cmpl_schema(nullptr, params_base, no_bias, data);
}
} // namespace

// test: a value inside the hard limits round-trips
EXPECT_EQ(parse_params({{"sse_ping_interval", 5}}).sse_ping_interval, 5);

// test: out-of-range and malformed values throw std::invalid_argument
EXPECT_THROW(parse_params({{"repeat_last_n", -1}}), std::invalid_argument);
EXPECT_THROW(parse_params({{"dry_sequence_breakers", json::array()}}), std::invalid_argument);
```

Note what the second line pins: `repeat_last_n` and `dry_penalty_last_n` carry
`set_hard_limits(0, INT32_MAX)` since **b10273**, so `-1` is simply out of range.  It does
**not** expand to the slot context size any more — an older version of this section said it did.

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
- Platform-specific native libraries must be pre-built and placed under `src/main/natives/` before packaging for distribution (CI does this; see "Natives jars").

## Javadoc Conventions

See [`../workspace/policies/javadoc-conventions.md`](../workspace/policies/javadoc-conventions.md).

## Java 8 bytecode floor — what may ship

This artifact targets **Java 8** (`release 8`), so **every class a consumer's JVM can load must be
class-file major 52 or lower**. Two entries in `llama/pom.xml` exist only for that, and both are
easy to undo by accident:

- **`slf4j-simple`, not logback, is the shipped SLF4J binding.** Every logback release from 1.4.0 on
  is Java 11 bytecode, so `LogbackServiceProvider` cannot load on Java 8 — SLF4J's `ServiceLoader`
  finds it at startup and the JVM throws `UnsupportedClassVersionError`. The Java 8 line (1.3.x) is
  end-of-life (last release 1.3.16, 2025-10-29) and every logback CVE disclosed since has been fixed
  only in 1.5.x/1.6.x with no backport, so it is not an option either. `slf4j-simple` is six classes
  from the same release train as `slf4j-api`, with no configuration or socket layer for a CVE to
  live in. Configure it with a classpath `simplelogger.properties` or `-Dorg.slf4j.simpleLogger.*`.
- **`checker-qual` is `provided` scope, not a pinned old version.** Its annotations are major 55
  from 4.0.0 on and `@Retention(RUNTIME)`, so anything reflecting over an annotated element (Jackson
  does) loads them and a Java 8 JVM throws `UnsupportedClassVersionError`. **Pinning the shipped copy
  to the last Java 8 line (3.55.1) does not work** — that was shipped in #411 and broke `main`
  outright: the Nullness Checker resolves its own qualifiers through javac's symbol table, i.e. the
  *compile classpath*, so a 3.x checker-qual under the 4.x processor fails every build with
  `Could not load type: org.checkerframework.framework.qual.DoesNotUnrefineReceiver`. Processor and
  qualifiers must share a major version. `provided` satisfies both constraints: 4.3.0 on the compile
  classpath where the checker needs it, and excluded from consumers' graph **and** from the fat jar
  (`jar-with-dependencies` takes scope `runtime`), so no checker-qual class of any version ships.
  `<optional>true</optional>` would not have been enough on its own — that descriptor filters on
  scope only. Safe because no source imports `org.checkerframework`.

**The gate: `.github/verify-bytecode-version.sh`.** Kept **byte-identical** across java-llama.cpp /
BitcoinAddressFinder / streambuffer / srcmorph (listed in `.github/shared-files.sha256`, checked by the `shared-files` job).
It opens every `.class` in every jar it is given and fails on any whose class-file major version
exceeds `--max-major`:

```bash
.github/verify-bytecode-version.sh --max-major 52 [--allow '<jar>:<entry>']... <jar-or-dir>...
```

Paths may be jars or directories (searched recursively for `*.jar`), so one invocation covers a whole
artifact set — here the classes jar, all 27 natives jars and every `all-<os>-<arch>` fat jar. `module-info.class`
and `META-INF/versions/**` are skipped unconditionally: a classpath JVM never loads either, which is
why a `release 9` `module-info` is fine. `--allow` is a repeatable glob matched against
`<jar-basename>:<entry-path>` for anything else that must be tolerated. Exit codes: 0 clean,
1 violations, **2 nothing to scan** (an empty input is a failure, never a pass — the first version of
this check reported a clean pass over a directory a failed build had left empty).

It runs twice: in the `package` job over `llama/target` (every jar plus the default fat
jar, as early as they exist), and again in the two Linux rows of `smoke-fatjar` over the downloaded fat jar —
`package-fatjars` rewrites those zips, and they are the artifacts users actually download.

**Surefire AND PIT both exclude `org.slf4j:slf4j-simple` from the test classpath**
(`classpathDependencyExcludes` in each). Runtime scope is on the test classpath too, and LogCaptor
(test scope) requires logback specifically — with both providers present SLF4J's `ServiceLoader`
picks one arbitrarily and the LogCaptor assertions fail. The exclusions leave logback the sole
provider in tests and do not touch the artifact: the jar and the fat jar still ship slf4j-simple.

**The PIT half is not redundant, and forgetting it is a trap worth naming.** `spotbugs:check` and
`spotless:check` bind to `verify`, so `mvn test` misses them — a different trap. This one is
sharper: **PIT builds its own classpath and never reads Surefire's configuration**, so a
Surefire-only exclusion leaves the mutation run with two providers. It then aborts the whole gate
with *"N tests did not pass without mutation … requires a green suite"* — a red gate on a suite
Surefire had just reported green, which reads like a PIT bug rather than a classpath one. This
shipped once (#411 added the binding with only the Surefire exclusion; the five affected
LogCaptor tests reddened `Java Tests Ubuntu` and the failure went unseen because every publish run
in between was cancelled). The sibling repo srcmorph hit the identical thing and solved it a
different way — there both providers arrive transitively, so it excludes at the dependency instead.
Same rule either way: **one SLF4J provider on the test classpath, enforced everywhere a test
classpath is built.**

## SpotBugs Suppressions

See [`../workspace/policies/spotbugs-suppressions.md`](../workspace/policies/spotbugs-suppressions.md).

**`spotbugs:check` binds to `verify`, so neither `mvn test` nor `mvn package` runs it.** A change
can pass every local gate and still red the pipeline in the `Code style (spotless) + package graph`
job. Before pushing, run what that job runs:

```bash
mvn -B --no-transfer-progress -f llama/pom.xml -DskipTests -Denforcer.skip=true compile spotbugs:check
```

This has actually shipped twice. Most recently the `--flash-attn` / `--lazy-mode` work reddened
`main`: the design-intent `OCP_OVERLY_CONCRETE_PARAMETER` suppression in
`llama/spotbugs-exclude.xml` lists methods **by name**, so renaming `setTensorReadLazy` to
`setLazyMode` left a dead entry while the new `setLazyMode` and `setFlashAttn` were uncovered. **Any
rename or addition of an enum-valued `ModelParameters` setter needs that list updated in the same
commit** — `setLoadMode` was added to it for exactly this reason — the same "FQN not updated after a rename" class as the stale PIT `targetClasses` and
`CMakeLists.txt` OSInfo repairs.

**Half of that is now a test.** `WireNameRegistryTest.everyOcpSuppressionStillNamesAnEnumValuedSetter`
asserts every method named in those suppressions still exists as an enum-valued setter, which is the
half nothing else covers: a suppression for a method that no longer exists is silently inert, so the
next real finding on the renamed method arrives as a surprise. The opposite direction — a flagged
setter *missing* from the list — already reds `spotbugs:check`, and is not derivable by reflection
anyway: SpotBugs raises OCP only when a method uses nothing beyond the interface, so `setPoolingType`
(compares a concrete constant) and `withMiroStat` (calls `ordinal()`) are legitimately absent.

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
318/318 killed, 0 NO_COVERAGE, test strength 100% in a fixture-less sandbox; the former
audio-fixture gotcha is resolved).
**PIT builds its own classpath and reads none of Surefire's settings**, so everything the tests need
from Surefire's configuration is repeated on the PIT plugin: the `slf4j-simple` exclusion (see "Java 8
bytecode floor") and `src/main/natives` as `additionalClasspathElements` (the natives left
`target/classes` with the natives jars; without it a model-backed test in a target-test package aborts
the gate in CI, where its model exists). `TestClasspathTest` is in PIT's `targetTests` and fails under
both runners when the native library is missing from the classpath, when the tests run in a named
module (Surefire's default with a `module-info.class`, hence `useModulePath=false`), or when Jackson is
not on `java.class.path` (which the router worker JVM is started with).

**`net.ladenthin.llama.value.*` is a target at `mutationThreshold` 100**, so a new getter on a
`value` type needs its own test or the gate reds — the `ServerMetrics` counters added for the
`getMetrics()` merge are covered by `ServerMetricsTest`.

**`parameters.JsonParameters` is on the gate too**, because it carries the one-JSON-value invariant
rather than plumbing. The rest of the `parameters` package is deliberately **not**: ~200 one-line
builder setters would add cost without signal. Getting `JsonParameters` to 100% needed one test more
than expected — the bounded excerpt in its rejection message is observable only *exactly* at the
limit, so it is pinned from both sides.

## JPMS Module Descriptor

This repo ships a `module-info.java` compiled in a separate `release 9` execution. Javadoc
currently runs in **classpath mode** (javadoc `<source>` is `1.8`), which is the *only* thing
keeping it clear of the JPMS module-mode javadoc trap that bit BAF. **Before raising the Java /
javadoc source level to ≥ 9, read**
[`../workspace/policies/jpms-module-descriptor.md`](../workspace/policies/jpms-module-descriptor.md).

## Repository layout — Maven reactor (`llama/` + `llama-langchain4j/` + `llama-kotlin/` + `llama-platform/`) + the `llama-android/` Gradle build

The repo root is a thin **aggregator/parent POM** (`net.ladenthin:llama-parent`,
`packaging=pom`) with four modules:

- **`llama/`** — the native JNI core (`net.ladenthin:llama`). *All the core sources and build
  files live here now:* `llama/src/`, `llama/CMakeLists.txt`, `llama/cmake/`, `llama/patches/`,
  `llama/pom.xml`, `llama/spotbugs-exclude.xml`, `llama/lombok.config`, `llama/.clang-format`.
  Its published coordinates are unchanged (`net.ladenthin:llama`), so consumers are unaffected.
- **`llama-langchain4j/`** — the LangChain4j adapters (see below).
- **`llama-kotlin/`** — the Kotlin coroutines façade (see "Android AAR + Kotlin façade" below).
- **`llama-platform/`** — a pom-packaging dependency list: the classes jar plus the CPU natives jars
  of every desktop platform (see "Natives jars" above).

All modules inherit the single `<version>` from the parent, so they **ship in lockstep by
construction** (no CI guard needed). The parent also holds the shared `release` profile (GPG +
Central Publishing) and a `<pluginManagement>` with every plugin version more than one module uses
(compiler, jar, resources, surefire, source, javadoc, gpg, central-publishing -- a module names such a
plugin without a version; without it a module that pins nothing builds with the default of whatever
Maven runs it, which differed between CI and a local build), so one reactor `mvn -P release,natives deploy` signs and publishes all five
Maven artifacts (`llama-parent` pom, `llama` with its natives jars, `llama-langchain4j`,
`llama-kotlin`, `llama-platform` pom) at the same version.

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
versioning** here), so a version change must be applied to **all five poms in lockstep**:

- `pom.xml` (root) — `<version>`
- `llama/pom.xml` — `<parent><version>`
- `llama-langchain4j/pom.xml` — `<parent><version>`
- `llama-kotlin/pom.xml` — `<parent><version>`
- `llama-platform/pom.xml` — `<parent><version>`

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
- **`llama-atmosphere-agent/pom.xml`** — its own `<version>`, which must equal the reactor's
  (standalone project outside the reactor, so `versions:set` skips it; `check-natives.py` fails until
  they agree), plus the JBang coordinates and the fat-jar filename
  `llama-<version>-jar-with-dependencies.jar` in the root README's "Local coding agent" section and
  the project's own README.
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
   profile signs and publishes every reactor module together at the same version).
   A separate **`test-java-llama-langchain4j-integration`** job runs the model-backed tests
   (chat/streaming/embedding/scoring adapters) by **reusing** the shared GGUF cache
   (`gguf-models-v1`, restore-only — no extra download) and the `natives-cpu-linux-x86-64` native
   artifact: it `needs: [crosscompile-linux-x86_64, download-models]` (so the cache is already
   populated and it runs in parallel), installs parent+core (classes), points the test JVM at the
   downloaded library with `-Dnet.ladenthin.llama.lib.path`, and passes the already-cached chat (`REASONING_MODEL_NAME`), nomic-embedding and
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
  `crosscompile-android-x86_64` dockcross job, which also ships as the `cpu-android-x86-64` natives
  jar; the OpenCL flavor stays arm64-only —
  Adreno is Qualcomm ARM hardware), a
  `minSdkVersion 28` manifest (AGP enforces the floor on consumers), and consumer R8/ProGuard
  rules (`consumer-proguard.txt` → `proguard.txt` in the AAR; keeps `net.ladenthin.llama.**` for
  the JNI `FindClass`/Jackson reflection surface). The AAR's `classes.jar` is the
  **byte-identical Maven-built classes jar** minus `module-info.class` (D8 rejects it) — it carries no
  natives, which ship as separate natives jars; on Android `LlamaLoader`
  resolves via `System.loadLibrary("jllama")`, which finds the AAR-installed `.so` — no loader
  change was needed. Built by the **standalone plain-Gradle build** in `llama-android/`
  (see "Repository layout" for why it is not a Maven module); the POM mirrors the core's
  compile-scope deps (jackson/slf4j-api/jspecify/checker-qual, versions parsed from
  `llama/pom.xml` — deliberately NOT the SLF4J binding, which is the JVM-only runtime dependency).
- **`net.ladenthin:llama-kotlin`** — Maven reactor module; pure-Kotlin (2.4, jvmTarget 1.8)
  coroutines façade: `generateFlow`/`generateChatFlow` (cold `Flow`, source closed on
  completion/error/cancellation) and `completeSuspend`/`chatSuspend`/`chatCompleteTextSuspend`/
  `embedSuspend` (`completeSuspend` wires coroutine cancellation into the cooperative
  `CancellationToken`). The core dep is **provided-scope** so Android consumers pair it with the
  AAR instead of transitively pulling the fat desktop JAR. 6 model-free unit tests fake the
  `Iterable & AutoCloseable` seam (`closeableIterableFlow`/`withCancellationToken` internals).

**16 KB page-size invariant (Google Play, Android 15+ targets):** `llama/CMakeLists.txt` pins
`-Wl,-z,max-page-size=16384` in the Android guard block, and `.github/buildcheck/nativedeps.py`
(`verify-native-deps.py`, run by the `package-android-aar` job on the staged libraries and by
`package` on every natives jar) asserts every LOAD segment of an Android `.so` is 16384-aligned,
reading the ELF program headers itself — a dockcross toolchain bump cannot silently regress Play
compatibility.

**dlopen-ability invariant (bionic-only DT_NEEDED):** the same Android guard block sets
`GGML_OPENMP OFF` (ggml uses its std::thread pool — Android ships no `libomp.so`; same trade
as the Windows-arm64 clang-cl job) and links `-static-libstdc++` (no `libc++_shared.so`
dependency — that runtime only exists when an app packages it itself). Without both, the
dockcross cross-clang emitted `DT_NEEDED` on `libomp.so` + `libc++_shared.so`, which made
`System.loadLibrary("jllama")` fail with `UnsatisfiedLinkError` on every device (caught by the
`test-android-emulator` job; the released 5.0.5 arm64 lib had the same latent defect). The
`nativedeps.ALLOWED` holds each Android directory to an exact `DT_NEEDED` list (`libc.so libm.so
libdl.so liblog.so libandroid.so`, plus `libOpenCL.so` for `Linux-Android/aarch64/opencl`) — the
same allowlist mechanism as every desktop CPU build, where the AAR job used to carry a copy of its
own; the AAR job additionally asserts the AAR's `jni/` library is the staged one. And
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

## Local coding agent with Atmosphere (`llama-atmosphere-agent/`, standalone)

A copy-and-run terminal agent (console, `--web`, `--acp`) pairing Atmosphere's OpenAI-compatible
agent runtime with this project's `OpenAiCompatServer`. A **standalone Maven project** (Java 21,
Atmosphere's floor), **not** a reactor module: an application, built without a parent so the folder
can be copied out and run. **Published to Maven Central** at the core's version as a thin jar whose
pom names `llama-platform` as a runtime dependency, so `jbang net.ladenthin:llama-atmosphere-agent:<v>`
starts it with the CPU natives of every desktop platform. Everything about the
agent itself -- the REPL, approval gate, file tools, JLine console and its eight carried JLine fixes,
the three front ends -- is in **[`llama-atmosphere-agent/CLAUDE.md`](llama-atmosphere-agent/CLAUDE.md)**,
which Claude Code loads when working in that directory. What matters from the rest of the repository:

- **Maven Central:** a step of its own in `publish-snapshot` / `publish-release`, right after the
  reactor deploy (`-f llama-atmosphere-agent/pom.xml -P release deploy`), resolving the core and its
  natives jars from the local repository that deploy just filled. Its CI jobs install only the classes
  and the `llama-platform` pom and pass `-Dllama.natives=none` (a profile that excludes everything that
  pom names), since no natives jar exists before `package`.
- **Release asset:** `llama-atmosphere-agent-<llama.version>-jar-with-dependencies.jar`, built
  **without** the core; its manifest `Class-Path` names the four `all-<os>-<arch>` fat jars and the
  default fat jar. Renaming a core fat jar means updating that list -- `check-natives.py` holds it
  to the fat-jar targets of `natives.csv`, and `smoke-agent-linux` launches the pair.
- **CI:** the model-free job, the model-backed integration job and `smoke-agent-linux` all gate both
  publish jobs.
- **Version bump:** the pom's `<version>` is the reactor's, and `versions:set` does not touch this
  standalone pom -- bump it by hand (`check-natives.py` fails until it agrees) with the JBang
  coordinates and fat-jar filename in the two READMEs; `llama.version` follows (`${project.version}`).
- **The one core change it needed:** `OpenAiBackend`, `ChunkSink` and
  `OpenAiCompatServer(OpenAiBackend, OpenAiServerConfig)` are public, so the agent's tests can drive
  the real server without a model. Keep them public.

## Android app "LLM Service" (`android-llmservice/`)

A fully-offline on-device chat app consuming the `llama-android` AAR + `llama-kotlin` façade
(applicationId `net.ladenthin.android.llmservice`), a **standalone Gradle/AGP build**, not a reactor
module and not published to Maven Central. Details -- structure, AGP/Gradle pins, signing, i18n --
are in **[`android-llmservice/CLAUDE.md`](android-llmservice/CLAUDE.md)**. Two rules that reach
outside the directory: **`requirements.md` is the spec of record** (any behaviour change updates it in
the same commit), and its two CI jobs, `build-android-llmservice` and `test-android-llmservice`, are
**not** publish gates (both are named with a reason in `.github/release-gate-exemptions.txt`).

## Dependency Convergence Pinning

`dependencyConvergence` is enabled (maven-enforcer, `llama/pom.xml`). Convention for pinning a
direct-vs-transitive version mismatch in `dependencyManagement`, the
`excludedScopes=[test,provided]` enforcer default gotcha (jspecify/logback-classic here are
pinned defensively because of it — see that file), and merge-discipline guidance are in
[`../workspace/policies/dependency-convergence-pinning.md`](../workspace/policies/dependency-convergence-pinning.md).

## Open TODOs

Open TODOs for this repo live in [`TODO.md`](TODO.md). Cross-repo status
tracking lives in [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md).
