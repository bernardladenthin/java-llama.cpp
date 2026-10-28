# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b11968**

## Natives jars: library jars and GPU module jars (`.github/natives.csv`)

`net.ladenthin:llama` is the **Java classes only**. The native code ships as jars of the same
artifact, classifier `<backend>-<os>-<arch>`, each holding exactly one directory
`net/ladenthin/llama/<OS>/<ARCH>/<backend>/` -- 23 today, of two kinds (the `kind` column of
`natives.csv`):

- **8 library jars**, one per platform (`cpu-*` for Linux x86-64 / aarch64 / s390x, Android aarch64 /
  x86-64, Windows x86-64 / aarch64; `metal-macos-aarch64`): `libjllama` with llama-common and the
  server linked in, plus -- on the modular platforms (`JLLAMA_BACKEND_DL`, see "Modular natives") --
  `libllama` and `libmtmd` as shared libraries of their own (upstream's C API, layer A), ggml's
  shared libraries, `ggml-rpc` and the CPU backend modules (on the static platforms llama and mtmd
  are inside `libjllama`). Exactly one is loaded, the one of
  the running platform; `LlamaLoader.LIBRARY_BACKENDS` (`metal`, then `cpu`) names the directory.
- **15 GPU module jars** (`cuda13`, `rocm`, `sycl`, `vulkan`, `opencl`, `openvino` per OS/arch,
  `LlamaLoader.MODULE_BACKENDS`): each holds **one** ggml backend module (`libggml-cuda.so`,
  `ggml-vulkan.dll`, ...) built with `JLLAMA_MODULE_ONLY`, and the two list files, nothing else. They
  are **additive**: the loader extracts the library directory and then every module jar of the
  platform it finds on the classpath into the same extraction directory, and `JNI_OnLoad` has ggml
  load whatever lies there (`ggml_backend_load_all_from_path`). A module whose vendor runtime is
  missing fails to open -- silently inside ggml, which is why `JNI_OnLoad` logs the backend registry
  and the devices at INFO and `LlamaLoader` logs `native backend '<lib>' loaded from <dir> with GPU
  module(s) a, b` -- and the model runs on the CPU modules. **Two measured exceptions, both in
  [`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md), and both reached by a
  consumer doing nothing wrong:** OpenVINO's CPU plugin opens fine and contributes a *device*, so on
  a host without an Intel accelerator it is preferred and runs **17x/172x slower** than the CPU
  modules; and a CUDA driver that is present but too old to JIT the module's PTX lets ggml offload
  every layer and then call `abort()`, killing the JVM instead of falling back (a Tesla T4 on driver
  595.91.07 -- Turing, A100 and Hopper ship as PTX only, so they depend on the driver's JIT). Several together are fine: llama.cpp
  dedups the devices by `device_id`, so the same GPU seen through CUDA and Vulkan is one device.
  **`openvino` is opt-in** (`LlamaLoader.OPT_IN_BACKENDS`): its jar on the classpath does nothing
  until `net.ladenthin.llama.backend` names it, and the loader says so on stderr. Every other module
  is additive because the rule above holds for it; OpenVINO breaks it at both ends (measured, see
  [`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md)) -- its CPU plugin opens
  and contributes a *device*, so llama.cpp prefers it and a host without an Intel accelerator drops
  from 490.4/34.5 to 29.4/**0.2** tokens per second, while without an OpenCL ICD loader
  (`libOpenCL.so.1`, which the OpenVINO archive does not ship) it does not open at all.
  **The `optin` column of `natives.csv` is where that is declared**, and `check-natives.py` holds
  `LlamaLoader.OPT_IN_BACKENDS` to it in both directions: a backend the list calls opt-in while the
  loader does not would ship a jar that takes over silently, and the reverse would leave a jar inert
  with no row saying why. The two smoke layers read the same column -- the model-backed rows do not
  expect an opt-in module in the loader line, while `smoke-natives-jars.sh` names every module
  through the property, which is the only place in CI that loads one.
  `net.ladenthin.llama.backend=<a>,<b>` is a **filter** over the module jars (`cpu` = none); a name
  that is no module backend, or whose jar is not on the classpath, fails the load with a message
  naming the jar to add. **A module jar needs the library jar of the same build**: every natives
  directory carries `jllama-build.txt` (`llama.cpp=<tag>`, `backend=`, `kind=` plus the provenance keys
  `llama.cpp.commit=`, `classifier=`, `jllama.version=`, `jllama.commit=`, `patches=`, `patches.sha256=`,
  `compiler=`, `toolkit=`, `signed=`, written by `cmake/jllama-build-info.cmake`, which also writes the
  CycloneDX `jllama-sbom.json` next to it; `LlamaModel.getJllamaBuildInfo()` exposes the keys), and
  `requireSameBuild` refuses a module whose tag differs from the library's **before** anything is
  extracted -- ggml checks only the module's exports at load time, so a mismatched ggml ABI would
  surface later and worse. The extraction directory is keyed by the library's jar entry **and** the
  module list (`extractionDirectoryName(backend, library, modules)`: `-m<hash>` over the module URLs
  and their entry keys), so a classpath with other modules gets a directory of its own and the
  reuse-across-starts logic ("Modular natives") never mixes two sets.

`net.ladenthin:llama-platform` is a **pom-packaging** module (a dependency list, no jar, nothing to
upload) naming the classes jar plus the library jars of every desktop platform; consumers use it with
`<type>pom</type>` and add GPU module jars next to it. **There are no fat jars any more** -- not the
default CPU one, not the `all-<os>-<arch>` ones, not the agent's: with modular natives a jar carrying
every backend of every platform is the opposite of the point, and a consumer's classpath is what
Maven resolves. The replacements are `jbang --deps net.ladenthin:llama:<v>:<classifier>
net.ladenthin:llama:<v>` (the classes jar's `Main-Class` is `ServerLauncher`) and
[`examples/server/pom.xml`](examples/server/pom.xml) (`llama-platform` + one profile per GPU module
jar, named after its classifier, and a local `-P assembly` for a machine without Maven; its
`<llama.version>` is held to the README's install snippet by `check-natives.py`). The GitHub Release
attaches the Maven artifacts with the `.asc` maven-gpg wrote for Central, and nothing else.

This replaced two older layouts: up to 5.1.0 the default jar carried the CPU natives of every
platform at `<OS>/<ARCH>/` and each GPU classifier was a complete replacement jar; in the first
5.2.0 snapshots each GPU jar was a complete library with the CPU backend linked in, tried in a fixed
priority order (`BACKEND_PRIORITY`) until one loaded, and the `all-<os>-<arch>` fat jars merged them.
Both are gone: the GPU jars hold a module each, there is no priority order (every module is loaded),
and 32-bit Windows (`cpu-windows-x86`, `msvc-windows-x86`), the MSVC jar (`msvc-windows-x86-64`,
still **built and tested**, see "Windows natives") and the second SYCL precision (`sycl-fp32`) went
with the fat jars (6.0.0, before the release).

**The list is `.github/natives.csv`** (classifier, directory, library, platform yes/no, kind, optin yes/no) — the
one place a natives jar is declared, in the same spirit as `models.csv`. Everything else either reads
it or is checked against it:

| Place | Relation to the list |
|---|---|
| `llama/CMakeLists.txt` | names the backend directory (`JLLAMA_BACKEND`: GGML flag → name; `metal` = Apple + `GGML_METAL`), writes to `src/main/natives/net/ladenthin/llama/<OS>/<ARCH>/<backend>/`, and -- through `cmake/jllama-build-info.cmake` -- writes `jllama-build.txt` and `jllama-sbom.json` there (plus `jllama-files.txt` on the modular platforms and in a module-only build); the build file's `classifier=` is derived from the directory and must equal the row's |
| `llama/pom.xml` `natives` profile | one `maven-jar-plugin` execution per row, **generated**: `python3 .github/check-natives.py pom` prints them; each also includes `META-INF/third-party/natives/**`, the license texts of `llama/legal/natives/` that the `natives-legal` execution copies next to the natives tree (`NOTICE.md` there is the index; the WebUI's npm texts come from `.github/webui-licenses.py`, which `build-webui` checks against the pinned `package-lock.json`) |
| `llama-platform/pom.xml` | depends on the rows with `platform=yes` |
| `publish.yml` build jobs | each uploads its tree as artifact `natives-<classifier>`; `package` waits for every one of them (transitively) |
| `LlamaLoader.LIBRARY_BACKENDS` / `MODULE_BACKENDS` | must name every backend directory of their kind |
| the smoke sets | **derived**: one `llama-smoke-<os>-<arch>` per desktop platform with a library jar (no Android, no s390x: no runner) — `check-natives.py smoke-targets` prints them, `smoke-set <t>` the classifiers of one; `package` assembles each with `package-smoke-sets.sh` (the classes jar, its runtime dependencies, the platform's library jar and every GPU module jar of the platform) and uploads it; every target must be downloaded by a smoke job (`smoke-natives` rows, `smoke-natives-macos`), which `check-natives.py` demands |
| `examples/jbang/Chat.java`, `examples/server/pom.xml` | the JBang `//DEPS` lines name the classes jar and every `platform=yes` jar; the server example's `<llama.version>` and profiles follow the list — both held to the README's install version |
| `.github/check-natives.py` | runs in `code-style` (first minutes of every run); fails when any row of this table disagrees with the list — including CMake's backend names, the dependency allowlists and the README rows |
| `.github/merge-native-artifacts.sh` | reads the list: every listed `natives-*` artifact present, no other, each holding its library (or module) and nothing outside its directory, each with a `jllama-build.txt` naming a tag, the provenance keys and the row's classifier, next to a `jllama-sbom.json` naming the same classifier, a module directory holding nothing but its module, the `jllama-*.txt` lists and the BOM, no path claimed twice; writes `jllama-extras.txt` (sibling files loaded before the library, e.g. OpenVINO's `OpenCL.dll` on Windows -- never the files a build's own `jllama-files.txt` names, which are only extracted; each of those must exist) |
| `.github/package-smoke-sets.sh` | reads the list through `check-natives.py smoke-targets` / `smoke-set`: fails when a natives jar of a set was not built; also writes the glibc-floor set (`floor-set`) |
| the glibc-floor set | **derived**: the library rows whose directory promises a glibc floor (`nativedeps.GLIBC_CEILING`: the three Linux CPU jars) — `check-natives.py floor-set` prints them; `package` uploads them with the classes jar and its dependencies as `llama-glibc-floor`, and `smoke-glibc-floor` loads each on RHEL 8.10 (glibc 2.28, GCC 8's libstdc++) on its own architecture, s390x under qemu (`smoke-glibc-floor.sh`, `NativeLoadSmoke.java` in the UBI 8 OpenJDK 21 runtime image); `check-natives.py` holds the job's matrix rows to the set |
| `.github/verify-native-deps.py` | every library of a directory: exact dependency allowlist per library directory (`cpu`/`metal`) plus the files next to it; **the export set of every `jllama` library** (`JNI_OnLoad`, `JNI_OnUnload`, `Java_*`, `jllama_*` and nothing else -- `cmake/jllama-exports.map` on ELF and `cmake/jllama-exports-macos.txt` on macOS keep the statically linked llama/ggml/mtmd/BoringSSL symbols local, Windows exports only `dllexport`; before the lists the Linux library exported 14,220 symbols, 47 of them JNI), and for the Android OpenCL module (bionic + `libOpenCL.so` + `libggml-base.so`); denylist for the other GPU modules, each of which must import `ggml-base` and, as ELF, reach it only through run path `$ORIGIN` (an Android library carries no run path -- bionic's search path is the app's native-library directory -- and only a build-tree one fails there); the manylinux_2_28 builds (both Linux CPU jars, CUDA) held to `GLIBC_2.28`; every Android library 16 KB page-aligned (Google Play); every typeinfo name (`_ZTS*`) of the s390x libraries at an even address (see "Linux s390x"). Runs in `package` and on the staged AAR libraries |
| `.github/smoke-natives-jars.sh` (`package` job) | loads the real jars: classes + all 23 natives jars at once, on the classpath **and** the module path; it passes `net.ladenthin.llama.backend` naming **every** module of the runner's platform, opt-in ones included, and the loader line must then name each -- so a module jar the loader skipped fails here, and this is also the only place in CI that loads an opt-in module at all |
| `.github/smoke-lib.sh` / `smoke-test-natives.ps1` | the model-backed smoke rows expect every module of the set **except** the opt-in ones (`optin` column), because the loader deliberately leaves those out -- it is the same list, read with one more condition |

**Adding a natives jar:** a row in `natives.csv`, the execution `check-natives.py pom` prints, a build
job uploading `natives-<classifier>` (`-DJLLAMA_MODULE_ONLY=ON -DGGML_<X>=ON` for a GPU module), the
backend name in CMake and in `MODULE_BACKENDS` / `LIBRARY_BACKENDS` if new, a README row, a profile in
`examples/server/pom.xml`. `check-natives.py` fails until they agree; nothing else needs touching — the
smoke sets follow by themselves (a new desktop OS/arch also needs its `smoke-natives` row, which
`check-natives.py` then demands). The checks are Python in `.github/buildcheck/` with unit tests (see
"Build checks, shared files and the release gate").

**Why the build jobs are not spawned from the list as one matrix** (considered and rejected): the 23
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
it silently keeps the first one and drops the others' natives** (measured; `check-natives.py` holds
the name in every pom execution, and `smoke-natives-jars.sh` launches all 23 on the module path for
that reason). Nothing `requires` a natives module, so a module-path launch needs `--add-modules` (or
the natives jars on the classpath). `module-info.java` `requires` Jackson and SLF4J — it did not
before, so the classes jar never worked on the module path.

**Locally**, CMake writes into `llama/src/main/natives/` (git-ignored); surefire adds that directory
to the test classpath (`additionalClasspathElements`), so `mvn test` finds the library. `-P natives`
is CI-only (a local tree holds one directory); a missing directory there is caught by the merge
step, not by Maven.

## Modular natives: `JLLAMA_BACKEND_DL` (library jars) and `JLLAMA_MODULE_ONLY` (GPU module jars)

Every platform a GPU backend can join is built **modular**, the way upstream builds its own release
binaries (`GGML_BACKEND_DL`), and so is s390x, for its CPU variants (see "Linux s390x"): ggml and ggml-base as shared libraries, every backend a module ggml
loads at run time -- the CPU backend as one small `ggml-cpu-<level>` module per instruction-set
level where ggml has a variant list (`GGML_CPU_ALL_VARIANTS`: Linux and Android x86-64 / aarch64,
Windows x86-64), of which `ggml_backend_load_best` loads the best for the running CPU, and as one
`ggml-cpu` module where it has none (Windows arm64). Two CMake options:

- **`-DJLLAMA_BACKEND_DL=ON`** builds a **library jar**: `libjllama` (llama, mtmd, llama-common and the
  server static inside it), `ggml`, `ggml-base`, `ggml-rpc` and the CPU modules, all in one directory.
  The option refuses every other platform (`FATAL_ERROR`) and `BUILD_TESTING` (see "Where C++ tests
  run"), and on Windows x86-64 it refuses `cl.exe`/`clang-cl` with variants, because ggml's
  `if (NOT MSVC)` block drops 5 of the 14 x86 variants for both and `clang-cl` cannot build
  `alderlake` at all -- that jar is built with **plain clang**.
- **`-DJLLAMA_MODULE_ONLY=ON -DGGML_<X>=ON`** (implies the first) builds a **GPU module jar**: exactly
  one of `GGML_CUDA` / `GGML_HIP` / `GGML_SYCL` / `GGML_VULKAN` / `GGML_OPENCL` / `GGML_OPENVINO`, with
  `GGML_CPU OFF` and `GGML_RPC OFF`, and the configure `return()`s before the llama.cpp subdirectory
  and the `jllama` target exist -- the build produces the one module (`ggml-cuda`, `ggml-hip`,
  `ggml-sycl`, `ggml-vulkan`, `ggml-opencl`, `ggml-openvino`) into the backend directory, with
  `$<TARGET_FILE_NAME>` in `jllama-files.txt`, and nothing else. It links the ggml-base the same
  configure built, so its ABI is the library jar's of the same tag (what `jllama-build.txt` pins).
  Upstream's GPU zips do the same (`--target ggml-cuda` etc., Vulkan/OpenCL with `GGML_CPU=OFF`).

**What stays static**: macOS (`metal-macos-aarch64`: Metal is compiled into the library, upstream
ships macOS static too). s390x was static until 6.1.0 (one scalar `libjllama.so`, no GPU can join
it) and is modular now for its IBM Z CPU variants. There the loader extracts and loads a single file (plus `jllama-extras.txt` and a
`ggml-metal.metal` a build does not embed).

**Every directory carries `jllama-build.txt`** (`llama.cpp=<tag>`, the `JLLAMA_GIT_TAG` number of this
file's pin, `backend=`, `kind=library|module`), written with `file(GENERATE)`: the loader's build
check and the merge step's completeness check read it (see "Natives jars").

**And since the delivery roadmap (TODO.md) the file is the artefact's provenance record**, written by
`llama/cmake/jllama-build-info.cmake`: `llama.cpp.commit` (the upstream commit actually fetched),
`classifier` (derived from the directory, held to the `natives.csv` row by the merge step),
`jllama.version` (the `<parent>` version of `llama/pom.xml`, or `-DJLLAMA_VERSION=`), `jllama.commit`
(this repository's commit, `-dirty` with local changes, `GITHUB_SHA` as the fallback), `patches` (the
ids of `llama/patches/*.patch` in order) and `patches.sha256` (one hash over `<name> <sha256>\n` per
patch, the lines the applier's stamp carries), `compiler`, `toolkit` (the vendor SDK a GPU module was
built against, probed per backend) and `signed` (`no` until the signature step exists). A CycloneDX
1.5 `jllama-sbom.json` next to it carries the same facts, with every patch as the llama.cpp
component's pedigree. `merge-native-artifacts.sh` requires every key and the BOM of every artifact;
`LlamaModel.getJllamaBuildInfo()` returns the map, `Diagnose` prints one line per directory, and
`docs/patches.md` -- generated by `check-patches.py --render`, held current by `code-style` -- is the
page the `patches=` ids point at. The `package` job and `attest-snapshot` / `attest-release` (every
file of the signed publication, see "Signed publication") additionally attest build provenance
through `actions/attest-build-provenance` (Sigstore): `gh attestation verify --owner bernardladenthin
<file>` proves a natives jar, a jar, a pom or an AAR came out of this workflow at a given commit, which
the `.asc` alone cannot (it proves the key, not the build). The reactor's `cyclonedx-maven-plugin`
attaches a `cyclonedx` json classifier to every Maven module for the Java side.

Before 6.0.0 each jar was **one** library for one level -- x86-64 the Haswell baseline (x86-64-v3,
a block that forced `GGML_AVX2` & co.: a CPU without AVX2 died with SIGILL, a JVM crash and not an
exception, at the first matrix multiplication, and AVX-512/VNNI/AMX stayed unused on CPUs that have
them), aarch64 plain ARMv8 -- and the GPU jars were complete libraries with the CPU backend linked
in, tried one after the other (`JLLAMA_CPU_VARIANTS`, the two-stage predecessor of this; the
measurements behind the Windows step are in `docs/handover/local-agent-report-b11538-windows.md`).

- **ggml, llama and mtmd become shared; llama-common stays static.** `FetchContent_MakeAvailable`
  is pointed at a non-existent `SOURCE_SUBDIR`, so it only fetches and patches; ggml is then added
  first with `BUILD_SHARED_LIBS ON` (llama.cpp skips its own `add_subdirectory(ggml)` when a `ggml`
  target exists; a module-only build never adds llama.cpp). **Layer A of the delivery roadmap
  (TODO.md)** then adds llama.cpp's root with `BUILD_SHARED_LIBS ON` and `LLAMA_BUILD_COMMON OFF`
  -- `src/` gives `libllama.so`, upstream's own C API (`llama.h`) from the patched source, exactly
  the file upstream's release zips ship -- and afterwards `vendor/cpp-httplib` plus `common/` with
  it `OFF` again, because upstream's `common/CMakeLists.txt` follows `BUILD_SHARED_LIBS` like
  `src/` does and llama-common is a C++ library (no stable ABI, `WINDOWS_EXPORT_ALL_SYMBOLS` when
  shared) whose only consumer here is the server code compiled into `jllama`. `common/` reads the
  build-info values (`b<number>-<hash>`, compiler, target) and the version pair from the root's
  directory scope, which is not visible from ours, so the configure sets them itself from the
  fetched tree (or from the archived tree's patch stamp, see "Source archive"). `mtmd` is added in
  its own block as before, shared in this build (`libmtmd.so`, `mtmd.h`). Both are unversioned like
  ggml and get the `$ORIGIN` run path; the configure refuses a `llama`/`mtmd` that came out static or
  a `llama-common`/`cpp-httplib` that came out shared (one more file to ship). On the static
  platform (macOS) all of it stays inside `libjllama`. **Every ggml default llama.cpp's own
  CMakeLists changes before its `add_subdirectory(ggml)` is repeated** -- `GGML_LLAMAFILE_DEFAULT`
  and `GGML_CUDA_GRAPHS_DEFAULT` at b11968 (unchanged since b11639) -- because adding ggml first skips that block; re-check with
  `grep -n _DEFAULT CMakeLists.txt` at the new tag on every bump. The second one was missing once:
  ggml's own default for CUDA graphs is OFF, and a CUDA module built without it lost two thirds of
  its token generation (265.6 -> 88.8 tg128 on an RTX 3070, prompt processing -20%; Vulkan, which
  has no equivalent, lost nothing). `ggml` and `ggml-base` are built **unversioned** (`VERSION` /
  `SOVERSION` cleared: the loader extracts plain files, so file name, SONAME and `DT_NEEDED` must
  agree).
- **The library directory holds 20 files on Linux, Android and Windows x86-64**: `libjllama.so`,
  `libllama.so`, `libmtmd.so`, `libggml.so`, `libggml-base.so`, 14 `libggml-cpu-<level>.so` (`x64`,
  `sse42`, `sandybridge`, `ivybridge`, `piledriver`, `haswell`, `skylakex`, `cannonlake`,
  `cascadelake`, `cooperlake`, `icelake`, `zen4`, `alderlake`, `sapphirerapids`) and
  `libggml-rpc.so`; **14 on Linux aarch64** (8 modules, `armv8.0_1` to `armv9.2_2`: dotprod, fp16,
  SVE, i8mm, SVE2, SME); **9 on Linux s390x** (3 modules: `z13`, scalar, `z15` with VXE2, `z16` with
  VXE2 + NNPA; see "Linux s390x"); **13 on Android aarch64** (ggml's own Android list of 7, `android_armv8.0_1`
  to `android_armv9.2_2`); **7 on Windows arm64** (one `ggml-cpu.dll`) -- ELF with run path
  `$ORIGIN` only, **except on Android,
  where the libraries carry no run path at all**: CMake's Android platform drops the flag
  (`Modules/Platform/Android.cmake`, "Android reportedly ignores RPATH"; dockcross's toolchain file
  sets `CMAKE_SYSTEM_NAME Android`), so `INSTALL_RPATH "$ORIGIN"` writes nothing there, and none is
  needed -- bionic resolves the siblings through the app's native-library directory, the lookup
  `System.loadLibrary` itself uses, which `test-android-emulator` proves on the device.
  `verify-native-deps.py` therefore accepts an absent run path on Android (a build-tree path still
  fails); the first full run of the modular natives (38065017595) applied the desktop rule there and
  failed 27 libraries that load fine. `jllama-files.txt` lists everything besides
  `libjllama`; `LlamaLoader` extracts them next to it **without loading them** (`BACKEND_FILES_FILE`,
  the counterpart of `jllama-extras.txt`, which loads). `JNI_OnLoad` then calls
  `ggml_backend_load_all_from_path(<its own directory>)` (found via `dladdr`, on Windows
  `GetModuleHandleExW` + `GetModuleFileNameW`, handed over as **UTF-8** -- a path with an umlaut loaded
  nothing as ANSI, and no error was logged): ggml's default searches next to the executable, which
  here is `java`. That call also picks up the GPU modules the loader put there.
- **Windows specifics.** There is no `$ORIGIN`, so the four libraries `jllama.dll` links
  (`ggml-base.dll`, `ggml.dll`, `llama.dll`, `mtmd.dll`, in that dependency order) are left out of
  `jllama-files.txt` and named in `jllama-extras.txt` instead, which
  `LlamaLoader` loads by full path **before** `jllama.dll` (otherwise Windows resolves the import from
  `java.exe`'s directory or `PATH`, where a foreign llama.cpp install can answer); CMake writes that
  file itself, byte-identical to what `merge-native-artifacts.sh` derives, so a local build is
  loadable without the merge step (before, 57 Java tests failed on the load). A GPU module whose
  runtime is missing fails silently (`SEM_FAILCRITICALERRORS`, ggml logs it at `GGML_LOG_DEBUG` only),
  which is why `JNI_OnLoad` logs the backend registry and the devices ggml ends up with at INFO. And
  the plain-clang jobs (this CPU job, the ROCm module) cache through sccache only because
  `llama/cmake/msvc-runtime-gnu-clang.cmake` keeps CMake's `-Xclang` runtime flags off their TUs (see
  "sccache on every Windows Ninja job"); before it every TU was a miss and the ROCm job built for
  3-4 h on every run.
  **And the `windows-2025-vs2026` runners are a CPU lottery** (an AMD EPYC 7763 and an Intel Xeon
  6973P-C in two consecutive runs), so which CPU module `ggml_backend_load_best` picks in a Windows
  test or smoke job -- `haswell` on the AMD box, `sapphirerapids` with AMX on the Intel one --
  differs per run. That is how `patches/0018` (the AMX buffer freed with `free()` after
  `_aligned_malloc()`) stayed invisible through the first full run and killed the second.
- **Android loads by soname, not by directory.** `System.loadLibrary("jllama")` finds the AAR's
  `jni/<abi>/` through the app's native-library search path, and the directory itself is not
  reliably known (an app's libraries may stay inside the APK, `extractNativeLibs=false`), so
  `ggml_backend_load_all_from_path` cannot be used. CMake generates `jllama_modules.h` with the
  sonames of the build's CPU modules and of `ggml-rpc` (`JLLAMA_CPU_MODULE_SONAMES` /
  `JLLAMA_OTHER_MODULE_SONAMES`) and defines `JLLAMA_MODULES_BY_SONAME`; `JNI_OnLoad` then
  `ggml_backend_score`s every CPU module by name, loads the best, loads the others, and tries the GPU
  candidates `libggml-opencl.so` and `libggml-vulkan.so` by name -- present when the app packages the
  OpenCL AAR (or a future Vulkan one), absent otherwise, logged either way. The CPU AAR therefore
  carries every `.so` of the directory under `jni/<abi>/`, and the OpenCL AAR holds the one module
  and depends on the CPU AAR (see "Android AAR + Kotlin façade").
- **The extraction directory is keyed by the build and the module set**
  (`LlamaLoader.extractionDirectoryName`: `jllama-backend-<backend>-<CRC and size of the library's
  jar entry>[-m<hash of the module jars>]`), because it is shared between JVMs and holds many files.
  Keyed by the backend alone, a JVM running another jllama build wrote into the directory of a
  running one; on Windows that replaces exactly the modules the first JVM scored and unloaded again
  (which ones depends on the CPU) and leaves a mixture of two builds for the next start -- measured
  with upstream's DLLs. **And it is reused across starts**: `cleanup()` spares the directories of the
  builds on the classpath (`extractionDirectoriesOfThisClasspath`, computed before anything is
  extracted, module set included), nothing is registered for `deleteOnExit` any more, and
  `extractFile`'s reuse branch -- compare the file with the jar byte for byte, write only what
  differs, atomically -- does the rest, so a second start copies nothing (18 files, ~1 s with an
  on-access scanner on Windows, on every JVM start before; the same for every surefire fork). What
  `cleanup()` removes: the flat files of the pre-6.0.0 layout, and the directory of a build that is
  **not** on this classpath once it is older than `STALE_EXTRACTION_AGE` (10 min) -- a start touches
  its directory's mtime before extracting, so a JVM that is still starting from a directory is never
  swept; every delete is best-effort, and a DLL a running JVM holds is locked on Windows and stays
  (the modules ggml scored and unloaded are not, and may go -- that JVM no longer needs them, and
  the next start of its build re-extracts exactly the missing ones). Isolating the extraction per
  process was considered and rejected for the same reason: a per-build directory shared by every JVM
  of that build, with content-checked reuse, is the cache; a per-process one would copy on every
  start again.
- **RPC is a module too** (in the library jar; a module-only build has `GGML_RPC OFF`), so
  `rpc_bridge.cpp`/`rpc_support.hpp` resolve its four entry points through
  `ggml_backend_reg_get_proc_address` (`rpc_proc<>()`, as upstream's `common/arg.cpp` does) in every
  build.
- **Built in manylinux_2_28** (AlmaLinux 8, gcc-toolset-14, glibc 2.28): x86-64 in the dockcross
  image the CUDA job already used, aarch64 natively on `ubuntu-24.04-arm` inside the pypa image
  (`.github/manylinux/manylinux_2_28-aarch64`, see "Linux aarch64"); the s390x libraries are
  cross-compiled against the pypa s390x image as a sysroot (see "Linux s390x"). manylinux2014's GCC 10
  lacked `-mavxvnni` and `-mamx-tile` (`alderlake`, `sapphirerapids`; ggml does not check). The glibc
  floors meet at **2.28**: x86-64 rises from 2.17, aarch64 falls from ~2.39, s390x from 2.38.
  `verify-native-deps.py` reads every library's version-needs table and holds these directories (and
  CUDA) to `GLIBC_2.28`, `GLIBCXX_3.4.25` and `CXXABI_1.3.11` (the system libstdc++ of manylinux_2_28
  is GCC 8's; gcc-toolset links the rest from `libstdc++_nonshared.a`); the modules themselves
  reference `GLIBC_2.27` at most (measured in the image). `smoke-glibc-floor` loads each of the three
  CPU jars on RHEL 8.10.
- **One `ctest` in a modular build job, `c_abi`, and no `jllama_test`**: `jllama_test` calls the
  CPU/RPC backends directly, so the option refuses `BUILD_TESTING`; what a library-jar build does
  run (`JLLAMA_C_ABI_TEST`, on by default, not when cross-compiling -- s390x excepted, which runs it under its toolchain's qemu) is `src/test/c/test_c_abi.c`
  -- plain C11, linked against the shared `llama` and `ggml-base` only, no JNI, no C++: it calls
  `ggml_backend_load_all_from_path` on the natives directory, `llama_backend_init`, requires at
  least one device (the CPU module must have loaded), prints `llama_print_system_info`, and loads a
  model when `JLLAMA_C_ABI_MODEL` names one. `build.sh`/`build.bat` run `ctest -R '^c_abi$'` after
  every build that registered it, so the first thing proven about a library jar is that upstream's
  C API in it works without a JVM (layer A). `test-cpp-linux-x86_64`, `test-cpp-linux-aarch64`,
  `test-cpp-windows-x86_64` and `test-cpp-windows-arm64` build the default static library with the
  tests, each with the compiler of the shipped build. The Java suite (`test-java-linux-x86_64`,
  `test-java-windows-x86_64`, the langchain4j integration job through `-Dnet.ladenthin.llama.lib.path`,
  the smoke sets) runs real inference through the modular build -- a load alone proves little: in the
  Windows measurement every crash came after a successful load with a correct device count.
- **The natives-jar checks know directories with many files**: `merge-native-artifacts.sh` requires
  every file `jllama-files.txt` names and keeps them out of `jllama-extras.txt`; `verify-native-deps.py`
  checks every library and allows a sibling as a dependency only through `$ORIGIN`; `package-fatjars`
  is gone with the fat jars.
- **Measured.** Locally (GCC 13, a Xeon with AVX-512 + VNNI, no VBMI/BF16): the x86-64 directory is
  36 MB (13.4 MB zipped), `libjllama.so` 15 MB of it; ggml picked `cascadelake`, and with the AVX-512
  modules removed it fell back to `haswell` (skipping `alderlake`, whose AVX-VNNI the CPU lacks).
  On a Ryzen 7 5800H (Zen 3; upstream's b11476 Windows binaries, Qwen3-0.6B Q4_K_M, 8 threads):
  `haswell` 313 t/s pp512 and 14.3 t/s tg128, the plain `x64` module 30.0 and 8.8 -- a baseline-only
  build would be unusable for prompt processing, and the 14 x86 modules together are ~20 MB. The gain
  on AVX-512/AMX hardware was measured on 2026-10-11 (Granite Rapids, Zen 4, Graviton 4:
  `sapphirerapids` +148 % prompt processing over `haswell`, `zen4` +35 %;
  [`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md)). The module-only path was verified locally with
  a real `libggml-opencl.so` (OpenCL headers + ICD loader staged in the sandbox): put next to the
  modular CPU library by the loader, ggml registered the OpenCL backend with the ICD loader on
  `LD_LIBRARY_PATH` ("3 backend(s) registered") and silently ran on the CPU without it; with
  `-Dnet.ladenthin.llama.backend=cpu` the module stayed out and a separate extraction directory was
  used. The GPU jobs themselves (12 module-only builds) are proven by the CI run.

## Build checks, the shared CI and the release gate (`.github/buildcheck/`, bernardladenthin/actions)

The pipeline's static checks are **stdlib-only Python** in two places. What the four Java
repositories (java-llama.cpp, BitcoinAddressFinder, srcmorph, streambuffer) share lives once in
[bernardladenthin/actions](https://github.com/bernardladenthin/actions) -- the `actionlib` library,
composite actions for single steps, reusable workflows for whole jobs and templates for the files
every repository must carry -- and this repository pins it by commit (`uses:
bernardladenthin/actions/<common|java>/<action>@<commit> # v1.x.y`; Dependabot moves the pin). Until
2026-10 that code was four byte-identical copies held together by `.github/shared-files.sha256` per
repository. What only this repository needs stays in `.github/buildcheck/`, with unit tests in
`.github/buildcheck/tests/` (`python3 -m unittest discover -s .github/buildcheck/tests -t .github`,
from the repo root; they run in the `repository-checks` job). The `.github/*.py` scripts are thin
command-line wrappers around it, so every check can be driven with literal inputs in a test --
including the ELF/PE/Mach-O parsers (synthesized binaries). `natives.py` reads the job graph with
actionlib's workflow parser: `.github/buildcheck/__init__.py` finds actionlib on `PYTHONPATH`
(`common/setup` in CI), in `ACTIONS_DIR`, or in a checkout next to this repository (`../actions`),
and `natives.py` imports it only in the checks that need it, so `check-natives.py smoke-targets` /
`smoke-set` run without it. Nothing uses PyYAML: the runners do not guarantee it, and the checks only
need the job graph, which the workflow parser reads from the two shapes these workflows use and
rejects anything else loudly.

| Module (here) | CLI | What it checks |
|---|---|---|
| `natives.py` | `check-natives.py` (`code-style`) | everything that names a natives jar agrees with `natives.csv` (see "Natives jars") |
| `models.py` | `check-natives.py` | every `*_MODEL_NAME` of publish.yml's `env:` is a filename of `models.csv` |
| `patches.py` | `check-patches.py` (`code-style`) | every hunk of `llama/patches/*.patch` declares exactly the lines it carries -- the half `git apply --check` does not see (see "Local llama.cpp source patches"); and `docs/patches.md` is what `check-patches.py --render` generates from the patch headers (a stale page fails) |
| `nativedeps.py` | `verify-native-deps.py` (`package`, `package-android-aar`, `build-linux-s390x`) | dependency allowlists, 16 KB Android alignment, every s390x typeinfo name at an even address |
| `hipoffload.py` | `verify-hip-offload-compressed.py` (the two ROCm jobs) | no uncompressed GPU code bundle |
| `aarpoms.py` | `verify-aar-poms.py` (`package-android-aar`, on the poms in mavenLocal) | both AAR poms carry exactly the core's compile-scope dependencies, at its versions and at `compile` (plus the CPU AAR, `type aar`, in the OpenCL one) |
| `postrelease.py` | `post-release-check.py` (`post-release.yml`, after the release) | the published natives jars against `natives.csv`, their provenance with `patches.sha256` recomputed from the tag, the source archive's stamp and reproducible pack, the AAR invariants and poms, the cache-free tag build (docs/post-release-check.md, checks 3, 8, 9, 13, 14) |
| `webuilicenses.py` | `webui-licenses.py` (`build-webui`, and `webui-licenses.yml` on every pull request and push) | the WebUI's npm license texts cover the pinned lockfile (digest over LF line ends) and its imports; it fails every run -- until the release run of v6.0.0 it only warned outside a release tag, and a stale list surfaced there first |

| Action (bernardladenthin/actions) | Job here | What it checks or does |
|---|---|---|
| `common/check-templates` | `repository-checks` | the files this repository carries equal their templates (`.github/actions-config.json`, `templates`) |
| `common/check-release-gate` | `repository-checks` | every job gates **both** publish jobs, unless `.github/release-gate-exemptions.txt` names it with a reason |
| `common/check-run-scripts` | `repository-checks` | every bash `run:` script of the workflows and local actions parses (`bash -n`) |
| `java/check-maven-versions` | `repository-checks` | **warns** where a Maven dependency or plugin is used in another version than in a sibling repo |
| `java/check-publication` | `repository-checks` | the publication jobs are wired as documented; a dry run of `.github/release-assets.json` collects what it promises |
| `java/check-release-ref`, `java/finalize-publication`, `java/publish-central`, `java/publish-github`, `java/attest-publication` | `check-snapshot` / `check-tag`, the publish jobs, the six consumers | the signed publication (see "Signed publication") |
| `java/audit-publication` | `audit` of `post-release.yml` | the published release from the outside: the run and its jobs, the files against the contract, Maven Central (polled), the signatures with the release key from a keyserver, every file's provenance, the bytecode floors, the next version (`post-release` in `.github/actions-config.json`) |

**The post-release check (`post-release.yml`).** Runs after a successful Publish run dispatched on a
`v*` tag, and by hand with a tag: `audit` (the action above, then `post-release-check.py` on the
downloaded files) and `consumer` (per platform: `post-release-consumer.sh` resolves `llama-platform`
from Central into an empty local repository, then the server smoke on Java 8, and on Linux x86-64
`post-release-webui.sh`). Tools from the default branch, the release's facts from a checkout of the
tag in `release/`. It is not in the release gate -- it runs after the publication, on what was
published (docs/post-release-check.md).

**The release gate.** A job nothing waits for can go red while a release still ships — the natives
builds `package` once forgot to wait for, and the aarch64 fat jars that were signed and attached
without any job launching them, were both that shape. So "not gating" is a written decision: a line
`<job>: <reason>` in `.github/release-gate-exemptions.txt`. The check fails both ways — a job neither
gating nor listed, and a listed job that gates after all (a stale exemption hides the next job of that
name). Introducing it found `vmlens` gating nothing in all four repos; it now gates both publish jobs.

**The files every repository carries (templates).** Community files, `LICENSES/{MIT,Apache-2.0}.txt`,
`.editorconfig`, `.gitattributes`, `.java-version`, `.mvn/jvm.config`, `lombok.config` (here
`llama/lombok.config`, the `relocate` entry), Dependabot, `scorecard.yml`, `claude.yml`,
`claude-code-review.yml` and the callers of the reusable workflows (`codeql.yml`,
`dependency-submission.yml`, `reuse.yml`, `osv-scanner.yml`) are templates in bernardladenthin/actions;
`common/check-templates` fails when a copy here differs (up to the repository name and the action
pins, which Dependabot moves per repository). To change one, change the template there and run
`python3 -m actionlib.common.templates --write` here (with its `lib/` on `PYTHONPATH`). This repository
excludes `sonarqube.yml` (its analysis is its own), `.mvn/settings.xml` and `CITATION.cff`.

**Maven versions across the four repos.** They pin the same build tooling and Dependabot bumps each
repo on its own, so the versions drift one pull request at a time; `java/check-maven-versions`
compares every `groupId:artifactId` the POMs use (plugins, dependencies, annotation-processor paths,
and the formatter version Spotless is configured with -- `${...}` resolved) with the default branches
of the `java-maven` family and **warns** per difference. Warnings only, because a bump lands in four
pull requests and the first one merged must not turn the other three red; the repos' own artifacts
(`net.ladenthin:*`, `ignore-groups`) are left out. It compares what is used, not property names: those
differ between repos and include values that must differ (`maven.compiler.release`). **The POM list
of this repository is `maven-versions.poms` in `.github/actions-config.json`** -- a new module's POM
goes there, and only there: a sibling reads it from this repository's configuration (here also
`examples/server/pom.xml`, whose `exec-maven-plugin` sat two patch releases behind the reactor's unseen
until it was listed). It also compares the Kotlin Gradle plugins of every `*.gradle.kts` with the Kotlin
the POMs use (`org.jetbrains.kotlin:*`): Dependabot's maven ecosystem bumps `kotlin.version` and never
sees the Gradle files, which is how the Compose compiler plugin stayed at 2.4.20 while the POMs went to
2.4.21.

**Run scripts parse.** A cleanup once lost the trailing ` \` of six continued lines in `publish.yml`
-- each left a line starting with `||`/`&&`, a syntax error in a step only a native build, the AAR job
or the snapshot release reaches. actionlint did not see it (it parses scripts only with shellcheck
installed), `bash -n` sees every one. So `common/check-run-scripts` runs `bash -n` over every `run:`
script of `.github/workflows/` and `.github/actions/*/action.yml` that runs in bash, deciding the
shell as the runner does (step `shell:`, job then workflow `defaults.run.shell`, else PowerShell on a
`windows` runner and bash elsewhere; a matrix/input runner is Windows only where the step's `if:` says
`runner.os == 'Windows'`). pwsh/powershell/cmd/python are skipped, an unknown shell is an error.

**And the workflows must parse as YAML (`code-style`, first step).** None of the checks above reads the
files as YAML -- the workflow parser and the run-script check read the two-space layout on purpose,
and a run script that parses in bash tells nothing about the document around it. So a step `name:`
with an unquoted `: ` inside it (`(full history: the commit count is checked)`, a plain scalar that
YAML reads as a nested mapping) passed every check, `code-style` included, and failed the **whole**
run of the 6.0.0 merge (38807436221) before a single job started: zero jobs, the run named after the
file path, `gh run view` saying "workflow file issue". The step loads every `.github/workflows/*.yml`
and `.github/actions/*/action.yml` with PyYAML (preinstalled on the Ubuntu runner, `pip install
--user` as the fallback) and fails on the first file that does not parse. It stays out of
`buildcheck/`, which is stdlib-only by design; a colon in a step name is best avoided rather than
quoted.

**License of `.github`:** every file there is `MIT OR Apache-2.0` -- the license of
bernardladenthin/actions, so a file can move between them without a second header -- except the
third-party ones annotated in `REUSE.toml` (the generated dockcross wrappers, the JNI headers).

**Composite actions (`.github/actions/`).** The shared ones are in bernardladenthin/actions (see
above and "Signed publication"); the local ones are this repo's own: `fetch-models` (`.github/fetch-models.sh`: the models of
the features a job names, from Hugging Face with `HF_TOKEN`, SHA-256 checked; the one place a job gets
its models — 9 call sites, plus the reusable `java-tests.yml` and `post-release.yml`) and
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

**Matrix for the natives smokes (`smoke-natives`).** See "Smoke sets" below. A matrix and not a
reusable workflow, because the rows differ only in data (target, runner).

**Check names changed with both** — a called job shows as `<caller name> / Java tests`, a matrix row
as `Smoke test natives jars (<target>)`. That was measured, not assumed, before doing it:
none of the four repositories has a required status check (each has only a ruleset — no deletion,
no force push, PR required — and classic branch protection is off), so no setting pins a check
name. Should required checks ever be added, name these new forms.

## Where C++ tests run: one build per shipped artifact

**The policy, which was implicit until it was written down here and therefore re-guessed at every
new job:**

> **One build per shipped artifact. The C++ tests ride along in that same job wherever the shipped
> configuration can host them, and get a job of their own only where it cannot. The risks that are
> specific to a configuration the tests cannot run in are covered by the layers that *can* see
> them -- the packaging check and the Java suite against the real artifact.**

Today that means 1 job builds a shipped artifact **and** runs `ctest` (`build-macos-arm64-metal-15`
-- the one static platform; every modular library-jar job additionally runs the one plain-C test
`c_abi` against the shared `llama` it ships, see "Modular natives", the s390x one under qemu) and 8
only test
(`test-cpp-linux-x86_64`, `test-cpp-linux-aarch64`, `test-cpp-linux-s390x`, `test-cpp-windows-x86_64`,
`test-cpp-windows-arm64` -- forced, see below -- plus `build-windows-x86_64-msvc`, the MSVC
toolchain build that ships no jar since the modular natives but still runs `ctest` and the Java
suite, and `build-macos-arm64-metal` and `build-macos-arm64-no-metal`, whose artifacts are not
shipped at all). The 12 GPU module jobs build no tests either: a module-only build has no `jllama`.

**Why riding along is safe, measured rather than assumed:** `-DBUILD_TESTING=ON` does not change
the shipped library. Compared on a Windows MSVC configure with the flag off and on, the `jllama`
target's `DEFINES`, `FLAGS` and `INCLUDES` are byte-identical
(`/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++17 -MT /utf-8`) and no gtest/gmock reaches its
link line -- the option only adds the googletest FetchContent and the `jllama_test` target beside
it. So a combined job ships exactly what it tested, which is a property worth having: the test sees
the same compiler, the same flags and the same sccache state as the artifact.

**Why splitting is not better as a default.** It doubles the compilation of llama.cpp for that
platform (134 model TUs plus the 16.6k-line `httplib.cpp` at `-O3`) and buys nothing where the
configuration can host the tests -- it even gives up the property above.

**Where it is forced:** `JLLAMA_BACKEND_DL` (every modular platform, see "Modular natives") turns
the CPU backend and ggml-rpc into loadable modules, so their symbols are not available at link time.
`src/test/cpp/test_rpc.cpp` calls `ggml_backend_cpu_init()` and the four
`ggml_backend_rpc_*` entry points **directly and on purpose** -- that is the runnable guard for
`patches/0015`, built to fail at *link* time if the patch is dropped -- so `jllama_test` cannot
link in that configuration, and `llama/CMakeLists.txt` refuses the combination with a
`FATAL_ERROR` rather than letting it fail obscurely.

**A second "variants + tests" job was considered and rejected, with numbers.** Exactly **one** of
the 14 test files is affected by `GGML_BACKEND_DL` -- `test_rpc.cpp`, which is also the only one
whose behaviour depends on ggml's device registry (populated by loaded modules there instead of at
link time). The other 13 test pure helpers and patch guards over statically linked llama code and
return the same result in any configuration. So such a job would either exclude `test_rpc.cpp` and
re-run 13 configuration-independent tests for the price of a full extra build (no information
gained), or adapt `test_rpc.cpp` to resolve the symbols through
`ggml_backend_reg_get_proc_address` under `#ifdef GGML_BACKEND_DL` -- which duplicates, in C++,
what `RpcIntegrationTest` already proves end to end against the shipped artifact on every platform.

**And the decisive check: the two defects the Windows variants work actually had would have been
caught by neither.** Both only appear when the library is loaded out of the jar:

| Defect | `ctest` would catch it | what does |
|---|---|---|
| the 14 module DLLs built into `<dir>/Release/` instead of beside the library | no -- they exist, just in the wrong place for packaging | **`merge-native-artifacts.sh`**, which fails with "`jllama-files.txt` names '<f>', which the build did not produce" |
| `ggml.dll` not pre-loaded, so `jllama.dll`'s import resolves from `java.exe`'s directory or `PATH` | no | the Java test job and the smoke sets -- `System.load` fails |

So the safety of a variants build rests on the packaging check and on the Java suite running real
inference against the **shipped** natives artifact (`test-java-windows-x86_64` takes
`natives-cpu-windows-x86-64`), not on `ctest`. That is the same argument this file already makes
for Linux: a successful load proves little, since in the Windows measurement every crash came
*after* a load that reported the right device count.

**`ctest` runs with `--no-tests=error`, and with `-C Release` on Windows.** The Windows jobs configure
with a multi-config generator (Visual Studio, Ninja Multi-Config), where `gtest_discover_tests`
registers the tests per configuration; a `ctest` without `-C` found none there, printed `No tests
were found!!!` and exited 0. That is how the three Windows C++ jobs passed while running nothing,
until 2026-10-25 (found in the review of PR #559, run 40185148270). `--no-tests=error` turns such
an empty run red in every job.

**Consequence for a new job:** if the configuration you ship can build `jllama_test`, run `ctest`
in the same job. If it cannot, add a `test-cpp-<platform>` job that builds the default static
library with `-DBUILD_TESTING=ON`, and make sure the artifact itself is exercised by a
model-backed Java job or a smoke job -- not by the C++ suite alone.

## Upgrading CUDA Version

Current CUDA version: **13.4** (Linux `cuda-toolkit-13-4` from NVIDIA's rhel8 repo; Windows 13.4 redist archives)

To change the CUDA version, update the following places:

1. **`.github/build_cuda_linux.sh`** — the `sudo dnf install -y cuda-toolkit-13-4` line and the
   `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc` line (plus the header comment).
2. **`.github/workflows/publish.yml`** — the `build-windows-x86_64-cuda` job's
   "Install CUDA Toolkit … (NVIDIA redist archives)" step: the `v13.x` directory, `CUDA_PATH_V13_x`,
   and the per-component archive list. **The explicit `CMAKE_CUDA_ARCHITECTURES`** sits in this job
   and in `build_cuda_linux.sh`, and the two lists must stay equal: ggml's default leaves `75` (T4),
   `80` (A100) and `90` (H100) as PTX, and a driver too old to JIT the toolkit's PTX makes ggml
   `abort()` the JVM after offloading every layer instead of falling back -- `-ngl 0` aborts too, so
   having the jar on the classpath is enough (measured on a Tesla T4, see
   [`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md)). Server applications run
   on exactly those GPUs, which is why this deviates from upstream, who ship to RTX owners and say in
   a comment that they rely on the default. On a bump, re-read ggml's list at the new tag and
   **extend** ours rather than dropping it. **The component set follows upstream's
   `.github/actions/windows-setup-cuda/action.yml` at the pinned `GIT_TAG`, the component versions
   follow NVIDIA's redist manifest** (`CUDA_REDIST_MANIFEST`, e.g. `13.4.3` = `redistrib_13.4.3.json`):
   the step reads each component's `relative_path` and `sha256` from the manifest, so the archive it
   downloads and the hash it checks cannot name two different versions. They did once -- the first
   full run of 6.0.0 (38808518362) paired upstream's hand-written `cuda_crt-…-13.4.59` with the newest
   manifest's hash for another `cuda_crt` and failed on the first archive. Component versions differ
   per component (cuBLAS and CCCL have their own numbering), which is why the manifest and not the
   CUDA version decides them. (This replaced `Jimver/cuda-toolkit`, which never shipped 13.4.)
3. **Only for a major bump:** the backend name `cuda13` — in `llama/CMakeLists.txt`
   (`JLLAMA_BACKEND`), `.github/natives.csv` (the two `cuda13-*` rows), the generated pom executions
   (`check-natives.py pom`), the build jobs' artifact names, `LlamaLoader.MODULE_BACKENDS` and the
   `examples/server/pom.xml` profiles. `check-natives.py` fails until they agree. No change for a
   minor bump.
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
dominant cost of the CUDA job. **nvcc goes through `sccache` again since 2026-10-25** (`build.sh` adds
`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache` when it detects `GGML_CUDA` in the cmake args;
`SCCACHE_WRAP_NVCC=false` opts out), so the per-arch `.cu` device passes cache over Depot alongside
the gcc C/C++ TUs -- with CUDA 13.2 that was ~51 min cold, ~15 min warm. From 2026-10-09 to 2026-10-25
it was off: with CUDA 13.4 and sccache 0.18.0 every run failed on the first `.cu` TUs (`Missing "cubin"
file output`, `Compiler killed by signal 126`) and the uncached retry rebuilt everything. The trigger
was ggml's default arch list, whose PTX-only `90-virtual` makes nvcc run a ptxas check pass without
`-o` (mozilla/sccache#2862, fix PR #2809); the list this project passes since 2026-10-11 (every arch
`-real`, only `121-virtual` PTX-only) does not hit it -- measured in `nvidia/cuda:13.4.3` with b11815:
108/108 hits warm with sccache 0.18.0, while ggml's list fails as CI did. The "signal 126" was sccache
printing an exit code as a signal, fixed in 0.19.0 (mozilla/sccache#2932), which is the pinned version
now. Either way CI keeps the **full arch set on every run** (release-safe everywhere), never a
reduced arch set; the cache is what makes it affordable.

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
that reach Central) runs on every GPU generation. The full-arch CI cost is paid by the job
on a cold cache; warm, the sccache-over-Depot cache covers the nvcc device passes and the gcc TUs
(the launcher is gated behind the same probe, and `build.sh`'s mid-build retry treats an sccache
`Compiler not supported` / `fatbinary fatal` failure like any other cache error and rebuilds the job
without any launcher rather than redding it).
**Measured with CUDA 13.2:** a warm run in the manylinux_2_28 container hit **100%** on CUDA / CUBIN /
device-code (139 CUDA hits, 99.86% overall, 3 misses) and cut the job from **~51 min cold to ~15 min
warm**. `build.sh` prints `sccache --show-stats` at the end of every run so the hit table stays visible.

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

- Its toolchain file sets `CMAKE_SYSTEM_NAME Android`, so CMake's **own** Android platform is
  active (`ANDROID` is 1 from `Platform/Android-Initialize.cmake`, ggml takes its `android_armv*`
  variant list, and no run path is ever emitted -- see "Modular natives"), but the NDK toolchain
  file's `ANDROID_ABI` / `ANDROID_PLATFORM` variables are never set, so any
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

A second Android arm64 build produces the OpenCL backend module with Adreno-tuned kernels
embedded: `libggml-opencl.so` alone (`JLLAMA_MODULE_ONLY`, see "Modular natives"). It ships as the
natives jar `opencl-android-aarch64` (directory `Linux-Android/aarch64/opencl/`) and as the
`llama-android-opencl` AAR, both **additive** to the CPU build (`cpu-android-aarch64` /
`llama-android`, which the AAR depends on): `JNI_OnLoad` loads the module by soname when it is in
the app's native-library directory. The `crosscompile-android-aarch64-opencl` job runs the
dockcross-android-arm64 build with `-DJLLAMA_MODULE_ONLY=ON -DGGML_OPENCL=ON
-DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON` and uploads
`natives-opencl-android-aarch64`; the rest is the natives-jar wiring (see "Natives jars" above).

Local sanity build:
```bash
.github/dockcross/dockcross-android-arm64 .github/build_opencl_android.sh \
  "-DOS_NAME=Linux-Android -DOS_ARCH=aarch64 -DJLLAMA_MODULE_ONLY=ON \
   -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
   -DGGML_OPENCL_USE_ADRENO_KERNELS=ON"
```
The module lands in `src/main/natives/net/ladenthin/llama/Linux-Android/aarch64/opencl/`.

The dockcross image does not ship OpenCL headers or a stub `libOpenCL.so`, so
`build_opencl_android.sh` first stages Khronos `OpenCL-Headers` and
cross-builds `OpenCL-ICD-Loader` into `/tmp/opencl-stage/` before invoking the
main project cmake with `-DOpenCL_INCLUDE_DIR=...` and `-DOpenCL_LIBRARY=...`.
At runtime the device must provide its own OpenCL ICD (`libOpenCL.so`);
Qualcomm Adreno drivers do. On a device without one the module fails to load
(logged) and the model runs on the CPU modules of the `llama-android` AAR.

## Windows natives (clang modular CPU + CUDA/Vulkan/OpenCL/ROCm/SYCL/OpenVINO modules + the MSVC build-and-test job)

The Windows x86-64 natives are the library jar `cpu-windows-x86-64` (in `llama-platform`: the
**modular build with plain clang**, Ninja Multi-Config, see "Modular natives") and six GPU module
jars, **`cuda13-windows-x86-64`**, **`vulkan-windows-x86-64`**, **`opencl-windows-x86-64`** (Ninja +
`cl.exe`), `rocm-windows-x86-64`, `sycl-windows-x86-64` and `openvino-windows-x86-64` (see
"Additional GPU-backend natives"). **32-bit Windows is gone** (`cpu-windows-x86` / `msvc-windows-x86`
were never tested by anyone; dropped with the modular natives), and so is the **MSVC jar**: the
Visual Studio generator + `cl.exe` build (`build-windows-x86_64-msvc`) is still built, `ctest`ed and
run through the full model-backed Java suite (`test-java-windows-x86_64-msvc`, artifact
`msvc-windows-x86_64`, outside the `natives-*` glob) as the second, independent toolchain that
proves the sources compile with MSVC -- but nothing ships from it, because a second static CPU
library next to the clang one would only be a second CPU library, and clang's is the faster one.

**Why plain clang.** Only plain clang produces all 14 x86 CPU variants (measured in the
`GGML_OPENMP=OFF` table below: clang is +5.6% on prompt processing and equal on generation against `cl.exe`, static
against static), and upstream's own `win-cpu` release uses the same LLVM toolchain. The Visual
Studio generator ignores `CMAKE_{C,CXX}_COMPILER_LAUNCHER`, so only Ninja Multi-Config can front
`cl.exe` with sccache over Depot WebDAV, which is why the GPU module jobs use Ninja + `cl.exe`. All
Windows builds use the same hybrid CRT (`CMAKE_MSVC_RUNTIME_LIBRARY` + the UCRT linker flags, same
Release flags, same runner), so the DLLs are **functionally equivalent with identical runtime
dependencies** across toolchains. (Upstream llama.cpp also builds its Windows artifacts with Ninja
Multi-Config.)

**`GGML_OPENMP=OFF` on every Windows CPU job -- it is faster, not a concession.** Both x86-64 CPU
jobs (clang modular, MSVC build-and-test) pass it, as the arm64 jobs already did for a dependency reason. Measured
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
arm64 job hit.

**Linux passes `GGML_OPENMP=OFF` now, and that is a deployment decision rather than a throughput
one** (`llama/CMakeLists.txt`, forced for `OS_NAME MATCHES "^Linux"` and not Android).
`libggml-base.so` -- the first file `LlamaLoader` loads -- declares `libgomp.so.1` whenever ggml
builds with OpenMP, and GCC's `-fopenmp` links it *dynamically*, so it is a runtime requirement of
the shipped jar that a plain Ubuntu server image and a distroless Java image do not satisfy; the
whole jar then fails with `The native library of backend 'cpu' could not be loaded`, which reads like
a packaging defect. It costs nothing: built twice on one 16-vCPU machine and benchmarked with
`llama-bench`, generation is slightly *faster* without OpenMP at 4, 8 and 16 threads and prompt
processing lower by 0.4-2.1%, all inside the error bars
([`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md)); the ~6% figure
in the paragraph below was a single sample and does not reproduce. **Only OpenMP:**
`libstdc++`/`libgcc_s` stay dynamic, because HotSpot's own `libjvm.so` links them -- every image that
can run this library has them -- and `-static-libstdc++` would give each shared object its own
unwinder for no gain. `verify-native-deps.py` enforces the *absence* (`libgomp.so.1` is out of the
three Linux `cpu` allowlists, with a unit test saying why widening it again defeats the purpose).
The older throughput measurement follows and still stands on its own terms:

**Linux used to keep OpenMP, and that was measured, not an omission.** The Linux CPU jobs passed no
`GGML_OPENMP`, so they shipped with it ON (ggml's default) through **libgomp** -- and libgomp does
*not* show the penalty. Measured the same way in the project's own `manylinux_2_28_x86_64` image
(gcc 14.2.1, the compiler and libgomp the `crosscompile-linux-x86_64` job uses; same model, 8
threads, interleaved): tg128 72.8 with OpenMP against 77.3 without, pp512 357.0 against 358.7 --
roughly 6% on generation with one ON sample at 76.1 +- 0.9, i.e. inside the spread rather than a
result, and nothing like the 1.9x/2.7x the Windows runtimes cost. libgomp evidently keeps its
thread team alive across regions where LLVM's `libomp` and MSVC's `vcomp` do not. So the Windows
numbers must **not** be generalised into a one-line change on Linux on *throughput* grounds.
**The dependency half of that sentence used to read "libgomp is present everywhere", and that is
measured and wrong:** `libggml-base.so` -- the first file `LlamaLoader` loads -- declares
`libgomp.so.1`, and a plain Ubuntu 24.04 server image does not have it; the jar then fails with
`UnsatisfiedLinkError: The native library of backend 'cpu' could not be loaded`, which reads like a
packaging defect and is a missing distro package (measured 2026-10-11 on an EC2 Ubuntu AMI, fixed by
installing `libgomp1`). Every shipped Linux file also needs `libstdc++.so.6` and `libgcc_s.so.1`, so
a distroless or musl image cannot run the jar at all. Of making the Linux jar self-contained the way
the Android build is (`GGML_OPENMP=OFF` plus `-static-libstdc++`), the OpenMP half is done (the
paragraph above) and the C++ runtime half declined, since every JVM that can load the library brings
`libstdc++`/`libgcc_s` itself. Unmeasured: a machine with many more cores than the 8 measured here.

**GPU runtime libraries are NOT bundled.** A GPU natives jar ships only its backend module
(`ggml-cuda.dll`, `ggml-vulkan.dll`, ...; see "Modular natives"). The consumer's driver/toolkit must supply the runtime: CUDA needs the
installed CUDA 13 Toolkit (`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll` on `PATH`); Vulkan
needs `vulkan-1.dll` (ships with current GPU drivers); OpenCL needs the vendor ICD
(`System32\OpenCL.dll`). Not bundling = no NVIDIA-EULA redistribution obligation. **GitHub-hosted
Windows runners have NO GPU**, so the GPU jobs **build the artifact only** (no `-DBUILD_TESTING`/`ctest`)
— a GPU-linked `jllama_test.exe` can't even be enumerated on a GPU-less runner (it errors probing for a
device, so `gtest_discover_tests` registers a failing `*_NOT_BUILT` sentinel). The CPU-only C++ unit
suite is fully covered by the `C++ Tests` job + the CPU Windows jobs; model-backed GPU inference is
local / self-hosted.

Windows-specific wiring (the rest is the natives-jar wiring, see "Natives jars" above):

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
   sccache v0.19.0 zip + Depot WebDAV), each uploading `natives-<classifier>`:
   - `build-windows-x86_64` — **modular CPU, plain clang** (`-DJLLAMA_BACKEND_DL=ON
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`; no `-DBUILD_TESTING` --
     `test-cpp-windows-x86_64` builds the static library with the tests, same compiler).
   - `build-windows-x86_64-msvc` — **MSVC CPU, build + `ctest`, no jar** (artifact `msvc-windows-x86_64`
     for `test-java-windows-x86_64-msvc`).
   - `build-windows-x86_64-cuda` — CUDA `13.4` assembled from NVIDIA's redist archives (upstream's
     `windows-setup-cuda` component list; `Jimver/cuda-toolkit` stops at 13.3.1) + `-DJLLAMA_MODULE_ONLY=ON -DGGML_CUDA=ON`.
   - `build-windows-x86_64-vulkan` — `jakoch/install-vulkan-sdk-action` + `-DJLLAMA_MODULE_ONLY=ON -DGGML_VULKAN=ON`.
   - `build-windows-x86_64-opencl` — `build_opencl_windows.bat -DJLLAMA_MODULE_ONLY=ON -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON`.
   Each GPU job uploads one `ggml-<backend>.dll` (plus `jllama-build.txt` / `jllama-files.txt`).

**First CI run (PR #276, run 28327740376):** the default Ninja CPU flip, the MSVC classifier, and the
**OpenCL** job were green on the first try. Two GPU jobs needed a toolchain fix: **CUDA** failed with
`Version not available: 13.0.0` because the pinned `Jimver/cuda-toolkit@v0.2.24` predated CUDA 13.x →
bumped to `@v0.2.35` + `13.2.0` (matches the Linux pin, classifier stays `cuda13-…`); **Vulkan** failed
`find_package(Vulkan)` because `humbletim/install-vulkan-sdk` set `VULKAN_SDK` but laid the SDK out in a
way CMake's `FindVulkan` couldn't read → switched to `jakoch/install-vulkan-sdk-action` (purpose-built,
FindVulkan-compatible). Because every Windows build job is in the `package`/publish `needs:` graph, a
GPU-toolchain failure blocks packaging — the same release-gating policy every build job follows.

**sccache on every Windows Ninja job.** Every Ninja build job installs sccache (x86_64 or the native
`aarch64` release) through the composite action `.github/actions/install-sccache-windows` (arch from
`RUNNER_ARCH`; its version must equal `SCCACHE_DL_VERSION` in `build.sh`) with the same `USE_CACHE` /
`SCCACHE_WEBDAV_*` env; only the MSVC build-and-test job cannot, because the Visual Studio generator
ignores compiler launchers.

**The plain-clang jobs cache because of `llama/cmake/msvc-runtime-gnu-clang.cmake`.** sccache refuses
any TU carrying an `-Xclang` argument it does not know (`Can't handle UnknownFlag arguments with
-Xclang`; 0.18.0 and 0.19.0), and CMake's `Platform/Windows-Clang.cmake` implements
`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` (the hybrid CRT every Windows job sets) for clang with the
GNU front end as `-Xclang -flto-visibility-public-std -D_MT -Xclang --dependent-lib=libcmt` -- and
`-flto-visibility-public-std` is the one sccache does not know (`--dependent-lib` and llama.cpp's own
`-Xclang -fno-pch-timestamp` it does). So `build-windows-x86_64` (plain clang, modular CPU; measured at
b11538: 719 of 720 requests non-cacheable) ran without sccache, and **the Windows ROCm job** -- TheRock's
clang is a plain clang, and upstream compiles the HIP sources as C++ on Windows (`CXX_IS_HIPCC` on
`WIN32`) -- sent its ~150 HIP TUs, each compiled for the 23 `GPU_TARGETS`, through a launcher that refused
every one: 2:40 h to 4:15 h for its build step, the pipeline's critical path. The override file, loaded
through `CMAKE_USER_MAKE_RULES_OVERRIDE` (set before `project()` in `llama/CMakeLists.txt`, so CMake
reads it after its platform files while enabling each language), maps `MultiThreaded[Debug]` to the
driver flag `-fms-runtime-lib=static[_dbg]`, which `llama/CMakeLists.txt` already passed next to CMake's
sequence: same runtime selection in cc1, no `-Xclang`. **Measured before the change** (locally, MSVC
14.44 as on the `windows-2022` runner, TheRock 10.0.0 clang 23, Ninja Multi-Config, sccache 0.18.0 over
a local WebDAV server, `GPU_TARGETS=gfx1100`): the build as it was, 157 requests all refused, 325 s;
with the override, 0 refused, 377 s cold and **25 s warm**. Nine TUs (C, C++, seven HIP) compiled with
a pinned `-cuid` from the same build directory: every section identical between the two flag sets --
`.text` and the HIP device code included -- except `.drectve`, where CMake's sequence repeated
`/DEFAULTLIB:libcmt.lib`; through sccache and without it identical up to the COFF timestamp clang
writes on every compile. Two things that did **not** work, so nobody tries them again: an empty
`CMAKE_MSVC_RUNTIME_LIBRARY` reaches `try_compile`, whose C compiler check then links `-nostdlib`
without a CRT (`undefined symbol: mainCRTStartup`, configure fails); and `-fuse-cuid=none` to compare
whole DLLs makes every TU define the same `__hip_cuid_` (`duplicate symbol` at link). The CUID is also
why a HIP DLL built through sccache is not byte-identical to one built without it: clang hashes the
command line into `__hip_cuid_<hash>`, sccache's command line differs slightly, and a hash with a
leading zero is one byte shorter and shifts the device code layout -- names and addresses, not code
(see "CI build cache"). It cannot red a build, by
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
:: the static test build (cl.exe) and the shipped modular CPU build (plain clang)
.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DBUILD_TESTING=ON
ctest --test-dir build --output-on-failure
.github\build.bat -G "Ninja Multi-Config" -DJLLAMA_BACKEND_DL=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOS_NAME=Windows -DOS_ARCH=x86_64
:: GPU module jars (needs the matching SDK installed + on PATH); each writes one ggml-<backend>.dll next to the CPU set
.github\build.bat -G "Ninja Multi-Config" -DJLLAMA_MODULE_ONLY=ON -DGGML_CUDA=ON   -DOS_NAME=Windows -DOS_ARCH=x86_64
.github\build.bat -G "Ninja Multi-Config" -DJLLAMA_MODULE_ONLY=ON -DGGML_VULKAN=ON -DOS_NAME=Windows -DOS_ARCH=x86_64
.github\build_opencl_windows.bat -G "Ninja Multi-Config" -DJLLAMA_MODULE_ONLY=ON -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DOS_NAME=Windows -DOS_ARCH=x86_64
```

## Linux Vulkan natives + Windows arm64 CPU and Vulkan

Four natives jars that extend the matrix toward upstream llama.cpp's release set.

**Linux Vulkan (`vulkan-linux-x86-64` + `vulkan-linux-aarch64`).** A vendor-neutral GPU module jar
for Linux (NVIDIA / AMD / Intel) with no CUDA toolkit. The build jobs are `build-linux-x86_64-vulkan`
(native `ubuntu-latest`, **not** dockcross — the Vulkan SDK is a trivial apt install and upstream
builds ubuntu-vulkan the same way) and `build-linux-aarch64-vulkan` (`ubuntu-24.04-arm` + GCC 14).
Both `apt-get install libvulkan-dev glslc glslang-tools`, build `-DJLLAMA_MODULE_ONLY=ON
-DGGML_VULKAN=ON -DGGML_NATIVE=OFF` (one `libggml-vulkan.so`; no `mvn compile`, a module-only build
has no JNI), and are **build-only** (no `ctest`: a module-only build has no `jllama_test`, and a
Vulkan-linked one would error enumerating devices on a GPU-less runner anyway — same as the Windows
GPU jobs). The module's glibc floor is the ubuntu baseline; it lands next to a manylinux_2_28
library, which is fine for a GPU artifact (a host with a Vulkan driver is not a RHEL 8 host). GPU
runtime `libvulkan.so.1` is supplied by the consumer's driver — nothing is bundled (same policy as
every GPU backend).

**Windows arm64 Vulkan (`vulkan-windows-aarch64`).** Added at the llama.cpp b11395 bump, when upstream
put a Windows arm64 Vulkan build into its release set (#29954). `build-windows-arm64-vulkan` uses the
arm64 CPU job's toolchain below (`windows-11-arm`, `clang-cl`, `GGML_OPENMP=OFF`) and installs the SDK
exactly as upstream's release job does: LunarG's x64 installer with the `com.lunarg.vulkan.arm64`
component (the arm64 import library), whose x64 `glslc` runs under the runner's x64 emulation; the SDK
version is upstream's `VULKAN_VERSION`, not the one the x86-64 Vulkan job pins through
`jakoch/install-vulkan-sdk-action`. Build-only like every GPU job (`-DJLLAMA_MODULE_ONLY=ON`, one
`ggml-vulkan.dll`), and part of the `windows-aarch64` smoke set, so that row of `smoke-natives`
puts it next to the CPU library and sees ggml skip it on the GPU-less runner.

**Windows arm64 CPU (`cpu-windows-aarch64`, in `llama-platform`).** `build-windows-arm64` runs
natively on GitHub's free `windows-11-arm` runner (`ilammy/msvc-dev-cmd` `arch: arm64`, Ninja
Multi-Config, `-DOS_ARCH=aarch64 -DJLLAMA_BACKEND_DL=ON`) and writes `Windows/aarch64/cpu/`: the
modular layout (`jllama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll` and **one** `ggml-cpu.dll`
-- ggml has no ARM variant list for Windows, so no `GGML_CPU_ALL_VARIANTS`), so the Vulkan and OpenCL
module jars can join it. `ctest` moved to **`test-cpp-windows-arm64`** (a static `clang-cl` build with
the tests on the same runner), as on every modular platform. No Java change
was needed: `OSInfo` already maps a Windows-on-ARM JVM (`os.arch=aarch64`)
to `Windows/aarch64` (it isn't in `archMapping`, so it falls through `translateArchNameToFolderName`).
sccache runs here too, from its native `aarch64-pc-windows-msvc` release (it wraps `clang-cl`; see
"sccache on every Windows Ninja job" above). **Compiler: `clang-cl`, not MSVC
`cl.exe`.** ggml's `ggml-cpu/CMakeLists.txt` aborts with *"MSVC is not supported for ARM, use clang"*
via `if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")`; `clang-cl` (LLVM's MSVC-compatible driver)
satisfies that guard (compiler id `"Clang"`) while keeping CMake's `MSVC=TRUE`, so the hybrid-CRT
block still applies and the generator stays Ninja Multi-Config. The job passes
`-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl`; `msvc-dev-cmd` supplies the MSVC
headers/libs/linker **and the bundled clang-cl/lld-link** (`VC\Tools\Llvm\ARM64`), so no separate
LLVM install is needed. It also passes **`-DGGML_OPENMP=OFF`**: with clang-cl, ggml links LLVM's
OpenMP (`libomp.lib` → `libomp140.aarch64.dll` at runtime), which — unlike MSVC's `vcomp140.dll`,
which at least ships in the VC++ redistributable — is in no redistributable at all, so the test exe
(and any consumer) failed to launch with `0xc0000135` (`STATUS_DLL_NOT_FOUND`). Disabling OpenMP
makes ggml use its own `std::thread` threadpool, leaving the arm64 `jllama.dll` self-contained.
**Every** Windows CPU job passes the flag now, arm64 and x86 alike -- for arm64 it was the
dependency, for x86 it is a measured ~2x on token generation (see "`GGML_OPENMP=OFF` on every
Windows CPU job" above), so the two reasons converged on one setting. (Upstream llama.cpp instead cross-compiles arm64 from an
x64 runner with `vcvarsall amd64_arm64` + a `clang`/`clang++` toolchain file and no arm64 tests; the
native-runner + `clang-cl` route here keeps the hybrid CRT and lets `ctest` run on real ARM hardware.)

## Additional GPU-backend natives (ROCm/HIP, SYCL, Win-arm64 OpenCL, OpenVINO)

Seven further GPU module jars extend the matrix toward upstream llama.cpp's full release set, with
the same natives-jar wiring as every other one (see "Natives jars"; each build job is in
`package.needs`, **fail-loud**). All are **build-only** (GitHub runners have no matching GPU),
module-only (`-DJLLAMA_MODULE_ONLY=ON`, one `libggml-<x>.so` / `ggml-<x>.dll`) and bundle **no**
vendor runtime.

| Classifier | GGML flag(s) | Job runner / toolchain | Directory / module |
|---|---|---|---|
| `rocm-linux-x86-64` | `GGML_HIP=ON -DCMAKE_HIP_COMPILER=… -DGPU_TARGETS=…` | `ubuntu-latest` + ROCm 10 TheRock wheels (pip, `rocm-sdk path`) | `Linux/x86_64/rocm/libggml-hip.so` |
| `rocm-windows-x86-64` | `GGML_HIP=ON` | `windows-2022` + ROCm 10 TheRock wheels (pip) | `Windows/x86_64/rocm/ggml-hip.dll` |
| `sycl-linux-x86-64` | `GGML_SYCL=ON -DGGML_SYCL_F16=ON` (`icx`/`icpx`) | `ubuntu-latest` + Intel Deep Learning Essentials 2026.1.4 + oneDNN 2026.0.2 (offline installers) + Level Zero 1.33.1 debs, as upstream's `ubuntu-24-sycl` | `Linux/x86_64/sycl/libggml-sycl.so` |
| `sycl-windows-x86-64` | `GGML_SYCL=ON -DGGML_SYCL_F16=ON` (`icx`) | `windows-2025-vs2026` + Deep Learning Essentials 2026.1.4 + oneDNN 2026.0.2 (offline installers) + the Level Zero SDK zip, as upstream's `windows-sycl` | `Windows/x86_64/sycl/ggml-sycl.dll` |
| `opencl-windows-aarch64` | `GGML_OPENCL=ON …ADRENO_KERNELS=ON` (clang-cl, `GGML_OPENMP=OFF`) | `windows-11-arm` (arm64 CPU job's toolchain) | `Windows/aarch64/opencl/ggml-opencl.dll` |
| `openvino-linux-x86-64` | `GGML_OPENVINO=ON` | `ubuntu-latest` + OpenVINO apt | `Linux/x86_64/openvino/libggml-openvino.so` |
| `openvino-windows-x86-64` | `GGML_OPENVINO=ON` | `windows-2025-vs2026` + OpenVINO archive | `Windows/x86_64/openvino/ggml-openvino.dll` |

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
Windows `jllama.dll` ~1 GB for its 23 targets (234 MB zipped, so the jar never showed it) when the
HIP backend was still linked into it; the same code sits in `ggml-hip.dll` now. That size also lands
on disk: `LlamaLoader` extracts every module jar on the classpath next to the library, i.e. on every
machine whose classpath carries the ROCm jar, with or without an AMD GPU. `llama/CMakeLists.txt`
therefore adds `--offload-compress` to the `ggml-hip` target only
(`$<COMPILE_LANGUAGE:HIP,CXX>`: its sources are HIP on Linux and CXX on Windows, where upstream
compiles HIP as C++), so clang stores every bundle zstd-compressed (a `CCOB` bundle) and the HIP
runtime inflates it when the module loads. Upstream llama.cpp does **not** do this; its
`ggml-hip.dll` carries the same uncompressed code (for 20 targets). `.github/verify-hip-offload-compressed.py` runs after the build in
both ROCm jobs (over `ggml-hip.dll` / `libggml-hip.so`, `hipoffload.NAMES`), prints the module size and bundle counts (also into the job summary), and fails on
any uncompressed bundle (`__CLANG_OFFLOAD_BUNDLE__`) or on none compressed — a toolchain or upstream
change that drops the flag reds the job instead of quietly shipping the 1 GB library again. The
jar barely shrinks (zip already compressed the code); what shrinks is the extracted library.

**SYCL is fp16 on both OSes, one jar each**, as upstream's SYCL release builds (`GGML_SYCL_F16=ON`).
The second Linux precision (`sycl-fp32-linux-x86-64`, fp32 accumulation) was dropped with the modular
natives: nobody had asked for it, and two modules of the same backend on one classpath would both
register the same device. **Both SYCL jobs install upstream's toolchain, pinned by installer URL**
(`KEEP IN SYNC WITH UPSTREAM` in `publish.yml`, read off `release.yml`'s `ubuntu-24-sycl` / `windows-sycl` jobs
at the pinned tag; the bump runbook lists it): Intel's Deep Learning Essentials bundle (DPC++, MKL, TBB) plus
the separate oneDNN installer (b11654, #30131 -- until the b11800 bump the Linux job took the same components
from Intel's apt repository, unpinned, and the Windows job the 2025.3.3 bundle that still carried oneDNN), and
the Level Zero loader/SDK upstream's jobs have had since #21597 / #29273 (`GGML_SYCL_SUPPORT_LEVEL_ZERO_API`,
default ON, links `ze_loader` when found and only warns without it). **The Level Zero half changed the shipped
modules**: the ones built from the apt packages / the 2025.3 bundle never found a loader (run 38870889665's
package job listed none), so both modules import it now -- `libze_loader.so.1` / `ze_loader.dll`, the Intel GPU
driver's -- and the Windows module needs the oneAPI 2026 runtime (`sycl9.dll`, `mkl_sycl_blas.6.dll`) instead of
2025.3's `sycl8.dll` / `mkl_sycl_blas.5.dll`; the README classifier table names both. **oneDNN is required**: since
llama.cpp #29107 (b11636) `GGML_SYCL_DNN` (default ON) fails the configure
when oneDNN is missing or is a CPU-only build, where it used to warn and build a module whose flash attention
runs the slower MKL path -- which is what the Linux module had been until the b11639 bump. The module then
needs `libdnnl.so.3` / `dnnl.dll` at run time, from the same oneAPI environment (`setvars.sh`) as MKL.

The Linux jobs that install a multi-GB vendor toolchain (CUDA, ROCm, SYCL) start with
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

**The end-to-end gate: `smoke-natives-macos`.** The three macOS Java test jobs each test the dylib
*their own job* built, so until now nothing exercised the **packaged** artifact on macOS — Linux and
Windows had smoke jobs downstream of `package`, macOS had none, which is why this bug reached
three releases with a fully green pipeline. The job (`needs: [package]`, `macos-15`, gates both
publish jobs) downloads the `llama-smoke-macos-aarch64` smoke set and runs
`.github/smoke-native-macos.sh`, which extracts `Mac/aarch64/metal/libjllama.dylib` from the
**`metal-macos-aarch64` natives jar** of the set and asserts two things: `codesign --verify
--strict` (re-hashes the code pages against the signature's stored hashes — the direct check for a
dylib assembled from two builds) and a real JVM load via `.github/smoke/NativeLoadSmoke.java`
(compiled with `javac` first and run as `java -cp <classes>:'<set>/*' NativeLoadSmoke` -- not through
the single-file source launcher, which Java 8 lacks and the post-release consumer runs this script
on), which forces
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

Two macOS specifics: its smoke set holds **one** natives jar (macOS has no GPU module jar: Metal is
compiled into the library), and it asserts native loadability plus the code signature rather than a
server round trip, so it does not reuse `smoke-test-natives.sh`. Deliberately cache-free (~1 min, no
GGUF restore; its HTTPS round downloads the 1 MB `stories260K.gguf` itself): a full model-backed
macOS server smoke would be strictly more, but this catches the failure class that actually shipped
and is cheap enough to always run. It is the macOS member of the cross-repo convention in
[`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md)
("No release asset is attached that CI has not run" -- here: no natives jar is published that CI has
not loaded), which BAF and srcmorph implement with the `java/smoke-fatjar-cli` action of bernardladenthin/actions.

## Smoke sets: the published jars launched before they are published

There are **no fat jars** (see "Natives jars"), so the artifacts CI has to launch are the natives
jars themselves, on the classpath a consumer gets from Maven. The `package` job assembles one
**smoke set** per desktop platform -- `.github/package-smoke-sets.sh <llama/target> <runtime-deps>
<out>`: the classes jar, its runtime dependencies (`mvn dependency:copy-dependencies
-DincludeScope=runtime`), the platform's library jar and every GPU module jar of the platform, the
jar list taken from `check-natives.py smoke-set <target>` -- and uploads each alone as
`llama-smoke-<target>` (`linux-x86-64`, `linux-aarch64`, `windows-x86-64`, `windows-aarch64`,
`macos-aarch64`), small enough for a matrix row to download. `check-natives.py` holds the uploads
and the rows of `smoke-natives` to the targets `natives.csv` derives, so a new platform cannot ship
unlaunched.

1. **`smoke-natives`** (matrix, `fail-fast: false`; rows `linux-x86-64` on `ubuntu-latest`,
   `linux-aarch64` on `ubuntu-24.04-arm`, `windows-x86-64` on `windows-2025-vs2026`,
   `windows-aarch64` on `windows-11-arm`; `needs: [package]`). Each row runs
   `java -cp '<set>/*' net.ladenthin.llama.server.ServerLauncher` (`smoke-test-natives.sh`, in
   PowerShell `smoke-test-natives.ps1`) on a GPU-less runner with the draft model: poll
   `/health` to 200, assert a `/v1/chat/completions` choice, and require the loader line
   `[jllama] native backend '<lib>' loaded ... with GPU module(s) ...` naming **every** module jar of
   the platform (read from `natives.csv`, `smoke-lib.sh` / `Get-SetModules`) -- so every GPU module
   put in place and skipped by ggml for lack of a runtime is exercised on the published jars; then
   the HTTPS round (see "HTTPS: BoringSSL, statically"). The Linux rows re-run the bytecode gate over
   the set and the `linux-x86-64` row the RPC smoke (`smoke-rpc.sh`, two JVMs from the same set).
2. **`smoke-natives-macos`** -- the `macos-aarch64` set, see "macOS arm64".
3. **`smoke-agent-linux`** -- the published agent's thin jar with its dependencies on one classpath
   with the `linux-x86-64` set (see "Local coding agent").

`publish-snapshot` / `publish-release` `need` all three (a matrix job is done when every row is;
fail-loud gating). They stage and sign every Central artifact locally, and `finalize-publication`
collects and freezes the publication (see "Signed publication"): the classes jar, every natives jar,
every module's pom (`llama-parent` and `llama-platform` included: maven-gpg writes
`target/<artifactId>-<version>.pom` next to its `.pom.asc`, also for a pom-only module), each reactor
module's CycloneDX JSON, sources, javadoc, the agent's thin jar, the signed llama.cpp source archive,
and the two AARs with the `.aar.asc` Gradle writes next to them, their poms
(`build/publications/<publication>/pom-default.xml`) and their sources/javadoc jars
(`build/distributions/`). `github-snapshot` / `github-release` attach exactly those files and sign or
assemble nothing (no `sign-fatjars.sh`, no `environment:`). Until 2026-10-12 the collect steps copied
jars and bare AARs only: every pom was missing, `llama-platform` was absent entirely, the AARs went out
unsigned and the `*.jar`-only signature check never noticed. The collection is shared across all
four Java repositories (`java/finalize-publication` of bernardladenthin/actions); the local
`.github/release-assets.json` declares the native/Android extras. It discovers all reactor modules from their POMs and all native classifiers from
`natives.csv`, includes `llama-ffm`, renames each `target/bom.json` to its Central name and reports
missing outputs/signatures. After a complete upload, `github-snapshot` deletes every asset of the
`snapshot` pre-release the run did not upload, each printed by name (`gh release upload --clobber`
never removed anything: 208 assets, 109 of them from 5.0.x). The prune refuses an empty upload,
treats an asset that is already gone as removed and fails only after trying every deletion; a
versioned release (`v<digit>...`) is refused as a rolling tag, so it is never pruned. An incomplete
collection never reaches the job (the gate fails first), and the snapshot update itself refuses one
before any change. The whole snapshot update is serialized; it rejects late older
commits/runs, and updates the source tag and notes to the uploaded commit and run after a complete
upload; the first snapshot of a repository is a draft until then, so a failed first upload leaves
neither a public pre-release nor a tag. Why it prunes asset by asset instead of
deleting and recreating the release -- the job's first form, which failed in every burst of main
pushes in May 2026 -- is recorded in `workspace/policies/fat-jar-release-assets.md`, "The `snapshot`
pre-release holds exactly the current upload".

The common part of the three smoke scripts is `.github/smoke-lib.sh` (`smoke_classpath`,
`smoke_set_modules`, `smoke_assert_loader_line`); the PowerShell script repeats it, since the Windows
rows run without bash.

## WebUI (llama.cpp Svelte UI) embedding

The llama.cpp WebUI is **built once in CI and shared to every native build**, then
compiled into `libjllama` so the embedded server (`server-http.cpp`) can serve it.
This repo commits no build outputs, so the assets are produced per-pipeline, never
checked in (same policy as the native libs).

Pipeline (`.github/workflows/publish.yml`):

1. **`build-webui` job** (ubuntu — the *only* job that runs `npm`): resolves the
   pinned `b<nnnn>` tag from `llama/CMakeLists.txt`'s `JLLAMA_GIT_TAG` (checking out
   `JLLAMA_GIT_COMMIT` instead when that is set, see "Upgrading/Downgrading"), sparse-checks-out
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

The WebUI version **auto-follows** the pinned `JLLAMA_GIT_TAG`: a llama.cpp version bump
needs no extra step here, `build-webui` re-reads the tag and rebuilds the matching UI.

**Building the WebUI locally** (optional — a plain `cmake` build uses the stub and
ships no UI):
```bash
# needs node/npm + network for the asset build; the embed step is plain cmake -P
git clone --depth 1 --branch b11968 https://github.com/ggml-org/llama.cpp /tmp/lc
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

Because `sccache` is **content-addressed** and llama.cpp is pinned (`JLLAMA_GIT_TAG b11968`), the
~280 upstream object files are byte-identical every run, so a warm cache recompiles only the
*changed* files. Depot's cache is **shared across all branches** (unlike GitHub's
per-branch `actions/cache`), so every branch builds incrementally; a `b<nnnn>` version bump
naturally invalidates the upstream entries (their content changed) with no manual step. It
stays `-O3`, and a hit is the object the same command line compiles (release-safe). That is
**bit-identical** to a clean build for C and C++; for HIP it is identical up to the per-TU CUID symbol
name, which clang derives from the command line sccache slightly changes (measured with a pinned
`-cuid`: byte-identical; see "sccache on every Windows Ninja job").

**HIP on Linux needs its own launcher.** There ggml compiles the HIP TUs as the CMake language HIP,
which `CMAKE_{C,CXX}_COMPILER_LAUNCHER` do not reach, so `build.sh` adds
`-DCMAKE_HIP_COMPILER_LAUNCHER=sccache` when `GGML_HIP` is in the cmake args. Before it the Linux ROCm
job counted 2 requests (both C/C++) for a 2:50 h build. Measured in Docker with the job's toolchain
(TheRock 10.0.0, clang 23, `GPU_TARGETS=gfx1100`, sccache 0.18.0 over a local WebDAV server): 310 s
uncached, 157 of 157 hits and 10 s warm. With sccache's **disk** backend (a local build, never CI) some
TUs fail with `Failed to parse included file path`: its preprocessor cache mode reads clang's `-E`
output as text, and with `--offload-compress` that output is a compressed bundle
(mozilla/sccache#2788, unmerged); `SCCACHE_DIRECT=false` avoids it, and the mode does not exist for
the WebDAV backend. Both scripts also default `SCCACHE_IDLE_TIMEOUT` to 0: with sccache's 600 s the
server stopped during long builds and the stats table counted only what came after (67 of ~160
requests in the Windows ROCm job, 2 in the Linux one).

**Safety / transparency.** It is **inert** until `DEPOT_TOKEN` is configured and on **fork
PRs** (secrets are hidden there) — those simply compile normally; the `Install sccache` step
is `continue-on-error`; `use_cache=false` forces a pristine, from-scratch build; and **a `v*` tag
build is always from scratch** (the workflow-level `USE_CACHE` is `false` on a tag ref whatever the
dispatch input says, TODO.md delivery roadmap item 8), so no release object comes out of the shared
cache -- the cache provider stays out of a release's trust chain. Crucially,
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
   ✅ **green, full-arch always; the gcc TUs cache, and the nvcc device passes do again since
   2026-10-25 (sccache 0.19.0)** -- uncached from 2026-10-09 to 2026-10-25 (the paragraph at the end
   of this item). With CUDA 13.2 `build.sh` also wrapped nvcc
   (`-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache`, scoped to CUDA builds), so both the gcc C/C++ TUs
   (134 model files + ggml + httplib) **and** the per-arch `.cu` device passes cached over Depot.
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
   like every other sccache/nvcc incompatibility instead of redding the job. **CUDA 13.4 with sccache 0.18.0: a third
   failure shape, and why nvcc wrapping was opt-in from 2026-10-09 to 2026-10-25.**
   `sccache: caused by: Missing "cubin" file output` followed by `sccache: Compiler killed by signal
   126`, on the first `.cu` TUs ten seconds into the build, in **every** CUDA job log examined since
   the 13.4 bump of 2026-09-27 -- runs 37643964068 and 37831180985 (2026-10-07/08), 37680715063 (the
   b11476 bump) and 37925157175 (#492, 2026-10-09). Each fell back to the uncached retry, which took
   47 to 80 minutes depending on the runner (job totals of 54 to 93 min; none came near the warm
   ~15 min), so the launcher never paid anything back and the retry threw the gcc cache away with it.
   `build.sh` then wrapped nvcc only with `SCCACHE_WRAP_NVCC=true`. **Resolved 2026-10-25:** the
   failure needs a PTX-only `compute_90`/`compute_120` target (ggml's default list, `90-virtual`; the
   cause is mozilla/sccache#2862, fix PR #2809), which this project's own list (since 2026-10-11, see
   "Upgrading CUDA Version") no longer has -- measured locally: 108/108 hits warm with sccache 0.18.0.
   The "signal 126" was an exit code printed as a signal (mozilla/sccache#2932, fixed in 0.19.0, as is
   the same misreport for every ordinary nvcc compile error, which used to trigger this retry too). So
   nvcc is wrapped by default again with sccache 0.19.0; `SCCACHE_WRAP_NVCC=false` opts out. The first
   warm run's stats table is the check that it pays back.
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
version is the `SCCACHE_DL_VERSION` knob in `build.sh` (default **0.19.0**; overridable per-job
to try a different build against a container that crashed another). **Windows** is handled
separately (the Visual Studio generator ignores `CMAKE_*_COMPILER_LAUNCHER`): see
"Windows native classifiers" below — the **default** Windows CPU JAR now uses the **Ninja
Multi-Config** generator (so it caches) with a `build.bat` sccache probe and a direct sccache zip
download (not `mozilla-actions/sccache-action`); the uncached MSVC build is build-and-test only
(no jar), and the Windows GPU module jobs (CUDA/Vulkan/OpenCL/ROCm/SYCL/OpenVINO) use the same Ninja
path.

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
- **`.github/check-patches.py`** (`buildcheck/patches.py`, the `code-style` job) -- audits every hunk of
  every patch **as text**: its header must declare exactly the lines its body carries. `git apply --check`
  does not see this -- it reads a hunk by its counts and skips what follows as the start of the next
  header -- so a new-file hunk with more `+` lines than its header says applies "cleanly" and writes a
  **truncated file**. That shipped in #489: two comment lines added inside `0017`'s `prefetch.h` without
  recounting, `prefetch.h` cut off inside its one function, run 37922404867 red in 24 jobs, corrected by
  #492. The check costs milliseconds, needs no llama.cpp source, and runs in the first minutes of every
  run; `test_patches.py` pins the exact historic patch text as its negative case. Checking that a patch
  applies is not checking that it applies correctly -- after editing a patch by hand, run it.
- **`llama/CMakeLists.txt`** — wired as the llama.cpp `FetchContent_Declare(... PATCH_COMMAND ...)`, so it
  runs for **every** C++ build (all CI jobs *and* local `cmake -B build`) from one place — no
  per-build-step plumbing.

**On a llama.cpp version bump, every patch must still apply** — if a bump shifts the patched code,
the configure fails with an "does not apply cleanly" error; refresh the diff against the new source
and recommit. Treat `patches/` as part of the upgrade checklist below.

Current patches:

| Patch | Fixes |
|-------|-------|
| `0001-win32-arg-parse-embed-guard.patch` | Windows JNI regression from llama.cpp **#24779** (introduced b9739): on Windows `common_params_parse` re-derived argv from the **process** command line (`GetCommandLineW`) and adopted it, so an embedded/JNI caller (`java.exe`) lost its `--model …` args → "Failed to parse model parameters". b9789 narrowed the unconditional override to a **count-guard** (`if (static_cast<int>(utf8.buf.size()) == argc) { argv = utf8.ptrs.data(); }`), but that is exactly the variant the project already found breaks its Windows server-integration tests (when the embedded argv length coincides with `java.exe`'s). The patch carries the **complete upstream change** (so it can be submitted to llama.cpp verbatim and then dropped here): **(1)** `common_params_parse` parses **exactly the argv it is given** (no `GetCommandLineW` magic) and a new `common_params_parse_main()` wrapper holds the UTF-8 recovery for the standalone tools' `main()` (`common/arg.{cpp,h}`); **(2)** the **~34 standalone `main()` call sites** (every `common_params_parse(argc, argv, …)` across `tools/*`, `examples/*` and the `tests/*` programs) flip to `common_params_parse_main()`; **(3)** a `tests/test-arg-parser.cpp` regression case pins that `common_params_parse` honors a caller-supplied argv. The embedded caller (`jllama.cpp`) keeps calling `common_params_parse` and is never overridden. **Our subproject build compiles only the `arg.{cpp,h}` core** — `LLAMA_BUILD_TOOLS`/`LLAMA_BUILD_TESTS` are OFF for a FetchContent subproject — so the flips + test are applied-but-not-compiled here; they were validated via a one-off `-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=ON` build (the new test compiles and its asserts pass; `test-arg-parser`'s only red there is the live `ggml.ai` download check, which is sandbox-network, not the patch). Because it spans **35 files** it must be refreshed on every llama.cpp bump (the applier fails loud). **Refreshed at the b10679 bump:** upstream rewrote `tests/test-save-load-state.cpp`'s `main()` to take a `--models DIR` option, which it strips itself into a `filtered_argv` before calling `common_params_parse(fargc, filtered_argv.data(), …)`. That call site therefore stopped qualifying for the `_main()` flip — by this patch's own rule a caller that builds its own argv must use `common_params_parse` directly, so its argv is kept — and the hunk was **dropped** rather than refreshed (37 → 36 files). Caveat for whoever submits this upstream: that `main()` now filters a possibly-mojibake Windows argv *before* any UTF-8 recovery, so the fully correct upstream form there is recover-then-filter, not a one-line flip. It is out of scope for the downstream carry because `LLAMA_BUILD_TESTS` is OFF here, so the file is never compiled. **Refreshed the same way at the b11236 bump:** upstream #29426 rewrote `tests/test-recurrent-state-rollback.cpp`'s `main()` into the identical `--models DIR` / `filtered_argv` shape, so that hunk was dropped too (36 → 35 files); the same recover-then-filter caveat applies to it. **Still required at b10679, verified rather than assumed:** `common_params_parse` in pristine `b10679:common/arg.cpp` still carries the `#ifdef _WIN32` count-guarded `argv = utf8.ptrs.data()` override, and `common_params_parse_main` appears nowhere in `b10679:common/arg.h` — upstream has not adopted the fix. The upstream-facing write-up, including a standalone reproducer that makes llama.cpp's own `test-arg-parser` fail on unmodified `master`, lives in [docs/upstream-investigation-win32-argv-substitution.md](docs/upstream-investigation-win32-argv-substitution.md). **Reported upstream as [ggml-org/llama.cpp#26416](https://github.com/ggml-org/llama.cpp/issues/26416)** (2026-08-01, label `bug-unconfirmed`, first bad commit `508a475`); the issue asks which of the two directions the maintainers prefer before a PR is opened, so this patch stays downstream until they answer. **Refreshed at the b11612 bump:** #30023 (`a17e94d22`, "tests : consolidate context state tests into test-llama-context") deleted `tests/test-state-restore-fragmented.cpp`, so its hunk was dropped (36 → 35 `diff --git` headers); the consolidated `tests/test-llama-context.cpp` has the `--models DIR` / `filtered_argv` `main()` shape, so by this patch's own rule it keeps calling `common_params_parse` directly and gets no flip. Still required, verified at pristine b11612: `common/arg.cpp:1288` carries the count-guarded `argv = utf8.ptrs.data()` override and `common/arg.h` has no `common_params_parse_main`. |
| `0002-server-preserve-caller-load-progress-callback.patch` | Load-progress-callback regression introduced in llama.cpp **b9789**: `server_context::load_model` (`tools/server/server-context.cpp`) now **unconditionally** installs the server's own load-progress reporter on `params_base.load_progress_callback` immediately before `common_init_from_params`, clobbering any callback the embedding caller already set. libjllama's `LoadProgressCallback` feature wires `common_params.load_progress_callback` to a JNI trampoline *before* calling `load_model`, so the bump silently killed it — `LoadProgressCallbackTest` saw zero progress updates and the abort-on-`false` path never threw. The patch guards the assignment with `if (params_base.load_progress_callback == nullptr)`, so the server installs its own reporter **only when the caller hasn't** — a caller-supplied callback survives and fires during load. Standalone `llama-server` (no caller callback, so the field is null) is unaffected. Same JNI-vs-standalone divergence class as `0001`. **The guard is `== nullptr || == load_progress_callback`, and the second disjunct must never be dropped:** `load_progress_text` is a **local** of `load_model()`, and upstream re-assigns both fields on every call so the `user_data` always points at the current frame. `load_model()` runs a **second** time when resuming from the sleeping state (`--sleep-idle-seconds`), and by then `params_base` holds *our own* callback from the first load — a bare nullptr check skips the re-assignment and leaves `user_data` pointing into a **dead stack frame**, which segfaults inside `load_progress_callback()` on the first request after an idle window. That was a latent defect in this patch from the day it was written; only a second `load_model()` can reach it, and nothing exercised sleep until `IdleSleepWakeIntegrationTest` was added. |
| `0003-pr22393-server-add-slot-prompt-similarity-getter-setter.patch` | **Upstream-PR carry** of [ggml-org/llama.cpp#22393](https://github.com/ggml-org/llama.cpp/pull/22393) ("server : add slot_prompt_similarity getter/setter"). Purely additive: adds `server_context::get_slot_prompt_similarity()` / `set_slot_prompt_similarity(float)` (`tools/server/server-context.{cpp,h}`) so an embedding/JNI caller can query and tune the slot-selection threshold at runtime without reloading the model. Verbatim copy of the PR, which **upstream closed without merging** (rejected as exposing unsafe internal state — see the patch header). Carried permanently; it will not be droppable via a version bump. |
| `0007-server-attach-http-frontend.patch` | **Adds `llama_server_attach(argc, argv, server_context &, server_routes &)`** so the `NativeServer` *attach mode* can serve an **already-loaded `LlamaModel`** over the upstream HTTP frontend — no second model load, no `start_loop()`; the LlamaModel's worker keeps driving the shared `server_context` and the HTTP routes post tasks to its queue (the queue is the synchronization point). Since b11881 (upstream #31113) it adds **only** `llama_server_attach`, which parses only the HTTP-side argv via `common_params_parse`, starts the stream-session GC + `server_http_context`, registers **upstream's own** `server_register_routes()` (the route table plus the GCP-compat shim) and `server_register_stream_routes()` with the instance (non-router) handlers -- exactly what `server_run_instance()` registers, so attach mode cannot drift from upstream's endpoint set -- and **403 "disabled" stubs for `/cors-proxy` + `/tools`** (attach mode does not wire the experimental CORS-proxy / MCP / built-in-tools host — those belong to a full `llama-server`, not an embedded model), marks ready immediately (model already loaded), and blocks on the HTTP thread until `llama_server_request_shutdown()` — never calling `common_init()`, backend init, `ctx_server.terminate()` or `llama_backend_free()` (the embedding caller owns those). Applies after `0001`+`0006` (same file); closes the "NativeServer — reuse an already-loaded LlamaModel" TODO. Upstream-submittable ("server: let embedding callers attach the HTTP frontend to an existing server_context"). **Refreshed at the b10519 bump:** upstream #26347 dropped the API key from the `/models` + `/v1/models` public-endpoint set and deleted the two trailing `// public endpoint (no API key check)` comments on those route registrations. Those two lines sit inside this patch's route-table removal block, so `git apply` failed ("patch does not apply", `server.cpp:258`) at **every** tag from b10519 on; the fix was to drop the now-wrong comment from all four affected lines (2 on the `-` side, 2 in the extracted helper on the `+` side), keeping the helper byte-identical to the block it replaces. (Until b11881 that was **the invariant to re-check on every bump**: the `+` side of the patch's own `llama_server_register_common_routes()` had to stay a verbatim copy of the route table it factored out of `llama_server()`.) **Refreshed at the b11104 bump** (upstream #28690, multi-address `--host`): `server_http_context` lost its single `thread` and `listening_address` members in favour of `join()` and a `listening_addresses` vector, one listener thread per bound address. The patch still *applied* cleanly there — only its own `+` lines named the removed members — so the applier could not see it; `llama_server_attach` now logs every address and blocks in `ctx_http.join()`, exactly as upstream's `llama_server()` does. **Refreshed at the b11361 bump** (#29818 adds `POST /v1/systemone` to the route table: one line, added on both sides so the helper stays verbatim) **and at b11401** (#29895 adds a `server_child &` overload of `llama_server()` beside the declarations and a `server_child child;` at the top of the argv entry point, the context of `0007`'s first two hunks; only context moved, the `+`/`-` lines are unchanged). **Changed in 5.2.x, not a refresh: `llama_server_attach(argc, argv, server_context &, server_routes &)` serves the caller's `server_routes`** (`jllama_context::routes`, the one the model load builds) **instead of constructing one on its own frame.** `server_routes`' constructor registers a sleeping-state callback on the model's task queue, `server_queue::on_sleeping_state` only appends, so the frame-local object left a callback into freed memory behind when the attach returned, and the model's next idle sleep (`--sleep-idle-seconds`) ran it -- undefined behaviour, found at the b11361 bump and pinned by `NativeServerAttachSleepIntegrationTest` (model-backed, CI). Consequence: the route handlers read the **model's** params, so the endpoint toggles (`--metrics`, `--props`, `--slots`), `--slot-save-path`, the slot count and the sampling defaults are `ModelParameters` now (`enableMetricsEndpoint()` & co., added for that), and the attach argv configures the HTTP side only. **Refreshed at the b11799 bump:** #26603 (`POST /tts`, b11781) and #30636 (the WebSocket `/v1/realtime`, b11780, registered with `ctx_http.ws()`, no `ex_wrapper`) joined the route table between the transcription and the Anthropic routes, so the removal block and the `+` side of `llama_server_register_common_routes()` both gained the two lines and attach mode serves them too; the helper stays the verbatim copy of the block it replaces, checked by script when the patch was regenerated. **Rewritten at the b11881 bump** (#31113, "server: refactor server entrypoint"): upstream split `llama_server()` into one function per role and factored the route table out itself (`server_register_routes`, which also registers the GCP shim, and `server_register_stream_routes`), so the patch's helper, its removal block and the verbatim-copy invariant are gone; the patch now adds the declaration and `llama_server_attach()` (2 hunks instead of 4), and attach mode serves the same endpoints as before. |
| `0008-server-models-worker-cmd-override.patch` | **Makes router mode usable in-JVM.** The router (`server-models.cpp`) spawns each model worker by re-executing its own binary (`get_server_exec_path()` = `/proc/self/exe` & friends) — inside a JVM that binary is `java`, not a llama-server, so embedded router workers could never start. The patch adds env `LLAMA_SERVER_WORKER_CMD` (whitespace-split; read in `server_model_meta::update_args`) which replaces only the leading binary-path token of the rendered worker args, letting an embedding host relaunch workers through its own bootstrap — e.g. `java -cp app.jar net.ladenthin.llama.server.NativeServer` (each worker is then a fresh JVM running the classic single-model `NativeServer`). Exposed in Java as `NativeServer.setWorkerCommand(String...)` (JNI `setenv`); exercised by `RouterModeIntegrationTest` (Linux CI). Upstream-submittable (also useful for containerized/wrapped deployments). **Refreshed at the b11429 bump:** #29987 (untagged `4d60b4d08`, the commit before b11429) moved `server_model_meta::update_args` and changed the function after it (`update_caps(const common_params &)`), the hunk's trailing context; the `+` lines are unchanged. Since b11401 (#29895) a worker is a router *child* whose stdout carries the state commands, which is why `LlamaLoader` prints its two diagnostic lines to stderr (a JVM worker's early `System.out` line tripped the router until 2026-10-09, `e9758054`). |
| `0006-server-embed-native-server-jni.patch` | **Makes `server.cpp`'s `llama_server` embeddable in the JVM** so the `NativeServer` JNI bridge can run the full upstream HTTP server (WebUI included) inside `libjllama` — see "Two server modes" below. b9870 already exposes `int llama_server(int, char**)` (non-static; no `main` in the file), so the patch only adds embedded-mode support: (1) a `g_llama_server_embedded` flag + `llama_server_set_embedded()` / `llama_server_request_shutdown()` (declared in the committed `src/main/cpp/native_server_bridge.h`); (2) skips installing the process-wide SIGINT/SIGTERM handlers when embedded (they would hijack the JVM's); (3) in embedded mode parses the **forwarded** argv via `common_params_parse` instead of `common_params_parse_main` (whose `GetCommandLineW` recovery would pick up `java.exe`'s command line — the same Windows class of bug `0001` fixes). `llama_server_request_shutdown()` mirrors the SIGTERM path (invokes the installed `shutdown_handler` → `ctx_server.terminate()` unblocks `start_loop()`), giving JNI an out-of-band stop since `ctx_server` is loop-local. **The handler is guarded, and both guards are load-bearing:** upstream's `shutdown_handler` lambdas capture `llama_server()`'s locals (`ctx_http`, `models_routes`, `mcp_mgr`, `ctx_server`) by reference and are never cleared, while `native_server.cpp` signals *repeatedly* until the worker reports it finished (a stop issued before the handler is installed, or before `start_loop()` begins, would otherwise be lost). A request landing after `llama_server()` returned therefore ran the lambda over destroyed objects — a `SIGSEGV` in `server_http_context::stop()` in `RouterModeIntegrationTest.tearDown` (run 36485159455), with router mode's `clean_up()` (unloading workers) making the window wide. The patch (1) takes `g_shutdown_handler_mutex` for every read, write and invocation of the handler (the caller is on another thread) and (2) declares a `llama_server_shutdown_handler_reset` **after** the captured locals, so the handler is cleared before any of them is destroyed; because invocation holds the same lock, that reset waits for an invocation in progress. `0007`'s `llama_server_attach` carries the same guard. Runnable guard: `src/test/cpp/test_native_server_shutdown.cpp`. Applies **after `0001`** (which flips this call site to `common_params_parse_main`), so its context is the post-`0001` tree; regenerate against `0001`+source on a bump. Only touches `tools/server/server.cpp`. **Refreshed at the b11881 bump** (#31113): `llama_server()` now dispatches through `server_run()` to `server_run_router` / `server_run_instance` / `server_run_download`, and upstream added its own `server_run_opts::install_signal_handlers` switch (off for llama-cli). The embedded mode clears that switch instead of guarding the `sigaction` block, and the reset object sits in the two role functions that install a handler, declared after the last local the handler captures -- `models_routes` in the router (declared after `ctx_http` there), `ctx_http` in the instance; the forwarded-argv parse, the mutex and `llama_server_request_shutdown()` are unchanged. |
| `0014-common-log-callback-sink.patch` | **Gives `common_log` a callback sink: `common_log_set_callback(log, cb, user_data)` (`common/log.{h,cpp}`).** This is what `LlamaModel.setLogger` hooks. Before it, the Java logger was a `llama_log_set()` callback, which has two holes, both found while chasing `slot print_timing` lines interleaving with the Atmosphere agent's streamed answer: **(a)** every model load runs `common_init()`, which re-points `llama_log_set()` at `common_log_default_callback` (`common.cpp:394`), so `setLogger(…)` *before* `new LlamaModel(…)` silently lost the callback; **(b)** the server's own `SRV_*`/`SLT_*` macros are `LOG_INF` and write straight into `common_log`, which `llama_log_set()` never carried, so the per-request `slot …`/`srv …` lines could not be routed to Java at all (the reason `LlamaModelTest#testLogText/JSON` sat `@Disabled` for years). `common_log` upstream offers file, colors, prefix, timestamps, verbosity and JSONL but no hook. The patch adds one: while a callback is set, the worker thread hands every entry to it **instead of** printing to stdout/stderr (a `--log-file` still receives them); the callback gets the bare formatted message (no prefix/timestamp/colors) with the `ggml_log_callback` signature; swapping the callback pauses the worker first, so queued entries reach the *previous* sink (which is what makes `setLogger(format, null)` a synchronous drain). With the sink, `common_init()`'s `llama_log_set()` reset is harmless — it points at the default callback that feeds `common_log`, i.e. exactly the path into the sink — so the ordering problem (a) disappears without any re-install logic, and the `srv`/`slot` lines (b) arrive because they are `common_log` entries. `jllama.cpp`'s `setLogger` therefore sets `common_log_set_callback(common_log_main(), trampoline)` **plus** `llama_log_set(common_log_default_callback)` (so llama/ggml lines feed `common_log` even before the first load). Two consequences to know: the callback runs on `common_log`'s **worker thread**, a plain `std::thread` llama.cpp re-creates on every pause/resume and never attaches to the JVM — the trampoline attaches it **once per worker thread** as a daemon, named `jllama-log-worker`, and a `thread_local` guard detaches it when the thread exits (`get_jni_env_attaching`; `common_log::pause()` joins the worker, a normal exit on which TLS destructors run, and `JNI_OnUnload` pauses it before anything else). Until 5.2.x it attached and detached per line, which created a `java.lang.Thread` object per log line (13 lines of a failed load, 13 `Thread` objects -- `LlamaLoggerTest` pins the single attach now). Each call runs inside a `PushLocalFrame`/`PopLocalFrame` pair, because a thread that stays attached and has no Java frame would otherwise accumulate local references, `setLogger` must call `common_log_set_callback` **outside** `g_log_mutex`, because the pause joins the worker, which needs that mutex to read the callback, and `setLogger` callers are serialized by a **separate** `g_set_logger_mutex`: two unserialized swaps race on the worker's `std::thread` (one joins it while the other assigns a fresh thread over the still-joinable object = `std::terminate`, the whole JVM), which `LlamaLoggerTest#concurrentSetLoggerCallsDoNotRaceOnTheLogWorker` reproduced before the mutex existed. Two caveats the Javadoc carries: a caller must not hold a lock the *previous* callback needs (the drain runs it on the worker while the caller waits), and the verbosity threshold is process-wide and reset by **every** load (`common_params_parse` ends with `common_log_set_verbosity_thold(params.verbosity)`, default 3), so a load without `-lv` puts it back to 3 — a review assumed the opposite, and `LlamaLoggerTest#verbosityThresholdIsProcessWideAndEveryLoadSetsIt` now pins the measured behaviour. And the verbosity threshold applies *before* the sink: at the default (`3`) the Java logger sees errors, warnings and the server's INFO lines, while llama/ggml INFO lines (`common_log_get_verbosity` maps them to TRACE = 4) arrive only from `setLogVerbosity(4)` on — the same filtering the console gets, and a behaviour change for consumers who captured the unfiltered `llama_log_set()` stream before. **Runnable guards:** `src/test/cpp/test_common_log_callback.cpp` (6 tests over a private `common_log_init()` instance: delivery, bare text under prefix+timestamps, clear, swap-drains-to-old-sink, file kept, levels pass through) links the function on every platform, so a bump that drops the patch reds `C++ Tests` at link time; `LlamaLoggerTest` (model-free, needs only `libjllama`: a logger set before a deliberately failing load on a non-GGUF file sees the `srv … loading model` INFO line and llama's ERROR line, in TEXT and JSON) and the re-enabled `LlamaModelTest#testLogText/testLogJSON` plus `#testLoggerSetBeforeLoadSurvivesTheLoad` (vocab-only load) cover the Java side. Upstream-submittable ("common : add a callback sink to common_log for embedding hosts"); **not yet filed upstream**. Touches only `common/log.{h,cpp}`, which no other patch touches. **Refreshed at the b11401 bump** (#29895 adds a `colors` member next to the sink's members and a `get_colors()` next to `set_callback()`; only context moved, the `+`/`-` lines are unchanged, and upstream still has no hook -- it added `common_log_get_colors` only). **Refreshed at the b11612 bump** (`common_log_set_stdo()`, a stdout/stderr on-off switch stored as a `stdo` member right where the sink's members sit, and `print_entry()` now prints to the console only `if (stdo)`): three hunks of `common/log.cpp` lost their context and were regenerated; the sink keeps precedence -- `if (callback) { callback(...) } else if (stdo) { e.print(); }` -- so a sink still replaces the console and `set_stdo(false)` without a sink still silences it. Still no hook upstream (`common/log.h` names only `common_log_default_callback`). **On a bump, check whether upstream added a hook of its own (grep `callback` in `common/log.h`) and, if so, DROP this patch and port `setLogger` to theirs rather than refreshing it.** |
| `0015-rpc-embeddable-client-and-stoppable-server.patch` | **Makes ggml-rpc embeddable.** Registration/HELLO failures report an ordinary error instead of aborting; cached registrations are rechecked; memory queries of a gone server return 0/0. Post-registration device loss remains fatal upstream. **Stoppable server:** the main listener and COMM_INIT peer listener poll cancellation every 50 ms; outgoing peer connects/retries also observe stop, and the main client plus all live peer sockets are shut down under one mutex before blocked caps I/O can survive stop (on Windows `shutdown()` also calls `CancelIoEx`: shutdown alone does not wake a `recv()` blocked in another thread there, which the first run of the Windows C++ suites found, 40195444689). Peers are tracked weakly so COMM_FREE still releases them immediately. No wake-up TCP connection can fill the main accept backlog. **COMM_INIT keeps the configured bind address:** a loopback server cannot open a wildcard peer listener. TCP errors close descriptors and do not raise SIGPIPE; stop leaves shared Winsock state alive. **Runnable guards:** `src/test/cpp/test_rpc.cpp` (registration, math, stop) and `src/test/cpp/test_rpc_comm.cpp` (protocol-v8 C-API subprocesses: loopback/network binding, absent peer, both stalled caps handshakes, successful setup, port cleanup and restart). No model/GPU/external network; deadlines make regressions fail instead of hang. Touches `common/arg.cpp`, `ggml/include/ggml-rpc.h` and `ggml/src/ggml-rpc/{ggml-rpc.cpp,transport.cpp,transport.h}`. **Drop-check on every bump:** inspect registration, COMM_INIT binding, cancellable socket waits and peer tracking; remove hunks only after upstream implements the same contracts. Upstream-submittable; nothing filed before the release. |
| `0017-ggml-cpu-x86-prefetch-byte-offset-and-msvc-type.patch` | **Makes `GGML_CPU_ALL_VARIANTS=ON` buildable with plain clang on Windows, and fixes a prefetch address that is wrong everywhere.** The four `_mm_prefetch` calls in `ggml/src/ggml-cpu/arch/x86/quants.c` (the SSSE3-without-AVX branch of `ggml_vec_dot_q4_0_q8_0`, and the only `_mm_prefetch` calls in the whole ggml tree) carry two independent defects; both are fixed by routing the address through one new `static inline` helper in the new header `ggml/src/ggml-cpu/arch/x86/prefetch.h`, which is included relatively, so **no CMake change is needed**. **(1) Type -- fatal only on a current clang:** MSVC declares the intrinsic as `void _mm_prefetch(char const *, int)`; clang's own `xmmintrin.h` defines a casting macro but only `#ifndef _MSC_VER`, so on a `*-windows-msvc` target clang leaves the declaration to Microsoft and a typed pointer is `-Wincompatible-pointer-types`. **clang 22 promoted that to an error by default** ([llvm-project #157364](https://github.com/llvm/llvm-project/pull/157364); the [22.1.0 release notes](https://releases.llvm.org/22.1.0/tools/clang/docs/ReleaseNotes.html) add that `-Wno-error=incompatible-pointer-types` downgrades it) -- *not* clang 16, which promoted only `-Wincompatible-FUNCTION-pointer-types`; an earlier version of this row said otherwise. Measured on the real file at `-msse4.2`, unpatched → patched: clang **20.1.8** 4 warnings → 0/0, clang **22.1.3** 4 errors → 0/0, clang **23.1.3** 4 errors → 0/0. Plain clang is exactly what `cmake/x64-windows-llvm.cmake` selects -- and plain clang is the only way to get all **14** CPU variants, because `clang-cl` makes CMake's `MSVC` true and the `if (NOT MSVC)` block in `ggml/src/CMakeLists.txt` then drops `zen4`, `cooperlake`, `sapphirerapids`, `ivybridge` and `piledriver`. **(2) Arithmetic:** `&x[ib] + sizeof(block_q4_0)` adds `sizeof()` *elements*, not bytes -- +324 and +1156 instead of +18 and +34 (the sizes `ggml-common.h` pins by `static_assert`), and +648 instead of +36 for the `2 *` pair. That half is wrong on every platform, including where the call compiles. **Measured:** no throughput difference (see the test-table row), so this is a correctness and portability fix, not a performance one. **Upstream-submittable and meant to be submitted.** **Why upstream's own build is green is measured, not an open question:** their `windows-cpu` release job builds exactly this with plain clang, `GGML_CPU_ALL_VARIANTS=ON` and no suppressing flag (`CMAKE_ARGS` carries only build-target toggles), and ships `ggml-cpu-sse42.dll` -- because the `windows-2025-vs2026` image has **LLVM 20.1.8** on `PATH`, where these are warnings. That is narrower than it looks: the same image also ships **clang 22.1.3** inside Visual Studio 2026 (`VC/Tools/Llvm/x64`), which *fails* on the unpatched file -- upstream reaches the working compiler only because `C:\Program Files/LLVM/bin` comes first on `PATH`. It is **not** a header difference: all three MSVC toolsets on that image (14.29.30133, 14.44.35207, 14.51.36231) declare it identically as `extern void _mm_prefetch(char const*_A, int _Sel);` -- that was the first hypothesis, and a one-off probe workflow (deleted again once it had answered) disproved it by printing the declaration from each toolset on the image. So this patch is **ahead of** upstream rather than diverging: their build breaks the moment the image ships a clang that makes the diagnostic fatal, and until then they compile four warnings and a prefetch address 18x/34x too far. **Runnable guard: `src/test/cpp/test_prefetch.cpp`.** On a bump, run `git grep -n _mm_prefetch ggml/src/ggml-cpu/arch/x86/quants.c` at the new tag -- a **locator, not a verdict**: it returns **four** hits at every tag, because the calls exist either way (b11639: lines 846, 847, 864, 865). Read them; the fix is upstream's only when the cast to a byte pointer happens **before** the addition, since `(const char *)(&x[ib] + sizeof(...))` casts the whole expression, silences the compiler and keeps the wrong address. Only then **drop this patch rather than refreshing it**. |
| `0018-ggml-amx-free-the-aligned-buffer.patch` | **The AMX weight buffer is freed with `free()` but allocated with `_aligned_malloc()` on Windows -- `STATUS_HEAP_CORRUPTION` (0xC0000374) the moment a model is unloaded.** `ggml_backend_amx_buffer_type_alloc_buffer()` (`ggml/src/ggml-cpu/amx/amx.cpp`) allocates with `ggml_aligned_malloc()`, which is `_aligned_malloc()` under `_MSC_VER` (plain clang on a `*-windows-msvc` target defines it too); `ggml_backend_amx_buffer_free_buffer()` called `free()` on it. Fine on Linux (`posix_memalign` + `free`), fatal on Windows, and only there. The patch frees with `ggml_aligned_free(buffer->context, buffer->size)`, exactly what `ggml_backend_cpu_buffer_free_buffer()` does; on Linux that is `free()`, so nothing changes there. **Reached by** the `sapphirerapids` CPU module (the only variant built with `AMX_TILE AMX_INT8`, which `ggml_backend_load_best` picks on Sapphire/Emerald/Granite Rapids; `ggml_amx_init()` is `return true` on `_WIN32`) and a model whose weights have AMX kernels (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS -- i.e. every CI model). **Found by the second full run of the modular natives** (38069595101): `Java Tests Windows 2025 x86_64 (clang)` landed on an `Intel(R) Xeon(R) 6973P-C` runner (Granite Rapids), loaded `ggml-cpu-sapphirerapids.dll`, ran `GrammarExample`'s two generations fine and died with exit code -1073740940 at `LlamaModel.close()`; the first run (38065017595) had the same job green on an AMD EPYC 7763 (no AVX-512, `haswell`). **The Windows runners are a CPU lottery**, so which CPU module a Windows job exercises differs per run; before the modular natives the Windows jar was a static Haswell-level build and this code never ran. **Of the six callers of `ggml_aligned_malloc` in the tree this was the only mismatched pair** (`ggml-backend.cpp`, `ggml-cpu.c` twice, `ggml-openvino.cpp`, `ggml-zdnn.cpp` and `ggml.c` all pair correctly), and `ggml-backend.cpp:2464` is the line the fix copies. **Reproduced deliberately rather than waiting for the lottery again**: on a rented Granite Rapids machine, one CPU module staged at a time with the same model and the DLLs CI itself built, `sapphirerapids` crashed **6 of 6** runs -- including at **one** thread, which rules out a data race -- while `cooperlake`, `icelake` and `haswell` were clean 6 of 6, and on Linux with the same CPU all six exit 0 with byte-identical output. The crash lands *after* the inference prints its timings, in cleanup: the signature of a heap corrupted earlier and detected at `free`. With the patch applied the same matrix on the same machine is **clean in all twelve runs**. **The guard is a configure-time check, not a `jllama_test` case**: the AMX code exists only in that one variant of a `GGML_CPU_ALL_VARIANTS` build and executes only on AMX hardware, which the static test build, the local agent's Zen 3 and the cloud sandboxes lack -- so **`llama/cmake/check-aligned-free-pairing.cmake`** fails the configure on **every** platform when a ggml source calls `ggml_aligned_malloc` and never `ggml_aligned_free` (verified red against the unpatched source, naming `amx.cpp`, and green with this patch). A run-time test cannot replace it: without an AMX CPU there is nothing to observe, which is exactly how this shipped. What exercises it: the Windows x86-64 Java test job and the `windows-x86-64` smoke row whenever GitHub hands them an Intel AMX runner; the patch itself is held by `check-patches.py`, applies at b11639, and the patched file compiles with the variant's flags (GCC 13, `-mamx-tile -mamx-int8 -mavx512vnni ...`). Upstream master still carried the `free()` on 2026-10-10; **not yet filed upstream** (nothing before the release). **Drop-check on every bump, note the inversion:** `grep -n "free(buffer->context)" ggml/src/ggml-cpu/amx/amx.cpp` at the new tag -- a **hit** means the bare `free()` is still there and the patch is still needed; **gone** means upstream fixed it, drop the patch. Measured at b11639 (`a0e6dc738`): one hit, line 46. |
| `0020-ggml-alloc-validate-leafs-before-reusing-an-allocation.patch` | **An `--embedding` model that is asked to generate crashes the process in `ggml_gallocr_alloc_graph`** (SIGSEGV on Linux, 0xC0000005 with both Windows toolchains), since upstream #30445 (`cb55ffe6e`, "graph: skip the LM head for pooled embeddings", b11574..b11612). `ggml_gallocr_alloc_graph()` assigns every tensor by **position** from the graph it was reserved with, and `ggml_gallocr_needs_realloc()` validates every node and every node's sources but of the **leafs only the count**. Since #30445 the LM head is expanded by `build_logits()` *after* `build_pooling()`, so an embeddings context reserves a graph that expands `t_embd` first and `t_logits` second, while a generation step on it (the server calls `llama_set_embeddings(ctx, false)` for a slot that samples) expands only `t_logits`, whose first source is `output.weight`: same node and leaf counts, every leaf after it shifted by one. An input leaf then lands on the slot of a weight (`buffer_id -1`) and, with the asserts compiled out, `ggml_vbuffer_tensor_alloc()` indexes `galloc->buffers[-1]`; a leaf landing on another input's slot is not size-checked either. The patch validates the leafs exactly as the nodes are validated, so an invalid leaf makes the allocator reserve again (one buffer) or return false, on which `ggml_backend_sched` reserves the graph itself. **Found by** Publish run 38648257387 (2026-10-14, the first full run since b11558): `ErrorHandlingTest` loads its model with `enableEmbedding()` and posts an infill request to it -- its first test in JUnit's order -- and the fork died in all three Java test jobs (`ggml_gallocr_alloc_graph <- ggml_backend_sched_alloc_graph <- llama_context::process_ubatch <- decode <- server_context_impl::update_slots`). **Reproduced locally without a downloaded model** (a random 2-layer llama GGUF with FIM tokens, context 128): infill crashes with `--embedding` and works without; a debug print showed leaf 31 `out_ids` arriving at a slot with no allocation; reverting `cb55ffe6e` makes it pass, and so does this patch -- with greedy sampling the `--embedding` model then returns the same tokens as the model without, run after run. Upstream master (`a0e6dc738`) still lacks the check; **not yet filed upstream**. **Runnable guard: `src/test/cpp/test_gallocr_leafs.cpp`** (two graphs with the same counts and the leafs swapped). **Drop-check on every bump:** read `ggml_gallocr_needs_realloc()` in `ggml/src/ggml-alloc.c` at the new tag -- once it walks `graph->leafs` itself, drop this patch rather than refreshing it. |
| `0021-server-clear-the-slot-cache-when-the-server-adapters-change.patch` | **A slot's prompt cache survived a change of the LoRA adapters it was computed with.** `launch_slot_with_task()` (`tools/server/server-context.cpp`) clears the cache when a request names its own adapters and they differ from the slot's (`lora_should_clear_cache`, which keeps it only for aLoRA), but a request that names none takes `params_base.lora_adapters` in the `else` branch without that check. So the cache survived `SET_LORA` (`LlamaModel.setLoraAdapter(s)`, upstream's `POST /lora-adapters`) and the return to the server's adapters after a request with its own, and the slot answered like neither adapter set, silently. **Measured** (J9 of the bug-hunt brief, b11639, upstream's own LoRA test pair `stories15M_MOE-F16.gguf` + `moe_shakespeare15M.gguf`, greedy, one slot): after `setLoraAdapter(0, 0f)` the warm slot answered ". He saw a picture of a bird ..." where an emptied slot with the same scale answers " was a little girl ..."; only `eraseSlot(0)` restored it (F67). The patch applies the per-request rule in the `else` branch too; equal sets keep the cache as before. **A second hunk starts every slot with the server's adapters in `init()`** (`slot.lora = params_base.lora_adapters;`): `slot.lora` used to start empty, so a never-used slot of a server started with `--lora` read as "computed under no adapters", and its first task logged `0 loras -> 1 loras` and cleared a prompt it already held -- a state restored into it (`SLOT_RESTORE`) or loaded from the host-memory prompt cache by `get_available_slot()`, both computed under the server's adapters (F240). Upstream has the same defect on its HTTP route, so the patch is upstream-shaped; **not yet filed upstream**. Not covered, and said in `setLoraAdapters`' Javadoc: the host-memory prompt cache (`--cache-ram`, on by default) stores slot states without their adapters. **Runnable guard: `src/test/cpp/test_lora_cache.cpp`** (a random llama model and a random LoRA adapter on K, V and the FFN output, written by the test; five cases). **Drop-check on every bump, one per hunk:** read the `else` branch of the "process per-request lora adapters" block in `launch_slot_with_task()` at the new tag -- once it compares `params_base.lora_adapters` with `slot.lora` and clears the prompt (or `SET_LORA` clears the slots), drop that hunk rather than refreshing it; and read the slot loop of `init()` -- once it sets `slot.lora` from `params_base.lora_adapters`, drop that hunk. The patch goes when both have. |
| `0022-ggml-cpu-repack-batch-the-tokens-of-an-expert.patch` | **The expert matmuls of a MoE model over repacked CPU weights ran one token at a time on a static thread split.** `forward_mul_mat_id()` (`ggml/src/ggml-cpu/repack.cpp`) -- the path of every MoE model whose experts are repacked at load: MXFP4 (gpt-oss), Q4_0, IQ4_NL and Q4_K on AVX2, their interleaved forms on ARM -- multiplied each token routed to an expert with its own gemv, although the gemm kernel for the type exists, and gave thread `ith` a fixed row range of every expert, so each expert matmul waited for the slowest thread. Measured with a random-weight gpt-oss-20b in ggml-org's MXFP4 layout on the `alderlake` module (the Core Ultra's), 4 threads: 72 % of prompt time was that gemv loop, 13.9 % of generation time the barrier wait after it ([`docs/benchmarks/moe-mxfp4-cpu-alderlake.md`](docs/benchmarks/moe-mxfp4-cpu-alderlake.md)). The patch adds a batched path -- every group of four tokens routed to an expert gathered and quantized interleaved (`ggml_quantize_mat_t`, as `forward_mul_mat` does) **once**, into a buffer shared by all threads, then (expert, column chunk) units handed out through `ggml_threadpool_chunk_add`, each multiplying up to 16 shared rows at a time with `gemm`, a rest of fewer than four by gemv -- and **keeps upstream's path behind a process-wide switch**: `ggml_backend_cpu_set/get_repack_mmid_batched` (`ggml-cpu.h`, and in the CPU backend's `get_proc_address` table, which is how libjllama reaches it in a `GGML_BACKEND_DL` build), on by default, `GGML_CPU_REPACK_MMID_BATCHED=0` in the environment starts it off. Thread 0 reads it once per op and publishes the choice in the work buffer, so a flip during a compute cannot split the threads; `work_size()` reserves the batched path's shared rows (one per routed token slot, 6.3 MB for gpt-oss at a 512-token batch) and per-thread scratch either way. **Quantizing once is load-bearing on ARM:** the first version quantized an expert's rows again in every column chunk, and with no SIMD Q8_K quantizer on ARM that made Q4_K prompt processing on a Neoverse-N1 18 % *slower* than upstream at 32 threads -- found with real weights ([`docs/benchmarks/moe-repack-real-weights.md`](docs/benchmarks/moe-repack-real-weights.md)), where the final patch gains +13 % to +66 % in prompt processing on Zen 3, the N1 and a z16 (Q4_0 only, the one type s390x repacks) for MXFP4, Q4_0 and Q4_K, generation unchanged; KLD 0 for MXFP4/Q4_0, 0.0075 for Q4_K, where gemm and gemv round differently and upstream's own repack path is as far from `--no-repack`. Exposed as `LlamaModel.setCpuMoeBatched` / `isCpuMoeBatched`, `jllama_set/get_cpu_moe_batched` and the FFM `Jllama`: the switch exists so a user can measure both paths on their own hardware -- the P-core/E-core effect the static split suggests is unmeasured (TODO.md). **Measured, one binary, the environment variable flipped:** prompt processing +47 % (42.63 -> 62.76 t/s), token generation +26 % (7.90 -> 9.97), logits equal to rounding (`llama-perplexity --kl-divergence`: mean KLD 0.000000, max 0.000002, same top token 100 %). **Runnable guard: `src/test/cpp/test_moe_repack.cpp`** (both paths against ggml's generic `MUL_MAT_ID` over the same weights; the switch resolved through `get_proc_address`). Upstream-shaped without the switch; **not yet filed upstream**. **Drop-check on every bump:** read `forward_mul_mat_id()` in `ggml/src/ggml-cpu/repack.cpp` at the new tag -- once it calls `gemm` for an expert's batched rows and hands its work out through `ggml_threadpool_chunk_add`, drop this patch rather than refreshing it; the switch goes with it, and `setCpuMoeBatched` then throws (`LlamaException`, no entry point), so the Java, C and FFM API needs the decision whether to remove it or keep it as a no-op. |
| `0023-ggml-cpu-s390x-variants-build-with-clang-and-a-z13-baseline.patch` | **The IBM Z CPU variants (`GGML_CPU_ALL_VARIANTS` on s390x) did not build with clang, and had no module below z15.** (1) `ggml-cpu-impl.h` and `llamafile/sgemm.h` both include `<vecintrin.h>` under `__VXE__ \|\| __VXE2__`; clang's header has no include guard (GCC's has `_VECINTRIN_H`), so `ggml-cpu.c`, which includes both, failed with "redefinition of 'vec_extract'". The include in `sgemm.h` came with ggml-org/llama.cpp#14273 and is not needed: the header declares no vector type, and both of its includers include `ggml-cpu-impl.h` -- `sgemm.cpp` after it, where `__VXE__` is not defined yet (`ggml-cpu-impl.h` defines it from `__VEC__`), so it never fired there. The patch removes it. (2) Upstream's list starts at `z15`, so a z13/z14 would have found no CPU backend at all; the patch adds `ggml-cpu-z13` (scalar, `-march=z13`, score 1), so the jar keeps the z13 floor the static library had. Measured on cfarm191 (z16, b11901, Qwen3-0.6B Q4_K_M big-endian, 4 threads): `z13` 19.7 / 13.6 t/s, `z15` 90.0-95.5 / 57.3-58.3, `z16` 82.6-88.3 / 51.9-53.9 (prompt / generation); Q4_K_M against a Q8_0 reference on z13: mean KLD 0.0777 on `z13`, 0.0746 on `z16`. **Guard:** `build-linux-s390x` builds the variants with clang 18 -- without part (1) the build fails -- and checks that `libggml-cpu-z13/z15/z16.so` all exist, which is what fails without part (2). Not yet filed upstream. **Drop-check on every bump:** `grep -n vecintrin ggml/src/ggml-cpu/llamafile/sgemm.h` (a hit: part 1 still needed) and the s390x block of `ggml_add_cpu_backend_variant` calls in `ggml/src/CMakeLists.txt` (no `z13`/scalar entry: part 2 still needed). Measured at b11901: both still needed. |

**Dropped patches** (`0004`, `0005`, `0009`, `0010`, `0011`, `0012`, `0013`, `0016`, `0019` -- upstream fixed
each defect itself, or for `0016` added the model; `0019` is this project's own fix, merged upstream as #30505) are recorded in [`docs/history/dropped-llama-patches.md`](docs/history/dropped-llama-patches.md):
why each went, and the runnable guards that stayed behind re-pointed at upstream's contract. Read the
matching note before reintroducing a patch for the same defect. **The standing drop-check on every
bump:** the applier detects "does not apply", never "upstream already fixed this" -- for each
remaining patch ask whether the *defect* is still there, not only whether the patch still applies.

**And read each drop-check's output rather than its exit code — both remaining ones (`0017`, `0018`) return
hits at every tag.** Each row above now says what its own output means, because the generic form of these
greps is not a true/false and reducing one to a boolean has the worst possible failure mode: it
deletes a fix that is still needed, silently, in the same commit that bumps the version. Measured at
b11639, while `0019` was still carried: its loose grep returned **3** hits (none in the branch it was about, while
`git apply --check` of the patch was clean — the two answers contradicted each other and the grep was the one that
was wrong; the sed over the `__ARM_NEON` branch alone printed nothing until b11760 and one line at b11761, which is
how that patch went, see `docs/history/dropped-llama-patches.md`), `0017`'s returns **4** at every tag because it
locates calls that exist either way, `0018`'s is **inverted** (a hit means the defect stands); `0016`'s was the
one clean true/false, and that patch went at the b11760 bump the way its row said -- upstream registered the
architecture at b11709, the applier fired on the file it would have created.
The same class cost a wrong answer once before, when a drop-check was run against a tag that did not
exist yet: `git show` returned an empty file, the grep found nothing, and "upstream has not fixed it"
was read out of a file that was never read. **A drop-check that cannot distinguish "no match" from
"nothing examined" has not been run.**

## Source archive: a release stays rebuildable without upstream

Every natives jar is compiled from the pinned upstream commit plus `llama/patches/`, a tree that
used to exist only transiently in each build job's `_deps/`. Two jobs of `publish.yml` keep it
(TODO.md, delivery roadmap, item 6):

- **`archive-llama-source`** (`.github/archive-llama-source.sh`): a full checkout of
  `ggml-org/llama.cpp` at the pinned commit (`JLLAMA_GIT_COMMIT`, else the tag; the commit count is
  checked against the number as the configure does), the applier run exactly as every build runs it
  (`cmake -P llama/cmake/apply-llama-patches.cmake`), `verify-patches-applied.sh` over the result,
  `.git` dropped, and the tree packed **reproducibly** (sorted, the commit's time as every mtime, no
  owner, `gzip -n`; measured: two packs of the same inputs hash the same) as
  `llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz`, artifact `llama-cpp-source`. The tree keeps the
  applier's stamp `.jllama-patches-applied` (the commit it was patched at + every patch's hash).
- **`test-source-archive`** (gates both publish jobs): unpacks the archive and builds the static
  library with the tests from it through `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<dir>`, then `ctest`.
  **FetchContent skips population entirely for an overridden source directory -- no download and no
  `PATCH_COMMAND`** (measured in `FetchContent.cmake`: "no population is required"), so the applier
  never runs there; the stamp is what proves the patches are in, and the job runs
  `verify-patches-applied.sh <dir>` against it before building. The applier itself also understands
  such a tree now: a non-git tree whose stamp names exactly the patches on disk with their hashes is
  "already applied", another patch set is a loud error (before, the legacy per-patch reverse check
  aborted on `0001`, whose region `0006`/`0007` rewrite). The two build-info writers read the commit
  from the stamp when `.git` is absent (`jllama_archived_llama_commit` in `llama/CMakeLists.txt`), so
  `llama_build_info()` and `jllama-build.txt` of an archive build name `b<nnnn>-<commit>` rather than
  `unknown`.
- The publish jobs sign the archive with the release key (`gpg --detach-sign --armor`, the key
  `setup-java` imported, the same passphrase name as maven-gpg) and collect it next to the Maven
  artifacts: a GitHub-only file of the signed publication, so the gate verifies its signature and
  `attest-*` covers it (see "Signed publication").
  A consumer rebuilds a release with `cmake -B build -DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<unpacked>`.

## The C API: `jllama_api.h` over the engine (layer B)

[`docs/c-api.md`](docs/c-api.md) is the consumer-facing page; this is the layout. Three files
carry the one implementation behind both call paths into `libjllama`:

- **`src/main/cpp/jllama_context.hpp`** -- `jllama_context` (the handle both paths hold: the Java
  field's jlong, the C API's opaque `jllama_model *`), `erase_reader`, the user-reference release
  and `jllama_context_guard`, and `acquire_jllama_context(jllama_context *)` for a handle already
  known to be live. No JNI in it; `jni_helpers.hpp` includes it and adds the Java-field half
  (`acquire_jllama_context_impl`, under `g_ctx_mutex`).
- **`src/main/cpp/jllama_engine.{hpp,cpp}`** (`namespace jllama::engine`) -- the C++ over
  `server_context`: `load` (argv -> params, `--vocab-only`, the `--rpc` guard, `common_init`, the
  worker thread -- a plain thread now, nothing on it calls into a host), `free` (closing flag, the
  readers cleared, wait for `users == 0`, terminate, join, delete), `wake_server` /
  `wake_and_get_meta`, `parse_oai_chat_params`, `populate_completion_task`, `wake_and_post`,
  `blocking_completion`, `streaming_request`, `stream_next`, `task_release`, the three chat
  operations over JSON text, `set_log_sink` (the swap mutex lives here). Errors are exceptions:
  `jllama_invalid_request` (400), `jllama_interrupted` (a wait cut by `free`), `std::runtime_error`.
- **`src/main/cpp/jllama_api.{h,cpp}`** -- the `extern "C"` shell: `jllama_status` codes, a
  thread-local `jllama_last_error()`, malloc'd strings (`jllama_string_free`), every call inside the
  user-reference guard, `JLLAMA_API` export macro (`JLLAMA_BUILD` -> dllexport; the ELF/Mach-O
  export lists already let `jllama_*` through), `jllama_api_version()` =
  `JLLAMA_VERSION_STRING`, the pom version `cmake/jllama-build-info.cmake` resolves
  (`JLLAMA_VERSION_RESOLVED`).

**`jllama.cpp` is the shell for Java.** Every `LlamaModel` and `LlamaQuantizer` entry point calls
the C API and maps a status with `throw_api_status` (`INVALID_REQUEST` -> `InvalidRequestException`,
else `LlamaException`); the out-strings cross through `api_string`, the task ids through
`api_task`. The file holds no server logic any more -- the old dispatch helpers, the batch and
result checks, the indexed token tasks and `post_and_wait` live in the engine. The same holds for
the other JNI files: `native_server.cpp` (`NativeServer`) is a shell over `jllama_server_*`, whose
engine is `native_server_engine.{hpp,cpp}` (compiled on every platform but Android, with
`JLLAMA_HAS_NATIVE_SERVER` guarding the C functions in `jllama_api.cpp`); `rpc_bridge.cpp`
(`RpcServer`) over `jllama_rpc_server_*` (rpc_support.hpp); the `TextToSpeech` entry points in
`jllama.cpp` over `jllama_tts_*` (`tts_engine.{h,cpp}`); `train_engine.cpp`'s one entry point over
`jllama_finetune` (`finetune_json`, the parse it used to do itself). `REQUIRE_SERVER_CONTEXT` stays
on every `LlamaModel` entry point: it reads the handle from the Java field under `g_ctx_mutex`,
which is what makes `close()` safe against a racing call; the C API acquires a second, nested user
reference on the raw pointer (the attach server holds one for its worker's lifetime). **A new native method of
`LlamaModel` is written as an engine function plus a `jllama_*` function plus a JNI shell, in that
order** -- the Java side never gets logic the C side lacks again.

**The header ships in every library directory**, named in `jllama-files.txt` so the loader extracts
it next to the library and never loads it (the static platforms get a one-line `jllama-files.txt`
for exactly that; without the list the merge step derives every unlisted file into
`jllama-extras.txt`, which `LlamaLoader` `System.load`s). **On Windows the import library
`jllama.lib` ships the same way** (`ARCHIVE_OUTPUT_DIRECTORY` of `jllama` is the natives directory
and the list names `$<TARGET_LINKER_FILE_NAME:jllama>`), so a C consumer links it instead of
resolving the entry points with `GetProcAddress`. `verify-native-deps.py` looks at library
suffixes only, and the merge step's "nothing else" rule applies to module directories only.

**Tests.** `src/test/cpp/test_c_api.cpp` drives the C API compiled into `jllama_test`
(`jllama_engine.cpp` + `jllama_api.cpp` are in its source list, with the same two definitions) over
a tiny random llama-architecture GGUF it writes itself -- hermetic, so it runs in every `C++ Tests`
job without `fetch-models`; the output of a random model is noise, the contract is what is
asserted. `test_c_abi.c` (the modular jobs' `c_abi`) links the shipped `libjllama` and resolves
`jllama_api_version` / `jllama_build_info` through its exports, holding the build info's number to the
`llama.cpp=` line of the directory's `jllama-build.txt` (`llama_build_info()` itself is llama-common,
static and hidden inside `libjllama`, so a plain-C consumer cannot compare against it). And every model-backed Java test of
the migrated entry points runs through the C API.

## RPC backend: `--rpc` client and the in-JVM `RpcServer`

llama.cpp's RPC backend (`ggml-rpc`) is compiled into **every library jar** with `GGML_RPC=ON` forced
in `llama/CMakeLists.txt` (a GPU module jar has `GGML_RPC OFF`: the module is in the library jar it
joins). On the static platforms it stays inside the **one** `jllama` library:
`ggml_add_backend_library` makes `ggml-rpc` a static library linked into `ggml`, and its
registration is compiled in (`GGML_USE_RPC`); on the modular platforms (`GGML_BACKEND_DL`, see
"Modular natives") `libggml-rpc.so` / `ggml-rpc.dll` is a module next to the library that
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
commands — and holds every library directory (`cpu`, `metal`) and the Android OpenCL module to an
exact per-directory allowlist (the other GPU directories only to a denylist, since they need their
vendor runtime). A new CPU
directory without an allowlist fails too. (That every build arrived is the merge step's check against
`natives.csv`.) **It found a pre-existing defect on its first run**: the macOS dylib linked Homebrew's
`openssl@3` (`/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib` + `libcrypto`), so it did not load on a
Mac without that formula. Fixed in 6.0.0 by linking BoringSSL statically (see "HTTPS: BoringSSL,
statically" below); the allowlist no longer carries those paths, so their return reds `package`.

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
`ggml_backend_rpc_stop_server()` (patch `0015`) through `jllama_rpc_server_stop()` -- which keeps a
stop issued during the serve call but before ggml binds the socket (ggml clears its own flag there)
and returns once the serve has, so the FFM and C callers get what `RpcServer`'s Java retry gave the
JNI path alone (F241); every serve call has an id, a stop is for the serves in progress when it is
made and waits for those only, so a serve started while the stop still waits (a restart from
another thread) is neither told to stop nor reached by the repeated ggml stop, which is issued under
the registration mutex (F271) -- and `start` waits for
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
`System.load`, clearing the temp files and, with every GPU jar on the classpath, extracting every
backend again over the library being loaded. The RPC smoke timed out on it (the doubled extraction
of the CUDA/ROCm/SYCL libraries, then still complete libraries); it now waits 300 s like the other
smoke and fails if the backend is selected more than once. Since 6.0.0 `initialize()` is also idempotent
after its first success (a changed `lib.path` or `backend` is named once on stderr and takes effect only
in a new JVM; F264); `BackendLoadTest` drives repeated loads through the package-private
`LlamaLoader.resetForTesting()`.
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
load). CI: `.github/smoke-rpc.sh` in the `linux-x86-64` row of `smoke-natives` runs **two JVMs from the
published jars** (the smoke set) — `RpcServer` in one, the default `NativeServer` with `--rpc` in the other — and requires a
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

## HTTPS: BoringSSL, statically

Every desktop natives jar can fetch a model from an `https://` URL (`--model-url`, `-hf`, the
`ModelParameters` setters behind them) and serve the embedded server behind its own TLS certificate
(`--ssl-key-file` / `--ssl-cert-file`). The SSL library is **BoringSSL, built from source and linked
statically** -- `LLAMA_BUILD_BORINGSSL ON` in `llama/CMakeLists.txt` (the "HTTPS" block), the flag
upstream's own release builds pass; upstream's `vendor/cpp-httplib/CMakeLists.txt` fetches the pinned
tag and links `ssl`/`crypto` into its `cpp-httplib` target. **Android, s390x and the Linux SYCL jars
are the exceptions** (`LLAMA_OPENSSL=OFF`, so `find_package` cannot link a system OpenSSL by
accident): the dockcross cross-clang is not a configuration BoringSSL is built for here, big-endian
is outside its supported set, and `icx`/`icpx` (compiler id `IntelLLVM`) is neither GCC nor Clang to
BoringSSL's CMake -- upstream's `ubuntu-24-sycl` release job builds against the system OpenSSL instead since
b11676 (#30456; `libssl-dev`, a dynamic `libssl` a shipped library here must not import), while its
`windows-sycl` job (C by `cl`, C++ by `icx`) builds BoringSSL. The SYCL jars are module-only builds -- ggml
alone, no cpp-httplib -- so no SSL code reaches them either way; the exception matters for a full Linux SYCL
build of `libjllama` only. `verify-native-deps.py` is what holds the line: a `libssl` of any kind in a shipped
library's dependency list fails `package`. The one import this adds is Windows' `crypt32.dll`
(cpp-httplib's certificate-store lookup), an OS component since Windows 2000, in `WINDOWS_OS`.

**The defect this replaced was a link-order lottery, not a missing flag.** Until 6.0.0 `jllama`
compiled `vendor/cpp-httplib/httplib.cpp` itself, deliberately without `CPPHTTPLIB_OPENSSL_SUPPORT`
(the embedded server was "plain HTTP, front it with a TLS proxy"), while upstream's `llama-common`
-- `common/download.cpp` -- linked its own `cpp-httplib` static library *with* the define whenever an
OpenSSL was found. Two copies of the same classes with different layouts in one link; the linker's
choice per platform decided the behaviour: on **Linux** our objects won, `libssl` was never pulled in
(measured: no `NEEDED` on it even though CMake had found OpenSSL 3.0.13), and an `https://` download
threw `HTTPS is not supported` -- in **every** Linux jar ever shipped; on **macOS** the archive member
was pulled and the dylib depended on `/opt/homebrew/opt/openssl@3` (found by `verify-native-deps.py`,
carried as a known defect); on **Windows** BoringSSL was already configured but the same duplicate
sat in the link. The fix is one copy: `jllama` and `jllama_test` link upstream's `cpp-httplib` target
(PUBLIC define, so `server-http.cpp` and the download code agree) and no longer compile `httplib.cpp`.

**Certificates.** cpp-httplib loads the OS store: `crypt32` on Windows, Security.framework on macOS
(the two frameworks were already in the Mac allowlist), and on Linux BoringSSL's compiled-in defaults
`/etc/ssl/certs` (hashed directory) and `/etc/ssl/cert.pem`, overridable with `SSL_CERT_FILE` /
`SSL_CERT_DIR` -- which is also how a private CA or a corporate proxy's certificate is trusted.
Upstream's download code has no `--insecure`; since b11699 (#21752, `common/http.h`) its HTTP client honours the
`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` environment variables (loopback never goes through the proxy). This project
adds nothing on top.

**Measured locally (Linux x86-64, the sandbox, BoringSSL `0.20260929.0`)** before the CI run: the
same `NEEDED` list as before (no `libssl`, no `libcrypto`), 56 `httplib::SSLClient` symbols linked,
603/603 C++ tests and the model-free Java suite green; the embedded server in router mode behind a
self-signed certificate answered `/health` and `/models` over TLS 1.3 and refused a plain-HTTP
request on the same port (curl exit 52); a download from a local TLS server was **refused** with
the certificate untrusted (`HEAD failed, status: -1`, nothing written) and **completed** with
`SSL_CERT_FILE` pointing at it (the 64 KB dummy landed at the `-m` path, then failed the GGUF magic,
as intended). Two things learned on the way, both now in the smokes: `--model-url` **alone** starts
the server in **router mode** (`is_router_server` looks at `model.path`, `hf_repo` and `docker_repo`
before the URL is resolved), so a download run names its target with `-m` too; and hostname
verification needs an IP SAN for an IP-literal host, which a bare `CN=127.0.0.1` certificate lacks
-- `curl -k` for the server round, a SAN for the client proof.

**What CI proves, per platform.** `verify-native-deps.py` (package): the allowlists above.
`smoke-natives` (Linux x86-64 / aarch64 in bash, Windows x86-64 / arm64 in PowerShell) and
`smoke-natives-macos`: a second server round behind a self-signed certificate, serving a model it
downloaded itself from the `https://` URL of `stories260K.gguf` (1 MB, `models.csv`) into a
`LLAMA_CACHE` of the run -- `/health` and a chat completion over HTTPS, plain HTTP on the TLS port
refused, the `.gguf` present in the cache. That download is the only place besides `fetch-models`
that reaches Hugging Face, on purpose: it is the end-to-end HTTPS client test with a real
certificate chain and the OS store, which no local check can give. The step passes the `HF_TOKEN`
secret, which llama.cpp sends as a bearer token (`-hft` reads it from the environment, also for a
`--model-url` download): anonymous, the runner IP's shared rate limit answered 429 once (run
40554119477). The macOS smoke thereby stopped being model-free; it still fetches no CI model and
takes ~1 min.

**Cost.** BoringSSL is ~400 C/C++ files compiled with `OPENSSL_NO_ASM` (upstream's setting, no
assembler needed), cached by sccache like everything else; the shipped library grows by its
`libssl` + `libcrypto` text. **On a BoringSSL or llama.cpp bump:** the tag is upstream's
(`BORINGSSL_VERSION` in `vendor/cpp-httplib/CMakeLists.txt`); it needs CMake >= 3.22, the floor of
every image here. A local build in a sandbox that cannot reach `boringssl.googlesource.com` passes
`-DBORINGSSL_GIT=https://github.com/google/boringssl` (the mirror carries the tags) or
`-DFETCHCONTENT_SOURCE_DIR_BORINGSSL=<clone>`.

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
stays in the loop. **The engine clears sequence 0 of the KV cache before `set_input()`**: the Qwen3-TTS
pipeline numbers every prompt from position 0 and leaves the cache to the caller (upstream's server
calls `prompt_clear()` right before `set_input()`; `tools/tts/tts.cpp` synthesizes once per process and
never needs to). Without it every synthesis after the first on one `TextToSpeech` failed with "prompt
processing failed" -- unseen until `TtsIntegrationTest` shared one instance across its tests;
`synthesizesAgainOnTheSameInstance` pins it.

**Why this replaced OuteTTS, not extended it.** Upstream #26254 ("mtmd: support Qwen3-TTS") deleted
the entire OuteTTS implementation from `tools/tts/tts.cpp` (it shrank from ~1450 to 205 lines) and
replaced the two-model OuteTTS-(text-to-codes)-+-WavTokenizer-(vocoder) design with the single
backbone+mmproj design above — there is no upstream code path for OuteTTS left at all past b10269
(`enum mtmd_gen_audio_type` has exactly `MTMD_GEN_AUDIO_TYPE_NONE` and `MTMD_GEN_AUDIO_TYPE_QWEN3TTS`).
See `docs/history/llama-cpp-breaking-changes.md`'s `b10269–b10270` row for the full investigation;
this was a breaking **public API** change (`TextToSpeech`'s constructor and `synthesize()` overloads
all changed shape) done deliberately — the project does not carry OuteTTS-compatibility shims.

**Nothing silent to re-verify on a llama.cpp bump, but the surface does move.** There is no generator or
extracted header any more, so a bump cannot break the TTS surface the way `patches/` or the old extraction
could: `mtmd_helper::gen_audio` is upstream's own committed public header, and a change there is a compile
error in `tts_engine.cpp` (`tools/mtmd/mtmd-helper.h` is on the priority list). It moved at b11796 (#30902):
the helper is default-constructed and initialised with `init(lctx, mctx)` / `valid()` instead of a
constructor, and `mtmd_helper_gen_audio_inp_default()` fills the input struct, which gained a `stream` flag
(off for a one-shot synthesis); `tts_engine.cpp` picked both up at the b11799 bump.

**A model with a `"tts"` chat template (LFM3-Audio, b11941, #31257) speaks the prompt rendered from it.**
The engine keeps the model's chat templates (`common_chat_templates_init(model, "")`, as `llama-tts`)
and asks `common_chat_get_tts_prompt(templates, text, lang, voice)` for every synthesis: non-empty for
such a model, which then gets the rendered prompt, empty for the others (Qwen3-TTS, Pocket TTS, Soprano),
which keep getting the text itself -- upstream's `/tts` route and `llama-tts` do exactly this. The voice
(`mtmd_helper_gen_audio_inp::voice`, LFM3-Audio: `female`/`male`) goes through `engine_synthesize`, the
C API's `jllama_tts_synthesize_voice` (6.1.0; `jllama_tts_synthesize` is it with a null voice, so its
callers are unchanged), the JNI `synthesizeNative`, `TextToSpeech.synthesize(text, ref, language, voice,
...)` and `FfmTextToSpeech`. **The language default depends on the template** (`tts_pipeline_lang` in
`tts_params.hpp`, unit-tested): a template-rendered prompt leaves an empty language empty, so LFM3-Audio
uses its template's default `en` (its codes are `en`, `de`, ...), while the pipelines without a template
keep this library's historical `"english"`. `TtsChatTemplateIntegrationTest` covers LFM3-Audio
(`-Dnet.ladenthin.llama.tts.template.model/.mmproj`, not in the CI set: 1.4 GB): language and voice, the
template's defaults, two voices giving different samples for one seed, reproducibility, no voice cloning.

## Upgrading/Downgrading llama.cpp Version

**Runbook (documentation root):** [`docs/upgrade/llama-cpp-version-bump.md`](docs/upgrade/llama-cpp-version-bump.md)
covers the full bump process end-to-end — picking the target (topmost GitHub release, via the atom
feed), **chunking by `git diff` byte-size** (bump straight to the target when the diff is < 100 KiB,
else step through the largest intermediate tag still under the threshold), the
`.github/scripts/llama-next-version.sh` helper that computes the next reviewable step, and the
edit/verify/commit loop below. Use it for any non-trivial bump; the steps here are the mechanical core.

To change the llama.cpp version, update the following **four** files (and re-verify `patches/`):

1. **llama/CMakeLists.txt** — the pin: `set(JLLAMA_GIT_TAG    b8831)`. (There is no second tag to
   keep in sync any more: the cosmetic `-DLLAMA_TAG=` that fed the old build-time TTS extraction went
   away with the Qwen3-TTS rework — see "Qwen3-TTS via `mtmd_helper::gen_audio`".) **A number
   upstream has not tagged yet** (the tags trail master, and a fix worth bumping to may sit in
   between) is pinned the same way plus `set(JLLAMA_GIT_COMMIT "<full hash>")`, the commit the
   number names -- `llama-next-version.sh --resolve b<nnnn>` prints it. Only the two fetches
   (`FetchContent`, the `build-webui` checkout) read the hash; `jllama-build.txt`, the loader,
   `LlamaCppVersion`, README and this file keep the number, and the configure asserts that the
   fetched commit's `git rev-list --count` is the number (upstream's own tag-numbering rule, and
   also how llama.cpp's `build-info` derives `b<nnnn>-<commit>` -- so the Java pin check holds
   against a virtual number too). **Keep `JLLAMA_GIT_COMMIT` after upstream publishes the tag**: a
   tag name is a moving reference, so the commit is the pin and the tag only the display name, and
   CI refuses a fetch by name (`JLLAMA_REQUIRE_COMMIT`, set by `build.sh`/`build.bat` under GitHub
   Actions; a local configure without the commit still works, by the tag). Runbook: "Pinning a
   number upstream has not tagged yet".
2. **README.md** — the badge and link line with the version number
3. **CLAUDE.md** — the "Current llama.cpp pinned version" line
4. **llama/src/main/java/net/ladenthin/llama/value/LlamaCppVersion.java** — the
   `LLAMA_CPP_VERSION` constant (the compile-time pin exposed to Java/Kotlin consumers, e.g. the
   Android version badge). It is the *pure-Java mirror* of `JLLAMA_GIT_TAG` and must stay equal to it —
   the native `LlamaModel.getLlamaCppBuildInfo()` getter reports the actually-linked build
   (`b<number>-<commit>`), and `NativeLibraryLoadSmokeTest.nativeBuildInfoMatchesPinnedVersionConstant`
   **fails the build** if this constant and the linked binary drift apart.

Example: To upgrade from b8808 to b8831:
```bash
# Edit llama/CMakeLists.txt: change JLLAMA_GIT_TAG b8808 to b8831
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
| `common/chat.h` | `common_chat_prompt` (b11612, #30389 "chat: replace output string with parts": `common_chat_params::prompt` and `common_chat_session::prompt()` are a list of template / user-input parts now, `str()` renders the text, `common_chat_template_direct_apply` returns one -- `jllama.cpp` never reads the prompt, so nothing here changed), `common_chat_session` (b11531, #30210: one object per generation owning the prompt and the parser state; `oaicompat_chat_params_parse` fills it as its last argument, `server_task::apply_chat_session` consumes it, `server_response_reader::post_tasks` takes it -- `jllama.cpp` threads one through every chat entry point), `common_chat_parser_params` (was `common_chat_syntax`), `to_json_oaicompat`, `common_chat_msg_diff_to_json_oaicompat`, `set_tool_call_ids` |
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
mvn package          # Build the classes JAR (the natives jars are the CI-only `natives` profile; there is no fat jar)
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

**Use Temurin 25 from `.java-version` for builds; the core still targets Java 8.**
The compiler must accept the configured `-XDaddTypeAnnotationsToSymbol` flag. Historically,
Temurin 21 accepted it while Oracle JDK 21 did not. NullAway's JSpecify mode needs it below JDK 22; on Oracle JDK 21.0.9 `mvn compile` dies in an
Error Prone `IllegalStateException` naming exactly that flag, while Temurin 21.0.12.1 compiles the
same tree (measured, `docs/handover/local-agent-report-b11538-windows.md`, deliverable 1). Check
`mvn -version` before blaming anything else.

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
| Windows x86_64 | `jllama.dll` | `src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu/` |

A plain `cmake -B build` is the **static** build: `CMakeLists.txt` forces `BUILD_SHARED_LIBS OFF`, so
upstream `llama` and `ggml` are static libraries linked into one `jllama` library (what ships for
macOS, and what every `test-cpp-*` job and the local test loop build). `LlamaLoader` then
extracts and loads that single file (plus the files in its `jllama-extras.txt`, and `ggml-metal.metal`
from a Metal build that does not embed it). The shipped Linux / Windows / Android library jars are
the **modular** build, `cmake -B build -DJLLAMA_BACKEND_DL=ON` (see "Modular natives"): ggml shared,
the directory holding its libraries and the CPU modules, which the loader extracts without loading
them (`jllama-files.txt`); a GPU module jar is `-DJLLAMA_MODULE_ONLY=ON -DGGML_<X>=ON` into the same
tree, and the loader of a local `mvn test` picks both up from `src/main/natives/`. Historical note:
upstream kherud once shipped split `ggml` + `jllama` libraries, which is where stale "three
co-located DLLs" claims came from.

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
(`TestConstants.DEFAULT_*`), so downloading a model from that list into `models/` is all it takes
(`bash .github/fetch-models.sh vision`, or with no argument the whole set) —
the tests find it (module-relative or at the reactor root) and self-skip while it is missing. The
properties below only **override** a default. The full property → consumer → default table for every
`net.ladenthin.llama.*` property the library understands (runtime + test) is the user-facing
**[System Properties Reference](README.md#system-properties-reference)** in the README. The summary
below covers the model bindings:

| Property | Default test that uses it | Default (`models/…` unless noted) |
|----------|---------------------------|-------|
| `net.ladenthin.llama.text.model` | `LlamaModelTest` and most model-backed tests (`MODEL_PATH`) | `codellama-7b.Q2_K.gguf` |
| `net.ladenthin.llama.draft.model` | speculative decoding + the small-model tests (`DRAFT_MODEL_PATH`) | `AMD-Llama-135m-code.Q2_K.gguf` |
| `net.ladenthin.llama.reasoning.model` | reasoning-budget tests (`REASONING_MODEL_PATH`) | `Qwen3-0.6B-Q4_K_M.gguf` (must emit a thinking block) |
| `net.ladenthin.llama.rerank.model` | reranking tests (`RERANKING_MODEL_PATH`) | `jina-reranker-v1-tiny-en-Q4_0.gguf` (must be a reranker) |
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
# CI's models, e.g. the vision pair (bash .github/fetch-models.sh vision) -- no property needed afterwards:
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
[root], `exception.LlamaException`, `exception.InvalidRequestException`, `value.LogLevel`, `args.LogFormat`,
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
- `TextToSpeech` — Separate AutoCloseable native type for speech synthesis over llama.cpp's upstream Qwen3-TTS pipeline (a backbone text GGUF + an mmproj GGUF bundling the speaker encoder, code predictor, and code2wav decoder); `synthesize(text)` returns a 24 kHz mono 16-bit WAV byte stream, with overloads for a cloned-voice speaker-reference clip and language; `supportsSpeakerReference()` says whether the pair clones voices (llama.cpp's `mtmd_helper_cap::audio_gen_spk_ref`, read once at load), and the engine refuses a reference clip by name for a pair that does not. Native orchestration in `tts_engine.{h,cpp}` drives upstream's `mtmd_helper::gen_audio` streaming API directly (see "Qwen3-TTS via `mtmd_helper::gen_audio`" below) — there is nothing extracted or hand-copied from llama.cpp source; the in-memory WAV writer is `tts_wav.hpp`.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes. Every wire name they can
  emit is an enum constant carrying the contract it must satisfy (`args.ModelOption` + `args.ModelFlag` for
  argv, `parameters.RequestField` for the request body), and the base classes accept nothing else — see
  "Wire-name registries" below. `InferenceParameters.toJson()` renders the request body; its `toString()`
  is a redacted debug view and deliberately not valid JSON.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.
- **`server` package — OpenAI-compatible HTTP endpoint (a single implementation).**
  - `server.OpenAiCompatServer` — built only on the JDK's `com.sun.net.httpserver` (no new dependency), embeddable and runnable via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …` (the classes jar's `Main-Class`, `ServerLauncher`, runs `NativeServer` by default — see "Two server modes"). Serves `POST /v1/chat/completions` (streaming via SSE + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST /infill`, `GET /v1/models` and `GET /health` (every route is also reachable without the `/v1` prefix), so editors that speak the OpenAI protocol (e.g. VS Code Copilot "Custom Endpoint", Cline, Roo Code, Continue) can drive a local model. Streaming chat uses the native OAI chunk path (`LlamaModel.streamChatCompletion` → `requestChatCompletionStream` / `receiveChatCompletionChunk` + the C++ `wrap_stream_chunk` helper), preserving `delta.tool_calls`; completions/embeddings/infill forward verbatim to the matching `LlamaModel.handle*`, and a streamed `/v1/completions` verbatim to `LlamaModel.streamCompletionsOai` (upstream's own `text_completion` chunks, `usage` on the last); rerank reshapes `handleRerank` into the OAI `results`/`data` shape with `usage`, or, for a request with TEI's `texts`, into TEI's bare `[{index, score[, text]}]` array (`OaiRerankSupport.toResponse`, as upstream's `format_response_rerank`). The chat mapper forwards `stream_options` and `response_format` and defaults `cache_prompt=true`; a CORS `Filter` answers `OPTIONS` preflights; `OpenAiSseFormatter.ensureUsageCachedTokens` guarantees `usage.prompt_tokens_details.cached_tokens` on the streamed usage chunk (Copilot crash fix, microsoft/vscode #273482). **Agentic tool-calling is the primary target**; a C++ guard (`test_server.cpp`) pins `tool_calls.function.arguments` as a JSON string (llama.cpp #20198).
  - **Alternative protocol surfaces** (pure translation over the OpenAI chat core — no second inference path; each reconstructs streamed tool calls via `ToolCallDeltaAccumulator`): **Ollama-native** (`GET /api/version`, `/api/tags`, `POST /api/show`, `/api/chat` with NDJSON streaming, `/api/generate` prompt-completion/FIM — `OllamaApiSupport`; `/api/show` advertises tools/insert/vision capabilities + context length for Copilot's Ollama provider), **Anthropic Messages** (`POST /v1/messages`, SSE event stream — `AnthropicApiSupport` + `AnthropicStreamTranslator`), and **OpenAI Responses** (`POST /v1/responses`, SSE event stream — `ResponsesApiSupport` + `ResponsesStreamTranslator`). Upstream's three token-count routes (`/v1/messages/count_tokens`, `/v1/responses/input_tokens`, `/v1/chat/completions/input_tokens`) translate like their generating route and count through `OpenAiBackend.countTokens` (template + tokenize; `501` for content only the projector can count). Every route answers its own path only (F112), and a failure after a stream started is the route's error event, not a silent end (F262). The llama.cpp-native `GET /props` (context length + `modalities`) is served via `OpenAiSseFormatter.propsJson` for autocomplete clients that size their context from it.
  - Supporting classes: `OpenAiServerConfig` (builder; optional bearer auth; binds `127.0.0.1`; port 9931 by default, llama.cpp's, as `NativeServer`; `corsAllowOrigin`; `supportsVision`), `OpenAiServerCli` (testable CLI arg parser → `ModelParameters` + `OpenAiServerConfig`; flags incl. `--mmproj`/`-mmdev,--mmproj-device`/`--embedding`/`--reranking`), `OpenAiRequestMapper` (OAI chat request → `InferenceParameters`), `OpenAiSseFormatter` (SSE/models/error JSON + usage normalization), `OaiRerankSupport` (pure rerank request/response shaping), and the model-free test seam `OpenAiBackend`/`ChunkSink` + `LlamaModelBackend`. The streaming envelope is parsed by `json.ChatStreamChunkParser`.
  - The `server` package is a dedicated top layer in the ArchUnit `layeredArchitecture` rule (the only layer allowed to access the root `Api`); `noInternalJdkImports` carries an explicit exception for the supported `com.sun.net.httpserver` (the exported `jdk.httpserver` module, which `module-info.java` `requires`). See README "OpenAI-compatible HTTP server".

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp. 39 native methods (34 `LlamaModel` + 4 `TextToSpeech` + 1 `LlamaQuantizer`) plus `JNI_OnLoad`/`JNI_OnUnload`; since the C API (see "The C API" below) every one of them is a shell over `jllama_api.cpp` / `jllama_engine.cpp`, where the model logic lives.
- `utils.hpp` — Helper utilities (format helpers, argv stripping, token-piece serialisation).
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- **The `json` alias is upstream's `common_json`, not `nlohmann::ordered_json` (since llama.cpp b10585, upstream #27511).** `tools/server/server-common.h` now says `using json = common_json;` — a deliberately small pimpl wrapper (`common/json.{h,cpp}`, compiled into `llama-common`) around the vendored nlohmann copy. Two traps this cost the project once, both of which **compile silently**:
  1. **An unscoped enum became a JSON boolean — fixed upstream at b11080, and still worth knowing.** From b10585 to b11069, `common_json_value`'s integral constructor template was `std::is_integral`-gated, which excludes enums, so an enum bound to `common_json_value(bool)` and serialised as `true`/`false`. Upstream [#28518](https://github.com/ggml-org/llama.cpp/pull/28518) added an `std::is_enum`-gated constructor delegating to the underlying type, which retired `patches/0010` (see the drop note under the patches table). **Keep casting anyway** — `jllama.cpp`'s two `"vocab_type"` sites still `static_cast<int>(...)`, which is equivalent under the fix and immune to a revert of it. Guards, both in CI: `test_json_helpers.cpp`'s `CommonJsonEnum` trio pins the mechanism (uncast enum is numeric, explicit cast equivalent, `bool` still boolean); `LlamaModelTest`'s `isIntegralNumber()` assertion pins the real wire value (the model-gated suite no longer self-skips — see "CI model policy" below).
  2. **`common_json` converts to `std::string` implicitly**, so it binds happily to a `const nlohmann::json &` parameter (via nlohmann's string-constructible converting constructor) and then throws `json::type_error 302` at runtime. Never declare a project helper as taking `nlohmann::json` when callers pass the `json` alias — `require_json_field_impl` is a template for exactly this reason.
  Other differences to know: no `get_ref`/`array_t`/`type_name()`; a braced list in *value* position does not build an array (write `json::array({...})`); `at(key)` needs an explicit `.get<T>()`; errors are `common_json_error`; and `get<T>()` is limited to the types explicitly specialised in `common/json.cpp`. `log_helpers.hpp` and `train_engine.cpp` keep their own `nlohmann::json` alias — they never touch the server's `json`.
- Uses `nlohmann/json` for JSON deserialization of parameters in the two files named above; everything on the server path uses `common_json`.
- The upstream server library (`server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-schema.cpp`, `server-models.cpp`, and — since b9829 — `server-stream.cpp`) is compiled directly into `jllama` via CMake — there is no hand-ported `server.hpp` fork. **`server-stream.cpp` is mandatory, not optional:** it defines the resumable-streaming SSE replay buffer (`g_stream_sessions`, `stream_session_attach_pipe`, `stream_aware_should_stop`, `stream_conv_id_from_headers`, the `stream_pipe_*` types) that `server-context.cpp` / `server-http.cpp` / `server-models.cpp` now `#include "server-stream.h"` and call, so omitting it fails the link with undefined references. It is platform-neutral (threads + std mutex/condvar, no `subprocess.h`/`posix_spawn_*`), so it builds on Android too and sits outside the `server-models.cpp` Android guard. `jllama` wires its own JNI routes and never calls `g_stream_sessions.start_gc()` (only the excluded standalone `server.cpp` `main()` does), so its GC thread stays dormant. **Two more units are mandatory for the same reason:** `server-realtime.cpp` (b11946, #31269; `server-context.cpp` creates a `server_realtime_slot` and serves `/v1/realtime` conversations through it -- it uses only the `ws_conn` interface, so it builds on Android too) and, next to `server-models.cpp` under its Android guard, `server-models-sched.cpp` (b11930, #21231; the router's request queue and memory accounting, which `server-models.cpp` calls on every spawn). Both were measured as link failures of `jllama_test` without them. **Phase 2:** the upstream HTTP transport (`tools/server/server-http.cpp`) is compiled into `jllama` too, and `jllama` links upstream's `cpp-httplib` static target (`vendor/cpp-httplib/`) instead of compiling `httplib.cpp` a second time (it did until 6.0.0, without SSL -- see "HTTPS: BoringSSL, statically" for what that cost), so the OpenAI-compatible server can be driven natively from JNI *inside* `libjllama` — no separate `llama-server` executable (a JNI shared library loads anywhere a JVM runs, which a standalone binary does not). `server-http.cpp` does `#include "ui.h"` (the WebUI asset table that `tools/ui`/`llama-ui` normally generates); since the Svelte WebUI is not shipped, `src/main/cpp/webui_stub/ui.h` supplies the upstream **empty-asset** interface and leaves `LLAMA_UI_HAS_ASSETS` undefined (all static-asset-serving blocks compile out). `<cpp-httplib/httplib.h>` already resolves through `llama-common` — since upstream #27304 (b10488) not from a `PUBLIC ../vendor` include dir of its own but transitively, via the `vendor::nlohmann` / `vendor::sheredom` INTERFACE targets it links PUBLIC, each of which exports the `vendor/` root -- and that root came **before** the then FetchContent `json-src` include dir on every `jllama` compile line (measured in `flags.make`), so even our own `#include <nlohmann/json.hpp>` in `log_helpers.hpp` / `train_engine.cpp` compiles against upstream's **vendored** copy: one copy per TU. Since b11556 that copy is 3.12.0 plus upstream's `json-deep-nesting.patch` (nlohmann/json#5387, applied by `scripts/sync_vendor.py`), so the FetchContent `v3.12.0` was not even the same code; it is gone, and `llama/CMakeLists.txt` ("json: upstream's vendored copy, and only that one") guards that upstream's copy is there. **HTTPS is on**: the `cpp-httplib` target carries `CPPHTTPLIB_OPENSSL_SUPPORT` as a PUBLIC define whenever an SSL library is configured, and BoringSSL is linked statically on every desktop platform, so `server-http.cpp` serves TLS with `--ssl-key-file`/`--ssl-cert-file` and `common/download.cpp` fetches `https://` URLs (not on Android and s390x, where `LLAMA_OPENSSL=OFF`). **`server.cpp`, `server-tools.cpp` and `server-mcp.cpp` are now compiled in too** (on non-Android — they pull in `subprocess.h`/`posix_spawn_*`, so they share `server-models.cpp`'s Android guard): b9870 exposes `server.cpp`'s entry as `int llama_server(int, char**)` (no `main` in the file), and `patches/0006` makes it embeddable (no process signal handlers, forwarded-argv parse, out-of-band shutdown). **`server-mcp.cpp` is new in b10154** (upstream MCP-server support): both `server.cpp` (`llama_server`'s `mcp_mgr` lifecycle) and `server-tools.cpp` (`tools.setup(..., mcp_mgr)` / `server_mcp::call_tool`) reference `server_mcp`, so it **must** be in the `target_sources` list or the link fails with undefined `server_mcp::{start,shutdown,call_tool,list_tools,~server_mcp}` — **latent on Linux** (a shared object tolerates undefined symbols) but a **hard link error on macOS/ld64 and Windows/MSVC**. It is compiled into `jllama` and — on non-Android — into `jllama_test` too, together with `server.cpp`, `server-tools.cpp` and `server-http.cpp` (and the linked `cpp-httplib` target), so `test_native_server_shutdown.cpp` can drive the real `llama_server()`. The `NativeServer` JNI bridge (`src/main/cpp/native_server.cpp`) calls `llama_server` on a worker thread, so the **full** upstream server — WebUI and all — runs inside `libjllama`. See "Two server modes" below.

### Two server modes (`OpenAiCompatServer` vs `NativeServer`)

The library exposes **two** ways to serve a model over HTTP, on two different transports. The classes jar's `Main-Class` is `server.ServerLauncher`, a tiny dispatcher (`jbang --deps net.ladenthin:llama:<v>:<natives classifier> net.ladenthin:llama:<v> …` starts it from Maven Central, `examples/server/pom.xml` with Maven): it runs `OpenAiCompatServer` when `--jllama-openai-compat` is present (that marker is stripped, the rest forwarded) and the default `NativeServer` otherwise. Both mains are also runnable directly by class name via `java -cp`. The two modes:

1. **`server.OpenAiCompatServer` (Java transport).** OpenAI/Ollama/Anthropic-compatible JSON API on the JDK's `com.sun.net.httpserver`, driving the compiled server *core* over JNI. Embeddable, no extra dependency, and it can share/reuse a `LlamaModel`. It serves **no** static assets — its `/` route is a 404, so **no WebUI**. It has its own `main` (run via `java -cp <jar> net.ladenthin.llama.server.OpenAiCompatServer …`); its CLI (`OpenAiServerCli`) maps a curated flag subset (`-m/-c/-b/-ub/-ngl/-t/-tb/-ctk/-ctv/--jinja/--chat-template-kwargs/--host/--port/--parallel/--mmproj/--api-key/--embedding/--reranking/--rpc`).
2. **`server.NativeServer` (native transport) — what `ServerLauncher` runs by default (when `--jllama-openai-compat` is absent).** Runs the **full upstream `llama_server`** (via `patches/0006` + `native_server.cpp`) inside `libjllama`, forwarding the raw llama-server argv verbatim — so **every** llama-server flag works and the **embedded WebUI is served** (when the assets are compiled in; CI's released jars have them, local `cmake` builds use the empty-asset stub). With the classic constructor it is an **independent lifecycle** (loads its own model from the argv, like `llama-server.exe`; owns the process's llama backend + stderr logging while running); the **attach constructor** (`NativeServer(LlamaModel, String...)`, via `patches/0007`'s `llama_server_attach`) instead serves an **already-loaded `LlamaModel`** — one copy of the weights, the model's worker keeps driving inference, the HTTP routes post to its queue; caller closes the server before the model. **Router mode** (start without a model argument: `--models-dir`, `GET/POST /models`, per-request model selection) works in-JVM after `NativeServer.setWorkerCommand(...)` redirects the worker spawn to a fresh JVM (`patches/0008` — upstream re-execs its own binary, which in a JVM is `java`); the typed `server.RouterClient` (+ `value.RouterModel`, `json.RouterModelsResponseParser`) wraps the model-management endpoints (list/load/unload/await-loaded with fail-fast on failed workers) so callers don't hand-roll HTTP+JSON, and its `apiKey` constructors send `Authorization: Bearer <key>` — required for **every** one of those calls against a router started with `--api-key` since b10519 (#26347 dropped `/models` + `/v1/models` from the public-endpoint set; `/models/load` and `/models/unload` were always gated). `awaitModelLoaded` cannot observe a model hidden by a preset with `dedup-cache-models` (b10505/#27346 omits it from `GET /models` although it still loads and serves by name), so its "not listed" message names that cause explicitly; such a model is reached by issuing the request directly instead. Either way it is **single-instance per process** (upstream keeps shutdown state in file-scope globals) and **not available on Android** (the `subprocess.h` guard). `libjllama` loading anywhere a JVM runs is what makes this "no separate `llama-server.exe`" possible.

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
`server_metrics_to_json`, `route_error_message`, `get_result_error_code`.

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
- **Messages go through `jni_utf8.hpp`** (`throw_utf8_impl`, `new_string_utf8_impl`): the same rule for
  error text, which quotes paths and request content. ASCII stays on `ThrowNew`/`NewStringUTF`
  (identical in both encodings), anything else crosses as bytes. Self-contained (jni.h only, no cached
  ids), so `train_engine.cpp`, which never includes this header, uses it too. A new `ThrowNew` or
  `NewStringUTF` with text that is not a literal goes through it.
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
convert anything that escapes into a Java exception**, and there are 49 of them across four TUs —
`jllama.cpp` (39), `native_server.cpp` (5), `rpc_bridge.cpp` (4), `train_engine.cpp` (1).

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

**Two Java classes, chosen by what upstream would answer.** `c_llama_error` (`LlamaException`) is the
backstop and the class for everything that is not the request's fault; `c_llama_invalid_request`
(`InvalidRequestException`, a subclass) is thrown wherever llama.cpp's own HTTP server answers `400`,
so that `OpenAiCompatServer` can answer `400` too without matching on message text (it maps the type
like the `IllegalArgumentException` of its own validation; everything else is `500`). Three helpers
in `jllama.cpp` decide it: `throw_invalid_request` for request *parsing* (an invalid request whatever
the parser threw — upstream's parsers mix `std::invalid_argument` and `std::runtime_error` for the
same kind of defect; also behind `parse_json_params` and `require_json_field`),
`throw_by_exception_type` for upstream code whose exception type carries the verdict
(`std::invalid_argument` and `common_json_error` → 400, else 500 — the rule of upstream's `ex_wrapper`
in `tools/server/server.cpp`; tokenization, `format_prompt_infill`, the `/v1/systemone` handler), and
`error_class_for` for a failed task result (`get_result_error_code` reads the code
`format_error_response` put into it: 400 for `ERROR_TYPE_INVALID_REQUEST` and
`ERROR_TYPE_EXCEED_CONTEXT_SIZE`). A `ThrowNew(c_llama_error, …)` is still right for a missing
capability (`--embedding`, `--reranking`, FIM tokens: upstream answers those `501`) and for a broken
result. Pinned model-free by `OpenAiCompatServerHttpTest` (a backend throwing the typed exception is
`400` on every surface, a plain `LlamaException` `500`) and model-backed by `ErrorHandlingTest` and
the two server integration tests (an empty `input` array, a `/v1/completions` body without a prompt).

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
1. System property `net.ladenthin.llama.lib.path` -- a **natives directory** (README, "Natives outside
   Maven"): the library jar's directory unpacked, optionally with module jars' files copied flat into
   it. `loadFromDirectory` loads it with the jar branch's logic minus the extraction: the files its
   `jllama-extras.txt` names by full path first (Windows: `ggml-base.dll`, `ggml.dll` -- a bare
   `System.load` of `jllama.dll` failed there until 5.2.x, because Windows resolves imports from
   `java.exe`'s directory and `PATH`, not from the DLL's), then the library; `jllama-build.txt` feeds
   `getBuildInfo()`; a `jllama-files.txt` entry that is missing is reported; the modules present are
   recognised by file name (`MODULE_LIBRARY_NAMES`, held to `natives.csv` by `check-natives.py`) and
   named in the same loader line; the `backend` filter has no effect and says so. The temp-directory
   cleanup and the module selection run only when the jars are reached, so a stale filter cannot fail
   a `lib.path` start. Proven by `NativeLibraryLoadSmokeTest` in a fresh JVM on every `test-java-*`
   job and by round 3 of every smoke row (`smoke_natives_directory`).
2. `System.loadLibrary` on Android (the AAR's `jni/`), then every `java.library.path` entry as a natives
   directory (same branch)
3. The natives jars on the classpath: the library directory `net/ladenthin/llama/{os}/{arch}/{backend}/`
   of the platform (`LIBRARY_BACKENDS`: `metal`, else `cpu`), extracted together with every GPU module
   directory present (`MODULE_BACKENDS`, filtered by `net.ladenthin.llama.backend`, each checked against
   the library's `jllama-build.txt`), then loaded (see "Natives jars"). `LlamaModel`'s static block warns
   when the loaded build's tag differs from `LlamaCppVersion` (either branch).

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for **Android** targets and the
two x86_64 manylinux_2_28 jobs (modular CPU, the CUDA module). **Linux `aarch64` is built natively** on a
GitHub `ubuntu-24.04-arm` runner inside the pypa manylinux_2_28 image, through
`.github/manylinux/manylinux_2_28-aarch64` (see "Linux aarch64" below). Deleted wrappers: the
`dockcross-linux-arm64-lts` cross-compiler (GCC 8.5), the never-wired 32-bit `dockcross-android-arm`,
and `dockcross-manylinux2014-x64` (GCC 10 cannot build the AVX-VNNI/AMX CPU variants).

### Linux aarch64: native build in the manylinux_2_28 image

The `crosscompile-linux-aarch64` job (id kept for its downstream `needs:` reference; display name
**"Build manylinux_2_28 aarch64 (CPU variants)"**) builds **natively on `ubuntu-24.04-arm`, inside
`quay.io/pypa/manylinux_2_28_aarch64`** (AlmaLinux 8, gcc-toolset-14 = GCC 14.2.1, glibc 2.28),
through the wrapper `.github/manylinux/manylinux_2_28-aarch64`, modular with the CPU variants
(`-DJLLAMA_BACKEND_DL=ON`, see "Modular natives").

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
- No `ctest` (the modular build has no static CPU backend for `jllama_test`): **`test-cpp-linux-aarch64`**
  builds the default static library with `gcc-14` on the same runner and runs the C++ suite on real
  ARM hardware, as this job did before.
- Branch protection: a required check pinned to "Build and Test Linux aarch64" must be repointed.

### Linux s390x: big-endian cross-build, IBM Z CPU variants, qemu test gate

`build-linux-s390x` adds **IBM Z (s390x, big-endian)** — the one target whose
byte order differs from every other platform. It **cross-compiles** at native x86 speed — no emulated
build — a **modular library jar with one CPU module per IBM Z level**, and `test-cpp-linux-s390x`
runs the **full C++ unit suite under `qemu-user`** against the static build of the same toolchain.

**The CPU variants (since 6.1.0).** `-DJLLAMA_BACKEND_DL=ON` with the toolchain below: `libjllama`,
`libllama`, `libmtmd`, `libggml`, `libggml-base`, `libggml-rpc` and `libggml-cpu-z13` (scalar),
`-z15` (VXE2) and `-z16` (VXE2 + NNPA), of which `ggml_backend_load_best` loads the best by the
kernel's `AT_HWCAP` (`arch/s390/cpu-feats.cpp`: VXE2 bit 15, NNPA bit 20). Everything but the two
vector modules is built at the toolchain's `-march=z13`, so the floor stays z13. **Two things in
upstream's s390x variant list needed `llama/patches/0023`:** clang could not build the vector modules
(`ggml-cpu-impl.h` and `llamafile/sgemm.h` both include `<vecintrin.h>`, and clang's copy has no
include guard -- "redefinition of 'vec_extract'"; GCC's has one, and upstream documents GCC as the
s390x compiler), and the list started at z15, so a z13/z14 would have had no CPU backend at all. The
patch drops the include from `sgemm.h` (it declares no vector type, and both its includers include
`ggml-cpu-impl.h` too; it came with ggml-org/llama.cpp#14273) and adds the scalar `z13` variant. z17
(`-march=arch15`) is skipped by ggml's own compiler check: clang 18 does not accept the flag.
**Measured on cfarm191** (z16, LinuxONE, Debian 13, clang 19.1.7, b11901 with the patch; Qwen3-0.6B
Q4_K_M converted to big-endian, `llama-bench -t 4 -p 64 -n 16`, two interleaved rounds): `z13`
19.7-19.8 / 13.6-13.8 t/s, `z15` 90.0-95.5 / 57.3-58.3, `z16` 82.6-88.3 / 51.9-53.9 (prompt /
generation) -- 4.5x and 4x over the scalar library every jar up to 6.0.x shipped. Correctness, by
`llama-perplexity --kl-divergence` against a Q8_0 reference computed on `z13`: the Q4_K_M model
lands at a mean KLD of 0.0777 on `z13` and 0.0746 on `z16` -- equally far from it; the 0.0087
between the two modules is about a tenth of what the quantization costs. The CI-built jar (run
40475405391) loads `libggml-cpu-z16.so` there and serves the same model through `ServerLauncher` at
29.9-35.6 t/s generation with 4 threads, against 9.9-10.0 for the static 6.1.0-SNAPSHOT jar before
it. **Under qemu** (the
`c_abi` run of `build-linux-s390x`, the RHEL 8 row of `smoke-glibc-floor`) ggml loads `z15`:
qemu-user's CPU model has VXE2 and no NNPA, so CI runs the vector code too, in emulation (measured
locally with the job's toolchain: `c_abi` reports `CPU : VXE = 1`, and the library loads in the UBI 8
OpenJDK 21 image on RHEL 8.10). `build-linux-s390x` checks that all three modules were built: a
variant whose `-march` the compiler rejects is skipped by the configure, not reported.

**The toolchain gives it the manylinux_2_28 floor** (glibc 2.28, `GLIBCXX_3.4.25`, `CXXABI_1.3.11`),
like the other two Linux CPU jars, without an s390x runner:
`.github/manylinux/manylinux_2_28-s390x-sysroot` unpacks the pypa `manylinux_2_28_s390x` image
(pinned by digest; nothing of it is executed) as a sysroot, and `.github/manylinux/s390x-toolchain.cmake`
compiles with Ubuntu 24.04's `clang-18` (`--target=s390x-linux-gnu --sysroot=…
--gcc-install-dir=<gcc-toolset-14 in the sysroot> -march=z13`) and links with `binutils-s390x-linux-gnu`
(`--ld-path`). clang takes the toolset's crt files, libgcc and libstdc++ exactly as a native
gcc-toolset build would — the system `libstdc++.so.6` of GCC 8 plus `libstdc++_nonshared.a` through
the toolset's linker script, whose absolute paths ld resolves inside the sysroot. No packaged x86 GCC
targets s390x with glibc 2.28, and running the toolset's own s390x gcc under qemu would take hours per
(uncached) tag build. `-march=z13` is the arch11 baseline the former GCC cross build defaulted to (and
RHEL 8's). The toolchain's emulator is `env QEMU_LD_PREFIX=<sysroot> qemu-s390x-static`, so `ctest`
runs against glibc 2.28 and GCC 8's libstdc++ -- the oldest runtime the jar promises; the prefix must
be an environment variable, not qemu's `-L`, because googletest's death tests re-execute the binary
through binfmt_misc without arguments (with `-L` the six `RpcComm` death tests failed on
`Could not open '/lib/ld64.so.1'`). The sysroot path is a cache variable, so a reconfigure without
`JLLAMA_S390X_SYSROOT` in the environment still finds it. The job then runs `verify-native-deps.py` on
its own tree, and `smoke-glibc-floor` loads the jar on RHEL 8.10 s390x under qemu.

**Mixing clang's objects with GCC's needs `s390x-typeinfo-names.ld`** (next to the toolchain, passed as
`-Wl,-T` with `INSERT`, so it only augments ld's default script). The s390x ABI promises every global
symbol an even address and GCC relies on it, loading addresses with `LARL` (`R_390_PC32DBL`, counted in
halfwords); clang emits typeinfo name strings (`_ZTS*`, COMDAT) with the alignment of `char`
(`GetAddrOfTypeName` in `ItaniumCXXABI.cpp`, unchanged on LLVM main in 2026-10), and the copy a link
keeps is clang's, from our objects. gcc-toolset-14's `libstdc++_nonshared.a` is GCC code
(`fs_path80.o`'s `_M_get_deleter` loads `_ZTSSt19_Sp_make_shared_tag`), so the link failed with
`misaligned symbol ... for relocation R_390_PC32DBL` whenever the strings before it added up to an odd
size: green on main, red in the next PR's `jllama_test` link (#577). The script puts every
`.rodata._ZTS*` into one output section with `SUBALIGN(2)`. Measured with Ubuntu's s390x cross GCC
standing in for the toolset (a GCC archive whose `_M_get_deleter` uses `LARL`, clang objects in front
of it, the real toolchain file over a sysroot in the RHEL layout): the link fails without the script
and succeeds with it, for an executable, a PIE and a shared object, and the program runs under qemu.
**The script reaches only names in a `.rodata._ZTS*` section of their own**, and clang gives one only
to COMDAT names; those of classes with a key function and of internal types (lambdas) sat in plain
`.rodata`, so the toolchain also passes **`-fdata-sections`**. **And the link is not the test:** it
fails only where GCC code happens to reference an odd name, so `verify-native-deps.py` reads the
static symbol table of every s390x library and fails any `_ZTS*` at an odd address -- by every
binding, because the export map makes all of `libjllama`'s local (`TYPEINFO_NAME_ALIGNMENT` in
`nativedeps.py`; a `libjllama` without any typeinfo name fails too, since it would pass vacuously;
the other libraries of the directory are checked when they carry any). It runs in
`build-linux-s390x` after the build. Measured on two local builds of the library with the
stand-in sysroot: script alone, 444 of 1833 typeinfo names odd; script plus `-fdata-sections`, 0.

Until 2026-10-25 the job used the runner's `g++-s390x-linux-gnu` (GCC 13) with its glibc 2.39
sysroot: the 6.0.0 jar referenced `GLIBC_2.38` (`__isoc23_strtol` & co., `fmod`), `GLIBCXX_3.4.32`
and `CXXABI_1.3.13`, and failed to load on RHEL 8/9 and Ubuntu 22.04 while every job was green --
`GLIBC_CEILING` held only the manylinux directories, and the qemu `ctest` ran against the same new
sysroot. Measured after the move (b11887): `GLIBC_2.27`, `GLIBCXX_3.4.25`, `CXXABI_1.3.11`, new
dependencies `libpthread.so.0` and `libdl.so.2` (glibc 2.28 has not merged them yet), 694/694 C++
tests under qemu against the sysroot and natively on a LinuxONE z16 (cfarm191, Debian 13), and the
jar loads on RHEL 8.10 s390x where the 6.0.0 one fails with `GLIBC_2.38 not found`.

That `ctest` run (`test-cpp-linux-s390x`) is a **real big-endian correctness gate** for the byte-order-sensitive surface — the
little-endian WAV writer (`tts_wav.hpp`), the JSON/token/embedding transforms, and the JNI helpers —
which is where an endian bug in *our* code could hide. Model-backed **Java** tests are deliberately
**not** run under emulation (a JVM + GGUF inference under `qemu-user` is slow and flaky); the Java↔JNI
boundary uses host-native array copies (endian-transparent), so the C++ gate covers the actual risk.
`-DGGML_OPENMP=OFF` sidesteps cross-libgomp issues (ggml uses its own `std::thread` pool).
`-DGGML_NATIVE=OFF` switches VXE off for everything outside the variant modules, which carry their
own `-march`; **do not pass `-DGGML_VXE=ON`**: ggml then adds VXE2 to every variant, the `z13` one
included, which would score 0 without VXE2 and leave a z13/z14 without a CPU backend (the static
build of the test job stays scalar; the `0013` note of `docs/history/dropped-llama-patches.md`
measured what the flag does to it). s390x is a
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
floor on the number of tests actually executed -- `.github/verify-test-counts.sh`, run by every
`test-java-*` job: no class with zero entries, and at least 1800 tests executed (run minus skipped;
the jobs execute 1856 to 1872, a checkout without models 1589). It is why several stale
assertions (e.g. `LlamaModelTest#testGetMetrics` against a payload shape upstream had dropped at
b10408) never failed in CI. The fix is **`TestConstants.resolveModelPath` /
`resolveModelProperty`**, which accept either layout — module-relative first, then the reactor root
— so a developer with models under `llama/models/` and CI with them at the workspace root both
work, with no workflow change. Every `TestConstants` path constant is routed through it, as is
every `-Dnet.ladenthin.llama.*` fixture property; `TestConstantsTest` pins both the resolver and
the wiring (a future edit that drops the wrapper from a constant fails that test rather than
silently re-muting the suite). `llama-langchain4j` had the identical defect and carries the same
resolver as `TestModelPaths` (test classes are not shared between modules).

`fetch-models.sh` treats every model a job asks for as **required** (a missing or changed model
hard-fails the job before tests run, so a download regression can never silently downgrade to a
skip). **The tests default to
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
The model set has a **single source of truth: `.github/models.csv`**, one `filename,feature,sha256,url`
row per model (`#` comments): the **feature** says what the model is in the set for (`text`, `draft`,
`reasoning`, `tools`, `rerank`, `embedding`, `vision`, `tts`, `train`), the **URL** is pinned to a
Hugging Face commit (`resolve/<commit>/`, never `main`) and the **SHA-256** is the file's. A job
gets its models from the composite action **`.github/actions/fetch-models`**, naming the features it
needs (`features: draft tts`; none = every row); it runs **`.github/fetch-models.sh`**, which downloads
those rows from Hugging Face with the **`HF_TOKEN`** secret (a fine-grained read-only token; on a
fork pull request it is absent and the download is anonymous), refuses a file whose SHA-256 differs,
and keeps a file that is already there with the right hash (so the same command fills a local
`models/`). What each job asks for:

| Job | Features |
|---|---|
| `test-java-linux-x86_64`, the five `java-tests.yml` callers | all |
| `test-java-llama-langchain4j-integration` | `reasoning embedding rerank tools` |
| `test-java-llama-ffm` | `draft tts` |
| `test-java-llama-atmosphere-agent-integration`, `smoke-agent-linux` | `tools` |
| `smoke-natives` (4 rows), `test-android-emulator`, `test-android-llmservice` | `draft` |
| `post-release.yml` `consumer` | `train` |

`buildcheck/models.py` (`check-natives.py`) holds the rows to that format (four fields, one
lower-case word as feature, a lower-case SHA-256, a pinned URL that names the row's file, no file
twice) and every `features:` of publish.yml to a feature of the list. **Moving a model** means a new
commit in its URL and the new SHA-256 in the same row (`/api/models/<repo>/revision/main` gives the
commit, `/api/models/<repo>/paths-info/<commit>` the `lfs.oid`, which is the SHA-256).

**Until 2026-10 the set went through the GitHub Actions cache**, and why that ended is worth keeping:
a `download-models` job fetched every row into one entry keyed `gguf-models-<hash of models.csv>`
(6.1 GB, `enableCrossOsArchive`), a three-OS `verify-model-cache` job proved it restorable, and
`restore-models` brought **the whole entry** to every job that needed any model -- one cache entry
cannot be restored in part -- about 20 times and 100 GB per Publish run. One restore failed with
"Egress is over the account limit" (run 40580317285), the repository's caches stood at 11.1 GB of the
10 GB GitHub keeps, and the one cold-cache download in between had to come from Hugging Face
anonymously, from a runner IP whose shared rate limit answered 429. A cache entry per feature would
have needed one fixed `actions/cache` step per entry (a composite action cannot loop over `uses:`)
and kept the egress on GitHub; Hugging Face with a token serves each file on its own and counts the
requests against the account. (The cache design had hardened after run 28805360584, when an entry
re-saved **empty** after an eviction and a `validate-models.bat` whose `exit /b 1` never fired let the
Windows jobs self-skip every model-backed test; the SHA-256 check is the stricter form of that gate.)
The remaining `*_MODEL_NAME` env vars of publish.yml
serve only the consumers outside the llama module's tests (smoke scripts, Android emulator jobs,
langchain4j/agent integration); `check-natives.py` fails when one is not a filename of `models.csv`.
Locally the model tests self-skip when a GGUF is absent (`Assume.assumeTrue`), so a partial local
checkout is fine.


**Never compare two generations across a prompt-cache hit.** A second request with the same (or a
prefix-equal) prompt and `cache_prompt` (the default) takes every prompt token but the last from the
slot and re-evaluates that one token in a batch of its own (upstream `[TAG_PROMPT_LOGITS]`); its
logits are not bit-identical to the batched prompt evaluation of the first request, and a greedy near
tie flips -- measured on the random test model over 4000 prompts: the first token's logprob differs
in 3999 first-vs-second pairs and the text in 4, while the second and the third request (both over
the cached prefix) agree in all 4000 and two requests with `cache_prompt=false` are bit-identical.
`ChatScenarioTest#testChatCompleteTextMatchesChatCompleteContent` was that lottery and lost once on
the MSVC Windows job (run 39702748614, F272). So a test that needs two generations to agree either
sets `withCachePrompt(false)` on both and proves it through the response's `timings` (`cache_n` 0,
`prompt_n` equal to the prompt tokens), or compares the second with the third cached request and
asserts the reuse (`cache_n == prompt_tokens - 1`, `prompt_n == 1`) -- never the first with the second.
A sweep of every test tree the same day rewrote eight more (the ledger's F272 Nachtrag names them);
the helpers to reuse are `MemoryManagementTest#assertCachedPrefixReused`,
`ChatScenarioTest#assertWholePromptEvaluated` and `test_c_api.cpp`'s `without_prompt_cache` /
`expect_whole_prompt_evaluated`.


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
| `src/test/cpp/test_server.cpp` | 208 | Upstream result types: `server_slot_stats` (the `timings` JSON payload; replaced `result_timings` in b10408), `task_params::to_json()` (incl. `dry_sequence_breakers`, `preserved_tokens`, `timings_per_token`), `completion_token_output`, `server_task_result_cmpl_partial` (non-oaicompat + `to_json_oaicompat` + logprobs + `to_json_oaicompat_chat` + `to_json_anthropic` + dispatcher), `server_task_result_cmpl_final` (non-oaicompat + `to_json_oaicompat` + `to_json_oaicompat_chat` + `to_json_oaicompat_chat_stream` + `to_json_anthropic` + `to_json_anthropic_stream` + tool_calls + dispatcher), `server_task_result_embd`, `server_task_result_rerank`, `server_task_result_metrics` (`to_metrics()` = the `/metrics` Prometheus exposition text; its `to_json()` has been unused since b10519 and returns `json{}` = JSON null), `server_task_result_slots` (`to_json()` = the `/slots` array, fed by the b10519 `SERVER_TASK_TYPE_SLOT_GET` task), `server_task_result_slot_save_load`, `server_task_result_slot_erase`, `server_task_result_apply_lora`, `server_task_result_get_lora`, `server_task_result_error`, `format_error_response`, `server_task::need_sampling()`, `server_task::n_tokens()`, `server_schema::eval_llama_cmpl_schema()` (parsing pipeline + grammar routing + error paths + per-request `dry_*` and `sse_ping_interval` field round-trips incl. hard-limit + server-default inheritance), `response_fields` projection |
| `src/test/cpp/test_json_helpers.cpp` | 72 | All functions in `json_helpers.hpp`: `get_result_error_message`, `get_result_error_code` (the HTTP code of a failed task result, from which `jllama.cpp` picks `InvalidRequestException` or `LlamaException`), `results_to_json`, `rerank_results_to_json` (incl. missing/out-of-range `index` rejection and the `tokens_evaluated` pass-through), `parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk`, `server_metrics_to_json`, `route_error_message` |
| `src/test/cpp/test_log_helpers.cpp` | 13 | All functions in `log_helpers.hpp`: `log_level_name`, `format_log_as_json` |
| `src/test/cpp/test_common_log_callback.cpp` | 7 | **The runnable guard for `patches/0014`**: `common_log_set_callback()` on a private `common_log_init()` instance (never `common_log_main()`, so the process-wide logger the other tests print through is untouched) — delivery of level + bare text, no prefix/timestamp even when both are on (what `common_init()` does), clearing stops delivery, a swap drains queued entries to the *previous* sink (the property behind `LlamaModel.setLogger(format, null)` being a synchronous flush), a `--log-file` keeps being written alongside the sink, every `ggml_log_level` passes through unchanged, and the sink is not gated on upstream's `common_log_set_stdo()` console switch (b11612: entries reach the callback with stdout/stderr output off). The Java half (`LlamaLoggerTest`, model-free) proves the JNI trampoline on top of it. |
| `src/test/cpp/test_jni_helpers.cpp` | 77 | All functions in `jni_helpers.hpp` using a zero-filled `JNINativeInterface_` mock (incl. the `utf8_to_jstring_impl` byte-array string path: emoji byte-preservation, truncated-UTF-8 replace-not-throw), and `jni_utf8.hpp` (six: an ASCII message stays on `ThrowNew`, a 4-byte character or an invalid byte reaches the exception constructor as the original bytes through `String(byte[], "UTF-8")`, the same for `new_string_utf8_impl` and for the JNI exception boundary; and seven: an exception already pending is left in place on both paths, as `jni_guard_impl` does). Seven of them pin `jni_guard_impl` — the JNI exception boundary every `Java_*` entry point runs inside — including the `catch (...)` arm that is the only backstop for a non-`std::exception` type, and its two refusals (never `ThrowNew` over a pending Java exception, never with a null class). |
| `src/test/cpp/test_tts_wav.cpp` | 2 | The in-memory WAV writer `pcm_to_wav16_bytes` in `tts_wav.hpp` (WAV header/payload + little-endian clamping) — our own code, not upstream. The Qwen3-TTS pipeline it pairs with (`mtmd_helper::gen_audio`) is entirely upstream-owned (no project-side DSP to unit-test here). The load path is additionally covered by `test_tts_params.cpp` (3 tests over `tts_params.hpp`'s `build_tts_params`, plus 2 pinning the upstream `-1` default it depends on), which pins the CPU-thread resolution whose absence used to crash the JVM on every platform — see the `TODO.md` entry for the mechanism. End-to-end coverage is `TtsIntegrationTest`, which is model-gated. |
| `src/test/cpp/test_tts_params.cpp` | 16 | The **three** builders every hand-assembled `common_params` goes through: `build_tts_params` (`tts_params.hpp`), `build_train_params` (`train_params.hpp`) and the shared `jllama::resolve_cpu_params` (`cpu_params.hpp`). Each builder is guarded separately on purpose — testing the resolver alone does **not** cover its call sites, because `train_engine.cpp` is compiled into `jllama` only, never into `jllama_test`, and `LlamaTrainerIntegrationTest` is gated on `net.ladenthin.llama.train.model`, which no CI job sets. Without these the JVM-abort bug could regress in the trainer on every platform, unseen. Three more pin `speaker_reference_refusal` (`tts_params.hpp`), the rule `engine_synthesize` applies before it touches the model: a pair without a voice encoder (`mtmd_helper_cap::audio_gen_spk_ref` false, b11901) refuses a speaker reference by name, no reference or a pair that clones voices goes on. |
| `src/test/cpp/test_prefetch.cpp` | 5 | **The runnable guard for `patches/0017`** (the x86 prefetch helper). The patch fixes two defects in the four `_mm_prefetch` calls of `ggml_vec_dot_q4_0_q8_0`'s SSSE3-without-AVX branch, and only one of them is a build error. The **type** half is caught by the compiler, but only by a recent one: plain clang on a `*-windows-msvc` target gets MSVC's `char const *` declaration (its own casting macro sits behind `#ifndef _MSC_VER`), and the resulting `-Wincompatible-pointer-types` is an **error by default from clang 22 on** -- measured on the real file at `-msse4.2`: 4 warnings on 20.1.8, 4 errors on 22.1.3 and 23.1.3, and 0/0 on all three once patched (`-msse4.2 -mavx` compiles either way, taking the AVX branch). So `GGML_CPU_ALL_VARIANTS=ON` stops building with plain clang on Windows once the compiler is new enough -- via the **`sse42`** variant, the only one that reaches `#elif defined(__SSSE3__)` (the `x64` baseline has no SSSE3 and takes the scalar branch; everything richer has AVX). Nothing in a test can cover that half. The **arithmetic** half is what this file guards, and it is silent: `&x[ib] + sizeof(block_q4_0)` adds `sizeof()` *elements*, so with the sizes `ggml-common.h` pins by `static_assert` (18 and 34) the prefetch targets +324 and +1156 bytes instead of +18 and +34, and the `+ 2 *` pair +648 instead of +36 -- wrong on **every** platform, including the ones where the call compiles. `ggml_prefetch_at()` casts to `const char *` first, which is what makes the offset byte-wise. **The case that makes the file worth having:** writing the cast around the whole expression, `(const char *)(&x[ib] + sizeof(block_q4_0))`, silences the compiler and keeps the wrong address -- that form builds and fails here. Verified both ways: 5/5 green against the patched header, 5/5 red against a deliberately element-wise one. Includes two internal headers (`ggml-common.h` for the block types, `prefetch.h` for the helper) via the `${llama.cpp_SOURCE_DIR}/ggml/src` and `.../ggml/src/ggml-cpu/arch/x86` include dirs added for it. **Measured, so the upstream report does not overclaim:** the wrong prefetch distance costs no measurable throughput (Ryzen 7 5800H forced onto the `sse42` kernel, 0.6B Q4_0: pp512 101.3 vs 100.9, tg128 37.3 vs 37.0, inside each other's error bars) -- it is a correctness and portability fix, not a performance one. |
| `src/test/cpp/test_kolibri1.cpp` | 3 | **The guard for upstream's Kolibri-1 support** (b11709, #30673; carried as `patches/0016` until the b11760 bump, see `docs/history/dropped-llama-patches.md`). Writes tiny random `kolibri1` GGUFs with the public `gguf` API in the format upstream's converter writes (`conversion/kolibri1.py` over `AfmoeModel`: gating function 2, pre-tokenizer `qwen2`, `expert_weights_norm`, an explicit `output.weight`, the per-layer sliding-window pattern), loads them through the real library on the CPU and compares every logit -- one batch, and token by token through the iSWA KV cache -- with an independent double-precision reference written from Aleph Alpha's vLLM implementation, not from llama.cpp. The layer pattern puts a full-attention (NoPE) layer between sliding ones, the sequence is longer than the window, and the expert biases are large enough that DeepSeek-V3's router would pick other experts (one test asserts that, so the comparison cannot pass vacuously); the second comparison turns `norm_topk_prob` on. A red comparison means upstream computes something else than Aleph Alpha's reference (router, window boundary, positional encoding) and is a finding, not a test to adjust. The two format rows of the community dialects the patch loaded (gating 5 + pre `kolibri1` + tied output; the rejection of gating 1) went with the patch: upstream reads the keys above as required, knows no pre-tokenizer `kolibri1`, and passes an unknown gating value on to `build_moe_ffn` instead of refusing it -- those GGUFs are reconverted, not shimmed (README, "Kolibri-1"). |
| `src/test/cpp/test_model_flags.cpp` | 4 | **The contract between the Java CLI-flag registries and llama.cpp's server argument parser.** CMake reads `ModelFlag.java` + `ModelOption.java` (`cmake/extract-java-wire-names.cmake` → a generated header of `{name, contract}` pairs), and this file asserts every `SERVER_PARSER` name is in `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options`. It exists because **no Java test can catch this class**: `ModelFlagTest`/`ModelParametersExtendedTest` pin the *string mapping* (`hasKey("--mlock")`), never that llama.cpp still accepts the string, so they stay green forever while the flag is dead — and `common_params_parse` treats an unregistered option as a hard error, so the affected builder method makes the model **unloadable**, not merely ineffective. **A grep over `arg.cpp` is not a substitute**: `--grp-attn-n`/`-w` are present there at every pinned tag but `set_examples()`-scoped to `LLAMA_EXAMPLE_COMPLETION`/`PASSKEY`, so the server parser rejects them exactly like a deleted flag — only the real option table sees that. `--vocab-only` is the one exemption, and it declares itself `CliContract.PROJECT_PSEUDO` on its own constant rather than appearing in a list inside this file; the test asserts such a name is **still unknown** to the parser (an exemption upstream later registers would be hiding a real check) and that the exempt set is non-empty. |
| `src/test/cpp/test_wire_contracts.cpp` | 6 | **The same contract for the two quieter surfaces.** `RequestField` against `server_schema::make_llama_cmpl_schema(...)` (5 tests) and `TrainingField` against `jllama_train::config_keys()` (1 test). Both receivers *silently ignore* an unknown key — the schema skips it, `train_engine.cpp` reads with `j.value(key, default)` and falls back — so a dead field produces no error anywhere and every string-mapping test keeps passing. `OAI_LAYER`-declared keys (consumed by `oaicompat_*_params_parse` before the schema) are exempt from the schema check, and are checked **both** ways: still unknown to the schema (the inverted check), and read by at least one upstream reader-shaped site (the configure-time sweep — this is what caught `chat_template`, a key a public builder wrote and nothing read). See [`docs/history/parameter-wire-surface.md`](docs/history/parameter-wire-surface.md). |
| `src/test/cpp/test_rpc.cpp` | 28 | **The runnable guard for `patches/0015` and for `rpc_support.hpp`.** The device-selection rules (pure, literal inputs: `--rpc` endpoints accumulate across repeated options, an explicit `--device`/`-dev` is never overridden, stale RPC devices are replaced by the local GPUs chosen the way llama.cpp's default does — duplicates by device id dropped, iGPUs only without a discrete GPU, `none` on a CPU-only host). Then the **real** ggml-rpc client and server over loopback, no model: a `mul_mat` graph computed on the RPC backend equals the local CPU result; `ggml_backend_rpc_stop_server()` ends the server, frees the port and disconnects a client that is still connected; an unreachable or malformed endpoint is `nullptr`/`std::invalid_argument` instead of an abort; a registered server that went away is re-checked on the next registration; its device reports 0/0 memory instead of aborting; and a stale server is kept out of a later load that did not ask for it. The loopback server picks a free port from a range so parallel jobs do not collide, and serves the **CPU** by name — a served GPU that lacks an op aborts the server (see the RPC section). Also: server devices chosen by name (case-insensitive, deduplicated), an unknown name listing the available devices, and a registered RPC device refused; the mmproj device pinned the way clip would choose it minus the stale server; and `exclude_stale_devices()`, the params-level guard for TTS and the trainer. |
| `src/test/cpp/test_rpc_comm.cpp` | 6 | **`patches/0015`'s COMM_INIT and stop contract over real TCP, protocol v8.** Each scenario runs the C API's RPC server in a fresh subprocess with a deadline, so a regression fails instead of hanging the runner: a loopback server opens no wildcard peer listener and a network bind stays an explicit opt-in; a stop with no peer closes both listeners and lets the port be bound again; a stop interrupts a stalled capability handshake of rank 0 and of rank 1; and a successful peer setup still works and stops. No model, GPU or external network. |
| `src/test/cpp/test_c_api.cpp` | 55 | **The C API (`jllama_api.h`, layer B) without a JVM.** Writes a tiny random two-layer llama GGUF with the GPT-2 byte tokenizer (the public `gguf` API, like `test_kolibri1.cpp`), loads it through `jllama_model_load` and drives the shipped contract: version and build info, null-argument refusal with a message, a missing model failing to load, the load-progress callback (reported, and a `false` return aborts the load), a chat completion that is an OpenAI `chat.completion` object, an invalid request answered as the caller's fault (`JLLAMA_STATUS_INVALID_REQUEST`), streaming until `stop` with the task released afterwards ("Task not found" on the next receive), a released task gone, and the log callback seeing the server's lines with a null set flushing synchronously. Hermetic, so it runs in every `C++ Tests` job without `fetch-models`; the output of a random model is noise, the contract is what is asserted. Two things the fixture does on purpose: the output rows of every non-ASCII and special token are scaled to almost nothing, because a byte above 0x7F is an incomplete UTF-8 piece the server holds back rather than streams (a random model that emits only those sends no partial chunk at all), and the task id is checked `>= 0` -- the server numbers its tasks from zero. Then every other `LlamaModel` operation in turn (the meta, raw and OAI completions, raw streaming, the streamed OAI completion (`text_completion` chunks with one id, probabilities, `usage` on the last), the template, tokenize/detokenize, metrics, slots, LoRA, configure, the refusals of a model loaded without `--embedding`/`--reranking`/FIM tokens, the grammar conversion, a Q8_0 quantisation that loads again, several prompts in one request and several completions of one (`"n"` > 1) refused on the blocking, raw and streamed paths while a token array stays one prompt, and every OpenAI answer and chunk carrying a `chatcmpl-` id and the model name, one id per stream), then the other engines: the embedded server in router mode and attached to the model (which still answers after the server stopped), the RPC server over loopback, text to speech refusing missing models, and a fine-tuning run over the random F32 model that writes a GGUF -- which is why `tts_engine.cpp`, `train_engine.cpp` and `native_server_engine.cpp` are in `jllama_test`'s source list. Five model-free cases (`CApiStatic`) run before any load. `CApiSlotFiles` loads the random model with `--slot-save-path`: a saved slot file lands under that directory by its name and restores, and a name with a directory part, `..`, an absolute path or nothing is `JLLAMA_STATUS_INVALID_REQUEST` with nothing written (red before F251: the name was the path, relative to the working directory); a model without the directory refuses save and restore with `JLLAMA_STATUS_ERROR`. Four `CApiRpc` cases pin `jllama_rpc_server_stop()` around the bind and the next serve: a stop issued before ggml binds the socket -- from the serving thread through the test seam `jllama::rpc::set_serve_hook_for_testing()` (`rpc_support.hpp`, not exported), or from another thread while the serve is held in that seam, where the stop must not return before the serve has -- ends the serve instead of being lost (patches/0015 clears ggml's stop flag at the bind; red before F241), a stop while nothing serves does not stop the next serve, and a stop does not reach a serve started while it waits for the one it is for (the stop seam `set_stop_hook_for_testing()` starts the next serve once the stop is recorded; red before F271: the stop waited 30 s for the newcomer and ended it too). Two `CApiFim` cases load the same random model with FIM tokens (`write_tiny_llama_gguf(prefix, true)`): a raw completion with `input_prefix`/`input_suffix`, blocking and streamed, evaluates exactly the prompt `jllama_infill` builds from the same fields, and one of the two fields alone is refused naming the other (red before F246: the completion path completed the empty prompt and dropped both). Two more `CApi` cases pin `messages` on the raw paths: blocking and streamed, the request evaluates exactly the tokens of a completion of the prompt `jllama_apply_template` renders from them (an empty `prompt` beside them counts as none), and a non-empty `prompt` or the infill fields beside them are refused (red before F247: the field was ignored). Two more: the chat fields the raw paths cannot apply beside `messages` (`response_format`, `tools`, `tool_choice`, `parallel_tool_calls`, `top_logprobs`, `thinking_budget_tokens`) are refused by name on both raw paths, and `CApiTemplateStops` loads the random model with an end-of-generation byte of its own (`write_tiny_llama_gguf(prefix, false, 4)`) and a template upstream adds the stop string `</assistant>` to, which a grammar forces: the raw paths stop there as the chat path does, also beside a `stop` of the request (red before F248: dropped, `stop_type` `eos` and the string in `content`). And `CApiReasoningBudget` loads it with a template that opens a `<think>` block in its generation prompt: a raw completion with `messages` and `reasoning_budget_tokens` 1 completes the forced `</think>` and then the chat path's answer, blocking and streamed, while an output-format grammar beside it still runs (the generation prompt the budget is primed with is left out there); a template without thinking tags leaves the request unchanged (red before F249: the tags were dropped and the budget did nothing). A third case loads the random model with `<think>`, `</think>` and `<tool>` as control tokens (`write_tiny_llama_gguf(prefix, false, 4, special)`): the forced end tag reads as text in the raw content, blocking and streamed, and a request's own `preserved_tokens` stays in force beside the template's (red before F250: the session's preserved tokens were not taken over, and the special end tag was an empty piece). A fourth, without `messages`, names the tags in the request with a logit bias that makes the model write the start tag: a budget of zero forces the end tag after each one, and a `reasoning_budget_message` before it (red before F252: without the message the budget forced nothing). Four cases pin the raw multimodal prompt: `CApiVision` loads the random model with a random llava-style vision projector the shared fixture writes too (`write_tiny_mmproj_gguf()`, `--mmproj`) and finds the server's `media_marker` in the model meta; a raw completion whose `prompt` is upstream's `{"prompt_string": "...<marker>...", "multimodal_data": ["<base64>"]}` evaluates the text tokens plus the image's four patch embeddings on the blocking, streamed and OpenAI paths (red before F258: refused as if the model had no projector), while markers that do not match the images, data that is not base64, no data and an image at the very end (an abort of the process in `update_slots`) are refused, and the same object on the model without a projector is the caller's fault. `CApiVocabOnly` loads the random model with `--vocab-only`: the meta, tokenize and detokenize calls answer, and a parameterized case per other C function on a model (22 instances: every completion, chat and stream path, the template, embeddings, rerank, infill, metrics, slots, LoRA, configure, systemone, the attached server) is `JLLAMA_STATUS_ERROR` with "`jllama_<function>` is not available in vocab-only mode" (red before F266: a `SIGSEGV` in `llama_vocab::token_bos()` or a wait that never ended). Two pin the `capabilities` object of the model meta (b11901, from upstream's server meta): the random model chats and neither generates nor streams audio, with the vision projector loaded as well (`CApiVision`), while a vocab-only load claims none; and `CApiTts` refuses `jllama_tts_supports_speaker_reference()` with a null handle or out-parameter, and `jllama_tts_synthesize_voice()` (6.1.0) with a null handle or text. |
| `src/test/cpp/test_native_server_shutdown.cpp` | 3 | **The runnable guard for the shutdown-handler guard in `patches/0006`/`0007`.** Runs the real `llama_server()` in router mode over an empty `--models-dir` on an ephemeral loopback port (no model, no worker) and stops it the way `native_server.cpp` does. Pins a clean stop (exit code 0), that `llama_server_request_shutdown()` **after** the server returned is a no-op (the deterministic form of the CI `SIGSEGV` — it crashed every run before the fix), and that requests hammered from another thread while the server tears down are safe (5 rounds). Compiled on non-Android only, like `server.cpp` itself. |
| `src/test/cpp/test_gallocr_leafs.cpp` | 1 | **The runnable guard for `patches/0020`.** Two graphs with the same two nodes in the same order (`n1 = add(x, x)`, `n2 = mul(w, n1)`) and only the leafs moved: one expands `n1` before `n2` and meets the input first, the other expands only `n2`, whose first source is the weight -- the shape of the llama.cpp case, where every node and node source still matches its slot, so the old check passed it. Reserved with the first, the second must get an allocation for the input inside the allocator's buffer and compute the right product on it. No model, every platform. **Verified both ways:** with the fix removed from the source tree the test dies with SIGSEGV (rc 139); a first version that swapped the operands of one `add` passed without the fix, because a swapped *node source* was already caught -- a guard has to move only the leafs. |
| `src/test/cpp/test_lora_cache.cpp` | 5 | **The runnable guard for `patches/0021`.** Writes a random llama model (the shared `tiny_llama_gguf.hpp`, also used by `test_c_api.cpp`) and a random LoRA adapter on K, V and the FFN output of every layer, loads them with one slot and `--cache-ram 0`, and answers the same prompt greedily. After `jllama_set_lora_adapters` (scale 1 -> 0 and back) and after a request with its own adapters, the warm slot must answer exactly like an emptied one. One case guards the others against passing vacuously: the adapter changes the answer, and a repeated request reuses the cache (`timings.cache_n > 0`; not `tokens_cached`, which counts every token the slot holds and is positive after any request). After each adapter change the next request must take nothing from the cache (`timings.cache_n == 0`), so those three cases are red without the patch whether or not the random model happens to answer differently. **Verified both ways:** without the patch 3 of 4 red (the warm slot answered from the old scales), with it 4 of 4 green in three runs; no model, every platform. A second fixture (`LoraCacheFreshSlot`: two slots, `--slot-save-path`) saves slot 0 (by its file name inside that directory) after a request and restores it into the never-used slot 1, whose request must take all of the prompt but at most two tokens from it and answer like slot 0 -- red before the `init()` hunk (`Expected: (prompt_n) <= (2), actual: 77 vs 2`), green with it (F240). |
| `src/test/cpp/test_moe_repack.cpp` | 2 | **The runnable guard for `patches/0022`.** `MUL_MAT_ID` over weights in the `CPU_REPACK` buffer -- MXFP4, Q4_0, IQ4_NL and Q4_K, random, quantized with `ggml_quantize_chunk` -- against ggml's generic `MUL_MAT_ID` over the same bytes in plain memory (independent code, the same activation quantization), for both settings of the switch on 1, 3 and 4 threads (3 does not divide the 9 column blocks, 1 leaves a unit past the last column). Routings: experts with 0, 1, 3, 4, 15 and 17 tokens (the gemv rest alone, one gemm group, a full 16-row group plus a rest), the same with one slot of -1 (its row stays zero), and a single token; each with one input row per token (gate/up) and one per slot (down): 24 instances. A type the running CPU does not repack is skipped, not passed -- which is why `test-cpp-linux-aarch64` builds with `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`: on plain ARMv8 nothing is repacked and all 24 cases skipped (they still do in the Windows arm64 and s390x test jobs). The second test resolves the switch through the CPU backend's `get_proc_address`, as libjllama does, and round-trips it, so a bump that drops the patch reds this file. |

**Current total: 681 tests** (`TEST`/`TEST_F`/`TEST_P` definitions; ctest counts 725 cases, because
the parameterized `CApiVocabOnly` case expands to 22 instances and `MoeRepack` to 24).

#### Upstream source location (in CMake build tree)

llama.cpp is fetched via CMake FetchContent, pinned to `JLLAMA_GIT_TAG b11968`.

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
     `test_prefetch.cpp` for `0017`. Give every patch that introduces a callable a
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

- **Java 8+** runtime required. Built with JDK 25 targeting bytecode 1.8 for broad compatibility.
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
  classpath where the checker needs it, and excluded from consumers' graph (and from a consumer's
  own `jar-with-dependencies`, which takes scope `runtime`), so no checker-qual class of any version
  ships.
  `<optional>true</optional>` would not have been enough on its own — that descriptor filters on
  scope only. Safe because no source imports `org.checkerframework` — but the **class files do
  reference it**: the Nullness Checker writes its qualifiers (`@NonNull`, `@UnknownKeyFor`,
  `@Initialized`, `@Pure`, `@SideEffectFree`, …) into the classes it checks, 133 of the core's 146
  (measured on the 6.0.0-SNAPSHOT jar). Harmless where checker-qual is absent: a JVM or ART that
  reflects over an annotated element skips an annotation type it cannot resolve. R8 does not skip it —
  it reports the missing class — hence `-dontwarn org.checkerframework.**` in the AAR's consumer rules.

**The gate: the `java/verify-bytecode` action of [bernardladenthin/actions](https://github.com/bernardladenthin/actions)** (its `verify-bytecode-version.sh`).
It opens every `.class` in every jar it is given and fails on any whose class-file major version
exceeds `max-major`:

```yaml
- uses: bernardladenthin/actions/java/verify-bytecode@<commit> # v1.x.y
  with:
    max-major: "52"
    paths: |          # jars or directories, one per line
      llama/target
    allow: |          # optional: '<jar-basename>:<entry-path>' globs to tolerate
```

Paths may be jars or directories (searched recursively for `*.jar`), so one invocation covers a whole
artifact set — here the classes jar and all 23 natives jars, then each smoke set. `module-info.class`
and `META-INF/versions/**` are skipped unconditionally: a classpath JVM never loads either, which is
why a `release 9` `module-info` is fine. `allow` takes globs matched against
`<jar-basename>:<entry-path>` for anything else that must be tolerated. Exit codes: 0 clean,
1 violations, **2 nothing to scan** (an empty input is a failure, never a pass — the first version of
this check reported a clean pass over a directory a failed build had left empty).

It runs twice: in the `package` job over `llama/target` (every jar, as early as they exist), and
again in the two Linux rows of `smoke-natives` over the downloaded smoke set — the jars plus the
runtime dependencies a consumer resolves with them, i.e. the classpath users actually get.

**Surefire AND PIT both exclude `org.slf4j:slf4j-simple` from the test classpath**
(`classpathDependencyExcludes` in each). Runtime scope is on the test classpath too, and LogCaptor
(test scope) requires logback specifically — with both providers present SLF4J's `ServiceLoader`
picks one arbitrarily and the LogCaptor assertions fail. The exclusions leave logback the sole
provider in tests and do not touch the artifact: the jar's pom still names slf4j-simple at runtime scope.

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

## Repository layout — Maven reactor (`llama/` + `llama-langchain4j/` + `llama-kotlin/` + `llama-ffm/` + `llama-platform/`) + the `llama-android/` Gradle build

The repo root is a thin **aggregator/parent POM** (`net.ladenthin:llama-parent`,
`packaging=pom`) with five modules:

- **`llama/`** — the native JNI core (`net.ladenthin:llama`). *All the core sources and build
  files live here now:* `llama/src/`, `llama/CMakeLists.txt`, `llama/cmake/`, `llama/patches/`,
  `llama/pom.xml`, `llama/spotbugs-exclude.xml`, `llama/lombok.config`, `llama/.clang-format`.
  Its published coordinates are unchanged (`net.ladenthin:llama`), so consumers are unaffected.
- **`llama-langchain4j/`** — the LangChain4j adapters (see below).
- **`llama-kotlin/`** — the Kotlin coroutines façade (see "Android AAR + Kotlin façade" below).
- **`llama-ffm/`** — the C API through the Java 22+ FFM API, no JNI (see "Java 22+ without JNI" below).
- **`llama-platform/`** — a pom-packaging dependency list: the classes jar plus the CPU natives jars
  of every desktop platform (see "Natives jars" above).

All modules inherit the single `<version>` from the parent, so they **ship in lockstep by
construction** (no CI guard needed). The parent also holds the shared `release` profile (GPG +
Central Publishing) and a `<pluginManagement>` with every plugin version more than one module uses
(compiler, jar, resources, surefire, source, javadoc, gpg, central-publishing -- a module names such a
plugin without a version; without it a module that pins nothing builds with the default of whatever
Maven runs it, which differed between CI and a local build; and, for the same reason, clean, install,
deploy and dependency, which no module configures but the lifecycle and CI's `mvn dependency:...` run), so one reactor `mvn -P release,natives deploy` signs and publishes all six
Maven artifacts (`llama-parent` pom, `llama` with its natives jars, `llama-langchain4j`,
`llama-kotlin`, `llama-ffm`, `llama-platform` pom) at the same version.

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
versioning** here), so a version change must be applied to **all six poms in lockstep**:

- `pom.xml` (root) — `<version>`
- `llama/pom.xml` — `<parent><version>`
- `llama-langchain4j/pom.xml` — `<parent><version>`
- `llama-kotlin/pom.xml` — `<parent><version>`
- `llama-ffm/pom.xml` — `<parent><version>`
- `llama-platform/pom.xml` — `<parent><version>`

(`llama-android/` needs **no** edit — its Gradle build reads the root pom's version at
configure time.)

(`versions:set` reaches all six.) The safe way is `mvn -q versions:set -DnewVersion=X.Y.Z -DgenerateBackupPoms=false` from the repo
root (it updates the parent and every child `<parent>` reference at once). Changing only the root
`<version>` leaves the children pointing at a non-existent parent and **fails the reactor build**
(`Could not find artifact net.ladenthin:llama-parent:pom:X.Y.Z`).

`versions:set` only rewrites the **poms**. Hardcoded release-version dependency snippets in the
README files and examples listed below must be bumped **manually and in the same commit** — miss one
and the published docs point consumers at the previous release. (The `llama-langchain4j/README.md`
snippet was exactly the one forgotten on the `5.0.4 → 5.0.5` bump; it is listed here so it is not
missed again.)

- **`README.md`** (root) — the install snippet, the two classifier-example snippets (default + the
  `<classifier>` template), and the `llama-langchain4j` snippet. The Maven Central **badge**
  auto-pulls the latest released version, so leave it. The **`-SNAPSHOT` line** in the "Snapshot
  builds" section documents the snapshot channel — set it to the *next* dev version, not the release.
  Update the runnable server, agent and native-download examples as well as dependency snippets.
- **`examples/jbang/Chat.java`** — the version in its `//DEPS` lines; `check-natives.py` holds it to
  the README install snippet's version, so forgetting it fails `code-style`.
- **`llama-langchain4j/README.md`** — its own `<dependency>` snippet.
- **`llama-atmosphere-agent/pom.xml`** — its own `<version>`, which must equal the reactor's
  (standalone project outside the reactor, so `versions:set` skips it; `check-natives.py` fails until
  they agree), plus the JBang coordinates in the root README's "Local coding agent" section and the
  project's own README, and `<llama.version>` in `examples/server/pom.xml` (held to the README's
  install snippet by `check-natives.py`).
- **`llama-android/README.md`** and **`llama-kotlin/README.md`** — their Gradle dependency
  snippets, plus the `llama-android`/`llama-kotlin` snippets in the root README's
  "Importing in Android" section.
- **`llama-ffm/README.md`** — its `<dependency>` snippets (the module and `llama-platform`).

(If single-source ergonomics are wanted, the Maven
CI-friendly `${revision}` property + `flatten-maven-plugin` would let a bump touch only the root —
that plugin is not configured today, so do not rely on "root only".)

## Java 22+ without JNI (`llama-ffm` reactor module)

`llama-ffm/` is the C API ("The C API: `jllama_api.h`") called through the Java 22+ Foreign Function
and Memory API: `net.ladenthin:llama-ffm`, package `net.ladenthin.llama.ffm`, **Java floor 22**
(`maven.compiler.release`; the FFM API is final there, JEP 454) while the core stays Java 8. It is
the "Java 22+ over the C API" item of the delivery roadmap (TODO.md item 10), decided 2026-10-15.

- **One loading path.** `JllamaNative`'s static initializer initializes `LlamaModel`, whose static
  block runs the core's `LlamaLoader.initialize()` -- the same natives jars, `lib.path` /
  `java.library.path`, module jars and build check as `LlamaModel` -- and resolves every `jllama_*`
  function with `SymbolLookup.loaderLookup()`, which sees what `System.load` put into the class
  loader. Through `LlamaModel` and not `LlamaLoader` on purpose: the core's `module-info.java` does
  not export `net.ladenthin.llama.loader` (javadoc in module mode refused the import, and a
  module-path consumer would hit the same wall), and the module surface of the core is not this
  module's to widen. There is no `System.load` and no jar handling in this module; a library of
  another version fails with an `UnsatisfiedLinkError` naming the missing symbol.
- **Hand-written bindings, no jextract** (not in the JDK, not in CI): one `MethodHandle` per
  declaration of `jllama_api.h` in `JllamaNative`, plus the marshalling helpers (`cString`,
  `cStrings` for an argv, `takeString` / `takeFloats` / `takeTokens` / `takeBytes` / `takeStrings` for
  the out-parameters, each releasing with the typed `jllama_*_free`). `size_t` is `JAVA_LONG`: every
  platform this project ships natives for is 64-bit. The callbacks are upcall stubs: the load-progress
  one in the confined arena of the load call (it runs on the loading thread), the log one in a shared
  arena `Jllama` keeps until the next `setLogCallback` swap -- the C side drains the previous sink
  before `jllama_set_log_callback` returns, so the old stub is closed after the call. An exception
  inside an upcall would end the JVM, so both bridges catch everything (the progress bridge aborts
  the load instead).
- **The core's exceptions.** `JllamaNative.check` maps `JLLAMA_STATUS_INVALID_REQUEST` to
  `InvalidRequestException`, `JLLAMA_STATUS_INVALID_ARGUMENT` to `IllegalArgumentException`, everything
  else to `LlamaException`, each with `jllama_last_error()` as the message -- the same verdicts the
  JNI shells give.
- **Public API**: `FfmLlamaModel` (the `LlamaModel` operations, JSON text in and out, `FfmTaskStream`
  for the streams: an `Iterator<String>` that releases the task at `stop` or on `close()`),
  `FfmNativeServer` (classic, attached, `setWorkerCommand`), `FfmRpcServer` (static; `serve` blocks
  like `RpcServer`'s thread body), `FfmTextToSpeech`, `Jllama` (version, build info, the log callback,
  grammar conversion, quantise, finetune), `SlotOp`. `ModelParameters.toArray()` is the argv
  (`argv[0]` is "", which the C API accepts as the program name).
- **No `module-info.java`**, on purpose: javadoc stays in classpath mode
  (`../workspace/policies/jpms-module-descriptor.md`), the manifest carries
  `Automatic-Module-Name: net.ladenthin.llama.ffm`. The compiler runs `-Xlint:all -Werror` minus
  `restricted` (every FFM call is one) and `classfile` (the core's checker-qual annotations are
  provided scope there and absent here). Surefire passes `--enable-native-access=ALL-UNNAMED`.
- **Toolchain.** The reactor uses JDK 25 from the shared `.java-version`. The core still targets
  Java 8, this optional module targets Java 22, and test release settings remain unchanged.
  CI needs one JDK installation. The `toolchain-jdk22` profile remains a local fallback for
  Maven running below JDK 22, selecting a registered JDK 22+ from `~/.m2/toolchains.xml`.
- **CI**: `test-java-llama-ffm` (`needs: [crosscompile-linux-x86_64]`) downloads
  the Linux x86-64 natives, fetches the `draft` and `tts` models, builds the core, runs `mvn -f llama-ffm/pom.xml
  verify` with `lib.path` and the draft model, and fails when `FfmLlamaModelIntegrationTest`,
  `FfmTextToSpeechIntegrationTest` or `JllamaNativeTest` skipped itself (a self-skip would hide a broken
  binding). It gates both publish jobs; the release gate check
  demands it. The bytecode gate (`java/verify-bytecode`) keeps scanning `llama/target` only: the module's classes
  are major 66 and are not part of the smoke sets.
- **Tests**: `JllamaNativeTest` (model-free: `apiVersion` equals the pom version surefire passes,
  `buildInfo` starts with `LlamaCppVersion`, a grammar conversion, a missing model's message, the log
  callback reached from the worker and flushed on `null`, an `Error` thrown by the log callback losing
  that line and nothing else, RPC device names, and three `takeStrings` cases: a populated array read
  in order and freed with its count, an empty non-null array still freed, a null array not freed,
  and `jllama_tts_supports_speaker_reference` refusing a null handle through its bool slot),
  `FfmNativeServerTest` and `FfmTextToSpeechTest` (no native library, fake `Backend`s: `close()` waits
  for an `isRunning()` / `synthesize` call inside the native call -- the lifecycle lock, a read/write
  lock for TTS --, a call after `close()` does not touch the handle -- `supportsSpeakerReference()`
  included --, a second `close()` has no effect), `FfmRpcServerTest` (loopback, CPU: a stop while nothing serves does not stop the next serve,
  a stop ends it) and `FfmLlamaModelIntegrationTest` (the draft model, or
  `-Dnet.ladenthin.llama.ffm.model`: progress callback, meta, chat completion, streaming to `stop` and
  an early close, raw completion and its stream, the OAI completion stream ending with `usage`,
  tokenize/detokenize round trip, template, metrics, LoRA and configure, slot erase, the save refused
  without `--slot-save-path` and a slot file named inside that directory (a path refused), the embedding refusal, invalid requests, the attached server answering `/health` and the model
  outliving it, an `Error` thrown by the progress callback aborting the load instead of the JVM, a
  closed model refusing calls, the meta's `capabilities`) and `FfmTextToSpeechIntegrationTest` (CI's
  Qwen3-TTS pair, or `-Dnet.ladenthin.llama.ffm.tts.model` / `.tts.mmproj`: the speaker-reference
  capability and a synthesis with a voice cloned from the core's `sample.wav`; the CI job fails when
  it skipped itself, like the other two). Locally the C++ test fixture's twin, a
  tiny random llama GGUF written with `gguf-py`, serves as the model where no download is possible.
- **`maven-versions.poms`** of `.github/actions-config.json` names `llama-ffm/pom.xml` (the siblings
  read this repository's POM list from there).

## Code signing of the Windows DLLs (SSL.com eSigner, one switch; inert until the certificate exists)

Decided 2026-10-15 (TODO.md, delivery roadmap, item 11): Authenticode over every DLL of every Windows
natives jar with an **SSL.com IV code-signing certificate held in eSigner** (SSL.com's cloud HSM),
Windows only (the macOS Developer ID was declined; the dylib stays ad-hoc signed by the linker, with
the `.asc` and the Sigstore attestation as its provenance). **Azure Trusted Signing was the first
choice and is a dead end for the owner**: Microsoft's identity validation does not take an individual
in Germany (country not supported), the organisation form needs a registered business, and the
account created for it costs 9.99 USD/month for nothing -- the first form of this section and of
`sign-windows-natives.sh` (OIDC to Azure, `TRUSTEDSIGNING`) was replaced on the same day. SignPath
Foundation (free for open source) was declined because the publisher on the signature would read
"SignPath Foundation", not the owner. The code side is in place and does nothing until the
certificate and the secrets exist.

- **One switch, `sign_windows`**, a `workflow_dispatch` input next to `publish_to_central` and
  `use_cache`, turned into the workflow-level `SIGN_WINDOWS` the way `USE_CACHE` is. Snapshot and
  release alike: the pipeline is the same at that point, a release is both switches on, and a push
  or a pull request never signs (nothing publishes from those either). `publish-release` warns
  (`::warning::`) when it deploys without the switch -- deliberately no failure, the switch is the
  decision.
- **Where**: a job of its own, `sign-windows-natives`, after the ten Windows build jobs and
  `test-java-windows-x86_64`, before `package`. It downloads the `natives-*-windows-*` artifacts
  unmerged (one `Windows/<arch>/<backend>/` directory each, as uploaded), signs, verifies and
  **re-uploads each under its own name with `overwrite: true`**, so `package` and both publish jobs
  merge signed DLLs, the Windows `smoke-natives` rows run the signed sets and the jars carry them
  ("no asset ships that CI has not run"). Not steps in `package`: the secrets live in the
  `maven-central` environment (the GPG key's, same trust class, same rules), and `package` runs on
  every pull request, where that environment's rules must not apply (it admits only `main` and
  `v*` tags, which is why `verify-signing-key` on the same environment runs only on a publishing
  dispatch). The job waits for the Windows Java test
  job because it reads `natives-cpu-windows-x86-64`: `overwrite` deletes the artifact for a moment,
  and a download in that moment fails. **The job runs on every run and is never skipped**: without
  the switch its steps are gated off (`env.SIGN_WINDOWS`) and it takes the unprotected environment
  `unsigned` -- the environment name is an expression, `maven-central` only when signing, so that
  environment's rules never apply to a pull request -- and `package` lists it in `needs` with the
  default condition: run and failed (wrong secrets, the environment refusing the branch) means
  `package` is skipped, because the signature was asked for. The first form had the job itself
  `if:`-skipped without the switch and `package` tolerating the skip (`!cancelled() &&
  !contains(needs.*.result, 'failure') && ...`), and that lost every job behind `package`: GitHub
  applies a skip "to all jobs in the dependency chain from the point of failure or skip onwards",
  past any `if:` in between, so run 38870889665 (attempt 2, the first run to reach `package` after
  the job existed, a dispatch on main) ended "success" with `package` green and the smoke sets, the
  report and both publish jobs skipped -- and so did every run since, until this was corrected on
  2026-10-22 (the fix had sat on the unmerged branch `claude/jolly-brahmagupta-0ydzby` since
  2026-10-16). A job that other jobs depend on must run or fail, never skip.
- **How**: `.github/sign-windows-natives.sh` -- **jsign** (`net.jsign:jsign:7.6` from Maven Central,
  pinned by version and SHA-256; Java, runs on the Linux runner, no signtool) with
  `--storetype ESIGNER --keystore <endpoint> --storepass "<username>|<password>" --alias
  <credential id> --keypass <TOTP secret>` (read from jsign's `KeyStoreType`: the store password must
  carry the `|`, the key password is stored as the TOTP secret and jsign generates each one-time
  code itself), SHA-256, an RFC 3161 timestamp from `ts.ssl.com`, every `Windows/*/*/*.dll` of the
  tree in one call; then `signed=no` -> `signed=yes` in each directory's `jllama-build.txt` (CMake
  writes `no`; the merge step requires the key with either value). `JSIGN_JAR` names a local jar
  instead of the Maven fetch (measured with a real DLL and dummy credentials: the options are
  accepted and the run fails at SSL.com's authentication, nothing flipped). The two passwords reach
  jsign through its `env:` prefix (`--storepass env:JSIGN_STOREPASS`, `--keypass env:ES_TOTP_SECRET`;
  both of `KeyStoreBuilder`'s password getters resolve it), so no secret is a process argument. No
  `set -x`, no echoed argument.
- **Secrets and variables**: `ES_USERNAME`, `ES_PASSWORD`, `CREDENTIAL_ID`, `ES_TOTP_SECRET` -- the
  names of SSL.com's own GitHub action, so their documentation maps 1:1 -- as secrets of the
  `maven-central` environment; optionally the repository variable `ES_ENDPOINT` (empty =
  `https://cs.ssl.com`; `https://cs-try.ssl.com` is SSL.com's sandbox, whose test certificate
  Windows does not trust, for a dry run of the pipeline). **The TOTP secret lets whoever holds it
  sign in the owner's name**, which is why it reaches exactly one job, only on the explicit switch,
  only under that environment's rules.
- **The gate on the result**: `verify-native-deps.py` carries `pe_signed()` (the attribute certificate
  table, `IMAGE_DIRECTORY_ENTRY_SECURITY`, presence only -- the chain is Windows' and jsign's business)
  and `signature_violations()`: a Windows directory whose `jllama-build.txt` says `signed=yes` must hold
  only signed DLLs, and with `--require-signed` a directory saying `signed=no` is a violation. The
  sign job runs it with `--require-signed` over the signed artifacts before re-uploading, and
  `package` and both publish jobs run it over the merged tree with the flag whenever `SIGN_WINDOWS`
  is on -- so a signed run cannot let an unsigned DLL into a jar, and a run without the switch is
  not blocked. Unit tests in `test_nativedeps.py` (`SignatureTest`, a synthetic PE with and without
  the security directory). `check-natives.py` holds the job itself (`natives.check_signing`): one
  `overwrite: true` upload per Windows row of `natives.csv`, the job waits for every job that
  uploads one of them, and it carries no job-level `if:` (the chain rule above; a step-level gate
  is what it uses).
- **What the owner does, in order** (TODO.md, delivery roadmap, item 11 carries the
  details and the prices): delete the Azure Trusted Signing account; SSL.com's IV validation; the IV
  code-signing certificate enrolled in eSigner; the TOTP secret and the credential id from the
  eSigner settings; the four secrets into the `maven-central` environment; a dispatch with
  `sign_windows` and without `publish_to_central`; `Get-AuthenticodeSignature` (or `signtool verify
  /pa /v`) on a `jllama.dll` unpacked from the run's smoke set must name the owner as the publisher.
  eSigner meters per signature: a signed run signs **35 DLLs** (20 in `cpu-windows-x86-64`, 7 in
  `cpu-windows-aarch64`, one in each of the 8 Windows module jars), so the plan's quota is read
  against that before the first run.
- **Not covered, on purpose**: the jars (`jarsigner` only on a consumer's request -- a fat jar breaks
  it), Linux and Android (no OS mechanism; `.asc` + attestation), macOS (declined).

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
   (chat/streaming/embedding/scoring/tool adapters) with the four models it fetches
   (`features: reasoning embedding rerank tools`) and the `natives-cpu-linux-x86-64` native
   artifact: it `needs: [crosscompile-linux-x86_64]`, installs parent+core (classes), points the test JVM at the
   downloaded library with `-Dnet.ladenthin.llama.lib.path`, and passes the chat (`REASONING_MODEL_NAME`), nomic-embedding,
   jina-reranker and tool model paths via the module's
   `-Dnet.ladenthin.llama.langchain4j.{embedding,rerank}.model` / `net.ladenthin.llama.model.path`
   properties.

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

- **`net.ladenthin:llama-android`** / **`llama-android-opencl`** — AARs (`<packaging>aar</packaging>`).
  The CPU AAR carries the core classes + every `.so` of the CI-built modular natives directory under
  `jni/<abi>/` (`libjllama.so`, `libggml.so`, `libggml-base.so`, `libggml-rpc.so` and the
  `libggml-cpu-<variant>.so` modules; see "Modular natives" for the soname loading) and is
  **multi-ABI** (`arm64-v8a` devices + `x86_64` emulators/Chromebooks, built by the
  `crosscompile-android-x86_64` dockcross job, which also ships as the `cpu-android-x86-64` natives
  jar). The OpenCL AAR holds **only** `jni/arm64-v8a/libggml-opencl.so` (Adreno is Qualcomm ARM
  hardware) and its POM depends on `llama-android` of the same version (type `aar`), so AGP merges the
  two `jni/` trees and `JNI_OnLoad` finds the module by name -- additive, like the module jars on the
  desktop. Both carry a `minSdkVersion 28` manifest (AGP enforces the floor on consumers), and
  consumer R8/ProGuard rules (`consumer-proguard.txt` → `proguard.txt` in the AAR; keeps
  `net.ladenthin.llama.**` for the JNI `FindClass`/Jackson reflection surface). The AAR's
  CPU AAR's `classes.jar` is the **byte-identical Maven-built classes jar** minus `module-info.class` (D8
  rejects it) — it carries no natives, which ship as separate natives jars; on Android `LlamaLoader`
  resolves via `System.loadLibrary("jllama")`, which finds the AAR-installed `.so` — no loader
  change was needed. `registerAarTask` copies `**/*.so` only (the `jllama-*.txt` lists are desktop
  loader input) and fails loud on the flavor's defining library per ABI; the CI job validates the
  full set. The additive OpenCL AAR has an empty `classes.jar`: its CPU AAR dependency already
  supplies the Java API, and duplicating it breaks D8/R8. Built by the **standalone plain-Gradle build** in `llama-android/`
  (see "Repository layout" for why it is not a Maven module); the POM mirrors the core's
  compile-scope deps — the list itself, versions **and scopes**, is read from `llama/pom.xml`
  (`coreCompileDependencies`: no `<scope>` or `compile`, not optional; today jackson-databind,
  slf4j-api, jspecify). Not mirrored: the `provided` ones (checker-qual, lombok, animal-sniffer) and the
  SLF4J binding, which is the JVM-only runtime dependency. The 6.0.0 AAR poms were written from a
  hand-kept list that also carried checker-qual at `compile`, which the core keeps `provided` (major
  55, see "Java 8 bytecode floor"); the versions were derived and stayed right, the scopes drifted.
  `.github/verify-aar-poms.py` checks the poms `publishToMavenLocal` wrote against the core pom in
  `package-android-aar`. The consumer rules carry `-dontwarn org.checkerframework.**`: the classes
  reference the checker's qualifiers (below), which no Android app has.
- **`net.ladenthin:llama-kotlin`** — Maven reactor module; pure-Kotlin (2.4, jvmTarget 1.8)
  coroutines façade: `generateFlow`/`generateChatFlow` (cold `Flow`, source closed on
  completion/error/cancellation) and `completeSuspend`/`chatSuspend`/`chatCompleteTextSuspend`/
  `embedSuspend` (`completeSuspend` wires coroutine cancellation into the cooperative
  `CancellationToken`). The core dep is **provided-scope** so Android consumers pair it with the
  AAR instead of transitively pulling the desktop classes jar. 6 model-free unit tests fake the
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
libdl.so liblog.so libandroid.so`, the siblings next to it -- which bionic finds through the app's
native-library directory, not through a run path: an Android library carries none, see "Modular
natives" -- plus `libOpenCL.so` and `libggml-base.so` for the module in `Linux-Android/aarch64/opencl`) — the
same allowlist mechanism as every desktop CPU build, where the AAR job used to carry a copy of its
own; the AAR job additionally asserts the AAR's `jni/` library is the staged one. And
`LlamaLoader` now includes the swallowed `System.loadLibrary` message in its
"Directly from .apk/lib (…)" tried-path entry so a future dlopen reason is never invisible.

**CI (`publish.yml`):** `test-java-llama-kotlin` (model-free unit tests);
`package-android-aar` (needs both Android native jobs) builds the core jar, stages the natives,
assembles both AARs, validates structure (entries, minSdk, classes.jar content, 16 KB alignment),
publishes to mavenLocal, and runs the **AGP consumer smoke test** — the minimal app fixture in
`.github/android-consumer-test/` resolves each AAR from mavenLocal (`jllamaBackend=cpu` and
`jllamaBackend=opencl`, the latter with its transitive CPU AAR) and runs a full R8
`assembleRelease` for each on the runner's preinstalled Android SDK (this is what actually validates
AGP/Android Studio consumption). **On-device runtime IS now CI-covered** via
`test-android-emulator`: a KVM-accelerated x86_64 emulator (API 30) runs the fixture's
`connectedDebugAndroidTest` — `System.loadLibrary` from the AAR's `jni/x86_64`, on-device
`GgufInspector`, and real native inference against the adb-pushed cached draft model
(AMD-Llama-135m). The job is a **release gate** (in both
publish `needs:` graphs) since PR #298, after running flake-free through the PR's validation
cycle. arm64 kernels + the Adreno/OpenCL flavor remain out of emulator scope —
the planned example app covers those on hardware.
Both publish jobs `need` these jobs (fail-loud release gating) and stage the signed AARs with
Gradle into `llama-android/build/staging-repo` (`publishAllPublicationsToStagingRepository`, a file
repository), a Central root of `.github/release-assets.json`; `central-snapshot` / `central-release`
transfer them with the rest of the frozen publication (see "Signed publication"). For a snapshot
Gradle writes timestamped names there (`llama-android-<base>-<yyyyMMdd.HHmmss>-<n>.aar`, also into a
file repository), which the gate maps back to `-SNAPSHOT`. `llama-kotlin` rides the reactor's
staged deploy.

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
- **No release asset:** the thin jar on Central is the deliverable (the `-P assembly` jar without
  the core and its `Class-Path` into the core fat jars went with the fat jars). A GPU is one more
  natives jar on the JBang line (`--deps net.ladenthin:llama:<v>:<backend>-<os>-<arch>`) or the
  `gpu-natives` profile (`-Dllama.classifier=…`).
- **CI:** the model-free job (which also collects the thin jar + its dependencies without
  `net.ladenthin:*` as artifact `llama-atmosphere-agent-jar`), the model-backed integration job and
  `smoke-agent-linux` (that directory on one classpath with the `linux-x86-64` smoke set of the core,
  `.github/smoke-agent.sh`) all gate both publish jobs.
- **Version bump:** the pom's `<version>` is the reactor's, and `versions:set` does not touch this
  standalone pom -- bump it by hand (`check-natives.py` fails until it agrees) with the JBang
  coordinates in the two READMEs; `llama.version` follows (`${project.version}`).
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

## Signed publication: one gate, three independent consumers

The actions of [bernardladenthin/actions](https://github.com/bernardladenthin/actions), shared by the four Java repositories; the full text is
[docs/publication.md](https://github.com/bernardladenthin/actions/blob/main/docs/publication.md) there, the JSON contract and its rules
[`../workspace/policies/release-asset-scripts.md`](../workspace/policies/release-asset-scripts.md).
The jobs stay in this workflow (environment, permissions, concurrency); `java/check-publication`
checks their wiring and dry-runs `.github/release-assets.json` in every pull request.

- **`check-snapshot` / `check-tag`** run `java/check-release-ref`: a release publishes only from the
  tag `v<version>` of a non-SNAPSHOT version, a snapshot only a `-SNAPSHOT` version and never from a
  tag (the snapshot check only when the run publishes: `main` carries the release version for the
  commit a release is tagged on).
- **`publish-snapshot` / `publish-release`** build and sign everything, stage the Central files
  locally (`-DskipUpload=true deploy`, here plus the Gradle staging repository of the AARs) and end in
  `java/finalize-publication`: the collection of what `.github/release-assets.json` names, then
  `publication prepare --channel <channel>`, the one gate. It checks the version against the
  channel and `GITHUB_REF`, exact membership of the Central staging (every artifact with its own
  signature, no orphan, no unknown file; Maven metadata and checksums are ignored, Gradle's
  timestamped snapshot names are mapped back to `-SNAPSHOT`) and of the GitHub files against the
  contract, names GitHub would rename, byte equality of each Central file with its GitHub twin, and
  every signature with `gpg --verify`. It freezes `central/`, `github/` and `manifest.json`
  (commit, version, channel, size and SHA-256 per file), and only after the Actions upload of that
  package does the job output `assets_complete=true`.
- **The consumers** each `need` only that job and that output, never each other:
  `github-snapshot` / `github-release` (`java/publish-github`: attach the `github/` files; the rolling
  `snapshot` pre-release is serialized, prunes only after a complete upload and starts as a draft),
  `central-snapshot` / `central-release` (`java/publish-central`: the release as one Portal bundle
  with `publishingType=AUTOMATIC`, no validation polling; the snapshot through
  `maven-deploy-plugin:deploy-file`, serialized by a concurrency group and skipped when another run
  already transferred a newer commit), and `attest-snapshot` / `attest-release`
  (`java/attest-publication`: Sigstore over `github/*`). Each verifies the manifest after the
  download; none rebuilds or re-signs. A failed Central transfer leaves the GitHub release and the
  attestation in place, and the other way round.
