# Handover prompt for the local agent -- PR A on `claude/hopeful-pascal-9jlbqb`

Paste the text below the line into the local agent (Windows workstation: Ryzen 7 5800H, RTX 3070,
AMD iGPU, MSVC + LLVM + Docker + JDK 21 + Maven). It describes what this branch already contains,
what the remote session cannot verify from its sandbox, and the exact deliverables.

**Status (2026-10-09).** PR A (#496) and the agent's own corrections to this file (#498) are merged;
the branch has since moved on (b11538, `InvalidRequestException`). The file stays for one reason:
deliverable 6 (B2) carries the agent's measurements and the four `OPEN` items that the Windows
CPU-variants CI job still needs, and `TODO.md` ("CPU variants") points here for them. Deliverables
1-5 were not reported and are not asked for again; delete the file when B2 lands.

**Deliverable 6 (B2) carries measurements taken on this same workstation on 2026-10-09**, marked
`MEASURED` or `OPEN` per item: four are done and only need confirming, three are genuinely open, and
the Java suite among them is the one that decides whether the variant path works at all. Verify a
`MEASURED` line rather than skipping it -- a different result there is a finding -- but spend the
time on the `OPEN` ones.

---

You are working in a checkout of `bernardladenthin/java-llama.cpp` on the branch
`claude/hopeful-pascal-9jlbqb`. Read `CLAUDE.md` first (it is long; the sections "Natives jars",
"CPU variants", "Building the native library for local Java tests" and "Testing" matter here).

## What the branch contains (do not redo it)

| Item | State | Where |
|---|---|---|
| **A1** llama.cpp b11529 → b11534 | done, 3 commits (b11530, b11531, b11534); 603/603 C++ tests and the model-free Java suite are green on Linux x86-64 | `llama/CMakeLists.txt`, `docs/history/llama-cpp-breaking-changes.md` |
| **A2** floor on *executed* tests | done: `.github/verify-test-counts.sh --min-executed 1800` in every `test-java-*` job | `.github/verify-test-counts.sh`, `java-tests.yml`, `publish.yml` |
| **A3** loader lines to stderr | done: `[jllama] using native backend '…'` and `[jllama] extracted …` print to `System.err` | `LlamaLoader.java` |
| **A4** one-file JBang chat | done: `examples/jbang/Chat.java` names the classes jar + the 7 desktop CPU natives jars (JBang treats a `pom` dependency as a BOM, so `llama-platform` cannot be used there); `check-natives.py` guards the lines | `examples/jbang/Chat.java`, README "Try it with JBang" |

The sandbox the branch was written in has no models (HuggingFace is blocked), no Windows and no GPU.
Everything below is what only your machine can show.

## Deliverables, in this order

1. **Toolchain inventory** (once): clang versions and paths (LLVM on `PATH`, the one inside Visual
   Studio under `VC\Tools\Llvm\x64`), Visual Studio and MSVC toolset versions, Windows SDK, Ninja,
   sccache, JDK(s), Docker images you have pulled, CUDA toolkit and driver version, Vulkan SDK,
   Android SDK/NDK. (There is no Mac and no Android device; macOS and Android are verified through
   CI runners and the CI emulator only, so do not plan work that needs either.)

2. **A1 -- the Java suite with models on Windows.** Build the Ninja CPU library at b11534 the way CI
   does (`mvn -q compile` in `llama/`, then `.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows
   -DOS_ARCH=x86_64 -DBUILD_TESTING=ON`, `ctest --test-dir build -C Release`), download the CI model set
   (`.github/models.csv`, into `models/` at the repo root) and run `mvn -f llama/pom.xml test`.
   Report: ctest result, Surefire's final `Tests run: N, Failures, Errors, Skipped` line, and the
   names of the skipped tests (`grep -l '<skipped' llama/target/surefire-reports/TEST-*.xml`).

3. **A2 -- the numbers and one red run.** From the run above, paste the line
   `.github/verify-test-counts.sh llama/target/surefire-reports --min-executed 1800` prints
   (run it under Git Bash). CI measured 1869 run / 13 skipped on both Windows jobs; yours should
   match within a few tests. Then rename one model (e.g. `models/stories260K.gguf`), run the suite
   again and show that the script goes red (the skipped count jumps, or a class reports zero
   entries). Restore the model afterwards.

4. **A3 -- router log without the warning.** `RouterModeIntegrationTest` is Linux-only, so run it
   in Docker (any Linux image with JDK 21 + Maven + a build of the library, or `cmake` inside the
   container -- the `manylinux_2_28` image of `.github/dockcross/dockcross-manylinux_2_28-x64` has
   the compilers): `mvn -f llama/pom.xml test -Dtest=RouterModeIntegrationTest` with the reasoning
   model present. Report whether the router output still contains
   `unexpected output on the command pipe` (it must not) and whether the test is green.

5. **A4 -- JBang against the local snapshot.** `mvn -f llama/pom.xml -DskipTests install` installs
   `net.ladenthin:llama:5.2.0-SNAPSHOT` (classes) into `~/.m2`; `-P natives` additionally installs the
   natives jars (only `cpu-windows-x86-64` has content on your machine, the others are empty jars,
   which is fine). Copy `examples/jbang/Chat.java` elsewhere, change every `//DEPS` version to
   `5.2.0-SNAPSHOT`, and run `jbang Chat.java models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (JBang resolves
   from `~/.m2` first). Report: does it resolve, load the Windows natives jar (`[jllama] using native
   backend 'cpu'` on stderr) and answer a prompt? Paste the first exchange.

6. **B2 pre-verification -- Windows CPU variants, before the CI job is written.** Build the variant
   library with plain clang (not `clang-cl`, not `cl.exe`: ggml drops 5 of the 14 variants under
   `MSVC`, see CLAUDE.md "CPU variants").

   **Four of these were already measured on this workstation on 2026-10-09, so VERIFY them rather
   than discover them -- a different result is a finding worth reporting.** The three that are open
   are marked as such, and the Java suite is the one that matters most, because it is the only step
   that proves the variant path end to end and the only one the measuring session could not run (its
   `mvn compile` dies in an Error Prone / NullAway crash on `args/package-info.java`, unrelated to
   the native work; disabling annotation processing then breaks Lombok, so there was no quick
   bypass).

   - **MEASURED -- which clang.** Plain clang **23.1.3** from `C:\Program Files\LLVM\bin` builds all
     14 variants with **0 errors**. Note two things the earlier draft of this prompt got wrong about
     this machine: `PATH` carries **23.1.3**, not 20.1.8, and **Visual Studio ships no clang here**
     (neither `2022\Community` nor `18\BuildTools` has a `VC\Tools\Llvm\x64`), so there is no "newer
     versus older clang" choice to make -- the 20.1.8 used in the earlier comparison was unpacked
     separately. 23.1.3 is the stricter of the two (it makes `-Wincompatible-pointer-types` an
     error, which clang 22 introduced), and `patches/0017` is what makes `quants.c` compile there at
     all: a failure on that file is a finding, not a toolchain problem.
   - **OPEN -- the exact `build.bat` command line that works.** The measurement used `cmake`
     directly, so `build.bat` itself is unverified for this configuration. Report the line that
     works (`-G "Ninja Multi-Config"`, `-DJLLAMA_CPU_VARIANTS=ON`, the clang toolchain,
     `-DGGML_OPENMP=OFF`, `-DOS_NAME=Windows -DOS_ARCH=x86_64`), with and without sccache on `PATH`
     -- does `build.bat`'s probe accept clang, or does it fall back to an uncached build? The probe
     only ever proved `cl.exe`, so this is genuinely unknown.
   - **MEASURED -- the directory.** 18 DLLs in
     `src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu/`: `jllama.dll`, `ggml.dll`,
     `ggml-base.dll`, 14 `ggml-cpu-<level>.dll` (`x64`, `sse42`, `sandybridge`, `ivybridge`,
     `piledriver`, `haswell`, `skylakex`, `cannonlake`, `cascadelake`, `icelake`, `cooperlake`,
     `zen4`, `alderlake`, `sapphirerapids`) and `ggml-rpc.dll`, nothing in a `Release/`
     subdirectory, and `jllama-files.txt` naming **15** of them -- the modules plus `ggml-rpc`.
     **`jllama-extras.txt` is absent, and that is correct, not a defect:** CMake never writes it;
     `merge-native-artifacts.sh` derives it in CI from everything in the directory that
     `jllama-files.txt` does *not* name, which on Windows is exactly `ggml.dll` and `ggml-base.dll`
     (they must be pre-loaded by full path -- there is no `$ORIGIN` on Windows).
   - **MEASURED -- `python .github/verify-native-deps.py llama/src/main/natives`.**
     "18 native libraries checked, 0 violations". Paste yours anyway; it is one line and it covers
     the hybrid-CRT guard in both directions.
   - **OPEN and the important one -- the Java suite** (`mvn -f llama/pom.xml test`) against the
     variant build, with the models in place. CLAUDE.md's warning applies directly here: a
     successful load proves little, because in the earlier Windows measurement *every* crash came
     after a load that reported the right device count. So report the Surefire summary line, and
     whether `RpcIntegrationTest` and the model-backed classes pass against a build whose CPU
     backend is a loadable module rather than a linked library.
   - **OPEN -- which variant ggml picks on the 5800H.** Zen 3 has AVX2 and no AVX-512, and
     `alderlake` needs AVX-VNNI it does not have, so `haswell` is the expectation. Caveat before you
     spend time on the log: CLAUDE.md records that ggml logs module decisions at `GGML_LOG_DEBUG`
     only, so `-lv 4` may not name the chosen one. If it does not, list the DLLs the JVM actually
     has mapped (Process Explorer, `listdlls`, or `tasklist /m`) instead of guessing.
   - **OPEN -- pp512 and tg128** for Qwen3-0.6B Q4_K_M at 8 threads, variant build versus the single
     static CPU build. For orientation, measured with `llama-bench` on this machine at
     `GGML_NATIVE=ON` (so a single kernel, not the variant selection): clang static reached pp512
     391.6 and tg128 82.3 t/s with `GGML_OPENMP=OFF`. A variant build landing near that means the
     module indirection costs nothing; well below it is a finding.

## How to work on the branch

- Pull before every push (`git pull --rebase=false origin claude/hopeful-pascal-9jlbqb`); never force-push,
  never rewrite the remote session's commits. Commit only files you own: your reports go to
  `docs/handover/local-agent-report-<topic>.md` (plain text, numbers in tables, command lines as
  code blocks); do not edit source files on this branch unless a step above says so -- propose the
  change in the report instead. End your commits with your own `Co-Authored-By` / session trailers,
  not the remote session's.
- Before pushing anything that touches `.java`, `.cpp`/`.hpp` or `.github/`: `mvn -f llama/pom.xml
  spotless:apply`, `clang-format` 23.1.3 on the C++ files, and the Python checks
  (`python .github/check-natives.py`, `check-patches.py`, `check-release-gate.py`,
  `check-run-scripts.py`, `check-shared-files.py`, `python -m unittest discover -s
  .github/buildcheck/tests -t .github`).
- Do not dispatch or re-run CI workflows; the owner does that.
- Measurements are only useful as text: paste the console lines, not a description of them.
