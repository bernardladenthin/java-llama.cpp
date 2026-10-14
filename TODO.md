<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# TODO — java-llama.cpp

Open work items for this repo. Cross-cutting tracking lives in
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md);
items here are jllama-specific or are this repo's slice of a
cross-cutting initiative.

**Completed work is not recorded here.** It lives in git history and in
`crossrepostatus.md`; a finished item is deleted from this file rather than annotated,
so everything below is genuinely still open.

## Where the modular natives stand

**The goal was exchangeable native libraries built with LLVM, replacing the fat jars. As of
2026-10-12 that is implemented and proven -- on real hardware and through the whole pipeline.** Two
consecutive `workflow_dispatch` runs on `main` finished **completely green** (67 jobs, 0 red): runs
**38186841655** and **38218935644**, both through `Package JARs`, all six smoke jobs, both GPG checks,
`Publish Snapshot to Central` and the GitHub pre-release. Before those, nothing behind `package` had
ever run with modular natives.

This section stays because a Publish run costs hours and "what is this run supposed to show" has to
be answerable before starting one -- but that question now has an answer for the state as shipped.

**Done and proven.** 8 library jars plus 15 additive GPU module jars, each holding one ggml backend
module; `jllama-build.txt` pinning a module to its library's llama.cpp tag before anything is
extracted; the extraction directory keyed by build *and* module set, reused across starts. Windows
x86-64 is built with **plain clang**, which is what yields all 14 CPU variants (`clang-cl` loses 5 of
them to ggml's `if (NOT MSVC)` block). All 14 x86 modules were measured on one Granite Rapids machine
and 6 of 8 aarch64 ones on a Graviton 4 -- before that, no module had ever been *chosen* on hardware
the project could inspect.

**Where it sticks, in the order that matters for a consumer:**

1. **The library jar is not self-contained** (item 1 below). Java runs in containers, and the usual
   slim base images have no `libgomp`; the jar then fails to load at all. Measured, decided, not yet
   changed.
2. **"A module that cannot help is skipped" holds for Vulkan and breaks for the other two**
   (items 2 and 3): CUDA on an incompatible driver `abort()`s the JVM in *every* configuration,
   OpenVINO on a CPU-only host takes over and runs 172x slower. The premise the whole additive design
   rests on is therefore conditional, and that was unknown until a module was actually used.
3. **Nothing behind `package` has ever run on the modular natives.** `package` was red until the
   Windows OpenVINO module jar was unblocked, so the smoke-set assembly, `smoke-natives-jars.sh` with
   all 23 jars on classpath *and* module path, the four `smoke-natives` rows, the macOS smoke, the
   agent smoke, the bytecode gate and `verify-native-deps.py` over all 23 jars have never executed
   together. **This is what the next Publish run is for.**

**So the next CI run should be the current `main`, unchanged.** It is the first opportunity to see
everything behind `package`, and bundling the decided changes into it would confound the result: a red
run would not say whether the pipeline or the change is at fault. Expect several rounds there -- a
stage that has never run is unlikely to pass first time -- and note that the `windows-x86-64` smoke
row draws the same CPU lottery as the Windows test job, so it exercises AMX only when GitHub hands it
an Intel runner.

**The batch for the run after that**, assembled while the first one runs: `GGML_OPENMP=OFF` plus
`-static-libstdc++ -static-libgcc` on Linux and s390x with the dependency allowlist inverted, the
explicit `CMAKE_CUDA_ARCHITECTURES`, the OpenVINO opt-in decision, and the shipped diagnostic entry
point (item 5). These are independent of each other and each is verifiable in the same run.

### Round 0 found one defect, and it was mine

Run 38146809329 (cancelled after the first failures) went red in **four Android jobs**, with one root
cause: `Package + Validate Android AARs` failed, and the three jobs that need the AAR followed.

```
Property <jackson.version> not found in ../llama/pom.xml <properties>
  -- keep the AAR pom dependencies in lockstep with the core
```

`llama-android/build.gradle.kts` mirrors the core's dependency versions into the AAR pom by **parsing
`llama/pom.xml`**, and the Jackson advisory fix moved `jackson.version` into the **parent** pom (where
it belongs: the parent now imports `jackson-bom` so `llama-langchain4j` cannot pull an older
`jackson-core` transitively). The Gradle build looked only in the core pom.

**That check working is the good news.** It is fail-loud by design, with a message naming exactly the
invariant it protects, and it caught a cross-build reference that no Maven gate sees -- the same
"reference not updated after a move" class as the stale `spotbugs-exclude.xml`, the PIT
`targetClasses` and the `CMakeLists.txt` OSInfo repairs. **Fixed by teaching the parser Maven's own
inheritance** (core pom, then parent) rather than by restoring a duplicate `jackson.version`, which
would drift the next time one of them is bumped. **Not verifiable locally** -- no Gradle on this
machine -- but the XML lookup it performs was reproduced against both poms: the property is absent
from the core and present in the parent at 2.22.3.

**What this says about round 0 itself:** it reached `Package + Validate Android AARs`, so the stages
before it ran. The Android jobs sit *beside* `package` rather than behind it, so this failure says
nothing yet about the smoke sets, `smoke-natives-jars.sh` or the three merged changes -- those still
need a run.

**The fix itself left a fragment, and the re-dispatch found it.** Run 38148401395 (`workflow_dispatch`
on `main` at `989534d5`, the first commit carrying the fix) went red in the same three Android jobs,
this time in `Assemble AARs + publish to mavenLocal`, i.e. before Gradle reached the pom parser at
all: the rewrite of `coreProperty()` kept the old function's tail -- a `?: error(...)` and a `}` --
at file level two lines below the new closing brace, which is a Kotlin script compilation error
(`build.gradle.kts:118:9: Expecting an element`). Fixed by deleting the two lines (bump session,
2026-10-11) and verified both ways, because the sandbox *does* have Gradle (`/opt/gradle`, 8.14.3):
the unfixed script fails with exactly that message, the fixed one configures. **`gradle --offline -q
help` in `llama-android/` is the minimum check for any change to that script** -- it needs no natives,
no network and no Android SDK, and it catches everything the AAR job can fail on before
`Assemble AARs`. When run 38148401395 was last read, 32 jobs were green, 16 still running and only
the three Android ones red, so the questions behind `package` were still open; the dispatch after
this fix is the one to read for them.

### Round 0, run 38164077038: `package` reached, and the opt-in decision had a loose end

**What it answered, and it is the first time any of it has been answered:** all **54** build jobs
green, both ROCm among them (Windows 3:08 h, Linux 2:57 h -- they *are* the three-hour pipeline; every
other job was done inside an hour), and **`verify-native-deps.py`: 99 native libraries checked, 0
violations.** So the SYCL run path is clean, and the Linux jars no longer declare `libgomp.so.1`
without anything noticing.

**What it found:** `smoke-natives-jars.sh` failed with
`classpath: the loader did not put the openvino module jar of Linux/x86_64 in place` -- a consequence
of making `openvino` opt-in, which I should have foreseen when I added `OPT_IN_BACKENDS`: that script
asserts the loader line names **every** GPU module of the platform, read from `natives.csv`, and the
list had no way to say that one of them is deliberately left out. The loader even printed the reason
("is on the classpath but not used: it is opt-in"); nothing read it.

Fixed by declaring it where every consumer already looks: an **`optin` column in `natives.csv`**.
`check-natives.py` now holds `LlamaLoader.OPT_IN_BACKENDS` to that column in both directions, the
model-backed smoke rows stop expecting an opt-in module (bash and PowerShell), and
`smoke-natives-jars.sh` names every module through the property instead -- which makes it **the only
place in CI that loads an opt-in module at all**, so the mechanism is exercised rather than merely
respected. 122 buildcheck tests, four of them new; both module lists and the property construction
were simulated against the real CSV locally.

**Still unanswered, and only a run can answer it:** everything behind `smoke-natives-jars.sh` --
the five smoke sets, the four `smoke-natives` rows with the RPC smoke on aarch64, the macOS smoke and
the agent smoke were all **skipped**, as they were in every run before. Two emulator jobs stayed red
in this run by construction (their fix landed afterwards, #526).

### Hand-over from the bump session, 2026-10-11

What that session did: the b11538-to-b11556 bump (#513 -- every patch re-verified, two history rows,
the nlohmann finding recorded below under "The FetchContent nlohmann/json is shadowed"), the two-line
Gradle fix above, and a read-only check of the 2026-10 plan (the 12-point table that followed the
Windows measurement report) against `main`. Result of that check, so it is not redone: **9 of the 12
points are on `main`** -- the publish dispatch, b11535 and beyond, the macOS OpenSSL defect, the
Windows CPU variants, the attach-mode sleep callback, the test-count floor, the agent's approval gate,
400 for invalid requests, and all of point 11 (the router worker's stdout line went to stderr on
2026-10-09, `e9758054`). Open from that table: **the release with its migration section** (new entry
under "Open" below), **the upstream submissions** (policy: after the release; inventory below) and the
two owner-only items of point 12 (license compliance, the RPC upstream issues). Every finding of the
Windows report -- D1 preload order, D1b hybrid CRT, D3 UTF-8 path, D4 registry logging, D5 keyed
extraction directory, D6 reuse across starts, E1/E3 manylinux_2_28 -- is implemented; the Windows
two-process measurement of D5 is round 2, item 5. Nothing else from that plan is missing from this
file.

## The next batch, in the order it should be done

Assembled 2026-10-11 after the libgomp / CUDA-arch / OpenVINO changes landed. The constraint that
shapes it: **a Publish run costs hours**, so work is grouped by what one run can answer, and anything
that needs no run is done before the run starts.

**Round 0 -- one Publish run on the current `main`, nothing added.** *(Started 2026-10-11 as run
38146809329, `workflow_dispatch` on `main` at `df5e1ba` -- which also carries the b11538-to-b11556
bump that landed just before it, so the run answers three questions rather than two. They stay
separable in the output: a packaging or smoke failure is not a flag failure, and a bump failure shows
up as a compile or patch error. Cancelled on the Android AAR defect above; re-dispatched as run
38148401395 at `989534d5`, which failed the same three jobs on the fragment that fix left behind.
**The run to read is the first dispatch after that fragment is gone.**)* It is the first run that can
reach everything behind `package`, *and* the first that exercises the three changes just merged. Those
two questions are separable in the output (a packaging or smoke failure is not a flag failure), so
they share a run rather than costing two. What it must show:
`package` green; `verify-native-deps.py` happy over all 23 jars **with `libgomp` gone from the Linux
ones** (the allowlist now fails if it is back); the smoke sets assembled; `smoke-natives-jars.sh`
loading all 23 jars on classpath **and** module path; the four `smoke-natives` rows with their loader
line, HTTPS round and -- new -- the RPC smoke on **aarch64**; `smoke-natives-macos`;
`smoke-agent-linux`. Plus two numbers to read off rather than assume: **what the explicit CUDA arch
list costs** in job time and module size, and whether the Windows x86-64 rows drew an AMX runner (the
host-CPU line says so).

**Round 1 -- things that need no run, done while round 0 is in flight.** None of these touch the same
files as each other, so they can go in one branch:
1. ~~**The shipped diagnostic entry point**~~ **done 2026-10-11**: `net.ladenthin.llama.Diagnose`
   (the main class) plus `loader.DiagnosticReport` (the half only the loader can answer). The split
   is not tidiness: every platform and classpath fact is a package-private member of `LlamaLoader`,
   and **Java access is per package, not per module** -- so that half has to live in the loader's
   package, while the main class cannot, because that package is deliberately not exported and a
   consumer on the module path could not run a main there. The architecture rule allows exactly this
   direction (`Api -> Loader`, never back), which is also why the pinned-version constant (`value`)
   and the load itself (`LlamaModel`) are read on the `Diagnose` side.
   It prints the pinned tag, the JVM, `os.name`/`os.arch` and what `OSInfo` derives from them, every
   set `net.ladenthin.llama.*` property, each natives directory of the platform **with the llama.cpp
   tag of its `jllama-build.txt`** (which is how a mismatched set becomes visible at all) and, on a
   failed load, the operating system's own reason. 9 tests, no natives and no model needed.
   **Two defects of its own, both found by running it rather than by reading it, and both the same
   mistake:** it composed the property name from `LlamaSystemProperties.PREFIX`, which carries **no**
   trailing dot, and so read `net.ladenthin.llamabackend` -- a name nothing sets, so a filter that
   was in effect was reported as unset; and `namedBy` compared case-insensitively while
   `LlamaLoader.selectModules` compares with `equals` and **fails loud** on an unknown name, so
   `OpenVino` would have been reported as an enabled module while the load throws on it. Both are now
   read through the loader's own accessor and matched by the loader's own comparison, each with a
   red/green test. The rule they both broke: **a diagnostic may not be kinder, or differently spelled,
   than the code it describes.**
2. ~~**README: the GPU jars' real requirements**~~ **done 2026-10-11**: the two `cuda13` rows name the
   driver requirement, the two `openvino` rows say they are opt-in and name the OpenCL ICD loader the
   OpenVINO archive does not ship, and an `[!IMPORTANT]` block carries the measured detail -- the
   CUDA abort (`ggml_abort`, not catchable, and reached even with `-ngl 0`), the 17x/172x OpenVINO
   regression that is the reason for the opt-in, and Vulkan as the one backend that behaves as
   designed. The existing note's sentence "simply registers none" now names its two exceptions
   instead of stating the rule as if it held. **The README did not mention the OpenVINO opt-in at
   all**, so a consumer adding that jar would have seen nothing happen and no reason why.
3. ~~**`TestConstantsTest` / `TestModelPaths` parity**~~ **done 2026-10-11, and the premise was
   mostly outdated**: `llama-langchain4j`'s test file had grown since that line was written and
   already covered nine of the core's ten cases, including the reactor-root branch non-vacuously
   (it resolves `llama-langchain4j/pom.xml`, which exists only when looked up from the parent, so a
   pass proves the second branch ran) and the "no test may read a model property without the
   resolver" source rule. The one genuine gap was `fromProperty`: only the **unset** case was
   tested, so a set property that bypassed the resolver would have been invisible -- which is
   exactly the failure the resolver exists for, and the one that stayed hidden in the core for
   hundreds of builds because a class-level assumption records no test entries. One test added.

**Round 2 -- ~~needs round 0 to have produced artifacts~~ DONE 2026-10-12.** All three were measured
against the **published** snapshot jars (`20261012.035453-12`) rather than a build tree, so the
numbers are what a consumer resolving from Maven gets. Written up in
[`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md) and the new
[`docs/benchmarks/native-loading.md`](docs/benchmarks/native-loading.md); the account-specific record
is in the private workspace. Two AWS boots, **under 0.08 USD together**, both terminated and four
regions swept afterwards.
4. ~~**A `g4dn` run proving a T4 works** with the new arch list~~ -- **done, and the abort is gone.**
   Same AMI, same driver 595.91.07, same Tesla T4; only the module changed. **Three exit-134 aborts
   became three clean runs**, including `-ngl 0`, which was the most absurd of them since no layer was
   on the GPU. The GPU gives **20.6x prompt processing and 7.0x generation** over the CPU baseline on
   that machine. `cuobjdump` on the published module confirms native code for `sm_75`, `sm_80`,
   `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120a` and `sm_121a` plus PTX for `sm_121a` -- so Turing,
   A100 and Hopper no longer depend on the driver and anything newer than Blackwell still has a
   fallback. The `a` suffixes are upstream's own rewrite of every `12x` architecture, not a defect in
   the list this project passes. **The README's figures were corrected with it**: the CUDA paragraph
   described the pre-fix state, and OpenVINO's cost was understated (see item 6).
5. ~~**The Windows extraction-reuse measurement**~~ -- **done, all four documented properties
   confirmed.** Extracting 18 files / 39 MB costs **~810 ms per JVM start** (cold ~2175 ms against
   warm ~1365 ms) and the reuse removes it -- per start, so also per surefire fork. The directory key
   visibly carries the module set (`...-11b5000` against `...-11b5000-mf39ba978`). Two JVMs of one
   build run concurrently without interfering. And the sharpest case: with an `RpcServer` holding the
   library and the directory backdated past `STALE_EXTRACTION_AGE`, a sweeper left **exactly the five
   files that JVM still had open** -- the library, the two pre-loaded ggml libraries, the one CPU
   module ggml kept, and `ggml-rpc` -- while the 13 it had scored and closed went and were
   re-extracted on the next start. The server kept running throughout.
6. ~~**The slim-image / distroless load test**~~ -- **done.** The published Linux jar declares only
   what a JVM brings with it (`libc`, `libm`, `libdl`, `libpthread`, `libstdc++`, `libgcc_s`, run path
   `$ORIGIN`, `GLIBC_2.27` at most) -- **`libgomp.so.1` is gone from the artifact**, not only from the
   allowlist. `net.ladenthin.llama.Diagnose` from the published jars loads in
   `gcr.io/distroless/java21-debian12` and `eclipse-temurin:21-jre`, and is refused by
   `eclipse-temurin:21-jre-alpine` **by name**: `No native library found for os.name=Linux-Musl`.
   That last row is not a defect -- musl was never a shipped platform -- and it was in the matrix on
   purpose, because a measurement that only admits successes cannot show where the boundary is.
   **Also measured, and it corrects this file and the README:** OpenVINO on an Intel Sapphire Rapids
   instance costs **21.7x prompt processing and 336x generation** against the baseline of the same
   machine (481.3/33.6 down to 22.2/**0.1**), not the 17x/172x recorded from the earlier machine.
   Without an OpenCL ICD the module does not open at all (2 backends, full CPU speed); with
   `ocl-icd-libopencl1` it registers a third and collapses. And with the jar on the classpath but
   `net.ladenthin.llama.backend` unset the loader says `is on the classpath but not used` and the
   machine runs at full speed -- **the only hardware proof the opt-in decision has**, since CI has no
   Intel accelerator and nothing happening there proves nothing.
   distroless Java, drop the classes jar plus the Linux library jar in, and load. This is the check
   that CI *structurally cannot* do, because it runs on full runners -- and it is the direct
   verification of what round 0's dependency check only implies.

**Deliberately not in the next batch**, with the reason: `cuda13-windows-aarch64` (a new build job --
worth doing, but it belongs in a run of its own so a toolchain failure is unambiguous); the Android
Vulkan module (same); code signing (needs the owner's identity validation first); and the upstream
submissions (policy: not before the release).

## What still has to be measured

Kept as a list because each entry names the machine it needs -- the harness is
[`docs/benchmarks/ec2-harness.md`](docs/benchmarks/ec2-harness.md) and a run is cents:

| what | needs | why it cannot wait for CI |
|---|---|---|
| ~~the cost of `GGML_OPENMP=OFF`~~ | ~~16-vCPU Linux~~ | **done 2026-10-11: no measurable cost** |
| ~~re-baseline the module tables at the current pin~~ | — | **done 2026-10-12 at b11558 on 64 cores**, and the suspicion behind it was **refuted**: #30270 changes nothing measurable ([`docs/benchmarks/upstream-f16c-30270.md`](docs/benchmarks/upstream-f16c-30270.md)), so the baseline moved with the machine size, not with the pin |
| ~~Does a self-contained jar load on a slim/distroless image?~~ | — | **done 2026-10-12**: distroless and `temurin:21-jre` load, `temurin:21-jre-alpine` is refused by name (`os.name=Linux-Musl`) -- see [`docs/benchmarks/native-loading.md`](docs/benchmarks/native-loading.md) |
| ~~Does `75-real 80-real 90-real` make a T4 work?~~ | — | **done 2026-10-12: yes.** Three `exit 134` aborts became three clean runs on the same AMI, driver and GPU |
| ~~Windows extraction reuse~~ | — | **done 2026-10-12 locally**, no rented machine needed: ~810 ms saved per start, and a sweep leaves exactly the five files a running JVM still holds |
| ~~the `sapphirerapids` vs `haswell` gap at high core counts~~ | — | **done 2026-10-12: the ranking reverses.** At 64 threads `haswell` leads token generation at 192.6 while `sapphirerapids` is last at 181.8 -- and ggml picks the latter |
| **OpenVINO on an Intel GPU/NPU** | hardware nobody rents | the 21.7x/336x figures are the CPU-plugin fallback only, so nothing is known about the hardware the jar is actually for |
| the two **SME** aarch64 modules | not rentable anywhere | — |

## Open — jllama-specific

### The release: 5.2.0 or 6.0.0, with an "Upgrading from 5.1" section (owner's decision)

**Nothing has been published since 5.1.0 on 2026-08-29.** `pom.xml` is `5.2.0-SNAPSHOT`, and
`CHANGELOG.md`'s `[Unreleased]` holds six weeks of work no consumer can get from Central: the modular
natives, the Windows generation speed-up, the Linux and Windows CPU variants, RPC, the agent,
Kolibri-1, the bump to b11556. It also holds changes that break every existing consumer at upgrade,
and the changelog states each one where it happened rather than in the one place a reader upgrading
from 5.1 would look:

- the **natives split** -- `net.ladenthin:llama` is the classes jar only, the natives are classifier
  jars collected by `llama-platform` (`<type>pom</type>`), the fat jars are gone -- so a 5.1
  `<dependency>` on `net.ladenthin:llama` resolves to a jar without natives and fails at the first
  load with `UnsatisfiedLinkError`;
- `slf4j-simple` instead of `logback-classic` as the shipped binding; the ROCm jars on ROCm 10; the
  `LazyMode` rename and the `enableFlashAttn()` rework; twelve dead builder methods removed;
  `NativeServer` on port 9931 (llama.cpp's new default) while `OpenAiServerConfig` stays on 8080
  (open item below); the floors Windows 10, macOS 15 and glibc 2.28.

Semantic versioning says that is **6.0.0**, not 5.2.0; the decision is the owner's, and so is the
timing. What the release needs either way, in order:

1. **Round 0 green through `package` and the smokes** -- the modular natives have never passed the
   stages that assemble and launch the published jars (top of this file); a release before that
   ships jars CI has not loaded, which the smoke jobs exist to prevent.
2. **An "Upgrading from 5.1" section at the top of the release's changelog entry**: the dependency
   change (`llama-platform` as `pom`, or the classifier jars by name, plus a GPU module jar where
   wanted), the logging binding, the port, the floors, the renamed and removed builder methods -- one
   place, written for someone who reads nothing else. The `[Unreleased]` bullets stay as the record.
3. **The version in every place the bump touches**: `mvn versions:set` for the five reactor poms,
   then by hand the two READMEs' snippets, `examples/jbang/Chat.java`, `examples/server/pom.xml`,
   `llama-atmosphere-agent/pom.xml` and its READMEs, the Android and Kotlin READMEs (CLAUDE.md
   "Version bump" lists them; `check-natives.py` fails on the ones it can see), then the CHANGELOG
   heading and footer links, which `../workspace/workflows/release-process.md` names as the
   most-missed step.
4. After the release: the upstream submissions (policy, below) and the `main` bump back to a
   `-SNAPSHOT` of the next version.

The longer this waits, the longer the migration section gets and the more snapshot-only behaviour
consumers discover on their own.

### ~~`package` is red: the Linux SYCL module ships the build host's oneAPI paths~~ -- fixed 2026-10-11

The next violation behind the OpenVINO one, and only visible once that was out of the way -- the
merge check runs before `verify-native-deps.py`, so `package` had never got this far. Run
38149667195, `Package JARs`: **99 native libraries checked, 1 violation**.

```
Linux/x86_64/sycl/libggml-sycl.so needs ['libggml-base.so'] next to it
but its run path is '$ORIGIN:/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/tbb/2023.1/lib/intel64/gcc4.8'
```

**What made the decision easy was counting the siblings**: of the 36 Linux libraries, **35 carry
exactly `$ORIGIN`** -- including CUDA, ROCm, OpenVINO and Vulkan, every one of which needs a vendor
runtime just as much and finds it through the consumer's own environment. SYCL is the **one
outlier**, so the thing to repair was the artifact, not the rule.

**Why the obvious CMake fix is not one:** `llama/CMakeLists.txt` already sets
`INSTALL_RPATH "$ORIGIN"` with `BUILD_WITH_INSTALL_RPATH` on the module target. oneAPI's MKL and TBB
targets hand the **linker** their install paths as `-Wl,-rpath`, and a target property cannot remove a
flag the linker was given. So the Linux SYCL job now normalises the shipped file with `patchelf
--set-rpath '$ORIGIN'` and **fails if the result is not exactly that** -- a job failure at the source
instead of a surprise two jobs later.

Shipping `/opt/intel/oneapi/mkl/2026.1/lib` to a consumer is build-host leakage of exactly the class
`verify-native-deps.py` was written to catch (the Homebrew `openssl@3` dylib that shipped in 5.0.6).

**And the check's message was wrong, not merely vague.** It reported this as *"it would be looked up
on the system instead"* -- the reason for a run path with **no** `$ORIGIN`, while here `$ORIGIN` was
present and first. It sent a reader looking for something that was not the problem, which is the same
defect as the `sdkmanager` step's "dl.google.com broken". The two cases are now separate messages,
with four unit tests (red against the old wording, green against the new).

### ~~`package` is red: the Windows OpenVINO module jar fails the merge check~~ -- done

**Done 2026-10-11 (`7870923f`, "Unblock package").** The cause was vcpkg, not the build and not the
check: with its toolchain file in use, vcpkg's app-local deployment put `OpenCL.dll` next to the
linked module in the link step itself, and `merge-native-artifacts.sh` requires a module jar to hold
exactly its module and the two `jllama-*.txt` lists. The job now passes `VCPKG_APPLOCAL_DEPS=OFF`; the
check was deliberately not relaxed (a GPU module whose runtime is absent does not load and the model
runs on the CPU modules -- the documented, smoke-tested behaviour of every other backend). Whether
`package` then goes green end to end is what round 0 shows; it had not been reached when this was
written. The original entry follows.

**This blocks every release and was untracked until 2026-10-11.** `Package JARs` fails with

    artifact 'natives-openvino-windows-x86-64' is a module jar:
    Windows/x86_64/openvino/ must hold ggml-openvino.dll (named in jllama-files.txt)

so `merge-native-artifacts.sh` rejects the artifact before anything is packaged. Because the smoke
jobs `need: package`, all of them **skip** rather than fail -- which is why the run reads as "3
failures" and the Java test jobs look green: nothing downstream of `package` ran at all. Seen on run
38112603436; the check itself came with `4f811a23` on `main`, so this is **pre-existing**, not a
consequence of the AMX work on the branch.

What to find out first: whether the Windows OpenVINO build produces a differently named module (the
other GPU jobs write `ggml-<backend>.dll`, and `JLLAMA_MODULE_ONLY` writes `$<TARGET_FILE_NAME>` into
`jllama-files.txt`, so a target name mismatch would show exactly like this) or produces nothing and
the upload's `if-no-files-found: error` did not catch it. The two GPG jobs red in the same run are
**unrelated and expected** on a branch: they need the `maven-central` environment secret.

### ~~Four defects in the published artefact set~~ -- found and fixed 2026-10-12

The first snapshot the green pipeline published was compared against `natives.csv` and against Maven
Central. **Central was complete and correct**; the **GitHub release** was not, and the three gaps were
all in one collect step that copied `*.jar`, `*.jar.asc` and `*.aar` only:

1. **no `*.pom` at all**, so `llama-platform` -- a pom-packaging module, the artifact consumers are
   told to depend on -- was absent from the release entirely;
2. **the AARs were attached unsigned**, while Central had `aar.asc` for both;
3. **the signature check globbed `*.jar` only**, which is why the job reported success while
   attaching something unsigned -- the check had exactly the hole the collect step had;
4. **`gh release upload --clobber` never removes anything**, so the rolling `snapshot` release had
   grown to **208 assets**: 109 from versions 5.0.0-5.0.7, 43 in shapes that no longer exist
   (`msvc-windows`, `all-*-jar-with-dependencies`) and three SpotBugs plugin jars an old collect step
   had swept in during 2026-05/06 -- the same three are still on srcmorph's and streambuffer's
   releases.

Fixed in #529 and #531 and **verified on the live release** after run 38218935644: **208 -> 98
assets**, every one of them `5.2.0-SNAPSHOT`, **49 artefacts with 49 `.asc`** (1:1, nothing unsigned),
**23 of 23** natives jars against the list, and `llama-platform-5.2.0-SNAPSHOT.pom` present. The
prune computes its keep-set from what the run actually uploaded rather than from a version pattern --
a pattern is what produced the 43 stale shapes -- deletes **after** the upload, runs only on the
rolling release (a `v*` release is history) and prints every deletion by name.

**Still to do in the siblings:** item 4 applies to BitcoinAddressFinder, srcmorph and streambuffer
too; all three accumulate, and two carry the same SpotBugs jars. `publish.yml#github-snapshot` and
`#github-release` are *shared jobs* there (byte-identical, hashed in `shared-files.sha256`), so a
change lands in all three plus `check-shared-files.py --write` in each. jllama's copies are not in its
own manifest and already diverge.

### Three measured gaps between "the jar is on the classpath" and "it works" (2026-10-11)

All from the first runs in which a GPU module jar was ever *used* rather than placed, and from
loading the Linux jar on an image that is not Amazon Linux. Numbers and method:
[`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md).

1. ~~**The Linux CPU jar needs `libgomp.so.1`.**~~ **Done:** `llama/CMakeLists.txt` forces `GGML_OPENMP=OFF` for Linux (non-Android), `verify-native-deps.py` enforces the absence, a unit test says why widening the allowlist again defeats it, and **only** OpenMP is touched -- `libstdc++`/`libgcc_s` stay dynamic because HotSpot links them anyway. **Still to verify in CI:** that the Linux jars come out with no `NEEDED libgomp` (the `package` job's own check now fails if they do) and one re-measurement in the gcc-toolset-14 image. The original entry follows.
   **The Linux CPU jar needs `libgomp.so.1`, and a plain Ubuntu server image does not have it.**
   **Measured 2026-10-11, and the decision is no longer a trade-off: turning OpenMP off costs
   nothing.** Upstream b11538 built twice on one 16-vCPU machine with identical flags except
   `GGML_OPENMP`, benchmarked with `llama-bench` (3 repetitions): token generation is *slightly
   faster* without it at 4, 8 and 16 threads (+0.9 %, +0.8 %, +0.7 %) and prompt processing is lower
   by 0.4-2.1 %, all inside the error bars -- and `libggml-base.so` loses its `NEEDED libgomp`. The
   ~6 % figure quoted below does not reproduce; it was one sample already flagged as inside its own
   spread. Table in [`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md).
   **Remaining work is the change itself**, not the decision: add `GGML_OPENMP=OFF` and
   `-static-libstdc++ -static-libgcc` to the Linux (and s390x) builds, extend
   `verify-native-deps.py`'s allowlist so the *absence* of `libgomp`/`libstdc++`/`libgcc_s` is what
   the check enforces, and re-measure once in CI's own gcc-toolset-14 image.
   **Why `libgomp` was a dependency at all, since this reads like a regression and is not one:** GCC's
   `-fopenmp` links `-lgomp` *dynamically*, so it was always a runtime requirement -- it simply never
   surfaced because every probe before this ran on Amazon Linux 2023, which ships it. It has nothing
   to do with `-static-libstdc++`, which does not cover libgomp.
   **And upstream is not one answer but three, per platform** (checked in their `release.yml` at
   b11538): **Android `GGML_OPENMP=OFF`** (no `libomp` on devices -- what this project already does),
   **Windows `GGML_OPENMP=ON` plus `GGML_OPENMP_FETCH=ON`** (they fetch and *bundle* LLVM's libomp),
   and **Linux unset**, i.e. ON against the distro's libgomp. Upstream's Linux choice is right for
   upstream's deployment -- a zip onto a developer's distro, where libgomp comes with the toolchain --
   and wrong for ours: a Maven dependency landing in whatever container the consumer runs, where the
   usual slim Java images have none. So there are **three** options here, not two: (a)
   `GGML_OPENMP=OFF`, measured to cost nothing and *removing* a file; (b) `GGML_OPENMP_FETCH=ON`,
   upstream parity on Windows' pattern, keeping OpenMP and *adding* a bundled `libomp.so`; (c) leave
   it and document the requirement. **(a) is the recommendation** -- the measurement says OpenMP buys
   nothing on Linux, and (a) is the only one that makes the directory smaller. Deviating from upstream
   here is the same shape of decision this project already took twice: Windows OpenMP off (measured
   1.9x/2.7x *faster* without) and the ROCm `GPU_TARGETS` going further than upstream's.
   The original wording follows for the record:
   `libggml-base.so` -- the first file the loader loads -- declares it, so the *whole* jar dies with
   `UnsatisfiedLinkError: The native library of backend 'cpu' could not be loaded`, which reads like
   a packaging defect. Every shipped Linux file also needs `libstdc++.so.6` and `libgcc_s.so.1`, so
   distroless and musl images cannot run it at all. **Decision needed:** make the Linux jar
   self-contained the way the Android build already is (`GGML_OPENMP=OFF` plus `-static-libstdc++`),
   at a measured cost of ~6% on token generation on 8 threads -- or keep the dependency and document
   it in the README. That 6% is one sample inside its own spread; re-measure it on the rented
   machines before deciding, which is cheap now that the harness exists.

2. ~~**`cuda13` does not fall back when the driver is too old.**~~ **Fixed and verified on the
   same hardware (2026-10-12).** Both CUDA jobs pass an explicit `CMAKE_CUDA_ARCHITECTURES` that
   makes `75 80 86 89 90 100 120 121` **real**, so Turing, A100 and Hopper ship native code
   instead of PTX; `121-virtual` keeps a JIT fallback for anything newer. Re-measured with the
   **same AMI, the same driver 595.91.07 and the same Tesla T4 -- only the module changed**:
   **three `exit 134` aborts became three clean runs** (including `-ngl 0`, the most absurd of
   them since no layer was on the GPU), and the GPU gives **20.6x prompt processing / 7.0x
   generation** over the CPU baseline. `cuobjdump` on the published jar confirms cubins for
   `sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120a sm_121a` plus PTX for `sm_121a`; the `a`
   suffixes are upstream's own rewrite of every `12x` architecture, not a defect in our list.
   Numbers and method: [`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md).
   **`cuda13` does not fall back when the driver is too old -- it aborts the JVM.** On a Tesla T4
   with driver 595.91.07 the module opened, ggml enumerated the GPU and offloaded 29/29 layers, and
   then `ggml_abort` killed the process: `the provided PTX was compiled with an unsupported
   toolchain`. Turing, A100 and Hopper ship as **PTX only** (`75-virtual 80-virtual 90-virtual` in
   ggml's default set; native cubins exist for `86`, `89` and Blackwell), so they depend entirely on
   the driver JIT-compiling CUDA 13.4 PTX. `abort()` cannot be caught from Java. **And there is no
   safe configuration: `-ngl 0` aborts as well** (0/29 layers offloaded, same exit 134) -- having the
   module on the classpath is enough, because the backend registers and llama.cpp uses the device
   during setup whatever the layer count says; the only escape is leaving the jar off the classpath or
   `-Dnet.ladenthin.llama.backend=cpu`. By contrast the Vulkan module on the same machine behaved
   exactly as designed: skipped for want of an ICD, throughput identical to the CPU baseline.
   **Two candidate
   fixes:** document a minimum driver version for the `cuda13` jars, and/or pass an explicit
   `CMAKE_CUDA_ARCHITECTURES` adding `75-real 80-real 90-real` so the common datacenter GPUs need no
   JIT -- at the price of binary size, the module already being 154 MB.
   **The gap is upstream's design, not a mistake, and that is exactly why deviating is justified
   here.** ggml's own comment states the policy: build *virtual* architectures "to cover all features
   needed for best performance" and *real* ones "for the most commonly used GPUs" -- and the real ones
   it picks are `86` (RTX 3000), `89` (RTX 4000) and consumer Blackwell. Upstream passes no
   `CMAKE_CUDA_ARCHITECTURES` at all and says so in a comment, so their CUDA binaries have the
   identical gap: a T4, A100 or H100 user of upstream's zip hits the same JIT path and the same abort
   on an older driver. **But their users and ours are the opposite population.** Upstream ships a zip
   to someone with an RTX card; this project ships a Maven dependency into server applications, which
   run on T4, A100, L4 and H100 -- precisely the GPUs upstream leaves as PTX. For us the "most
   commonly used GPUs" are the datacenter ones.
   **Cost, so the decision is informed:** nvcc compiles every `.cu` once per *real* architecture, so
   going from three real (86, 89, Blackwell) to six roughly doubles the device-compile half of a job
   that already takes 47-80 minutes uncached, and the module grows from 154 MB by roughly the same
   proportion. `100-real` (datacenter Blackwell, B200) would be a seventh and is rarer. There is no
   way to have both the small jar and no JIT dependency; the question is only which one the consumer
   should pay for.

3. ~~**`openvino` takes over silently and is 172x slower.**~~ **Done, and re-measured on Intel
   hardware (2026-10-12): the cost was understated.** `openvino` is in
   `LlamaLoader.OPT_IN_BACKENDS` and declared by the `optin` column of `natives.csv`, so its jar
   does nothing until `net.ladenthin.llama.backend` names it. On an `m7i.xlarge` (Sapphire
   Rapids, no accelerator) the real figures are **21.7x slower prompt processing and 336x
   slower generation** -- 481.3/33.6 down to 22.2/**0.1** tokens per second, about ten seconds
   per token -- not the 17x/172x first recorded. Two things that run settled which the earlier
   one could not: **without an OpenCL ICD the module does not open at all** (2 backends, full
   CPU speed; `ocl-icd-libopencl1` is what changes the answer), and **with the jar on the
   classpath and the property unset the loader leaves it out** and the machine runs at full
   speed -- the only hardware proof the opt-in decision has, since CI has no Intel accelerator
   and nothing happening there proves nothing. The README carries the corrected numbers.
   **`openvino` on a host without an Intel accelerator is 172x slower, and takes over silently.**
   Same CPU library, same machine: `cpu` only 490.4 pp / 34.5 tg, with the OpenVINO module present
   29.4 / **0.2** -- five seconds per token, over 27000 log lines mentioning OpenVINO. Its CPU plugin
   contributes a *device*, so llama.cpp prefers it; this is the case the "a module whose vendor
   runtime is missing is skipped" rule does not cover. It additionally needs **`libOpenCL.so.1`**,
   which the OpenVINO archive does **not** contain and Ubuntu does not install by default
   (`ocl-icd-libopencl1`) -- without it the module silently never opens, so a user can follow
   OpenVINO's own instructions and still get nothing. That dependency is documented for Windows only,
   where the OS supplies the loader. **Decision needed:** make the OpenVINO module **opt-in** instead
   of additive, or document in the README that the jar needs an Intel GPU/NPU plus an ICD loader and
   must otherwise stay off the classpath.


4. ~~**The loader's exception does not name the missing library.**~~ **Fixed.** It used to end in
   "see the messages above" and print the `dlopen` reason to `System.err` separately, so a consumer
   with only the stack trace could not tell "library missing from the jar" from "library present, one
   `DT_NEEDED` dependency absent" -- and the second is the case that happens. The reason is now
   carried into the thrown message, the way the Android branch of the same method already did it,
   and `LlamaLoaderTest` pins both halves: a failed load records a reason, and a file that is simply
   absent does not (so no stale message is reported as a load failure).

5. **A shipped diagnostic entry point, for the support cost rather than for a feature.** Today a
   consumer whose load fails opens an issue; the answer is almost always one of a handful of things
   this project can already determine -- which platform the JVM reports, which natives jars are on the
   classpath, which library backend and modules the loader selected, which `DT_NEEDED` dependency is
   absent, which CPU module ggml picked and what the backend registry ended up holding. A
   `net.ladenthin.llama.Diagnose` main printing exactly that turns a support thread into a paste.
   The pieces exist: `NativeLoadSmoke` does the load half in CI, `JNI_OnLoad` already logs the
   registry and devices at INFO, `buildcheck/nativedeps.py` parses ELF/PE/Mach-O (the Java side would
   need only the `DT_NEEDED` list, or can simply report the `dlopen` reason that item 4 now carries).
   Cheap, and it is the one item on this list that pays off per *user* rather than per release.

### Modular natives -- follow-ups

Every platform a GPU can join ships modular since 5.2.0 (CLAUDE.md "Modular natives": `JLLAMA_BACKEND_DL`
library jars, `JLLAMA_MODULE_ONLY` GPU module jars, no fat jars). What is still open:

1. **GPU modules on real hardware.** The 12 module-only jobs are build-only in CI; what no session can
   see is a module *registering a device* out of the published jars: `cuda13-windows-x86-64` and
   `vulkan-windows-x86-64` next to `cpu-windows-x86-64` on the local agent's machine (RTX 3070 + AMD
   iGPU), the loader line naming both, llama.cpp listing the devices once each (`device_id` dedup),
   all layers offloaded, and the tg128 of the CUDA module against the 265.6 the static build measured
   (`GGML_CUDA_GRAPHS_DEFAULT` is repeated in the modular path, so the 88.8 of addendum 3 of
   `docs/handover/local-agent-report-b11538-windows.md` should be gone). Device indices are not stable
   across the set of loaded modules (`Vulkan0` was the NVIDIA GPU with CUDA loaded, the AMD iGPU
   without), so any device setting must go by name, never by index -- worth a README sentence once
   measured.
2. ~~**Benchmark on AVX-512/VNNI/AMX hardware**~~ **-- done 2026-10-11, see
   [`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md).** All 14 x86
   modules now have numbers from one Granite Rapids machine (the first on which every level loads),
   plus a Zen 4 machine and the Graviton 4 run. **AMX is worth +148 % on prompt processing** over the
   Haswell baseline, and the plain `x64` module is **66x** slower than the chosen one, not the 10.4x
   the Zen 3 figure suggested. The finding that outgrew the item: on all three machines
   `ggml_backend_load_best` gives up **10-11 % of token generation** by picking the richest module,
   and on aarch64 it also gives up 27 % of prompt processing -- it scores by feature level, not by
   measured throughput. That belongs upstream (see "Upstream submissions"), not in a downstream
   override. The original wording follows for the record:
   **Benchmark on AVX-512/VNNI/AMX hardware** (Sapphire Rapids, Zen 4/5, Core Ultra). The Zen 3
   measurement could only show the floor: the plain x86-64 module is 10.4x slower at prompt
   processing (313 -> 30 t/s, Qwen3-0.6B Q4_K_M) and 1.6x at generation than `haswell`, which is what
   the single-level build was. The gain upwards (`zen4` BF16, `sapphirerapids` AMX) is unmeasured.
3. **Unsigned binaries.** Upstream ships its DLLs unsigned, and so do we; on a WDAC/AppLocker-managed
   client, loading unsigned DLLs from `%TEMP%` is blocked. The README's troubleshooting section now
   says so and names the way out (`net.ladenthin.llama.lib.path` to an allow-listed directory). The
   project's GPG key cannot sign them: Windows checks Authenticode signatures (an X.509 code-signing
   certificate, `signtool`/`osslsigncode`), not OpenPGP. **Owner's decision**, not scheduled: a
   code-signing certificate (Azure Trusted Signing or an OV/EV certificate, both paid, identity-
   validated) plus a signing step over every `.dll` before `package`.
4. **CUDA for Windows arm64 (`cuda13-windows-aarch64`) -- upstream builds it, we do not.** Their
   `windows-cuda` job is a matrix over `x64` and `arm64`, the latter through
   `cmake/arm64-windows-msvc-cuda` plus the arm64 `cudart`/`cublas` redistributables. This project has
   `cuda13-windows-x86-64` only. The arm64 Windows toolchain already exists here (the `cpu` and
   `vulkan` jars for that platform), so the missing piece is the CUDA toolkit's arm64 components and
   the toolchain file. Niche hardware today (Windows-on-ARM with a discrete NVIDIA GPU), but it is a
   straight coverage gap against upstream rather than a judgement call, and the "support more rather
   than fewer" policy that governs the ROCm targets applies. **Not** a gap: Linux aarch64 CUDA
   (Jetson, Grace-Hopper) -- upstream does not build that either, so it would be new ground.

5. **A Vulkan module for Android** (`vulkan-android-aarch64`, upstream ships none either): `JNI_OnLoad`
   already tries `libggml-vulkan.so` by soname, so it is one more module-only dockcross job plus the
   natives-jar wiring and an AAR flavor -- if anyone asks.
6. **The Windows ROCm job is the pipeline's critical path, and it builds uncached.** **Measured in run
   38149667195: both ROCm jobs were still compiling after 2 h 23 min**, with every other build job
   finished (50 of 56 green) and `Package JARs` not started -- so everything behind `package`, which
   is the whole point of a round-0 run, waits on them. The dllimport flood above sits in the same job
   but is not the cause: 23 GPU targets' kernels are. Suppressing the warnings makes the log readable,
   not the build faster.
   The original entry follows:
   **The Windows ROCm job is the pipeline's critical path, and it builds uncached.** Run 38069595101
   (the second full run of the modular natives): 4:15 h for its build step, against 2:00 h for the
   Linux ROCm job and 1:41 h for the Windows CUDA job; every other Windows build is under 20 min. Of
   its 158 ninja steps ~140 are HIP TUs, each compiled for the 23 `GPU_TARGETS`; on Windows upstream
   compiles them as C++ with TheRock's clang, a plain clang, so they go through sccache -- and sccache
   0.18.0 refuses every one of them for the `-Xclang` that CMake puts on any plain-clang Windows TU
   (CLAUDE.md "sccache on every Windows Ninja job but one"; the job's stats table: 67 requests, all
   non-cacheable, 0 compilations). What would shorten it: a sccache that accepts `-Xclang` -- re-test
   on every `SCCACHE_DL_VERSION` bump, the stats table in the job summary answers at once, and do it
   on a branch with one dispatched run, because `build.bat`'s unconditional retry turns a sccache that
   accepts the flag but mishandles the 23-target `--offload-arch` compile into a second 4 h build --
   or fewer `GPU_TARGETS`, against the policy of building every target TheRock builds (CLAUDE.md
   "Additional GPU-backend natives").

### The FetchContent nlohmann/json is shadowed by upstream's vendored copy

Measured at the b11556 bump (`llama/build/CMakeFiles/jllama.dir/flags.make`): `llama.cpp-src/vendor/nlohmann/..`
is include dir number 7 on every `jllama` compile line and the FetchContent `json-src/include` number 12, so
`log_helpers.hpp` and `train_engine.cpp` -- the two project TUs that `#include <nlohmann/json.hpp>` directly --
compile against upstream's **vendored** copy, as every upstream TU does. That is the good case (one copy per
TU, no ODR mix), and it became visible because b11556 (#30253) applies `vendor/nlohmann/json-deep-nesting.patch`
to that copy, so it is no longer byte-identical to the `v3.12.0` our `FetchContent_Declare(json ...)` fetches.
The FetchContent copy is therefore dead weight: a clone per fresh build directory, a `nlohmann_json` link on
`jllama` and `jllama_test`, and a tag (`v3.12.0`) that looks like something to keep in sync. **To do:** drop
the FetchContent block and the two `nlohmann_json` links, include the vendored copy through `llama-common`'s
interface (what already happens), and keep one guard: a configure-time check that exactly one `nlohmann/json.hpp`
is on `jllama`'s include path, so a future upstream move of the vendor directory cannot silently flip the order
back. Not done in the bump PR on purpose -- a bump changes the pin, nothing else.

### Waiting on sccache: four open issues, and they cost the pipeline hours

**Nothing to do here but watch.** Checked 2026-10-12: `SCCACHE_DL_VERSION` is **0.18.0 and that is the
newest release** (2026-09-14), so there is no bump to take, and the cause is a family of **open**
issues upstream:

| issue | what it is |
|---|---|
| [mozilla/sccache#1122](https://github.com/mozilla/sccache/issues/1122) | *"Generically assume that unknown flags are `PassThrough`"* -- **the structural fix**: it would make every one of the others moot |
| [#1603](https://github.com/mozilla/sccache/issues/1603) | handling `-Xclang` custom plugins and flags |
| [#2522](https://github.com/mozilla/sccache/issues/2522) | `-Xclang -fopenmp` errors with `UnknownFlag` |
| [#2717](https://github.com/mozilla/sccache/issues/2717) | add `-fno-cxx-modules` to the known `-Xclang` flags |

**Why this is not cosmetic.** sccache refuses any translation unit carrying an `-Xclang` argument, and
CMake's `Platform/Windows-Clang.cmake` puts one on **every** plain-clang Windows TU, because that is
how it implements `CMAKE_MSVC_RUNTIME_LIBRARY` -- the hybrid CRT every Windows job here sets
(`-Xclang -flto-visibility-public-std -D_MT -Xclang --dependent-lib=libcmt`). So two jobs cache
nothing at all:

* **`build-windows-x86_64`** (plain clang, modular CPU): measured at b11538, **719 of 720 compile
  requests non-cacheable, 0.00% hit rate**. The job builds cold on every run, which is why it has no
  sccache install step at all.
* **`build-windows-x86_64-rocm`**: TheRock's clang is a plain clang too, and upstream compiles the HIP
  sources as C++ on Windows, so its ~140 HIP TUs -- each compiled for **23** GPU targets -- all go
  through the launcher and are all refused (run 38069595101: 67 requests, every one non-cacheable, 0
  compilations). **This is the pipeline's critical path**: 2:40 h to 3:08 h for its build step in the
  measured runs, against ~1 h for everything else in the run put together. A pipeline run is
  three hours because of this one job, and a working cache is the only thing that would shorten it.

**What is already in place, so nothing is blocked meanwhile:** `build.sh`/`build.bat` probe-compile
through sccache before trusting it as the launcher, retry once without it when a build fails with a
cache error, and pass `-DGGML_CCACHE=OFF` so ggml cannot re-enable a launcher behind the retry's back.
The refusal therefore costs time, never a red build.

**The re-check, when it is worth doing.** On the next sccache **release** (not on a schedule): bump
`SCCACHE_DL_VERSION` in `build.sh`, add the install step back to `build-windows-x86_64`, and read the
`### sccache statistics` table the build script writes into the job summary. A hit rate above zero on
that job is the answer; anything else and the pin goes back. The ROCm job keeps its install step
already, so its summary table answers the same question for free on any run.

**Worth one comment upstream** rather than a new issue: this project's case -- CMake putting `-Xclang`
on every Windows TU, so the flag is not exotic but unavoidable -- belongs in #1122, which is where the
general fix is being discussed. That is the only action available until something merges.

### CUDA job: nvcc through sccache (`SCCACHE_WRAP_NVCC`, off since the second failure)

**Checked 2026-10-12: there is nothing to bump to.** `SCCACHE_DL_VERSION` is 0.18.0 and that is the
newest release (2026-09-14). The `-Xclang` refusal that also costs the plain-clang Windows jobs their
cache is tracked in open sccache issues (#1603, #2522, #2717, #1122), none merged. So the
re-test recipe below is still the right one -- it just cannot be run yet. Re-check on the next
sccache release rather than on a schedule.

- Run 37680715063 (b11476 bump) and run 37925157175 (#492, 2026-10-09): the Linux CUDA build logged
  `sccache: caused by: Missing "cubin" file output` and `sccache: Compiler killed by signal 126` on
  the first three `.cu` TUs, one minute into the build, and `build.sh`'s retry rebuilt everything
  without any launcher (green; 86 min in total the second time, longer than a cold build). Not the
  CUDA 13.3 `-virtual` `.ptx` failure, and not a storage error (the gcc probe passed and the gcc TUs
  were being served): sccache 0.18.0 and CUDA 13.4's nvcc disagree on the device-compile command
  line, in every job log examined since the 13.4 bump of 2026-09-27 (also runs 37643964068 and
  37831180985; the retry alone takes 47-80 min depending on the runner, and no job came near the warm
  ~15 min). Since the second confirmed occurrence `build.sh`
  wraps nvcc only with `SCCACHE_WRAP_NVCC=true` (CLAUDE.md "Fast local CUDA builds"), so the gcc TUs
  stay cached and nvcc runs directly from the start. **Open:** re-test the opt-in with a newer
  sccache (bump `SCCACHE_DL_VERSION` on a branch, set the variable on the CUDA job for one PR run,
  look for CUDA hits in the stats table) and flip the default back when a warm run shows them -- the
  win was ~51 -> ~15 min with CUDA 13.2.

### RPC backend — follow-ups

- **A server lost mid-inference still aborts the JVM.** Patch `0015` makes *registration* fail
  softly; every call after it (`get_dispatcher()`, `RPC_STATUS_ASSERT` in the dispatcher's `work()`)
  still ends in `GGML_ABORT`, because the ggml backend interface has no error return for a lost
  device (`graph_compute` returns a status, but buffer `set/get_tensor` are `void`). A real fix is an
  upstream change: carry a failed-state flag through the dispatcher, fail the pending futures, and
  surface it as a `GGML_STATUS_FAILED` at the next `graph_compute`, which llama.cpp already turns into
  a decode error. File upstream first; do not carry it downstream.
- **A served device that cannot run an operation aborts the server process.** ggml-rpc's client
  answers every `supports_op` with `true` (upstream `//TODO: call the remote backend and cache the
  results` in `ggml_backend_rpc_device_supports_op`), so the scheduler never falls back and the
  server hits `GGML_ABORT("unsupported op")` in the device's graph compute. Seen on the macOS CI
  runners, whose paravirtual Metal GPU has no `MUL_MAT`: an in-JVM `RpcServer` serving it takes the
  JVM down on the first inference. Mitigated, not fixed: `RpcServer`'s device list (`--device CPU`)
  and the tests serve the CPU. The fix is upstream — forward `supports_op` over the protocol (a new
  command, with a per-op cache on the client) — and belongs there, not in `0015`.
- **Android RPC is untested on a device.** Bionic sockets build (upstream ships RPC in its Android
  release too), but the app needs `android.permission.INTERNET` even for loopback, which the AAR
  deliberately does not declare and the emulator fixture does not have. A loopback test on the
  emulator needs that permission in the fixture's manifest only.
- **RPC smoke on the other smoke-set platforms.** `smoke-rpc.sh` runs in the `linux-x86-64` row
  of the `smoke-natives` matrix only (widening it to `linux-aarch64` is a change of that step's `if:`;
  Windows needs a PowerShell port of the script); the Java `RpcServerTest`/`RpcIntegrationTest` already run on every `test-java-*` job
  (Windows and macOS included), so this is about the published jars, not the code path.
- **Authentication / TLS.** Upstream has none; the documented answer is a trusted network or a
  tunnel. Only worth doing if it lands upstream.
- **RDMA transport** (`GGML_RPC_RDMA`) as its own classifier, since it needs `libibverbs` at runtime.
- **Several clients at once.** Upstream's server serves one connection at a time.
- **Server-to-server comm (`-sm tensor`, llama.cpp b11450) widens what a client can make the server do.**
  #26610 lets a client tell two RPC servers to form a pair (`RPC_CMD_COMM_INIT`): the rank-0 server
  then *listens on `0.0.0.0`* on a port the client names (`socket_t::create_server("0.0.0.0", port)`
  in `rpc_server::comm_init`) and blocks in `accept()` until the rank-1 server connects. Two
  consequences for the in-JVM `RpcServer`: (1) `startLocal` binds loopback only, but a local client can
  still open a listener on every interface; (2) `close()` cannot end that wait --
  `ggml_backend_rpc_stop_server()` shuts down the *client* socket and wakes the *main* listener, not
  the comm listener -- so if the peer never connects, the server thread and `close()` hang. Not
  reproduced; found reading the diff at the bump. Fix candidates for `0015`: bind the comm listener to
  the server's own host and register it with the stop machinery so a stop also wakes it.

### Atmosphere coding agent (`llama-atmosphere-agent/`) — follow-ups

The headless loop is verified, including the model-backed CI job (run 35600558852: tool call
answered, read→write→read loop changed the file). Still open:

- **Tool rounds across REPL turns ride as a note, not as `tool_calls` messages -- and that cannot
  change on this Atmosphere.** The earlier entry here asked to keep the full `ChatMessage` list
  (incl. `tool_calls`/`tool`) per turn; measured against `atmosphere-ai` 4.0.72, that is impossible:
  `AbstractAgentRuntime.assembleMessages` rebuilds every history entry as
  `new ChatMessage(h.role(), h.content())` (checked in the bytecode), so the tool-call array and the
  tool-call id never leave the framework. What ships instead: `AgentSession.toolNote` carries each
  turn's calls and results (cut at 400 chars) in front of the next user message, pinned by
  `LocalAgentTest.aToolCallStaysInTheHistorySoTheNextTurnSeesItHappened`; the full placement
  argument is in `llama-atmosphere-agent/CLAUDE.md`, point 8. Reopen only with an Atmosphere
  release whose `assembleMessages` preserves tool calls (then a protocol-faithful replay is the
  better form, and the note goes).
- **In-stream engine errors are swallowed by Atmosphere** (pinned in
  `AtmosphereWireContractTest.midStreamEngineFailureCompletesSilentlyRatherThanErroring`): an SSE
  `data: {"error":…}` after HTTP 200 is ignored by `OpenAiCompatibleClient.processSSELine` (it reads
  only `choices[0]`). Worth an upstream PR to Atmosphere; until then the console shows an empty turn.
- **Spring Boot `@Agent` variant** (WebSocket/SSE UI via `atmosphere-ai-spring-boot-starter` and
  `LLM_BASE_URL`) is expected to work on the same runtime but is not CI-covered; a smoke that boots
  the starter against the scripted `OpenAiCompatServer` would close that.
- **Anthropic Messages surface.** The server also speaks `/v1/messages`; the Anthropic adapter
  (`org.atmosphere:atmosphere-anthropic`) was not tested against it.
- **Model recommendation table** for the agent (which local GGUFs actually complete an
  edit→build→test loop) — needs a GPU host, not CI.

### OpenAI-compatible HTTP endpoint — open follow-ups (Java transport; deprioritized)

The `OpenAiCompatServer` surface itself is shipped (routes, protocol translations, integration
round-trips — see CLAUDE.md "Two server modes"). **Owner priority: the native-transport
`NativeServer` comes first; Java-transport-only items below are deliberately deprioritized.**

- **Streaming raw-completion remainder:** (a) streaming `POST /v1/completions` is DONE; remaining are
  (b) token-streaming Ollama `/api/generate` (translate `text_completion` chunks to NDJSON, mirroring
  the chat→Ollama translator) and (c) Continue's native `POST /completion` route in the llama.cpp-native
  streaming shape (`{"content":…,"stop":…}` per chunk). Java-only server wiring.
- **Future *output* modalities (audio / image) — design note, not yet actionable.** llama.cpp's server
  produces text (plus embeddings/rerank) only; the integration points are isolated (a new
  `OpenAiBackend.stream*` primitive + `OpenAiSseFormatter.*Chunk` per modality). Two future hooks:
  the existing `TextToSpeech` (Qwen3-TTS since llama.cpp b10270 — OuteTTS no longer exists upstream)
  behind an `/v1/audio/speech`-style route; proxying image/audio generation to an external model.
  Keep chunk formatters modality-neutral.
- **Incremental tool-call streaming on the alternative surfaces.** Ollama/Anthropic/Responses emit each
  tool call whole at end-of-stream (`ToolCallDeltaAccumulator`); revisit only if a client needs
  incremental `input_json_delta` / `function_call_arguments.delta` fidelity.
- **Per-model FIM template registry** — only needed if `/v1/completions`-with-`suffix` FIM is exposed;
  `/infill` applies the model's FIM tokens server-side, so low value.
- **Multi-model registry (Java transport).** The native surface has this via router mode +
  `RouterClient`; the Java `OpenAiCompatServer` still advertises/serves a single model id.
- **Manual real-client validation.** Server-side round-trips exist for every surface; what remains is
  pointing the actual editor clients (Copilot Ollama provider / Custom Endpoint, Claude Code, a
  Responses client) at a running server, since round-trips confirm wire shapes but not each client's
  parser.

### Code-sign the Windows DLLs (Azure Trusted Signing, owner's decision -- it costs money)

**What.** Microsoft's **Azure Trusted Signing** (formerly Azure Code Signing) signs Windows PE files
from CI without a hardware token, on a monthly subscription -- the **Basic** tier is in the ~10
USD/month range and includes a signature quota; **verify the current price and quota before
committing**, and whether the quota covers signing *18 files per Windows natives jar* on every run
(`jllama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll` and the 14 `ggml-cpu-<level>.dll`, plus
one module per Windows GPU jar). The traditional alternative -- an OV/EV certificate -- is several
hundred euros a year **and** has required a hardware token since 2023, which a GitHub runner cannot
use, so it is not a real option for this pipeline.

**Why it is worth considering here specifically.** This project's DLLs are **extracted from a jar
into a temp directory at run time and loaded with `System.load`**, which is exactly the shape
antivirus and EDR products treat as suspicious: an unsigned native library appearing in `%TEMP%` and
being loaded by `java.exe`. A signature does not silence a SmartScreen installer prompt (there is no
installer), but it is what lets an enterprise allowlist the files by publisher instead of by hash,
and it removes the "unknown publisher" class of false positive that a consumer cannot debug. There
is no evidence of a user hitting this yet -- so this is prevention, not a fix, which is why it is a
decision rather than a task.

**The gatekeeper, and the reason to check it first:** Trusted Signing requires **identity
validation**. An organisation needs a verifiable legal entity; an **individual** must have a
verifiable public history (Microsoft has required ~3 years), and the account must be in a supported
region. **Find out whether the account qualifies before paying for anything** -- the subscription is
useless without a validated identity, and that is the step that fails, not the plumbing.

**How it would hook into GitHub.** `azure/trusted-signing-action` with federated OIDC credentials
(no long-lived secret), run in the Windows build jobs *after* the build and *before* the artifact
upload, so what gets packaged is already signed. Note what that touches on our side: signing
**modifies the files**, so it must happen before `merge-native-artifacts.sh` and
`package-smoke-sets.sh` -- both compare files byte for byte -- and the signature must survive the
jar round trip (it does; a PE signature is part of the file). Signing *after* packaging would mean
re-writing jars and re-checking them, which is strictly worse.

**What it does not cover.** Linux has no equivalent and needs none. **macOS is a separate purchase
and a separate problem**: a Developer ID certificate (Apple Developer Program, ~99 USD/year) plus
notarisation, and the dylib is already ad-hoc signed by the linker -- which is what
`smoke-natives-macos` verifies with `codesign --verify --strict` after a corrupt dylib shipped in
three releases. Do not bundle the two decisions; the Windows one is cheap and self-contained.

### License compliance gate (owner's decision)

The FSFE REUSE check is green on every run. A FOSSA-style *dependency-license* gate (the scanner
whose "17 issues found" status once posted on PRs #248/#298; it posts nothing on `main` today) was
never set up deliberately. Whether the project wants one, with which policy, is the owner's call;
nothing here blocks on it.

### ~~Drop patch `0012` at the first bump that carries #29677~~ -- done at the b11632 bump

**Done at b11632** (the first number carrying the fix is b11626, `df18bee11`; the bump went to master's head of the day, b11632, pinned through `JLLAMA_GIT_COMMIT` ahead of the tag). The applier fired as predicted in reverse: `0012` does **not** apply at `df18bee11` (upstream's line sits inside the block the patch replaces), so the drop was forced, not silent. The guard went with it -- see the `0012` note in `docs/history/dropped-llama-patches.md` for why no replacement exists. The text below is the plan as it stood, kept for the reasoning.

**The drop-check of `0012` fires at the first bump to a tag that contains `df18bee11` -- not at b11612, and not at b11617 either** (checked: no tag up to b11618 carries it; b11617 still divides by `split_sum` unguarded at `src/llama-model.cpp:1648`). On 2026-10-14 upstream
merged [ggml-org/llama.cpp#29677](https://github.com/ggml-org/llama.cpp/pull/29677), commit
[`df18bee11`](https://github.com/ggml-org/llama.cpp/commit/df18bee11e96557b07aa7b12db0976f3c7ac9165)
("model: split layers evenly when no device reports free memory"), one line in `load_tensors()`:

```cpp
splits[i] = split_sum != 0.0f ? splits[i] / split_sum : float(i + 1) / n_devices();
```

That is the even-split fallback `patches/0012` carries in `llama_model_splits_normalize()`, 15 commits
**after** b11612 (`10da8c463`) on master -- at b11612 `src/llama-model.cpp:1648` is still the bare
`splits[i] /= split_sum` (measured on the pristine tag), so the patch is still carried at this pin and
applies unchanged. The PR touches `src/llama-model.cpp` only: no warning, no diagnostic on the device
lookup, **no test**.

**The rule for the bump that crosses it:**

1. `git grep -n split_sum src/llama-model.cpp` at the new tag. The guarded line is there → **drop
   `0012`, do not refresh it.** The applier will not say so: `0012` applies *next to* upstream's line
   (its hunks lift the two blocks out of `load_tensors()` and touch other lines), so it keeps applying
   cleanly -- the `0010` case, "still applies, no longer needed". Delete the patch file; its `CLAUDE.md`
   row moves to `docs/history/dropped-llama-patches.md` with the note below.
2. What goes with it, deliberately: the `LLAMA_LOG_WARN` naming the cause, the bounds check in
   `llama_model_splits_select_device()` whose message names layer, index and split points (upstream keeps
   `devices.at()`, so a malformed split still reads `"vector"`), and the never-compiled upstream
   `tests/test-model-split.cpp`. None of that is worth a patch on its own. One thing upstream's line does
   **not** cover, for the record: a NaN weight (`-ts nan,1` -- `std::stof` accepts it, `NaN != 0.0f` is
   true) still produces NaN split points and the bare `"vector"`; `0012`'s message named it. If that ever
   matters it is a separate upstream issue, not a reason to keep the patch.
3. **The guard `src/test/cpp/test_model_split.cpp` cannot stay as it is.** It links
   `llama_model_splits_normalize` and `llama_model_splits_select_device`, which exist only through the
   patch -- without it `jllama_test` fails to link on every platform. That was the design (a dropped
   patch reds `C++ Tests`), and at the drop it works against us; and upstream shipped no test the guard
   could be re-pointed at. So the guard is replaced by one over upstream's contract, like `0010` / `0011`:
   - **Preferred: an end-to-end test through the real load path**, the only public route to the fixed
     line. The backend-independent trigger the `0012` row documents is `--tensor-split 1,-1`, which needs
     **two** devices: two loopback RPC servers serving the CPU (`test_rpc.cpp`'s `loopback_server`)
     register two GPU-type devices; load a tiny GGUF written with the `gguf` API (as `test_kolibri1.cpp`
     does) with `tensor_split = {1, -1}` and every layer offloaded. At an unfixed tag the load fails
     (`std::out_of_range` from `devices.at()`, the bare `"vector"` in the log); at a fixed one it loads
     with the layers split evenly. That pins upstream's line on every platform in `C++ Tests` without a
     patch. **Verify first** that two `ggml_backend_rpc_start_server` instances coexist in one process
     (`0015`'s stop mechanism is global state); if they cannot, this option is out.
   - **Otherwise the guard goes with the patch**, and the drop note says why no unit-level guard is
     possible (the line sits inside a member function with no seam). Do **not** keep a refactor-only
     residue of `0012` just so the test keeps linking: that is a patch to refresh on every bump for a line
     upstream owns, the exact thing the drop rule exists to avoid.
4. `patches/0012`'s header and `test_model_split.cpp`'s header carry the same pointer to this section.

### Upstream submissions -- not before the release (policy)

**Nothing is filed upstream before the release is out**, neither issues nor PRs; the drafts are not
written before that either. Everything below is the inventory for that day, so it is not rediscovered.

#### The four issues to file, prepared

These are the ones with **no patch to offer** -- a measurement and a question. The work left on each is
writing it; the evidence exists and is linked. File them as **four separate issues**, since they touch
three different subsystems and a combined report gets triaged as one.

| # | title to use | the evidence to paste | the question for the maintainers |
|---|---|---|---|
| 1 | `ggml_backend_load_best picks the richest CPU module, not the fastest` | the three tables in [`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md): on Granite Rapids the pick gives up 10-11% of token generation, on Graviton 4 **27% of prompt processing** as well | scoring is by feature level by construction. Is a throughput-aware choice wanted at all, or should a consumer be able to override the pick? There is no downstream lever today -- `net.ladenthin.llama.backend` cannot select a *CPU* module |
| 2 | `ggml-rpc: a server lost mid-inference aborts the process` | `patches/0015`'s header plus the RPC section of `CLAUDE.md`: registration was made non-aborting downstream, every path *after* registration still ends in `GGML_ABORT` | the ggml backend interface has no error return for a device that went away. Is adding one in scope, or is aborting the intended contract for a lost device? This is the blocker for using RPC from a long-lived host process |
| 3 | `ggml-rpc: the client answers every supports_op with true` | the RPC section of `CLAUDE.md`: a served device that lacks an op **aborts the server**. Reproduced on the macOS CI runners' paravirtual Metal GPU, which has no `MUL_MAT` and took the first run down; it is why every test here serves `CPU` by name | upstream's own TODO comment admits it. Should the client forward the query to the server, or should `rpc-server -d` refuse a device whose op set it cannot report? |
| 4 | `windows-rocm: 83,362 compiler warnings, against 9 on linux-rocm` | the table below in this section (714,345 log lines against 2,455; the per-class counts; all of them in `ggml/src/ggml-cuda/*.cu`) | the cause is the language, not the arch count: HIP sources are compiled as plain C++ on Windows (`CXX_IS_HIPCC` on `WIN32`) and as HIP on Linux. Fix the warnings, or the C++-on-Windows choice? Their own `windows-rocm` job pays the identical cost |

#### ~~A fifth: the ARM fp16 conversion~~ -- FILED as llama.cpp#30505

**Done 2026-10-14.** [ggml-org/llama.cpp#30505](https://github.com/ggml-org/llama.cpp/pull/30505),
one commit, `+13/-3` in `ggml/src/ggml-cpu/simd-mappings.h`, open. Carried downstream meanwhile as
`llama/patches/0019-ggml-cpu-select-the-neon-fp32-to-fp16-conversion.patch`; the full measurement
record is [`docs/benchmarks/arm-fp16-conversion.md`](docs/benchmarks/arm-fp16-conversion.md).

**Rebased the same day, at the maintainer's request, and the rebase changed the change.** `#30343`
(`685f70894`) landed 26 minutes before that request and takes back half of `#30270`: F16C keeps
`fp32->fp16` and `fp16->fp32` goes back on the lookup table, because clang lowers the intrinsic into
the live accumulator register of the quantized dot products and serialises the loop (45-65 % slower
Q8_0/Q4_0 `MUL_MAT_ID`, measured by its author). Two consequences, both measured here with `clang -E`
before anything was pushed: the guard's `FP16_TO_FP32` half would have **fired on every x86 F16C
build**, so it carries `&& !defined(__F16C__)` with the issue link on the condition line; and the
F16C *move* was dropped from the PR, since the block is now a documented exception rather than a
stray placement. A third thing fell out of it -- the guard's position is no longer merely the better
of two, because directly after the `#elif` chain it does not build at all on x86+F16C (#30343 leaves
the F16C selection *below* the lookup-table declarations). The downstream patch is the rebased diff
byte for byte and applies at the pin **and** on master.

What remains to do here is the **drop-check on every llama.cpp bump**, which the patch header states:
`git grep -n "GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16" ggml/src/ggml-cpu/simd-mappings.h`
at the new tag -- if the `__ARM_NEON` branch carries that line the PR merged, and the patch is
**dropped rather than refreshed**.

**Two rules for the day.** State the measurement and ask the question; do not propose the fix for 1-3,
because each is a design decision in code this project does not own. And say plainly what was done
downstream instead (the opt-in, the non-aborting registration, the `-w` on `ggml-hip`), so a
maintainer can see what a merged change would let this project delete.

A fifth report is **not** for llama.cpp: sccache refuses every TU carrying an `-Xclang` argument, and
CMake's runtime-library selection puts one on every plain-clang Windows TU. Checked 2026-10-12:
**0.18.0 is the newest release** (2026-09-14), so there is no bump to take, and the flag handling is a
family of **open** sccache issues -- mozilla/sccache#1603, #2522, #2717 and #1122 (the structural one:
treat an unknown flag as pass-through). So the measurement stands as recorded and the only action is
to add this case to one of those issues, or wait. Re-check on the next sccache release before
spending a CI run on it.

There are **twelve** patches today (`0001`-`0003`, `0006`-`0008`, `0014`-`0019`). Each
accepted upstream change deletes a patch from the bump checklist once the pin is bumped past it.

- **Submittable as they are** (self-contained, no jllama specifics): `0001` (Windows arg-parse embed
  guard, already reported as ggml-org/llama.cpp#26416 with the reproducer in
  `docs/upstream-investigation-win32-argv-substitution.md`; the maintainers have not picked a
  direction), `0002` (preserve the caller's load-progress callback), `0006` (embeddable
  `llama_server`), `0007` (`llama_server_attach`), `0008` (`LLAMA_SERVER_WORKER_CMD`), ~~`0012`~~ (zero
  split-sum guard -- **fixed upstream independently on 2026-10-14 by #29677**, nothing left to submit;
  dropped at the first bump that carries #29677, see the section right above), `0014` (`common_log_set_callback`), `0015`
  (non-aborting RPC registration, stoppable server, transport fixes), `0017` (x86 prefetch helper),
  **`0018`** (the AMX buffer released with `free()` instead of `ggml_aligned_free` -- one line, with
  the 6/6-against-0/6 measurement in its header and a configure-time guard that is red without it;
  the clearest submission of the set, since the defect reproduces on any AMX machine and the fix
  copies the line `ggml-backend.cpp` already has).
- **Permanent:** `0003` carries upstream PR #22393, which upstream closed without merging.
- **Temporary carry, not a submission:** `0016` (Kolibri-1). Upstream will add the architecture itself
  ([ggml-org/llama.cpp#29922](https://github.com/ggml-org/llama.cpp/issues/29922)). Drop it on the
  first bump whose tag registers `kolibri1` (`git grep -n kolibri src/llama-arch.cpp`) and keep
  `src/test/cpp/test_kolibri1.cpp`: its numerical comparisons must stay green (red there is a finding
  about upstream, not a test to adjust), its GGUF-format rows follow upstream's converter -- a red row
  means the published GGUFs of that dialect stop loading without the patch, which is decided
  deliberately (a small compatibility patch, or a note that those files must be reconverted). Open
  verification gaps: no run of the real 78B model (60 GB of shards need more than the 63 GB of RAM
  the measuring machine had) and no GPU backend.
- **Two upstream-only defects worth an issue on that day, no patch here:** a lost RPC server
  mid-inference still aborts the process (the ggml backend interface has no error return for a lost
  device), and ggml-rpc's client answers every `supports_op` with `true`, so a served device without
  an op aborts the server (both under "RPC backend" above). And sccache 0.18.0 refuses any TU that
  carries an `-Xclang` argument, which on Windows every plain-clang TU does -- CMake's runtime-library
  selection puts it there -- so no plain-clang Windows build caches at all (CLAUDE.md "Windows
  natives"); that one is an sccache issue, if none is open.
- **A fourth, measured 2026-10-11 and again 2026-10-12: the Windows HIP build produces 83,362
  compiler warnings, and the Linux one produces 9.** The dllimport note below was the first
  symptom; the full measurement came from run 38172107721, where the Windows ROCm job's log was
  **714,345 lines with 83,362 warning messages** against **2,455 lines and 9 warnings** in the Linux
  ROCm job -- the same upstream sources, the same TheRock clang, the same 23 GPU targets. It is
  **60x larger than the next biggest job of that run** and carries **98.5%** of all the warnings its
  Windows jobs emit:

  | job | log lines | warnings |
  |---|---:|---:|
  | **Windows ROCm/HIP** | **714,345** | **83,362** |
  | Windows SYCL | 11,981 | 724 |
  | Windows CPU (clang) | 2,627 | 407 |
  | Linux CUDA | 3,266 | 9 |
  | Linux ROCm/HIP | 2,455 | 9 |
  | C++ Tests Windows | 1,655 | 66 |
  | Windows arm64 | 1,499 | 24 |
  | Windows CUDA | 1,068 | 0 |

  The cause is **the language, not the arch count**: upstream compiles the HIP sources as plain C++
  on Windows (`CXX_IS_HIPCC` on `WIN32`) and as the HIP language on Linux, and clang's C++ frontend
  reports what the HIP one does not -- `-Wsign-compare` 29,673 times, `-Wnested-anon-types` 17,964,
  `-Wmissing-noreturn` 16,378, then unused variables, missing braces, unused parameters, cast-qual
  and unreachable breaks. **Every one is in `ggml/src/ggml-cuda/*.cu`**, upstream's own code: not a
  line of ours. Mitigated downstream with `-w` on the `ggml-hip` target, Windows only, which
  subsumes the dllimport suppression; the Linux job keeps every warning, so the information stays
  where it is legible. **Worth an upstream issue with this table** -- the fix is theirs (either the
  warnings or the C++-on-Windows choice), and their own `windows-rocm` job pays the same cost.

- **The original dllimport note, which is one of the eight classes above: `ggml.h`'s `dllimport`
  form reaches the HIP *device* pass.** `GGML_API` expands to `__declspec(dllimport) extern` for a consumer of a shared
  ggml on `_WIN32`, and HIP compiles every translation unit once per GPU architecture for the amdgcn
  **device** target, which has no DLL storage class -- so clang reports
  `__declspec attribute 'dllimport' is not supported` for each `GGML_API` declaration, times the 23
  Windows GPU targets. Counted in run 38149667195: **at least 1,374,762 warning lines** in the Windows
  ROCm job, against 0 in the CUDA job and 8 in the SYCL one. That is a **floor rather than a total**,
  and how it was found is worth keeping: two counts of the same log disagreed, with the *narrower*
  pattern returning more matches than the broader one read later -- impossible unless the log was
  still growing, which it was, because the job had not finished. A count off a running job is a
  snapshot; say so or re-count at the end. Nothing is broken (the attribute applies on the
  host pass and is meaningless on the device one), but it makes the log of the pipeline's
  **critical-path, uncached** job unreadable for a real error and large enough that GitHub truncates
  it. Suppressed downstream on the `ggml-hip` target only (`-Wno-ignored-attributes`, Windows only --
  see `llama/CMakeLists.txt`), which is a mitigation rather than the fix: **the fix belongs in
  `ggml.h`**, where a device pass should not see the dllimport form at all. Upstream's own
  `windows-rocm` job has the identical flood, so this is an issue to file with the counts, not a
  patch -- a change to the `GGML_API` macro touches every consumer of ggml and is theirs to make.

- **A third, measured 2026-10-11: `ggml_backend_load_best` picks a slower CPU module than one it
  rejects.** On a Graviton 4 it selects `armv8.6_2` (198.0 t/s pp512) while `armv8.2_2` reaches
  **271.4** on the same machine and model -- the SVE/SVE2/i8mm paths losing to plain DOTPROD+FP16 --
  with generation inside the noise. The scoring function ranks by feature *level*, not by measured
  throughput, so it is by construction rather than a bug, but it costs 37 % of prompt processing on
  the hardware AWS rents most cheaply, and it is invisible to anyone who only runs the module their
  own CPU picks. Worth an issue with the table from `docs/local-test-plan.md`; **no patch here** --
  overriding upstream's scoring downstream would be a maintenance burden with no end.

(`0009`, `0010`, `0011` and `0013` are gone -- upstream fixed each defect itself -- and recorded in
`docs/history/dropped-llama-patches.md`.)

### llama.cpp upstream feature exposure (queued, deferred by policy)

These are JNI plumbing items for upstream API additions. Policy: add only after a real user request — they are mostly relevant to specific model families or specialized workflows.

- **Three upstream flags found by the b10878 flag audit, deliberately NOT implemented there.** The
  audit that produced `test_model_flags.cpp` swept every option `common/arg.cpp` registers for
  `LLAMA_EXAMPLE_SERVER` against what the Java layer emits (now `args.ModelFlag` + `args.ModelOption`;
  at the time of the audit, the string literals in `ModelParameters`). Beyond the seven dead
  flags it retired, it found ten option groups upstream had added since b10456 that the Java API
  does not expose. Seven were already covered (`--kv-unified-per-slot`, `--mmproj-device`/`-mmdev`,
  `--video-fps`, `--video-timestamp-interval`, `--video-ffmpeg-dir`, `--lazy-mode`/`-lzm`,
  `--n-cpu-ffn`/`-ncffn`). These three are the remainder, left out of the correction PR on purpose
  — it was a *fix* for an unloadable-model bug, and adding surface would have widened it:

  - **`--log-jsonl` / `--no-log-jsonl`** (a positive/negative flag pair, so it would fit `ModelFlag`
    directly). The only one of the three with real consumer value, but it is **not a free addition**:
    it flips `common_log_set_jsonl(common_log_main(), …)`, i.e. the process-wide llama.cpp logger,
    whose console output (and, since `patches/0014`, the sink `LlamaModel.setLogger` hooks) it would
    reformat. The project already has its own JSON logging at the Java level — the `args.LogFormat`
    enum plus `log_helpers.hpp`'s `format_log_as_json` — so the two would overlap and could contradict
    each other on the same stream. Deciding which layer owns the format is a **feature decision**, not a correctness fix,
    and needs its own change with its own tests.
  - **`--spec-synth-len` and `--spec-synth-rates`** — a documented non-goal, not deferred work. The
    reasoning lives in its own entry below (**"deliberately NOT exposed, and this should stay that
    way"**); it is not repeated here.

  Nothing is broken by leaving these out: `NativeServer` forwards raw llama-server argv verbatim, so
  all three remain reachable that way. The gap is only in the typed `ModelParameters` surface.

- **Request-key exposure, measured rather than guessed.** With `parameters.RequestField` in place the
  gap is countable instead of arguable. The Java layer writes **57** request keys (47 checked against
  llama.cpp's completion-request schema, 10 consumed by the OpenAI layer ahead of it); upstream's
  schema declares **68** primary fields, and **22** of those nothing here writes: `logprobs`, `lora`,
  `response_fields`, `return_progress`, `n`, `echo`, `max_tokens`/`max_completion_tokens`, the
  `reasoning_*` family, `grammar_lazy`/`grammar_triggers`, `preserved_tokens`, `chat_format`,
  `parse_tool_calls`, `adaptive_target`/`adaptive_decay` and `backend_sampling`. Same policy as the
  flags above — add on a real request, not speculatively — but the list is no longer something a
  future audit has to rediscover: re-derive it by diffing `RequestField.values()` against the field
  table `src/test/cpp/test_wire_contracts.cpp` already walks. (Counts are from the b10883 pin; the
  two numbers move independently, so re-measure rather than trusting them after a bump.)

- **Video input (`ContentPart.videoFile(...)`).** `mtmd` has had an end-to-end video path since
  llama.cpp **b9562** (#24269) — `mtmd_helper_video_init_params` was already present at the previous
  pin, b10456. What **b10647** (#24318, commit `f29551215`) added is the surfacing: a fourth
  `mtmd_helper_init_opt` parameter on the bitmap/tokenize helpers and the CLI flags `--video-fps`,
  `--video-timestamp-interval`, `--video-ffmpeg-dir`. Older notes cite b10649 for all of it because
  that was the *bump step* that carried it; b10647 is the tag that introduced it, and the video path
  itself is older still.

  **The three flags are now exposed** as `ModelParameters.setVideoFps` /
  `setVideoTimestampInterval` / `setVideoFfmpegDir`. They were initially refused at the b10649 bump
  as "inert without a way to submit a video"; a later audit showed that reasoning was wrong on two
  counts. First, they are not inert: `server_context::load_model` copies them into its own
  `init_opt` when the projector loads, and that `init_opt` is what `server-context.cpp` passes to
  `process_mtmd_prompt` on the task path this binding uses — so they take effect for any media the
  caller attaches. Second, video decoding is genuinely compiled in: `MTMD_VIDEO` defaults to `ON`
  (it needs only `LLAMA_SUBPROCESS`, also `ON`), and the shipped `libjllama.so` carries the ffmpeg
  invocation strings. `setVideoFfmpegDir` is the one that matters most, because upstream otherwise
  looks the binaries up on `PATH`, which a JVM process frequently does not have them on.

  What is still missing is the content part. Upstream's wire type is
  `{"type":"input_video","input_video":{"data":"<base64>"}}`, handled in
  `oaicompat_chat_params_parse` and gated on `allow_video = mtmd_helper_support_video(mctx)`. Note it
  calls `handle_media(..., accept_base64_uri = false)`, i.e. **raw base64 only** — unlike `image_url`,
  it will not take a `data:` URI, so `ContentPart.videoFile(Path)` must emit the bare base64 payload,
  not the `data:video/mp4;base64,...` form the image factories build.

  (An earlier draft of this entry suggested smuggling video through
  `ContentPart.imageBytes(bytes, "video/mp4")`, on the reasoning that `mtmd_helper_bitmap_init_from_buf`
  sniffs the container. That is plausible — the `image_url` branch does pass
  `accept_base64_uri = true` and does not validate the MIME string — but it is **untested here** and
  additionally gated on `allow_image`, so it is not documented as a supported route.)

  Remaining work: the factory, a bytes overload, and an integration test. Note the runtime cost:
  upstream **shells out to `ffmpeg`/`ffprobe`**, so a consumer needs those binaries, which makes the
  feature untestable on a CI runner without them.

- **`--spec-synth-len` / `--spec-synth-rates` — deliberately NOT exposed, and this should stay that
  way.** Added in b10649. Upstream's own help text marks both **"(benchmarking only)"**: they
  synthesise fake per-position acceptance probabilities so the speculative-decoding harness can be
  measured without a real draft model. They are an instrument for benchmarking llama.cpp itself, not a
  knob for an application, and exposing them as library API would invite callers to "tune" numbers
  that fabricate rather than measure acceptance. Anyone who genuinely wants them already has them:
  `NativeServer` forwards raw llama-server argv verbatim.

  **This is the single record for these two flags.** The b10878 flag audit (entry above) swept them up
  again as "upstream options the Java API does not expose" and briefly carried its own copy of the
  reasoning; that copy is now a pointer here. An audit re-finding them is expected and is not a signal
  to reopen the decision — the audit answers "is this name reachable from Java", which is a different
  question from "should it be".

- **Expose `--spec-draft-backend-sampling` toggle via `ModelParameters.setSpecDraftBackendSampling(boolean)`.** Added in b9437 (env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING`). Backend sampling for the speculative draft is enabled by default upstream but auto-disabled on `LLAMA_SPLIT_MODE_TENSOR` setups; an explicit Java-side setter lets callers force-disable it for benchmarking or for backends with sampler bugs. Speculative-decoding power users.

- **Expose runtime reasoning control via `InferenceParameters.setReasoningControl(boolean)` + `LlamaModel.endReasoning(...)`.** Added in b9444–b9490: new `common_params_sampling::reasoning_control` flag arms the budget sampler so reasoning can be ended at runtime, and new `common_sampler_reasoning_budget_force(common_sampler *)` triggers the end-of-thinking token injection on the next sample. Upstream also adds a `POST /v1/chat/completions/control` server endpoint accepting `{"id": "...", "action": "reasoning_end"}`. Java mapping would be: (a) `InferenceParameters.setReasoningControl(boolean)` arms the sampler on the inference run, (b) a new `LlamaModel.endReasoning(int slotId)` (or per-streaming-task-id) JNI method calls the upstream `common_sampler_reasoning_budget_force` against the slot's sampler. Useful for interactive UIs that want a "skip thinking and answer now" button. Relevant only for reasoning-trained models (DeepSeek-R1, Qwen3-Thinking, GPT-OSS-Reasoner, etc.).

- **Expose `llama_context_params::n_outputs_max` via `ModelParameters.setMaxOutputs(int)`.** Added in b9444–b9490 (default `-1` = derived from `n_batch`). Caps the number of output slots allocated per context; relevant for memory-constrained setups that always run with `logits_all=false` and want to prevent over-allocation when `n_batch` is large. Trivial JNI plumbing (one `cparams` field passthrough); add when a user reports OOM on context creation tied to output slot pre-allocation.

- **Expose Multi-Token Prediction toggle via `ModelParameters.setMtp(boolean)`.** Existed since the Qwen3.5 MTP work; b9444–b9490 extends it to Step-3.5. CLI flags `--mtp`/`--no-mtp` (env `LLAMA_ARG_MTP`) control whether the draft head runs alongside the main model for accelerated decoding. Java setter would route to `common_params_speculative::type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP`. Relevant only for MTP-trained models.

- **Expose `llama_vocab::get_suppress_tokens()` via `LlamaModel.getSuppressTokens()`.** Added in b9490–b9495 alongside the new `tokenizer.ggml.suppress_tokens` GGUF key and the `LLM_KV_TOKENIZER_SUPPRESS_TOKENS` constant. When a GGUF declares this array, upstream stores it on `llama_vocab::impl::suppress_tokens` and exposes it via the new `llama_vocab::get_suppress_tokens()` accessor. The bias is **applied automatically** inside the model forward graph — the Gemma4 Unified graph (`src/models/gemma4.cpp`) reads the list and adds a `-INFINITY` logit bias to those token IDs via a new `llm_graph_input_logits_bias` input so the model cannot emit them (used to block `<image|>` / `<audio|>` placeholders). A Java mirror would be `public int[] getSuppressTokens()` on `LlamaModel`: a read-only inspector returning the suppression list for debugging or for callers running their own sampling who want to replicate the same bias. Value is low (the bias is auto-applied, Java callers cannot change it; java-llama.cpp does not expose custom logit-bias hooks at this level); cost is trivial (one JNI passthrough + a `getSuppressTokens()` Java method).

### Feature backlog from similar projects

The consolidated investigation lives in
[`docs/feature-investigation-similar-projects.md`](docs/feature-investigation-similar-projects.md)
(18 candidates across the 5 pure-Java sibling runtimes + llamacpp4j, with effort sizing). Everything
high-value from it has shipped, the jbang one-file example (`examples/jbang/Chat.java`) included.
Further per-repo findings in the doc can be pulled on demand; none is prioritized.

### The ROCm Windows job runs with a CRT strategy that was never measured

**Recovered from the session transcript on 2026-10-11; it had been stated repeatedly and recorded
nowhere.** The hybrid-CRT change reaches the ROCm Windows build through its plain-clang branch, and
that branch applies because the job compiles with HIP's own clang (`lib/llvm/bin/clang.exe` under
driver, so CMake's `MSVC` is false there. Before the change that job had **no** branch and therefore
the dynamic CRT; it now gets the hybrid one (`-fms-runtime-lib=static` plus
`/nodefaultlib:libucrt.lib`). That may be an improvement (fewer dependencies) or it may break the
link, and it is exactly the `/MT`-against-`/MD` mixture across library boundaries that the comment in
`CMakeLists.txt` warns about for SYCL and OpenVINO: **ROCm's device libraries may be built against
the dynamic CRT.**

The job has been **green** since, so the link holds. What is unmeasured is the **runtime** behaviour
of the resulting module on an actual Radeon GPU -- and that cannot be rented (see the rented-hardware
plan), because the only AMD-GPU instance class has been withdrawn and never officially supported
ROCm anyway. So this needs a real Radeon machine, or a deliberate decision to ship it on the strength
of a green link plus `verify-native-deps.py`'s dependency allowlist. Do not quietly treat "the job is
green" as the measurement.

### Needs a machine the sessions do not have

Everything here is implemented or decided; what is missing is hardware or a model the cloud sessions
cannot reach. **Most of it is rentable** -- see
[`docs/rented-hardware-test-plan.md`](docs/rented-hardware-test-plan.md) for the queue, the
invariants and what rented hardware explicitly cannot answer. The items that stay with a developer
machine are the **interactive agent** (console behaviour needs a human at a real terminal) and a real
**arm64 Android device**.

- **The GPU module jars on a GPU**: item 1 of "Modular natives -- follow-ups" above (CUDA + Vulkan
  modules next to the Windows CPU jar, devices, offload, the CUDA graphs re-measurement).
- **Extraction reuse on Windows**: a second JVM start of the same build must copy nothing (no
  `[jllama] extracted` lines), a running JVM's locked DLLs must survive another build's cleanup, and
  the start must not block on them.
- **On-device Android**: the AAR + Kotlin facade are exercised on the x86_64 emulator in CI
  (`test-android-emulator`) and the LLM Service app (`android-llmservice/`) is the example app, built
  and UI-tested there too; what no session has is a real arm64 device, and the Adreno/OpenCL flavor
  needs one. Nothing to write, only to run.
- **AVX-512/AMX hardware** (Sapphire Rapids, Zen 4/5, Core Ultra): item 2 of "Modular natives" above.
  Partly answered since: a Cascade Lake machine (`avx512f` + `avx512_vnni`, no AMX) picks
  `cascadelake` and loads cleanly, which lifted the modules ever *chosen* anywhere from six to eight.
  AMX itself was that **suspected defect** and is now a found and fixed one (`patches/0018`, proven on
  a rented Granite Rapids machine and verified there with the patch -- see the rented-hardware section
  below; it is additionally backed by a GCC compile of the patched file with the variant's own
  flags, `-mamx-tile -mamx-int8 -mavx512vnni`). What stays open is the original question, the *throughput* gain of `sapphirerapids` and
  `zen4` over `haswell`, which needs the same hardware but no longer blocks anything.
- **Kolibri-1 on the real 78B model: done, `patches/0016` holds.** Measured 2026-10-11 on a rented
  16-vCPU / 123 GB instance, both published dialects fetched from Hugging Face on the machine at
  80-94 MB/s. **Both load and both answer correctly** through a chat turn on the model's own template:
  `content` `"Berlin."` for a German question and `"51"` for 17x3, the chain of thought in
  `reasoning_content`, `finish_reason` `stop`. Generation 13.4 t/s for Q4_K_M (44 GB) and 14.6 for
  Q3_K_S (31 GB). **The strongest single piece of evidence is the tokenizer**: both dialects produce
  the *same* token ids (`[452 22090 493 1678 2459 5513 46]`) and round-trip to the input sentence, so
  mapping `kolibri1` to the Qwen2 pre-tokenizer is demonstrated rather than assumed -- and a broken
  router would not have produced two correct answers in two quantizations. What is still unverified is
  **GPU backends** (the patch's own caveat) -- the first run of this measurement was wrong in a way
  worth keeping on file: it used a raw completion at temperature 0.0, which loops on an instruct model
  regardless, and read `chatCompleteText`, which returns only `content` and is therefore *empty* when a
  reasoning model spends its budget inside the thinking block. Both failures looked like the patch.
- **`TestConstantsTest.theShippedModelConstantsGoThroughTheResolver` is vacuous when the fixture is
  absent.** Mutating `MODEL_PATH = resolveModelPath("models/…")` to the bare literal leaves the test
  green with `models/` empty, and only goes red once the GGUF actually exists. In CI that is the
  normal case (`validate-models.sh` hard-fails first), so residual risk is low — but the two
  `src/test/resources/...` constants resolve from the module basedir either way, so their wrapper is
  undetectable **even in CI**. Fix: assert the wiring structurally rather than by value — reflect
  over the `String` constants and require each `models/…`-shaped one to equal
  `resolveModelPath(literal)` against a `@TempDir` fixture planted at the reactor root, so the
  assertion does not depend on a real model being present.

### Test-coverage debt found during the b10649 review (PR #403)

Each item below was verified against pristine upstream tags and is real, but none is a regression
introduced by the version bump — they were deferred to keep that PR landable. (The seven dead CLI
flags, the `acquire`/`release` context-guard unit tests, the `OSInfo` alias test and the dead
`Java8CompatibilityHelper` from the original list are done.)

- ~~**`LlamaLoader`'s jar-extraction internals are tested only through directory fixtures.**~~
  **Closed 2026-10-11 as not worth a test, after looking for the branch it would cover.** The premise
  was that a jar behaves differently from a directory somewhere in the loader. It does, in exactly
  **one** place -- `extractionKey`, which takes a `jar:<url>!/<entry>` URL apart by hand to read the
  entry's CRC-32 and size out of the central directory (deliberately not through a
  `JarURLConnection`, which would hold the jar open in the JVM's jar cache) -- and that place **is**
  already tested in `LlamaLoaderTest`, against a real jar built by the test and against a jar that
  does not exist. Everywhere else there is no branch of ours to cover: `extractFile` and
  `resourceMatchesFile` read a resource only as a stream (`getResourceAsStream`), so the
  jar-versus-directory difference lives inside the JDK's class loader. A unit test for it would have
  to add a settable class loader to `LlamaLoader` -- `classLoader()` is hardcoded to the loader's own,
  by design -- i.e. new mutable static API in production code to exercise a path where the only thing
  under test would be the JDK. CI loads all 23 natives jars for real anyway
  (`smoke-natives-jars.sh`, on the classpath **and** the module path), which is the coverage that
  matters. Recorded with the reasoning so this is not reopened as an oversight.

- **`ContentPart.videoFile(...)` — see the video-input entry above** for the wire shape upstream
  expects (`input_video`, raw base64, not a `data:` URI).

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** JSpecify + NullAway are now enforced at compile time in **strict JSpecify mode** with the extra options `CheckOptionalEmptiness`, `AcknowledgeRestrictiveAnnotations`, `AcknowledgeAndroidRecent`, `AssertsEnabled` (see `pom.xml`); `@NullMarked` on the three packages via `package-info.java`; JDK module exports in `.mvn/jvm.config`. The legacy `org.jetbrains.annotations` dep has been removed; all nullability annotations are JSpecify. Public-API methods that may legitimately have no value use `Optional<T>` rather than `@Nullable T` (`ChatResponse.getFirstMessage`, `ChatMessage.getParts`, `ChatRequest.buildToolsJson`). Open follow-up: review remaining unannotated public API surfaces for places where `@Nullable` would be more precise than the implicit non-null default.

- **Drop the project-wide `OPM_OVERLY_PERMISSIVE_METHOD` suppression in
  `spotbugs-exclude.xml`** once the package-architecture refactor lands
  (see [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md)
  under "Affects BAF + jllama (multi-package repos)"). The single-root
  package today makes every "method called only by same-package callers
  → could be package-private" finding correct-but-unstable; once layers
  split, cross-layer calls will need public. Snapshot at suppression
  (`07109cc`): 25 sites. The same rule is suppressed in BAF
  (`52c8c95`) for identical reasons.

- **Additional ArchUnit rules to consider** — the full **`layeredArchitecture()`** rule and a **per-module banned-import** rule (`jacksonBannedFromContractsAndLoader` — Jackson kept out of `args`/`callback`/`exception`/`loader`) are now DONE. Still open: more per-module banned-imports if useful, public-API-surface constraints (no public mutable static state, etc.). Partial progress: `7b6667d` covers the "no public field that is not final" sub-rule.

- **Cross-repo code-quality TODOs** — see [`../workspace/policies/code-quality-todos.md`](../workspace/policies/code-quality-todos.md) for the canonical `@VisibleForTesting` design-fit review, package hierarchy review, and class/method naming review. This repo has no `@VisibleForTesting` usages today; package and naming reviews remain open.

## GPU backends on rented hardware

**AWS blocks this outright on a new account: the G-instance vCPU quota is 0.** A `g4dn.xlarge`
launch (Tesla T4, ~0.21 USD/h, so cents for a probe) is refused with
`VcpuLimitExceeded ... your current vCPU limit of 0`, and raising it is a support request the
account owner has to file -- the probe package (CPU + `cuda13` + `vulkan` module jars, 169 MB) and
the driver-equipped AMI (`Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`) are
otherwise ready, so this is a one-line unblock once the quota exists. Cost was never the obstacle.

**Which GPU backends are reachable on rented hardware at all, measured 2026-10-11:**

| backend | rentable | how |
|---|---|---|
| `cuda13` | yes | G instances (T4 / A10G / L4 / L40S). A T4 proves as much as an H100 for "the module loads and computes", so the **P** buckets (V100/A100/H100) are deliberately not requested |
| `vulkan` | yes | the same NVIDIA cards -- the driver supplies `vulkan-1` / `libvulkan.so.1` |
| `opencl` | yes | the same cards -- NVIDIA ships an ICD |
| `openvino` | **yes, with no GPU at all** | OpenVINO has a CPU plugin, so an `m7i`/`m8i` can load `libggml-openvino.so`. Currently build-only; this is the cheapest open coverage gap |
| `rocm` | **no** | `g4ad` (Radeon Pro V520) is **gone from every region checked** (us-east-1, us-east-2, us-west-2, eu-central-1) and was AWS's only AMD GPU; ROCm never officially supported Navi 12 either. Stays build-only |
| `sycl` (GPU) | **no** | AWS has no discrete Intel GPU. Intel's CPU OpenCL runtime covers part of it |
| `metal` | **no** | EC2 Mac exists but bills a 24-hour minimum, and macOS ships the backend inside the library anyway |

**Spot cannot be stopped**, only terminated, so a persistent switch-off-in-between machine needs the
**On-Demand** G quota (`L-DB2E81BA`), not the Spot one. Measured spot price for `g4dn.xlarge` on
2026-10-11: **0.2925 USD/h**; On-Demand is roughly 0.53. A stopped instance costs only its EBS
volume (40 GB gp3 ~ 3 EUR/month), and starts in ~40 s with the toolchain and models still on it --
against 10 to 15 minutes for the boot-from-scratch probes used so far.

**`SME` is not rentable anywhere, and an Apple instance does not help.** AWS does offer
`mac-m4.metal` / `mac-m4pro.metal` / `mac-m4max.metal`, and Apple M4 does have SME -- but macOS
ships **one static library** (`metal-macos-aarch64`, Metal compiled in), not the aarch64 CPU module
set, so `armv9.2_1`/`armv9.2_2` cannot even be loaded there. They would need **Linux aarch64 with
SME**, which Graviton 4 lacks and no current instance provides. EC2 Mac also bills a **24-hour
minimum** (dedicated host), so it is the expensive way to answer nothing.


## Handed over from the modular-natives session (2026-10-11) -- read the first item before merging #507

Appended on `local/cpu-module-isolation-test` by the session that shipped the modular natives
(#504 to #506) and watched their two full runs (38065017595, 38069595101). Nothing above was moved.
Items `main` already carries through #506 are named, not repeated.

### Before #507 merges: two `0018` patches for the same hunk

`main` (cfc10fd2, #506) carries `llama/patches/0018-ggml-amx-free-the-aligned-buffer.patch`; this
branch carries `llama/patches/0018-ggml-amx-aligned-free.patch`. Both replace the same line of
`ggml/src/ggml-cpu/amx/amx.cpp` (main's adds a four-line comment above the call, this one is bare).
After a merge both files exist, the applier takes them in filename order, the first applies, the
second no longer finds its `-` line and aborts the configure with "does not apply cleanly" -- every
native build job red, on every platform. `git merge-tree main HEAD` (2026-10-11) reports a textual
conflict in `TODO.md` **only**; the two patch files, CLAUDE.md (two `0018` rows in the patch table)
and CHANGELOG.md (two "Fixed" bullets for the same defect) merge *silently*. Resolve in the merge
commit: keep exactly one patch file (the comment is the only difference), one patch-table row and
one CHANGELOG bullet, and keep this branch's `llama/cmake/check-aligned-free-pairing.cmake` guard
either way (main has none). For the `TODO.md` conflict: take main's wording for the three places
#506 touched -- a fifth "Modular natives -- follow-ups" item (the Windows ROCm job is the pipeline's
critical path at 4:15 h and builds uncached), the corrected sccache sentence under "Upstream
submissions" (the `-Xclang` that sccache 0.18.0 refuses comes from CMake's
`Platform/Windows-Clang.cmake` runtime-library selection on every plain-clang Windows TU, not only
from llama.cpp's `-fno-pch-timestamp`; CLAUDE.md "sccache on every Windows Ninja job but one" on
main has the measurement) and the AMX bullet under "Needs a machine" -- and this branch's structure
for everything else.

### Open from that session, recorded nowhere else

- **WER minidumps are switched on but unproven.** #506 makes the Windows test step set
  `Disabled=0` / `DontShowUI=1` under both WER keys next to the `LocalDumps\java.exe` entry
  (`DumpType` 2), because the AMX heap corruption left no dump behind. No Windows crash has happened
  since, so nothing shows that a `.dmp` now lands in the failure artifact. Prove it once rather
  than wait: a throwaway step that crashes a `java.exe` on purpose (an access violation through
  `sun.misc.Unsafe`, or `jshell` writing to address 0) on a branch run, then look for the dump in
  the artifact -- and if none appears, replace the WER route with `procdump -ma -e` around the
  surefire run.
- ~~**The emulator package pre-install is unproven.**~~ **Diagnosed and fixed 2026-10-11: it never
  worked, and it reported the wrong cause.** The step calls `sdkmanager` bare, but it is **not on
  `PATH`** on the hosted runner -- the SDK sits at `$ANDROID_HOME` with the command under
  `cmdline-tools/<version>/bin`. So every run failed four times with `sdkmanager: command not found`,
  slept 20+40+60 seconds between attempts, and then printed
  `the Android emulator package could not be installed in 4 attempts (dl.google.com download or
  archive broken)` -- a cause that had nothing to do with it. **The step therefore converted a job
  that was only occasionally flaky into one that is always red** (runs 38146809329, 38148401395,
  38149667195), and the misleading message is what kept that invisible: a reader sees a download
  problem at Google and waits for it to pass.
  Fixed in two steps, and the first one was wrong in a way the second could only find *because* it
  was wrong. The first (#519) kept the two failure modes apart -- "not found" says it is a
  runner-image change and not a download problem -- and resolved the command with
  `find "$root/cmdline-tools" -maxdepth 2`. Run 38164077038 then reported, in one line,
  `sdkmanager not found -- looked ... ANDROID_HOME='/usr/local/lib/android/sdk'`: the directory is
  there, and the file sits at `cmdline-tools/latest/bin/sdkmanager`, which is **three** levels below
  `cmdline-tools` -- the search stopped at the `bin` directory. An off-by-one, mine; what the
  message rewrite bought was seeing it at a glance instead of suspecting Google again.
  The second (#526) drops the find for the three known layouts in order (`cmdline-tools/latest/bin`,
  `cmdline-tools/<version>/bin`, the old `tools/bin`), guards the outer loop so a later root cannot
  overwrite the found path, and -- the part the first version lacked -- **lists what is actually
  under each root** when nothing is found, because naming only the places searched leaves three
  causes (missing directory, missing version subdirectory, missing file) behind one message.
  Validated against five faked layouts locally, including the failure case. **Still to verify in CI**
  -- the emulator jobs have not been green since the step was added, so the thing it was built to
  guard against (the corrupt archive of run 38069595101) also remains unproven.
  One trap worth keeping for whoever tests such a step locally: on this Windows checkout `chmod +x`
  does not stick on a plain file (`ls -l` still shows `rw-r--r--`), so `[ -x ]` answers *false* and
  every probe "failed" until the fixtures were given a `#!/bin/sh` line. The first run of that probe
  reported all five layouts broken, including the one that demonstrably works in CI -- the harness,
  not the logic.
- **Everything behind `package` has never run on the modular natives.** Run 38069595101 skipped
  `package` (it `needs` the Windows clang test job, red with the AMX crash); this branch's run
  38112603436 reaches it and fails on the Windows OpenVINO module jar (the blocker at the top of
  this file). Still unexercised after that: the smoke-set assembly (`package-smoke-sets.sh`),
  `smoke-natives-jars.sh` (classes + all 23 natives jars on the classpath **and** the module path),
  the bytecode gate and `verify-native-deps.py` over all 23 jars, the four `smoke-natives` rows
  (loader line naming every module jar of the platform, the HTTPS round, the RPC smoke, the
  bytecode gate over the set), `smoke-natives-macos`, `smoke-agent-linux`, and the publish jobs
  attaching the Maven artifacts. Expect further rounds there; the `windows-x86-64` smoke row runs
  on the same CPU lottery as the Windows test job, so it exercises AMX only when GitHub hands it an
  Intel runner.
- **The two server defaults disagree, by design or not -- owner's call.** Upstream #30159 (b11529)
  moved llama-server's default port from 8080 to 9931, which `NativeServer.DEFAULT_PORT` follows
  because it inherits the forwarded argv; `OpenAiServerConfig.DEFAULT_PORT` stayed 8080 as the
  project's own constant. Raised 2026-10-09, never answered: align `OpenAiCompatServer` to 9931
  (one constant, its tests, README) or keep the split and say so in the README.
- **Maven plugins without a pinned version, and no `requirePluginVersions` enforcer rule** (all
  four repos; raised 2026-10-01, never answered). `clean`, `resources`, `install`, `deploy`, `site`
  and in places `compiler`/`jar` run at whatever version the Maven on the runner defaults to, which
  differs between CI and a local build (the parent's `<pluginManagement>` here covers the plugins
  more than one module uses, not these). `check-versions.py` would then also compare them across
  the siblings. One pull request per repo, warnings-only until all four have landed.
- **`SECURITY.md` and `CONTRIBUTING.md` are the two per-repo files never compared.** Every other
  file that was "probably the same" across the four repos is either in `shared-files.sha256`
  byte-for-byte or as a `?repo` entry; these two were left out on 2026-10-01 as "differs per repo,
  probably wanted, not checked". Diff them once and either add `?repo` entries or write down why
  they differ.
- **SonarCloud reported "1 new issue" on #486 and nobody looked.** The sandbox's proxy answers 403
  for sonarcloud.io, so the finding was never read; the quality gate has passed since, so it is
  low, but it is unread.
- **The local-agent assignment of 2026-10-10 exists only in the session, not in `docs/handover/`.**
  Its deliverables map onto "Modular natives -- follow-ups" item 1 (the module jars on the RTX 3070
  + iGPU, throughput against 265.6 / 88.8 / 243.7, `--device` names across module sets, the
  filter's negative paths), the first two "Needs a machine" bullets (extraction reuse, stale
  cleanup against a running JVM) and the Kolibri bullet; the agent was asked to write
  `docs/handover/local-agent-report-5.2.0-modular-windows.md` on a branch
  `local/modular-natives-windows-report`. One question in it is not in item 1 yet: NVIDIA's and
  AMD's drivers ship OpenCL ICDs, so CUDA, Vulkan **and** OpenCL may each see the same RTX 3070 --
  whether llama.cpp folds the three into one device by `device_id` and which backend wins is
  unmeasured. The input it waits for is the first green `llama-smoke-windows-x86-64` artifact.
- **REUSE attribution is probably wrong in two places, and changing it needs the owner's word.**
  `REUSE.toml` credits the generated dockcross wrappers (`.github/dockcross/dockcross-*`) to the
  original fork author and the owner, while they are output of the dockcross project; and about 35
  Java files plus the documentation block still name the fork author, which a `git blame` pass
  would have to confirm or refute file by file. Raised 2026-10-01; no attribution is removed
  without an explicit decision.
- **srcmorph's smoke tests a fat jar it never ships** (its thin Central jars are launched by no job,
  its 15 classifier fat jars are checked structurally only). Whether srcmorph follows jllama -- thin
  jar plus `jbang --deps`, smoke on the consumer's classpath -- is the owner's decision and is
  recorded in srcmorph's `TODO.md` and the workspace fat-jar policy, not here.
