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

## Where the modular natives stand, and what the next CI run is for

**The goal was exchangeable native libraries built with LLVM, replacing the fat jars. That is
implemented and, as of 2026-10-11, partly proven on real hardware.** This section is the state of it
in one place, because a Publish run costs hours and the question "what is this run supposed to show"
has to be answerable before starting one.

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
1. **The shipped diagnostic entry point** (item 5 of the modular-natives follow-ups). The only item on
   any list here that pays off *per user* rather than per release: it turns "it does not work" into a
   paste. Testable without natives.
2. **README: the GPU jars' real requirements.** `cuda13` needs a driver new enough for the toolkit
   (and what happens when it is not: the JVM aborts); `openvino` is opt-in now and needs an Intel
   GPU/NPU *plus* an OpenCL ICD loader; `vulkan` needs only a driver. None of that is documented, and
   all three are measured.
3. **`TestConstantsTest` / `TestModelPaths` parity.** `llama-langchain4j` carries its own copy of the
   resolver; the vacuity guard added to the core's test has no counterpart there.

**Round 2 -- needs round 0 to have produced artifacts.**
4. **A `g4dn` run proving a T4 works** with the new arch list (round 0 builds the module; the probe
   package and harness already exist).
5. **The Windows extraction-reuse measurement** -- two JVMs, file locks, the one item on the
   measurement list that is both cheap and still completely unmeasured.
6. **A slim-image load test**: build a container from `eclipse-temurin:21-jre-noble` and from
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
| **re-baseline the module tables at the current pin** | the same instances | the tables are b11538, and **#30270 in the b11556 range fixes F16C on AVX2**, i.e. the `haswell` class every "against haswell" column is computed against -- so nothing measured from b11556 on may be compared with that page (noted there too) |
| Does a self-contained jar load on a **slim/distroless** image? | any Linux instance, a container | CI runs on full runners, so it cannot see this class at all |
| Does **`75-real 80-real 90-real`** make a T4 work? | a rebuilt CUDA module + a `g4dn` instance | the module has to be built first, so this follows a CI run |
| **Windows extraction reuse**: second start copies nothing, a running JVM's locked DLLs survive another build's cleanup | a rented Windows instance, two JVMs | file locking is Windows-only and needs two processes |
| the **`sapphirerapids` vs `haswell`** gap at high core counts | a large Intel instance | one 4-vCPU table exists; generation ranking can reverse with threads |
| **OpenVINO on an Intel GPU/NPU** | hardware nobody rents | the 172x figure is the CPU-plugin fallback only, so nothing is known about the hardware it is for |
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

2. ~~**`cuda13` does not fall back when the driver is too old.**~~ **Mitigated:** both CUDA jobs now pass an explicit `CMAKE_CUDA_ARCHITECTURES` making `75 80 86 89 90 100 120 121` **real** with a `121-virtual` tail, so the datacenter GPUs no longer depend on the driver's JIT. **Still to verify:** the build time and module size this costs (nvcc compiles each `.cu` per real arch -- expect roughly double on a job already at 47-80 min uncached), and then a `g4dn` run proving a T4 works. A minimum driver version for the README is still open, because PTX JIT remains the path for any architecture not in the list. The original entry follows.
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

3. ~~**`openvino` takes over silently and is 172x slower.**~~ **Done:** `openvino` is in `LlamaLoader.OPT_IN_BACKENDS`, so its jar does nothing until `net.ladenthin.llama.backend` names it, and the loader prints why on stderr. Pinned by `LlamaLoaderTest` from both sides (left out by default, selected when named, and every other module still additive). **Still open:** the README needs to say the jar requires an Intel GPU/NPU *and* an OpenCL ICD loader, and nothing is known about its behaviour on the hardware it is actually for. The original entry follows.
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
6. **The Windows ROCm job is the pipeline's critical path, and it builds uncached.** Run 38069595101
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

### CUDA job: nvcc through sccache (`SCCACHE_WRAP_NVCC`, off since the second failure)

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

### Upstream submissions -- not before the release (policy)

**Nothing is filed upstream before the release is out**, neither issues nor PRs; the drafts are not
written before that either. Everything below is the inventory for that day, so it is not rediscovered.

There are **twelve** patches today (`0001`-`0003`, `0006`-`0008`, `0012`, `0014`-`0018`). Each
accepted upstream change deletes a patch from the bump checklist once the pin is bumped past it.

- **Submittable as they are** (self-contained, no jllama specifics): `0001` (Windows arg-parse embed
  guard, already reported as ggml-org/llama.cpp#26416 with the reproducer in
  `docs/upstream-investigation-win32-argv-substitution.md`; the maintainers have not picked a
  direction), `0002` (preserve the caller's load-progress callback), `0006` (embeddable
  `llama_server`), `0007` (`llama_server_attach`), `0008` (`LLAMA_SERVER_WORKER_CMD`), `0012` (zero
  split-sum guard, ships `tests/test-model-split.cpp`), `0014` (`common_log_set_callback`), `0015`
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
- **The emulator package pre-install is unproven.** Both emulator jobs died in one minute inside the
  action's own `sdkmanager --install emulator --channel=0` ("Error on ZipFile unknown archive", run
  38069595101) an hour after being green; #506 installs the package in a step before the action,
  four attempts, so the action's install is a no-op. Neither job has run since. If the download
  keeps flaking, the sturdier form is `actions/cache` over `$ANDROID_HOME/emulator` keyed on the
  emulator package version `sdkmanager --list` reports, so dl.google.com is reached once per
  version instead of once per job.
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
