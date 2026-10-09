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

## Open — jllama-specific

### CPU variants (`JLLAMA_CPU_VARIANTS`) -- Windows, GPU modules, loader follow-ups

Linux x86-64 and aarch64 ship the variant build since 5.2.0 (CLAUDE.md "CPU variants"). The rest,
with what a measurement on a Windows 11 machine (Ryzen 7 5800H, RTX 3070, JDK 21, upstream b11476
binaries; 2026-10-08) established:

1. **Windows x86-64 build.** Only plain `clang`/`clang++` with the GNU driver -- upstream's
   `cmake/x64-windows-llvm.cmake`, four lines -- produces all 14 x86 variants: CMake sets `MSVC` for
   `clang-cl` as for `cl.exe`, and ggml's `if (NOT MSVC)` then drops `ivybridge`, `piledriver`,
   `cooperlake`, `zen4` and `sapphirerapids` (a Zen 4/5 machine falls back to `icelake` and loses the
   BF16 kernels); the `clang-cl` build of `alderlake` also fails (`/arch:AVX2` + `__AVXVNNI__` without
   `-mavxvnni`, an upstream gap). Consequences, all measured -- **(a) and (c) were re-measured on
   2026-10-09 and the earlier answer to both is superseded; read this version:**

   **(a) The CRT dependency is removable entirely -- Hybrid CRT, no DLLs to ship, no licence
   question.** The earlier finding ("the GNU driver ignores `CMAKE_MSVC_RUNTIME_LIBRARY`", a static
   `/MT` crashing with `0xC0000409`, so ship `msvcp140`/`vcruntime140`/`vcruntime140_1` app-local or
   require the VC++ redistributable) rested on two mistakes. The driver does *not* ignore the
   variable -- it emits `-D_DLL -D_MT` from it, which is why a naive static attempt failed to link;
   the control that works is **`-fms-runtime-lib=static`**. And the `0xC0000409` came from a *fully*
   static CRT (static UCRT included) **combined with shared libraries**, i.e. the known-bad
   configuration where every DLL gets its own CRT heap -- reproduced again on 2026-10-09, where such
   a build does not even start. **Microsoft's own Hybrid CRT** (static STL + vcruntime, dynamic
   UCRT) is the answer:

   ```
   -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
   -DCMAKE_C_FLAGS="-fms-runtime-lib=static"   (same for CXX)
   -DCMAKE_{SHARED,EXE}_LINKER_FLAGS="-Xlinker /nodefaultlib:libucrt.lib -Xlinker ucrt.lib"
   ```

   Measured on the **complete** `jllama.dll` built through this project's own `CMakeLists.txt`
   (plain clang 23.1.3, MSVC 14.51.36231 headers, llama + mtmd + cpp-httplib + the server sources,
   14.6 MB): its imports are `KERNEL32`, `ADVAPI32`, `SHELL32`, `WS2_32` and 11 `api-ms-win-crt-*`
   forwarders -- **no non-OS dependency at all**. UCRT stays, by design: `ucrtbase.dll` is an OS
   component from Windows 10 on, and it is not the DLL behind the README incident. That incident
   disappears *structurally* rather than being worked around -- the loader race over an old resident
   `msvcp140.dll` cannot happen when there is no import to resolve. So **the licence/REUSE decision
   on Microsoft "Distributable Code" is void**, and `nativedeps.py`'s `Windows/*/cpu` allowlist gets
   *shorter*, not longer. One wiring consequence: `llama/CMakeLists.txt`'s static-CRT block is
   `if(MSVC AND ...)`, and CMake's `MSVC` is **false** for plain clang -- widen that guard or pass
   the flags from the job, or the clang build silently gets the dynamic CRT back.

   **(b) `GGML_OPENMP=OFF` -- and it is a throughput *win*, not a trade.** It was known to save the
   `libomp.dll` dependency (confirmed: without it the clang build imports `libomp140.x86_64.dll`,
   which no redistributable carries -- the same class as the Windows-arm64 `0xc0000135`). Measured
   on 2026-10-09 (Ryzen 7 5800H, Qwen3-0.6B Q4_0, 8 threads, `GGML_NATIVE=ON`, all four builds
   static, runs interleaved, t/s):

   | | pp512 | tg128 |
   |---|---:|---:|
   | MSVC, OpenMP on | 359.8 | 41.7 |
   | MSVC, OpenMP off | 371.0 | **79.9** |
   | clang, OpenMP on | 390.6 | 30.2 |
   | clang, OpenMP off | 391.6 | **82.3** |

   Token generation is **1.9x (MSVC) / 2.7x (clang)** faster without OpenMP; prompt processing is
   unchanged within error on both. Cause, confirmed by thread scaling (clang, tg64): OpenMP peaks at
   4 threads and *degrades* above it -- 36.0 (t=2), 37.6 (t=4), 29.1 (t=8), 21.5 (t=16) -- while
   without it 54.5 / 74.2 / 71.4. Generation is synchronisation-bound (little work per barrier), so
   the runtime's barrier cost swamps it; prompt processing has enough work per barrier to hide it.
   **This applies to the artifact shipped today**, which is MSVC with OpenMP on -- and which
   therefore also imports `vcomp140.dll`, a third redistributable DLL this entry did not list. One
   line in the build job roughly doubles interactive generation throughput, independently of
   everything else here; it is set on all four Windows x86-64/x86 CPU jobs now, and
   `nativedeps.py`'s allowlist no longer carries `vcomp140.dll`, so dropping the flag again fails
   the `package` job instead of silently costing the throughput back.
   **It is a Windows-runtime property, not an OpenMP one -- do NOT generalise it to Linux.**
   Measured the same way in the project's own `manylinux_2_28_x86_64` image (gcc 14.2.1, i.e. the
   compiler and libgomp `crosscompile-linux-x86_64` uses; same model, 8 threads, interleaved):
   tg128 72.8 with OpenMP against 77.3 without, pp512 357.0 against 358.7 -- about 6% with one ON
   sample at 76.1 +- 0.9, inside the spread rather than a result. libgomp evidently keeps its thread
   team alive across parallel regions where LLVM's `libomp` and MSVC's `vcomp` do not. The Linux
   jobs therefore keep OpenMP, deliberately. What is still unmeasured is a machine with many more
   cores than the 8 here -- ggml's own pool could scale differently there. An earlier reading of a non-interleaved run suggested OpenMP was 8.6% ahead
   on pp512; that did **not** reproduce once the runs were interleaved (thermal skew, +-22 t/s
   spread) -- there is no trade-off to weigh.

   **(c) Do NOT pin clang to 20.1.8 -- `patches/0017` fixes the cause.**
   `-Wincompatible-pointer-types` became an error by default in **clang 22**
   ([llvm-project #157364](https://github.com/llvm/llvm-project/pull/157364)), not 16, and the four
   `_mm_prefetch` calls in `ggml-cpu/arch/x86/quants.c` are the only thing it hits here: a full
   `GGML_CPU_ALL_VARIANTS=ON` build with clang 23.1.3 produced **exactly four errors, all of them
   these**, and nothing in the AVX512/BF16/AMX/AVX-VNNI paths. With the patch that build is green
   and emits all 14 module DLLs (17.03 MB in total; `x64` 0.85 MB to `sapphirerapids` 1.56 MB).
   Upstream is pinned to clang 20 for a *second* reason worth knowing: `GGML_OPENMP_FETCH`
   `FATAL_ERROR`s unless the clang major matches its bundled LLVM OpenMP 20.1.8 -- so `0017`
   together with `GGML_OPENMP=OFF` is what makes a current clang possible here, while upstream ships
   `libomp.dll` next to its binaries rather than having no dependency.

   **(d) clang is not slower than MSVC -- it is ahead**, which removes the last argument for keeping
   MSVC as the primary build. Static against static, both with OpenMP off: generation equal within
   error (82.3 vs 79.9), prompt processing **+5.6%** for clang (391.6 vs 371.0). A first comparison
   appeared to favour MSVC; that was an invalid measurement pitting *static* MSVC against *shared*
   clang -- the same clang source gives 71.0 t/s shared and 82.3 static, so shared libraries cost
   ~13% on generation. The MSVC build stays as `msvc-windows-x86-64` (9 variants, static CRT, one
   library), tried before `cpu` when present; its value is being a second, independent toolchain,
   not its user count. Packaging stays as it is -- one CPU backend in the default jar, `msvc` opt-in
   by dependency, `net.ladenthin.llama.backend` as the switch: shipping both in one jar would add
   14.6 MB for every Windows consumer, and both the structure and the switch already exist.
   32-bit Windows and Windows arm64 get no variants (upstream builds none).

   **Measurement hygiene, learned the hard way on 2026-10-09:** this machine has the winget package
   `ggml.llamacpp` on `PATH`, carrying its own `llama-bench.exe` *and* `ggml-base.dll` /
   `ggml-vulkan.dll` / `ggml-cpu-haswell.dll`. `cmd /c "llama-bench.exe ..."` ran **that** binary
   even from inside the build's own directory -- caught only by `build: 689e227db (10357)` in the
   output where `(11512)` was expected, and by Vulkan results of 7047 t/s. Always invoke the full
   path and check the `build:` line. And a `grep -c` for a marker reports 0 for *empty* output too,
   so it cannot tell "ran clean" from "did not run".
2. **Windows loader.** `System.load` with a full path does NOT add the DLL's directory to the
   dependency search, and a foreign llama.cpp on `PATH` (winget `ggml.llamacpp`, Ollama, LM Studio)
   silently satisfies `ggml-base.dll` with another build's binary -- the probe passed falsely until
   the PATH was cleaned. So `ggml-base.dll` and `ggml.dll` (after the CRT files) are preloaded by
   full path in that order through `jllama-extras.txt`; the modules themselves stay in
   `jllama-files.txt`. `JNI_OnLoad` finds its own directory with `GetModuleHandleExW` +
   `GetModuleFileNameW` and must hand it to `ggml_backend_load_all_from_path` as **UTF-8** (a path
   with an umlaut loaded nothing as ANSI, and no error was logged). A GPU module whose runtime is
   missing fails silently and without a dialog (`SEM_FAILCRITICALERRORS`, exit 0) -- ggml logs it at
   `GGML_LOG_DEBUG` only, so the loader should log which modules it extracted and which devices
   ggml reports afterwards.
3. **GPU backends as modules (stage 3).** Measured feasible from a JVM: `ggml-cuda.dll` and
   `ggml-vulkan.dll` from upstream's zips load side by side from one directory with the CPU set
   (4 devices, incl. the AMD iGPU through Vulkan). Upstream's GPU zips carry a byte-identical copy of
   the whole CPU set; a GPU natives jar holding only its module would share ours and is then not
   usable without the CPU jar of the **same build**. Decide: consumers take `llama-platform` + GPU jar
   (the loader fails loud when the CPU part is missing or from another build -- a build key per
   natives jar, compared across jars), or GPU jars become artifacts of their own with a POM
   dependency. Also: `ggml-cuda.dll` imports only `cublas64_13.dll` (cudart is static) -- the
   requirement is smaller than the README says; and device indices are not stable across the set of
   loaded backends (`Vulkan0` was the NVIDIA GPU with CUDA loaded, the AMD iGPU without), so any
   device setting must go by name, never by index.
4. **Loader: reuse the extraction across runs.** The directory is keyed per build now
   (`extractionDirectoryName`), but every start still re-extracts: `cleanup()` deletes every
   `jllama*` path first and `deleteOnExit` removes the files at exit. Copying the 18 files costs ~1 s
   (measured, Defender on). With the key, a start could reuse a directory of its own build and
   cleanup could leave directories younger than a few minutes alone (a JVM still extracting). A
   loaded DLL is locked on Windows, an unloaded variant is not -- which is why the key, not a lock,
   separates builds.
5. **Benchmark on AVX-512/VNNI/AMX hardware** (Sapphire Rapids, Zen 4/5, Core Ultra). The Zen 3
   measurement could only show the floor: the plain x86-64 module is 10.4x slower at prompt
   processing (313 -> 30 t/s, Qwen3-0.6B Q4_K_M) and 1.6x at generation than `haswell`, which is what
   the single-level build was. The gain upwards (`zen4` BF16, `sapphirerapids` AMX) is unmeasured.
6. **Smoke tests must run inference through the loaded module.** Every crash in the Windows
   measurement came after a successful load with a correct device count; `--list-devices` does not
   even show the CPU. The fat-jar smokes and the Java test jobs do run completions; the
   `smoke-natives-jars.sh` load check alone is not evidence.
7. **Unsigned binaries.** Upstream ships its DLLs unsigned, and so do we; on a WDAC/AppLocker-managed
   client, loading unsigned DLLs from `%TEMP%` is blocked. Out of scope here; worth a README note.

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

### macOS dylib links Homebrew OpenSSL (found by `verify-native-deps.py`)

- **The shipped `Mac/aarch64/metal/libjllama.dylib` needs `/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib`
  and `libcrypto.3.dylib`** (verified on the published 5.1.0 jar and the current snapshot). The macOS
  build finds the runner's Homebrew OpenSSL and links it dynamically, so the macOS natives do not load
  on a Mac without `brew install openssl@3` — and the macOS smoke cannot see it, because the runner
  has it. Likely fix: build BoringSSL statically on macOS as on Windows
  (`LLAMA_BUILD_BORINGSSL`, `llama/CMakeLists.txt`), or turn HTTPS off there (`-DLLAMA_OPENSSL=OFF`;
  the library only needs it for URL model downloads). Then delete the two allowlist lines marked
  KNOWN DEFECT in `.github/buildcheck/nativedeps.py`. Needs a macOS CI run to verify, which is why it is
  not folded into the RPC PR that surfaced it.

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
- **File patch `0015` upstream** (non-aborting registration, `ggml_backend_rpc_stop_server()`,
  `ggml_backend_rpc_server_listening()`, the transport fd/SIGPIPE fixes) and drop it once merged.
- **Android RPC is untested on a device.** Bionic sockets build (upstream ships RPC in its Android
  release too), but the app needs `android.permission.INTERNET` even for loopback, which the AAR
  deliberately does not declare and the emulator fixture does not have. A loopback test on the
  emulator needs that permission in the fixture's manifest only.
- **RPC smoke on the other fat-jar platforms.** `smoke-rpc-fatjar.sh` runs in the `linux-x86-64` row
  of the `smoke-fatjar` matrix only (widening it to `linux-aarch64` is a change of that step's `if:`;
  Windows needs a PowerShell port of the script); the Java `RpcServerTest`/`RpcIntegrationTest` already run on every `test-java-*` job
  (Windows and macOS included), so this is about the packaged asset, not the code path.
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

### Logging sink (`patches/0014`) — follow-ups

- **Keep the log worker attached instead of attaching per line.** `LlamaModel.setLogger`'s trampoline
  runs on `common_log`'s worker thread, which llama.cpp creates (and re-creates on every
  pause/resume) and which is not ours; today `get_jni_env_attaching` does `AttachCurrentThread` +
  `DetachCurrentThread` **per log line**. Leak-free and simple, but every attach creates a
  `java.lang.Thread` object and fires JVMTI `ThreadStart`/`ThreadEnd`, which is noticeable at
  `--verbose` volumes and makes profilers/debuggers crawl. The cheaper shape is a `thread_local`
  guard object whose destructor detaches once at thread exit (C++ TLS destructors run on normal
  thread exit on glibc/macOS/MSVC, including for a `dlopen`'d library, and `std::thread::join` in
  `common_log::pause()` is a normal exit). Caveats to design in: attach as daemon
  (`AttachCurrentThreadAsDaemon`, so `DestroyJavaVM` never waits for the leaked singleton's worker),
  skip the detach when `g_vm` is already gone (`JNI_OnUnload` ran), and pin the behaviour with the
  existing model-free `LlamaLoggerTest` plus a count of `java.lang.Thread` objects seen by the
  callback — `deliveryIsAsynchronousOnTheLogWorkerAndRemovingTheLoggerDrains` already prints it
  (measured: 13 lines of a failed load → 13 distinct `Thread` objects, i.e. one per line). Not a
  correctness issue; measure the time cost before doing it.
- **File the patch upstream.** `common_log_set_callback` is a small, self-contained addition to
  `common/log.{h,cpp}` with no jllama specifics; upstream acceptance would retire the carry.

### Atmosphere coding agent (`llama-atmosphere-agent/`) — follow-ups

The headless loop is verified, including the model-backed CI job (run 35600558852: tool call
answered, read→write→read loop changed the file). Still open:

- **Tool rounds are not carried across REPL turns** — only `user`/`assistant` text is replayed, so a
  second question cannot refer to a tool result of the first. Keep the full Atmosphere
  `ChatMessage` list (incl. `tool_calls`/`tool` messages) per turn instead.
- **Approval for destructive tools.** `write_file`/`delete`/`run_command` run unasked. Atmosphere's
  `ToolDefinition.requiresApproval` + an `ApprovalStrategy` on the context would give a Claude-Code
  style "allow this?" prompt on the console.
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

### NativeServer attach mode leaves a sleep callback behind (found at the b11361 bump, not reproduced)

`llama_server_attach` (`patches/0007`) builds a `server_routes` on its own stack frame over the
`LlamaModel`'s `server_context`. Its constructor registers a sleeping-state callback on the model's
queue (`server_queue::on_sleeping_state` only appends, there is no unregister), and that callback
captures the `server_routes`. When the attached `NativeServer` is closed, the frame returns and the
object is gone, but the callback stays in the queue of the model, which lives on. The next time that
model enters idle sleep, the callback runs on a destroyed object. Reachable only with a model loaded
with `--sleep-idle-seconds` that was served by an attached `NativeServer` and then kept in use after
the server closed. Since b11361 `LlamaModel` holds a `server_routes` of its own for its whole lifetime
(`jllama_context::routes`, for `handleSystemOne`); the natural fix is to let attach mode serve
*that* object instead of building a second one, which changes `llama_server_attach`'s signature in
`0007` and `native_server.cpp`. Needs a test with sleep enabled (`IdleSleepWakeIntegrationTest` is the
template) before the fix, to show it red first.

### LlamaLoader extraction-directory isolation (optional follow-up, low priority)

Left over from the 2026-06-20 code audit (18/18 findings fixed in PRs #258/#260, regression tests in
#261/#262): full per-process extraction **directory** isolation + a `cleanup()`
that recursively removes dead-process dirs. Since extraction writes are atomic and content-checked,
this is a tidiness improvement (stops the shared-tmpdir `cleanup()` racing a live peer's flat file),
not a correctness fix — and it needs Windows locked-file co-design.

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
- **400 vs. 500 for an invalid request body.** Since llama.cpp b11337 (#29060) upstream's server
  answers a malformed or empty embedding `"prompt"` (and any `common_json_error`) with 400. The JNI
  layer throws a plain `LlamaException` for both, and `LlamaModelBackend` does not translate it into
  the `IllegalArgumentException` that `completeNonStreaming` maps to 400, so `OpenAiCompatServer`
  answers 500. A fix needs a typed signal from native (an invalid-request exception subclass, or the
  `throw_invalid_request` JSON shape parsed on the Java side) rather than message matching.
- **Manual real-client validation.** Server-side round-trips exist for every surface; what remains is
  pointing the actual editor clients (Copilot Ollama provider / Custom Endpoint, Claude Code, a
  Responses client) at a running server, since round-trips confirm wire shapes but not each client's
  parser.

### SonarCloud "Security Rating on New Code" gate — PR #248 (open)

The PR's **only** red is SonarCloud's "Security Rating on New Code" gate (every build/test job is
green; SonarCloud is **not** a merge-blocking build job). The findings are GitHub-Actions/Java
analyzer issues from the Maven scanner — **"C" is the rating *grade* (A–E), not the C language**;
there is no CFamily/C-C++ scan configured. Addressed:

- **`clang-format.yml`** — `pip install` without `--only-binary :all:` can run a package's `setup.py`;
  forced wheels-only (`84297e0`, block scalar so `:all:` doesn't break YAML). *If Sonar still flags it,
  try the `--only-binary=:all:` equals form.*
- **`osv-scanner.yml` / `scorecard.yml`** — top-level `permissions: read-all` → `contents: read`
  (`84297e0`); safe because every job in both files already declares its own exact permissions.
- **`publish.yml`** — workflow-level `permissions: contents: read` (Sonar wants it per-job); **owner
  marked it Accept/"Won't fix" on the dashboard** rather than spreading perms across ~25 release jobs.
  Alternative if ever desired: add `permissions: contents: read` to the ~19 read-only jobs (the 5
  publish/report jobs already declare `contents: write`) and drop the top-level block.
- **`PairTest.java`** — 3 Critical *Reliability* bugs (`assertNotNull` on the primitive `hashCode()`)
  replaced with a determinism check (`9f0d377`). Reliability rating, **not** the Security gate.

**Still open:** the gate was still red as of `9f0d377`. SonarCloud's issues API is auth-gated (403 from
CI), so the exact remaining new-code Vulnerability must be read off the dashboard. Resolve the last
finding, accept it on the dashboard, or merge on the green build/test checks.

### License Compliance (FOSSA-style dependency-license gate) — PR #248 (open)

Separate from the FSFE **REUSE** check (which is green — `reuse lint` reports 266/266 files compliant)
and from SonarCloud: the PR's combined commit status shows a **"License Compliance" check failing with
"17 issues found"** (an error-state commit status posted by a license-scanner GitHub App, not a
workflow in `.github/workflows/`). It contributes to the `mergeable_state: blocked` on #248.

- **Almost certainly pre-existing**, not introduced by this PR: #248 changes **no dependencies** (the
  `pom.xml` edit only adds the `windows-ninja` build profile), so the 17 are dependency-license policy
  findings already present on `main` (e.g. GPL-2.0 carried by the llama.cpp sources).
- **Not yet inspected** — the scanner's dashboard/host is outside this sandbox's egress allowlist, same
  as `sonarcloud.io`. To triage: open the check's details link from the PR (or allowlist the host), read
  the 17 findings, then accept policy-OK licenses on the dashboard or adjust the policy. Confirm whether
  it is a *required* status (if so it blocks merge; if advisory it does not).
- **Still red on PR #298 (2026-07-05):** the same status ("17 issues found") posts on every head there
  too and contributes to its `mergeable_state: blocked`. Same triage path: read the findings on the
  scanner's dashboard, accept policy-OK licenses or adjust the policy.

### Upstream PR submissions — drop the carried patches (open)

There are **eight** patches today (`0001`–`0003`, `0006`–`0008`, `0012`, `0014`). **Seven are
upstream-submittable verbatim**; each accepted PR (once the pin is bumped past it) deletes a patch
from the bump checklist. The exception is **`0003`**, a carry of upstream PR #22393, which upstream
**closed without merging** — it is permanent and will never be droppable via a bump. (`0003` used to
be described here as "drops automatically when that merges"; it will not.)

**`0016` (Kolibri-1) is not a submission candidate but a temporary carry**: upstream will add the
architecture itself (request [ggml-org/llama.cpp#29922](https://github.com/ggml-org/llama.cpp/issues/29922)).
Drop it on the first bump whose tag registers `kolibri1` (`git grep -n kolibri src/llama-arch.cpp`),
keep `src/test/cpp/test_kolibri1.cpp` (it compiles without the patch). Its **numerical comparisons**
must stay green -- red there means upstream computes something else than Aleph Alpha's reference, a
finding to report, not a test to adjust. Its **GGUF-format rows** (gating function 2 *and* 5, no gating
key, pre-tokenizer `qwen2` *and* `kolibri1`, rejection of gating 1) follow whatever format upstream's
converter fixes: a red row means the published GGUFs of that dialect stop loading without the patch.
Decide that deliberately -- keep a small compatibility patch, or document that those files must be
reconverted (and say so upstream rather than lose them silently) -- and only then move the row's
`{gating, pre, ...}` entry to upstream's format. Open verification gaps of the carry:
no run of the real 78B model and no GPU backend from here (see the patch header).

- **`0001` Windows arg-parse embed guard** (against #24779): `common_params_parse` trusts the caller's
  argv; `common_params_parse_main()` keeps the standalone tools' UTF-8 recovery. Ship with the
  standalone-safe repro (synthetic argv discarded on Windows because `GetCommandLineW()` returns the
  host process line) — written up, with the reproducer executed, in
  `docs/upstream-investigation-win32-argv-substitution.md`. Reported upstream as
  ggml-org/llama.cpp#26416; waiting on the maintainers to pick a direction before a PR.
- **`0002` preserve caller load-progress callback** (b9789 regression: server clobbers
  `params_base.load_progress_callback`).
- **`0006` embeddable `llama_server`** (no process signal handlers, forwarded-argv parse, out-of-band
  shutdown).
- **`0007` `llama_server_attach`** (HTTP frontend on an existing `server_context`).
- **`0008` `LLAMA_SERVER_WORKER_CMD` router worker override** (also useful for containerized/wrapped
  deployments).
- **`0012` guard the zero split-sum and name the device index** (a GPU reporting zero free memory —
  or a cancelling `--tensor-split` such as `-ts 1,-1` on any backend — makes every model load fail
  with the unactionable `error loading model: vector`). Ships an upstream `tests/test-model-split.cpp`.
  **Not yet filed upstream.**
- **`0014` add a callback sink to `common_log`** (`common_log_set_callback`, what `LlamaModel.setLogger`
  hooks; upstream has file/colors/prefix/timestamps/verbosity/JSONL but no hook, so an embedding host
  cannot route the server's own `SRV_*`/`SLT_*` lines anywhere). **Not yet filed upstream.**

(`0009` is **not** in this list and the number is burned: upstream merged the subprocess.h fix via
ggml-org/llama.cpp#26606, so the patch was dropped at the b10280 bump. `0013` is likewise gone —
upstream merged this project's own PR ggml-org/llama.cpp#28775 and it was dropped at b10948. `0011`
went the same way at b11069: upstream fixed the invalid-UTF-8 PEG-parser failure independently and
more broadly via ggml-org/llama.cpp#29161 (one U+FFFD per undecodable run, text after it kept) before
the patch was ever filed, so the `ContentOnlyParseUtf8` guard now pins upstream's contract instead.
`0010` followed at b11080: upstream gave `common_json_value` an enum constructor via
ggml-org/llama.cpp#28518, fixing at the root the enum-to-bool trap the patch cast around, so it became
a redundant carry — note that it still *applied* cleanly, which is why the by-hand drop-check exists.
All four drops are recorded in `docs/history/dropped-llama-patches.md`.)

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

### Feature backlog from similar projects (remainder: jbang example)

The consolidated investigation lives in
[`docs/feature-investigation-similar-projects.md`](docs/feature-investigation-similar-projects.md)
(18 candidates across the 5 pure-Java sibling runtimes + llamacpp4j, with effort sizing). Everything
high-value from it has shipped — README system-properties table, per-run timing line
(`TimingsLogger`), UTF-8 boundary safety (native `utf8_to_jstring_impl` path), runtime LoRA control,
typed batch embeddings, in-JVM router mode, in-JVM GGUF quantization, GGUF metadata inspector,
session fork/rewind. **Remaining:**

- **jbang single-file example** (XS-S): a `//DEPS net.ladenthin:llama` one-file runnable demo so new
  users can try the binding without a Maven project.
- Further per-repo unique findings in the doc can be pulled on demand; none is currently prioritized.

### Android example app (own session; the remaining Android item)

The AAR + Kotlin façade + multi-ABI (arm64-v8a/x86_64) + emulator CI shipped, and the emulator job is
a release gate (see CLAUDE.md "Android AAR + Kotlin façade"). Remaining: a minimal
sample app under e.g. `examples/android-sample/` (single Activity, model picker, streaming text view)
consuming `net.ladenthin:llama-android` + `llama-kotlin` — it validates what the emulator cannot:
real arm64 hardware and the Adreno/OpenCL flavor. Treat LLaMAndroid as prior art.

### GraalVM Native Image evaluation

- **Evaluate GraalVM Native Image as an alternative distribution target.** Reference: [GraalVM Native Image](https://www.graalvm.org/latest/reference-manual/native-image/). The pure-Java sibling projects in the README's "Similar Projects" list (mukel's `llama3.java` / `gemma4.java` / `gptoss.java` / `qwen35.java` / `nemotron3.java`) demonstrate that single-jar, no-JNI Java inference is viable for individual model architectures. Native Image opens an orthogonal direction for THIS project: AOT-compile the Java layer + JNI bridge to a self-contained binary that bundles the libjllama.so (or per-OS equivalent) and starts in milliseconds without a JVM, which would make jllama usable in CLI tools, serverless functions, and short-lived processes where JVM startup is the dominant cost.

  **What to investigate before committing**:
  - **JNI-loading shape.** Native Image supports JNI but requires `--enable-native-access=ALL-UNNAMED` + reflection/JNI configuration files (`reflect-config.json`, `jni-config.json`, `resource-config.json`) describing every class/method/field reachable across the JNI boundary. The 34 native methods in `jllama.cpp` plus the JNI-side `FindClass` / `GetFieldID` / `GetMethodID` calls at `JNI_OnLoad` need to be mapped. The GraalVM tracing agent (`-agentlib:native-image-agent=config-output-dir=...`) can auto-generate the config during a representative test run, but the `LlamaLoader` JAR-extraction path needs at least one resource-config rule for `net/ladenthin/llama/{OS}/{ARCH}/lib*.so`.
  - **Native-library packaging.** The current `LlamaLoader` extracts the OS-specific `.so`/`.dll`/`.dylib` from the JAR to a tmp dir at first use. Native Image needs the same file at AOT-execution time, so either (a) ship the native lib alongside the produced binary as a sidecar file and adjust `LlamaLoader` to find it on the same directory, or (b) embed the native lib as a resource and keep the existing extract-to-tmpdir flow (which Native Image supports via `resource-config.json`).
  - **CUDA / Metal / OpenCL backend selection.** `LlamaLoader` already selects at runtime among the natives jars on the classpath (one directory per backend, fixed priority order). Native Image would need those directories as bundled resources (`resource-config.json`) or as sidecar files next to the binary.
  - **Startup-time benchmark to justify the work.** Measure cold-start of a current java-llama.cpp `LlamaModel(new ModelParameters().setModel("...").setNPredict(1))` invocation: how much is JVM startup + class load vs JNI load + model parse + tokenize + 1 token? If JVM startup is < 10 % of cold-start, Native Image yields little. If JVM startup is > 50 %, it's a clear win for CLI / serverless use cases.
  - **Maintenance cost.** Native Image adds a second build matrix (per OS × per backend × per JDK) and a new failure surface (Native Image config drift when a llama.cpp version bump adds new JNI-reachable types). Should ship only with a CI job that exercises the Native Image build on at least one OS, otherwise the config files will rot silently.

  **Out of scope until evidence supports it**: actually implementing any of the above. This entry exists so that when someone asks "can I ship java-llama.cpp as a single 30 MB binary?" the answer points to a concrete investigation plan rather than restarting from zero.

### macOS packaged-artifact gate — landed cheap, optional depth remains

**Done:** `smoke-fatjar-macos` (`needs: [package]`, gates both publish jobs) now verifies the dylib
inside the packaged jar — `codesign --verify --strict` plus a real JVM load and JNI round-trip via
`.github/smoke/NativeLoadSmoke.java`. See CLAUDE.md, "macOS arm64: three build jobs, one shipped
dylib". This closes the gap that let a SIGKILL-on-load binary ship through three releases with a
green pipeline, and is the macOS member of the cross-repo convention in
[`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md).

**Optional depth, not scheduled:** a full model-backed macOS server smoke (poll `/health`, assert a
`/v1/chat/completions` choice) as Linux and Windows run. It would need `verify-model-cache` +
a cache restore, and — since there is no `all-macos-*` fat jar — either a macOS variant from
`package-fatjars` or running `smoke-test-fatjar.sh` against the default jar. Worth doing only if a macOS-specific *inference* regression ever appears;
the load-time failure class is already covered, and a slow smoke tends to get made non-gating.

**Not yet observed green in CI** — the job and the two sibling-repo smokes landed in one change set
and have only run locally so far.

### Test-coverage gaps found by the b10679 mutation audit (PR #403)

> **Update.** The `IdleSleepWakeIntegrationTest` added to close the `wake_and_post` gap immediately
> found a real JVM crash (SIGSEGV on all six CI platforms) — see the CHANGELOG "Fixed" entry. Both
> facets are fixed in that PR via the `wake_server()` choke point. This is the clearest evidence for
> the entry below about a floor on executed tests: the defect had been reachable from public API for
> as long as `--sleep-idle-seconds` has existed, and nothing ran that path.

A mutation pass over the branch applied 27 mutations and 26 went red on the test that claims them,
so no test here passes with its subject deleted. What it did find is code with **no runnable guard**.
Two of the three were closed in that PR (a model-free `jsonSchemaToGrammar` test in
`NativeLibraryLoadSmokeTest`, and `IdleSleepWakeIntegrationTest` for the `wake_and_post` path); the
third was `patches/0010`'s `(int)` cast, reachable only from a model-gated Java test, and it went
away with the patch itself at the b11080 bump. This is what remains.

- **`TestConstantsTest.theShippedModelConstantsGoThroughTheResolver` is vacuous when the fixture is
  absent.** Mutating `MODEL_PATH = resolveModelPath("models/…")` to the bare literal leaves the test
  green with `models/` empty, and only goes red once the GGUF actually exists. In CI that is the
  normal case (`validate-models.sh` hard-fails first), so residual risk is low — but the two
  `src/test/resources/...` constants resolve from the module basedir either way, so their wrapper is
  undetectable **even in CI**. Fix: assert the wiring structurally rather than by value — reflect
  over the `String` constants and require each `models/…`-shaped one to equal
  `resolveModelPath(literal)` against a `@TempDir` fixture planted at the reactor root, so the
  assertion does not depend on a real model being present.

- **Nothing asserts a floor on the number of tests actually executed.** A class-level `@BeforeAll`
  assumption makes Surefire record `tests="0" errors="0" skipped="0"` — the class contributes no
  entries at all, so "did the run skip anything?" is structurally blind to it. This is exactly how
  the model-gated suite stayed silently muted for months. Summing `tests=` across
  `target/surefire-reports/TEST-*.xml` in each `test-java-*` job and failing below a pinned minimum
  is the one check that would have caught it directly, and it is cheap.

### Test-coverage debt found during the b10649 review (PR #403)

Each item below was verified against pristine upstream tags and is real, but none is a regression
introduced by the version bump — they were deferred to keep that PR landable.

- ~~**`ModelParameters` emits five CLI flags the server arg parser rejects, so any caller of them
  cannot load a model.**~~ **DONE** — and it turned out to be **seven**, not five. The fix is the one
  this entry prescribed: `cmake/extract-java-cli-flags.cmake` extracts every `"--flag"` literal
  `ModelFlag.java`/`ModelParameters.java` can emit into a generated header, and
  `src/test/cpp/test_model_flags.cpp` asserts each is registered in
  `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options`, exempting only `--vocab-only`
  (which `strip_flag_from_argv` removes on purpose). Run against the pre-fix Java sources it reported
  exactly the predicted set, which is how the count grew: the five named here plus `--mlock` and
  `--no-mmap`, deleted upstream at b10878 while this entry was open. Those two have a faithful
  replacement, so `enableMlock()`/`disableMmap()` were **repointed** to upstream's own deprecation-shim
  mapping (`--load-mode mlock` / `--load-mode none`) behind a new `setLoadMode(LoadMode)` rather than
  retired — no API loss. The other five became no-ops (`@Deprecated`, never write the map), and
  `ModelFlag.MLOCK`/`NO_MMAP`/`DUMP_KV_CACHE` were removed from the enum so a broken argv is not
  reachable through `setFlag` either — the same reasoning that already excluded `FLASH_ATTN`. The
  replaced Java assertions now compare against a pristine `ModelParameters`, not against the old
  "still has this key" shape that would have passed forever.

- **`acquire_jllama_context_impl` / `release_jllama_context_impl` / `jllama_context_guard` have no
  model-free unit guard.** These three (`jni_helpers.hpp`) are the whole `close()`-vs-inference
  use-after-free defence, and grep finds zero references across all seven `test_*.cpp` files, while
  their sibling `get_jllama_context_impl` has three tests. A dropped `fetch_add`, or a guard whose
  destructor stops calling release, produces a use-after-free during `close()` or a `close()` that
  hangs forever. `LlamaModelTest#testCloseDuringInference` covers the mechanism end to end but only
  bluntly. They are absent from `jllama_test` only because they are `inline` and never odr-used
  there: `g_ctx_mutex` is `extern` in the header and defined in `jllama.cpp`, which `jllama_test`
  does not compile — a test-local definition at global scope unblocks it.

- **`OSInfo`: the `archMapping` alias branch is untested.** `getArchName()`'s map lookup has no
  assertion anywhere — the two test call sites either take the override early-return or only assert
  non-empty — so a lost `amd64 -> x86_64` entry would send `LlamaLoader` to a resource directory
  that does not exist. Cheap to close: set `os.arch`, assert the non-identity aliases only (identity
  entries such as `s390x` are behaviourally redundant with the `\W`-stripping fallback).

- **`LlamaLoader`'s jar-extraction internals are tested only through directory fixtures.**
  `BackendLoadTest` drives backend probing, extras, fallthrough and forcing over the committed
  `Linux/backendtest*/` trees on the test classpath (directories, not jars); extraction out of a real
  jar is exercised in CI by `smoke-natives-jars.sh` and the fat-jar smokes, not by a unit test.

- **`Java8CompatibilityHelper` is mostly dead code — decide delete vs. test.** Six of its seven
  public methods have zero call sites repo-wide; the only live one is
  `toString(ByteArrayOutputStream, Charset)`, used once in `ProcessRunner`. Writing tests for the
  rest would pin dead code.

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
