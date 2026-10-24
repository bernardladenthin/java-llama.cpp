<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# TODO — java-llama.cpp

Open work items and the decisions that constrain them. Cross-cutting tracking lives in
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md);
items here are jllama-specific or are this repo's slice of a
cross-cutting initiative.

**Completed work is not recorded here.** It lives in git history and in
`crossrepostatus.md`; a finished item is deleted from this file rather than annotated,
so entries below describe remaining work or an explicit deferral policy.

## Release 6.0.0 — prerequisites

- [ ] Merge the fixes on `refactor/shared-release-checks` and obtain a complete green Publish
      run for the resulting release candidate, including packaging, classpath/module-path and
      macOS smokes, Android CPU + additive OpenCL consumers, FFM, source-archive and integration
      jobs. The latest completed snapshot run,
      [39922567350](https://github.com/bernardladenthin/java-llama.cpp/actions/runs/39922567350),
      is green on `fb6b693a`; it predates this branch's packaging and Git-credential fixes.
- [ ] Strip `-SNAPSHOT` from **all six reactor poms and the standalone agent pom** in one release
      commit. Follow [docs/RELEASE.md](docs/RELEASE.md) for dependency snippets, JBang and the server
      example; keep the existing "Upgrading from 5.1" list current for any further breaking change.
- [ ] Set the CHANGELOG release heading/date and footer links, and update `SECURITY.md`'s
      supported line from `5.x` to `6.x` as part of that release commit.
- [ ] Verify the mirror credential and destination access described in roadmap item 6 below.
- [ ] **Owner:** tag/dispatch `v6.0.0` using the
      [workspace release procedure](../workspace/workflows/release-process.md). Check the reactor,
      standalone agent and both Android publications after upload. Android Central validation
      polling remains deliberately outside the build gate (see `docs/RELEASE.md`).
- [ ] After publication: bump `main` to the next `-SNAPSHOT` in all seven poms and carry out the
      upstream submissions queued below.

## Delivery roadmap — remaining work

The remaining items retain the numbering of the 2026-10-15 roadmap referenced by `CLAUDE.md`.
Windows code signing stays deferred by the owner's decision. Vendor hash pinning and the
measurements below remain open hardening/coverage work; they do not establish a runtime defect.

### 3. Hard pinning (C.2) — four repos

- [ ] **Pin the vendor checksums into `publish.yml`** (`OPENVINO_LINUX_SHA256`,
      `OPENVINO_WINDOWS_SHA256`, `VULKAN_SDK_SHA256`). The fallback verifies downloads against
      vendor-hosted checksums, but an archive and checksum replaced together would still pass.
      Obtain the values from a trusted download or the successful Publish logs and verify them
      before pinning. `CUDA_REDIST_MANIFEST` is already version-pinned.
- [ ] **Hash the oneAPI installers and Level Zero downloads.** Both SYCL jobs use versioned
      `ONEAPI_DLE_URL` / `ONEAPI_ONEDNN_URL` installers (Deep Learning Essentials 2026.1.4 and
      oneDNN 2026.0.2). Download them on a machine that can reach Intel, then add pinned SHA-256
      checks; do the same for the Level Zero debs and SDK zip.
- [ ] **Hash the ROCm wheels.** `stable.repo.amd.com/rocm/whl-next/` is a moving index; versions
      are constrained with `==`, but hashes are absent. Generate a complete hash-pinned
      requirements file for `rocm[libraries,devel]` using `pip download` / `pip hash`, then use
      `pip install --require-hashes`.
- [ ] BoringSSL's tag pin (`BORINGSSL_VERSION` in upstream's
      `vendor/cpp-httplib/CMakeLists.txt`) remains an upstream follow-up; no downstream override.

### 5. Layer A: model-backed plain-C smoke (B.A)

- [ ] Set `JLLAMA_C_ABI_MODEL` in a suitable CI job and exercise the model-backed half of
      `c_abi`. The existing plain-C compile/smoke runs without a model; Java/C++/FFM inference
      coverage does not exercise this particular consumer. macOS and s390x remain static builds.

### 6. Release mirror — destination credential and first real push

- [ ] **Owner:** verify/provision `LLAMA_MIRROR_TOKEN` with contents-write access to
      `bernardladenthin/llama.cpp` in the `maven-central` environment before the release.
      Its current availability cannot be inferred from a snapshot run: `mirror-llama-source`
      is release-only and was skipped in run 39922567350. The credential-selection fix is on
      this branch; tests with temporary repositories do not prove real destination access.
- [ ] Check the first actual `mirror-llama-source` release run and the immutable upstream tag
      it creates. Source-archive creation and `test-source-archive` are already CI-covered.

### 8. Cached/uncached reproducibility measurement (C.3)

- [ ] Compare native hashes from cached and uncached builds of the same commit before considering
      a reproducibility job as a replacement for disabling the shared cache on release tags.
      The release-tag cache bypass is already implemented.

### 11. Windows code signature — deferred by the owner (C.4, C.5)

The signing implementation and `--require-signed` checks exist. A successful
`sign-windows-natives` job without the `sign_windows` switch is an unsigned pass-through and
does not prove Authenticode signing. The owner selected SSL.com IV + eSigner; macOS remains
ad-hoc signed with GPG/Sigstore provenance, without an Apple Developer subscription.

- [ ] **Owner:** delete the unusable Azure Trusted Signing account (9.99 USD/month at the time
      of the decision; individual identity validation was unavailable for Germany).
- [ ] **Owner:** obtain the SSL.com IV certificate, enroll it in eSigner and verify the quota
      against **35 DLL signatures per run**. Configure `ES_USERNAME`, `ES_PASSWORD`,
      `CREDENTIAL_ID`, `ES_TOTP_SECRET` in `maven-central`; keep the TOTP secret scoped to the
      signing job. Recheck pricing before ordering (the original estimate was ~129 USD/year
      for the certificate plus ~15–20 USD/month, metered, for eSigner).
- [ ] Dispatch a first signed run on `main` with `sign_windows` and without
      `publish_to_central`. Verify the publisher on an unpacked `jllama.dll` with
      `Get-AuthenticodeSignature` / `signtool verify /pa /v`, signed Windows smoke artifacts and
      `package`'s `--require-signed` check; then verify a signed snapshot publication.
      `ES_ENDPOINT=https://cs-try.ssl.com` can prove the sandbox flow, but its certificate is
      not trusted by Windows. Unsigned delivery remains possible under the existing owner decision.

## What still has to be measured

Kept as a list because each entry names the machine it needs -- the harness is
[`docs/benchmarks/ec2-harness.md`](docs/benchmarks/ec2-harness.md) and a run is cents:

| what | needs | why it cannot wait for CI |
|---|---|---|
| **OpenVINO on an Intel GPU/NPU** | suitable Intel device, unavailable to these sessions | the 21.7x/336x figures are CPU-plugin fallback only; they do not prove GPU/NPU execution |
| the two **SME** aarch64 modules | Linux aarch64 with SME, absent from current CI/cloud machines | macOS's static Metal library does not exercise these modules |

## Open — jllama-specific

### Deferred API features from the b11800–b11827 range

- **Server-side TTS through `LlamaModel` / the Java transport.** `POST /tts` (b11781, #26603) runs
  `SERVER_TASK_TYPE_TTS` on a chat model loaded with a TTS mmproj and streams `server_task_result_tts` chunks
  (`audio`, `sample_rate`, `final`; WAV or PCM per `out_type`). `NativeServer` serves it already (classic and
  attach mode, the route is in `0007`'s common table). A `LlamaModel.tts(...)` over the C API and a
  `/v1/audio/speech`-shaped route on `OpenAiCompatServer` would make it reachable without the native
  frontend; `TextToSpeech` (its own context over `mtmd_helper::gen_audio`) stays the one-shot path. Not before
  the release.
- **Multimodal rerank through `LlamaModel` / the Java transport.** Since b11820 (#29625) upstream's rerank
  route takes `{"text": ..., "image": ...}` items and splices the images through `process_mtmd_prompt`
  (`format_prompt_rerank(..., query_files, doc_files, ...)`, `handle_media()` for the decoding); `NativeServer`
  serves it. `engine::rerank` passes empty file lists, `jllama_rerank` and `LlamaModel.rerank` take strings,
  and `OpenAiCompatServer` refuses an `image` with a `400` naming `NativeServer` rather than ranking the text
  alone. Wiring it means items in the C API (text plus base64 media, as the multimodal prompt of
  `jllama_completions` -- `multimodal_data` -- already carries) and a typed overload on `LlamaModel`; it
  needs a multimodal reranker in the CI model set to be tested. Not before the release.
- **Realtime transcription** (`/v1/realtime`, WebSocket, b11780, #30636: `mtmd_audio_stream`,
  `SERVER_TASK_TYPE_TRANSCRIBE_RT`, `SERVER_CONTROL_ACTION_AUDIO_*`). Served by `NativeServer`; the JDK's
  `com.sun.net.httpserver` has no WebSocket, so the Java transport cannot follow without a dependency. Not
  planned.
- **`server_context_meta::has_cap_chat` / `has_cap_tts` / `has_audio_stream`** are not in the meta JSON
  (`ModelMeta` shows `"audio"` among the output modalities for a TTS-capable model, which covers the second).
  Add a `capabilities` object when the first item lands; `value.*` is PIT-gated at 100 %, so each getter
  needs its test.
- **`--connect` / `--connect-code`** (b11744, #28277): the server spawns the separate `llama-connect`
  binary (WebRTC tunnel to a browser outside the LAN). `LLAMA_CONNECT` stays OFF (a prebuilt download at
  configure time), `tools/server/server-connect.cpp` is compiled because `server.cpp` references it, and the
  flag answers with upstream's "not available" message. No plan to bundle a binary.

### Modular natives -- follow-ups

1. **Windows GPU modules on real hardware.** Verify device registration from the published jars: `cuda13-windows-x86-64` and
   `vulkan-windows-x86-64` next to `cpu-windows-x86-64` on the local agent's machine (RTX 3070 + AMD
   iGPU), the loader line naming both, llama.cpp listing the devices once each (`device_id` dedup),
   all layers offloaded, and the tg128 of the CUDA module against the 265.6 the static build measured
   (`GGML_CUDA_GRAPHS_DEFAULT` is repeated in the modular path, so the 88.8 of addendum 3 of
   `docs/handover/local-agent-report-b11538-windows.md` should be gone). Device indices are not stable
   across the set of loaded modules (`Vulkan0` was the NVIDIA GPU with CUDA loaded, the AMD iGPU
   without), so any device setting must go by name, never by index -- worth a README sentence once
   measured.
   Also check whether CUDA, Vulkan and OpenCL deduplicate the same NVIDIA GPU by `device_id`
   and which backend wins. Linux CUDA on a Tesla T4 has already been measured; see
   [the GPU module measurements](docs/benchmarks/gpu-module-jars.md).
2. **CUDA for Windows arm64 (`cuda13-windows-aarch64`) -- upstream builds it, we do not.** Their
   `windows-cuda` job is a matrix over `x64` and `arm64`, the latter through
   `cmake/arm64-windows-msvc-cuda` plus the arm64 `cudart`/`cublas` redistributables. This project has
   `cuda13-windows-x86-64` only. The arm64 Windows toolchain already exists here (the `cpu` and
   `vulkan` jars for that platform), so the missing piece is the CUDA toolkit's arm64 components and
   the toolchain file. Niche hardware today (Windows-on-ARM with a discrete NVIDIA GPU), but it is a
   straight coverage gap against upstream rather than a judgement call, and the "support more rather
   than fewer" policy that governs the ROCm targets applies. **Not** a gap: Linux aarch64 CUDA
   (Jetson, Grace-Hopper) -- upstream does not build that either, so it would be new ground.

3. **A Vulkan module for Android** (`vulkan-android-aarch64`, upstream ships none either): `JNI_OnLoad`
   already tries `libggml-vulkan.so` by soname, so it is one more module-only dockcross job plus the
   natives-jar wiring and an AAR flavor -- if anyone asks.
4. **Measure warm ROCm CI cache performance.** Read the second run's build timings and cache
   size for both ROCm jobs with the full GPU target lists. Local single-target measurements already
   exist in CLAUDE.md ("sccache on every Windows Ninja job" / "CI build cache"); the implemented
   Windows runtime-flag workaround and Linux HIP launcher no longer need implementation work.

### sccache — optional upstream and nvcc follow-ups

- **After the release:** propose support for `-Xclang -flto-visibility-public-std` in sccache
  (`src/compiler/clang.rs`, `PassThroughFlag`, with a test of CMake's full sequence). The Windows
  workaround in `llama/cmake/msvc-runtime-gnu-clang.cmake` already permits caching; see CLAUDE.md
  "sccache on every Windows Ninja job". This is independent of the nvcc issue below.
- **Re-test nvcc caching when a newer sccache is available.** sccache 0.18.0 with CUDA 13.4 failed
  with `Missing "cubin" file output` / `Compiler killed by signal 126` (runs 37680715063 and
  37925157175). `build.sh` therefore uses direct nvcc by default and caches the host compiler.
  On a branch, bump `SCCACHE_DL_VERSION`, enable `SCCACHE_WRAP_NVCC=true` for one CUDA job and
  inspect the compilation results and warm-run hits before changing the default. The existing
  uncached retry can otherwise cost another 47–80 minutes.

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

### Atmosphere coding agent (`llama-atmosphere-agent/`) — follow-ups

- **Tool rounds across REPL turns ride as a note, not as `tool_calls` messages -- and that cannot
  change on this Atmosphere.** The earlier entry here asked to keep the full `ChatMessage` list
  (incl. `tool_calls`/`tool`) per turn; the recorded measurement used `atmosphere-ai` 4.0.72:
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

- **Streaming raw-completion remainder:** token-streaming Ollama `/api/generate` (translate
  `text_completion` chunks to NDJSON, mirroring the chat→Ollama translator) and Continue's native
  `POST /completion` route in the llama.cpp-native
  streaming shape (`{"content":…,"stop":…}` per chunk). Java-only server wiring.
- **Future *output* modalities (audio / image) — design note, not yet actionable.** The Java
  transport's integration points are isolated (a new `OpenAiBackend.stream*` primitive +
  `OpenAiSseFormatter.*Chunk` per modality). Two future hooks:
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

### License compliance gate (owner's decision)

The FSFE REUSE check is green on every run. A FOSSA-style *dependency-license* gate (the scanner
whose "17 issues found" status once posted on PRs #248/#298; it posts nothing on `main` today) was
never set up deliberately. Whether the project wants one, with which policy, is the owner's call;
nothing here blocks on it.

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

#### Submission rules and remaining patches

**Two rules for the day.** State the measurement and ask the question; do not propose the fix for 1-3,
because each is a design decision in code this project does not own. And say plainly what was done
downstream instead (the opt-in, the non-aborting registration, the `-w` on `ggml-hip`), so a
maintainer can see what a merged change would let this project delete.

There are **twelve** patches today (`0001`–`0003`, `0006`–`0008`, `0014`, `0015`,
`0017`, `0018`, `0020`, `0021`). Each
accepted upstream change deletes a patch from the bump checklist once the pin is bumped past it.

- **Submittable as they are** (self-contained, no jllama specifics): `0001` (Windows arg-parse embed
  guard, already reported as ggml-org/llama.cpp#26416 with the reproducer in
  `docs/upstream-investigation-win32-argv-substitution.md`; the maintainers have not picked a
  direction), `0002` (preserve the caller's load-progress callback), `0006` (embeddable
  `llama_server`), `0007` (`llama_server_attach`), `0008` (`LLAMA_SERVER_WORKER_CMD`), `0014` (`common_log_set_callback`), `0015`
  (non-aborting RPC registration, stoppable server, transport fixes), `0017` (x86 prefetch helper),
  **`0018`** (the AMX buffer released with `free()` instead of `ggml_aligned_free` -- one line, with
  the 6/6-against-0/6 measurement in its header and a configure-time guard that is red without it;
  the clearest submission of the set, since the defect reproduces on any AMX machine and the fix
  copies the line `ggml-backend.cpp` already has), `0020` (validate leaf allocations before
  reusing a graph allocation), `0021` (invalidate the slot prompt cache when server LoRA adapters
  change; both hunks have independent drop-checks in `CLAUDE.md`).
- **Permanent:** `0003` carries upstream PR #22393, which upstream closed without merging.
- **Two upstream-only defects worth an issue on that day, no patch here:** a lost RPC server
  mid-inference still aborts the process (the ggml backend interface has no error return for a lost
  device), and ggml-rpc's client answers every `supports_op` with `true`, so a served device without
  an op aborts the server (both under "RPC backend" above). The separate sccache flag proposal is
  listed under "sccache — optional upstream and nvcc follow-ups"; the local workaround already
  permits plain-clang Windows caching.
- **A fourth, measured 2026-10-11 and again 2026-10-12: the Windows HIP build produces 83,362
  compiler warnings, and the Linux one produces 9.** The full measurement came from run
  38172107721, where the Windows ROCm job's log was
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

- **A third, measured 2026-10-11: `ggml_backend_load_best` picks a slower CPU module than one it
  rejects.** On a Graviton 4 it selects `armv8.6_2` (198.0 t/s pp512) while `armv8.2_2` reaches
  **271.4** on the same machine and model -- the SVE/SVE2/i8mm paths losing to plain DOTPROD+FP16 --
  with generation inside the noise. The scoring function ranks by feature *level*, not by measured
  throughput, so it is by construction rather than a bug, but it costs 37 % of prompt processing on
  the hardware AWS rents most cheaply, and it is invisible to anyone who only runs the module their
  own CPU picks. Worth an issue with the table from `docs/local-test-plan.md`; **no patch here** --
  overriding upstream's scoring downstream would be a maintenance burden with no end.

- **A fourth, measured 2026-10-19 (F187): `--lora-scaled` / `--control-vector-scaled` cannot name an
  absolute Windows path.** `common/arg.cpp` splits each `FNAME:SCALE` item with `string_split(item, ':')`
  and demands exactly two parts, so `X:\models\a.gguf:1.0` is rejected as three. Measured against the
  real parser in `BUGHUNT-MEASUREMENTS.md` M6 (of four argv forms only a relative path and plain
  `--lora` were accepted). The obvious fix is to split on the **last** colon; small enough for a PR,
  so draft it then. Downstream `ModelParameters` refuses such a path with the reason meanwhile.

- **From the branch review of 2026-10-20 (F221): upstream's Anthropic answer names a matched stop
  word in `stop_sequence` but says `stop_reason` `end_turn`**, where the Messages API specifies
  `stop_reason` `stop_sequence`. The fix is one line in each of the two `to_json_anthropic*` functions
  of `tools/server/server-task.cpp` plus an assertion in `test_compat_anthropic.py`; downstream
  `AnthropicApiSupport` corrects it meanwhile. Small enough for a PR, so draft it then.

Dropped-patch decisions, including `0012`, `0016` and `0019`, live in
[docs/history/dropped-llama-patches.md](docs/history/dropped-llama-patches.md).

### llama.cpp upstream feature exposure (queued, deferred by policy)

Add typed API surface only after a real user request. Raw native-server argv already reaches
upstream options; absence of a Java builder method alone does not make a release blocker.

- **JSONL logging:** decide whether the process-wide upstream logger or Java's `args.LogFormat`
  owns the format before exposing `--log-jsonl` / `--no-log-jsonl`; the callback sink is shared.
- **Request-key coverage:** re-derive remaining gaps by comparing `RequestField.values()` and
  the current upstream schema walked by `test_wire_contracts.cpp`. The old b10883 counts are not
  a current API inventory; expose additional fields on demand.
- **Video input (`ContentPart.videoFile(...)` / bytes overload):** the three video configuration
  flags are exposed, but the typed content part and its integration test remain. Emit upstream's
  `input_video` wire shape with base64/data URI/URL as supported by the current parser. Do not
  advertise the untested `imageBytes(..., "video/mp4")` workaround. A real test needs video-capable
  model/projector fixtures and `ffmpeg`/`ffprobe` on the runner, because upstream invokes them.
- **`--spec-synth-len` / `--spec-synth-rates`: deliberately not exposed.** They synthesize fake
  acceptance probabilities for upstream benchmarks. Keep them out of application builder API;
  they remain reachable through raw `NativeServer` argv.
- **Draft backend sampling:** consider
  `ModelParameters.setSpecDraftBackendSampling(boolean)` for speculative-decoding users who
  need to disable `--spec-draft-backend-sampling` explicitly.
- **Runtime reasoning control:** `InferenceParameters.setReasoningControl(boolean)` and a
  corresponding `LlamaModel.endReasoning(...)` action must land together. The flag only arms the
  sampler; the action needs access to the running slot/task. The existing reasoning budget/tag
  fields do not implement this runtime control (F252).
- **Output-slot allocation:** consider a `ModelParameters.setMaxOutputs(int)` mapping for
  `llama_context_params::n_outputs_max` when a consumer needs the memory limit.
- **Multi-Token Prediction:** consider `ModelParameters.setMtp(boolean)` for MTP-trained models,
  following upstream's current speculative-decoding configuration.
- **Suppressed-token inspection:** consider `LlamaModel.getSuppressTokens()` as a read-only
  view of the model's `tokenizer.ggml.suppress_tokens`. Upstream already applies the bias;
  this would expose diagnostics, not fix inference behaviour.

### ROCm Windows — real Radeon runtime verification

The ROCm Windows build uses HIP's clang as a plain C++ compiler, so it takes the hybrid CRT path
(`-fms-runtime-lib=static` plus `/nodefaultlib:libucrt.lib`). Linking succeeds; runtime behaviour
still needs measurement across the `/MT`/`/MD` library boundary, because **ROCm's device libraries
may use the dynamic CRT**. The same class of boundary is why CMake treats SYCL/OpenVINO separately.

The job has been **green** since, so the link holds. What is unmeasured is the **runtime** behaviour
of the resulting module on an actual Radeon GPU -- and that cannot be rented (see the rented-hardware
plan), because the only AMD-GPU instance class has been withdrawn and never officially supported
ROCm anyway. So this needs a real Radeon machine, or a deliberate decision to ship it on the strength
of a green link plus `verify-native-deps.py`'s dependency allowlist. Do not quietly treat "the job is
green" as the measurement.

### Hardware and test-fixture coverage

- **Real arm64 Android / Adreno OpenCL device:** run the AAR + Kotlin facade and LLM Service app
  on-device. The x86_64 emulator and app UI jobs already run in CI.
- **Kolibri-1 on GPU:** CPU inference and both published GGUF dialects were measured on the real
  78B model; GPU inference remains unmeasured. `0016` was dropped when upstream added Kolibri-1;
  see the retained history and numerical guard, not the former patch, for the current contract.
- **Structural test-constant wiring:** `TestConstantsTest.theShippedModelConstantsGoThroughTheResolver`
  now skips explicitly when the fixture layout cannot distinguish a missing resolver wrapper.
  The `src/test/resources/...` constants are indistinguishable by value even in CI. Consider a
  structural check of the wrapper calls; the resolver itself already has a hermetic fixture test.
- **Windows module measurements:** see "Modular natives -- follow-ups"; the remaining Intel
  GPU/NPU and Linux SME measurements are in "What still has to be measured".
- **Interactive agent:** real console/terminal behaviour needs a person at a terminal; model-backed
  headless integration coverage does not replace that check.

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** Review remaining public API surfaces for places where `@Nullable` is more precise than the `@NullMarked` default, respecting the existing `Optional<T>` policy and strict JSpecify/NullAway checks.

- **Review the project-wide `OPM_OVERLY_PERMISSIVE_METHOD` suppression in
  `llama/spotbugs-exclude.xml`.** The package layers now exist. Recheck the findings against the
  current public API before removing the suppression; align with BitcoinAddressFinder and the
  workspace's "OPM scope-tightening" policy. The former "single-root package" premise is stale.

- **Additional ArchUnit rules to consider:** further per-module import restrictions and public-API constraints if useful. Existing layer, Jackson-import and non-final-public-field rules already cover those cases.

- **Cross-repo code-quality TODOs** — see [`../workspace/policies/code-quality-todos.md`](../workspace/policies/code-quality-todos.md) for the canonical `@VisibleForTesting` design-fit review, package hierarchy review, and class/method naming review. This repo has no `@VisibleForTesting` usages today; package and naming reviews remain open.

## GPU backends on rented hardware

Use [docs/rented-hardware-test-plan.md](docs/rented-hardware-test-plan.md) and
[the EC2 harness](docs/benchmarks/ec2-harness.md) for new probes. Recheck current region capacity,
GPU vCPU quotas and prices before launching; the old quota-0 failure predates successful T4
measurements and is not evidence that GPU testing is currently blocked.

| remaining target | hardware needed |
|---|---|
| Windows CUDA/Vulkan/OpenCL coexistence and throughput | real Windows GPU host (RTX 3070 + AMD iGPU target in the local handover) |
| ROCm runtime, including Windows CRT behaviour | actual supported Radeon/Instinct GPU; the retired AWS g4ad is not evidence for current ROCm support |
| OpenVINO GPU/NPU and SYCL GPU execution | suitable Intel GPU/NPU host; CPU fallback does not prove GPU execution |
| `armv9.2_1` / `armv9.2_2` CPU modules | Linux aarch64 with SME; Graviton 4 lacks it, and macOS's static Metal library does not exercise these modules |

## Other open verification and maintenance

- **WER minidumps are switched on but unproven.** #506 makes the Windows test step set
  `Disabled=0` / `DontShowUI=1` under both WER keys next to the `LocalDumps\java.exe` entry
  (`DumpType` 2), because the AMX heap corruption left no dump behind. No Windows crash has happened
  since, so nothing shows that a `.dmp` now lands in the failure artifact. Prove it once rather
  than wait: a throwaway step that crashes a `java.exe` on purpose (an access violation through
  `sun.misc.Unsafe`, or `jshell` writing to address 0) on a branch run, then look for the dump in
  the artifact -- and if none appears, replace the WER route with `procdump -ma -e` around the
  surefire run.
- **Maven plugin policy:** the default release lifecycle plugins (`clean`, `resources`,
  `install`, `deploy`, `compiler`, `jar`) are pinned in this reactor. Still open across the four
  repositories: an explicit `site` version where that lifecycle is used and an enforcer
  `requirePluginVersions` rule, initially warnings-only while aligning the repositories.
- **Unread SonarCloud finding:** inspect the "1 new issue" reported on #486. The quality gate
  subsequently passed, but that does not identify or explain the finding; the sandbox proxy
  previously blocked access to sonarcloud.io.
- **REUSE attribution review:** check the remaining Java/documentation attribution to the original
  fork author with `git blame`, file by file. Attribution is not removed without an explicit owner
  decision. The generated dockcross wrappers already carry dockcross's own copyright attribution.
