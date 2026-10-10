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

Linux x86-64 and aarch64 ship the variant build since 5.2.0, Windows x86-64 since 5.2.x
(CLAUDE.md "CPU variants"; the measurements behind it are in
`docs/handover/local-agent-report-b11538-windows.md`). What is still open:

1. **GPU backends as modules (stage 3).** Measured feasible from a JVM: `ggml-cuda.dll` and
   `ggml-vulkan.dll` from upstream's zips load side by side from one directory with the CPU set
   (4 devices, incl. the AMD iGPU through Vulkan). Upstream's GPU zips carry a byte-identical copy of
   the whole CPU set; a GPU natives jar holding only its module would share ours and is then not
   usable without the CPU jar of the **same build**. Decide: consumers take `llama-platform` + GPU jar
   (the loader fails loud when the CPU part is missing or from another build -- a build key per
   natives jar, compared across jars), or GPU jars become artifacts of their own with a POM
   dependency. Also: `ggml-cuda.dll` imports only `cublas64_13.dll` (cudart is static) -- the
   requirement is smaller than the README says; and device indices are not stable across the set of
   loaded backends (`Vulkan0` was the NVIDIA GPU with CUDA loaded, the AMD iGPU without), so any
   device setting must go by name, never by index. **Measured through this project's own CMake at
   b11538** (`docs/handover/local-agent-report-b11538-windows.md`, addendum 3): `-DJLLAMA_CPU_VARIANTS=ON`
   together with `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` builds (nvcc accepts the plain-clang host
   compiler), and a directory holding the 14 CPU modules plus the GPU module offloads all layers;
   `verify-native-deps.py` accepts it. Two things stand between that and a shippable jar: **(A)** the
   GPU module is built into `build/bin/Release/` and never copied -- the file-list block in
   `llama/CMakeLists.txt` knows only the CPU modules, `ggml`, `ggml-base` and `ggml-rpc`, so
   `jllama-files.txt` and the copy would have to learn about GPU backend modules; **(B)** the CUDA
   module lost two thirds of its token generation because the variants path did not repeat
   `GGML_CUDA_GRAPHS_DEFAULT ON` -- fixed in the CMakeLists (both llama.cpp defaults are repeated now,
   CLAUDE.md "CPU variants"), not yet re-measured with CUDA. And such a jar is 18 files plus a 44-50 MB
   GPU module that `LlamaLoader` extracts once per build (reused across starts since 5.2.x).
2. **Benchmark on AVX-512/VNNI/AMX hardware** (Sapphire Rapids, Zen 4/5, Core Ultra). The Zen 3
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

### License compliance gate (owner's decision)

The FSFE REUSE check is green on every run. A FOSSA-style *dependency-license* gate (the scanner
whose "17 issues found" status once posted on PRs #248/#298; it posts nothing on `main` today) was
never set up deliberately. Whether the project wants one, with which policy, is the owner's call;
nothing here blocks on it.

### Upstream submissions -- not before the release (policy)

**Nothing is filed upstream before the release is out**, neither issues nor PRs; the drafts are not
written before that either. Everything below is the inventory for that day, so it is not rediscovered.

There are **eleven** patches today (`0001`-`0003`, `0006`-`0008`, `0012`, `0014`-`0017`). Each
accepted upstream change deletes a patch from the bump checklist once the pin is bumped past it.

- **Submittable as they are** (self-contained, no jllama specifics): `0001` (Windows arg-parse embed
  guard, already reported as ggml-org/llama.cpp#26416 with the reproducer in
  `docs/upstream-investigation-win32-argv-substitution.md`; the maintainers have not picked a
  direction), `0002` (preserve the caller's load-progress callback), `0006` (embeddable
  `llama_server`), `0007` (`llama_server_attach`), `0008` (`LLAMA_SERVER_WORKER_CMD`), `0012` (zero
  split-sum guard, ships `tests/test-model-split.cpp`), `0014` (`common_log_set_callback`), `0015`
  (non-aborting RPC registration, stoppable server, transport fixes), `0017` (x86 prefetch helper).
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
  an op aborts the server (both under "RPC backend" above). And sccache 0.18.0 cannot cache a TU that
  carries llama.cpp's unconditional `-Xclang -fno-pch-timestamp` (CLAUDE.md "Windows natives").

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

### Needs a machine the sessions do not have

Everything here is implemented or decided; what is missing is hardware or a model the cloud sessions
cannot reach. The local agent (Windows, Ryzen 7 5800H + RTX 3070 + AMD iGPU) covers the first three.

- **CUDA as a module, re-measured with the graphs default** (`GGML_CUDA_GRAPHS_DEFAULT` repeated in
  5.2.x): the 88.8 tg128 against 265.6 static should close; same build command as addendum 3 of
  `docs/handover/local-agent-report-b11538-windows.md`.
- **Extraction reuse on Windows**: a second JVM start of the same build must copy nothing (no
  `[jllama] extracted` lines), a running JVM's locked DLLs must survive another build's cleanup, and
  the start must not block on them.
- **On-device Android**: the AAR + Kotlin facade are exercised on the x86_64 emulator in CI
  (`test-android-emulator`) and the LLM Service app (`android-llmservice/`) is the example app, built
  and UI-tested there too; what no session has is a real arm64 device, and the Adreno/OpenCL flavor
  needs one. Nothing to write, only to run.
- **AVX-512/AMX hardware** (Sapphire Rapids, Zen 4/5, Core Ultra): item 2 of "CPU variants" above.
- **Kolibri-1 on the real 78B model** (after a full build; the shards need more than 64 GB of RAM).

### GraalVM Native Image evaluation

- **Evaluate GraalVM Native Image as an alternative distribution target.** Reference: [GraalVM Native Image](https://www.graalvm.org/latest/reference-manual/native-image/). The pure-Java sibling projects in the README's "Similar Projects" list (mukel's `llama3.java` / `gemma4.java` / `gptoss.java` / `qwen35.java` / `nemotron3.java`) demonstrate that single-jar, no-JNI Java inference is viable for individual model architectures. Native Image opens an orthogonal direction for THIS project: AOT-compile the Java layer + JNI bridge to a self-contained binary that bundles the libjllama.so (or per-OS equivalent) and starts in milliseconds without a JVM, which would make jllama usable in CLI tools, serverless functions, and short-lived processes where JVM startup is the dominant cost.

  **What to investigate before committing**:
  - **JNI-loading shape.** Native Image supports JNI but requires `--enable-native-access=ALL-UNNAMED` + reflection/JNI configuration files (`reflect-config.json`, `jni-config.json`, `resource-config.json`) describing every class/method/field reachable across the JNI boundary. The 34 native methods in `jllama.cpp` plus the JNI-side `FindClass` / `GetFieldID` / `GetMethodID` calls at `JNI_OnLoad` need to be mapped. The GraalVM tracing agent (`-agentlib:native-image-agent=config-output-dir=...`) can auto-generate the config during a representative test run, but the `LlamaLoader` JAR-extraction path needs at least one resource-config rule for `net/ladenthin/llama/{OS}/{ARCH}/lib*.so`.
  - **Native-library packaging.** The current `LlamaLoader` extracts the OS-specific `.so`/`.dll`/`.dylib` from the JAR to a tmp dir at first use. Native Image needs the same file at AOT-execution time, so either (a) ship the native lib alongside the produced binary as a sidecar file and adjust `LlamaLoader` to find it on the same directory, or (b) embed the native lib as a resource and keep the existing extract-to-tmpdir flow (which Native Image supports via `resource-config.json`).
  - **CUDA / Metal / OpenCL backend selection.** `LlamaLoader` already selects at runtime among the natives jars on the classpath (one directory per backend, fixed priority order). Native Image would need those directories as bundled resources (`resource-config.json`) or as sidecar files next to the binary.
  - **Startup-time benchmark to justify the work.** Measure cold-start of a current java-llama.cpp `LlamaModel(new ModelParameters().setModel("...").setNPredict(1))` invocation: how much is JVM startup + class load vs JNI load + model parse + tokenize + 1 token? If JVM startup is < 10 % of cold-start, Native Image yields little. If JVM startup is > 50 %, it's a clear win for CLI / serverless use cases.
  - **Maintenance cost.** Native Image adds a second build matrix (per OS × per backend × per JDK) and a new failure surface (Native Image config drift when a llama.cpp version bump adds new JNI-reachable types). Should ship only with a CI job that exercises the Native Image build on at least one OS, otherwise the config files will rot silently.

  **Out of scope until evidence supports it**: actually implementing any of the above. This entry exists so that when someone asks "can I ship java-llama.cpp as a single 30 MB binary?" the answer points to a concrete investigation plan rather than restarting from zero.

### macOS model-backed server smoke (optional depth, not scheduled)

`smoke-fatjar-macos` verifies the dylib inside the packaged jar (code signature + a real JVM load
and JNI round-trip; green in CI since run 37985786772). A full model-backed macOS server smoke as
Linux and Windows run would need `verify-model-cache` plus a cache restore and -- there is no
`all-macos-*` fat jar -- `smoke-test-fatjar.sh` against the default jar. Worth doing only if a
macOS-specific *inference* regression ever appears; the load-time failure class is covered.

### Test-coverage gaps found by the b10679 mutation audit (PR #403)

> **Update.** The `IdleSleepWakeIntegrationTest` added to close the `wake_and_post` gap immediately
> found a real JVM crash (SIGSEGV on all six CI platforms) — see the CHANGELOG "Fixed" entry. Both
> facets are fixed in that PR via the `wake_server()` choke point. This is the clearest evidence for
> the floor on executed tests that `.github/verify-test-counts.sh` enforces since: the defect had been
> reachable from public API for as long as `--sleep-idle-seconds` has existed, and nothing ran that path.

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

### Test-coverage debt found during the b10649 review (PR #403)

Each item below was verified against pristine upstream tags and is real, but none is a regression
introduced by the version bump — they were deferred to keep that PR landable. (The seven dead CLI
flags, the `acquire`/`release` context-guard unit tests, the `OSInfo` alias test and the dead
`Java8CompatibilityHelper` from the original list are done.)

- **`LlamaLoader`'s jar-extraction internals are tested only through directory fixtures.**
  `BackendLoadTest` drives backend probing, extras, fallthrough and forcing over the committed
  `Linux/backendtest*/` trees on the test classpath (directories, not jars); extraction out of a real
  jar is exercised in CI by `smoke-natives-jars.sh` and the fat-jar smokes, not by a unit test.

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
