# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from version 5.0.0 onward. Pre-fork releases (`1.x`–`4.2.0`) were authored by
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp).

## [Unreleased]

### Changed
- **BREAKING (runtime): the shipped SLF4J binding is now `slf4j-simple`, not `logback-classic`.**
  Two independent reasons, and the first is a hard failure rather than a preference:

  1. **This artifact targets Java 8 and logback no longer does.** Every logback release from 1.4.0 on
     is class-file major 55 (Java 11). SLF4J's `ServiceLoader` loads `LogbackServiceProvider` at JVM
     startup, so a Java 8 consumer got `UnsupportedClassVersionError` before a single line of library
     code ran. Measured: all 181 classes of logback-classic 1.6.3 are major 55.
  2. **The Java 8 logback line is end-of-life with unfixed CVEs.** 1.3.16 (2025-10-29) is its last
     release; CVE-2026-1225, CVE-2026-9828 and CVE-2026-10532 were fixed only in 1.5.x, and
     CVE-2026-19880 only in 1.6.3 — which is Java 11 bytecode and therefore unreachable from here.
     Downgrading would have traded a crash for permanent unpatchability.

  `slf4j-simple` is six classes from the same release train as `slf4j-api`, with no configuration
  parser, socket server or deserialization — the subsystems essentially every logback CVE lives in.

  **What changes for you:** `logback.xml` is no longer read. Configure with a classpath
  `simplelogger.properties` or `-Dorg.slf4j.simpleLogger.*`. With no configuration at all, output is
  quieter than before (logback defaulted the root logger to DEBUG; slf4j-simple defaults to INFO).
  To keep logback, exclude `org.slf4j:slf4j-simple` and declare your own binding — which is what the
  SLF4J api/binding split is for.

  **The runnable fat jar carries logging defaults; the library jar deliberately does not.**
  `simplelogger.properties` (INFO, stdout, timestamps, thread + short logger name — what the previous
  logback default emitted) is added by the assembly, from `src/main/assembly-resources/`. It is *not*
  under `src/main/resources`, because from there it would be published inside the library jar and land
  on every consumer's classpath: slf4j-simple reads whichever file the classloader hands it first, so
  a consumer with their own configuration would get a coin flip. A library must not decide that. The
  `assembly` profile therefore uses its own descriptor — a verbatim copy of the predefined
  `jar-with-dependencies` plus that one file.

- **`checker-qual` pinned to 3.55.1 and marked optional.** 4.x is major 55 and its annotations are
  `@Retention(RUNTIME)`, so a Java 8 JVM throws `UnsupportedClassVersionError` the moment anything
  reflects over an annotated element. The optional flag keeps it out of consumers' transitive graph;
  the version pin is what protects the fat jar, since `jar-with-dependencies` filters on scope only.
  The build-time Checker Framework processor stays on 4.2.2 under its own property.

### Added
- **`ModelParameters.setFlashAttn(FlashAttn)` — the only way to express `--flash-attn` correctly.**
  llama.cpp turned that option from a bare flag into a value-taking one in **b10273**: the
  `on|off|auto` value is mandatory, so emitting the key alone makes the parser consume whatever argv
  token happens to follow it. The failure is as misleading as it sounds — the load dies naming a flag
  the caller never set, e.g. `error: unknown value for --flash-attn: '--reasoning-format'`.

  The new `args.FlashAttn` enum follows the existing `CacheType` / `TensorReadLazyMode` pattern, so
  the option is now expressible: `AUTO` is upstream's own default, `ON` forces it and fails the load
  where the backend cannot provide it, `OFF` disables it.

### Changed
- **llama.cpp `b10731` → `b10797`.** No project-source change: every header move in the range is
  additive or a **widening** const-qualification, and the server wire contract is byte-identical
  (request-field set, `set_hard_limits` bounds and response keys all verified mechanically, which is
  the check that catches the contract-behind-a-stable-signature breaks a header diff cannot see).
  All 8 local patches still apply.

  Two upstream changes are worth knowing even though nothing broke:

  1. **`preserve_reasoning` now defaults to enabled** (llama.cpp #28174). `common_params_parse_ex`
     sets it when the caller did not, where it previously followed the chat template's own default.
     On a template advertising `supports_preserve_reasoning`, the full history now carries reasoning
     traces instead of only the last assistant message — better continuity, **more prompt tokens**.
     It applies to every entry point that parses argv, so `NativeServer` in both modes and
     `LlamaModel`'s own parameter parse. Pass `--no-reasoning-preserve` to restore the old behaviour.
  2. **`data:` URLs are now accepted for `input_audio` and `input_video`**, not only images
     (llama.cpp #27735) — so an OpenAI-style request may inline base64 audio/video the same way it
     could already inline an image.

  Also picked up: a fix for **Qwen3-TTS-0.6b** (llama.cpp #28231), where an F16 `ffn_down` overflowed
  on intermediate peaks past the 65504 ceiling and turned the residual into NaN. That is on the
  `TextToSpeech` path this project ships.

  The final `b10792` → `b10797` step touches **nothing** the project links against — not one file
  under `common/`, `include/`, `tools/server/`, `tools/mtmd/` or `ggml/include/`. It carries a
  **big-endian correctness fix** that does matter to a platform this project ships in the default
  JAR: `ggml-cpu`'s s390x `q5_1` path used an uninitialized `v_acc` accumulator (llama.cpp #28332),
  so `Q5_1`-quantised inference on IBM Z could return garbage. There is also an upstream build
  change (#28278) that moves `LLAMA_VERSION`/`LLAMA_COMMIT` from compile definitions into a
  generated `llama-version.h`; it does not reach `getLlamaCppBuildInfo()`, which reads
  `llama_build_info()` from `common/build-info.cpp` instead — verified by the smoke test that
  cross-checks the pin against the linked binary.

- **llama.cpp `b10682` → `b10731`.** One project-source change came out of it, and it is the kind a
  header diff does not surface: upstream renamed `--tensor-read-lazy` to `-lzm` / `--lazy-mode`
  (env `LLAMA_ARG_TENSOR_READ_LAZY` → `LLAMA_ARG_LAZY_MODE`) **with no alias**. The binding emitted
  the old spelling, so every model load with the knob set would have failed on an unknown argument —
  a contract change behind an unchanged signature.

  Everything else in the range was ruled out mechanically: `common/speculative.h` is byte-unchanged
  (only the `.cpp` moved), the two touched ggml headers have **zero deletions**, and the 42 files the
  eight patches touch were intersected against the changed-file list — the sole hit is
  `common/arg.cpp`, whose change sits at line ~2729 while patch `0001`'s hunks there are at
  1201/1242. Confirmed by the real fail-loud applier: fresh `cmake -B build-b10731` configured clean,
  `ggml commit: 0eadefebd`, stamp written over all eight patches.

- **`TensorReadLazyMode` → `LazyMode`, `setTensorReadLazy` → `setLazyMode` — breaking.** The binding
  follows upstream's rename rather than papering over it; keeping the old names would leave the API
  describing a flag that no longer exists.

### Removed
- **`ModelParameters.enableFlashAttn()` and `ModelFlag.FLASH_ATTN` — breaking.** Both modelled
  `--flash-attn` as a valueless flag, which it has not been since b10273. Keeping either would leave
  the broken argv reachable: the method directly, the enum constant through the public
  `setFlag(ModelFlag)`. Replacement: `setFlashAttn(FlashAttn)`.

  Deprecating instead was considered and dropped. A deprecated method that still emits an argv
  llama.cpp misparses is a trap with a warning label on it, and this is a major-version window.

### Fixed
- **A test pinned the broken argv shape as correct.** `ModelParametersExtendedTest`'s
  complex-combination case asserted a 9-token argv built with `enableFlashAttn()` — i.e. it encoded
  the valueless emission as the expected contract, which is why no gate ever flagged it. It now uses
  `setFlashAttn(FlashAttn.ON)` and asserts 10 tokens, and a separate test pins the deprecated
  method's emission explicitly as the defect it is, so the two cannot be confused again.

## [5.1.0] - 2026-08-29

> The entries below also cover the **b9917 → b10456** window (PRs #341–#394), which went unrecorded
> here while it happened; they were reconstructed from the git history and from
> [`docs/history/llama-cpp-breaking-changes.md`](docs/history/llama-cpp-breaking-changes.md), which
> has a row per upgrade range and stays authoritative for the per-range detail.

### Fixed
- **JVM crash (SIGSEGV) on the first request after an idle-sleep window.** With
  `--sleep-idle-seconds` set, upstream's `handle_sleeping_state(true)` calls `destroy()`, which frees
  the model and context and nulls `ctx_tgt`/`model_tgt`. Two things in the JNI layer assumed they
  outlived that: `server_context::get_meta()`, read on every request before the task is posted, and
  the `jctx->vocab` pointer captured once after the initial load — dangling after the reload replaces
  the model. The first was a null dereference that aborted the JVM at `llama_context::get_model()`;
  the second a use-after-free on every tokenize/detokenize/rerank path. Both now go through a single
  `wake_server()` choke point that waits out the sleep and re-reads the vocab, called from every entry
  point that touches the model. The earlier `wake_and_post()` fix was necessary but not sufficient:
  it woke at *post* time, and these reads happen before the post.
  Fixing that exposed a third, latent defect in our own `patches/0002`: it guarded upstream's
  progress-callback install on `== nullptr`, but `load_progress_text` is a **local** of
  `load_model()` whose address upstream re-assigns on every call. On resume the guard saw our own
  callback from the first load, skipped the re-assignment, and left `user_data` pointing into a dead
  stack frame — a second SIGSEGV, this time inside `load_progress_callback()`. The guard now also
  accepts its own callback, so our `user_data` is refreshed on every load while a caller-supplied
  callback still survives.
- **`ModelParameters.setSleepIdleSeconds`** now rejects `0` and values below `-1`, which upstream's
  own handler throws on. Emitting them aborted the whole argv parse and surfaced only as
  `"Failed to parse model parameters"`, naming neither the flag nor the reason. Its Javadoc also said
  the server "shuts down" after the idle window; it does not — it releases the model and reloads it on
  the next request.

### Added
- **`ModelParameters.setCpuMoeLayers(int)` / `setCpuFfnLayers(int)`** — keep the first N layers'
  Mixture-of-Experts weights, or dense FFN weights, on the CPU (upstream `--n-cpu-moe` / `-ncmoe` and
  `--n-cpu-ffn` / `-ncffn`). The companions to `setGpuLayers`: where that moves whole layers, these move
  only the weight class that dominates a model's size, usually fitting a much larger model into the same
  VRAM at a smaller speed cost. Only `--n-cpu-ffn` is new (llama.cpp b10645); `--n-cpu-moe` has existed
  upstream since b6089 but had never been exposed here.
- **`ModelParameters.setVideoFps(float)` / `setVideoTimestampInterval(long)` / `setVideoFfmpegDir(String)`**
  — the video-decoding knobs upstream added in llama.cpp b10647 (`--video-fps`,
  `--video-timestamp-interval`, `--video-ffmpeg-dir`). They apply to any media attached to a request
  once a projector is loaded: `server_context` copies them into the `mtmd_helper_init_opt` it passes
  to `process_mtmd_prompt`, and video decoding is compiled into the shipped library (`MTMD_VIDEO`
  defaults on). `setVideoFfmpegDir` is the significant one — upstream otherwise looks `ffmpeg` and
  `ffprobe` up on `PATH`, which a JVM process often does not have them on.
- **`ModelParameters.setKvUnifiedPerSlot(int)`** — caps the context each parallel slot may use
  (upstream `--kv-unified-per-slot`, new in llama.cpp b10662). The cap reaches this binding through
  `server_context_meta::slot_n_ctx`: it becomes every `slot.n_ctx` and is the context budget passed to
  `format_prompt_infill`. Upstream's second effect — sizing the
  shared KV pool to `n_parallel * N` when no context size is given — lives in `llama_server()` and
  therefore applies to `NativeServer` only, not to a model loaded from `ModelParameters`; the
  Javadoc says so.
- **`ModelParameters.setTensorReadLazy(TensorReadLazyMode)`** and the new
  **`net.ladenthin.llama.args.TensorReadLazyMode`** enum (`OFF` / `AUTO` / `ON`) — on-demand reading
  of tensors the model architecture marks as lazy-loadable, such as per-layer embeddings (upstream
  `--tensor-read-lazy`, new in llama.cpp b10653, mapping to `llama_lazy_mode`). Trades resident
  memory for disk reads and requires mmap. It reaches the plain `LlamaModel` load path too, because
  `common_model_params_to_llama` copies `lazy_mode` into `llama_model_params`.
- **`ServerMetrics.getWindowPromptProcessingMillis()` / `getWindowTokenGenerationMillis()` /
  `getWindowTimings()`** — typed access to the current-window timing keys `t_prompt_processing` and
  `t_tokens_generation`. Both were always emitted; only the cumulative `_total` variants had accessors.
- **`ModelMeta.supportsVideo()`**, and `getModelMeta()` now emits `modalities.video`. Upstream has
  tracked `has_inp_video` on `server_context_meta` for releases and emits all three modalities from its
  own `/props`; this binding emitted only vision and audio, so feature detection concluded no model
  ever accepts video.
- `QuantizationType.Q2_0` — maps the new upstream `LLAMA_FTYPE_MOSTLY_Q2_0` (llama.cpp b9916) for `LlamaQuantizer`.
- **Voice cloning and language selection for `TextToSpeech`**: `synthesize(String text, String speakerReferenceAudioPath, String language, int maxFrames, int topK, int seed)` — a speaker-reference clip makes the model imitate that voice. Part of the Qwen3-TTS rework (see Changed).
- **`ModelParameters.setMmprojDevice(String)`** — places the multimodal projector on a device of its own
  (llama.cpp `--mmproj-device`, added upstream in b10541), independently of `setDevices(...)`. Exactly one
  device may be named; the literal `"none"` keeps the projector on the CPU. `OpenAiCompatServer`'s CLI
  accepts the same flag as `-mmdev`/`--mmproj-device`; `NativeServer` already forwarded it verbatim.
- **`RouterClient` API-key constructors** (`RouterClient(int, String)`, `RouterClient(String, int, String)`) —
  send `Authorization: Bearer <key>`, which a router started with `--api-key` requires for *every* call:
  `/models/load` and `/models/unload` were always gated, and since b10519 (upstream #26347) the listing
  endpoints are too. An empty key behaves like none, and `toString()` never prints it.
- **`ServerMetrics` cache and speculative-decoding counters** — `getCumulativeCachedPromptTokens()`,
  `getDraftTokensTotal()`, `getDraftAcceptedTotal()`, `getDraftVerifyStepsTotal()`,
  `getDraftAcceptedPerPosition()` and the derived `getDraftAcceptanceRate()`. Upstream exposes these only
  as Prometheus text; they now arrive in the JSON payload.

### Changed
- **Upgraded the pinned llama.cpp from b10679 to b10682.** No project-source change, and the range
  cannot require one: the whole delta is ten files (575 insertions / 59 deletions, 53 KB) confined to
  `ggml/src/` backend implementations — Metal flash-attention vec tunings for M1 Max
  ([ggml-org/llama.cpp#27932](https://github.com/ggml-org/llama.cpp/pull/27932)) and a Vulkan
  `mul_mat_id` change that pads K rather than N
  ([ggml-org/llama.cpp#27925](https://github.com/ggml-org/llama.cpp/pull/27925)) — plus the Snapdragon
  Windows SDK scripts ([ggml-org/llama.cpp#27903](https://github.com/ggml-org/llama.cpp/pull/27903)),
  documentation, and one upstream test that this project never compiles. Nothing under `common/`,
  `include/`, `tools/server/`, `tools/mtmd/` or `ggml/include/` moved, and the 42 files the eight
  local patches touch have an empty intersection with the changed-file list, so all eight apply
  unchanged. The Metal and Vulkan classifier artifacts pick the backend work up by rebuilding.
- **Deprecated `InferenceParameters.withTfsZ`, `withPenalizeNl` and both `withPenaltyPrompt` overloads.**
  `tfs_z`, `penalize_nl` and `penalty_prompt` appear nowhere in upstream `common/` or `tools/server/`
  at the pinned build, and the request schema discards unknown fields rather than rejecting them — so
  these have been silently doing nothing. Kept compiling for now; they will be removed.
- `ModelParameters.setMmprojDevice` and `setMmprojOffload` now clear each other. Both write upstream's
  single `mmproj_use_gpu` field, and the rendered argv comes out of a `HashMap`, so leaving both present
  left the winner to hash order. Clearing in only one direction still lost the race whenever
  `setMmprojOffload` was called second; the contract is now simply "the last of the two calls wins".
- **Deprecated `InferenceParameters.withUseChatTemplate` and `withChatTemplate`.** Both are load-time
  settings upstream, not per-request ones: `common_params::use_jinja` is set only by `--jinja` /
  `--no-jinja`, and the only `"chat_template"` string in upstream `common/` or `tools/server/` is the one
  the server *emits* from `/props`. Neither key is ever read from a request body, so both calls were
  silently doing nothing — including at three call sites in this library that used
  `withUseChatTemplate(true)` to "enable jinja for tools", which those calls could not do. Use
  `ModelParameters.enableJinja()` / `setChatTemplate(String)` instead. Tool calling was unaffected in
  practice only because upstream defaults `use_jinja` to true.
- `ch.qos.logback:logback-classic` bumped 1.6.2 → 1.6.3 (test/runtime binding only).
- CI actions bumped to latest: `actions/setup-java` v5 → v6.
- Upgraded llama.cpp from **b9894 to b9917** (all eight local patches re-verified across the range).
- **BREAKING — `TextToSpeech` was reworked onto Qwen3-TTS** (llama.cpp **b10270**, upstream #26254, which
  upstream itself labels a breaking change). llama.cpp deleted the OuteTTS pipeline outright:
  `tools/tts/tts.cpp` shrank from ~1450 to 205 lines and `mtmd_gen_audio_type` has only
  `NONE`/`QWEN3TTS`, so there is no OuteTTS code path left anywhere upstream and no compatibility shim
  was possible. The two-argument constructor keeps its **signature** but changes **meaning**:
  `(ttcModelPath, vocoderModelPath)` → `(modelPath, mmprojPath)`, i.e. a Qwen3-TTS backbone plus the
  mmproj that bundles speaker encoder, code predictor and code2wav decoder — an OuteTTS + WavTokenizer
  pair no longer works and fails at load, not at compile time. `synthesize`'s `maxCodeTokens` parameter
  became `maxFrames`, and the single-argument overload's default dropped 4096 → 512.
- **BREAKING — `-1` is no longer accepted for the repetition-penalty windows** (llama.cpp **b10273**).
  `repeat_last_n` and `dry_penalty_last_n` used to take `-1` for "the whole context"; upstream removed
  the sentinel, moving the request schema's hard limits to `[0, INT32_MAX]` and making
  `common_params_parse` throw on a negative value. `ModelParameters.setRepeatLastN` /
  `setDryPenaltyLastN` and `InferenceParameters.withRepeatLastN` / `withDryPenaltyLastN` had kept
  advertising and accepting `-1`, so the value reached llama.cpp and failed there — at model load for
  the launch flags, as a rejected request for the per-request withers. All four now reject a negative
  value with a message naming the change; pass the context size explicitly for the old behaviour.
  (Verified exhaustively: these are the **only** two request-field limits that moved in the whole
  b9994 → b10618 range.)
- Upgraded llama.cpp from **b9917 to b10456** across PRs #341–#394. Local patches `0005` (b9981) and
  `0004` (b9982) were dropped after upstream merged equivalent — and broader — fixes, and `0009`
  (`subprocess.h` old-glibc build break) was dropped at b10280 once upstream vendored the same fix.
- `server-mcp.cpp` is compiled into `libjllama` (llama.cpp **b10154** added upstream MCP-server
  support; `server.cpp` and `server-tools.cpp` reference `server_mcp`, so omitting it is latent on
  Linux but a hard link error on macOS/ld64 and Windows/MSVC). The `subprocess.h` `addchdir_np` use is
  guarded for old glibc in the same change.
- Android/Gradle toolchain: Gradle pins moved 8.14.3 → 9.6.1 and the dockcross cross-compile images
  were bumped, alongside the AGP/Compose pin updates the Android builds needed.
- **Post-upgrade audit of the whole b10456→b10644 range** — three independent sweeps over the
  upstream diff (completeness, adaptation correctness, test integrity) against the files the binding
  actually consumes. No missed adaptation was found: the request-field set, their bounds and the
  emitted response keys are identical at both ends of the range, and `libjllama` links with zero
  undefined upstream symbols. The audit did surface documentation and coverage gaps, fixed here:
  - The `-1` context-size sentinel was dropped upstream at **b10273** (#26524), not b10275 — corrected
    in 4 Javadoc blocks, 4 exception messages and every doc that cited it. The `server-schema.h`
    signature break is **the same** upstream commit, not an unrelated one: #26524 at b10273 dropped
    `eval_llama_cmpl_schema`'s `n_ctx_slot` parameter in the same change that removed the sentinel
    (`git diff b10273 b10275 -- tools/server/server-schema.h` is empty).
  - `LlamaModel.saveSlot`/`restoreSlot` now document that the on-disk format is version-locked to the
    linked llama.cpp build, and that a mismatch surfaces as upstream's misleading
    `"No available space in KV cache or invalid slot save file"`.
  - `getMetrics()` documents that the merged payload is not an atomic snapshot and that it defers
    idle-sleep, which upstream's own `/metrics` stopped doing at b10519 (#27376, which introduced
    `task_resets_idle_timer`). It cannot have been b10644: `git diff --name-only b10639 b10644 --
    tools/server/` is empty.
  - **`--tools get_datetime` no longer starts.** Upstream deleted that built-in tool in this range and
    an unknown name is fatal (`server_tools::setup` throws), so a `NativeServer` command line carrying
    it now fails at startup. Same block: `server_tool::type()` reports `"server"` instead of
    `"builtin"`, changing the `/tools` payload in full `NativeServer` mode.
  - The four `t_*` keys in `getMetrics()` are now fractional rather than whole milliseconds, because
    the merge divides upstream's microseconds. `ServerMetrics` reads them as doubles; a consumer
    parsing the raw JSON with an integer parser sees a type change.

- Upgraded llama.cpp from **b10649 to b10679**. No project-source change: all twelve
  `tools/server/*.h` headers, `server-schema.cpp`, `server-task.cpp`, `server-common.cpp`,
  `common/chat.h` and `tools/mtmd/mtmd-helper.h` are byte-identical across the range (compared by blob
  SHA), so the request-field set, its bounds and the emitted response keys cannot have moved and the
  three mechanical contract checks are moot. The whole in-scope delta is 8 files, 172 insertions and
  15 deletions — the rest of the 159-file range is `tools/ui` (rebuilt from `GIT_TAG` by CI), the ggml
  backends, and `conversion/`, `gguf-py/`, `tests/`, `.github/`, `docs/`, `scripts/` and the standalone
  `tools/` binaries, none of which this project compiles.
  Two additive upstream features are new and both are now exposed (see Added):
  `--kv-unified-per-slot` and `--tensor-read-lazy` / `llama_lazy_mode`. Three patch-target files were
  touched (`common/arg.cpp`, `tools/server/server-context.cpp`, `tools/server/server.cpp`) and all
  eight patches still apply with zero fuzz; patch `0007`'s invariant holds because the new
  KV-pool-sizing block in `llama_server()` sits before the extracted route table, not inside it.
  `llama_model_quantize_params` gained `max_buf_size`, which needs no adaptation because
  `LlamaQuantizer` builds its params from `llama_model_quantize_default_params()`. Upstream's private
  `get_slot_n_ctx()` → `n_ctx_slot()` rename is invisible here — the project reads the value through
  the unchanged `server_context_meta::slot_n_ctx`.
  Patch `0001` shrank from 37 to 36 files: upstream rewrote `tests/test-save-load-state.cpp`'s
  `main()` to build its own filtered argv, so by the patch's own rule that call site now wants
  `common_params_parse` and no longer the `_main()` flip. The patch itself is still required —
  `common_params_parse` at b10679 still carries the count-guarded `GetCommandLineW` override and
  `common_params_parse_main` does not exist upstream.
- Upgraded llama.cpp from **b10644 to b10649**. The first range in this bump to break the project's own
  compile: upstream threaded a new `mtmd_helper_init_opt` (video-decode settings) through every helper
  that can ingest media, changing the signature of `mtmd_helper_bitmap_init_from_file`,
  `tokenize_input_prompts` and `format_prompt_rerank`. Four call sites were adapted — all of them pass
  `mctx = nullptr` or handle audio, so each now passes upstream's own `mtmd_helper_init_opt_default()`.
  The wire contract is unchanged: 68 request fields and 23 bounds identical across the range, and
  the emitted response-key set identical for every server TU the project compiles (the exact key count
  depends on which TUs are swept — the load-bearing half is that it does not move). Zero CLI flags were
  removed or renamed, and all eight local patches apply unchanged even though six patch-target files
  were touched.
  Of the 6 new upstream flags, four are now exposed (see Added): `--n-cpu-ffn` and the three
  `--video-*` knobs. `--n-cpu-moe` is exposed alongside them but is not new — it has existed upstream
  since b6089 and had simply never been surfaced here. The two `--spec-synth-*` flags stay unexposed:
  upstream marks them "benchmarking only" — they synthesise fake acceptance probabilities to measure
  llama.cpp's own speculative harness. The `--video-*` trio was initially refused as inert without a
  `ContentPart` video factory; a follow-up audit showed that was wrong on both counts (they reach the
  task path this binding drives, and `MTMD_VIDEO` is compiled into the shipped library), so they are
  exposed. The content part itself — upstream's `input_video`, which takes raw base64 rather than a
  `data:` URI — remains in `TODO.md`.
- Upgraded llama.cpp from **b10639 to b10644**. No project-source change, and the only file on the
  priority API-review list that the range touches is `include/llama.h`, whose entire diff is two
  constants: `LLAMA_SESSION_VERSION` 9 → 10 and `LLAMA_STATE_SEQ_VERSION` 2 → 3. They follow from a new
  `tok` field on `llama_kv_cell_ext` (n-gram input embeddings) that has to survive a state save/restore.
  Everything else is the Snapdragon/Hexagon backend rework, a one-line fix in the nanbeige model graph,
  and the WebUI. Nothing under `common/`, `tools/server/` or `tools/mtmd/` changed, so no request field,
  no bound and no response key can have moved, and all eight local patches apply unchanged.
  **One consumer-visible consequence:** the version bumps are a *state-file format* break. A slot state
  saved by an earlier build — via the public `LlamaModel.saveSlot(int, String)`, or the server's
  `/slots/{id}?action=save` — is rejected by `LlamaModel.restoreSlot` after this upgrade and has to be
  regenerated. No Java or native signature changed. The rejection is graceful but its message is
  upstream's misleading `"No available space in KV cache or invalid slot save file"`, which does not
  name the version mismatch; `saveSlot`'s Javadoc now spells this out. Slot state files are a cache to
  regenerate on upgrade, not durable storage. The in-memory `Session` snapshot/fork feature is
  unaffected — it never writes a file.
- Upgraded llama.cpp from **b10631 to b10639**, in two reviewed steps. Neither range changes any
  project source. b10631→b10636 is ggml-cuda quantised-matmul configs for Pascal, ggml-metal
  SSM/Mamba kernels, an upstream `LLAMA_BUILD_UI` default flip that is inert here (this project
  compiles its own `webui-generated/ui.cpp`), and the WebUI. b10636→b10639 is the RPC backend's
  event/async APIs (#18626, protocol 5.1 → 6.0 — `GGML_RPC` is never enabled in this project, so
  `ggml-rpc.cpp` is not compiled) plus Vulkan `cross_entropy_loss` kernels (#27216) and a warptile
  clamp for warp sizes > 64 (#27726). Neither range touches `common/`, `include/llama.h`,
  `tools/server/` or `tools/mtmd/`, so no request field, no bound and no response key can have
  moved. All seven local patches apply unchanged.
- Upgraded llama.cpp from **b10618 to b10631**. No project-source change. The only **project-relevant** edits in the
  range are a narrowing input validation in `oaicompat_chat_params_parse` (continuing a final
  assistant message that carries `tool_calls` now throws), a Qwen3-Coder-only grammar refinement
  in `common_chat_params_init_qwen3_coder`, a cosmetic `LLAMA_VERSION_MINOR` bump, and the WebUI.
  `server-schema.cpp`, `server-task.cpp`, `server-context.cpp`, the `tools/server/*.h` headers,
  `common/common.h`, `include/llama.h` and `mtmd-helper.h` are byte-identical across the range, so
  neither the request-field set and its bounds nor the emitted response keys can have moved. All
  seven local patches re-verified against a clean b10631 checkout; C++ suite 499/499.
- Upgraded llama.cpp from **b10456 to b10618**, in 25 reviewed steps. Patch `0007` refreshed (upstream
  #26347 deleted comments inside its removal block, breaking `git apply` at every tag from b10519 on) and
  a new patch `0010` carries a one-line upstream fix: `GET /models` emitted `vocab_type` as a JSON boolean
  after the `common_json` switch (#27511), because an unscoped enum binds to the `bool` constructor.
  The project's own C++ moved to `common_json` in the same range.
- **`apply-llama-patches.cmake` is now genuinely idempotent**, via a stamp file (llama.cpp commit plus each
  patch's SHA-256) gated on git's clean/dirty state. Reconfiguring an existing build directory is a no-op
  instead of aborting with a misleading "does not apply cleanly"; a real mismatch fails with an accurate
  message. A source tree supplied via `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP` that is not a git work tree
  keeps the previous per-patch behaviour.
- `ServerMetrics.getStartTimestamp()` is documented correctly: `t_start` is a monotonic-clock **microsecond**
  reading (`ggml_time_us()`), not milliseconds since the epoch. The value is unchanged.

### Fixed
- **With `setSleepIdleSeconds(> 0)`, the model became permanently unusable after the first idle
  period.** Once llama.cpp's task queue enters its sleeping state, posting a task does not leave it:
  `server_queue::post()` only notifies the condition variable, whose sleeping predicate tests
  `req_stop_sleeping`, so the loop woke, re-tested, and went straight back to sleep with the task
  still queued. Upstream performs the wake on the caller's behalf in `server_res_generator`'s
  constructor (`wait_until_no_sleep()`), but only for readers built through `create_response()`;
  this binding builds its readers with the CLI-facing `get_response_reader()`, which does not, and
  nothing in the JNI layer called `wait_until_no_sleep()` at all. Every subsequent call then either
  blocked until `close()` (completions, embeddings, rerank, infill) or threw `"No result"`
  (`getMetrics`, LoRA and slot operations), for the lifetime of the process. All six post sites now
  wake the queue first. Idle-sleep is off by default (`-1`), so a default configuration was never
  affected.
- **A single malformed UTF-8 byte in a model's output turned a finished generation into an HTTP 500.**
  The server parses *every* completion through `common_chat_parse()`; with no chat parser configured
  (plain `/completion`) that is llama.cpp's content-only fallback, whose scan tolerates an incomplete
  trailing UTF-8 sequence in lenient mode — which is the only mode the chat parser ever uses — but
  rejected an *invalid* byte outright. The request then failed with `"The model produced output that
  does not match the expected Content-only format"` even though generation had completed normally.
  Carried as local patch `0011`, which makes the invalid-byte branch respect leniency the same way
  (keeping the text up to the bad byte); strict-mode parsing is unchanged. Upstream-submittable.
- **`TextToSpeech` crashed the JVM on every platform when loading a model.** A hand-built
  `common_params` never passes through `common_params_parse`, and `common/arg.cpp` is upstream's
  only caller of `postprocess_cpu_params` — `common_init_from_params` does not call it. So
  `cpuparams_batch.n_threads` kept its `-1` default, `common_threadpools::init` created a second
  threadpool with -1 threads, and `ggml_threadpool_new` sized its worker array as
  `sizeof(ggml_compute_state) * -1` — a huge `size_t`, so the allocation returned `NULL` and the
  unchecked `memset` that follows it faulted at address 0. `tts_engine.cpp` and `train_engine.cpp` now mirror `arg.cpp`'s two
  calls; the `LlamaModel` paths were never affected because their params are parsed. Guarded by
  five model-free C++ tests over the extracted `build_tts_params`.
- **`LlamaQuantizer` never worked in any published jar — every call threw `UnsatisfiedLinkError`.**
  The `extern "C"` declarations that give the JNI entry points C linkage come from the
  javac-generated `jllama.h`, which covers **only** `LlamaModel`; a JNI function for any other class
  has to declare its own (as `train_engine.cpp` and `native_server.cpp` do).
  `Java_net_ladenthin_llama_LlamaQuantizer_quantizeNative` did not, so it was exported under its
  C++-mangled name and the JVM could never resolve it — on every platform, not just the two Windows
  jobs that reported it. The only coverage was `QuantizerIntegrationTest`, which gates on a GGUF and
  so skipped in CI for as long as the model paths resolved to the wrong directory. Fixed, and guarded
  model-free by `NativeLibraryLoadSmokeTest.quantizerNativeEntryPointResolves` so a future entry point
  that forgets `extern "C"` fails a test that runs wherever the library exists.
- **The macOS arm64 native library shipped corrupt in 5.0.6 and in several 5.0.7 snapshots.** All three
  macOS arm64 build jobs uploaded their dylib under a `*-libraries` artifact name, and the packaging
  job collects those with one globbed download — so three builds landed on the same
  `Mac/aarch64/libjllama.dylib` and the survivor could be a byte-level hybrid of two of them rather
  than either input. Its ad-hoc signature then no longer matched its own `__TEXT` pages (66/4078 and
  1141/4097 code pages failed their stored hashes) and macOS **SIGKILLed every process that loaded
  it**. Fixed by naming the test-only variants outside the glob and selecting the shipped variant by
  an explicit download step (thanks to **@linking12**, #388), plus two guards so it cannot recur:
  `merge-native-artifacts.sh` fails the build when any relative path is claimed by more than one
  artifact — checked *before* the merge, since a collision leaves exactly one file behind and is
  invisible afterwards — and the new `smoke-fatjar-macos` job runs `codesign --verify --strict` and a
  real JVM load of the dylib extracted from the **packaged** fat jar (#390).
- **`LlamaModel.getMetrics()` returned the wrong shape.** Upstream reduced the payload to a bare slot array
  at b10408 (#26920) and split the task in two at b10519 (#27376), so the counter getters on
  `value.ServerMetrics`, `LlamaModelTest#testGetMetrics` and `OpenAiCompatServer`'s metrics routes had all
  been reading keys that no longer existed. The JNI layer now posts both tasks and merges them, restoring the
  documented object rather than following upstream's transport split.
- **`GET /slots` answered HTTP 200 with a zero-length body** whenever the metrics payload carried no `slots`
  key (`MissingNode.toString()` is `""`). It now always answers with a JSON array.
- **Model-gated Java tests silently self-skipped in CI.** Surefire's working directory is the module basedir
  while the shared GGUF cache is restored to the reactor root, so every `models/…` path resolved to nothing,
  every such class aborted in its `@BeforeAll`, and the job still reported success — which is why the stale
  `getMetrics()` assertions above never failed. Test paths now resolve against either layout.
  `llama-langchain4j` had the identical defect.
- **`RouterClient.awaitModelLoaded` misdiagnosed hidden router models.** A cache model deduplicated by a
  preset with `dedup-cache-models` (b10505, #27346) is omitted from `GET /models` although it still loads and
  serves by name; the error now names that cause instead of sending callers to re-check `--models-dir`.
- **CVE-2026-49844** (GHSA-qv9r-c865-cp47, moderate): `org.apache.logging.log4j:log4j-api`
  2.25.3 arrives as a **test-scope** transitive of `io.github.hakky54:logcaptor` 2.12.6, and
  Dependabot could not update it on its own. Pinned `log4j-api` **and** `log4j-to-slf4j` to
  **2.26.1** in `dependencyManagement` — both together, since `log4j-to-slf4j` requires a
  matching `log4j-api` and the two must not skew. Neither reaches a published artifact.

## [5.0.6] - 2026-07-07

Feature release. Headline additions are the Android AAR + Kotlin coroutines
façade, the `NativeServer` attach and in-JVM router modes, GGUF tooling
(quantizer + inspector), and all-backends server fat jars as GitHub release
assets. Tracks llama.cpp **b9870 → b9894**.

### Added
- **Android AARs** (`net.ladenthin:llama-android`, `net.ladenthin:llama-android-opencl`): consumable Android artifacts carrying the core classes + CI-built `libjllama.so` natives — the CPU AAR is multi-ABI (`arm64-v8a` devices + `x86_64` emulators/Chromebooks), minSdk 28, with consumer R8/ProGuard rules. Built by a standalone Gradle build (version-locked to the Maven reactor); validated in CI by an AGP consumer smoke test (full R8 `assembleRelease`) and an on-emulator job running real native inference (release gate).
- **Kotlin coroutines façade** (`net.ladenthin:llama-kotlin`, new reactor module): `generateFlow`/`generateChatFlow` cold `Flow`s plus `completeSuspend`/`chatSuspend`/`chatCompleteTextSuspend`/`embedSuspend`, with coroutine cancellation wired into the cooperative `CancellationToken`.
- **`NativeServer` attach mode** (`NativeServer(LlamaModel, String...)`, patch `0007`): serve an **already-loaded** `LlamaModel` over the full upstream HTTP frontend — one copy of the weights, no second model load.
- **In-JVM router mode** (patch `0008` + `NativeServer.setWorkerCommand(...)`): `--models-dir` multi-model routing with per-request model selection, worker processes relaunched as fresh JVMs; typed `server.RouterClient` + `value.RouterModel` API for the model-management endpoints.
- **GGUF tooling**: `LlamaQuantizer` (native GGUF quantization) and `GgufInspector` (metadata reader; works on Android).
- **Session fork/rewind**, **runtime LoRA control**, and **batch embeddings** on the core API.
- **LangChain4j**: blocking tool calling (`ToolSpecification` round-trip), JSON mode (`json_object` + `json_schema` structured output), multimodal user input (`ImageContent`/`AudioContent`), and full streaming via `StreamingChunkAssembler` — streamed tool calls, per-token thinking events, real finish reason and token usage.
- **All-backends server fat jars** attached to GitHub releases (never Maven Central): `llama-<version>-all-<os>-<arch>-jar-with-dependencies.jar` for Linux/Windows x86-64 + aarch64, each bundling every GPU backend's natives with a priority manifest. `LlamaLoader` tries each backend in order and falls back to CPU; the `net.ladenthin.llama.backend` system property forces one. Smoke-tested via real `java -jar` runs on Linux + Windows.
- Committed audio prompt fixture (`src/test/resources/audios/sample.wav`) for `AudioInputIntegrationTest`.

### Fixed
- **Android `System.loadLibrary("jllama")` failure on every device**: the cross-clang emitted `DT_NEEDED` on `libomp.so` and `libc++_shared.so`, which don't exist on stock Android — fixed by disabling OpenMP and linking `-static-libstdc++` (the released 5.0.5 arm64 lib carried this latent defect). A per-`.so` `DT_NEEDED` whitelist and the 16 KB page-size alignment are now CI-enforced.
- **UTF-8-safe JNI strings**: payload text no longer goes through `NewStringUTF` (which expects *Modified* UTF-8), so supplementary-plane characters (emoji) are preserved and Android CheckJNI no longer aborts.
- Stale Windows docs claiming three co-located DLLs corrected (a single monolithic `jllama.dll` ships per arch); leftover extracted `ggml-metal.metal` cleanup.

### Changed
- Upgraded llama.cpp from **b9870 to b9894** (all local patches refreshed across the range).
- CI model downloads single-sourced from `.github/models.csv`: one download job is the only cache writer, the cache entry is cross-OS, and a 3-OS verification gate proves it restorable and complete before any model-consuming job starts.

## [5.0.5] - 2026-07-04

Feature release. Headline addition is `NativeServer` — the full upstream
llama.cpp server (embedded WebUI included) running in-process over JNI — plus
a large native-artifact matrix expansion (Linux Vulkan, Windows arm64, eight
ROCm/SYCL/OpenVINO/OpenCL classifiers, Linux s390x). Tracks llama.cpp
**b9859 → b9870**.

### Added
- **`server.NativeServer`**: runs the full upstream `llama_server` — WebUI and all — inside `libjllama` via JNI (patch `0006`), forwarding the raw llama-server argv verbatim, so every llama-server flag works with no separate `llama-server` executable. The fat jar's `Main-Class` is now `server.ServerLauncher`: `NativeServer` by default, `--jllama-openai-compat` selects the Java-transport `OpenAiCompatServer`.
- **Linux Vulkan classifiers** (`vulkan-linux-x86-64`, `vulkan-linux-aarch64`): vendor-neutral GPU jars for NVIDIA/AMD/Intel without a CUDA toolkit.
- **Windows arm64 CPU natives** in the default JAR (built natively on `windows-11-arm` with clang-cl; self-contained `/MT` CRT, OpenMP off).
- **Eight further GPU-backend classifiers**: `rocm-linux-x86-64`, `rocm-windows-x86-64`, `sycl-fp16-linux-x86-64`, `sycl-fp32-linux-x86-64`, `sycl-windows-x86-64`, `opencl-windows-aarch64`, `openvino-linux-x86-64`, `openvino-windows-x86-64`.
- **Linux s390x (big-endian) natives** in the default JAR, cross-compiled and gated by the full C++ unit suite under `qemu-user` (real big-endian correctness check for the byte-order-sensitive surface).
- `sse_ping_interval` and further audited completion parameters on `InferenceParameters`; model ftype/quantization surfaced through the Java API and `/v1/models`; additional `OpenAiServerCli` flags (`-b`/`-ub`/`-tb`/`-ctk`/`-ctv`/`--jinja`/`--chat-template-kwargs`).
- llama.cpp version-bump automation: `.github/scripts/llama-next-version.sh` + the runbook `docs/upgrade/llama-cpp-version-bump.md`.

### Fixed
- **Multi-turn tool-calling checkpoint starvation** for recurrent/hybrid models (e.g. Granite-4), patch `0005`: agentic conversations no longer re-prefill the whole conversation tail every turn — prefill is constant per turn (≈5.4× less prefill by turn 6, growing with conversation length), validated output-identical.

### Changed
- Upgraded llama.cpp from **b9859 to b9870**.
- CI: per-job sccache statistics table appended to GitHub job summaries.
- Bumped checker-qual 4.2.0 → 4.2.1 and spotless-maven-plugin 3.7.0 → 3.8.0.

## [5.0.4] - 2026-07-02

Feature release. Adds in-process LangChain4j adapters, an experimental
fine-tuning API, and richer model introspection, and restructures the build
into a Maven reactor (published coordinates unchanged). Tracks llama.cpp
**b9842 → b9859**.

### Added
- **LangChain4j integration** (`llama-langchain4j` module): in-process adapters for LangChain4j's `ChatModel`, `StreamingChatModel`, `EmbeddingModel`, and `ScoringModel` over JNI (no HTTP hop). Shipped as a separate artifact `net.ladenthin:llama-langchain4j` (Java 17), versioned and released in lockstep with the core so a Java-8 `net.ladenthin:llama` consumer is unaffected.
- **In-process fine-tuning** (`LlamaTrainer`): an experimental training API with configurable `TrainingParameters` and `Optimizer` (`args.Optimizer`) driving llama.cpp's optimizer through the JNI binding.
- **Model introspection via `ModelMeta`** (`value.ModelMeta`): exposes the model's chat template, special tokens, and full key/value metadata.

### Changed
- Restructured the build into a **Maven reactor**: the native JNI core moved into the `llama/` module under a new aggregator parent POM (`net.ladenthin:llama-parent`, `packaging=pom`), alongside the `llama-langchain4j` module. Both modules inherit a single version, so all artifacts ship in lockstep. Published coordinates (`net.ladenthin:llama`) are **unchanged** — no consumer action required.
- Upgraded llama.cpp from **b9842 to b9859**. All four local patches (`0001`–`0004`) apply unchanged across the range.
- CI: the GGUF model set is now downloaded once upfront by a dedicated job and restored (not re-fetched) by every test job, de-duplicating the pipeline.
- Bumped `palantir-java-format` 2.92.0 → 2.94.0.

## [5.0.3] - 2026-06-29

Feature release. Headline addition is a full OpenAI-compatible embedded HTTP
server with multi-protocol surfaces, plus end-to-end multimodal (vision, audio
input, text-to-speech) and slot-bound sessions. Tracks llama.cpp **b9555 → b9842**.

### Added
- **OpenAI-compatible HTTP server** (`server` package, built on the JDK's `com.sun.net.httpserver` — no new runtime dependency; embeddable and the fat-jar `Main-Class`). Serves `POST /v1/chat/completions` (streaming SSE + non-streaming), `/v1/completions` (token-by-token streaming), `/v1/embeddings`, `/v1/rerank`, `/infill`, `GET /v1/models`, `GET /health`, and `GET /props` (every route also reachable without the `/v1` prefix), with optional bearer auth and CORS — drives editor clients such as VS Code Copilot, Cline, Roo Code, and Continue.
- **Multi-protocol surfaces** over the same inference core (pure translation, no second inference path): **Ollama-native** (`/api/version`, `/api/tags`, `/api/show`, `/api/chat` NDJSON, `/api/generate`), **Anthropic Messages** (`POST /v1/messages`, SSE), and **OpenAI Responses** (`POST /v1/responses`, SSE).
- **Agentic tool-calling**: `parallel_tool_calls` support (`ChatRequest.withParallelToolCalls(Boolean)`, `InferenceParameters.withParallelToolCalls(boolean)`, server-mapper pass-through), the `ToolCallingAgent` chat loop (JSON-serialized tool-result errors), and `ToolCallDeltaAccumulator` for reconstructing streamed tool calls; real-model integration tests (`ToolCallingIntegrationTest`, Qwen2.5-1.5B-Instruct).
- **Text-to-speech** (`TextToSpeech`): OuteTTS (text-to-codes) + WavTokenizer (codes-to-speech) pipeline; `synthesize(text)` returns a 24 kHz mono 16-bit WAV byte stream. The OuteTTS DSP is derived at build time from upstream `tts.cpp` rather than hand-copied.
- **Audio input** via OpenAI `input_audio` content parts (`ContentPart.audioFile`), for Ultravox / Qwen2.5-Omni-class models.
- **End-to-end vision input** across blocking, typed `ChatRequest`, streaming, and OpenAI-compatible request mapping; real-model tests verify distinct red/blue images produce the correct semantic answers. Explicit `setMmprojAuto(boolean)` / `setMmprojOffload(boolean)` controls (`--no-mmproj-auto` / `--no-mmproj-offload`).
- Per-request KV controls: `InferenceParameters.withSlotId(int)` and `withCacheReuse(int)`.
- Per-request DRY sampling on `InferenceParameters` (`dry_multiplier` / `dry_base` / `dry_allowed_length` / `dry_penalty_last_n` / `dry_sequence_breakers`).
- `ModelParameters.enableSwaFull()` (`--swa-full`): keep a full-size SWA KV cache to enable cross-request prompt-prefix reuse.
- Typed cache observability: `Usage.getCachedTokens()`, `Usage.getProcessedPromptTokens()`, `SlotMetrics`, `ServerMetrics.getSlotMetrics()`, plus authenticated JSON `GET /metrics` and `GET /slots`.
- **Windows GPU native classifiers**: `cuda13-windows-x86-64`, `vulkan-windows-x86-64`, `opencl-windows-x86-64`, and the `msvc-windows` CPU classifier (the default Windows CPU JAR flipped to the Ninja Multi-Config generator).
- `log_helpers.hpp` — pure, unit-tested log-formatting helpers (`log_level_name`, `format_log_as_json`).

### Changed
- Upgraded llama.cpp from **b9555 to b9842** across eleven incremental upgrades. Notable upstream features now reachable: DRY sampling, `--swa-full`, DFlash block-diffusion speculative decoding (`--spec-type draft-dflash`), the MiniCPM5 XML tool-call chat template, the server `--reasoning-preserve` flag, Jinja `min`/`max` array filters, and the **DeepSeek-V4** architecture (b9840). The b9829 bump additionally compiles the new upstream `server-stream.cpp` (resumable-streaming SSE replay buffer) into `libjllama`. The final b9840→b9842 step is internal-only (preset INI section-tag canonicalization in `common/preset.cpp`; a Vulkan graph-submission heuristic switched from weight-matrix bytes to estimated FLOPs) — no project source changes, no API impact, all four local patches (`0001`–`0004`) apply unchanged across the range.
- Replaced the `--skip-download` flag with `--offline` (llama.cpp b9803).
- `Session` now pins every inference request to its configured slot, so generation and slot save/restore/erase target the same KV state (`SessionState` extracted as a testable concurrency contract).
- `configureParallelInference` now applies `slot_prompt_similarity` live via `server_context::set_slot_prompt_similarity()` (upstream PR ggml-org/llama.cpp#22393, carried as `patches/0003`), instead of validating and discarding the value.
- **Android minimum API level raised from 24 to 28** (Android 9.0 Pie), satisfied via bionic's weak-symbol mechanism rather than `__ANDROID_API__`.
- CI: rolled out the sccache → Depot shared compiler cache across all native build jobs (incl. nvcc wrapping for full-arch CUDA and the Windows Ninja path), fork-PR token-gating, and a shared GGUF model cache.
- `LlamaLoader` native-library extraction is now race-safe (atomic write) and uses a private lock object instead of `synchronized` methods.
- SpotBugs (effort=Max, threshold=Low) made clean and wired into CI; C++ unit suite grown to 459 tests.

### Fixed
- Per-request `reasoning_budget_tokens` is now honored (via `patches/0004`, upstream PR ggml-org/llama.cpp#23116): `reasoning_budget_tokens=0` suppresses thinking.
- Preserved decoded image buffers across the JNI chat boundary and submitted media requests through llama.cpp's multimodal task path instead of silently tokenizing them as text-only prompts; preserved multipart image content in the typed `ChatRequest` serializer.
- The standalone OpenAI-compatible server now advertises vision only when the loaded model confirms usable vision support.
- Cached-token usage is preserved through typed Java responses and the OpenAI Responses / Anthropic blocking and streaming adapters.
- Stabilized flaky reasoning-budget tests on Metal by using greedy sampling.

## [5.0.2] - 2026-06-08

Tracks llama.cpp **b9151 → b9555**.

### Added
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.0).
- `docs/RELEASE.md` capturing the maintainer-facing release procedure (moved out of CHANGELOG).
- OpenSSF Best Practices badge (project 12862) on README.
- Reasoning-budget tests (Qwen3-0.6B).

### Changed
- **Reorganized the Java API into subpackages** — `parameters` (`ModelParameters`, `InferenceParameters`, …), `value` (`LogLevel`, …), `callback`, `exception` (`LlamaException`, …), and `loader` (`LlamaLoader`, `OSInfo`). Source-incompatible for consumers: import statements for the moved types must be updated.
- Unified `CONTRIBUTING.md` and `SECURITY.md` structure with sibling repositories, and migrated cross-repo `CLAUDE.md` sections to `workspace` pointers.
- Reconciled Java baseline to **11+** across `pom.xml`, README badge, `CLAUDE.md`, and `CONTRIBUTING.md`.
- README license badge corrected from "Apache 2.0" to "MIT" (matches `LICENSE` file and `pom.xml`).
- `pom.xml` SCM URL: `tree/master` → `tree/main` (default branch renamed).
- Upgraded Maven dependencies (incl. `logback-classic` 1.5.32 → 1.5.33).
- Upgraded llama.cpp from **b9151 to b9555** across multiple incremental upgrades.

## [5.0.1] - 2026-05-14

### Added
- `InferenceParameters.setContinueFinalMessage(boolean)` for the vLLM/transformers-compatible prefill-assistant heuristic (llama.cpp b9134+).
- Tests for `setContinueFinalMessage`.
- Comprehensive Javadoc on public APIs (PR #129).
- Maven Central badge on README (PR #130).

### Changed
- Bumped project version to 5.0.1-SNAPSHOT (PR #127), then released as 5.0.1 (PR #135).
- Refactored GitHub release workflow to parallelise snapshot and release jobs (PR #128).
- Removed snapshot build documentation and badge (PR #131).
- Upgraded Windows CI to `windows-2025` with Visual Studio 2026 (PR #132).
- Switched Windows MSVC runtime from dynamic (`/MD`) to static (`/MT`) to eliminate the `msvcp140.dll` runtime dependency (PR #133).
- Upgraded llama.cpp from b9106 to b9134 (PR #134), then to b9150 (PR #136), then to b9151 (PR #139).
- Refactored CI workflow with explicit snapshot/tag check gates (PR #137).
- Removed `setCtxSizeDraft()` — the underlying CLI flag was deleted upstream in llama.cpp b9106.

### Fixed
- `fix(publish):` quoted gate job names to avoid YAML colon-in-scalar parse errors (PR #138).
- Release routing in the publish workflow now correctly distinguishes snapshot vs. tag pushes.

## [5.0.0] - 2026-05-11

First release of the fork under the `net.ladenthin:llama` Maven coordinates. ~100 merged pull requests since baseline `49be664` (the last pre-fork upstream commit).

### Added
- First publish to Maven Central under `net.ladenthin:llama`.
- Pre-built native libraries for Linux (x86-64, aarch64), macOS (x86-64, arm64), and Windows (x86-64, x86).
- Java API surface: `LlamaModel`, `ModelParameters`, `InferenceParameters`, `LlamaIterator` / `LlamaIterable` for streaming, chat completion (`chatComplete`, `generateChat`, `chatCompleteText`), embeddings, reranking, infilling, raw JSON endpoint handlers, slot management (`saveSlot`, `restoreSlot`, `eraseSlot`), and `getModelMeta()`.
- `chatComplete()` for OpenAI-compatible chat completions, re-implemented from scratch based on a patch by @vaiju1981 (PR #61; see `docs/history/CHAT_INTEGRATION_SUMMARY.md`).
- `mmproj`, reasoning-budget, sigma, and sleep-idle parameters added to `ModelParameters`.
- JaCoCo code-coverage reporting integrated with Coveralls and Codecov (PR #124).
- CodeQL static-analysis workflow on push, PR, and a weekly schedule.
- Automated Claude Code review workflow on pull requests.
- Dependabot for Maven and GitHub Actions dependency updates.
- Automatic snapshot release workflow on `main` push (PR #105) publishing to the Sonatype Central snapshot repository.
- CUDA, Metal, and Vulkan build support via local CMake build.
- Android integration documented in README.
- All system properties (`net.ladenthin.llama.*`) and `LogLevel` values documented.
- `CLAUDE.md` maintainer guide covering upstream upgrade procedure and the b5022→b9172 breaking-change table.

### Changed
- Migrated Maven group and artifact from `de.kherud:java-llama.cpp` to `net.ladenthin:llama` (PR #101).
- Migrated Maven Central publishing from OSSRH (Legacy) to the Sonatype Central Publisher Portal.
- Deleted the hand-ported `server.hpp` fork (~3,780 lines) and linked the upstream `llama.cpp` server source files directly into `jllama`. ~4,100 C++ lines removed in total; future upstream upgrades become a CMake version bump. **The Java API is unchanged.** See `docs/history/REFACTORING.md`.
- Compiled upstream server-context / queue / task / models directly into jllama (PR #96).
- Unified CI into a single `publish.yml` workflow with cross-compilation, testing, coverage, and release stages.
- Upgraded CUDA from 12.1 to 13.2 (PR #50).
- Upgraded llama.cpp from b8913 through b9106 across multiple incremental upgrades.
- `setDraftMax` / `setDraftMin` now emit the canonical `--spec-draft-n-max` / `--spec-draft-n-min` flags (llama.cpp b9016 removed the old aliases).
- Bumped CI GitHub Actions: `actions/checkout` v4 → v6, `actions/upload-artifact` v6 → v7, `actions/download-artifact` v6 → v8, `codeql-action` v3 → v4.

### Fixed
- Javadoc warnings resolved across the public API by adding missing comments.
- `cache_idle_slots` slot-parameter handling aligned with the upstream rename (b8841 → b8854).

## Pre-fork history (kherud/java-llama.cpp 1.x–4.2.0)

Releases `1.1.1` through `4.2.0` were authored by [@kherud](https://github.com/kherud) on the upstream repository. The full upstream release notes are at
<https://github.com/kherud/java-llama.cpp/releases>. The fork's baseline is upstream commit `49be664` (tagged `v4.2.0`, 2025-06-20).

For an architecture-level diff between the pre-fork baseline (`49be664`) and the first 5.0.0 candidate (`24918e4`), see [`docs/history/49be664_24918e4.md`](docs/history/49be664_24918e4.md). For the server-fork-deletion refactor that culminated in 5.0.0, see [`docs/history/REFACTORING.md`](docs/history/REFACTORING.md). For the chat-completion integration that landed in 5.0.0, see [`docs/history/CHAT_INTEGRATION_SUMMARY.md`](docs/history/CHAT_INTEGRATION_SUMMARY.md).

[Unreleased]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.1.0...HEAD
[5.1.0]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.6...v5.1.0
[5.0.6]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.5...v5.0.6
[5.0.5]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.4...v5.0.5
[5.0.4]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.3...v5.0.4
[5.0.3]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.2...v5.0.3
[5.0.2]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.1...v5.0.2
[5.0.1]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/bernardladenthin/java-llama.cpp/releases/tag/v5.0.0
