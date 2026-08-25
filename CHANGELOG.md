# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from version 5.0.0 onward. Pre-fork releases (`1.x`–`4.2.0`) were authored by
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp).

## [Unreleased]

### Added
- `QuantizationType.Q2_0` — maps the new upstream `LLAMA_FTYPE_MOSTLY_Q2_0` (llama.cpp b9916) for `LlamaQuantizer`.

### Changed
- `ch.qos.logback:logback-classic` bumped 1.6.2 → 1.6.3 (test/runtime binding only).
- CI actions bumped to latest: `actions/setup-java` v5 → v6.
- Upgraded llama.cpp from **b9894 to b9917** (all eight local patches re-verified across the range).

### Fixed
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

[Unreleased]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.6...HEAD
[5.0.6]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.5...v5.0.6
[5.0.5]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.4...v5.0.5
[5.0.4]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.3...v5.0.4
[5.0.3]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.2...v5.0.3
[5.0.2]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.1...v5.0.2
[5.0.1]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/bernardladenthin/java-llama.cpp/releases/tag/v5.0.0
