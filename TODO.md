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

Six of the seven `patches/` are upstream-submittable verbatim; each accepted PR (once the pin is
bumped past it) deletes a patch from the bump checklist. (`0003` is a carry of an already-open
upstream PR #22393 — it drops automatically when that merges.)

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
- **`0009` guard `posix_spawn_file_actions_addchdir_np` on old glibc** (b10154 cross-compile break on
  manylinux2014 / glibc 2.17 and manylinux_2_28 / glibc 2.28; adds an overridable
  `SUBPROCESS_HAVE_CWD` probe via `__GLIBC_PREREQ(2, 29)` — submitted as sheredom/subprocess.h#104,
  drops automatically once llama.cpp bumps the vendored pin).

### llama.cpp upstream feature exposure (queued, deferred by policy)

These are JNI plumbing items for upstream API additions. Policy: add only after a real user request — they are mostly relevant to specific model families or specialized workflows.

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

- **Expose `--spec-draft-backend-sampling` toggle via `ModelParameters.setSpecDraftBackendSampling(boolean)`.** Added in b9437 (env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING`). Backend sampling for the speculative draft is enabled by default upstream but auto-disabled on `LLAMA_SPLIT_MODE_TENSOR` setups; an explicit Java-side setter lets callers force-disable it for benchmarking or for backends with sampler bugs. Speculative-decoding power users.

- **Expose runtime reasoning control via `InferenceParameters.setReasoningControl(boolean)` + `LlamaModel.endReasoning(...)`.** Added in b9444–b9490: new `common_params_sampling::reasoning_control` flag arms the budget sampler so reasoning can be ended at runtime, and new `common_sampler_reasoning_budget_force(common_sampler *)` triggers the end-of-thinking token injection on the next sample. Upstream also adds a `POST /v1/chat/completions/control` server endpoint accepting `{"id": "...", "action": "reasoning_end"}`. Java mapping would be: (a) `InferenceParameters.setReasoningControl(boolean)` arms the sampler on the inference run, (b) a new `LlamaModel.endReasoning(int slotId)` (or per-streaming-task-id) JNI method calls the upstream `common_sampler_reasoning_budget_force` against the slot's sampler. Useful for interactive UIs that want a "skip thinking and answer now" button. Relevant only for reasoning-trained models (DeepSeek-R1, Qwen3-Thinking, GPT-OSS-Reasoner, etc.).

- **Expose `llama_context_params::n_outputs_max` via `ModelParameters.setMaxOutputs(int)`.** Added in b9444–b9490 (default `-1` = derived from `n_batch`). Caps the number of output slots allocated per context; relevant for memory-constrained setups that always run with `logits_all=false` and want to prevent over-allocation when `n_batch` is large. Trivial JNI plumbing (one `cparams` field passthrough); add when a user reports OOM on context creation tied to output slot pre-allocation.

- **Expose Multi-Token Prediction toggle via `ModelParameters.setMtp(boolean)`.** Existed since the Qwen3.5 MTP work; b9444–b9490 extends it to Step-3.5. CLI flags `--mtp`/`--no-mtp` (env `LLAMA_ARG_MTP`) control whether the draft head runs alongside the main model for accelerated decoding. Java setter would route to `common_params_speculative::type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP`. Relevant only for MTP-trained models.

- **Expose `llama_vocab::get_suppress_tokens()` via `LlamaModel.getSuppressTokens()`.** Added in b9490–b9495 alongside the new `tokenizer.ggml.suppress_tokens` GGUF key and the `LLM_KV_TOKENIZER_SUPPRESS_TOKENS` constant. When a GGUF declares this array, upstream stores it on `llama_vocab::impl::suppress_tokens` and exposes it via the new `llama_vocab::get_suppress_tokens()` accessor. The bias is **applied automatically** inside the model forward graph — the Gemma4 Unified graph (`src/models/gemma4.cpp`) reads the list and adds a `-INFINITY` logit bias to those token IDs via a new `llm_graph_input_logits_bias` input so the model cannot emit them (used to block `<image|>` / `<audio|>` placeholders). A Java mirror would be `public int[] getSuppressTokens()` on `LlamaModel`: a read-only inspector returning the suppression list for debugging or for callers running their own sampling who want to replicate the same bias. Value is low (the bias is auto-applied, Java callers cannot change it; java-llama.cpp does not expose custom logit-bias hooks at this level); cost is trivial (one JNI passthrough + a `getSuppressTokens()` Java method).

### JNI safety and server hardening (from PR #251 contributor)

Raised by [@vaiju1981](https://github.com/vaiju1981) in
[PR #251 comment](https://github.com/bernardladenthin/java-llama.cpp/pull/251#issuecomment-4761363838).
Feel free to contribute fixes — PRs welcome.

- **Unhandled C++ exceptions cross the JNI boundary → JVM abort (UB).** Any `std::exception`
  (or worse, an exception of unknown type) that escapes a native method and crosses the JNI
  boundary causes undefined behaviour on most JVMs and typically aborts the process. Each native
  method in `jllama.cpp` should wrap its body in `try { … } catch (const std::exception& e) {
  env->ThrowNew(llamaExceptionClass, e.what()); return <zero>; } catch (...) { env->ThrowNew(…,
  "unknown C++ exception"); return <zero>; }` so that errors surface as `LlamaException` on the
  Java side instead of crashing the JVM.

- **`parse_string_array` — null deref + JNI local-reference leak.** The helper that reads a
  JSON string array from JNI can dereference a null pointer when an array element is absent,
  and leaks JNI local references when an early exit skips the matching `DeleteLocalRef`. Fix:
  guard every `GetObjectArrayElement` result and pair each reference acquisition with a
  `DeleteLocalRef` before the next iteration or return.

- **`close()` / native `delete()` double-free under concurrent close.** If two threads race to
  call `LlamaModel.close()`, both can reach the native `delete` path and free the same
  `jllama_context` pointer twice → heap corruption. Fix: use `AtomicBoolean closed` + a
  `synchronized` guard (or `compareAndSet`) on the Java side so `close()` is idempotent and
  the native pointer is nulled before the second caller can reach it.

- **Unbounded request-body read → OOM DoS.** The HTTP handler reads the entire request body
  into a `String`/`byte[]` before parsing it, with no size cap. A client that streams a
  multi-gigabyte body can exhaust heap memory and crash the JVM. Fix: add a configurable
  `maxRequestBodyBytes` limit (e.g. default 4 MB) and reject oversized requests with
  `HTTP 413 Content Too Large` before buffering them.

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
  - **CUDA / Metal / OpenCL backend selection.** Today the choice between CPU-only / `cuda13-linux-x86-64` / `opencl-android-aarch64` JARs is at Maven-classifier time. Native Image would need either one binary per backend (multiplying the release matrix) or a runtime selector inside `LlamaLoader` that picks among bundled backend libs. The latter is a bigger refactor.
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
`package-fatjars` or a `smoke-test-fatjar.sh` flag making the backend-manifest grep optional for the
manifest-less default jar. Worth doing only if a macOS-specific *inference* regression ever appears;
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
`NativeLibraryLoadSmokeTest`, and `IdleSleepWakeIntegrationTest` for the `wake_and_post` path);
these are what remains.

- **`patches/0010` has no guard that runs on a model-free host.** Reverting the patch's
  `(int)` cast in the fetched `tools/server/server-context.cpp` leaves `ctest` at a clean **520/520** —
  the always-run `C++ Tests` job cannot see the regression at all. The only guard is
  `NativeServerAttachIntegrationTest.models_reportNumericVocabType`, which is model-gated; it *does*
  run on all six CI Java jobs (the full model set is downloaded there), so this is a coverage gap
  rather than a shipping risk today. It becomes one the moment a platform stops downloading models.
  `CommonJsonEnumTrap` in `test_json_helpers.cpp` cannot help — it builds its own JSON literals and
  calls no project code. A direct unit test is impossible as things stand: `get_res_model_info` is
  `static` inside `server-context.cpp` and unreachable from `jllama_test`. Cheapest real fix is a
  CI assertion in the `C++ Tests` job that the patch is present in the fetched tree
  (`grep -c '(int) meta.model_vocab_type'` plus a non-empty `git -C _deps/llama.cpp-src diff`).

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

### Release/build robustness gaps found by the b10679 audit (PR #403)

Both are **pre-existing** and orthogonal to a version bump, so they were recorded rather than folded
into that PR.

- **Two `all-*-aarch64` fat jars are attached to releases with no smoke job.**
  `.github/package-fatjars.sh` emits four OS/arch fat jars (`linux-x86-64`, `linux-aarch64`,
  `windows-x86-64`, `windows-aarch64`), all uploaded as `llama-fatjars` and attached by
  `github-release-signed` / `github-snapshot`. Only the two **x86-64** ones are smoked
  (`smoke-fatjar-linux`, `smoke-fatjar-windows`); grepping `publish.yml` for `all-linux-aarch64` or
  `all-windows-aarch64` returns nothing, so neither is ever downloaded or launched.

  That directly violates the cross-repo rule in
  [`../workspace/policies/fat-jar-release-assets.md`](../workspace/policies/fat-jar-release-assets.md)
  — *"No release asset is attached that CI has not run"* — which exists because a corrupt macOS dylib
  shipped in three releases under a fully green pipeline. The fix is cheap: the workflow **already**
  uses the free ARM runners elsewhere (`ubuntu-24.04-arm` for the aarch64 CPU and Vulkan builds,
  `windows-11-arm` for the Windows arm64 build), so `smoke-fatjar-linux-aarch64` and
  `smoke-fatjar-windows-arm64` can mirror the existing smoke jobs and join both publish jobs'
  `needs:`. Not done in the bump PR because it widens a version bump into CI work and would gate that
  PR on a pre-existing defect if either jar turns out to be broken.

- **The patch applier silently accepts a partially-reverted source tree.** The stamp file records the
  checked-out llama.cpp commit plus each patch's SHA-256 — **nothing about the resulting file
  contents**. Reverting one patched file after a successful apply leaves the stamp valid and the tree
  still dirty (the other patched files are still modified), so the dirty-tree branch reports
  "already applied — skipping", exits 0, and the build compiles unpatched code. Reproduction:

  ```bash
  # with the tree fully patched and the stamp written:
  git -C <llama.cpp-src> checkout -- common/peg-parser.cpp     # drops patch 0011's fix
  cmake -DPATCH_DIR=... -DLLAMA_SRC=... -P llama/cmake/apply-llama-patches.cmake
  # -> "8 patch(es) already applied — skipping", exit 0, patch NOT restored
  ```

  Every other path is correctly fail-loud (committed-patch state, stamp/HEAD mismatch on a dirty tree,
  and a non-git-worktree re-run all exit 1). The fix is a content oracle in the manifest — cheapest is
  to append `git -C <src> diff --no-color | sha256`, or per-patched-file blob hashes — so a reverted or
  hand-edited file invalidates the stamp. **CI is unaffected** (every job configures into a fresh build
  directory); this only bites a local reconfigure, which is why it was not rushed. Note the stamp
  format change will make every existing local build dir abort with the applier's
  "configure into a fresh build directory" message — that is the designed fail-loud path, not a
  regression.

### Test-coverage debt found during the b10649 review (PR #403)

Each item below was verified against pristine upstream tags and is real, but none is a regression
introduced by the version bump — they were deferred to keep that PR landable.

- **`ModelParameters` emits five CLI flags the server arg parser rejects, so any caller of them
  cannot load a model.** `--dump-kv-cache`, `--hf-repo-v` and `--hf-file-v` no longer exist anywhere
  in llama.cpp (absent at both b10456 and b10649); `--grp-attn-n` and `--grp-attn-w` still exist but
  are `set_examples({LLAMA_EXAMPLE_COMPLETION, ...})`, so `add_opt` never registers them for
  `LLAMA_EXAMPLE_SERVER` — the example jllama parses with. An unregistered flag is not ignored:
  `arg.cpp` throws, `common_params_parse` returns false, and `load_model_impl` throws
  `LlamaException("Failed to parse model parameters")`. Four existing tests pin the dead literals and
  would pass forever. Fix: deprecate the five members the way this PR handled
  `withTfsZ`/`withPenalizeNl` (keep source compatibility, never write the map), and add a hermetic
  `jllama_test` contract test that walks `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER)`'s
  `ctx.options` (upstream's own `test-arg-parser` pattern; the symbols already link into
  `jllama_test`) and asserts every flag `ModelParameters`/`ModelFlag` can emit is in that set,
  excluding only `--vocab-only`, which `strip_flag_from_argv` removes on purpose. A grep-based sweep
  is **not** sufficient — it is structurally blind to example scoping, which is exactly how
  `--grp-attn-w` hides.

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

- **`LlamaTrainer`'s end-to-end path runs on no CI platform.** `LlamaTrainerIntegrationTest`
  self-skips everywhere: `net.ladenthin.llama.train.model` is set by no job and its model is in no
  `.github/models.csv` row, so `validate-models.{sh,bat}` does not treat it as required. The C++ half
  is now mitigated (`test_tts_params.cpp`'s `TrainParams` + `ResolveCpuParams` suites), but nothing
  exercises the Java → JNI → native trainer round trip. Adding a small training model to `models.csv`
  plus the matching property to the Java test jobs would close it.

- **`LlamaLoader`'s jar-extraction internals need synthetic jar fixtures.** `readBackendManifest`,
  `tryLoadBackend`, `extractFile`, `moveIntoPlace`, `cleanPath` and `hasNativeLib` are named in no
  test; `BackendManifestLoadTest` and `LlamaLoaderTest` drive the class only from outside via system
  properties. Covering the multi-backend fat-jar path (per-backend temp subdir extraction,
  manifest-extras-first ordering, `UnsatisfiedLinkError` fallback to the next backend and then to the
  default CPU natives) means building jars carrying a `jllama-backends.txt` and dummy payloads.

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
