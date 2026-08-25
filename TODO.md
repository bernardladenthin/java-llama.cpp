<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# TODO — java-llama.cpp

Open work items for this repo. Cross-cutting tracking lives in
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md);
items here are jllama-specific or are this repo's slice of a
cross-cutting initiative.

## Open — jllama-specific

### LlamaLoader extraction-directory isolation (optional follow-up, low priority)

Left over from the 2026-06-20 code audit (18/18 findings fixed in PRs #258/#260, regression tests in
#261/#262 — see the Done section): full per-process extraction **directory** isolation + a `cleanup()`
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
  OuteTTS behind an `/v1/audio/speech`-style route; proxying image/audio generation to an external
  model. Keep chunk formatters modality-neutral.
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

- **`ServerMetrics.getCumulativeTimings()` truncation — STALE / REFUTED.** This claim was
  verified false: `getCumulativeTimings` reads `n_prompt_tokens_processed_total` /
  `n_tokens_predicted_total` via `asLong(0L)` into `long` fields and constructs a `Timings`
  with `long promptN` / `predictedN` (`ServerMetrics.java` / `Timings.java`) — no int cast
  occurs. The real token-count truncation (now fixed) lived in the protocol shims
  (`ResponsesStreamTranslator`, `AnthropicStreamTranslator`, `ResponsesApiSupport`,
  `AnthropicApiSupport`, `OllamaApiSupport`, `ChatResponseParser.extractUsageField`), which used
  `.asInt(0)`; those now use `.asLong(0L)` so sessions above ~2.1B tokens no longer overflow.

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
session fork/rewind (see the Done section). **Remaining:**

- **jbang single-file example** (XS-S): a `//DEPS net.ladenthin:llama` one-file runnable demo so new
  users can try the binding without a Maven project.
- Further per-repo unique findings in the doc can be pulled on demand; none is currently prioritized.

### Android example app (own session; the remaining Android item)

The AAR + Kotlin façade + multi-ABI (arm64-v8a/x86_64) + emulator CI shipped, and the emulator job is
a release gate (see the Done section / CLAUDE.md "Android AAR + Kotlin façade"). Remaining: a minimal
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

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** JSpecify + NullAway are now enforced at compile time in **strict JSpecify mode** with the extra options `CheckOptionalEmptiness`, `AcknowledgeRestrictiveAnnotations`, `AcknowledgeAndroidRecent`, `AssertsEnabled` (see `pom.xml`); `@NullMarked` on the three packages via `package-info.java`; JDK module exports in `.mvn/jvm.config`. The legacy `org.jetbrains.annotations` dep has been removed; all nullability annotations are JSpecify. Public-API methods that may legitimately have no value use `Optional<T>` rather than `@Nullable T` (`ChatResponse.getFirstMessage`, `ChatMessage.getParts`, `ChatRequest.buildToolsJson`). Open follow-up: review remaining unannotated public API surfaces for places where `@Nullable` would be more precise than the implicit non-null default.

- **SpotBugs `effort=Max` + `threshold=Low`** — **DONE (already enabled in `pom.xml`)**, with fb-contrib +
  findsecbugs, bound to `verify`. The legacy "flip the pom / ~65 findings" note is stale: only a handful
  of unexcluded findings remain at any time, and `spotbugs:check` is kept green. Most recent pass fixed the
  6 introduced by the audit Tier-1–3 fixes — `withScalar` uses a single `instanceof Number` (no
  `ITC_INHERITANCE_TYPE_CHECKING`); `ChatMessage.getToolCalls` returns a fresh unmodifiable view (no
  `EI_EXPOSE_REP`); the `LlamaModel` batch methods' deliberate re-throw and the `ChatMessage` public
  constructor's `List` param carry narrow `<Match>` rationale suppressions.
  **Note:** `spotbugs:check` is bound to the `verify` phase, which the model-backed CI test jobs
  (`mvn test` / `mvn package`) do not reach — run `mvn verify` (or a dedicated job) to gate it in CI.

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

## Done (kept for history)

### 2026-08-25 — the five gaps recorded during the b10618 bump, now fixed

All five were found while bumping llama.cpp to b10618, recorded there as diagnoses rather than fixes
(the bump commit had to stay a bump), and closed in a follow-up. Details in CLAUDE.md; one-liners:

- **Model-gated Java tests silently self-skipped in CI** — Surefire's working directory is the
  module basedir while the GGUF cache is restored to the reactor root, so every `models/…` path
  resolved to nothing, every model-gated class aborted in `@BeforeAll`, and the job still went green.
  Fixed with `TestConstants.resolveModelPath` / `resolveModelProperty` (accept either layout), routed
  through every path constant and every `-Dnet.ladenthin.llama.*` fixture property, plus
  `TestConstantsTest` pinning the resolver **and** the wiring. `llama-langchain4j` had the same
  defect and got the same resolver as `TestModelPaths`. Verified end-to-end: with a placeholder at
  `<repo>/models/`, `LlamaModelTest` reports `Skipped: 0` and actually attempts the load.
- **`getMetrics()` payload contract drifted at b10408** — restored in the native layer instead of
  bending the Java contract: `handleSlotAction(0, …)` now posts both `SERVER_TASK_TYPE_METRICS` and
  `SERVER_TASK_TYPE_SLOT_GET` and merges them via the pure `server_metrics_to_json`. All three
  consumers keep working unchanged; `value.ServerMetrics` additionally exposes the cache and
  speculative-decoding counters that previously existed only in the Prometheus text.
  `LlamaModelTest#testGetMetrics` now asserts the parsed shape rather than substrings (the old
  assertion was satisfiable by the slot entries themselves), and `GET /slots` answers `[]` instead of
  an empty body when the payload carries no `slots` key.
- **`RouterClient` had no API-key support** — added `RouterClient(port, apiKey)` /
  `RouterClient(host, port, apiKey)` sending `Authorization: Bearer <key>`; an empty key behaves like
  none, `toString()` never prints it, `equals` includes it.
- **`RouterClient.awaitModelLoaded` vs hidden router models** — the TODO's first suggestion (poll to
  the timeout) was wrong: upstream filters hidden models out of `GET /models` permanently, so no
  amount of polling observes one. Fixed the honest way — the "not listed" message now names the
  `dedup-cache-models` cause and the javadoc documents the direct-request path.
- **`apply-llama-patches.cmake` was not idempotent** — replaced per-patch reverse-checking with a
  stamp file (llama.cpp commit + per-patch SHA-256) gated on git's clean/dirty state. A reconfigure
  over a patched tree is now a no-op; a genuine mismatch fails with an accurate message instead of a
  misleading "does not apply cleanly". Verified against the real build tree and a purpose-built
  two-patches-one-file fixture that reproduces the old failure.


### 2026-07-05 feature wave (PR #298) + follow-ups

One-liners for the sections removed from "Open" (full detail: PR #298, CLAUDE.md, git history):

- **NativeServer attach mode** — `NativeServer(LlamaModel, String...)` via `patches/0007`
  (`llama_server_attach`); serves an already-loaded model over the full upstream HTTP frontend.
- **Typed router API** — `server.RouterClient` + `value.RouterModel` + parser; router mode in-JVM via
  `patches/0008` + `NativeServer.setWorkerCommand`.
- **GGUF metadata inspector** — pure-Java `GgufInspector` + `value.GgufMetadata` (LE/BE, fail-loud).
- **Session fork/rewind** — `Session.checkpoint/rewind/fork` + `value.SessionCheckpoint`.
- **LangChain4j v1 + streaming** — tool calling, JSON mode, multimodal; streamed tool calls +
  per-token thinking via `StreamingChunkAssembler`.
- **UTF-8 JNI path, runtime LoRA control, typed batch embeddings, in-JVM quantizer** (first batches).
- **Android AAR + Kotlin façade + x86_64 ABI + emulator CI** — incl. the dlopen fix (`GGML_OPENMP OFF`
  + `-static-libstdc++`, DT_NEEDED whitelist; also fixed the latent 5.0.5 arm64 defect); emulator job
  promoted to a release gate; committed audio fixture (`audios/sample.wav`) wired as the
  `AudioInputIntegrationTest` default.
- **PIT gate hermeticity** — verified 295/295, 0 NO_COVERAGE with no fixtures; stale gotcha removed.
- **llama.cpp b9870 → b9873 → b9876** — all 8 patches re-verified each step.
- **Windows native classifiers (Ninja default flip + MSVC classifier + CUDA/Vulkan/OpenCL)** — shipped
  earlier; docs live in CLAUDE.md "Windows native classifiers".
- **b9739 Windows JNI arg-parse regression** — fixed via `patches/0001`; upstream submission tracked
  in "Upstream PR submissions" above.
- **Code audit (18 findings)** — fixed in PRs #258/#260 (+ tests #261/#262); only the optional
  extraction-directory isolation remains (own section above).
- **Branch protection aarch64 check rename** — closed as a no-op per owner.


### b9739 upgrade + PR #248 (Windows Ninja, native aarch64, patches mechanism)

- **llama.cpp b9682 → b9739** (#247, merged) + build fixes: `server-schema.cpp` added to the
  `jllama_test` sources (b9739 link fix, `38be6db`); `test_server.cpp` `ParamsFromJsonCmpl`
  expectations updated to b9739 schema behavior (`aaba886`).
- **Windows Ninja artifact** — `ninja-windows` classifier JAR built with Ninja Multi-Config + sccache,
  shipped alongside the permanent MSVC default; both build + Java-test jobs green (`e113ed3`,
  `48f0863`). (See the open section above for the design rationale; verification is done.)
- **Linux aarch64 → native `ubuntu-24.04-arm` build** (`ed9ecbb`). The dockcross `linux-arm64-lts`
  image (GCC 8.5 / glibc 2.17) could no longer compile b9739's C++17 CTAD-in-`new`; now builds natively
  with GCC 14 (mirroring upstream), runs `ctest` on real ARM (446 tests green), and warms sccache
  (99.66% hits). Trade-off: glibc floor 2.17 → ~2.39 (same envelope as upstream's ARM binaries);
  documented in the README classifier table. `build.sh` sccache auto-fetch generalized to aarch64.
- **Generic `patches/` mechanism** — drop `*.patch`/`*.diff` in `llama/patches/`, applied to the
  FetchContent'd llama.cpp source by `llama/cmake/apply-llama-patches.cmake` via the llama.cpp
  `PATCH_COMMAND` (cross-platform, idempotent, fail-loud). Covers every C++ build from one place.
  First patch fixes the Windows JNI arg-parse regression (`1d875b1` → deterministic form `f651b53`).
  REUSE annotated via `patches/**` glob (`0cffac1`).
- **CUDA sccache verified** — the `manylinux_2_28 (CUDA)` job caches all gcc C/C++ TUs (247/248 hits,
  99.60%); the nvcc `.cu` kernels remain uncached (sccache limitation), and `CUDA_FAST_BUILD` keeps
  PR/validation runs single-arch. (Doc/observation; no code change.)

### Layered package restructure (flat root package → layered hierarchy)

The flat `net.ladenthin.llama` root package was split (via `git mv`, history
preserved) into layered packages so boundaries align with the layers, enforced
by a new `layeredArchitecture()` ArchUnit rule (Api → Loader → Marshalling →
Foundation):

- **Foundation**: `value` (18 DTOs: ChatMessage, ContentPart, Pair, LlamaOutput,
  …), `callback` (CancellationToken, LoadProgressCallback, ToolHandler),
  `exception` (LlamaException, ModelUnavailableException), `args` (existing leaf).
- **Marshalling**: `json` (response parsers + `TimingsLogger`, its only consumer),
  `parameters` (Inference/Model/Json/Cli parameters + `ParameterJsonSerializer` +
  `ChatRequest`).
- **Loader** (internal, NOT exported): `loader` (LlamaLoader, OSInfo,
  ProcessRunner, NativeLibraryPermissionSetter, Java8CompatibilityHelper,
  OfflineModelGuard, LlamaSystemProperties).
- **Api** (root): LlamaModel, Session, LlamaIterable, LlamaIterator.

Cycle-breaking moves: `TimingsLogger` root→`json`, `ParameterJsonSerializer`
`json`→`parameters`, `ChatRequest` root→`parameters` (it carries an
`InferenceParameters` customizer). Test classes mirrored into their subjects'
packages; cross-layer members promoted to `public`. Cross-package Javadoc
`{@link}` references fully-qualified (palantir's `removeUnusedImports` strips
javadoc-only imports). `module-info` exports the new public-API packages and
keeps `loader` internal. All 11 ArchUnit rules green; `javadoc:jar` clean.

**Breaking change**: public-API FQNs changed (e.g. `net.ladenthin.llama.ChatMessage`
→ `net.ladenthin.llama.value.ChatMessage`) — ship under a major version bump.

- **Reactive `LlamaPublisher` removed in favour of consumer-side adapters.**
  The hand-rolled `LlamaPublisher` + `LlamaModel.streamPublisher` /
  `streamChatPublisher` (shipped in PR #188 as §2.3 of the Kotlin SDK
  feature comparison) had zero non-test callers. `LlamaIterable` is
  already `Iterable<LlamaOutput> & AutoCloseable`, and every mainstream
  reactive library wraps it in a few lines via its own resource-management
  primitive (`Flux.using`, `Flowable.using`, Kotlin `use {}`). The real-world
  Android consumer [LLaMAndroid](https://github.com/Rattlyy/LLaMAndroid)
  already uses `LlamaIterable` inside a Kotlin `flow {}` block — bypassing
  the publisher entirely. README "Reactive integration" section documents
  the Reactor / RxJava 3 / Kotlin Flow / Akka patterns; correctness is
  pinned end-to-end by a new `ReactorIntegrationTest` using
  test-scope `reactor-core` (zero runtime deps added; `org.reactivestreams`
  runtime dep dropped). Cleared 6 fb-contrib Max+Low findings on
  `LlamaPublisher$LlamaSubscription` as a side effect.

- **Error Prone bug-pattern promotions to `ERROR`** — `855f447` (12 patterns promoted; `-Xlint:all` enabled).
- **`javac -Werror` + `-Xlint:all,-serial,-options,-classfile,-processing`** — `3e2efbb`. ~20 EP warnings addressed first (EqualsGetClass on `Pair` via instanceof; MissingOverride on `PoolingType` / `RopeScalingType`; JdkObsolete `LinkedList` → `ArrayList` in `LlamaLoader`; StringSplitter inline-suppressed; 3× StringCaseLocaleUsage `Locale.ROOT` in `OSInfo`; EmptyCatch in `OSInfo.isAlpineLinux`; FutureReturnValueIgnored in `LlamaModel.completeAsync`; Finalize on `LlamaModel.finalize`; MixedMutabilityReturnType in 4 parser methods; EnumOrdinal in `InferenceParameters.setMiroStat`; EscapedEntity in `InferenceParameters` javadoc; 4× TypeParameterUnusedInFormals; AnnotateFormatMethod on `Java8CompatibilityHelper.formatted`; SafeVarargs + varargs on `Java8CompatibilityHelper.listOf`).
- **`-parameters` javac arg** — `4350cf2`.
- **`--release N`** — `4350cf2` (`<release>8</release>`).
- **Mutation-testing threshold enforcement (PIT)** — `62f8a00` + `bb93a8f` (docs) + `3bfa51f` (README badge). Runs every CI build with `<mutationThreshold>100</mutationThreshold>`. **Scope expanded 2026-06-07** from the original single `Pair` target (which was stale after the restructure — `llama.Pair`→`value.Pair` matched nothing) to `value.*` + `exception.*` + `args.*` + `json.TimingsLogger` = 27 classes / 163 mutations, all killed. Still open (optional): `json.ChatResponseParser` / `CompletionResponseParser` private-helper survivors (`RerankResponseParser` is excluded — equivalent empty-list mutant).
- **Checker Framework as a second static-nullness pass** — `c63870b`. The original
  `@PolyNull` on `JsonParameters.toJsonString` was simplified to plain `@Nullable`
  (the only `@PolyNull` site in production; eliminated in a later cleanup).
  Native-method constructor calls in `LlamaModel` carry
  `@SuppressWarnings("method.invocation")` (Checker's `@UnderInitialization`
  cannot see that the native callee does not dereference `this`); `Pair.equals`
  and `Usage.equals` declare `@Nullable Object`; `LlamaSystemProperties` getters
  return `@Nullable String`; `getPackage()` and resource-stream null derefs are
  guarded.
- **JPMS `module-info.java` with module-level `@NullMarked`** — `0fd066a` + `9528e79`. The module `net.ladenthin.llama` exports the three hand-written public packages (`net.ladenthin.llama`, `.args`, `.json`). Two-execution `maven-compiler-plugin` pattern; module-level `@NullMarked` lives on the module descriptor.
- **Banned-API enforcement** — Maven Enforcer (`8baae0c`), ArchUnit `System.exit` / `new Random` / `Thread.sleep` (`329d764`), `sun.*` / `com.sun.*` / `jdk.internal.*` (`e6069da`).
- **ArchUnit public-fields-final** — `7b6667d`.
- **LogCaptor smoke test** — `LoggingSmokeTest` (`3cedc6e`).
- **Offline / air-gapped model loading** — `ModelFlag.OFFLINE` + `ModelParameters.setOffline(boolean)` + `hasFlag` helper + public `ModelUnavailableException` (extends now-public `LlamaException`) + deterministic pre-check `OfflineModelGuard`. Unit tests in `LlamaModelOfflineTest`. No JNI rebuild required. *(Originally shipped as `SKIP_DOWNLOAD`/`setSkipDownload` over a parse-failure heuristic; reworked when llama.cpp b9803 removed `common_params::skip_download` and `common_skip_download_exception` — `--skip-download` was never a registered upstream arg, so it never actually skipped a download. `--offline` is the real upstream flag with the intended load-from-cache semantics.)*
- **`LlamaSystemProperties` registry cleanup** — `getLibName()` deleted (`6bb63e1` upstream forensic trace); `OSInfo.getArchName()` now routes through `LlamaSystemProperties.getOsinfoArchitecture()` (`3ae6c81`).
- **Abstract the Java and test writing guidelines to a workspace-level shared layer.** Workspace version chain at [`../workspace/guides/src/CODE_WRITING_GUIDE-8.md`](../workspace/guides/src/CODE_WRITING_GUIDE-8.md) and [`../workspace/guides/test/TEST_WRITING_GUIDE-8.md`](../workspace/guides/test/TEST_WRITING_GUIDE-8.md); canonical TDD skill at [`../workspace/.claude/skills/java-tdd-guide/SKILL.md`](../workspace/.claude/skills/java-tdd-guide/SKILL.md).
- **Standardised CLAUDE.md template** — [`../workspace/templates/CLAUDE.md.template`](../workspace/templates/CLAUDE.md.template).
