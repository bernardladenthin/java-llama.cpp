# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from version 5.0.0 onward. Pre-fork releases (`1.x`–`4.2.0`) were authored by
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp).

## [Unreleased]

### Upgrading from 5.1

Everything a 5.1 consumer has to change, in one place; the bullets below record each change where it
happened. Nothing was published between 5.1.0 (2026-08-29) and this release, which is 6.0.0 because
the first two items break every existing consumer at upgrade.

1. **The natives are separate jars.** `net.ladenthin:llama` is the Java classes only; a 5.1-style
   `<dependency>` on it alone resolves to a jar without natives and fails at the first load with
   `UnsatisfiedLinkError`. Depend on **`net.ladenthin:llama-platform`** with `<type>pom</type>`
   (Gradle: `implementation("net.ladenthin:llama-platform:6.0.0")`) -- the classes plus the CPU
   natives jar of every desktop platform -- or name the natives jars you need by classifier
   (`cpu-linux-x86-64`, `metal-macos-aarch64`, ...). A GPU is one **more** jar next to them
   (`cuda13-linux-x86-64`, `vulkan-windows-x86-64`, ...: one ggml backend module each, loaded when
   its driver is installed, skipped otherwise); the 5.1 GPU classifiers were complete replacement
   jars, and the `-jar-with-dependencies` fat jars are gone. The natives need the library jar of
   the **same version**: mixing a 6.0.0 module jar with another build's library jar fails the load
   before anything is extracted. Without Maven: a natives jar is a zip of one directory, and
   `-Dnet.ladenthin.llama.lib.path=<that directory>` loads it (README, "Natives outside Maven").
2. **The shipped SLF4J binding is `slf4j-simple`, not `logback-classic`.** `logback.xml` is no
   longer read; configure with `simplelogger.properties` or `-Dorg.slf4j.simpleLogger.*`, or
   exclude `org.slf4j:slf4j-simple` and declare your own binding. `checker-qual` no longer reaches
   your classpath at all (`provided` scope).
3. **Builder methods renamed or removed.** `setTensorReadLazy` / `TensorReadLazyMode` are
   `setLazyMode` / `LazyMode`; `enableFlashAttn()` and `ModelFlag.FLASH_ATTN` are gone, use
   `setFlashAttn(FlashAttn)`; twelve methods that wrote a name llama.cpp no longer reads are
   deleted without a deprecation window (`withTfsZ`, `withPenalizeNl`, `withPenaltyPrompt`,
   `withUseChatTemplate`, `withChatTemplate`, `setGrpAttnN`/`-W`, `enableDumpKvCache`,
   `setHfRepoV`/`setHfFileV`, `enableMlock`, `disableMmap` -- the Removed section names the
   replacement where one exists, e.g. `setLoadMode(LoadMode)`). The attach-mode server reads the
   endpoint toggles from `ModelParameters` now (`enableMetricsEndpoint()` & co.), not from its argv.
4. **`NativeServer` listens on 9931 when no `--port` is given** (llama.cpp's new default);
   `OpenAiCompatServer` stays on 8080. Pass `--port` where a fixed port matters.
5. **Platform floors:** Windows 10 or newer (Universal CRT), macOS 15.0, glibc 2.28 on Linux
   (RHEL 8, Ubuntu 20.04, Debian 10 and later), Android API 28; the ROCm jars need a ROCm 10
   runtime, the CUDA jars CUDA 13. 32-bit Windows and the `msvc-windows-x86-64` jar no longer ship.
6. **Rejected requests** come back as `InvalidRequestException` (a `LlamaException`, so existing
   handlers still match) with the bare reason as the message; the OpenAI-compatible server answers
   `400` for them.
7. **The Java floor is still 8.** The C API (`jllama_api.h` in every natives directory), the
   provenance record (`jllama-build.txt`, `LlamaModel.getJllamaBuildInfo()`) and upstream's own
   `libllama` next to `libjllama` are additions; nothing else about loading or calling changed.
8. **Values that were ignored or misread are refused when they are set** (`IllegalArgumentException`,
   see Changed): `TrainingParameters` -- and with it both `LlamaTrainer.finetune` overloads -- checks
   its documented bounds (`epochs` at least 1, `valSplit` in `[0, 1)`, exactly one of `trainingText` /
   `trainingFile`, `modelPath` and `outputPath` set); `ModelParameters.setPredict` refuses values below
   `-1` (`-2` stopped every request after one token; `0` behaves like `1`);
   `addLoraScaledAdapter` / `addControlVectorScaled` refuse a path containing a colon, i.e. every
   absolute Windows path, which llama.cpp's `FNAME:SCALE` format cannot express (load the adapter with
   `addLoraAdapter` and set its scale with `LlamaModel.setLoraAdapter`, or use a relative path); an
   empty or blank JSON fragment (`withJsonSchema("")`, `withToolsJson("")`, ...) and a `NaN` or
   `+Infinity` token bias are refused instead of being sent as `null` or dropped.
9. **The repeatable adapter options add up.** A second `addLoraAdapter`, `addLoraScaledAdapter`,
   `addControlVector` or `addControlVectorScaled` call adds to the first instead of replacing it, as
   their Javadoc always said; code that relied on the last call winning now loads every adapter.
10. **One request, one prompt, one completion.** Several prompts in one request
    (`"prompt": ["a", "b"]`) and `"n"` above 1 are refused as invalid requests
    (`InvalidRequestException`, `400` in the servers) on every native path -- `LlamaModel.handle*` and
    the streams, the C API, `llama-ffm` and every route of `OpenAiCompatServer` -- where the first prompt
    and a single choice used to be answered without a word. Send separate requests; a token array is
    still one prompt.
11. **`ServerMetrics.getCumulativeUsage()` counts the cached prompt tokens**: `getPromptTokens()` is now
    evaluated + cached tokens, like the `usage` of a response. Read `getProcessedPromptTokens()` (or
    `getCumulativeProcessedPromptTokens()`) where the evaluated tokens alone were meant.
12. **`LlamaModel.chatBatch` runs at most 64 requests at once** on threads of its own, and
    `completeBatch` / `completeBatchWithStats` submit every request from the calling thread; neither
    uses the common `ForkJoinPool` any more.
13. **`OpenAiCompatServer`**: `--model-id` / `--model-alias` refuse a comma, an empty value and
    leading or trailing whitespace, and a model file name with a comma is advertised with `_` in its
    place; `/api/generate` without `"raw": true` applies the model's chat template, as Ollama does
    (send `raw: true` for a bare completion); a route answers its own path only, so a sub-path such as
    `/v1/messages/count_tokens` is `404`; a negative rerank `top_n` is `400`.

### Added
- **The C API of `libjllama` (`jllama_api.h`, shipped in every natives directory).** Every operation
  of `LlamaModel`, `LlamaQuantizer`, `NativeServer` (classic, attach and the router's worker
  command), `RpcServer`, `TextToSpeech` and `LlamaTrainer` is a plain C function with an opaque
  handle, JSON text in and out, status codes with a thread-local `jllama_last_error()`, typed
  `*_free` functions for what the library returns, and C function pointers for the load-progress
  and log callbacks -- callable from C, C++, Python's ctypes or Rust without a JVM. The JNI entry
  points are thin shells over the same functions, so one implementation serves both call paths
  (`jllama_engine.{hpp,cpp}` holds it; the JNI files hold no server logic any more).
  `jllama_api_version()` is the library's Maven version, and a function present in `x.y.z` keeps
  its semantics in every later `x.y.*`. On Windows the linker's import library `jllama.lib` ships
  next to the header. `test_c_api.cpp` drives the whole surface inside `jllama_test` over a tiny
  random GGUF it writes itself, in every `C++ Tests` job; `docs/c-api.md` is the consumer page.
- **`net.ladenthin:llama-ffm`: Java 22+ without JNI.** The C API through the Foreign Function and
  Memory API -- `FfmLlamaModel`, `FfmTaskStream`, `FfmNativeServer`, `FfmRpcServer`,
  `FfmTextToSpeech`, `Jllama` -- over the same natives jars, loaded by the core's `LlamaLoader` (so
  `llama-platform`, the GPU module jars and `lib.path` apply unchanged), JSON in and out, the core's
  exceptions. Java floor 22 for this module only; the core stays Java 8. Built with a JDK 25 toolchain
  next to the reactor's JDK 21 (`maven-toolchains-plugin`), tested model-backed in CI
  (`test-java-llama-ffm`, gating both publish jobs). Not for Android.
- **Upstream's own C API ships next to `libjllama` (layer A).** On every modular platform (Linux
  x86-64 / aarch64, Android, Windows x86-64 / arm64) `libllama` and `libmtmd` are shared libraries
  of their own in the natives directory, exporting exactly upstream's `llama.h` / `mtmd.h` from the
  patched source -- the same file names and symbols as upstream's release zips. A plain-C test
  program (`src/test/c/test_c_abi.c`, `ctest c_abi`) links them in every modular build job, the
  first `ctest` those jobs run. macOS and s390x stay one static library.
- **Build provenance in every natives directory.** `jllama-build.txt` names the fetched llama.cpp
  commit, the classifier, the `net.ladenthin:llama` version and commit, the carried patches and one
  hash over them, the compiler, the vendor toolkit and the signature state, next to a CycloneDX
  `jllama-sbom.json`; `LlamaModel.getJllamaBuildInfo()` returns the keys and `Diagnose` prints them.
  `docs/patches.md` is generated from the patch headers and held current by CI. The `package` and
  publish jobs attest build provenance through Sigstore (`gh attestation verify --owner
  bernardladenthin <file>` proves a jar, a natives jar, a pom or an AAR came out of this workflow
  at a given commit, which the `.asc` alone cannot), and every Maven module attaches a CycloneDX BOM.
- **The natives directory is a contract, loaded from `lib.path` with the jar branch's logic.**
  `-Dnet.ladenthin.llama.lib.path=<dir>` (and `java.library.path`) now preloads what the
  directory's `jllama-extras.txt` names -- which is what makes the modular Windows build loadable
  from a directory at all, since Windows does not resolve `jllama.dll`'s imports next to the DLL --
  reads `jllama-build.txt`, reports listed files that are missing, recognises the GPU modules next
  to the library and prints the same loader line as the jar branch; the backend filter and the
  temp-directory cleanup run only when the jars are reached, so a stale filter cannot fail a
  `lib.path` start, and a failed load names its reason. `LlamaModel` warns when the loaded natives
  name another llama.cpp tag than the classes jar pins. `NativeLibraryLoadSmokeTest` launches a
  fresh JVM through `lib.path` on every Java test job, and every natives smoke unpacks the published
  jars into a directory and starts from it with no natives jar on the classpath. README, "Natives
  outside Maven", documents the directory, how to fetch a natives jar without Maven and the WDAC /
  AppLocker case.
- **The patched llama.cpp source of a release is an asset of its own, and mirrored.**
  `archive-llama-source` packs the pinned commit with the patches applied, reproducibly, as
  `llama.cpp-b<nnnn>-jllama-<version>-src.tar.gz`, signed and attached with the Maven artifacts;
  `test-source-archive` builds the C++ suite from that archive before either publish job runs, and a
  consumer rebuilds a release with `-DFETCHCONTENT_SOURCE_DIR_LLAMA.CPP=<unpacked>`. After a release
  `mirror-llama-source` pushes the pinned commit to `bernardladenthin/llama.cpp` as `jllama/b<nnnn>`
  (`JLLAMA_GIT_REPOSITORY` selects the mirror for a build); the patches stay files in this
  repository, never fork commits.
- **Modular natives: the GPU backends are additive module jars, and every platform a GPU can join is
  built modular.** A GPU natives jar (`cuda13-linux-x86-64`, `vulkan-windows-aarch64`, ... -- 15 of
  them) now holds **only** ggml's backend module for that GPU (`libggml-cuda.so`, `ggml-vulkan.dll`,
  ...; CMake option `JLLAMA_MODULE_ONLY`), and the library jar of the platform (`cpu-<os>-<arch>`,
  `metal-macos-aarch64` -- 8, what `llama-platform` collects) holds `libjllama` with ggml's shared
  libraries and the CPU backend modules (`JLLAMA_BACKEND_DL`; Linux x86-64 / aarch64, Windows x86-64
  / arm64 and Android x86-64 / aarch64 are modular, s390x and macOS stay static). `LlamaLoader` puts
  every GPU module jar of the platform it finds on the classpath next to the library, and ggml loads
  each module whose vendor runtime is installed; a module without its runtime is skipped (the loader
  logs `native backend '<lib>' loaded ... with GPU module(s) a, b`, `JNI_OnLoad` logs the backends and
  devices ggml registered) and the model runs on the CPU modules. Several GPU jars on one classpath
  are fine (llama.cpp dedups devices by `device_id`). A module jar alone is not enough: it needs the
  library jar of its platform **from the same build** -- every natives directory carries
  `jllama-build.txt` (the llama.cpp tag, written by CMake), and the loader refuses a module whose tag
  differs from the library's before extracting anything (`UnsatisfiedLinkError` naming both jars).
  Consumers: `llama-platform` + the GPU module jar(s), or
  `jbang --deps net.ladenthin:llama:<v>:<classifier> net.ladenthin:llama:<v>` (the classes jar's
  `Main-Class` is `ServerLauncher`). The GPU jars are built like upstream's GPU release zips
  (`ggml-cuda` & co. as the only target); the CPU jars like upstream's `win-cpu` / `ubuntu-cpu`.
- **`examples/server/pom.xml`**: the llama.cpp server from Maven Central with Maven -- `mvn exec:java
  -Dexec.args="-m model.gguf"` on the CPU of every desktop platform (`llama-platform`), one profile
  per GPU module jar named after its classifier (`-P cuda13-linux-x86-64`), and `-P assembly package`
  for a local `jar-with-dependencies` of exactly that combination. `check-natives.py` holds its
  version and profiles to `natives.csv` and the README.
- **Windows on ARM is modular too** (`cpu-windows-aarch64`: `jllama.dll`, `ggml.dll`, `ggml-base.dll`,
  `ggml-rpc.dll` and one `ggml-cpu.dll` -- ggml builds no ARM variant set for Windows), so the
  `vulkan-windows-aarch64` and `opencl-windows-aarch64` module jars join it; its C++ suite moved to a
  job of its own, `test-cpp-windows-arm64` (a static `clang-cl` build with the tests), as on every
  modular platform.
- **The Android AARs are modular too.** `llama-android` carries `libjllama.so`, `libggml.so`,
  `libggml-base.so`, `libggml-rpc.so` and one `libggml-cpu-<variant>.so` per ARM feature level
  (ggml's Android list, `android_armv8.0_1` ... `android_armv9.2_2`: dotprod, fp16, i8mm, SVE/SVE2,
  SME; x86-64 the 14 desktop levels) under `jni/<abi>/`, of which `JNI_OnLoad` scores and loads the
  best **by soname** (an app's libraries may stay inside the APK, so there is no directory to scan;
  the libraries carry no run path either -- bionic's search path is the app's native-library
  directory, and CMake's Android platform emits none); `llama-android-opencl` holds
  `libggml-opencl.so` alone and **depends on `llama-android`** (type `aar`), so declaring the OpenCL
  AAR is enough and AGP merges the two `jni/` trees -- additive, like the module jars on the desktop.
  A device without an OpenCL ICD runs on the CPU modules with a log line instead of needing the other
  flavor.
- **`net.ladenthin.llama.backend` is a filter over the GPU module jars**: `vulkan`, `cuda13,vulkan`,
  or `cpu` for none; a named module whose jar is not on the classpath, or an unknown name, fails the
  load with a message naming the jar to add. Unset, every module jar on the classpath is used.
- **The smoke sets.** `package` assembles, per desktop platform, the classpath a consumer gets --
  the classes jar, its runtime dependencies, the library jar and every GPU module jar of the platform
  (`package-smoke-sets.sh`, the jar list from `check-natives.py smoke-set`) -- and the `smoke-natives`
  matrix (Linux x86-64 / aarch64, Windows x86-64 / arm64), `smoke-natives-macos` and
  `smoke-agent-linux` launch the published jars from it: the loader line must name every GPU module
  of the platform, so each module put in place and skipped by ggml on the GPU-less runner is proven
  on the jars that go to Central. `check-natives.py` derives the targets from `natives.csv` and fails
  on one without a smoke job.
- **The loaded backends and devices are logged at start-up.** `JNI_OnLoad` logs, after loading the CPU
  modules next to the library, how many backends ggml registered and which devices they report (reaches
  `LlamaModel.setLogger`), and `LlamaLoader` prints how many module files it extracted for a backend:
  a module ggml cannot load fails silently (`GGML_LOG_DEBUG` only, no dialog on Windows), and without
  these lines a GPU module with a missing runtime was indistinguishable from one that was never there.
- **README: Windows clients under WDAC / AppLocker.** The DLLs are not Authenticode-signed and are
  extracted to `%TEMP%`; the troubleshooting section names the way out (`net.ladenthin.llama.lib.path`
  pointing at an allow-listed directory) and why the OpenPGP release key cannot sign them.
- **`ModelParameters.enableMetricsEndpoint()` / `enablePropsEndpoint()` / `enableSlotsEndpoint()` /
  `disableSlotsEndpoint()`** (`--metrics`, `--props`, `--slots`, `--no-slots`): the HTTP frontend's
  endpoint toggles, needed since `NativeServer` attach mode serves the model's own routes (see Fixed).
- **`test-cpp-windows-x86_64`**: the C++ suite on Windows x86-64 with plain clang (a static build with
  the tests), the compiler the shipped `cpu-windows-x86-64` natives are built with now; the two MSVC
  jobs keep running it with `cl.exe`. Gates both publish jobs like every other job.
- **`-D` overrides for the four shipped test models.** `net.ladenthin.llama.text.model`, `.draft.model`,
  `.reasoning.model` and `.rerank.model` override `TestConstants.MODEL_PATH`, `DRAFT_MODEL_PATH`,
  `REASONING_MODEL_PATH` and `RERANKING_MODEL_PATH` the way every capability-specific model already had
  one, so a developer can run the suite against models already on the machine (README "System
  Properties Reference"). The defaults stay the CI model set, which `TestConstantsTest` pins over the
  `DEFAULT_*` constants.
- **`InvalidRequestException`** (`net.ladenthin.llama.exception`, a `LlamaException`): the JNI layer now
  throws it wherever llama.cpp's own HTTP server answers `400` (`invalid_request_error`) -- a body that is
  not valid JSON, a missing `"prompt"` / `"messages"` / `"input_prefix"`, an empty embedding input, an
  unknown `encoding_format`, a parameter outside its hard limits, a prompt over the context size, and a
  `std::invalid_argument` or `common_json_error` out of tokenization, prompt formatting or the
  `/v1/systemone` handler (the same rule as upstream's `ex_wrapper`). Everything else stays a plain
  `LlamaException`. A `catch (LlamaException)` keeps catching both.
- **`RouterModel.getContextLength()`** (llama.cpp b11538, #30228): the context length a model was
  trained with, which the router now reads offline from the GGUF for `GET /models` (`context_length`,
  `<arch>.context_length`) -- known before the first load, like the modalities; `0` from a server before
  b11538 or for a file without the key.
- **CI proves HTTPS in both directions on the published jars.** The natives smokes (Linux x86-64 /
  aarch64, Windows x86-64 / arm64) and the macOS smoke each start the server behind a self-signed
  certificate (`--ssl-key-file` / `--ssl-cert-file`; a plain-HTTP request to the port must be refused)
  serving a model it downloaded itself from an `https://` URL (the 1 MB `stories260K.gguf` of
  `models.csv`) with the operating system's certificate store -- the check that would have caught the
  Linux defect below, and the only one that can see the macOS one (the runner has Homebrew's OpenSSL).
- **`examples/jbang/Chat.java`**, a one-file console chat that [JBang](https://www.jbang.dev) runs without a
  checkout or a build (`jbang https://github.com/bernardladenthin/java-llama.cpp/blob/main/examples/jbang/Chat.java
  model.gguf`). Its `//DEPS` lines name the classes jar and the CPU natives jar of every desktop platform
  themselves -- JBang treats a `pom` dependency such as `llama-platform` as a BOM and puts nothing of it on the
  classpath -- and `check-natives.py` holds them to the `platform=yes` rows of `natives.csv` and to the README's
  release version.
- **`ModelParameters.setMoeCacheMib(int)`** (`--moe-cache-mib`, llama.cpp b11480, EXPERIMENTAL upstream): a
  GPU cache for the MoE expert weights a `setCpuMoeLayers` / `--cpu-moe` setup keeps in host memory,
  split among several GPUs like the layers; `0` (the default) disables it.
- **Kolibri-1 support** (Aleph Alpha, architecture `kolibri1`, 78B German/English reasoning MoE) ahead of upstream
  llama.cpp ([ggml-org/llama.cpp#29922](https://github.com/ggml-org/llama.cpp/issues/29922)), as the carried patch
  `0016-model-kolibri1.patch`. It combines the two community ports and, unlike either of them, loads the GGUFs of
  both community converters. Guarded by `test_kolibri1.cpp`, which compares tiny random models with an
  independent reference written from Aleph Alpha's vLLM implementation. The patch is dropped once upstream adds
  the architecture.
- **`GpuSplitMode.TENSOR`** (`--split-mode tensor`, tensor parallelism, EXPERIMENTAL upstream). The mode
  existed upstream before; since llama.cpp b11450 (#26610) it also works across RPC servers.
- **Input/output modalities on `RouterModel` and `ModelMeta`** (llama.cpp b11429, #29987):
  `getInputModalities()`, `getOutputModalities()` and `isDecisionModel()`, from upstream's new
  `architecture` object of `GET /models` (the router computes it offline, so a decision model is
  recognisable before its first load) and, for a loaded `LlamaModel`, from the same metadata in
  `getModelMeta()`. Empty against a server before b11429.
- **`vulkan-windows-aarch64` natives jar** (Windows on ARM with a Vulkan 1.2+ driver), following
  upstream's new Windows arm64 Vulkan release (llama.cpp b11395, #29954). Built natively on
  `windows-11-arm` with `clang-cl`; a module jar, additive to `cpu-windows-aarch64` like
  `opencl-windows-aarch64`.
- **`ModelParameters.setDraftSampling(DraftSampling)`** (`--spec-draft-sampling`, llama.cpp b11368):
  `PROBABILISTIC` samples the speculative draft and has the target verify it by rejection sampling,
  which accepts more drafted tokens at a temperature above zero; `GREEDY` is upstream's default. Applies
  to a draft model and to a model's own MTP heads.
- **Decision models: `LlamaModel.handleSystemOne(String)`**, llama.cpp's TypeSafe-compatible
  `/v1/systemone` API (upstream b11361): typed `choice` / `score` / `noul` questions about a state,
  answered with probabilities in one forward pass, for the decision models upstream supports (laya,
  julia-1, lev, openjev, kev, ...). The JNI method forwards to upstream's own route handler, so the
  request and response are exactly the HTTP endpoint's; `NativeServer` serves `POST /v1/systemone` in
  classic and attach mode. A model that is not a decision model throws a `LlamaException`.
- **`net.ladenthin:llama-atmosphere-agent` on Maven Central**, at the core's version: the agent's thin jar
  (with `Main-Class`), sources and javadoc, published right after the reactor. Its pom names
  `llama-platform` as a runtime dependency, so `jbang net.ladenthin:llama-atmosphere-agent:<version>`
  starts it with the CPU natives of every desktop platform, no checkout and no download by hand. The
  release-asset jar without the core stays on the GitHub release. The agent's version now moves with the
  core's; `check-natives.py` fails when they differ.
- **`InferenceParameters.withLogitBiasJson(String)`**: `logit_bias` as one checked JSON value in either
  form llama.cpp's request schema reads -- an array of `[token, bias]` pairs or OpenAI's object of
  token ids to biases, `false` banning a token. The OpenAI request mapper forwards a client's
  `logit_bias` with it, and `n_probs` with `withNProbs`, on every chat route.
- **`InferenceParameters.withResponseFields(String...)`** (request field `response_fields`,
  `RequestField.RESPONSE_FIELDS`): upstream's projection of the native answer, and with `withVerbose`
  of the `__verbose` object of an OpenAI answer, onto slash-separated paths (`timings/predicted_n`);
  the projected key is the whole path, and a path the result lacks is dropped without an error, as
  upstream does. The OpenAI mapper forwards a client's `response_fields` and answers `400` for
  anything but an array of strings.
- **`RouterModel.getAliases()` and `isNamed(String)`**: the aliases `GET /models` lists for a model
  (`--alias` of a preset), parsed by `RouterModelsResponseParser`; `RouterClient.findModel` and
  `awaitModelLoaded` match a model by its id or any alias, as the router resolves the name of a
  request.
- **`OpenAiServerConfig.Builder.totalSlots(int)` / `getTotalSlots()`** (advertised as `total_slots` in
  `GET /props`) and the `OpenAiServerCli.Options.toServerConfig` overloads taking the context size
  of one slot and the slot count of the loaded model, which `OpenAiCompatServer.main` now passes.
- **`RerankResponseParser.toDocumentScores(...)`**: the document-to-score map of a rerank result in
  the order of the documents (see Fixed).

### Changed
- Dependencies and build tooling bumped in step with the sibling repositories: Jackson 2.22.3 ->
  2.22.4 (`jackson.version`, which the `jackson-bom` and the AAR pom follow), LogCaptor 2.12.7 ->
  2.13.0 (test; still requests log4j 2.26.1, so the log4j floor guard is unchanged); in the agent
  Atmosphere 4.0.72 -> 4.0.75 and central-publishing-maven-plugin 0.11.0 -> 0.12.0 (the reactor
  already had it); jsign 7.2 -> 7.6 for the Windows code signing (`sign-windows-natives.sh`, new
  SHA-256; same options, same eSigner authentication path); Gradle 9.8.1 -> 9.8.2 (`GRADLE_VERSION`
  and the shared `verify-signing-key-gradle` job); `anthropics/claude-code-action` v1.0.255.
- **Values that were ignored, misread or left to fail at load time are refused when they are set,
  with an `IllegalArgumentException` naming the value:**
  - `TrainingParameters.toJson()` (and so `LlamaTrainer.finetune`) checks the documented bounds before
    the JNI call: `epochs(0)` ran no epoch and wrote the unchanged model to `outputPath` as a silent
    success, both corpus fields set let `trainingFile` win, a `valSplit` outside `[0, 1)` (`NaN`
    included) reached ggml unchecked, and a missing `modelPath` / `outputPath` was a bare
    `NullPointerException`.
  - `ModelParameters.setPredict` refuses values below `-1`. Its Javadoc promised `-2` = "until
    context filled", which is a setting of llama.cpp's completion tool: in the server a slot counts
    only `-1` as unlimited, so every request stopped after its first token without an error
    (measured). The Javadoc of `setPredict` and `InferenceParameters.withNPredict` now says that `0`
    behaves like `1`, and that the server answers a per-request `n_predict` of `-2` as an invalid
    request.
  - `setRopeScaling(LONGROPE)` and `setRopeScaling(MAX_VALUE)` throw, where they made the model
    unloadable (llama.cpp's parser takes only `none`, `linear` and `yarn`); `UNSPECIFIED` removes the
    option and leaves the model's default, as `setPoolingType(UNSPECIFIED)` now does too (it left an
    earlier `--pooling` in place).
  - A token bias of `NaN` or `+Infinity` (`withTokenIdBias`, `withTokenBias`) is refused, and
    `-Infinity` bans the token: Jackson wrote non-finite floats as strings, which llama.cpp skips, so
    banning a token that way had no effect.
  - `JsonParameters` refuses an empty or blank fragment (`withJsonSchema("")`,
    `withResponseFormat("")`, `withToolsJson("")`, ...), which the strict reader parsed to nothing and
    the request carried as `"key": null`, i.e. generation ran without the schema the caller set.
  - `OpenAiServerConfig.Builder.heartbeatMillis` refuses a value of `0` or below: as the period of the
    heartbeat timer it made every streamed request fail with `200` and an
    `IllegalArgumentException` error event, reported as the client's fault.
  - `OpenAiServerCli` refuses a `--port` outside 0..65535 as a usage error before the model loads;
    `OpenAiCompatServer.main` used to load the model, which can take minutes, and then die with an
    uncaught "port out of range".
- **`LlamaModel.completeBatch`, `completeBatchWithStats` and `chatBatch` no longer run on the common
  `ForkJoinPool`**, whose `availableProcessors() - 1` threads the application shares: 8 requests on
  8 slots ran 3 at a time on a 4-core machine (measured). `completeBatch` and
  `completeBatchWithStats` submit every request from the calling thread and then receive the results
  in order, releasing the requests already submitted when a submission fails; `chatBatch`, whose
  native call submits and waits in one, runs each request on a daemon thread of its own, at most 64
  at once, and throws the first failure of any kind only after every request has finished (an
  `Error` used to end the wait while other requests still ran).
- **Ollama `/api/generate` follows Ollama's semantics.** It completed every prompt raw; now a request
  with `suffix` is fill-in-the-middle as before, one with `"raw": true` a bare completion (a `system`,
  `template` or `context` next to it is refused with 400, as Ollama does), and any other goes through
  the model's chat template with `system` as the system message and the prompt's `images`, `format`
  and `think` honoured, answering `thinking` next to `response`. A request with neither a prompt nor
  a suffix -- how clients preload a model -- is answered at once with Ollama's load response
  (`done: true`, empty `response`, `done_reason` `"load"`; a single object even when streamed)
  instead of answering an empty user turn.
- **`OpenAiCompatServer --model-id` / `--model-alias` take a value llama.cpp keeps as one alias.**
  llama.cpp splits `--alias` at commas and strips every part, so `--model-id my,model` was advertised
  as `my,model` in `/v1/models` and named `model` in every answer, and a blank id fell back to the
  model path. A comma, an empty value and leading or trailing whitespace are a usage error now; an id
  derived from a file name replaces a comma with `_` and is trimmed (blank: the default id).
- **`libjllama` exports only its interface.** Hidden visibility for the project's own translation
  units, a version script on ELF and an exported-symbols list on macOS keep the statically linked
  llama, llama-common, mtmd, cpp-httplib and BoringSSL symbols local: the Linux library exported
  14,220 symbols, 47 of them JNI -- an accidental C++ ABI with no header contract -- and exports
  `JNI_OnLoad`, `JNI_OnUnload`, `Java_*` and `jllama_*` now, which `verify-native-deps.py` holds it
  to (ELF `.dynsym`, PE export directory, Mach-O symbol table) in `package`.
- **Every build input is pinned by content.** All `uses:` of the workflows carry the action's commit
  SHA (Dependabot keeps them current); CI refuses a llama.cpp fetch by tag name
  (`JLLAMA_REQUIRE_COMMIT`), and `JLLAMA_GIT_COMMIT` stays set after upstream publishes the tag;
  googletest is pinned by commit; the sccache archives are checked against mozilla's SHA-256s, the
  CUDA redist archives against NVIDIA's manifest, the OpenVINO archives and the Vulkan arm64
  installer against the checksum each vendor publishes next to the file.
- **The Windows DLLs are Authenticode-signed with the project's SSL.com certificate (eSigner) on a
  publish run started with the new `sign_windows` switch**, snapshot or release alike -- the job
  `sign-windows-natives` (`.github/sign-windows-natives.sh`: jsign, the secrets of the `maven-central`
  environment, an RFC 3161 timestamp) re-uploads the Windows natives artifacts signed, `signed=yes`
  in each directory's `jllama-build.txt`, and `verify-native-deps.py` refuses an unsigned DLL on such a
  run. Without the switch every build says `signed=no`, as before. (Azure Trusted Signing, the first
  choice, takes no individual in Germany and was dropped before it ever ran.)
- **A `v*` tag release is built without the shared compiler cache.** `USE_CACHE` is `false` on every
  tag ref whatever the dispatch input says, so each natives build of a release compiles every object
  in the job that ships it; snapshots and PR builds keep the sccache/Depot cache. The cache provider
  is thereby out of a release's trust chain.
- **Lombok-generated code carries JSpecify nullness** (`lombok.addNullAnnotations = jspecify` in the
  shared `lombok.config`): the generated `equals(Object)` / `canEqual` parameter is `@Nullable`, as
  `Object.equals` promises. Under the `@NullMarked` packages it read as non-null before, which
  SpotBugs 4.10.5 (JSpecify-aware since this release) reported as
  `NP_METHOD_PARAMETER_TIGHTENS_ANNOTATION` on 27 value and parameter classes, and which told a
  consumer's nullness checker the same wrong contract about the published bytecode. Build tooling:
  spotbugs-maven-plugin 4.10.4.1 -> 4.10.5.0, JLine 4.4.7 -> 4.4.8 (the agent).
- **The local agent speaks ACP through SDK 0.80.0** (`com.agentclientprotocol:acp-core` +
  `acp-json-jackson2`, from 0.18.0 -- upstream's binary-incompatible release of 2026-10-08). The wire
  format is unchanged, so `AcpServerTest`, which drives the server with plain JSON, is unchanged too; the
  port is mechanical: session updates go out through `sendSessionUpdate` without the discriminator
  argument the records now write themselves, `session/request_permission` is a call on the prompt
  context's `client()`, `awaitTermination()` replaces `await()`, and the constructors that gained the
  protocol's new tool-call `name` carry the agent's tool name there (`read_file`, `shell`, ...), so an
  editor can show it next to the title.
- **Build tooling in step with the sibling repositories:** Spotless 3.10.4, Checker Framework 4.3.1,
  `exec-maven-plugin` 3.6.4 in `examples/server/pom.xml` -- which `check-versions.py` reads now, so the
  drift it had (3.6.2 against the reactor's 3.6.4) can no longer go unseen -- and the Compose compiler
  plugin 2.4.21, the Kotlin the POMs already used. `check-versions.py` compares the Kotlin Gradle
  plugins of every `*.gradle.kts` with the POMs' Kotlin from now on: Dependabot bumps `kotlin.version`
  and never sees the Gradle files.
- **`print-host-info.sh` is a shared file** (`.github/shared-files.sha256`, in all four repositories):
  the runner's CPU at the top of every test job, which is how a chosen CPU module or a timing is read
  against the hardware the job drew.
- **Upgraded the pinned llama.cpp from b11538 to b11556**, one reviewed step (18 commits, 96 files, no
  `tools/ui` change, so the WebUI is unchanged). All twelve patches apply unchanged and every drop-check
  still finds its defect. In the range: a request pinned to a **busy** slot (`id_slot`,
  `InferenceParameters.withSlotId`) no longer runs the prompt-cache update on that slot while another
  generation uses it -- a RAM-cache match used to be loaded under the running generation, which then
  continued on the wrong context (#30295); AVX2 CPUs with F16C convert fp16 in hardware (#30270 -- the
  `haswell` CPU module and every x86 module below AVX-512); a race in the CUDA/HIP flash-attention tile is
  fixed (#30103); upstream's **vendored nlohmann/json carries a backport of the deep-nesting fix**
  (#30253, nlohmann/json#5387: copying, `dump()` and comparing a value nested deep enough crashed) -- and
  every TU of `libjllama` compiles against that vendored copy, measured from the include order, so the
  FetchContent `v3.12.0` is shadowed (CLAUDE.md, `TODO.md`); `common_batch` gained a per-token `state`
  embedding for MTP drafting (#30257, no project code builds those tokens); `--slot-save-path` reads
  `LLAMA_ARG_SLOT_SAVE_PATH` (#30272); new model support (Prism Bonsai 2 27B #29600, MiniCPM-V 4.7 #29416)
  and backend work (Vulkan `mul_mat_id`, SYCL MXFP4 MoE, OpenCL MoE kernels, MMVQ on sm_60) need nothing on
  the Java side. Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **Upgraded the pinned llama.cpp from b11556 to b11558**, one reviewed step (2 commits, 7 files, no
  `tools/ui` change, so the WebUI is unchanged). In the range: the RPC backend's logging is one header
  with **`GGML_RPC_DEBUG` as a verbosity level 0-3** instead of an on/off flag (#29544), and its server
  lines (`Starting RPC server`, `Accepted client connection`, `Client connection closed`) go through
  ggml's log callback instead of `printf` -- stderr by default, and a `LlamaModel.setLogger` sink sees
  them at verbosity 4 like every ggml INFO line; the OpenCL module's Adreno kernel selection respects the
  image size limit and the weights' batch dimensions (#30310). Patch `0015` (embeddable RPC) is
  refreshed onto the rewritten logging context with the same changes; the other eleven apply unchanged
  and every drop-check still finds its defect. Per-step record:
  `docs/history/llama-cpp-breaking-changes.md`.
- **Upgraded the pinned llama.cpp from b11558 to b11574**, in 5 reviewed steps, each ending at a tag
  (b11559, b11560, b11570, b11573, b11574; 16 commits, 69 files; one `tools/ui` fix, which the
  auto-followed WebUI picks up). What reaches this project: **ggml's shared libraries hide their
  symbols by default** (#30014, under `BUILD_SHARED_LIBS` only, i.e. the modular library jars) --
  everything `libjllama` and the modules take from `ggml` / `ggml-base` is public API, verified by a
  local modular build and the Java native-load smoke over it; `llama_perf_context_data` reports
  **compute-graph reallocations** (#30329: two appended fields and a `graph reallocs` log line per
  slot, no JSON change, so `Timings` and `getMetrics()` are untouched); two new **decision model
  types** for `/v1/systemone` (#30378 vllm-decision2 with a strict payload, #30383 vllm-d3 with
  non-causal attention) behind an unchanged contract for the existing ones; `--kv-unified` reaches the
  speculative example (#30095). Patch `0012` is refreshed onto #30299's `tests/CMakeLists.txt`
  (`LINK llama`); the other eleven apply unchanged and every drop-check still finds its defect.
  Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **No fat jars, no release assets beyond the Maven artifacts.** The all-backends
  `llama-<v>-all-<os>-<arch>-jar-with-dependencies.jar`, the default CPU fat jar and the agent's
  `llama-atmosphere-agent-<v>-jar-with-dependencies.jar` are gone (see Removed): with modular natives
  a jar of every backend of every platform is the opposite of the point, and a consumer's classpath is
  what Maven resolves. The GitHub Release and the rolling `snapshot` pre-release attach the Maven
  artifacts -- classes jar, natives jars, poms, sources, javadoc, the AARs -- with the `.asc`
  maven-gpg wrote for Central, and nothing else; `github-release-signed` / `github-snapshot` sign
  nothing themselves any more. Replacements: `jbang --deps net.ladenthin:llama:<v>:<classifier>
  net.ladenthin:llama:<v> …` and `examples/server/pom.xml` (above); the agent through
  `jbang net.ladenthin:llama-atmosphere-agent:<v>`, a GPU as one more `--deps` jar.
- **The two Linux CPU jars and `cpu-windows-x86-64` keep their modular layout, now built by
  `-DJLLAMA_BACKEND_DL=ON`** (the option replaced `JLLAMA_CPU_VARIANTS`, whose only difference was
  that no GPU module could join it); `ggml` / `ggml-base` are built unversioned, and the extraction
  directory is keyed by the library's jar entry **and** the module set, so a classpath with other GPU
  jars never shares a directory with one without them.
- **SYCL is one jar per OS, fp16** (`sycl-linux-x86-64`, `sycl-windows-x86-64`; `GGML_SYCL_F16=ON`
  on both, as upstream's SYCL release builds): `sycl-fp16-linux-x86-64` is renamed, `sycl-fp32`
  dropped (see Removed).
- **The MSVC build no longer ships** (`msvc-windows-x86-64`): `build-windows-x86_64-msvc` still
  builds with the Visual Studio generator + `cl.exe`, runs `ctest` and the full model-backed Java
  suite (`test-java-windows-x86_64-msvc`) as the second, independent toolchain, but uploads no
  natives jar -- a second static CPU library next to the clang one was only a second CPU library.
- **The extracted natives are reused across JVM starts.** `LlamaLoader` used to delete every extracted
  file at exit and every `jllama*` entry of the temp directory at the next start, so each JVM copied
  the whole backend directory again (18 files for a CPU-variants build, ~1 s with an on-access scanner
  on Windows, for every surefire fork too). The per-build extraction directory
  (`jllama-backend-<backend>-<key>`) now stays: a later start of the same build compares each file
  with the jar byte for byte and writes only what differs; a directory of a build that is not on the
  classpath is removed once it is 10 minutes old (a JVM that is still starting touches its directory
  first). `net.ladenthin.llama.tmpdir` still chooses the location.
- **The log sink attaches llama.cpp's log worker to the JVM once per thread, not once per line.**
  `LlamaModel.setLogger`'s trampoline used to `AttachCurrentThread` + `DetachCurrentThread` for every
  log line, i.e. one `java.lang.Thread` object and a pair of JVMTI thread events per line (13 lines of a
  failed load, 13 `Thread` objects). A `thread_local` guard now keeps the attach (as a daemon thread
  named `jllama-log-worker`, visible in thread dumps) until the worker thread exits, which
  `common_log::pause()` makes a normal exit; each delivery runs in its own local-reference frame.
  `LlamaLoggerTest` pins both.
- **`cpu-windows-x86-64` is the modular CPU build, built with clang** (`build-windows-x86_64`,
  `-DJLLAMA_BACKEND_DL=ON`): like the two Linux CPU jars since 6.0.0, the directory holds
  `jllama.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll` and one `ggml-cpu-<level>.dll` per
  instruction-set level (14), of which ggml loads the best for the running CPU -- an AVX-512 / AVX-VNNI
  / AMX machine uses its kernels and a CPU without AVX2 works, where the one static library was built
  for one level. Plain clang because only the GNU driver yields all 14 variants (`clang-cl` and
  `cl.exe` lose five, CLAUDE.md "Modular natives"); same hybrid CRT, no new runtime dependency
  (`verify-native-deps.py` holds the directory to the OS imports). `LlamaLoader` pre-loads
  `ggml-base.dll` and `ggml.dll` by full path from `jllama-extras.txt` before `jllama.dll`, so the
  import never resolves against a foreign llama.cpp install on `PATH`. The job no longer installs
  sccache (sccache 0.18.0 refuses the `-Xclang` CMake puts on every plain-clang Windows TU, so every
  TU is a cache miss) and no longer runs `ctest` (the option refuses `BUILD_TESTING`; see
  `test-cpp-windows-x86_64` under Added).
- **The message of a rejected request is the bare reason** (`"prompt" must not be empty`), no longer the
  JSON object `{"code":400,"message":…,"type":"invalid_request_error"}` the old `throw_invalid_request`
  wrapped it in: the type is the exception class now (`InvalidRequestException`), as it is for every
  other `LlamaException`.
- **`NativeServer` binds port 9931 when no `--port` is given** (llama.cpp b11521,
  [#30159](https://github.com/ggml-org/llama.cpp/pull/30159) moved the server default off 8080), in classic and
  attach mode alike; `NativeServer.getPort()` reports 9931 accordingly. `OpenAiCompatServer` keeps its own default
  of 8080 (`OpenAiServerConfig.DEFAULT_PORT`, a public constant of this project with no upstream coupling). Pass
  `--port` explicitly where a fixed port matters; every README example already does.
- **CI: the Linux CUDA build no longer routes nvcc through sccache by default** (`SCCACHE_WRAP_NVCC=true`
  in `build.sh` opts back in). With CUDA 13.4 and sccache 0.18.0 the nvcc launcher failed on the first `.cu`
  files in two confirmed runs (`Missing "cubin" file output`, `Compiler killed by signal 126`), and the
  uncached retry that followed made those jobs slower than a cold build; the C/C++ objects stay cached.
- **Upgraded the pinned llama.cpp from b11534 to b11538**, one reviewed step. All eleven patches apply
  unchanged and every drop-check still finds its defect. In the range: the router's `GET /models` carries
  `context_length` per model (#30228, the `RouterModel` getter above; `common_get_decision_type(fname)`
  became `common_get_gguf_info()`, which no project code called), the embedding `get_rows` is built after
  the inputs (#30160), a CUDA rounding mismatch between CPU and GPU under MSVC is fixed (#30229), and
  `llama-bench` respects `-fitc` (#28331). Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **CI: the three dockcross images are pinned by digest** (`android-arm64`, `android-x86_64`,
  `manylinux_2_28-x64`, all tag `20260712-79e54f9`), as the manylinux aarch64 image already was.
- **Upgraded the pinned llama.cpp from b11529 to b11534**, in 3 reviewed steps, each ending at a tag. No
  carried patch needed a refresh, and every drop-check still finds its defect. The one step with project
  code is b11531, **#30210 "chat : refactor API"**: the prompt and the parser state of a chat generation now
  live in a `common_chat_session` that `oaicompat_chat_params_parse` fills and the server task applies,
  replacing the `chat_format` / `chat_parser` / `generation_prompt` / `parse_tool_calls` request fields
  (none was a `RequestField`, so the Java wire surface is unchanged); `jllama.cpp` threads a session through
  its four chat entry points and the C++ tests follow the new signatures. Also in the range: OpenCL kernels
  that compile on Adreno A6x (#30176, the `opencl-android-aarch64` jar and the `llama-android-opencl` AAR on
  that family), exact GELU for ModernBERT encoders (#30108), and no redundant CUDA copies after `SSM_SCAN`
  (#29807). Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **CI: every `test-java-*` job now requires at least 1800 tests *executed*** (`verify-test-counts.sh
  --min-executed`, run minus skipped), where it required 1500 *run* before. The run count cannot see the
  other shape of the muted-suite failure -- method-level assumptions skipping in bulk -- while the executed
  count does: the jobs execute 1856 (Windows) to 1865 (Linux) tests, a checkout without the models 1589.
- **Upgraded the pinned llama.cpp from b11512 to b11529**, in 5 reviewed steps, each ending at a tag. No
  carried patch needed a refresh, and every drop-check still finds its defect. In the range: the server's
  default port moves from 8080 to 9931 (#30159, the `NativeServer` entry above); CUDA top-k selection is
  reworked (#28713) and the MMQ config helpers take the `src1` precision (#30168); SYCL gains a Q5_K
  reorder-layout MMVQ and a fused GLU (#29375); Metal gets flash-attention kernels for the 128/96 head-size
  pair (#30209); Vulkan's `rms_norm` no longer overflows its workgroup count (#30145); `ggml_acc` no longer
  writes out of bounds for a large offset (#30135); the meta backend handles host views under
  `--split-mode tensor` (#30217); the WebUI gains a models manager (#29583, auto-followed by `build-webui`).
  Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **Windows natives: no Visual C++ redistributable, and `ucrtbase.dll` stays serviceable — but
  Windows 10 or newer is now required.** The Windows CPU jars (`cpu-windows-x86-64`,
  `cpu-windows-aarch64`) and the GPU module jars use the **hybrid CRT**: the C++
  standard library and vcruntime are linked statically, so `msvcp140.dll` / `vcruntime140.dll` / `vcruntime140_1.dll` are not needed on the
  target machine, while the Universal CRT remains the operating system's. A plain `/MT` would link
  the UCRT in as well and freeze a copy of it inside `jllama.dll`, where Microsoft's UCRT security
  updates could never reach it — for a library that parses network data and untrusted GGUF files
  that is the deciding argument (it is also Microsoft's own recommendation for self-contained
  binaries, and the library shrinks slightly: 10.1 MB → 9.8 MB). **The one cost is the new
  floor:** the UCRT is an OS component from Windows 10 on, so the Windows natives no longer run on
  Windows 7 / 8.1 without the Universal CRT update. `.github/buildcheck/nativedeps.py` enforces
  both halves per release — `msvcp140` appearing fails the build, and the `api-ms-win-crt-*`
  forwarders disappearing fails it too. **The SYCL and OpenVINO jars are the exception** and keep
  the dynamic `/MD` runtime, because `icx -fsycl` rejects `/MT` outright and the OpenVINO import
  libraries are built `/MD`; they require their vendor runtime on the host anyway, so the
  self-contained-library argument does not apply to them.
- **Windows CPU inference generates tokens about twice as fast** (`-DGGML_OPENMP=OFF` on all four
  Windows x86-64/x86 CPU build jobs, which the arm64 jobs already used for a dependency reason).
  Measured on a Ryzen 7 5800H with Qwen3-0.6B Q4_0 at 8 threads, four static builds from one
  source tree with the runs interleaved: token generation 41.7 → 79.9 t/s with MSVC (1.9×) and
  30.2 → 82.3 t/s with clang (2.7×), prompt processing unchanged within error (359.8 → 371.0 and
  390.6 → 391.6). ggml uses OpenMP only to manage threads, never for the arithmetic: with it every
  graph computation enters a `#pragma omp parallel` region, and one generated token *is* one graph
  computation, so the per-region barrier cost is paid per token and grows with the thread count
  (37.6 → 29.1 → 21.5 t/s at 4/8/16 threads), while prompt processing amortises it over 512 tokens
  of work. ggml's own `std::thread` pool is persistent and does not pay it. `vcomp140.dll` leaves
  the dependency list as a side effect. **Linux keeps OpenMP deliberately** — measured in the
  project's own manylinux image, libgomp does not show the effect (72.8 → 77.3 t/s, inside the
  spread), so the Windows result must not be generalised there.
- **`cpu-linux-x86-64` and `cpu-linux-aarch64` ship CPU backend variants** -- upstream's
  `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`, the way its own release binaries are built: one
  `libggml-cpu-<level>.so` per instruction-set level (x86-64: baseline, SSE4.2, AVX, AVX2, AVX-512,
  AVX-VNNI, AMX -- 14 modules; aarch64: `armv8.0_1` to `armv9.2_2` with dotprod, fp16, SVE, i8mm, SVE2,
  SME -- 8) next to `libjllama.so`, `libggml.so`, `libggml-base.so` and `libggml-rpc.so`, of which ggml
  loads the best for the running CPU at start-up. Before, each jar was one library for one level
  (x86-64: the Haswell/AVX2 baseline; aarch64: plain ARMv8): a CPU without AVX2 crashed the JVM with
  SIGILL, and AVX-512/VNNI/AMX stayed unused. Measured with upstream's binaries, the plain x86-64
  module alone is 10x slower at prompt processing than the AVX2 one, and the 14 modules cost ~20 MB
  per jar. `LlamaLoader` extracts the modules next to the library without loading them (a new
  `jllama-files.txt` per backend directory), into a directory keyed by the library's build so that two
  JVMs running different jllama versions never share one; the RPC entry points are resolved through
  ggml's proc-address table in every build. Build option `-DJLLAMA_CPU_VARIANTS=ON`; the other
  platforms still ship the single static library.
- **The Linux glibc floors meet at 2.28 (manylinux_2_28).** `cpu-linux-x86-64` rises from 2.17
  (manylinux2014, whose GCC 10 cannot build the AVX-VNNI/AMX variants) to 2.28 -- RHEL 8, Ubuntu 20.04,
  Debian 10 and later; `cpu-linux-aarch64` falls from ~2.39 (the native Ubuntu 24.04 build) to 2.28,
  built in the pypa manylinux_2_28 aarch64 image on the arm64 runner, with the C++ tests moved to a job
  of their own (`test-cpp-linux-aarch64`). `verify-native-deps.py` now checks every library of a
  natives directory, allows a sibling as a dependency only through run path `$ORIGIN`, and holds the
  manylinux builds (both CPU jars and CUDA) to `GLIBC_2.28` by reading each library's version-needs
  table. The `dockcross-manylinux2014-x64` wrapper is gone.
- **The macOS library now names macOS 15.0 as its minimum, explicitly.** Nothing set a deployment
  target before, so `libjllama.dylib` required whatever macOS the build runner had; moving a build
  job to a newer image would have dropped the older release unnoticed. `CMAKE_OSX_DEPLOYMENT_TARGET`
  is now `15.0`, every macOS build logs the dylib's `minos`, and the macOS smoke fails a shipped
  library above 15.0.
- **CI: the host-native macOS Metal build and its Java tests moved from `macos-14` to `macos-26`.**
  GitHub retires the `macos-14` image by 2026-11-02 and fails every `macos-14` job during its October
  brownouts. The label is pinned rather than `macos-latest`, and it is not `macos-15`, which the
  shipped Metal build and the no-Metal build already run on.
- **CI: every hunk of `llama/patches/*.patch` is checked as text** (`.github/check-patches.py`,
  `code-style` job): the header must declare exactly the lines the body carries. `git apply --check`
  cannot see a header that undercounts -- it skips the surplus lines and writes a truncated file -- which
  is how two comment lines added to `prefetch.h` in patch `0017` cut the header off inside its function
  and reddened 24 jobs before the count was corrected. The check needs no llama.cpp source and runs in
  the first minutes of a run; the historic broken patch text is its pinned negative case.
- **Upgraded the pinned llama.cpp from b11476 to b11512**, in 7 reviewed steps, each ending at a tag.
  No carried patch needed a refresh, and every drop-check still finds its defect. The chat parser now
  receives the generated tokens next to the text (#29876, `common_chat_input`) -- a change in the C++
  tests here, not on the wire: request fields, their bounds and the response keys are identical across
  the range. Two things a user can notice: the embedded server speaks cpp-httplib 0.60.1 (b11505), and
  a slot state file written from b11512 on carries the context checkpoints (#26004) in an appendix an
  older file simply lacks, so files from earlier releases still restore. The new `--moe-cache-mib` is
  under *Added*. Per-step record: `docs/history/llama-cpp-breaking-changes.md`.
- **Upgraded the pinned llama.cpp from b11320 to b11476**, in 25 reviewed steps, each ending at a tag.
  Every carried patch that broke was traced to the one upstream commit that broke it, and the step
  containing that commit ends at the first tag after it: `0007` at #29818 (b11361) and #29895 (b11401), `0014` at #29895, `0008`
  at #29987 (the commit just before b11429), `0015` at #26610 (b11450). Each refresh moved context only,
  and every patch is still needed. The new
  upstream features this binding now exposes are listed under *Added*; the build follows upstream's
  CUDA CCCL pin (v3.4.3) and OpenVINO 2026.4.1. Per-step record:
  `docs/history/llama-cpp-breaking-changes.md`.
- **RPC protocol 8** (llama.cpp b11450, #26610): `RPC_PROTO_MAJOR_VERSION` 7 → 8. An `RpcServer` or
  `--rpc` client of this release talks only to RPC peers of the same protocol -- upgrade the
  `rpc-server`s and every JVM using `RpcServer` together.
- **Slot state files from earlier releases no longer restore** (llama.cpp b11411, #28498): upstream
  now stores the exact KV-cache rotation in a state file and rejects one restored under a mismatched
  rotation, which bumps `LLAMA_SESSION_VERSION` 10 → 11 and `LLAMA_STATE_SEQ_VERSION` 3 → 4. A file
  written by `LlamaModel.saveSlot` (or the server's `/slots/{id}?action=save`) with an earlier jar is
  rejected by `restoreSlot` with upstream's generic "invalid slot save file" message; regenerate it.
  `Session` snapshots taken and restored within one process are not affected.
- **`ProcessRunner` rewritten on `ProcessBuilder`** (the helper `OSInfo` runs `uname` with): the timeout
  is now real -- a command that does not end in time is killed and reported as an `IOException`, where
  the old timeout overload ignored the result of `waitFor` and then blocked reading the output -- and the
  plain call waits at most 10 s instead of forever. `ProcessRunnerTest` pins both.
- **`models/README.md` describes the current setup**: nothing downloads models automatically on a local
  checkout; the list is `.github/models.csv`, which CI's `download-models` job and the tests' defaults
  share.
- **The model-backed tests default to CI's model set** (`.github/models.csv`): the vision, TTS, nomic
  and trainer tests read `models/<file>` unless a `-Dnet.ladenthin.llama.*` property names another file,
  so a model downloaded into `models/` needs no property (new: `net.ladenthin.llama.train.model`
  defaults to `models/stories260K.gguf`). `TestConstantsTest` asserts the defaults and the list are the
  same set.
- **Build checks are a tested Python library** (`.github/buildcheck/`) and check more: that `package`
  waits for every natives build, that the smoke sets derived from `natives.csv` are each
  launched and every classifier documented in the README (`package-smoke-sets.sh` asks the check
  for the sets instead of hard-coding them), CMake's backend names, the dependency allowlists, and that
  every job gates both publish jobs unless `.github/release-gate-exemptions.txt` says why (`vmlens`
  gated nothing; it now gates). The Android AAR libraries are held to the same allowlist and 16 KB
  alignment check as every natives jar.
- **Files shared with the sibling repositories are checksummed** in `.github/shared-files.sha256` and
  checked by a `shared-files` job (crash-log printing, the signing-key preflight and the build-check
  library are now such shared scripts). The workflow lost ~760 lines to two composite actions
  (`restore-models`, `install-sccache-windows`) and `print-host-info.sh`; `validate-models.bat` is gone
  (Windows runs the bash script).
- **Workflow jobs kept identical across the repositories are checked too**: a
  `.github/shared-files.sha256` entry `.github/workflows/publish.yml#<job>` hashes one job (`startgate`,
  `shared-files`, `verify-signing-key`, `check-snapshot`, `check-tag`, and where present
  `verify-signing-key-gradle`, `github-snapshot`, `github-release`).
- **The dockcross wrappers are attributed to dockcross** in `REUSE.toml` (MIT, its copyright holders):
  they are the generated output of dockcross's wrapper template, not this project's code.
- **Unused dockcross wrappers removed**: `dockcross-linux-arm64-lts` (Linux aarch64 builds natively on
  `ubuntu-24.04-arm` with GCC 14) and `dockcross-android-arm` (32-bit Android was never built in CI
  and is not published). With them went the `stdc++fs` link for GCC < 9 in `llama/CMakeLists.txt`:
  llama.cpp needs GCC >= 12 since b9789, so no compiler that could use it builds the project.
- **`.github/dockcross/update.sh` removed**: it regenerated wrappers that no longer exist
  (`manylinux2014-x86`, `android-x86`) from unpinned images, while the wrappers in use are pinned to
  a dockcross tag. A wrapper is regenerated with `docker run --rm dockcross/<image>:<tag> > dockcross-<image>`
  (the command each wrapper prints at its end).
- **`LlamaModel`, `CliParameters` and `ModelFlag` rewritten where they still held upstream text**: the
  class and method documentation of `LlamaModel` now describes the current API (chat, structured output,
  embeddings, reranking, `Session`) instead of the original four-item list, its code examples use
  `ChatMessage`, and `rerank`/`decode` were rewritten; `CliParameters` builds argv and `toString` from one
  `arguments()` method and keeps the options in a `LinkedHashMap`, so argv comes out in the order the
  options were set rather than in hash order; `ModelFlag`'s eight inherited one-line descriptions say
  what the flag actually does. What remains in common with upstream is the public API and the JNI
  declarations, so the upstream copyright line went from these three files. `LICENSE` names the current
  holder first.
- **`.clang-format` is a short file of our own, `.clang-tidy` llama.cpp's current one**: the style is
  `BasedOnStyle: LLVM` plus the four options that differ (column limit 120, indent 4, attributes on the
  same line, no include sorting) instead of a 230-line `--dump-config`; clang-format 23.1.1 resolves it to
  the same configuration and leaves all 27 C++ files unchanged. `.clang-tidy`, an older copy of
  llama.cpp's, is now its current version and attributed to the ggml authors.
- **The four examples are rewritten for the current API and run in CI**: `MainExample` (blocking
  completion with token counts and speed, then streaming), `ChatExample` (a multi-turn console chat on
  `Session`, streamed), `GrammarExample` (a GBNF grammar, and a JSON schema bound to a Java object with
  `completeAsJson`) and `InfillExample` (fill-in-the-middle). Each defaults to a model of
  `.github/models.csv` and takes another GGUF as its argument; `ExamplesTest` runs all four on every Java
  test job, so an example can no longer go stale unnoticed (the old `ChatExample` was `@Disabled`).
- **Copyright lines checked against the upstream code that is actually left**: the upstream author's
  `SPDX-FileCopyrightText` line had been stamped onto every file when REUSE was introduced. Each file
  was compared with the upstream source tree at its last upstream commit (`49be664`, token sequences,
  robust to reformatting and moves); 80 files whose upstream share is nil or only generic boilerplate
  (enum and value-class skeletons, separator comments, API calls every example makes, Maven/`.gitignore`
  templates) no longer carry it. It stays on the 20 files that still hold upstream code or text (the
  JNI layer, `LlamaModel`, `LlamaLoader`, `ModelParameters`, the examples, the clang configs, ...),
  on `README.md`, `models/README.md` and `llama/CMakeLists.txt`, and in `LICENSE`.
- **More shared files, and files identical up to the repository name**: a shared-files entry ending
  in `?repo` is hashed with the repository's name replaced by `{repo}`. Added: `.editorconfig`,
  `.gitattributes` (now with `*.gguf binary` everywhere), `FUNDING.yml`, `CODEOWNERS`, the license texts,
  `SUPPORT.md`, `ISSUE_TEMPLATE/config.yml` and further files listed in `.github/shared-files.sha256`;
  the signing self-test now runs on Gradle 9.8.0 in all four repositories.
  The skip-flag list of the core build (9 jobs) and the CPU-AAR staging (3 jobs) are composite actions
  now (`build-core`, `publish-cpu-aar-local`).
- **The JDK is named once, in `.java-version`**: every workflow reads it through setup-java's
  `java-version-file` (the `JAVA_VERSION` env and the literal `21`s are gone); `.java-version` and
  `codeql.yml` are now byte-identical in all four sibling repositories and in the shared-files manifest.
- **CI files are licensed `MIT OR Apache-2.0`**: every own `.github` file now has the same license
  header in all four sibling repositories, so the shared ones are byte-identical. The upstream
  copyright line stamped onto the redesigned CI files was dropped. `CODE_OF_CONDUCT.md`, `claude.yml`,
  `claude-code-review.yml`, `scorecard.yml`, `reuse.yml`, `osv-scanner.yml` and `dependabot.yml` joined
  the shared-files manifest.
- **Workflow run scripts are parsed in the `shared-files` job**: `check-run-scripts.py` runs `bash -n`
  over every `run:` script of the workflows and composite actions that runs in bash (shell decided as
  the runner does), so a broken script fails within minutes instead of in the job that runs it.
- **Maven versions are compared with the sibling repositories**: `check-versions.py` (in the
  `shared-files` job) warns where a dependency or plugin -- incl. annotation-processor paths and the
  Spotless formatter version -- is used in another version than in a sibling's default branch.
- **Fewer copies of the same job in the workflow.** The four natives smoke jobs are one `smoke-natives`
  matrix (its rows checked against the targets `natives.csv` derives; the per-platform artifacts are
  `llama-smoke-<target>`), the macOS and Windows Java test jobs call one reusable workflow
  (`.github/workflows/java-tests.yml`), and the JDK version is `.java-version`, read by every
  `setup-java` step. Check names of those jobs changed (`<name> / Java tests`, `Smoke test natives
  jars (<target>)`); no required status check referred to them.
- **Dependency updates in the published side artifacts**: `llama-langchain4j` builds against
  `langchain4j-core` 1.21.0; `llama-atmosphere-agent` against Atmosphere 4.0.72, Jetty 12.1.14 and JLine
  4.4.7 (none of the files the agent's carried JLine fixes touch changed between 4.4.6 and 4.4.7).

### Removed
- **The fat jars** -- `llama-<v>-jar-with-dependencies.jar`, the four
  `llama-<v>-all-<os>-<arch>-jar-with-dependencies.jar` and
  `llama-atmosphere-agent-<v>-jar-with-dependencies.jar` (with its manifest `Class-Path` into the core
  fat jars and the agent's `assembly` profile / `src/assembly/agent-jar.xml`), their `.sha256` /
  `.asc`, the `assembly` profile of the core, `package-fatjars.sh`, `sign-fatjars.sh`,
  `check-natives.py fatjar-targets` and the `smoke-fatjar` jobs (now `smoke-natives`,
  `smoke-natives-macos`, `smoke-agent-linux` on the smoke sets). See Changed.
- **32-bit Windows**: the `cpu-windows-x86` and `msvc-windows-x86` natives jars and their build jobs
  (`build-windows-x86`, `build-windows-x86-msvc`), untested by anyone; `llama-platform` no longer
  names `cpu-windows-x86`, and the `examples/jbang/Chat.java` `//DEPS` line for it is gone. The
  32-bit x86 compatibility shim (`compat/ggml_x86_compat.c`) went with it.
- **`msvc-windows-x86-64`** as a natives jar (the build and its tests stay, see Changed) and
  **`sycl-fp32-linux-x86-64`** (fp32 accumulation; `sycl-fp16-linux-x86-64` is `sycl-linux-x86-64`
  now).
- **The backend priority order** (`LlamaLoader.BACKEND_PRIORITY`): there is nothing to order any more
  -- the one library jar of the platform is loaded, every GPU module jar on the classpath is put
  next to it, and ggml, not the loader, decides which backends come up. `net.ladenthin.llama.backend`
  changed meaning accordingly (see Added).

### Fixed
- **`OpenAiCompatServer`: what every surface shares.**
  - **`--api-key` accepts Anthropic clients.** Only `Authorization: Bearer <key>` with a capital
    `Bearer` passed, so every Anthropic SDK -- which sends `x-api-key` -- got `401` on `/v1/messages`
    once a key was set; `x-api-key` is accepted now and the scheme compared case-insensitively
    (RFC 7235), both still in constant time.
  - **The CORS preflight allows the headers the browser asks for.** With `corsAllowOrigin` set it
    always answered `Access-Control-Allow-Headers: Content-Type, Authorization`, so a browser dropped
    the requests of the Anthropic SDK (`x-api-key`, `anthropic-version`, `x-stainless-*`). It mirrors
    `Access-Control-Request-Headers` now (the origin stays the gate), and without one allows
    `Content-Type, Authorization, x-api-key, anthropic-version`.
  - **A route answers its own path only.** `com.sun.net.httpserver` dispatches by longest prefix, so
    `POST /v1/messages/count_tokens` (Claude Code counts tokens there) ran a full generation and
    `/v1/modelsXYZ` listed the models; every other path is `404` now, as upstream, and `/v1/reranking`
    is registered as the alias of `/v1/rerank` it only was by prefix.
  - **The advertised context matches what a request can use** (`maxInputTokens` / `maxOutputTokens`,
    the context length of `/api/show` and `/props`, which Copilot & co. size their prompts by). The
    input budget was `-c` minus 2048 whatever the output budget was (`-c 2048` advertised 1 input
    token); with `--parallel N` the whole `-c` was advertised although each slot holds `-c / N`; and
    without `-c` the fixed defaults 8192 + 2048 were advertised whatever the model loaded with (10240
    for slots of 2048, measured). The server now advertises the context of one slot of the loaded
    model, else `-c` divided by an explicit `--parallel`, and splits it between input and output.
  - **`/props` reports the slot count** of the loaded model (else `--parallel`, else 1) as
    `total_slots`, where it was always 1.
  - **`/v1/models` carries `created`**, one of the four required fields of OpenAI's model object;
    strictly typed clients (Rust's `async-openai`, schema validators) could not read the list.
  - **A heartbeat no longer swallows the opening event of a stream.** The heartbeat timer was
    scheduled before the opening of `/v1/messages` and `/v1/responses` was set, so a heartbeat that
    fired first started the stream without `message_start` / `response.created`, which Anthropic SDKs
    reject. The opening is now always the first event.
- **`OpenAiCompatServer`: the chat request mapping**, which every chat request goes through -- OpenAI
  directly and the Ollama, Anthropic and Responses routes after their translation:
  - **The reasoning controls reach the model.** `chat_template_kwargs` (Qwen3's `enable_thinking`),
    `reasoning_effort` (`"none"` switches thinking off through `enable_thinking: false`, another value
    becomes the template variable `reasoning_effort`) and `reasoning_budget_tokens` /
    `thinking_budget_tokens` were dropped; they follow the rules of upstream's chat parser now.
  - **llama.cpp's own sampling fields are forwarded**: `min_p`, `typical_p`, `repeat_penalty`,
    `repeat_last_n`, `top_n_sigma`, `xtc_*`, `dynatemp_*`, `min_keep`, `n_keep`, `ignore_eos`,
    `grammar` and `json_schema` (unless a `grammar` is given), and `n_predict`, which wins over
    `max_tokens` / `max_completion_tokens` as upstream; so are `logit_bias` and `n_probs`, and
    `logprobs: true` becomes `n_probs` (`top_logprobs`, else 20, as upstream).
  - **`id_slot: -1` means any slot**, upstream's default, instead of a `400`; and `max_tokens: 0` is
    passed on as upstream reads it instead of meaning "no limit".
  - **A `tool_choice` naming a function** (`{"type": "function", ...}`, also the Responses form) became
    `auto`, so the model could answer without calling it; it is `required` now, the closest value
    llama.cpp knows.
- **Ollama routes (`/api/*`).**
  - **Images reach the model.** `/api/chat` ignored a message's `images`, although `/api/show`
    advertises `vision` with `--mmproj`; they become `image_url` parts (the MIME type from the magic
    bytes).
  - **The second turn of a conversation with tools works.** Ollama sends earlier tool calls without
    `type`, which llama.cpp's parser requires, so every request carrying one was refused with `400`;
    `"function"` is set now. A tool result's `tool_name` becomes the `name` llama.cpp reads, and a
    tool call without arguments is answered with `"arguments": {}` instead of `null`.
  - **`think` works**: `false` switches thinking off, a level string (`"low"`, ...) is passed as
    `reasoning_effort`. The thinking itself is answered as `message.thinking`, blocking and per stream
    line, and `thinking` in the history goes back to the model as `reasoning_content`; both were lost.
  - **The sampling options are forwarded on all three routes** (`/api/chat`, `/api/generate`, and
    fill-in-the-middle with `suffix`, which took only `temperature` and `num_predict`):
    `temperature`, `top_p`, `top_k`, `seed`, `stop`, `repeat_penalty`, `repeat_last_n`, `min_p`,
    `typical_p`, `presence_penalty`, `frequency_penalty` and `num_keep` (as `n_keep`). Ollama's
    `num_predict: -2` ("fill the context") is passed as `-1` instead of being refused, and
    `repeat_last_n: -1` is left to the server default.
  - **`done_reason` and the token counts are right.** `done_reason` is `length` for an answer cut off
    by `num_predict` and `stop` otherwise on all four ways (the blocking chat passed OpenAI's
    `tool_calls` through, the others always said `stop`); the streamed `/api/chat` done line and both
    `/api/generate` forms report `prompt_eval_count` / `eval_count`.
  - **`/api/version` reports `0.9.0`.** VS Code Copilot's Ollama provider refuses a server below
    0.6.4, and the route said `0.1.0`, so the whole Copilot path failed at that check.
  - **`/api/show` names the context length where Continue reads it**: `parameters` carries
    `num_ctx <n>`.
- **Anthropic Messages (`/v1/messages`).**
  - **Images reach the model**: `image` blocks of a message (`base64` or `url` source) and of a
    `tool_result` become `image_url` parts; both were dropped.
  - **Several text blocks of one message are joined with a newline**, as upstream does for a template
    that takes text only; `first` + `second` became `firstsecond`.
  - **Thinking works in both directions.** `thinking.budget_tokens` becomes the reasoning budget
    (upstream's default 10000 without one), so a thinking block can no longer use up the whole answer;
    the answer carries the reasoning as a `thinking` block first, and streams it as `thinking_delta`s
    closed with a `signature_delta`; `thinking` blocks of earlier turns go back to the model as
    `reasoning_content`.
  - **A request without `max_tokens` is capped at 4096**, as upstream, instead of running to the end
    of the context, and `chat_template_kwargs` is passed on.
  - **`tool_choice: {"type": "none"}` forbids tool calls**; it ran as `auto`.
  - **Tool calls without arguments**: `tool_use.input` is `{}` instead of `null`, and a `tool_use`
    without `input` in the history is sent as `"{}"` -- `""` or `"null"` made llama.cpp refuse the whole
    follow-up request.
  - **Claude Code's prompt cache hits.** Its system prompt starts with a billing header whose `cch`
    stamp changes on every request, so no prefix ever matched and every turn re-evaluated the whole
    prompt; the stamp is normalized as upstream does.
  - **A server error is an `api_error`**, a rejected request an `invalid_request_error`; every error
    was called `invalid_request_error`, so a client deciding by the type whether to retry took a
    server fault for its own.
  - **The stop word is fetched alone**: the translated request asks for
    `response_fields: ["stopping_word"]` with `verbose`, instead of upstream's whole debug object with
    the rendered prompt.
- **OpenAI Responses (`/v1/responses`).**
  - **Images and tool output parts reach the model**: `input_image` parts of a message and an
    array-valued `function_call_output` (`input_text` / `input_image`) were flattened to their text or
    to nothing.
  - **The history is rebuilt the way upstream does**: text parts of one item are joined with a
    newline; `reasoning` items become the `reasoning_content` of the assistant turn, where they were
    turned into **user** messages; consecutive `function_call` items and a following assistant
    `message` join the preceding assistant turn instead of opening one each; a `function_call`
    without `arguments` is sent as `"{}"`; and a `refusal` part is kept as text instead of being
    dropped.
  - **Reasoning works in both directions**: `reasoning.effort` is passed on (`"none"` switches
    thinking off), and the answer carries the reasoning as a `reasoning` item first, streamed as
    `response.reasoning_text.delta`s.
  - **The request fields the chat path reads are passed on** -- `top_k`, `seed`, `stop`, the
    penalties, `chat_template_kwargs`, `response_format`, the cache fields and llama.cpp's sampling
    fields; only `temperature`, `top_p` and `max_output_tokens` got through. **`text.format`
    (structured output) becomes `response_format`**, so a JSON schema constrains the answer.
  - **What cannot be served is refused with `400`** instead of answered as if it were absent:
    `previous_response_id` (there is no response store; send the earlier turns in `input`) and
    `input_file` parts, both with upstream's message.
  - **A streamed failure ends with `response.failed`** (with `error.code` / `error.message`) before
    the `error` event; Responses clients such as Codex read the message only from there and reported
    "stream closed before response.completed".
  - **`created_at` is the current time**, not `0`, and a response that only calls tools has no empty
    `message` item in front of the `function_call` items, as upstream and the streamed answer.
- **Reranking.** `/rerank` accepts Hugging Face text-embeddings-inference requests (`texts` instead of
  `documents`) and answers them in TEI's form, as upstream; `top_n: 0` returns no result instead of
  all, a negative `top_n` is `400`, and the OpenAI-shaped answer carries `usage` (the native rerank
  result passes `tokens_evaluated` on). `LlamaModel.rerank(String, String...)` returns its map in the
  order of the documents instead of hash order.
- **Java API.**
  - **`InferenceParameters.withMessages(List<ChatMessage>)` keeps `tool_calls` and `tool_call_id`**,
    which only `ChatRequest` added, so a tool round built through it sent an assistant turn without
    its calls and a result without its id. Empty tool-call arguments are written as `"{}"`, which
    llama.cpp parses (`""` failed the whole follow-up request with `400`).
  - **Five "`null` clears" withers clear.** `withPrompt`, `withInputPrefix`, `withInputSuffix`,
    `withGrammar` and `withToolChoice` returned the request unchanged for `null`, so
    `withGrammar(g).withGrammar(null)` still sent the grammar.
  - **JSON values are taken as given or refused, never cut short.** A `ToolDefinition` schema with
    trailing content (`{"type":"object"} {"required":["x"]}`) was sent as its first value, and a
    `chat_template_kwargs` value holding two JSON values as the first of them; both now take the
    documented fallback to a string, as does an empty schema or kwarg value, which became `null`.
  - **`LlamaIterator` stays finished after `cancel()` from another thread**; a result that arrived
    afterwards revived it, and the next `next()` failed with "Task not found".
  - **`TextToSpeech.close()` waits for a running `synthesize()`** instead of freeing the engine under
    it (a use-after-free, i.e. a JVM crash); syntheses still run side by side.
  - **`NativeServer` start, `isRunning()` and `close()` are serialized**: two `close()` calls at once
    (a shutdown hook and try-with-resources) could stop and free the native server twice, and
    `isRunning()` could read a freed handle.
  - **A failed `RpcServer.start` stops the server.** A stop that arrived before the native server
    bound its socket was lost, so after a start timeout or an interrupt the server kept listening on a
    daemon thread although `start` had thrown; the stop is now repeated until the thread ends.
  - **`RouterClient`**: `awaitModelLoaded` with `Long.MAX_VALUE` overflowed into an immediate timeout,
    and with `0` reported a timeout without asking; it polls at least once now, counts a `sleeping`
    model as loaded (as the router itself does), and finds a model by its alias; an IPv6 literal host
    (`"::1"`) is bracketed in the URL instead of failing with `MalformedURLException`.
  - **`LlamaTrainer.finetune` accepts any Unicode**: a character outside the Basic Multilingual Plane
    (an emoji) in the training text or a path failed the whole run with "invalid training config".
  - **`GgufInspector.read` rejects nested arrays with an `IOException`**, as llama.cpp's reader does; a
    crafted file ended it with a `StackOverflowError`.
  - **`Session.fork` refuses a negative slot id before touching the model**, with the constructor's
    `IllegalArgumentException`, instead of writing the slot file and failing in `restoreSlot`.
  - **An offline load (`--offline`) of a path the file system cannot represent** throws the documented
    `ModelUnavailableException` instead of an `InvalidPathException`; and `ChatStreamChunkParser` treats
    an empty envelope as the end of the stream, as its Javadoc says.
- **`LlamaLoader` deletes only its own files from the temp directory.** The clean-up of the pre-6.0.0
  layout deleted, recursively, every entry of `java.io.tmpdir` (or `net.ladenthin.llama.tmpdir`) whose
  name starts with `jllama`, `llama` or `ggml` -- a `/tmp/llama.cpp` checkout, a `/tmp/ggml-models/`
  folder, a downloaded `/tmp/llama-2-7b.Q4_K_M.gguf`. It removes regular files with a library suffix
  (`.so`, `.dll`, `.dylib`, `.jnilib`, `.metal`) now, nothing else.
- **`llama-ffm`.** `FfmTextToSpeech.close()` waits for a running `synthesize()` and
  `FfmNativeServer.close()` for a running `isRunning()`, as their JNI twins do, instead of freeing the
  native object under them; and an empty, non-null string array returned by the library is freed.
- **`llama-langchain4j`.** The blocking `JllamaChatModel` carries the reasoning as
  `AiMessage.thinking()` like the streaming model; the streamed response carries the token usage (the
  request now asks for the usage chunk); a streamed tool call no longer fails at its first fragment
  without arguments or without a name (langchain4j's `PartialToolCall` refuses both, the fragment
  still joins the assembled call); and several `TextContent`s of a user message are joined with a
  newline instead of being glued together.
- **A scaled LoRA adapter or control vector loads at all.** `ModelParameters.addLoraScaledAdapter` and
  `addControlVectorScaled` rendered their value as `FNAME,SCALE`, while llama.cpp has read these options
  as `FNAME:SCALE,...` since b7448 (#18056): the comma is the list separator, the colon the field
  separator. Every value they emitted was rejected by the argv parser (`invalid stof argument`), so the
  model did not load and the caller saw only `Failed to parse model parameters`. They now render
  `FNAME:SCALE`. One upstream limit comes with it and is refused loudly rather than silently: the parser
  splits each item on *every* colon and demands exactly two parts, so an absolute Windows path cannot be
  expressed — `addLoraScaledAdapter` throws `IllegalArgumentException` naming the cause and the two ways
  out (a relative path, or `addLoraAdapter` plus `LlamaModel.setLoraAdapter` after the load).
- **The repeatable adapter options accumulate instead of replacing.** `addLoraAdapter`,
  `addLoraScaledAdapter`, `addControlVector` and `addControlVectorScaled` all say "can be repeated" and
  all four silently dropped everything but the last call, because the parameter set is keyed by the wire
  string. They now append to one comma-separated value, as llama.cpp reads these options, and a value
  carrying a comma or a quote is written as a quoted CSV field.
- **`/v1/models` and the answers name the model the same without `--model-id` too.** The id was passed on
  as llama.cpp's `--alias` only when the flag was given; without it `/v1/models` reported the model's file
  name while every answer and stream chunk reported the full model path, because upstream's fallback for
  the model name resolves to the path for a local file. `OpenAiCompatServer` now always sets the alias
  from the advertised id, so the two agree and no local path reaches an answer. An embedding caller that
  loads its own `LlamaModel` still has to set `ModelParameters.setAlias(…)` itself — the server cannot,
  the model is already loaded; `OpenAiServerConfig.getModelId()` says so.
- **A streamed `/v1/completions` is upstream's stream.** `OpenAiCompatServer` built the chunks of a
  streamed completion itself from the Java generator: no `usage` on the last chunk, no probabilities
  (a request for them was refused with a pointer to `stream: false`), and a prompt given as a token
  array refused
  while the same request without `stream` was accepted. The body now goes to llama.cpp verbatim, as
  the blocking request does, and the chunks are upstream's own `text_completion` events. New
  `LlamaModel.streamCompletionsOai(String, Consumer<String>)`, the C function
  `jllama_completions_oai_stream_request` (received with `jllama_chat_completions_stream_receive`)
  and `FfmLlamaModel.completionsOaiStream`.
- **OpenAI answers carry an id and the model name.** Every native OpenAI answer and stream chunk had
  `"id": ""` and `"model": ""`; upstream gives each completion a `chatcmpl-` id (one per stream) and
  names the model it runs (the alias, else the model's name). They do now. New
  `ModelParameters.setAlias(String)` (`--alias`), and `OpenAiCompatServer --model-id` passes the id on
  as the alias, so the answers name the model `/v1/models` lists.
- **Several prompts in one request are refused instead of answered with the first.** A task carries
  one prompt, and `/v1/completions` (and `handleCompletions`, the raw stream) took the first of
  `"prompt": ["a", "b"]` and dropped the rest without a word -- two prompts, one choice. Upstream
  answers each; here the request is now an invalid request (`400`) naming the count. A token array is
  one prompt and stays accepted. **The same for `"n"` above 1:** llama.cpp's schema accepts up to the
  slot count and answers the further completions in child tasks its server creates, which this
  library's engine never did, so `n: 2` came back as one choice on every native path -- the C API,
  `llama-ffm`, `LlamaModel.handle*`, both `/v1/completions` routes and the chat paths; the server's
  request mapper had dropped `n` on the chat routes as well. It is an invalid request now, checked in
  one place for the blocking, streamed and chat paths.
- **`/v1/messages` reports the matched stop word as `stop_sequence`**, blocking and streamed, as
  upstream does (`stop_reason` stays `end_turn`); it was always `null`. The OpenAI answer the route
  translates from does not carry the word, so the translated request asks for llama.cpp's `verbose`
  answer and reads `__verbose.stopping_word`, which never reaches the client. New
  `InferenceParameters.withVerbose(boolean)` (`RequestField.VERBOSE`), and the OpenAI mapper forwards
  a client's `verbose`.
- **A streamed request that fails before its first chunk is answered with the error status.**
  `OpenAiCompatServer` sent `200` and the stream's headers before the engine ran, so a request
  llama.cpp refused at once arrived as `200` with an error event -- on all five streaming routes
  (`/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/v1/messages`, `/v1/responses`). Upstream's
  server reads the first result before it starts a stream; this one now does the same: the response
  starts with the first write (a chunk, the protocol's opening events or a heartbeat), and a failure
  before it is `400` or `500` with the route's own error body, as without `stream`. A failure after the
  stream started is still an event. One visible effect: the Atmosphere agent now reports such a failure
  instead of ending the turn with an empty answer.
- **Error messages with a 4-byte character reach Java intact.** The JNI shells handed messages to
  `ThrowNew` and `NewStringUTF`, which read Modified UTF-8; a message quoting a path or request text
  with a supplementary-plane character (an emoji) was mangled on HotSpot -- a trainer message for
  `missing-😀.gguf` came back as `missing-`, four wrong characters and `.g` -- and aborts the process under
  Android's CheckJNI. A non-ASCII message now crosses as bytes through `String(byte[], "UTF-8")`
  (`jni_utf8.hpp`), on every path that carries llama.cpp's text: the C API status, the model load,
  the JNI exception boundary, `NativeServer`, `RpcServer` and `LlamaTrainer`. Called while a Java
  exception is already pending (after a failed `GetStringUTFChars`, say), both helpers leave that
  exception in place, as the JNI exception boundary already did, instead of replacing it with a less
  precise one or calling JNI functions the specification forbids in that state, which Android's
  CheckJNI aborts on.
- **A LoRA adapter change no longer answers from the old cache** (`patches/0021`). After
  `setLoraAdapter(s)` -- and after a request with its own adapters -- a slot reused the prompt cache it
  had computed under the previous scales and answered like neither adapter set (measured with
  llama.cpp's own LoRA test pair: a third, unrelated text until the slot was erased). Upstream clears
  the cache when a request names its own adapters but not when the server's adapters change; the patch
  applies the same rule there, which also fixes upstream's `POST /lora-adapters` on the embedded
  server. `setLoraAdapters`' Javadoc no longer tells callers to erase the slots; it names the one cache
  that still does not record adapters, the host-memory prompt cache (`setCacheRamMib(0)` disables it).
- **`ServerMetrics.getCumulativeUsage()` counts the cached prompt tokens.** It returned
  `Usage(processed, predicted)`, so `getPromptTokens()` was the evaluated tokens only and
  `getCachedTokens()` always `0`, against `Usage`'s own contract (prompt tokens include the cached
  ones, which are their subset). The method's Javadoc gave the reason "upstream exposes no cumulative
  cached total", which stopped being true when `getMetrics()` started carrying
  `n_prompt_tokens_cached_total`. Now `getPromptTokens()` is evaluated + cached,
  `getCachedTokens()` the cached ones and `getProcessedPromptTokens()` the evaluated ones -- read the
  same way as the `usage` of a response. A caller who read `getPromptTokens()` as "evaluated" reads
  `getProcessedPromptTokens()` or `getCumulativeProcessedPromptTokens()` now. `getWindowUsage()` is
  unchanged: upstream keeps no cache count for the window.
- **The GitHub `snapshot` pre-release and the versioned releases carry what Maven Central carries,
  and the `snapshot` release no longer accumulates.** The collect steps copied `*.jar` / `*.jar.asc`
  and the bare AARs, so every `.pom` with its `.pom.asc` was missing, `llama-platform` (a pom-only
  module, the artifact consumers are told to depend on) was absent from the release entirely, the
  AARs were attached **unsigned** although Gradle had written their `.asc` next to them, and the
  agent's thin jar was never attached; the signature check globbed `*.jar` only, so the unsigned AARs
  never turned the job red. Now every module's jars and poms (`llama-parent` and `llama-platform`
  included), the AARs with their signatures, poms and sources/javadoc jars, and the agent's thin jar
  are collected in both the snapshot and the release job, the check counts jars, AARs and poms, and
  `github-snapshot` deletes every asset the run did not upload, after the upload and each printed by
  name -- `gh release upload --clobber` never removed anything, so the release held 208 assets, 109 of
  them from 5.0.x and 43 in shapes that no longer exist (`msvc-windows`,
  `all-*-jar-with-dependencies`), plus three SpotBugs plugin jars an old collect step attached in
  2026-05. The prune refuses an empty upload, treats an asset that is already gone as removed and
  fails only after trying every deletion (asset by asset on purpose: the job's first form deleted
  and recreated the release, and failed in every burst of main pushes in May 2026). Versioned
  releases are never pruned; the next snapshot run clears the backlog by itself
  (`workspace/policies/fat-jar-release-assets.md`).
- **A Windows build on an AMX CPU (Sapphire, Emerald or Granite Rapids) crashed the JVM with
  `STATUS_HEAP_CORRUPTION` when a model was unloaded.** ggml's AMX weight buffer -- used by the
  `sapphirerapids` CPU module for Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS weights -- is allocated with
  `_aligned_malloc()` on Windows and was freed with `free()`; `patches/0018` frees it with
  `ggml_aligned_free()`, as the plain CPU buffer does (no change on Linux). Never shipped: the
  5.1.0 Windows jar was a static Haswell-level build that carried no AMX code. Found by the second
  full CI run of the modular natives, whose Windows Java test job landed on an Intel Xeon 6973P-C
  runner; the first run's AMD EPYC 7763 never reached the code (the Windows runners are a CPU
  lottery, CLAUDE.md "Modular natives"). Of the six callers of `ggml_aligned_malloc` upstream this
  was the only mismatched pair. Reproduced deliberately afterwards on a rented Granite Rapids
  machine with one CPU module staged at a time: `sapphirerapids` crashed 6 of 6 runs and
  `cooperlake`, `icelake` and `haswell` none, it reproduces at a **single thread** (so it is no data
  race), and the same matrix with the patch applied is clean in all twelve runs. A configure-time
  check (`llama/cmake/check-aligned-free-pairing.cmake`) now fails the build on **every** platform
  when a ggml source allocates aligned and frees plain, because a machine without AMX cannot observe
  the defect at run time -- which is exactly how it survived.
- **`NativeServer` attach mode no longer leaves a sleep callback into freed memory behind.**
  `llama_server_attach` (`patches/0007`) built a `server_routes` on its own stack frame; its
  constructor registers a sleeping-state callback on the model's task queue, which has no unregister,
  so after `NativeServer.close()` the model's next idle sleep (`--sleep-idle-seconds`) ran the callback
  on a destroyed object. The attached server now serves the model's own `server_routes`
  (`jllama_context::routes`, alive as long as the model); pinned by
  `NativeServerAttachSleepIntegrationTest`. **Behaviour change:** route-level settings therefore come
  from the model's `ModelParameters` -- the endpoint toggles `--metrics` / `--props` / `--slots` (new
  setters, see Added), `--slot-save-path`, the slot count, the sampling defaults -- and an attach
  argument of that kind is parsed and ignored; the attach argv keeps the HTTP side (`--host`, `--port`,
  `--api-key`, `--ssl-*`, `--threads-http`, ...).
- **`GGML_CUDA_GRAPHS_DEFAULT` is repeated in the CPU-variants CMake path.** That path adds ggml before
  llama.cpp and therefore skips the block in which llama.cpp changes two ggml defaults; it repeated
  only `GGML_LLAMAFILE_DEFAULT`. Latent for the shipped CPU jars, but a CUDA backend built as a module
  in this path ran without CUDA graphs and lost two thirds of its token generation (measured at b11538
  on an RTX 3070, `docs/handover/local-agent-report-b11538-windows.md`). The CMake comment now states
  the rule: every `*_DEFAULT` llama.cpp sets before its own `add_subdirectory(ggml)` is repeated.
- **A Windows `JLLAMA_CPU_VARIANTS` build is loadable without the CI merge step.** `CMakeLists.txt`
  writes `jllama-extras.txt` (`ggml-base.dll`, `ggml.dll`, the two libraries `jllama.dll` links) next to
  the library, byte-identical to what `merge-native-artifacts.sh` derives in CI; without it
  `LlamaLoader` extracted `jllama.dll` alone, Windows resolved its `ggml.dll` import from `java.exe`'s
  directory or `PATH`, and every model-backed test of a local variants build failed on the load.
- **`OpenAiCompatServer` answers `400`, not `500`, for a request the native layer rejects** -- an empty
  `input` array for `/v1/embeddings`, a `/v1/completions` body without a `prompt`, a malformed body, a
  message the chat template cannot take, a prompt over the context size -- on every surface (OpenAI,
  Ollama, Anthropic, Responses; as an `invalid_request_error` SSE event once a stream has started).
  Upstream's server has answered these with `400` since b11337 (#29060); the Java transport reported them
  as `server_error`, because the JNI layer threw the same `LlamaException` for a rejected request and a
  failed inference. The server now maps `InvalidRequestException` like the `IllegalArgumentException` of
  its own validation, without matching on message text; a plain `LlamaException` stays `500`.
- **HTTPS works out of every desktop natives jar, and the macOS dylib no longer needs Homebrew's OpenSSL.**
  `jllama` compiled cpp-httplib itself, without SSL, next to the SSL-enabled copy upstream's `llama-common`
  (the download code) links -- two incompatible copies of the same classes in one link, and which one the
  linker took was decided per platform: on Linux `--model-url https://…` and `-hf` threw
  "HTTPS is not supported", and on macOS the dylib came to depend on `/opt/homebrew/opt/openssl@3`
  and failed to load on a Mac without that formula (`verify-native-deps.py` carried the two paths as a
  known defect). Now `jllama` links upstream's `cpp-httplib` target, and BoringSSL is built from source
  and linked statically on Linux x86-64 / aarch64, macOS and Windows (where it already was), as
  upstream's own release builds do; Android, s390x and the Linux SYCL jars stay without SSL
  (`LLAMA_OPENSSL=OFF`, as upstream's `ubuntu-24-sycl` job). The
  embedded server's `--ssl-key-file` / `--ssl-cert-file` work as a result. Certificates are verified
  against the OS store: crypt32 on Windows (the one new import of `jllama.dll`, allowlisted),
  Security.framework on macOS, `/etc/ssl/certs` and `/etc/ssl/cert.pem` on Linux (`SSL_CERT_FILE` /
  `SSL_CERT_DIR` override). The library grows by about 2.5 MB, the code of BoringSSL's two static libraries.
- **`LlamaLoader` prints its two diagnostic lines to stderr**, `[jllama] using native backend '…'` and
  `[jllama] extracted '…'`, where they went to stdout before. A router worker JVM
  (`NativeServer.setWorkerCommand`) printed them onto the router's command pipe, which since llama.cpp b11401
  the router reports as `unexpected output on the command pipe` (harmless, misleading); and the stdout of a
  `java -jar` server is otherwise the upstream server's. The CI smokes read both streams already.
- **`llama/patches/0017`** fixes two defects in the four `_mm_prefetch` calls of ggml's x86
  `quants.c` (the SSSE3-without-AVX branch of `ggml_vec_dot_q4_0_q8_0`, and the only
  `_mm_prefetch` calls in the whole ggml tree), both by routing the address through one new
  `static inline` helper. **(1)** MSVC declares the intrinsic as `void _mm_prefetch(char const *,
  int)` and clang's own casting macro sits behind `#ifndef _MSC_VER`, so a typed pointer is
  `-Wincompatible-pointer-types` — which clang 22 promoted to an error by default
  ([llvm-project #157364](https://github.com/llvm/llvm-project/pull/157364)). Measured on the real
  file at `-msse4.2`: 4 warnings with clang 20.1.8, 4 errors with 22.1.3 and 23.1.3. This is what
  stops `GGML_CPU_ALL_VARIANTS=ON` from building with a current plain clang on Windows, which is
  the only toolchain that produces all 14 CPU variants. **(2)** `&x[ib] + sizeof(block_q4_0)` adds
  `sizeof()` *elements* rather than bytes, so the prefetch targets +324 and +1156 bytes instead of
  +18 and +34 — wrong on every platform, including those where the call compiles. Measured as
  having no effect on throughput, so this half is a correctness fix rather than a performance one.
  Runnable guard: `src/test/cpp/test_prefetch.cpp` (598 → 603 C++ tests). Upstream-submittable and
  not yet filed.
- **CI: snapshot deploy failed with HTTP 401 on Maven 3.10.** The runners moved to Maven 3.10, which
  sends a `<server>`'s credentials only to the origins declared for it; for the id `central` that is
  `https://repo.maven.apache.org`, so the upload to `central.sonatype.com/repository/maven-snapshots/`
  went out without credentials ("Not using credentials of server 'central'"). Every `deploy` step now
  passes `-Dmaven.repository.credentialScope=id`, Maven's own switch back to id-only matching, until a
  setup-java release can write `<repositoryOrigins>` (`mvn-server-repository-origins`, merged upstream
  but not yet released).
- **`SessionConcurrencyTest` timed out on `Java Tests macOS 15 arm64 (Metal)` from llama.cpp b11457 on**
  (green through b11320, red in both runs since). Only this class slowed down -- about 130x, from
  0.45 s to 57-110 s per token, while every other class ran as fast as before. It is the one test
  that loads the 7B model with a 4096-token context onto the GPU without `--fit`, so the weights
  (~2.7 GB) plus the F16 KV cache (~2 GB) over-commit the runner's 4.7 GB Metal device. The context
  is now 2048, which the longest transcript of the class (under 200 tokens) is far from. Which upstream
  change made the over-commit expensive was not bisected.
- **Both CUDA build jobs failed** after the CCCL pin: CCCL calls `include(CTest)`, which turned our
  later `option(BUILD_TESTING ... OFF)` into a no-op, so the GPU-less runners built `jllama_test` and
  its test discovery could not load the CUDA driver. The option is now declared before any subproject.
- **`ToolCallingIntegrationTest#requiredToolCallIsParsedFromStreamingResponse` failed on both Windows
  x86-64 jobs after the b11211 bump** (the Ubuntu run and the blocking twin stayed green). The streamed
  request generated its full 512 tokens without a tool call. The prompt ("Write an example") never asked
  for the tool, so the grammar-constrained answer was a fragile ~90-token call even when it worked, and a
  numerically different CPU path on those runners (most likely upstream's new tiled k-quant matmul,
  which picks its microkernel by ISA) tipped greedy decoding over. The test pins how a tool call is
  parsed and streamed, not whether a 1.5B model infers one, so the user message now asks for the call
  outright; both assertions carry the streamed content, `finish_reason` and chunk count, so a future
  failure says what the model did instead of `but: was ""`.
- **`LlamaModel.setLogger` was silently overridden by every model load, and never saw the server's own
  log lines.** llama.cpp's `common_init()` — run on each load — re-points `llama_log_set()` at its own
  default callback, so a logger set *before* `new LlamaModel(…)` (the natural order) stopped receiving
  anything; and the `srv …` / `slot …` lines (per-request timings, slot state) are written by the server
  macros straight into `common_log`, which `llama_log_set()` never carried, so they went to stderr no
  matter what Java configured. The logger is now a sink on `common_log` itself
  (`patches/0014-common-log-callback-sink.patch`, `common_log_set_callback`): it survives loads, it
  receives every line — the server's and llama/ggml's — and it replaces the console output instead of
  duplicating it (a `setLogFile` file keeps being written). Messages are delivered from llama.cpp's log
  worker thread; replacing or removing the logger flushes what is queued to the previous callback first,
  so `setLogger(format, null)` is a synchronous drain. Behaviour change to know: the verbosity threshold
  now applies before the callback (as on the console), so llama/ggml INFO lines reach the logger only
  from `setLogVerbosity(4)` on. `LlamaModelTest#testLogText/testLogJSON` are re-enabled (they were
  `@Disabled` because of exactly this), `#testLoggerSetBeforeLoadSurvivesTheLoad` pins the ordering, and
  the model-free `LlamaLoggerTest` plus six C++ tests guard the sink on every platform.
- The `setLogger` Javadoc and the README "Logging" section claimed JSON to stdout as the default; the
  default is llama.cpp's text format on stderr. `enableLogPrefix()` / `enableLogTimestamps()` are
  documented as the no-ops they are (`common_init()` forces both on), `setLogFile` as additive.

### Added
- **Distributed inference over llama.cpp RPC**, client and server, in every artifact with no new
  runtime dependency. `ModelParameters.setRpcServers(RpcEndpoint...)` (and `--rpc host:port[,…]` on
  both HTTP servers) offloads layers to RPC servers on other machines; `RpcServer` serves this
  machine's devices (every GPU found, else the CPU), loopback-only unless `startOnNetwork` is used,
  also runnable as `java -cp <jar> net.ladenthin.llama.RpcServer`. `value.RpcEndpoint` validates
  endpoints (IPv4 or host name; the transport has no IPv6). Upstream aborts the whole process on
  every client-side connection problem; the new `patches/0015` turns an unreachable, malformed or
  non-RPC endpoint into a `LlamaException` naming it, makes the server stoppable, and keeps a
  stopped server's registered device from aborting later loads. Because llama.cpp never forgets a
  registered RPC server, a load that does not ask for one gets an explicit device list without it —
  including the multimodal projector's device, and `TextToSpeech` / `LlamaTrainer` loads.
  A server lost in the middle of inference still terminates the process (upstream limitation).
  The served devices can be chosen by name (`RpcServer.startLocal(…, List<String> devices)`,
  `--device CPU`), because a served device that cannot run an operation terminates the server
  (llama.cpp's RPC client reports every operation as supported).
- **`LlamaLoader` no longer loads the native library twice** when the first class to load it is not
  `LlamaModel` (e.g. `TextToSpeech`, `LlamaQuantizer`, `RpcServer`): `JNI_OnLoad` initializes
  `LlamaModel`, whose static block re-entered the loader and ran a second full load — with every
  GPU jar on the classpath that meant re-extracting every backend over the library being loaded.
- **`.github/verify-native-deps.py`** holds every shipped native library to its known runtime
  dependencies (ELF, PE incl. Windows arm64, Mach-O). It found that the macOS dylib has always
  linked Homebrew's `openssl@3` (see `TODO.md`).
- **The published agent is smoked on the published core.** The thin jar on Maven Central carries
  **no core**: its pom resolves `net.ladenthin:llama` and the natives through `llama-platform`. A new
  CI job, `smoke-agent-linux`, puts the agent's jar with its own dependencies (what JBang resolves for
  `jbang net.ladenthin:llama-atmosphere-agent:<v>`) on one classpath with the Linux x86-64 smoke set of
  the core and launches that (bytecode ≤ Java 21, the agent alone must fail for the missing core, `--help`, a one-shot answer and
  a `read_file` round on the cached tool model), and it, the model-free agent job and the model-backed
  agent integration test now gate both publish jobs.
- **`llama-atmosphere-agent`: `--log-verbosity <n>` (default `2`) and `--verbose`** for the in-process
  `--model` mode. llama.cpp's per-request INFO lines go to stderr, the console the streamed answer is
  printed to, and interleaved with it; the agent now loads the model with warnings-and-errors only. A
  `.mvn/jvm.config` pins `-Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8` for the `mvn exec:java`
  JVM, because on Windows `common_init()` switches the console to UTF-8 after the JVM fixed its stdout
  encoding from the old code page (umlauts/emoji in answers rendered as `�`/`?`).
- **`llama-atmosphere-agent/` — a local, offline JVM coding agent** (Claude Code / OpenCode reduced to
  the essentials) that drives [Atmosphere](https://github.com/Atmosphere/atmosphere)'s built-in
  OpenAI-compatible agent runtime **headless** (no Spring Boot, no servlet container) against this
  project's `OpenAiCompatServer`: streaming, the model→tool→model loop, Atmosphere's workspace-confined
  file tools and an opt-in `run_command` tool. Standalone Maven project (not a reactor module, not
  published): `mvn compile exec:java -Dexec.args="--base-url http://127.0.0.1:8080/v1 …"` against a
  running java-llama.cpp / llama-server, or `--model x.gguf` to host the model in-process. Verified two
  ways and wired into CI: model-free wire-contract tests drive the *real* `OpenAiCompatServer` with a
  scripted engine (tool-call deltas by index, parallel calls, four consecutive rounds with full history,
  401 handling, the one known Atmosphere gap on in-stream errors), and a model-backed job runs the loop
  against the Qwen2.5-1.5B tool model. Result: Atmosphere works **unchanged** (verdict A).
- `OpenAiBackend`, `ChunkSink` and `OpenAiCompatServer(OpenAiBackend, OpenAiServerConfig)` are now
  **public** — the inference-engine seam behind the OpenAI-compatible server, previously package-private
  and used only by the core's own tests, so that sibling projects can drive the real HTTP surface
  without a native library or model.

### Changed
- **llama.cpp `b11214` → `b11222`.** Eight upstream commits. Two touch argument parsing:
  **#29518** makes `string_split<T>` throw `invalid value: "…"` for a list element that does not parse
  (only the benchmark options `-npp`/`-ntg`/`-npl` use a numeric split, so nothing a server or `jllama`
  argument reaches changes); and **#29537** registers `--rpc`
  in every build and rejects it at parse time with `RPC not supported in this build` (this project builds
  with `GGML_RPC=OFF`), where before the option did not exist at all. The rest is CUDA/SYCL/OpenCL kernel
  work, a Jinja `dict` builtin and conversion scripts. `patches/0001` and `0006` were refreshed for a
  moved log line in `tools/server/server.cpp`; their content is unchanged.
- **llama.cpp `b11211` → `b11214`.** Three upstream commits, version-only from this project's side:
  a HIP flash-attention kernel choice for CDNA (#28907), a Vulkan argsort fix for Adreno (#29469), and
  **#29516**, which makes `common_sampler_init` *throw* `failed to parse grammar: llguidance is not
  enabled` instead of calling `GGML_ABORT` when a `%llguidance` grammar reaches a build without
  llguidance. That one matters in a JVM: these builds do not enable llguidance, so such a grammar used
  to abort the whole process; it now surfaces as an ordinary request error. All patches apply unchanged.
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

  **The library jar deliberately carries no logging defaults.** A `simplelogger.properties` under
  `src/main/resources` would be published inside the library jar and land on every consumer's
  classpath: slf4j-simple reads whichever file the classloader hands it first, so a consumer with
  their own configuration would get a coin flip. A library must not decide that. (The runnable fat
  jar used to carry one -- INFO, stdout, timestamps -- from `src/main/assembly-resources/`; it went
  with the fat jars, see Removed. A server started with `jbang` or `examples/server/pom.xml` logs
  slf4j-simple's defaults, INFO to stderr, unless `-Dorg.slf4j.simpleLogger.*` says otherwise.)

- **`checker-qual` is `provided` scope and ships to nobody.** 4.x is major 55 and its annotations are
  `@Retention(RUNTIME)`, so a Java 8 JVM throws `UnsupportedClassVersionError` the moment anything
  reflects over an annotated element (Jackson does). A first version pinned the shipped copy to the
  last Java 8 line, 3.55.1, which broke every build -- the Nullness Checker resolves its qualifiers
  through the compile classpath, so processor and qualifiers must share a major version. `provided`
  satisfies both: the 4.x qualifiers where the checker needs them, and nothing of any version in a
  consumer's dependency graph or fat jar (`jar-with-dependencies` takes scope `runtime`).

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
- **llama.cpp `b11080` → `b11211`, in six steps sized by what they change here.** 131 upstream
  commits. Five steps are version-only from this project's side (`b11103`, `b11160`, `b11163`,
  `b11209`, `b11211`); the one incompatible change got a step of its own: **`b11104`** (llama.cpp #28690)
  lets `--host` take a comma-separated list of addresses and removed `server_http_context::thread`
  and `::listening_address`. `patches/0007` still applied cleanly there but named both members, so
  it was refreshed to upstream's new `join()` / `listening_addresses` shape; and
  `NativeServer` gains **`getHosts()`**, with `getHost()` now returning the first address instead
  of the raw comma-separated string. New upstream surface that needs no project change: the
  extended batch API (`llama_batch_ext` + `llama_process`, #24669, additive), `input_image` accepted
  as a Responses-API `function_call_output` and OpenAI's `video_url` as an alias of `input_video`
  on the native server (#22575, #27921), cleanup of K/V and recurrent state after a failed state
  restore (#27530), and new model support (Ling 3.0 VL, Gemma 4 DSpark draft, MiMo-V2.6).
  cpp-httplib moves to 0.58.0. All eight local patches are still required.
- **BREAKING (runtime): the ROCm classifiers are built against ROCm 10.0 (TheRock), not 6.3.**
  AMD now releases ROCm through [TheRock](https://github.com/ROCm/TheRock); both `rocm-*` jobs install
  its wheels the way upstream llama.cpp's own release jobs do, replacing the `repo.radeon.com` 6.3.4 apt
  repo (Linux) and the HIP SDK 26.Q1 installer (Windows). The GPU target lists are now every target
  TheRock builds per OS — a superset of upstream llama.cpp's: Linux adds gfx90c, gfx942/gfx950, RDNA1,
  the rest of RDNA2/RDNA3, RDNA3.5 (gfx1150–1153) and RDNA4 (gfx1200/1201) to the previous eight;
  Windows goes from four RDNA2/3 targets to 23, gfx900 through RDNA4. The extras upstream omits
  (gfx900/gfx906/gfx90c/gfx1153, build-passing only in TheRock) are kept as long as they build
  without patches. Consumers need a ROCm 10 runtime.
- **CUDA 13.3 → 13.4** for both `cuda13-*` classifiers, matching upstream llama.cpp. Linux installs
  `cuda-toolkit-13-4`; Windows assembles the toolkit from NVIDIA's redist archives (upstream's
  component list) because `Jimver/cuda-toolkit` never shipped 13.4. Classifier names are unchanged.
- **llama.cpp `b11069` → `b11080`, and local patch `0010` dropped — upstream fixed the
  enum-to-JSON-boolean trap at its root.** Eleven upstream commits, 1244 KiB, no project-source
  change. The size is one commit that does not concern this project (llama.cpp #29197 rewrites 46
  files under `ggml/src/ggml-hexagon/`; no hexagon classifier is built here); the rest of `ggml/src`
  is additive ARM repack kernels, a Metal fusion-list simplification and a SYCL softmax tweak, and
  `ggml/include` is byte-identical. The one that matters is llama.cpp #28518 ("json: Fixed json enum
  handling"): `common_json_value` gains an `std::is_enum`-gated constructor delegating to the
  underlying type, so an unscoped enum no longer binds to `common_json_value(bool)` and serialises
  as `true`/`false`. That is exactly the defect `patches/0010` cast around in upstream's own
  `get_res_model_info()`, so the patch became a redundant carry and was dropped rather than kept
  (the `0009`/`0011`/`0013` precedent). **Nothing observable changes for consumers** —
  `GET /models` and `GET /v1/models` reported a numeric `vocab_type` with the patch and still do
  without it — but the drop is worth flagging because `0010` *still applied cleanly*: the fail-loud
  applier can only detect "does not apply", never "upstream already fixed this", which is why that
  patch carries a by-hand drop-check on every bump. Its guard was kept and re-pointed: the
  `CommonJsonEnumTrap` pair in `test_json_helpers.cpp` is now the `CommonJsonEnum` trio and pins
  upstream's contract (uncast enum is numeric, an explicit cast is equivalent, a real `bool` is
  still a boolean), so a bump that loses the constructor reds `C++ Tests` everywhere instead of
  shipping a boolean. `jllama.cpp` keeps its own two `"vocab_type"` casts — correct either way.
  Also in range: six existing sampling flags gained environment defaults (#27380 —
  `LLAMA_ARG_TEMPERATURE`, `_TOP_P`, `_MIN_P`, `_REPEAT_PENALTY`, `_PRESENCE_PENALTY`,
  `_FREQUENCY_PENALTY`), which adds no option but does mean a host with those variables set now
  inherits them; and a router no longer forwards `LLAMA_ARG_API_KEY_FILE` to spawned children
  (#28938). `tools/server/`'s schema, task and context translation units are byte-identical, and
  the request-field set (68), bounded-field set (23) and response-key set (142) were all verified
  unchanged mechanically, so the server wire contract cannot have moved. The other eight patches
  apply unchanged and all four remaining drop-checks still say "still required".
- **llama.cpp `b11062` → `b11069`, and local patch `0011` dropped — malformed UTF-8 in a
  completion is now replaced, not truncated.** Seven upstream commits, 57 KiB, no project-source
  change; the one that matters is llama.cpp #29161 ("common/peg : handle invalid utf-8 sequences in
  the AST", first tagged b11063). It fixes the failure `0011` had carried since 5.1.0 — a single
  undecodable byte in the model's output made the content-only parse `FAIL` and the request 500 —
  independently and more broadly than the patch did, so the patch no longer applies and was dropped
  rather than refreshed (the `0009`/`0013` precedent). The observable difference: `0011` returned the
  text *up to* the bad byte, whereas upstream consumes every undecodable run and substitutes exactly
  one U+FFFD for it (the Unicode "maximal subpart" rule — `\xE4\xB8` followed by `c` is one run,
  `\xFF\xFE` is two), so the text *after* the byte is now delivered too. A trailing sequence that is
  still incomplete at the end of the input keeps being withheld, as before. The `ContentOnlyParseUtf8`
  C++ tests that guarded the patch now pin upstream's replacement contract on every platform. The
  rest of the range is CUDA/Metal/WebGPU kernel tuning and a converter flag; `tools/server/`,
  `common/arg.*`, `src/llama-model.*` and `ggml/include` are byte-identical across it, so the other
  eight patches apply unchanged and the server wire contract cannot have moved.
- **llama.cpp `b10731` → `b10850`.** No project-source change: every header move in the range is
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

  The final `b10797` → `b10817` step touches seven files across the paths this project links
  against, `+12 / −13` in total, and none of them needs a source change here:

  - `common/build-info.h` gains an optional `FILE *` on `llama_print_build_info` (llama.cpp
    #28322) — a **default argument**, so every existing one-argument call still compiles. The
    project does not call it; it uses `llama_build_info()`, which is unchanged.
  - `ggml/include/ggml-backend.h` **removes** `ggml_backend_op_alloc_size_may_expand` (the ggml
    sync, #28379). No source here references it, so the removal is inert — checked, not assumed,
    because a deletion from a public header is the one shape that breaks a build silently.
  - `tools/mtmd/mtmd.cpp` corrects Gemma-4V causal-attention handling for the E2B/E4B embedding
    sizes and `clip.cpp` raises an image-token limit (#28335) — both model behaviour inside
    upstream translation units, no API movement.
  - The remaining three are upstream's own version bump, a `LLAMA_VERSION_MINOR` change and a
    BoringSSL pin that only applies with `LLAMA_BUILD_BORINGSSL`, which this project leaves off.

  The final `b10817` → `b10819` step is two commits over three files, none of which this project
  links against: a **Metal memory-leak fix** on an early-return path
  (`ggml-metal-context.m`, llama.cpp #28399), a SYCL Kronecker-product/FWHT restore (#28254) and
  the matching upstream test. Nothing in the review surface, nothing in the server contract.

  The `b10819` → `b10850` step is 31 commits and 466 KB in total, but **10 files / +181 / −98**
  across the paths this project links against — `common/arg.cpp`, `common/jinja/{caps,runtime}.cpp`,
  `common/log.{cpp,h}` and `tools/server/server-models.{cpp,h}` plus that tool's CMake, README and
  a router test. Everything else is GPU backends, the WebUI, docs and upstream CI. Two of those
  files are patch targets (`common/arg.cpp` for `0001`, `tools/server/server-models.cpp` for
  `0008`, the latter with 159 lines rewritten), so the intersection was **not** empty this time and
  the applier had to prove it rather than the file list implying it: a fresh configure applied all
  eight patches clean and stamped them at head `f114f91f`. The server wire contract was re-checked
  mechanically and is byte-identical — request-field set, `set_hard_limits` bounds, and response
  keys in **both** emit forms.

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

### Added
- **Wire-name registries with a declared contract, checked in CI against the real receiver.** The
  three surfaces that leave this library as names on a wire — CLI options, request keys, trainer
  configuration keys — are now enum constants (`args.ModelOption` + `args.ModelFlag`,
  `parameters.RequestField`, `parameters.TrainingField`), each declaring the contract it satisfies.
  The parameter base classes accept nothing else, so an undeclared name cannot reach the wire at
  all. CMake extracts the declarations at configure time and three C++ test files feed them to the
  actual receivers — llama.cpp's server argument parser, its completion-request schema, and the
  trainer's own key list — on **every** platform, which is the only place these can be checked: the
  request schema and the trainer both ignore an unknown key silently. `WireNameRegistryTest` checks
  the other direction, that every declared constant is still reachable from a public builder method.

  This found the twelve dead names removed below. It also found one that a schema check alone could
  not: a key exempted as "consumed by the OpenAI layer before the schema" is only proven *absent*
  from the schema, which a key nothing reads at all satisfies equally well. That exemption now
  additionally requires a reader upstream, and `chat_template` was the one key that had none.

### Removed
- **Twelve builder methods that wrote a name no llama.cpp receiver reads — breaking, no
  deprecation window.** Each one looked like configuration and behaved as a no-op, or worse:

  | Removed | Why |
  |---|---|
  | `InferenceParameters.withTfsZ` | `tfs_z` — upstream deleted the tail-free sampler |
  | `InferenceParameters.withPenalizeNl` | `penalize_nl` — deleted upstream |
  | `InferenceParameters.withPenaltyPrompt(String)` / `(int...)` | `penalty_prompt` — deleted upstream |
  | `InferenceParameters.withUseChatTemplate` | `use_jinja` is a **server start** flag, never a request key |
  | `InferenceParameters.withChatTemplate` | `chat_template` is a **load-time** option; the server only ever *emits* that name, in `/props` |
  | `ModelParameters.setGrpAttnN` / `setGrpAttnW` | `--grp-attn-n`/`-w` exist in `arg.cpp` but are `set_examples()`-scoped away from the server, so the parser rejects them |
  | `ModelParameters.enableDumpKvCache` | `--dump-kv-cache` — deleted upstream |
  | `ModelParameters.setHfRepoV` / `setHfFileV` | `--hf-repo-v`/`--hf-file-v` — deleted upstream |
  | `ModelParameters.enableMlock` / `disableMmap` | `--mlock`/`--no-mmap` — deleted at b10878 in favour of `--load-mode` |

  The two failure modes differ and neither was visible from Java. A dead **CLI** name is a hard parse
  error, so `loadModel()` throws `"Failed to parse model parameters"` and the model does not load. A
  dead **request** key is discarded by llama.cpp's schema without a word, so the parameter simply
  stops having an effect. Either way the Java tests asserting the string mapping
  (`hasKey("--mlock")`) stayed green. Replacements where one exists: `setLoadMode(LoadMode)` for the
  last row, `ModelParameters.setChatTemplate(String)` for `withChatTemplate`, and `--jinja` at server
  start for `withUseChatTemplate`.

  Deprecating was considered and rejected for the same reason as `enableFlashAttn` below: a method
  that keeps writing a name nothing reads is a trap with a warning label on it.

- **`ModelParameters.enableFlashAttn()` and `ModelFlag.FLASH_ATTN` — breaking.** Both modelled
  `--flash-attn` as a valueless flag, which it has not been since b10273. Keeping either would leave
  the broken argv reachable: the method directly, the enum constant through the public
  `setFlag(ModelFlag)`. Replacement: `setFlashAttn(FlashAttn)`.

  Deprecating instead was considered and dropped. A deprecated method that still emits an argv
  llama.cpp misparses is a trap with a warning label on it, and this is a major-version window.

### Fixed
- **Every streaming generation sent the native parser an unparseable body.** Splitting the parameter
  object's single renderer into `toJson()` (the wire form) and `toString()` (a redacted debug view,
  deliberately not valid JSON) turned every surviving `toString()` payload call site into a silent
  trap. Six were repointed; `LlamaIterator` was missed, so `generate()`, `generateChat()`, the
  `LlamaIterable` paths and the Kotlin `generateFlow` / `generateChatFlow` all shipped
  `InferenceParameters{keys=[…], values=redacted}` where a request body belonged. It is caught by an
  ArchUnit rule now — no class outside the `parameters` package may call a parameter object's
  `toString()` at all — and the stale class javadoc that described `toString` as "consumed by the
  native server" is corrected.

  Nothing local could see it: every test that exercises streaming is model-gated and self-skips
  without a GGUF, so a green `mvn test` with 269 skips said nothing about it. It surfaced on the
  first full-matrix CI run, on all five model-backed test jobs at once — which is the behaviour the
  redacted form was designed for, an unparseable body failing loudly at the parser rather than a
  plausible-looking one succeeding with different values.
- **A caller-supplied JSON fragment could inject sibling fields into a request body.**
  `InferenceParameters` stored every value as a raw string and built the request by concatenating
  `"key": value` pairs, so a fragment passed to `withJsonSchema` / `withResponseFormat` /
  `withStreamOptions` / `withMessagesJson` / `withToolsJson` — anywhere a caller supplies JSON text —
  could close its own object and append arbitrary keys. Duplicate keys resolve last-wins in the
  native parser, so an injected `n_predict` or `grammar` silently beat the one the builder wrote.
  The body is now built as a real JSON tree, and every stored value is parsed and required to be
  **exactly one well-formed JSON value** at write time — a fragment with a trailing sibling is
  rejected with the offending key named. `toString()` on a parameter object is now a redacted debug
  view (keys only) and deliberately not valid JSON; use `toJson()` for the wire form.
- **The Android "LLM Service" app applied its chat-template override per request**, where llama.cpp
  discarded it. It is now set at model load (`ModelParameters.setChatTemplate`), which is where
  upstream reads it. Only the `CHAT_TEMPLATE` test hook set it, so no shipped UI path changed
  behaviour — but the on-device test was proving less than it looked.
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
