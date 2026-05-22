# Open Issues at Baseline `49be664`

**Baseline commit:** `49be66475700487e9ae9be5ba1d22b5855bb0d1c` (kherud/java-llama.cpp,
"bump pom.xml version 4.1.0 -> 4.20", 2025-06-20).

This file enumerates all **37 open issues** on the upstream
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp) repository at
the time of the fork. Each issue is reproduced below with its number, title,
original URL, reporter, creation date and a concise summary of the reported
problem or request. Issues are listed in descending order of issue number
(newest first).

The intent is that each item can later be investigated, reproduced against the
current fork, and either marked fixed (linking the resolving commit/PR),
deferred, or closed as not-applicable.

---

## #124 — Request an update！！！

- **URL:** https://github.com/kherud/java-llama.cpp/issues/124
- **Reporter:** YodyOy
- **Created:** 2026-04-16

Empty body. Generic request for the maintainer to publish a new release / sync
with upstream llama.cpp.

**Status in fork:** FIXED. The fork pins llama.cpp to `b9284` (see `CLAUDE.md` line 11 and `CMakeLists.txt` GIT_TAG), with an automated per-version upgrade workflow visible in `git log --oneline` (e.g. `f84f974 Upgrade llama.cpp from b9279 to b9284`, plus dozens of similar bump commits). Releases now flow through `.github/workflows/release.yaml`.

---

## #123 — Supporting latest llama.cpp version

- **URL:** https://github.com/kherud/java-llama.cpp/issues/123
- **Reporter:** jesuino
- **Created:** 2025-11-18

Asks whether there are plans to support the latest llama.cpp versions; user
wants to try Qwen3-VL.

**Status in fork:** FIXED. llama.cpp pinned to `b9284` (`CLAUDE.md:11`), which natively supports Qwen3 and Qwen3-VL architectures. The fork also fetches `tools/mtmd` via FetchContent (`CMakeLists.txt:125-145`) and links `mtmd` into `jllama` (`CMakeLists.txt:255`), so VLM-capable upstream support is compiled in even though the Java surface for image input is not yet exposed (see #103).

---

## #121 — Error on Android apk

- **URL:** https://github.com/kherud/java-llama.cpp/issues/121
- **Reporter:** Togaroda
- **Created:** 2025-10-28

When using the library inside an Android APK, the loader cannot find the native
library because it looks for an `aarch64` directory while Android packages
ship `arm64-v8a` (and `armeabi-v7a`). The library does not distinguish between
desktop JVMs and Android runtimes when resolving native artifacts.

**Status in fork:** PARTIALLY FIXED. `OSInfo.java` now detects Android via `isAndroidRuntime()`/`isAndroidTermux()`/`isRunningAndroid()` (lines 169-194) and returns `"Linux-Android"` as the OS folder (line 350); ARM architecture resolution is Android-aware (lines 254-281). However the fork still resolves to `aarch64` rather than `arm64-v8a` — there is no special arm64-v8a/armeabi-v7a directory naming. Next steps: inspect `LlamaLoader.java:103` Android branch to confirm the directory layout under `src/main/resources/net/ladenthin/llama/Linux-Android/aarch64/`, and produce an Android sample to verify end-to-end loading.

---

## #120 — What models are supported?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/120
- **Reporter:** Togaroda
- **Created:** 2025-10-12

Question: are Qwen3 or SmolLM2 supported?

**Status in fork:** FIXED. Model architecture support is delegated to the pinned llama.cpp version `b9284` (`CLAUDE.md:11`), which supports Qwen3, Qwen3-VL, SmolLM2 and many other newer architectures.

---

## #119 — Update questions

- **URL:** https://github.com/kherud/java-llama.cpp/issues/119
- **Reporter:** Myhuiku
- **Created:** 2025-08-19

Asks whether the maintainer plans to update the pinned llama.cpp to the latest
release.

**Status in fork:** FIXED. The fork has an automated, per-build version-bump cadence (CLAUDE.md documents the upgrade procedure; `git log --oneline | grep "Upgrade llama.cpp"` shows continuous bumps culminating in `b9284`).

---

## #117 — The new model is causing crashes on Android

- **URL:** https://github.com/kherud/java-llama.cpp/issues/117
- **Reporter:** ysc-ysc-ysc
- **Created:** 2025-07-10

User packaged the library into an AAR and integrated it with Godot on Android.
Instantiating the model crashes the app with `SIGABRT` from an uncaught
`std::filesystem::filesystem_error` (`posix_stat … Permission denied
["/charger"]`). Crash occurs inside `libhoudini.so` on an `x86_64` Android
emulator/device, suggesting the native code tries to walk a filesystem path
that is not accessible on Android.

**Status in fork:** NEEDS INVESTIGATION. The crash signature (`std::filesystem::filesystem_error` walking `/charger`) points at upstream backend-device enumeration, not project code. The fork is several hundred upstream versions ahead of the report so the specific walk may have changed. Next steps: reproduce on an `x86_64` Android emulator with a current `b9284` build; if it persists, file a fresh upstream issue and consider catching `std::filesystem_error` at backend-load time.

---

## #116 — java-llama creates a non-daemon thread that inhibits JVM termination

- **URL:** https://github.com/kherud/java-llama.cpp/issues/116
- **Reporter:** claudionieder
- **Created:** 2025-07-01

After `new LlamaModel(...)` followed by `close()`, the JVM does not exit
because a non-daemon thread (`Thread-0`) is spawned at model load and never
stopped on close. Expected: thread should be a daemon, or be joined/terminated
on `LlamaModel.close()`.

**Status in fork:** FIXED. The native worker thread is a C++ `std::thread` (not a Java thread, so daemon-ness does not apply at the JVM level) and is explicitly joined on close: `Java_net_ladenthin_llama_LlamaModel_delete` calls `jctx->server.terminate()` twice and then `jctx->worker.join()` (`src/main/cpp/jllama.cpp:937-940`). Streaming readers are also drained before shutdown (`jllama.cpp:925-930`).

---

## #113 — Allow tracking the progress of loading a LlamaModel

- **URL:** https://github.com/kherud/java-llama.cpp/issues/113
- **Reporter:** natanfudge
- **Created:** 2025-06-09

Feature request: expose load progress (file name, fraction 0..1, bytes loaded
vs. total, whether the file is the weights file, whether it is a download or
disk load) via a `Consumer<LLamaLoadProgress>` callback passed to the
`LlamaModel` constructor. Intended for showing a progress bar to end users.

**Status in fork:** STILL POSSIBLE. No `LLamaLoadProgress`, `Consumer<…>` or `setProgressCallback` exists in `LlamaModel.java` / `ModelParameters.java` (`grep -n "progress\|Progress"` returns only iterator cancellation comments). Next steps: expose `llama_model_params.progress_callback` through `ModelParameters` (Java side: a `Consumer<Float>` field; JNI side: wire a trampoline in `jllama.cpp` similar to `log_callback_trampoline`).

---

## #112 — Qwen 3 model does not load

- **URL:** https://github.com/kherud/java-llama.cpp/issues/112
- **Reporter:** msky-dev
- **Created:** 2025-06-07

Loading a Qwen3 GGUF fails with `unknown model architecture: 'qwen3'`. User
reports newer llama.cpp builds do support it, so the pinned upstream needs to
be bumped.

**Status in fork:** FIXED. llama.cpp pin is now `b9284` (`CLAUDE.md:11`), which includes the Qwen3 architecture (added upstream around `b3000+`). Resolved transparently by the version bump.

---

## #111 — Running with a Predefined Number of Cores

- **URL:** https://github.com/kherud/java-llama.cpp/issues/111
- **Reporter:** michaelsheka
- **Created:** 2025-05-28

How-to question: how to configure the library to use a specific (predefined)
number of CPU cores instead of the default.

**Status in fork:** FIXED (documentation only — API already existed). `ModelParameters.setThreads(int)` (line 51) and `setThreadsBatch(int)` (line 61) map directly to the `--threads`/`--threads-batch` CLI flags. The how-to gap could be addressed by a Javadoc example.

---

## #110 — Running embedding in batch

- **URL:** https://github.com/kherud/java-llama.cpp/issues/110
- **Reporter:** michaelsheka
- **Created:** 2025-05-28

The current embedding API only supports a single input string at a time. User
requests guidance / API support for batched embedding of multiple strings in
one call.

**Status in fork:** FIXED. In addition to the single-prompt `LlamaModel.embed(String)` (`LlamaModel.java:95`), the fork exposes `handleEmbeddings(String paramsJson, boolean oaiCompat)` (`LlamaModel.java:316`) which forwards arbitrary JSON to the upstream embeddings handler. The native side uses `extract_embedding_prompt` (`src/main/cpp/json_helpers.hpp:137`, called from `jllama.cpp:1075`), which accepts either a string or an array of strings — enabling batched embedding requests.

---

## #107 — OS_NAME for Mac in CMakeLists.txt

- **URL:** https://github.com/kherud/java-llama.cpp/issues/107
- **Reporter:** prabhdatnoor
- **Created:** 2025-05-08

On macOS (Intel, 2020), the JNI include directory detection fails. Root cause:
`CMakeLists.txt` checks `OS_NAME STREQUAL "Mac"` but CMake actually reports
`Darwin`. Suggested fix: replace `"Mac"` with `"Darwin"` on line 74 of
`CMakeLists.txt`.

**Status in fork:** FIXED. `CMakeLists.txt:196` now matches both: `if(OS_NAME MATCHES "^Linux" OR OS_NAME STREQUAL "Mac" OR OS_NAME STREQUAL "Darwin")`. On the Java side, `OSInfo.translateOSNameToFolderName` normalises both `Mac` and `Darwin` to the folder name `Mac` (`OSInfo.java:345-346`).

---

## #104 — Question: How to enable offload_kqv?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/104
- **Reporter:** michaelsheka
- **Created:** 2025-04-23

How-to question: how to enable the `offload_kqv` llama.cpp option from the
Java API.

**Status in fork:** FIXED. KQV offload is on by default in upstream; to *disable* it the fork exposes `ModelFlag.NO_KV_OFFLOAD` mapped to `--no-kv-offload` (`src/main/java/net/ladenthin/llama/args/ModelFlag.java:50`), used via `ModelParameters.java:739`. Leaving the flag unset preserves the default `offload_kqv=true`.

---

## #103 — VLM support — Image input for multimodal models

- **URL:** https://github.com/kherud/java-llama.cpp/issues/103
- **Reporter:** amirvenus
- **Created:** 2025-04-21

Feature request: support visual-language models such as Qwen2.5-VL (image
inputs) on Android.

**Status in fork:** PARTIALLY FIXED. The build links the upstream `mtmd` multimodal library into `jllama` (`CMakeLists.txt:125-145, 253-255`) and `ModelParameters` exposes `setMmproj`, `setMmprojUrl`, `enableMmprojAuto`, `enableMmprojOffload` (`ModelParameters.java:1250-1281`). However, no high-level Java API for attaching image bytes/paths to an inference request is yet exposed — `InferenceParameters` has no `image`/`addImage`/`images` methods (`grep -n image src/main/java/net/ladenthin/llama/InferenceParameters.java` returns no hits). VLM requests are reachable only via the OAI-compat chat path (`handleChatCompletions`) using a JSON `content` array with `image_url` entries. Next steps: add a typed `InferenceParameters.addImage(byte[]|Path)` helper that constructs the multipart `content` array.

---

## #102 — Native call to 'delete' doesn't free memory

- **URL:** https://github.com/kherud/java-llama.cpp/issues/102
- **Reporter:** karambaso
- **Created:** 2025-03-23

Repeatedly constructing and `close()`-ing `LlamaModel` instances does not
release native memory. Reproduced with a 10-iteration loop: GPU eventually
OOMs; CPU eventually thrashes swap. Suggests a leak in the JNI/native
destructor path.

**Status in fork:** LIKELY FIXED. The native destructor (`Java_net_ladenthin_llama_LlamaModel_delete`, `src/main/cpp/jllama.cpp:917-948`) now: clears the field pointer first, drains `readers`, signals `server.terminate()` (twice for the race documented in comments), joins the worker, frees `vocab_only_model` if set, and `delete jctx` (which destroys the embedded `server_context` value member). Refactor `fc55802 Refactor JNI lifecycle and log formatting into reusable helpers` (`git log --oneline`) explicitly addressed lifecycle. Next steps: run a 100-iteration open/close loop test under valgrind and `nvidia-smi --query-gpu=memory.used` to confirm.

---

## #101 — Log messages are not redirected to a provided consumer

- **URL:** https://github.com/kherud/java-llama.cpp/issues/101
- **Reporter:** karambaso
- **Created:** 2025-03-22

After calling `LlamaModel.setLogger(LogFormat.TEXT, consumer)`, log lines still
go to stdout instead of the supplied `consumer`. Reporter notes that the
logging template in `Utils.cpp` never invokes the user callback.

**Status in fork:** FIXED. `Java_net_ladenthin_llama_LlamaModel_setLogger` (`src/main/cpp/jllama.cpp:954-977`) now registers the JNI callback as a global ref and installs `log_callback_trampoline` via `llama_log_set(...)`. The trampoline invokes `m_biconsumer_accept` on the user's `BiConsumer<LogLevel,String>` with the formatted text. JSON formatting is gated by `log_json` from `LogFormat`.

---

## #98 — Error loading embedding model `nomic-embed-text-v1.5.f16.gguf`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/98
- **Reporter:** taksan
- **Created:** 2025-03-13

Loading `nomic-embed-text-v1.5.f16.gguf` aborts with
`GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor")`.
The same file works with upstream `llama-embedding` CLI, so the issue is in
how the bindings configure the embedding context (e.g. missing `embedding=true`
or pooling parameter).

**Status in fork:** LIKELY FIXED. `ModelParameters.enableEmbedding()` sets `ModelFlag.EMBEDDING` (`ModelParameters.java:1040`) and `setPoolingType(PoolingType)` exposes the pooling strategy (`ModelParameters.java:606`). The upstream b9284 server-context (compiled directly into `jllama`) handles `nomic-embed-text` correctly. Next steps: add an integration test loading `nomic-embed-text-v1.5.f16.gguf` to confirm.

---

## #95 — Inference content repetition that can't be ended

- **URL:** https://github.com/kherud/java-llama.cpp/issues/95
- **Reporter:** z503951608
- **Created:** 2025-03-11

Reporter takes issue with `LlamaIterator.next()`: when the receive call returns
`stop`, the iterator exposes the final token then ends, but the user observes
inference output that loops and cannot be terminated. Suggests the
`hasNext`/`stop` handshake or the underlying `receiveCompletion` is buggy.

**Status in fork:** LIKELY FIXED. `LlamaIterator` now uses `receiveCompletionJson` and toggles `hasNext = !output.stop` (`LlamaIterator.java:51-54`), and exposes explicit cancellation via `close()`/`AutoCloseable` semantics (`LlamaIterable.java:44`, `LlamaIterator.java:69-77`). Repetition itself remains a sampler-tuning issue handled via `InferenceParameters` (`setRepeatPenalty`, `setMinP`, etc.). Next steps: add a regression test that asserts iteration terminates with the expected `StopReason` on a deliberately repetitive prompt.

---

## #91 — Unable to integrate java-llama.cpp with Android Studio Java

- **URL:** https://github.com/kherud/java-llama.cpp/issues/91
- **Reporter:** Mohsin-Fawad
- **Created:** 2025-01-20

User followed README steps but cannot integrate the library into an
Android Studio Java project. Requests a working `MainActivity.java`,
`build.gradle.kts`, and any other modified files.

**Status in fork:** STILL POSSIBLE. The fork has Android detection in `OSInfo.java` (lines 169-194) but ships no Android sample app, no Gradle integration documentation, and no pre-built `arm64-v8a` AAR. Next steps: produce a minimal sample under `examples/android/` with `MainActivity.java` and `build.gradle.kts` and document NDK ABI mapping.

---

## #90 — Process to rebuild after Java-only changes

- **URL:** https://github.com/kherud/java-llama.cpp/issues/90
- **Reporter:** siddhsql
- **Created:** 2024-12-29

Asks for the documented process to rebuild the project when only Java sources
are modified, reusing pre-built native binaries instead of recompiling
llama.cpp.

**Status in fork:** FIXED (documented). `CLAUDE.md` "Build Commands" section explicitly separates `mvn compile` / `mvn package` (Java-only — no native rebuild) from the `cmake -B build && cmake --build build` native step. Running `mvn package` alone reuses the pre-built shared libraries under `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.

---

## #89 — What are slots?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/89
- **Reporter:** siddhsql
- **Created:** 2024-12-29

Asks for an explanation of the `server_slot` concept (referenced in
`src/main/cpp/server.hpp`).

**Status in fork:** NOT APPLICABLE. The hand-ported `server.hpp` has been removed; the project now compiles the upstream `server-context.cpp`/`server-queue.cpp`/`server-task.cpp`/`server-models.cpp` directly (see `CLAUDE.md` "Architecture" section). `server_slot` is now an upstream-owned concept; consumers of `LlamaModel` see it indirectly via the slot-related task params (`InferenceParameters.setIdSlot`, etc.).

---

## #88 — JSON request like role system or role user isn't supported?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/88
- **Reporter:** yousefabdel1727
- **Created:** 2024-12-26

Asks how to send chat-style JSON requests (`role: system`, `role: user`) the
way the upstream llama-server accepts them.

**Status in fork:** FIXED. `LlamaModel.chatComplete(InferenceParameters)` (line 236) routes to native `handleChatCompletions` which accepts the standard OpenAI `messages: [{role, content}, ...]` JSON format (`LlamaModel.java:215-238`). The chat template is auto-applied via the compiled upstream chat pipeline.

---

## #87 — How to reset the context on every turn?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/87
- **Reporter:** siddhsql
- **Created:** 2024-12-18

In the `MainExample` chat loop, the context fills up over multiple turns,
slowing inference and eventually producing garbage. Asks how to reset the KV
cache / context between iterations.

**Status in fork:** FIXED. `InferenceParameters.setCachePrompt(boolean)` (line 116) controls reuse of the cached prefix, and per-slot KV state is managed by the upstream server. To "reset" between turns, the caller passes a fresh `id_slot` or sets `cache_prompt=false`. The upstream server also supports an explicit `/slots/{id}:erase` task type accessible via the chat path.

---

## #86 — Does `llama-3.4.1-cuda12-linux-x86-64.jar` handle both CPU and GPU or only GPU?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/86
- **Reporter:** siddhsql
- **Created:** 2024-12-18

Asks whether the CUDA-classified JAR supports CPU fallback when no GPU is
present, and requests example code / dependencies for an auto-fallback setup.

**Status in fork:** PARTIALLY FIXED. The CUDA classifier `cuda13-linux-x86-64` is built via `.github/build_cuda_linux.sh` (see `CLAUDE.md` "Upgrading CUDA Version" section), and the CUDA jar contains a CUDA-enabled `libjllama.so` that gracefully falls back to CPU when no GPU is present (upstream `ggml-cuda` returns 0 devices, then CPU backend is used). Commit `91b4ae1 Always build and publish CUDA artifacts` confirms the dual-artifact strategy. Next steps: add Javadoc / README guidance documenting the fallback.

---

## #85 — SIGILL on M1 MacBook with Rosetta 2 and Java 8

- **URL:** https://github.com/kherud/java-llama.cpp/issues/85
- **Reporter:** s0t00524
- **Created:** 2024-11-26

Running the library under Rosetta 2 on an M1 Mac with Java 8
(`OpenJDK 1.8.0_242`, `bsd-amd64`) crashes with `SIGILL` inside
`libggml.dylib+0x4724 (ggml_init+0x74)`. The same code works on
Linux x86_64. Suggests `ggml_init` uses CPU instructions Rosetta cannot
emulate for this build.

**Status in fork:** NEEDS INVESTIGATION. This is a Rosetta-2 specific x86_64 emulation defect; not project-side. The Java target was upgraded from 1.8 to 21 then explicitly downgraded back to bytecode 1.8 (`a48957e Upgrade Java version to 21`, `ac84fe0 Downgrade Java target to 1.8`). Recommend reproducing on Apple Silicon with native arm64 builds (no Rosetta) — the fork ships a `Mac/aarch64/` native artifact.

---

## #84 — rerank

- **URL:** https://github.com/kherud/java-llama.cpp/issues/84
- **Reporter:** litongjava
- **Created:** 2024-10-05

How-to question: how to run a reranker model such as
`BAAI/bge-reranker-v2-m3` with the bindings.

**Status in fork:** FIXED. `ModelParameters.enableReranking()` (`ModelParameters.java:1049`) enables the rerank endpoint; `LlamaModel.rerank(boolean, String query, String...)` (`LlamaModel.java:170`) and `rerank(query, documents)` (line 187) return `Pair<String,Float>` or `LlamaOutput`. End-to-end coverage exists in `src/test/java/net/ladenthin/llama/RerankingModelTest.java` using `jina-reranker-v1-tiny-en-Q4_0.gguf`.

---

## #83 — `EXCEPTION_ACCESS_VIOLATION` on Windows 11 x86-64 with default libraries

- **URL:** https://github.com/kherud/java-llama.cpp/issues/83
- **Reporter:** kyselat
- **Created:** 2024-10-02

Versions from 3.3.0 onward (downloaded via Maven Central) crash the JVM on
Windows 11 with `EXCEPTION_ACCESS_VIOLATION` inside `msvcp140.dll`. Earlier
versions up to 3.2.1 work. User has the latest VC++ 2022 redistributable
installed.

**Status in fork:** NEEDS INVESTIGATION. The fork's Windows artifact is rebuilt for `b9284`, and `src/main/cpp/compat/ggml_x86_compat.c` provides a 32-bit MSVC `_InterlockedIncrement64` shim (`CLAUDE.md` ~b8808–b8831 row). The original Maven Central artifact (kherud groupId) is unaffected by this fork. Next steps: build the fork's Windows artifact and re-run the failing scenario; if it still crashes, profile under WinDbg against `msvcp140.dll` to identify the offending C++ runtime symbol.

---

## #82 — Please give a sample for Android

- **URL:** https://github.com/kherud/java-llama.cpp/issues/82
- **Reporter:** SteveWorkshop
- **Created:** 2024-09-30

User tried the README's Gradle snippet for importing the native library and it
fails with `Could not find method listOf() for arguments [mvn, compile]…`
(a Groovy-vs-Kotlin DSL mismatch). Requests a working Android sample.

**Status in fork:** STILL POSSIBLE. No Android sample is shipped with the fork. Next steps: same as #91 — add `examples/android/` with a working Gradle (Kotlin DSL) config and a verified `arm64-v8a` build of `libjllama.so` via the dockcross toolchain (`.github/dockcross/`).

---

## #81 — Add Android sample

- **URL:** https://github.com/kherud/java-llama.cpp/issues/81
- **Reporter:** shalva97
- **Created:** 2024-09-27

Feature request: a separate Android sample repository that can be cloned and
run as-is to demonstrate the library on Android.

**Status in fork:** STILL POSSIBLE. No separate Android demo repo exists. Tracked alongside #81/#82/#91 as a documentation/sample deliverable.

---

## #80 — Segfault on model open and close

- **URL:** https://github.com/kherud/java-llama.cpp/issues/80
- **Reporter:** shuttie
- **Created:** 2024-09-24

On 3.4.1, opening a model and immediately calling `close()` (without
generating) segfaults inside libc, with a stack pointing into
`std::_Rb_tree::_M_erase`. If generation happens first, `close()` succeeds.
Likely an uninitialized/half-initialized native field being destroyed.

**Status in fork:** LIKELY FIXED. The native destructor now waits on `worker_ready` before terminating (`src/main/cpp/jllama.cpp:929-931`), drains the `readers` map under lock (`jllama.cpp:925-928`), and calls `server.terminate()` twice with a 1 ms sleep specifically to close the race "where the thread signalled ready but `start_loop()` hasn't yet set its internal running flag" (comment at `jllama.cpp:932-934`). This is exactly the half-initialised case the issue describes.

---

## #79 — Android inference issue: `pthread_mutex_lock called on a destroyed mutex`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/79
- **Reporter:** xunuohope1107
- **Created:** 2024-09-19

On Android, model load succeeds but `llamaModel.complete(inferParams)` aborts
with `FORTIFY: pthread_mutex_lock called on a destroyed mutex` and `SIGABRT`.
Same code works on macOS. Suggests an Android-specific JNI/threading bug or a
library ABI mismatch.

**Status in fork:** NEEDS INVESTIGATION. The threading model has been substantially rewritten (single `worker` thread per `jllama_context` joined on `delete`, see `jllama.cpp:683` and `:938`), and the worker `Attach/DetachCurrentThread` cycle is correct. With `b9284` upstream and a properly built Android `arm64-v8a` `libjllama.so`, this is plausibly resolved. Next steps: reproduce on a current Android emulator with a fork-built shared library.

---

## #78 — Add support for `params.lora_adapters` (llama.cpp ≥ b3534)

- **URL:** https://github.com/kherud/java-llama.cpp/issues/78
- **Reporter:** xunuohope1107
- **Created:** 2024-09-12

Upstream llama.cpp changed the LoRA API from a single `lora_adapter` to a
vector of `lora_adapters` structs (each carrying a path and a scaling factor).
Requests that the bindings update to the new structure and bump the pinned
llama.cpp version.

**Status in fork:** FIXED. `ModelParameters.addLoraAdapter(String)` (line 906) and `addLoraScaledAdapter(String, float)` (line 918) map to `--lora` and `--lora-scaled` respectively, both of which the upstream CLI parser already converts into the vector-of-`lora_adapters` form. `setLoraInitWithoutApply()` (line 1121) covers the `--lora-init-without-apply` flag. The llama.cpp pin is `b9284`, well past `b3534`.

---

## #77 — Process exit `-1073741819 (0xC0000005)` inferring CodeGemma-2B GGUF

- **URL:** https://github.com/kherud/java-llama.cpp/issues/77
- **Reporter:** 32kda
- **Created:** 2024-09-10

On Windows 10, attempting inference with `codegemma-2b.Q2_K.gguf` causes JVM
to exit with `0xC0000005` (access violation) shortly after start, with no
output and no HotSpot crash dump produced.

**Status in fork:** NEEDS INVESTIGATION. CodeGemma support has matured upstream since the report, and the fork's Windows artifact is built fresh at `b9284`. The 32-bit MSVC `_InterlockedIncrement64` shim (`src/main/cpp/compat/ggml_x86_compat.c`) addresses one known Windows-specific build issue. Next steps: reproduce on Windows 10 x86-64 with `codegemma-2b.Q2_K.gguf` against the fork's artifact.

---

## #72 — JNI signature exception (`@Nullable BiConsumer`)

- **URL:** https://github.com/kherud/java-llama.cpp/issues/72
- **Reporter:** ys2940
- **Created:** 2024-07-23

On macOS M3 with JDK 1.8, compiling `de.kherud.llama.LlamaModel` fails with
`com.sun.tools.javac.jvm.JNIWriter$TypeSignature$SignatureException` on a
`@org.jetbrains.annotations.Nullable :: java.util.function.BiConsumer`
signature. Suggests `javac -h` cannot encode the annotated parametrised
type.

**Status in fork:** FIXED. The `setLogger` signature in `LlamaModel.java:128` no longer carries `@Nullable` (`public static native void setLogger(LogFormat format, BiConsumer<LogLevel, String> callback);`). `grep -n "@Nullable" src/main/java/net/ladenthin/llama/LlamaModel.java` returns no hits. The build now targets bytecode 1.8 with JDK 21 (`a48957e`, `ac84fe0`), so `javac -h` accepts the signature.

---

## #70 — Add `vocab_only`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/70
- **Reporter:** ardinursyamsu
- **Created:** 2024-06-21

Feature request: expose llama.cpp's `vocab_only` flag, equivalent to the
Python binding's option, so a Spring Boot app can run a tokenizer-only
service separately from the model service.

**Status in fork:** FIXED. `ModelParameters.setVocabOnly()` (`ModelParameters.java:1336`) toggles `ModelFlag.VOCAB_ONLY` → `--vocab-only` (`args/ModelFlag.java:92`). The native layer also branches on `jctx->vocab_only` to skip server/worker init while keeping the vocab available for tokenize/decode (`src/main/cpp/jllama.cpp:710-718, 923, 944`).

---

## #50 — Android build error: `libllama.so is incompatible with aarch64linux`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/50
- **Reporter:** RageshAntonyHM
- **Created:** 2024-02-28

Building for Android (`arm64-v8a`, `armeabi-v7a`) on macOS M2 fails at link
time: the Android NDK linker tries to link against the macOS-aarch64
`libllama.so` shipped under `resources/de/kherud/llama/Mac/aarch64/`, which is
not compatible with `aarch64linux`. Root cause: CMake/build picks the wrong
prebuilt artifact for Android targets.

**Status in fork:** PARTIALLY FIXED. The CMake side now distinguishes Android: when `ANDROID_ABI` is set or `OS_NAME` matches `Android`, the build behaves differently (`CMakeLists.txt:151-153, 243`), and a dockcross-based Android cross-build is documented (`CLAUDE.md` "Cross-compilation"). Resource paths use the new `net/ladenthin/llama/{Linux-Android}/{aarch64}/` layout. Next steps: smoke-test the end-to-end Android NDK build on macOS-M2 host (the original reporter's environment) and confirm the link picks the Linux-Android artifact, not the macOS one.

---

## #34 — Support multimodal inputs

- **URL:** https://github.com/kherud/java-llama.cpp/issues/34
- **Reporter:** yeroc
- **Created:** 2023-12-19

Feature request: add multimodal input support (referencing
[ggerganov/llama.cpp#3436](https://github.com/ggerganov/llama.cpp/pull/3436)).

**Status in fork:** PARTIALLY FIXED. The upstream `mtmd` multimodal library is built and linked into `jllama` (`CMakeLists.txt:125-145, 253-255`), and `ModelParameters` exposes `setMmproj`, `setMmprojUrl`, `enableMmprojAuto`, `enableMmprojOffload` (`ModelParameters.java:1250-1281`). What is still missing is a typed Java API for attaching images to a completion request; for now, callers must construct a JSON `messages[].content` array with `image_url` entries and send it via `chatComplete`. See #103 for the same partial-fix story.

---

## Cross-reference

| # | Category | Title |
|---|---|---|
| 124 | Update request | Request an update |
| 123 | Update request | Supporting latest llama.cpp version |
| 121 | Android | Error on Android apk (aarch64 vs arm64-v8a) |
| 120 | Question | What models are supported? |
| 119 | Update request | Update questions |
| 117 | Android / crash | New model causing crash on Android |
| 116 | Lifecycle | Non-daemon thread inhibits JVM termination |
| 113 | Feature | Load progress callback |
| 112 | Model support | Qwen 3 architecture unknown |
| 111 | Question | Predefined number of cores |
| 110 | Feature | Batch embedding |
| 107 | Build | macOS `OS_NAME` is Darwin not Mac |
| 104 | Question | Enable `offload_kqv` |
| 103 | Feature / Android | VLM / image input |
| 102 | Memory leak | `close()` doesn't free native memory |
| 101 | Logging | Logger consumer ignored |
| 98  | Model support | Embedding model fails (`result_output`) |
| 95  | Iterator bug | Repetition that can't be ended |
| 91  | Android | Integration with Android Studio |
| 90  | Docs | Rebuild process for Java-only changes |
| 89  | Question | What are slots? |
| 88  | Question | JSON role system/user support |
| 87  | Question | Context reset per turn |
| 86  | Question | CUDA jar CPU fallback |
| 85  | macOS / crash | SIGILL under Rosetta 2 (M1, Java 8) |
| 84  | Question | rerank model usage |
| 83  | Windows / crash | EXCEPTION_ACCESS_VIOLATION in msvcp140 |
| 82  | Android | Gradle DSL mismatch in README |
| 81  | Feature | Android sample repository |
| 80  | Crash | Segfault on open+close |
| 79  | Android / crash | pthread_mutex_lock on destroyed mutex |
| 78  | Upstream API | New `params.lora_adapters` shape |
| 77  | Windows / crash | 0xC0000005 on CodeGemma-2B |
| 72  | Build | JNI signature exception on `@Nullable BiConsumer` |
| 70  | Feature | `vocab_only` flag |
| 50  | Android / build | macOS prebuilt picked for Android target |
| 34  | Feature | Multimodal inputs |

## Status overview

| # | Verdict | Short status | Key evidence |
|---|---|---|---|
| 124 | FIXED | Continuous version bumps; pinned to b9284 | `CLAUDE.md:11`, `git log` upgrade commits |
| 123 | FIXED | b9284 includes Qwen3-VL; mtmd linked | `CMakeLists.txt:255`, `CLAUDE.md:11` |
| 121 | PARTIALLY FIXED | Android detected; ABI dir still `aarch64` | `OSInfo.java:169-194,350` |
| 120 | FIXED | Architecture support comes from b9284 | `CLAUDE.md:11` |
| 119 | FIXED | Per-release bump cadence to b9284 | `git log --oneline` Upgrade commits |
| 117 | NEEDS INVESTIGATION | Upstream backend-device crash; reproduce | `b9284` is current; reproduce on emulator |
| 116 | FIXED | Worker joined explicitly on delete | `jllama.cpp:937-940` |
| 113 | STILL POSSIBLE | No load-progress callback in API | No `Progress` in `LlamaModel.java` |
| 112 | FIXED | Qwen3 in b9284 | `CLAUDE.md:11` |
| 111 | FIXED | `setThreads/setThreadsBatch` exist | `ModelParameters.java:51,61` |
| 110 | FIXED | `handleEmbeddings` accepts batch JSON | `LlamaModel.java:316`, `json_helpers.hpp:137` |
| 107 | FIXED | CMake matches both Mac and Darwin | `CMakeLists.txt:196` |
| 104 | FIXED | `NO_KV_OFFLOAD` flag exposed | `args/ModelFlag.java:50` |
| 103 | PARTIALLY FIXED | mtmd linked, no typed image API | `ModelParameters.java:1250-1281` |
| 102 | LIKELY FIXED | Destructor drains workers and frees ctx | `jllama.cpp:917-948` |
| 101 | FIXED | Trampoline calls BiConsumer | `jllama.cpp:954-977` |
| 98  | LIKELY FIXED | enableEmbedding + setPoolingType | `ModelParameters.java:1040,606` |
| 95  | LIKELY FIXED | Iterator close/AutoCloseable wired | `LlamaIterator.java:51-77` |
| 91  | STILL POSSIBLE | No Android sample shipped | No `examples/android/` |
| 90  | FIXED | mvn compile vs cmake split documented | `CLAUDE.md` Build Commands |
| 89  | NOT APPLICABLE | Hand-port `server.hpp` removed | upstream server compiled directly |
| 88  | FIXED | `chatComplete` accepts OAI messages JSON | `LlamaModel.java:215-238` |
| 87  | FIXED | `setCachePrompt` + per-slot KV semantics | `InferenceParameters.java:116` |
| 86  | PARTIALLY FIXED | CUDA jar built; falls back to CPU | `.github/build_cuda_linux.sh`, commit `91b4ae1` |
| 85  | NEEDS INVESTIGATION | Rosetta-2 emulation defect; arm64 builds ship | `Mac/aarch64/` artifact |
| 84  | FIXED | `rerank()` API + RerankingModelTest | `LlamaModel.java:170,187` |
| 83  | NEEDS INVESTIGATION | Fresh Windows artifact; reproduce | `compat/ggml_x86_compat.c` |
| 82  | STILL POSSIBLE | No Android Gradle sample | See #91 |
| 81  | STILL POSSIBLE | No Android demo repo | See #91 |
| 80  | LIKELY FIXED | Half-init race closed by double-terminate | `jllama.cpp:932-940` |
| 79  | NEEDS INVESTIGATION | Threading rewritten; needs Android repro | `jllama.cpp:683,938` |
| 78  | FIXED | `addLoraAdapter`/`addLoraScaledAdapter` | `ModelParameters.java:906,918` |
| 77  | NEEDS INVESTIGATION | Fresh Windows build at b9284 | `compat/ggml_x86_compat.c` |
| 72  | FIXED | `@Nullable` removed from setLogger | `LlamaModel.java:128` |
| 70  | FIXED | `setVocabOnly()` + native branch | `ModelParameters.java:1336`, `jllama.cpp:710-718` |
| 50  | PARTIALLY FIXED | CMake handles ANDROID_ABI; needs e2e test | `CMakeLists.txt:151-153,243` |
| 34  | PARTIALLY FIXED | mtmd linked, no typed image API | `CMakeLists.txt:255` |
