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

---

## #123 — Supporting latest llama.cpp version

- **URL:** https://github.com/kherud/java-llama.cpp/issues/123
- **Reporter:** jesuino
- **Created:** 2025-11-18

Asks whether there are plans to support the latest llama.cpp versions; user
wants to try Qwen3-VL.

---

## #121 — Error on Android apk

- **URL:** https://github.com/kherud/java-llama.cpp/issues/121
- **Reporter:** Togaroda
- **Created:** 2025-10-28

When using the library inside an Android APK, the loader cannot find the native
library because it looks for an `aarch64` directory while Android packages
ship `arm64-v8a` (and `armeabi-v7a`). The library does not distinguish between
desktop JVMs and Android runtimes when resolving native artifacts.

---

## #120 — What models are supported?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/120
- **Reporter:** Togaroda
- **Created:** 2025-10-12

Question: are Qwen3 or SmolLM2 supported?

---

## #119 — Update questions

- **URL:** https://github.com/kherud/java-llama.cpp/issues/119
- **Reporter:** Myhuiku
- **Created:** 2025-08-19

Asks whether the maintainer plans to update the pinned llama.cpp to the latest
release.

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

---

## #116 — java-llama creates a non-daemon thread that inhibits JVM termination

- **URL:** https://github.com/kherud/java-llama.cpp/issues/116
- **Reporter:** claudionieder
- **Created:** 2025-07-01

After `new LlamaModel(...)` followed by `close()`, the JVM does not exit
because a non-daemon thread (`Thread-0`) is spawned at model load and never
stopped on close. Expected: thread should be a daemon, or be joined/terminated
on `LlamaModel.close()`.

---

## #113 — Allow tracking the progress of loading a LlamaModel

- **URL:** https://github.com/kherud/java-llama.cpp/issues/113
- **Reporter:** natanfudge
- **Created:** 2025-06-09

Feature request: expose load progress (file name, fraction 0..1, bytes loaded
vs. total, whether the file is the weights file, whether it is a download or
disk load) via a `Consumer<LLamaLoadProgress>` callback passed to the
`LlamaModel` constructor. Intended for showing a progress bar to end users.

---

## #112 — Qwen 3 model does not load

- **URL:** https://github.com/kherud/java-llama.cpp/issues/112
- **Reporter:** msky-dev
- **Created:** 2025-06-07

Loading a Qwen3 GGUF fails with `unknown model architecture: 'qwen3'`. User
reports newer llama.cpp builds do support it, so the pinned upstream needs to
be bumped.

---

## #111 — Running with a Predefined Number of Cores

- **URL:** https://github.com/kherud/java-llama.cpp/issues/111
- **Reporter:** michaelsheka
- **Created:** 2025-05-28

How-to question: how to configure the library to use a specific (predefined)
number of CPU cores instead of the default.

---

## #110 — Running embedding in batch

- **URL:** https://github.com/kherud/java-llama.cpp/issues/110
- **Reporter:** michaelsheka
- **Created:** 2025-05-28

The current embedding API only supports a single input string at a time. User
requests guidance / API support for batched embedding of multiple strings in
one call.

---

## #107 — OS_NAME for Mac in CMakeLists.txt

- **URL:** https://github.com/kherud/java-llama.cpp/issues/107
- **Reporter:** prabhdatnoor
- **Created:** 2025-05-08

On macOS (Intel, 2020), the JNI include directory detection fails. Root cause:
`CMakeLists.txt` checks `OS_NAME STREQUAL "Mac"` but CMake actually reports
`Darwin`. Suggested fix: replace `"Mac"` with `"Darwin"` on line 74 of
`CMakeLists.txt`.

---

## #104 — Question: How to enable offload_kqv?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/104
- **Reporter:** michaelsheka
- **Created:** 2025-04-23

How-to question: how to enable the `offload_kqv` llama.cpp option from the
Java API.

---

## #103 — VLM support — Image input for multimodal models

- **URL:** https://github.com/kherud/java-llama.cpp/issues/103
- **Reporter:** amirvenus
- **Created:** 2025-04-21

Feature request: support visual-language models such as Qwen2.5-VL (image
inputs) on Android.

---

## #102 — Native call to 'delete' doesn't free memory

- **URL:** https://github.com/kherud/java-llama.cpp/issues/102
- **Reporter:** karambaso
- **Created:** 2025-03-23

Repeatedly constructing and `close()`-ing `LlamaModel` instances does not
release native memory. Reproduced with a 10-iteration loop: GPU eventually
OOMs; CPU eventually thrashes swap. Suggests a leak in the JNI/native
destructor path.

---

## #101 — Log messages are not redirected to a provided consumer

- **URL:** https://github.com/kherud/java-llama.cpp/issues/101
- **Reporter:** karambaso
- **Created:** 2025-03-22

After calling `LlamaModel.setLogger(LogFormat.TEXT, consumer)`, log lines still
go to stdout instead of the supplied `consumer`. Reporter notes that the
logging template in `Utils.cpp` never invokes the user callback.

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

---

## #95 — Inference content repetition that can't be ended

- **URL:** https://github.com/kherud/java-llama.cpp/issues/95
- **Reporter:** z503951608
- **Created:** 2025-03-11

Reporter takes issue with `LlamaIterator.next()`: when the receive call returns
`stop`, the iterator exposes the final token then ends, but the user observes
inference output that loops and cannot be terminated. Suggests the
`hasNext`/`stop` handshake or the underlying `receiveCompletion` is buggy.

---

## #91 — Unable to integrate java-llama.cpp with Android Studio Java

- **URL:** https://github.com/kherud/java-llama.cpp/issues/91
- **Reporter:** Mohsin-Fawad
- **Created:** 2025-01-20

User followed README steps but cannot integrate the library into an
Android Studio Java project. Requests a working `MainActivity.java`,
`build.gradle.kts`, and any other modified files.

---

## #90 — Process to rebuild after Java-only changes

- **URL:** https://github.com/kherud/java-llama.cpp/issues/90
- **Reporter:** siddhsql
- **Created:** 2024-12-29

Asks for the documented process to rebuild the project when only Java sources
are modified, reusing pre-built native binaries instead of recompiling
llama.cpp.

---

## #89 — What are slots?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/89
- **Reporter:** siddhsql
- **Created:** 2024-12-29

Asks for an explanation of the `server_slot` concept (referenced in
`src/main/cpp/server.hpp`).

---

## #88 — JSON request like role system or role user isn't supported?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/88
- **Reporter:** yousefabdel1727
- **Created:** 2024-12-26

Asks how to send chat-style JSON requests (`role: system`, `role: user`) the
way the upstream llama-server accepts them.

---

## #87 — How to reset the context on every turn?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/87
- **Reporter:** siddhsql
- **Created:** 2024-12-18

In the `MainExample` chat loop, the context fills up over multiple turns,
slowing inference and eventually producing garbage. Asks how to reset the KV
cache / context between iterations.

---

## #86 — Does `llama-3.4.1-cuda12-linux-x86-64.jar` handle both CPU and GPU or only GPU?

- **URL:** https://github.com/kherud/java-llama.cpp/issues/86
- **Reporter:** siddhsql
- **Created:** 2024-12-18

Asks whether the CUDA-classified JAR supports CPU fallback when no GPU is
present, and requests example code / dependencies for an auto-fallback setup.

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

---

## #84 — rerank

- **URL:** https://github.com/kherud/java-llama.cpp/issues/84
- **Reporter:** litongjava
- **Created:** 2024-10-05

How-to question: how to run a reranker model such as
`BAAI/bge-reranker-v2-m3` with the bindings.

---

## #83 — `EXCEPTION_ACCESS_VIOLATION` on Windows 11 x86-64 with default libraries

- **URL:** https://github.com/kherud/java-llama.cpp/issues/83
- **Reporter:** kyselat
- **Created:** 2024-10-02

Versions from 3.3.0 onward (downloaded via Maven Central) crash the JVM on
Windows 11 with `EXCEPTION_ACCESS_VIOLATION` inside `msvcp140.dll`. Earlier
versions up to 3.2.1 work. User has the latest VC++ 2022 redistributable
installed.

---

## #82 — Please give a sample for Android

- **URL:** https://github.com/kherud/java-llama.cpp/issues/82
- **Reporter:** SteveWorkshop
- **Created:** 2024-09-30

User tried the README's Gradle snippet for importing the native library and it
fails with `Could not find method listOf() for arguments [mvn, compile]…`
(a Groovy-vs-Kotlin DSL mismatch). Requests a working Android sample.

---

## #81 — Add Android sample

- **URL:** https://github.com/kherud/java-llama.cpp/issues/81
- **Reporter:** shalva97
- **Created:** 2024-09-27

Feature request: a separate Android sample repository that can be cloned and
run as-is to demonstrate the library on Android.

---

## #80 — Segfault on model open and close

- **URL:** https://github.com/kherud/java-llama.cpp/issues/80
- **Reporter:** shuttie
- **Created:** 2024-09-24

On 3.4.1, opening a model and immediately calling `close()` (without
generating) segfaults inside libc, with a stack pointing into
`std::_Rb_tree::_M_erase`. If generation happens first, `close()` succeeds.
Likely an uninitialized/half-initialized native field being destroyed.

---

## #79 — Android inference issue: `pthread_mutex_lock called on a destroyed mutex`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/79
- **Reporter:** xunuohope1107
- **Created:** 2024-09-19

On Android, model load succeeds but `llamaModel.complete(inferParams)` aborts
with `FORTIFY: pthread_mutex_lock called on a destroyed mutex` and `SIGABRT`.
Same code works on macOS. Suggests an Android-specific JNI/threading bug or a
library ABI mismatch.

---

## #78 — Add support for `params.lora_adapters` (llama.cpp ≥ b3534)

- **URL:** https://github.com/kherud/java-llama.cpp/issues/78
- **Reporter:** xunuohope1107
- **Created:** 2024-09-12

Upstream llama.cpp changed the LoRA API from a single `lora_adapter` to a
vector of `lora_adapters` structs (each carrying a path and a scaling factor).
Requests that the bindings update to the new structure and bump the pinned
llama.cpp version.

---

## #77 — Process exit `-1073741819 (0xC0000005)` inferring CodeGemma-2B GGUF

- **URL:** https://github.com/kherud/java-llama.cpp/issues/77
- **Reporter:** 32kda
- **Created:** 2024-09-10

On Windows 10, attempting inference with `codegemma-2b.Q2_K.gguf` causes JVM
to exit with `0xC0000005` (access violation) shortly after start, with no
output and no HotSpot crash dump produced.

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

---

## #70 — Add `vocab_only`

- **URL:** https://github.com/kherud/java-llama.cpp/issues/70
- **Reporter:** ardinursyamsu
- **Created:** 2024-06-21

Feature request: expose llama.cpp's `vocab_only` flag, equivalent to the
Python binding's option, so a Spring Boot app can run a tokenizer-only
service separately from the model service.

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

---

## #34 — Support multimodal inputs

- **URL:** https://github.com/kherud/java-llama.cpp/issues/34
- **Reporter:** yeroc
- **Created:** 2023-12-19

Feature request: add multimodal input support (referencing
[ggerganov/llama.cpp#3436](https://github.com/ggerganov/llama.cpp/pull/3436)).

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
