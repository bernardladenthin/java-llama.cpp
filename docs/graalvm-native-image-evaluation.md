<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# GraalVM Native Image as a distribution target -- evaluation

**Verdict:** feasible, not free, and not worth doing before a consumer asks for it. The work is
bounded and known (this document is the plan); the gain is a start-up time that nobody has measured
against the time the model load itself takes, which is the dominant cost of every short-lived use.
Nothing in the library stands in the way, and nothing in the library should be changed for it now.

What this is **not**: a way to drop JNI. The pure-Java sibling runtimes in the README's "Similar
Projects" (mukel's `llama3.java` and friends) run inference in Java; this project runs llama.cpp.
Native Image would compile the *Java layer* ahead of time and still load `libjllama` through JNI --
the same library, the same loader, one process image instead of a JVM.

## What a consumer would get

A single executable per OS/arch that starts in tens of milliseconds, with no JVM to install, for the
shapes a JVM start hurts: a command-line tool, a serverless function, a short-lived subprocess, a
desktop launcher. The `NativeServer` and `OpenAiCompatServer` entry points are long-running and gain
little; `llama-atmosphere-agent` (Java 21) is the most plausible first target, since it is an
application and already ships as a fat jar.

## What jllama needs from Native Image, measured against the source

Native Image resolves JNI and reflection at build time, from metadata it is given. The surface is
small and enumerable, and the GraalVM tracing agent writes it out from one representative run.

| Surface | What the source does today | Native Image needs |
|---|---|---|
| **JNI lookups in `JNI_OnLoad`** | `jllama.cpp`: 16 `FindClass`, 16 `GetMethodID`, 1 `GetFieldID`, 7 `GetStaticFieldID`; `native_server.cpp` and `rpc_bridge.cpp` two lookups each (the `ctx` field, the exception class) | `jni-config.json` naming each class, method and field. The set is closed: every name is a string literal in three files. A class the config misses is a `NoClassDefFoundError` out of `JNI_OnLoad`, i.e. the same failure `NativeLibraryLoadSmokeTest` exists for. |
| **45 `native` methods** | `Java_*` symbols resolved by the JVM's `System.load` | nothing extra: Native Image links JNI libraries loaded at run time the way the JVM does (`System.load` is supported); the exported `Java_*` names are found in the loaded library. |
| **The natives on the classpath** | `LlamaLoader` looks up `net/ladenthin/llama/<OS>/<ARCH>/<backend>/{library, jllama-extras.txt, jllama-files.txt}` through the `ClassLoader` (three call sites) and extracts them to the temp directory | either **(a)** `resource-config.json` including those directories, so they live inside the image and the loader keeps extracting (36 MB for a CPU-variants directory, embedded in the executable), or **(b)** the natives next to the executable as sidecar files, found through `net.ladenthin.llama.lib.path` -- which already exists and already means "load from this directory, extract nothing". (b) needs no loader change and keeps the executable small; (a) keeps the single-file promise. |
| **Backend selection** | one directory per backend, tried in `BACKEND_PRIORITY` order until one loads | unchanged under (a); under (b) the directory layout next to the executable is one backend, chosen at packaging time -- a GPU image and a CPU image, as the natives jars already separate them. |
| **Jackson** | tree model only: 31 `readTree`, one `writeValueAsString`; the one reflective use is the caller's own type in `completeAsJson` | Jackson's tree model needs no reflection metadata of ours; Jackson's own internals ship reachability metadata through the GraalVM metadata repository. A caller's type for `completeAsJson` is the caller's `reflect-config.json`, as it is the caller's `opens` on the module path today. |
| **SLF4J** | `slf4j-simple` through `ServiceLoader` | `ServiceLoader` is supported; the provider is found at image build time. |
| **`com.sun.net.httpserver`** | `OpenAiCompatServer` | the `jdk.httpserver` module is supported by Native Image. |
| **Threads crossing JNI** | the log sink attaches llama.cpp's worker thread with `AttachCurrentThreadAsDaemon`; the model worker and the attach-mode HTTP threads are JVM-attached | both are part of Native Image's JNI invocation interface. |
| **`System.loadLibrary` on Android** | the Android path | out of scope: Native Image does not target Android. |

Nothing in the table is a redesign. The two decisions a real attempt has to take are the natives
placement, (a) or (b), and which entry point to build first.

## The measurement that decides it

Native Image removes JVM start-up and class loading. It does **not** remove the model load, the
first-token latency or the inference itself, which the native library does identically either way.
So the question is one number: what share of a cold `LlamaModel(...).complete(one token)` is JVM
start-up plus class loading? Measure it on a machine with a model, in this order:

1. `java -Xshare:auto -cp <classes + natives> <a main that loads the draft model and generates one
   token>`, wall clock from process start to exit, three runs after a warm-up (the page cache holds
   the GGUF after the first run; the comparison wants it warm, because that is the steady state of a
   tool that is run repeatedly).
2. The same with `-verbose:class | wc -l` and `-Xlog:startuptime` (or simply a timestamp as the
   first statement of `main`), which gives JVM start + class loading as a separate number.
3. `LlamaModel` construction alone (model load) and the one-token completion alone, from the
   log timestamps the library already writes (`TimingsLogger`, the load log).

If JVM start-up is below ~10% of the total, Native Image buys a rounding error for a CLI and nothing
for a server. Above ~50%, it is worth the second build matrix below. The draft model (AMD-Llama-135m,
~100 MB) is the right fixture for the smallest-model case; `Qwen3-0.6B` for a realistic one. The
cloud sessions have no model and HuggingFace is unreachable from them, so this is a task for the
local agent (`docs/local-test-plan.md`), not for CI.

## The cost, so it is not underestimated

- **A second build matrix.** One image per OS/arch (and per backend under (b)); each needs a
  GraalVM JDK on the runner and a `native-image` step, and the Linux image should be built in the
  manylinux_2_28 container like the natives, or its glibc floor rises to the runner's.
- **Metadata that rots silently.** A llama.cpp bump that adds a JNI-reachable Java type, or a new
  `FindClass` in `JNI_OnLoad`, breaks only the image, and only at run time. The guard is a CI job that
  builds one image and runs the model-free smoke against it (`NativeLibraryLoadSmokeTest`'s shape:
  load, `JNI_OnLoad`, `getLlamaCppBuildInfo()`), gating the publish jobs like every other job.
- **The agent's reflective surface.** `llama-atmosphere-agent` pulls Atmosphere, Jetty and JLine,
  each with its own reflection and resource needs; the tracing agent handles it, but that image is a
  bigger project than the library's.
- **Java 8 is irrelevant here, and so is the module path.** An image is built from one classpath at
  one Java level (GraalVM 21+); the library's Java 8 floor and the `module-info` stay as they are for
  the jar consumers.

## Recommendation

Do nothing until either a consumer asks for an executable or the measurement above shows JVM
start-up dominating a use that matters. When it does: start with (b) (natives as sidecar files
through `net.ladenthin.llama.lib.path`, no loader change), the `NativeServer` main as the entry
point (the fat-jar default, the smallest reflective surface), the tracing agent for the metadata,
and the CI image-build-plus-smoke job from day one. Record the measurement in
`docs/local-test-results-<cpu>.md` like every other local result.
