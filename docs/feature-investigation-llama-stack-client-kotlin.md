# Feature Investigation — ideas from `llama-stack-client-kotlin`

Comparison source: [ogx-ai/llama-stack-client-kotlin](https://github.com/ogx-ai/llama-stack-client-kotlin)
(version 0.2.14, a Stainless-generated Kotlin SDK for the Llama Stack REST API
with an optional ExecuTorch-backed local-inference path).

This document inventories candidate features for `java-llama.cpp` derived from
that comparison, with rough effort estimates. Effort is given in
T-shirt sizes:

| Size | Calendar effort (1 engineer) | Description |
|------|------------------------------|-------------|
| XS   | < 0.5 day                    | Trivial Java-side change, no JNI |
| S    | 0.5 – 2 days                 | Java surface + minor JNI/JSON wiring |
| M    | 2 – 5 days                   | New JNI methods, native plumbing, tests |
| L    | 1 – 2 weeks                  | New native subsystem or large API surface |
| XL   | > 2 weeks                    | Architectural addition |

## 1. What `java-llama.cpp` already covers

| Capability                                              | Status |
|---------------------------------------------------------|--------|
| Chat / completion (blocking + streaming via `LlamaIterable`) | ✅ |
| Embeddings (`embed(String)`)                             | ✅ |
| Rerank (`rerank(query, docs)`)                           | ✅ |
| Grammar + JSON-schema constrained output                 | ✅ |
| Rich sampling (DRY, mirostat, dyn-temp, XTC, top-n-σ)    | ✅ |
| Reasoning budget / reasoning format                      | ✅ |
| Slot save / restore / erase                              | ✅ |
| `continueFinalMessage`                                   | ✅ |
| Tokenize / decode / template apply                       | ✅ |
| Metrics string (`getMetrics()`)                          | ✅ |
| Speculative draft model wiring                           | ✅ |

These do not need work — they already match or exceed the Kotlin client.

## 2. Recommended additions (in priority order)

### 2.1 Multimodal image input (mtmd) — **L**

**Gap.** Upstream llama.cpp ships `mtmd` (vision + audio for some models) and
the compiled-in server already pulls it in via `mtmd.h` / `mtmd-helper.h`. No
Java method currently accepts image input. Kotlin examples show base64 image
chat against vision models.

**Proposal.**
- `InferenceParameters.addImage(byte[] png)` / `addImage(Path)` / `addImageBase64(String)`.
- `ModelParameters.setMmproj(Path)` to load the mmproj projector file.
- JNI: feed images into the server task params (`mtmd_*` API).

**Effort: L** — non-trivial JNI plumbing, lifecycle of `mtmd_context`,
test fixtures for vision models, but most of the heavy lifting is already
upstream.

**Value.** Biggest user-visible capability missing today. Unlocks Qwen-VL,
Gemma 3, MiniCPM-V, LLaVA, etc.

---

### 2.2 Typed `ChatMessage` / `ChatResponse` model + tool calling — **M**

**Gap.** Today: `setMessages(String system, List<Pair<String,String>>)` and
`chatComplete → String`. The server *parses* tool calls
(`common_chat_*` infrastructure) but Java callers must scrape JSON
themselves. Kotlin exposes typed `ChatCompletionRequest` / `ChatResponse`
with `toolCalls`, `finishReason`, `usage`.

**Proposal.**
- `ChatMessage` record: `role` (enum: system/user/assistant/tool), `content`
  (text + optional image parts), `toolCalls`, `toolCallId`, `name`.
- `ToolDefinition` record (name, description, JSON-schema parameters) and
  `InferenceParameters.setTools(List<ToolDefinition>)`.
- `ChatResponse` record: `content`, `finishReason` (enum), `toolCalls`,
  `usage` (see §2.5), `timings`, optional `logprobs`.
- New `LlamaModel.chat(ChatRequest)` / `chatStream(ChatRequest)`.

**Effort: M** — Java-side data model + JSON marshalling + a wrapper around
the existing native chat path. No new native code needed.

**Value.** Brings the library in line with every modern Java LLM SDK
(LangChain4j, Spring AI). Removes the "parse the JSON string yourself"
papercut.

---

### 2.3 Async / non-blocking API — **S–M**

**Gap.** All `LlamaModel` methods are blocking. Kotlin offers
`suspend fun` + Flow variants. JVM users currently dedicate platform
threads per inference.

**Proposal.**
- `CompletableFuture<String> completeAsync(InferenceParameters)`
- `CompletableFuture<ChatResponse> chatAsync(ChatRequest)`
- Reactive Streams `Publisher<String> generateReactive(...)` (or
  `Flow.Publisher` from `java.util.concurrent.Flow` to avoid a new
  dependency).
- Backed by the existing native worker thread inside `jllama_context`;
  no extra Java thread pool.

**Effort: S** for `CompletableFuture` wrappers, **M** for a proper
backpressure-aware `Flow.Publisher` streaming token producer.

**Value.** Enables composition with virtual threads, project Reactor,
RxJava, Kotlin coroutines from Java consumers.

---

### 2.4 Batch inference across slots — **M**

**Gap.** llama.cpp natively serves parallel slots; the compiled-in server
handles concurrent tasks. `LlamaModel` exposes no batch entry point.

**Proposal.**
- `List<String> completeBatch(List<InferenceParameters>)`.
- `List<ChatResponse> chatBatch(List<ChatRequest>)`.
- `Stream<String> generateBatch(...)` returning results in submission order.
- Configure parallelism via existing `ModelParameters` (e.g. n_parallel).

**Effort: M** — server already supports it; needs a Java entry that
dispatches N tasks, awaits all, and maps results.

**Value.** Throughput multiplier for embedding / classification /
rerank pipelines; close to a free win.

---

### 2.5 Typed `Usage` / `Timings` result — **XS–S**

**Gap.** `getMetrics()` returns a raw JSON `String`. Kotlin exposes
`Usage(promptTokens, completionTokens, totalTokens)` plus a richer
`Timings` (`tokensPerSecond`, `promptMs`, `predictedMs`, `cacheHit`,
`drafted`, `accepted`).

**Proposal.**
- `Usage` and `Timings` records.
- Returned as part of `ChatResponse` (§2.2) and from a new
  `LlamaModel.getMetricsTyped()` accessor.

**Effort: XS** if scoped to `Timings`/`Usage` parsing only,
**S** when wired into the new `ChatResponse`.

**Value.** Removes ad-hoc JSON parsing. Cheap.

---

### 2.6 `Session` helper (multi-turn) — **S–M**

**Gap.** Slots exist as a low-level primitive. Kotlin offers
"agents/sessions/turns" with persistence and resume.

**Proposal.**
- `Session` (`AutoCloseable`) owning a slot id, an accumulating
  `List<ChatMessage>`, and `save(Path)` / `restore(Path)` that delegate
  to `saveSlot` / `restoreSlot`.
- `session.send(userMessage)` → `ChatResponse`.
- `session.stream(userMessage)` → `LlamaIterable`.

**Effort: S** if it stays a thin wrapper over existing slot APIs,
**M** if we add proper concurrency / per-session locking.

**Value.** Common conversational use case becomes a one-liner.

---

### 2.7 Stream cancellation & `AutoCloseable` iterator — **S**

**Gap.** `LlamaIterable` / `LlamaIterator` cannot be cancelled mid-stream;
the underlying slot task keeps running until natural stop. Kotlin marks
streaming returns `@MustBeClosed`.

**Proposal.**
- Make `LlamaIterator` implement `AutoCloseable` with `close()` cancelling
  the server task (slot stop).
- Document try-with-resources usage.

**Effort: S** — small JNI addition (`cancel_task(task_id)`), straightforward
Java side.

**Value.** Prevents wasted compute when the consumer gives up early.

---

### 2.8 Structured-output convenience helpers — **S**

**Gap.** `setJsonSchema` / `setGrammar` already exist on `ModelParameters`
but not on `InferenceParameters`. No typed-result helper.

**Proposal.**
- Add `setJsonSchema(String)` / `setGrammar(String)` to
  `InferenceParameters` (per-request schema).
- `<T> T completeAsJson(Class<T>, InferenceParameters)` — derive the
  JSON schema from the class via Jackson (`JacksonModule` already on
  classpath?) or document a manual-schema variant.

**Effort: S**.

**Value.** Matches the "structured outputs" UX of OpenAI / Llama Stack
SDKs.

---

### 2.9 Logprobs in the typed result — **S**

**Gap.** `setNProbs` exists; the result type is a plain `String`, so
per-token probabilities are not surfaced.

**Proposal.**
- Add `List<TokenLogprob> logprobs` to `ChatResponse` (§2.2) and to a
  new `LlamaOutput` field for non-chat completion.
- `TokenLogprob { String token; int tokenId; float logprob;
  List<TokenLogprob> topLogprobs; }`.

**Effort: S**.

**Value.** Required for evaluation / uncertainty estimation use cases.

---

### 2.10 Cancellation token / abort for blocking calls — **S**

**Gap.** A blocking `complete(...)` cannot be aborted from another thread.

**Proposal.**
- `complete(params, CancellationToken token)` overload.
- Token wraps a native task id; `token.cancel()` issues a stop.

**Effort: S** — overlaps with §2.7's `cancel_task` primitive.

**Value.** Enables UI timeouts and request-scoped cancellation in
web frameworks.

---

## 3. Considered but **not recommended**

| Kotlin feature                | Why we should not port it |
|------------------------------|---------------------------|
| Agents framework / Turns     | Belongs in a higher-level "stack" layer (LangChain4j, Spring AI). Out of scope for a thin JNI binding. |
| Shields / Safety service     | Same — separate library concern. |
| Eval / Benchmarks / Scoring  | Separate test/eval tooling, not inference. |
| SyntheticDataGeneration      | Out of scope. |
| Telemetry spans (OTEL)       | Users can add their own around `LlamaModel` calls. |
| VectorDB / VectorStore / RAG | Would duplicate ObjectBox / Lucene / pgvector / Chroma integrations. Higher-level libs do this better. |
| Files / Datasets             | REST-only concept; no on-device analogue. |
| HTTP client conveniences (retries, proxy, timeouts) | N/A for in-process JNI. |

## 4. Suggested rollout order

1. **§2.1 Multimodal (L)** — biggest capability gap, isolated subsystem.
2. **§2.2 Typed Chat model + tool calling (M)** — foundational; other
   features (usage, logprobs, async) all return / accept these types.
3. **§2.5 Usage / Timings (XS)** — quick, lands inside §2.2.
4. **§2.7 + §2.10 Cancellation primitive (S)** — small JNI add, unblocks §2.3.
5. **§2.3 Async API (S–M)** — `CompletableFuture` first, reactive later.
6. **§2.4 Batch inference (M)**.
7. **§2.6 Session helper (S)**.
8. **§2.8 Structured-output helpers (S)**.
9. **§2.9 Logprobs in result (S)**.

Total realistic effort for the full list: **~5–7 engineer-weeks**, with
multimodal alone accounting for roughly a third of that.

## 5. Open questions

- **Multimodal scope** — vision only first, or also audio (Qwen-Audio,
  Whisper-style mmproj)?
- **Reactive dep** — use `java.util.concurrent.Flow` (JDK 9+ but project
  targets bytecode 1.8) or add Reactive Streams as an optional Maven
  classifier?
- **Tool calling** — expose the raw `common_chat_*` parser output, or
  normalize to an OpenAI-compatible shape?
- **Session persistence format** — reuse llama.cpp's slot binary file, or
  also store the message history as JSON next to it?
