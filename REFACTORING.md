# Refactoring Plan: java-llama.cpp → Lean JNI Wrapper

## Context

`java-llama.cpp` ships ~**6,154 lines of custom C++** dominated by `server.hpp`
(3,780 lines), a hand-ported copy of llama.cpp's pre-split `server.cpp`. When that
port was written, upstream had a single monolithic `server.cpp` glued to
`cpp-httplib`, so the only way to drive the slot/task machinery from JNI was to fork
and strip all HTTP.

**Upstream has since done exactly that refactor.** `tools/server/` is now split into
library-grade translation units with a clean public API, and two reference
implementations (`tools/cli/cli.cpp`, `tools/completion/completion.cpp`) demonstrate
the whole `get_response_reader → post_task → next` loop in ~40 lines each.

This plan **deletes `server.hpp`**, links upstream's server source files directly into
`jllama`, and rewrites `jllama.cpp` as a thin JNI shim. Outcome: ~5,000 C++ lines
removed; every duplicate (base64, slot_params, result formatters, task dispatch) gone;
future llama.cpp upgrades become a CMake version bump instead of a 100-line sync patch.

**Incompatibility is acceptable** — explicitly authorised.

---

## Baseline (before refactoring)

| File | Lines | Nature |
|------|-------|--------|
| `src/main/cpp/server.hpp` | 3,780 | Adapted copy of llama.cpp server logic — no HTTP |
| `src/main/cpp/jllama.cpp` | 1,270 | JNI bridge — 17 native methods |
| `src/main/cpp/jni_helpers.hpp` | 398 | JNI type-conversion helpers |
| `src/main/cpp/json_helpers.hpp` | 243 | Pure JSON transforms |
| `src/main/cpp/utils.hpp` | 322 | Misc utilities (50 lines copied base64) |
| **Total** | **6,154** | |

---

## Upstream Server Library (llama.cpp `tools/server/`)

At b8913, `tools/server/` is split into library-grade compilation units:

| File | Purpose |
|------|---------|
| `server-context.{h,cpp}` | Pimpl `server_context` — `load_model`, `start_loop`, `terminate`, `get_response_reader`, `get_meta`, `get_llama_context`, `on_sleeping_changed` |
| `server-queue.{h,cpp}` | `server_queue`, `server_response`, **`server_response_reader`** — the explicit non-HTTP embedder API |
| `server-task.{h,cpp}` | `server_task`, `task_params`, `server_task_type` enum, `task_response_type` enum, `server_task::params_from_json_cmpl()` |
| `server-common.{h,cpp}` | `oaicompat_chat_params_parse`, `format_response_*`, `tokens_to_str`, `tokenize_input_prompts`, base64 |
| `server-chat.{h,cpp}` | OAI/Anthropic chat parsing, streaming diffs |
| `server-models.{h,cpp}` | Model/LoRA registry |
| `server-http.{h,cpp}` | **HTTP transport only — NOT compiled into jllama** |
| `server.cpp` | `main()` entry point — NOT compiled into jllama |

### `server_response_reader` — key non-HTTP API (`server-queue.h`)

```cpp
class server_response_reader {
    int  get_new_id();
    void post_task(server_task task);
    void post_tasks(std::vector<server_task>);
    // blocks until next result or should_stop() returns true:
    server_task_result_ptr next(std::function<bool()> should_stop);
    // drains all results for a task id (blocking):
    std::vector<server_task_result_ptr> wait_for_all(std::function<bool()> should_stop);
    void stop();
};
```

### Reference pattern (`tools/completion/completion.cpp`)

```cpp
server_context ctx_server;
ctx_server.load_model(params);
server_response_reader rd = ctx_server.get_response_reader();

server_task task{SERVER_TASK_TYPE_COMPLETION};
task.id         = rd.get_new_id();
task.cli        = true;          // skip HTTP tokenization
task.cli_prompt = prompt_text;
rd.post_task(std::move(task));

while (true) {
    server_task_result_ptr result = rd.next(should_stop);
    if (!result || result->is_error()) break;
}
```

---

## Target Architecture

```
BEFORE:
  jllama.cpp → #include "server.hpp"  (3,780-line forked copy)

AFTER:
  jllama.cpp → #include <server-context.h>   ← upstream
               #include <server-task.h>       ← upstream
               #include <server-queue.h>      ← upstream
               #include <server-common.h>     ← upstream
               // thin JNI glue only (~400 lines)
```

**The Java API stays completely unchanged.** All 17 native method signatures in
`LlamaModel.java` remain identical.

---

## Implementation Phases

Each phase ends with a commit. Every commit leaves the build runnable and recoverable.

### Phase 0 — Safety net ✅ DONE

- Branch `claude/refactor-java-llama-d3lua` checked out.
- Baseline recorded: 6,154 lines of custom C++.

---

### Phase 1 — CMakeLists: compile upstream server files into `jllama`

**File:** `CMakeLists.txt`

1. Add upstream server units **directly** to the `add_library(jllama SHARED …)` source
   list (not via a separate target, to avoid the LLAMA_BUILD_TOOLS flag issue).
   Added files: `server-context.cpp`, `server-queue.cpp`, `server-task.cpp`,
   `server-models.cpp` (wrapped in `if(NOT ANDROID_ABI)` guard).
   Already present: `server-common.cpp`, `server-chat.cpp`.
2. Include path `${llama.cpp_SOURCE_DIR}/tools/server` is **already in place** (line 223).
3. **Not added:** `server.cpp`, `server-http.cpp`, `server-tools.cpp`.
4. `server.hpp` kept — no behaviour change yet (additive only).

**Risk to verify in build:** Does `server-context.cpp` transitively include
`server-http.h` at link time? If yes, a stub TU will be needed.

**Status:** [TO BE UPDATED AFTER BUILD]

---

### Phase 2 — Replace `server.hpp` with a thin shim + update `jllama_context`

**Files:** `src/main/cpp/server.hpp`, `src/main/cpp/jni_helpers.hpp`

Replace the 3,780-line body with a ~20-line include shim:
```cpp
#pragma once
#include <server-context.h>
#include <server-task.h>
#include <server-queue.h>
#include <server-common.h>
#include <server-chat.h>
```

Rewrite `jllama_context` to hold an upstream `server_context` + reader map:
```cpp
struct jllama_context {
    server_context server;
    std::thread    worker;
    std::atomic<bool> worker_ready{false};
    bool vocab_only = false;
    std::mutex readers_mutex;
    std::unordered_map<int, server_response_reader> readers;
};
```

**Status:** [PENDING Phase 1 completion]

---

### Phase 3 — Migrate pure `llama.h` methods

**File:** `src/main/cpp/jllama.cpp`

| JNI method | Upstream replacement |
|---|---|
| `encode` | `common_tokenize(vocab, text, false, true)` |
| `decodeBytes` | `tokens_to_str(ctx, tokens)` — `server-common.h` |
| `handleTokenize` | Same + inline `{"tokens": [...]}` JSON |
| `handleDetokenize` | Same + inline `{"content": "..."}` JSON |
| `jsonSchemaToGrammarBytes` | Already direct — no change |
| `applyTemplate` | `oaicompat_chat_params_parse(...).prompt` |
| `setLogger` | Already direct — no change |
| `getModelMetaJson` | Serialize `ctx_server.get_meta()` fields |

**Status:** [PENDING Phase 2 completion]

---

### Phase 4 — Migrate embeddings and reranking

**File:** `src/main/cpp/jllama.cpp`

```cpp
auto rd = ctx->server.get_response_reader();
server_task t{SERVER_TASK_TYPE_EMBEDDING};
t.id = rd.get_new_id();
t.cli = true;
t.cli_prompt = prompt;
rd.post_task(std::move(t));
auto results = rd.wait_for_all([]{ return false; });
```

Delete from `jni_helpers.hpp` / `json_helpers.hpp`: `extract_first_embedding_row`,
`rerank_results_to_json`, `build_embeddings_response_json`, `dispatch_and_collect`,
`append_task`.

**Status:** [PENDING Phase 3 completion]

---

### Phase 5 — Migrate completions, chat completions, infill

**File:** `src/main/cpp/jllama.cpp`

**Blocking methods** (`handleCompletions`, `handleChatCompletions`, `handleInfill`):
```cpp
server_task task{SERVER_TASK_TYPE_COMPLETION};
task.id     = rd.get_new_id();
task.cli    = true;
task.params = server_task::params_from_json_cmpl(vocab, params_base, n_ctx, logit_bias_eog, data);
task.params.res_type = TASK_RESPONSE_TYPE_OAI_CMPL;
rd.post_task(std::move(task));
auto results = rd.wait_for_all([]{ return false; });
return results.back()->to_json().dump();
```

**Streaming methods** (`requestCompletion`, `receiveCompletionJson`, `cancelCompletion`,
`releaseTask`): store `server_response_reader` in `jllama_context::readers` map, keyed
by task id. `receiveCompletionJson` calls `rd.next(...)`. `cancelCompletion` calls
`rd.stop()`. `releaseTask` erases the map entry.

**Status:** [PENDING Phase 4 completion]

---

### Phase 6 — Slot management, metrics, configureParallelInference

- `handleSlotAction`: map action codes to upstream `SERVER_TASK_TYPE_SLOT_SAVE/RESTORE/ERASE`.
- `getMetrics`: `SERVER_TASK_TYPE_METRICS` task.
- `configureParallelInference`: **no-op returning `true`** (no upstream post-load
  equivalent; document as deprecated).

**Status:** [PENDING Phase 5 completion]

---

### Phase 7 — Delete dead code

- Delete `src/main/cpp/server.hpp` (replaced by thin shim or inlined).
- `utils.hpp` → shrink to ~40 lines (drop base64 duplicate).
- `jni_helpers.hpp` → shrink to handle/error helpers (~80 lines).
- `json_helpers.hpp` → remove helpers replaced by upstream.
- `src/test/cpp/test_server.cpp` → **delete** (internals now in upstream).
- Trim `test_jni_helpers.cpp`, `test_json_helpers.cpp` to surviving helpers.
- Update `BUILD_TESTING` block in `CMakeLists.txt`.

**Status:** [PENDING Phase 6 completion]

---

### Phase 8 — Verification

```bash
cmake -B build -DBUILD_TESTING=ON && cmake --build build --config Release
ctest --test-dir build --output-on-failure
mvn compile
mvn test -Dtest=StopReasonTest,InferenceParametersTest,LlamaLoaderTest,OSInfoTest
wc -l src/main/cpp/*   # target: <1,000 lines total
```

**Must pass:** `LlamaModelTest` tokenization/vocab/embedding/grammar/complete/generate/
chatComplete/cancel/getModelMeta, `LlamaEmbeddingsTest`, `ModelParametersTest`,
`InferenceParametersTest`, `LlamaOutputTest`.

**Acceptable failures:** `ConfigureParallelInferenceTest` (deprecated to no-op).

**Status:** [PENDING Phase 7 completion]

---

## Expected Code Reduction

| File | Before | After |
|------|--------|-------|
| `server.hpp` | 3,780 | **0** (deleted) |
| `jllama.cpp` | 1,270 | ~500 |
| `jni_helpers.hpp` | 398 | ~80 |
| `json_helpers.hpp` | 243 | ~60 |
| `utils.hpp` | 322 | ~40 |
| **Total** | **6,154** | **~680** (~89% reduction) |

---

## Upstream types/functions to reuse

- `server_context`, `server_context_meta` — `tools/server/server-context.h`
- `server_response_reader` — `tools/server/server-queue.h`
- `server_task`, `task_params`, `server_task_type`, `task_response_type`,
  `server_task::params_from_json_cmpl` — `tools/server/server-task.h`
- `oaicompat_chat_params_parse`, `tokens_to_str` — `tools/server/server-common.h`
- Reference pattern: `tools/completion/completion.cpp`, `tools/cli/cli.cpp`

## Risks

1. `server-context.cpp` may drag in `server-http.h` link symbols → add stub TU if needed.
2. `task.cli = true` token output must match legacy path — smoke-test after Phase 3.
3. `server_response_reader` map must be mutex-guarded and destroyed on `releaseTask`.
4. `configureParallelInference` deprecated to no-op — document in CLAUDE.md.
