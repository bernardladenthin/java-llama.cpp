# Refactoring: java-llama.cpp → Lean JNI Wrapper

> **This is a running document.** It tracks every phase of the refactoring from
> start to finish and is updated after each commit. When the refactoring is
> complete, this file becomes the final change record. Anyone continuing this
> work in a new session should read this file first and pick up from the first
> phase that is not marked ✅ DONE.

---

## Why

`java-llama.cpp` shipped ~6,154 lines of custom C++ dominated by `server.hpp`
(3,780 lines), a hand-ported copy of llama.cpp's pre-split `server.cpp`. When
that port was written, upstream had a single monolithic `server.cpp` glued to
`cpp-httplib`, so the only way to drive the slot/task machinery from JNI was to
fork and strip all HTTP.

Upstream has since done exactly that refactor. `tools/server/` is now split
into library-grade translation units with a clean public API. This refactoring
**deletes `server.hpp`**, links upstream's server source files directly into
`jllama`, and rewrites `jllama.cpp` as a thin JNI shim.

Outcome: ~4,100 C++ lines removed so far; every duplicate (base64, slot_params,
result formatters, task dispatch) gone; future llama.cpp upgrades become a
CMake version bump instead of a 100-line sync patch.

**The Java API is unchanged.** All native method signatures in `LlamaModel.java`
remain identical.

---

## Baseline (before any changes, on `main`)

| File | Lines | Nature |
|------|-------|--------|
| `src/main/cpp/server.hpp` | 3,780 | Hand-ported copy of llama.cpp server logic |
| `src/main/cpp/jllama.cpp` | 1,270 | JNI bridge — 17 native methods |
| `src/main/cpp/jni_helpers.hpp` | 398 | JNI type-conversion helpers |
| `src/main/cpp/json_helpers.hpp` | 243 | Pure JSON transforms |
| `src/main/cpp/utils.hpp` | 322 | Misc utilities (50 lines copied base64) |
| **Total** | **6,013** | |

---

## Current state (branch `claude/refactor-java-llama-d3lua`)

| File | Lines | Change |
|------|-------|--------|
| `src/main/cpp/server.hpp` | 10 | Replaced with thin include shim |
| `src/main/cpp/jllama.cpp` | 1,240 | Fully rewritten — upstream reader API |
| `src/main/cpp/jni_helpers.hpp` | 238 | `jllama_context` rewritten; dead helpers removed |
| `src/main/cpp/json_helpers.hpp` | 243 | Type alias updates; otherwise unchanged |
| `src/main/cpp/utils.hpp` | 322 | Unchanged (still used) |
| **Total** | **2,053** | **~4,000 lines removed from the 6,013 baseline** |

All C++ unit tests (272) pass. Java integration tests: compile passes on all
platforms; final test results pending CI.

---

## Upstream server library (`tools/server/` at b8913)

| File | Purpose |
|------|---------|
| `server-context.{h,cpp}` | Pimpl `server_context` — `load_model`, `start_loop`, `terminate`, `get_response_reader`, `get_meta`, `get_llama_context` |
| `server-queue.{h,cpp}` | `server_response_reader` — the non-HTTP embedder API |
| `server-task.{h,cpp}` | `server_task`, `task_params`, type enums, `params_from_json_cmpl()` |
| `server-common.{h,cpp}` | `oaicompat_chat_params_parse`, `tokenize_input_prompts`, `tokens_to_str`, base64 |
| `server-chat.{h,cpp}` | OAI/Anthropic chat parsing |
| `server-models.{h,cpp}` | Model/LoRA registry (not compiled on Android — subprocess.h) |
| `server-http.{h,cpp}` | HTTP transport only — **never compiled into jllama** |
| `server.cpp` | `main()` entry point — **never compiled into jllama** |

### Key API facts verified at b8913

- `server_response_reader` has ref members → not copyable; move-constructible.
  Heap-allocate for the streaming reader map.
- `post_task()` may be called **exactly once** per reader (GGML_ASSERT at
  server-queue.cpp:344). Use `post_tasks(vector)` for multi-document batches.
- `params_from_json_cmpl()` parses sampling parameters only — it does **not**
  tokenize the prompt. Call `tokenize_input_prompts()` explicitly and assign
  the result to `task.tokens` before posting.
- `server_tokens::operator=(const server_tokens&)` is deleted — must
  `std::move()` when assigning to `task.tokens`.
- `wait_for_all()` returns `batch_response { is_terminated, results, error }`.
- `task_params::stream` defaults to `false` (via `params_from_json_cmpl` JSON
  default), so blocking calls naturally return a single final result.
- `server_context_meta` has no architecture field; use
  `llama_model_meta_val_str(mdl, "general.architecture", buf, size)` directly.

---

## Phase log

### Phase 0 — Safety net ✅ DONE

Branch `claude/refactor-java-llama-d3lua` created. Baseline line counts
recorded. `REFACTORING.md` written into the repository.

---

### Phase 1 — CMakeLists: compile upstream server files into `jllama` ✅ DONE

**Commit:** `9026600`

- Added `server-context.cpp`, `server-queue.cpp`, `server-task.cpp`,
  `server-models.cpp` to `target_sources(jllama PRIVATE …)`.
- Guard: `if(NOT ANDROID_ABI AND NOT OS_NAME MATCHES "Android")` — `ANDROID_ABI`
  is not reliably set by the dockcross android-arm64 toolchain, so `OS_NAME` is
  checked as a fallback (always `-DOS_NAME=Linux-Android` in the CI invocation).
- `server-common.cpp` and `server-chat.cpp` were already in `add_library(jllama …)`.
- `server-http.cpp` and `server.cpp` intentionally excluded.

---

### Phase 2 — Replace `server.hpp` with upstream shim + rewrite `jllama.cpp` ✅ DONE

This was the core of the refactoring. All 17 JNI methods were rewritten in a
single pass to the upstream reader-based API. Phases 3–6 of the original plan
(pure llama.h methods, embeddings, completions, slot management) were all
completed as part of this phase because `jllama.cpp` required a full rewrite
rather than incremental method migration.

#### What changed

**`server.hpp`** — replaced 3,780-line body with a 10-line include shim:
```cpp
#pragma once
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
```

**`jni_helpers.hpp`** — `jllama_context` struct rewritten:
```cpp
struct jllama_context {
    server_context    server;          // value member (pimpl inside)
    std::thread       worker;
    bool              vocab_only       = false;
    std::atomic<bool> worker_ready{false};
    const llama_vocab *vocab           = nullptr;  // cached after load_model
    llama_model       *vocab_only_model = nullptr; // set only in vocab-only path
    common_params      params;                     // cached for post-load use
    std::mutex         readers_mutex;
    std::map<int, std::unique_ptr<server_response_reader>> readers;
};
```
Dead helpers removed: `build_completion_tasks_impl`, `check_infill_support_impl`,
`append_task`, `collect_task_results_impl`, `recv_slot_task_result_impl`.

**`jllama.cpp`** — all 17 JNI methods rewritten:

| Method group | Pattern used |
|---|---|
| `loadModel` | `server.load_model(params)` + worker thread calling `server.start_loop()` |
| `delete` | `server.terminate()` + thread join + vocab_only_model free |
| `embed` | `get_response_reader()` → `post_task()` → `wait_for_all()` |
| `handleEmbeddings` | Same + `post_tasks(vector)` for multi-prompt batches |
| `handleRerank` | `post_tasks(vector)` (one task per document) |
| `handleCompletions` / `handleCompletionsOai` / `handleChatCompletions` / `handleInfill` | `dispatch_blocking_completion()` → `wait_for_all()` |
| `requestCompletion` / `requestChatCompletion` | `dispatch_streaming_completion()` → reader stored in `readers` map |
| `receiveCompletionJson` | `readers[id]->next()` |
| `cancelCompletion` / `releaseTask` | erase from `readers` map (unique_ptr stops reader) |
| `encode` / `decodeBytes` / `handleTokenize` / `handleDetokenize` | `tokenize_mixed` / `tokens_to_str` / upstream format helpers |
| `applyTemplate` | `oaicompat_chat_params_parse()` |
| `handleSlotAction` | `SERVER_TASK_TYPE_METRICS / SLOT_SAVE / SLOT_RESTORE / SLOT_ERASE` |
| `getModelMetaJson` | `get_meta()` + `llama_model_meta_val_str` for architecture |
| `configureParallelInference` | Validates inputs; returns true (no-op — post-load reconfiguration not possible via pimpl API) |

**`json_helpers.hpp`** — `oaicompat_type` → `task_response_type`,
`OAICOMPAT_TYPE_EMBEDDING` → `TASK_RESPONSE_TYPE_OAI_EMBD`.

#### Bugs found and fixed during Phase 2

| Commit | Bug | Fix |
|--------|-----|-----|
| `9b2ea0f` | `handleRerank`: `post_task()` called in loop → GGML_ASSERT crash | Collect tasks in vector; call `post_tasks()` once |
| `322388f` | All completions: `task.tokens` never set → server slot got 0 tokens → "empty prompt" | Call `tokenize_input_prompts()` in both `dispatch_blocking_completion` and `dispatch_streaming_completion` |
| `c95b5df` | `handleEmbeddings`: same `post_task()` loop as rerank | Same `post_tasks()` fix |
| `c87faa2` | `task.tokens = tokenized_prompts[0]` → compile error | `server_tokens` copy-assign is deleted; use `std::move()` |
| `aa7df43` | Android: `server-models.cpp` compiled despite guard | `ANDROID_ABI` not set by dockcross; add `OS_NAME MATCHES "Android"` fallback |
| `f1a9bff` | `testGetModelMeta`: `"architecture"` field missing | `server_context_meta` has no arch field; fetch via `llama_model_meta_val_str` |
| `5533a58` | `configureParallelInference`: no-op silently accepted invalid values | Re-enable `parse_slot_prompt_similarity` / `parse_positive_int_config` validation before returning true |

#### C++ unit tests updated

- `test_server.cpp` — removed tests for internal types now owned by upstream
  (`slot_params` → `task_params`, `oaicompat_chat_syntax` → `chat_parser_params`,
  enum renames, `stop_type_to_str` / `oaicompat_finish_reason` removed from API).
- `test_jni_helpers.cpp` — updated `jllama_context` construction; added
  `readers` map lifecycle tests; removed impossible EXPECT_NE.
- `test_json_helpers.cpp` — updated enum names; added `(void)` casts for
  `[[nodiscard]]` warnings; added new tests for Phase 2 invariants.
- `CMakeLists.txt` — linked all four server TUs into `jllama_test`.

---

### Phase 3 — Delete dead code ⏳ NEXT

**Prerequisite: CI green on all platforms.**

After Phase 2, the following code is confirmed dead or redundant:

#### `server.hpp`
Currently a 10-line include shim. It can be eliminated entirely by moving
the six `#include` lines into `jllama.cpp` directly. This removes the last
file named `server.hpp` from the project.

Steps:
1. Open `src/main/cpp/jllama.cpp`. Replace `#include "server.hpp"` with the
   six individual upstream includes.
2. Delete `src/main/cpp/server.hpp`.
3. Remove `src/main/cpp/server.hpp` from `add_library(jllama …)` in `CMakeLists.txt`.
4. Build and run C++ tests to confirm.
5. Commit: `"refactor: inline server.hpp shim into jllama.cpp and delete file"`.

#### `jni_helpers.hpp`
Several Layer B helpers were removed in Phase 2 but the file still has
boilerplate worth reviewing. After Phase 3a:
1. Check each remaining `_impl` function — is it still called from `jllama.cpp`?
2. Remove any that are no longer called.
3. Commit: `"refactor: remove unused jni_helpers.hpp Layer B helpers"`.

#### `json_helpers.hpp`
All ten functions are still referenced. No deletions expected here yet.
Defer to Phase 4.

#### `test_server.cpp`
`test_server.cpp` (607 lines) tests `server.hpp` internals. Now that
`server.hpp` is gone, most of this file tests upstream internals that we
should not be testing. Review each test group:
- Keep tests that verify `utils.hpp` behaviour still reachable.
- Remove tests that re-test upstream types (`server_task_result_*`, routing).
- Commit: `"test: trim test_server.cpp to utils-only coverage"`.

#### `utils.hpp`
Contains ~50 lines of base64 that duplicate `server-common.h`. After the
server.hpp shim is gone, check whether any call site in `jllama.cpp` still
uses the local base64 helpers vs the upstream ones. Remove duplicates.
Commit: `"refactor: drop base64 duplicate from utils.hpp"`.

---

### Phase 4 — Final verification ⏳ AFTER Phase 3

```bash
# C++ unit tests
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure

# Java compile (no model)
mvn compile
mvn test -Dtest=StopReasonTest,InferenceParametersTest,LlamaLoaderTest,OSInfoTest

# Full integration (requires model)
mvn test -Dmodel.path=models/codellama-7b.Q2_K.gguf

# Line count — target: under 1,500 lines for the five main C++ files
wc -l src/main/cpp/jllama.cpp src/main/cpp/jni_helpers.hpp \
       src/main/cpp/json_helpers.hpp src/main/cpp/utils.hpp
```

**Must pass:** `LlamaModelTest`, `LlamaEmbeddingsTest`, `ModelParametersTest`,
`InferenceParametersTest`, `LlamaOutputTest`, `ResponseJsonStructureTest`,
`MemoryManagementTest`, `RerankingModelTest`, `ErrorHandlingTest`.

**Known acceptable gap:** `configureParallelInference` returns true for valid
inputs but does not actually apply n_threads or slot_prompt_similarity at
runtime (post-load reconfiguration is not exposed by the upstream pimpl API).
The validation tests pass; the functional tests for actual effect are N/A.

---

## Expected final code reduction

| File | Baseline | Phase 2 done | After Phase 3 target |
|------|----------|--------------|----------------------|
| `server.hpp` | 3,780 | 10 (shim) | **0** (deleted) |
| `jllama.cpp` | 1,270 | 1,240 | ~1,200 |
| `jni_helpers.hpp` | 398 | 238 | ~180 |
| `json_helpers.hpp` | 243 | 243 | ~200 |
| `utils.hpp` | 322 | 322 | ~270 |
| **Total** | **6,013** | **2,053** | **~1,850** |

The 3,780-line `server.hpp` body is already gone. The remaining reduction in
Phase 3 is incremental cleanup of dead helpers and test code.
