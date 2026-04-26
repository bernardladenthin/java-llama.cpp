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

## Current state (after Pass 1 and Pass 2)

| File | Lines | Change |
|------|------:|--------|
| `src/main/cpp/server.hpp` | 0 | **Deleted** — includes inlined directly (Pass 1) |
| `src/main/cpp/jllama.cpp` | 1,259 | Fully rewritten (Pass 1) + runtime n_threads applied (Pass 2) |
| `src/main/cpp/jni_helpers.hpp` | 196 | `jllama_context` rewritten; dead helpers removed (Pass 1) |
| `src/main/cpp/json_helpers.hpp` | 196 | Type alias updates; stale comments fixed (Pass 1) |
| `src/main/cpp/utils.hpp` | 74 | Base64 copy + dead slot macros removed (Pass 1); `format_infill` + `format_rerank` replaced by upstream (Pass 2) |
| **Total** | **1,725** | **~4,288 lines removed from the 6,013 baseline (71%)** |

413 C++ unit tests pass. Java integration tests pass on all platforms
(Linux, macOS, Windows, Android).

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

## Pass 1 — `server.hpp` removal (branch `claude/refactor-java-llama-d3lua`)

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

### Phase 3 — First dead-code pass ✅ DONE

**Commits:** `0a5a396`, `c19ccfe`

#### What was done

**`server.hpp` deleted** (`0a5a396`):
- The 10-line include shim was the last remnant of the old `server.hpp`.
- Replaced by inlining its 6 upstream includes directly into `jllama.cpp`
  and all 3 test TUs.
- Removed from `add_library(jllama …)` in `CMakeLists.txt`.
- Updated stale comments in `jni_helpers.hpp`, `test_jni_helpers.cpp`,
  `test_json_helpers.cpp`, `test_server.cpp`.

**Dead code removed from `utils.hpp` and tests** (`c19ccfe`):
- Deleted 46-line `base64_decode` copy (tested-only, not used in production).
- Removed `#include "base64.hpp"` (the `base64::` class was never called).
- Removed `SLT_*` / `QUE_*` macro overrides (workarounds for old `server.hpp`
  slot layout; jllama.cpp never calls these macros).
- Removed corresponding `Base64Decode.*` test cases from `test_utils.cpp`.
- Fixed stale "server.hpp" include-order comment in `json_helpers.hpp`.

**`test_server.cpp` header updated** (same commit):
- Removed stale "collect_task_results_impl() is tested in test_jni_helpers.cpp".
- Rewritten to accurately describe the file as upstream API regression coverage.

---

### Phase 4 — Upstream API migration (embeddings) ✅ DONE

`embed` and `handleEmbeddings` migrated to use `dynamic_cast<server_task_result_embd*>`
for direct struct access, removing the JSON-roundtrip extraction path.

Deleted from `json_helpers.hpp`: `extract_first_embedding_row`, `build_embeddings_response_json`.
Deleted from `test_json_helpers.cpp`: 15 tests for those two functions.

Test count after: 409 tests (−15 from Phase 3 total).

---

### Phase 5 — Second dead-code pass ✅ DONE

**Commits:** `71485d5`, and follow-up cleanup commit.

Functions confirmed dead (zero callers in `jllama.cpp`) and deleted:

| Symbol | File | Reason |
|--------|------|--------|
| `format_logit_bias` | `utils.hpp` | Replaced by upstream `format_logit_bias_oaicompat` |
| `parse_lora_request(base, data)` | `utils.hpp` | 2-arg wrapper; upstream 1-arg version is called directly |
| `require_single_task_id_impl` | `jni_helpers.hpp` | Streaming now uses per-task `server_response_reader` objects |
| `get_server_context_impl` | `jni_helpers.hpp` | All production code uses `get_jllama_context_impl` instead |
| `#include <iostream>` | `jllama.cpp` | Unused after rewrite |
| `#include "download.h"` | `utils.hpp` | `common_remote_*` not used in utils.hpp |
| `#include <random>` | `utils.hpp` | No random number generation in utils.hpp |

Deleted tests: 10 (`FormatLogitBias`×3, `ParseLoraRequest`×7) + 5 (`GetServerContext_*`×4, contrast test×1) = 15 tests removed.

Test count after: **413 tests**.

---

### Phase 6 — Duplication elimination ✅ DONE

**Commit:** `95cbe55`

A `find-cpp-duplication` audit identified five recurring patterns across
`jllama.cpp`. All extracted into named helpers:

| Helper | Pattern absorbed | Sites |
|--------|------------------|-------|
| `result_ok_or_throw(env, result)` | 4-line single-result null/error guard | 4 |
| `batch_ok_or_throw(env, br)` | 3-line batch-error guard | 4 |
| `dispatch_one_shot_task(env, ctx, task)` | reader → post → wait → check → return-json pipeline; absorbed `exec_slot_file_task`'s body and both inline switch arms in `handleSlotAction` | 3 |
| `populate_completion_task(task, jctx, ...)` | identical tokenize+`params_from_json_cmpl` block in streaming and blocking dispatch | 2 |
| Wrapper removal | thin `results_to_jstring` / `json_to_jstring` / `jint_array_to_tokens` forwarders deleted; all 12 call sites now invoke the `_impl` versions directly (matching the architecture rule already documented in CLAUDE.md) | 12 |

Net change: **−35 lines** in `jllama.cpp` (1,250 → 1,215). Tests: 413 still passing.

---

### Phase 7 — Final verification ✅ DONE

```bash
# C++ unit tests
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j$(nproc)
ctest --test-dir build --output-on-failure

# Java compile (no model)
mvn compile
mvn test -Dtest=StopReasonTest,InferenceParametersTest,LlamaLoaderTest,OSInfoTest

# Full integration (requires model)
mvn test -Dmodel.path=models/codellama-7b.Q2_K.gguf

# Line count
wc -l src/main/cpp/jllama.cpp src/main/cpp/jni_helpers.hpp \
       src/main/cpp/json_helpers.hpp src/main/cpp/utils.hpp
```

**Must pass:** `LlamaModelTest`, `LlamaEmbeddingsTest`, `ModelParametersTest`,
`InferenceParametersTest`, `LlamaOutputTest`, `ResponseJsonStructureTest`,
`MemoryManagementTest`, `RerankingModelTest`, `ErrorHandlingTest`.

**Known acceptable gap (after Pass 1, partially closed in Pass 2):**
`configureParallelInference` originally returned true for valid inputs but did
not actually apply any of `n_threads`, `n_threads_batch`, or
`slot_prompt_similarity` at runtime. Pass 2 wired `n_threads` and
`n_threads_batch` through the public `llama_set_n_threads` API.
`slot_prompt_similarity` remains validated-only — the field is private inside
`server_context_impl` (pimpl) and upstream b8913 exposes no setter. See
`llama-cpp.patch.md` for the proposed upstream patch and the Pass 2 section
below.

---

---

## Pass 2 — Second-pass deduplication at b8913 (branch `claude/analyze-refactoring-changes-ovOFq`)

After Pass 1 landed, four observations remained:

1. `handleInfill` checked FIM-token availability through three direct
   `llama_vocab_fim_*` calls, even though the very next read of model state in
   the same function went through `server_context::get_meta()`. Inconsistent
   style — both paths return the same data because upstream populates
   `server_context_meta::fim_*_token` from those exact `llama_vocab_fim_*`
   calls.
2. `format_infill` and `format_rerank` in `utils.hpp` were hand-ported copies
   of upstream `format_prompt_infill` / `format_prompt_rerank`. Duplicated
   logic with no semantic divergence.
3. The two index/tokens loop bodies in `handleRerank` and `handleEmbeddings`
   were copy-paste twins.
4. **`configureParallelInference` was a documented no-op.** The JNI handler
   validated the JSON shape and threw on out-of-range values, but applied
   nothing because the affected fields lived inside `server_context_impl`
   (pimpl) and there was no public setter. Three configurable fields —
   `n_threads`, `n_threads_batch`, `slot_prompt_similarity` — all silently
   ignored after the model was loaded.

(1)–(3) are pure cleanup. (4) is a behaviour change that closes a real gap.

### Pass-2 commits

| # | Commit | Summary |
|--:|--------|---------|
| 1 | `0fe1620` | Replace `format_infill` with upstream `format_prompt_infill`. Deleted 96 lines from `utils.hpp`. Byte-for-byte identical signature/implementation to upstream `server-common.h:356` / `server-common.cpp:1440`. |
| 2 | `e2c6d04` | Replace `format_rerank` with upstream `format_prompt_rerank`. Deleted 19 lines from `utils.hpp`; rewrote `handleRerank` to pass raw query/document strings (upstream's `tokenize_input_subprompt` is called internally). Pruned now-unused `build-info.h` and `mtmd-helper.h` includes. **Behavioural delta:** models without a rerank chat template (incl. CI Jina-Reranker) now receive the canonical `[BOS?] q [EOS?] [SEP?] doc [EOS?]` token sequence (each token gated by `llama_vocab_get_add_*`) instead of the doubled-BOS/EOS sequence produced by the previous `tokenize_input_prompts(..., add_special=true) + format_rerank` chain. The new sequence matches what upstream's HTTP `/rerank` endpoint emits. Logit magnitudes can dip slightly negative for poorly-matched (query, document) pairs (the doubled-token format previously kept them just inside `[0, 1]`). Test contracts updated in commits 6/7 below. Models shipping a dedicated rerank chat template (some Jina v3 variants) additionally pick it up via `llama_model_chat_template(model, "rerank")`. |
| 3 | `a9d42f4` | Extract `build_indexed_token_task` helper. Both `handleRerank` and `handleEmbeddings` built the same `id`/`tokens`/`index`/`res_type` task pattern; collapsed to one named helper. Pattern is now testable in isolation. |
| 4 | `100b419` | Route `handleInfill` FIM-token check through `server_context_meta`. Replaced three `llama_vocab_fim_pre/suf/mid` calls with `meta.fim_pre_token` / `meta.fim_sub_token` / `meta.fim_mid_token`. Upstream populates these from the identical `llama_vocab_fim_*` calls (`server-context.cpp:3120`). Same data path; brings the FIM check into the same `get_meta()` idiom the function already used for `meta.slot_n_ctx`. |
| 5 | `d051b1c` | Apply runtime `n_threads` / `n_threads_batch` in `configureParallelInference`. Rewrote the JNI handler from validate-and-no-op stub to active reconfigurer of the live `llama_context` via the public C API `llama_set_n_threads(ctx, n, nb)` (`include/llama.h:946`). The setter requires both values, so a single-field update fills the missing one from the cached `common_params` captured at load time; the cache is written back so a follow-up partial update reads the just-applied value. `slot_prompt_similarity` remains validated but **not yet applied** — the field is private inside `server_context_impl`, upstream b8913 exposes no setter. The future call site is reserved as a commented block in the handler, ready to uncomment once the upstream patch in `llama-cpp.patch.md` lands and the pin advances past b8913. This commit also fixes a build regression: commit 2 dropped the `build-info.h` include from `utils.hpp`, but `jllama.cpp:668` still calls `llama_build_info()`; include is now added explicitly to `jllama.cpp`. |
| 6 | `718fbd1` | Fix `RerankingModelTest.testRerankScoreRange` for canonical-format scores. Previous assertion required `score >= 0.0f && score <= 1.0f` based on a wrong mental model — that rerank scores are probabilities. They are not: upstream returns the raw classification-head logit `embd[0]` (`server-context.cpp` `send_rerank()`) with no sigmoid applied. The CI failure on Ubuntu/macOS reported `-0.0013` for a weakly-matched (query, document) pair, which is correct behaviour with the canonical token sequence. Two changes: (a) loosen `testRerankScoreRange` — drop the `[0, 1]` bound, keep NaN/Inf, add `|score| < 10` magnitude bound; (b) add `testRerankSpreadAndSign_canonicalFormatSentinel` — sentinel that asserts the canonical format produces a plausible logit spread and that document sign tracks relevance (ML doc > 0; Paris doc < machine doc). Initial spread bound was `> 0.3` (calibrated in commit 7). |
| 7 | `3b6c47b` | Calibrate sentinel spread threshold from empirical CI run. The first version of the sentinel in commit 6 used `> 0.3` based on a guess. CI on Ubuntu and macOS reported the actual spread on `jina-reranker-v1-tiny-en-Q4_0`: `0.19975` (Ubuntu) / `0.19972` (macOS) — bit-identical modulo quantisation rounding, well below the guessed 0.3. Lowered to `> 0.1`. Comfortably below the observed ~0.20, well above any plausible noise-floor cluster. The other two sentinel assertions (`mlScore > 0`, `parisScore < machineScore`) already passed unchanged. |

### Pass-2 reduction

| File | After Pass 1 | After Pass 2 | Δ |
|------|-------------:|-------------:|---:|
| `src/main/cpp/jllama.cpp`       | 1,215 | 1,259 | +44 |
| `src/main/cpp/utils.hpp`        |   199 |    74 | −125 |
| `src/main/cpp/jni_helpers.hpp`  |   196 |   196 | 0 |
| `src/main/cpp/json_helpers.hpp` |   196 |   196 | 0 |
| **Total**                       | **1,806** | **1,725** | **−81** |

`jllama.cpp` grows by 44 lines because commit 5 *adds function* (real thread
setter + cache write-back + nullptr guard + reserved comment for the future
similarity setter) where there used to be a documented no-op. Pass 2 is
not solely a size win; it closes a known gap.

### Investigated and kept (with reasoning)

The Pass 2 audit also reviewed every other helper in `utils.hpp`,
`json_helpers.hpp`, `jni_helpers.hpp`, and `jllama.cpp`. The following
helpers stay:

| Symbol | Why we keep it |
|--------|----------------|
| `str_to_bytes`, `token_piece_value` (`utils.hpp`) | Upstream's `completion_token_output::str_to_bytes` returns `vector<unsigned char>` for binary serialisation; ours returns a `json::array` of integers for the `/tokenize` wire format. Different contracts. |
| `format_tokenizer_response`, `format_detokenized_response` (`utils.hpp`) | Trivial 1-line wrappers, but extracted named helpers are preferred over inlined literals. No upstream equivalent. |
| `strip_flag_from_argv` (`utils.hpp`) | No upstream equivalent; well-isolated; well-tested. |
| `parse_encoding_format`, `extract_embedding_prompt` (`json_helpers.hpp`) | Upstream has the same logic but inline at `server-context.cpp:4263-4272` / `4249-4261`. Not exposed as free functions. |
| `is_infill_request` (`json_helpers.hpp`) | No upstream equivalent — upstream uses HTTP route splitting (`post_infill` vs `post_completion`) instead of body-content sniffing. |
| `parse_slot_prompt_similarity`, `parse_positive_int_config` (`json_helpers.hpp`) | Config validators specific to `configureParallelInference`. |
| `results_to_json`, `rerank_results_to_json` (`json_helpers.hpp`) | Project-specific output shapes that the Java client depends on. |
| All JNI plumbing (`jllama.cpp` cache, attach/detach, log forwarding, dispatch helpers) | JNI-specific; no upstream equivalent possible. |
| Vocab-only mode | Project feature; not in upstream. |

`getModelMetaJson` still calls `llama_model_meta_val_str(mdl,
"general.architecture", ...)` manually because `server_context_meta` does not
include an architecture field at b8913. An upstream feature request to add
`architecture` to the meta struct would let us delete this 5-line dance, but
is deferred — low value, requires upstream coordination.

### Forward references

- `llama-cpp.patch.md` — proposed upstream PR adding
  `server_context::get_slot_prompt_similarity()` /
  `set_slot_prompt_similarity()`. When that lands and the pin moves past
  b8913, a tiny follow-up commit on this repo uncomments the reserved block
  in `configureParallelInference` and removes the gap note above.
- Future llama.cpp upgrades may surface new helpers worth re-auditing on
  each `b<NEW>` bump. The CLAUDE.md upgrade procedure is unchanged.

---

## Code reduction achieved (combined Pass 1 + Pass 2)

| File | Baseline | After Pass 1 | After Pass 2 | Total reduction |
|------|---------:|-------------:|-------------:|----------------:|
| `server.hpp`        | 3,780 | **0** (deleted)  | **0**  | 3,780 |
| `jllama.cpp`        | 1,270 | 1,215 | 1,259 (+44 for real `llama_set_n_threads` apply) | 11 |
| `jni_helpers.hpp`   |   398 |   196 |   196 | 202 |
| `json_helpers.hpp`  |   243 |   196 |   196 | 47 |
| `utils.hpp`         |   322 |   199 |    74 | 248 |
| **Total**           | **6,013** | **1,806** | **1,725** | **4,288 lines (71%)** |

The 3,780-line `server.hpp` was the dominant cost. The codebase is now a thin
JNI wrapper over the upstream server library with no duplicated logic.
Pass 2 (`utils.hpp` 199 → 74) finished the deduplication of formatters and
closed two of three `configureParallelInference` fields; the third is
parked behind the upstream patch tracked in `llama-cpp.patch.md`.
