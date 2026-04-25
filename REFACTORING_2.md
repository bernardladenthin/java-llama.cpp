# Refactoring 2: Closing the `configureParallelInference` gap (b8913)

> **Continuation of `REFACTORING.md`.** That document delivered the
> ~4,200-line cleanup that replaced `server.hpp` with upstream sources.
> This second pass audits the remaining 1,806-line surface against
> upstream b8913 and closes the small follow-up items left behind.

---

## Why

After REFACTORING.md landed, four observations remained:

1. `handleInfill` checked FIM-token availability through three direct
   `llama_vocab_fim_*` calls, even though the very next read of model
   state in the same function went through `server_context::get_meta()`.
   Inconsistent style — both paths return the same data because
   upstream populates `server_context_meta::fim_*_token` from those
   exact `llama_vocab_fim_*` calls.

2. `format_infill` and `format_rerank` in `utils.hpp` were hand-ported
   copies of upstream `format_prompt_infill` / `format_prompt_rerank`.
   Duplicated logic with no semantic divergence.

3. The two index/tokens loop bodies in `handleRerank` and
   `handleEmbeddings` were copy-paste twins.

4. **`configureParallelInference` was a documented no-op.** The JNI
   handler validated the JSON shape and threw on out-of-range values,
   but applied nothing because the affected fields lived inside
   `server_context_impl` (pimpl) and there was no public setter.
   Three configurable fields — `n_threads`, `n_threads_batch`,
   `slot_prompt_similarity` — all silently ignored after the model was
   loaded.

(1)–(3) are pure cleanup. (4) is a behaviour change that closes a real
gap.  This document covers all four.

---

## Baseline (post-REFACTORING.md, on `main`)

| File | Lines |
|------|------:|
| `src/main/cpp/jllama.cpp`       | 1,215 |
| `src/main/cpp/utils.hpp`        | 199 |
| `src/main/cpp/jni_helpers.hpp`  | 196 |
| `src/main/cpp/json_helpers.hpp` | 196 |
| **Total**                       | **1,806** |

---

## Done — five commits on branch `claude/analyze-refactoring-changes-ovOFq`

### Commit 1 — replace `format_infill` with upstream `format_prompt_infill`

Deleted the 96-line local `format_infill` from `utils.hpp`. The upstream
signature (`server-common.h:356`) and implementation
(`server-common.cpp:1440`) are byte-for-byte identical. `handleInfill`
now calls `format_prompt_infill` directly.

### Commit 2 — replace `format_rerank` with upstream `format_prompt_rerank`

Deleted the 19-line local `format_rerank` from `utils.hpp`. Rewrote
`handleRerank`:

- Removed manual `tokenize_mixed` of the query and `tokenize_input_prompts`
  per-document loop — upstream's `format_prompt_rerank` calls
  `tokenize_input_subprompt` internally.
- Pass raw query/document strings directly. `task.tokens` now receives
  `server_tokens` from upstream.

Behavioural delta: models shipping a `rerank` chat template (some Jina
v3 variants) now use it via `llama_model_chat_template(model, "rerank")`
instead of always producing `[BOS]q[EOS][SEP]doc[EOS]`. Models without
a rerank template (including the CI Jina-Reranker) produce bit-identical
output.

Also pruned two now-unused includes from `utils.hpp`
(`build-info.h` and `mtmd-helper.h` — only needed by the deleted
formatters).

### Commit 3 — extract `build_indexed_token_task` helper

`handleRerank` and `handleEmbeddings` both built a `vector<server_task>`
with the same id / tokens / index / res_type assignment pattern.
Extracted into a single named helper; reduced each loop body to one call.
Pattern is now testable in isolation.

### Commit 4 — route `handleInfill` FIM-token check through `server_context_meta`

Replaced the three `llama_vocab_fim_pre/suf/mid` calls with the
equivalent fields on `server_context_meta` (`fim_pre_token`,
`fim_sub_token`, `fim_mid_token`). Same data path; upstream populates
these fields from the identical `llama_vocab_fim_*` calls
(`server-context.cpp:3120`). Brings the FIM check into the same idiom
the function already used three lines below for `meta.slot_n_ctx`.

### Commit 5 — apply runtime `n_threads` / `n_threads_batch` in `configureParallelInference`

Rewrote the JNI handler. Previously a validate-and-no-op stub; now it
actively reconfigures the live `llama_context` via the public C API
`llama_set_n_threads(ctx, n, nb)` (`include/llama.h:946`). The setter
requires both values, so a single-field update fills the missing one
from the cached `common_params` captured at load time. The cache is
written back so a follow-up partial update reads the just-applied value
instead of the original.

`slot_prompt_similarity` remains validated but **not yet applied** —
the field is private inside `server_context_impl` and upstream b8913
exposes no setter. The future call site is reserved as a commented
block in the handler:

```cpp
// if (slot_sim_opt.has_value()) {
//     ctx_server->set_slot_prompt_similarity(*slot_sim_opt);
// }
```

The proposed upstream patch is tracked in `llama-cpp.patch.md` (the
companion doc). Once it is merged into a tagged llama.cpp build and
this repo's pin is bumped, the comment becomes a live call and the
gap closes for the third field.

This commit also fixes a build regression: commit 2 dropped the
`build-info.h` include from `utils.hpp`, but `jllama.cpp` still calls
`llama_build_info()` at line 668. The include is now added explicitly
to `jllama.cpp` (it was previously transitive through `utils.hpp`).

---

## Reduction table

| File | Pre-2 | Post-2 | Δ |
|------|------:|-------:|---:|
| `src/main/cpp/jllama.cpp`       | 1,215 | 1,259 | +44 |
| `src/main/cpp/utils.hpp`        |   199 |    74 | −125 |
| `src/main/cpp/jni_helpers.hpp`  |   196 |   196 | 0 |
| `src/main/cpp/json_helpers.hpp` |   196 |   196 | 0 |
| **Total**                       | **1,806** | **1,725** | **−81** |

`jllama.cpp` grows by 44 lines because Commit 5 *adds function* (real
thread setter + cache write-back + nullptr guard + reserved comment for
the future similarity setter) where there used to be a documented no-op.
This is the correct outcome — REFACTORING_2 is not solely a size win;
it closes a known gap.

Combined with REFACTORING.md (which removed 4,207 lines from the 6,013
baseline), the project now sits at **1,725 / 6,013 = 71% reduction**
from the original C++ surface, with one less behavioural gap.

---

## Tests

C++ unit tests: **413 passing** (unchanged across all five commits).

Java integration tests:
- `ConfigureParallelInferenceTest` — 11 existing tests still pass
  unmodified. Their assertions were "no exception thrown" / "throws
  LlamaException for out-of-range" — both contracts still hold. The
  new behaviour (threads actually applied) is observable but not
  asserted by these tests.
- `LlamaModelTest`, `RerankingModelTest`, `LlamaEmbeddingsTest` — all
  green; the upstream formatter swap in Commit 2 produces bit-identical
  output for models without a rerank template.

---

## Investigated and kept (with reasoning)

The audit also reviewed every other helper in `utils.hpp`,
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
"general.architecture", ...)` manually because `server_context_meta`
does not include an architecture field at b8913. An upstream feature
request to add `architecture` to the meta struct would let us delete
this 5-line dance, but is deferred — low value, requires upstream
coordination.

---

## Forward references

- `llama-cpp.patch.md` — the proposed upstream PR adding
  `server_context::get_slot_prompt_similarity()` /
  `set_slot_prompt_similarity()`.  When that lands and the pin moves
  past b8913, a tiny follow-up commit on this repo uncomments the
  reserved block in `configureParallelInference` and removes the gap
  note.
- Future llama.cpp upgrades may surface new helpers worth re-auditing
  on each `b<NEW>` bump. The CLAUDE.md upgrade procedure is unchanged.
