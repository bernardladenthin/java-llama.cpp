# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b8953**

## Upgrading CUDA Version

Current CUDA version: **13.2**

To change the CUDA version, update the following **three** places:

1. **`.github/build_cuda_linux.sh`** — Line 10: `sudo dnf install -y cuda-toolkit-13-2`
2. **`.github/build_cuda_linux.sh`** — Line 12: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`
3. **`pom.xml`** — The `<classifier>` tag in the `cuda` jar execution: `cuda13-linux-x86-64`

Also update the header comment in `build_cuda_linux.sh` and the job name in `.github/workflows/release.yaml` for clarity.

Available CUDA versions for RHEL8/Manylinux_2_28 can be browsed at:
```
https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/
```

**Note:** Each CUDA version supports only certain GCC versions. If the dockcross container uses a newer GCC than CUDA supports, the build will fail with `unsupported GNU version`. Check NVIDIA's compatibility table before downgrading CUDA.

Example: To upgrade from 13.2 to a hypothetical 13.3:
```bash
# Edit .github/build_cuda_linux.sh:
#   line 10: cuda-toolkit-13-2 -> cuda-toolkit-13-3
#   line 12: /usr/local/cuda-13.2/bin/nvcc -> /usr/local/cuda-13.3/bin/nvcc
# Edit pom.xml classifier: cuda13-linux-x86-64 (major version only, no need to change for minor bumps)
# Edit CLAUDE.md line: Current CUDA version: **13.2** -> **13.3**
git add .github/build_cuda_linux.sh pom.xml CLAUDE.md
git commit -m "Upgrade CUDA from 13.2 to 13.3"
```

## Optional CUDA build flag (CI feedback-loop workaround)

**Status: temporary — revert when the feedback loop is no longer the bottleneck.**

The `crosscompile-linux-x86_64-cuda` job in `.github/workflows/release.yaml` is the
slowest job in the pipeline (CUDA toolkit install inside dockcross + nvcc compile).
It used to run on every PR, which dominated CI wall time even for changes that had
nothing to do with CUDA.

To shorten the PR feedback loop, the job is now gated behind a `workflow_dispatch`
boolean input named **`enable_cuda_build`** (default `false`):

```yaml
crosscompile-linux-x86_64-cuda:
  if: github.event_name == 'release' || github.event.inputs.enable_cuda_build == 'true'
```

| Trigger | CUDA job runs? |
|---|---|
| `pull_request` | no (skipped — fast feedback) |
| `workflow_dispatch` (defaults) | no |
| `workflow_dispatch` with `enable_cuda_build=true` | yes |
| `release` event | yes (always) |

Two downstream jobs were adjusted to tolerate skipped CUDA:

1. **`package`** — gained `if: always() && !contains(needs.*.result, 'failure') && !contains(needs.*.result, 'cancelled')` so it still runs when CUDA is skipped, and its CUDA-artifact download step is now conditional on `needs.crosscompile-linux-x86_64-cuda.result == 'success'`.

2. **`publish`** — its trigger now also requires `enable_cuda_build=true` for manual dispatches: `github.event_name == 'release' || (release_to_maven_central == 'true' && enable_cuda_build == 'true')`. Otherwise a manual publish would fail mid-step trying to download a non-existent CUDA artifact.

### How to revert

When CI capacity allows running CUDA on every PR again:

1. Delete the `enable_cuda_build` input from the `workflow_dispatch.inputs` block.
2. Remove the `if:` line from the `crosscompile-linux-x86_64-cuda` job (and its
   surrounding 3-line comment).
3. Restore `package` to its original form: drop the `if:` block, drop the
   `if: needs.crosscompile-linux-x86_64-cuda.result == 'success'` line on the
   CUDA-artifact download step.
4. Restore `publish`'s `if:` to the original `github.event_name == 'release' || github.event.inputs.release_to_maven_central == 'true'`.
5. Delete this section from `CLAUDE.md`.

Reference commit that introduced the flag: search the git log for
`enable_cuda_build` on branch `claude/refactor-java-llama-d3lua`.

## Upgrading/Downgrading llama.cpp Version

To change the llama.cpp version, update the following **three** files:

1. **CMakeLists.txt** — the `GIT_TAG` line for llama.cpp: `GIT_TAG        b8831`
2. **README.md** — the badge and link line with the version number
3. **CLAUDE.md** — the "Current llama.cpp pinned version" line

Example: To upgrade from b8808 to b8831:
```bash
# Edit CMakeLists.txt: change GIT_TAG b8808 to b8831
# Edit README.md: change b8808 to b8831 (in both badge and link)
# Edit CLAUDE.md: change b8808 to b8831
git add CMakeLists.txt README.md CLAUDE.md
git commit -m "Upgrade llama.cpp from b8808 to b8831"
git push -u origin <your-branch>
```

**Note:** Always test the build with `cmake -B build && cmake --build build --config Release` after version changes to catch compatibility issues early.

### Inspecting API changes between versions

Use the GitHub compare URL to diff any two llama.cpp builds:

```
https://github.com/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

Example — what changed between b6721 and b6732:
```
https://github.com/ggml-org/llama.cpp/compare/b6721...b6732
```

The GitHub HTML page may time out for large ranges; fall back to the API:
```
https://api.github.com/repos/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

For individual file content at a specific build:
```
https://raw.githubusercontent.com/ggerganov/llama.cpp/b<VERSION>/common/chat.h
```

### Files to check for API compatibility

The three project C++ files (`jllama.cpp`, `server.hpp`, `utils.hpp`) pull in the following
llama.cpp headers. Any of these can introduce breaking changes on upgrade.

**Include dependency graph:**
```
jllama.cpp / server.hpp / utils.hpp
│
├── arg.h ──────────────────────────► common.h ─┐
├── common.h ──────────────────────────────────►├── ggml-opt.h ──► ggml.h
├── chat.h ─────────────► common.h, peg-parser.h └── ggml-backend.h ──► ggml-alloc.h
├── speculative.h ──────► llama.h, common.h
├── sampling.h ─────────► llama.h, common.h
├── download.h ─────────► (stdlib only, no deps)
├── log.h ──────────────► ggml.h
├── llama.h ────────────────────────────────────► ggml.h, ggml-cpu.h, ggml-backend.h, ggml-opt.h
│                                                  └── llama-cpp.h ──► llama.h
├── json-schema-to-grammar.h
├── base64.hpp
├── mtmd.h
└── mtmd-helper.h
```

**Priority-ordered review list for upgrade diffs** (highest break risk first)

The top 8 rows cover all known API-level breaking changes from b5022 → b8831.
For future upgrades, provide diffs for at least these 8 files rather than the full patch.
Also review the project `CMakeLists.txt` for build-system-level breaks (e.g. renamed link targets, new required headers) — those are not visible in header file diffs alone.

| File | What to watch for |
|------|-------------------|
| `common/common.h` | `common_params`/`common_params_speculative` struct fields, `model_alias` container type, `common_init_result` shape, `build_info` symbol (removed in b8831 — now `llama_build_info()` from `build-info.h`) |
| `common/chat.h` | `common_chat_parser_params` (was `common_chat_syntax`), `to_json_oaicompat`, `common_chat_msg_diff_to_json_oaicompat`, `set_tool_call_ids` |
| `common/speculative.h` | `common_speculative_init`, `common_speculative_draft`, `common_speculative_accept` signatures, struct names |
| `tools/mtmd/mtmd.h` | `mtmd_context_params` fields, `image_marker`/`media_marker` API, deprecated symbols (was `common/mtmd.h` before ~b8190) |
| `include/llama-cpp.h` | `common_init_result_ptr` type, access pattern changes (`.get()` vs `->method()`) |
| `common/arg.h` | `n_parallel` sentinel value, what moved to `download.h` across versions |
| `include/llama.h` | Core llama_ function signatures, token types, `llama_model_ptr`, renamed structs |
| `common/download.h` | `common_remote_params` struct, `headers` field format (string vs key-value pair) |
| `common/common.cpp` | Implementation of any inline API used directly |
| `common/speculative.cpp` | Speculative decoding implementation details |
| `common/chat.cpp` | Chat parsing implementation |
| `common/sampling.h` | Sampler API, `common_sampler_*` functions |
| `common/log.h` | Log macro signatures |
| `tools/mtmd/mtmd-helper.h` | Multimodal helper functions |
| `common/json-schema-to-grammar.h` | Grammar API |
| `ggml/include/ggml.h` | `ggml_type` enum values (e.g. `GGML_TYPE_F16`), tensor primitives |
| `ggml/include/ggml-backend.h` | Backend/device abstraction types |
| `ggml/include/ggml-opt.h` | Optimizer params pulled in via `common.h` |

**Safe to skip** (have never caused a break; not used directly by project code):
`common/sampling.h`, `common/log.h`, `tools/mtmd/mtmd-helper.h`, `common/json-schema-to-grammar.h`,
`ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `ggml/include/ggml-opt.h`,
`ggml-alloc.h`, `ggml-cpu.h`, `peg-parser.h`, `base64.hpp`

**Known breaking changes by version range** (b5022 → b8953):

| Version | File | Change |
|---------|------|--------|
| ~b7217–b7433 | `common/common.h`, `include/llama-cpp.h` | `common_init_result` became `common_init_result_ptr`; access changed to `->model()` / `->context()` / `->free_context()` |
| ~b7433 | `common/arg.h` | `n_parallel` default changed to sentinel `-1` (auto); Java bindings must resolve to `1` before model load |
| ~b7217–b7783 | `common/arg.h` → `common/download.h` | `common_remote_get_content` and `common_remote_params` split into new `download.h`; `headers` changed from `vector<string>` to `vector<pair>` |
| ~b7783 | `common/common.h` | `build_info` string moved into `common.h`; local definition must be removed |
| ~b7783–b7858 | `common/chat.h` | `common_chat_syntax` renamed to `common_chat_parser_params`; `to_json_oaicompat<json>()` template removed (no template arg); `ensure_tool_call_ids_set()` → `set_tool_call_ids()` |
| ~b7858–b7864 | `common/speculative.h` | Full redesign: `common_speculative_init(ctx_tgt, ctx_dft)` → `common_speculative_init(params_speculative, ctx)`; `common_speculative_gen_draft` → `common_speculative_draft`; new `common_speculative_accept()`; `common_speculative_params` struct replaced by `common_params_speculative`; draft model loaded via `llama_model_load_from_file` into `llama_model_ptr` |
| ~b7858–b7864 | `common/common.h` | `params_speculative`: `.model.path`/`.hf_repo` replaced by `.has_dft()`/`.mparams_dft`; new `.model_dft` and `.cparams_dft` fields; `speculative.type` enum added (`COMMON_SPECULATIVE_TYPE_NONE`) |
| ~b7858–b7864 | `server.hpp` (internal) | `slot_action.slot_id` → `slot_action.id_slot`; `llama_init_dft` removed from `server_context`; `model_dft` changed from `llama_model*` to `llama_model_ptr`; `slot.ctx_tgt`/`ctx_dft` removed |
| ~b7864 | `common/mtmd.h` | `mtmd_init_params.verbosity` field removed |
| ~b7904–b8190 | `common/common.h` | `params_base.model_alias` changed from `std::string` to a container; use `*model_alias.begin()` instead of direct string cast |
| ~b8778–b8808 | `tools/mtmd/mtmd.h` | `MTMD_DEFAULT_IMAGE_MARKER` macro removed; `mtmd_image_tokens_get_nx/ny` deprecated; new `mtmd_decoder_pos` struct + `mtmd_image_tokens_get_decoder_pos()`; `mtmd_context_params_default()` now sets `image_marker = nullptr` (throws `"custom image_marker is not supported anymore"` if non-null); upstream server adds randomized `get_media_marker()` in `server-common.h` — our `server.hpp` is unaffected since it does not include that header and uses `mtmd_default_marker()` consistently |
| ~b8808–b8831 | project `CMakeLists.txt` | CMake target `common` renamed to `llama-common`; update `target_link_libraries` for `jllama` and `jllama_test` |
| ~b8808–b8831 | `common/common.h` → new `common/build-info.h` | `build_info` `std::string` removed; replaced by `llama_build_info()` (`const char*`) in new `build-info.h`; add `#include "build-info.h"` in `server.hpp` and `utils.hpp`; call sites: `std::string(llama_build_info())` in `server.hpp` (6×), `llama_build_info()` in `jllama.cpp` (1×) and `utils.hpp` (1×) |
| ~b8808–b8831 | `ggml/src/ggml.c` | New `ggml_graph_next_uid()` calls `_InterlockedIncrement64` via `<intrin.h>` on x86; intrinsic unavailable on 32-bit MSVC; fix: `src/main/cpp/compat/ggml_x86_compat.c` provides `__cdecl _InterlockedIncrement64` via `InterlockedIncrement64` (CMPXCHG8B), added to `ggml-base` via `target_sources` guarded by `MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4` |
| ~b8838–b8841 | `src/llama-model.h` | Attention bias fields renamed: `bq`→`wq_b`, `bk`→`wk_b`, `bv`→`wv_b`, `bo`→`wo_b`, `bqkv`→`wqkv_b`; internal to llama.cpp, no impact on this project |
| ~b8841–b8854 | `common/common.h` | `common_params::clear_idle` renamed to `cache_idle_slots`; new `common_context_seq_rm_type` enum + `common_context_can_seq_rm()` replacing `common_speculative_is_compat()`; `get_model_endpoint()` → `common_get_model_endpoint()` |
| ~b8841–b8854 | `tools/mtmd/mtmd.h` + `mtmd-helper.h` | `mtmd_decoder_pos` gains `z` field; `mtmd_image_tokens_get_decoder_pos()` + `mtmd_helper_image_get_decoder_pos()` gain new `pos_0` parameter |
| ~b8841–b8854 | project `utils.hpp` / `server.hpp` | `server_tokens::get_text_tokens()` split: `get_tokens()` returns raw `const llama_tokens &`; new `get_text_tokens()` returns filtered copy (removes `LLAMA_TOKEN_NULL` mtmd placeholders); save/load and context-shift call sites updated to `get_tokens()` |
| ~b8854–b8887 | `common/chat.h` | `common_chat_msg_diff_to_json_oaicompat` removed; moved to `tools/server/server-chat.cpp`; project defines it locally in `server.hpp` — importing server-chat.cpp is impractical because it pulls in `convert_transcriptions_to_chatcmpl` → `get_media_marker` → `server-common.cpp` |
| ~b8854–b8887 | `common/common.h` | `common_params::reasoning_budget` and `reasoning_budget_message` moved into `common_params::sampling` sub-struct as `reasoning_budget_tokens`; update: `params_base.reasoning_budget` → `params_base.sampling.reasoning_budget_tokens` |
| ~b8854–b8887 | `common/fit.h` (new) | `llama_params_fit` and `llama_memory_breakdown_print` removed from `include/llama.h`; now `common_fit_params` / `common_memory_breakdown_print` in new `common/fit.h`; not used directly by project |
| ~b8887–b8913 | `tools/server/server-chat.h` | `convert_transcriptions_to_chatcmpl` gained a new `const common_chat_templates * tmpls` second parameter; not called by project's `server.hpp` — handled automatically by upstream `server-chat.cpp` |
| ~b8887–b8913 | `tools/server/server-task.cpp` | `n_discard` clamped to non-negative: `params.n_discard = std::max(0, params.n_discard)`; applied in project's `server.hpp` after the `json_value` parse |
| ~b8887–b8913 | `tools/server/server-common.cpp` | `parallel_tool_calls` now defaults to `caps["supports_parallel_tool_calls"]` instead of hardcoded `false`; handled automatically by upstream file |
| ~b8887–b8913 | `common/chat.h` | New additive `common_chat_prompt_preset` struct and `common_chat_get_asr_prompt()` function; no project changes required |
| ~b8887–b8913 | `common/common.h` | New `string_starts_with(std::string_view, char)` overload added; no project changes required |
| ~b8887–b8913 | `tools/mtmd/mtmd.cpp` | Added `LLAMA_ROPE_TYPE_NONE` case to rope-type switch; internal fix, no project changes required |
| ~b8913–b8953 | `common/debug.h` | `base_callback_data` renamed to `common_debug_cb_user_data`; template `common_debug_cb_eval<false/true>` replaced by plain `common_debug_cb_eval`; not used by this project |
| ~b8913–b8953 | `tools/server/server-http.h` | New `uploaded_file` struct; `files` map type changed from `map<string, raw_buffer>` to `map<string, uploaded_file>`; upstream server sources compiled directly — no project impact |
| ~b8913–b8953 | `src/llama-quant.cpp` | Default quantization ftype changed from `LLAMA_FTYPE_MOSTLY_Q5_1` to `LLAMA_FTYPE_MOSTLY_Q8_0`; upstream only |
| ~b8913–b8953 | `src/models/llama.cpp`, `qwen3.cpp`, `qwen3moe.cpp` | Removed duplicate `ggml_mul` for `wo_s` scale (now handled exclusively by `build_attn`); upstream only |

## Build Commands

### Java (Maven)
```bash
mvn compile          # Compiles Java and generates JNI headers
mvn test             # Run all tests (requires native library and model files)
mvn package          # Build JAR
mvn test -Dtest=LlamaModelTest#testGenerate  # Run a single test method
```

### Native Library (CMake)
Must run `mvn compile` first to generate JNI headers, then:
```bash
# CPU only
cmake -B build
cmake --build build --config Release

# CUDA (Linux)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (macOS)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

# Optional: enable model downloading via URL
cmake -B build -DLLAMA_CURL=ON
```

Built libraries are placed in `src/main/resources/de/kherud/llama/{OS}/{ARCH}/`.

### Code Formatting
```bash
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp   # Format C++ code
```

## Architecture

### Two-Layer Design

**Java layer** (`src/main/java/de/kherud/llama/`):
- `LlamaModel` — Main API class (AutoCloseable). Wraps native context for inference, embeddings, re-ranking, and tokenization.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes that serialize to JSON (extend `JsonParameters`) for passing to native code.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp. ~1,215 lines; 17 native methods.
- `utils.hpp` — Helper utilities (format helpers, argv stripping, token-piece serialisation).
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- Uses `nlohmann/json` for JSON deserialization of parameters.
- The upstream server library (`server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-models.cpp`) is compiled directly into `jllama` via CMake — there is no hand-ported `server.hpp` fork.

### Native Helper Architecture

The project C++ helpers follow a strict semantic split:

**`json_helpers.hpp`** — Pure data transforms.
- Input: `nlohmann::json`, `server_task_result_ptr`, plain C++ types.
- Output: `json`, `std::vector`, `std::optional`, plain C++ types.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama state (`llama_context*`, `llama_vocab*`, `server_context*` never appear).
- Functions are named without `_impl` suffix — they are the canonical implementation.
- Testable with JSON literals and fake result objects; no JVM and no loaded model required.
- Upstream server headers must be included by the translation unit first (they define `server_task_result_ptr`, `json`, etc.).

Functions: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`,
`parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`,
`parse_slot_prompt_similarity`, `parse_positive_int_config`.

**`jni_helpers.hpp`** — JNI bridge helpers, split into two layers:

*Layer A* (no server headers required): handle management.
- `jllama_context` struct — owns `server_context` (value member, pimpl inside), background
  worker thread, cached `vocab`, saved `params`, and a `readers` map for streaming tasks.
- `get_jllama_context_impl` — reads Java `ctx` handle, returns the `jllama_context*` wrapper.
  Does NOT throw on zero handle (valid no-op for destructor-style calls).
- `require_json_field_impl` — throws `"<field> is required"` if key is absent.
- `jint_array_to_tokens_impl` — reads a Java `int[]` into `std::vector<int32_t>`.

*Layer B* (requires upstream server headers in the TU before `jni_helpers.hpp`): orchestration.
Includes `json_helpers.hpp` so all bridge helpers can call transforms directly.
- `json_to_jstring_impl` — serialises any `json` value to a JNI string via `dump()`.
- `results_to_jstring_impl` — delegates to `results_to_json` then `json_to_jstring_impl`.
- `vec_to_jarray_impl<JArray,JElem,CppElem>` — generic C++ vector → JNI primitive array.
- `embedding_to_jfloat_array_impl` — converts `std::vector<float>` to `jfloatArray`.
- `tokens_to_jint_array_impl` — converts `std::vector<int32_t>` to `jintArray`.

Functions with `_impl` suffix are called directly from `jllama.cpp`.

**Include order rule:**
```
// In jllama.cpp and any TU that uses Layer B helpers:
#include "server-context.h"   // upstream server headers must come first
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "jni_helpers.hpp"    // includes json_helpers.hpp internally
```

**Adding a new pure transform** (e.g. a new JSON field parser):
- Add it to `json_helpers.hpp`. No JNI, no llama types.
- Add tests to `src/test/cpp/test_json_helpers.cpp`.

**Adding a new JNI bridge helper:**
- Add it to `jni_helpers.hpp` in the appropriate layer.
- If it needs upstream server types, put it in Layer B (after the `json_helpers.hpp` include).
- Add tests to `src/test/cpp/test_jni_helpers.cpp`.

### Parameter Flow
Java parameters are serialized to JSON strings and passed to native code, which deserializes them using nlohmann/json. This avoids complex JNI field mapping for the many llama.cpp parameters.

### Native Library Resolution
`LlamaLoader` tries in order:
1. System property `de.kherud.llama.lib.path`
2. `java.library.path`
3. Extracts from JAR resources at `de/kherud/llama/{os}/{arch}/`

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for ARM/Android targets. CI workflows use these for non-x86 Linux builds.

## Testing

### Java tests
Require a model file. The CI downloads models from HuggingFace:
- **LlamaModel tests**: CodeLlama-7B-GGUF (`codellama-7b.Q2_K.gguf`)
- **RerankingModel tests**: Jina-Reranker model

Set the model path via system property or environment variable (see test files for exact property names).

Test files are in `src/test/java/de/kherud/llama/` and `src/test/java/examples/`.

### C++ unit tests

**No JVM and no model file required.** All tests run on pure data structures using mock
objects. The binary is named `jllama_test` and is built by CMake when `BUILD_TESTING=ON`.

#### Commands

```bash
# 1. Configure (once per fresh clone or after CMakeLists.txt changes)
cmake -B build -DBUILD_TESTING=ON

# 2. Build (incremental; -j$(nproc) uses all CPU cores)
cmake --build build --config Release -j$(nproc)

# 3. Run all tests
ctest --test-dir build --output-on-failure

# Count tests across all files
grep -rn "^TEST\b\|^TEST_F\b\|^TEST_P\b" src/test/cpp/ | wc -l

# Run a single named test (GoogleTest filter syntax)
ctest --test-dir build --output-on-failure -R "ResultsToJson"
```

#### Test files

| File | Tests | Scope |
|------|-------|-------|
| `src/test/cpp/test_utils.cpp` | 156 | Upstream helpers: `server_tokens`, `server_grammar_trigger`, `gen_tool_call_id`, `json_value`, `json_get_nested_values`, UTF-8 helpers, `format_response_rerank`, `format_embeddings_response_oaicompat`, `oaicompat_completion_params_parse`, `oaicompat_chat_params_parse`, `are_lora_equal`, `strip_flag_from_argv`, `token_piece_value`, `json_is_array_and_contains_numbers`, `format_oai_sse`, `format_oai_resp_sse`, `format_anthropic_sse` |
| `src/test/cpp/test_server.cpp` | 179 | Upstream result types: `result_timings`, `task_params::to_json()` (incl. `dry_sequence_breakers`, `preserved_tokens`, `timings_per_token`), `completion_token_output`, `server_task_result_cmpl_partial` (non-oaicompat + `to_json_oaicompat` + logprobs + `to_json_oaicompat_chat` + `to_json_anthropic` + dispatcher), `server_task_result_cmpl_final` (non-oaicompat + `to_json_oaicompat` + `to_json_oaicompat_chat` + `to_json_oaicompat_chat_stream` + `to_json_anthropic` + `to_json_anthropic_stream` + tool_calls + dispatcher), `server_task_result_embd`, `server_task_result_rerank`, `server_task_result_metrics`, `server_task_result_slot_save_load`, `server_task_result_slot_erase`, `server_task_result_apply_lora`, `server_task_result_error`, `format_error_response`, `server_task::need_sampling()`, `server_task::n_tokens()`, `server_task::params_from_json_cmpl()` (parsing pipeline + grammar routing + error paths), `response_fields` projection |
| `src/test/cpp/test_json_helpers.cpp` | 42 | All functions in `json_helpers.hpp`: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`, `parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`, `parse_positive_int_config` |
| `src/test/cpp/test_jni_helpers.cpp` | 36 | All functions in `jni_helpers.hpp` using a zero-filled `JNINativeInterface_` mock |

**Current total: 413 tests (all passing).** Branch: `claude/refactor-java-llama-d3lua`.

#### Upstream source location (in CMake build tree)

llama.cpp is fetched via CMake FetchContent, pinned to `GIT_TAG b8953`.

```
build/_deps/llama.cpp-src/tools/server/   ← server-task.h, server-common.h, etc.
build/_deps/llama.cpp-src/include/        ← llama.h, llama-cpp.h
build/_deps/llama.cpp-src/common/         ← common.h, chat.h, arg.h, etc.
```

When reading a `to_json()` implementation to write tests against it, read from:
`build/_deps/llama.cpp-src/tools/server/server-task.cpp`

#### Mock JNI pattern used in test_jni_helpers.cpp

```cpp
// Zero-fill the interface so all unpatched fn pointers are nullptr
JNINativeInterface_ iface = {};
// Patch only the stubs this test needs, e.g.:
iface.GetLongField  = [](JNIEnv*, jobject, jfieldID) -> jlong { return some_handle; };
iface.ThrowNew      = [](JNIEnv*, jclass, const char*) -> jint { return 0; };
// Wire up the env
JNIEnv_ fake_env = {};
fake_env.functions = &iface;
JNIEnv *env = &fake_env;
```

Any stub that is called but not patched will crash (null function pointer) — deliberately,
so missing stubs are caught immediately rather than silently.

#### How to add a new C++ test

1. Open the appropriate `src/test/cpp/test_*.cpp`:
   - Pure JSON transform → `test_json_helpers.cpp`
   - JNI helper → `test_jni_helpers.cpp`
   - Upstream result type `to_json()` → `test_server.cpp`
   - `utils.hpp` function or upstream utility → `test_utils.cpp`
2. Add a `TEST(SuiteName, TestName) { ... }` block using GoogleTest macros.
3. Rebuild: `cmake --build build --config Release -j$(nproc)`
4. Run: `ctest --test-dir build --output-on-failure`
5. Commit with message summarising coverage added and new test total.

#### Finding untested code paths

```bash
# List all functions defined in a header
grep -n "^inline\|^static\|^\[\[nodiscard\]\]" src/main/cpp/utils.hpp

# Check which functions already have tests
grep -n "function_name" src/test/cpp/*.cpp

# Find all fields in an upstream to_json() method
grep -n "\"field_name\"" build/_deps/llama.cpp-src/tools/server/server-task.cpp

# Check which JSON fields Java actually reads (important: must test these)
grep -rn "field_name" src/main/java/de/kherud/llama/
```

#### Testing complex scenarios — methodology

Simple tests verify individual field values on a default-constructed struct.
Complex tests verify **control flow**: switch dispatchers, cross-cutting flags, and
multi-step parameter pipelines.  The same build/run/commit loop applies.

**1. Dispatcher (switch) coverage**

Every `to_json()` that is a switch on `res_type` has one test per arm:

```cpp
// Pattern: set is_updated=true, set res_type, call to_json(), check the
// distinguishing field that differs between arms.
server_task_result_cmpl_final f;
f.is_updated = true;
f.stream     = false;
f.res_type   = TASK_RESPONSE_TYPE_OAI_CMPL;
// ... set required fields ...
const json j = f.to_json();
EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
```

The same pattern handles the `stream` flag fork inside `OAI_CHAT`:
`stream=false` → single object with `"object":"chat.completion"`;
`stream=true`  → JSON array of chunks with `"object":"chat.completion.chunk"`.

**2. Cross-cutting flag interaction**

Some flags (verbose, include_usage, timings.prompt_n) cut across multiple formatters.
Test each flag in one formatter only — they share the same code path:

```cpp
// verbose=true must add __verbose to the first chunk/top-level object
f.verbose = true;
EXPECT_TRUE(j.contains("__verbose"));

// timings absent when prompt_n < 0 (default), present when >= 0
f.timings.prompt_n = 5;
EXPECT_TRUE(j.contains("timings"));
```

**3. Parameter parsing (`params_from_json_cmpl`) without a model**

`server_task::params_from_json_cmpl(vocab, params_base, n_ctx_slot, logit_bias_eog, data)`
can be called with `nullptr` vocab **if the JSON does not trigger grammar/preserved_tokens
tokenisation** (those are the only vocab-dependent paths).  This lets us test the full
parsing pipeline including error throws:

```cpp
common_params          params_base;
std::vector<llama_logit_bias> no_bias;
const int n_ctx = 512;

// test: repeat_last_n=-1 is expanded to n_ctx_slot
json data = {{"repeat_last_n", -1}};
auto p = server_task::params_from_json_cmpl(nullptr, params_base, n_ctx, no_bias, data);
EXPECT_EQ(p.sampling.penalty_last_n, n_ctx);

// test: invalid value throws std::runtime_error
json bad = {{"dry_sequence_breakers", json::array()}};  // empty → error
EXPECT_THROW(server_task::params_from_json_cmpl(nullptr, params_base, n_ctx, no_bias, bad),
             std::runtime_error);
```

**4. Array-returning formatters**

Some methods (e.g. `to_json_oaicompat_chat_stream()`) return a JSON array of event objects,
not a single object.  Check with `is_array()` first, then iterate or index:

```cpp
const json j = f.to_json_oaicompat_chat_stream();
ASSERT_TRUE(j.is_array());
ASSERT_GE(j.size(), 1u);
// Last chunk always has a non-null finish_reason
EXPECT_FALSE(j.back().at("choices")[0].at("finish_reason").is_null());
```

**5. `response_fields` projection**

`to_json_non_oaicompat()` supports a projection list via `response_fields`.
When non-empty, only those dot-separated paths survive:

```cpp
f.response_fields = {"content", "tokens_predicted"};
const json j = f.to_json_non_oaicompat();
EXPECT_TRUE(j.contains("content"));
EXPECT_FALSE(j.contains("stop_type"));  // filtered out
```

## Key Constraints

- **Java 11+** required.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.
