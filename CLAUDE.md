# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b8778**

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

## Upgrading/Downgrading llama.cpp Version

To change the llama.cpp version, update the following **three** files:

1. **CMakeLists.txt** — Line 28: `GIT_TAG        b5022`
2. **README.md** — Line 2: Badge and link with version number
3. **CLAUDE.md** — Line 9: This documentation

Example: To upgrade from b5016 to b5022:
```bash
# Edit CMakeLists.txt, line 28: change b5016 to b5022
# Edit README.md, line 2: change b5016 to b5022 (in both badge and link)
# Edit CLAUDE.md, line 9: change b5016 to b5022
git add CMakeLists.txt README.md CLAUDE.md
git commit -m "Upgrade llama.cpp from b5016 to b5022"
git push -u origin claude/upgrade-llama-cpp-b4927-EaJcb
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

**Priority-ordered review list** (highest break risk first):

| File | What to watch for |
|------|-------------------|
| `include/llama.h` | Core llama_ function signatures, token types, `llama_model_ptr`, renamed structs |
| `include/llama-cpp.h` | C++ smart pointer types: `llama_model_ptr`, `common_init_result_ptr`; access pattern changes (`.get()` vs `->method()`) |
| `common/common.h` | `common_params` / `common_params_speculative` struct fields, `model_alias` type, `common_init_result` shape |
| `common/common.cpp` | Implementation of any inline API used directly |
| `common/speculative.h` | `common_speculative_init`, `common_speculative_draft`, `common_speculative_accept` signatures |
| `common/speculative.cpp` | Speculative decoding implementation details |
| `common/chat.h` | `common_chat_parser_params` (was `common_chat_syntax`), `to_json_oaicompat`, `common_chat_msg_diff_to_json_oaicompat`, `set_tool_call_ids` |
| `common/chat.cpp` | Chat parsing implementation |
| `common/arg.h` | Parameter parsing; check what moved to `download.h` across versions |
| `common/download.h` | `common_remote_params` struct, `headers` field format (string vs key-value pair) |
| `common/sampling.h` | Sampler API, `common_sampler_*` functions |
| `common/log.h` | Log macro signatures |
| `common/mtmd.h` | Multimodal API, `mtmd_init_params` fields |
| `common/mtmd-helper.h` | Multimodal helper functions |
| `common/json-schema-to-grammar.h` | Grammar API |
| `ggml/include/ggml.h` | `ggml_type` enum values (e.g. `GGML_TYPE_F16`), tensor primitives |
| `ggml/include/ggml-backend.h` | Backend/device abstraction types |
| `ggml/include/ggml-opt.h` | Optimizer params pulled in via `common.h` |

**Safe to skip** (stable leaf headers, not used directly by project code):
`ggml-alloc.h`, `ggml-cpu.h`, `peg-parser.h`, `base64.hpp`

**Known breaking changes by version range** (b5022 → b8190):

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
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp.
- `server.hpp` — Inference server logic (adapted from llama.cpp's server).
- `utils.hpp` — Helper utilities.
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- Uses `nlohmann/json` for JSON deserialization of parameters.

### Native Helper Architecture

The project C++ helpers follow a strict semantic split:

**`json_helpers.hpp`** — Pure data transforms.
- Input: `nlohmann::json`, `server_task_result_ptr`, plain C++ types.
- Output: `json`, `std::vector`, `std::optional`, plain C++ types.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama state (`llama_context*`, `llama_vocab*`, `server_context*` never appear).
- Functions are named without `_impl` suffix — they are the canonical implementation.
- Testable with JSON literals and fake result objects; no JVM and no loaded model required.
- Requires `server.hpp` to be included by the translation unit first (TU convention — `server.hpp` has no include guard).

Functions: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`,
`build_embeddings_response_json`, `extract_first_embedding_row`, `parse_encoding_format`,
`extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`,
`parse_positive_int_config`.

**`jni_helpers.hpp`** — JNI bridge helpers, split into two layers:

*Layer A* (no `server.hpp` required): handle management.
- `jllama_context` struct — owns `server_context*` and background worker thread.
- `get_server_context_impl` — reads Java `ctx` handle, throws on null.
- `get_jllama_context_impl` — like above but returns the wrapper (delete path only).
- `require_single_task_id_impl` — validates exactly one task ID was created.
- `require_json_field_impl` — throws `"<field> is required"` if key is absent.
- `jint_array_to_tokens_impl` — reads a Java `int[]` into `std::vector<int32_t>`.

*Layer B* (requires `server.hpp` in the TU before `jni_helpers.hpp`): server orchestration.
Includes `json_helpers.hpp` so all bridge helpers can call transforms directly.
- `json_to_jstring_impl` — serialises any `json` value to a JNI string.
- `build_completion_tasks_impl` — tokenises prompt and populates `server_task` vector.
- `recv_slot_task_result_impl` — receives one slot result, throws on error.
- `collect_task_results_impl` — receives all results for a task-id set, throws on error.
- `results_to_jstring_impl` — delegates to `results_to_json` then `json_to_jstring_impl`.
- `check_infill_support_impl` — validates FIM prefix/suffix/middle tokens present.
- `append_task` — constructs and appends a `server_task` of a given type.
- `embedding_to_jfloat_array_impl` — converts `std::vector<float>` to a Java `jfloatArray`; throws OOM on allocation failure.
- `tokens_to_jint_array_impl` — converts `std::vector<int32_t>` to a Java `jintArray`; throws OOM on allocation failure.

Functions with `_impl` suffix have a thin module-level wrapper in `jllama.cpp`; functions
without the suffix (in `json_helpers.hpp`) are called directly.

**Include order rule:**
```
// In jllama.cpp and any TU that uses Layer B helpers:
#include "server.hpp"     // must come first — no include guard
#include "jni_helpers.hpp"  // includes json_helpers.hpp internally
```

**Adding a new pure transform** (e.g. a new JSON field parser):
- Add it to `json_helpers.hpp`. No JNI, no llama types.
- Add tests to `src/test/cpp/test_json_helpers.cpp`.

**Adding a new JNI bridge helper:**
- Add it to `jni_helpers.hpp` in the appropriate layer.
- If it needs `server.hpp` types, put it in Layer B (after the `json_helpers.hpp` include).
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
No JVM or model file required. Built as `jllama_test` via CMake when `BUILD_TESTING=ON`.

| File | What it tests |
|------|---------------|
| `test_json_helpers.cpp` | All functions in `json_helpers.hpp` — pure JSON transforms, using fake result objects |
| `test_jni_helpers.cpp` | All functions in `jni_helpers.hpp` — mock `JNIEnv`, pre-seeded `server_response` queue |
| `test_server.cpp` | Selected `server.hpp` internals (result types, error formatting, routing helpers) |
| `test_utils.cpp` | Utilities from `utils.hpp` |

Run C++ tests:
```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## Key Constraints

- **Java 11+** required.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.
