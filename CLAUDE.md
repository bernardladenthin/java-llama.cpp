# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b9264**

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

**Known breaking changes by version range** (b5022 → b9022):

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
| ~b8953–b8962 | `common/common.h` | `struct cpu_params` → `struct common_cpu_params`; `cpu_get_num_physical_cores()` → `common_cpu_get_num_physical_cores()`; `cpu_get_num_math()` → `common_cpu_get_num_math()`; not used directly by project |
| ~b8953–b8962 | `common/common.h` | `common_params_speculative` fully restructured with nested sub-structs: `.mparams_dft`/`.model_dft`/`.cparams_dft`/`.n_max`/`.n_min`/`.p_split`/`.p_min` → `.draft.mparams`/`.draft.model`/`.draft.cparams`/`.draft.n_max`/`.draft.n_min`/`.draft.p_split`/`.draft.p_min`; ngram fields moved to `.ngram_cache`/`.ngram_mod`/`.ngram_simple`/etc sub-structs; not referenced by project directly |
| ~b8953–b8962 | `common/arg.h` | `is_sparam` bool split into `is_sampling` + `is_spec`; `set_sparam()` split into `set_sampling()` + `set_spec()`; not used by project |
| ~b8953–b8962 | `tools/server/server-task.cpp` | `task_params::to_json()` drops `"speculative.n_max"`, `"speculative.n_min"`, `"speculative.p_min"` from output; only `"speculative.type"` remains; test `SlotParamsToJson.SpeculativeFields_Present` updated accordingly |
| ~b8953–b8962 | `common/speculative.h` | New public API: `common_speculative_n_max()` and `common_speculative_n_min()` added; server-context.cpp uses these instead of direct field access; no project changes required |
| ~b8962–b8982 | `common/sampling.h` | `common_sampler_accept` 3rd param renamed `accept_grammar` → `is_generated`; semantics broadened: `false` now also skips reasoning budget update (not just grammar); no project call sites affected |
| ~b8962–b8982 | `common/reasoning-budget.h` | Two overloads merged: `prefill_tokens` variant removed; new single overload takes `initial_state = REASONING_BUDGET_IDLE`; prefill now fed via `llama_sampler_accept()` loop after init; not called directly by project |
| ~b8962–b8982 | `ggml/src/ggml-cuda/ssm-conv.cuh` | `ggml_cuda_op_ssm_conv` gained optional `bias_add_node` param; `SSM_CONV + ADD + SILU` fusion now supported; internal CUDA code, no project changes required |
| ~b8962–b8982 | `common/speculative.cpp` | Draft token confidence check (`p_min`) moved before push to result: low-confidence tokens are now discarded entirely rather than included then ignored; behavior fix, no project changes required |
| ~b8962–b8982 | `tools/server/server-context.cpp` | `n_draft_total` accounting moved to draft generation site instead of acceptance site (bug fix); upstream only |
| ~b8982–b8994 | `ggml/src/ggml-cuda.cu` | `ggml_backend_cuda_i` struct: `.get_tensor_2d_async` and `.set_tensor_2d_async` function pointers were swapped (get pointed to set impl and vice versa); corrected; internal CUDA backend, no project changes required |
| ~b8982–b8994 | `ggml/src/ggml-vulkan.cpp` | `ggml_vk_buffer_write_2d_async` and `ggml_vk_buffer_write_2d` gained a `dpitch` parameter; Vulkan now implements `set_tensor_2d`/`get_tensor_2d` in buffer interface; internal backend code, no project changes required |
| ~b8982–b8994 | `common/speculative.cpp` | Checkpoint helpers renamed: `draft_create_checkpoint` → `create_checkpoint`, `draft_restore_checkpoint` → `restore_checkpoint`; `ckpt_size` field removed (size computed from context directly); internal speculative module, not called by project |
| ~b8982–b8994 | `common/arg.cpp` | CLI option typo fixed: `--spec--draft-p-split` → `--spec-draft-p-split` (extra dash removed); CLI-only, no project changes required |
| ~b8982–b8994 | `src/llama-mmap.cpp` | Windows large-file (>2 GB) fix: `ftell`/`fseek` replaced with `_ftelli64`/`_fseeki64`; upstream only |
| ~b8982–b8994 | `tools/server/httplib.h` | cpp-httplib bumped to v0.43.2: Windows `FILE_SHARE_WRITE` fix, Linux DNS cancel race fix, mbedTLS `close_notify` fix; upstream server header, no project changes required |
| ~b8982–b8994 | `tools/server/server-context.cpp` | New `LLAMA_TRACE` env variable enables slot acceptance tracing; upstream only |
| ~b8994–b9004 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `vk_fa_pipeline_state` gains `k_type`/`v_type` fields; `get_fa_tuning_params_coopmat2` now takes separate `k_type`/`v_type` params; mixed K/V type FA pipeline creation refactored to `CREATE_FA_CM2_MIXED()` macro; `flash_attn_cm2.comp` shader uses runtime `FaTypeK`/`FaTypeV` spec constants (spec constants 12–15 added); `DECODEFUNC`/`NEEDS_INIT_IQ_SHMEM` macros removed; internal Vulkan backend, no project changes required |
| ~b8994–b9004 | `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp` | `get_mul_mat_fast_pipeline` vectorized-path condition fixed: `dst->ne[1] % 4 == 0` check removed (was preventing vectorization for non-multiple-of-4 batch sizes); internal WebGPU backend, no project changes required |
| ~b8994–b9004 | `ggml/src/ggml-hexagon/` | Hexagon HTP backend: FA `exp2` half-precision option, unary-op non-contiguous tensor fix; internal DSP backend, no project changes required |
| ~b8994–b9004 | `tools/server/webui/` | Major frontend component reorganization (Svelte/TypeScript); purely UI, no C++ or JNI impact |
| ~b9004–b9016 | `src/llama-io.h` | `llama_io_read_i` interface changed: `read(size_t)→read(void*,size_t)`, `read_to(void*,size_t)` removed, new `read_tensor(tensor,offset,size)` added; `llama_io_write_buffer`/`llama_io_read_buffer` now batch backend tensor ops in destructors for performance; internal state-save/load path, not called by project |
| ~b9004–b9016 | `tools/server/server-context.cpp` | Static `server_get_checkpoint()` (returns by value) renamed to `server_prompt_checkpoint_update()` (takes `server_prompt_checkpoint &` by reference, in-place update); compiled directly into jllama, no call site in project code |
| ~b9004–b9016 | `common/arg.cpp` + docs | Speculative decoding CLI args renamed: `--draft`/`--draft-n`/`--draft-max` and `--draft-min`/`--draft-n-min` were **REMOVED** (handler `throw`s `std::invalid_argument` at parse time, not just deprecated); other draft flags (`--draft-p-min`, `--ctx-size-draft`, `--device-draft`, `--gpu-layers-draft`, `--model-draft`) kept as aliases for new canonical `--spec-draft-*` names. **Java impact**: `ModelParameters.setDraftMax`/`setDraftMin` produced removed flags → threw at model load; fixed to canonical `--spec-draft-n-max`/`--spec-draft-n-min`. Other `set*Draft` methods updated to canonical names for forward compatibility. Env vars also renamed (`LLAMA_ARG_DRAFT_MAX`→`LLAMA_ARG_SPEC_DRAFT_N_MAX`, etc.) |
| ~b9004–b9016 | `ggml/src/ggml-cuda/ggml-cuda.cu` | PCI bus ID detection replaced `snprintf` with `cudaDeviceGetPCIBusId` (buffer 16→32 bytes); HIP/MUSA compat headers gain `cudaDeviceGetPCIBusId` alias; internal CUDA backend |
| ~b9004–b9016 | `ggml/src/ggml-opencl/` | Adreno MoE MXFP4: new `kernel_convert_block_mxfp4_trans4_ns`/`restore` kernels in `cvt.cl`; new `gemm_moe_mxfp4_f32_ns`, `gemv_moe_mxfp4_f32_ns`, `moe_reorder_b`, `moe_sort_by_expert` kernel files; GPU-side router reorder replaces CPU-side preprocessing; `q_img` created for GEMM path; internal OpenCL backend |
| ~b9004–b9016 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `GGML_VK_MAX_NODES 8192` macro removed (node limit now determined differently); internal Vulkan backend |
| ~b9004–b9016 | `ggml/src/ggml-webgpu/` | `ggml_webgpu_row_norm_pipeline_key` gains `src_type`/`dst_type` fields; `GGML_OP_NORM` now supported alongside `GGML_OP_RMS_NORM`/`GGML_OP_L2_NORM`; `row_norm.wgsl` gains SRC_TYPE/DST_TYPE parameterization and NORM two-pass algorithm; internal WebGPU backend |
| ~b9004–b9016 | `src/llama-model.cpp` | `rope_yarn_log_mul` `get_key` call changed from `required=0.0f` to `required=false`; fixes Mistral YaRN log_mul loading; internal model loading, no project impact |
| ~b9004–b9016 | `common/chat.cpp` | `common_chat_templates_generation_prompt()` extracted from `common_chat_templates_apply_jinja()`; internal refactor, no API change |
| ~b9016–b9022 | `src/llama-model.h` + `src/llama-model.cpp` + `src/models/` | `llama_model` becomes abstract base with pure virtual methods (`load_stats`, `load_hparams`, `load_vocab`, `load_tensors`, `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`); `load_arch()` removed; new intermediate `llama_model_base` class provides concrete implementations; per-arch subclasses (e.g. `llama_model_llama`, `llama_model_gemma2`) in `src/models/`; factory `llama_model_create(llm_arch, params)` and `llama_model_create(ml, params)` replace direct instantiation; `LLAMA_LOAD_LOCALS` convenience macro added; public C API (`llama_model_load_from_file` etc.) unchanged — no project impact |
| ~b9016–b9022 | `src/models/` | Many model files renamed: `cohere2-iswa.cpp`→`cohere2.cpp`, `gemma2-iswa.cpp`→`gemma2.cpp`, `gemma3n-iswa.cpp`→`gemma3n.cpp`, `gemma4-iswa.cpp`→`gemma4.cpp`, `mimo2-iswa.cpp`→`mimo2.cpp`, `openai-moe-iswa.cpp`→`openai-moe.cpp`, `pangu-embedded.cpp`→`pangu-embed.cpp`, `qwen3vl-moe.cpp`→`qwen3vlmoe.cpp`, `step35-iswa.cpp`→`step35.cpp`; new model files added (`deepseek2ocr.cpp`, `glm-dsa.cpp`, `granite-moe.cpp`, `hunyuan-vl.cpp`, `jina-bert-v2/v3.cpp`, `lfm2moe.cpp`, `llama-embed.cpp`, `mamba2.cpp`, `minicpm.cpp`, `mistral4.cpp`, `nemotron-h-moe.cpp`, `nomic-bert.cpp`, `nomic-bert-moe.cpp`, `phimoe.cpp`); upstream only, no project changes required |
| ~b9016–b9022 | `tools/server/server-context.cpp` | `server_prompt_checkpoint_update` (the renamed function from b9016) static function signature changed from returning by value to taking `server_prompt_checkpoint &` by reference; compiled directly into jllama, no project call site |
| ~b9016–b9022 | `tools/server/server-tools.cpp` | New built-in `get_datetime` tool added via new `server_tool_get_datetime` struct in `build_tools()`; no project changes required (handled automatically by compiled upstream source) |
| ~b9016–b9022 | `common/chat-auto-parser-generator.cpp` | `force_tools` variable removed from `build_tool_parser_json_native`, `build_tool_parser_tag_json`, `build_tool_parser_tag_tagged`; content before tool calls is now always `p.optional(p.content(...))` regardless of `tool_choice=required`; upstream only, no project changes required |
| ~b9016–b9022 | `common/chat-peg-parser.h/cpp` | New `optspace(const std::string & tag)` method added to `common_chat_peg_builder`; makes leading/trailing spaces in reasoning tags optional; upstream only, no project changes required |
| ~b9016–b9022 | `common/reasoning-budget.cpp` | Forced token logit now set to `+INFINITY` (previously left at whatever the model computed); reasoning budget enforcement is now absolute; upstream only, no project changes required |
| ~b9016–b9022 | `common/chat.cpp` | `thinking_start_tag` and `thinking_end_tag` now trimmed via `trim_whitespace()`; upstream only, no project changes required |
| ~b9016–b9022 | `examples/diffusion/` | `diffusion_generate` extracted from `diffusion-cli.cpp` to new `diffusion.h`/`diffusion.cpp` static library; enum names prefixed: `ORIGIN`→`DIFFUSION_ALGORITHM_ORIGIN`, `TIMESTEP_BASED`→`DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED` etc.; examples only, no project changes required |
| ~b9022–b9049 | `include/llama.h` | New `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2` macro added alongside existing `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1`; enables on-device KV cache state save/restore without host round-trip via `llama_state_seq_get_size_ext`/`get_data_ext`/`set_data_ext`; no project call-site changes required (not used by JNI layer) |
| ~b9022–b9049 | `src/llama-context.cpp` | State seq data format breaking change: `llama_state_seq_get_data`/`set_data` now prepend a 4-byte magic (`0xaf143cd8`) + 4-byte `seq_id` header; state data saved with ≤b9022 is **incompatible** with b9049+; internal I/O classes renamed `llama_io_write_buffer`→`llama_io_write_host`, `llama_io_read_buffer`→`llama_io_read_host`; new `llama_io_write_device`/`llama_io_read_device` classes for on-device paths; no project changes required (not called by JNI layer) |
| ~b9022–b9049 | `ggml/include/ggml.h` | New `ggml_op_hint` enum (`GGML_HINT_DEFAULT=0`, `GGML_HINT_SRC0_IS_HADAMARD=1`) and `ggml_mul_mat_set_hint()` function added for FWHT (Fast Walsh-Hadamard Transform) support; used internally in `llama-graph.cpp` / `llama-kv-cache.cpp`; no project call-site changes required |
| ~b9022–b9049 | `src/llama.cpp` | `llama_backend_init()` now auto-calls `ggml_backend_load_all()` if no backends are yet registered; `ggml_backend_load_all()` removed from `common_params_parser_init()` (was in `common/arg.cpp`); no project changes required — backend loading still happens correctly |
| ~b9022–b9049 | `tools/server/server-context.cpp` | `server_prompt_checkpoint_update()` gained an `on_device` bool parameter; speculative checkpoints now use `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY \| LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`; compiled directly into jllama from upstream source — no project call-site changes required |
| ~b9022–b9049 | `src/llama-model.cpp` | Unsupported model architecture now throws `std::runtime_error` instead of calling `GGML_ABORT`; allows callers to catch unknown-arch errors gracefully; no project changes required |
| ~b9022–b9049 | `ggml/CMakeLists.txt` | GGML version bumped 0.10.2 → 0.11.0; no project changes required |
| ~b9022–b9049 | `vendor/cpp-httplib/` | Updated to 0.43.3: `str2tag` converted to iterative loop (eliminates recursion stack depth risk), `res.body.reserve` now OOM-safe; upstream server header, no project changes required |
| ~b9049–b9071 | `common/chat.h` | `contains_media()` method added to `common_chat_msg`; `to_json_oaicompat()` now forces text concatenation when message contains media markers; additive change, no project impact |
| ~b9049–b9071 | `src/llama-arch.h/cpp` + `src/llama-hparams.h` | New `LLM_KV_ATTENTION_VALUE_SCALE` KV key and `f_attn_value_scale` hparam field added for MiMo-V2 attention value scaling; additive, no project changes required |
| ~b9049–b9071 | `src/llama.cpp` | `llama_supports_gpu_offload()` and `llama_supports_rpc()` now auto-call `ggml_backend_load_all()` if no backends are registered; behavior fix, no project changes required |
| ~b9049–b9071 | `src/llama-context.cpp` | `state_seq_set_data`: removed too-strict seq_id matching guard that was gated on `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`; KV slot restorer now checks tensor shapes and view offsets before deciding to reallocate (avoids unnecessary realloc on shape-compatible updates); both are bug fixes, no project API changes required |
| ~b9049–b9071 | `src/models/mimo2.cpp` | MiMo-V2 extended with MTP (Multi-Token Prediction) layer support via `nextn_predict_layers`; fused `wqkv` projection; `attention_value_scale` post-attention scaling; all internal model-loading changes, no project changes required |
| ~b9049–b9071 | `ggml/src/ggml-sycl/` | SYCL implementations added for `CUMSUM`, `DIAG`, `FILL`, `SSM_SCAN`, `SOLVE_TRI` ops; additive, no project changes required |
| ~b9049–b9071 | `ggml/src/ggml-cuda/out-prod.cu` | CUDA outer-product uses `cublasSgemmStridedBatched` for batched path (dps2==1, ne2>1); HIP/MUSA compat headers gain the alias; performance improvement, no project changes required |
| ~b9049–b9071 | `tools/mtmd/` | MiniCPM-V 4.6 multimodal support added (`PROJECTOR_TYPE_MINICPMV4_6`, ViT merger graph, new tensor names); additive, no project changes required |
| ~b9049–b9071 | `tools/server/webui/` | LLM-based conversation title generation; CSS animation `fill-mode-forwards` fixes; UI-only changes compiled into upstream server, no project changes required |
| ~b9071–b9094 | `ggml/src/ggml-cuda/allreduce.cu` + `allreduce.cuh` (NEW) | 2-GPU PCIe AllReduce pipeline for tensor parallelism (no NVLink required); requires Volta+ (sm70+); enabled via `GGML_CUDA_ALLREDUCE` env var (`nccl`/`internal`/`none`); compiled automatically via FetchContent, no project changes required |
| ~b9071–b9094 | `ggml/src/ggml-cuda/snake.cu` + `snake.cuh` (NEW) | Fused CUDA Snake activation kernel (`y = x + sin(a*x)^2 * inv_b`) for BigVGAN/Vocos audio models; fuses 5-op chain `MUL→SIN→SQR→MUL→ADD` at graph level; F32/F16/BF16; compiled automatically, no project changes required |
| ~b9071–b9094 | `ggml/src/ggml-cuda/ggml-cuda.cu` | Flash attention head size 192 (DKQ=192, DV=128) for MiMo-V2.5/V2.5-Pro/V2-Flash with GQA ratio 8/16; multi-GPU comm context refactored to `ggml_backend_cuda_comm_context` with `try_allreduce` function pointer; PCI bus IDs lowercased; compiled automatically, no project changes required |
| ~b9071–b9094 | `ggml/src/ggml-sycl/` | Q5_K reordered memory layout + MMVQ kernel for Intel GPUs; PAD op supports non-contiguous src0; dedicated growing K/V buffer for flash attention; all internal SYCL backend, no project changes required |
| ~b9071–b9094 | `ggml/src/ggml-hexagon/` | GATED_DELTA_NET and L2_NORM HVX-vectorized on Hexagon HTP backend; internal DSP backend, no project changes required |
| ~b9071–b9094 | `src/models/sarvam.cpp` (NEW) | Sarvam-MoE model (`sarvamai/sarvam-30b`); reuses BailingMoeV2 arch; new vocab pre-type `LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE = 51`; additive, no project changes required |
| ~b9071–b9094 | `src/models/gemma4.cpp` | Gemma4 split gate/up experts: `ffn_gate_up_exps` now TENSOR_NOT_REQUIRED; fallback to separate `ffn_gate_exps`/`ffn_up_exps`; NVFP4 per_expert_scale folding; internal model-loading, no project changes required |
| ~b9071–b9094 | `tools/server/server-context.h` + `server-context.cpp` | New `get_model_info()` method on `server_context`; `/v1/models` response now includes `"n_ctx"` field (value: `slot_n_ctx`); compiled from upstream sources, no JNI changes required (Java callers of model info APIs receive the new field transparently) |
| ~b9071–b9094 | `tools/server/server-http.h` + `server.cpp` | `handlers` map moved from private to public in `server_http_context`; new `register_gcp_compat()` method exposes GCP/Vertex AI Prediction Protocol endpoint reading `AIP_MODE`/`AIP_PREDICT_ROUTE`/`AIP_HEALTH_ROUTE`/`AIP_HTTP_PORT` env vars; compiled from upstream sources, no project changes required |
| ~b9071–b9094 | `tools/server/server-models.h` + `server.cpp` | Router child→parent model info propagation: new `CMD_CHILD_TO_ROUTER_INFO` command; `setup_child_server()` gains `const json & model_info` parameter; new `update_loaded_info()` method; `server_model_meta` gains `loaded_info` field; all internally consistent across compiled upstream sources, no project changes required |
| ~b9071–b9094 | `common/reasoning-budget.cpp` | Forced token logit no longer set to `+INFINITY`; only competing tokens set to `-INFINITY`; internal sampler behavior change, no project changes required |
| ~b9071–b9094 | `tools/server/webui/` | Settings registry refactored (`settings-config.ts`/`settings-fields.ts`/`settings-sections.ts` merged into `settings-registry.ts`); MCP route `#/settings/mcp` → `#/mcp-servers`; settings route `/settings/chat/[section]` → `/settings/[[section]]`; UI-only, no project changes required |
| ~b9094–b9102 | `ggml/src/ggml-cuda/allreduce.cu` + `allreduce.cuh` | Internal CUDA AllReduce pipeline refactored with `ggml_cuda_ar_pipeline` struct; `ggml_cuda_ar_pipeline_init(devices, n_devices)` / `_free` / `_allreduce` APIs; supports 2-GPU PCIe AllReduce without NCCL (Volta+ / sm70+); chunked kernel path (small tensors) vs copy-engine path (large tensors); `GGML_CUDA_ALLREDUCE` env = `nccl`/`internal`/`none`; env tuning vars `GGML_CUDA_AR_COPY_THRESHOLD` / `GGML_CUDA_AR_COPY_CHUNK_BYTES` / `GGML_CUDA_AR_BF16_THRESHOLD`; HIP/MUSA builds return nullptr stub; compiled automatically via FetchContent, no project changes required |
| ~b9094–b9102 | `ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_LOG_WARN_ONCE` macro added; `ggml_backend_cuda_comm_context` gains `try_allreduce` fn pointer and `ar_pipeline`; three dispatch fns: `try_allreduce_nccl`, `try_allreduce_internal`, `try_allreduce_butterfly`; init chain: `comm_init_nccl` → `comm_init_internal` → `comm_init_none`; platform default Linux→NCCL, Windows→internal; no project changes required |
| ~b9094–b9102 | `ggml/src/ggml-sycl/ggml-sycl.cpp` + `im2col.cpp` + `im2col.hpp` | New `ggml_sycl_im2col_3d` function; `GGML_OP_IM2COL_3D` now supported on Intel GPU via SYCL; 2D im2col kernel rewritten with tile-based `IC_KH_KW` thread decomposition; new `SYCL_IM2COL_BLOCK_SIZE 256`; additive, no project changes required |
| ~b9094–b9102 | `ggml/CMakeLists.txt` | GGML version patch bumped 0.11.0 → 0.11.1; no project changes required |
| ~b9094–b9102 | `common/sampling.cpp` | Bug fix in `common_sampler_sample`: `set_logits` now called at the top before backend-sampling check; backend sampling token-selection now scans all of `cur_p.data` to find matching token (instead of artificial 1-element array), fixing `cur_p.selected` for downstream `n_probs`; post-sampling probabilities now work correctly with backend sampling |
| ~b9094–b9102 | `tools/server/server-context.cpp` | `need_logits` renamed to `need_pre_sample_logits`; only set when `n_probs > 0 && !post_sampling_probs`; backend sampling now works with `post_sampling_probs`; 0.0-probability tokens filtered from `result.probs`; compiled from upstream, no project JNI changes required |
| ~b9094–b9102 | `src/llama-model.cpp` | `n_vocab` loading moved from `llama_model_base::load_hparams()` to per-model `load_arch_hparams()` (e.g. `src/models/deepseek2.cpp`, `src/models/llama.cpp`); internal model-loading refactor, no project changes required |
| ~b9094–b9102 | `src/llama-model.cpp` | `ggml/src/ggml-virtgpu/ggml-backend-device.cpp` gains `#include <mutex>` for `std::once_flag`; internal backend fix, no project changes required |
| ~b9094–b9102 | `vendor/cpp-httplib/httplib.cpp` + `httplib.h` | Security fix: chunk-size parsing replaced `strtoul` with manual hex-digit scanning to prevent overflow and reject invalid chunk extensions; version bumped to 0.43.4; compiled automatically, no project changes required |
| ~b9102–b9103 | `vendor/cpp-httplib/httplib.cpp` + `httplib.h` | cpp-httplib bumped to v0.44.0: (1) RFC 9110 §5.5 compliance — header field values are no longer percent-decoded by the recipient in `parse_header`; `Location`/`Referer` special-casing removed; callers that need URI-component decoding must call `decode_uri_component()` explicitly; (2) `ThreadPool` constructor is now exception-safe — if thread creation fails partway through, already-started workers are signalled to exit and joined before rethrowing, preventing `std::terminate` from joinable threads in the destructor; compiled automatically, no project changes required |
| ~b9103–b9106 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` + Vulkan shaders | Vulkan flash attention refactored: `pipeline_flash_attn_f32_f16` changed from a per-type array of maps to a single map; mixed K/V quant types (e.g. Q4_0 K + F16 V) now supported on all Vulkan FA paths (scalar, cm1, cm2) rather than coopmat2 only; per-type SPIR-V variants replaced by two generic modules (`flash_attn_f32_f16` and `flash_attn_f32_f16_int8`) that select K/V type at runtime via `FaTypeK`/`FaTypeV` spec constants; new `flash_attn_dequant.glsl` contains aliased SSBO views and an uber `dequantize4()` switch; the K/V type mismatch guard removed from `ggml_backend_vk_device_supports_op`; internal Vulkan backend refactor, no project changes required |
| ~b9103–b9106 | `ggml/src/ggml-cuda/argsort.cu` | Added `#include <cuda/iterator>` for CCCL ≥ 3.1 strided-iterator path; internal CUDA backend, no project changes required |
| ~b9103–b9106 | `convert_hf_to_gguf.py` | Mistral Medium 3.5 mmproj support: `n_embd_text` now reads `"dim"` key instead of `"hidden_dim"`; negative `img_break_tok_id` placeholders resolved from `tekken.json` or `tokenizer.json`; conversion tool only, no project changes required |
| ~b9106–b9134 | `common/arg.cpp` | CLI option `--spec-draft-ctx-size` / `-cd` / `--ctx-size-draft` REMOVED — throws `std::invalid_argument` at parse time; `ModelParameters.setCtxSizeDraft()` removed; no replacement (context size now managed internally by speculative engine) |
| ~b9106–b9134 | `common/arg.cpp` | CLI option `--spec-draft-replace` / `--spec-replace` REMOVED — throws `std::invalid_argument` at parse time; no corresponding Java method existed |
| ~b9106–b9134 | `common/speculative.h` | Full redesign: `common_speculative_type` enum values renamed `DRAFT`&#x2192;`DRAFT_SIMPLE`, `EAGLE3`&#x2192;`DRAFT_EAGLE3`; `common_params_speculative.type` (single enum) &#x2192; `.types` (vector); `common_speculative_n_max()` / `common_speculative_n_min()` REMOVED; new `common_speculative_init(params, n_seq)` no longer takes ctx; new `common_speculative_begin(spec, seq_id, prompt)`, `common_speculative_draft(spec)`, `common_speculative_accept(spec, seq_id, n)`, `common_speculative_process(spec, batch)` signatures; `common_speculative_draft_params` struct added; server sources compiled directly, no project JNI changes required |
| ~b9106–b9134 | `common/common.h` | New `common_prompt_checkpoint` struct (contains `data_tgt` + `data_dft`) replaces the old `server_prompt_checkpoint` in `server-task.h`; compiled from upstream server sources, no project JNI changes required |
| ~b9106–b9134 | `tools/server/server-task.cpp` | `task_params::to_json()` renamed field `"speculative.type"` &#x2192; `"speculative.types"` (now serialises the vector); test `SlotParamsToJson.SpeculativeFields_Present` updated accordingly |
| ~b9106–b9134 | `include/llama.h` | New `LLAMA_STATE_SEQ_FLAGS_NONE = 0` macro added; additive, no project changes required |
| ~b9134–b9145 | `tools/server/server-common.cpp` | New `continue_final_message` boolean request field in `oaicompat_chat_params_parse`; vLLM/transformers-compatible alias for the prefill-assistant heuristic — when `true`, the last assistant message is extended without appending an end-of-turn token; mutually exclusive with `add_generation_prompt=true` (throws 400); compiled from upstream server sources; `InferenceParameters.setContinueFinalMessage(boolean)` added |
| ~b9134–b9145 | `ggml/src/ggml-sycl/` | Level Zero API integration for SYCL device memory allocation (`GGML_SYCL_SUPPORT_LEVEL_ZERO` build option, `GGML_SYCL_ENABLE_LEVEL_ZERO` runtime env); reduces system RAM usage on Intel dGPUs; internal SYCL backend, no project changes required |
| ~b9134–b9145 | `ggml/src/ggml-opencl/` | Q5_0 and Q5_1 MoE GEMM/GEMV kernels added for Adreno (Qualcomm) GPUs; internal OpenCL backend, no project changes required |
| ~b9134–b9145 | `ggml/src/ggml-cuda/allreduce.cu` | AllReduce accumulation now routed through `float` intermediate for precision (avoids BF16 truncation); internal CUDA backend, no project changes required |
| ~b9134–b9145 | `ggml/src/ggml-hexagon/` | `GGML_UNARY_OP_TANH` added to Hexagon HTP backend; internal DSP backend, no project changes required |
| ~b9134–b9145 | `ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp` | `use_subgroup_matrix` condition now also checks `sg_mat_k > 0 && sg_mat_n > 0` and alignment; prevents crash on devices reporting subgroup matrix support with zero k/n; internal WebGPU backend, no project changes required |
| ~b9145–b9150 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Bug fix: `mul_mat_l_int[i]` / `mul_mat_m_int[i]` / `mul_mat_s_int[i]` / `mul_mat_id_l_int[i]` / `mul_mat_id_m_int[i]` / `mul_mat_id_s_int[i]` were unconditionally set to `true` instead of mirroring the actual device pipeline capabilities from `mul_mat_l[i]` etc.; now properly initialized; internal Vulkan backend bug fix, no project changes required |
| ~b9145–b9150 | `src/unicode.cpp` | New `unicode_regex_split_custom_qwen35()` function registered for the Qwen 3.5 tokenizer regex pattern; uses `[\p{L}\p{M}]+` letter-plus-combining-mark runs vs. Qwen2's `\p{L}+`; additive internal tokenizer change, no project changes required |
| ~b9145–b9150 | `ggml/src/ggml-cpu/ggml-cpu-riscv64-spacemit/` | SpaceMIT RISC-V IME backend major refactor: IME2 kernels, expanded quantization (Q2_K, Q3_K, Q6_K, Q8_0, Q5_0, Q5_1, Q5_K, MXFP4), TCM (Tightly Coupled Memory) pool; new source files `ime2_kernels.cpp`, `ime_env.cpp`, `repack.cpp`, `rvv_kernels.cpp`, `spine_mem_pool.cpp`; guarded by `GGML_CPU_RISCV64_SPACEMIT` build flag; no project changes required |
| ~b9150–b9151 | `common/log.h` | New `LOG_TRC` macro added at `LOG_LEVEL_TRACE = 4` (between INFO=3 and DEBUG=5); `LOG_LEVEL_DEBUG` bumped from 4 to 5; new `LOG_TRCV` verbosity variant; additive, no project changes required |
| ~b9150–b9151 | `common/common.h` + `common/common.cpp` | New `common_params_print_info(const common_params &)` function: prints verbosity level, per-device memory (name, total, free), and system info at `LOG_INF` level; replaces the two-line pattern `LOG_INF("build_info: %s\n", llama_build_info()); LOG_INF("%s\n", common_params_get_system_info(params).c_str());` — updated in `jllama.cpp` |
| ~b9150–b9151 | `common/common.cpp` | `common_init()` now unconditionally calls `common_log_set_prefix(…, true)` and `common_log_set_timestamps(…, true)` before setting the log callback; log output will always include prefix and timestamps unless explicitly disabled with `--no-log-prefix` / `--no-log-timestamps` |
| ~b9150–b9151 | `common/arg.cpp` | `--log-prefix` and `--log-timestamps` now also accept negated forms `--no-log-prefix` / `--no-log-timestamps` (lambda receives a `bool value`); backing env vars renamed `LLAMA_LOG_PREFIX` → `LLAMA_ARG_LOG_PREFIX` and `LLAMA_LOG_TIMESTAMPS` → `LLAMA_ARG_LOG_TIMESTAMPS`; Java layer does not expose these, so no project changes required |
| ~b9150–b9151 | `tools/server/server-common.h` | New `SLT_TRC` and `SRV_TRC` macros (emit at `LOG_TRC` level); additive, no project changes required |
| ~b9150–b9151 | `tools/server/server-context.cpp` | New `server_slot::t_print_last` field + `print_timings_tg()` / `print_timings_pp()` methods: emit periodic in-flight token-generation and prompt-processing throughput to `SLT_INF` (throttled to ≥100 decoded tokens and ≥3 s interval); `server_context_impl` constructor now calls `mtmd_helper_log_set` unconditionally (was guarded by `!is_resume`); many `SLT_INF`/`SRV_WRN` downgraded to `SLT_TRC`/`SRV_INF`; compiled from upstream, no project JNI changes required |
| ~b9150–b9151 | `tools/server/server-task.cpp` | Several `SRV_WRN` calls downgraded to `SRV_INF`; one `SRV_WRN` upgraded to `SRV_ERR` for failed state restore; compiled from upstream, no project changes required |
| ~b9151–b9172 | `tools/mtmd/clip.h` | `clip_has_whisper_encoder()` removed from public API; not referenced by project — no changes required |
| ~b9151–b9172 | `tools/server/CMakeLists.txt` + `scripts/webui-download.cmake` (new) | WebUI assets no longer committed (`tools/server/public/` gitignored); provisioned at build time via HF bucket (`LLAMA_USE_PREBUILT_WEBUI=ON` default) or built from source (`LLAMA_BUILD_WEBUI`); project sets `LLAMA_BUILD_WEBUI=OFF CACHE BOOL "" FORCE` before FetchContent to skip asset download |
| ~b9151–b9172 | `common/common.h` | `common_params::webui` default made conditional on `LLAMA_WEBUI_DEFAULT_ENABLED` macro (falls back to `true` when undefined); compiled server sources unaffected |
| ~b9151–b9172 | `common/reasoning-budget.cpp` | `common_reasoning_budget_clone` rewritten to use `llama_sampler_init` properly; pure bug fix, no API change, no project changes required |
| ~b9151–b9172 | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` + `mma.cuh` | AMD RDNA3 WMMA flash attention support; new `DATA_LAYOUT_I_MAJOR_SCRAMBLED`, `tile<16,16,half2,I_MAJOR_SCRAMBLED>`, extended config tables; internal CUDA backend, no project changes required |
| ~b9151–b9172 | `tools/server/server-chat.cpp` | Non-function Responses API tools now silently skipped (`continue`) instead of throwing; server behavior fix, no Java API change required |
| ~b9172–b9198 | project `CMakeLists.txt` | Option `LLAMA_BUILD_WEBUI` renamed to `LLAMA_BUILD_UI` (and `LLAMA_USE_PREBUILT_WEBUI` → `LLAMA_USE_PREBUILT_UI`); upstream keeps a backward-compat shim that forwards the old cache variable with a `DEPRECATION` message, so this project's `set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE)` still works unchanged |
| ~b9172–b9198 | `common/common.h` | `common_params::webui` / `webui_mcp_proxy` / `webui_config_json` deprecated in favour of `ui` / `ui_mcp_proxy` / `ui_config_json`; both pairs of fields are kept and synced by `common/arg.cpp`, compiled upstream sources unaffected; new `common_params::ctx_type` and `cparams.n_rs_seq` fields added (default `LLAMA_CONTEXT_TYPE_DEFAULT` / `0`), additive |
| ~b9172–b9198 | `common/common.cpp` + `common.h` | `common_params_print_info` gained optional `print_devices` parameter (default `true`); upstream `tools/server/server.cpp` passes `!is_router_server` to skip GPU enumeration on the router process; this project does not compile `server.cpp`, no impact |
| ~b9172–b9198 | `common/speculative.h` + `speculative.cpp` | New enum value `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` (count is now 9); new `common_speculative_need_embd()` API; MTP draft implementation added (`common_speculative_state_draft_mtp`); `--spec-type draft-mtp` CLI flag added in `common/arg.cpp`; additive, no project changes (could be exposed later as a `ModelParameters` enhancement) |
| ~b9172–b9198 | `include/llama.h` | New `enum llama_context_type { LLAMA_CONTEXT_TYPE_DEFAULT, LLAMA_CONTEXT_TYPE_MTP }`; new `llama_context_params::n_rs_seq` (recurrent-state snapshots per seq for rollback) and `ctx_type` fields; new `llama_n_rs_seq()` accessor; all additive, default-zero, no project impact |
| ~b9172–b9198 | `src/llama-ext.h` (new) + `src/llama-context.cpp` | New pre-norm embedding extraction path: `llama_set_embeddings_pre_norm` / `llama_get_embeddings_pre_norm[_ith]` APIs and an `embd_pre_norm` output buffer in `llama_context`; used by the MTP draft loop only, additive |
| ~b9172–b9198 | `src/llama-memory-recurrent.cpp` | Recurrent-state rollback support: per-seq `rs_idx` snapshot index and `set_rs_idx()` helper; tensors widened to `(1 + n_rs_seq)` groups; `seq_rm` now rolls back via snapshot when within `n_rs_seq` bounds. Backwards-compatible when `n_rs_seq == 0` (this project's default), no project changes |
| ~b9172–b9198 | `tools/server/server-context.cpp` | Embedding endpoint default now reads `params.embd_normalize` (was hard-coded `2`); compiled upstream, no project changes |
| ~b9172–b9198 | `tools/server/CMakeLists.txt` + new `tools/ui/CMakeLists.txt` | WebUI asset wiring moved into a new `llama-ui` static library; `tools/server` now links `llama-ui`; project does not build the `llama-server` binary (only compiles `server-context.cpp` / `server-queue.cpp` / `server-task.cpp` / `server-models.cpp` directly into `jllama`), so no impact. HF bucket name renamed `LLAMA_WEBUI_HF_BUCKET` → `LLAMA_UI_HF_BUCKET` (old name still honoured) |
| ~b9172–b9198 | `vendor/cpp-httplib/httplib.{h,cpp}` | Bumped to v0.45.0: RFC 9112 §6 message-body framing — requests without `Content-Length` / `Transfer-Encoding` no longer scan for stray body bytes on persistent connections (fixes #2450 keep-alive misframing); X-Forwarded-For parser falls back to the connection remote address when the header is empty/malformed; compiled automatically, no project changes |
| ~b9172–b9198 | `ggml/CMakeLists.txt` | GGML version bumped 0.11.1 → 0.12.0; no project changes |
| ~b9172–b9198 | `ggml/src/ggml.c` + `ggml-cuda/gated_delta_net.cu` + `ggml-metal/ggml-metal.metal` + `ggml-vulkan/vulkan-shaders/gated_delta_net.comp` | `ggml_gated_delta_net` state tensor reshaped from 2D `(S_v*S_v*H, n_seqs)` to 3D `(S_v*S_v*H, K, n_seqs)` where `K` is the snapshot slot count (`K=1` is final-state-only, `K>1` keeps last `min(n_tokens, K)` per-token snapshots); internal Qwen3.5 / Qwen3-Next recurrent-attention kernel, no project changes |
| ~b9198–b9219 | `common/chat.{h,cpp}` | New `common_chat_continuation` enum (`NONE`/`AUTO`/`REASONING`/`CONTENT`); new `common_chat_msg::render_content(delimiter)` method; new `continue_final_message` field on `common_chat_templates_inputs`; new `common_chat_continuation_parse()` accepts both `bool` and `"reasoning_content"`/`"content"` strings; `common_chat_template_generation_prompt()` extracted; `oaicompat_chat_params_parse` refactored to route the prefill-assistant heuristic through the new continuation enum. Existing `bool` wire-format unchanged; the new string variants are exposed via `InferenceParameters.setContinueFinalMessage(ContinuationMode)` |
| ~b9198–b9219 | `common/hf-cache.{h,cpp}` + `common/arg.cpp` | `hf_cache::migrate_old_cache_to_hf_cache()` and `hf_file::size` field removed; the migration call in `common_params_parse_ex` was dropped. Internal to `arg.cpp`, no project impact |
| ~b9198–b9219 | `common/speculative.{h,cpp}` + `src/llama-ext.h` + `src/llama-context.{h,cpp}` + `src/llama-cparams.h` | `llama_set_embeddings_pre_norm(ctx, value)` → `llama_set_embeddings_pre_norm(ctx, value, masked)` (3rd `bool` arg distinguishes "embeddings for outputs only" from "embeddings for every token"); new `cparams.embeddings_pre_norm_masked`; new `common_speculative_need_embd_pre_norm()` API; MTP draft path now uses pre-norm extraction. Project does not call any of these APIs (speculative decoding is configured via `ModelParameters` only), no source changes required |
| ~b9198–b9219 | `tools/server/server-task.{h,cpp}` | `task_result_state` ctor moved from header into `.cpp` — now seeds `chat_msg` via `common_chat_parse("", true, …)` when `!echo` so the assistant prefill is not echoed back as a delta; new `bool echo` field on `chat_parser_params` (default `false`, populated from request body via `json_value(data, "echo", false)`). Project compiles `server-task.cpp` from upstream and does not instantiate `task_result_state` directly, no source changes required |
| ~b9198–b9219 | `tools/server/server-context.cpp` + `server-models.cpp` | New `cors_proxy_enabled` boolean field added to `/props` and `/v1/models` JSON responses (set from `params.ui_mcp_proxy \|\| params.webui_mcp_proxy`). Additive, no Java consumer in this project |
| ~b9198–b9219 | upstream `CMakeLists.txt` | Backward-compat shim widened: `if(DEFINED LLAMA_BUILD_WEBUI AND NOT DEFINED LLAMA_BUILD_UI)` → `if(DEFINED LLAMA_BUILD_WEBUI)` — setting the old name now always forwards to the new one (and emits the existing `DEPRECATION` message). Project sets only `LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE` (`CMakeLists.txt:107`), behaviour unchanged |
| ~b9198–b9219 | `ggml/src/ggml-cuda/ssm-conv.cu` + `top-k.cu` | Added kernel size 15 to SSM-conv launcher (now supports 3/4/5/9/15); `top-k.cu` includes `<cuda/iterator>` for CCCL ≥ 3.1; internal CUDA backend, no project changes |
| ~b9198–b9219 | `ggml/src/ggml-sycl/ggml-sycl.cpp` + `vecdotq.hpp` | SYCL GEMM now falls back to direct MKL for small problems (gemm_flops < 256³); Q6_K dot product refactored to a single scalar fast-path helper `vec_dot_q6_K_q8_1_impl_mmvq_scalar`; internal SYCL backend, no project changes |
| ~b9219–b9222 | `ggml/src/ggml-hexagon/` + `htp/pad-ops.c` (new) + `htp/unary-ops.c` | Hexagon HTP backend gains `GGML_OP_PAD` (HVX + optional VTCM/DMA double-buffered, both zero-pad and circular-pad variants) and `GGML_OP_TRI` (HVX-vectorised triangular masking) support; new `HTP_OP_PAD` / `HTP_OP_TRI` opcodes; internal Qualcomm DSP backend, no project changes |
| ~b9219–b9222 | `.devops/*.Dockerfile` + `.github/workflows/docker.yml` | OCI image labels (`org.opencontainers.image.*`) added via `BUILD_DATE`/`APP_VERSION`/`APP_REVISION` build args; new `skip_s390x` workflow_dispatch input; manifest annotations on `docker buildx imagetools create`; upstream packaging/CI only, no project changes |
| ~b9222–b9245 | `common/common.h` + `common.cpp` | `common_init_result(common_params &, bool model_only = false)` and `common_init_from_params(common_params &, bool model_only = false)` gain an optional `model_only` flag that skips context/sampler/lora/warmup setup and returns only the loaded model. Additive with default value; no project call sites in `src/main/cpp/`, no source changes required |
| ~b9222–b9245 | `common/common.h` | `common_params_speculative_draft` defaults retuned: `n_max` 16→3, `p_min` 0.75f→0.0f. Defaults only; Java `ModelParameters` sets these explicitly via JSON, so behaviour is unchanged for this project |
| ~b9222–b9245 | `common/speculative.{h,cpp}` | `common_speculative_impl::accept()` virtual gains a 3rd `bool is_other` parameter; `common_speculative_accept()` now broadcasts the accepted-token count to every registered impl (with `is_other=true` for impls that did not generate the draft). `common_speculative_impl_ngram_map_k` ctor signature simplified (no longer takes `common_params_speculative`). Lots of new `LOG_INF` startup banners per impl. Internal to upstream-compiled `server-context.cpp`; no project call sites |
| ~b9222–b9245 | `common/arg.cpp` + `common/common.cpp` + `tools/fit-params/fit-params.cpp` | `--verbosity` levels relabeled: level `4` now means "trace (more info)" and level `5` means "debug"; `LOG_LEVEL_DEBUG` constant value moved from `4` to `5`. Direct `params.verbosity >= 4` comparisons in upstream `common.cpp` and `fit-params.cpp` replaced with `>= LOG_LEVEL_DEBUG`. Project does not reference `LOG_LEVEL_DEBUG` or numeric verbosity thresholds in `src/main/cpp/`; no source changes required |
| ~b9222–b9245 | `common/arg.cpp` | `--spec-type` duplicate-arg DEPRECATED warning suppressed (the flag legitimately accepts repeated values to form the comma-list). Behaviour-only |
| ~b9222–b9245 | `common/ngram-map.cpp` | One per-draft `LOG_INF` downgraded to `LOG_DBG`. Log-level only |
| ~b9222–b9245 | `src/llama-graph.h` | `llm_graph_params::operator==` adds a third disjunct so ubatches with both `token` and `embd` arrays present compare equal (graph reuse fix for MTP pre-norm path). Internal |
| ~b9222–b9245 | `src/llama-memory-recurrent.{h,cpp}` + `src/llama-memory-hybrid.cpp` + `src/llama-memory-hybrid-iswa.cpp` | `init_batch()` now forces sequential split (`split_seq`) instead of equal split when `n_rs_seq > 0` (recurrent-state rollback is incompatible with equal splits). Internal upstream model code, no project impact |
| ~b9222–b9245 | `src/models/delta-net-base.cpp` + `src/models/models.h` + `src/models/qwen35.cpp` | `llm_build_delta_net_base::keep_rs()` helper removed; conv-state and recurrent-attn paths reworked to read `cparams.n_rs_seq` directly and loop `K = n_rs_seq + 1` snapshot slots. Comment fix in `qwen35.cpp` MTP layer index. All internal upstream model code |
| ~b9222–b9245 | `tools/server/server-context.cpp` | `pos_min_thold` lowered by one (`pos_next - n_swa` → `pos_next - n_swa - 1`); checkpoint trigger guard relaxed from `n_past < slot.prompt.n_tokens()` to `<=`; per-slot `print_timings_pp`/`print_timings_tg` lines split into separate `SLT_INF` calls; new `graphs reused` and `draft acceptance` lines; `n_draft_total` log moved from `SLT_CNT` to `SLT_INF`. Compiled upstream-as-is, no project changes |
| ~b9222–b9245 | `ggml/src/ggml-cuda/mmvq.cu` | `calc_nwarps` table tweak: Q6_K returns 2 warps (was grouped with the 8-warp tier). Internal CUDA backend |
| ~b9222–b9245 | `ggml/src/ggml-hexagon/` (`htp/rope-ops.c`, `htp/unary-ops.c`, `htp-ops.h`, `main.c`, `ggml-hexagon.cpp`) | New `HTP_OP_NORM` opcode (mean+variance norm); `rope-ops.c` adds MROPE / IMROPE position-id support via new `mrope_cache_init()`. Internal Qualcomm DSP backend |
| ~b9222–b9245 | `ggml/src/ggml-opencl/` (`ggml-opencl.cpp`, `kernels/cvt.cl`, six new `gemm_moe_q{4,5,6}_k_f32_ns` + `gemv_moe_q{4,5,6}_k_f32_ns` kernels) | Adreno MoE pipeline extended to Q4_K / Q5_K / Q6_K (image1d_buffer_t transposed layout, dedicated convert/restore kernels, GEMM + GEMV paths). Internal OpenCL backend |
| ~b9222–b9245 | `ggml/src/ggml-rpc/ggml-rpc.cpp` | `last_graph_uid` field moved from `ggml_backend_rpc_context` (per-backend) into `ggml_backend_rpc_device_context` (per-device) so multiple backends sharing a device reuse cached graphs. Internal RPC backend |
| ~b9222–b9245 | `ggml/src/ggml-sycl/ggml-sycl.cpp` | New `GGML_SYCL_USE_ASYNC_MEM_OP` env (default `1`) decouples async USM alloc/free from the graph path. Internal SYCL backend |
| ~b9222–b9245 | `ggml/src/ggml-webgpu/ggml-webgpu.cpp` + `wgsl-shaders/gated_delta_net.wgsl` | Gated-delta-net shader gains a `K` snapshot-count param; per-slot snapshot write path added. Internal WebGPU backend |
| ~b9222–b9245 | `convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, `examples/save-load-state/save-load-state.cpp`, `examples/llama-eval/*`, `tools/cli/README.md`, `tools/server/README.md`, `docs/speculative.md`, `docs/backend/SYCL.md` | Doc/example/tooling updates only. Not compiled by this project |
| ~b9222–b9245 | `tools/ui/*` | WebUI source reorganisation (enum file renames `*.ts` → `*.enums.ts`, new chat components, Tailwind plugin imports). Project sets `LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE` in `CMakeLists.txt`, so the UI is never built — no impact |
| ~b9245–b9264 | `src/llama-chat.{h,cpp}` | `LLM_CHAT_TEMPLATE_HUNYUAN_OCR` renamed to `LLM_CHAT_TEMPLATE_HUNYUAN_VL` (HunyuanOCR and HunyuanVL now share one template). Not referenced by project — no source changes required |
| ~b9245–b9264 | `tools/mtmd/clip-impl.h` + `tools/mtmd/models/` | `PROJECTOR_TYPE_HUNYUANOCR` removed and merged into `PROJECTOR_TYPE_HUNYUANVL`; `hunyuanocr.cpp` renamed to `hunyuanvl.cpp`; clip graph class `clip_graph_hunyuanocr` renamed to `clip_graph_hunyuanvl`. Not referenced by project — no source changes required |
| ~b9245–b9264 | `tools/mtmd/clip.h` | `clip_is_minicpmv()` and `clip_is_glm()` removed from public API. Not referenced by project — no source changes required |
| ~b9245–b9264 | `tools/mtmd/clip.h` (`struct clip_context_params`) | New `bool no_alloc` field added (initialized via `mtmd_context_params_default()`). Additive default-zero — no project changes required |
| ~b9245–b9264 | `tools/mtmd/mtmd.h` | New `mtmd_get_memory_usage()` C++ API for estimating mmproj VRAM/RAM usage. Additive, not called by project |
| ~b9245–b9264 | `tools/mtmd/clip-model.h` | New `enum pad_style { PAD_NONE, PAD_CEIL, PAD_NEAREST }` replacing the `bool image_resize_pad` flag (allows Pillow-byte-parity nearest-integer rounding for DeepSeek-OCR). Internal to mtmd, project links `mtmd` as-is |
| ~b9245–b9264 | `common/common.h` (`struct common_params_speculative_draft`) | New `bool backend_sampling = true` field — offloads draft sampling to the backend. Additive default-on; Java `ModelParameters` doesn't set it, so the upstream default applies. Backend sampler auto-disables when `split_mode == TENSOR` in `src/llama-context.cpp` — safe |
| ~b9245–b9264 | `common/speculative.cpp` | `common_speculative_impl_draft_mtp` now registers a per-seq backend sampler chain (top-k 10) on `ctx_dft` via `llama_set_sampler`; cleaned up in destructor. Falls back to CPU sampler if `llama_set_sampler` fails. Internal to upstream-compiled speculative module, no project call sites |
| ~b9245–b9264 | `app/` (new) | New optional unified `llama` binary (`llama-app` target) dispatching to `serve`/`cli`/`completion`/`bench`. Guarded by `LLAMA_BUILD_APP=OFF` default — project doesn't enable it |
| ~b9245–b9264 | `tools/{cli,completion,llama-bench,server}/CMakeLists.txt` | Each tool split into a `*-impl` static library (the logic) plus a thin `main.cpp` wrapper; the `main()` in `cli.cpp`/`completion.cpp`/`llama-bench.cpp`/`server.cpp` is renamed to `llama_cli`/`llama_completion`/`llama_bench`/`llama_server` and now satisfies `-Wmissing-declarations` via a forward decl. Project does NOT compile any of these `.cpp` files — only `server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-models.cpp` (see `CMakeLists.txt:237`/`:302`) — so no impact |
| ~b9245–b9264 | `tools/server/server-context.cpp` | Adds mmproj memory estimation: when `params_base.fit_params` is set, calls `mtmd_get_memory_usage(mmproj_path, mparams)` and adds the per-device cost into `params_base.fit_params_target` before `common_init_from_params`. Also calls `mtmd_helper_log_set(common_log_default_callback, nullptr)` once when `!is_resume`. Compiled upstream-as-is, no project call sites |
| ~b9245–b9264 | `src/llama-context.cpp` | New `llama_context::set_sampler()` short-circuits with a one-shot `LLAMA_LOG_WARN` and returns `false` when `model.split_mode() == LLAMA_SPLIT_MODE_TENSOR` (backend sampling not supported with tensor split). Internal safety check, no project call sites |
| ~b9245–b9264 | `common/arg.cpp` | New CLI flags `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling` and env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING` to toggle the new `backend_sampling` field. Not exposed by `ModelParameters`; could be added later as a Java-side enhancement |
| ~b9245–b9264 | `ggml/src/ggml-cuda/CMakeLists.txt` + `common.cuh` + `binbcast.cu`, `concat.cu`, `cpy.cu`, `fattn-*.cu`, `gated_delta_net.cu`, `getrows.cu`, `mean.cu`, `mmvf.cu`, `mmvq.cu`, `norm.cu`, `quantize.cu`, `reduce_rows.cuh`, `rope.cu`, `scale.cu`, `set-rows.cu`, `softcap.cu`, `ssm-conv.cu`, `ssm-scan.cu`, `sumrows.cu`, `topk-moe.cu`, `unary.cu` | New PDL (Programmatic Dependent Launch) infrastructure: `GGML_CUDA_USE_PDL` build flag (CUDART ≥ 11.8, non-HIP/MUSA); `ggml_cuda_pdl_sync()` / `ggml_cuda_pdl_lc()` device helpers (active on Hopper sm_90+); `ggml_cuda_kernel_launch_params` + `ggml_cuda_kernel_launch()` host template that calls `cudaLaunchKernelEx` with stream-serialization attribute when `GGML_CUDA_PDL` env var allows. Adds `90-virtual` (Hopper) to default `CMAKE_CUDA_ARCHITECTURES` when CUDA ≥ 11.8. Internal CUDA backend, no project changes required |
| ~b9245–b9264 | `ggml/src/ggml-metal/ggml-metal-{device,ops}.cpp` + `ggml-metal.metal` | New 4-element `kernel_pad_*_4` variant (currently disabled — `is_c4 = false`); `kernel_pad` rewritten with 1024-element-per-block tiling for larger tensors; `kernel_cpy_*` rewritten to use `tpitg` rows-per-threadgroup batching; Q quantization cpy paths use 256-thread limit. Internal Metal backend |
| ~b9245–b9264 | `ggml/src/ggml-hexagon/htp/` (`hmx-matmul-ops.c`, `hmx-ops.h`, `matmul-ops.c`, `main.c`) | HMX matmul refactor: K-loop tiled in 32-tile blocks with `Q6_activation_hf_mxmem_RR_deep`; the out-stationary fallback path for large M·K·N was deleted; function rename `hmx_mat_mul_permuted_w16a32` → `hmx_matmul_f16_f32`, `hmx_mat_mul_permuted_qk_0_d16a32` → `hmx_matmul_q_f32`, `hmx_mat_mul_permuted_w16a32_batched_params_t` → `hmx_matmul_f16_f32_batched_params_t`. HMX power-up code reorganized (`HAP_power_set_HMX_v2` now combines power-on + clock in one step for `__HVX_ARCH__ ≥ 75`). Internal Qualcomm DSP backend |
| ~b9245–b9264 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | Lazy kernel compilation: `argsort` and `flash_attn` programs are now built only when first needed (`load_cl_kernels_argsort` / `load_cl_kernels_flash_attn` called from `supports_op`); new device-supported probe in `ggml_opencl_is_device_supported` runs at registration time; renamed `ggml_cl2_init`/`ggml_cl2_free` → `ggml_cl_init`/`ggml_cl_free`; OpenCL contexts now live as long as the process. Internal OpenCL backend |
| ~b9245–b9264 | `ggml/src/ggml-vulkan/vulkan-shaders/im2col.comp` | Refactor: precomputed base input coords and step deltas; running pointer/index for destination; one inlined unrolled loop iteration writes `BLOCK_SIZE` outputs per step. Internal Vulkan backend |
| ~b9245–b9264 | `src/models/delta-net-base.cpp` | Renamed local variables (`state_in_3d`→`s_3d`, `state_3d`→`s_3d_pad`) when reshaping the recurrent state; behaviour unchanged |
| ~b9245–b9264 | `tools/mtmd/mtmd-image.cpp` | `img_tool::resize()` takes a `pad_style` enum (was `bool add_padding`); new `PAD_NEAREST` rounding path for Pillow byte-parity; `mtmd_image_preprocessor_deepseekocr::preprocess` rewritten with `static constexpr` resolution table and `RESIZE_ALGO_BICUBIC_PILLOW` + `PAD_NEAREST`. Internal mtmd, project links as-is |
| ~b9245–b9264 | `tools/mtmd/models/deepseekocr.cpp` | Extracted `build_sam(ggml_tensor *inp_raw)` member function from the monolithic build path; FA mask casting to F16 only when `flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED`. Internal |
| ~b9245–b9264 | `conversion/hunyuan.py`, `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` | HunyuanOCR / HunyuanVL unified in conversion: `VisionProjectorType.HUNYUANOCR` removed; `HunYuanVLForConditionalGeneration` registers a single `HunyuanVLVisionModel` + `HunyuanVLTextModel`; `vit.perceive.*` tensor mappings now only mention `HunyuanVL`. Python tooling, not compiled by project |
| ~b9245–b9264 | `CMakeLists.txt` (upstream) | New `LLAMA_BUILD_APP` option (default OFF); deprecation shims for `LLAMA_BUILD_WEBUI`/`LLAMA_USE_PREBUILT_WEBUI` → `LLAMA_BUILD_UI`/`LLAMA_USE_PREBUILT_UI` preserved. Project's `set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE)` still works unchanged |
| ~b9245–b9264 | `.devops/*.Dockerfile`, `.github/workflows/build-and-test-snapdragon.yml`, `scripts/snapdragon/`, `docs/backend/snapdragon/`, `tools/cli/README.md`, `tools/server/README.md`, `tools/mtmd/tests/` | Docker images add `conversion/` dir; snapdragon toolchain bumped v0.3 → v0.6 with `+dotprod+i8mm`; mtmd test rewritten to use CER/chrF metrics; doc-only updates. Not compiled by project |

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

Built libraries are placed in `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.

### Code Formatting
```bash
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp   # Format C++ code
```

## Architecture

### Two-Layer Design

**Java layer** (`src/main/java/net/ladenthin/llama/`):
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
1. System property `net.ladenthin.llama.lib.path`
2. `java.library.path`
3. Extracts from JAR resources at `net/ladenthin/llama/{os}/{arch}/`

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for ARM/Android targets. CI workflows use these for non-x86 Linux builds.

## Testing

### Java tests
Require a model file. The CI downloads models from HuggingFace:
- **LlamaModel tests**: CodeLlama-7B-GGUF (`codellama-7b.Q2_K.gguf`)
- **RerankingModel tests**: Jina-Reranker model

Set the model path via system property or environment variable (see test files for exact property names).

Test files are in `src/test/java/net/ladenthin/llama/` and `src/test/java/examples/`.

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

**Current total: 417 tests (all passing).** Branch: `claude/determined-volta-T8AoQ`.

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
grep -rn "field_name" src/main/java/net/ladenthin/llama/
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

- **Java 8+** runtime required. Built with JDK 21 targeting bytecode 1.8 for broad compatibility.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.

## Javadoc Conventions

### HTML Entities

In Javadoc comments, never use bare Unicode characters for operators and symbols. Use HTML entities instead:

| Symbol | HTML entity |
|---|---|
| `<` | `&lt;` |
| `>` | `&gt;` |
| `≤` | `&#x2264;` |
| `≥` | `&#x2265;` |
| `→` | `&#x2192;` |
| `←` | `&#x2190;` |
| `≠` | `&#x2260;` |

Use numeric hex entities (`&#xNNNN;`) for any Unicode symbol outside ASCII. Named entities (`&lt;`, `&gt;`) are acceptable for `<` and `>`.
