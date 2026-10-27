<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# gpt-oss-20b on the `alderlake` CPU module: where the time goes

**The question.** The Windows guide [`../windows-intel-core-ultra.md`](../windows-intel-core-ultra.md)
puts a Core Ultra 7 165U on `ggml-cpu-alderlake` (AVX2 + AVX-VNNI, no AVX-512, no AMX). Where does
that module lose compute, on which models, and what would make gpt-oss-20b faster there?

**The answer, measured 2026-10-26/27 on b11887 + `llama/patches/`, the module staged alone:**

1. **Dense Q4_K_M models lose little.** Qwen3-4B spends 89-92 % of its thread time in the Q4_K / Q6_K
   matmuls, Qwen3-0.6B 75-86 % (plus 16 % flash attention in prompt processing); both wait 3-8 % at
   barriers. There is no lever there short of the kernels themselves.
2. **gpt-oss-20b loses a lot, in one place: ggml-cpu's MoE path for repacked weights**
   (`forward_mul_mat_id` in `ggml/src/ggml-cpu/repack.cpp`, identical on upstream master on
   2026-10-26). It has two defects:
   - **Prompt processing computes every token on its own** (one GEMV per token per expert, never the
     GEMM kernel that exists for the same type). 72 % of prompt time is that loop.
   - **Its work is split statically**, one fixed row range per thread. Token generation therefore
     waits for the slowest thread after every expert matmul: 13.9 % of all thread time in token
     generation is that wait. Every other matmul hands out chunks dynamically and waits 1-2 %.
3. **A fix for both** (dynamic work units plus GEMM for groups of ≥ 4 tokens per expert), first as a
   prototype ([`../../.github/scripts/bench/mmid-dynamic-gemm.diff`](../../.github/scripts/bench/mmid-dynamic-gemm.diff)),
   then as [`llama/patches/0022`](../patches.md#0022-ggml-cpu-repack-batch-the-tokens-of-an-expertpatch),
   gives **+46-47 % prompt processing and +23-26 % token generation** at 4 threads, **+59 % prompt
   processing single-threaded**, with logits identical to rounding. With one core emulated at half
   speed, prompt processing on upstream's path halves (−49 %), on the batched path it loses 24 %.
   That emulation is the closest this machine gets to a P-core/E-core laptop.
4. **No configuration knob makes it faster, and one makes it much slower.** Repack on and flash
   attention on (both defaults) are already the fastest settings, the sampler chain costs nothing
   measurable, and the server path (the code `NativeServer` runs) is within 3-5 % of `llama-bench`.
   But a **quantized KV cache** (`-ctk q8_0 -ctv q8_0`, as in the README's gpt-oss-20b example) cuts
   prompt processing at an 8k context to **a quarter** (30.76 → 7.45 t/s): ggml's fast
   flash-attention paths need an f16/f32 cache with K and V of one type.

The patch keeps upstream's path behind a **process-wide switch**, on by default, so the comparison
can be repeated on any machine without a second build: `LlamaModel.setCpuMoeBatched(boolean)` in
Java, `jllama_set_cpu_moe_batched()` in C, or `GGML_CPU_REPACK_MMID_BATCHED=0` in the environment
of any process ([Patch 0022 and its switch](#patch-0022-and-its-switch)).

## What this run can and cannot show

**Can:** which code `alderlake` runs, where its time goes, and what a change to that code is worth on
the same machine. The kernels are the module's own: `-mavx2 -mavxvnni -mfma -mf16c -mbmi2`, built with
plain clang like the shipped `cpu-windows-x86-64` jar.

**Cannot:**

- **The laptop's numbers.** The host is a 4-vCPU KVM guest on an Intel Xeon (family 6, model 173,
  Granite Rapids generation) with 15 GiB RAM. Single-threaded it reads 15.4 GB/s, with 4 threads
  55.7 GB/s (a 2 GiB AVX2 read probe). The 165U has a hybrid core set (2 P + 8 E + 2 LP-E),
  LPDDR5 shared with the iGPU and a 15 W budget. **Absolute t/s here are not a forecast for it.**
  The ratios between code variants on one machine are what transfers, and even those need the
  device to confirm them.
- **Linux, clang 18.1.3.** The shipped Windows jar is built by clang 20 (the runner's LLVM) and runs
  on Windows' scheduler.
- **Real routing.** The mock's router has random weights, so tokens spread evenly over the 32
  experts (≈ 64 per expert in a 512-token batch). A trained router is less even. That changes how
  many tokens each expert's GEMM gets, not whether GEMM applies.
- **Hybrid cores, properly.** The slow-core emulation (a busy loop sharing one pinned core) halves
  that core's throughput, but for token generation the OS's time slices dominate. A thread that is
  descheduled for a whole slice stalls hundreds of barriers, which a slower but continuously running
  E-core does not do. Only the prompt-processing half of that table means what it says.
- **Vulkan / the Arc iGPU.** No GPU here.

## Setup

### The build

llama.cpp at the pinned commit (`6050b467`, b11887) with `llama/patches/*.patch` applied in order,
built the way the modular library jars are: `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`,
`GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`, `BUILD_SHARED_LIBS=ON`, Release (`-O3`), plain clang. On top,
upstream's own tools: `llama-bench`, `llama-quantize`, `llama-perplexity`, `llama-server`.

```bash
git clone --depth 1 --branch b11887 https://github.com/ggml-org/llama.cpp && cd llama.cpp
for p in $(ls <java-llama.cpp>/llama/patches/*.patch | sort); do git apply "$p"; done
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build build -j --target llama-bench llama-quantize llama-perplexity llama-server
mkdir build/cpu-modules && mv build/bin/libggml-cpu-*.so build/cpu-modules/
```

**Forcing one module.** ggml loads the best CPU module it finds next to the executable, which on this
host is `sapphirerapids`. With every module moved out of `bin/`, `GGML_BACKEND_PATH` names the one to
load. [`../../.github/scripts/bench/bench.sh`](../../.github/scripts/bench/bench.sh) does that and
refuses a `bin/` that still holds modules. The load line confirms it:
`load_backend: loaded CPU backend from .../libggml-cpu-alderlake.so`.

### The models: random weights in the real shapes

Hugging Face is not reachable from the machine, and nothing here depends on what a model *says*.
Throughput is set by every tensor's shape and type: the work and the bytes a token costs.
[`../../.github/scripts/bench/mock_gguf.py`](../../.github/scripts/bench/mock_gguf.py) writes those
shapes with random weights, one tensor at a time (a 12 GB file needs no 12 GB of RAM):

| mock | written as | size | real file |
|---|---|--:|--:|
| Qwen3-0.6B (`Qwen/Qwen3-0.6B` config) | F16 → `llama-quantize Q4_K_M` | 393.9 MB | ~397 MB |
| Qwen3-4B-Instruct-2507 (config of that name) | F16 → `llama-quantize Q4_K_M` | 2,494 MB | ~2.5 GB |
| gpt-oss-20b (`openai/gpt-oss-20b` config) | directly in ggml-org's layout: experts MXFP4, every other matrix Q8_0 (llama-quantize's `MXFP4_MOE` rule), norms/biases/router/sinks F32 | 12,100.7 MB, 20.91 B params | `gpt-oss-20b-mxfp4.gguf`, 12.1 GB |

The Qwen mocks go through `llama-quantize`, so their tensor-type mix *is* the real Q4_K_M mix
(Q6_K for the tied embedding/head and half of `attn_v`/`ffn_down`). The gpt-oss layout follows
`llama-quant.cpp`'s `MXFP4_MOE` branch, and the size check confirms it: Q8_0 for the non-expert
matrices adds up to 12.07 GB, BF16 would be 13.75 GB (the size of the published F16 file). The
vocabularies are the real sizes (151,936 and 201,088), since the output head is a large part of
every token.

```bash
export PYTHONPATH=<llama.cpp>/gguf-py   # gguf-py of the same tag
python3 .github/scripts/bench/mock_gguf.py gpt-oss-20b gpt-oss-20b-mxfp4.gguf        # 67 s, 12.1 GB
python3 .github/scripts/bench/mock_gguf.py qwen3-4b qwen3-4b-f16.gguf
llama-quantize qwen3-4b-f16.gguf qwen3-4b-q4_k_m.gguf Q4_K_M
# --layers N writes fewer layers with the same per-layer work, for quick iterations
```

### Bytes per generated token

What one generated token has to read, which sets the memory-bound ceiling:

| model | bytes / token | of which |
|---|--:|---|
| Qwen3-0.6B Q4_K_M | 391 MB | the whole model, a third of it the Q6_K head (151,936 × 1,024) |
| Qwen3-4B Q4_K_M | 2,494 MB | the whole model |
| gpt-oss-20b MXFP4 | 2,570 MB | 4 of 32 experts × 24 layers 1,269 MB, attention (Q8_0) 677 MB, **output head (Q8_0, 201,088 × 2,880) 615 MB** |

## Baseline: `alderlake`, unchanged code

`llama-bench`, defaults otherwise (repack on, flash attention auto, f16 KV cache), Qwen3-0.6B `-r 5`,
the others `-r 3`. t/s ± standard deviation:

| model | threads | pp512 | tg128 |
|---|--:|--:|--:|
| Qwen3-0.6B Q4_K_M | 1 | 127.33 ± 0.68 | 18.75 ± 0.35 |
|  | 2 | 244.09 ± 4.95 | 31.75 ± 0.16 |
|  | 4 | 452.17 ± 17.92 | 59.70 ± 1.01 |
| Qwen3-4B Q4_K_M | 1 | 18.00 ± 0.12 | 3.01 ± 0.02 |
|  | 2 | 35.15 ± 0.43 | 5.39 ± 0.05 |
|  | 4 | 65.65 ± 1.13 | 9.91 ± 0.07 |
| gpt-oss-20b MXFP4 | 1 | 10.22 ± 0.02 | 3.06 ± 0.04 |
|  | 2 | 20.47 ± 0.22 | 5.24 ± 0.05 |
|  | 4 | 43.79 ± 0.69 | 7.72 ± 0.03 |

Two things stand out before any profile:

- **Token generation is not memory-bound here.** Single-threaded, every model reads ~7.5 GB/s of
  weights, half of what one core can stream (15.4). The AVX2 kernels are instruction-bound on this
  host, which is also why AVX-512 changes so much (below).
- **gpt-oss scales worst.** From 1 to 4 threads its token generation rises 2.52×, Qwen3-4B's 3.29×,
  Qwen3-0.6B's 3.18×. Its prompt processing runs at 62 GFLOP/s per thread (6.06 GFLOP of active
  weights per token × 10.22 t/s), Qwen3-4B's at 131.

## Where the time goes

Two instruments. `perf record -e cpu-clock` (this VM exposes no hardware counters) for a flat
per-function profile. And a small instrumentation of ggml's graph loop,
[`../../.github/scripts/bench/ggml-opprof.diff`](../../.github/scripts/bench/ggml-opprof.diff)
(compiled in only with `-DGGML_OPPROF`). It sums, per op and source type, each thread's time *inside*
the op and *in the barrier after it*. The barrier time after an op is that op's load imbalance:
the time the faster threads wait for the slowest.

4 threads, share of all thread time:

| model / phase | biggest op | 2nd | 3rd | barrier wait (of which after the biggest op) |
|---|---|---|---|--:|
| gpt-oss tg | MoE MXFP4 `MUL_MAT_ID` 42.7 % | Q8_0 `MUL_MAT` 38.4 % | flash attn 0.8 % | **17.0 % (13.9 %)** |
| gpt-oss pp | MoE MXFP4 `MUL_MAT_ID` **72.3 %** | Q8_0 `MUL_MAT` 18.6 % | flash attn 2.6 % | 3.9 % (2.5 %) |
| Qwen3-4B tg | Q4_K 68.9 % | Q6_K 23.5 % | flash attn 1.4 % | 5.5 % (4.0 %) |
| Qwen3-4B pp | Q4_K 77.9 % | Q6_K 11.2 % | flash attn 6.3 % | 2.7 % (2.1 %) |
| Qwen3-0.6B tg | Q4_K 51.3 % | Q6_K (the head) 35.1 % | flash attn 4.2 % | 7.9 % (4.6 %) |
| Qwen3-0.6B pp | Q4_K 64.3 % | flash attn 16.2 % | Q6_K 10.6 % | 4.7 % (3.0 %) |

The perf profile names the functions behind the two gpt-oss rows: in prompt processing **65 % of all
samples are `ggml_gemv_mxfp4_8x8_q8_0`**, the *single-row* kernel, and 16 % the llamafile Q8_0 GEMM
(attention). In token generation it is 39 % that GEMV, 35 % `ggml_vec_dot_q8_0_q8_0` (attention and
the head) and 18 % `ggml_graph_compute_thread`, which is threads spinning at a barrier.

### Why: `forward_mul_mat_id` in the repack buffer

MXFP4 expert weights are repacked at load into the 8×8 interleaved layout (`mxfp4_8x8_q8_0`, chosen
for every AVX2 CPU). For `MUL_MAT_ID` the repack buffer's `forward_mul_mat_id`:

1. quantizes the activations, groups the token rows by expert, and passes one barrier;
2. gives **thread `ith` the fixed rows `[ith·n/nth, (ith+1)·n/nth)`** of every expert, and
3. loops over the expert's tokens calling **`gemv` once per token**.

The plain `MUL_MAT` path of the same buffer (`forward_mul_mat`) does neither: it calls `gemm` for
every four rows and hands out chunks through `ggml_threadpool_chunk_add`. The repack `gemm` for
MXFP4 (`ggml_gemm_mxfp4_8x8_q8_0`, AVX2 with a 16-row tile) exists and is simply never reached from
the MoE path. In ggml's generic `MUL_MAT_ID` the new tiled kernels take over for batches, but they
cover K-quants and IQ types, not MXFP4. Upstream master's `forward_mul_mat_id` is byte-identical to
b11887's (checked 2026-10-26).

What the static split costs depends on how equal the threads are. On this VM's identical vCPUs it
is the 13.9 % above. On a P-core + E-core machine every expert matmul runs at the E-cores' pace.

### Configuration does not reach it

gpt-oss, 4 threads, `-r 3`:

| repack | flash attn | pp512 | tg128 |
|---|---|--:|--:|
| off | off | 31.41 ± 0.11 | 7.16 ± 0.07 |
| off | on | 32.58 ± 0.19 | 7.27 ± 0.01 |
| **on** | off | 40.39 ± 1.22 | **7.99 ± 0.02** |
| **on** | **on** | **43.41 ± 0.42** | 7.73 ± 0.05 |

Repack (the default) is the faster path despite the per-token GEMV. Without it the generic
`vec_dot` path is slower still. Flash attention is a wash at an empty context (off at a long context
was not measured). **The server path** (`llama-server`, i.e. the
upstream `llama_server` that `NativeServer` runs in-process; `-c 4096 -np 1`, three 512-token
prompts with 128 tokens each) measured pp 42.61 ± 0.18 and tg 7.45 ± 0.04. That is 3 % and 5 % under
`llama-bench`, and identical with greedy sampling (7.45 ± 0.02), so neither the sampler chain over
201,088 logits nor the server loop is where time goes.

### The KV cache type, at a long context

An agent re-sends a growing conversation, so what matters there is a prompt *on top of* a filled
cache. `-d 8192` fills 8,192 tokens first, then measures. Unchanged code, flash attention on,
4 threads, `-r 3` (q8_0/q8_0: `-r 2`):

| K / V cache | pp512 at depth 8192 | tg32 at depth 8192 |
|---|--:|--:|
| **f16 / f16** (default) | **30.76 ± 0.37** | 4.57 ± 0.04 |
| q8_0 / f16 | 11.21 ± 0.30 (−64 %) | 5.01 ± 0.08 (+10 %) |
| q8_0 / q8_0 | **7.45 ± 0.02 (−76 %)** | *not measured on this build* |

On the prototype build (`-r 2`, generation only) the four combinations gave tg32 4.88 ± 0.21 (f16/f16),
4.72 ± 0.02 (f16/q8_0), 5.65 ± 0.09 (q8_0/f16) and 4.91 ± 0.11 (q8_0/q8_0).

The cause is a type check, not the arithmetic. `ggml_compute_forward_flash_attn_ext` in
`ggml/src/ggml-cpu/ops.cpp` takes its tiled path for a batch of queries and its split-KV path for a
single query only when K and V are F32 or F16 **and of the same type** (`kv_is_f32_or_f16`,
`k->type == v->type`). Any quantized or mixed cache falls back to the per-row path. A quantized K
saves some bandwidth in generation, and the prompt side pays for it four times over. **On the CPU,
leave the cache at f16.** For gpt-oss the memory is not a reason either: at 65,536 tokens the f16 cache
of the 12 full-attention layers is 1.6 GB, and the 12 sliding-window layers keep only their
128-token window (plus one batch).

(`llama-bench` caches the filled context by depth only and then fails to restore an f16 state into a
q8_0 cache, `incompatible key rotation`, and refills. The numbers are unaffected, the runs just take
longer.)

### The instruction set

Same machine, same build, one module at a time, 4 threads:

| module | gpt-oss pp512 | gpt-oss tg128 | Qwen3-4B pp512 | Qwen3-4B tg128 |
|---|--:|--:|--:|--:|
| `haswell` (AVX2) | 38.41 ± 0.77 | 7.36 ± 0.03 | 66.74 ± 0.87 | 10.17 ± 0.08 |
| `alderlake` (+ AVX-VNNI) | 43.79 ± 0.69 | 7.72 ± 0.03 | 65.65 ± 1.13 | 9.91 ± 0.07 |
| `sapphirerapids` (AVX-512, AMX) | 50.73 ± 1.10 | 11.60 ± 0.07 | 146.29 ± 4.42 | 12.68 ± 0.08 |

AVX-VNNI is worth 14 % / 5 % on gpt-oss (its MXFP4 and Q8_0 kernels use `vpdpbusd`) and nothing on
Q4_K. The Core Ultra's module is `alderlake` and cannot be anything else: AVX-512 is fused off in
Core Ultra series 1. The `sapphirerapids` row shows how much the 256-bit kernels leave on the table
on a core that is not memory-starved: 50 % of gpt-oss's token generation.

## The fix, first as a prototype

[`mmid-dynamic-gemm.diff`](../../.github/scripts/bench/mmid-dynamic-gemm.diff) changes only
`forward_mul_mat_id` and its `work_size`, behind the compile-time `-DJLLAMA_MMID_V2`, so one tree built
both sides of the A/B below. `llama/patches/0022` is the same code with the choice moved to run time
(next section):

- **Work units.** After the grouping barrier, thread 0 lists the experts that received tokens. The
  work becomes `n_active × nck` units of (expert, column chunk), with `nck` chosen for ~16 units per
  thread. Threads take units from `ggml_threadpool_chunk_add`, as `forward_mul_mat` does.
- **GEMM.** Within a unit, every group of up to 16 of the expert's tokens (in multiples of 4) is
  gathered from `src1`, interleave-quantized with `ggml_quantize_mat_t` (the same quantizer
  `forward_mul_mat` uses for its GEMM rows), multiplied with `gemm` into a per-thread tile, and
  scattered back to the token rows. Fewer than four leftover tokens keep the `gemv` path.
- **Cost:** ~420 KB of scratch per thread for gpt-oss's shapes, plus one list of `n_expert` integers.

### Correctness

`llama-perplexity` over 4 × 512 tokens of text, unchanged code with `--kl-divergence-base`, then the
prototype with `--kl-divergence`:

| | |
|---|---|
| PPL, unchanged / prototype | 321390.708865 / 321390.684826 (ratio 1.000000 ± 0.000023) |
| mean / max KL divergence | 0.000000 / 0.000002 |
| same top token | **100.000 %** |

The two paths quantize the activations identically and differ only in the order of the float
accumulation, which is what a KLD of 10⁻⁶ is. (A random model's perplexity is meaningless as a
number. Its agreement between two builds is the check.)

### Speed

Unchanged code and prototype alternated, `alderlake`, gpt-oss:

| threads | test | unchanged | prototype | change |
|--:|---|--:|--:|--:|
| 1 | pp256 | 10.53 ± 0.03 | 16.74 ± 0.16 | **+59 %** |
| 1 | tg32 | 3.09 ± 0.01 | 3.23 ± 0.00 | +5 % |
| 2 | pp512 | 20.38 ± 0.23 | 32.39 ± 0.55 | **+59 %** |
| 2 | tg128 | 5.32 ± 0.12 | 5.58 ± 0.12 | +5 % |
| 4 | pp512 | 43.94 ± 0.25 | 64.10 ± 0.44 | **+46 %** |
| 4 | tg128 | 7.84 ± 0.05 | 9.63 ± 0.09 | **+23 %** |

(1 thread `-r 2`, otherwise `-r 2` twice, n = 4 samples.) The single-threaded prompt row isolates
GEMM against GEMV: the same work, 37 % less time. The 4-thread generation row is the balance: the
barrier wait after `MUL_MAT_ID` falls from 13.9 % to 1.7 % of thread time and the total from 17.0 %
to 5.2 % (same instrumentation). Prompt processing now runs at 101 GFLOP/s per thread, three
quarters of the dense Q4_K model's 131.

**One core at half speed**, all four threads pinned (`-C F --cpu-strict 1`), a busy loop sharing
core 3, `-r 3`:

| | unchanged | prototype |
|---|--:|--:|
| pp256, no load | 43.50 ± 0.36 | 60.20 ± 4.10 |
| pp256, core 3 shared | **22.19 ± 0.79** (−49 %) | **45.77 ± 0.05** (−24 %) |
| tg64, no load | 7.96 ± 0.01 | 9.60 ± 0.13 |
| tg64, core 3 shared | 4.65 ± 0.02 | 5.49 ± 0.02 |

With the static split the whole prompt runs at the slow core's pace. With work units the other
three take over its share, and what is lost is about that core's missing half. The generation rows
fall alike in both builds. That is the time-slice artefact described above, not a property of either
build, which is why they are not read as a hybrid-core result.

### After the fix

The prototype's profile at 4 threads. Token generation: MoE 47.0 %, **Q8_0 GEMV 45.5 %**, barrier 5.2 %.
Prompt processing: MoE 62.0 %, Q8_0 GEMM 27.4 %, flash attention 3.9 %, barrier 2.7 %.

At a long context the picture shifts. With 8,192 tokens already in the cache (`-d 8192`, flash
attention on) the prototype measures pp512 **39.28** and tg32 **4.99**, against 64.10 and 9.63 at an
empty context, and flash attention becomes the second-largest op.

## Patch 0022 and its switch

`llama/patches/0022` is the prototype with the `#ifdef` replaced by a run-time switch, reachable from
every layer:

| layer | how |
|---|---|
| ggml | `ggml_backend_cpu_set_repack_mmid_batched(bool)` / `..._get_...()` in `ggml-cpu.h`, and in the CPU backend's `get_proc_address` table -- in a `GGML_BACKEND_DL` build the CPU backend is a module loaded at run time, and that table is how anything outside it reaches the switch |
| C API | `jllama_set_cpu_moe_batched(bool)`, `jllama_get_cpu_moe_batched(bool *)` |
| Java | `LlamaModel.setCpuMoeBatched(boolean)`, `LlamaModel.isCpuMoeBatched()`; FFM: `Jllama.setCpuMoeBatched` / `isCpuMoeBatched` |
| any process | `GGML_CPU_REPACK_MMID_BATCHED=0` in the environment makes upstream's path the initial value -- also for `llama-bench` and a server started from the command line |

It is process-wide. Thread 0 reads it once per `MUL_MAT_ID`, while it groups the tokens, and
publishes the choice in the work buffer before the barrier, so a flip during a running computation
cannot send the threads down different paths; the computation finishes on the path it started with.
`work_size()` reserves the batched path's scratch whatever the switch says, since it may flip between
planning a graph and running it.

The same measurement as above, now **with one binary**, the switch flipped through the environment
(b11887 + all patches, `alderlake`, gpt-oss, 4 threads, `-r 2` twice, n = 4):

| | upstream's path (`=0`) | batched (default) | change |
|---|--:|--:|--:|
| pp512 | 42.63 ± 1.16 | 62.76 ± 2.76 | **+47 %** |
| tg128 | 7.90 ± 0.07 | 9.97 ± 0.06 | **+26 %** |

The guard is `llama/src/test/cpp/test_moe_repack.cpp`: both paths against ggml's generic
`MUL_MAT_ID` over the same weights (MXFP4, Q4_0, IQ4_NL, Q4_K), on 1, 3 and 4 threads, with experts
that get 0, 1, 3, 4, 15 and 17 tokens, an empty slot and a single token.

## Next steps

1. **Upstream.** The change applies to every repacked expert type, not only MXFP4, and is
   upstream-shaped without the switch. The release policy for upstream submissions ("nothing before
   the release") no longer blocks it.
2. **Measure on the Core Ultra 7 165U**, both paths through the switch, `-t 6/8/10`: that is where the
   static split meets real P-, E- and LP-E-cores, and the hybrid half of this report can only be
   settled there. The KV-cache table is worth repeating there too, at the context an agent session
   actually reaches.
3. **The README's gpt-oss-20b `NativeServer` example** runs on the CPU (`-ngl 0`) with
   `-ctk q8_0 -ctv q8_0`, the configuration the KV table above measures at a quarter of the prompt
   speed. Whether the example keeps those flags, for a reason not visible here, is the owner's call.
4. **The next candidates, unmeasured as fixes:** the Q8_0 GEMV for token generation (45.5 % after the
   fix; x86 has no repacked Q8_0 path, ARM has one), and flash attention at long contexts (64 query
   heads over 8 KV heads).
