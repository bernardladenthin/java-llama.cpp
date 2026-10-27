<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Patch 0022 with real weights

**The question.** [`llama/patches/0022`](../patches.md#0022-ggml-cpu-repack-batch-the-tokens-of-an-expertpatch)
batches the tokens of an expert in the CPU backend's repacked `MUL_MAT_ID` and hands its work out
dynamically. It was measured on random weights in gpt-oss-20b's shape, on one x86 module
([`moe-mxfp4-cpu-alderlake.md`](moe-mxfp4-cpu-alderlake.md)). Does it hold with trained weights,
whose routers spread tokens unevenly, on the other kernels it calls: ARM's, K-quants?

**The answer, measured 2026-10-27 on b11901 + `llama/patches/`:**

1. **Prompt processing gains on every model and machine measured**: +13 % to +66 %, on x86
   (Zen 3), ARM (Neoverse-N1) and s390x (z16).
2. **Token generation is unchanged** within noise. With one token per expert, nothing is batched.
3. **The logits are those of upstream's path for MXFP4 and Q4_0** (KLD 0.000000, same top token
   100 %). **For Q4_K they differ by a KLD of 0.0075**, which is not a defect of the patch:
   upstream's own repack path is as far from the unrepacked computation (below).
4. **It found a defect in the first version of the patch.** On a Neoverse-N1, Q4_K prompt
   processing was 18 % slower than upstream's path at 32 threads, because every column chunk
   quantized the expert's activations again. ARM has no SIMD form of the Q8_K quantizer. Fixed:
   each group of four rows is now quantized once.
5. **The ARM C++ tests never ran the patch's guard.** `test-cpp-linux-aarch64` built for plain
   ARMv8, where nothing is repacked, so all 24 `test_moe_repack` cases skipped. It builds with
   dotprod now; on the N1 all 25 tests pass.

## Setup

- **Models** (trained weights from Hugging Face):
  - gpt-oss-20b MXFP4 (`ggml-org/gpt-oss-20b-GGUF`, 12.1 GB);
  - Qwen3-30B-A3B Q4_K_M (`unsloth/Qwen3-30B-A3B-GGUF`, 18.6 GB; 128 experts, 8 used);
  - OLMoE-1B-7B-0125-Instruct Q4_0 (`allenai/OLMoE-1B-7B-0125-Instruct-GGUF`, 3.9 GB; 64 experts,
    8 used).
- **The build:** llama.cpp b11901 with the 14 patches, built as the modular library jars are
  (`GGML_BACKEND_DL`, every CPU variant, no OpenMP, clang). ggml picks the module, upstream's
  `llama-bench` and `llama-perplexity` measure.
- **The switch:** `GGML_CPU_REPACK_MMID_BATCHED=0` gives upstream's path, `1` the batched one, in
  the same binary.
- **Benchmarks:** `llama-bench -p 512 -n 128 -r 3`, off and on interleaved per thread count.
- **KLD:** `llama-perplexity --kl-divergence` on wikitext-2 (20 chunks of 512), the batched path
  against upstream's.
- **The machines:**
  - **x86:** Ryzen 7 5800H (Zen 3, 8 cores / 16 threads, module `haswell`), Docker on Windows;
    downloads were running during part of the first series.
  - **ARM:** Neoverse-N1, cfarm424 (64 cores, dotprod, no i8mm, module `armv8.2_2`), a shared
    host; some rounds ran next to other users' load.
  - **s390x:** IBM z16, cfarm191 (8 CPUs, module `z16`), OLMoE only ([below](#s390x)).

## Prompt processing (pp512, t/s, off → on)

| model (type) | machine | threads | upstream | batched (final) | |
|---|---|--:|--:|--:|--:|
| gpt-oss-20b (MXFP4) | Zen 3 | 4 | 28.56 | 37.23 | +30 % |
|  |  | 8 | 41.68 | 59.58 | +43 % |
|  |  | 16 (SMT) | 54.44 | 69.80 | +28 % |
|  | N1 | 8 | 49.32 | 80.85 | +64 % |
|  |  | 16 | 98.28 | 152.46 | +55 % |
|  |  | 32 | 176.97 | 278.70 | +57 % |
| Qwen3-30B-A3B (Q4_K) | Zen 3 | 4 | 39.96 | 53.27 | +33 % |
|  |  | 8 | 58.32 | 84.65 | +45 % |
|  |  | 16 (SMT) | 71.72 | 82.27 | +15 % |
|  | N1 | 8 | 56.81 | 66.77 | +18 % |
|  |  | 16 | 105.99 | 121.84 | +15 % |
|  |  | 32 | 190.66 | 215.69 | +13 % |
| OLMoE-1B-7B (Q4_0) | Zen 3 | 4 | 81.11 | 116.78 | +44 % |
|  |  | 8 | 138.81 | 189.95 | +37 % |
|  |  | 16 (SMT) | 204.83 | 236.12 | +15 % |
|  | N1 | 8 | 192.28 | 274.53 | +43 % |
|  |  | 16 | 378.50 | 506.39 | +34 % |
|  |  | 32 | 705.85 | 989.98 | +40 % |

Rows with the final patch: Zen 3 at 8 threads, and N1 Qwen3 plus every N1 row at 32 threads. The
other rows were measured with the first version. For MXFP4 and Q4_0 the two versions do the same
work, and the activation quantizers have SIMD forms on both machines.

**Token generation** (tg128) stays within the spread at every point, e.g. gpt-oss on N1 at 32
threads 42.60 → 42.10, Qwen3 on Zen 3 at 8 threads 16.42 → 16.51. One N1 run at 32 threads read
38.03 → 31.86 ± 3.76 for Qwen3. Repeated in alternating rounds on that host (load average 33 from
other users), it read 38.00 / 35.10 and 32.20 / 35.11: noise.

## The defect the real weights found

The first version of the batched path quantized an expert's rows inside every (expert, column
chunk) unit. With more threads there are more column chunks per expert (`nck`), so the same rows
were quantized up to four times at 32 threads. On the N1, Qwen3-30B (Q4_K):

| threads | upstream | first version | final |
|--:|--:|--:|--:|
| 8 | 56.66 | 62.22 (+10 %) | 66.77 (+18 %) |
| 16 | 107.62 | 104.95 (−2 %) | 121.84 (+15 %) |
| 32 | 200.91 | 164.04 (**−18 %**) | 215.69 (+13 %) |

The upstream column of the final run is from a later series on the shared host: 56.81 / 105.99 /
190.66.

`perf` at 32 threads: upstream's path spends 36.5 % in `ggml_gemv_q4_K_8x4_q8_K` and 13.9 % in
`ggml_gemv_q6_K_8x4_q8_K`. The first version spent 30.3 % in `ggml_gemm_q4_K_8x4_q8_K` and 5.5 % in
the Q6_K gemm, but **20.8 % inside `forward_mul_mat_id` itself** (the inlined
`ggml_quantize_mat_q8_K_4x4_generic`) plus 7.5 % in `memcpy`. x86 has an AVX2
`ggml_quantize_mat_q8_K_4x8`; ARM has no SIMD Q8_K quantizer at all. MXFP4 and Q4_0 quantize to
Q8_0, which has NEON forms, so only K-quants on ARM showed it.

The final version gathers and quantizes every group of four rows once, into a buffer the units
share (`work_size` reserves one row per routed token slot: 9.6 MB for Qwen3-30B at a 512-token
batch). The groups are split statically over the threads, then a barrier; both are skipped when no
expert got four tokens.

## KLD: why Q4_K is not zero

| model | machine | batched vs upstream: mean KLD | max | same top token |
|---|---|--:|--:|--:|
| gpt-oss-20b (MXFP4) | Zen 3 | 0.000000 | 0.000052 | 100 % |
|  | N1 | 0.000000 | 0.000048 | 100 % |
| OLMoE-1B-7B (Q4_0) | Zen 3 | 0.000000 | 0.000063 | 100 % |
|  | N1 | 0.000000 | 0.000056 | 100 % |
| Qwen3-30B-A3B (Q4_K) | Zen 3 | 0.007511 | 0.929 | 95.8 % |
|  | N1 | 0.007316 | 0.897 | 96.1 % |

For MXFP4 and Q4_0, gemm and gemv round alike, so batching changes nothing. For Q4_K they round
differently (another summation order of the Q8_K block sums). In an MoE model, a router near a tie
then picks another expert, which turns a rounding difference into a different token now and then.

**The reference settles which path is "right".** Both paths were compared with the unrepacked
generic computation (`--no-repack`, 8 chunks, N1):

- upstream's repack path: mean KLD 0.007986, same top token 95.5 %;
- the batched path: mean KLD 0.008496, same top token 95.0 %.

Both are as far from it as from each other. The final version gives the same KLD as the first,
digit for digit, since it computes the same products.

## Through the jar

`LlamaModel.setCpuMoeBatched(false/true)` alternated in one JVM on the same chat request, prompt
cache off, the CI-built 6.1.0-SNAPSHOT `cpu-linux-aarch64` jar (first version of the patch), N1,
16 threads:

- gpt-oss-20b: prompt 79.4 / 81.1 → 130.7 / 130.6 t/s (+63 %), generation 29.9 → 29.3;
- Qwen3-30B: 82.5 / 84.5 → 82.6 / 82.7 (the first version's Q4_K result at 16 threads).

The switch reaches the native path through the jar, and the loader picked `armv8.2_2`.

## s390x

ggml repacks only Q4_0 on s390x (the VXE `q4_0_4x8` kernels), so OLMoE-1B-7B Q4_0, converted to
big-endian with `gguf_convert_endian.py`, on cfarm191 (IBM z16, 8 CPUs, module `z16`; a shared host,
load average 5 from other users during the run):

| threads | pp512 upstream | batched | | tg128 upstream | batched |
|--:|--:|--:|--:|--:|--:|
| 2 | 24.34 | 38.97 | +60 % | 8.30 | 8.26 |
| 4 | 45.79 | 76.23 | +66 % | 17.04 | 17.37 |
| 8 | 43.33 | 59.03 | +36 % | 0.31 | 0.33 |

At 8 threads every CPU was shared with other users' processes, which spinning threads punish:
generation collapsed on both paths alike. KLD of the batched path against upstream's (8 threads,
20 chunks): mean 0.000000, max 0.000059, same top token 99.98 %.

## Not measured here

AVX-512/AMX (`sapphirerapids`) and Zen 4 (`zen4`), which need a paid cloud instance; ARM with i8mm
(Graviton 3/4: the `q4_K_8x8` / `q4_0_4x8` kernels); and the hybrid-core case the original
measurement emulated.
