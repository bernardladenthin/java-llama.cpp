<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Isolating upstream PR #30270 (F16C on AVX2 CPUs)

Measured **2026-10-12**. [ggml-org/llama.cpp#30270](https://github.com/ggml-org/llama.cpp/pull/30270)
("cpu: fix AVX2 CPUs not using F16C when available", commit `c28e174141dd`, merged 2026-10-11) is
eight lines in `ggml/src/ggml-cpu/simd-mappings.h`. It was merged with, in the author's own words,
*"a better by-eyeball measurement"* on a FreeBSD machine with mismatched RAM clocks, **no numbers and
no review comments**.

This page is what a quantified answer looks like. **Short version: the fix is correct, and on this
machine and workload it changes nothing measurable.** Both halves matter.

## What the fix does

The header selects a conversion macro from a chain of per-architecture branches:

| | |
|---|---|
| the `#elif defined(__F16C__)` branch | defines only `GGML_CPU_**COMPUTE**_FP16_TO_FP32` -- the raw intrinsic |
| **the eight new lines** | define `GGML_CPU_FP16_TO_FP32` to point at it |
| the tail of the file | `#if !defined(GGML_CPU_FP16_TO_FP32)` → `ggml_lookup_fp16_to_fp32`, a 64K-entry table |

So before the fix, a CPU with the hardware instruction used **the lookup table**: the F16C branch
defined half of its own decision. The NEON, POWER9 and RISC-V branches each define *both* macros; only
F16C forgot. The fix is right.

## The measurement: one tree, built twice, one variable

`m8i.16xlarge` (Xeon 6975P-C, Granite Rapids, 64 vCPU), Ubuntu 24.04, gcc 13.3. llama.cpp **b11574**
cloned once and copied; the second copy had exactly those seven lines removed (printed into the run
log, so the removal is auditable). Both built with `GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON
GGML_NATIVE=OFF` (46 s and 48 s, 14 CPU modules each). Measured with **upstream's own `llama-bench`**,
5 repetitions, one module staged at a time, Qwen3-0.6B **Q4_K_M**.

The variant tiers come from upstream's own `ggml/src/CMakeLists.txt`, so each row carries a prediction:

| module | flags it is built with | prediction |
|---|---|---|
| `sse42`, `sandybridge` | **no F16C** | **must not move** -- the control |
| `ivybridge`, `piledriver` | F16C, no AVX2 | should gain |
| `haswell`, `alderlake` | F16C + AVX2 | **the case the PR names** ("i5-8th gen or i7-4790") |
| `cascadelake`, `sapphirerapids` | + AVX512 | should gain less: AVX512 has its own vector block |

### Result: nothing moves, and the controls prove the method

16 threads, tokens/s with the spread `llama-bench` reports:

| module | pp512 fixed | pp512 reverted | Δ | tg128 fixed | tg128 reverted | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `sse42` *(control)* | 224.49 ± 0.22 | 225.41 ± 0.11 | −0.4 % | 106.90 ± 0.17 | 107.60 ± 0.36 | −0.7 % |
| `sandybridge` *(control)* | 894.14 ± 1.59 | 891.68 ± 1.45 | +0.3 % | 156.32 ± 1.52 | 155.59 ± 1.54 | +0.5 % |
| `ivybridge` | 900.48 ± 0.70 | 903.00 ± 1.27 | −0.3 % | 174.52 ± 2.28 | 173.63 ± 2.44 | +0.5 % |
| `piledriver` | 938.45 ± 0.60 | 939.83 ± 3.88 | −0.1 % | 171.41 ± 2.20 | 172.86 ± 3.13 | −0.8 % |
| **`haswell`** | 1433.81 ± 3.87 | 1409.04 ± 2.07 | **+1.8 %** | 196.47 ± 1.85 | 196.01 ± 2.21 | +0.2 % |
| `alderlake` | 1431.95 ± 6.73 | 1435.12 ± 2.30 | −0.2 % | 197.56 ± 2.23 | 194.58 ± 2.02 | +1.5 % |
| `cascadelake` | 1987.48 ± 7.57 | 1975.39 ± 3.48 | +0.6 % | 197.62 ± 2.46 | 199.48 ± 2.69 | −0.9 % |
| `sapphirerapids` | 2985.25 ± 2.81 | 2981.14 ± 2.06 | +0.1 % | 201.15 ± 0.61 | 198.83 ± 0.82 | +1.2 % |

And the PR's own case at a second thread count, so it is not one observation:

| module | threads | pp512 fixed | pp512 reverted | Δ |
|---|---:|---:|---:|---:|
| `haswell` | 8 | 772.74 ± 1.98 | 778.63 ± 0.52 | **−0.8 %** |
| `alderlake` | 8 | 779.58 ± 0.39 | 771.06 ± 0.60 | +1.1 % |

**Every difference is inside ±2 % with random signs.** The controls, which the fix *cannot* touch,
move as much as the test cases -- and `haswell` is +1.8 % at 16 threads but **−0.8 % at 8**. That is
the shape of noise, not of an effect.

### Why it is plausible that the gain is real elsewhere

The fix removes a lookup table from the conversion path, and the table is ~256 KB -- L2-resident on
this Xeon, and competing for bandwidth on a memory-starved machine. The PR author describes exactly
such a machine ("clock-mismatched RAM") and even cautions that "gains like these may be greater than
reasoned". With **Q4_K_M** the weights are not fp16 either, so the conversion sits at the margins of
the work rather than in the hot loop.

So this is not a case against the change. It is a case against the number: **the gain is machine- and
workload-dependent, and on a bandwidth-rich server CPU with a 4-bit quant it is zero.**

## The follow-up that does not depend on performance

Two structural points, which stand whatever the throughput does.

**1. The two lines belong inside the branch.** They sit **88 lines** below the `#elif
defined(__F16C__)` branch they complete, outside the chain -- which is why the new code has to explain
the ordering in its own comment (*"checks below don't test for `__F16C__`"*). NEON, POWER9 and RISC-V
each define both macros inside their branch. As written, the two halves of one decision are far apart,
and a future reordering of the file breaks it **silently**, back to the lookup table: precisely the
defect that went unnoticed until now.

**2. A guard makes the omission impossible, and here it is proven both ways.** Placed after the branch
chain and **before** the fallback:

```c
#if defined(GGML_CPU_COMPUTE_FP16_TO_FP32) && !defined(GGML_CPU_FP16_TO_FP32)
#error "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 but not GGML_CPU_FP16_TO_FP32"
#endif
```

Compiled with clang, `-fsyntax-only`, against the real header of b11574 plus the four headers it
includes:

| tree | flags | result |
|---|---|---|
| fixed + guard | `-mavx2 -mf16c -mfma` | **passes** |
| **reverted + guard** | `-mavx2 -mf16c -mfma` | **fails** with the `#error` |
| fixed + guard | `-msse4.2` (no F16C) | passes -- the branch is not taken, nothing to catch |
| reverted + guard | `-msse4.2` | passes -- same |
| *control:* fixed, **no** guard | `-mavx2 -mf16c -mfma` | passes |
| *control:* reverted, **no** guard | `-mavx2 -mf16c -mfma` | passes |

The last two rows are the ones that make the first two mean something: both trees compile on their
own, so the guard is the only thing that changes the outcome. The two no-F16C rows show it does not
fire spuriously.

## Two mistakes made while producing this, both worth recording

**The first attribution was wrong.** A `haswell` jump from 221.8 to 422.7 t/s between a b11538
4-vCPU run and a b11558 64-vCPU one was read here as *consistent with #30270*, because `haswell` had
gained far more than the AMX module (+91 % against +54 %). This isolation refutes it: #30270
contributes ~0, so that jump was **the instance size** -- 4 threads on a 64-vCPU machine have far more
memory bandwidth to themselves than a 4-vCPU machine does. An asymmetry between two rows is not an
isolation, and the correction is kept in [`cpu-backend-modules.md`](cpu-backend-modules.md) rather
than quietly removed.

**The first guard test proved nothing.** It appended the guard *after* `#include "simd-mappings.h"`,
and the header's own fallback defines the macro unconditionally-if-undefined -- so after the include
both macros are always defined and the guard could never fire. It reported `exit=0` for both trees,
which reads like "the guard does not work" and actually meant "the test does not test". A guard that
passes in the broken case has to be suspected of being in the wrong place.

## What this run cannot show

* **One CPU, one model, one quantisation.** A memory-bound machine, an fp16 model or a GPU-less laptop
  could all show the gain the author saw. The claim here is narrow on purpose: *on this machine, with
  this quant, at 8 and 16 threads, no measurable effect.*
* **gcc 13.3 only.** A different compiler could emit different code around the macro; clang and MSVC
  were not measured.
* **Nothing about correctness.** Both builds produced the same output; no bit-exactness comparison of
  the conversion itself was made.
