<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Every ARM build converts fp32 to fp16 in software, including ours and upstream's

Measured **2026-10-13**. One `#define` is missing from the NEON branch of
`ggml/src/ggml-cpu/simd-mappings.h`, so `GGML_CPU_FP32_TO_FP16` falls through to a software bit
manipulation while the branch's own single-instruction conversion sits there unused. Same shape of
defect as the one [ggml-org/llama.cpp#30270](https://github.com/ggml-org/llama.cpp/pull/30270) fixed
for x86 F16C on 2026-10-11 -- in the other direction, on a different architecture, still open.

Found while isolating that PR ([`upstream-f16c-30270.md`](upstream-f16c-30270.md)): a `#error` guard
written to stop the omission recurring fired immediately on upstream's unmodified header.

**Two findings, and they must not be confused:**

| | |
|---|---|
| **The function is 8.9x to 13.3x slower than it needs to be**, in 26 of 26 shipped ARM binaries | measured, static and timed, four ways |
| **Inference does not get faster** | measured: the function is **0.03 %** of cycles, because the dominant path never calls it |

So this is a correctness, consistency and portability fix. It is **not** an inference speed-up on a
current default build, and saying otherwise would repeat exactly the mistake #30270 made in the
other direction.

## The chain, branch by branch

The header picks the conversion macros from a chain of per-architecture branches. Each defines a
`GGML_CPU_COMPUTE_*` pair -- the raw intrinsic -- and is expected to **select** it as
`GGML_CPU_FP16_TO_FP32` / `GGML_CPU_FP32_TO_FP16`. Anything left unselected falls through to the end
of the section: a 64K lookup table in one direction, and `ggml_compute_fp32_to_fp16` -- a software
bit manipulation -- in the other.

Read at b11576, all four branches:

| branch | defines COMPUTE pair | selects fp16 to fp32 | selects fp32 to fp16 |
|---|:-:|:-:|:-:|
| `__ARM_NEON && __ARM_FP16_FORMAT_IEEE` | yes | yes | **no** |
| `__F16C__` (x86) | yes | from outside the chain, by #30270 | same |
| `__POWER9_VECTOR__` | yes | yes | yes |
| `__riscv && __riscv_zfhmin` | yes | yes | yes |

Two of four were incomplete. #30270 completed one of them -- from 88 lines below, outside the chain.
The ARM one is open, in the direction that fix did not cover.

**The sharpest way to state it:** the fallback reads
`#define GGML_CPU_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)`, and `GGML_COMPUTE_FP32_TO_FP16` is
ggml-base's deliberately portable software conversion -- the one `ggml.c` uses everywhere because
ggml-base may not assume CPU features. The whole purpose of the `GGML_CPU_*` pair is to be the
*accelerated* variant. On ARM it is defined to be the unaccelerated one.

That it is an omission and not a decision is upstream's own comment on that fallback, which has
always promised both:

> On ARM NEON, it's quicker to directly convert x -> x instead of calling into
> `ggml_lookup_fp16_to_fp32`, so we define `GGML_CPU_FP16_TO_FP32` **and `GGML_CPU_FP32_TO_FP16`**
> elsewhere for NEON. This is also true for POWER9.

POWER9 defines both. NEON defines one. And `neon_compute_fp32_to_fp16` -- a one-line `__fp16`
conversion -- sits in the branch with nothing referencing it.

## Every shipped ARM binary, disassembled

`ggml_cpu_fp32_to_fp16`, counting fp32-to-fp16 conversion instructions (`fcvt h`, `fcvtn`,
`fcvtn2`), with the **other direction in the same file** as an in-file control -- that one the
branch selects correctly:

| artifact | files | fp32 to fp16 | fp16 to fp32 *(control)* |
|---|:-:|:-:|:-:|
| ours `cpu-linux-aarch64` | 8 variants | **0 in each** | 9 / 9 / 9 / SVE / SVE / SVE / 2 / 2 |
| ours `cpu-android-aarch64` | 7 variants | **0 in each** | 5 / 5 / 5 / 5 / 1 / 1 / 1 |
| ours `cpu-windows-aarch64` | `ggml-cpu.dll` | **0** | 6 |
| ours `metal-macos-aarch64` | `libjllama.dylib` | **0** | 9 |
| **upstream** `bin-ubuntu-arm64` b11576 | 8 variants | **0 in each** | 9 / 9 / 9 / SVE / SVE / SVE / 2 / 2 |
| **upstream** `bin-macos-arm64` b11576 | `libggml-cpu.dylib` | **0** | 6 |

The control column is the point: **in one and the same object file** the widening direction carries
hardware conversions and the narrowing direction carries none.

Two caveats on the method, both worth stating because the first one bit this measurement:

* A pattern of `fcvt` alone is **wrong**: `fcvtzs` / `fcvtns` are float-to-**integer**, i.e. the
  quantisation step, and they are everywhere. The shipped `quantize_row_q8_0` contains 8 `fcvtns`
  and **no** fp16 conversion -- which would have been reported as 8 by a loose pattern.
* Where the control column reads *SVE*, the widening uses the predicated `fcvt z, z` form, which a
  scalar-shaped pattern does not match -- a limit of the pattern, not an absence.

The fp32-to-fp16 column needs no caveat at all: a bare search for **any** `fcvt` of any form, across
all 26 files, returns **zero**. That number depends on no pattern.

### The x86 control, at artifact level

The same sweep over this project's 14 x86-64 CPU modules, counting `vcvtps2ph`:

| modules | count |
|---|---|
| `ivybridge`, `piledriver`, `haswell`, `skylakex`, `cannonlake`, `cascadelake`, `cooperlake`, `icelake`, `zen4`, `alderlake`, `sapphirerapids` | **5-6 each** |
| `x64`, `sse42`, `sandybridge` | **0** |

Those three zeros are exactly the three variants **without F16C** (which arrived with Ivy Bridge), so
they are correct -- there is no instruction to emit. Same project, same build system, same function:
where the hardware has the instruction, x86 uses it; where it does not, x86 does not. Only ARM has it
everywhere and uses it nowhere.

## What it costs, read out of the shipped binary

The loop of `ggml_cpu_fp32_to_fp16` in `libggml-cpu-armv8.0_1.so`, as published:

| | shipped | with the one `#define` |
|---|---|---|
| instructions per element | **25** | `fcvtn v0.4h, v0.4s` -- **4 elements per instruction** |
| serial float dependency chain | 6 (`fabs`, `fmul`, `fmov`, `fmadd`, `fmov`) | none |
| branches in the loop body | 3, one of them **data-dependent** (inf/NaN) | none |

The shipped loop is the classic software fp16 encoder: `fabs`, multiply by `0x77800000`, compare
against `0x71000000`, `csel`, `fmadd` by `0x08800000`, then the mantissa and sign stitched back
together with shifts and masks. With the selection in place it is a load, one `fcvt` and a store --
and clang then vectorises it to four elements per conversion.

The latency chain, not the instruction count, is what dominates: five serial FP operations at
several cycles each, per element.

## Throughput of the function, where a machine can be trusted

On **cfarm185** (GCC Compile Farm, Ampere eMAG, 32 cores), a 16M-element bulk conversion, best of 7,
with the **opposite direction measured in the same binary and the same run** as a control the change
cannot touch:

| compiler | fp32 to fp16, upstream | fixed | factor | fp16 to fp32 *(control)* |
|---|---:|---:|---:|---|
| GCC 8.5 `-O3` | 82.0 Melem/s | 1087.2 | **13.3x** | 4259.3 to 4258.3 (**-0.02 %**) |
| clang 21.1.8 `-O3` | 341.9 | 3031.6 | **8.9x** | 4255.3 to 4268.1 (+0.3 %) |
| GCC `-O3 -march=armv8.2-a+fp16` | 82.0 | 1087.2 | 13.3x | 4246.1 to 4244.9 (-0.03 %) |
| clang `-O3 -march=armv8.2-a+fp16` | 342.0 | 3037.9 | 8.9x | 4252.8 to 4254.2 (+0.03 %) |

clang recovers part of the loss by vectorising the software path (342 against GCC's 82) and the fix
is still worth **8.9x** on top of that. The control stays within +-0.3 % in all eight runs -- same
file, same binary, same run -- which is what separates this from a timing artefact.

## End to end: nothing, and the reason is measured

One tree built twice on cfarm185 with clang, differing only in that `#define`; the two artifact sets
run interleaved with upstream's own `llama-bench`, `-r 5`, on a **1.5 GB F16** model and on
**Qwen3-0.6B-Q4_K_M**:

| model, `-t 16` | upstream | fixed | delta |
|---|---:|---:|---:|
| F16 `pp512` | 140.88 | 141.07 | +0.13 % |
| F16 `tg128` | 37.61 | 37.65 | +0.11 % |
| Q4_K_M `pp512` | 84.63 | 84.59 | -0.05 % |
| Q4_K_M `tg128` | 40.98 | 40.95 | -0.07 % |

All four inside the run-to-run spread (two runs of the *same* binary differ by up to 0.09 %).

The `-t 32` rows of the same run are **discarded**, not reported, and they are worth showing as the
reason why. That machine carries a permanent 8-core foreign load, so 32 threads oversubscribe it:
two runs of the same binary spread by up to **10.5 %** there, and `llama-bench`'s own error bar
reaches **+-9.8 %** within a single run. Taken at face value those rows would say the fix makes
generation **12 % to 18 % slower** -- a confident-looking result, with error bars, that is plainly
false. A number from a machine whose load is unknown is not a number, and the check that exposes it
is cheap: interleave the variants and compare two runs of the *same* binary first.

**Why there is nothing to see**, from `perf` on the F16 `pp512` run:

| share of cycles | function |
|---:|---|
| **83.93 %** | `tinyBLAS<...>::gemm` |
| 9.07 % | `ggml_compute_forward_flash_attn_ext` |
| 0.16 % | `ggml_vec_dot_f16` |
| **0.03 %** | `ggml_cpu_fp32_to_fp16` |

`GGML_LLAMAFILE` is on by default, and tinyBLAS takes `unsigned short` as an operand type and
converts **inside its own microkernel** -- so for an F16 model's matmuls the broken function never
sees the activations at all. 13x on 0.03 % of the time is a 0.03 % saving, which is precisely what
the table above shows.

This was predicted wrong here before it was measured: the expectation written down in advance was a
large gain on F16 prompt processing, on the reasoning that `GGML_TYPE_F16`'s `vec_dot_type` is
`GGML_TYPE_F16` and so `mul_mat` would convert the whole activation matrix per call. It would --
if tinyBLAS were not there first. The prediction was recorded before the run, which is the only
reason the error is visible at all.

## Why nobody noticed, in full

Three things have to be true at once, and they are:

1. **tinyBLAS bypasses it** for F16 matmuls -- 84 % of the cycles, measured above.
2. **Quantized models never reach it**: their activations go to Q8_0/Q8_K through a different
   `from_float` that was never broken. Everybody benchmarks quantized models.
3. **What is left is tiny**: the KV cache writes (`cache_type_k`/`cache_type_v` default to
   `GGML_TYPE_F16`, written through `ggml_compute_forward_dup`/`cpy`), spread over 16 threads.

And it cannot be caught by a build: `ggml_cpu_fp32_to_fp16` compiles and produces bit-identical
results either way (see below), so nothing fails, nothing warns, and no test notices.

## Bit-exact for every non-NaN value -- exhaustively, and the NaN case argues for the fix

Both conversions were applied to **every one of the 2^32 float32 bit patterns** and compared (44 s
on cfarm185):

| inputs checked | 4 294 967 296 -- every float32 bit pattern |
|---|---|
| differing, **non-NaN** | **0** |
| differing, NaN payload | 16 744 448 |

So for every finite value, every zero, every denormal and both infinities the two paths agree bit for
bit. They differ only on **NaN inputs**, where the software encoder emits a canonical quiet NaN
(`0x7E00` with the sign, visible as `orr w7, w5, #0x7e00` in the shipped disassembly) while the
hardware instruction propagates the truncated payload, as IEEE 754 recommends.

**That difference is not a reason against the fix -- it is a reason for it**, and this is measured
rather than argued: the identical comparison run on **x86 with F16C**, where upstream already ships
the hardware path, gives

| | non-NaN | NaN payload |
|---|---|---|
| ARM, software vs `fcvt` | 0 | 16 744 448 |
| **x86, software vs `vcvtps2ph`** | **0** | **16 744 448** |

the same numbers. The eleven F16C x86 modules this project ships already behave the hardware way, so
making ARM select its own instruction brings the two architectures into agreement instead of
splitting them. And nothing in llama.cpp reads a NaN payload: a NaN anywhere in weights or
activations is already a lost computation, and both encodings are quiet NaNs that propagate.

## The guard, proven both ways on real aarch64

```c
#if defined(GGML_CPU_COMPUTE_FP32_TO_FP16) && !defined(GGML_CPU_FP32_TO_FP16)
#error "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 but did not select it"
#endif
```

Placed after the branch chain and before the fallback, compiled on cfarm185 against the real b11576
header: it **fires** on upstream's unmodified file and **passes** once the `#define` is added. It
cannot fire spuriously -- POWER9 and RISC-V select both macros, and a platform with no branch at all
(s390x) defines no COMPUTE macro for it to catch.

## Limits of this measurement

* **One ARM core generation.** cfarm185 is an Ampere eMAG, a 2018 design, and it is the only ARM
  machine on the compile farm that accepts our key. The static evidence is generation-independent;
  the 8.9x to 13.3x factor is not, and a Neoverse or an Apple core may differ.
* **A shared machine.** Permanent 8-core foreign load. `-t 16` reproduced to 0.05 % between runs of
  the same binary and is reported; `-t 32` spread 7.5 % and is not.
* **Two compilers**, GCC 8.5 and clang 21.1.8. MSVC on Windows-on-ARM is untested, though its
  shipped `ggml-cpu.dll` shows the same zero.
* **No claim about Apple Silicon throughput.** The macOS rows are disassemblies of the shipped
  binaries, which is why they appear in the static table and not in the timing one.
* **Only one model size and two quantisations.** A build with `GGML_LLAMAFILE=OFF`, a very large KV
  cache, or an embedder calling the CPU backend's `from_float` directly would weight the function
  differently; none of those was measured.
