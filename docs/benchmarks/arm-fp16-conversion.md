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

**Submitted upstream as [ggml-org/llama.cpp#30505](https://github.com/ggml-org/llama.cpp/pull/30505)**
and carried here until it merges as
[`llama/patches/0019-ggml-cpu-select-the-neon-fp32-to-fp16-conversion.patch`](../../llama/patches/0019-ggml-cpu-select-the-neon-fp32-to-fp16-conversion.patch).

> **Read "One day later" before taking the guard or the F16C move in this page at face value.** The
> PR was rebased on 2026-10-14 after upstream #30343 reverted half of #30270; the guard grew a
> carve-out, the F16C move was dropped, and the ARM measurements below are unaffected (the function
> and the NEON branch are byte-identical across that base change).

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
| `__F16C__` (x86) | yes | from outside the chain, by #30270 (**reverted to the table by #30343**, see "One day later") | from outside the chain, by #30270 |
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

## Throughput of the function, on five ARM core types and three operating systems

A 16M-element bulk conversion, best of 7, with the **opposite direction measured in the same binary
and the same run** as a control the change cannot touch. All three machines are GCC Compile Farm
hosts:

| machine | core | OS / notes | compiler | upstream | fixed | factor | control |
|---|---|---|---|---:|---:|---:|---|
| cfarm185 | Ampere eMAG (2018) | Linux | gcc 8.5 | 82.0 Melem/s | 1087.2 | **13.3x** | -0.02 % |
| cfarm185 | Ampere eMAG | Linux | clang 21.1.8 | 341.9 | 3031.6 | **8.9x** | +0.3 % |
| cfarm423 | **Cortex-A55** (in-order) | Linux, **64K pages** | gcc 14.2 | 265.1 | 1136.2 | **4.3x** | **0.00 %** |
| cfarm424 | **ARM Neoverse-N1**, 64 cores | Linux | gcc 14 | 323.8 | 1310.1 | **4.0x** | -2.0 % |
| cfarm424 | ARM Neoverse-N1 | Linux | clang 19.1.7 | 976.2 | 3030.3 | **3.1x** | +2.2 % |
| cfarm427 | ARM Neoverse-N1 | **FreeBSD 16** | FreeBSD clang 21.1.8 | 886.1 | 2168.4 | **2.45x** | -2.4 % |
| `c8g.large` | **ARM Neoverse-V2** (Graviton 4) | Linux | gcc 11 | 583.0 | 2442.0 | **4.2x** | -3.2 % |
| `c8g.large` | ARM Neoverse-V2 | Linux | clang 15.0.7 | 1595.7 | 6127.1 | **3.8x** | -1.5 % |
| cfarm104 | **Apple M1** | macOS 25.5.0 | Apple clang 21 | 2751.7 | 9811.2 | **3.6x** | -2.5 % |

**The factor narrows on wider cores and never vanishes: 2.45x to 13.3x**, across every ARM server
core generation from 2018 to Graviton 4, an in-order little core, a 64K-page machine, Apple Silicon,
and Linux, macOS and FreeBSD. The control on the Cortex-A55 is identical to the digit (3683.67 both
ways), which is as tight as this measurement can get. That shape makes sense --
a wider, deeper core executes the software path's serial float chain faster, so it loses less by
taking it, but the hardware instruction is still several times better. Where the older machine's
controls hold to ±0.3 %, the two newer ones move ±2.5 %; both are shared machines and that is an
order of magnitude below the effect.

On a newer compiler the *starting point* also improves, because clang vectorises the software path:
on the eMAG gcc 8.5 manages 82 Melem/s where clang 21 manages 342. The fix is worth multiples on
top of either.

**Apple Silicon is measured, not inferred.** `cfarm104` is a Mac mini M1 running macOS, and a plain
`cc` there predefines `__ARM_NEON` and `__ARM_FP16_FORMAT_IEEE` exactly as Linux does -- so
`metal-macos-aarch64` takes the same branch, which the disassembly of the shipped dylib already
showed and this confirms from the other side.

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

Run on three different architectures -- Ampere eMAG, Graviton 4 and x86-64 -- with identical counts
in each (7.4 s on Graviton 4, 44 s on the eMAG).

The 16M-element checksum of the microbenchmark is **byte-identical on all five core types and every compiler** (`00924503571709770412`: Ampere eMAG, Cortex-A55, Neoverse-N1, Neoverse-V2, Apple M1; nine compiler builds across Linux, macOS and FreeBSD), which is the same equality seen from a second direction.

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

## Where the guard goes, decided by mutation rather than by argument

Two positions are plausible: **A**, where #30270's out-of-chain block stood, immediately before the
portable fallback; and **B**, directly after the branch chain, which is the adjacent-to-the-cause
reading. Both compile, both pass the full eight-variant ARM build, and both are silent on a correct
tree -- so the question cannot be settled by trying one.

It can be settled by **mutation**: build a header for every way the invariant can be violated, and
for ways it can look violated but is not, then compile each with the guard at A and at B. Two kinds
of case are kept apart, because only one of them is a bug:

* **correctness** -- a `COMPUTE` macro is defined and never selected anywhere, so the build silently
  takes the software fallback. This is the defect that shipped.
* **style** -- a `COMPUTE` macro is defined in the chain but selected far below it. Builds correctly
  today; fragile, because reordering the file breaks it silently.

| case | kind | A | B |
|---|---|:-:|:-:|
| NEON defines, never selects (the shipped defect) | correctness | FIRES | FIRES |
| the NEON selection removed from the fixed tree again | correctness | FIRES | FIRES |
| a new first-in-chain branch defines but never selects | correctness | FIRES | FIRES |
| **a `COMPUTE` defined after the chain, never selected, no branch taken** | correctness | **FIRES** | **silent** |
| F16C defines in the chain and selects below the tables (#30270) | style | silent | FIRES |
| a `COMPUTE` defined after the chain and selected right there | style | silent | silent |
| the fixed tree, x86+F16C / NEON / plain x86 | clean | silent | silent |
| a new first-in-chain branch, complete | clean | silent | silent |

| position | correctness caught | correctness missed | false alarms | style flagged |
|---|---:|---:|---:|---:|
| **A** | **4 of 4** | **0** | 0 | 0 |
| B | 3 of 4 | **1** | 0 | 1 |

**A is strictly safer**: it catches every correctness defect, while B misses one -- a `COMPUTE`
definition added after the chain and never selected, which is exactly the silent-software-fallback
class this change is about. What B catches instead is a *style* case, and that one is already
removed by moving the F16C selection into its branch; the guard does not have to carry it.

So the chosen shape is **A plus the move**: the move makes the chain regular, the guard at A catches
everything that silently costs the instruction, and the two are independent -- the guard works with
or without the move, which keeps the change reviewable in pieces. **The move did not survive the day;
see "One day later" below, which also turns A from the safer position into the only one that builds.**

The first version of this measurement was invalid and is worth naming for anyone repeating it: it set
the expectation per mutant instead of per configuration (on x86 without F16C no branch is taken, so
there is nothing to catch), its injected branches were unreachable behind an earlier `#elif`, and its
error filter counted `1 error generated.` as a compile failure. All three made a guard look blind
when the test was.

## One day later: #30343 took back half of #30270, and the change had to answer for it

**2026-10-14.** The maintainer asked for a rebase. 26 minutes before that request,
[#30343](https://github.com/ggml-org/llama.cpp/pull/30343) (`685f70894`) had landed and **reverted
half of #30270**: F16C keeps `fp32->fp16`, while `fp16->fp32` goes back on the lookup table, with the
selecting `#define` commented out in place and an issue link above it. The reason is measured by its
author and is not about this function at all -- clang 20.1.8 lowers `_cvtsh_ss` into a `vpinsrw` that
merges into the live accumulator register of the quantized dot products and serialises the loop,
**45-65 % slower Q8_0/Q4_0 `MUL_MAT_ID`** kernels.

Three things followed, all measured with `clang -E` against the real header before anything was
pushed.

**1. The guard, unchanged, breaks the x86 build.** The `__F16C__` branch now defines
`GGML_CPU_COMPUTE_FP16_TO_FP32` and deliberately does not select it -- exactly the state the guard
refuses:

| upstream master + the guard as submitted | result |
|---|---|
| `-mavx2 -mf16c -mfma` | **fires** `fp16_to_fp32` |
| `-mavx512f -mf16c` | **fires** `fp16_to_fp32` |
| `-msse4.2` | silent |
| NEON | fires `fp32_to_fp16` -- the defect this change fixes |

So the invariant the guard encodes had acquired a deliberate exception. The `FP16_TO_FP32` half now
carries `&& !defined(__F16C__)`, with the issue link **on the condition line** so that whoever
resolves #30342 sees the clause that has to go with it. Commenting the whole half out, paired with
#30343's own commented-out `#define`, was the first idea and was rejected on measurement: the
carve-out keeps the check alive for NEON, POWER9, RISC-V and any future branch, while a commented-out
`#error` checks nothing anywhere.

Re-measured in that shape against current master:

| | result |
|---|---|
| x86 +F16C+AVX2, +F16C+AVX512, sse42, plain x86, NEON | **5 of 5 silent** |
| NEON selection removed again (the defect restored) | **fires** |
| a new chain branch defining both COMPUTE macros, selecting neither | **fires, both halves** |
| a `COMPUTE_FP32_TO_FP16` defined after the chain, unselected | **fires** |
| a `COMPUTE_FP16_TO_FP32` defined after the chain, unselected, non-F16C target | **fires** |

**4 of 4 correctness cases still caught** on every architecture without the documented exception. The
blind spot is exactly that exception and nothing wider: on x86 with F16C the `fp16->fp32` direction is
unchecked, which is the state #30343 intends.

**2. Position B stopped being merely weaker and became unbuildable.** #30343 leaves the F16C
*selection* in the block below the lookup-table declarations -- that is, **after** the end of the
chain. The same guard placed directly after the chain therefore runs before that selection exists:

| the guard at position B, clean tree | result |
|---|---|
| x86 +F16C | **`no select fp32_to_fp16`** -- every x86 F16C build red |
| x86 -F16C, NEON | silent |

The mutation result above said A catches 4 of 4 against B's 3 of 4. It now also says B does not
compile. The reviewing bot had independently endorsed A on the first ground; the second was not
available to it, because #30343 had not landed yet.

**3. The F16C move was dropped.** Its argument was that the two selections sat 88 lines below the
branch they complete, explaining their own position in a comment. After #30343 that block is a
*documented exception carrying an issue link*, not a stray placement -- and it is the whole conflict
surface of the rebase. Moving upstream's fresh comment while the maintainer asked for a narrow rebase
buys nothing, so the block is untouched. One consequence worth having: **the diff now applies both at
the pinned tag and on upstream master**, verified with `git apply --check` against each.

**And one correction to the replacement comment.** The first draft of the fallback comment read "for
any architecture whose branch above selected neither" -- which #30343 makes false, since F16C selects
one of the two directions. It is phrased per *direction* now. The reviewing bot had suggested almost
that wording ("when no accelerated branch selected"); it would have been inaccurate for the same
reason, one day later.

**What this says about the earlier numbers.** Nothing in the throughput, bit-exactness or
disassembly sections depends on #30343: both `ggml_cpu_fp32_to_fp16` and the `__ARM_NEON` branch of
this header are **byte-identical** between the base those were measured on and current master
(`722bc30bc9865793` and `f4ab08fd209502c4` on either side), so the ARM measurements still describe
the tree as it stands. What changed is the guard and the scope of the diff, both re-measured above.

## Limits of this measurement

* ~~One ARM core generation~~ -- **closed**: five core types measured (Ampere eMAG 2018, Cortex-A55,
  Neoverse-N1, Neoverse-V2 via AWS Graviton 4, Apple M1) on Linux, macOS and FreeBSD, with nine
  compiler builds, including a 64K-page machine. The factor does depend on the core, from 2.45x to
  13.3x, which is why it is reported as a range. Still untested: Neoverse N2, Apple M2 and later.
* **A shared machine.** Permanent 8-core foreign load. `-t 16` reproduced to 0.05 % between runs of
  the same binary and is reported; `-t 32` spread 7.5 % and is not.
* **Nine compiler builds** (gcc 8.5, gcc 11, gcc 14, gcc 14.2, clang 15.0.7, clang 19.1.7,
  clang 21.1.8, FreeBSD clang 21.1.8, Apple clang 21). MSVC on
  Windows-on-ARM is untested, though its shipped `ggml-cpu.dll` shows the same zero.
* **Only one model size and two quantisations.** A build with `GGML_LLAMAFILE=OFF`, a very large KV
  cache, or an embedder calling the CPU backend's `from_float` directly would weight the function
  differently; none of those was measured.
