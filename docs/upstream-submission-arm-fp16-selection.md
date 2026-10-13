<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Upstream submission, prepared: `ggml-cpu: select the NEON fp32->fp16 conversion`

Everything needed to open the pull request, ready to paste. **Nothing is filed and nothing is
committed.** The change sits uncommitted in the working tree of
`X:\Privat\OpenSource\github\llama.cpp.bernardladenthin`.

The measurement record behind it is [`benchmarks/arm-fp16-conversion.md`](benchmarks/arm-fp16-conversion.md).

---

## The patch: +10 / -6, one file

```diff
--- a/ggml/src/ggml-cpu/simd-mappings.h
+++ b/ggml/src/ggml-cpu/simd-mappings.h
@@ -42,6 +42,7 @@ extern "C" {
     #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)
 
     #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
+    #define GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
 
     static inline float neon_compute_fp16_to_fp32(ggml_fp16_t h) {
@@ -63,6 +64,8 @@ extern "C" {
         #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
         #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
     #endif
+    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
+    #define GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
 #elif defined(__POWER9_VECTOR__)
@@ -140,12 +143,13 @@ extern float ggml_table_f32_ue4m3[1 << 8];
 #define GGML_CPU_UE4M3_TO_FP32(x) ggml_ue4m3_to_fp32(x)
 #endif
 
-// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
-// do not produce codegen as checks below don't test for __F16C__
-// but this check does.
-#if defined(__F16C__)
-#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
-#define GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
+// A branch above that defines a GGML_CPU_COMPUTE_* macro must also select it: left unselected it
+// falls through to the portable fallback below, which is correct but never uses the instruction.
+#if defined(GGML_CPU_COMPUTE_FP16_TO_FP32) && !defined(GGML_CPU_FP16_TO_FP32)
+#error "a branch defines GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
+#endif
+#if defined(GGML_CPU_COMPUTE_FP32_TO_FP16) && !defined(GGML_CPU_FP32_TO_FP16)
+#error "a branch defines GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
 #endif
 
 // On ARM NEON, it's quicker to directly convert x -> x instead of calling into ggml_lookup_fp16_to_fp32,
```

Three details that make it read well in review:

* **The fix carries no comment**, because `__POWER9_VECTOR__` and `__riscv_zfhmin` have the same pair
  bare -- a comment there would be the inconsistency. After the change the NEON branch is identical
  in shape to both.
* **The guard replaces the out-of-chain block at its own position**, rather than sitting next to it.
  Where a selection used to live outside the chain, the check that makes out-of-chain selections
  impossible now lives.
* **It sits immediately above upstream's own comment** promising both macros for NEON, so the promise
  and the check that enforces it are read together.

**Applies cleanly to master at `9d7d50498` and at `296538a`** -- master moved during preparation and
both were tested.

## Suggested commit split

Three commits in one PR, so either of the last two can be dropped without touching the first.
Maintainers squash-merge, so this is for review readability only.

1. `ggml-cpu: select the NEON fp32->fp16 conversion` -- the one line.
2. `ggml-cpu: move the F16C selection into its own branch` -- behaviour-preserving, proven at object
   level.
3. `ggml-cpu: refuse a branch that defines a COMPUTE macro without selecting it` -- the guard.

## Conventions, checked against the repository rather than assumed

* `CONTRIBUTING.md` asks for `<module> : <commit title> (#<issue_number>)`, and **maintainers
  squash-merge and write that title themselves** -- so the PR *title* becomes the commit title and
  the `(#N)` is added at merge.
* **The module prefix this file's history uses is `ggml-cpu`** -- `ggml-cpu: vectorize fp32 to fp16
  conversion (#30157)`, `ggml-cpu: gate __fp16 on __ARM_FP16_FORMAT_IEEE (#26860)`; #30270 used
  `cpu:`.
* A PR needs **two independent reviews** unless a maintainer adds the "merge ready" label, and
  CONTRIBUTING suggests granting maintainers write access to the branch for faster reviews.

## Archaeology, so the PR does not misstate the history

* **Not a regression.** The NEON branch has never carried the fp32->fp16 selection: absent in all 25
  most recent commits touching the file, back to **2025-04-10**. The second half of that branch was
  never written.
* **The function was worked on three days earlier.** `ggml-cpu: vectorize fp32 to fp16 conversion`
  (#30157, 2026-10-10) added the `__VXE__`/`__VXE2__` vector path to `ggml_cpu_fp32_to_fp16` -- the
  same function. It now has vector paths for F16C, RISC-V and s390x and **none for NEON**, which is
  exactly why a missing macro selection is felt there and nowhere else. Context, not blame.
* **#26860 is not the cause** either: it only tightened the branch *conditions* with
  `__ARM_FP16_FORMAT_IEEE`, leaving the selections untouched.

---

## PR description, ready to paste

> ### ggml-cpu: select the NEON fp32->fp16 conversion
>
> The `__ARM_NEON` branch of `simd-mappings.h` defines both `GGML_CPU_COMPUTE_FP16_TO_FP32` and
> `GGML_CPU_COMPUTE_FP32_TO_FP16`, but selects only the first. The second therefore falls through to
> `GGML_COMPUTE_FP32_TO_FP16` at the end of that section -- ggml-base's portable software bit
> manipulation -- while `neon_compute_fp32_to_fp16`, a single `fcvt`, stays defined and referenced by
> nothing. `__POWER9_VECTOR__` and `__riscv_zfhmin` select both; the comment on that fallback already
> says NEON does too:
>
> > so we define GGML_CPU_FP16_TO_FP32 **and GGML_CPU_FP32_TO_FP16** elsewhere for NEON. This is also
> > true for POWER9.
>
> Same shape of omission #30270 fixed for F16C, in the other direction. It is not a regression -- the
> selection has been absent since at least 2025-04-10.
>
> **It is in the released binaries.** Disassembling `ggml_cpu_fp32_to_fp16` and searching for any
> `fcvt` of any form finds **none** in `llama-b11576-bin-ubuntu-arm64` (all 8 CPU variants) or
> `llama-b11576-bin-macos-arm64`, nor in the 17 ARM artifacts a downstream project ships across
> Linux, Android, Windows-on-ARM and macOS -- **26 binaries, zero conversion instructions**. In the
> same object file the *other* direction carries 9 of them, which is the asymmetry in one picture. The
> x86 side behaves correctly by comparison: the 11 F16C-enabled CPU variants emit 5-6 `vcvtps2ph`
> each, and the three without F16C (`x64`, `sse42`, `sandybridge`) emit none, as they should.
>
> **What it costs.** In the shipped `libggml-cpu-armv8.0_1.so` the loop is 25 instructions per
> element, with a six-operation serial float dependency chain and a data-dependent branch for the
> inf/NaN case. With the selection it is a load, one `fcvt` and a store, which the compiler then
> vectorises to `fcvtn v0.4h, v0.4s` -- four elements per instruction. Measured on three ARM core
> generations, a 16M-element bulk conversion, best of 7, with the opposite direction measured in the
> same binary and the same run as a control the change cannot touch:
>
> | machine | compiler | before | after | factor |
> |---|---|---:|---:|---:|
> | Ampere eMAG (2018) | gcc 8.5 | 82.0 Melem/s | 1087.2 | 13.3x |
> | Ampere eMAG | clang 21.1.8 | 341.9 | 3031.6 | 8.9x |
> | ARM Neoverse-N1 | gcc 14 | 323.8 | 1310.1 | 4.0x |
> | ARM Neoverse-N1 | clang 19.1.7 | 976.2 | 3030.3 | 3.1x |
> | Apple M1, macOS | Apple clang 21 | 2751.7 | 9811.2 | 3.6x |
>
> The factor narrows on newer cores and never vanishes -- **3.1x to 13.3x**. In every one of the ten
> runs the control direction stays within 2.5 %, against an effect of 3x and up.
>
> **It is bit-exact.** Both paths were applied to every one of the 2^32 float32 bit patterns: **0
> differences** for every finite value, zero, denormal and infinity. They differ only on NaN inputs
> (16 744 448 of them), where the software encoder emits a canonical quiet NaN and the instruction
> propagates the truncated payload. The identical comparison on x86 with F16C gives the same two
> numbers, so this makes ARM agree with the behaviour x86 already ships rather than splitting them.
> The 16M-element checksum is byte-identical on all three cores and all five compiler builds.
>
> **It does not make inference faster, and that should be said plainly.** `perf` on an F16 `pp512`
> run attributes **83.9 %** of cycles to `tinyBLAS::gemm`, which converts inside its own microkernel,
> and **0.03 %** to `ggml_cpu_fp32_to_fp16`; end to end the change is +0.13 % against a run-to-run
> spread of 0.09 %, i.e. nothing. Quantized models never reach the function at all, since their
> activations go through a different `from_float`. So this is a correctness and consistency fix: the
> `GGML_CPU_*` pair exists to be the accelerated variant, and on ARM it was defined to be the
> unaccelerated one.
>
> ### Second commit: the F16C selection moves into its branch
>
> #30270's two selecting `#define`s sit 88 lines below the chain, after the lookup-table
> declarations, which is why that code has to explain its own position ("checks below don't test for
> `__F16C__` but this check does"). Moving them into the `__F16C__` branch beside the intrinsics they
> select gives the chain one shape -- every branch defines and selects in the same place -- and makes
> the third commit's invariant a plain one rather than one with an exception.
>
> **Behaviour-preserving, proven at object level** rather than by reading: a translation unit calling
> both macros, compiled on x86 and disassembled, emits `vcvtps2ph` before and after with
> `-mavx2 -mf16c -mfma` and with `-mavx512f -mf16c`, and the software path before and after with
> `-msse4.2`. Macro resolution is identical in four x86 configurations.
>
> ### Third commit: the guard
>
> Two of the four branches in that chain had this defect at the same time -- F16C until #30270, NEON
> until this PR -- and neither was caught by a build, a warning or a test, because the fallback
> silently supplies a correct software implementation. The only symptom is an unused instruction.
>
> The `#error` pair checks exactly that state and cannot fire otherwise: POWER9 and riscv_zfhmin
> select both macros, and a platform with no branch at all defines no `COMPUTE` macro to catch.
> Verified three ways with clang against this file: on unmodified `master` it fires **only** in the
> NEON configuration, with the fix it is silent, and both trees compile unchanged when the guard is
> absent -- so the guard is the only thing that changes the outcome.
>
> ### Build and test, in upstream's own ARM release configuration
>
> Built with clang 21.1.8 and
> `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF`, i.e. all **eight** ARM CPU
> variant modules, each compiling this header with a different feature set (dotprod, fp16, SVE, i8mm,
> SVE2, SME). All eight build, **the guard fires nowhere in the whole build**, and
> `ggml_cpu_fp32_to_fp16` now contains `fcvt`, `fcvtn` and `fcvtn2` -- six occurrences -- in every one
> of them. Reverting the patch in the same build tree and rebuilding puts that count back to **0**.
>
> `ctest`: **52 of 54 pass**. The two failures -- `test-tokenizers-ggml-vocabs` (#16) and
> `test-jinja-py` (#33) -- are the same two, with the same numbers, on unpatched `master` in that
> same tree: they need test vocabularies and a Python jinja2 the machine does not have. So the suite
> is unchanged by the patch, measured rather than assumed.
>
> ### Testing summary
>
> * `clang -E` resolution matrix over five configurations (`x86 +F16C`, `x86 +F16C+AVX512`,
>   `x86 -F16C`, `x86` plain, NEON) for `master`, `master`+guard and the final tree.
> * A real x86 translation unit compiled and disassembled, before and after, three flag sets.
> * Full build with all 8 ARM CPU variants plus `ctest`, and the revert round trip in the same tree.
> * `llama-bench` on an Ampere eMAG (f16 and Q4_K_M, interleaved, `-r 5`) plus `perf record`.
> * Microbenchmark on three ARM cores and five compiler builds, with an in-run control.
> * Exhaustive 2^32 equivalence, on ARM and on x86.
> * Disassembly of 26 released ARM binaries, upstream's own two included.

---

## What is deliberately not claimed

* **No inference speed-up.** Measured and stated above. A prediction of a large gain was written down
  before the benchmark and was wrong; the `perf` profile explains why.
* **Three ARM core generations measured** (Ampere eMAG 2018, Neoverse-N1, Apple M1), on Linux and
  macOS, with five compiler builds. Untested: Neoverse V2/N2, Apple M2 and later. The factor depends
  on the core -- 3.1x to 13.3x -- which is why the PR text states a range rather than a number.
* **MSVC on Windows-on-ARM is untested** as a compiler, although the shipped `ggml-cpu.dll` built
  with it shows the same absence of `fcvt`.

## Three mistakes made while preparing this, kept because they are the method

* A pattern matching `fcvt` alone is **wrong**: `fcvtzs` / `fcvtns` are float-to-**integer**, the
  quantisation step. It would have reported 8 phantom fp16 conversions in `quantize_row_q8_0`. The
  headline number is therefore a bare search for any `fcvt` of any form, which depends on no pattern.
* The first end-to-end attempt ran at 32 threads on a machine with a permanent 8-core foreign load.
  Two runs of the same binary spread up to 10.5 % there, and those rows -- taken at face value --
  would have reported the fix making generation 12-18 % **slower**, with error bars. They are
  discarded; the reported rows are at 16 threads, where the same check reproduces to 0.05 %.
* The first real build was **void**: the script deleted the directory the patch had been uploaded
  into, then carried on and reported `BUILD OK` for unpatched source. A failed precondition has to
  exit, and what was applied has to be asserted -- the rebuilt script checks `+10/-6` before it
  configures.
