<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Upstream submission, prepared: `ggml-cpu: select the NEON fp32->fp16 conversion`

**Nothing is filed and nothing is committed.** The change sits uncommitted in the working tree of
`X:\Privat\OpenSource\github\llama.cpp.bernardladenthin`; the patch is also exported as a file.

Everything below the line **PR text** is meant to be pasted as-is. Everything above it is for us.

The full measurement record is [`benchmarks/arm-fp16-conversion.md`](benchmarks/arm-fp16-conversion.md).

---

## For us: the patch, and why this exact shape

**+10 / -6 in one file.** Applies cleanly to master at `9d7d50498` and at `296538a`.

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

Why this and not something smaller or larger:

* **The fix carries no comment.** `__POWER9_VECTOR__` and `__riscv_zfhmin` have the same pair bare, so
  a comment on the NEON one would itself be the inconsistency. Afterwards all three branches are
  identical in shape.
* **The F16C selection moves into its branch**, which is what turns the guard's invariant from "every
  branch selects what it defines, except F16C which does it 88 lines further down" into a plain one.
  It also retires the comment that had to explain that position.
* **The guard replaces the out-of-chain block at its own position** rather than sitting beside it, and
  lands immediately above upstream's own comment promising both macros for NEON -- promise and
  enforcement are read together.

## For us: commit split and conventions

Three commits in one PR, so either of the last two can be dropped without touching the first.
Maintainers squash-merge, so the split is for review readability only and the **PR title becomes the
commit title**.

1. `ggml-cpu: select the NEON fp32->fp16 conversion`
2. `ggml-cpu: move the F16C selection into its own branch`
3. `ggml-cpu: refuse a branch that defines a COMPUTE macro without selecting it`

`CONTRIBUTING.md` asks for `<module> : <title> (#<issue>)` and the maintainer adds the number at
merge. The module prefix this file's own history uses is **`ggml-cpu`**. A PR needs two independent
reviews unless a maintainer adds the "merge ready" label.

---

# PR text

> ### ggml-cpu: select the NEON fp32->fp16 conversion
>
> The `__ARM_NEON` branch of `simd-mappings.h` defines both `GGML_CPU_COMPUTE_FP16_TO_FP32` and
> `GGML_CPU_COMPUTE_FP32_TO_FP16`, but selects only the first. The second therefore falls through to
> `GGML_COMPUTE_FP32_TO_FP16` at the end of that section -- ggml-base's portable software bit
> manipulation, the one `ggml.c` uses everywhere because ggml-base may not assume CPU features --
> while `neon_compute_fp32_to_fp16`, a single `fcvt`, stays defined and referenced by nothing. The
> `GGML_CPU_*` pair exists to be the accelerated variant; on ARM it is defined to be the
> unaccelerated one.
>
> `__POWER9_VECTOR__` and `__riscv_zfhmin` select both macros. The comment on that fallback already
> says NEON does too:
>
> > so we define GGML_CPU_FP16_TO_FP32 **and GGML_CPU_FP32_TO_FP16** elsewhere for NEON. This is also
> > true for POWER9.
>
> Same shape of omission #30270 fixed for F16C, in the other direction. **Not a regression** -- the
> NEON selection is absent in all 25 most recent commits touching the file, back to 2025-04-10.
>
> Why it is felt on ARM and nowhere else: `ggml_cpu_fp32_to_fp16` -- the `from_float` of
> `GGML_TYPE_F16` -- has vector paths for F16C, RISC-V and s390x (the last added in #30157) and
> **none for NEON**, so there the whole function is the scalar loop through the macro.
>
> ### It is in the released binaries
>
> Disassembling `ggml_cpu_fp32_to_fp16` and searching for **any** `fcvt` of any form finds **none** in
> `llama-b11576-bin-ubuntu-arm64` (all 8 CPU variants) or `llama-b11576-bin-macos-arm64`, nor in the
> 17 ARM artifacts a downstream project ships across Linux, Android, Windows-on-ARM and macOS -- **26
> binaries, zero conversion instructions**. In the same object file the *other* direction carries 9 of
> them, which is the asymmetry in one picture.
>
> The x86 side is the comparison: of this project's 14 x86-64 CPU variant modules, the 11 with F16C
> emit 5-6 `vcvtps2ph` each and the three without (`x64`, `sse42`, `sandybridge`) emit none -- exactly
> matching the hardware. Only ARM has the instruction everywhere and uses it nowhere.
>
> ### What it costs
>
> In the shipped `libggml-cpu-armv8.0_1.so` the loop is **25 instructions per element**, with a
> six-operation serial float dependency chain (`fabs`, `fmul`, `fmov`, `fmadd`, `fmov`) and a
> data-dependent branch for the inf/NaN case. With the selection it is a load, one `fcvt` and a store,
> which the compiler then vectorises to `fcvtn v0.4h, v0.4s` -- four elements per instruction.
>
> Measured on four ARM core generations: a 16M-element bulk conversion, best of 7, with the
> **opposite direction measured in the same binary and the same run** as a control the change cannot
> touch.
>
> | machine | compiler | before | after | factor |
> |---|---|---:|---:|---:|
> | Ampere eMAG (2018) | gcc 8.5 | 82.0 Melem/s | 1087.2 | 13.3x |
> | Ampere eMAG | clang 21.1.8 | 341.9 | 3031.6 | 8.9x |
> | ARM Neoverse-N1 | gcc 14 | 323.8 | 1310.1 | 4.0x |
> | ARM Neoverse-N1 | clang 19.1.7 | 976.2 | 3030.3 | 3.1x |
> | ARM Neoverse-V2 (Graviton 4) | gcc 11 | 583.0 | 2442.0 | 4.2x |
> | ARM Neoverse-V2 | clang 15.0.7 | 1595.7 | 6127.1 | 3.8x |
> | Apple M1, macOS | Apple clang 21 | 2751.7 | 9811.2 | 3.6x |
>
> The factor narrows on newer cores and never vanishes -- **3.1x to 13.3x** -- across every ARM
> server core generation from 2018 to Graviton 4, plus Apple Silicon. In all fourteen runs the
> control direction stays within 3.2 %, against an effect of 3x and up.
>
> ### It is bit-exact
>
> Both paths applied to **every one of the 2^32 float32 bit patterns**: **0 differences** for every
> finite value, zero, denormal and infinity. They differ only on NaN inputs (16 744 448), where the
> software encoder emits a canonical quiet NaN and the instruction propagates the truncated payload.
> The identical comparison on x86 with F16C gives the same two numbers, so this brings ARM into
> agreement with the behaviour x86 already ships rather than splitting them. The 16M-element output
> checksum is byte-identical on all four cores and all seven compiler builds, and the exhaustive run
> was repeated on three architectures (Ampere eMAG, Graviton 4, x86-64) with identical counts.
>
> ### It does not make inference faster
>
> `perf` on an F16 `pp512` run attributes **83.9 %** of cycles to `tinyBLAS::gemm`, which converts
> inside its own microkernel, and **0.03 %** to `ggml_cpu_fp32_to_fp16`. End to end the change is
> +0.13 % against a run-to-run spread of 0.09 %, i.e. nothing; quantized models never reach the
> function at all, since their activations go through a different `from_float`. This is a correctness
> and consistency fix, and the 3x-13x belongs to the function, not to a model.
>
> ### Second commit: the F16C selection moves into its branch
>
> #30270's two selecting `#define`s sit 88 lines below the chain, after the lookup-table declarations,
> which is why they need a comment explaining their own position. Moving them into the `__F16C__`
> branch beside the intrinsics they select gives the chain **one shape** -- every branch defines its
> `COMPUTE` pair and selects it in the same place -- and makes the third commit's invariant a plain
> rule rather than one with an exception. It also removes a fragility: as written, a future reordering
> of the file silently returns F16C to the software fallback, which is the defect this PR is about.
>
> **Behaviour-preserving, proven at object level:** a translation unit calling both macros, compiled
> on x86 and disassembled, emits `vcvtps2ph` before and after with `-mavx2 -mf16c -mfma` and with
> `-mavx512f -mf16c`, and the software path before and after with `-msse4.2`. Macro resolution is
> identical in four x86 configurations.
>
> ### Third commit: the guard
>
> Two of the four branches in that chain had this defect at the same time -- F16C until #30270, NEON
> until this PR -- and neither was caught by a build, a warning or a test, because the fallback
> silently supplies a correct software implementation. The only symptom is a hardware instruction that
> is never used.
>
> The `#error` pair checks exactly that state and cannot fire otherwise: POWER9 and riscv_zfhmin
> select both macros, and a platform with no branch at all defines no `COMPUTE` macro to catch.
> Verified three ways with clang against this file: on unmodified `master` it fires **only** in the
> NEON configuration, with the fix it is silent, and both trees compile unchanged when the guard is
> absent -- so the guard is the only thing that changes the outcome.
>
> ### Build and test
>
> Built in the configuration used for the ARM releases -- `-DGGML_BACKEND_DL=ON
> -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF`, clang 21.1.8 -- i.e. all **eight** ARM CPU variant
> modules, each compiling this header with a different feature set (dotprod, fp16, SVE, i8mm, SVE2,
> SME). All eight build, **the guard fires nowhere in the whole build**, and `ggml_cpu_fp32_to_fp16`
> contains `fcvt`, `fcvtn` and `fcvtn2` -- six occurrences -- in every one of them. Reverting the
> patch in the same build tree and rebuilding puts that count back to **0**.
>
> `ctest`: **52 of 54 pass**. The two failures, `test-tokenizers-ggml-vocabs` (#16) and
> `test-jinja-py` (#33), are the same two with the same numbers on unpatched `master` in that same
> tree -- they need test vocabularies and a Python jinja2 the machine does not have. The suite is
> unchanged by the patch.
>
> ### Testing summary
>
> * `clang -E` resolution matrix over five configurations (`x86 +F16C`, `x86 +F16C+AVX512`,
>   `x86 -F16C`, `x86` plain, NEON) for `master`, `master`+guard and the final tree.
> * A real x86 translation unit compiled and disassembled, before and after, three flag sets.
> * Full build with all 8 ARM CPU variants plus `ctest`, and the revert round trip in the same tree.
> * Microbenchmark on four ARM cores and seven compiler builds, with an in-run control.
> * `llama-bench` (f16 and Q4_K_M, interleaved, `-r 5`) plus `perf record`.
> * Exhaustive 2^32 equivalence, on ARM and on x86.
> * Disassembly of 26 released ARM binaries.
>
> ### Not claimed
>
> * No inference speed-up, as measured above.
> * Four ARM core generations were measured (Ampere eMAG, Neoverse-N1, Neoverse-V2, Apple M1) on
>   Linux and macOS with seven compiler builds. Neoverse N2 and Apple M2 and later are untested,
>   which is why the factor is given as a range.
> * MSVC on Windows-on-ARM is untested as a compiler, although the `ggml-cpu.dll` built with it shows
>   the same absence of `fcvt`.
